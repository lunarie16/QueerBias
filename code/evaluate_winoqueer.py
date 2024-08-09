import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from dotenv import load_dotenv
from PromptTuning import PromptTuningModelDDP
from peft import PeftModel, AutoPeftModelForCausalLM, PeftConfig
import difflib
import time
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

def get_log_prob_unigram_autoregressive(prev_token_ids, full_token_ids, tgt_idx, lm):
    """Given a sequence of token ids, with one masked token, return the log probability of the masked token."""
    model = lm["model"]
    log_softmax = torch.nn.LogSoftmax(dim=0)
    with torch.no_grad():
        prev_token_ids = prev_token_ids.to(torch.long).to(args.device)
        full_token_ids = full_token_ids.to(torch.long).to(args.device)
        output = model(prev_token_ids)
        hidden_states = output[0].squeeze(0).to(torch.bfloat16)

        hs = hidden_states[-1]  # use logits for next word prediction
        target_id = full_token_ids[0][tgt_idx].long()
        log_probs = log_softmax(hs)[target_id]

        return log_probs

def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    tokenizer = lm["tokenizer"]
    uncased = lm["uncased"]

    sent1_token_ids, sent2_token_ids = data["sent_x"], data["sent_y"]

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked

    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0

    # pass to model one word at a time for autogressive models
    # start at 1 because BOS token is prepended
    for i in range(1, N):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()[:, :template1[i]].to(torch.long).to(args.device)
        sent2_masked_token_ids = sent2_token_ids.clone().detach()[:, :template1[i]].to(torch.long).to(args.device)
        total_masked_tokens += 1

        score1 = get_log_prob_unigram_autoregressive(sent1_masked_token_ids, sent1_token_ids,
                                                     template1[i], lm)
        score2 = get_log_prob_unigram_autoregressive(sent2_masked_token_ids, sent2_token_ids,
                                                     template2[i], lm)

        sent1_log_probs += score1.item()
        sent2_log_probs += score2.item()

    score = {}

    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs

    return score

def evaluate(args):
    short_model_name = args.model_name.split("/")[-1]
    adapter_model_name = f"data/results/{args.mode}/peft/{short_model_name}"

    # load models in dependence on mode
    if args.mode == 'lora':
        if os.getenv('CHECKPOINT'):
            checkpoint = str(os.getenv('CHECKPOINT', '1000'))
            adapter_model_name = f'data/results/{args.mode}/peft/{short_model_name}/checkpoint-{checkpoint}'
        # if adapter path is set, load adapter model from path and overwrite checkpoint path
        if args.adapter_path:
            adapter_model_name = args.adapter_path
        logger.info(f"Loading adapter model for LoRA from {adapter_model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
        peft_config = PeftConfig.from_pretrained(adapter_model_name + '/queernews')
        peft_config.init_lora_weights = False
        logger.info(f"Loading adapter model from config {peft_config}")
        # add adapter
        model.add_adapter(peft_config, 'queernews')
        # enable loaded adapter
        model.enable_adapters()
        # verify that adapter is loaded as errors can occur
        logger.info(model.active_adapters())

    elif args.mode == "soft-prompt":
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     torch_dtype=torch.bfloat16).to(args.device)
        logger.info(f"Loading soft prompts from: {adapter_model_name}")
        model = PeftModel.from_pretrained(model, adapter_model_name)
    elif args.mode == 'pretrained':
        logger.info(f"Loading model from {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     torch_dtype=torch.bfloat16,
                                                     token=args.token).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if hasattr(tokenizer, 'do_lower_case'):
        uncased = tokenizer.do_lower_case
    else:
        uncased = False
    model.eval()
    lm = {"model": model,
          "tokenizer": tokenizer,
          "log_softmax": torch.nn.LogSoftmax(dim=0),
          "uncased": uncased}

    # Load input file
    df_data = pd.read_csv(args.dataset_path)

    if args.reduce_dataset:
        logger.info(f"Reducing dataset to size {args.dataset_size}")
        df_data = df_data.groupby('Gender_ID_x', group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.dataset_size)))

    df_score = pd.DataFrame(columns=['sent_more', 'sent_less',
                                     'sent_more_score', 'sent_less_score',
                                     'score', 'bias_target_group'])
    total_pairs = 0
    stereo_score = 0

    # dict for keeping track of scores by category
    category_scores = {group: {'count': 0, 'score': 0, 'metric': None} for group in
                       df_data.Gender_ID_x.unique()}

    N = 0
    neutral = 0
    start = time.time()

    # tokenize dataset columns
    df_data['sent_x'] = df_data['sent_x'].apply(lambda x: tokenizer.encode(tokenizer.bos_token + x, return_tensors='pt', add_special_tokens=False).to(args.device))
    df_data['sent_y'] = df_data['sent_y'].apply(lambda x: tokenizer.encode(tokenizer.bos_token + x, return_tensors='pt', add_special_tokens=False).to(args.device))

    with tqdm(total=len(df_data.index)) as pbar:
        for index, data in df_data.iterrows():
            bias = data['Gender_ID_x']
            score = mask_unigram(data, lm)

            # round all scores to 3 places
            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            category_scores[bias]['count'] += 1
            pair_score = 0
            pbar.update(1)
            if score['sent1_score'] == score['sent2_score']:
                neutral += 1
            else:
                total_pairs += 1
                if score['sent1_score'] > score['sent2_score']:
                    stereo_score += 1
                    category_scores[bias]['score'] += 1
                    pair_score = 1

            sent_more = data['sent_x']
            sent_less = data['sent_y']
            sent_more_score = score['sent1_score']
            sent_less_score = score['sent2_score']

            # Sample data to append
            new_row = {
                'sent_more': sent_more,
                'sent_less': sent_less,
                'sent_more_score': sent_more_score,
                'sent_less_score': sent_less_score,
                'score': pair_score,
                'bias_target_group': bias
            }

            # Convert the new row to a DataFrame and use pd.concat to append it
            df_score = pd.concat([df_score, pd.DataFrame([new_row])], ignore_index=True)
    end = time.time()
    elapsed_time = end - start
    # add model name without company
    file_extension = args.model_name.split("/")[-1]
    # add information if gender or sexual identity
    file_extension += args.dataset_path.split("/")[-1].split("_")[1]
    # add dataset size
    file_extension += str(len(df_data.index))
    output_file = f"data/results/winoqueer/detailed_{file_extension}-{args.mode}.csv"
    summary_path = f"data/results/winoqueer/summary_{file_extension}-{args.mode}.csv"
    df_score.to_csv(output_file)



    # Assuming these variables are already defined
    # elapsed_time, N, neutral, stereo_score, category_scores, df_data, args, summary_path

    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write summary statistics
        writer.writerow(['Summary Statistics'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Elapsed time', elapsed_time])
        writer.writerow(['Total examples', N])
        writer.writerow(['Num. neutral', neutral])
        writer.writerow(['% neutral', round(neutral / N * 100, 2)])
        writer.writerow(['Winoqueer Overall Score', round(stereo_score / N * 100, 2)])
        writer.writerow([])

        # Write score breakdown by category
        writer.writerow(['Score Breakdown by Target of Bias'])
        writer.writerow(['Category', 'Number of Examples', 'Bias Score (%)'])
        for k, v in category_scores.items():
            if v['count'] > 0:
                v['metric'] = round(v['score'] / v['count'] * 100, 2)
            else:
                v['metric'] = 0
            writer.writerow([k, v['count'], v['metric']])
        writer.writerow([])

        # Write data for pasting into a spreadsheet
        writer.writerow(['Data for Spreadsheet'])
        gender_categories = df_data['Gender_ID_x'].unique()
        writer.writerow(['Order Overall'] + list(gender_categories))
        row = [round(stereo_score / N * 100, 2)]
        row.extend([category_scores[key]['metric'] if key in category_scores else 0 for key in
                    gender_categories])
        writer.writerow(row)

    logger.info('=' * 100)
    logger.info("Output written to: " + args.output_file)
    logger.info("summary stats written to: " + summary_path)
    logger.info(
        f"For pasting into spreadsheet (Order Overall, {', '.join(df_data['Gender_ID_x'].unique())}:\n")
    logger.info(str(row) + "\n")
    logger.info('=' * 100)


if __name__ == "__main__":
    class Args:
        dataset_path = os.getenv('DATASET_PATH')
        output_file = os.getenv('OUTPUT_FILE')
        model_name = os.getenv('MODEL_NAME', 'gpt2')
        mode = os.getenv('MODE')
        token = os.getenv('HF_TOKEN')
        dataset_size = int(os.getenv('DATASET_SIZE'))
        reduce_dataset = bool(int(os.getenv('REDUCE_DATASET', 0)))
        adapter_path = os.getenv('ADAPTER_PATH', None)
        @staticmethod
        def get_device() -> torch.device:
            """Utility function to get the available device."""
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        device = get_device()

    args = Args()

    # log settings
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Dataset Path: {args.dataset_path}")
    logger.info(f"Output File: {args.output_file}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Reduce dataset {args.reduce_dataset}")
    logger.info(f"Reduce to size {args.dataset_size}")


    if not args.dataset_path or not args.output_file or not args.mode:
        raise ValueError("Please set the DATASET_PATH, OUTPUT_FILE, and MODE environment variables.")
    if '.csv' not in args.dataset_path:
        files = os.listdir(args.dataset_path)
        file_path = args.dataset_path
        for f in tqdm(files):
            if f.startswith('winoqueer'):
                logger.info(f"Evaluating {f}")
                args.dataset_path = f'{file_path}{f}'
                evaluate(args)
            else:
                continue
    else:
        evaluate(args)
