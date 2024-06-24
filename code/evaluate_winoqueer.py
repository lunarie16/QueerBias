import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from dotenv import load_dotenv
from PromptTuning import PromptTuningModel
import difflib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

def get_log_prob_unigram_autoregressive(prev_token_ids, full_token_ids, tgt_idx, lm):
    """Given a sequence of token ids, with one masked token, return the log probability of the masked token."""
    model = lm["model"]
    log_softmax = torch.nn.LogSoftmax(dim=0)

    output = model(prev_token_ids)
    hidden_states = output[0].squeeze(0)

    hs = hidden_states[-1]  # use logits for next word prediction
    target_id = full_token_ids[0][tgt_idx]
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
        # each op is a list of tuple:
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
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
    model = lm["model"]

    sent1, sent2 = data["sent_x"], data["sent_y"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    # append BOS token for conditional generation
    sent1_token_ids = tokenizer.encode(tokenizer.bos_token + sent1, return_tensors='pt',
                                       add_special_tokens=False).to(args.device)
    sent2_token_ids = tokenizer.encode(tokenizer.bos_token + sent2, return_tensors='pt',
                                       add_special_tokens=False).to(args.device)

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked

    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0

    # pass to model one word at a time for autogressive models
    # start at 1 because BOS token is prepended
    for i in range(1, N):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()[:, :template1[i]]
        sent2_masked_token_ids = sent2_token_ids.clone().detach()[:, :template1[i]]
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
    # Load model and tokenizer based on mode
    if args.mode == "soft-prompt":
        trainer = PromptTuningModel(model_name=args.model_name, token=args.token,
                                    prompt_length=args.prompt_length, device=args.device)
        trainer.load_trained_prompts()
        tokenizer = trainer.tokenizer
    else:
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
    elapsed_time = start - end
    file_extension = args.model_name.split("/")[-1]
    file_extension += args.dataset_path.split("/")[-1].split("_")[1]
    file_extension += str(len(df_data.index))
    output_file = f"data/results/winoqueer/detailed_{file_extension}-{args.mode}.csv"
    summary_path = f"data/results/winoqueer/summary_{file_extension}-{args.mode}.csv"
    df_score.to_csv(output_file)

    with open(summary_path, 'w') as f:
        f.wirte(f'Elapsed time: {elapsed_time}')
        f.write('Total examples: ' + str(N) + '\n')
        logger.info('Total examples: ' + str(N))
        f.write("Num. neutral:" + str(neutral) + ", % neutral: " + str(
            round(neutral / N * 100, 2)) + '\n')
        logger.info("Num. neutral:" + str(neutral) + ", % neutral: " + str(
            round(neutral / N * 100, 2)))
        f.write('Winoqueer Overall Score: ' + str(round(stereo_score / N * 100, 2)) + '\n')
        logger.info('Winoqueer Overall Score: ' + str(round(stereo_score / N * 100, 2)))
        f.write('Score Breakdown by Target of Bias:\n')
        logger.info('Score Breakdown by Target of Bias:')
        for k, v in category_scores.items():
            f.write("Category: " + k + '\n')
            logger.info("Category: " + k)
            f.write("    Number of examples: " + str(v['count']) + '\n')
            logger.info("    Number of examples: " + str(v['count']))
            if v['count'] > 0:
                v['metric'] = round(v['score'] / v['count'] * 100, 2)
                f.write("    Bias score against group " + k + ": " + str(v['metric']) + '\n')
                logger.info("    Bias score against group " + k + ": " + str(v['metric']))

        f.write(
            f"For pasting into spreadsheet (Order Overall, {', '.join(df_data['Gender_ID_x'].unique())}):")
        logger.info(
            f"For pasting into spreadsheet (Order Overall, {', '.join(df_data['Gender_ID_x'].unique())}):")
        # use list of keys instead of category_scores.items() to force order to match the spreadsheet
        f.write(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join(
            [str(category_scores[key]['metric']) for key in
             df_data['Gender_ID_x'].unique()]))
        logger.info(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join(
            [str(category_scores[key]['metric']) for key in
             df_data['Gender_ID_x'].unique()]))

    logger.info('=' * 100)
    logger.info("Output written to: " + args.output_file)
    logger.info("summary stats written to: " + summary_path)
    logger.info(
        f"For pasting into spreadsheet (Order Overall, {', '.join(df_data['Gender_ID_x'].unique())}:\n")
    # use list of keys instead of category_scores.items() to force order to match the spreadsheet
    logger.info(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join(
        [str(category_scores[key]['metric']) for key in df_data['Gender_ID_x'].unique()]) + "\n")
    logger.info('=' * 100)

if __name__ == "__main__":
    class Args:
        dataset_path = os.getenv('DATASET_PATH')
        output_file = os.getenv('OUTPUT_FILE')
        lm_model_path = os.getenv('LM_MODEL_PATH')
        model_name = os.getenv('MODEL_NAME', 'gpt2')
        prompt_length = int(os.getenv('PROMPT_LENGTH', 10))
        mode = os.getenv('MODE')
        token = os.getenv('HF_TOKEN')
        dataset_size = int(os.getenv('DATASET_SIZE'))
        reduce_dataset = bool(int(os.getenv('REDUCE_DATASET', 0)))
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
    logger.info(f"Prompt Length: {args.prompt_length}")
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
            logger.info(f"Evaluating {f}")
            args.dataset_path = f'{file_path}{f}'
            evaluate(args)
    else:
        evaluate(args)
