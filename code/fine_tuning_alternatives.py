
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, \
    AutoModelForCausalLM, logging, GenerationConfig,  default_data_collator, get_linear_schedule_with_warmup, \
    BitsAndBytesConfig
from transformers.optimization import Adafactor, AdafactorSchedule
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, PrefixTuningConfig, AutoPeftModel
from datasets import load_dataset, Dataset, DatasetDict, enable_caching, load_from_disk
from dotenv import load_dotenv
from PromptTuning import PromptTuningModelDDP, SoftPromptTrainerDDP
from logging import getLogger
import time
import pickle
import gc
import torch.distributed as dist

logging.set_verbosity_info()

logger = getLogger(__name__)
logger.setLevel("INFO")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

enable_caching()


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dist.init_process_group(backend='nccl')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1']


@staticmethod
def get_device() -> torch.device:
    """Utility function to get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def remove_columns_from_dataset(dataset):
    columns_to_remove = [x for x in dataset['train'].column_names if x not in ['input_ids', 'attention_mask', 'sentence']]
    logger.info("columns that could possibly be removed: " + str(columns_to_remove))
    if columns_to_remove:
        try:
            logger.info("Removing columns: " + str(columns_to_remove))
            if columns_to_remove in dataset['train'].column_names:
                dataset['train'] = dataset['train'].remove_columns(columns_to_remove)
            if columns_to_remove in dataset['validation'].column_names:
                dataset['validation'] = dataset['validation'].remove_columns(columns_to_remove)
            if columns_to_remove in dataset['test'].column_names:
                dataset['test'] = dataset['test'].remove_columns(columns_to_remove)
            dataset.save_to_disk(f"data/datasets/tokenized_{short_model_name}_queer_news.hf")
        except Exception as e:
            logger.error(f"Error removing columns: {e}")
    return dataset



modes = ['soft-prompt',  'lora', 'prefix-tuning',]


model_name = os.getenv("MODEL_NAME", 'google/gemma-7b')
dataset_path = os.getenv("DATASET_PATH", 'data/datasets/queer_news.pkl')
mode = os.getenv("MODE", 'soft-prompt').lower()
prompt_length = int(os.getenv("PROMPT_LENGTH", 10))
batch_size = int(os.getenv("BATCH_SIZE", 8))
learning_rate = float(os.getenv("LEARNING_RATE", 1e-5))
epochs = int(os.getenv("EPOCHS", 3))
device = get_device()
deepspeed_path = os.getenv("DEEPSPEED_PATH", None)
num_gpu = int(os.getenv("NUM_GPU", 1))
gradient_accum_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 8))
adapter_path = str(os.getenv("ADAPTER_PATH", None))

local_rank = int(os.environ['LOCAL_RANK'])
# torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
logger.info(f"Local rank: {local_rank}")

# device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

dist.init_process_group(backend='nccl', rank=local_rank, world_size=num_gpu)
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{num_gpu}"
# os.environ["OMP_NUM_THREADS"] = "1"


# log all environment variables
for k, v in os.environ.items():
    logger.info(f"{k}: {v}")


gc.collect()

if mode == 'all':
    for m in modes:
        os.environ["MODE"] = m
        os.system("python code/main_dp.py")


def tokenize_function(example):
    if 'sentence' in example:
        return tokenizer(example['sentence'], max_length=512, truncation=True,
                         padding='max_length')
    elif 'original' in example:
        return tokenizer(example['original'], max_length=512, truncation=True,
                         padding='max_length')


"""Loads the model and tokenizer based on the provided mode."""
def define_peft_config(mode, model_name):
    if mode == 'soft-prompt':
        peft_config = PromptTuningConfig(
            peft_type= PeftType.PROMPT_TUNING,
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=prompt_length,
        )
    elif mode == 'prefix-tuning':
        peft_config = PrefixTuningConfig(
            peft_type= PeftType.PREFIX_TUNING,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=prompt_length
        )

    elif mode == 'lora':
        """
        task_type: the task to train for 
        inference_mode: whether using the model for inference or not
        r: the dimension of the low-rank matrices
        lora_alpha: the scaling factor for the low-rank matrices
        lora_dropout: the dropout probability of the LoRA layers
        
        settings from: https://huggingface.co/docs/peft/quicktour
        """
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            # r=16,
            r=8,
            # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0.1
        )

    return peft_config

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             # attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16)
#     .to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    logger.info("Setting pad token to eos token")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Resizing token embeddings to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

model.config.gradient_checkpointing = True
peft_config = define_peft_config(mode, model_name)
# model = AutoPeftModel.from_pretrained(model_name, config=peft_config, adapter_name='queernews', is_trainable=True)
model = get_peft_model(model, peft_config, adapter_name='queernews')
logger.info(model.print_trainable_parameters())
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)


"""Loads and splits the dataset into training and evaluation sets."""
logger.info(f"Loading dataset {dataset_path}")

if dataset_path.endswith(".pkl"):
    df = pickle.load(open(dataset_path, 'rb'))
    if 'news' in dataset_path.lower():
        dataset_pd = Dataset.from_pandas(df)
        train_test = dataset_pd.train_test_split(test_size=.1)
        test_eval = train_test['test'].train_test_split(test_size=.5)
        dataset =  DatasetDict({
                                "train": train_test["train"],
                                "test": test_eval["test"],
                                "validation": test_eval["train"]})
        dataset.save_to_disk("data/datasets/queer_news.hf")
elif dataset_path.endswith(".csv"):
    dataset = load_dataset("csv", data_files=dataset_path)
    # drop all sentences that are none
    logger.info(f"Dataset length: {len(dataset)}")
    columns_to_remove =[x for x in dataset['train'].column_names if x != 'sentence']
    if [x for x in columns_to_remove if x in dataset.column_names]:
        logger.info("Removing columns: " + str(columns_to_remove))
        dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.filter(lambda x: x['sentence'] is not None)
    logger.info(f"Dataset length: {len(dataset)} after removing Nones")
    dataset = dataset.filter(lambda x: len(x['sentence'].split(' ')) > 1)
    # give length of dataset
    logger.info(f"Dataset length: {len(dataset)} after filtering for len > 1 sentences")
    if not 'validation' in dataset.data:
        train_test = dataset['train'].train_test_split(test_size=.1)
        test_eval = train_test['test'].train_test_split(test_size=.5)
        dataset = DatasetDict({
            "train": train_test["train"],
            "test": test_eval["test"],
            "validation": test_eval["train"]})
        dataset.save_to_disk("data/datasets/queer_news.hf")
        logger.info(f"Dataset saved to data/datasets/queer_news.hf")
else:
    dataset = load_from_disk(dataset_path)
    logger.info(f"Column names: {dataset.column_names}")



short_model_name = model_name.split("/")[-1]
output_dir = f"data/results/{mode}/peft/{short_model_name}/no-eval/"
logger.info(f"Output directory: {output_dir}")

reduce_dataset = bool(int(os.getenv("REDUCE_DATASET", 1)))
reduced_size = int(os.getenv("REDUCE_DATASET_SIZE", 10000))
if reduce_dataset:
    logger.info(f"Reducing dataset size to {reduced_size}")
    if len (dataset['train']) > reduced_size:
        dataset['train'] = dataset['train'].select(range(reduced_size))
    if len (dataset['validation']) > reduced_size:
        dataset['validation'] = dataset['validation'].select(range(reduced_size))

    output_dir = f"data/results/{mode}/reduced-{reduced_size}/models"

os.makedirs(os.path.dirname(output_dir), exist_ok=True)
#
if 'tokenized' not in dataset_path:
    logger.info("Dataset loaded. Now tokenizing...")
    dataset = dataset.map(tokenize_function, batched=True)
    short_model_name = model_name.split("/")[-1].replace('.', '-')
    dataset = remove_columns_from_dataset(dataset)
    dataset.save_to_disk(f"data/datasets/tokenized_{short_model_name}_queer_news.hf")
    logger.info(f"Tokenization complete. Saved to {f'data/datasets/tokenized_{short_model_name}_queer_news.hf'}")


logger.info(
    f"Dataset loading complete. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")


train_dataset = dataset['train'].remove_columns([x for x in dataset['train'].column_names if x not in ['input_ids', 'attention_mask']])
eval_dataset = dataset['validation'].remove_columns([x for x in dataset['validation'].column_names if x not in ['input_ids', 'attention_mask']])

logger.info(f"Final column names {train_dataset.column_names}")




training_args = TrainingArguments(
    output_dir=output_dir,
    disable_tqdm=False,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accum_steps,
    # gradient_checkpointing=True, # not available with ddp
    # learning_rate=learning_rate,
    # prediction_loss_only=True,
    report_to=["tensorboard"],
    weight_decay=0.01,
    save_steps=5000,
    save_total_limit=2,
    dataloader_num_workers=num_gpu,
    ddp_find_unused_parameters=False,
    local_rank=local_rank,
    remove_unused_columns=False,
    bf16=True,
    # optim="adafactor"
)
if deepspeed_path:
    training_args.deepspeed = deepspeed_path

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# lr_scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=(len(train_dataloader) * epochs),
# )
optimizer = Adafactor(model.parameters(), scale_parameter=True,
                      warmup_init=True, lr=None, relative_step=True)
lr_scheduler = AdafactorSchedule(optimizer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(optimizer, lr_scheduler)
)

torch.cuda.empty_cache()

logger.info("Training started...")
start = time.time()
trainer.train()
logger.info("Training complete.")


trainer.save_model(output_dir)
logger.info(f"Model saved to {output_dir}")

end = time.time()
elapsed_time = end - start
end = time.strftime("%Y%m%d-%H%M%S")
model_name = model_name.replace("/", "-")[-1]
with open(f"data/results/{mode}/summary-{model_name}-{end}.txt", 'w') as f:
    f.write(f"Time elapsed for training: {elapsed_time}\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Prompt length: {prompt_length}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Epochs: {epochs}\n")
