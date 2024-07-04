import os

import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, \
    AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict, enable_caching
from dotenv import load_dotenv
from PromptTuning import PromptTuningModel, SoftPromptTrainer
from logging import getLogger
import time
import pickle
import gc

logger = getLogger(__name__)
logger.setLevel("INFO")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

enable_caching()

@staticmethod
def get_device() -> torch.device:
    """Utility function to get the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def tokenize_function(example):
    if 'sentence' in example:
        return tokenizer(example['sentence'], max_length=512, truncation=True,
                         padding='max_length')
    elif 'original' in example:
        return tokenizer(example['original'], max_length=512, truncation=True,
                         padding='max_length')

model_name = os.getenv("MODEL_NAME", 'google/gemma-7b')
dataset_path = os.getenv("DATASET_PATH", 'data/datasets/queer_news.pkl')
mode = os.getenv("MODE", 'soft-prompt')
prompt_length = int(os.getenv("PROMPT_LENGTH", 10))
batch_size = int(os.getenv("BATCH_SIZE", 8))
learning_rate = float(os.getenv("LEARNING_RATE", 1e-5))
epochs = int(os.getenv("EPOCHS", 3))
device = get_device()
deepspeed_path = os.getenv("DEEPSPEED_PATH", None)
gc.collect()



"""Loads the model and tokenizer based on the provided mode."""
if mode == 'soft-prompt':
    logger.info(f"Loading Prompt Tuning model {model_name}...")
    model = PromptTuningModel(model_name=model_name, token=HF_TOKEN,
                              num_soft_prompts=prompt_length, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    # Check if pad_token is already in the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

else:
    logger.info(f"Loading pre-trained model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    # Check if pad_token is already in the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))

logger.warning(f"Tokenizer vocab size: {tokenizer.vocab_size}")
logger.warning(f"Tokenizer special tokens: {tokenizer.all_special_tokens}")

model = model.to(device)

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
    dataset = dataset.filter(lambda x: x['sentence'] is not None)
    if not 'validation' in dataset.data:
        train_test = dataset['train'].train_test_split(test_size=.1)
        test_eval = train_test['test'].train_test_split(test_size=.9)
        dataset = DatasetDict({
            "train": train_test["train"],
            "test": test_eval["test"],
            "validation": test_eval["train"]})
        dataset.save_to_disk("data/datasets/queer_news.hf")
else:
    dataset = load_dataset(dataset_path)

if bool(int(os.getenv("REDUCE_DATASET", 1))):
    reduced_size = int(os.getenv("REDUCE_DATASET_SIZE", 10000))
    logger.info(f"Reducing dataset size to {reduced_size}")
    dataset['train'] = dataset['train'].select(range(reduced_size))
    dataset['validation'] = dataset['validation'].select(range(reduced_size))

logger.info("Dataset loaded. Now tokenizing...")
#take small subset
dataset = dataset.map(tokenize_function, batched=True)

logger.info(
    f"Dataset loading complete. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

if deepspeed_path is not None:
    training_args = TrainingArguments(
        output_dir=f"data/results/{mode}",
        eval_strategy="epoch",
        disable_tqdm=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        prediction_loss_only=True,
        report_to=["tensorboard"],
        weight_decay=0.01,
        deepspeed=deepspeed_path
    )
else:
    training_args = TrainingArguments(
        output_dir=f"data/results/{mode}",
        eval_strategy="epoch",
        disable_tqdm=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        prediction_loss_only=True,
        report_to=["tensorboard"],
        weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
if mode == 'soft-prompt':
    trainer = SoftPromptTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
logger.info("Training started...")
start = time.time()
trainer.train()
logger.info("Training complete.")


trainer.save_model('data/results/fine-tuning/models/')

logger.info("Evaluation started...")
trainer.evaluate()
end = time.time()
elapsed_time = start - end
end = time.strftime("%Y%m%d-%H%M%S")
model_name = model_name.replace("/", "-")[-1]
with open(f"data/results/{mode}/summary-{model_name}-{end}.txt", 'w') as f:
    f.write(f"Time elapsed for training: {elapsed_time}")
    f.write(f"Model: {model_name}")
    f.write(f"Dataset: {dataset_path}")
    f.write(f"Prompt length: {prompt_length}")
    f.write(f"Batch size: {batch_size}")
    f.write(f"Learning rate: {learning_rate}")
    f.write(f"Epochs: {epochs}")
    f.write(f"Device: {device}")
