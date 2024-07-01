import os

import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, \
    AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from datasets.fingerprint import Hasher

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


class LanguageModelTrainer:
    def __init__(self, model_name: str,
                 mode: str = "soft-prompt",
                 prompt_length: int = 10) -> None:
        self.model_name = model_name
        self.mode = mode
        self.prompt_length = prompt_length
        self.device = self.get_device()
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        gc.collect()

    @staticmethod
    def get_device() -> torch.device:
        """Utility function to get the available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load_model_and_tokenizer(self) -> tuple:
        """Loads the model and tokenizer based on the provided mode."""
        if self.mode == 'soft-prompt':
            logger.info(f"Loading Prompt Tuning model {self.model_name}...")
            model = PromptTuningModel(model_name=self.model_name, token=HF_TOKEN,
                                      num_soft_prompts=self.prompt_length, device=self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
            # Check if pad_token is already in the tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        else:
            logger.info(f"Loading pre-trained model {self.model_name}...")
            model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                         torch_dtype=torch.bfloat16,
                                                         token=HF_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
            # Check if pad_token is already in the tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.resize_token_embeddings(len(tokenizer))

        logger.warning(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.warning(f"Tokenizer special tokens: {tokenizer.all_special_tokens}")

        model = model.to(self.device)

        logger.warning(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.warning(f"Tokenizer special tokens: {tokenizer.all_special_tokens}")
        return model, tokenizer

    def tokenize_function(self, tokenizer, example):
        if 'sentence' in example:
            return tokenizer(example['sentence'], max_length=512, truncation=True,
                             padding='max_length')
        elif 'original' in example:
            return tokenizer(example['original'], max_length=512, truncation=True,
                             padding='max_length')

    Hasher.hash(tokenize_function)
    def load_data(self, dataset_path: str):
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
                test_eval = train_test['test'].train_test_split(test_size=.5)
                dataset = DatasetDict({
                    "train": train_test["train"],
                    "test": test_eval["test"],
                    "validation": test_eval["train"]})
                dataset.save_to_disk("data/datasets/queer_news.hf")
        else:
            dataset = load_dataset(dataset_path)
        logger.info("Dataset loaded. Now tokenizing...")
        #take small subset
        # Take small subset and apply the standalone tokenize function
        dataset = dataset.map(lambda x: self.tokenize_function(self.tokenizer, x), batched=True, num_proc=4)

        logger.info(
            f"Dataset loading complete. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")
        return dataset['train'], dataset['validation']

    def train(self, dataset_path: str, num_train_epochs: int = 3, batch_size: int = 8,
              learning_rate: float = 2e-5) -> None:
        """Trains the model based on the specified parameters."""
        train_dataset, eval_dataset = self.load_data(dataset_path)

        training_args = TrainingArguments(
            output_dir=f"data/results/{self.mode}",
            disable_tqdm=False,
            # overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            report_to=["tensorboard"]
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        if self.mode == 'soft-prompt':
            trainer = SoftPromptTrainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        logger.info("Training started...")
        start = time.time()
        trainer.train()
        logger.info("Training complete.")
        if self.mode == 'soft-prompt':
            soft_prompt_path = dataset_path.split("/")[-1].split(".")[0]
            self.model.save_prompt_embeddings(f"data/results/soft-prompt/{soft_prompt_path}_trained_prompt_embeddings.pt")
            logger.info("Prompt embeddings saved.")
        else:
            trainer.save_model('data/results/fine-tuning/models/')
            logger.info("Model saved.")

        logger.info("Evaluation started...")
        trainer.evaluate()
        end = time.time()
        elapsed_time = start - end
        with open(f"data/results/{self.mode}/summary-{self.model_name}-{end}.txt", 'w') as f:
            f.write(f"Time elapsed for training: {elapsed_time}")
            f.write(f"Model: {self.model_name}")
            f.write(f"Dataset: {dataset_path}")
        logger.info("Evaluation complete.")


    # def load_trained_prompts(self, prompt_path: str = "trained_prompt_embeddings.pt") -> None:
    #     """Loads the trained prompt embeddings."""
    #     if not os.path.exists(prompt_path):
    #         dataset_path = self.dataset_path.split("/")[-1].split(".")[0]
    #     prompt_path = f"data/results/soft-prompt/{dataset_path}_trained_prompt_embeddings.pt"
    #     self.model.load_prompt_embeddings(prompt_path)


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", 'google/gemma-7b')
    dataset_path = os.getenv("DATASET_PATH", 'data/datasets/queer_news.pkl')
    mode = os.getenv("MODE", 'soft-prompt')
    prompt_length = int(os.getenv("PROMPT_LENGTH", 10))
    batch_size = int(os.getenv("BATCH_SIZE", 8))
    learning_rate = float(os.getenv("LEARNING_RATE", 1e-5))
    epochs = int(os.getenv("EPOCHS", 3))


    # list settings
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Prompt Length: {prompt_length}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")



    lm_trainer = LanguageModelTrainer(model_name=model_name,
                                      mode=mode,
                                      prompt_length=prompt_length)

    lm_trainer.train(dataset_path=dataset_path,
                     num_train_epochs=epochs,
                     batch_size=batch_size,
                     learning_rate=learning_rate)

    # trainer.evaluate(dataset_path=dataset_path)
    # trainer.soft_prompt("Once upon a time")

# Example usage:
# trainer = LanguageModelTrainer(model_name="gpt2", mode="soft-prompt", prompt_length=10)
# trainer.train(dataset_path="path/to/dataset.csv")
# trainer.evaluate(dataset_path="path/to/dataset.csv")
# trainer.soft_prompt("Once upon a time")
