from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, PrefixTuningConfig, AutoPeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, \
    AutoModelForCausalLM, logging, GenerationConfig,  default_data_collator, get_linear_schedule_with_warmup, \
    BitsAndBytesConfig
import torch
import json

from logging import getLogger
logging.set_verbosity_info()

logger = getLogger(__name__)
logger.setLevel("INFO")

def define_peft_config(mode, model_name):
    if mode == 'soft-prompt':
        peft_config = PromptTuningConfig(
            peft_type=PeftType.PROMPT_TUNING,
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=10,
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


modes = ['lora', 'soft-prompt']
models = ['google/gemma-7b', 'meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3']

result = {}
for model_name in models:

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 # attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16)
    for mode in modes:
        logger.info(f"Model: {model_name}, Mode: {mode}")
        peft_config = define_peft_config(mode, model_name)
        # model = AutoPeftModel.from_pretrained(model_name, config=peft_config, adapter_name='queernews', is_trainable=True)
        model = get_peft_model(model, peft_config, adapter_name='queernews')
        logger.info(f"{model} - {mode} - {model.print_trainable_parameters()}")
        result[f"{model_name}_{mode}"] = model.print_trainable_parameters()

with open('trainable_params_all_modes_mode.json', 'w') as f:
    json.dump(result, f, indent=4)