import os
import torch
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, GenerationConfig
from transformers.optimization import Adafactor, AdafactorSchedule
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTuningModel(nn.Module):
    def __init__(self, model_name, token, num_soft_prompts, device):
        super(PromptTuningModel, self).__init__()
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.generation_config = GenerationConfig.from_pretrained(model_name, token=token,
                                                                  use_cache=True)
        # Check if pad_token is already in the self.tokenizer
        if self.tokenizer.pad_token is None:
            pad_token_id = getattr(self.generation_config, 'pad_token_id', self.tokenizer.eos_token_id)
            logger.info(f"Pad token id: {pad_token_id}")
            if pad_token_id is not None:
                pad_token = self.tokenizer.convert_ids_to_tokens([pad_token_id])[0]
                self.tokenizer.pad_token_id = pad_token_id
                self.tokenizer.pad_token = pad_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Pad token: {self.tokenizer.pad_token}, Pad token id: {self.tokenizer.pad_token_id}")


        num_devices = torch.cuda.device_count()
        if num_devices == 1:
            self.device = torch.device('cuda:0')
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                              token=token).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                          token=token).to(self.device)
        if model_name != 'google/gemma-7b':
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_size = self.embedding_layer.embedding_dim


        self.prompt_length = num_soft_prompts
        self.soft_prompts = nn.Parameter(self.initialize_embedding(self.embedding_layer, self.prompt_length)).to(self.device)
        logger.info(f"Soft prompts initialized with shape: {self.soft_prompts.size()}")


        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        """Initializes learned embedding"""
        if initialize_from_vocab:
            logger.info(f"Initializing learned embedding from vocab")
            return wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.model_name == 'google/gemma-7b':
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            batch_size = input_ids.size(0)
            logger.debug(f"Batch size: {batch_size}, Prompt length: {self.prompt_length}")

            prompt_ids = torch.arange(self.prompt_length, device=input_ids.device).unsqueeze(
                0).expand(input_ids.size(0), -1)
            prompt_embedding = self.soft_prompts[prompt_ids]
            inputs_embeds = self.embedding_layer(input_ids)
            inputs_embeds = torch.cat((prompt_embedding, inputs_embeds), dim=1)

            if attention_mask is not None:
                prompt_attention_mask = torch.ones((batch_size, self.prompt_length),
                                                   device=input_ids.device)
                attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

            if labels is not None:
                prompt_labels = torch.full((batch_size, self.prompt_length), -100, dtype=torch.long,
                                           device=input_ids.device)
                labels = torch.cat((prompt_labels, labels), dim=1)
                logger.debug(f"Concatenated labels shape: {labels.shape}")

            # print types of each input
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 labels=labels)
                return outputs
        else:
            # Move inputs to the device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            batch_size = input_ids.size(0)

            # Check the dimensions of soft_prompts
            soft_prompts_size = self.soft_prompts.size(0)

            # Create prompt embeddings
            prompt_ids = torch.arange(self.prompt_length,
                                      device=input_ids.device).unsqueeze(0).expand(
                batch_size, -1)

            # Check the maximum index of prompt_ids
            max_prompt_id = torch.max(prompt_ids).item()
            if max_prompt_id >= soft_prompts_size:
                raise IndexError(
                    f"Max prompt id {max_prompt_id} is out of bounds for soft_prompts with size {soft_prompts_size}")

            prompt_embedding = self.soft_prompts[prompt_ids]

            inputs_embeds = self.embedding_layer(input_ids)

            inputs_embeds = torch.cat((prompt_embedding, inputs_embeds), dim=1)

            # Adjust attention mask if provided
            if attention_mask is not None:
                prompt_attention_mask = torch.ones((batch_size, self.prompt_length),
                                                   device=input_ids.device)

                attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

            # Adjust labels if provided
            if labels is not None:
                prompt_labels = torch.full((batch_size, self.prompt_length), -100,
                                           dtype=torch.long, device=input_ids.device)

                labels = torch.cat((prompt_labels, labels), dim=1)

            # Model forward pass
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 labels=labels)
            return outputs


    def save_model(self, save_path, _internal_call=False):
        """Save the model with special handling for shared weights"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.endswith('.pt'):
            model_name = self.model_name.split('/')[-1]
            save_path = os.path.join(save_path, f'{model_name}.pt')
        model_to_save = {
            'model_state_dict': self.model.state_dict(),
            'soft_prompts': self.soft_prompts.data
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        torch.save(model_to_save, save_path)

    @classmethod
    def load_model(cls, load_path, model_name, token, num_soft_prompts, device):
            """Load the model with special handling for shared weights"""
            logger.info(f"Loading model from {load_path}")
            checkpoint = torch.load(load_path, map_location='cuda:0')
            model = cls(model_name, token, num_soft_prompts, device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.soft_prompts.data = checkpoint['soft_prompts']
            # Ensure shared weights are correctly referenced
            # model.model.get_input_embeddings().weight = model.model.lm_head.weight = model.model.model.embed_tokens.weight
            return model





class SoftPromptTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Create an optimizer for only the prompt embeddings
        self.optimizer = Adafactor([self.model.module.soft_prompts], scale_parameter=True,
                                   warmup_init=True, lr=None,
                                   relative_step=True)

        self.lr_scheduler = AdafactorSchedule(self.optimizer)

    def save_model(self, output_dir, _internal_call=False):
        self.model.module.save_model(output_dir, _internal_call)
