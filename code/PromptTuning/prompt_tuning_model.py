import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTuningModel(nn.Module):
    def __init__(self, model_name, token, num_soft_prompts, device):
        super(PromptTuningModel, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_auth_token=token).to(device)
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_size = self.embedding_layer.embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_soft_prompts = num_soft_prompts
        self.soft_prompts = nn.Parameter(self.initialize_embedding(self.embedding_layer, num_soft_prompts)).to(device)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        """Initializes learned embedding"""
        if initialize_from_vocab:
            return wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Debugging: log shapes
        logger.warning(f"input_ids shape: {input_ids.shape}")

        # Check the range of input_ids
        logger.warning(f"input_ids max value: {input_ids.max().item()}, input_ids min value: {input_ids.min().item()}")
        vocab_size = self.embedding_layer.num_embeddings
        logger.warning(f"Vocab size: {vocab_size}")

        if input_ids.max().item() >= vocab_size or input_ids.min().item() < 0:
            raise ValueError(f"input_ids contain indices outside the valid range (0, {vocab_size-1})")

        input_embeddings = self.embedding_layer(input_ids)
        logger.warning(f"input_embeddings shape: {input_embeddings.shape}")

        batch_size = input_embeddings.size(0)
        seq_length = input_embeddings.size(1)
        # Expand prompt embeddings to match the batch size
        expanded_prompt_embeddings = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        logger.warning(f"expanded_prompt_embeddings shape: {expanded_prompt_embeddings.shape}")

        # Concatenate prompt embeddings with input embeddings
        extended_input_embeddings = torch.cat([expanded_prompt_embeddings, input_embeddings], dim=1)
        logger.warning(f"extended_input_embeddings shape: {extended_input_embeddings.shape}")

        # Adjust attention mask to account for prompt tokens
        if attention_mask is not None:
            extended_attention_mask = torch.cat([torch.ones(batch_size, self.num_soft_prompts, device=input_ids.device), attention_mask], dim=1)
        else:
            extended_attention_mask = None
        logger.warning(f"extended_attention_mask shape: {extended_attention_mask.shape if extended_attention_mask is not None else 'None'}")

        # Pass through the model
        outputs = self.model(inputs_embeds=extended_input_embeddings, attention_mask=extended_attention_mask, labels=labels)
        return outputs

    def save_soft_prompts(self, file_path: str):
        torch.save(self.soft_prompts, file_path)

    def load_soft_prompts(self, file_path: str):
        self.soft_prompts = torch.load(file_path).to(self.device)
        self.soft_prompts = nn.Parameter(self.soft_prompts)
