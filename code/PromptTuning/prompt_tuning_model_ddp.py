import os
import torch
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTuningModel(nn.Module):
    def __init__(self, model_name, token, num_soft_prompts, device):
        super(PromptTuningModel, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                          token=token).to(self.device)
        self.embedding_layer = self.model.get_input_embeddings()
        self.embedding_size = self.embedding_layer.embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_length = num_soft_prompts
        self.soft_prompts = nn.Parameter(self.initialize_embedding(self.embedding_layer, self.prompt_length)).to(self.device)
        logger.info(f"Soft prompts initialized with shape: {self.soft_prompts.size()}")

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        """Initializes learned embedding"""
        if initialize_from_vocab:
            return wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        batch_size = input_ids.size(0)
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
        logger.debug(f"Labels: {labels}")
        outputs = self.model.module(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    # def save_soft_prompts(self, file_path: str):
    #     torch.save(self.soft_prompts, file_path)
    #
    # def load_soft_prompts(self, file_path: str = None):
    #     if not file_path:
    #         dataset_path = 'queer_news'
    #         file_path = f'data/results/soft-prompt/{dataset_path}_trained_prompt_embeddings.pt'
    #     self.soft_prompts = torch.load(file_path).to(self.device)
    #     self.soft_prompts = nn.Parameter(self.soft_prompts)

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
            checkpoint = torch.load(load_path)
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

        self.optimizer = optim.AdamW([self.model.module.soft_prompts], lr=self.args.learning_rate)

        self.lr_scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )

    def save_model(self, output_dir, _internal_call=False):
        self.model.module.save_model(output_dir, _internal_call)
