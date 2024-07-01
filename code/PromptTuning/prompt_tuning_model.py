import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.optimization import AdamW
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
        self.prompt_length = num_soft_prompts
        self.soft_prompts = nn.Parameter(self.initialize_embedding(self.embedding_layer, self.prompt_length)).to(device)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        """Initializes learned embedding"""
        if initialize_from_vocab:
            return wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, input_ids, attention_mask=None, labels=None):
        prompt_ids = torch.arange(self.prompt_length, device=input_ids.device).unsqueeze(
            0).expand(input_ids.size(0), -1)
        prompt_embedding = self.soft_prompts[prompt_ids]
        inputs_embeds = self.embedding_layer(input_ids)
        inputs_embeds = torch.cat((prompt_embedding, inputs_embeds), dim=1)

        attention_mask = torch.cat((torch.ones(prompt_embedding.size()[:-1],
                                               device=attention_mask.device), attention_mask),
                                   dim=1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs

    def save_soft_prompts(self, file_path: str):
        torch.save(self.soft_prompts, file_path)

    def load_soft_prompts(self, file_path: str = None):
        if not file_path:
            dataset_path = 'queer_news'
            file_path = f'data/results/soft-prompt/{dataset_path}_trained_prompt_embeddings.pt'
        self.soft_prompts = torch.load(file_path).to(self.device)
        self.soft_prompts = nn.Parameter(self.soft_prompts)


class SoftPromptTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Create an optimizer for only the prompt embeddings
        self.optimizer = AdamW([self.model.soft_prompts], lr=self.args.learning_rate)

        self.lr_scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer
        )
