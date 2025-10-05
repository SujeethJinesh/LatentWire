
# models.py (CCoT): ThoughtHead to produce latent thoughts, then consume them
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ThoughtHead(nn.Module):
    def __init__(self, d_model: int, k: int, d_thought: int):
        super().__init__()
        self.k = k
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_thought * k))
    def forward(self, final_hidden: torch.Tensor) -> torch.Tensor:
        out = self.proj(final_hidden)  # [B, k*d_thought]
        return out.view(final_hidden.size(0), self.k, -1)

class LMWrapper(nn.Module):
    def __init__(self, model_id, dtype=torch.float16, device='cuda',
                 use_ccot=True, k_latent=6, d_thought=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device
        self.use_ccot = use_ccot
        self.thought_head = ThoughtHead(self.model.config.hidden_size, k_latent, d_thought) if use_ccot else None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, return_dict=True, labels=labels, **kw)
        return out

    def disable_adapters(self):
        from contextlib import contextmanager
        @contextmanager
        def _cm(): yield
        return _cm()
