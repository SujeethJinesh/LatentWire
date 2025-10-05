
# models.py (listen_lora variant)
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_utils import apply_listen_lora

class LMWrapper(nn.Module):
    def __init__(self, model_id, dtype=torch.float16, device='cuda',
                 listen_lora_enable=False, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 lora_target=("q_proj","k_proj","v_proj","o_proj"), lora_firstN=12):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device

        if listen_lora_enable:
            # apply LoRA broadly; for real firstN you'd filter module names
            self.model = apply_listen_lora(self.model, list(lora_target),
                                           r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def disable_adapters(self):
        # PEFT models expose disable_adapters(); if not, provide a no-op context manager
        from contextlib import contextmanager
        @contextmanager
        def _cm():
            try:
                yield from self.model.disable_adapters()
            except Exception:
                yield
        return _cm()

