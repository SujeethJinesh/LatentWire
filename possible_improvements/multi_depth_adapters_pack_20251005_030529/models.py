
# models.py (multi-depth adapters)
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_utils import LatentAdapterBlock

class LMWrapper(nn.Module):
    def __init__(self, model_id, dtype=torch.float16, device='cuda',
                 use_adapters=True, adapter_layers=(5,10,15), d_z=256, n_heads=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device
        self.use_adapters = use_adapters
        self.adapter_layers = adapter_layers
        if use_adapters:
            self.adapters = nn.ModuleDict({str(l): LatentAdapterBlock(self.model.config.hidden_size, d_z, n_heads) for l in adapter_layers})
        else:
            self.adapters = nn.ModuleDict()

    def forward(self, input_ids=None, attention_mask=None, latent=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kw)
        if self.use_adapters and latent is not None:
            hs = list(out.hidden_states)  # tuple per layer
            for l in self.adapter_layers:
                h = hs[l]  # [B,T,D]
                h2 = self.adapters[str(l)](h, latent)  # [B,T,D]
                # simple residual injection into last hidden
                hs[l] = h2
            # recompute logits from modified final representation (approx via LM head)
            last = hs[-1]
            logits = self.model.lm_head(last)
        else:
            logits = out.logits
        out.logits = logits
        return out

    def disable_adapters(self):
        from contextlib import contextmanager
        @contextmanager
        def _cm():
            old = self.use_adapters
            self.use_adapters = False
            try:
                yield
            finally:
                self.use_adapters = old
        return _cm()
