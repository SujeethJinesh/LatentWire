
# models.py (latent coprocessor KV augmentation)
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_utils import LatentCoprocessor

class LMWrapper(nn.Module):
    def __init__(self, model_id, dtype=torch.float16, device='cuda',
                 use_coprocessor=True, d_z=256, heads_per_layer=None, width=256, kv_scale=0.8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device
        self.use_coprocessor = use_coprocessor
        cfg = self.model.config
        if heads_per_layer is None:
            heads_per_layer = [cfg.num_attention_heads] * cfg.num_hidden_layers
        self.copro = LatentCoprocessor(d_z, cfg.hidden_size, cfg.num_hidden_layers, heads_per_layer, width=width, kv_scale=kv_scale) if use_coprocessor else None

    def forward(self, input_ids=None, attention_mask=None, latent=None, use_cache=False, past_key_values=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache,
                         past_key_values=past_key_values, return_dict=True, **kw)
        return out

    def generate_step(self, input_ids, attention_mask=None, latent=None, past_key_values=None, **gen_kw):
        # one-step helper showing where to augment pkv
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values, return_dict=True)
        pkv = out.past_key_values
        if self.use_coprocessor and latent is not None:
            K_layers, V_layers = self.copro(latent)  # latent: [B,M,d_z]
            pkv = self.copro.augment_past_kv(pkv, K_layers, V_layers, [self.model.config.num_attention_heads]*self.model.config.num_hidden_layers)
        return out.logits, pkv

    def disable_adapters(self):
        from contextlib import contextmanager
        @contextmanager
        def _cm(): yield
        return _cm()
