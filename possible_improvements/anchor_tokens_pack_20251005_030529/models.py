
# models.py (anchor tokens): model wrapper unchanged; anchors applied via masks/kv operations externally if needed
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LMWrapper(nn.Module):
    def __init__(self, model_id, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device

    def forward(self, **kw):
        return self.model(**kw)

    def disable_adapters(self):
        from contextlib import contextmanager
        @contextmanager
        def _cm(): yield
        return _cm()
