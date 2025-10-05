
# sample_code/listen_lora.py
"""
Tiny LoRA on early attention + gating support.
"""
from typing import List
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    def get_peft_model(model, cfg): return model

def apply_listen_lora(model, target_modules: List[str], r=8, alpha=16, dropout=0.05):
    if LoraConfig is None:
        return model
    peft_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_cfg)

class LoraGate(nn.Module):
    def __init__(self, init_on: bool = True):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(1.0 if init_on else 0.0))

    def forward(self, delta: torch.Tensor, is_text_batch: bool):
        g = torch.sigmoid(self.gate)
        if is_text_batch:
            return delta * 0.0
        return delta * g
