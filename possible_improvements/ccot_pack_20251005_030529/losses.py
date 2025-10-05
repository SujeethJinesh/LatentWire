
# losses.py (drop-in)
# - First-token CE, K-token CE
# - KD first-K with temperature and PEFT adapter disabling for clean teacher

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

def first_token_ce(logits, target_ids):
    # logits: [B,T,V], target_ids: [B,T]
    first_logits = logits[:, 0, :]  # assume first step aligned
    first_targets = target_ids[:, 0]
    return F.cross_entropy(first_logits, first_targets, ignore_index=-100)

def k_token_ce(logits, target_ids, K=8):
    B,T,V = logits.shape
    K = min(K, T)
    loss = 0.0
    count = 0
    for t in range(K):
        step_logits = logits[:, t, :]
        step_targets = target_ids[:, t]
        loss += F.cross_entropy(step_logits, step_targets, ignore_index=-100)
        count += 1
    return loss / max(1, count)

def kd_first_k(student_model, teacher_model, input_ids, attention_mask, gold_ids, K=8, tau=2.0):
    """
    Compute KL(teacher || student) over first K steps. Teacher is run with adapters disabled if available.
    """
    with torch.no_grad():
        # Create teacher inputs per step (teacher-forced with gold so far)
        B,T = gold_ids.shape
        K = min(K, T)
        # Use no_peft context if available
        disable = getattr(teacher_model, "disable_adapters", None)
        ctx = disable() if callable(disable) else nullcontext()
        with ctx:
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask,
                                           use_cache=False, return_dict=True).logits[:, :K, :]

    student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask,
                                   use_cache=False, return_dict=True).logits[:, :K, :]

    t_probs = F.softmax(teacher_logits / tau, dim=-1)
    s_logp = F.log_softmax(student_logits / tau, dim=-1)
    kd = F.kl_div(s_logp, t_probs, reduction="batchmean") * (tau * tau)
    return kd
