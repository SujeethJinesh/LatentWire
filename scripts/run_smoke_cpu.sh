#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# CPU-friendly tiny models for plumbing checks
python - <<'PY'
import torch
from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer

# Tiny HF models for CPU smoke test (safetensors-backed)
mA = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
mB = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

A = LMWrapper(LMConfig(model_id=mA, dtype=dtype))
B = LMWrapper(LMConfig(model_id=mB, dtype=dtype))

enc = InterlinguaEncoder(d_z=128, latent_len=4).to(device)
adpA = Adapter(d_z=128, d_model=A.d_model).to(device)
adpB = Adapter(d_z=128, d_model=B.d_model).to(device)

bt = ByteTokenizer(max_bytes=256)
texts = [
    "Question: What is 2+2?\nContext: Arithmetic basics.\nAnswer:",
    "Question: The capital of France?\nContext: Europe geography.\nAnswer:",
]
answers = ["4", "Paris"]

# Build inputs
import torch
ids = [bt.encode(t) for t in texts]
maxT = max(x.numel() for x in ids)
batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.numel(), dtype=torch.long)], 0) for x in ids], 0).to(device)

with torch.no_grad():
    Z = enc(batch)                             # [B, M, d_z]
    pA = adpA(Z)                               # [B, M, d_model_A]
    pB = adpB(Z)                               # [B, M, d_model_B]
    outA = A.generate_from_prefix(pA, max_new_tokens=16)
    outB = B.generate_from_prefix(pB, max_new_tokens=16)

print("A:", [A.tokenizer.decode(o, skip_special_tokens=True) for o in outA])
print("B:", [B.tokenizer.decode(o, skip_special_tokens=True) for o in outB])
print("✅ Smoke test done.")
PY
