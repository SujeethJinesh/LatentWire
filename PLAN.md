# PLAN_v2.md — Pure Latent Prompting Under Honest Compression

### Scope (what this plan is):

A clean, task‑agnostic evaluation of whether learned continuous latents can stand in for text prompts under real byte pressure.

### What this plan is not:

No span heads, no pointer indices, no micro‑prompts, no answer‑position hints. Those are moved to an optional hybrid track (Appendix H) with explicit disclaimers.

---

## Success Criteria & Decision Gates

### Primary bar (pure latent):

- **FirstTok@1:** 12–20% at M∈{64,48,32} (from ~2–5%).
- **F1 (SQuAD‑style):** 0.10–0.20 at M=64 with fp16 latents; expect lower at M=32.
- **Compression honesty:** measured wire bytes for latents strictly ≤ measured UTF‑8 bytes of the text prompt for at least one configuration (after quantization), without sending any task‑specific side info.

### Go / No‑Go:

If, after implementing A0–A2 and tuning within the grid below, F1 < 0.10 and FirstTok@1 < 12% at M=64, log the result as an empirical limit for pure latent prompting on extractive QA and consider pivoting tasks/approaches (see §Pivot Options).

---

## Phase A — Core, Task‑Agnostic Improvements

_The only phase required to claim “pure latent prompting.”_

### A0) Tokenization & Alignment Sanity Checks (do first)

- Verify exact `t=0` alignment after the prompt anchor (e.g., "Answer: ").
- Ensure the same anchor text is used in train & eval.
- Confirm `append_bos_after_prefix` matches pretraining expectations.

```python
# tools/sanity_checks.py
import torch
from transformers import AutoTokenizer

def assert_t0_alignment(tokenizer_name: str, answer_prefix="Answer: "):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    examples = [
        ("Q: Who wrote Hamlet?", "C: Shakespeare wrote many plays.", "Shakespeare"),
        ("Q: Capital of France?", "C: Paris is the capital.", "Paris"),
    ]
    for q, c, gold in examples:
        prompt = f"{c}\n\n{q}\n{answer_prefix}"
        ids_all = tok(prompt + gold, add_special_tokens=False).input_ids
        ids_pref = tok(prompt, add_special_tokens=False).input_ids
        ids_gold = tok(gold, add_special_tokens=False).input_ids
        assert ids_all[len(ids_pref)] == ids_gold[0], (
            "t=0 mismatch",
            tok.convert_ids_to_tokens([ids_all[len(ids_pref)], ids_gold[0]])
        )
    print(f"[OK] t=0 alignment for {tokenizer_name}")

if __name__ == "__main__":
    assert_t0_alignment("meta-llama/Meta-Llama-3.1-8B-Instruct")
A1) K‑token teacher‑forced cross‑entropy (K=4)
One‑token supervision is too weak; supervise t=0…K−1.

Python

# losses/k_token_ce.py
import torch
import torch.nn.functional as F

def k_token_ce(student_llm, prefix_embeds, scaffold_ids, gold_ids, K: int = 4):
    """
    student_llm: LLM called with (prefix_embeds, input_ids, attention_mask)
    prefix_embeds: [B, P, d_model] projected latent prefix
    scaffold_ids: [B, T] text prompt up to and including the anchor (context+question+anchor)
    gold_ids:     [B, A] gold answer token ids
    """
    B, A = gold_ids.shape
    loss = 0.0
    T = scaffold_ids.size(1)
    base_mask = scaffold_ids.new_ones(B, T)
    for t in range(min(K, A)):
        step_ids = torch.cat([scaffold_ids, gold_ids[:, :t+1]], dim=1)          # [B, T+t+1]
        step_mask = torch.cat([base_mask, base_mask.new_ones(B, t+1)], dim=1)   # [B, T+t+1]
        logits = student_llm(prefix_embeds=prefix_embeds,
                             input_ids=step_ids,
                             attention_mask=step_mask,
                             use_cache=False).logits
        loss = loss + F.cross_entropy(logits[:, -1, :], gold_ids[:, t])
    return loss / float(min(K, A))
A2) Prefix KD vs. text‑prompted teacher (t=0…K−1)
Match the distribution the base model would produce with the real text prefix.

Python

# losses/kd_first_k.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _teacher_step_logits(teacher_llm, scaffold_ids, gold_ids, t):
    ids_t = torch.cat([scaffold_ids, gold_ids[:, :t+1]], dim=1)
    mask_t = ids_t.new_ones(ids_t.shape)
    return teacher_llm(input_ids=ids_t, attention_mask=mask_t, use_cache=False).logits[:, -1, :]

def kd_first_k(student_llm, teacher_llm, prefix_embeds, scaffold_ids, gold_ids, K=4, tau=1.0):
    loss = 0.0
    A = gold_ids.size(1)
    for t in range(min(K, A)):
        T_logits = _teacher_step_logits(teacher_llm, scaffold_ids, gold_ids, t)     # [B, V]
        ids_t = torch.cat([scaffold_ids, gold_ids[:, :t+1]], dim=1)
        mask_t = ids_t.new_ones(ids_t.shape)
        S_logits = student_llm(prefix_embeds=prefix_embeds,
                               input_ids=ids_t,
                               attention_mask=mask_t,
                               use_cache=False).logits[:, -1, :]                   # [B, V]
        T = F.softmax(T_logits / tau, dim=-1)
        S_log = F.log_softmax(S_logits / tau, dim=-1)
        loss = loss + F.kl_div(S_log, T, reduction="batchmean") * (tau * tau)
    return loss / float(min(K, A))
A3) Thin, task‑agnostic prefix adapter (capacity > scalar RMS)
Keep it generic; small MLP over the shared latent to produce model‑specific prefix embeddings.

Python

# adapters/prefix_adapter.py
import torch.nn as nn

class PrefixAdapter(nn.Module):
    def __init__(self, d_latent: int, d_model: int, P: int, hidden_mult: int = 4):
        super().__init__()
        self.P = P
        self.token_proj = nn.Linear(d_latent, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_mult * d_model),
            nn.SiLU(),
            nn.Linear(hidden_mult * d_model, d_model),
        )
        self.gamma = nn.Parameter(nn.init.ones_(nn.Parameter(torch.zeros(d_model))).data + 0.0)  # γ init to 1
        self.beta  = nn.Parameter(torch.zeros(d_model))                                          # β init to 0

    def forward(self, Z):  # Z: [B, P, d_latent] (slice/reshuffle from shared latent if needed)
        H = self.token_proj(Z)
        H = H + self.mlp(H)
        return H * self.gamma + self.beta
A4) Decode nudges (first token only; model‑agnostic)
first_token_temperature: 0.7

first_token_top_p: 0.95

eos_ban_steps: 4

(Apply only at t=0 to avoid changing the task; this is a decoding prior, not a task hack.)

A5) Training loop integration
Python

# train_step.py (sketch)
loss = 0.0
input_ids_scaffold, gold_answer_ids = batch["scaffold_ids"], batch["gold_ids"]  # [B,T], [B,A]
Z_shared = encoder(batch["context_feats"])                                       # [B, M, d_latent]
prefix_embeds = prefix_adapter(Z_shared[:, :P, :])                               # [B, P, d_model]

loss += cfg.lambda_ce * k_token_ce(student_llm, prefix_embeds, input_ids_scaffold, gold_answer_ids, K=cfg.K)
loss += cfg.lambda_kd * kd_first_k(student_llm, teacher_llm, prefix_embeds, input_ids_scaffold, gold_answer_ids, K=cfg.K, tau=cfg.kd_tau)

loss.backward()
optimizer.step(); optimizer.zero_grad(set_to_none=True)
Phase B — Compression Accounting (still pure latent)
This phase measures and reduces bytes without adding task‑specific signals.

B1) Honest wire‑bytes meter (both sides)
Python

# tools/bytes_meter.py
import math
import numpy as np

def text_bytes_utf8(prompt_str: str) -> int:
    return len(prompt_str.encode("utf-8"))

def latent_bits_count(M: int, d_latent: int, bits_per_param: int,
                      group_size: int = None, scale_bits: int = 16) -> int:
    """
    M: latent length (tokens)
    d_latent: latent dims per token
    bits_per_param: 16 (fp16), 8, 6, 4, etc.
    group_size: if not None, per-group scaling overhead in 'scale_bits' per group
    scale_bits: bits used to store scale per group (e.g., 16)
    """
    core = M * d_latent * bits_per_param
    overhead = 0
    if group_size:
        groups = math.ceil((M * d_latent) / group_size)
        overhead = groups * scale_bits
    header = 8 * 16  # 16 bytes header: M, d_latent, bits_per_param, group_size, seed, etc.
    return core + overhead + header

def latent_bytes(**kwargs) -> int:
    return math.ceil(latent_bits_count(**kwargs) / 8)

# Example sanity:
# M=32, d=32, 6-bit, group_size=32, scale_bits=16 ->
# core=6144b (768B), overhead≈(1024 vals / 32)*16b = 512b (64B), header=128b (16B) => total ≈ 848B
Interpretation: With M=32, d=32:

fp16 → 323216/8 = 2,048B (≈2.0 KB) → larger than text.

6‑bit + group scale → ~0.83–0.90 KB depending on header/metadata → smaller than text for many prompts.

4‑bit → ~0.57 KB (+overhead).

Measure actual text bytes via text_bytes_utf8(prompt). Report frontiers: (F1, FirstTok@1) vs bytes.

B2) Quantization schedule (no task info added)
Train with fp16 latents to hit stability.

Post‑train evaluate with 8‑, 6‑, and 4‑bit symmetric quantization on Z_shared (group‑wise scales).

If quality collapses under 4–6 bits, record the trade‑off transparently; do not add side‑channels.

Phase C — Evaluation Protocol & Ablations (still pure latent)
Model grid: M∈{64,48,32}, d_latent∈{32,48}, K∈{1,4}, τ∈{1.0,0.7}.

Metrics: EM, F1, FirstTok@1, FirstTok@5, length‑normalized log‑prob of gold for t=0…3.

Bytes: UTF‑8 text bytes vs latent bytes (fp16, int8, int6, int4).

Ablations:

K=1 vs K=4 (expect large drop at K=1).

KD on/off (should help early tokens).

Adapter on/off (MLP vs linear‑only).

Configuration Diffs
JSON

# configs/run_latent_pure.json (conceptual)
- "latent_len": 32,
+ "latent_len": 64,                  // start easier; then 48 -> 32

 "debug": {
   "latent_anchor_text": "Answer: ",
   "calibration_mode": "embed_rms",
   "append_bos_after_prefix": "no",
   "decode": {
-    "first_token_top_p": 1.0,
-    "first_token_temperature": 0.0,
-    "eos_ban_steps": 6
+    "first_token_top_p": 0.95,
+    "first_token_temperature": 0.7,
+    "eos_ban_steps": 4
   }
 },
 "training": {
-  "K": 1,
+  "K": 4,
   "loss_weights": {
-    "k_token_ce": 1.0
+    "k_token_ce": 1.0,
+    "kd_first_k": 1.0
   },
+  "kd_tau": 1.0
 },
 "adapter": {
   "type": "mlp_affine",
   "hidden_mult": 4,
   "prefix_tokens_P": 16
}
Expected Ranges (pure latent; conservative)
After A0–A2, M=64, fp16:

FirstTok@1: 12–20%

F1: 0.10–0.20

M=32, fp16: expect a drop (e.g., FirstTok@1 8–15%, F1 0.08–0.15).

With 6‑bit quantization: small additional drop (record it honestly).

Hitting 0.8× of text baseline under pure latent prompting on extractive QA is unlikely. If results plateau below the bar, log the negative result cleanly.

Pivot Options (if pure latent underperforms)
Change task: classification or short‑form generation where 50–70% of baseline might be acceptable under strict compression.

Change compression target: compress the text itself (token selection or learned token masks) instead of learning a separate latent space.

System‑level: KV‑cache reuse or retrieval to reduce wire bytes without changing the prompt semantics.

Logging & Repro
Save per‑example: prompt_utf8_bytes, latent_bytes_{fp16,int8,int6,int4}, first_token_gold, first_token_pred, gold logprob at t=0..3.

Plot Accuracy vs Bytes curves for each (M, bits).

Minimal Trainer Glue (drop‑in)
Python

# trainer.py (condensed)
from adapters.prefix_adapter import PrefixAdapter
from losses.k_token_ce import k_token_ce
from losses.kd_first_k import kd_first_k

def build(prefix_tokens_P, d_latent, d_model):
    adapter = PrefixAdapter(d_latent=d_latent, d_model=d_model, P=prefix_tokens_P)
    return adapter

def train_step(batch, encoder, adapter, student_llm, teacher_llm, cfg, optimizer):
    scaffold_ids, gold_ids = batch["scaffold_ids"], batch["gold_ids"]
    Z = encoder(batch["context_feats"])                       # [B, M, d_latent]
    prefix_embeds = adapter(Z[:, :cfg.prefix_tokens_P, :])    # [B, P, d_model]

    loss = 0.0
    loss += cfg.lambda_ce * k_token_ce(student_llm, prefix_embeds, scaffold_ids, gold_ids, K=cfg.K)
    loss += cfg.lambda_kd * kd_first_k(student_llm, teacher_llm, prefix_embeds, scaffold_ids, gold_ids, K=cfg.K, tau=cfg.kd_tau)
    loss.backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return loss.item()
Appendix H — Optional Hybrid Track (Not part of the pure‑latent claim)
Only use if your goal shifts from “can latents replace prompts?” to “can we build a competitive, compressed system?”.

These ideas improve scores but change the research question:

Copy bias & span head (extractive QA‑specific).

Pointer indices or micro‑prompt scaffolds.

Any side‑channel carrying answer location or task hints.

If you explore these, report results on a separate axis and include full byte accounting (latent bytes + indices + metadata). Expect better F1 but weaker generality.

FAQ / Known Concerns
“Quantization math doesn’t work out.”
For M=32, d=32:

fp16: 323216/8 ≈ 2.0 KB → larger than typical prompts.

6‑bit w/ group scales (32‑val groups, 16‑bit scales): ~0.83–0.90 KB (see bytes_meter.py).

This beats many SQuAD prompt sizes (~1.1–1.3 KB) if you do not add task‑specific payloads.

“Is 0.80× of baseline feasible?”
Not for pure latent prompting on extractive QA in our estimation. The aim here is to measure the limit honestly and either accept it or pivot.

TL;DR
Implement A0–A2 exactly as above.

Measure bytes and accuracy together.

Expect modest but real gains; report negative results if the bar isn’t met.

Keep any hybrid tricks quarantined in Appendix H with full byte accounting.
```
