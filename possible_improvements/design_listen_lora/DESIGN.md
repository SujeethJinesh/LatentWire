
# Teach the Base to Listen (Tiny LoRA + KD)

**Goal:** Co-finetune **small LoRA adapters in early attention** so the base model *learns to listen* to the latent prefix, while keeping most weights frozen.

## Before vs After


```mermaid
flowchart LR
  subgraph Current Pipeline (Before)
  A[Input text] --> E[Latent Encoder (frozen/lt)]
  E -->|Z (M×d_z)| Z[(Shared latent wire)]
  Z --> AM1[Per-model Adapter (MLP)]
  AM1 --> P1[Prefix at input-only (shallow)]
  P1 --> T1[Chat Template]
  T1 --> LLM1[LLM (frozen)]
  end
  LLM1 --> O[Answer]
```


**After:**


```mermaid
flowchart LR
  subgraph After: Teach Base to Listen
  Z[(Shared latent)] --> P[Prefix (deep)]
  P --> LORA[Tiny LoRA (early attn)]
  LORA --> LLM[LLM (mostly frozen)]
  LLM --> O[Answer]
  end
```


## What changes
- Apply LoRA (r=8–16) to Q/K/V (and possibly O) in early N layers (e.g., 8–12).
- Keep deep prefix active; train LoRA + prefix jointly with strong **first-token CE** and **KD(τ≈2)**.

## Training
- Batch mix: 70–80% latent, 20–30% text-only (rehearsal).
- KD teacher = base model with adapters disabled.
- Optional adapter gating (turn LoRA off on text batches) to avoid interference.

## Expected benefits
- Minor trainables (<1–2%) but large gains in acceptance and stability.

## Integration steps
1. Apply PEFT LoRA to early attention modules only.
2. Add gating flag to disable LoRA on text-only batches.
3. Keep `first_token_ce_weight` high and constant; use `kd_first_k_K=8, tau=2`.

## Risks
- Overfitting if too many layers use LoRA; start with first 8–12 layers, r=8.
