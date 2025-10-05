
# Compressed Chain-of-Thought (CCoT) with LoRA

**Goal:** Train the model to generate **k latent thought tokens** (continuous vectors) internally, then use them to produce the answer—compressing reasoning.

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
  subgraph After: Compressed CoT (CCoT)
  A[Input] --> LLM[LLM]
  LLM --> T[(k latent thought tokens)]
  T --> LLM
  LLM --> O[Answer]
  end
```


## What changes
- Insert a phase where the model emits **k learned latent vectors** (not text).
- Re-feed these vectors (as prefix or cache entries) before final decoding.
- Train with answer CE; optionally distill from a teacher that uses full CoT.

## Training
- Stage 1: collect teacher CoT (or use larger model) and distill into latent tokens (KD/hidden-state KD).
- Stage 2: student learns to **produce and consume** k latent thoughts with LoRA + prefix.

## Expected benefits
- Dramatically fewer tokens compared to textual CoT, with similar/better accuracy.

## Integration steps
1. Add special "latent-thought phase" between prompt and answer.
2. Implement latent thought head producing continuous vectors.
3. Re-insert thoughts via prefix/kv at next step.
4. Losses: answer CE, first-token CE (+KD), optional reconstruction of teacher hidden states.

## Risks
- Scheduling: ensure thoughts are short (k=4–10) and useful; add MI/aux loss to avoid degenerate solutions.
