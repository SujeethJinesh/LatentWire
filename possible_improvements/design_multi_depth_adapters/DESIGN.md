
# Multi-Depth Latent Adapters (IAA-style)

**Goal:** Provide the latent to the frozen model at **multiple depths** (several layers), not just the input. Each adapter merges latent info with the layer's hidden state (cross-attn or MLP fusion).

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
  subgraph After: Multi-Depth Adapters (IAA-style)
  Z[(Shared latent)] --> ENC[Small projection]
  ENC --> A5[Adapter@L5]
  ENC --> A10[Adapter@L10]
  ENC --> A15[Adapter@L15]
  A5 --> LLM[LLM (frozen)]
  A10 --> LLM
  A15 --> LLM
  LLM --> O[Answer]
  end
```


## What changes
- Insert 2–3 **adapter blocks** at intermediate layers (e.g., 5, 10, 15).
- Each adapter: LN → cross-attn(h, z) → MLP → residual; train adapters + latent projection only.

## Training
- Same acceptance/KD losses; optionally add hidden-state alignment at adapter layers.

## Expected benefits
- Early layers get coarse latent cues, later layers get high-level latent context – better integration.

## Integration steps
1. Define `LatentAdapterBlock` with cross-attn from hidden→latent.
2. Patch chosen transformer blocks to include the adapter in forward.
3. Provide per-layer scalars to control adapter strength; log per-layer norms.

## Risks
- Too many adapters can overfit; start with 3 layers, small width.
