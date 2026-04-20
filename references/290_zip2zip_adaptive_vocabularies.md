# zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression

- Title: `zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression`
- Date: `2025-06-01`
- Link: `https://arxiv.org/abs/2506.01084`

## Why it matters

- It treats the token interface itself as adjustable at inference time rather
  than fixed, which is a useful lateral direction now that several local bridge
  families have saturated.
- The paper is relevant as a vocabulary/interface reference rather than as a
  direct KV-transport baseline.

## Transplantable idea

- Create a smaller adaptive shared token interface:
  - compress multiple source-side token pieces into a learned shared unit,
  - supervise transport/bridge operations on that adaptive interface instead of
    raw tokenizer positions.

## Use in our stack

- Best used as inspiration for:
  - token/span grouping before calibration alignment,
  - adaptive shared units or byte-span buckets before the local bridge,
  - or a future vocab-side ablation once the current dynamic teacher lane is
    exhausted.
