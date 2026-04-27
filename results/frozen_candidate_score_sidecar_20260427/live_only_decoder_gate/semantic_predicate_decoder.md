# SVAMP Source Semantic Predicate Decoder

- date: `2026-04-27`
- status: `semantic_predicate_decoder_fails_smoke`
- mode: `learned_logodds`

This CPU-only probe learns fold-local semantic predicate weights over
source generated text and uses an erasure rule to preserve the target
fallback unless the source-supported candidate is sufficiently unique.

## Surfaces

### live

- target set: `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`
- matched correct: `21/70`
- matched accepted: `1`
- matched clean source-necessary: `0`
- matched accepted harm: `0`
- control clean union: `0`
- clean IDs: `[]`

### holdout

- target set: `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`
- matched correct: `9/70`
- matched accepted: `1`
- matched clean source-necessary: `0`
- matched accepted harm: `0`
- control clean union: `0`
- clean IDs: `[]`

## Decision

Do not promote this semantic-predicate decoder on current artifacts. If revived, it needs stronger source surfaces or model-collected target-side likelihood/uncertainty features.
