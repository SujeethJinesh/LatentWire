# SVAMP Source Semantic Predicate Decoder

- date: `2026-04-27`
- status: `semantic_predicate_decoder_fails_smoke`
- mode: `learned_logodds`

This CPU-only probe learns fold-local semantic predicate weights over
source generated text and uses an erasure rule to preserve the target
fallback unless the source-supported candidate is sufficiently unique.

## Surfaces

### live

- target set: `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`
- matched correct: `0/3`
- matched accepted: `0`
- matched clean source-necessary: `0`
- matched accepted harm: `0`
- control clean union: `0`
- clean IDs: `[]`

### holdout

- target set: `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`
- matched correct: `0/3`
- matched accepted: `0`
- matched clean source-necessary: `0`
- matched accepted harm: `0`
- control clean union: `0`
- clean IDs: `[]`

## Decision

Do not promote this semantic-predicate decoder on current artifacts. If revived, it needs stronger source surfaces or model-collected target-side likelihood/uncertainty features.
