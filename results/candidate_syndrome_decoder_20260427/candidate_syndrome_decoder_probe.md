# Candidate-Syndrome Decoder Probe

Date: 2026-04-27

Status: `candidate_syndrome_decoder_fails_smoke`

This is a CPU-only artifact probe, not a learned method. It tests a tiny
hash-syndrome over target-side numeric candidate pools with source-destroying
controls.

## Surfaces

### live

- status: `candidate_syndrome_decoder_fails_smoke`
- target set: `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json`
- matched clean source-necessary: `1`
- matched target-self harms: `17`
- control clean union: `0`
- pass rule: `{'min_clean_source_necessary': False, 'control_clean_union_zero': True, 'target_self_harm_zero': False, 'numeric_coverage_parity': True}`

### holdout

- status: `candidate_syndrome_decoder_fails_smoke`
- target set: `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json`
- matched clean source-necessary: `4`
- matched target-self harms: `14`
- control clean union: `0`
- pass rule: `{'min_clean_source_necessary': True, 'control_clean_union_zero': True, 'target_self_harm_zero': False, 'numeric_coverage_parity': True}`

## Decision

Do not promote this hash-syndrome artifact probe. If the family is revived, it needs learned source predicates or a stronger source surface rather than another numeric-hash sidecar.
