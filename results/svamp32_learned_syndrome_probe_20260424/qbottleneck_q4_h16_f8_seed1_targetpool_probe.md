# SVAMP32 Learned Syndrome Probe

- date: `2026-04-24`
- status: `learned_syndrome_probe_fails_gate`
- reference rows: `32`
- moduli: `2,3,5,7`
- query count: `4`
- hidden dim: `16`
- epochs: `80`
- outer folds: `8`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 10 | 0 | 2 |
| zero_source | 11 | 0 | 3 |
| shuffled_source | 10 | 0 | 1 |
| label_shuffled | 13 | 0 | 3 |
| same_norm_noise | 14 | 0 | 3 |
| target_only | 14 | 0 | 3 |
| slots_only | 8 | 0 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: none

## Interpretation

This probe trains a cross-fitted tiny query bottleneck over frozen source token states to predict C2C residue classes. It is a strict source-syndrome diagnostic, not a generation method.
