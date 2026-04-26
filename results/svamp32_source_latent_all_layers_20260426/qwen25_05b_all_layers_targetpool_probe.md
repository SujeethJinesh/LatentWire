# SVAMP32 Source-Latent Syndrome Probe

- date: `2026-04-26`
- status: `source_latent_syndrome_probe_fails_gate`
- reference rows: `32`
- moduli: `2,3,5,7`
- ridge lambda: `1.0`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 9 | 0 | 2 |
| zero_source | 14 | 0 | 3 |
| shuffled_source | 10 | 0 | 1 |
| label_shuffled | 14 | 0 | 3 |
| target_only | 14 | 0 | 3 |
| slots_only | 8 | 0 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: none

## Interpretation

This probe trains leave-one-out ridge classifiers from frozen source hidden summaries to C2C residue classes. It tests source-latent predictability of the previously cleared syndrome bound, but remains a small-slice diagnostic.
