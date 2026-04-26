# SVAMP32 Source-Latent Syndrome Probe

- date: `2026-04-26`
- status: `source_latent_syndrome_probe_fails_gate`
- reference rows: `32`
- moduli: `2,3,5,7`
- probe model: `ridge`
- ridge lambda: `1.0`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 8 | 0 | 0 |
| zero_source | 8 | 0 | 0 |
| shuffled_source | 7 | 0 | 0 |
| label_shuffled | 8 | 0 | 0 |
| target_only | 8 | 0 | 0 |
| slots_only | 7 | 1 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: `de1bf4d142544e5b`

## Interpretation

This probe trains leave-one-out ridge classifiers from frozen source hidden summaries to C2C residue classes. It tests source-latent predictability of the previously cleared syndrome bound, but remains a small-slice diagnostic.
