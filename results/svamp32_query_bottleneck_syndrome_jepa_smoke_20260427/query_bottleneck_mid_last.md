# SVAMP32 Source-Latent Syndrome Probe

- date: `2026-04-27`
- status: `source_latent_syndrome_probe_fails_gate`
- reference rows: `32`
- moduli: `2,3,5,7`
- probe model: `query_bottleneck`
- ridge lambda: `1.0`
- teacher numeric coverage: `32/32`
- provenance issues: `0`
- query slots: `8`
- query epochs: `80`
- query lr: `0.01`
- query weight decay: `0.001`
- query seed: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 10 | 0 | 3 |
| zero_source | 13 | 0 | 3 |
| shuffled_source | 10 | 0 | 1 |
| label_shuffled | 12 | 0 | 3 |
| target_only | 14 | 0 | 3 |
| slots_only | 8 | 0 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: none

## Interpretation

This probe trains leave-one-out learned query bottlenecks over source hidden summary tokens to predict C2C residue classes. It tests source-latent predictability of the previously cleared syndrome bound, but remains a small-slice diagnostic.
