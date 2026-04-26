# SVAMP32 C2C Mechanism Syndrome Probe

- date: `2026-04-26`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- reference rows: `32`
- feature family: `c2c_prefill_token_layer_tail_residual`
- moduli: `2,3,5,7`
- ridge lambda: `1.0`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 8 | 0 | 0 |
| zero_source | 8 | 0 | 0 |
| shuffled_source | 8 | 0 | 0 |
| label_shuffled | 8 | 0 | 0 |
| target_only | 8 | 0 | 0 |
| slots_only | 7 | 1 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: `de1bf4d142544e5b`

## Interpretation

This probe trains leave-one-out classifiers from C2C prefill projector traces to the compact C2C residue syndrome. The feature extractor does not decode or parse C2C final answers; labels still come from C2C final numeric residues, so this remains a strict small-slice distillation diagnostic rather than a paper claim.
