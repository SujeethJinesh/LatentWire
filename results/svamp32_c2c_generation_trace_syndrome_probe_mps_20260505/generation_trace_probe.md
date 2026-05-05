# SVAMP32 C2C Mechanism Syndrome Probe

- date: `2026-05-05`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- reference rows: `32`
- feature family: `c2c_generation_projector_and_logit_trace_history`
- moduli: `2,3,5,7`
- ridge lambda: `1.0`
- teacher numeric coverage: `32/32`
- provenance issues: `0`

## Summary

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 13 | 4 | 0 |
| zero_source | 14 | 4 | 0 |
| shuffled_source | 12 | 3 | 0 |
| label_shuffled | 15 | 4 | 0 |
| target_only | 14 | 4 | 0 |
| slots_only | 8 | 0 | 0 |

- clean source-necessary IDs: `0`
- source-necessary IDs: none
- control clean union IDs: `4c84ebf42812703b`, `4d780f825bb8541c`, `b1200c32546a34a5`, `de1bf4d142544e5b`

## Interpretation

This probe trains leave-one-out classifiers from C2C prefill projector traces to the compact C2C residue syndrome. The feature extractor does not decode or parse C2C final answers; labels still come from C2C final numeric residues, so this remains a strict small-slice distillation diagnostic rather than a paper claim.
