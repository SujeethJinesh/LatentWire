# HellaSwag Source-Family Stress Card

- pass gate: `True`
- source families represented: `2`
- Qwen full-validation pass: `True`
- TinyLlama heldout-slice pass: `True`
- TinyLlama delta vs best label-copy: `0.051758`
- TinyLlama CI95 low vs best label-copy: `0.025391`
- ICLR ready: `False`

## Rows

| Source family | Scope | Rows | Accuracy | Best label-copy | Delta | CI95 low | Jackknife | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5 | same_family_full_validation_global | 10042 | 0.526688 | 0.480880 | 0.045808 | 0.039634 | 3/3 | True |
| TinyLlama | non_qwen_source_family_heldout_slice | 1024 | 0.501953 | 0.450195 | 0.051758 | 0.025391 | 3/3 | True |
| TinyLlama | non_qwen_source_family_full_validation | 10042 | 0.619199 | 0.558753 | 0.060446 | 0.053423 | 3/3 | True |

## Interpretation

The TinyLlama evidence weakens the concern that the HellaSwag hidden-innovation packet only works because of Qwen-specific hidden coordinates. A full TinyLlama pass, if present, promotes the source-family robustness claim; it still preserves the no-overclaim boundary that ICLR needs true receiver-family transfer, benchmark diversity, and native systems evidence.
