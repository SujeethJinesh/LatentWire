# Conditional PQ Innovation Summary

- pass gate: `True`
- rows: `12`
- decisive n500 pass rows: `8/8`
- budget-2 n500 pass rows: `2/2`
- cross-family pass rows: `0/2`
- min decisive source accuracy: `0.996`
- max decisive best control accuracy: `0.288`

| Run | Basis | Bytes | Remap | Train->Eval | Pass | Source | Target | Best control | CI95 low | Unique ratio |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| n256_shared_text_remap101_uph_constrained_label | shared_text | 4 | 101 | all->all | `True` | 1.000 | 0.250 | 0.277 | 0.672 | 0.895 |
| n500_shared_text_remap101_uph_constrained_label | shared_text | 4 | 101 | all->all | `True` | 1.000 | 0.250 | 0.274 | 0.684 | 0.872 |
| n500_shared_text_remap103_uph_constrained_label | shared_text | 4 | 103 | all->all | `True` | 1.000 | 0.250 | 0.272 | 0.686 | 0.872 |
| n500_shared_text_remap107_uph_constrained_label | shared_text | 4 | 107 | all->all | `True` | 1.000 | 0.250 | 0.274 | 0.684 | 0.872 |
| n256_anchor_relative_remap101_uph_constrained_label | anchor_relative | 4 | 101 | all->all | `True` | 0.996 | 0.250 | 0.258 | 0.684 | 0.973 |
| n500_anchor_relative_remap101_uph_constrained_label | anchor_relative | 4 | 101 | all->all | `True` | 0.996 | 0.250 | 0.260 | 0.694 | 0.950 |
| n500_anchor_relative_remap103_uph_constrained_label | anchor_relative | 4 | 103 | all->all | `True` | 0.996 | 0.250 | 0.262 | 0.692 | 0.942 |
| n500_anchor_relative_remap107_uph_constrained_label | anchor_relative | 4 | 107 | all->all | `True` | 0.998 | 0.250 | 0.256 | 0.702 | 0.940 |
| n500_anchor_relative_remap101_budget2_uph_constrained_label | anchor_relative | 2 | 101 | all->all | `True` | 1.000 | 0.250 | 0.276 | 0.682 | 0.594 |
| n500_shared_text_remap101_budget2_uph_constrained_label | shared_text | 2 | 101 | all->all | `True` | 1.000 | 0.250 | 0.288 | 0.668 | 0.532 |
| n256_core_to_holdout_shared_text_remap101_uph_constrained_label | shared_text | 4 | 101 | core->holdout | `False` | 0.281 | 0.250 | 0.273 | -0.012 | 0.625 |
| n256_holdout_to_core_shared_text_remap101_uph_constrained_label | shared_text | 4 | 101 | holdout->core | `False` | 0.297 | 0.250 | 0.277 | -0.012 | 0.500 |

## Interpretation

Conditional PQ innovation passes same-family disjoint-ID n500 gates across shared-text and anchor-relative bases, including 2-byte low-uniqueness rows. Bidirectional held-out-family rows remain negative, so the method should be framed as shared-schema disjoint communication rather than unseen-family latent transfer.
