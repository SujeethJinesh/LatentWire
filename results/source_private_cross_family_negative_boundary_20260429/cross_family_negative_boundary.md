# Source-Private Cross-Family Negative Boundary

- pass gate: `True`
- total rows: `27`
- method families: `6`
- claim-ready cross-family methods: `0`
- oracle-headroom rows: `6`

## Family Summary

| Family | Rows | Negative boundary | Asymmetric/incomplete | Best accuracy | Best delta vs control | Claim status |
|---|---:|---:|---:|---:|---:|---|
| anchor-relative sparse packet | 8 | 6 | 2 | 0.496 | 0.246 | `not_claimed_cross_family` |
| canonical RASP | 2 | 1 | 1 | 0.492 | 0.242 | `not_claimed_cross_family` |
| consistency posterior negative ablation | 2 | 1 | 1 | 0.495 | 0.245 | `not_claimed_cross_family` |
| learned Wyner-Ziv / scalar syndrome | 6 | 5 | 1 | 0.623 | 0.373 | `not_claimed_cross_family` |
| learned target-preserving receiver | 4 | 4 | 0 | 0.453 | 0.143 | `not_claimed_cross_family` |
| masked innovation receiver | 5 | 5 | 0 | 0.266 | 0.016 | `not_claimed_cross_family` |

## Boundary Rows

| Row | Surface | Method | Bytes | Acc | Target | Best control | Oracle | Claim status | Reason |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `anchor_sparse::core_to_holdout::budget2` | core_to_holdout | 2-byte AR-SIP | 2.0 | 0.242 | 0.250 | 0.453 | - | `negative_boundary` | does not beat target (-0.008); insufficient matched-control delta (-0.211) |
| `anchor_sparse::core_to_holdout::budget4` | core_to_holdout | 4-byte AR-SIP | 4.0 | 0.250 | 0.250 | 0.271 | - | `negative_boundary` | does not beat target (0.000); insufficient matched-control delta (-0.021) |
| `anchor_sparse::core_to_holdout::budget6` | core_to_holdout | 6-byte AR-SIP | 6.0 | 0.125 | 0.250 | 0.283 | - | `negative_boundary` | does not beat target (-0.125); insufficient matched-control delta (-0.158) |
| `anchor_sparse::core_to_holdout::budget8` | core_to_holdout | 8-byte AR-SIP | 8.0 | 0.250 | 0.250 | 0.375 | - | `negative_boundary` | does not beat target (0.000); insufficient matched-control delta (-0.125) |
| `anchor_sparse::holdout_to_core::budget2` | holdout_to_core | 2-byte AR-SIP | 2.0 | 0.496 | 0.250 | 0.250 | - | `asymmetric_or_incomplete_not_claimed` | valid_rate,ci95_low_vs_target,ci95_low_vs_comparator |
| `anchor_sparse::holdout_to_core::budget4` | holdout_to_core | 4-byte AR-SIP | 4.0 | 0.248 | 0.250 | 0.250 | - | `negative_boundary` | does not beat target (-0.002); insufficient matched-control delta (-0.002) |
| `anchor_sparse::holdout_to_core::budget6` | holdout_to_core | 6-byte AR-SIP | 6.0 | 0.270 | 0.250 | 0.250 | - | `negative_boundary` | insufficient matched-control delta (0.020) |
| `anchor_sparse::holdout_to_core::budget8` | holdout_to_core | 8-byte AR-SIP | 8.0 | 0.373 | 0.250 | 0.262 | - | `asymmetric_or_incomplete_not_claimed` | valid_rate,ci95_low_vs_target,ci95_low_vs_comparator |
| `canonical_rasp_cross_family_core_to_holdout` | core -> holdout | 4-byte canonical RASP | 4.0 | 0.207 | 0.250 | 0.494 | - | `negative_boundary` | does not beat target (-0.043); insufficient matched-control delta (-0.287) |
| `canonical_rasp_cross_family_holdout_to_core` | holdout -> core | 4-byte canonical RASP | 4.0 | 0.492 | 0.250 | 0.250 | - | `asymmetric_or_incomplete_not_claimed` | valid_rate,ci95_low_vs_target,ci95_low_vs_comparator |
| `consistent_posterior_core_to_holdout_large` | core -> holdout large | 4-byte consistent posterior packet | 4.0 | 0.354 | 0.250 | 0.355 | - | `negative_boundary` | insufficient matched-control delta (-0.002) |
| `consistent_posterior_holdout_to_core_large` | holdout -> core large | 4-byte consistent posterior packet | 4.0 | 0.495 | 0.250 | 0.250 | - | `asymmetric_or_incomplete_not_claimed` | valid_rate,ci95_low_vs_target,ci95_low_vs_comparator |
| `wyner_ziv_cross::core_to_holdout::budget2` | core_to_holdout | 2-byte scalar WZ packet | 2.0 | 0.127 | 0.250 | 0.623 | - | `negative_boundary` | does not beat target (-0.123); insufficient matched-control delta (-0.496) |
| `wyner_ziv_cross::core_to_holdout::budget4` | core_to_holdout | 4-byte scalar WZ packet | 4.0 | 0.174 | 0.250 | 0.529 | - | `negative_boundary` | does not beat target (-0.076); insufficient matched-control delta (-0.355) |
| `wyner_ziv_cross::core_to_holdout::budget6` | core_to_holdout | 6-byte scalar WZ packet | 6.0 | 0.146 | 0.250 | 0.584 | - | `negative_boundary` | does not beat target (-0.104); insufficient matched-control delta (-0.438) |
| `wyner_ziv_cross::holdout_to_core::budget2` | holdout_to_core | 2-byte scalar WZ packet | 2.0 | 0.328 | 0.250 | 0.275 | - | `negative_boundary` | insufficient matched-control delta (0.053) |
| `wyner_ziv_cross::holdout_to_core::budget4` | holdout_to_core | 4-byte scalar WZ packet | 4.0 | 0.338 | 0.250 | 0.250 | - | `negative_boundary` | insufficient matched-control delta (0.088) |
| `wyner_ziv_cross::holdout_to_core::budget6` | holdout_to_core | 6-byte scalar WZ packet | 6.0 | 0.623 | 0.250 | 0.250 | - | `asymmetric_or_incomplete_not_claimed` | valid_rate,ci95_low_vs_target,ci95_low_vs_comparator |
| `candidate_embedding_receiver_heldout_anchor_relative_code_similarity_budget8_seed29_30` | heldout-family anchor-relative code-similarity seed29->30 | 8-byte candidate-embedding receiver | 8.0 | 0.281 | 0.250 | 0.258 | 0.756 | `negative_boundary` | insufficient matched-control delta (0.023) |
| `candidate_embedding_receiver_heldout_anchor_relative_ridge_budget8_seed29_30` | heldout-family anchor-relative ridge seed29->30 | 8-byte candidate-embedding receiver | 8.0 | 0.303 | 0.250 | 0.438 | 0.342 | `negative_boundary` | insufficient matched-control delta (-0.135) |
| `candidate_embedding_receiver_heldout_code_similarity_budget8_seed29_30` | heldout-family core-train/holdout-eval code-similarity seed29->30 | 8-byte candidate-embedding receiver | 8.0 | 0.256 | 0.250 | 0.285 | 1.000 | `negative_boundary` | insufficient matched-control delta (-0.029) |
| `candidate_embedding_receiver_heldout_core_to_holdout_budget8_seed29_30` | heldout-family core-train/holdout-eval seed29->30 | 8-byte candidate-embedding receiver | 8.0 | 0.453 | 0.250 | 0.311 | 0.809 | `negative_boundary` | insufficient matched-control delta (0.143) |
| `masked_innovation_anchor_relative_core_to_holdout::budget4` | core -> holdout | 4-byte masked innovation receiver | 4.0 | 0.258 | 0.258 | 0.258 | 1.000 | `negative_boundary` | full diagnostic oracle is high but learned source-private packet stays at target/control floor |
| `masked_innovation_shared_text_core_to_holdout::budget4` | core -> holdout | 4-byte masked innovation receiver | 4.0 | 0.266 | 0.250 | 0.250 | 1.000 | `negative_boundary` | full diagnostic oracle is high but learned source-private packet stays at target/control floor |
| `masked_innovation_anchor_relative_core_to_holdout::budget8` | core -> holdout | 8-byte masked innovation receiver | 8.0 | 0.250 | 0.250 | 0.250 | 1.000 | `negative_boundary` | full diagnostic oracle is high but learned source-private packet stays at target/control floor |
| `masked_innovation_shared_text_core_to_holdout::budget8` | core -> holdout | 8-byte masked innovation receiver | 8.0 | 0.250 | 0.250 | 0.250 | 1.000 | `negative_boundary` | full diagnostic oracle is high but learned source-private packet stays at target/control floor |
| `masked_innovation_anchor_relative_core_to_holdout::budget12` | core -> holdout | 12-byte masked innovation receiver | 12.0 | 0.250 | 0.250 | 0.250 | 1.000 | `negative_boundary` | full diagnostic oracle is high but learned source-private packet stays at target/control floor |

Interpretation: Cross-family latent/source-private learned communication is not a headline claim. Several rows retain oracle headroom, so the benchmark can represent source information, but current learned/static interfaces do not transfer it bidirectionally under controls.
