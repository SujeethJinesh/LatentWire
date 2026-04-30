# Source-Private Systems Comparison Table

- headline learned pass rows: `3`
- headline learned min delta vs target: `0.625`
- same-surface text max accuracy: `0.250`
- same-surface text max delta vs target: `0.000`
- scalar source-code comparator accuracy: `1.000`
- QJL-style comparator accuracy: `1.000`
- min endpoint non-packet QJL 1-bit byte ratio vs packet: `10752.0x`

## Rows

| Group | Method | Surface | Bytes | Accuracy | Target | Best control | Pass | Paper use |
|---|---|---|---:|---:|---:|---:|---|---|
| learned_packet | learned synonym dictionary packet | core_to_holdout synonym_stress | 4 | 1 | 0.25 | 0.25 | pass | headline learned calibrated-dictionary packet |
| same_surface_control | same-byte structured text relay | core_to_holdout synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | core_to_holdout synonym_stress | 4 | 0.246094 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | core_to_holdout synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| learned_packet | learned synonym dictionary packet | holdout_to_core synonym_stress | 4 | 0.875 | 0.25 | 0.257812 | pass | headline learned calibrated-dictionary packet |
| same_surface_control | same-byte structured text relay | holdout_to_core synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | holdout_to_core synonym_stress | 4 | 0.257812 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | holdout_to_core synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| learned_packet | learned synonym dictionary packet | same_family_all synonym_stress | 4 | 0.9375 | 0.25 | 0.25 | pass | headline learned calibrated-dictionary packet |
| same_surface_control | same-byte structured text relay | same_family_all synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | same_family_all synonym_stress | 4 | 0.246094 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | same_family_all synonym_stress | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| heldout_boundary | learned synonym dictionary packet | core_to_holdout heldout_synonym | 4 | 0.5 | 0.25 | 0.375 | fail | negative boundary for held-out paraphrase generalization |
| same_surface_control | same-byte structured text relay | core_to_holdout heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | core_to_holdout heldout_synonym | 4 | 0.234375 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | core_to_holdout heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| heldout_boundary | learned synonym dictionary packet | holdout_to_core heldout_synonym | 4 | 0.375 | 0.25 | 0.375 | fail | negative boundary for held-out paraphrase generalization |
| same_surface_control | same-byte structured text relay | holdout_to_core heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | holdout_to_core heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | holdout_to_core heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| heldout_boundary | learned synonym dictionary packet | same_family_all heldout_synonym | 4 | 0.4375 | 0.25 | 0.375 | fail | negative boundary for held-out paraphrase generalization |
| same_surface_control | same-byte structured text relay | same_family_all heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | direct same-byte text control |
| same_surface_control | random same-byte packet | same_family_all heldout_synonym | 4 | 0.246094 | 0.25 |  | control_ok | source-destroying byte control |
| same_surface_control | answer-label text truncated to 4 bytes | same_family_all heldout_synonym | 4 | 0.25 | 0.25 |  | control_ok | answer-only leakage control |
| compression_baseline | scalar quantized source projection | all_to_all slot | 6 | 1 | 0.25 | 0.25 | pass | direct vector/source-coding comparator |
| compression_baseline | QJL-style residual projection | all_to_all slot | 6 | 1 | 0.25 | 0.25 | pass | direct quantization/sketch comparator |
| compression_control | raw source sign sketch | all_to_all slot | 6 | 0.306641 | 0.25 | 0.25 | unpromoted | raw sketch ablation |
| endpoint_systems | matched_packet | core n160 label_strict | 2 | 0.675 |  |  | pass | Mac endpoint packet systems row |
| endpoint_systems | matched_byte_text_2 | core n160 label_strict | 2 | 0.25 |  |  | fail | same-byte text endpoint control |
| endpoint_systems | query_aware_diag_span | core n160 label_strict | 14 | 0.69375 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | structured_free_text_diag | core n160 label_strict | 17 | 0.7125 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | structured_json_diag | core n160 label_strict | 21 | 0.575 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | full_hidden_log | core n160 label_strict | 366.5 | 0.4625 |  |  | higher_rate_comparator | full-log rate baseline |
| endpoint_systems | matched_packet | holdout n160 label_strict | 2 | 0.6875 |  |  | pass | Mac endpoint packet systems row |
| endpoint_systems | matched_byte_text_2 | holdout n160 label_strict | 2 | 0.25 |  |  | fail | same-byte text endpoint control |
| endpoint_systems | query_aware_diag_span | holdout n160 label_strict | 14 | 0.6875 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | structured_free_text_diag | holdout n160 label_strict | 17 | 0.71875 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | structured_json_diag | holdout n160 label_strict | 21 | 0.59375 |  |  | higher_rate_comparator | structured text rate comparator |
| endpoint_systems | full_hidden_log | holdout n160 label_strict | 373.5 | 0.53125 |  |  | higher_rate_comparator | full-log rate baseline |

Claim boundary: The learned packet has clean same-surface 4-byte rows and seed stability, but held-out paraphrase generalization fails; KV/TurboQuant rows are byte-floor accounting only.
