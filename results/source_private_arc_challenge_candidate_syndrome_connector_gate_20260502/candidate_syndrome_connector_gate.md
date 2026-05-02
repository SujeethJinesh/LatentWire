# ARC Candidate-Syndrome Connector Gate

- date: `2026-05-02`
- pass gate: `False`
- test disagreement rows: `473`
- selected primary view: `tiny_score_shape_connector`

| View | Primary | Validation | Test | Qwen | Delta | CI95 low | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_packet_only_connector | True | 0.304 | 0.246 | 0.317 | -0.071 | -0.142 | 0.586 |
| tiny_score_shape_connector | True | 0.344 | 0.288 | 0.317 | -0.029 | -0.091 | 0.586 |
| paired_family_diagnostic_connector | False | 0.418 | 0.316 | 0.317 | -0.001 | -0.027 | 0.586 |

## Interpretation

This gate tests whether a low-capacity learned candidate scorer can recover the ARC TinyLlama-vs-Qwen disagreement oracle headroom from cached packet and source-score features. A negative result rules out cached packet/score-shape connectors and promotes true hidden-state or query-resampler connectors as the next branch.
