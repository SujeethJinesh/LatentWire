# HellaSwag Repair Systems Acceptance Card

## Headline

- Method gate pass: `True`
- Systems audit pass: `True`
- Native queue allowed: `False`
- Best delta vs source-label copy: `0.037109`
- Best delta vs trained label-copy control: `0.0400390625`
- Trained label-copy control rows: `3`
- Best repair row: `hidden_innovation_repair`
- Strict promotion rule: `Promote a HellaSwag repair only if it beats source-label-copy by at least 0.02 on the frozen validation slice, also beats trained label-bias copy controls by at least 0.02 when those controls are available, has paired CI95 low > 0 when paired samples are available, and exposes no source text, source KV, raw hidden vectors, or raw score vectors.`

## Rows

| Row | Accuracy | Source-label copy | Delta | Trained copy | Delta trained | Method gate | Systems audit | Bytes | Kill reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fixed_packet_vs_source_label_copy` | 0.460 | 0.462 | -0.002 | - | - | `False` | `True` | 2B/5B | source_label_copy_control_beats_packet |
| `score_margin_packet` | 0.467 | 0.465 | 0.002 | - | - | `False` | `True` | 2B/5B | delta_below_0p02_source_label_copy_margin |
| `public_receiver_top2_repair` | 0.413 | 0.462 | -0.049 | - | - | `False` | `True` | 1B/5B | source_label_copy_control_beats_packet |
| `train_source_score_repair` | 0.447 | 0.462 | -0.015 | 0.459 | -0.012 | `False` | `True` | 2B/5B | source_label_copy_control_beats_packet |
| `hidden_summary_repair` | 0.413 | 0.462 | -0.049 | - | - | `False` | `True` | 2B/5B | source_label_copy_control_beats_packet |
| `top2_contrastive_switch_repair` | 0.449 | 0.462 | -0.013 | 0.459 | -0.010 | `False` | `True` | 2B/5B | source_label_copy_control_beats_packet |
| `hidden_innovation_repair` | 0.499 | 0.462 | 0.037 | 0.459 | 0.040 | `True` | `True` | 2B/5B | method_gate_clear |

## Interpretation

The hidden-innovation denoiser is the first HellaSwag repair in this card to clear the strict source-label/trained-label copy margin with paired uncertainty while preserving source-private byte accounting. Native queueing remains blocked only because native NVIDIA/vLLM/SGLang rows are not yet available.

## Checks

- `all_rows_have_label_copy_delta`: `True`
- `all_rows_have_byte_latency_exposure_fields`: `True`
- `systems_audit_passes`: `True`
- `method_gate_status_matches_margin_rule`: `True`
- `trained_label_copy_control_available`: `True`
- `trained_label_copy_control_respected_when_available`: `True`
- `source_private_boundary_preserved`: `True`
- `label_copy_margin_gate_clears_best_or_blocks_all`: `True`
- `oracle_headroom_documented`: `True`
- `native_queue_blocked`: `True`
- `systems_comparator_available`: `True`
- `strict_method_gate_rule_recorded`: `True`
