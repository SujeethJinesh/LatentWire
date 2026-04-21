# Headwise Route-Atom Ratio Sweep

Checkpoint:

- `checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`

Protocol:

- `data/gsm8k_5.jsonl`
- Qwen2.5-0.5B-Instruct -> Qwen3-0.6B
- sparse `K-only`
- `position_selection_metric=attention`
- `position_selection_ratio=0.5`
- fixed gate `0.10`
- `runtime_head_selection_metric=headwise_route_atom`

| Runtime head ratio | Accuracy | Avg bytes | Route-atom sharpness | Route-atom JS | Orientation span | Notes |
|---:|---:|---:|---:|---:|---:|---|
| none | 0.4000 | 686026.6 | - | - | - | dense-head `dynalign_prefdist` reference |
| 0.25 | 0.4000 | 176381.8 | 0.8562 | 0.0967 | 0.2494 | best bytes/accuracy point so far |
| 0.50 | 0.2000 | 346263.4 | 0.7793 | 0.0861 | 0.5910 | too many extra heads reintroduced |
| 0.75 | 0.2000 | 516145.0 | 0.7149 | 0.0801 | 0.6810 | still below the 25% setting |

Controlled GSM10 follow-up:

| Setting | Accuracy | Avg bytes | Paired delta vs dense-head dynalign | Method-only | Baseline-only |
|---|---:|---:|---:|---:|---:|
| dense-head dynalign reference | 0.1000 | 681668.4 | - | - | - |
| route-atom ratio 0.25 | 0.1000 | 175249.2 | 0.0000 | 0 | 0 |

Interpretation:

- The useful route-atom setting is non-monotonic: 25% preserves the smoke score
  while 50% and 75% degrade it.
- The 25% setting keeps sharper heads and lower orientation span, suggesting
  the extra heads admitted at higher ratios may add interference rather than
  useful transport.
- Controlled GSM10 ties the dense-head branch exactly, so this is still not an
  accuracy-positive result. It is a strong bytes/control lead: same score and
  same paired correctness at roughly 3.9x fewer bytes.

GSM30 scale check:

| Setting | Accuracy | Avg bytes | Paired delta vs target-alone | Method-only | Baseline-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| target-alone | 0.0667 | - | - | - | - | - | - |
| route-atom ratio 0.25 | 0.0333 | 172643.3 | -0.0333 | 0 | 1 | 1 | 28 |

Route telemetry on GSM30:

- `head_keep_fraction_avg`: `0.2500`
- `route_atom_score_entropy_avg`: `1.9455`
- `route_atom_score_gap_avg`: `0.0460`
- `route_atom_sharpness_mean_avg`: `0.8588`
- `route_atom_js_divergence_mean_avg`: `0.0954`
- `route_atom_orientation_span_avg`: `0.2442`

Scale interpretation:

- The 25% route-atom selector does **not** preserve the GSM5/GSM10 behavior on
  the larger GSM30 check.
- It remains byte-efficient and interpretable, but the paired GSM30 read has no
  method-only wins and one target-only win.
- Treat route atoms as a compression/control and collapse-diagnostic tool until
  they are stacked with a learned query-conditioned interface or a stronger
  teacher.
