# Separable Asymmetric K/V Qwen Smoke

Date: 2026-04-21

Checkpoint:

`checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`

Model pair:

- Source: `Qwen/Qwen2.5-0.5B-Instruct`
- Target: `Qwen/Qwen3-0.6B`

Common settings:

- `--source-use-chat-template --target-use-chat-template`
- `--source-enable-thinking false --target-enable-thinking false`
- `--source-reasoning-mode brief_analysis`
- `--gate-mode fixed --fixed-gate 0.10`
- `--fusion-rule static --kv-transport both`
- `--kv-route-selection-ratio 0.25 --kv-value-selection-ratio 0.75`

## Results

| Split | Route metric | Value metric | Target | RotAlign | Delta | Avg bytes | Route/value overlap | Jaccard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GSM5 | attention | energy | 0.20 | 0.40 | +0.20 | 1369158.0 | 0.674 | 0.206 |
| GSM5 | random | random | 0.20 | 0.20 | 0.00 | 1369158.0 | 0.837 | 0.270 |
| GSM10 | attention | energy | 0.10 | 0.10 | 0.00 | 1366231.3 | 0.666 | 0.202 |

## Interpretation

The new metric-separated selector fixes the prior nested-mask failure: route
and value masks are no longer identical, and the attention/energy run shows
lower overlap than the random/random control. The GSM5 result has one
method-only win, but GSM10 is neutral, so this is a promising ablation lane
rather than a settled positive method.

Next run should scale to GSM30 with three matched controls:

- route attention / value energy
- route attention / value attention
- route random / value random
