# C1 Covariance Headroom

Status: `NEEDS_GPU_OR_CACHE_FOR_TRUE_COVARIANCE`.

This worker found activation magnitude caches and compact activation-mean
summaries, but no centered activation covariance matrices for the rotation
surfaces. I therefore computed a proxy, not a covariance result:

- surface: block / transformer-layer output only
- early position: 100
- late position: 10,000 for Granite and Nemotron; 20,000 for DeepSeek and
  Falcon from the Stage-1 compact summaries
- proxy vector drift: `||mean_abs_late - mean_abs_early||_2 / ||mean_abs_early||_2`
- diagonal proxy: `||late_abs^2 - early_abs^2||_2 / ||early_abs^2||_2`
- top-1% leaving proxy: strict leaving of top-1% mean-absolute channels

No off-diagonal covariance decomposition is supported by the available compact
artifacts. Any KLT, pairing, or rotation-refresh decision needs a covariance
cache or a small GPU capture that stores centered second moments.

## Proxy Summary

| model | layers | vector drift | diagonal proxy | late/early p99 | top-1% leaving proxy | interpretation |
|---|---:|---:|---:|---:|---:|---|
| Granite | 40 | 0.376 | 0.564 | 0.995 | 0.337 | Scale distribution is stable at p99, but diagonal energy moves; clip/tail retune is plausible. |
| Nemotron | 52 | 0.406 | 0.461 | 1.082 | 0.377 | Moderate diagonal drift; ParoQuant already very strong, so retune is mainly robustness/tail work. |
| DeepSeek | 28 | 0.456 | 0.658 | 1.269 | 0.453 | Strong diagonal/range drift; rotation smoke should be prioritized before channel rescue. |
| Falcon-H1 | 36 | 0.343 | 0.539 | 1.105 | 0.321 | Moderate block-output diagonal drift; branch/surface cache is needed before claiming rotation headroom. |

## Gate

- `ScaleRefresh` / `ClipRetune`: promote cheap smoke for Granite first, and
  optional Nemotron confirmation. Granite has a positive ParoQuant median but a
  severe negative tail, so CVaR tuning is the most direct DriftRot check.
- `RotationRefresh` / `Pairing` / `KLT`: do not promote from this artifact
  alone. The necessary off-diagonal covariance is missing.
- Falcon/DeepSeek ParoQuant smoke remains higher priority than broad grid
  tuning because those models decide whether rotation is a four-model remedy.

## Sources

- Granite activation magnitudes:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/activation_magnitudes.jsonl.gz`
- Nemotron activation magnitudes:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/activation_magnitudes.jsonl.gz`
- DeepSeek compact activation means:
  `experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/deepseek_r1_distill_qwen_1_5b/activation_means.npz`
- Falcon compact activation means:
  `experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/falcon_h1_0_5b/activation_means.npz`

