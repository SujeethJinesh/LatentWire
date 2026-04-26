# SVAMP32 Source-Latent All-Layer Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a deployable source-derived positive method plus
  medium, seed-repeat, source-control, and cross-family gates
- current story: the candidate-pool syndrome bound is real, but source hidden
  summaries have not replaced the C2C oracle residue
- blocker: source-latent signal is too weak under leave-one-ID-out residue
  readout

## Gate

After killing raw GSM70 dynalign scale-up, the next selected branch was the
strongest existing source-latent sidecar variant: reuse the strict SVAMP32
candidate-pool syndrome analyzer, but extract all Qwen2.5-0.5B hidden layers
instead of only `last` or `mid,last`.

The run also exposed a harness issue: the ridge classifier used a primal
feature-dimension solve, which is pathological for all-layer features. The
solver now uses an equivalent centered dual ridge solve when feature dimension
exceeds sample count, while preserving the unregularized intercept.

## Evidence

| Feature set | Feature dim | Matched | Target-only | Zero-source | Label-shuffle | Target-self | Clean source-necessary | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| last | 1,792 | 9/32 | 14/32 | 13/32 | 13/32 | 2/3 | 0/6 | fail |
| mid,last | 3,584 | 9/32 | 14/32 | 14/32 | 13/32 | 3/3 | 0/6 | fail |
| all layers | 44,800 | 9/32 | 14/32 | 14/32 | 14/32 | 2/3 | 0/6 | fail |

All-layer provenance:

- teacher numeric coverage: `32/32`
- exact ordered ID parity: pass
- provenance issues: `0`
- source-destroying controls clean union: `0/6`
- failing criteria: `min_correct`, `preserve_fallback_floor`,
  `min_clean_source_necessary`

## Decision

Kill direct linear source-hidden syndrome readout as the next live branch,
including all-layer pooled hidden summaries. The failure is not lack of layer
coverage: richer source features still underperform the target-only decoder
floor and recover no clean source-necessary IDs.

The candidate-pool syndrome remains useful as a bound. The deployable branch
now needs a stronger source-derived mechanism, not another pooled-feature
ridge readout.

## Next Gate

Use a targeted literature pass and implement a bounded query-bottleneck or
token/layer C2C-residual distillation gate. The next experiment should train
against residue targets with cross-fitting and the same controls, then pass a
teacher-forced matched-only clean-ID gate before any generation or scale-up.

## Artifacts

See `results/svamp32_source_latent_all_layers_20260426/manifest.md`.
