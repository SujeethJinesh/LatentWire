# OutlierMigrate Stage-1 Scoop Hedges

Date: 2026-05-26

Purpose: record the source boundaries added during the autonomous flight-window
paper pass before any Stage-1 GPU experiments.

## Sources Added to the Paper

- DecDEC, Park et al. OSDI 2025: qualitative precursor for dynamic decode-step
  salient channel changes. The paper now states DecDEC's short-horizon
  Transformer static-recall result and scopes our novelty to long-reasoning
  horizons, strict set-membership drift, and hybrid/parallel-hybrid coverage.
- Dissecting Outlier Dynamics in LLM NVFP4 Pretraining, Dong et al. 2026,
  arXiv:2602.02047: training-time hot-channel counterpart. The paper now
  distinguishes pretraining-step hot-channel persistence from inference-step
  decode drift.
- Rotated Runtime Smooth, Yi et al. 2024, arXiv:2409.20361: closest
  rotation-plus-runtime-adjustment adjacency. The paper differentiates it from
  per-layer top-K EMA protection.
- ParoQuant, Liang et al. 2026 plus project page: rotation baseline and
  source-reported Qwen3-4B AIME-24 comparison (AWQ 62.2, ParoQuant 73.3,
  FP16 75.6).
- Quamba2 and MambaQuant remain cited as static or rotation-oriented SSM
  quantization baselines.

## GPU Status

`swarm/state.json` records `gpu_hours_used = 278.643`. The newly authorized
roadmap has a hard cap of 280 cumulative GPU hours. That leaves approximately
1.36 GPU hours, which is below the estimated cost of E1 and triggers the
roadmap override condition. No Stage-1 GPU experiment was started.
