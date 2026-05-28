# WJAC Prefilter Artifactization

Created: 2026-05-28T04:14:37Z

## Paper Gate Status

Paper readiness: not ICLR-ready. The project still lacks a positive method that survives larger frozen slices, seed stability, and strict cross-family falsification.

Current story: HybridKernel remains the active positive-method branch; WJAC is only a CPU-side sensitivity prefilter for the outlier-migration funnel, testing whether weight-column norms add an independent signal beyond EMA/M11b activation scores.

Blocking gap: do not spend GPU on WJAC unless this artifactized prefilter fails to kill it under the corrected two-diagnostic rule.

## Inputs And Scope

- `RUN_LEDGER.md`
- `DECISIONS.md`
- `paper/reviewer_feedback.md`
- `docs/wjac_prefilter.md`
- `experimental/outlier_migrate/phase9/preregister_om_phase9_mwjac.md`
- `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z`: DeepSeek-R1-Distill-Qwen-1.5B, layers 28/28, full cached layer set
- `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z`: Falcon-H1-0.5B-Instruct, layers 36/36, full cached layer set
- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`: Granite-4.0-H-Small, layers 9/40, representative early/mid/late CPU-feasible slice
- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`: Nemotron-3-Nano-30B-A3B-BF16, layers 9/52, representative early/mid/late CPU-feasible slice

No GPU jobs were run. Writes were restricted to `artifacts/wjac_prefilter/`.

## Method

- Activation score: EMA with `alpha=0.3` over prompt-mean squared cached activation magnitudes at positions 100..10000.
- Weight sensitivity proxy: `q_{l,i}=||W_{l,:,i}||_2^2`, summed over eligible layer-local linear tensors whose last axis matches the protected activation channel count.
- WJAC score: `q_{l,i} * EMA_i`.
- Comparison top-k: matched to cached `m11b_top10` protected count per covered layer.
- Large Granite/Nemotron runs use representative early/mid/late layer slices because full CPU streaming over all sharded expert weights is not a cheap prefilter.

## Corrected Kill Rule

KILL only if at least two diagnostics are true. Counted diagnostics: overlap >= 0.85, Spearman >= 0.90, nearly flat weight norms, and WJAC churn >= 0.25 without a layer-allocation change under the matched per-layer budget.

## Numeric Diagnostics

| Model | Coverage | WJAC vs M11b overlap | WJAC vs EMA overlap | Jaccard | Spearman WJAC/EMA | q CV | q p95/p05 | Churn | Allocation L1 | True diagnostics | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 28/28 | 0.9156 | 0.9740 | 0.8443 | 0.9985 | 0.0502 | 1.1520 | 0.0844 | 0.0 | 3/4 | KILL |
| Falcon-H1-0.5B-Instruct | 36/36 | 0.8447 | 0.9612 | 0.7311 | 0.9988 | 0.0669 | 1.2115 | 0.1553 | 0.0 | 2/4 | KILL |
| Granite-4.0-H-Small | 9/40 | 0.8585 | 0.9854 | 0.7521 | 0.9997 | 0.0462 | 1.0700 | 0.1415 | 0.0 | 3/4 | KILL |
| Nemotron-3-Nano-30B-A3B-BF16 | 9/52 | 0.8959 | 0.9517 | 0.8114 | 0.9977 | 0.0716 | 1.2275 | 0.1041 | 0.0 | 3/4 | KILL |

Additional signal: median Spearman(q, EMA) by model was deepseek=-0.2564, falcon=-0.6691, granite=0.1170, nemotron=-0.0520.

## Decision

Packet decision: `KILL_WJAC_PREFILTER`.

WJAC is killed as a positive-method funnel branch under the corrected offline prefilter. The covered surfaces all trip at least two diagnostics: the WJAC ranking is highly correlated with EMA, the weight norms are nearly flat, and most surfaces also have high M11b top-k overlap. WJAC does not create the high-churn/no-allocation-change failure mode; it mostly behaves like a near-no-op sensitivity multiplier.

## Output Artifacts

- `artifacts/wjac_prefilter/report.md`
- `artifacts/wjac_prefilter/decision.json`
- `artifacts/wjac_prefilter/wjac_scores.parquet`
