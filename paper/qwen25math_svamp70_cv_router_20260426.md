# Qwen2.5-Math -> Qwen3 SVAMP70 CV Router

- date: `2026-04-26`
- status: original-slice pass, disjoint-holdout fail
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- method: 1-byte source residue sidecar plus 5-fold decision-stump router

## Start Status

- current ICLR readiness: not ready
- current story: the fixed textless length-ratio guard is positive on the
  original SVAMP70 slice but fails a disjoint holdout
- exact blocker: determine whether a tiny cross-fitted router can replace the
  fixed hand threshold without source-control leakage

## Router

For each fold, the router trains on the other four folds and selects one
feature, threshold, and direction to decide whether to accept the source
sidecar or keep target fallback.

Features are existing JSONL fields only:

- source prediction character count
- source/target prediction length ratio
- source numeric count
- source generated tokens
- source final-marker indicator

Training objective:

`help - harm - 0.10 * accept_count`

The same fold rule is applied to matched source and to zero-source,
shuffled-source, label-shuffle, same-norm-noise, target-only, and slots-only
controls.

## Original SVAMP70

| Row | Correct | Clean Source-Necessary | Control Clean Union | Accepted Harm |
|---|---:|---:|---:|---:|
| CV router, moduli 2,3,5,7 | 25/70 | 4/6 | 0/6 | 1 |
| CV router, moduli 97 | 25/70 | 4/6 | 0/6 | 1 |

Paired comparisons:

| Comparison | Delta | Candidate-only | Baseline-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| CV router vs target | +0.0571 | 5 | 1 | [-0.0143, +0.1286] | 0.2207 |
| CV router vs text | +0.0429 | 12 | 9 | [-0.0857, +0.1714] | 0.6625 |
| CV router vs C2C | -0.0857 | 10 | 16 | [-0.2286, +0.0429] | 0.3268 |

## Disjoint Holdout

On SVAMP `chal-101` through `chal-170`, the same CV router family fails:

| Row | Correct | Clean Source-Necessary | Control Clean Union | Accepted Harm |
|---|---:|---:|---:|---:|
| CV router, moduli 2,3,5,7 | 6/70 | 0/2 | 0/2 | 2 |
| CV router, moduli 97 | 6/70 | 0/2 | 0/2 | 2 |

Paired comparisons:

| Comparison | Delta | Candidate-only | Baseline-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| CV router vs target | -0.0286 | 0 | 2 | [-0.0714, +0.0000] | 0.4795 |
| CV router vs text | -0.1714 | 2 | 14 | [-0.2714, -0.0571] | 0.0060 |
| CV router vs C2C | -0.4429 | 2 | 33 | [-0.5571, -0.3143] | 0.0000 |

## Decision

The CV router is a useful diagnostic but not a promotable method. It proves the
original slice can be matched by an auditable fold-learned rule with zero clean
control leakage, but the same family collapses on the disjoint holdout. This
points to source-surface instability and weak source-alone signal, not only
overfitting of a single fixed threshold.

Do not scale the current sidecar/router family until source-surface discovery
finds a slice where source-alone has enough clean target-complementary IDs, or
a stronger source encoder produces a stable source-derived signal.

## Next Gate

Continue source-surface discovery before C2C spend. Materialize source/target
and text relay first; run C2C only when clean source-only IDs after text
exclusion are high enough to support a meaningful sidecar test.

## Artifacts

- CV router script:
  - `scripts/analyze_svamp_source_sidecar_cv_router_gate.py`
- original-slice CV router:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_cv_router_penalty010_sidecar.json`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_cv_router_penalty010_predictions.jsonl`
- original-slice paired comparisons:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/cv_router_penalty010_paired_vs_target.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/cv_router_penalty010_paired_vs_text.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/cv_router_penalty010_paired_vs_c2c.md`
- holdout CV router:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_cv_router_penalty010_sidecar.json`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_cv_router_penalty010_predictions.jsonl`
- holdout paired comparisons:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/cv_router_penalty010_paired_vs_target.md`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/cv_router_penalty010_paired_vs_text.md`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/cv_router_penalty010_paired_vs_c2c.md`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_sidecar_cv_router_gate.py tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_sidecar_cv_router_gate.py scripts/analyze_svamp32_source_only_sidecar_router_gate.py
```
