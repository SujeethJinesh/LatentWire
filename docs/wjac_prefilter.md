# WJAC Offline Prefilter

Date: 2026-05-28

## Current Status

Paper readiness: not ICLR-ready. The project still lacks a positive method that
survives large frozen slices, seed repeats, and strict cross-family
falsification.

Current paper story: long-decode outlier-channel migration is real, but
activation-channel protection is not yet architecture-agnostic. M11b is useful
on Nemotron and ambiguous or weak on Granite, DeepSeek, and Falcon; the current
positive-method funnel must avoid spending GPU time on variants that only
relabel M11b.

Blocking gap for this prefilter: determine whether WJAC adds an independent
sensitivity signal, or whether `||W[:, i]||^2 * EMA(x_i^2)` collapses back to
activation-only EMA while preserving the same layer allocation.

## Inputs Read

- Reviewer/evaluation pressure: `paper/reviewer_feedback.md` and
  `colm_final/audits/reviewer_panel_feedback.md`.
- Current model-selection telemetry: `docs/architectural_decision_rule.md`.
- WJAC branch motivation and prior-art positioning:
  `docs/positive_method_scoop_check.md` and
  `experimental/outlier_migrate/phase9/preregister_om_phase9_mwjac.md`.
- Cached phase9 artifacts:
  - `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z`
  - `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z`
  - `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`
  - `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`

No GPU jobs were run. No shared ledger or result packet was edited.

## Method

This is an offline diagnostic, not endpoint scoring.

For each analyzed layer, I reconstructed the final activation score from cached
`activation_magnitudes.jsonl.gz`:

```text
EMA_i <- alpha * mean_prompts(|x_i(position)|^2) + (1 - alpha) * EMA_i
alpha = 0.3
positions = 100, 200, ..., 10000
```

I derived the WJAC sensitivity proxy from local Hugging Face safetensor shards:

```text
q_i = ||W[:, i]||_2^2
WJAC_i = q_i * EMA_i
```

Implementation assumptions:

- `q_i` sums hidden-input column norms for per-layer weight tensors whose input
  axis matches the protected channel count.
- For 3D expert banks, norms are summed across expert and output dimensions.
- Hidden-output axes, layernorms, convolutions, state tensors, and outside-stack
  tied heads are excluded because the WJAC formula is column/input sensitivity.
- WJAC top-k uses the same per-layer protected count as cached `m11b_top10`.

Coverage:

- DeepSeek: full 28/28 layers.
- Falcon: full 36/36 layers.
- Granite: representative 9/40 layers, early/mid/late
  `{0,1,2,19,20,21,37,38,39}`.
- Nemotron: representative 9/52 layers, early/mid/late
  `{0,1,2,25,26,27,49,50,51}`.

Large Granite/Nemotron full-layer WJAC would require streaming many multi-GB
expert and hybrid shards on CPU. The representative slices are sufficient for a
prefilter because the smaller full models and both large-model slices agree on
the decision.

## Kill Rule

Corrected WJAC kill rule for this prefilter: kill only if at least 2 of 4
diagnostics are true.

Diagnostic thresholds:

1. High M11b overlap: median WJAC-vs-M11b top-k overlap >= 0.75.
2. High EMA dependence: median Spearman(`WJAC_i`, `EMA_i`) >= 0.80.
3. Flat weight norms: median `q` CV <= 0.15 and median `q` p95/p05 <= 1.5.
4. Churn without allocation change: median churn >= 0.25 while layer-allocation
   L1 delta is 0.0.

## Numeric Diagnostics

| Model | Coverage | Top-k overlap median | Jaccard median | Spearman WJAC vs EMA median | q CV median | q p95/p05 median | Churn median | Layer allocation L1 delta | True diagnostics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B | 28/28 | 0.9156 | 0.8443 | 0.9985 | 0.0502 | 1.1520 | 0.0844 | 0.0 | 3/4 |
| Falcon-H1-0.5B-Instruct | 36/36 | 0.8447 | 0.7311 | 0.9988 | 0.0669 | 1.2115 | 0.1553 | 0.0 | 3/4 |
| Granite-4.0-H-Small | 9/40 | 0.8585 | 0.7521 | 0.9997 | 0.0462 | 1.0700 | 0.1415 | 0.0 | 3/4 |
| Nemotron-3-Nano-30B-A3B-BF16 | 9/52 | 0.8959 | 0.8114 | 0.9977 | 0.0716 | 1.2275 | 0.1041 | 0.0 | 3/4 |

Additional checks:

- Spearman(`q_i`, `EMA_i`) was not a stable positive signal:
  DeepSeek median -0.2564, Falcon median -0.6691, Granite median 0.1170,
  Nemotron median -0.0520.
- The only notable exception was Nemotron layer 1: top-k overlap 0.4758,
  `q` CV 0.3866, `q` p95/p05 19.6062, and Spearman WJAC-vs-EMA 0.6333. This is
  not enough to rescue the branch because the median behavior across the
  representative Nemotron slice still collapses to EMA.

## Decision

Decision: KILL WJAC as a positive-method funnel branch under the corrected
offline prefilter rule.

Reason: every analyzed surface triggers 3 of 4 kill diagnostics. WJAC mostly
preserves M11b top-k selections, is almost perfectly rank-correlated with the
activation-only EMA score, and uses weight norms that are flat enough that the
weight factor rarely changes the ordering. The fourth diagnostic is false:
WJAC does not add large churn. Instead, it mostly adds no useful churn while
retaining the same per-layer budget.

## Next Implication

Do not spend GPU time on M-WJAC endpoint scoring or WJAC-shuffled controls in
the current positive-method sprint. The sensitivity axis is not promoted by
this proxy. If sensitivity is revisited later, it should require a sharper
source of downstream sensitivity than weight-column norms and a fresh
preregistration before inspecting new rows.
