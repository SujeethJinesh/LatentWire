# LayerKeep / No-Gap CPU Artifact

Date: 2026-05-28T06:37:16Z

## Required Status

- Paper readiness: not ICLR-ready. The positive-method funnel still lacks a benchmark-backed method with larger frozen slices, seed repeats, paired uncertainty, and strict same-family vs cross-family separation.
- Current story: OutlierMigrate-style channel protection is regime-dependent. DeepSeek/Falcon remain the live cheap rescue surface for HYST/LAMBDA; LayerKeep is only a coarse fallback if channel-level selection is unstable but layer-level sensitivity is separated.
- Blocking gap: cached CPU signals do not provide causal per-layer recovery curves, and Granite has a substantial no-gap surface where static 1% has no recoverable loss gap.

## Inputs And Scope

No GPU jobs were run. I read reviewer feedback, the run ledger/decision state, the external review pack, `artifacts/funnel_prefilters/`, and Phase 9 cached summaries. I wrote only under `artifacts/layerkeep_nogap/`.

Primary source files are listed in `decision.json`. The layer ranking uses cached per-layer endpoint drift, strict set-leaving, global final top-10 allocation skew, and late protected-set churn from `artifacts/funnel_prefilters/decision.json`. The no-gap probe uses only prompt length plus activation summaries from decode positions 100/200/300 in cached `activation_magnitudes.jsonl.gz`; it does not use `static_gap`, losses, recovery, or no-gap labels as features.

## Layer Sensitivity Proxy

Score per layer:

`0.35 * norm(top10 endpoint drift) + 0.25 * norm(log1p(final top10 allocation ratio)) + 0.25 * norm(late top10 churn median) + 0.15 * norm(top1 strict set-leaving)`

The model-level recoverable surface is recorded as context but not used to reorder layers within a model. This is a sensitivity proxy, not causal evidence that keeping a layer improves recovery.

### Falcon Ranking

Falcon is the only model where LayerKeep is recommended as a bounded fallback candidate. Channel selection is unstable (`late churn median=0.298`, `p90=0.463`) and layer-level separation is meaningful (`top10 drift CV=0.340`, `range=0.528`, `allocation max=7.19x`, no-gap `0/12`).

| Rank | Layer | Score | Drift | Alloc ratio | Late churn | Strict leaving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 0.840 | 0.720 | 7.194 | 0.463 | 0.182 |
| 2 | 34 | 0.823 | 0.713 | 6.466 | 0.463 | 0.182 |
| 3 | 30 | 0.817 | 0.704 | 2.466 | 0.451 | 0.364 |
| 4 | 33 | 0.788 | 0.736 | 5.107 | 0.427 | 0.182 |
| 5 | 32 | 0.784 | 0.720 | 4.136 | 0.463 | 0.182 |
| 6 | 31 | 0.767 | 0.736 | 3.097 | 0.463 | 0.182 |
| 7 | 28 | 0.717 | 0.662 | 1.049 | 0.439 | 0.364 |
| 8 | 29 | 0.691 | 0.671 | 1.563 | 0.427 | 0.273 |
| 9 | 27 | 0.687 | 0.654 | 0.796 | 0.427 | 0.364 |
| 10 | 26 | 0.647 | 0.636 | 0.650 | 0.403 | 0.364 |

Recommended Falcon candidates in `layerkeep_candidates.csv`:

- `falcon_top3_peak`: `[35 34 30]`
- `falcon_top6_ranked`: `[35 34 30 33 32 31]`
- `falcon_late_contig_30_35`: `[30 31 32 33 34 35]`
- `falcon_late_contig_28_35`: `[28 29 30 31 32 33 34 35]`

The contiguous `30-35` block is the cleanest fallback implementation target. The ranked top-6 variant is slightly more score-aligned but has less implementation simplicity.

### Granite / DeepSeek / Nemotron

Granite has a visible late-layer ridge but the trace surface is weak for this fallback: no-gap is `4/12 = 0.333`, and the LAMBDA trace-surface gate already failed. Keep the Granite rows as diagnostics only.

| Rank | Layer | Score | Drift | Alloc ratio | Late churn | Strict leaving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 0.889 | 0.827 | 9.007 | 0.423 | 0.341 |
| 2 | 37 | 0.830 | 0.848 | 7.305 | 0.389 | 0.268 |
| 3 | 39 | 0.789 | 0.805 | 9.695 | 0.444 | 0.195 |
| 4 | 35 | 0.718 | 0.805 | 3.880 | 0.353 | 0.317 |
| 5 | 36 | 0.707 | 0.815 | 5.202 | 0.367 | 0.220 |
| 6 | 0 | 0.474 | 0.839 | 0.022 | 0.259 | 0.220 |
| 7 | 34 | 0.469 | 0.721 | 0.951 | 0.302 | 0.317 |
| 8 | 33 | 0.462 | 0.733 | 0.556 | 0.302 | 0.317 |

DeepSeek has a coherent late ridge and low no-gap (`1/12`), but this task was Falcon-first and the existing funnel already routes DeepSeek to LAMBDA/HYST smoke rather than LayerKeep.

| Rank | Layer | Score | Drift | Alloc ratio | Late churn | Strict leaving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 0.948 | 0.581 | 5.818 | 0.353 | 0.500 |
| 2 | 27 | 0.900 | 0.600 | 7.266 | 0.362 | 0.250 |
| 3 | 25 | 0.833 | 0.533 | 4.169 | 0.344 | 0.438 |
| 4 | 24 | 0.786 | 0.554 | 3.091 | 0.308 | 0.438 |
| 5 | 21 | 0.696 | 0.540 | 1.071 | 0.335 | 0.375 |
| 6 | 22 | 0.683 | 0.547 | 1.448 | 0.298 | 0.375 |
| 7 | 23 | 0.679 | 0.505 | 2.266 | 0.279 | 0.438 |
| 8 | 19 | 0.669 | 0.547 | 0.494 | 0.317 | 0.438 |

Nemotron is not a priority for LayerKeep because its late channel churn is locally saturated/low; the current positive story there is M11b/ParoQuant-related, not a coarse LayerKeep fallback.

| Rank | Layer | Score | Drift | Alloc ratio | Late churn | Strict leaving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47 | 0.853 | 0.685 | 4.067 | 0.058 | 0.370 |
| 2 | 49 | 0.839 | 0.681 | 5.219 | 0.051 | 0.370 |
| 3 | 46 | 0.834 | 0.662 | 3.922 | 0.058 | 0.407 |
| 4 | 48 | 0.819 | 0.675 | 4.691 | 0.051 | 0.370 |
| 5 | 51 | 0.813 | 0.658 | 6.245 | 0.065 | 0.185 |
| 6 | 50 | 0.803 | 0.658 | 5.874 | 0.051 | 0.333 |

## No-Gap Detector

Target label: `no_recoverable_static_gap` from cached `per_trace_metrics.json`.

Non-leaky feature set:

- `input_token_count`
- activation distribution summaries at decode positions 100/200/300: mean, top-1% mean, top-10% mean
- early top-10 channel-set Jaccard drift from position 100 to 300
- early late-layer activation share
- early layer-mean CV

I evaluated a model-specific leave-one-out decision stump. This is intentionally simple because `n=12` per model and Granite has only four positives. It is an offline warmup detector at best: deployment would require a cheap BF16/activation warmup through position 300 before deciding whether to run/interpret a protected policy. It is not a pre-run prompt-only classifier.

| Model | Positives | Accuracy@0.5 | Balanced acc@0.5 | ROC AUC | Predicted positives | Feature notes |
|---|---:|---:|---:|---:|---:|---|
| Granite | 4/12 | 0.750 | 0.625 | 0.250 | 1 | stump:input_token_count:le:86=2; stump:layer_mean_cv_100_300:le:1.2024=1; stump:layer_mean_cv_100_300:le:1.20697=8; stump:layer_mean_cv_100_300:le:1.20764=1 |
| DeepSeek | 1/12 | 0.583 | 0.318 | 0.318 | 4 | none_nonleaky_train_single_class=1; stump:input_token_count:ge:159=9; stump:late_layer_mean_share:ge:0.585534=2 |
| Falcon | 0/12 | 1.000 | n/a | n/a | 0 | none_nonleaky_train_single_class=12 |
| Nemotron | 2/12 | 0.667 | 0.400 | 0.400 | 2 | stump:input_token_count:le:102=9; stump:input_token_count:le:109.5=1; stump:pos100_mean:ge:2.40342=1; stump:row_top1_mean:le:18.0794=1 |

Granite full-batch post-hoc stump: `layer_mean_cv_100_300 le 1.206973; full-batch balanced accuracy 0.875`. This is useful only as an interpretation hint: some no-gap rows separate on unusually low early layer-mean CV, but the LOOCV detector does not hold up.

## Decision

- `recommend_layerkeep`: **true, Falcon-only bounded fallback candidate**. The evidence satisfies the gate condition: channel-level selection is unstable and layer-level signals are separated. This should be a small CPU-configured/GPU-later candidate, not a paper result.
- `recommend_nogap_filter`: **false**. The non-leaky Granite LOOCV stump gets one no-gap positive right at a fixed 0.5 cutoff, but misses 3/4 positives and has ROC AUC 0.25; the post-hoc full-batch rule is too small and unstable to deploy.

## Saturated / Alive / Priority

- Saturated: Granite no-gap filtering from these cached early features; Nemotron LayerKeep fallback.
- Alive: Falcon LayerKeep candidate subsets, especially contiguous layers `30-35` and broader `28-35` if boundary simplicity matters less than coverage.
- Highest-priority next gate: if the orchestrator opens LayerKeep, run only the Falcon fixed-trace smoke packet with candidate `30-35` versus a matched late-layer random/control block and report paired uncertainty. Do not claim progress until it beats the current Falcon smoke surface on the fixed traces.

## Rulings

- Promoted: Falcon late-block LayerKeep as a bounded fallback candidate.
- Weakened: Granite no-gap filtering as a deployable rule from cached early features.
- Ruled out for now: any no-gap filter that uses `static_gap`, recovery, or protected-policy losses as inputs; that would be label-leaky for deployment.
