# Granite DriftRot Clip/CVaR Confirmation Screen

Status: `PASS_TIGHT_CLIP_GRANITE_POSITIVE_TRACES`.

This is a **screen**, not a final method claim. The run directly scored the
held-out confirmation traces `[4, 7, 9, 10]` after the full 12-trace Granite
clip screen proved too expensive. Because the confirmation split was inspected
directly, a paper-level DriftRot-Config claim still needs a repeat split or a
full finalist packet. The result is still useful because it tests the known
Granite ParoQuant tail trace under a bounded config change.

## Compared Configs

| Candidate | Scale clip | Median | Worst | Mean | Decision |
|---|---:|---:|---:|---:|---|
| Baseline ParoQuant | [0.25, 4.0] | 0.640 | -26.371 | -6.022 | reference |
| Loose clip | [0.125, 8.0] | 0.686 | -79.783 | -19.359 | KILL: tail worsens |
| Tight clip | [0.5, 2.0] | 0.901 | 0.601 | 2.085 | PROMOTE: tail fixed on this screen |

## Interpretation

The loose clip is killed. It slightly improves the median on the four-trace
subset but makes the catastrophic tail much worse.

The tight clip is promising as a Tier-2 tail-control candidate. It converts the
known Granite ParoQuant tail trace from `-26.371` to `+5.940` recovery and keeps
all four confirmation traces positive. This is the first evidence that a
DriftRot-style tail objective can robustify static ParoQuant rather than merely
replicate it.

## Repeat Hold-Out

After this screen, a separate positive-gap hold-out subset was run on prompts
`[1, 2, 5]`, which were not in the inspected `[4, 7, 9, 10]` screen.

| Prompt | Baseline ParoQuant | Tight clip | Margin |
|---:|---:|---:|---:|
| 1 | 1.567 | 1.647 | +0.081 |
| 2 | 0.477 | 0.847 | +0.370 |
| 5 | 0.997 | 0.996 | -0.001 |

Hold-out median recovery is `0.996` with CI95 `[0.847, 1.647]`. This passes
the repeat split gate: the weak prompt improves, the strong prompts do not
meaningfully regress, and all three held-out recoveries remain positive.

## Caveat

## Positive-Trace Completion

The remaining recoverable Granite trace, prompt `8`, was then scored with tight
clip and reached `0.812` recovery versus the original ParoQuant `0.792`.

Across all eight recoverable Granite traces:

| Metric | Original ParoQuant | Tight clip `[0.5, 2.0]` | Margin |
|---|---:|---:|---:|
| Median recovery | 0.754 | 0.922 | +0.0509 median per-trace |
| Bootstrap CI95 | [0.477, 1.004] | [0.781, 1.647] | [0.0166, 0.370] |
| Worst trace | -26.371 | 0.601 | +32.311 |
| Mean recovery | -2.532 | 1.581 | +4.112 |

Source table: `artifacts/scale_cvar_clip/granite_tight_positive_trace_table.csv`.

## Caveat

This result must still be framed carefully. Config/clip retuning is Tier 2 in
the novelty table, not the headline method. The current defensible claim is
that a drift/tail-aware clip objective robustifies Granite's rotation baseline
on the recoverable positive-gap traces. The next valid gate is:

1. integrate tight clip as a tail-control DriftRot candidate, not as a generic
   claim that hyperparameter tuning beats ParoQuant;
2. test the idea on another model/tail surface only if we need cross-model
   evidence for the tail-control mechanism.

## Cross-Model Check: DeepSeek

DeepSeek-R1-Distill-Qwen-1.5B was then run with the same tight clip `[0.5, 2.0]`
to test whether the Granite tail-control behavior transfers. It does not pass
as a clean cross-model improvement.

| Metric | Baseline DeepSeek ParoQuant | Tight clip `[0.5, 2.0]` | Margin |
|---|---:|---:|---:|
| Median recovery | 0.756 | 0.518 | -0.232 median per-trace |
| Bootstrap CI95 | [-0.246, 0.855] | [0.265, 0.797] | [-0.591, 0.228] |
| Worst trace | -3.056 | -3.524 | -0.468 |
| Mean recovery | 0.176 | 0.073 | -0.103 |

The tighter clip improves the lower bootstrap bound, but it materially reduces
the median and worsens the worst trace. Treat tight clip as Granite-specific
tail-control evidence unless a new model-specific selection rule is added.

Source table: `artifacts/scale_cvar_clip/deepseek_tight_clip_comparison.csv`.

## Source Runs

- loose: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z`
- tight: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`
- tight hold-out: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z`
- tight final positive trace: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z`
- DeepSeek tight clip: `experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z`
- baseline: `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
