# Granite DriftRot Clip/CVaR Confirmation Screen

Status: `PROMOTE_TIGHT_CLIP_FINAL_POSITIVE_TRACE`.

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

This result must still be framed carefully. Config/clip retuning is Tier 2 in
the novelty table, not the headline method. The next valid gate is:

1. run the last remaining recoverable Granite trace, prompt `8`, so that all
   positive-gap Granite traces have tight-clip scores;
2. integrate tight clip as a tail-control DriftRot candidate, not as a generic
   claim that hyperparameter tuning beats ParoQuant;
3. test the idea on another model/tail surface only after the Granite positive
   trace table is complete.

## Source Runs

- loose: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z`
- tight: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`
- tight hold-out: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z`
- baseline: `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
