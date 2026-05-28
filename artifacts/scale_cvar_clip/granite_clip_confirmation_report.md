# Granite DriftRot Clip/CVaR Confirmation Screen

Status: `PROMOTE_TIGHT_CLIP_REPEAT_CONFIRMATION`.

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

## Caveat

This result must not be presented as a final positive method yet. Config/clip
retuning is Tier 2 in the novelty table, and this run directly inspected the
confirmation subset. The next valid gate is one of:

1. repeat the tight clip on a different frozen split;
2. run the full 12-trace tight-clip packet despite cost;
3. test tight clip on another model/tail surface;
4. use this as motivation for a Tier-1 residual-correction or rotation-refresh
   gate.

## Source Runs

- loose: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z`
- tight: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`
- baseline: `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
