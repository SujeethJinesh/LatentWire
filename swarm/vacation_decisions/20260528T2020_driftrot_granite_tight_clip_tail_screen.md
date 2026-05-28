# Decision: DriftRot Granite Tight Clip Tail Screen

Date: 2026-05-28T20:20Z

Decision: `PROMOTE_TIGHT_CLIP_REPEAT_CONFIRMATION`

Runs:
- loose: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z`
- tight: `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`

Confirmation traces: `[4, 7, 9, 10]`

Results:
- baseline ParoQuant [0.25, 4.0]: median 0.640, worst -26.371
- loose clip [0.125, 8.0]: median 0.686, worst -79.783 -> killed
- tight clip [0.5, 2.0]: median 0.901, worst 0.601 -> promoted for repeat

Interpretation:
Tight clipping is a promising Tier-2 tail-control candidate. It fixes the known
Granite tail trace in this screen, but it cannot be a final DriftRot method
claim because the confirmation subset was inspected directly. The next valid
step is repeat confirmation, full 12-trace finalist evaluation, or use this as
motivation for a Tier-1 residual-correction/rotation-refresh gate.
