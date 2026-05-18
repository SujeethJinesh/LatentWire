# Vacation Decision: Restart M11b With Reused Dense Activation Evidence

## Situation

The M11b Granite-Small run
`experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_fullgrid_20260518T025900Z`
was launched without `--reuse-activation-run-dir`, so it began regenerating the
full 100-position activation grid. After roughly 12 minutes it had produced
only a partial activation artifact and was on track to spend several GPU-hours
duplicating calibration evidence already present in the completed M11 packet.

The completed M11 packet at
`experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z`
has the same model snapshot (`b8c0982bab7fde4eb48110f5a069527c008fab39`),
the same 12-trace vacation-mode prompt slice, the same prompt SHA, and the
same 100-position update grid from 100 through 10000.

## Options Considered

1. Let the duplicate capture continue. This would be scientifically valid but
   waste GPU time and delay the next result.
2. Stop the duplicate capture and rerun M11b using the already-complete M11
   dense activation artifact, while still producing a fresh M11b result packet
   and checker output.
3. Skip M11b. This would violate the current mechanism-distinguishing queue.

## Decision

I chose option 2. This preserves the M11b scientific question because the
reused artifact is exactly the calibration evidence M11b requires; the runner
already supports this reuse path and records the source artifact in the new
packet manifest. The interrupted duplicate run is treated as superseded
infrastructure work, not as a scientific M11b result.

## What Would Invalidate This Decision

If the prior M11 activation artifact is later found not to match the M11b
prompt slice, model snapshot, capture positions, or capture semantics, then
the reuse run should be discarded and M11b should be rerun from fresh
activation capture.
