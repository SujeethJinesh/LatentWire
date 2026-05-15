# Vacation Decision: M10 12-Trace Granite-Small Slice

## Situation

The committed M10 preregistration permits a deterministic 12-trace vacation
revision if throughput is more than 2x slower than estimated or the default
configuration approaches OOM. The immediately preceding M2 Granite-Small run
required the same fixed 12-trace slice after the full 24-trace packet OOMed
entering dynamic scoring, and the completed 12-trace M2 packet still took more
than a day of wall-clock time because dynamic regimes repeatedly reload the
model.

M10 has more dynamic score arms than M2: static SmoothQuant plus M10,
midpoint matched-cost, and random-bin scale assignment. A full 24-trace M10
run would likely exceed the vacation-mode priority of landing interpretable
results and keeping the paper current.

## Options Considered

1. Run the full 24-trace M10 packet first.
2. Run the fixed 12-trace vacation slice used by M2, preserving prompt indices
   `0-11` and unchanged decision thresholds.
3. Skip M10 based on the M2 random-bin failure pattern.

## Decision

Use option 2. Run M10 on the deterministic first-12 trace slice, with the
same thresholds and all preregistered controls. This preserves the scientific
question while limiting wall-clock risk. The packet will include
`vacation_adaptation.json` and the checker requires the reduced-slice label.

## Invalidates If

The human requires a full 24-trace M10 packet before interpreting M10, or if
reviewers reject the 12-trace vacation slice as insufficient for the Phase 9
method comparison.
