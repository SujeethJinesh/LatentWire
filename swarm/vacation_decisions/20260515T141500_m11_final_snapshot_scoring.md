# Vacation Decision: M11 Final-Snapshot Scoring

## Situation

The M11 preregistration specifies a protected-set update every 100 decode
positions. A literal implementation that reloads or requantizes the
Granite-4-H-Small model at every 100-position boundary would require roughly
100 model state transitions per trace. The immediately preceding M2 and M10
measurements show that even 3-4 transitions per trace are expensive enough to
dominate the vacation sprint wall-clock.

## Options Considered

1. Implement literal online switching every 100 positions.
2. Skip M11 as infeasible and move directly to M17.
3. Preserve the EMA decision surface by computing the full 100-position EMA
   protected-set trajectory, then score the final position-10000 protected
   snapshot on the preregistered 512-token endpoint window.

## Decision

Use option 3 for the Granite-Small M11 gate. The runner captures activation
evidence at every 100-position update point, constructs the full EMA
trajectory for alpha values 0.1, 0.3, and 0.5, records the trajectory, and
scores the final protected set selected at position 10000. This tests whether
EMA smoothing selects a useful endpoint protected set while avoiding the hard
boundary-switching path that M2 exposed as harmful and too slow.

This is a vacation-mode V2 adaptation. It preserves the scientific question
most relevant to the current paper's endpoint metric, but it does not measure
the cost or cache-history effects of changing weights online at every update.

## Invalidation Condition

If the human requires literal online protected-set switching before interpreting
M11, the M11 result should be treated as a final-snapshot proxy only, not as the
full online M11 intervention.
