# C2 Scale / CVaR / Clip

Status: `PROMOTE_GRANITE_CLIP_RETUNE_SMOKE`.

The existing ParoQuant baseline uses:

- group size: 128
- rotations: 8
- scale clip: `[0.25, 4.0]`
- pairing: deterministic independent high-low within each group

No new scoring was run. This artifact only prepares candidate clip configs and
a CVaR-tail objective for a future smoke run.

## Why This Is Live

Granite ParoQuant is a positive method overall, but its included traces contain
one severe negative outlier:

- Granite median recovery: 0.754
- Granite CI95: [0.477, 1.004]
- Granite worst included trace: -26.371
- Granite 25% CVaR over included recoveries: -12.947

Nemotron ParoQuant is already strong and tail-stable:

- Nemotron median recovery: 1.047
- Nemotron CI95: [1.007, 1.292]
- Nemotron worst included trace: 0.960
- Nemotron 25% CVaR: 0.962

That pattern makes clip/CVaR tuning useful for Granite tail reduction and only
optional for Nemotron robustness confirmation.

## Candidate Clips

The candidate clips are `[0.125, 8.0]`, `[0.25, 4.0]`, and `[0.5, 2.0]`.
The middle candidate is the current ParoQuant baseline and must be present in
every smoke comparison.

## Objective

Use confirmation traces, not calibration traces:

`J(theta) = median_loss(theta) + lambda * CVaR_tail(theta) + mu * overhead(theta)`

Operational smoke gates:

- pass if median recovery beats baseline ParoQuant by at least 0.05;
- or CI lower improves by at least 0.10;
- or worst-trace / CVaR improves without median recovery loss greater than 0.05.

## Caveat

The current runner supports clip arguments, but not an explicit
calibration/confirmation split. Either add split support before claiming
DriftRot, or run full smoke only as an exploratory screen and evaluate the split
offline from the resulting score cache.

