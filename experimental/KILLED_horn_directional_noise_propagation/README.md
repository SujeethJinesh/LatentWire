# KILLED: HORN Directional Noise Propagation

Date killed: 2026-05-07

Evidence reviewed at head: `5c31c3685b11502eb3d72c514602fe0019c8b980`

## What Was Tried

HORN tested whether hybrid attention/SSM boundaries have directional outlier
and noise-propagation asymmetry, motivating direction-aware activation
precision.

The live H2 scout injected FP4-equivalent noise around the H1a-selected boundary
direction on Granite Tiny and measured paired noisy-continuation drift.

## Why It Died

The preregistered H2 directional-noise contract failed:

- packet: `experimental/shared/results/horn_h2_noise_replay_scout_20260507/`
- raw gate status: `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
- directional drift ratio: `1.037`
- signed selected-direction lower bound: `0.324`
- selected-direction support fraction: `0.5`
- paired units: `6/6`
- hook-off max delta: `0.0`

The effect is near-null and support is split. This kills HORN as an active
standalone COLM positive-method branch under the current preregistered
hypothesis. Do not spend GPU time on HORN standalone.

## Audit Trail

Primary stop manifest:

- `experimental/horn/phase2/h2_noise_replay_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/horn/paper/reviewer_pack.md`

## Salvage Value

The scaffold remains useful for future work:

- boundary tensor capture and recomputation checks
- direction-label permutation controls
- H2/H3 follow-up contract checker
- negative/control appendix evidence for hybrid-boundary quantization claims

## Revival Condition

Revival requires a new preregistered full H2/H3 scope on a fresh surface before
any new rows are inspected, plus a concrete reason the near-null H1a/H2 Granite
Tiny scouts should reverse on larger prompts or models.

