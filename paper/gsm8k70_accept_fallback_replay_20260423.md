# GSM8K70 Accept/Fallback Replay

Date: 2026-04-23

## Status

This is an offline accept/fallback replay over the existing
`dynalign_module_replace_residrank16` GSM70 artifacts. It is not yet a runtime
method result.

The replay promotes source-control-aware target fallback as the next live
method direction. Raw dynalign remains too seed-fragile to promote without a
frozen selector.

## Command

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_accept_fallback.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --candidate seed0=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --candidate seed3=.debug/gsm8k70_integrated_source_controls_20260423/seed3/dynalign_module_replace_residrank16.jsonl \
  --control zero_source=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16/source_controls/zero_source.jsonl \
  --control shuffled_source_salt0=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16/source_controls/shuffled_source_salt0.jsonl \
  --score-field selector_gap_min \
  --score-quantile 0.6 \
  --score-quantile 0.7 \
  --score-quantile 0.75 \
  --score-quantile 0.8 \
  --score-quantile 0.9 \
  --output-json results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.json \
  --output-md results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.md
```

`results/` is ignored by git, so this note records the artifact hashes:

```text
3ac363245ac963c4354175ae29ccfc454209ab3bb50b9489fef590da0b7330f9  results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.json
af42eb83c5c50b2443fb2793af6f8f4064df371aa1ab36f0e6cb32906a313370  results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.md
```

## Replay Contract

The replay chooses the candidate answer only when:

- the candidate has a valid numeric answer,
- the candidate numeric answer differs from target,
- the candidate numeric answer is not a degenerate long numeric string,
- `selector_gap_min` clears a quantile threshold calibrated on the seed-0
  candidate distribution.

Otherwise it falls back to the target-only answer.

`selector_gap_min` is label-free route telemetry from `selector_trace[*].score_gap`.
It is not an oracle score. The quantile choice is still post-hoc until the next
runtime run freezes it before evaluation.

## Readout

Target baseline: `4/70`.

| Policy | Gate | Seed 0 | Seed 3 | Zero/shuffled controls |
|---|---|---:|---:|---:|
| `target_only` | fail | `4/70`, `0/0/70` | `4/70`, `0/0/70` | `4/70`, accepts `0` |
| `numeric_changed` | fail | `8/70`, `6/2/62` | `2/70`, `1/3/66` | `4/70`, accepts `0` |
| `selector_gap_min_ge_q0p6_numeric_changed` | fail | `7/70`, `4/1/65` | `4/70`, `1/1/68` | `4/70`, accepts `0` |
| `selector_gap_min_ge_q0p7_numeric_changed` | pass | `7/70`, `3/0/67` | `5/70`, `1/0/69` | `4/70`, accepts `0` |
| `selector_gap_min_ge_q0p75_numeric_changed` | pass | `7/70`, `3/0/67` | `5/70`, `1/0/69` | `4/70`, accepts `0` |
| `selector_gap_min_ge_q0p8_numeric_changed` | pass | `6/70`, `2/0/68` | `5/70`, `1/0/69` | `4/70`, accepts `0` |
| `selector_gap_min_ge_q0p9_numeric_changed` | pass | `5/70`, `1/0/69` | `5/70`, `1/0/69` | `4/70`, accepts `0` |

Thresholds:

```text
q0.60  0.0287264883518219
q0.70  0.029237359762191772
q0.75  0.02981119602918625
q0.80  0.03003109246492386
q0.90  0.030762314796447754
```

The strongest offline policy is `q0.70`: it keeps `3` seed-0 wins with zero
target losses, turns seed 3 from target-negative to `+1/0`, and accepts no
zero-source or shuffled-source control examples.

## Interpretation

Alive:

- Control-safe selector-gated target fallback for same-family GSM70.
- `selector_gap_min` as a cheap route-confidence signal for deciding when to
  use the latent intervention.

Promoted:

- Runtime accept/fallback validation as the next method gate.

Weakened:

- Raw dynalign without selection. It has real seed-0 source dependence but
  unacceptable seed-repeat harms.

Still blocked:

- The policy is offline and threshold-calibrated after seeing seed-0 artifacts.
- Controls are seed-0 controls only; seed-3 controls were not run because raw
  seed 3 failed the live gate.
- This is same-family only; no strict cross-family falsification has cleared.

## Top Next Moves

1. Freeze `selector_gap_min_ge_q0p7_numeric_changed` as a runtime
   accept/fallback policy and rerun seed 0, seed 3, and matched source controls
   from scratch. This is the highest-value move because it directly tests
   whether the replayed positive survives as an actual method.
2. If runtime validation fails, implement a learned contrastive innovation
   connector that penalizes zero-source and shuffled-source deltas while
   allowing bounded matched-source updates. This is the next method branch with
   the cleanest reviewer-facing story.
3. Add accepted-ID provenance plus byte/latency accounting to the GSM70
   wrappers before widening. This improves reproducibility and makes any
   accepted-row result easier to defend.

## Verification

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_gsm8k_accept_fallback.py \
  tests/test_analyze_gsm8k_source_controls.py \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_build_gsm8k_contract_manifest.py
```

Result: `43 passed in 0.07s`.

## Decision

Do not claim this as the paper result yet. Treat it as a promoted gate: freeze
the q0.70 selector-gap accept/fallback policy, run it online with matched
controls, and only then decide whether to widen to larger frozen slices,
additional finite seeds, or one strict cross-family falsification pair.
