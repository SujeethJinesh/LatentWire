# Source-Private Learned Syndrome Smoke

- date: `2026-04-29`
- gate: `source_private_learned_syndrome_smoke_20260429`
- status: passed as synthetic learned side-information smoke; not yet a headline claim
- scale rung: micro/strict-small method smoke on Mac-local CPU

## Current Readiness

The paper now has a plausible fifth technical contribution candidate:
**learned syndrome packets with decoder side information**. This directly
addresses the reviewer concern that the current diagnostic packet is too
hand-coded. It does not yet replace the main diagnostic-packet method because
the first gate is synthetic and uses frozen generated latent/candidate features.

## Method

The source observes a private noisy projection of the correct candidate latent.
The target has public candidate latents and a target prior. A small ridge-fitted
encoder maps the private source observation to a predicted candidate latent,
then emits a compact binary syndrome using random hyperplane signs. The target
decodes by Hamming distance against candidate-side codes.

This matches the Wyner-Ziv/decoder-side-information story:

- source-only observation: private evidence correlated with the correct
  candidate;
- decoder side information: public candidate pool and target prior;
- message: rate-capped learned syndrome at `1/2/4/8` bytes;
- distortion: candidate selection error.

## Results

Seed `29`, train/eval `512/256`, four candidates:

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full text |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `True` | `0.820` | `0.250` | `0.281` | `+0.539` | `1.000` |
| 2 | `True` | `0.949` | `0.250` | `0.262` | `+0.688` | `1.000` |
| 4 | `True` | `0.992` | `0.250` | `0.262` | `+0.730` | `1.000` |
| 8 | `False` | `1.000` | `0.250` | `0.305` | `+0.695` | `1.000` |

Seed `30`, train/eval `512/256`, four candidates:

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full text |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `True` | `0.797` | `0.250` | `0.281` | `+0.516` | `1.000` |
| 2 | `True` | `0.902` | `0.250` | `0.266` | `+0.637` | `1.000` |
| 4 | `False` | `0.988` | `0.250` | `0.305` | `+0.684` | `1.000` |
| 8 | `False` | `1.000` | `0.250` | `0.309` | `+0.691` | `1.000` |

The important outcome is the low-rate frontier: `1-2` byte learned syndromes
pass on both seeds, and seed29 also passes at `4` bytes. Higher budgets are not
automatically safer because source-free binary/text controls can occasionally
match candidate codes above the allowed tolerance.

## Controls

Each budget includes:

- target-only;
- zero-source;
- shuffled-source;
- answer-masked source;
- random same-byte;
- target-derived sidecar;
- answer-only;
- matched-byte structured text;
- wrong-projection source;
- full text oracle.

The pass rule requires matched learned syndrome to beat the best no-source row
by at least `15` points, all source-destroying controls to stay within
`target_only + 0.05`, and the full-text oracle to reach `1.000`.

## Interpretation

This is the first positive learned communication gate in the source-private
portfolio. It is materially stronger than another prompt-compliance source
emitter row because the message is learned from source observations and decoded
against target-side candidate information.

However, it remains a smoke result. It should be described as a candidate
method contribution, not as final ICLR evidence, until it clears one harder
surface:

1. learned syndrome over real tool-trace features;
2. learned syndrome over cached model hidden/candidate features;
3. cross-family source/target feature mismatch with the same controls.

## Artifacts

- `scripts/run_source_private_learned_syndrome_smoke.py`
- `tests/test_run_source_private_learned_syndrome_smoke.py`
- `results/source_private_learned_syndrome_smoke_20260429/summary.json`
- `results/source_private_learned_syndrome_smoke_20260429_seed30/summary.json`

## Decision

Promote learned syndrome packets to the top candidate for a stronger method
contribution. The next exact gate is a real-feature learned syndrome row on the
tool-trace benchmark, preserving the same controls and 1/2/4-byte rate frontier.
