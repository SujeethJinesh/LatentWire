# C9 Paper Delta Report

## Status

The V1 ParoQuant-on-Nemotron result changes the paper queue. Rotation is no
longer only the Granite positive baseline: ParoQuant-style rotation also beats
the previous Nemotron M11b top-10 result on the same BF16/static baseline.

## Source Evidence

- V1 pack: `artifacts/external_review_pack/v1_rotation_delta_pack_20260528_1506/`
- V1 decision: `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`
- Nemotron ParoQuant median recovery: `1.0470852994064899`
- Nemotron ParoQuant CI95: `[1.0071831033257677, 1.2920493786337606]`
- Margin over Nemotron M11b top-10: `0.2323455005030597`
- Granite ParoQuant median recovery: `0.7537768488911776`
- Granite ParoQuant CI95: `[0.4770444524050311, 1.003769554323054]`

## Framing Implication

The strongest current framing is rotation-first but still conditional:

1. If ParoQuant also passes DeepSeek and Falcon, the paper should become a
   rotation-dominant mechanism/protocol paper: channel identity drifts, and
   rotation removes most of the basis dependence that channel-set protection
   assumes.
2. If ParoQuant fails Falcon or DeepSeek, the paper should remain
   regime-aware: rotation is first-line, and branch/surface/channel rescue is
   needed only where rotation fails.
3. DriftRot should be framed as an attempted improvement over static
   ParoQuant only if it beats ParoQuant on held-out confirmation traces,
   improves the CI lower bound or tail risk, or rescues a model where
   ParoQuant fails.

## Recommended Next Paper Delta

- Promote rotation to the first positive-method baseline in the abstract and
  contribution list.
- Do not call ParoQuant our method.
- Use "ParoQuant-style rotation" or "rotation baseline" for the reproduced
  result.
- Introduce DriftRot only as a candidate family: drift-aware rotation
  calibration, surface/branch selection, and residual correction.
- Keep Falcon and DeepSeek unresolved until their ParoQuant smoke results
  land.

## Gate

`method_gate.json` sets the immediate writing gate to
`ROTATION_FIRST_PENDING_FALCON_DEEPSEEK`.


## Novelty Lock-In: Rotation Positioning Paragraph

Prior rotation methods construct static calibration-time transforms, learned
rotations, or runtime smoothing rules that improve low-bit inference by changing
the basis in which quantization error appears. Our question is different: under
long reasoning decode, channel identity itself drifts, so we first measure when
channel-set protection becomes ill-posed, then test whether rotation removes the
basis dependence or whether drift-aware selection, surface choice, branch-local
rotation, or rotated-basis residual correction can improve on a fixed ParoQuant
baseline. ParoQuant is therefore reported as a strong baseline, not as our
method.

## DriftRot Candidate Novelty Ranking

Tier 1 candidates:

1. Rotated-basis residual correction: `y = x W_pq + x_P (W_fp - W_pq)_P`, with
   `P` selected by post-rotation residual error under long-decode traces.
2. Decode-adaptive rotation refresh, only if early/late covariance or range
   drift is measurable and confirmation traces improve over static ParoQuant.
3. Surface-selected rotation/protection, using measured internal-surface drift
   to distinguish block-output instability from internal persistence.
4. Branch-local Falcon rotation, only if branch-local drift/range is materially
   lower than post-mixer drift/range.

Tier 2 candidates:

1. CVaR/tail-aware calibration.
2. Static config/clip retuning.

Config/clip retuning must not be the headline method. It is useful as a tail
screen or as evidence that static ParoQuant is already close to saturated.

## Guardrails for DriftRot-Config / CVaR

Any DriftRot-Config or CVaR result must include the original ParoQuant config as
baseline, use a calibration/confirmation split, give a drift/tail-based reason
for selecting the config, and explicitly report if gains disappear on
confirmation traces. Without those conditions, it is ParoQuant hyperparameter
screening rather than a new method.

## Residual Correction Definition

Rotated-basis residual correction is defined as:

`y = x W_pq + x_P (W_fp - W_pq)_P`

The selected set `P` must be chosen using post-rotation residual error under
long-decode traces, not original-basis channel magnitude. Any result must report
HBM bytes/token, working-set size, and estimated pJ/token.

## Surface-Distinction Wording

Use the following Quamba2-safe wording for M-SURFACE:

"Quamba2-style internal persistence does not imply stable block-output top-k
protection. We therefore measure the surface at which protection is applied
before concluding whether the channel identity assumption fails internally or
only after block-level mixing."

Do not imply Quamba2 is wrong unless internal-surface measurements show that.

## Title / Abstract Candidate Update

Preferred title:

"Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and When
Rotation/Residual Correction Helps"

Avoid:

"A Better Rotation Quantizer for Reasoning LLMs"
