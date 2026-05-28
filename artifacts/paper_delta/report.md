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

