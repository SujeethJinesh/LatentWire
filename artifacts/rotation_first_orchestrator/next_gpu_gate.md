# Next GPU Gate

Run rotation smoke before any channel rescue.

## G0 ParoQuant Falcon Smoke

Status: COMPLETE. Full 12-trace packet at
`experimental/outlier_migrate/phase9/results/om_v1_paroquant_falcon_20260528T154653Z`
returned `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE` with median recovery
0.381, CI95 [0.0645, 0.547], and +0.337 median margin over Falcon M11b
top-10.

Command:

`artifacts/rotation_first_orchestrator/command_paroquant_falcon_smoke.sh`

Pass gates:

- median recovery >= 0.20, or
- median recovery beats Falcon M11b top-10 by at least 0.10, or
- checker returns `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE`.

Falcon HYST/LAMBDA/BranchRot are now lower priority unless later DriftRot
analysis specifically needs a Falcon channel fallback.

## G1 ParoQuant DeepSeek Smoke

Status: COMPLETE. Full 12-trace packet at
`experimental/outlier_migrate/phase9/results/om_v1_paroquant_deepseek_20260528T162858Z`
returned `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES` with median recovery
0.756, CI95 [-0.246, 0.855], and +0.379 median margin over DeepSeek
static-top10.

Command:

`artifacts/rotation_first_orchestrator/command_paroquant_deepseek_smoke.sh`

Pass gates:

- checker returns `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES`, meaning
  ParoQuant beats the static-top10 reference median 0.376752614594403.

DeepSeek FISH/GATE are lower priority. DriftRot retuning remains live because
the negative lower CI shows tail risk even when the median is strong.

## G2 DriftRot Scale/CVaR/Clip

Next gate: Granite-focused clip/CVaR retune from
`artifacts/scale_cvar_clip/`.

Run only with the exact ParoQuant baseline present. A positive DriftRot claim
requires held-out confirmation, CI/tail improvement over ParoQuant, or median
gain over ParoQuant; do not present pure ParoQuant hyperparameter tuning as a
new method without confirmation.
