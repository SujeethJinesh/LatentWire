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

Command:

`artifacts/rotation_first_orchestrator/command_paroquant_deepseek_smoke.sh`

Pass gates:

- checker returns `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES`, meaning
  ParoQuant beats the static-top10 reference median 0.376752614594403.

If pass, rotation-dominant four-model story strengthens after Falcon is known.
If weak, DeepSeek FISH/GATE or DriftRot retuning remains live.
