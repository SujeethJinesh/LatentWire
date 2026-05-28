# Next GPU Gate

Run rotation smoke before any Falcon channel rescue.

## G0 ParoQuant Falcon Smoke

Command:

`artifacts/rotation_first_orchestrator/command_paroquant_falcon_smoke.sh`

Pass gates:

- median recovery >= 0.20, or
- median recovery beats Falcon M11b top-10 by at least 0.10, or
- checker returns `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE`.

If pass, lower priority of Falcon HYST/LAMBDA/BranchRot. If weak, resume Falcon
fallback queue with HYST margin-5 first.

## G1 ParoQuant DeepSeek Smoke

Command:

`artifacts/rotation_first_orchestrator/command_paroquant_deepseek_smoke.sh`

Pass gates:

- checker returns `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES`, meaning
  ParoQuant beats the static-top10 reference median 0.376752614594403.

If pass, rotation-dominant four-model story strengthens after Falcon is known.
If weak, DeepSeek FISH/GATE or DriftRot retuning remains live.
