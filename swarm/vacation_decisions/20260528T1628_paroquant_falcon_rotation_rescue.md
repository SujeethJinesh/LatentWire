# Decision: ParoQuant Falcon Rotation Rescue

Date: 2026-05-28T16:28Z

Run:
`experimental/outlier_migrate/phase9/results/om_v1_paroquant_falcon_20260528T154653Z`

Decision: `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE`

Result:
- median recovery: 0.38061550869015837
- CI95: [0.0645493637145208, 0.5474138321928452]
- included traces: 12/12
- margin vs Falcon M11b top-10: +0.3366757784989901

Interpretation:
ParoQuant-style rotation rescues Falcon relative to the failed/weak channel
policies. Falcon channel fallbacks (HYST/LAMBDA/BranchRot) remain available,
but their priority drops because rotation now clears the preregistered Falcon
rescue threshold.

Operational note:
The first runner attempt completed all GPU scoring and then failed during
metrics summarization because the Falcon checker wrapper did not expose
`bootstrap_median`. The wrapper was fixed, and metrics/checker artifacts were
repaired from the completed BF16/static/ParoQuant score caches without rerunning
GPU scoring.

Next gate:
Run ParoQuant DeepSeek smoke/full packet. If DeepSeek passes, the
rotation-dominant four-model story becomes the main framing; if it fails,
DeepSeek-specific DriftRot/FISH/GATE work remains live.
