# V1 ParoQuant-on-Nemotron Decision Note

Date: 2026-05-28T13:22Z

## Result

V1 ParoQuant-on-Nemotron completed in:

`experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z`

Checker:

`experimental/outlier_migrate/phase9/check_om_v1_paroquant_nemotron.py`

Decision: `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`

Headline numbers:

- Median recovery: 1.047
- CI95: [1.007, 1.292]
- Included traces: 10/12
- M11b Nemotron top-10 reference median: 0.815
- ParoQuant minus M11b top-10: +0.232

## Interpretation

This is a headline-changing baseline-vetting result. ParoQuant no longer looks
like only the Granite-positive baseline; it also beats the strongest Nemotron
M11b result by a large margin under the V1 checker.

The positive-method framing should move from "M11b is the Nemotron-specific
remedy" toward "rotation is the strongest observed remedy on both measured
Granite and Nemotron regimes, while channel-set policies remain useful
diagnostically and may still require architecture-local rescue on Falcon and
DeepSeek."

## Action Taken

Falcon HYST smoke was not launched after V1, despite being ready, because the
queue explicitly required pausing when V1 revealed ParoQuant dominance on
Nemotron.

## Next Decision

Before spending more GPU on Falcon HYST, choose whether the next positive-method
gate should:

1. verify rotation/ParoQuant on DeepSeek and Falcon, or
2. continue Falcon-specific channel-set rescue with HYST/LayerKeep/SURFACE.

The current evidence favors reassessing the queue under a rotation-dominant
positive-method story.
