# Executive Summary

V1 decision: `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`.

Key numbers from `experiments/v1_paroquant_nemotron/check_script_output.txt`
and `tables/v1_vs_m11b_nemotron.csv`:

- Median recovery: 1.047
- CI95: [1.007, 1.292]
- Margin vs M11b top-10: +0.232
- Included traces: 10/12

Meaning for paper framing: ParoQuant/rotation is no longer only the Granite
positive baseline. It is also stronger than the prior Nemotron M11b top-10
positive result on the same BF16/static trace baseline. The paper should now
consider a rotation-dominant framing or a regime-aware protocol where rotation
is the first remedy to test.

Still unresolved:

- DeepSeek/Falcon rotation results are not measured.
- Falcon channel rescue remains live but paused behind the V1 gate.
- M-SURFACE remains a diagnostic, not yet run after V1.
- RISKGUARD remains deferred because its offline trigger was not robust under
  leave-one-trace-out checks.
