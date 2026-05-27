# Architectural Decision-Rule Analysis

Timestamp: 2026-05-27T19:45Z

## Decision

The V1/V2 vetting results support a conservative architecture-aware selection rule, not a universal M11b positive-method claim.

## Evidence

- Granite-Small: ParoQuant recovers 0.754 on the validated Granite packet; M11b top-5 is 0.449 but wide-CI, and top-10 is weaker. ParoQuant+M11b composition is sub-additive.
- Nemotron-3: M11b top-10 recovers 0.815 with CI [0.254, 0.923] and beats static top-10 by 0.220. This remains the strongest channel-set positive result.
- DeepSeek-R1-Distill: M11b top-10 median is 0.335 with CI [-0.409, 0.494], and static top-10 median is 0.377. This is ambiguous, not PASS.
- Falcon-H1: M11b top-10 median is 0.044 with CI [-0.144, 0.203], and static top-10 median is -0.025. This is ambiguous, not PASS.
- V1 ParoQuant-on-Nemotron was deferred after 2/12 traces because projected runtime exceeded the 5 GPU-hour cap; no ParoQuant-on-Nemotron claim is authorized.

## Rule Added

The paper now presents a model-local gate:

1. Measure strict set-leaving at the target horizon.
2. Validate budgeted EMA against a matched static top-10 control before claiming M11b.
3. Compare rotation directly when a rotation baseline is available.
4. Avoid claiming a channel-set positive method when high drift is present but the matched-budget gate remains ambiguous.

## Paper Update

Added `experimental/outlier_migrate/paper/sections/decision_rule.tex` and integrated it into the main paper after the composition result. The appendix now includes V1/V2 provenance and DeepSeek/Falcon top-10 rows.

## Next Queue Entry

Run the M-PRED scoop check before any new positive-method experiment.
