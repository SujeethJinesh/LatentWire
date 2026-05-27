# V2 M11b Falcon-H1 Result

Timestamp: 2026-05-27T19:25Z

## Run

- Experiment: V2, M11b top-10 on Falcon-H1-0.5B
- Run directory: `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z`
- Trace source: reused Stage 1 E1 Falcon traces from `experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/falcon_h1_0_5b`
- Runtime: 2026-05-27T14:34:40Z to 2026-05-27T19:20:07Z, approximately 4.76 GPU hours

## Checker Decision

`AMBIGUOUS_V2_M11B_FALCON`

Artifact validation passed. The checker did not find a PASS regime because high-budget M11b did not clear the preregistered median and CI thresholds.

## Headline Numbers

Recovery is measured against the static 1% baseline on 12 deterministic AIME-2025 traces.

| Regime | Median recovery | CI95 | Included traces | No-gap fraction |
|---|---:|---:|---:|---:|
| M11b top-1 | -0.099 | [-0.251, 0.017] | 12/12 | 0.000 |
| M11b top-5 | 0.070 | [-0.077, 0.102] | 12/12 | 0.000 |
| M11b top-10 | 0.044 | [-0.144, 0.203] | 12/12 | 0.000 |
| Static top-10 | -0.025 | [-0.126, 0.172] | 12/12 | 0.000 |

## Interpretation

Falcon does not replicate the Nemotron M11b top-10 PASS. Top-10 M11b has a small positive median but a CI crossing zero and does not beat the preregistered PASS threshold. Combined with the DeepSeek V2 ambiguity, the current evidence supports an architecture-dependent remedy story rather than an architecture-agnostic M11b positive method.

## Queue Implication

V1/V2 baseline vetting is complete. Next non-GPU step is the architectural decision-rule analysis, followed by M-PRED only after the required scoop check.
