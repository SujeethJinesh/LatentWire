# E3 Composition Result

Created: 2026-05-26T12:00:00Z

## Result Packet

`experimental/outlier_migrate/phase9/results/om_stage1_e3_granite_20260526T034442Z`

Checker decision: `KILL_E3_SUB_ADDITIVE`

## Headline Numbers

Positive-static-gap traces included: 8/12.

| Regime | Median recovery | 95% CI |
|---|---:|---:|
| ParoQuant W4A16 | 0.754 | [0.477, 1.004] |
| M11b top-10 | 0.241 | [-0.727, 0.495] |
| ParoQuant + M11b top-10 | 0.565 | [0.0029, 0.999] |
| ParoQuant + random top-10 | 0.798 | [0.640, 0.997] |

Composition minus best individual: -0.189.
Composition minus M11b top-10: +0.324.
Composition minus ParoQuant: -0.189.
Composition minus random control: -0.233.

## Decision

E3 does not become the paper headline. The composition improves over M11b
alone, but it trails ParoQuant alone and trails the random matched-budget
control. The useful paper interpretation is negative and scoped: on
Granite-Small, rotation appears to absorb most of the recoverable benefit, and
decode-time budget protection does not add value on top of ParoQuant under this
composition protocol.

Nemotron E3 was deferred for throughput under the 15 GPU-hour cap, as
pre-authorized by the revised Stage 1 order.

## Next Step

Proceed to E2 narrowed cross-prompt replication using the documented MATH-500
fallback because GPQA-Diamond access is gated in this environment.
