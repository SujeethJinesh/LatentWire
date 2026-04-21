# Toy Test-Before-Repair Bridge

- Candidate pool with one output-corrupted near miss and one semantically noisy route trace.
- Repair-only spends its repair pass immediately on the highest-surface candidate.
- Test-before-repair runs discriminative checks on the pool first, then spends the same repair budget only when needed.

- Seed: `21`
- Examples: `192`
- Pool size: `6`
- Chain length: `4`
- Test threshold: `0.6900`
- Repair threshold: `0.5500`

| Method | Accuracy | Oracle accuracy | Repair app. rate | Repair change rate | Test pass rate | Help vs repair-only | Harm vs repair-only | Change vs repair-only | Bytes est. | Test bytes | Repair bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair_only | 0.0312 | 0.0312 | 0.0000 | 0.0000 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 61.4792 | 0.0000 | 0.0000 |
| test_before_repair | 0.9531 | 0.9531 | 0.0000 | 0.0000 | 0.2865 | 0.9219 | 0.0000 | 0.9219 | 368.6146 | 368.6146 | 0.0000 |
| oracle | 1.0000 | 1.0000 | 0.9844 | 0.0000 | 0.0104 | 0.9688 | 0.0000 | 0.9688 | 428.9896 | 368.6146 | 60.3750 |

## Severity Subgroups

| Severity level | Count | Repair-only acc. | Test-before-repair acc. | Oracle acc. | Test-before-repair pass rate |
|---|---:|---:|---:|---:|---:|
| 0 | 64 | 0.0312 | 0.9844 | 1.0000 | 0.6562 |
| 1 | 64 | 0.0312 | 0.8750 | 1.0000 | 0.1250 |
| 2 | 64 | 0.0312 | 1.0000 | 1.0000 | 0.0781 |

## Interpretation

The route-noisy candidate is the hard case: it is internally consistent enough that output-only repair does not fix it, but the test suite detects the semantic drift before repair is spent.
