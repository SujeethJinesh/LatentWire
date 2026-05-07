# HBSM Local Sensitivity Runner

Decision: `RESOURCE_LIMITED_HBSM_B1_PACKET_WRITTEN_NOT_PROMOTABLE`

This is a resource-limited local HBSM B1 packet. It cannot promote B1.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompt: `hrsmoke_0001`
- Input tokens: `8`
- Layers: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Load seconds: `14.11`
- Baseline forward seconds: `89.02`
- Checker OK: `True`
- Checker decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`

| Layer | Symmetric KL drift |
|---:|---:|
| 0 | 0.00044458249 |
| 1 | 0.0055582314 |
| 2 | 0.0057776608 |
| 3 | 0.010679769 |
| 4 | 0.0064839497 |
| 5 | 0.027301367 |
| 6 | 0.0032552567 |
| 7 | 0.0095941052 |
