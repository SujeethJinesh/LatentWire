# HBSM Local Sensitivity Runner

Decision: `RESOURCE_LIMITED_HBSM_B1_PACKET_WRITTEN_NOT_PROMOTABLE`

This is a resource-limited local HBSM B1 packet. It cannot promote B1.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompts: `['hrsmoke_0001', 'hrsmoke_0002']`
- Input tokens: `8`
- Layers: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Load seconds: `9.92`
- Baseline forward seconds: `79.50`
- Checker OK: `True`
- Checker decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`

| Layer | Symmetric KL drift |
|---:|---:|
| 0 | 0.0020578391 |
| 1 | 0.0040636142 |
| 2 | 0.035678787 |
| 3 | 0.0077762371 |
| 4 | 0.012432819 |
| 5 | 0.032428432 |
| 6 | 0.0085097566 |
| 7 | 0.0080630165 |
