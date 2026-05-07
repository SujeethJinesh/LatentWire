# SSQ-LR All-Layer Scout

Decision: `RESOURCE_LIMITED_SCOUT_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`

This is a metrics-only resource-limited local scout. It cannot promote SSQ-LR S1.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompts: `hrsmoke_0001, hrsmoke_0002, hrsmoke_0003, hrsmoke_0004, hrsmoke_0005, hrsmoke_0006, hrsmoke_0007, hrsmoke_0008, hrsmoke_0009, hrsmoke_0010, hrsmoke_0011, hrsmoke_0012`
- Max input tokens: `8`
- SSM layers scanned: `4`
- Passing layers: `3` / `3`
- Selected S1 ratio: `2.561113`

| Layer | Max-abs ratio | Std ratio | Kurtosis ratio | Local pass |
|---:|---:|---:|---:|---|
| 0 | 2.6714 | 1.8736 | 1.8580 | `True` |
| 12 | 1.0844 | 2.1811 | 0.2745 | `True` |
| 18 | 1.7701 | 1.6991 | 0.9805 | `False` |
| 30 | 3.0383 | 4.1043 | 0.3852 | `True` |
