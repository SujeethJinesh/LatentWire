# SSQ-LR All-Layer Scout

Decision: `RESOURCE_LIMITED_SCOUT_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`

This is a metrics-only resource-limited local scout. It cannot promote SSQ-LR S1.

- Model: `ibm-granite/granite-4.0-h-tiny`
- Prompts: `hrs1b_0001, hrs1b_0002, hrs1b_0003, hrs1b_0004, hrs1b_0005, hrs1b_0006, hrs1b_0007, hrs1b_0008, hrs1b_0009, hrs1b_0010, hrs1b_0011, hrs1b_0012`
- Max input tokens: `8`
- SSM layers scanned: `4`
- Passing layers: `3` / `3`
- Selected S1 ratio: `2.459378`

| Layer | Max-abs ratio | Std ratio | Kurtosis ratio | Local pass |
|---:|---:|---:|---:|---|
| 0 | 2.9619 | 1.9086 | 2.2293 | `True` |
| 12 | 1.0336 | 2.0522 | 0.2676 | `True` |
| 18 | 1.6581 | 1.6322 | 0.7200 | `False` |
| 30 | 2.8625 | 4.1476 | 0.2951 | `True` |
