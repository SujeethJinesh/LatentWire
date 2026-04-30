# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `True`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `heldout_synonym`
- candidate calibration: `all_public`
- exact eval surface overlap count: `1024`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 1.000 | 0.250 | 0.250 | 0.750 | 0.695 | 1.000 | 0.938 |
