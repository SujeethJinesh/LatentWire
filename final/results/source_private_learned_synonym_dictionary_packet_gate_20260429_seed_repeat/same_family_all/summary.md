# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `True`
- train/eval families: `all -> all`
- candidate atom view: `synonym_stress`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- exact eval surface overlap count: `1024`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `True` | 0.938 | 0.250 | 0.250 | 0.688 | 0.629 | 1.000 | 1.000 |
