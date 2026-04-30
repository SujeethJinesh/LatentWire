# Learned Synonym Dictionary Direction

- direction: `holdout_to_core`
- pass gate: `True`
- train/eval families: `holdout -> core`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_low_rank_query`
- contrastive negative sources: `2`
- contrastive rank: `8`
- receiver effective rank: `8`
- min decision score: `0.3`
- exact eval surface overlap count: `76`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.375 | 0.250 | 0.250 | 0.125 | 0.031 | 1.000 | 0.875 |
| 4 | `True` | 0.625 | 0.250 | 0.250 | 0.375 | 0.219 | 1.000 | 0.875 |
