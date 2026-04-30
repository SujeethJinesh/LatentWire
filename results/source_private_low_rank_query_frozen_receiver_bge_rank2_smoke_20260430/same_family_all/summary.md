# Learned Synonym Dictionary Direction

- direction: `same_family_all`
- pass gate: `False`
- train/eval families: `all -> all`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_low_rank_query`
- contrastive negative sources: `2`
- contrastive rank: `2`
- receiver effective rank: `2`
- min decision score: `0.3`
- exact eval surface overlap count: `44`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.500 |
| 4 | `False` | 0.312 | 0.250 | 0.250 | 0.062 | 0.000 | 1.000 | 0.500 |
