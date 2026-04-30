# Learned Synonym Dictionary Direction

- direction: `core_to_holdout`
- pass gate: `False`
- train/eval families: `core -> holdout`
- candidate atom view: `heldout_synonym`
- calibration atom view: `synonym_stress`
- candidate calibration: `all_public`
- text feature mode: `hf_mid_last_mean`
- receiver mode: `contrastive_low_rank_factor`
- contrastive negative sources: `2`
- contrastive rank: `16`
- low-rank factor epochs: `220`
- low-rank factor lr: `0.02`
- low-rank factor loss: `bce`
- low-rank factor seed: `947`
- receiver effective rank: `16`
- min decision score: `0.3`
- exact eval surface overlap count: `12`
- exact ID parity: `True`

| Budget | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Top knockout reduction | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.250 |
| 4 | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.250 |
