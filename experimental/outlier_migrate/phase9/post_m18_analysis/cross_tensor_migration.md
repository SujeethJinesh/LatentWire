# Post-M18 Analysis 1: Cross-Tensor Migration Measurement
## Verdict
The requested K/V cache set-leaving measurement is **not identifiable from existing packets**. Existing OutlierMigrate packets contain activation-channel magnitudes across positions for Granite-Small, Nemotron-3, DeepSeek-R1-Distill-Qwen-1.5B, and Falcon-H1, but they do not contain K-cache or V-cache channel magnitudes at both an early and late decode position for those four models. The only K/V evidence currently present is the M18 Granite-Small endpoint evidence at decode position 10000.
## Available evidence
- M18 K/V endpoint evidence: `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z/kv_cache_channel_evidence.json`.
- M18 protected-set mapping: `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z/protected_sets.json`.
- It covers Granite-Small attention layers `[5, 15, 25, 35]` at selection position 10000. It does not include position-100 K/V top sets, so K/V set-leaving cannot be computed.
## Endpoint diagnostics from M18
| Layer | KIVI key top count | activation-mapped key count | Jaccard(KIVI key, activation-mapped key) | Jaccard(K top, V top) | Jaccard(random key, KIVI key) |
|---:|---:|---:|---:|---:|---:|
| 5 | 11 | 62 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 15 | 11 | 63 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 25 | 11 | 63 | 0.027777777778 | 0.047619047619 | 0.000000000000 |
| 35 | 11 | 58 | 0.000000000000 | 0.000000000000 | 0.000000000000 |

## Interpretation
The endpoint overlap is very low, which is consistent with activation-channel and key-cache salience not being trivially identical. This is only an endpoint diagnostic; it must not be described as K/V migration. A real cross-tensor migration measurement would require K and V channel magnitude snapshots at matching decode positions, minimally positions 100 and 10000, for each model.
