# OutlierMigrate Layer-Stratified Migration Analysis

Generated from existing Phase 0/1/2 activation packets. No Phase 3 intervention data is used.

Definitions:
- **strict set-leaving**: a channel in the position-100 top-1% set has final rank greater than the zero-based top-1% boundary rank.
- **within-set rank shuffling**: a channel stays inside the final top-1% set but moves by more than two rank positions.
- **original migration**: the preregistered metric, which counts any top-1% channel whose final rank differs by more than two positions.

Granite exposes Mamba-vs-attention mixer layer types; its MoE feed-forward is configured across blocks and is not separately isolatable from layer-output rows. Nemotron-3 exposes Mamba, attention, and MoE-only blocks via `hybrid_override_pattern`.

## Phase 0 Granite-4.0-H-Tiny

- Run dir: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`
- Model: `ibm-granite/granite-4.0-h-tiny`
- Positions: `[100, 500, 1000, 5000, 10000]`
- Overall original migration fraction: `0.817839` (CI95 `0.797266`, `0.836849`)
- Layer classification: `config.layer_types`

| Layer type | Layers | Strict set-leaving | Within-set shuffling | Original migration |
|---|---:|---:|---:|---:|
| `attention` | 4 | 0.618490 | 0.186198 | 0.798177 |
| `ssm_mamba` | 36 | 0.635995 | 0.174045 | 0.820023 |

## Phase 1 Granite-4.0-H-Small

- Run dir: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`
- Model: `ibm-granite/granite-4.0-h-small`
- Positions: `[100, 500, 1000, 5000, 10000, 20000]`
- Overall original migration fraction: `0.843166` (CI95 `0.833435`, `0.851143`)
- Layer classification: `config.layer_types`

| Layer type | Layers | Strict set-leaving | Within-set shuffling | Original migration |
|---|---:|---:|---:|---:|
| `attention` | 4 | 0.557165 | 0.282266 | 0.843242 |
| `ssm_mamba` | 36 | 0.567243 | 0.269676 | 0.843157 |

## Phase 2 Nemotron-3-Nano partial

- Run dir: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Positions: `[100, 500, 1000, 5000, 10000, 20000]`
- Overall original migration fraction: `0.820810` (CI95 `0.786533`, `0.854493`)
- Layer classification: `config.hybrid_override_pattern`

| Layer type | Layers | Strict set-leaving | Within-set shuffling | Original migration |
|---|---:|---:|---:|---:|
| `attention` | 6 | 0.563014 | 0.254630 | 0.832562 |
| `moe` | 23 | 0.526503 | 0.277778 | 0.820384 |
| `ssm_mamba` | 23 | 0.533280 | 0.264157 | 0.818170 |

## Interpretation Notes

- The strict set-leaving fraction is the most relevant number for static protected-channel lists, because it counts channels that leave the protected top-1% set entirely.
- Within-set shuffling remains relevant for scale refresh and ordering-sensitive schemes, but it is not by itself evidence that a static top-1% membership list has failed.
- Layer-type summaries average per-layer means equally, so very wide MoE blocks do not dominate the aggregate solely by parameter count.
