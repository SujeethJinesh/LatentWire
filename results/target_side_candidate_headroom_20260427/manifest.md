# Target-Side Candidate Headroom Manifest

- date: `2026-04-27`
- status: `canonical_live_target_side_pool_saturated`
- scale-up rung: source-surface discovery

## Command

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set svamp70_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=old_live,note=canonical_sidecar_saturated \
  --target-set svamp70_holdout=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json,role=old_holdout,note=weak_one_id_signal \
  --target-set svamp70_chal241_310=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_clean_surface \
  --target-set svamp70_chal311_380=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_weak_surface \
  --target-set svamp70_chal171_240=path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_weak_surface \
  --target-set gsm70_qwen25math=path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json,role=next_candidate,note=gsm_surface \
  --date 2026-04-27 \
  --output-json results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.json \
  --output-md results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.md
```

## Result

| Surface | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool |
|---|---:|---:|---:|---:|---:|
| `svamp70_holdout` | 8/70 | 8/70 | 28/70 | 20 | 2/2 |
| `svamp70_chal171_240` | 22/70 | 8/70 | 39/70 | 17 | 1/1 |
| `svamp70_chal241_310` | 10/70 | 5/70 | 23/70 | 13 | 1/4 |
| `svamp70_chal311_380` | 21/70 | 8/70 | 34/70 | 13 | 0/2 |
| `svamp70_live` | 21/70 | 13/70 | 33/70 | 12 | 0/6 |
| `gsm70_qwen25math` | 4/70 | 3/70 | 14/70 | 10 | 0/2 |

## Decision

Canonical `svamp70_live` is saturated for target-side side-information
decoders: none of its six clean source-only IDs have gold in the target-side
candidate pool. This explains why stricter target-side sidecars recover zero
clean live IDs.

Do not tune more numeric/likelihood/semantic sidecars on canonical `svamp70_live`
until a new candidate-surface generator exposes target-side alternatives
without source-only value leakage.

Next branch: candidate-surface generation. Generate target-side candidate pools
from target self-repair, stochastic target routes, or non-source candidate
decoders, then re-run the target-side headroom audit before testing any source
sidecar.

## Hashes

- `target_side_candidate_headroom.json`:
  `30f92ea80c55c330a96bc9771bef54f2b66532706366fb9af9342a43e1facf1d`
- `target_side_candidate_headroom.md`:
  `c2a729773f3d487224997924226a3be2b2b1651a39bbb17e691d674983e88237`
