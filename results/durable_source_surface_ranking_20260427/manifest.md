# Durable Source-Surface Ranking Manifest

Date: 2026-04-27

## Status

`primary_surface_selected`

The durable ranker selects `svamp70_live` as the primary next method gate when
ranking existing `source_contrastive_target_set.json` artifacts by clean
source-only IDs rather than raw source-only IDs.

## Command

```bash
./venv_arm64/bin/python scripts/rank_source_contrastive_target_sets.py \
  --target-set svamp70_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=primary_live,note=canonical_live_surface \
  --target-set svamp70_holdout=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json,role=canonical_holdout,note=canonical_holdout_surface \
  --target-set svamp70_chal241_310=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json,role=adjacent_falsifier,note=best_adjacent_clean_surface \
  --target-set svamp32_qwen25math=path=results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json,role=smoke_debug,note=tiny_debug_surface \
  --target-set gsm70_qwen25math=path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_gsm_surface \
  --target-set svamp70_chal171_240=path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_adjacent_surface \
  --target-set svamp70_chal311_380=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_adjacent_surface \
  --target-set svamp32_qwen25math_instruct=path=results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_instruct_surface \
  --min-clean-source-only 5 \
  --date 2026-04-27 \
  --output-json results/durable_source_surface_ranking_20260427/source_surface_ranking.json \
  --output-md results/durable_source_surface_ranking_20260427/source_surface_ranking.md
```

## Result

- Top surface: `svamp70_live`
- Decision: `primary_ready`
- Clean source-only: `6/70`
- Raw source-only: `9/70`
- Target/source oracle gain: `9/70`
- Canonical holdout: `weak_clean_headroom`, clean source-only `2/70`
- Adjacent falsifier `svamp70_chal241_310`: clean source-only `4/70`

## Output Hashes

- `source_surface_ranking.json`:
  `7e665698c206f748074ea567754e1f7392b0391ee60dc514bb41619e706a038f`
- `source_surface_ranking.md`:
  `99fe4631a973dcc09b4f97ef6a5b0d26c6dc833fd4968b24bb13a574cf7294e8`

## Decision

Use `svamp70_live` for the next method gate, with exact clean-ID reporting on:

`14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`,
`4d780f825bb8541c`, `bd9d8da923981d69`, `ce08a3a269bf0151`.

Promotion from this surface requires immediate replay on canonical
`svamp70_holdout` with identical controls, despite its weak clean headroom.
