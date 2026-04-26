# Source Surface Reselection Manifest

- date: `2026-04-26`
- status: `svamp70_live_and_holdout_rank_highest`
- git base commit: `e5a7be224d2ce53bfca747a5106add17a4f74b03`

## Result

The consolidated scan ranks existing exact-ID surfaces by target-complementary
source headroom:

1. `svamp70_live`: strong source-complementary surface, source-only `9`,
   target/source oracle `30/70`.
2. `svamp70_holdout`: strong source-complementary surface, source-only `6`,
   target/source oracle `14/70`.
3. `svamp32_qwen25math`: weak source-complementary surface, source-only `5`,
   target/source oracle `13/32`.

GSM70, DeepSeek SVAMP32, and Qwen2.5-Math-Instruct SVAMP32 remain weak
immediate surfaces.

## Files

- `source_headroom_surfaces.json`
- `source_headroom_surfaces.md`
