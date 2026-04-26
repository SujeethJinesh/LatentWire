# SVAMP70 Holdout Finalish Guard Manifest

- date: `2026-04-26`
- status: `source_finalish_guard_fails_holdout_gate`
- base surface: `qwen25math_qwen3_svamp70_holdout_source_surface_20260426`

## Files

- `source_finalish_guard_sidecar.json`
  - sha256: `dc5b99e4500e414dae02241e7472734ee9aef51772cd55d9de9149c6c4dd9c1d`
- `source_finalish_guard_sidecar.md`
  - sha256: `d5b9c88a414ae71d796d8f742724d14ba5ce22ab92d0b0869af9676fcbc5fcd4`
- `source_finalish_guard_predictions.jsonl`
  - sha256: `a0b7d2336c515b38c1e053fea09d94fb39e6fc224390e2500bf067928652e45a`

## Result

Best row reaches `9/70`, with `0` clean source-necessary IDs and `2` clean
IDs recovered by source-destroying controls. The fixed finalish-short-numeric
guard fails the holdout gate and should not be tuned further in this family.
