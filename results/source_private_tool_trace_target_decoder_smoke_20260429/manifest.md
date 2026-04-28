# Source-Private Tool-Trace Target-Decoder Smoke Manifest

- gate: `source_private_tool_trace_target_decoder_smoke_20260429`
- date: `2026-04-29`
- status: passed as smoke

## Outcome

| Surface | N | Target | Matched packet | Best control | Pass |
|---|---:|---:|---:|---:|---|
| core seed 29 | `16` | `0.250` | `0.688` | `0.250` | `True` |
| held-out seed 30 | `16` | `0.250` | `0.750` | `0.312` | `False` |
| held-out seed 30 | `32` | `0.250` | `0.750` | `0.281` | `True` |

## Artifacts

- `core_seed29_qwen3_n16/summary.json`
- `core_seed29_qwen3_n16/summary.md`
- `core_seed29_qwen3_n16/manifest.json`
- `core_seed29_qwen3_n16/manifest.md`
- `holdout_seed30_qwen3_n16/summary.json`
- `holdout_seed30_qwen3_n16/summary.md`
- `holdout_seed30_qwen3_n16/manifest.json`
- `holdout_seed30_qwen3_n16/manifest.md`
- `holdout_seed30_qwen3_n32/summary.json`
- `holdout_seed30_qwen3_n32/summary.md`
- `holdout_seed30_qwen3_n32/manifest.json`
- `holdout_seed30_qwen3_n32/manifest.md`

## Decision

Treat this as an ablation that reduces the hand-coded decoder novelty risk.
It is not the main large-slice evidence and should not be overclaimed as a
fully learned target bridge.
