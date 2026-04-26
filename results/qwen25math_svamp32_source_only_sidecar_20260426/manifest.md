# Qwen2.5-Math -> Qwen3 SVAMP32 Source-Only Sidecar Manifest

- date: `2026-04-26`
- git commit at run time: `ecd8b0946956636073875186add64d16c68af7a6`
- status: `source_only_sidecar_router_fails_gate`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- decision surface:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`

## Command

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --min-correct 9 \
  --min-target-self 0 \
  --min-clean-source-necessary 2 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 26 \
  --output-json results/qwen25math_svamp32_source_only_sidecar_20260426/source_only_sidecar_router.json \
  --output-md results/qwen25math_svamp32_source_only_sidecar_20260426/source_only_sidecar_router.md
```

## Result Summary

- source numeric coverage: `26/32`
- best matched: `8/32`
- clean source-necessary IDs: `0/6`
- control clean union: up to `3/6`

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_svamp32_source_only_sidecar_20260426/source_only_sidecar_router.json` | `457b74ce65e2e1dffc5b0b8b53f40a078d9534d7f90bbde7ed1a328e3d96385b` |
| `results/qwen25math_svamp32_source_only_sidecar_20260426/source_only_sidecar_router.md` | `2b05ecffab478a573175fd984bf9dbf859d05d0c44d4b56129a2f054b9e74f0f` |

## Decision

Kill raw source-generated numeric residue sidecars on this surface.

