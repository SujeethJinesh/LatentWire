# Qwen2.5-Math -> Qwen3 SVAMP70 Source Sidecar Manifest

- date: `2026-04-26`
- status: medium positive versus target/text, not C2C
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- method: target/text agreement guard plus 1-byte source residue sidecar

## Summary

- target-alone: `21/70`
- source-alone: `13/70`
- text relay: `22/70`
- C2C: `31/70`
- guarded sidecar: `25/70`
- clean source-necessary: `4/6`
- control clean union: `0/6`
- sidecar vs target paired delta: `+0.0571`, bootstrap
  `[-0.0286, +0.1429]`
- sidecar vs text paired delta: `+0.0429`, bootstrap
  `[-0.0571, +0.1429]`
- sidecar vs C2C paired delta: `-0.0857`, bootstrap
  `[-0.2143, +0.0571]`

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/manifest.json` | `c55ee94dcd1940a6d5cfde58f794e5ebf1a9a6d4ecf993ccb0a624b4806049d6` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/manifest.md` | `1155f7f1eee547d0d36e6d62fb6305d9c01cf7042758e19e3cd293383033b0fa` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json` | `fc5eb4ca577ca33e01a9b23f427ff5d45346e1d2435392a7ce01ca25631fc729` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_only_sidecar_router_t2t_guard.json` | `0d5971c8152650b31e2fda9ccf0b1263061f6adc045e488ed5feb92841e8389d` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_sidecar_t2t_guard_predictions.jsonl` | `4f8662b051d6df452f97c3c9791d48dcad361a15459bc31f50fdbe7c0e9ac83d` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_target.md` | `e10c069362919592a414b550ec0e7322080fc15b144d4fdd97bdc7bde14d3e7d` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_text.md` | `eb4c8eb66b6882fcb97eb7fc094555cd47ed0a83d87cf6a829f51bad559de38d` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_c2c.md` | `2de7635ecec14de137db4e178d62940a9e6485cd8975f235f9a8c170166d2037` |
| `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_sidecar_c2c_fallback_t2t_guard.json` | `17e4a88171e737f52d5b8a106de9f7156b83641fd41533e33b30543244a91e07` |

