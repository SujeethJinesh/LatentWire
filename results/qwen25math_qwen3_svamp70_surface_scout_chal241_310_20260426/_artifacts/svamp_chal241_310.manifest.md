# JSONL Range Materialization

- date: `2026-04-26`
- source: `data/svamp_1000.jsonl`
- output: `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.jsonl`
- start index: `241`
- count: `70`
- source sha256: `c56db9096c1dc61f132b135256ffe50849faf42ff9a5752811801bb90353bdce`
- output sha256: `29b0ebf00df9c0bd552bc0dd84e3f0f9f566b8c1f1beaa34bdb118a5fe960f05`

## IDs

- first metadata id: `chal-241`
- last metadata id: `chal-310`

## Command

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py --source data/svamp_1000.jsonl --output results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.jsonl --start-index 241 --count 70 --manifest-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.manifest.json --manifest-md results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.manifest.md --run-date 2026-04-26
```
