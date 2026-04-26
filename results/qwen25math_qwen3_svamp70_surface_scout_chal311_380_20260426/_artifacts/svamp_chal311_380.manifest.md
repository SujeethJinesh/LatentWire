# JSONL Range Materialization

- date: `2026-04-26`
- source: `data/svamp_1000.jsonl`
- output: `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl`
- start index: `311`
- count: `70`
- source sha256: `c56db9096c1dc61f132b135256ffe50849faf42ff9a5752811801bb90353bdce`
- output sha256: `f503455a810222bbc5652a58824c5f5090d6a9d7d80973eab2caac5d51612227`

## IDs

- first metadata id: `chal-311`
- last metadata id: `chal-380`

## Command

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py --source data/svamp_1000.jsonl --output results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl --start-index 311 --count 70 --manifest-json results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.manifest.json --manifest-md results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.manifest.md --run-date 2026-04-26
```
