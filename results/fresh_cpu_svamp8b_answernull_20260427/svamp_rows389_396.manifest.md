# JSONL Range Materialization

- date: `2026-04-27`
- source: `data/svamp_1000.jsonl`
- output: `results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.jsonl`
- start index: `389`
- count: `8`
- source sha256: `c56db9096c1dc61f132b135256ffe50849faf42ff9a5752811801bb90353bdce`
- output sha256: `753b2778348768e3ff6b72cd0c070454ce5baf52a142f8fc0ed3a0db78138280`

## IDs

- first metadata id: `chal-389`
- last metadata id: `chal-396`

## Command

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py --source data/svamp_1000.jsonl --output results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.jsonl --start-index 389 --count 8 --manifest-json results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.manifest.json --manifest-md results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.manifest.md --run-date 2026-04-27
```
