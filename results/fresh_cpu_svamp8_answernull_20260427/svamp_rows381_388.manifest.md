# JSONL Range Materialization

- date: `2026-04-27`
- source: `data/svamp_1000.jsonl`
- output: `results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl`
- start index: `381`
- count: `8`
- source sha256: `c56db9096c1dc61f132b135256ffe50849faf42ff9a5752811801bb90353bdce`
- output sha256: `7cc0e9d8388778fca31b7dde69293d5400e0c645f03c21ec5c677a482c50daf1`

## IDs

- first metadata id: `chal-381`
- last metadata id: `chal-388`

## Command

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py --source data/svamp_1000.jsonl --output results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl --start-index 381 --count 8 --manifest-json results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.json --manifest-md results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.md --run-date 2026-04-27
```
