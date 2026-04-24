# Generation Baseline Materialization

- date: `2026-04-24`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized eval file: `results/svamp32_stronger_source_baselines_20260424/_artifacts/svamp_eval_70_32_32.jsonl`
- limit: `32`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/svamp32_stronger_source_baselines_20260424/source_alone.jsonl` | `results/svamp32_stronger_source_baselines_20260424/logs/source.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 3/32 | 0.094 | `True` | 32/32 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-1.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp32_stronger_source_baselines_20260424/_artifacts/svamp_eval_70_32_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp32_stronger_source_baselines_20260424/.source_alone.jsonl.tmp.source.13267 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
