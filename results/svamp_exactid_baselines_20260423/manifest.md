# Generation Baseline Materialization

- date: `2026-04-23`
- eval file: `data/svamp_eval_70.jsonl`
- materialized eval file: `results/svamp_exactid_baselines_20260423/_artifacts/svamp_eval_70_5.jsonl`
- limit: `5`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `target` | `skipped_existing` |  | `results/svamp_exactid_baselines_20260423/target_alone.jsonl` | `results/svamp_exactid_baselines_20260423/logs/target.log` |
| `source` | `skipped_existing` |  | `results/svamp_exactid_baselines_20260423/source_alone.jsonl` | `results/svamp_exactid_baselines_20260423/logs/source.log` |
| `t2t` | `skipped_existing` |  | `results/svamp_exactid_baselines_20260423/text_to_text.jsonl` | `results/svamp_exactid_baselines_20260423/logs/t2t.log` |
| `c2c` | `skipped_existing` |  | `results/svamp_exactid_baselines_20260423/c2c_generate.jsonl` | `results/svamp_exactid_baselines_20260423/logs/c2c.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `target` | 2/5 | 0.400 | `True` | 5/5 | 0 |
| `source` | 2/5 | 0.400 | `True` | 5/5 | 0 |
| `t2t` | 0/5 | 0.000 | `True` | 5/5 | 0 |
| `c2c` | 1/5 | 0.200 | `True` | 5/5 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `c2c` | 0 | 1 | 1 | 2/5 |
| `source` | 1 | 1 | 1 | 3/5 |
| `t2t` | 0 | 2 | 0 | 2/5 |

## Commands

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/_artifacts/svamp_eval_70_5.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/target_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/_artifacts/svamp_eval_70_5.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/source_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/_artifacts/svamp_eval_70_5.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/text_to_text.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### c2c

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/_artifacts/svamp_eval_70_5.jsonl --device mps --max-new-tokens 64 --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines_20260423/c2c_generate.jsonl
```
