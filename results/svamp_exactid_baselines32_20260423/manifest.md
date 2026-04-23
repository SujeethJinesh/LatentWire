# Generation Baseline Materialization

- date: `2026-04-23`
- eval file: `data/svamp_eval_70.jsonl`
- materialized eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- limit: `32`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `target` | `skipped_existing` |  | `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` | `results/svamp_exactid_baselines32_20260423/logs/target.log` |
| `source` | `skipped_existing` |  | `results/svamp_exactid_baselines32_20260423/source_alone.jsonl` | `results/svamp_exactid_baselines32_20260423/logs/source.log` |
| `t2t` | `skipped_existing` |  | `results/svamp_exactid_baselines32_20260423/text_to_text.jsonl` | `results/svamp_exactid_baselines32_20260423/logs/t2t.log` |
| `c2c` | `skipped_existing` |  | `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl` | `results/svamp_exactid_baselines32_20260423/logs/c2c.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `target` | 8/32 | 0.250 | `True` | 32/32 | 0 |
| `source` | 5/32 | 0.156 | `True` | 31/32 | 0 |
| `t2t` | 2/32 | 0.062 | `True` | 32/32 | 0 |
| `c2c` | 16/32 | 0.500 | `True` | 32/32 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `c2c` | 10 | 2 | 6 | 18/32 |
| `source` | 3 | 6 | 2 | 11/32 |
| `t2t` | 1 | 7 | 1 | 9/32 |

## Commands

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/target_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/source_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/text_to_text.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### c2c

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl --device mps --max-new-tokens 64 --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl
```
