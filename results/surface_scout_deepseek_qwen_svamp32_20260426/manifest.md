# Generation Baseline Materialization

- date: `2026-04-26`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized eval file: `results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- limit: `32`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `skipped_existing` |  | `results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl` | `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/source.log` |
| `target` | `skipped_existing` |  | `results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl` | `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/target.log` |
| `t2t` | `skipped_existing` |  | `results/surface_scout_deepseek_qwen_svamp32_20260426/text_to_text.jsonl` | `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/t2t.log` |
| `c2c` | `failed` | 1 | `results/surface_scout_deepseek_qwen_svamp32_20260426/c2c_generate.jsonl` | `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/c2c.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 5/32 | 0.156 | `True` | 32/32 | 0 |
| `target` | 8/32 | 0.250 | `True` | 32/32 | 0 |
| `t2t` | 5/32 | 0.156 | `True` | 32/32 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 1 | 4 | 4 | 9/32 |
| `t2t` | 2 | 5 | 3 | 10/32 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/text_to_text.jsonl --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### c2c

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl --device mps --max-new-tokens 64 --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_deepseek_qwen_svamp32_20260426/.c2c_generate.jsonl.tmp.c2c.89485
```
