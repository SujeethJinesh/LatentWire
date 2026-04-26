# Generation Baseline Materialization

- date: `2026-04-26`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized eval file: `results/surface_scout_qwen25math_qwen3_svamp16_20260426/_artifacts/svamp_eval_70_32_16.jsonl`
- limit: `16`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `skipped_existing` |  | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/source_alone.jsonl` | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/logs/source.log` |
| `target` | `skipped_existing` |  | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/target_alone.jsonl` | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/logs/target.log` |
| `t2t` | `skipped_existing` |  | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/text_to_text.jsonl` | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/logs/t2t.log` |
| `c2c` | `skipped_existing` |  | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/c2c_generate.jsonl` | `results/surface_scout_qwen25math_qwen3_svamp16_20260426/logs/c2c.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 1/16 | 0.062 | `True` | 16/16 | 0 |
| `target` | 0/16 | 0.000 | `True` | 15/16 | 0 |
| `t2t` | 5/16 | 0.312 | `True` | 16/16 | 0 |
| `c2c` | 5/16 | 0.312 | `True` | 16/16 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `c2c` | 5 | 0 | 0 | 5/16 |
| `source` | 1 | 0 | 0 | 1/16 |
| `t2t` | 5 | 0 | 0 | 5/16 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/_artifacts/svamp_eval_70_32_16.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/source_alone.jsonl
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/_artifacts/svamp_eval_70_32_16.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/target_alone.jsonl
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/_artifacts/svamp_eval_70_32_16.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/text_to_text.jsonl
```

### c2c

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/_artifacts/svamp_eval_70_32_16.jsonl --device mps --max-new-tokens 64 --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/surface_scout_qwen25math_qwen3_svamp16_20260426/c2c_generate.jsonl
```
