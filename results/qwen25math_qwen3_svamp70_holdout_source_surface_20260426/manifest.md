# Generation Baseline Materialization

- date: `2026-04-26`
- eval file: `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170.jsonl`
- materialized eval file: `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl`
- limit: `70`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl` | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/logs/source.log` |
| `target` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl` | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/logs/target.log` |
| `t2t` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl` | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/logs/t2t.log` |
| `c2c` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/c2c_generate.jsonl` | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/logs/c2c.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 8/70 | 0.114 | `True` | 64/70 | 0 |
| `target` | 8/70 | 0.114 | `True` | 70/70 | 0 |
| `t2t` | 18/70 | 0.257 | `True` | 70/70 | 0 |
| `c2c` | 37/70 | 0.529 | `True` | 70/70 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `c2c` | 31 | 2 | 6 | 39/70 |
| `source` | 6 | 6 | 2 | 14/70 |
| `t2t` | 13 | 3 | 5 | 21/70 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/.source_alone.jsonl.tmp.source.979 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/.target_alone.jsonl.tmp.target.979 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/.text_to_text.jsonl.tmp.t2t.979 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### c2c

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl --device mps --max-new-tokens 64 --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/.c2c_generate.jsonl.tmp.c2c.979
```
