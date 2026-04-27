# Generation Baseline Materialization

- date: `2026-04-27`
- eval file: `data/svamp_eval_70.jsonl`
- materialized eval file: `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl`
- limit: `70`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `target` | `ran` | 0 | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/target_alone.jsonl` | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/logs/target.log` |
| `source` | `ran` | 0 | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_alone.jsonl` | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/logs/source.log` |
| `t2t` | `ran` | 0 | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/text_to_text.jsonl` | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `target` | 21/70 | 0.300 | `True` | 70/70 | 0 |
| `source` | 15/70 | 0.214 | `True` | 70/70 | 0 |
| `t2t` | 12/70 | 0.171 | `True` | 70/70 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 8 | 14 | 7 | 29/70 |
| `t2t` | 5 | 14 | 7 | 26/70 |

## Commands

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/.target_alone.jsonl.tmp.target.44479 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/.source_alone.jsonl.tmp.source.44479 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/.text_to_text.jsonl.tmp.t2t.44479 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
