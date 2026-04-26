# Generation Baseline Materialization

- date: `2026-04-26`
- eval file: `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl`
- materialized eval file: `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380_70.jsonl`
- limit: `70`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/logs/source.log` |
| `target` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/logs/target.log` |
| `t2t` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/text_to_text.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 8/70 | 0.114 | `True` | 63/70 | 0 |
| `target` | 21/70 | 0.300 | `True` | 70/70 | 0 |
| `t2t` | 19/70 | 0.271 | `True` | 70/70 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 3 | 16 | 5 | 24/70 |
| `t2t` | 9 | 11 | 10 | 30/70 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/.source_alone.jsonl.tmp.source.20558 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/.target_alone.jsonl.tmp.target.20558 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/.text_to_text.jsonl.tmp.t2t.20558 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
