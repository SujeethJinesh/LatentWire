# Generation Baseline Materialization

- date: `2026-04-27`
- eval file: `results/mps_qwen25_7b_historical_clean6_discovery_20260427/clean6_eval.jsonl`
- materialized eval file: `results/mps_qwen25_7b_historical_clean6_discovery_20260427/_artifacts/clean6_eval_6.jsonl`
- limit: `6`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/source_alone.jsonl` | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/logs/source.log` |
| `target` | `ran` | 0 | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/target_alone.jsonl` | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/logs/target.log` |
| `t2t` | `ran` | 0 | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/text_to_text.jsonl` | `results/mps_qwen25_7b_historical_clean6_discovery_20260427/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 1/6 | 0.167 | `True` | 6/6 | 0 |
| `target` | 0/6 | 0.000 | `True` | 6/6 | 0 |
| `t2t` | 1/6 | 0.167 | `True` | 6/6 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 1 | 0 | 0 | 1/6 |
| `t2t` | 1 | 0 | 0 | 1/6 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/_artifacts/clean6_eval_6.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/.source_alone.jsonl.tmp.source.19307 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/_artifacts/clean6_eval_6.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/.target_alone.jsonl.tmp.target.19307 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-7B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/_artifacts/clean6_eval_6.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_historical_clean6_discovery_20260427/.text_to_text.jsonl.tmp.t2t.19307 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
