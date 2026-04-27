# Generation Baseline Materialization

- date: `2026-04-27`
- eval file: `results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl`
- materialized eval file: `results/fresh_cpu_svamp8_answernull_20260427/baselines/_artifacts/svamp_rows381_388_8.jsonl`
- limit: `8`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/fresh_cpu_svamp8_answernull_20260427/baselines/source_alone.jsonl` | `results/fresh_cpu_svamp8_answernull_20260427/baselines/logs/source.log` |
| `target` | `ran` | 0 | `results/fresh_cpu_svamp8_answernull_20260427/baselines/target_alone.jsonl` | `results/fresh_cpu_svamp8_answernull_20260427/baselines/logs/target.log` |
| `t2t` | `ran` | 0 | `results/fresh_cpu_svamp8_answernull_20260427/baselines/text_to_text.jsonl` | `results/fresh_cpu_svamp8_answernull_20260427/baselines/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 1/8 | 0.125 | `True` | 8/8 | 0 |
| `target` | 1/8 | 0.125 | `True` | 8/8 | 0 |
| `t2t` | 4/8 | 0.500 | `True` | 8/8 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 1 | 1 | 0 | 2/8 |
| `t2t` | 3 | 0 | 1 | 4/8 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/_artifacts/svamp_rows381_388_8.jsonl --task-type generation --device cpu --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/.source_alone.jsonl.tmp.source.9911 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/_artifacts/svamp_rows381_388_8.jsonl --task-type generation --device cpu --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/.target_alone.jsonl.tmp.target.9911 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/_artifacts/svamp_rows381_388_8.jsonl --task-type generation --device cpu --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/fresh_cpu_svamp8_answernull_20260427/baselines/.text_to_text.jsonl.tmp.t2t.9911 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
