# Generation Baseline Materialization

- date: `2026-04-27`
- eval file: `data/svamp_eval_70.jsonl`
- materialized eval file: `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/_artifacts/svamp_eval_70_8.jsonl`
- limit: `8`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/source_alone.jsonl` | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/logs/source.log` |
| `target` | `ran` | 0 | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/target_alone.jsonl` | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/logs/target.log` |
| `t2t` | `ran` | 0 | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/text_to_text.jsonl` | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 2/8 | 0.250 | `True` | 7/8 | 0 |
| `target` | 2/8 | 0.250 | `True` | 8/8 | 0 |
| `t2t` | 2/8 | 0.250 | `True` | 8/8 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 1 | 1 | 1 | 3/8 |
| `t2t` | 1 | 1 | 1 | 3/8 |

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/_artifacts/svamp_eval_70_8.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/.source_alone.jsonl.tmp.source.11351 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/_artifacts/svamp_eval_70_8.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/.target_alone.jsonl.tmp.target.11351 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/_artifacts/svamp_eval_70_8.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/.text_to_text.jsonl.tmp.t2t.11351 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
