# Competitor Batch Status: 2026-04-21

## Status
Batch stopped early. Both attempted rows were slower than the "cheap control" expectation on this machine, so I did not continue into the full matrix.

## Commands Attempted

1. `./venv_arm64/bin/python scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --max-new-tokens 64 --prediction-output results/competitor_next_runnable_20260421/c2c_gsm70_qwen25_to_qwen3.jsonl`
2. `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none.jsonl`

## Outputs Produced

- C2C GSM70 printed the model fetch stage:
  - `Fetching 59 files: 100%`
  - `Sliding Window Attention is enabled but not implemented for sdpa; unexpected results may be encountered.`
- KVPress GSM70 printed:
  - `Device set to use mps`

No benchmark row completed to a final metrics JSON / JSONL artifact in this batch.

## Blockers

- C2C GSM70 was not cheap in practice: the initial run sat in model fetch / generation long enough that it was no longer the fastest path for this batch.
- KVPress GSM70 `none` also did not finish quickly enough to qualify as a quick control row here.
- Because this repo is on `mps`, both paths are sensitive to model load and generation latency, so the next batch should use the shortest control row available and keep the timeout budget explicit.

## Next Exact Command

`./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050.jsonl`

If that still hangs, fall back to SVAMP70 only after GSM70 control finishes or is explicitly timed out.
