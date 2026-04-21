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

## Follow-Up Limit-1 Smokes

After adding `--limit` support to `scripts/run_kvpress_eval.py`, the GSM70
same-model KVPress controls do complete as one-example smokes:

| Row | Limit | Max new tokens | Accuracy | Latency sec | Tokens/sec | Output |
|---|---:|---:|---:|---:|---:|---|
| KVPress `none` | `1` | `8` | `0.0000` | `2.1374` | `1.8715` | `kvpress_gsm70_none_limit1.jsonl` |
| KVPress `expected_attention` `0.5` | `1` | `8` | `0.0000` | `2.1806` | `2.7516` | `kvpress_gsm70_expected_attention_c050_limit1.jsonl` |

These are smoke rows only and must not be used as paper accuracy results.
They show that the local benchmark path is viable once the full-row run is
split into bounded chunks.

## Blockers

- C2C GSM70 was not cheap in practice: the initial run sat in model fetch / generation long enough that it was no longer the fastest path for this batch.
- KVPress GSM70 `none` also did not finish quickly enough to qualify as a quick control row here.
- Because this repo is on `mps`, both paths are sensitive to model load and generation latency, so the next batch should use the shortest control row available and keep the timeout budget explicit.

## Next Exact Command

`./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --limit 5 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit5.jsonl`

If that completes, run the matched `expected_attention --compression-ratio 0.5
--limit 5` row before scaling to the full GSM70/SVAMP70 rows.
