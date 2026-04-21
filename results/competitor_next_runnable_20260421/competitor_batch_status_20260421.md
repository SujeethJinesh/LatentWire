# Competitor Batch Status: 2026-04-21

## Limit-10 Widening Update

The KVPress smoke has been widened to `limit=10` for the required GSM70 rows and, since time remained, for the SVAMP70 pair as well.
This still remains a runnable-control checkpoint, not a paper claim, but it gives us a better matched-budget read than the earlier
limit-5 pass.

## Completed Limit-10 Rows

| Dataset | Press | Compression | Limit | Accuracy | Tokens/sec | Examples/sec | Latency sec | Generated tokens avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K70 | none | - | 10 | 0.1000 | 3.6120 | 0.0879 | 11.3788 | 41.1000 |
| GSM8K70 | expected_attention | 0.5000 | 10 | 0.1000 | 4.4913 | 0.1085 | 9.2177 | 41.4000 |
| SVAMP70 | none | - | 10 | 0.2000 | 4.8366 | 0.1157 | 8.6424 | 41.8000 |
| SVAMP70 | expected_attention | 0.5000 | 10 | 0.5000 | 5.4349 | 0.1419 | 7.0471 | 38.3000 |

## Interpretation

- GSM8K limit-10 remains neutral on accuracy: expected-attention matches the uncompressed row at `0.1`, but it is faster.
- SVAMP limit-10 remains positive: expected-attention improves from `0.2` to `0.5` and also lowers latency.
- The widened rows are still too small for a claim about LatentWire versus KVPress. They are only the control harness we needed to reopen.

## Commands Run

1. `timeout 1800s arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press none --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit10.jsonl`
2. `timeout 1800s arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit10.jsonl`
3. `timeout 1800s arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press none --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit10.jsonl`
4. `timeout 1800s arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit10.jsonl`

## Outputs Produced

- `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit10.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit10.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit10.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit10.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit10.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit10.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit10.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit10.jsonl.meta.json`

## Blockers

- The widening stayed on CPU fallback because the earlier MPS run was unstable for this local checkpoint.
- These rows still remain smoke-scale; the next useful step is limit-20 before any benchmark comparison is made against LatentWire.

## Checkpoint Update

The KVPress limit-5 paired smoke is complete for GSM8K70 and SVAMP70 with `Qwen/Qwen3-0.6B`.
This is only a runnable-control checkpoint: sample size is too small for a paper claim, but it reopens the
competitor harness and gives us a concrete next widening step.

## Completed Limit-5 Rows

| Dataset | Press | Compression | Limit | Accuracy | Tokens/sec | Examples/sec | Latency sec | Generated tokens avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K70 | none | - | 5 | 0.2000 | 4.3839 | 0.1154 | 8.6682 | 38.0000 |
| GSM8K70 | expected_attention | 0.5000 | 5 | 0.2000 | 4.3832 | 0.1112 | 8.9888 | 39.4000 |
| SVAMP70 | none | - | 5 | 0.4000 | 4.8396 | 0.1110 | 9.0091 | 43.6000 |
| SVAMP70 | expected_attention | 0.5000 | 5 | 0.6000 | 5.0484 | 0.1262 | 7.9233 | 40.0000 |

## Interpretation

- GSM8K limit-5 is neutral: expected-attention compression matches the uncompressed row at `0.2`.
- SVAMP limit-5 is positive but tiny: expected-attention improves from `0.4` to `0.6` and is slightly faster.
- The result is not statistically meaningful yet. Treat it as a harness sanity check and widen to limit-10/20 before comparing against LatentWire.

## Commands Run

1. `arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --limit 5 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit5.jsonl`
2. `arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --limit 5 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit5.jsonl`
3. `arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --limit 5 --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit5.jsonl`
4. `arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --limit 5 --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit5.jsonl`

## Outputs Produced

- `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit5.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit5.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit5.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit5.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit5.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit5.jsonl.meta.json`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit5.jsonl`
- `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit5.jsonl.meta.json`

## Blockers

- The SVAMP expected-attention `mps` run failed after Hugging Face HEAD retries and an MPS backend error, so the completed paired row used `--device cpu`.
- The earlier non-limited C2C and KVPress rows are still too expensive for this checkpoint and should be resumed only through bounded limit-10/20 controls.

## Next Exact Commands

Widen KVPress first:

`arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press none --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit10.jsonl`

`arch -arm64 ./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device cpu --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --limit 10 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit10.jsonl`

Then repeat the same limit-10 pair on `data/svamp_eval_70.jsonl`.
