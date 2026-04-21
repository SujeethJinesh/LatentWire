# Competitor Benchmarks: Current Local State and Next Fair Comparisons

## Current local results

### KVPress wrapper on our local GSM5 slice

Command used:

```bash
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python \
/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvpress_eval.py \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_5.jsonl \
  --press none \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/tmp_kvpress_none_gsm5.jsonl
```

Result:

- `Qwen/Qwen3-0.6B`, `press=none`
- `accuracy=0.2`
- `tokens_per_sec=4.9582`
- `examples_per_sec=0.1298`
- `latency_sec=7.7045`
- `generated_tokens_avg=38.2`

Paired run on the same slice:

```bash
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python \
/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvpress_eval.py \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_5.jsonl \
  --press expected_attention \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/tmp_kvpress_expected_gsm5.jsonl
```

Result:

- `Qwen/Qwen3-0.6B`, `press=expected_attention`, `compression_ratio=0.5`
- `accuracy=0.2`
- `tokens_per_sec=4.1516`
- `examples_per_sec=0.1070`
- `latency_sec=9.3458`
- `generated_tokens_avg=38.8`

Read:

- On this small local slice, `ExpectedAttention` ties no-press on accuracy and is slower.
- This is a valid smoke check, but it is not evidence that KVPress beats our methods.

### Archived KVPress GSM smoke and needle smoke

Already recorded in `results/kvpress_expected_20260420/` and the `.debug/kvpress_nih/` smoke outputs:

- GSM5 and GSM10 `no_press` vs `expected_attention` both tie on accuracy in the archived GSM runs.
- Native needle smoke on a single-row `needle_in_haystack` config also ties no-press:
  - `rouge-l f=0.75` for both no-press and expected_attention.

Interpretation:

- The competitor code path is runnable locally.
- The current `ExpectedAttention` smoke does not yet beat the no-press baseline on the small local probes we have.

## Next fair comparisons

### 1. KVPress native benchmark replay

Best next run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file <small_benchmark_config.yaml>
```

Use the repo-native benchmark configs for one of:

- `needle_in_haystack`
- `ruler`
- `longbench`
- `loogle`

Fairness rules:

- Keep the same `model`, `max_context_length`, `fraction`, `compression_ratio`, `query_aware`, and `needle_depth` settings across methods.
- Use the benchmark's own `calculate_metrics.py` / `evaluate.py` path, not our GSM parser, when comparing KVPress methods.

### 2. C2C replay on our held-out GSM splits

We already have the published C2C bundle bootstrapped in `references/repos/C2C` and local result anchors in:

- `results/c2c_gsm70_20260418/summary.md`
- `results/c2c_gsm100_20260418/summary.md`
- `results/c2c_bootstrap_20260418/summary.md`

Next fair comparison command:

```bash
python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/<new_c2c_run>.jsonl
```

Fairness rules:

- Compare only against the same source/target pair and the same held-out GSM split.
- Do not compare a prompt-local GSM5 smoke to a held-out GSM70 or GSM100 number.
- Use the existing C2C bootstrap summaries as the benchmark anchor, not a re-tokenized or re-parsed variant.

### 3. Quest / DeltaKV / tokenkit only after split parity is clear

These repos are present and inspectable, but they are not the next fastest fair comparison unless we can match their evaluation split and scoring exactly.

Runnable entrypoints to keep in mind:

- Quest: `references/repos/Quest/evaluation/LongBench/eval.py`
- DeltaKV / SGLang: `references/repos/DeltaKV_sparse_vllm/benchmark/scbench/run_scbench.py`
- tokenkit: `scripts/eval_lockstep.py`

These are likely to require extra model/tokenizer setup and are not the first choice for a quick smoke.

## How to avoid invalid scoring

- Never mix metrics across benchmark families. GSM exact-match, needle ROUGE, LongBench F1/ROUGE, and C2C paired accuracy are not directly interchangeable.
- Do not compare results with different decoding budgets, different `max_new_tokens`, or different context lengths as if they were the same experiment.
- Do not score KVPress GSM slices with KVPress benchmark-native needle metrics, or vice versa.
- Keep `query_aware`, `compression_ratio`, and `needle_depth` fixed when comparing press variants.
- For paired deltas, use the same split and the same example order. A method can look better on a smoke slice and still lose on the controlled held-out split.
- Treat token/sec and latency as operational metrics only; they are useful for throughput, not for accuracy claims.

## Bottom line

The local competitor state is runnable, but the current KVPress `ExpectedAttention` smoke only matches no-press on GSM5 and needle retrieval. The next fair comparison should be a split-matched replay against C2C on our held-out GSM slices, using the exact same source/target pair and score parser.
