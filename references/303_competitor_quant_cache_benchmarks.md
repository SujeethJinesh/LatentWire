# Competitor Quant / Cache Benchmarks for Cross-Model Communication

This memo is the comparison shortlist for two adjacent problem classes:

1. KV/cache compression and long-context eviction methods that can change our throughput/accuracy tradeoff.
2. Tokenizer / interface transfer methods that can expose whether our blocker is cache geometry, vocabulary mismatch, or both.

The goal is not to compare everything to everything. The goal is to keep split, tokenizer, context budget, and decoding budget fixed, then compare only methods that actually solve the same bottleneck.

## Runnable Here

These repositories are already cloned in `references/repos/` and have repo-native scripts or tests we can run locally.

| Repo | Why it matters | Runnable status | Primary local entrypoints |
| --- | --- | --- | --- |
| `kvpress` | Baseline KV cache compression library with `ExpectedAttention`, `SnapKV`, and related press policies. Good first comparator for our own cache-side ablations. | Local clone present; repo-native eval scripts found. | `references/repos/kvpress/evaluation/evaluate.py`, `scripts/run_kvpress_eval.py` |
| `Quest` | Query-aware long-context sparsity. Useful for route/interface comparisons because it is query-conditioned but still cache-side. | Local clone present; LongBench / passkey / PG19 scripts present. | `references/repos/Quest/evaluation/LongBench/eval.py`, `references/repos/Quest/evaluation/passkey/passkey.py`, `references/repos/Quest/scripts/bench_textgen.py` |
| `DeltaKV_sparse_vllm` | Broader KV compression benchmark suite with LongBench, SCBench, Needle, and model-side baselines. Good for stress-testing cache methods on multiple suites. | Local clone present; benchmark harnesses present. | `references/repos/DeltaKV_sparse_vllm/benchmark/long_bench/eval.py`, `references/repos/DeltaKV_sparse_vllm/benchmark/scbench/run_scbench.py`, `references/repos/DeltaKV_sparse_vllm/benchmark/niah/test_niah.py` |
| `KVzip` | Query-agnostic KV eviction with context reconstruction. Very relevant for future-query robustness. | Local clone present; `test.py` and `eval.py` entrypoints present. | `references/repos/KVzip/test.py`, `references/repos/KVzip/eval.py`, `references/repos/KVzip/data/needle/data.py` |
| `tokenkit` | Cross-tokenizer distillation / tokenizer transfer. This is the best local comparator for our tokenizer-vs-cache interface hypothesis. | Local clone present; `eval_lockstep.py`, `cross_tokenizer_distill.py`, and related configs present. | `references/repos/tokenkit/scripts/eval_lockstep.py`, `references/repos/tokenkit/scripts/cross_tokenizer_distill.py`, `references/repos/tokenkit/configs/` |
| `C2C` | Cross-model communication baseline and held-out GSM-style replay anchor. This is the closest local benchmark family to our own method. | Local clone present; GSM replay scripts present. | `scripts/run_c2c_eval.py`, `references/repos/C2C/script/` |
| `r-kv` | Redundancy-aware KV cache compression for reasoning acceleration. Useful as a reasoning-aware cache baseline rather than a pure long-context baseline. | Local clone present; repo source available. | `references/repos/r-kv/` |

## External Benchmarks / Baselines Worth Pulling In

These are relevant even if we do not compare them first because they help separate cache compression from tokenizer/interface mismatch.

| Method / suite | Link | Why it belongs here |
| --- | --- | --- |
| `SVDq` | https://arxiv.org/abs/2502.15304 | Very aggressive key-cache compression; useful stress test for our geometry and quantization intuition. |
| `FastKV` | https://arxiv.org/abs/2502.01068 | Training-free token-selective propagation; good cache-side control. |
| `Q-Filters` | https://arxiv.org/abs/2503.02812 | Query/key geometry baseline with cheap context-agnostic projection. |
| `LogQuant` | https://arxiv.org/abs/2503.19950 | Low-bit KV quantization baseline with a stronger compression story. |
| `Expected Attention` / KVPress | https://arxiv.org/abs/2510.00636 | Current mainstream cache-compression comparator; already runnable locally via `kvpress`. |
| `R-KV` | https://arxiv.org/abs/2505.24133 | Reasoning-aware cache compression baseline. |
| `KVzip` | https://arxiv.org/abs/2505.23416 | Query-agnostic future-query robustness baseline. |
| `KQ-SVD` | https://arxiv.org/abs/2512.05916 | Later 2025 geometric compression baseline; worth tracking if we need a fidelity-focused comparator. |
| `TurboQuant` | https://arxiv.org/abs/2504.19874 | Near-optimal KV cache quantization; important for any cache-size argument. |
| `AWQ` / `GPTQModel` / `EXL2` | https://github.com/mit-han-lab/llm-awq, https://github.com/modelcloud/gptqmodel | Weight-quantization controls. Not a direct match for our method, but necessary for a fair systems-level comparison. |
| `LongBench` / `InfiniteBench` / `RULER` / `SCBench` | https://github.com/OpenBMB/InfiniteBench, https://github.com/henryzhongsc/longctx_bench, repo-native suites in `Quest` and `DeltaKV_sparse_vllm` | Needed to check whether any apparent win is only a GSM artifact. |

## Current Local Evidence

The local comparator smoke checks already show the right kind of separation for a paper memo:

- `kvpress` on our GSM5 slice: `none` and `expected_attention` both sit at `0.2` accuracy.
- `kvpress` native needle smoke: `none` and `expected_attention` both tie at `rouge-l f=0.75`.
- That means the runner works, but the comparator is not yet beating the no-press baseline on our small probes.

This is still useful because it gives us a fair external control. It does not support a positive-method claim yet.

## Fair-Comparison Rules

- Keep the same `model`, tokenizer, context length, decoding budget, and evaluation parser inside each benchmark family.
- Do not compare GSM exact-match numbers to Needle ROUGE or LongBench F1 as if they are interchangeable.
- For KVPress-style comparisons, hold `query_aware`, `compression_ratio`, `needle_depth`, and `max_context_length` fixed across methods.
- For tokenizer-transfer comparisons, use the same source/target pair and the same split; otherwise the numbers are not comparable.
- For paired communication comparisons, use the same source/target pair, the same example order, and the same held-out split.
- Treat throughput, tokens/sec, and latency as operational metrics only. They are not accuracy evidence.

## Next Commands

Run these in order of expected signal.

```bash
# 1) Replay the local KVPress benchmark path on the current GSM slice.
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python \
/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvpress_eval.py \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_5.jsonl \
  --press none \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/tmp_kvpress_none_gsm5.jsonl
```

```bash
# 2) Compare the same slice with ExpectedAttention.
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python \
/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvpress_eval.py \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_5.jsonl \
  --press expected_attention \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/tmp_kvpress_expected_gsm5.jsonl
```

```bash
# 3) Re-run the native KVPress needle benchmark at the same context budget.
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_no_press.yaml
```

```bash
# 4) Run Quest LongBench / passkey smoke once the model and dataset are aligned.
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/evaluation/LongBench
python eval.py --help
```

```bash
# 5) Run the C2C held-out GSM replay on the same source/target pair as our bridge method.
python /Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file /Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/<new_c2c_run>.jsonl
```

## Recommended Interpretation

If a method improves accuracy but increases latency or narrows context coverage, it is a systems tradeoff, not a clean paper win.
If a method only helps after changing tokenizer or benchmark parsing, the effect belongs to the tokenizer/interface bucket, not the cache bucket.
If a method only improves a single small smoke slice and fails on held-out GSM, treat it as a debugging signal, not evidence of a positive method.
