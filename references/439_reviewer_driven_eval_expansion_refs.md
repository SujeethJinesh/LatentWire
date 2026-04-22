# Reviewer-Driven Evaluation Expansion References

Date: `2026-04-22`

## Why This Memo Exists

Reviewer feedback sharpened the evaluation problem:

- a frozen `32`-example slice is good for falsification, not for trusting
  `+1` or `+2` example gains
- tiny deltas need seed repeats, paired uncertainty, and stronger diagnostics
- benchmark ordering should be explicit and budget-matched rather than
  opportunistic

This memo collects the most relevant outside references for that evaluation
pivot.

## Core Benchmark Stack

- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
  Best synthetic long-context stress suite for controlled retrieval, tracing,
  aggregation, and QA.
- [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/abs/2412.10319)
  Most directly aligned external benchmark for cache-centric communication and
  KV-style methods.
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)
  Useful lower bar before moving to harder, more realistic long-context suites.
- [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204)
  Stronger realism and reasoning pressure than older retrieval-centric suites.
- [∞Bench: Extending Long Context Evaluation Beyond 100K Tokens](https://aclanthology.org/2024.acl-long.814/)
  Good stretch benchmark once the main long-context stack is stable.

## Evaluation Reliability / Uncertainty

- [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
  Useful both as support for careful small-sample design and as a warning that
  tiny curated slices can mislead without uncertainty reporting.
- [Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores](https://arxiv.org/abs/2410.03492)
  Strong support for seed repeats, interval estimates, and uncertainty-aware
  benchmark reporting.
- [Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence](https://arxiv.org/abs/2402.09880)
  Useful citation for why underspecified single-number benchmark claims are
  fragile.
- [On Robustness and Reliability of Benchmark-Based Evaluation of LLMs](https://arxiv.org/abs/2509.04013)
  Good recent support for format sensitivity and the need for robust evaluation
  rather than single static scores.
- [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://arxiv.org/abs/2406.19314)
  Good contamination-resistant benchmark reference if we later want a fresh
  general-purpose evaluation lane.

## Practical Takeaways For This Repo

1. Keep the exact GSM8K32 contract as a regression/falsification check only.
2. Build a larger frozen slice with explicit seeds and paired diagnostics.
3. Publicly fix benchmark order:
   - `RULER`
   - `SCBench`
   - `LongBench v2`
   - only then broader long-context or contamination-resistant suites
4. Report accuracy together with latency, tokens, and budget-matched settings.

## Candidate Cross-Family Pairs To Use Next

- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) ↔ [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  Cleanest strict same-size cross-family pair.
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) ↔ [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  Cheaper pair for seed repeats and early falsification.
- [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) ↔ [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
  Strong tokenizer/context mismatch stress pair, though less tightly
  size-matched.
