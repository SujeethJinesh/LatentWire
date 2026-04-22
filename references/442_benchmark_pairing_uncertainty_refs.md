# Benchmark Pairing / Uncertainty References

Date: `2026-04-22`

## Why This Memo Exists

The reviewer feedback and the latest benchmark pass agree on the same failure
mode:

- tiny frozen slices are useful for falsification, not for ranking nearby
  variants
- same-family and cross-family claims should not share one table
- latent communication rows need uncertainty and efficiency reporting, not just
  point accuracy

This memo freezes the next benchmark order and the comparator contract.

## Recommended Benchmark Order

1. Larger frozen same-pair GSM8K campaign
   - first larger practical slice in-repo: `data/gsm8k_eval_70.jsonl`
   - report paired deltas vs `target_alone`, bootstrap intervals, and seed
     spread
2. One matched cross-family small-model pair
   - purpose: falsify the “same-family calibration only” hypothesis before
     spending on broader long-context suites
3. [RULER](https://arxiv.org/abs/2404.06654)
   - cheapest controlled long-context falsifier
4. [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/abs/2412.10319)
   - closest benchmark to the KV-cache / shared-context medium
5. [LongBench v2](https://arxiv.org/abs/2412.15204)
   - first realistic broad reasoning expansion

Only after that:

6. [LongBench](https://arxiv.org/abs/2308.14508)
7. [∞Bench](https://aclanthology.org/2024.acl-long.814/)
8. multimodal extensions

## Best Strict Cross-Family Small-Model Pairs

- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) ↔ [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  Best strict same-size cross-family pair for a first falsification run.
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) ↔ [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  Cheapest pair for seed repeats and interface-mismatch stress.
- [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) ↔ [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
  Good tokenizer and style mismatch stress pair, though less tightly
  size-matched.

## Public Comparators To Keep Visible

- [C2C: Scaling Cross-Model Collaboration via Conditional Latent Space Fusion](https://arxiv.org/abs/2510.03215)
  Closest direct external method baseline for cache-to-cache communication.
- [KVComm: Cache-Efficient Collaborative Inference of Large Language Models](https://arxiv.org/abs/2510.03346)
  Strong communication-efficiency comparator for KV-style transport claims.
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)
  Strong public comparator for direct latent-space KV alignment across model
  families.
- [Q-KVComm](https://arxiv.org/abs/2512.17914)
  Useful adaptive-compression comparator if we need a tighter bandwidth-aware
  baseline stack.
- [RULER](https://arxiv.org/abs/2404.06654)
  Controlled long-context baseline suite.
- [SCBench](https://arxiv.org/abs/2412.10319)
  Must-run cache-centric benchmark once the same-pair lane survives.
- [LongBench v2](https://arxiv.org/abs/2412.15204)
  Strong broad reasoning benchmark once the cache-centric story holds.

## Larger Appendix-Scale Cross-Family Pairs

- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) ↔ [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  Strong tokenizer and interface mismatch pair once the small-model story is
  stable.
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) ↔ [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  Good medium-mismatch appendix pair at similar scale.
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) ↔ [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
  Good fairness pair for cross-family appendix results if we reach that stage.

## Uncertainty / Reliability References

- [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
  Useful support for careful small-slice design, but also a warning against
  over-reading tiny deltas.
- [Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores](https://arxiv.org/abs/2410.03492)
  Strong support for explicit seed repeats and interval reporting.
- [Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence](https://arxiv.org/abs/2402.09880)
  Useful citation against underspecified benchmark claims.
- [On Robustness and Reliability of Benchmark-Based Evaluation of LLMs](https://arxiv.org/abs/2509.04013)
  Recent support for paired uncertainty, format robustness, and stronger
  reliability reporting.
- [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://arxiv.org/abs/2406.19314)
  Good contamination-resistant future expansion once the main story is stable.

## Repo-Relevant Takeaways

1. Keep GSM8K32 as the falsification and regression gate only.
2. Treat `gsm8k_eval_70` plus `3-5` seeds as the first larger frozen campaign.
3. Keep same-family and cross-family tables separate.
4. Report efficiency on every main row:
   - bytes
   - latency
   - TTFT when relevant
   - transport depth / layers / heads / tokens
5. Do not widen to broader suites until one branch survives the larger frozen
   same-pair campaign and one matched cross-family pair.
