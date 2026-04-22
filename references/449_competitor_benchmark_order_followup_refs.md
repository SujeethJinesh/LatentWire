# Competitor Benchmark Order Follow-Up References

Date: `2026-04-22`

## Why This Memo Exists

The next comparison stack should stay narrow and fair. The goal is to compare
against the closest public communication methods before broadening to generic
long-context benchmarks.

## Closest Public Comparators

- [Cache-to-Cache (C2C)](https://arxiv.org/abs/2510.03215)
  Current closest public direct semantic communication baseline.
- [KVComm](https://arxiv.org/abs/2510.03346)
  Selective KV sharing baseline; closest efficiency-oriented comparator.
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)
  Closest public latent-KV alignment baseline.

## Benchmark Order

1. larger frozen same-pair slice
2. closest public comparators on that same-pair or matched setting
3. one strict matched cross-family pair
4. [RULER](https://arxiv.org/abs/2404.06654)
5. [SCBench](https://arxiv.org/abs/2412.10319)
6. [LongBench v2](https://arxiv.org/abs/2412.15204)

Keep [LongBench](https://arxiv.org/abs/2308.14508) as a compatibility or
appendix benchmark, not the main widening target.

## Strict Cross-Family Pair To Freeze Next

- primary:
  [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  -> [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- backup:
  [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  -> [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

Why this matters:

- similar enough size to be a fair first cross-family row
- different enough tokenizer / architecture / formatting to be a real
  transport test

## What To Keep Visible In Tables

- `target_alone`
- text relay
- live latent candidate
- closest external comparator
- oracle bound

This keeps the tables interpretable and prevents the story from drifting into
generic long-context benchmarking too early.
