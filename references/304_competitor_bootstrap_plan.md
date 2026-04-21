# Competitor Bootstrap Plan: Quest / KVzip / tokenkit

This is the next practical benchmark bootstrap after KVPress / C2C. The goal is to separate three blockers that keep showing up in our paper work:

1. cache-side compression and future-query robustness
2. query-aware sparsity / routing
3. tokenizer and interface mismatch

The main point is not to compare everything to everything. The point is to keep model family, tokenizer, context budget, and scoring family fixed within each benchmark family, then ask which bottleneck is actually limiting cross-model communication.

## Recommended Order

1. `KVzip` first
2. `Quest` second
3. `tokenkit` third

That order is the highest signal-to-effort path from the local clones already present in `references/repos/`.

## 1) KVzip

Why this is next:

- best local control for query-agnostic cache compression and reconstruction
- supports Qwen2.5 / Qwen3 / Llama3 families already
- directly tests whether our bridge is really better than compression-only cache pruning

Required models and datasets:

- `Qwen/Qwen2.5-7B-Instruct-1M` or `llama3.1-8b`
- `squad`
- `scbench_kv`
- `scbench_prefix_suffix_short`
- `scbench_repoqa_short`
- `needle` / `NIAH` style slices if we want a retrieval smoke

Runnable commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
python -B test.py -m qwen2.5-7b -d scbench_kv --kv_type retain --ratio 0.6
python -B test.py -m llama3.1-8b -d scbench_repoqa --save_head_score
python -B eval.py -m qwen2.5-7b -d scbench_kv --kv_type retain --num 100
```

Expected artifacts:

- `references/repos/KVzip/results/<data_name>/`
- `references/repos/KVzip/utils/head_score/*.pt`
- parsed summaries from `python -B -m results.parse`

Fair-comparison caveats:

- compare only within the same model family and same dataset split
- do not compare KVzip exact-match or F1 to GSM exact-match from our bridge
- KVzip is a compression/reconstruction control, not a heterogeneous communication baseline

## 2) Quest

Why this is next:

- query-aware sparsity is the closest baseline family to our own routing story
- it tells us whether query-conditioned pruning alone already explains most of the gain
- it gives a stronger long-context control than our current KVPress smoke

Required models and datasets:

- `LongChat-7B-v1.5-32K` for LongBench
- `Yarn-Llama-2-7B-128K` for long-context comparisons if needed
- `Llama-3.1-8B-Instruct` for passkey smoke
- LongBench tasks: `qasper`, `narrativeqa`, `hotpotqa`, `multifieldqa_en`, `gov_report`, `triviaqa`
- Passkey length smoke: `100000`

Runnable commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh

cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/evaluation/LongBench
python -u pred.py --model longchat-v1.5-7b-32k --task qasper
python -u pred.py --model longchat-v1.5-7b-32k --task qasper --quest --token_budget 2048 --chunk_size 16
python -u eval.py --model longchat-v1.5-7b-32k
```

Expected artifacts:

- `references/repos/Quest/evaluation/passkey/results/<MODEL>/`
- `references/repos/Quest/evaluation/LongBench/*` prediction files
- `references/repos/Quest/evaluation/LongBench` eval summaries

Fair-comparison caveats:

- Quest is same-architecture long-context sparsity, not cross-model transport
- use repo-native metrics only within Quest
- do not compare Quest ROUGE/F1 to our GSM exact-match as if they were the same number
- if we want Quest on Qwen3, that is a compatibility patch, not a clean bootstrap

## 3) tokenkit

Why this is next:

- it isolates tokenizer/interface mismatch, which is one of our main suspected blockers
- it is the cleanest local way to test whether a shared tokenizer or byteified interface improves downstream alignment
- it gives us a training-heavy control that can separate “cache geometry” from “tokenization geometry”

Required models and datasets:

- model/tokenizer pairs such as `llama_qwen` or `gemma_llama_qwen`
- `benjamin/Llama-3.2-3B-Instruct-flax`
- `meta-llama/Llama-3.2-3B-Instruct:source=Llama3`
- `Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3`
- `allenai/tulu-3-sft-mixture`
- eval tasks: `arc_easy`, `arc_challenge`, `piqa`, `hellaswag`, `boolq`, `arithmetic`, `mmlu`, `ifeval`, `agieval_en`, `agieval_cn`

Runnable commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/tokenkit
bash examples/llama3_to_qwen2_tokenizer_gpu.sh
bash examples/llama3_to_byte_tokenizer_gpu.sh
python3 scripts/eval_lockstep.py models=llama_qwen +eval.limit=100
python3 scripts/eval.py \
  model.pretrained_model_name_or_path='benjamin/Gemma2-2B-IT-with-Qwen2-Tokenizer' \
  model.tokenizer_name='benjamin/Gemma2-2B-IT-with-Qwen2-Tokenizer:source=Gemma2:conversion=prebyteified' \
  eval.tasks=[mmlu]
```

Expected artifacts:

- `references/repos/tokenkit/outputs/cross_tokenizer_distill/`
- `references/repos/tokenkit/outputs/eval_lockstep/`
- exported checkpoints from `scripts/export_checkpoint.py`

Fair-comparison caveats:

- tokenkit is tokenizer/interface transfer, not cache compression
- any result here should be interpreted as evidence about tokenizer mismatch, not direct evidence about latent transport quality
- use the same tokenizer pair and the same task split when comparing transfer variants

## Practical Interpretation Rules

- Keep metrics inside their own family: GSM exact-match, LongBench F1/ROUGE, passkey retrieval, and lockstep eval are not interchangeable.
- Keep decoding budget fixed inside each comparison.
- Keep the same model family fixed when comparing Quest or KVzip variants.
- Use tokenkit to decide whether tokenizer mismatch is a first-order blocker before adding more routing complexity.

## What This Bootstrap Should Answer

- If `KVzip` wins but `Quest` does not, our bottleneck is probably future-query robustness, not raw compression.
- If `Quest` wins but `KVzip` does not, the bottleneck is more likely query-aware sparsity than cache reconstruction.
- If tokenkit materially improves downstream alignment or lockstep eval, the bottleneck is likely tokenizer/interface mismatch, not just KV geometry.
- If none of them help on controlled smoke runs, the next branch should be learned query-conditioned slots or tokenizer-aware bridge preconditioning, not another fixed sparse rule.
