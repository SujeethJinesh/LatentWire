# Competitor Benchmark Bootstrap for LatentWire

Scope: competitor baselines we can run now, plus the minimum fair-comparison ladder for LatentWire. I am not cloning anything because the required local repos are already present under `references/repos/`.

## Bottom Line

Keep the benchmark story split into three buckets:

1. Direct cross-model communication peers: `C2C`, `KVComm`
2. Matched-byte / cache-side controls: `Quest`, `KVzip`, `kvpress`, `DeltaKV_sparse_vllm`
3. Interpretation controls: tokenizer/interface-transfer and geometry-only ablations from the existing references

Do not collapse these into one table. They answer different questions.

## Runnable Status

| Repo | Local status | Primary entrypoints | Notes |
| --- | --- | --- | --- |
| `C2C` | Clean local clone present at `references/repos/C2C` | `script/evaluation/unified_evaluator.py`, `script/evaluation/kv_cache_evaluate.py`, `bash/eval/run_eval.sh`, `script/dataset/run_generation.sh` | Best direct peer for semantic cache communication. README shows example pair `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`. |
| `KVComm` | Local clone present, but dirty (`model_attn.py`, `models.py` modified locally) | `com.py`, `eval.py`, `eval_online.py`, `eval_ms.py` | Direct selective-KV-sharing peer. README exposes cross-model communication modes and dataset/task list. |
| `Quest` | Clean local clone present at `references/repos/Quest` | `bash scripts/passkey.sh`, `bash scripts/longbench.sh`, `bash scripts/ppl_eval.sh`, `evaluation/LongBench/eval.py`, `evaluation/passkey/passkey.py` | Query-aware sparsity control; best for matched query-conditioned cache reduction. |
| `KVzip` | Clean local clone present at `references/repos/KVzip` | `test.py`, `eval.py`, `utils/tester.py` | Query-agnostic KV eviction with context reconstruction. Good same-family compression ceiling. |
| `kvpress` | Local clone present, but dirty (`kvpress/pipeline.py`, `kvpress/presses/base_press.py`, `kvpress/utils.py` modified locally) | `evaluation/evaluate.py`, `kvpress` pipeline API, `Makefile` | Main cache-compression control library. Use `ExpectedAttention`, `SnapKV`, `StreamingLLM`, `Quest`, `KVzip`, `AdaKV`, `PerLayerCompressionPress`, etc. |
| `DeltaKV_sparse_vllm` | Clean local clone present at `references/repos/DeltaKV_sparse_vllm` | `benchmark/long_bench/eval.py`, `benchmark/scbench/run_scbench.py`, `benchmark/niah/test_niah.py`, `benchmark/math_bench/eval.py`, `scripts/run_llama_ablation.sh` | Sparse-first engine and hybrid compression controls; useful for systems-level matched-byte comparisons. |

## What To Run

### 1) C2C

Use this when the claim is direct latent communication between different model families.

Published/reference anchor:

- Paper: https://arxiv.org/abs/2510.03215
- Code: https://github.com/thu-nics/C2C

Local smoke path:

```bash
python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --prediction-output results/c2c_gsm70_<run>.jsonl
```

Likely metrics:

- exact-match accuracy
- paired delta vs target-alone
- latency
- generated tokens

Required setup:

- source/target pair fixed across all runs
- held-out GSM split fixed
- same decoding budget across methods

### 2) KVComm

Use this when the claim is selective KV sharing between contexts.

Published/reference anchor:

- Paper: https://arxiv.org/abs/2510.12872
- Code: local clone under `references/repos/KVComm`

Local smoke path:

```bash
cd references/repos/KVComm
python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct

python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3
```

Datasets/tasks from the repo README:

- `hotpotqa`
- `qasper`
- `musique`
- `multifieldqa_en`
- `twowikimqa`
- `tipsheets`
- `countries`
- `tmath`

Likely metrics:

- task score / accuracy
- paired delta vs baseline
- selected top-layer set
- calibration sweep trace
- token usage and latency

Required setup:

- `transformers==4.53.3` per repo README
- same task split and decoding budget across all communication variants

### 3) Quest

Use this when the claim is query-aware cache sparsity.

Published/reference anchor:

- Paper: https://arxiv.org/abs/2406.10774
- Code: https://github.com/mit-han-lab/Quest

Local smoke path:

```bash
cd references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
bash scripts/ppl_eval.sh
```

Datasets/models called out in the repo README:

- `LongChat-7B-v1.5-32K`
- `Yarn-Llama2-7B-128K`
- Passkey retrieval
- LongBench
- PG19 perplexity

Likely metrics:

- passkey exact retrieval
- LongBench task scores
- PG19 perplexity
- kernel latency / throughput
- end-to-end speedup

Required setup:

- flash-attn and kernel build steps from the repo README
- model family aligned to the chosen benchmark

### 4) KVzip

Use this when the claim is query-agnostic compression with reconstruction.

Published/reference anchor:

- Paper: https://arxiv.org/abs/2505.23416
- Code: https://github.com/snu-mllab/KVzip

Local smoke path:

```bash
cd references/repos/KVzip
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
```

Datasets/models called out in the repo README:

- `SQuAD`
- `NIAH`
- `SCBench`
- `GSM8K`
- `Qwen2.5-7B-Instruct-1M`
- `LLaMA3`
- `Gemma3`

Likely metrics:

- accuracy / F1 / ROUGE depending on task
- compression ratio
- reconstruction quality
- runtime overhead

Required setup:

- the repo’s `requirements.txt`
- flash-attn per README
- model/dataset naming matched to the repo’s `data/load.py` and `model/load.py`

### 5) kvpress

Use this when the claim is a general cache-compression baseline family.

Published/reference anchors:

- Paper: https://arxiv.org/abs/2510.00636
- Code: https://github.com/NVIDIA/kvpress

Local smoke path:

```bash
python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --prediction-output results/tmp_kvpress_none_gsm5.jsonl

python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --prediction-output results/tmp_kvpress_expected_gsm5.jsonl
```

Repo-native benchmark families called out in the README:

- `longbench`
- `longbenchv2`
- `ruler`
- `needle_in_haystack`
- `infinite_bench`
- `math500`
- `aime25`
- `loogle`

Likely metrics:

- accuracy / task score
- compression ratio
- latency
- tokens/sec
- peak memory

Required setup:

- `pip install kvpress` or local `uv sync`
- model family supported by the press
- benchmark-native metric parser for the chosen suite

### 6) DeltaKV_sparse_vllm

Use this when the claim is sparse-first systems support for compression and runtime efficiency.

Published/reference anchor:

- Paper: https://arxiv.org/abs/2602.08005
- Local repo: `references/repos/DeltaKV_sparse_vllm`

Local smoke path:

```bash
python benchmark/long_bench/eval.py --help
python benchmark/scbench/run_scbench.py --help
python benchmark/niah/test_niah.py --help
python benchmark/math_bench/eval.py --help
```

Common benchmark families in the repo:

- `LongBench`
- `SCBench`
- `NIAH`
- `MathBench`

Likely metrics:

- task accuracy / F1 / ROUGE depending on benchmark
- throughput
- peak memory
- prefill / decode split

Required setup:

- the repo’s `pyproject.toml` / `requirements.txt`
- model path and tokenizer path aligned
- sparse-vLLM backend parameters fixed across compared methods

## Minimal Fair-Comparison Ladder

Use this order.

1. `Target-alone` baseline on the exact held-out split, exact same decoding budget.
2. Direct peer baseline:
   - `C2C`
   - `KVComm`
3. Same-family cache control:
   - `Quest`
   - `KVzip`
   - `kvpress`
   - `DeltaKV_sparse_vllm`
4. Byte-matched or context-matched ablations inside our own method family.
5. Tokenizer/interface transfer controls only after cache-side comparisons are stable.

## What To Measure

Collect these everywhere, even if a benchmark has its own native score.

- exact-match / task score / F1 / ROUGE
- paired delta vs target-alone
- generated token count
- latency
- tokens/sec
- peak memory
- effective bytes or cache budget
- calibration trace, if the method has one
- selected layer IDs / head IDs / token IDs, if relevant
- split, seed, and decoding budget

## Non-Negotiables For Fairness

- Keep the same source/target pair inside each direct communication benchmark.
- Keep the same split and example order for paired deltas.
- Do not compare different metric families as if they were interchangeable.
- Do not mix a compression-ratio claim with a latency claim.
- If a run requires a different tokenizer or context budget, record that as a different control bucket.
- Do not count a toy-slice smoke test as a paper result.

## Current Blockers

- `KVComm` and `kvpress` are runnable locally, but the repos are dirty, so any code-level comparison must avoid assuming a clean vendor state.
- `Quest` and `DeltaKV_sparse_vllm` are heavier setup paths; the entrypoints are visible, but exact benchmark parity still needs model/dataset alignment.
- `C2C` is the cleanest direct peer, but the local smoke path still needs the exact held-out split and parser parity to be paper-safe.

## Web Sources Used

- C2C: https://arxiv.org/abs/2510.03215 and https://github.com/thu-nics/C2C
- KVComm: https://arxiv.org/abs/2510.12872
- Quest: https://arxiv.org/abs/2406.10774 and https://github.com/mit-han-lab/Quest
- KVzip: https://arxiv.org/abs/2505.23416 and https://github.com/snu-mllab/KVzip
- kvpress: https://github.com/NVIDIA/kvpress and https://arxiv.org/abs/2510.00636
- DeltaKV_sparse_vllm: https://arxiv.org/abs/2602.08005
