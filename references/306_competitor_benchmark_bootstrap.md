# Competitor Benchmark Bootstrap

Scope: competitor baselines we can actually run now, or approximate cleanly enough to be useful for the paper. I am not touching other files.

## Bottom Line

The paper should keep three buckets separate:

1. `cross-model communication` baselines that are direct peers (`C2C`, `KVComm`, `LatentMAS`)
2. `same-family cache-compression` controls (`KVPress`, `KVzip`, `Quest`, `SnapKV`)
3. `tokenizer / interface-transfer` controls (`TokenKit` / cross-tokenizer distillation)

The first bucket is the direct competitor table. The second and third buckets are control tables that explain which bottleneck is actually dominating.

## 1) C2C

Status:

- Already cloned locally at `references/repos/C2C`
- Exact local wrapper exists in [scripts/bootstrap_c2c.py](/Users/sujeethjinesh/Desktop/LatentWire/scripts/bootstrap_c2c.py) and [scripts/run_c2c_eval.py](/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_c2c_eval.py)
- This is the cleanest direct semantic-communication comparator for our current `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` pair

Minimal smoke:

```bash
python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --download

python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --limit 5 \
  --prediction-output results/c2c_gsm70_20260420/qwen_gsm70_c2c.jsonl
```

Expected outputs:

- JSON metrics printed to stdout
- `results/c2c_gsm70_20260420/qwen_gsm70_c2c.jsonl`
- matching `.meta.json` sidecar with run config and metrics

Paper numbers to collect:

- exact-match accuracy on `gsm8k_eval_70`, `gsm8k_100`, `svamp_eval_70`
- paired delta vs target-alone
- `tokens/sec`, `latency_sec`, generated token count
- if we run any calibration sweep, keep the selected layer map and the sweep trace

Useful local evidence already in the tree:

- `latent_bridge/competitor_baselines_20260418.md`
- `scripts/build_bytes_accuracy_table.py`

## 2) KVComm

Status:

- Already cloned locally at `references/repos/KVComm`
- Exact local wrapper exists in [scripts/run_kvcomm_eval.py](/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvcomm_eval.py)
- This is the direct selective-KV-sharing baseline

Minimal smoke:

```bash
python scripts/run_kvcomm_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file data/gsm8k_100.jsonl \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --top-layers-grid 0.25,0.5,0.75,1.0 \
  --prediction-output results/kvcomm_gsm70_20260420/qwen_gsm70_kvcomm.jsonl
```

Tiny repo-native proof of life:

```bash
cd references/repos/KVComm
python com.py --test_task hotpotqa --do_test_baseline --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --limit 5
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --top_layers 0.3 --limit 5
```

Repo-native reference command:

```bash
cd references/repos/KVComm
python com.py --test_task hotpotqa --do_test_baseline --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --top_layers 0.3
```

Expected outputs:

- JSON metrics from the wrapper
- `results/kvcomm_gsm70_20260420/qwen_gsm70_kvcomm.jsonl`
- `.meta.json` with the selected layer subset and calibration sweep

Paper numbers to collect:

- exact-match accuracy on `gsm8k_eval_70`, `gsm8k_100`, `svamp_eval_70`
- selected layer count / selected layer IDs
- paired delta vs target-alone and vs `C2C`
- calibration sweep trace
- latency / generated tokens

Notes:

- The repo itself is not Qwen3-native; our wrapper already carries the compatibility lift.
- If a run fails, that is itself a useful blocker signal and should be logged, not hidden.

## 3) LatentMAS

Status:

- Already cloned locally at `references/repos/LatentMAS`
- It is a multi-agent latent-collaboration control, not a strict KV-transport peer
- Best used as a broader latent-collaboration context comparator, not the main direct table row

Minimal smoke:

```bash
cd references/repos/LatentMAS
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples 5 --max_new_tokens 256
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples 5 --max_new_tokens 256
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples 5 --max_new_tokens 256
```

Expected outputs:

- task accuracy / score in their native logs
- token usage
- wall-clock time
- optional `--latent_space_realign` ablation if we want to test whether geometry repair matters

Paper numbers to collect:

- GSM8K / AIME / GPQA / ARC-E / ARC-C / MBPP+ / HumanEval+ / MedQA if we can afford them
- token count reduction
- wall-clock reduction
- effect of `--latent_space_realign`

Interpretation:

- This is a latent-collaboration systems control, not a direct transport baseline.
- Use it to argue whether “latent communication at all” helps, not whether our cache bridge beats a specific KV-sharing protocol.

## 4) KVPress

Status:

- Already cloned locally at `references/repos/kvpress`
- Exact local wrapper exists in [scripts/run_kvpress_eval.py](/Users/sujeethjinesh/Desktop/LatentWire/scripts/run_kvpress_eval.py)
- The wrapper currently exposes `none` and `expected_attention`
- The vendored repo exposes the full press registry, including `snapkv`

Minimal smoke with the local wrapper:

```bash
python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --prediction-output results/kvpress_wrapper_smoke_20260420/qwen3_gsm70_no_press.jsonl

python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --prediction-output results/kvpress_wrapper_smoke_20260420/qwen3_gsm70_expected_attention_cr050.jsonl
```

Exact SnapKV smoke through the vendored repo:

```bash
cd references/repos/kvpress/evaluation
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snapkv --compression_ratio 0.5
```

Expected outputs:

- wrapper JSONL plus `.meta.json`
- direct KVPress evaluation summary and leaderboard-ready output directory

Paper numbers to collect:

- same-model compression accuracy on a held-out slice
- compression ratio
- latency / tokens/sec
- `no_press` floor versus `expected_attention` and `snapkv`

Interpretation:

- This is a strong same-model compression boundary, not a cross-model communication baseline.
- Use it to separate “query-aware compression” from “heterogeneous transport.”

## 5) KVzip

Status:

- Already cloned locally at `references/repos/KVzip`
- Exact repo-native smoke is `test.py` / `eval.py`
- This is the strongest same-model query-agnostic compression control we have locally

Minimal smoke:

```bash
cd references/repos/KVzip
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
```

Expected outputs:

- `results/<data_name>/...`
- `utils/head_score/*.pt` when head scores are saved
- summary parsed with `python -B -m results.parse`

Paper numbers to collect:

- same-family accuracy / F1 / ROUGE on SQuAD, NIAH, SCBench, GSM8K
- compression ratio
- context-reconstruction quality
- runtime overhead

Interpretation:

- Compare only within the same model family and benchmark family.
- Treat KVzip as a compression/reconstruction ceiling, not a direct peer to cross-model communication.

## 6) Quest

Status:

- Already cloned locally at `references/repos/Quest`
- Exact repo-native scripts exist: `scripts/passkey.sh`, `scripts/longbench.sh`, `scripts/ppl_eval.sh`, `scripts/example_demo.py`
- This is the cleanest query-aware sparsity control

Minimal smoke:

```bash
cd references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
```

Optional kernel / correctness smoke:

```bash
cd references/repos/Quest/quest/tests
PYTHONPATH=$PYTHONPATH:../../ pytest
```

Expected outputs:

- passkey retrieval results
- LongBench prediction files and eval summaries
- kernel-level correctness or speed benchmarks if you are on a CUDA host

Paper numbers to collect:

- retrieval accuracy / LongBench F1 or ROUGE
- token budget
- latency / speedup
- if we do any budget sweep, report the best budget and the curve, not just one point

Interpretation:

- Quest is a long-context sparsity control, not a cross-model transport method.
- Use it as a query-aware pruning comparator.

## 7) SnapKV

Status:

- No separate local SnapKV clone is needed for a first pass because KVPress already vendors `SnapKVPress`
- If we want the original codepath for paper parity or kernel-level inspection, clone:

```bash
git clone https://github.com/FasterDecoding/SnapKV.git
```

Recommended first approximation:

- run `SnapKVPress` through `references/repos/kvpress/evaluation/evaluate.py`
- do not wait for the standalone clone unless you need an exact implementation check

Paper numbers to collect:

- same-model long-context accuracy under the same compression ratio as `ExpectedAttentionPress`
- latency and memory
- whether the gain comes from query-awareness or just window retention

## 8) TokenKit / Cross-Tokenizer Distillation

Status:

- Already cloned locally at `references/repos/tokenkit`
- Exact local scripts exist:
  - `scripts/cross_tokenizer_distill.py`
  - `scripts/zett.py`
  - `scripts/eval_lockstep.py`
  - `scripts/eval.py`
  - examples under `examples/`
- This is the cleanest tokenizer/interface control

Minimal smoke:

```bash
cd references/repos/tokenkit
bash examples/llama3_to_qwen2_tokenizer_gpu.sh
bash examples/llama3_to_byte_tokenizer_gpu.sh
python3 scripts/eval_lockstep.py models=llama_qwen eval.tasks=[mmlu]
python3 scripts/cross_tokenizer_distill.py --config=configs/cross_tokenizer_distill.yaml
```

Byte-level / transferred-model evaluation smoke:

```bash
python3 scripts/eval.py \
  model.pretrained_model_name_or_path='benjamin/Gemma2-2B-IT-Byte' \
  model.tokenizer_name='google/gemma-2-2b-it:source=Gemma2:conversion=byte' \
  expand_model.pretrained_model_name_or_path='benjamin/gemma-2-2b-it-flax' \
  expand_model.tokenizer_name='google/gemma-2-2b-it:source=Gemma2' \
  eval.tasks=[mmlu]
```

Expected outputs:

- transferred checkpoints under `outputs/`
- LM-eval style task scores
- lockstep / ensemble scores
- tokenizer info / alignment metadata if computed

Paper numbers to collect:

- downstream accuracy after tokenizer transfer
- zero-shot tokenizer transfer vs ALM vs baselines
- lockstep / ensemble task scores
- byteification and overlap statistics

Interpretation:

- This is not cache compression.
- It isolates whether tokenizer mismatch is a first-order blocker before we keep adding routing complexity.

## Missing Repo To Clone If Useful

- `FasterDecoding/SnapKV` is the only obvious missing clone worth adding if we want the original SnapKV codepath instead of KVPress-only approximation.
- Everything else in the requested list already exists locally.

## What Numbers We Need In The Paper

For the main paper, the useful numbers are:

1. Direct peer baselines: exact-match / F1 / pass@k / task score on the same held-out split, plus paired delta vs target-alone.
2. Communication cost: bytes transferred, effective cache ratio, generated token count, and any calibration cost.
3. Runtime cost: latency, tokens/sec, TTFT, and any build / setup overhead that materially changes the method story.
4. Control-family metrics: separate tables for compression, sparsity, and tokenizer transfer. Do not mix these with direct cross-model communication scores.
5. Interpretability telemetry: selected layers / heads, entropy, dead-slot or dead-head rate, and paired-win counts.

## Recommended Table Order

1. `C2C` as the main external bar
2. `KVComm` as the direct selective-KV-sharing comparator
3. `LatentMAS` as the latent-collaboration control
4. `KVPress` / `SnapKV` / `Quest` / `KVzip` as compression-side controls
5. `TokenKit` as the tokenizer/interface control

If a baseline cannot run cleanly on the current hardware, that should be stated explicitly and the memo should treat it as a CUDA-only or repo-native control rather than pretending it is comparable today.
