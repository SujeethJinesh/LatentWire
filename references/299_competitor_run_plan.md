# Competitor Run Plan After Span-ALM Negative

This note is a concise, execution-oriented triage of the most relevant
competitor baselines we can still use against the current LatentWire story.
It focuses on what is actually cheap / runnable locally on this Mac/MPS setup
versus what is only fair on a CUDA box.

Primary-source anchors:
- **KVPress / Expected Attention** — arXiv 2510.00636, 2025-10-01
  https://arxiv.org/abs/2510.00636
- **KVzip** — arXiv 2505.23416, 2025-05-29
  https://arxiv.org/abs/2505.23416
- **Quest** — arXiv 2406.10774, 2024-06-16
  https://arxiv.org/abs/2406.10774
- **KVComm** — arXiv 2510.03346, 2025-10-02; OpenReview ICLR 2026
  https://arxiv.org/abs/2510.03346
  https://openreview.net/forum?id=F7rUng23nw
- **ALM / tokenkit** — arXiv 2503.20083, 2025-03-25
  https://arxiv.org/abs/2503.20083
  https://github.com/bminixhofer/tokenkit
- **C2C** — arXiv 2510.03215, 2025-10-03
  https://arxiv.org/abs/2510.03215

## What is already cloned locally

- `references/repos/kvpress`
- `references/repos/KVzip`
- `references/repos/Quest`
- `references/repos/KVComm`
- `references/repos/tokenkit`
- `references/repos/C2C`

## Current local evidence

I did not find any checked-in competitor result files in these clones:
- no `metrics.json`
- no generated prediction CSV/JSONL
- no benchmark outputs

So there is no local competitor telemetry to reuse yet.

## Best next local benchmark: KVPress

This is the cheapest run that is still directly useful after the span-ALM
negative. It gives a same-model proxy for query-aware KV compression, plus a
null.

### Local setup

Repo docs say:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress
pip install -r requirements.txt
pip install -e .
```

For this Mac/MPS environment, use `attn_implementation: eager` in config.
Do **not** depend on flash-attn.

### Exact run plan

Use `math500` as the cheapest reasoning proxy.
Compare:

1. `no_press`
2. `expected_attention`

Suggested config shape:

```yaml
dataset: math500
model: Qwen/Qwen3-0.6B
query_aware: true
fraction: 0.2
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500
log_level: INFO
model_kwargs:
  attn_implementation: eager
```

Commands:

```bash
cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_no_press.yaml <<'YAML'
dataset: math500
model: Qwen/Qwen3-0.6B
press_name: no_press
compression_ratio: 1.0
query_aware: true
fraction: 0.2
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML

cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_expected_attention.yaml <<'YAML'
dataset: math500
model: Qwen/Qwen3-0.6B
press_name: expected_attention
compression_ratio: 0.5
query_aware: true
fraction: 0.2
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML

cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_no_press.yaml
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_expected_attention.yaml
```

### Fair comparison to LatentWire

- Treat `no_press` as the floor.
- Treat `expected_attention` as the query-aware compression comparator.
- Compare against LatentWire only as a **same-model compression/null proxy**,
  not as cross-model communication.
- If the `ExpectedAttention` result ties the `no_press` floor, that is still
  useful: it says query-blind expected-attention compression is not rescuing
  the current branch.

### Blockers on this Mac/MPS box

- Expect slower-than-CUDA execution.
- `flash-attn` is not the right path here.
- If `Qwen/Qwen3-0.6B` is not already cached in the HF store, swap to any
  locally cached Qwen3 checkpoint supported by `kvpress`.
- This run is still cheap enough to be useful because it does not need a
  custom harness.

## Next if we want a stronger compression comparator: KVzip

### What it gives

- Query-agnostic KV compression with context reconstruction.
- Strong same-model compression baseline for a more demanding comparison.

### Exact run plan from repo docs

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
```

### Fair comparison to LatentWire

- Same-model compression comparator only.
- Useful as a ceiling for compression quality, not for cross-model semantic
  communication.

### Blockers on Mac/MPS

- Heavy CUDA / flash-attn assumptions.
- README examples are built around Qwen2.5-7B-Instruct-1M and similar large
  GPU-oriented checkpoints.
- This is likely **not** a cheap local run on this machine.

### Harness glue needed

- None for the repo itself, but a fair LatentWire comparison would need a
  separate reporting table because the benchmark and output format differ from
  our current GSM-style eval loop.

## Strong query-aware sparsity comparator: Quest

### What it gives

- Query-aware sparsity on long-context inference.
- Good if we want a direct “query matters” comparator.

### Exact run plan from repo docs

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
```

### Fair comparison to LatentWire

- Good query-aware long-context baseline.
- Still not a cross-model communication benchmark.

### Blockers on Mac/MPS

- CUDA / FlashInfer / kernel-build overhead.
- The README is centered on LongChat and Yarn-Llama2 setups, so this is not a
  natural match to the current heterogeneous Qwen pair.

### Harness glue needed

- No LatentWire code edits, but you would need a benchmark-specific result
  table if you want to compare it honestly against our current GSM readout.

## Direct communication comparator: KVComm

### What it gives

- The nearest direct inter-model KV sharing baseline.

### Exact run plan from repo docs

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
pip install -r requirements.txt
python com.py --test_task hotpotqa --do_test_baseline --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --top_layers 0.3
```

### Fair comparison to LatentWire

- This is the direct communication bar.
- For the current heterogeneous Qwen pair, the LatentWire replay remains the
  fairer apples-to-apples comparison.

### Blockers on Mac/MPS

- Very likely CUDA-first in practice.
- Uses large Llama-family models in the README examples.
- Not a cheap local smoke for this machine.

### Harness glue needed

- Yes, if we want a fair LatentWire comparison table: we need to keep the
  pairing and dataset framing consistent with our current controlled Qwen
  readout.

## Future tokenizer-side pivot: tokenkit / ALM

### What it gives

- Cross-tokenizer distillation and tokenizer transfer.
- Best future bridge if transport keeps saturating.

### Exact run plan from repo docs

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/tokenkit
pip install -r requirements.txt
pip install -e .
bash examples/llama3_to_byte_tokenizer_gpu.sh
bash scripts/cross_tokenizer_distill.py
bash scripts/zett.py
```

### Fair comparison to LatentWire

- Not a current inference-time competitor baseline.
- Best framed as a tokenizer-side bridge / future pivot.

### Blockers on Mac/MPS

- JAX-first stack.
- README warns it currently wants Python <= 3.10.
- Several examples are GPU-oriented.

### Harness glue needed

- Yes. This is training / transfer code, so it is not a drop-in comparator for
  the current LatentWire inference harness.

## External communication ceiling: C2C

### What it gives

- The strongest direct cache-to-cache communication competitor already cloned.

### Exact run plan from repo docs

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
pip install -e .
pip install -e ".[training,evaluation]"
python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml
```

### Fair comparison to LatentWire

- Use as the external ceiling / quality comparator.
- Do not claim byte parity with LatentWire unless you explicitly normalize
  payloads.

### Blockers on Mac/MPS

- Big-model / GPU-heavy.
- Needs a proper eval recipe and likely a CUDA-capable environment to be
  practical.

### Harness glue needed

- Yes, if we want a fair paired comparison to the current heterogeneous Qwen
  LatentWire setup.

## Bottom line

If we want one cheap baseline **right now on this Mac/MPS box**, run
`kvpress`:

- `no_press`
- `expected_attention`

If we want a stronger compression comparator later, use `KVzip`.
If we want a query-aware sparsity comparator, use `Quest`.
If we want the direct communication bar, use `KVComm` and `C2C`.
If we want a tokenizer-side pivot, use `tokenkit` / ALM, but that is not a
cheap comparator for the current benchmark loop.

## Local Smoke Executed 2026-04-20

I bootstrapped KVPress enough to run its evaluator on the current Mac/MPS
environment:

- installed the missing optional eval packages in `venv_arm64`: `jieba`,
  `rouge`, `fuzzywuzzy`, `bert-score`, and `nltk`
- used `PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress`
  because `evaluation/evaluate.py` does not add the repo root by default
- used `torch_dtype: auto` instead of `dtype: auto` because the local
  Transformers build rejects `dtype` for Qwen3 model construction
- used `attn_implementation: eager` for `no_press`, but `sdpa` for
  `expected_attention` because ExpectedAttention asserts against eager mode

Exact smoke:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
  /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_no_press.yaml
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
  /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_expected_attention.yaml
```

Smoke metrics on `math500`, `Qwen/Qwen3-0.6B`, `fraction=0.01`, `5` examples,
`max_new_tokens=64`, `query_aware=true`:

| Press | Compression | Accuracy | Answered | Total | Result path |
|---|---:|---:|---:|---:|---|
| `no_press` | `0.00` | `0.0000` | `0` | `5` | `.debug/kvpress_math500/math500__Qwen--Qwen3-0.6B__no_press__0.00__fraction0.010__query_aware/1/metrics.json` |
| `expected_attention` | `0.50` | `0.0000` | `0` | `5` | `.debug/kvpress_math500/math500__Qwen--Qwen3-0.6B__expected_attention__0.50__fraction0.010__query_aware/1/metrics.json` |

Interpretation:

- the KVPress harness is runnable locally now
- this smoke is not yet a meaningful competitive accuracy bar because the
  underlying Qwen3 math500 run produced `answered=0/5` even for `no_press`
- the next fair competitor step should tune the prompt/answer extraction for
  Qwen3 math500 or switch to a scorer where Qwen3 outputs are recognized before
  comparing against LatentWire claims
