# Competitor Bootstrap Status

Scope: what is already cloned or locally runnable for LatentWire-style
comparison, with exact commands, blockers, and fairness notes.

Primary-source dates:
- **KVPress / Expected Attention** — arXiv 2510.00636, **2025-10-01**
  https://arxiv.org/abs/2510.00636
- **KVzip** — arXiv 2505.23416, **2025-05-29**
  https://arxiv.org/abs/2505.23416
- **Quest** — ICML 2024 / arXiv 2406.10774, **2024-06-16**
  https://arxiv.org/abs/2406.10774
- **KVComm** — arXiv 2510.03346, **2025-10-02**; ICLR 2026 OpenReview
  https://arxiv.org/abs/2510.03346
  https://openreview.net/forum?id=F7rUng23nw
- **Cross-Tokenizer Distillation via Approximate Likelihood Matching (ALM)** / `tokenkit`
  — arXiv 2503.20083, **2025-03-25**
  https://arxiv.org/abs/2503.20083
  https://github.com/bminixhofer/tokenkit
- **Cache-to-Cache (C2C)** — arXiv 2510.03215, **2025-10-03**
  https://arxiv.org/abs/2510.03215

## Local clones already present

- `references/repos/kvpress`
- `references/repos/KVzip`
- `references/repos/Quest`
- `references/repos/KVComm`
- `references/repos/tokenkit`
- `references/repos/C2C`

## No generated results found yet

I found no checked-in `metrics.json`, prediction CSV/JSONL, or other generated
benchmark outputs inside the cloned competitor repos. The only local result-ish
artifacts are code helpers:

- `references/repos/KVzip/results/parse.py`, `metric.py`, `repo_qa_utils.py`
- `references/repos/kvpress/evaluation/evaluate_config.yaml`

So there is no existing competitor telemetry to reuse yet.

## Cheapest run next: KVPress

This is the best immediate local comparator because it is already cloned,
supports `ExpectedAttentionPress` and `no_press`, and can be used as a
same-model compression/null proxy.

### What it compares to in LatentWire

- `no_press` = null / floor
- `expected_attention` = query-aware compression baseline

This is **not** cross-model communication. It is a fair same-model proxy for
the query-aware compression side of the story.

### Exact setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress
pip install -r requirements.txt
pip install -e .
```

### Exact commands

Create two temporary configs:

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
```

Run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_no_press.yaml
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_expected_attention.yaml
```

### Expected blockers

- On this machine, `eager` attention is the safe path; do not expect a fast
  CUDA-style run.
- If `Qwen/Qwen3-0.6B` is not cached locally, swap to another supported Qwen3
  checkpoint already present in the HF cache.
- Use `math500` as a cheap reasoning proxy; it is not a cross-model benchmark.

## Heavier but still local candidates

### KVzip

What it is:
- Query-agnostic KV compression with context reconstruction.

Why it matters for LatentWire:
- Good same-model compression ceiling.
- Best as a stronger comparator once the cheap `kvpress` proxy is in place.

Local entry points:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
```

Fair comparison framing:
- Same-model compression comparator, not cross-model communication.

Blockers:
- Heavier CUDA / flash-attn stack.
- README examples target Qwen2.5/3, LLaMA3, Gemma3, but it is still more
  GPU-centric than KVPress.

### Quest

What it is:
- Query-aware sparsity for long-context inference.

Why it matters for LatentWire:
- Useful if we want a clean query-aware sparsity baseline and a “query matters”
  comparator.

Local entry points:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
```

Fair comparison framing:
- Strong query-aware sparsity baseline, but not a native cross-model
  communication comparator.

Blockers:
- CUDA / FlashInfer / kernel build overhead.
- README targets LongChat/Yarn-Llama2-style setups, so it is not a direct drop
  in for the current heterogeneous Qwen pair.

### KVComm

What it is:
- Direct inter-model communication through selective KV sharing.

Why it matters for LatentWire:
- It is the closest paper-level competitor to our claim.

Local entry points:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
pip install -r requirements.txt
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --top_layers 0.3
```

Fair comparison framing:
- Use it as the direct communication bar.
- For the current heterogeneous Qwen pair, the LatentWire replay remains the
  fairer apples-to-apples comparison than the stock repo.

Blockers:
- Not a cheap local smoke on the Mac.
- Native settings are mostly same-family / same-architecture communication.

### tokenkit / ALM

What it is:
- Cross-tokenizer distillation and tokenizer-transfer toolkit.

Why it matters for LatentWire:
- Best future tokenizer-side bridge if dense transport saturates.

Local entry points:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/tokenkit
pip install -r requirements.txt
pip install -e .
bash scripts/cross_tokenizer_distill.py
```

Fair comparison framing:
- Not an immediate inference baseline.
- Best treated as a future tokenizer-side pivot rather than a current
  competitor result.

Blockers:
- Training-heavy JAX stack.
- Python <= 3.10 guidance in README.

### C2C

What it is:
- Direct cache-to-cache semantic communication.

Why it matters for LatentWire:
- The strongest direct cross-model communication ceiling already cloned locally.

Local entry points:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
pip install -e .
pip install -e ".[training,evaluation]"
python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml
```

Fair comparison framing:
- Treat as the external ceiling / quality comparator.
- It is not byte-matched to the current LatentWire sparse branches unless we
  explicitly normalize payloads.

Blockers:
- More setup than `kvpress`.
- The easiest fair run likely needs a custom eval recipe for the Qwen pair.

## Bottom line

If the goal is one cheap, locally runnable competitor check next, run
`kvpress`:

- `no_press`
- `expected_attention`

If the goal is a stronger but heavier comparator, go to `KVzip`.
If the goal is a direct communication ceiling, use `KVComm` and `C2C`.
If the goal is a future tokenizer-side pivot, `tokenkit` / ALM is the right
place to look.
