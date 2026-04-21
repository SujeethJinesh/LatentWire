# Competitor Next Runs

Scope: cheap competitor baselines we can run locally next, with emphasis on
KV compression / cross-model communication comparators that help explain the
current LatentWire blocker story.

Primary sources used for the triage:
- **Expected Attention / KVPress** — arXiv:2510.00636, Oct 1 2025
  https://arxiv.org/abs/2510.00636
- **KVzip** — arXiv:2505.23416, May 29 2025
  https://arxiv.org/abs/2505.23416
- **Quest** — ICML 2024, Jun 16 2024
  https://arxiv.org/abs/2406.10774
- **KVComm** — arXiv:2510.03346, Oct 2 2025; OpenReview/ICLR 2026
  https://arxiv.org/abs/2510.03346
  https://openreview.net/forum?id=F7rUng23nw
- **Approximate Likelihood Matching (ALM)** / tokenkit — arXiv:2503.20083, Mar 25 2025; tokenkit release Apr 2 2025
  https://arxiv.org/abs/2503.20083
  https://github.com/bminixhofer/tokenkit

## Cheapest runnable benchmark to do next

### 1) `kvpress` on `math500`

Why this one:
- It is the cheapest local comparator that still maps to our current
  Qwen-side story.
- It gives a clean same-model proxy for query-aware compression using
  `ExpectedAttentionPress`.
- It also gives the null floor via `no_press`.

What to run:
- `no_press`
- `expected_attention`

Suggested setup:
- model: `Qwen/Qwen3-0.6B` if locally available
- dataset: `math500`
- fraction: `0.2`
- query-aware: `true`
- attention backend: `eager`

Exact commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress
pip install -r requirements.txt
pip install -e .
```

Create temp configs:

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

How to compare it to LatentWire:
- Treat `no_press` as the null / floor.
- Treat `expected_attention` as the query-aware compression comparator.
- Compare the score trend, not bytes, because this is a same-model proxy and
  not a cross-model communication baseline.

Expected blockers:
- This is not cross-model communication.
- On Apple Silicon, it will be slower than CUDA, but it is the cheapest
  immediately runnable option.
- If `Qwen/Qwen3-0.6B` is not cached locally, switch to the nearest supported
  Qwen3 checkpoint already available in the HF cache.

## Stronger but heavier next baselines

### 2) `KVzip`

Why:
- Query-agnostic KV compression with context reconstruction.
- Good same-model compression bar, especially if we later want a stronger
  compression ceiling than `kvpress`.

What is runnable:
- `pip install -r requirements.txt`
- `pip install flash-attn==2.7.4.post1 --no-build-isolation`
- `python -B eval.py -m [model_name] -d [data_name] --kv_type retain --num 100`

Best comparison role:
- Same-model compression benchmark, not cross-model comms.

Blockers:
- Heavier setup than `kvpress`.
- Best-supported models in README are LLaMA3 / Qwen2.5 / Qwen3 / Gemma3,
  but it is still a larger, more GPU-centric stack.

### 3) `Quest`

Why:
- Strong query-aware sparsity baseline.
- Useful if we want to argue about query-conditioned token criticality.

What is runnable:
- `bash scripts/passkey.sh`
- `bash scripts/longbench.sh`
- kernel tests under `quest/tests`

Best comparison role:
- Query-aware long-context sparsity comparator, not a cross-model benchmark.

Blockers:
- CUDA / FlashInfer / kernel build overhead.
- README targets LongChat / Yarn-Llama2-style setups, so it is not a direct
  drop-in for the current heterogeneous Qwen pair.

### 4) `KVComm`

Why:
- The closest direct cross-model communication comparator.
- Best paper-side contrast for our core claim, but not the cheapest local run.

What is runnable:
- `pip install -r requirements.txt`
- `python com.py --test_task hotpotqa --do_test --model_A ... --model_B ... --top_layers 0.3`

Best comparison role:
- Cross-model communication baseline when the pair/model setup is fair.

Blockers:
- Not a cheap local smoke on the current Mac setup.
- The upstream repo is native to its own communication setting; for the current
  heterogeneous Qwen pair, the LatentWire replay remains the fairer control.

### 5) `tokenkit` / ALM

Why:
- Best tokenizer-side bridge candidate.
- Good future pivot if dense transport stays saturated.

What is runnable:
- `scripts/cross_tokenizer_distill.py`
- `scripts/zett.py`
- examples under `examples/`

Best comparison role:
- Future tokenizer-transfer bridge, not an immediate inference baseline.

Blockers:
- Training-heavy, JAX-first stack.
- Not a cheap benchmark for the current paper’s local run loop.

## Recommendation

If we only run one competitor benchmark next, run `kvpress`:

1. `no_press`
2. `expected_attention`

That is the cheapest local comparator that still tells us something useful
about the current Qwen-side control story.

If we want a stronger compression bar afterward, use `KVzip`.
If we want a query-aware long-context baseline, use `Quest`.
If we want the true direct communication bar, keep `KVComm` as the paper-side
benchmark and the LatentWire replay as the fair heterogeneous-pair control.
If the transport lane fully saturates, `tokenkit`/ALM becomes the best future
tokenizer-side pivot rather than an immediate baseline.
