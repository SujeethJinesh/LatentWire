# Competitor Run Plan: Cheap Local Baseline

Recommended benchmark:

- **Repo:** `references/repos/kvpress`
- **Why this one:** it is the cheapest local comparator that is still paper-relevant to the current LatentWire blocker story. It gives a fair same-model proxy for query-aware KV compression and includes an explicit null (`no_press`) plus `ExpectedAttentionPress`.
- **Suggested model:** `Qwen/Qwen3-0.6B` if available locally; if that exact checkpoint is unavailable, use the closest locally cached Qwen3 model that `kvpress` already supports.
- **Suggested dataset:** `math500` as the closest cheap reasoning proxy to the current GSM-style readout.

## What to compare

Run two conditions:

1. `no_press` — the null / no-compression floor
2. `expected_attention` — the query-aware comparator

Keep everything else fixed:

- same model
- same dataset
- same fraction of examples
- same seed
- same output dir root

## Exact setup

`kvpress` is easiest to run in an isolated env because it has its own dependency stack.
On this machine, avoid flash-attn on Apple Silicon and use eager attention.

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress

# If you want a clean isolated env, create one outside LatentWire source.
# Example:
# python -m venv ../../.debug/kvpress_env
# source ../../.debug/kvpress_env/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Exact run commands

Create two temporary YAML configs under `.debug/`:

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

Then run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_no_press.yaml
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_math500_expected_attention.yaml
```

## Why this is the right cheap baseline

- It is a **same-model proxy**, so it is fair enough to compare as a compression/control baseline.
- `kvpress` explicitly supports `ExpectedAttentionPress`, `no_press`, `Qwen3ForCausalLM`, and `math500`.
- It gives a direct read on whether the current Qwen-side story is really above a query-aware null.

## Blockers

- This is **not** cross-model communication. It should be framed as a compression/null comparator.
- On Apple Silicon, `attn_implementation: eager` is the safe choice; the run will be slower than CUDA.
- If `Qwen/Qwen3-0.6B` is not available in the local HF cache, switch to another supported Qwen3 checkpoint already available locally.
- `query_aware: true` includes the question in the compression context; that is the fairest mode for a question-answer benchmark.
- If you want throughput or longer-context benchmarks, `KVzip` and `Quest` are better but more expensive and less directly aligned to the current cheap comparator goal.

## Why not the other candidates first

- `KVzip`: stronger compression baseline, but still same-model and more setup than `kvpress`.
- `KVComm`: direct communication baseline, but not a cheap same-model proxy and not native to the current heterogeneous pair.
- `R-KV`: useful for reasoning compression, but model/setup cost is higher.
- `tokenkit` / `DSKD`: training-heavy tokenizer-transfer tools, better treated as future pivots than immediate local baselines.

