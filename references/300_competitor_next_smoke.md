# Competitor Next Smoke After `math500` Answered=0/5

The `kvpress` math500 smoke was not a useful signal because the scorer expects
boxed math answers and Qwen3 often answered in a free-form style. The next
smoke should use a scorer/prompt where Qwen3 outputs are recognized by
substring or retrieval match instead of boxed formatting.

## Priority 1: KVPress on string-match / retrieval tasks

### A. `needle_in_haystack` first

Why:
- Cheapest exact-match smoke.
- The scorer checks whether the inserted needle appears in the answer, so
  Qwen3 free-form outputs are much more likely to be recognized than in
  `math500`.

Local run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress
pip install -r requirements.txt
pip install -e .
```

Create configs:

```bash
cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_no_press.yaml <<'YAML'
dataset: needle_in_haystack
model: Qwen/Qwen3-0.6B
press_name: no_press
compression_ratio: 1.0
query_aware: true
fraction: 0.1
max_context_length: 10000
needle_depth: 50
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML

cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_expected_attention.yaml <<'YAML'
dataset: needle_in_haystack
model: Qwen/Qwen3-0.6B
press_name: expected_attention
compression_ratio: 0.5
query_aware: true
fraction: 0.1
max_context_length: 10000
needle_depth: 50
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML
```

Run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_no_press.yaml
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_expected_attention.yaml
```

Fair comparison to LatentWire:
- Treat `no_press` as the floor.
- Treat `expected_attention` as the query-aware compression comparator.
- This is a same-model proxy, not a cross-model communication result.

Blockers on Mac/MPS:
- Use `attn_implementation: eager`.
- `device: mps` may work if the torch build supports it; otherwise CPU is the
  safe fallback.
- Still slower than CUDA, but it is the cheapest runnable smoke that avoids the
  boxed-answer issue from `math500`.

### B. `ruler` second

Why:
- Benchmark-y string-match scorer with more variety than `needle_in_haystack`.
- Still recognizes free-form Qwen3 outputs better than boxed math scoring.

Local run:

```bash
cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_no_press.yaml <<'YAML'
dataset: ruler
data_dir: "4096"
model: Qwen/Qwen3-0.6B
press_name: no_press
compression_ratio: 1.0
query_aware: true
fraction: 0.1
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML

cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_expected_attention.yaml <<'YAML'
dataset: ruler
data_dir: "4096"
model: Qwen/Qwen3-0.6B
press_name: expected_attention
compression_ratio: 0.5
query_aware: true
fraction: 0.1
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML
```

Run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_no_press.yaml
python evaluate.py --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_expected_attention.yaml
```

Fair comparison to LatentWire:
- Same logic as `needle_in_haystack`: null floor vs query-aware compression.
- Better for a benchmark table because it is less synthetic than a single
  needle-retrieval smoke.

Custom harness glue:
- None needed for either KVPress smoke.

## Priority 2: heavier local comparators

### KVzip

Best use:
- Stronger same-model compression comparator if we have a CUDA box.

Local command from repo docs:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
```

Blockers on this Mac/MPS machine:
- CUDA / flash-attn assumptions.
- Large model / GPU-centric setup.

Fair comparison to LatentWire:
- Same-model compression ceiling only.
- Good as a compression comparator, not a cross-model communication claim.

Custom harness glue:
- Yes, if we want to place it side-by-side with the current LatentWire GSM
  readout.

### Quest

Best use:
- Query-aware sparsity comparator for long-context retrieval.

Local command from repo docs:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
```

Blockers on this Mac/MPS machine:
- CUDA / FlashInfer / kernel build overhead.
- README targets LongChat/Yarn-Llama2-style setups, not the current Qwen pair.

Fair comparison to LatentWire:
- Query-aware sparsity baseline, not a direct cross-model communication
  result.

Custom harness glue:
- Yes, for a fair comparison table.

### KVComm

Best use:
- Direct communication baseline.

Local command from repo docs:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
pip install -r requirements.txt
python com.py --test_task hotpotqa --do_test_baseline --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct
python com.py --test_task hotpotqa --do_test --model_A meta-llama/Llama-3.1-8B-Instruct --model_B meta-llama/Llama-3.1-8B-Instruct --top_layers 0.3
```

Blockers on this Mac/MPS machine:
- Not a cheap local smoke.
- CUDA-heavy in practice and uses large Llama-family models in the README.

Fair comparison to LatentWire:
- Use as the direct communication bar.
- For the current heterogeneous Qwen pair, the LatentWire replay remains the
  fairer apples-to-apples comparison.

Custom harness glue:
- Yes, for any side-by-side comparison to the current Qwen pipeline.

### C2C

Best use:
- Strongest direct cache-to-cache communication ceiling.

Local command from repo docs:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
pip install -e .
pip install -e ".[training,evaluation]"
python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml
```

Blockers on this Mac/MPS machine:
- Big-model / GPU-heavy.
- More setup than the KVPress smoke.

Fair comparison to LatentWire:
- External quality ceiling.
- Do not claim byte parity unless you normalize payloads.

Custom harness glue:
- Yes, if we want a fair paired table.

### tokenkit / ALM

Best use:
- Future tokenizer-side bridge if transport stays saturated.

Local command from repo docs:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/tokenkit
pip install -r requirements.txt
pip install -e .
bash scripts/cross_tokenizer_distill.py
```

Blockers on this Mac/MPS machine:
- Training-heavy JAX stack.
- Python <= 3.10 guidance in the README.

Fair comparison to LatentWire:
- Not an immediate inference baseline.
- Better treated as a future pivot than a near-term competitor smoke.

Custom harness glue:
- Yes. It is not a drop-in comparator for the current LatentWire eval loop.

## Recommendation

If we only run one more cheap smoke now, run `kvpress` in this order:

1. `needle_in_haystack`
2. `ruler`

Then, if we have CUDA resources, move to `KVzip`, `Quest`, `KVComm`, and
`C2C`. Treat `tokenkit` / ALM as a future tokenizer-side pivot, not the next
cheap benchmark.
