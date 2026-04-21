# Competitor Benchmark Bootstrap After KVPress Math500 Answered=0/5

Date: 2026-04-20

This note is execution-focused. The prior KVPress Math500 run produced
`answered=0/5`, so the next competitor benchmark should avoid boxed-answer
math scoring and first verify that the competitor harness can produce
recognized answers on a string-match retrieval task.

## Primary Sources Consulted

- KVPress repository: https://github.com/NVIDIA/kvpress
- KVPress / Expected Attention paper: https://arxiv.org/abs/2510.00636
- KVzip repository: https://github.com/snu-mllab/KVzip
- KVzip paper: https://arxiv.org/abs/2505.23416
- Quest repository: https://github.com/mit-han-lab/Quest
- Quest paper: https://arxiv.org/abs/2406.10774
- SnapKV paper: https://arxiv.org/abs/2404.14469
- H2O repository: https://github.com/FMInference/H2O
- H2O paper: https://arxiv.org/abs/2306.14048
- KVComm repository: https://github.com/FastMAS/KVCOMM
- KVComm paper: https://arxiv.org/abs/2510.12872
- C2C repository: https://github.com/thu-nics/C2C
- C2C project page: https://fuvty.github.io/C2C_Project_Page/

## Local Competitor Inventory

Already cloned or wrapped locally:

- `references/repos/kvpress`
- `references/repos/KVzip`
- `references/repos/Quest`
- `references/repos/KVComm`
- `references/repos/C2C`
- `scripts/run_kvpress_eval.py`
- `scripts/run_kvcomm_eval.py`
- `scripts/run_c2c_eval.py`

The local wrappers matter because they normalize predictions into our JSONL
generation metrics where possible. For the immediate smoke, prefer the wrapper
over upstream evaluation CLI because it is already patched for the current
Transformers cache API and Qwen3 path.

## Recommended First Smoke To Run Now

Run KVPress on `data/gsm8k_5.jsonl` through the local wrapper, but treat this
as a harness sanity check, not the final competitor benchmark. It avoids the
upstream Math500 boxed-answer scorer that gave `answered=0/5` and uses the same
LatentWire generation scorer as our current results.

```bash
mkdir -p results/kvpress_wrapper_smoke_20260420

PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress:/Users/sujeethjinesh/Desktop/LatentWire \
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --prediction-output results/kvpress_wrapper_smoke_20260420/qwen3_gsm5_no_press.jsonl

PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress:/Users/sujeethjinesh/Desktop/LatentWire \
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --prediction-output results/kvpress_wrapper_smoke_20260420/qwen3_gsm5_expected_attention_cr050.jsonl
```

Interpretation:

- If `no_press` still cannot produce recognized answers, the issue is
  model/prompt/scoring and not compression.
- If `no_press` works but `expected_attention` drops, we have a valid same-model
  compression baseline to compare against target-alone and LatentWire.
- If both work and tie, move to retrieval benchmarks because GSM5 is too small
  and too noisy for KV compression.

Mac/MPS blockers:

- Use `--dtype float32`; Qwen3 plus MPS can be brittle under fp16/bf16.
- The local wrapper only exposes `none` and `expected_attention`; SnapKV/H2O
  are available in upstream KVPress but not exposed by this wrapper.
- This smoke is same-model compression, not cross-model communication.

## Executed KVPress Smokes on 2026-04-20

Local LatentWire wrapper on the controlled GSM10 slice:

| Press | Accuracy | Tokens/sec | Examples/sec | Latency sec | Generated tokens avg |
|---|---:|---:|---:|---:|---:|
| `none` | 0.1000 | 7.1584 | 0.1742 | 5.7415 | 41.1 |
| `expected_attention`, compression `0.5` | 0.1000 | 6.6484 | 0.1606 | 6.2270 | 41.4 |

Local LatentWire wrapper on GSM5:

| Press | Accuracy | Tokens/sec | Examples/sec | Latency sec | Generated tokens avg |
|---|---:|---:|---:|---:|---:|
| `none` | 0.2000 | 6.7396 | 0.1764 | 5.6680 | 38.2 |
| `expected_attention`, compression `0.5` | 0.2000 | 6.1384 | 0.1582 | 6.3210 | 38.8 |

Native KVPress `needle_in_haystack`, Qwen3-0.6B, `max_context_length=4096`,
`needle_depth=50`, single available row:

| Press | Compression | Rouge-1 F | Rouge-2 F | Rouge-L F |
|---|---:|---:|---:|---:|
| `no_press` | 1.0 | 0.7500 | 0.6667 | 0.7500 |
| `expected_attention` | 0.5 | 0.7500 | 0.6667 | 0.7500 |

Interpretation:

- The external harness is runnable locally after switching the single-row
  `needle_in_haystack` smoke to `fraction: 1.0`.
- On these tiny sanity checks, ExpectedAttention ties no-press and is slower on
  GSM, so it is a valid boundary comparator but not a stronger local bar yet.
- The next competitor run should be a larger retrieval/RULER slice on a Linux
  or CUDA host, not more one-row Needle smoke on this laptop.

## Next KVPress Benchmark: Needle/RULER

Use upstream KVPress evaluation next because it includes `needle_in_haystack`
and `ruler`. These are better after Math500 failed because the scorer is
string/retrieval oriented.

Setup:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python -m pip install -e ".[eval]"
```

Needle configs:

```bash
mkdir -p /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih

cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_no_press.yaml <<'YAML'
dataset: needle_in_haystack
model: Qwen/Qwen3-0.6B
press_name: no_press
compression_ratio: 1.0
query_aware: true
fraction: 1.0
max_context_length: 4096
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
fraction: 1.0
max_context_length: 4096
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
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_no_press.yaml
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_nih_expected_attention.yaml
```

RULER configs:

```bash
mkdir -p /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler

cat > /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_no_press.yaml <<'YAML'
dataset: ruler
data_dir: "4096"
model: Qwen/Qwen3-0.6B
press_name: no_press
compression_ratio: 1.0
query_aware: true
fraction: 0.05
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
fraction: 0.05
output_dir: /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler
log_level: INFO
model_kwargs:
  attn_implementation: eager
YAML
```

Run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress/evaluation
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_no_press.yaml
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress \
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python evaluate.py \
  --config_file /Users/sujeethjinesh/Desktop/LatentWire/.debug/kvpress_ruler_expected_attention.yaml
```

Expected blockers:

- Upstream eval extras may pull CUDA-oriented optional packages if installed
  carelessly; avoid `flash-attn` on macOS.
- `RULER` data generation/download may be slower or fail if upstream paths
  assume prebuilt data.
- If upstream evaluation fails, keep the local wrapper smoke as the committed
  same-model compression comparator and defer RULER to CUDA/Linux.

## KVPress SnapKV / H2O-Style Baselines

KVPress primary source lists `SnapKVPress`, `StreamingLLMPress`, `TOVAPress`,
`ObservedAttentionPress`, `PyramidKVPress`, `DuoAttentionPress`, `KVzipPress`,
`KVzapPress`, and other scorer/composition presses. This is the fastest route
to SnapKV/H2O-style comparisons without maintaining separate forks.

Recommended upstream config variants after Needle/RULER works:

```yaml
press_name: snapkv
compression_ratio: 0.5
```

```yaml
press_name: streaming_llm
compression_ratio: 0.5
```

```yaml
press_name: observed_attention
compression_ratio: 0.5
```

Expected blockers:

- Press names must match KVPress evaluation registry exactly; inspect
  `references/repos/kvpress/evaluation/evaluate_registry.py` before running.
- SnapKV-style methods may require returned attentions or architecture-specific
  hooks; use `attn_implementation: eager` on macOS.
- These are still same-model compression baselines. They do not transfer a
  source-model cache into a target-model cache.

## KVzip

KVzip is the strongest query-agnostic compression comparator in this set, but
the public code path is CUDA-oriented.

Setup and smoke from upstream README:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python -m pip install -r requirements.txt
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
python -B test.py -m qwen2.5-7b -d squad --kv_type evict --ratio 0.3
```

Context-independent run:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B test.py -m qwen2.5-7b -d scbench_qa_eng --save_head_score
python -B eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100
python -B -m results.parse -m qwen2.5-7b -d squad
```

Expected blockers on this Mac/MPS machine:

- `flash-attn` and custom kernels are CUDA-only in practice.
- README examples target Qwen2.5-7B/Qwen3-class long-context models, not the
  small Qwen3-0.6B local smoke model.
- Use this on CUDA/Linux first; do not burn time trying to make it native MPS.

Fair comparison:

- Same-model compression ceiling.
- Not direct cross-model communication.

## Quest

Quest is the strongest query-aware sparsity comparator, but it is also
CUDA/kernel heavy.

Setup and passkey/LongBench commands from upstream README:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
conda create -yn quest python=3.10
conda activate quest
pip install -e .
pip install ninja packaging
pip install flash-attn==2.6.3 --no-build-isolation
conda install cmake
cd kernels/3rdparty/raft
./build.sh libraft
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/quest/ops
bash setup.sh
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
bash scripts/longbench.sh
```

Expected blockers on this Mac/MPS machine:

- CUDA, FlashAttention, FlashInfer, RAFT, and PyBind operator setup.
- Upstream examples focus on LongChat/Yarn-Llama2/Llama3.1/Mistral-style
  long-context models, not our small heterogeneous Qwen pair.

Fair comparison:

- Query-aware same-model long-context sparsity.
- Use as a long-context speed/quality comparator, not a cross-model bridge.

## H2O Direct Repo

H2O is older but still useful as the canonical heavy-hitter eviction baseline.
Prefer KVPress `streaming_llm`, `observed_attention`, or related scorer presses
for quick local comparison. Use the direct repo only if we need historical
faithfulness to H2O.

Likely direct-repo path:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/H2O
```

Expected blockers:

- The local inventory currently does not show `references/repos/H2O`; clone it
  only if direct H2O reproduction becomes necessary:

```bash
mkdir -p /Users/sujeethjinesh/Desktop/LatentWire/references/repos
git clone https://github.com/FMInference/H2O.git \
  /Users/sujeethjinesh/Desktop/LatentWire/references/repos/H2O
```

- Direct H2O implementations are dated relative to current Transformers cache
  APIs.
- CUDA/GPU assumptions are likely.

Fair comparison:

- Baseline eviction policy for same-model generation.
- Not direct cross-model communication.

## KVComm

KVComm is a closer conceptual competitor because it communicates KV cache
information across contexts/agents. The local repo is cloned and LatentWire has
a wrapper with Qwen3 compatibility patching.

Local LatentWire wrapper smoke:

```bash
mkdir -p results/kvcomm_smoke_20260420

./venv_arm64/bin/python scripts/run_kvcomm_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file data/gsm8k_5.jsonl \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25,0.5 \
  --prediction-output results/kvcomm_smoke_20260420/qwen25_to_qwen3_gsm5.jsonl
```

Upstream README baseline/direct commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
pip install -r requirements.txt
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

Expected blockers on this Mac/MPS machine:

- The wrapper loads both source and target models simultaneously. Qwen
  0.5B/0.6B may fit on MPS, but CPU fallback may be needed if MPS memory
  fragments.
- Upstream scripts default to CUDA-oriented large Llama models.
- Layer-ranking calibration is extra work before held-out eval; keep the grid
  small for smoke.

Fair comparison:

- More relevant than same-model compression because it is communication-like.
- Still not identical to LatentWire if it expects same-family architectures or
  context reuse rather than heterogeneous source-to-target reasoning transfer.

## C2C

C2C is the most direct public cache-to-cache communication comparator, and the
local wrapper can use published artifacts when a supported pair is available.

Local LatentWire wrapper smoke:

```bash
mkdir -p results/c2c_smoke_20260420

./venv_arm64/bin/python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --limit 5 \
  --prediction-output results/c2c_smoke_20260420/qwen25_to_qwen3_gsm5.jsonl
```

Upstream setup/eval commands:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
conda create -n rosetta python=3.10
conda activate rosetta
pip install -e .
pip install -e ".[training,evaluation]"
python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml
```

Expected blockers on this Mac/MPS machine:

- Published C2C artifacts may not exist for the exact Qwen2.5-0.5B to Qwen3
  0.6B pair. If artifact resolution fails, this is a pair-support blocker, not
  a LatentWire failure.
- Upstream examples use CUDA devices and larger model pairs.
- The local wrapper downloads Hugging Face artifacts; first run may be network
  and disk heavy.

Fair comparison:

- Best conceptual competitor if the pair is supported.
- If not supported for our pair, report it as external method unavailable for
  the current heterogeneous Qwen pair and use upstream paper numbers only as
  context, not as a direct table row.

## Recommended Ordering

1. Run local KVPress wrapper `none` vs `expected_attention` on `gsm8k_5`.
2. Run upstream KVPress `needle_in_haystack` `no_press` vs
   `expected_attention` at 4k context and `fraction: 0.05`.
3. If Needle works, run RULER 4k with the same two presses.
4. Add KVPress SnapKV/StreamingLLM/ObservedAttention variants only after the
   `no_press` floor is nonzero on Needle/RULER.
5. Run local KVComm wrapper on GSM5 as the first communication-like competitor.
6. Try local C2C wrapper; treat missing Qwen pair artifacts as expected.
7. Defer KVzip and Quest to CUDA/Linux unless we specifically want to spend
   setup time on competitor reproduction.

## Reporting Schema For Interpretable Results

Every competitor run should emit or be converted into:

- `method`: exact competitor and compression/communication setting.
- `model_pair`: same-model or source-target pair.
- `benchmark`: GSM5, Needle, RULER, HotpotQA, etc.
- `n_examples`: evaluated sample count.
- `accuracy` or benchmark-native score.
- `answered_rate` when benchmark scoring can reject format.
- `compression_ratio` or retained cache fraction.
- `device`, `dtype`, `attn_implementation`.
- `bytes_or_cache_proxy`: if available, cache tokens retained or KV bytes.
- `blocker`: explicit reason if a method cannot run locally.

The key interpretability rule: keep same-model compression baselines separate
from cross-model communication baselines. They answer different questions and
should not be collapsed into one win/loss table.
