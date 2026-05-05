# 00 — Shared Setup (Macbook)

This applies to all three projects. Do this in week 1, day 1.

## Hardware assumed
- Macbook (Apple Silicon M-series)
- No CUDA. PyTorch MPS for some ops. CPU fallback always.
- All 5090 work is later — see project docs.

## Repository layout
Use this single LatentWire checkout. Each project owns a subfolder under
`experimental/` and keeps local environments, scratch clones, and generated
artifacts inside ignored project-local directories.

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
mkdir -p experimental/hybridkernel experimental/sinkaware experimental/thoughtflow_fp8
```

Shared HuggingFace cache:

```bash
mkdir -p .debug/hf_cache
export HF_HOME="$PWD/.debug/hf_cache"
```

## Python envs
Use repo-local virtual environments. The main LatentWire work should continue
to prefer `./venv_arm64`; these side experiments each get their own ignored
project-local `.venv`.

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
python3 -m venv experimental/hybridkernel/.venv
python3 -m venv experimental/sinkaware/.venv
python3 -m venv experimental/thoughtflow_fp8/.venv
```

Install per-project requirements only after a project reaches its setup gate:

```bash
source experimental/hybridkernel/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experimental/hybridkernel/requirements.txt
```

## Repos to clone (Phase 0)

### Universal — for all three projects
```bash
mkdir -p experimental/shared_external && cd experimental/shared_external
git clone --depth 1 https://github.com/vllm-project/vllm
git clone --depth 1 https://github.com/triton-lang/triton
git clone --depth 1 https://github.com/Dao-AILab/flash-attention
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer
git clone --depth 1 https://github.com/huggingface/transformers
```

### HybridKernel-specific
```bash
git clone --depth 1 https://github.com/state-spaces/mamba
# Mamba-3 release (March 2026, CMU/Princeton/Cartesia/Together)
git clone --depth 1 https://github.com/sustcsonglin/flash-linear-attention
git clone --depth 1 https://github.com/foundation-model-stack/bamba  # Bamba v2 (May 2025)
# Granite-4.0 reference: HuggingFace ibm-granite/granite-4.0-h-tiny / h-small
# Nemotron-H reference: HuggingFace nvidia/Nemotron-H-8B-Base
# Nemotron-3-Nano: HuggingFace nvidia/Nemotron-3-Nano-30B-A3B (FP8 + BF16 variants)
# Apriel-H1-15B-Thinker: HuggingFace ServiceNow-AI/Apriel-H1-15B-Thinker
```

### SinkAware-specific
```bash
# BLASST — find URL from arxiv 2512.xxxxx (Dec 2025)
# Block-Sparse Flash Attention — same
git clone --depth 1 https://github.com/mit-han-lab/streaming-llm
# EARN — check arxiv 2507.xxxxx
# OrthoRank — check arxiv 2507.xxxxx (July 2025)
# SinkTrack — check release
# DeepSeek V3.2-Exp DSA kernels (TileLang + CUDA): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
# DeepSeek V4 attention (c4a/c128a + DSA): vLLM blog post for impl reference
# FlashMLA: DeepSeek's MLA kernel (used by V3.2/V4 DSA)
# NSA (Native Sparse Attention) reference: arxiv 2502.11089
```

### ThoughtFlow-specific
```bash
# LongFlow — check OpenReview ICLR 2026 (withdrawn submission); may need to scrape arxiv
# ThinKV — check arxiv
# R-KV, RaaS, LazyEviction, ForesightKV, PM-KVQ — collect arxiv links
# "Pitfalls of KV Cache Compression" — ICLR 2026 withdrawal; find arxiv
# Open-R1 — for reasoning traces
git clone --depth 1 https://github.com/huggingface/open-r1

# Current reasoning models (May 2026):
# GPT-OSS-20B: openai/gpt-oss-20b (16GB MXFP4, configurable reasoning effort)
# Qwen3.6-27B: Qwen/Qwen3.6-27B (April 2026, hybrid thinking/non-thinking modes)
# Apriel-H1-15B-Thinker: ServiceNow-AI/Apriel-H1-15B-Thinker (hybrid reasoner, vLLM-deployable)
# Nemotron-3-Nano-30B-A3B-FP8: nvidia/Nemotron-3-Nano-30B-A3B (reasoning budget control)
# DeepSeek-V4-Flash: deepseek-ai/DeepSeek-V4-Flash (284B, won't fit 5090; reference only)

# DEPRECATION ALERT: DeepSeek R1 line retires July 24, 2026.
# Do NOT anchor experiments on R1 or its Qwen-distills as the headline target.
# The R1-distill traces in Open-R1 remain useful for offline analysis.
```

## Cached datasets / traces
Download once, share across all projects via `HF_HOME`.

```bash
huggingface-cli download HuggingFaceH4/aime_2024 --repo-type dataset
huggingface-cli download HuggingFaceH4/aime_2025 --repo-type dataset  # newer
huggingface-cli download Idavidrein/gpqa --repo-type dataset
huggingface-cli download AI-MO/aimo-validation-aime --repo-type dataset
# Open-R1 reasoning traces (sample, not full)
huggingface-cli download open-r1/OpenR1-Math-220k --repo-type dataset
# For ThoughtFlow: also generate fresh traces from GPT-OSS-20B / Qwen3.6 in Phase 2
# More as needed per project
```

## Tooling notes for Mac

### What runs natively
- PyTorch CPU (everything correct, slow)
- PyTorch MPS (some ops; `torch.nn.functional.scaled_dot_product_attention` works)
- Triton CPU backend (compiles kernels, can inspect IR; **does not give GPU perf numbers**)
- HuggingFace transformers with `device_map="mps"` for ≤7B models
- NumPy / pure-Python reference impls

### What does NOT run on Mac (defer to 5090)
- vLLM (CUDA-only)
- FlashAttention (CUDA)
- FlashInfer (CUDA)
- mamba-ssm CUDA selective scan (use CPU fallback for reference only)
- Real Triton kernel benchmarks
- Anything FP8 or FP4 native (Mac has no Tensor Cores)

### Agent-friendly conventions
- Each agent runs in its own tmux window or VS Code window
- Each project folder has its own README, requirements, progress log, and tests
- All deliverables stay under `experimental/<project>/` unless a shared asset is explicitly needed
- Use project folders in this checkout; avoid extra branches unless the user asks

## Gate review process
At the end of each Macbook phase, the agent posts to the project folder's
`progress.md`:
- Which deliverables exist (with paths)
- Which kill criteria were checked
- Status: PASS / KILL / PIVOT (with rationale)

Final gate review before any GPU spend: human-in-the-loop check that all three project docs' Phase 0–4 deliverables exist and pass criteria.

## What "Phase complete" means
A phase is complete only when:
1. All checkboxes in that phase are ticked
2. All deliverable files exist in the project folder at the specified paths
3. `progress.md` is updated
4. Tests (where applicable) pass

Do not mark a phase complete based on "I did most of it." Either it's done or it isn't.
