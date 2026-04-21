# Competitor Bootstrap Triage

Scope: inspect candidate codebases for cross-model communication or KV-cache
compression comparisons. No LatentWire source changes were made.

Cloned for later inspection:
- `references/repos/tokenkit` (`bcacdef`) — https://github.com/bminixhofer/tokenkit
- `references/repos/r-kv` (`c8169f5f`) — https://github.com/zefan-cai/r-kv
- `references/repos/DSKD` (`9962b49`) — https://github.com/songmzhang/DSKD

## What is runnable locally now

| Repo | Paper / date | Local entry point | Fair baseline value | Main blocker |
|---|---|---|---|---|
| `kvpress` | *Expected Attention* / 2025-10-02 (arXiv 2510.00636) | `pip install kvpress`; `kvpress` pipeline or `evaluation/` CLI | Best low-friction **compression control**. Supports `ExpectedAttentionPress`, `QFilterPress`, `DuoAttentionPress`, `KVzipPress`, `CAMPress`, etc. and supports Qwen2/3, Gemma3, Llama/Mistral in README. | It is **single-model compression**, not cross-model communication. It is a fair comparator for nulls / query-aware compression, not for source-to-target transfer. |
| `KVzip` | *KVzip* / 2025-05-29 (arXiv 2505.23416) | `pip install -r requirements.txt`; `python -B test.py ...`; `python -B eval.py ...` | Strong **query-agnostic compression** comparator. Supports Qwen2.5/3, LLaMA3, Gemma3; precomputed head scores exist for some models. | Same-model KV compression, not cross-model comms. Best as a compression bar for our transport story. |
| `DeltaKV_sparse_vllm` | *DeltaKV* / 2026-02-08 (arXiv 2602.08005) | `pip install -e .`; `scripts/bench_sparse_vllm.py`; `benchmark/math_bench/pred.py`; `benchmark/long_bench/pred.py` | Useful **systems-side sparse inference** harness; can benchmark Qwen-family backbones and sparse-vLLM methods. | DeltaKV-specific compressor checkpoints are “about to be uploaded” in README, so the compressor branch itself may be incomplete. Good for systems baselines, not a direct communication comparator. |
| `Quest` | *Quest* / 2024-06-16 (arXiv 2406.10774) | `bash scripts/passkey.sh`, `bash scripts/longbench.sh`, kernel tests in `quest/tests` | Strong **query-aware sparsity** comparator. Good for the “query matters” side of the story. | README targets Llama/Mistral family and needs CUDA + FlashInfer/NVBench tooling. Not a drop-in fair comparator for the current Qwen pair without porting. |
| `R-KV` | *R-KV* / 2025-05-30 (arXiv 2505.24133) | `pip install -r requirements.txt`; `bash scripts/run.sh`; `python3 run_math.py ...` | Strong **reasoning-oriented KV compression** comparator for math / CoT-style decoding. | Uses large reasoning checkpoints (e.g. DeepSeek-R1-Distill-Llama-8B, Qwen14B). Fair on reasoning tasks, but not a drop-in comparator for the current small heterogeneous Qwen pair. |
| `KVComm` | *KVComm* / 2025-10-02 (arXiv 2510.03346, ICLR 2026) | `pip install -r requirements.txt`; `python com.py ...` | Best **direct inter-model communication** comparator. It is the closest paper-side baseline to our task framing. | Upstream repo is native to same-architecture / same-family communication. The current heterogeneous Qwen2.5-0.5B → Qwen3-0.6B setting is not apples-to-apples without adaptation; the LatentWire replay is therefore the fairer hetero-pair view. |

## Not fair as immediate baselines, but useful future pivots

| Repo | Paper / date | Why it is interesting | Why it is blocked right now |
|---|---|---|---|
| `tokenkit` | *Cross-Tokenizer Distillation via Approximate Likelihood Matching* / 2025-03-25; tokenkit initial release 2025-04-02 | Best tokenization-side bridge candidate. Supports ALM, ZeTT/FVT, token-level ensembles, and explicit tokenizer transfer. | Training-heavy, JAX-first, Python <=3.10, and aimed at tokenizer transfer/distillation, not inference-only benchmarking. Good future pivot, not current baseline. |
| `DSKD` | *Dual-Space Knowledge Distillation for Large Language Models* / 2024-06-25 | Useful tokenizer-mismatch KD ancestor and a concrete teacher/student transfer codebase. | Training-heavy deepspeed KD, not a local inference benchmark. Good for future tokenizer-side experiments, not for the current fair comparison table. |

## Recommendation

For the current LatentWire paper, the strongest local competitor stack is:

1. `kvpress` for null / compression controls and query-aware selector baselines.
2. `KVzip` for query-agnostic compression and head-budget comparators.
3. `KVComm` for the direct cross-model communication bar.
4. `R-KV` only if we want a reasoning-specific compression comparator on the right model family.

I would **not** treat `tokenkit` or `DSKD` as immediate baselines in the paper table. They are better treated as **future tokenizer-side pivots** if the transport lane fully saturates.

## Concrete blocker summary

- `kvpress`: easy to run, but only same-model compression.
- `KVzip`: easy-ish to run, but still same-model compression.
- `DeltaKV_sparse_vllm`: runnable systems harness, but the DeltaKV compressor path may still be incomplete.
- `Quest`: strong query-aware sparsity baseline, but model-family mismatch and kernel/tooling overhead.
- `R-KV`: strong reasoning compression baseline, but large-model / CUDA-heavy.
- `KVComm`: direct communication comparator, but heterogeneous Qwen pair is not a native setting.
- `tokenkit` / `DSKD`: useful future tokenizer-transfer bridges, but training-heavy and not immediate baselines.
