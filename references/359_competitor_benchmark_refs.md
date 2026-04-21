# Competitor Benchmark References for LatentWire

Web check: 2026-04-21. This memo is a scouting list for the closest competitors and orthogonal controls around efficient cross-model reasoning/communication, KV compression, prompt/context compression, token pruning, cache transfer, and adapter/feature-dictionary baselines.

The goal is to keep the comparison set small enough to run, but broad enough that a reviewer cannot dismiss the result as a narrow cache trick or a prompt-format artifact.

## Read This First

- `Clone/runnable here` means the repo is already present under `references/repos/` or is available as a stable upstream repo/package.
- `Likely eval tasks` are the tasks that are the best fit for the method, not an exhaustive list of everything the paper reports.
- `Bootstrap plan` is the first fair experiment to run before widening the sweep.

## Direct Semantic Communication / Cache Transfer

| Family | Primary sources | Clone / runnable here | Likely eval tasks | Bootstrap plan |
|---|---|---|---|---|
| **C2C** | [paper](https://arxiv.org/abs/2510.03215), [repo](https://github.com/thu-nics/C2C) | **Yes.** Local clone present; repo README exposes training and eval entrypoints. | GSM8K, SVAMP, MMLU, HumanEval, plus any held-out pairwise communication split. | Run the published fuser on one fixed source/target pair, then compare against LatentWire at matched prompt bytes, matched repair budget, and matched decode budget. |
| **KVCOMM** | [paper](https://arxiv.org/abs/2510.03346), [repo](https://github.com/HankYe/KVCOMM) and current pointer in the README to [FastMAS/KVCOMM](https://github.com/FastMAS/KVCOMM) | **Yes.** Local clone present; repo README gives GSM8K, MMLU, and HumanEval scripts. | GSM8K, MMLU, HumanEval, multi-agent reuse traces. | Freeze agent count, model, and entropy threshold, then compare `default` vs `allow_kv_reuse` on the same split before testing LatentWire. |
| **aLoRA / activated adapters** | [PEFT docs](https://huggingface.co/docs/peft/main/developer_guides/lora), [paper](https://huggingface.co/papers/2504.12397), [IBM repo](https://github.com/IBM/activated-lora) | **Not cloned here, but runnable via PEFT.** The PEFT docs are the cleanest current implementation reference. | Multi-turn switching, correction / verification chains, agentic flows where the base model dominates most tokens. | Compare base-cache reuse against standard LoRA and aLoRA on a switching workload; count recomputed prefix tokens as the main cost. |

## KV Compression, Retrieval, and Token Pruning

| Family | Primary sources | Clone / runnable here | Likely eval tasks | Bootstrap plan |
|---|---|---|---|---|
| **KVPress** | [repo](https://github.com/NVIDIA/kvpress), [Expected Attention paper](https://arxiv.org/abs/2510.00636) | **Yes.** Local clone present; package and evaluation CLI are available. | LongBench, RULER, Needle-in-a-Haystack, passkey, retrieval QA. | Use `none`, `ExpectedAttention`, `SnapKV`, `QFilter`, and `AdaKV`-style presses at one fixed compression ratio and one fixed downstream model. |
| **Quest** | [paper](https://arxiv.org/abs/2406.10774), [repo](https://github.com/mit-han-lab/Quest) | **Yes.** Local clone present; README shows Passkey and LongBench scripts. | LongBench, passkey retrieval, PG19. | Hold the model and context length fixed, then sweep the top-k page budget and report accuracy, latency, and loaded-page count. |
| **KVzip** | [paper](https://arxiv.org/abs/2505.23416), [project](https://janghyun1230.github.io/kvzip/), [repo](https://github.com/snu-mllab/KVzip) | **Yes.** Local clone present; eval/test entrypoints are visible locally. | Question answering, retrieval, reasoning, code comprehension. | Compare query-agnostic compression against query-aware methods at the same retained-byte budget and record reuse stability on held-out prompts. |
| **H2O** | [paper](https://arxiv.org/abs/2306.14048), [repo](https://github.com/FMInference/H2O) | **Yes.** Local clone present; upstream repo has runnable code. | Long-context generation, retrieval, reasoning, passkey/needle-style probes. | Test heavy-hitter plus recency retention against a uniform retention baseline and log overlap with the retained-token set. |
| **SnapKV** | [paper](https://arxiv.org/abs/2404.14469), [OpenReview](https://openreview.net/forum?id=poE54GOq2l) | **Yes.** Local clone present; repo-native scripts are already mirrored locally. | LongBench, Needle-in-a-Haystack, passkey, long-sequence QA. | Use a fixed observation window and compare it to uniform retention and query-aware selection under the same KV budget. |
| **Ada-KV** | [paper](https://arxiv.org/abs/2407.11550), [repo](https://github.com/FFY0/AdaKV) and KVPress integration notes in the repo README | **Yes.** Local clone present; can be treated as a head-wise budget allocator on top of other press methods. | RULER, LongBench, and any benchmark where head importance is uneven. | Apply Ada-KV on top of SnapKV or ExpectedAttention and measure whether head-wise budget reallocation beats the same base press at identical bytes. |
| **FastKV** | [paper](https://arxiv.org/abs/2502.01068), [repo](https://github.com/dongwonjo/FastKV) | **Not cloned here, but official repo exists and is runnable.** | LongBench, Needle, prefill/decode latency-sensitive workloads. | Separate the token-selective propagation rate from the KV retention rate, then compare against a full-context baseline and a decode-only compression baseline. |
| **MInference** | [project / repo](https://github.com/microsoft/minference), [Microsoft project page](https://www.microsoft.com/en-us/research/project/minference-million-tokens-prompt-inference-for-long-context-llms/downloads/) | **Not cloned here, but official repo exists and is runnable.** | SCBench, LongBench, RULER, 1M-token prefill stress tests. | Use it as the sparse-attention prefill ceiling: fixed model, fixed prompt, fixed kernel settings, then measure whether LatentWire still wins once prefill speed is normalized. |
| **LLMLingua / LongLLMLingua / LLMLingua-2** | [repo](https://github.com/microsoft/LLMLingua), [project](https://www.llmlingua.com/), [LLMLingua paper](https://arxiv.org/abs/2310.05736), [LongLLMLingua paper](https://arxiv.org/abs/2310.06839), [LLMLingua-2](https://arxiv.org/abs/2403.12968) | **Yes.** Local clone present; the Microsoft project page is the clean source of truth for the series. | GSM8K, BBH, ShareGPT, Arxiv-March23, NaturalQuestions, LongBench, ZeroScrolls, summarization, code completion. | Use the prompt compressor as a front-end control and keep the downstream model, tokenizer, and decode settings fixed so the only difference is compressed prompt bytes. |

## KV Quantization / Geometry Controls

| Family | Primary sources | Clone / runnable here | Likely eval tasks | Bootstrap plan |
|---|---|---|---|---|
| **KIVI** | [paper](https://arxiv.org/abs/2402.02750), [repo](https://github.com/jy-yuan/KIVI) | **Not cloned here, but official repo exists and is runnable.** | LongBench, passkey retrieval, GSM8K, long-context perplexity / throughput checks. | Test asymmetric K/V bit splits first, then compare pre-RoPE vs post-RoPE key quantization at matched bytes and identical prompts. |
| **KVQuant** | [paper](https://arxiv.org/abs/2401.18079), [repo](https://github.com/SqueezeAILab/KVQuant) | **Not cloned here, but official repo exists and is runnable.** | Wikitext-2, C4 perplexity, 1M-token and 10M-token context tests. | Use KVQuant as the stronger low-bit cache quantizer and compare against KIVI at the same average bit budget. |

## Adapter / Feature-Dictionary Baselines

| Family | Primary sources | Clone / runnable here | Likely eval tasks | Bootstrap plan |
|---|---|---|---|---|
| **LoRA** | [PEFT docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora), [original paper](https://arxiv.org/abs/2106.09685) | **Yes via PEFT.** This is a standard control rather than a direct efficiency competitor. | Multi-task adaptation, cross-domain transfer, agent switching, merged-adapter sanity checks. | Treat LoRA as the adapter baseline and compare it to aLoRA on the same base model so any cache reuse benefit is explicit. |
| **Sparse Autoencoders / dictionary learning** | [ICLR paper](https://openreview.net/forum?id=F76bwRSLeK), [Anthropic writeup](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | **Paper-first; not cloned here.** Good source for feature dictionaries and causal feature selection. | Feature interpretability, feature transfer, causal feature localization. | Train a small SAE on the source model, use it as a shared dictionary, and check whether feature selection is more stable than raw activation magnitude. |
| **Universal Sparse Autoencoders (USAE)** | [paper](https://proceedings.mlr.press/v267/thasarathan25a.html), [OpenReview](https://openreview.net/forum?id=UoaxRN88oR), [project](https://yorkucvil.github.io/UniversalSAE/) | **Paper-first; code exists on the project page but is not cloned here.** | Cross-model concept alignment, shared feature space analysis, dictionary transfer. | Use USAE as the strongest shared-dictionary baseline: train one universal feature space, then test whether LatentWire can reuse it for cross-model communication or routing. |
| **Model stitching / linear feature transfer** | [OpenReview](https://openreview.net/forum?id=Qvvy0X63Fv), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/118079) | **Paper-first; not cloned here.** | Cross-model representation alignment, SAE transfer, probe transfer, steering-vector transfer. | Use affine stitching to transfer SAE weights from a smaller model to a larger one, then compare feature overlap and downstream recovery against a learned bridge. |
| **Sparse crosscoders** | [Transformer Circuits draft](https://transformer-circuits.pub/drafts/crosscoders/index.html) | **Paper-first; no local clone.** | Cross-layer feature persistence, model diffing, circuit simplification. | Use crosscoders as the cross-layer dictionary control when you want to know whether the latent bridge is preserving a persistent feature basis or just a shallow layer-local one. |

## What This Set Suggests for LatentWire

- The first paper-safe claim should come from **direct semantic communication**: `C2C` and `KVCOMM` are the cleanest apples-to-apples peers.
- The strongest **cache-side controls** are `KVPress`, `Quest`, `KVzip`, `H2O`, `SnapKV`, and `Ada-KV`; if LatentWire does not beat these on a matched budget, the bridge is not yet a robust communication primitive.
- The most useful **prompt/context controls** are `LLMLingua` / `LongLLMLingua` and `FastKV`; they separate prompt-byte savings from actual cross-model transfer.
- The most informative **orthogonal controls** are `KIVI`, `KVQuant`, `aLoRA`, `SAE`/`USAE`, and model stitching; these test whether the gain is really semantic transfer, cache reuse, or feature reuse.

## Recommended Bootstrap Order

1. `C2C` on GSM8K and SVAMP with the exact source/target pair used by LatentWire.
2. `KVCOMM` on GSM8K, MMLU, and HumanEval with fixed agent count and fixed threshold.
3. `KVPress`, `Quest`, and `KVzip` under a single matched long-context harness.
4. `LLMLingua` / `LongLLMLingua` and `FastKV` as prompt and prefill controls.
5. `KIVI`, `KVQuant`, `aLoRA`, and one shared-dictionary baseline (`USAE` or SAE) as orthogonal controls.

## Telemetry To Standardize

- exact byte budget
- prompt token budget
- repair budget
- KV retention ratio
- retained-token histogram
- pre-repair and post-repair accuracy
- latency p50 and p95
- model pair, tokenizer, chat template, and decode settings
- selected layers, heads, or anchors for any cache-sharing method
- reconstruction loss or cosine drift for any quantization or dictionary baseline
