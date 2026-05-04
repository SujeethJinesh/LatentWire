# References: Target Self-Resonance Oracle-Distill Gate

Date: 2026-05-04

## Why These References Matter

The oracle-distill gate failed to beat the target-only slots baseline. This
narrows the novelty boundary: the paper cannot claim a new soft-prompt,
context-compression, or Q-Former-style connector result. The next method must
show that source-conditioned information improves a frozen target over a
matched target-only slot cache at the same byte/slot budget.

## Compact Prefix And Context Compression Priors

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation,"
  ACL-IJCNLP 2021.
  https://aclanthology.org/2021.acl-long.353/

- Lester, Al-Rfou, and Constant, "The Power of Scale for Parameter-Efficient
  Prompt Tuning," EMNLP 2021.
  https://aclanthology.org/2021.emnlp-main.243/

- Mu, Li, and Goodman, "Learning to Compress Prompts with Gist Tokens,"
  NeurIPS 2023.
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html

- Chevalier et al., "Adapting Language Models to Compress Contexts,"
  EMNLP 2023.
  https://aclanthology.org/2023.emnlp-main.232/

- Ge et al., "In-context Autoencoder for Context Compression in a Large
  Language Model," ICLR 2024.
  https://openreview.net/forum?id=uREj4ZuGJE

Boundary: these are single-model prompting or context-compression methods.
Oracle-prefix distillation remains in this family until a source-conditioned
signal adds held-out gain over the target-only slot interface.

## Query Bottlenecks And Common-Basis Interfaces

- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs &
  Outputs," arXiv 2021.
  https://arxiv.org/abs/2107.14795

- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning,"
  NeurIPS 2022.
  https://arxiv.org/abs/2204.14198

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," ICML 2023.
  https://proceedings.mlr.press/v202/li23q.html

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  https://openreview.net/forum?id=SrC-nwieGJ

Boundary: learned query slots and relative/common-basis coordinates are strong
architectural inspiration, not novelty by themselves. LatentWire needs strict
source-destroying and target-cache controls to show communication rather than
ordinary adapter learning.

## Direct Communication And Systems Comparators

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  https://openreview.net/forum?id=LeatkxrBCi
  https://arxiv.org/abs/2510.03215

- Shi et al., "KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing," ICLR 2026.
  https://openreview.net/forum?id=F7rUng23nw
  https://arxiv.org/abs/2510.03346

- Ye et al., "KVCOMM: Online Cross-context KV-cache Communication for Efficient
  LLM-based Multi-agent Systems," arXiv 2025.
  https://arxiv.org/abs/2510.12872

- Kwon et al., "Efficient Memory Management for Large Language Model Serving
  with PagedAttention," SOSP 2023.
  https://arxiv.org/abs/2309.06180

- Zheng et al., "SGLang: Efficient Execution of Structured Language Model
  Programs," arXiv 2023.
  https://arxiv.org/abs/2312.07104

Boundary: C2C and KVComm are the closest direct semantic communication
baselines; vLLM/PagedAttention and SGLang/RadixAttention are serving-system
baselines. LatentWire can claim a bounded source-private interface only after
quality clears target-only controls; throughput superiority must wait for
native GPU serving rows.

## Quantization And Packet-Rate Systems Baselines

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,"
  ICML 2024.
  https://arxiv.org/abs/2402.02750

- Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead," arXiv 2024.
  https://arxiv.org/abs/2406.03482

- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," ICLR 2026.
  https://openreview.net/forum?id=tO3ASKZlok
  https://arxiv.org/abs/2504.19874

Boundary: these compress vectors or KV caches; they do not solve
source-private semantic transfer. They are mandatory systems/rate baselines,
not evidence for LatentWire's method unless implemented in the same native
serving harness.

## Reviewer-Safe Claim Boundary

If the next query-resampler/source-conditioned slot gate passes, safe claims
are:

1. A compact source-conditioned target-slot interface can improve a frozen
   receiver over target-only slots at the same slot/byte budget.
2. The gain is not reducible to prompt compression, static soft prompts,
   candidate-index copying, or target-cache priors if it survives shuffled
   source, wrong-row, zero-source, and candidate-deranged controls.
3. The interface gives a rate-distortion surface for model-to-model
   communication that can be compared against C2C/KVComm and KV quantization
   baselines under explicit byte/exposure accounting.

Do not claim:

- soft prompts are new;
- query slots are new;
- oracle-prefix distillation is cross-model communication;
- target self-compression is sufficient evidence;
- native systems superiority before NVIDIA/vLLM/SGLang rows exist.
