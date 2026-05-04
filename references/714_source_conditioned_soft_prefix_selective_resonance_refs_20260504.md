# Reference Memo 714: Source-Conditioned Soft-Prefix Selective Resonance

Date: 2026-05-04

## Current Paper Status

Paper readiness: not ICLR-ready. Estimated distance: one strong positive
method gate plus seed-stable and cross-family validation.

Current story: score/rank packet branches are saturated, but target-native
soft-prefix capacity is alive. The latest oracle shows that optimized 8-token
target prefixes can reproduce HellaSwag Qwen0.5 validation `0:64` full-prompt
behavior with agreement `0.9375` and mean KL `0.003533`.

Exact blocker: a learned source-conditioned receiver must beat target-only,
source-index/rank/score, same-byte text/code, and destructive controls on the
same decision surface. Oracle target self-resonance is capacity evidence only.

## What Is Genuinely Unique If Implemented Correctly

The defensible novelty is narrow:

1. A per-example source-conditioned encoder emits a compact target-native
   continuous prefix or slot packet.
2. The frozen target consumes that packet directly as soft input.
3. The training objective is selective target-logit resonance: match or improve
   the target's decision-relevant candidate logit geometry only where source
   evidence is expected to help, instead of reconstructing full source states.
4. The packet beats matched target-only soft slots and target-derived slot
   caches at the same slot/byte budget.
5. Source-destroying controls fail: zero-source, wrong-row source, row-shuffled
   source, candidate-deranged source, same-byte visible text/code, source
   index/rank/score packets, and label-shuffled training controls.

That combination is distinct from static soft prompting, prompt compression,
logit fusion, hidden-state alignment, and KV-cache communication. If any of the
controls above ties it, the method is not a cross-model communication result.

## Reviewer Attacks To Expect

- "This is just prefix/prompt tuning." Answer only if the packet is
  per-example, source-conditioned, and beats a target-only learned slot cache.

- "This is prompt/context compression." Answer only if the encoded signal comes
  from source evidence and not from compressing the receiver's own prompt.

- "This is distillation or proxy-tuning through another name." Answer only if
  full source logits are not available at inference and logit KL is used as a
  training signal, not as a raw score-fusion channel.

- "The channel copies candidate labels or answer indices." Answer with
  candidate derangement, label shuffle, source-rank/score packets, and
  same-byte text/code baselines.

- "The target can solve this from target-only slots." This is the central
  attack; target-only slot caches and source-free prefixes must be first-class
  controls, not appendix rows.

- "Same-family geometry is doing the work." A strict non-Qwen source-to-target
  falsification pair is required before claiming cross-family generality.

- "The result is an oracle/memorization artifact." Need larger frozen slices,
  multiple seeds, held-out training split, paired uncertainty, and row-shuffle
  controls.

- "Byte accounting hides continuous-vector cost." Report both conceptual slot
  budget and actual serialized packet bytes; do not compare to C2C/KVComm
  systems claims without native measurements.

- "C2C/KVComm already solved latent communication." The boundary is that they
  expose/fuse dense KV state, while this branch tests a compact, source-private
  target-soft-prefix packet. They are mandatory baselines, not prior art to
  ignore.

- "It is uninterpretable latent steering." Add slot ablations, per-candidate
  logit deltas, source-feature attribution, and destructive controls. Do not
  overclaim mechanistic interpretability.

## Exact Citations And Relevance

### Prefix, Prompt, And Soft-Prompt Compression

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation,"
  ACL-IJCNLP 2021.
  https://aclanthology.org/2021.acl-long.353/
  Relevance: establishes learned continuous prefixes for frozen LMs; LatentWire
  must not claim soft prefixes themselves are new.

- Lester, Al-Rfou, and Constant, "The Power of Scale for Parameter-Efficient
  Prompt Tuning," EMNLP 2021.
  https://aclanthology.org/2021.emnlp-main.243/
  Relevance: establishes soft prompts as frozen-model conditioning; the control
  is a learned target-only prompt/slot baseline.

- Mu, Li, and Goodman, "Learning to Compress Prompts with Gist Tokens,"
  NeurIPS 2023.
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html
  Relevance: compresses prompts into reusable gist tokens; LatentWire must
  distinguish source-conditioned evidence from prompt compression.

- Chevalier et al., "Adapting Language Models to Compress Contexts," EMNLP
  2023.
  https://aclanthology.org/2023.emnlp-main.232/
  Relevance: AutoCompressors turn long context into summary vectors used as soft
  prompts; target-only context compression is a direct confound.

- Ge et al., "In-context Autoencoder for Context Compression in a Large
  Language Model," ICLR 2024.
  https://openreview.net/forum?id=uREj4ZuGJE
  Relevance: ICAE uses memory slots for context compression; LatentWire must
  beat same-slot target self-compression.

- Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of
  Large Language Models," EMNLP 2023.
  https://aclanthology.org/2023.emnlp-main.825/
  Relevance: strong text-token compression baseline for same-byte visible
  communication.

- Pan et al., "LLMLingua-2: Data Distillation for Efficient and Faithful
  Task-Agnostic Prompt Compression," Findings ACL 2024.
  https://aclanthology.org/2024.findings-acl.57/
  Relevance: learned extractive compression with distillation; same-byte
  text/code controls should be competitive, not straw baselines.

### Query Bottlenecks And Soft Interfaces

- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs &
  Outputs," ICLR 2022.
  https://arxiv.org/abs/2107.14795
  Relevance: learned query bottlenecks are established; LatentWire novelty is
  the source-conditioned communication protocol and controls, not query slots.

- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning,"
  NeurIPS 2022.
  https://arxiv.org/abs/2204.14198
  Relevance: pretrained model bridging via learned connectors is prior art for
  the architecture pattern.

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," ICML 2023.
  https://proceedings.mlr.press/v202/li23q.html
  Relevance: Q-Former bridges frozen encoders and LMs; this makes bridge
  modules mandatory related work, not a novelty claim.

### Distillation And Logit Composition

- Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network,"
  arXiv 2015.
  https://arxiv.org/abs/1503.02531
  Relevance: dark-knowledge/logit matching is the ancestor of resonance losses;
  LatentWire must not call logit KL itself novel.

- Liu et al., "DExperts: Decoding-Time Controlled Text Generation with Experts
  and Anti-Experts," ACL 2021.
  https://arxiv.org/abs/2105.03023
  Relevance: combines model distributions at decoding time; raw logit fusion is
  a mandatory comparator.

- Li et al., "Contrastive Decoding: Open-ended Text Generation as
  Optimization," ACL 2023.
  https://arxiv.org/abs/2210.15097
  Relevance: shows expert-amateur logit differences can steer decoding without
  new training; selective resonance must beat equal-byte score/logit controls.

- Liu et al., "Tuning Language Models by Proxy," COLM 2024.
  https://arxiv.org/abs/2401.08565
  Relevance: proxy-tuning shifts target predictions using another model's score
  delta; LatentWire must avoid collapsing into black-box proxy score fusion.

### Cross-Model Communication And KV State Transport

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  https://openreview.net/forum?id=LeatkxrBCi
  Relevance: closest direct semantic communication competitor; C2C projects and
  fuses source KV caches into target caches.

- Shi et al., "KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing," ICLR 2026.
  https://openreview.net/forum?id=F7rUng23nw
  Relevance: selective KV-pair sharing is a strong systems/quality comparator
  with much larger state exposure.

- Liu et al., "DroidSpeak: KV Cache Sharing for Cross-LLM Communication and
  Multi-LLM Serving," arXiv 2024.
  https://arxiv.org/abs/2411.02820
  Relevance: cross-LLM KV reuse under same-architecture constraints; LatentWire
  must separate fixed-byte source-private packets from KV reuse systems.

### Alignment, Relative Coordinates, SAEs, And Crosscoders

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  https://openreview.net/forum?id=SrC-nwieGJ
  Relevance: anchor-relative representation is prior art for latent-space
  communication; LatentWire must show causal target improvement, not only
  coordinate alignment.

- Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in
  Language Models," arXiv 2023.
  https://arxiv.org/abs/2309.08600
  Relevance: SAEs give sparse interpretable feature bases; they can support
  diagnostics but are not a communication method by themselves.

- Lan et al., "Quantifying Feature Space Universality Across Large Language
  Models via Sparse Autoencoders," arXiv 2024.
  https://arxiv.org/abs/2410.06981
  Relevance: feature universality motivates shared feature packets, but it also
  creates a baseline attack that common features may explain any gain.

- Anthropic, "Insights on Crosscoder Model Diffing," 2025.
  https://www.anthropic.com/research/crosscoder-model-diffing
  Relevance: crosscoders jointly model activation spaces for model comparison;
  useful for interpretability, not sufficient evidence of communication.

- Chaudhari, Hundia, and Gulati, "Sparse Crosscoders for diffing MoEs and Dense
  models," arXiv 2026.
  https://arxiv.org/abs/2603.05805
  Relevance: recent crosscoder work reinforces that shared-vs-specific features
  are a natural diagnostic and possible attack surface.

### Continuous Latent Reasoning

- Hao et al., "Training Large Language Models to Reason in a Continuous Latent
  Space," COLM 2025.
  https://arxiv.org/abs/2412.06769
  Relevance: Coconut shows continuous latent tokens can support reasoning
  inside one model; it does not solve cross-model communication.

- Zhang et al., "Soft Thinking: Unlocking the Reasoning Potential of LLMs in
  Continuous Concept Space," arXiv 2025.
  https://arxiv.org/abs/2505.15778
  Relevance: continuous concept tokens are prior art for token-free reasoning;
  LatentWire must not claim continuous reasoning tokens are new.

- Liu et al., "Deliberation in Latent Space via Differentiable Cache
  Augmentation," arXiv 2024.
  https://arxiv.org/abs/2412.17747
  Relevance: trains a frozen-LM cache coprocessor with LM loss; this is close to
  target-cache augmentation and strengthens the target-only control requirement.

## What Not To Claim

- Do not claim soft prefixes, soft prompts, query slots, or latent tokens are
  new.
- Do not claim oracle-optimized target prefixes are cross-model communication.
- Do not claim prompt compression is beaten unless same-byte LLMLingua-style
  text/code baselines are run on the identical rows.
- Do not claim logit resonance is new; claim only a source-conditioned compact
  communication channel if it beats raw source score/logit controls.
- Do not claim cross-family generality from Qwen-to-Qwen or same-family rows.
- Do not claim interpretability from SAEs/crosscoders unless slot-level causal
  ablations and feature attribution survive controls.
- Do not claim systems superiority over C2C/KVComm/vLLM/SGLang until native
  GPU serving rows and honest packet serialization are in the ledger.

## Consequence For The Next Gate

The next branch should be framed as:

"A source-conditioned, rate-limited target-soft-prefix communication protocol
trained with selective logit-resonance loss, evaluated against target-only slot
caches and source-destroying controls."

The first pass/fail gate should remain the learned encoder, not another oracle:
train on official training rows, freeze the target, emit 8 target-native slots,
and compare against target-only slots, source-free slots, source-index/rank/score
packets, same-byte text/code, row-shuffled source, candidate-deranged source,
label-shuffled training, and Qwen-substituted controls with paired uncertainty.
