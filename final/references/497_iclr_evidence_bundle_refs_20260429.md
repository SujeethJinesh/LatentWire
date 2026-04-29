# ICLR Evidence Bundle Related Work

- date: `2026-04-29`
- blocker: the paper needs a concise novelty matrix separating LatentWire from
  cache communication, prompt compression, tool handoff, quantization, source
  coding, and JEPA/diffusion latent prediction.
- role: primary-source memo supporting
  `results/source_private_iclr_evidence_bundle_20260429/novelty_matrix.csv`.

## Sources

1. C2C cache-to-cache communication (`https://arxiv.org/abs/2510.03215`;
   OpenReview version: `https://openreview.net/forum?id=LeatkxrBCi`).
   - blocker helped: closest cross-LLM non-text communication competitor.
   - mechanism/design idea: project and fuse source KV/cache state into the
     receiver.
   - next experiment change: compare as high-rate internals-access baseline,
     not a same-byte source-private packet baseline.
   - use: baseline and framing.

2. KVCOMM / online cross-context KV-cache communication
   (`https://arxiv.org/abs/2510.12872`) and Q-KVComm
   (`https://arxiv.org/abs/2512.17914`).
   - blocker helped: establishes that multi-agent KV sharing and compressed KV
     communication are close systems baselines.
   - mechanism/design idea: transmit selected or compressed KV state between
     contexts/agents.
   - next experiment change: keep a systems-axis distinction for model-internal
     cache access and payload size.
   - use: baseline and systems framing.

3. LLMLingua prompt compression
   (`https://aclanthology.org/2023.emnlp-main.825/`).
   - blocker helped: reviewers may ask whether compressed text explains the
     packet.
   - mechanism/design idea: compress visible prompt/context tokens.
   - next experiment change: preserve query-aware text relay and rate-frontier
     rows; do not call text weak at all budgets.
   - use: text-channel baseline framing.

4. Toolformer (`https://arxiv.org/abs/2302.04761`) and Model Context Protocol
   tools/resources (`https://modelcontextprotocol.io/specification/2025-06-18/server/tools`).
   - blocker helped: practical agent communication often uses symbolic tool
     calls and exposed resources, not latent packets.
   - mechanism/design idea: tool/API handoff is interpretable but higher-level
     symbolic communication.
   - next experiment change: position LatentWire as a compact private-evidence
     sidecar, not a replacement for tool APIs.
   - use: framing.

5. TurboQuant (`https://arxiv.org/abs/2504.19874`).
   - blocker helped: modern quantization can make vector/cache transport cheap.
   - mechanism/design idea: online vector quantization with rotation and
     residual correction.
   - next experiment change: keep KV byte lower-bound rows and add vector
     matched-byte baselines only if a latent/vector branch is promoted.
   - use: systems baseline and future ablation.

6. QJL (`https://arxiv.org/abs/2406.03482`).
   - blocker helped: one-bit sign sketches are strong low-bit vector baselines.
   - mechanism/design idea: quantized Johnson-Lindenstrauss projections preserve
     inner products.
   - next experiment change: use QJL-style sign sketches for future latent
     candidate-similarity branches.
   - use: baseline and theory support.

7. Slepian-Wolf and Wyner-Ziv source coding
   (`https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources`,
   `https://www.sciencedirect.com/science/article/pii/S0019995878900347`).
   - blocker helped: decoder side information and syndrome coding are classical
     prior art.
   - mechanism/design idea: encoder sends compact task-relevant residual;
     decoder uses side information.
   - next experiment change: frame novelty as empirical LLM/agent
     source-private instantiation plus controls.
   - use: theory framing.

8. I-JEPA (`https://arxiv.org/abs/2301.08243`) and diffusion/flow latent
   prediction, including Flow Matching (`https://arxiv.org/abs/2210.02747`).
   - blocker helped: motivates learned representation-space prediction while
     clarifying that current paper should not overclaim it.
   - mechanism/design idea: predict target-compatible latent innovations rather
     than reconstructing full source evidence.
   - next experiment change: future shared-dictionary/crosscoder receiver needs
     feature knockout and cross-family controls before promotion.
   - use: inspiration and future-method framing.

## Novelty Position

LatentWire is closest in theory to side-information source coding and closest
in LLM systems space to cache/KV communication. The differentiating empirical
surface is: source-private task evidence, explicit byte cap, decoder candidate
side information, strict source-destroying controls, and rate/systems accounting
at the far-left byte point.
