# ARC Hidden/Query Common-Basis Related-Work Boundary

Date: 2026-05-02

## Local Result

The ARC hidden/query common-basis gate is negative:

- artifact:
  `results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement/arc_challenge_hidden_query_common_basis_gate.json`
- test matched/Qwen-substituted/cached-Tiny mean:
  `0.229598 / 0.317125 / 0.269345`
- matched minus Qwen-substituted mean/min: `-0.087526 / -0.105708`
- CI95 lower bound versus Qwen-substituted: `-0.159672`
- pass gate: `False`

## Reviewer-Safe Novelty Boundary

Do not claim novelty for latent communication, hidden-state communication,
query bottlenecks, shared latent spaces, common bases, sparse cross-model
dictionaries, or quantized KV/cache transport in general.

The defensible LatentWire boundary is narrower:

> fixed-byte, source-private evidence packets decoded with public target-side
> side information, with destructive controls that separate real source signal
> from target priors, row shuffles, candidate rolls, same-byte text, and
> source-family cache substitution.

## Primary Related Work

- Cache-to-Cache (C2C), arXiv:2510.03215 / ICLR 2026:
  learns to project and fuse source-model KV cache into the target KV cache.
  This is high-bandwidth cache-state transfer, not a fixed-byte source-private
  packet.
  https://arxiv.org/abs/2510.03215
- KVComm, OpenReview ICLR 2026:
  selectively shares KV pairs and reports near-upper-bound performance with a
  reduced fraction of layer KV pairs. This remains KV exposure/transfer rather
  than source-private packet evidence.
  https://openreview.net/forum?id=F7rUng23nw
- KVCOMM, arXiv:2510.12872:
  reuses and aligns KV caches across multi-agent contexts via anchor deviations.
  This is a serving/cache-reuse system, not task-level fixed-byte source
  evidence.
  https://arxiv.org/abs/2510.12872
- Communicating Activations Between Language Model Agents, arXiv:2501.14082:
  directly combines intermediate activations between LM agents. It validates
  activations as a communication medium, but does not impose the LatentWire
  fixed-byte/source-private/destructive-control packet boundary.
  https://arxiv.org/abs/2501.14082
- InterLat, arXiv:2511.09149:
  communicates continuous last hidden states, with learned compression for
  latent-space reasoning. This is close in motivation but still broader and
  higher-bandwidth than our packet protocol.
  https://arxiv.org/abs/2511.09149
- Latent Space Communication via K-V Cache Alignment, arXiv:2601.06123:
  learns shared K/V-cache spaces using model adapters. This is a learned
  shared-cache interface, not a source-private public-basis packet.
  https://arxiv.org/abs/2601.06123
- Sparse crosscoders and universal sparse autoencoders:
  support the idea that model activations can be aligned through shared sparse
  concept dictionaries, but they are interpretability/alignment frameworks, not
  the same downstream fixed-byte packet test.
  https://transformer-circuits.pub/2024/crosscoders/index.html
  https://arxiv.org/abs/2502.03714
- TurboQuant and QJL/KV quantization:
  motivate random rotations and low-bit latent/cache sketches, but their goal
  is vector/KV fidelity or cache compression rather than source-private
  downstream evidence under destructive controls.
  https://arxiv.org/abs/2504.19874

## Method Decision

The negative ARC result means our next claim should not be “TinyLlama hidden
states can speak the Qwen ARC packet language.” The safer story is:

1. public coordinate agreement is real and useful for Qwen-source ARC packets;
2. stronger same-family source decisions can be transferred through the packet;
3. Mac-local non-Qwen source-family repair remains unsolved;
4. systems value should be framed as byte/exposure boundary accounting until
   native C2C/KVComm/KVCOMM/TurboQuant/vLLM/SGLang rows are measured.
