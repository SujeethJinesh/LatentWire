# Candidate-Conditioned Packet Builder References, 2026-05-01

## Purpose

This memo supports the learned source-to-candidate packet-builder smoke. The
local result promotes sender-side packet construction over receiver-side
calibration, while keeping the same source-private packet boundary and strict
destructive controls.

Local artifacts:

- `results/source_private_candidate_conditioned_packet_builder_smoke_20260501/`
- `results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed53/`
- `results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/`

Code:
`scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py`

## Primary Sources and Boundaries

- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  arXiv:2510.03215. https://arxiv.org/abs/2510.03215
  - Boundary: C2C is the closest direct model-to-model communication baseline,
    but it fuses projected KV caches. The local method transmits a tiny
    source-private packet and exposes no source KV.
- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  arXiv:2510.03346. https://arxiv.org/abs/2510.03346
  - Boundary: KVComm is a selective KV-sharing systems baseline, not a
    source-private candidate-side-information packet.
- Relative Representations. ICLR 2023.
  https://openreview.net/forum?id=SrC-nwieGJ
  - Boundary: anchor-relative representation transfer is a strong common-basis
    competitor, but does not provide the source-destroying packet controls used
    here.
- Error-Correcting Output Codes: A General Method for Improving Multiclass
  Inductive Learning Programs. JAIR 1995. https://arxiv.org/abs/cs/9501101
  - Boundary: codeword/syndrome views of candidate classes are prior art. The
    local novelty is the source-private LLM packet protocol and controls.
- Learning to Decode Linear Codes Using Deep Learning. arXiv:1802.04741.
  https://arxiv.org/abs/1802.04741
  - Boundary: learned syndrome/code decoding is prior work. This result should
    not claim novelty for neural decoding itself.
- Slepian-Wolf Coding. IEEE TIT 1973.
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Boundary: decoder-side information coding is the right theoretical frame.
    The receiver candidate set is side information.
- Wyner-Ziv Coding. IEEE TIT 1976.
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Boundary: lossy coding with decoder side information predates this work.
    The local contribution is the controlled LLM packet instantiation.
- Distributed Source Coding Using Syndromes (DISCUS). IEEE TIT 2003.
  https://www.researchgate.net/publication/2352091_Distributed_Source_Coding_Using_Syndromes_DISCUS_Design_and_Construction
  - Boundary: syndrome/binning language is prior art. Use it as framing, not as
    a novelty claim.
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders
  and Large Language Models. arXiv:2301.12597.
  https://arxiv.org/abs/2301.12597
  - Boundary: Q-Former-style learned bottlenecks motivate learned interfaces
    between frozen systems, but the local result is not a multimodal connector.
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  arXiv:2504.19874. https://arxiv.org/abs/2504.19874
  - Boundary: low-bit vector/KV compression is a systems comparison wall. The
    local method should only claim a private packet boundary unless native
    serving rows are run.
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
  - Boundary: random projection/sign sketch compression is prior work and a
    relevant future baseline for vector packets.
- vLLM / PagedAttention. SOSP 2023. https://arxiv.org/abs/2309.06180
  - Boundary: required production-serving comparison for ICLR systems claims;
    Mac-local packet latency is not a vLLM throughput result.

## Local Finding

The learned packet builder passes all three public-disjoint seed repeats:

- seed 47: `3/3` directions pass;
- seed 53: `3/3` directions pass;
- seed 59: `3/3` directions pass.

Across the nine n512 rows, learned packet accuracy is `0.875`, live base
accuracy ranges from `0.500` to `0.625`, and best strict destructive-control
accuracy stays at or below `0.258`.

## Novelty Boundary

Safe claim:

> We instantiate source-private side-information coding for cross-model
> candidate disambiguation: a sender-side learned packet maps private source
> evidence into the receiver candidate basis, transmits only an 8-byte packet,
> and is validated against strict source-destroying controls.

Do not claim:

- first latent communication;
- first model-to-model KV communication;
- first syndrome/codeword method;
- broad zero-shot unseen-family transfer;
- GPU serving speedup before native NVIDIA/vLLM rows exist.
