# Packet-Integrity Sideband Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

Weighted corruption-to-no-op training preserved a useful matched packet lift on
the strict ARC n16 slice, but candidate-roll and top-atom-knockout controls
still produced matched-scale residuals. This branch tested whether a
source-answer-free integrity sideband could reject damaged or misaligned sparse
packets before residual fusion.

## Error Detection And Semantic Communication

- [Deep Source-Channel Coding for Sentence Semantic Transmission with
  HARQ](https://arxiv.org/abs/2106.03009) combines semantic coding with
  retransmission/error-detection ideas. The analogy for LatentWire is useful:
  sparse latent packets need an accept/reject signal, but SRP should not claim
  generic HARQ or CRC novelty.
- [Neural Joint Source-Channel Coding](https://arxiv.org/abs/1811.07557)
  learns compression and error correction jointly under a fixed bit budget. SRP
  differs by using a frozen source/receiver LLM pair, strict source-private
  controls, and task-level utility-per-byte reporting.
- [Distributed Deep Joint Source-Channel Coding with Decoder-Only Side
  Information](https://arxiv.org/abs/2310.04311) is a neural Wyner-Ziv-style
  setup. This supports framing the target model's own computation as decoder
  side information.

## Direct Competitors And Novelty Boundary

- [Cache-to-Cache](https://openreview.net/pdf?id=LeatkxrBCi) and
  [KVComm](https://openreview.net/forum?id=F7rUng23nw) remain direct
  high-bandwidth baselines. They move or fuse KV/cache state; SRP moves a tiny
  packet and can add explicit trust/integrity diagnostics.
- [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/) and soft-prompt
  transfer methods make it unsafe to describe SRP as merely continuous prompt
  transfer. SRP must emphasize per-example source-derived packets, byte
  accounting, source-private controls, and destructive packet audits.
- Sparse shared-basis work, including
  [Universal Sparse Autoencoders](https://arxiv.org/abs/2502.03714),
  [SPARC](https://arxiv.org/abs/2507.06265), and
  [Delta-Crosscoder](https://arxiv.org/abs/2603.04426), means the sparse atom
  basis is not novel by itself. The novelty must be communication utility,
  packet integrity, and strict source-use diagnostics.

## Diagnostic Consequence

The first candidate/atom integrity gate failed held-out coverage: it accepted
many matched train packets but rejected all matched held-out packets. That
weakens simple train-fit integrity classifiers on the current linear behavior
atoms and promotes a stronger atom-learning branch. Packet integrity remains a
good protocol story, but only after the atom bank itself has stable
candidate-aligned semantics.
