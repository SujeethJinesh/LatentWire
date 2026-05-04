# Paired BatchTopK Crosscoder Atom Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

The source-only BatchTopK behavior atom bank fit the tiny train slice but did
not improve held-out ARC accuracy. This branch tested the next smallest
crosscoder-style variant: learn the sparse packet basis from paired
source-hidden and target-hidden public innovations, while emitting only the
source-side sparse atom packet at runtime.

## Primary Sources And Novelty Boundary

- [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/)
  jointly model multiple activation spaces with shared sparse features. This is
  the nearest conceptual prior for a paired source/target atom basis.
- [Cross-Architecture Model Diffing with Crosscoders](https://arxiv.org/abs/2602.11729)
  introduces Dedicated Feature Crosscoders for separating shared and
  model-exclusive features across different LLM architectures. This is a strong
  novelty threat for any shared/private atom claim.
- [Anthropic's model diffing tool](https://www.anthropic.com/research/diff-tool)
  frames DFCs around shared, model-A-only, and model-B-only features and validates
  some features through steering. LatentWire must add a communication objective,
  byte budget, and strict source-private controls to be distinct.
- [Relative Representations](https://openreview.net/forum?id=SrC-nwieGJ) are
  prior art for latent communication through coordinate canonicalization. Any
  anchor-relative or shared-coordinate packet must be framed as a baseline unless
  it also proves downstream source-conditioned utility.
- [Gist Tokens](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html)
  and [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/) are
  target-side continuous conditioning/context-compression baselines. They matter
  because a source packet must beat same-capacity target-only or target-derived
  conditioning to prove communication.
- [Cache-to-Cache](https://arxiv.org/abs/2510.03215) and
  [KVComm](https://openreview.net/forum?id=F7rUng23nw) define the dense
  cache/KV-transfer competitor class. LatentWire's differentiator must remain
  low-rate source-only packets plus utility-per-byte and destructive controls.

## Scout Outcome

A new `--atom-basis-mode paired_batchtopk_behavior` mode was added to the strict
ARC behavior-atom harness. The target hidden states are used only for train-time
basis calibration on the fit rows; held-out runtime packets are source-only.

| Variant | Packet | Matched | Target | Best required control | Helps | Harms | Decision |
|---|---|---:|---:|---|---:|---:|---|
| paired BatchTopK top-1 | rank16 top1 q4 | 0.2500 | 0.3750 | qwen_substituted_packet, 0.6250 | 1 | 2 | fail |
| paired BatchTopK top-2 | rank16 top2 q4 | 0.2500 | 0.3750 | qwen_substituted_packet, 0.6250 | 0 | 1 | fail |

The paired atom bank again fits train behavior well (`source_fit_r2 ~= 0.73`
top-1 and `0.90` top-2; `target_fit_r2 ~= 0.93` and `0.96`), but the held-out
packet harms target accuracy. The paired alignment did make destructive
candidate-roll controls collapse on the n8 slice, but only because matched also
collapsed. This weakens naive paired BatchTopK crosscoder atoms as the next
ICLR-positive branch.

## Consequence

Do not scale this paired BatchTopK atom bank as-is. If sparse atoms remain the
main path, the next step should either:

1. use an explicit DFC partition with shared/source-only/target-only atoms and
   a receiver loss that penalizes matched harms; or
2. run a target-side behavior-transcoder feasibility probe first, proving that
   sparse target atoms can causally steer ARC margins before asking the source
   to transmit them.

The code path remains useful because it now enforces train-only target hidden
calibration and keeps packet byte accounting source-only.
