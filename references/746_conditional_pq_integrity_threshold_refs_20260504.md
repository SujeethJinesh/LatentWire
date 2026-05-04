# Conditional PQ Integrity Threshold References

Date: 2026-05-04

## Why This Memo Exists

The conditional-PQ scalar integrity threshold gate tests whether a low-rate
source-private packet can be accepted only when it is compatible with target
side information, and rejected to a target-prior no-op when corrupted. The gate
fails, but the references below remain useful for framing why this was the
right falsification test and what novelty boundaries reviewers will expect.

## Decoder Side Information And Syndromes

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources,"
  IEEE Transactions on Information Theory 1973.
  URL: https://doi.org/10.1109/TIT.1973.1055037

  Use: theoretical motivation for compressing correlated sources when decoding
  uses side information. Boundary: LatentWire does not prove an asymptotic
  theorem; it must show task-level packet utility under frozen model controls.

- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder," IEEE Transactions on Information Theory 1976.
  URL: https://doi.org/10.1109/TIT.1976.1055508

  Use: motivates source-private lossy packets decoded against target-side
  state. Boundary: novelty is in empirical LLM packet communication,
  source-private controls, and systems accounting, not the coding theorem.

- Pradhan and Ramchandran, "Distributed Source Coding Using Syndromes
  (DISCUS)," IEEE Transactions on Information Theory 2003.
  URL: https://doi.org/10.1109/TIT.2002.808103

  Use: constructive syndrome/coset precedent for sending parity-like
  information that becomes useful only with decoder side information.
  Boundary: LatentWire packets are learned task packets, not classical
  syndrome codes; reviewers may still expect wrong-packet/no-op controls.

- Liveris, Xiong, and Georghiades, "Compression of Binary Sources With Side
  Information at the Decoder Using LDPC Codes," IEEE Communications Letters
  2002.
  URL: https://doi.org/10.1109/4234.1001660

  Use: practical side-information decoding with parity checks. Boundary:
  classical parity baselines are not directly model-to-model latent
  communication, but they raise the bar for claims about packet integrity.

## Selective Prediction Boundary

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  NeurIPS 2017.
  URL: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

  Use: risk/coverage framing for accept/no-op decisions. Boundary:
  LatentWire's integrity gate is not novel if it is merely target confidence
  thresholding; it must reject corrupted packet families while preserving valid
  source gains.

- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an
  Integrated Reject Option," ICML 2019.
  URL: https://proceedings.mlr.press/v97/geifman19a.html

  Use: learned reject head precedent. Boundary: our scalar threshold gate
  currently fails the stronger communication-specific condition because
  label-shuffled and corrupted packet controls remain competitive.

## Quantization And Systems Baselines

- Jegou, Douze, and Schmid, "Product Quantization for Nearest Neighbor
  Search," IEEE TPAMI 2011.
  URL: https://inria.hal.science/inria-00514462

  Use: canonical compact vector-code baseline. Boundary: PQ itself is not the
  contribution; LatentWire must demonstrate source-private task utility and
  integrity under destructive controls.

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,"
  arXiv 2024.
  URL: https://arxiv.org/abs/2402.02750

  Use: low-bit KV-cache compression comparator. Boundary: KIVI compresses a
  model's own KV cache; LatentWire transmits tiny source-private packets across
  model computations.

- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," arXiv 2025.
  URL: https://arxiv.org/abs/2504.19874

  Use: strong modern vector/KV quantization baseline. Boundary: TurboQuant
  raises the compression bar; LatentWire should differentiate by
  source-private packet decoding, auditability, and corruption/no-op controls.

## Current Gate Consequence

The scalar integrity threshold result weakens simple accept/reject layers over
the current public-zscore conditional-PQ receiver. It accepts real source
packets often enough to improve over target-only, but also accepts corrupted
packet controls too often and loses to a label-shuffled encoder. Future
integrity claims need either a richer learned verifier with held-out
corruption controls, a new packet basis, or a different source/target task
surface.
