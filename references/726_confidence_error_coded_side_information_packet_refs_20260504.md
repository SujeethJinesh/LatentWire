# Reference Memo 726: Confidence/Error-Coded Side-Information Packets

Date: 2026-05-04

## Current Paper Status

Paper readiness: not ICLR-ready. The Sparse Resonance Packet pivot has a clean
strict ARC gate, but current source-PCA and target-aligned PCA packet branches
are negative.

Current story: LatentWire should now treat the packet as a tiny source-private
decision syndrome decoded against the target model's own uncertainty, rather
than as a hidden-coordinate reconstruction problem.

Exact blocker: a strict ARC packet must beat target-only, target-derived,
same-byte, source-index/rank/score, candidate-roll, atom/bit/header shuffles,
and source-family substitution controls with paired positive uncertainty.

## Fresh Primary Sources

### Side-Information Coding

- Slepian and Wolf, 1973, "Noiseless Coding of Correlated Information
  Sources."
  Source: https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
  Use: independent source encoding can be decoded jointly with correlated
  information. For LatentWire, target logits/cache are decoder side information.
  Boundary: this is a framing precedent, not a novelty claim.

- Wyner and Ziv, 1976, "The Rate-Distortion Function for Source Coding with
  Side Information at the Decoder."
  Source: https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  Use: optimize rate-distortion when only the decoder has side information.
  Boundary: the distortion must be target decision loss, not hidden-state MSE.

- Pradhan and Ramchandran, 2003, "Distributed Source Coding Using Syndromes
  (DISCUS): Design and Construction."
  Source: https://doi.org/10.1109/TIT.2002.808103
  Use: practical syndrome/coset code analogy. Encode a sparse diagnostic parity
  packet and decode it against target-side side information.

### Error-Coded Decisions And Lists

- Hamming, 1950, "Error Detecting and Error Correcting Codes."
  Source: https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
  Use: minimum-distance code design and destructive bit-flip/header tests.

- Dietterich and Bakiri, 1995, "Solving Multiclass Learning Problems via
  Error-Correcting Output Codes."
  Source: https://arxiv.org/abs/cs/9501101
  Use: fixed candidate codewords, Hamming-distance decoding, and probability
  estimates from output codes.

- Allwein, Schapire, and Singer, 2000, "Reducing Multiclass to Binary: A
  Unifying Approach for Margin Classifiers."
  Source: https://jmlr.csail.mit.edu/papers/v1/allwein00a.html
  Use: margin-based output-code decoding. For ARC, decode by target logit
  margin plus packet-code agreement, not by label copying alone.

- Sudan, 1997, "Decoding of Reed Solomon Codes beyond the Error-Correction
  Bound."
  Source: https://people.csail.mit.edu/madhu/papers/1996/reeds-journ.pdf
  Use: list decoding as a principled way to return multiple plausible
  candidates under high uncertainty.

- Guruswami and Sudan, 1999, "Improved Decoding of Reed-Solomon and
  Algebraic-Geometric Codes."
  Source: https://madhu.seas.harvard.edu/papers/1998/venkat-conf.pdf
  Use: decode to a list beyond the unique-decode radius. For LatentWire, let the
  packet produce a list or constrained candidate set and let target side
  information choose within it.

- Koetter and Vardy, 2003, "Algebraic Soft-Decision Decoding of Reed-Solomon
  Codes."
  Source: https://www.itsoc.org/publications/papers/algebraic-soft-decision-decoding-of-reed-solomon-codes
  Use: soft reliability should affect decoding multiplicity/weight. For
  LatentWire, calibrated source margin/header bits should weight the packet,
  not act as an unconditional override.

### Selective And Calibrated Gating

- Chow, 1970, "On Optimum Recognition Error and Reject Tradeoff."
  Source: https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff
  Use: always-on packet updates are the wrong default; the receiver should
  abstain unless estimated gain exceeds expected harm.

- El-Yaniv and Wiener, 2010, "On the Foundations of Noise-free Selective
  Classification."
  Source: https://jmlr.org/papers/v11/el-yaniv10a.html
  Use: report risk-coverage curves for packet firing, not only full-coverage
  accuracy.

- Geifman and El-Yaniv, 2017, "Selective Classification for Deep Neural
  Networks."
  Source: https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
  Use: train/calibrate a risk-controlled selective receiver over target
  uncertainty and source packet reliability.

- Shafer and Vovk, 2008, "A Tutorial on Conformal Prediction."
  Source: https://jmlr.csail.mit.edu/papers/v9/shafer08a.html
  Use: target conformal sets turn the target posterior into a valid list
  surface for packet decoding.

- Romano, Sesia, and Candes, 2020, "Classification with Valid and Adaptive
  Coverage."
  Source: https://proceedings.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
  Use: adaptive classification sets are the right target-side uncertainty
  object for ARC candidate lists.

- Mozannar and Sontag, 2020, "Consistent Estimators for Learning to Defer to an
  Expert."
  Source: https://proceedings.mlr.press/v119/mozannar20b.html
  Use: cost-sensitive defer/override loss. Boundary: the expert here is not a
  full source answer; it is a fixed-byte source-private packet.

### Control And Packet-Header Precedents

- Kalman, 1960, "A New Approach to Linear Filtering and Prediction Problems."
  Source: https://cs-www.cs.yale.edu/homes/yry/readings/general/Kalman1960.pdf
  Use: source packet as a noisy measurement of target decision-state error;
  log normalized innovation and gate updates when the measurement is useful.

- Tabuada, 2007, "Event-Triggered Real-Time Scheduling of Stabilizing Control
  Tasks."
  Source: https://doi.org/10.1109/TAC.2007.904277
  Use: packet firing should be event-triggered by target uncertainty and source
  reliability, not periodic/full-coverage.

- Jacobson, 1990, "Compressing TCP/IP Headers for Low-Speed Serial Links."
  Source: https://www.rfc-editor.org/rfc/rfc1144.html
  Use: low-bit headers work by shared context and delta fields. For ARC, header
  bits should identify packet mode, reliability, and list shape; payload should
  be decoded against target context.

- Bormann et al., 2001, "RObust Header Compression (ROHC)."
  Source: https://www.rfc-editor.org/rfc/rfc3095
  Use: compressed headers need explicit damage/loss propagation controls and
  internal verification. For LatentWire, add header-shuffle, bit-flip, and
  checksum/parity controls.

- Polyanskiy, Poor, and Verdu, 2010, "Channel Coding Rate in the Finite
  Blocklength Regime."
  Source: https://doi.org/10.1109/TIT.2010.2043769
  Use: tiny packets are finite-blocklength, so report empirical packet failure
  rates and paired uncertainty instead of asymptotic coding rhetoric.

## Concrete Strict ARC Recipe

Name: `confidence_error_coded_side_information_arc_gate`.

Decision surface: ARC-Challenge multiple-choice, answer-key-forbidden. Use
official train only for fitting packet calibrators, receiver thresholds, ECOC
weights, temperature scaling, and conformal thresholds. Use a predeclared frozen
validation/eval slice for the reported gate; n8/n16 is scouting only, while a
paper-facing row needs at least 500 rows and five seeds.

Source/target pair: keep the current TinyLlama-to-Qwen3 ARC pair for continuity,
then repeat one strict cross-family falsification pair before promotion.

Packet format, fixed before eval:

1. Shared candidate codebook: assign each candidate index a fixed 8-bit
   Hadamard-style codeword with pairwise Hamming distance 4. For variable ARC
   option count, truncate the candidate set but not the code length.
2. Header, 8 bits:
   - 2 bits packet mode: abstain, top1-syndrome, top2-list, error-repair;
   - 2 bits calibrated source margin bin;
   - 2 bits source entropy/conformal-set-size bin;
   - 1 bit source top1/top2 ambiguity flag;
   - 1 parity/check bit over the payload.
3. Payload, 8-16 bits:
   - 8 bits source top1 ECOC codeword, or its syndrome under a fixed parity
     check matrix;
   - optional 4 bits rival/top2 syndrome;
   - optional 4 bits signed score-shape quantization.
4. Framed packet budget: run exact 2B, 3B, and 4B framed variants. Compare
   against same-byte random, same-byte visible text/code, source-index,
   source-rank, calibrated source-score, and source top2-list baselines.

Decoder:

1. Target computes calibrated candidate probabilities, entropy, margin, and a
   target conformal candidate set.
2. Packet produces a source candidate list by ECOC distance, weighted by source
   reliability header bits. If the parity/check bit fails, force abstain.
3. Score candidate `y` with:
   `log p_T(y) + alpha(header, target_uncertainty) * code_agreement(y, packet)
   + beta(header) * rival_support(y)`.
4. Gate the packet only when all train-calibrated conditions pass:
   target conformal set size > 1, target margin below threshold, source
   reliability bin above threshold, normalized innovation is inside a beneficial
   band, and estimated override value has positive dev lower confidence bound.
5. Otherwise return target-only. Report both full-coverage accuracy and
   risk/coverage for packet-fired rows.

Required controls:

- Target/cache controls: `target_only`, `slots_only`, `target_derived_header`,
  `target_derived_payload`, target-native same-capacity soft slots.
- Source absence and causality: `zero_source`, `source_row_shuffle`,
  wrong-row-within-ARC, wrong-row-cross-benchmark, same-byte random.
- Source-choice controls: source-index, source-rank, calibrated source-score,
  source top1 ECOC without confidence header, source top2-list without header.
- Packet destructive controls: ECOC bit shuffle, header shuffle, reliability-bin
  shuffle, parity/check flip, codeword permutation, top1/top2 payload swap,
  candidate-roll, label permutation, wrong canonical remap, same-byte visible
  text/code.
- Family controls: same-family source substitution and at least one strict
  cross-family source substitution; if substituted packets match or beat the
  method, the branch fails as source communication.
- Oracle/headroom: target top2 oracle, source top2 oracle, union top2 oracle,
  source-unique repair count, packet-fire helps/harms, and abstention regret.

Pass rule:

- Scouting pass: on n16/n32, method must beat target-only, same-byte,
  source-index/rank/score, candidate-roll, and best destructive control without
  losing to target-derived or substitution controls.
- Promotion pass: n >= 500, five seeds, paired bootstrap CI95 low > 0 over
  target-only and best required control, and positive risk/coverage on packet
  fired rows.
- ICLR branch pass: repeat the promoted row on one strict cross-family pair and
  keep the utility-per-byte numerator positive before discussing byte advantage.

## Branch Decision

Promote this as the next highest-value ARC branch because it changes the
objective from coordinate reconstruction to side-information decoding under a
selective harm gate. Do not widen any hidden-coordinate sparse packet branch
until this decision-loss packet has been falsified on the scouting gate.
