# References: HellaSwag Wyner-Ziv Residual-Logit Packet Gate

## Claim Boundary

This memo supports the failed HellaSwag Wyner-Ziv residual-logit packet gate.
The safe claim is that a train-only, fixed-byte residual-logit packet was
tested and did not improve over the compact packet-only baseline. It does not
claim working learned syndrome communication, solved cross-model latent
reasoning, prefix-token equivalence, or superiority to KV/cache compression.

## Primary Sources And Why They Matter

1. Wyner-Ziv source coding with decoder side information
   - https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
   - Why it matters: the conceptual inspiration for sending a source code that
     is decoded using receiver-side information.

2. DISCUS: distributed source coding using syndromes
   - https://doi.org/10.1109/TIT.2002.808103
   - Why it matters: classic syndrome/coset framing for decoder-side
     side-information coding.

3. Neural Distributed Source Coding
   - https://arxiv.org/abs/2106.02797
   - Why it matters: a learned conditional VQ-VAE approach to distributed
     source coding with side information available only to the decoder.

4. Task-oriented semantic communication
   - https://arxiv.org/abs/2211.08747
   - Why it matters: supports the distinction between task utility and raw
     reconstruction. Our packet optimizes answer choice utility, not vector or
     KV reconstruction.

5. Cache-to-Cache and KV communication
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - https://arxiv.org/abs/2512.17914
   - Boundary: these communicate or compress KV/cache states. The residual-logit
     gate transmits a task-level packet and explicitly sends no source KV,
     source text, raw hidden vectors, or raw score vectors.

6. QJL, TurboQuant, KIVI, and KVQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Boundary: these are vector/KV-cache compression methods. The residual
     packet is not a vector fidelity codec and cannot claim memory/throughput
     wins over them without native systems rows.

7. Prefix-Tuning and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - Boundary: prefix/prompt tuning learns persistent continuous conditioning
     vectors or virtual tokens. The residual-logit packet is a per-example
     discrete source-private code.

8. HellaSwag
   - https://arxiv.org/abs/1905.07830
   - Why it matters: the frozen four-choice commonsense completion benchmark
     used by this gate.

## Reviewer-Facing Framing

Safe:

- We evaluated a train-only Wyner-Ziv-style residual-logit packet under a
  fixed `2B` raw / `5B` framed contract.
- The method failed to beat compact packet-only on full HellaSwag validation.
- The failure is useful because it narrows the live method search: receiver
  selectors, prototype/common-basis receivers, sparse-query receivers, and
  residual-logit packets are all saturated on this surface.

Unsafe:

- Claiming learned syndrome packets work.
- Claiming the receiver learned a cross-model common language.
- Claiming equivalence to prefix tokens or prompt tuning.
- Claiming GPU throughput or HBM traffic improvements.
- Claiming superiority to C2C, KVComm, QJL, TurboQuant, KIVI, or KVQuant.
