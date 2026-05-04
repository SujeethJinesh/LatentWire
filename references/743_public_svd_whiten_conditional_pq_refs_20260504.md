# Public-SVD Conditional PQ Gate References

- date: `2026-05-04`
- status: local memo for the failed public-SVD whitening conditional-PQ gate
- linked experiment memo:
  `paper/source_private_conditional_pq_public_svd_whiten_gate_20260504.md`

## Why This Memo Exists

The public-SVD whitening branch tested whether each row's public candidate
subspace could act like receiver side information for a tiny source-private
product-quantized innovation packet. This is a natural extension of
side-information-aware vector quantization, but it failed the held-out-family
destructive controls: permuted-code and random same-byte packets became more
predictive than the matched source packet.

## Primary Sources And Novelty Boundary

1. QINCo: Residual Quantization with Implicit Neural Codebooks
   - OpenReview: https://openreview.net/forum?id=NBAc36V00H
   - arXiv: https://arxiv.org/abs/2401.14732
   - Relevance: QINCo makes residual quantization codebooks depend on previous
     reconstruction state. Our conditional-PQ branch is much smaller and
     source-private: the packet stays a fixed byte string and receiver
     conditioning uses only public row/candidate geometry. The failed SVD result
     says a simple public candidate subspace is not enough.

2. QINCo2: Vector Compression and Search with Improved Implicit Neural
   Codebooks
   - OpenReview: https://openreview.net/forum?id=2zMHHZ569S
   - arXiv: https://arxiv.org/abs/2501.03078
   - Relevance: QINCo2 improves implicit neural codebook encoding and decoding.
     This strengthens the case that a serious learned conditional codebook may
     be needed if we stay in the PQ family; the current public-SVD whitening is
     closer to a cheap deterministic diagnostic than to QINCo-style learned
     conditional quantization.

3. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - arXiv: https://arxiv.org/abs/2510.03215
   - Relevance: C2C is the high-bandwidth direct communication baseline because
     it projects/fuses source KV cache into the target model. LatentWire's
     defensible distinction remains low-rate source-private packets with
     destructive controls and byte accounting, not raw accuracy over dense cache
     transfer.

4. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
   - arXiv: https://arxiv.org/abs/2504.19874
   - Relevance: TurboQuant is a strong vector/KV quantization comparator because
     it uses random rotations and scalar/residual quantization to reduce
     distortion. LatentWire should not claim systems superiority over
     TurboQuant-like KV compression without native serving measurements; use it
     as a byte/quality floor for dense-state compression.

## What This Gate Rules Out

- Simple public candidate-subspace SVD whitening is not a safe held-out-family
  fix for conditional-PQ packets.
- Better public geometry alone is insufficient if permuted-code and random
  same-byte controls decode above source.
- The next PQ-family attempt must either train a real conditional receiver
  that rejects corrupted packet codes, or leave the PQ branch and test a
  target-native/self-resonance encoder with the same controls.

## Paper Implication

For COLM_v2, keep the conditional-PQ claim narrow: same-schema/source-private
packet transfer with strict failure modes visible. For ICLR, do not frame
deterministic public-basis conditioning as the main method unless a learned
receiver later clears the controls.
