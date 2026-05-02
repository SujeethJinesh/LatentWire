# ARC Candidate-Alignment Receiver References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked by method-side matched-source evidence.
- Current story: the live method goal is not generic latent alignment, but a
  source-private candidate packet plus receiver whose gains disappear under
  wrong-row, zero-source, candidate-roll, and target-only controls.
- Exact gap: the new candidate-alignment receiver did not clear the n8/n16
  source-necessity gate.

## Primary Sources

1. Relative Representations Enable Zero-Shot Latent Space Communication.
   <https://openreview.net/forum?id=SrC-nwieGJ>
   - Boundary: relative/anchor coordinates are a common-basis precedent, but
     they do not by themselves prove source-private task communication under
     candidate-roll controls.

2. Deep Sets. <https://arxiv.org/abs/1703.06114>
   - Boundary: permutation-invariant/equivariant set scoring is prior art. A
     LatentWire novelty claim must be the controlled source-private packet
     protocol and matched-source necessity evidence, not set functions alone.

3. Set Transformer. <https://arxiv.org/abs/1810.00825>
   - Boundary: attention over unordered candidate sets is a known architecture
     family. A future Set Transformer receiver would be an implementation
     choice, not the contribution by itself.

4. SVCCA. <https://arxiv.org/abs/1706.05806>
   - Boundary: representation comparison and CCA-style axes help diagnose
     common-basis structure, but are not a fixed-byte communication baseline.

5. Similarity of Neural Network Representations Revisited / CKA.
   <https://arxiv.org/abs/1905.00414>
   - Boundary: CKA is useful for probing whether source and target candidate
     features share geometry; it does not solve the receiver gate.

6. Unsupervised Alignment of Embeddings with Wasserstein Procrustes.
   <https://arxiv.org/abs/1805.11222>
   - Boundary: Procrustes/OT alignment is a representation alignment neighbor.
     A learned packet receiver still needs destructive source controls.

7. BLIP-2 / Q-Former. <https://arxiv.org/abs/2301.12597>
   - Boundary: frozen-model query bottlenecks motivate receiver architectures,
     but LatentWire must differ by source-private bytes and target-side
     candidate controls.

8. Cache-to-Cache. <https://arxiv.org/abs/2510.03215>
   - Boundary: C2C transfers/fuses source cache state. The candidate-alignment
     receiver intentionally avoids target LM prefix/KV injection and scores a
     fixed candidate packet externally.

9. QJL. <https://arxiv.org/abs/2406.03482>
   - Boundary: QJL is a KV/cache sketch comparator. The int8/sign candidate
     sketches here are task-packet diagnostics, not KV-cache quantization.

10. TurboQuant. <https://arxiv.org/abs/2504.19874>
    - Boundary: TurboQuant motivates rate-distortion-aware quantization
      pressure, but current evidence says hard sign sketches destroy the
      candidate-alignment signal.

## Experiment Implication

The candidate-alignment receiver branch is scientifically useful because it
separates three effects:

- unquantized hidden-only candidate features can show a tiny n8 accuracy signal
  against candidate-roll/derangement;
- sign-quantized packet sketches destroy that signal;
- pairwise ranking improves target-public-only enough that matched source is no
  longer necessary.

This weakens the current low-capacity external receiver. The next live branch
should either train a residual source correction on top of a frozen target
public scorer or move to a genuine permutation-equivariant Set Transformer /
DeepSets receiver with source-control contrastive training.
