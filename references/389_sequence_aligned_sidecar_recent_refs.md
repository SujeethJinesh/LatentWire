# 389. Sequence-Aligned Sidecar Recent References

Date: 2026-04-21

This memo narrows the next interface-additive branch after the byte-sidecar
toy. The main read is that recent cross-tokenizer work is converging on
sequence-aware alignment rather than static token remap tables, and that this
maps cleanly onto the current LatentWire low-shot lane.

## Primary sources

1. Cross-Tokenizer LLM Distillation through a Byte-Level Interface
   - Link: https://arxiv.org/abs/2604.07466
   - Date: April 2026
   - Read: A tokenizer-agnostic byte carrier is a stronger default than trying
     to force one-to-one token transfer. This remains the main justification
     for keeping a sidecar rather than only a remap table.

2. DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation
   - Link: https://arxiv.org/abs/2602.21669
   - Date: February 2026
   - Read: Sequence-aware alignment with confidence weighting is the cleanest
     conceptual match for the current sidecar extension. The relevant idea is
     not “more tokens,” but “better-aligned interface trajectories.”

3. TokAlign: Efficient Vocabulary Adaptation via Token Alignment
   - Link: https://arxiv.org/abs/2506.03523
   - Date: June 2025
   - Read: Structured token alignment plus light recovery training can work as
     a cheap adaptation stage. For LatentWire, this suggests that explicit
     alignment features can be additive even without a full retraining loop.

4. Model-Aware Tokenizer Transfer
   - Link: https://arxiv.org/abs/2510.21954
   - Date: October 2025
   - Read: Interface transfer should preserve model-relevant structure, not
     just surface token identity. This supports using alignment features that
     encode tokenizer disagreement structure rather than only bytes.

5. The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems
   - Link: https://arxiv.org/abs/2602.15382
   - Date: February 2026
   - Read: A side channel can carry information that the main communication
     path cannot stably preserve. For LatentWire, sequence alignment should be
     viewed as an additive side-channel refinement, not as a replacement for
     the shared latent basis.

## Concrete read for LatentWire

1. Plain byte sidecars were already better than remap-only controls.
2. The next question was whether sequence-aware interface features add signal
   on top of that sidecar.
3. The new toy says yes: the sequence-aligned sidecar improves the current
   low-shot shared-basis lane again.

## Current local result

- On the strong interface-stress toy, the sequence-aligned sidecar is now the
  best shared-basis branch at `1-2` shots/class and remains the best
  shared-basis branch even at `4-8` shots/class.
- That makes the interface story sharper: the best current additive lane is
  no longer `remap`, then `byte sidecar`, but `byte sidecar + sequence-aware
  alignment`.

## What this means next

1. Keep the sequence-aligned sidecar as the current interface candidate.
2. Do not spend more time on remap-only tweaks as a main branch.
3. Only promote this lane if it survives the frozen `32`-example GSM8K smoke
   and at least one real cross-family mismatch pair.
