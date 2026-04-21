# 386. Cross-Tokenizer Interface Alignment Recent References

Date: 2026-04-21

This memo narrows the next interface ablation queue. The main read is that
recent cross-tokenizer and cross-model transfer work is moving away from static
token remap tables and toward tokenizer-agnostic side channels, sequence-level
alignment, and model-aware selective transfer.

## Primary sources

1. Cross-Tokenizer LLM Distillation through a Byte-Level Interface
   - Link: https://arxiv.org/abs/2604.07466
   - Date: April 2026
   - Read: A byte-level interface can act as a tokenizer-agnostic carrier rather
     than forcing a brittle token-to-token mapping. For LatentWire, this points
     to a `latent + byte sidecar` ablation instead of spending more time on
     discrete remap tables alone.

2. DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation
   - Link: https://arxiv.org/abs/2602.21669
   - Date: February 2026
   - Read: Soft-DTW style alignment and confidence-weighted token transfer are
     a stronger fit for variable-length interface drift than simple coverage or
     remap counts. This is a clean inspiration for sequence-aligned route-atom
     or span-trajectory losses.

3. Model-Aware Tokenizer Transfer
   - Link: https://arxiv.org/abs/2510.21954
   - Date: October 2025
   - Read: The main insight is to make tokenizer transfer model-aware using
     attention influence rather than surface or embedding similarity alone. For
     our setting, interface adaptation should probably preserve communication
     structure across heads/layers, not just token identity.

4. TokAlign: Efficient Vocabulary Adaptation via Token Alignment
   - Link: https://arxiv.org/abs/2506.03523
   - Date: June 2025
   - Read: A structural vocab-alignment step plus light recovery training can
     work surprisingly well. This motivates a clean contrast between
     `structural vocab replacement + light recovery` and our current latent
     bridge.

5. Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
   - Link: https://arxiv.org/abs/2503.20083
   - Date: March 2025
   - Read: ALM is a real cross-tokenizer distillation baseline, not just a
     heuristic remap. If we broaden the paper toward cross-family
     communication, a baseline in this family is more honest than another ad
     hoc remap control.

6. tokenkit
   - Link: https://github.com/bminixhofer/tokenkit
   - Read: Practical toolkit for byteification, tokenizer transfer, and related
     evaluation. Useful if we want a lightweight real-world control beyond the
     current toy remap harness.

7. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Date: October 2025
   - Read: C2C highlights selective layer exposure rather than a single uniform
     interface. The relevant implication for LatentWire is that interface
     control may need layer-conditioned or route-conditioned masks.

8. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Date: October 2025
   - Read: Selective KV sharing again suggests that the next interface ablation
     should be selective and model-aware under a fixed budget, not just
     token-boundary aware.

9. Reasoning with Latent Tokens in Diffusion Language Models
   - Link: https://arxiv.org/abs/2602.03769
   - Date: February 2026
   - Read: Latent tokens can serve as a controllable reasoning interface. This
     is relevant because a small learned planning side channel may be less
     brittle than forcing explicit tokenizer alignment across models.

## Concrete ablations for LatentWire

1. Latent plus byte sidecar
   - Keep the quotient + GPA + sparse-dictionary lane.
   - Add a byte-level decoder head or byte-sidecar communication stream.
   - Compare `latent-only`, `byte-only`, and `latent+byte` under matched byte
     budgets on cross-family pairs.

2. Sequence-aligned interface loss
   - Replace remap-table-only supervision with Soft-DTW or OT over span or
     route-atom trajectories.
   - Weight alignment by sender confidence or boundary confidence.
   - Evaluate whether the gain concentrates in the current `1-2` shot regime
     where the shared-basis method is strongest.

3. Model-aware selective interface
   - Score heads/layers/atoms by attention influence, KV importance, or route
     utility.
   - Compare `full bridge`, `selected bridge`, and `selected bridge +
     quantization` at a fixed budget.

## Why this memo matters now

- The current interface-stress toy says byte/span remap can help the composed
  low-shot lane under strong corruption, but the gain is modest.
- That means remap-only work should now be treated as a robustness control,
  not the center of the method story.
- The stronger next interface moves are sequence-aligned, side-channel, or
  model-aware selective interfaces.
