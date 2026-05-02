# ARC Equivariant Set-Delta Accept References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full remains
  blocked.
- Current story: source-private packets are novel only if the matched source
  beats target-public, same-byte visible text, and source-destroyed controls.
- Exact gap: an equivariant set receiver with accept/abstain is a principled
  receiver shape, but this Mac-local ARC gate converges to abstention rather
  than a positive learned communication method.

## Primary Sources

1. Deep Sets. <https://arxiv.org/abs/1703.06114>
   - Boundary: permutation-invariant/equivariant set functions are established
     prior art. LatentWire should not claim novelty for using a set-structured
     candidate receiver.

2. Set Transformer. <https://arxiv.org/abs/1810.00825>
   - Boundary: attention over unordered sets is also prior art. A future
     neural set receiver would be a baseline architecture, not the main
     contribution.

3. Selective Classification for Deep Neural Networks.
   <https://arxiv.org/abs/1705.08500>
   - Boundary: reject/abstain mechanisms are known risk-coverage tools. Our
     accept head is a reviewer-safety device for preventing harm, not a novel
     selective-classification method.

4. SelectiveNet: A Deep Neural Network with an Integrated Reject Option.
   <https://arxiv.org/abs/1901.09192>
   - Boundary: learned accept/reject heads are known. LatentWire's novelty has
     to come from source-private communication and destructive controls.

5. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models. <https://openreview.net/forum?id=LeatkxrBCi>
   - Boundary: C2C is a direct latent/KV communication competitor that moves
     internal source state. It strengthens the need for our source-exposure and
     byte accounting.

6. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://openreview.net/forum?id=F7rUng23nw>
   - Boundary: KVComm shares selected source key-value pairs. LatentWire must
     be positioned as fixed-byte task evidence without raw source KV exposure.

7. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. <https://arxiv.org/abs/2406.03482>
   - Boundary: QJL is a systems comparator for low-bit KV state compression,
     not a private task-packet communication method.

8. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://arxiv.org/abs/2504.19874>
   - Boundary: TurboQuant raises the systems bar for KV-cache compression. Our
     systems claim should compare bytes, exposure, and object scaling, not
     pretend packet bytes replace all serving-state baselines.

9. Prefix-Tuning. <https://arxiv.org/abs/2101.00190>
   - Boundary: continuous target conditioning is established. Any future soft
     or latent receiver needs target-only and zero-source controls.

10. Learning to Compress Prompts with Gist Tokens.
    <https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html>
    - Boundary: prompt/context compression is not enough for novelty; the
      paper must show per-example source necessity.

## Reviewer Objection

The strongest objection after this gate is that learned receivers can simply
learn target-public behavior or abstain when source signals are unreliable.
The result supports that objection for linear ARC candidate receivers. It does
not kill the paper, but it means the paper should not headline a learned ARC
receiver until a new method clears a larger frozen slice, seed repeats,
paired uncertainty, and source-destroyed controls.

## Decision

Treat DeepSets/Set Transformer/selective prediction as baseline machinery.
The paper-facing novelty should be the source-private packet protocol, the
source-destroying evaluation ladder, and systems byte/exposure accounting. If
we revisit learned receivers, the next credible version should be a real
trainable set/transformer receiver or tokenwise target-forward connector on
NVIDIA, not another Mac-local linear feature repair.
