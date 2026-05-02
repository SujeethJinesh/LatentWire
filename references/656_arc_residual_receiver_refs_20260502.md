# ARC Residual Receiver References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full remains
  blocked.
- Current story: fixed-byte source-private packets and systems byte/exposure
  accounting are credible; the missing result is still a receiver whose gain
  requires matched source information.
- Exact gap: the frozen-target-public residual receiver did not clear the
  source-necessity gate.

## Primary Sources

1. Learning to Compress Prompts with Gist Tokens.
   <https://arxiv.org/abs/2304.08467>
   - Boundary: latent/gist prompt compression is existing prior art. LatentWire
     must differ through source-conditioned, fixed-byte communication and
     destructive source controls.

2. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://aclanthology.org/2021.acl-long.353/>
   - Boundary: learned continuous prefixes for frozen LMs are not novel. Any
     soft-token or residual receiver must include target-only prompt/prefix
     controls.

3. The Power of Scale for Parameter-Efficient Prompt Tuning.
   <https://arxiv.org/abs/2104.08691>
   - Boundary: target-side soft prompts can be strong without source
     communication. This motivates zero-byte target-only controls.

4. Distilling the Knowledge in a Neural Network.
   <https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/>
   - Boundary: logit/soft-target matching is established behavioral
     supervision, not a new contribution.

5. DistilBERT. <https://arxiv.org/abs/1910.01108>
   - Boundary: mixed behavioral and hidden-state distillation is established.
     Hidden matching must not become the novelty claim by itself.

6. MiniLM. <https://arxiv.org/abs/2002.10957>
   - Boundary: relational attention/value distillation motivates matching
     useful structure rather than raw hidden-state MSE.

7. Activation Engineering. <https://arxiv.org/abs/2308.10248>
   - Boundary: residual-stream steering is prior art. LatentWire must prove
     per-example matched-source necessity rather than generic steering.

8. Inference-Time Intervention. <https://arxiv.org/abs/2306.03341>
   - Boundary: targeted activation shifts are prior art and motivate controls
     that distinguish source communication from target-side intervention.

9. TurboQuant. <https://arxiv.org/abs/2504.19874>
   - Boundary: vector quantization and QJL residual correction are systems
     inspirations, not LatentWire novelty.

10. QJL. <https://arxiv.org/abs/2406.03482>
    - Boundary: JL/sign sketches are known for KV-cache inner-product
      preservation. Current evidence says sign sketches are too lossy for the
      residual receiver on ARC n8.

11. Consistency Models. <https://arxiv.org/abs/2303.01469>
    - Boundary: few-step denoising/repair is prior art. A future LatentWire
      repair receiver should be framed as a controlled task-packet repair map,
      not as a new diffusion model.

12. Scalable Diffusion Models with Transformers.
    <https://arxiv.org/abs/2212.09748>
    - Boundary: DiT motivates latent-token denoising with transformers, but a
      full DiT is too heavy for the current Mac-local gate.

## Experiment Implication

The user-proposed resonance framing is useful, but the paper should avoid a
raw "match hidden activations" claim. Better framing:

`source packet -> target receiver -> behaviorally equivalent target state`

with logit/answer behavior and source-destroying controls first, and selective
hidden/relational probes second. The residual receiver tested here is a clean
control step because target-public scores are frozen before source-dependent
correction is learned.

## Decision

This branch is not paper-positive. It does, however, fix an important
evaluation weakness: zero-source exactly reproduces the frozen target-public
base because the residual design excludes public-only features and uses no
intercept. The next method branch should be a consistency-style receiver repair
gate or a true permutation-equivariant DeepSets/Set Transformer receiver, not
another linear residual scorer.
