# Answer-Null Side-Information References

- date: `2026-04-27`
- problem: source sidecars that look positive are explained by direct
  source-final numeric answer copying or by target-side candidate pools that
  already contain answer-explained clean IDs.
- role: inspiration / theory support / baseline design for answer-masked
  communication channels.

## Primary Sources

1. **Process supervision / step verification**
   - Source: `Let's Verify Step by Step`, arXiv:2305.20050
   - URL: https://arxiv.org/abs/2305.20050
   - Helps with: replacing answer-value sidecars with step-level process
     validity signals.
   - Experiment implication: process-verifier sidecars must mask final numeric
     answers and report whether non-answer step predicates survive controls.
   - Role: inspiration and verifier baseline framing.

2. **Process- and outcome-feedback for GSM8K**
   - Source: `Solving math word problems with process- and outcome-based
     feedback`, arXiv:2211.14275
   - URL: https://arxiv.org/abs/2211.14275
   - Helps with: distinguishing process feedback from final-answer supervision
     in math-word-problem settings.
   - Experiment implication: log process-only vs outcome-only sidecar variants
     separately; outcome-only is an upper bound, not communication evidence.
   - Role: baseline and ablation design.

3. **Distributed source coding with decoder side information**
   - Source: `Neural Distributed Source Coding`, arXiv:2106.02797
   - URL: https://arxiv.org/abs/2106.02797
   - Helps with: framing source messages as compressed predicates decoded using
     target-side candidate context.
   - Experiment implication: predicate syndromes should encode source-side
     non-answer structure while relying on receiver candidate pools as side
     information.
   - Role: theory support and mechanism inspiration.

4. **Contrastive chain-of-thought / error rationales**
   - Source: `Contrastive Chain-of-Thought Prompting`, arXiv:2311.09277
   - URL: https://arxiv.org/abs/2311.09277
   - Helps with: transmitting what mistakes to avoid without naming the final
     answer.
   - Experiment implication: test masked contrastive error tags and polarity
     flips as controls before any free-form rationale channel.
   - Role: inspiration for answer-null negative-rationale sidecars.

5. **Speculative accept/reject as target-preserving erasure**
   - Source: `Accelerating Large Language Model Decoding with Speculative
     Sampling`, arXiv:2302.01318
   - URL: https://arxiv.org/abs/2302.01318
   - Helps with: target-preserving acceptance rules where a proposal can be
     rejected without changing the target distribution.
   - Experiment implication: future source sidecars should use erasure/accept
     gates and count accepted harm, not force every source signal into the
     receiver.
   - Role: systems baseline and erasure-gate framing.

6. **Conformal abstention**
   - Source: `Mitigating LLM Hallucinations via Conformal Abstention`,
     arXiv:2405.01563
   - URL: https://arxiv.org/abs/2405.01563
   - Helps with: calibrated abstention when a sidecar is uncertain or likely to
     harm target-correct examples.
   - Experiment implication: report abstention rate, accepted harm, and
     calibration curves for any sidecar router.
   - Role: abstention/routing baseline.

7. **Query bottleneck connectors**
   - Source: `BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
     Image Encoders and Large Language Models`, arXiv:2301.12597
   - URL: https://arxiv.org/abs/2301.12597
   - Helps with: frozen-model query bottlenecks that can mediate a compact
     side-information interface.
   - Experiment implication: if CPU predicate sidecars fail, revisit
     zero-init gated query bottlenecks with an answer-null compatibility
     objective and source-destroying controls from the first gate.
   - Role: connector inspiration.

## Next Experiment Change

The immediate next gate should be a structured answer-null predicate syndrome:
operation sequence, quantity-role graph, equation-shape bucket, unit relation,
and sign/order relation. The sidecar must not include candidate IDs, candidate
values, source final numbers, verified answer numbers, or residue hashes.

Promotion requires matched masked predicates to recover at least one clean
source-necessary ID on a surface where target-side pools already contain clean
answers, while zero-source, shuffled-source, target-only, slots-only, random
same-byte, and polarity/label-shuffle controls keep clean union `0`.
