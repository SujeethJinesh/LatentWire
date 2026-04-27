# Target-Preserving Receiver Gate References

Date: 2026-04-27

## Problem Helped

The current blocker is target-self destruction. Source-derived messages can
recover live source-necessary IDs, but they must be accepted only when the
target receiver has evidence that the source proposal is safe. This memo
collects mechanisms for receiver-side acceptance, erasure, and final-answer
constraints.

## Sources And Mechanisms

1. Speculative Sampling / Decoding
   - Source: `https://arxiv.org/abs/2302.01318`
   - Helps with: target-preserving acceptance of source proposals.
   - Mechanism: a draft proposal is verified by the target distribution through
     a rejection/acceptance rule that preserves the target distribution.
   - Experiment impact: motivated CPU target-likelihood receiver scoring and
     the rule that source proposals must be externally verified before
     replacing target outputs.
   - Role: inspiration and systems framing.

2. Distributed Deep JSCC with Decoder-Only Side Information
   - Source: `https://arxiv.org/abs/2310.04311`
   - Helps with: treating source communication as decoder-side information
     rather than answer overwrite.
   - Mechanism: side information is injected at the receiver and can be ignored
     when unhelpful.
   - Experiment impact: supports erasure-aware semantic predicate decoding.
   - Role: inspiration.

3. CRANE: Reasoning with Constrained LLM Generation
   - Source: `https://arxiv.org/abs/2502.09061`
   - Helps with: avoiding trace-level overconstraint.
   - Mechanism: constrain only selected regions/final-answer format rather than
     the entire reasoning trajectory.
   - Experiment impact: the CPU decoder scores final numeric candidates and
     leaves target reasoning untouched.
   - Role: ablation design and caution.

4. Mitigating LLM Hallucinations via Conformal Abstention
   - Source: `https://arxiv.org/abs/2405.01563`
   - Helps with: calibrated refusal to inject unsafe source information.
   - Mechanism: abstain when uncertainty/risk exceeds a calibrated threshold.
   - Experiment impact: explicit `erasure`/fallback behavior and accepted-harm
     reporting.
   - Role: ablation design and source-fault inspiration.

5. Self-Consistency Improves Chain-of-Thought Reasoning
   - Source: `https://arxiv.org/abs/2203.11171`
   - Helps with: candidate-pool semantic agreement as a diagnostic.
   - Mechanism: aggregate over multiple reasoning paths/candidate answers
     rather than trusting a single greedy trace.
   - Experiment impact: semantic predicate decoders should be treated as
     candidate-pool filters, not proof of communication unless they beat
     target/text/C2C controls.
   - Role: diagnostic inspiration.

## Next Experiment Impact

This literature does not justify scaling the current semantic predicate branch.
The strict CPU gate recovered live IDs with no accepted harm but failed holdout
clean recovery. The practical next experiment remains stronger source-surface
discovery after MPS cleanup. Receiver-side likelihood/scoring can be revisited
only with a stronger surface and C2C kept as an external baseline.
