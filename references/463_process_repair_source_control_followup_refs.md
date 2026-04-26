# Process-Repair Source-Control Follow-Up References

Focused memo for the SVAMP70/GSM70 process-repair route after old results
showed positive held-out accuracy but before source-destroying controls decide
whether the branch is communication or target-side repair.

## Sources

1. Cobbe et al., "Training Verifiers to Solve Math Word Problems."
   <https://arxiv.org/abs/2110.14168>
   - Problem: verifier selection can improve math accuracy without source
     communication.
   - Mechanism: generate many candidate solutions, then rank with a verifier.
   - Experiment impact: add target-only candidate generation plus verifier
     reranking at the same candidate count, repair calls, latency, and token
     budget as source-assisted repair.
   - Role: baseline.

2. Lightman et al., "Let's Verify Step by Step."
   <https://arxiv.org/abs/2305.20050>
   - Problem: final-answer scoring cannot show whether a source route improves
     reasoning steps.
   - Mechanism: process supervision scores intermediate solution steps.
   - Experiment impact: log first bad step, accepted repair step, and whether
     matched source changes the step trajectory more than zero or shuffled
     source.
   - Role: baseline and theory support.

3. Snell et al., "Scaling LLM Test-Time Compute Optimally can be More
   Effective than Scaling Model Parameters."
   <https://arxiv.org/abs/2408.03314>
   - Problem: extra repair/search compute can explain gains independently of
     communication.
   - Mechanism: allocate test-time compute across search and verifier-guided
     updates.
   - Experiment impact: every source-repair row needs equal-call target-only
     repair/search and best-of-N controls with paired uncertainty.
   - Role: baseline and ablation.

4. Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback."
   <https://arxiv.org/abs/2303.17651>
   - Problem: target self-feedback is the direct confound for process-repair
     claims.
   - Mechanism: a model generates, critiques, and refines its own answer.
   - Experiment impact: keep `target_self_repair` mandatory and report repair
     help/harm, changed-answer rate, and route-specific lift.
   - Role: baseline.

5. Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet."
   <https://arxiv.org/abs/2310.01798>
   <https://openreview.net/forum?id=IkmD3fKBPQ>
   - Problem: distinguishes intrinsic self-correction from correction using
     external information.
   - Mechanism: tests whether models fix reasoning without feedback.
   - Experiment impact: require zero-source, shuffled-source, slots-only, and
     target-only repair controls before calling a repair gain communication.
   - Role: ablation and theory support.

6. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
   Language Models."
   <https://arxiv.org/abs/2203.11171>
   - Problem: target-only sampling and voting can beat greedy decoding without
     any source signal.
   - Mechanism: sample multiple reasoning paths and marginalize final answers.
   - Experiment impact: compare source-assisted repair against target-only
     self-consistency at matched sample count and token budget.
   - Role: baseline.

7. Du et al., "Improving Factuality and Reasoning in Language Models through
   Multiagent Debate."
   <https://arxiv.org/abs/2305.14325>
   - Problem: candidate exchange can look like communication while only adding
     diversity and selection.
   - Mechanism: multiple model instances propose answers and debate over rounds.
   - Experiment impact: include a text-visible multi-agent or source-candidate
     baseline when claiming latent repair beats explicit candidate exchange.
   - Role: baseline and inspiration.

8. Turpin et al., "Language Models Don't Always Say What They Think:
   Unfaithful Explanations in Chain-of-Thought Prompting."
   <https://arxiv.org/abs/2305.04388>
   <https://openreview.net/forum?id=bzs4uPLXvi>
   - Problem: repair rationales can hide prompt/order/source-label leakage.
   - Mechanism: controlled bias interventions show explanations may not reflect
     causal drivers.
   - Experiment impact: blind source identity, randomize candidate order and
     labels, hash trace provenance, and require matched-source gains to
     disappear under source shuffle but not order shuffle.
   - Role: leakage-control ablation and theory support.

## Decision Rule

Promote process repair only if matched-source route repair beats target-alone,
target self-repair, and budget-matched target-only candidate search, while the
same route-specific wins disappear under zero-source, shuffled-source,
target-only, slots-only, and source-label/order-shuffle controls.
