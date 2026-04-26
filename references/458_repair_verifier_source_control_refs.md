# Repair, Verifier, and Source-Control References

Focused memo for the SVAMP32 process-repair branch after target-side repair
failed to produce source-selected clean residual recovery.

## Sources

1. Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective
   than Scaling Model Parameters." arXiv:2408.03314.
   <https://arxiv.org/abs/2408.03314>
   - Problem: fixed repair/search budgets can inflate target-prior gains.
   - Mechanism: allocate test-time compute by prompt difficulty and compare
     search against verifier-guided updates.
   - Experiment impact: future repair stacks need budget-matched target-only
     self-consistency and equal-call controls.
   - Role: baseline and ablation.

2. Lightman et al., "Let's Verify Step by Step." arXiv:2305.20050.
   <https://arxiv.org/abs/2305.20050>
   - Problem: final-answer verification cannot show whether source transfer
     improved reasoning.
   - Mechanism: process supervision scores intermediate reasoning steps.
   - Experiment impact: log first bad step, accepted repair step, and whether a
     source intervention changes the step trajectory.
   - Role: baseline and theory support.

3. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
   Language Models." arXiv:2203.11171.
   <https://arxiv.org/abs/2203.11171>
   - Problem: target-only sampling and voting are strong arithmetic baselines.
   - Mechanism: sample multiple reasoning paths and marginalize final answers.
   - Experiment impact: compare any source-assisted repair against target-only
     self-consistency under the same candidate count and decode budget.
   - Role: baseline.

4. Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback."
   arXiv:2303.17651.
   <https://arxiv.org/abs/2303.17651>
   - Problem: target self-repair is already a hard comparator in this repo.
   - Mechanism: generate, critique, and refine with the same model.
   - Experiment impact: keep `target_self_repair` as a required comparator and
     report repair harm, not only final accuracy.
   - Role: baseline and ablation.

5. Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet."
   ICLR 2024.
   <https://openreview.net/forum?id=IkmD3fKBPQ>
   - Problem: no-source repair gains can be target-prior or prompt artifacts.
   - Mechanism: distinguishes intrinsic self-correction from correction with
     external feedback.
   - Experiment impact: require zero-source, shuffled-source, and target-only
     repair controls before calling a repair gain communication.
   - Role: ablation and theory support.

6. Gou et al., "CRITIC: Large Language Models Can Self-Correct with
   Tool-Interactive Critiquing." arXiv:2305.11738.
   <https://arxiv.org/abs/2305.11738>
   - Problem: SVAMP arithmetic can be checked externally, so tool feedback is a
     confound for source communication.
   - Mechanism: verify with tools and feed concrete critique back into repair.
   - Experiment impact: compare against verifier-only or tool-only repair when
     using arithmetic checks.
   - Role: inspiration and ablation.

7. Turpin et al., "Language Models Don't Always Say What They Think:
   Unfaithful Explanations in Chain-of-Thought Prompting." arXiv:2305.04388.
   <https://arxiv.org/abs/2305.04388>
   - Problem: repair rationales may be post-hoc and biased by prompt artifacts.
   - Mechanism: controlled bias interventions show explanations can hide causal
     drivers.
   - Experiment impact: randomize candidate order/labels and hide source
     identity from repair/verifier prompts.
   - Role: ablation and theory support.

8. Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
   Language Models." arXiv:2510.03215 / ICLR 2026.
   <https://arxiv.org/abs/2510.03215>
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Problem: strongest direct cache-communication competitor.
   - Mechanism: project and fuse source KV cache into the target KV cache.
   - Experiment impact: treat C2C as the matched communication baseline, not
     only as an oracle bound.
   - Role: baseline and inspiration.

## Design Rule

Do not promote a repair stack unless it beats target-self repair, target-only
self-consistency, verifier/tool-only repair when applicable, zero-source,
shuffled-source, and source-label/order-shuffle controls under the same
candidate, decode, byte, and latency budget.
