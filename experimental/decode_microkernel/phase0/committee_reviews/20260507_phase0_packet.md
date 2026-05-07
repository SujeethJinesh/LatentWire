# DMC Phase 0 Committee Review

Reviewed packet: `experimental/decode_microkernel/phase0/results/decode_microkernel_phase0_20260507T233130Z`

Preregistration: `experimental/decode_microkernel/phase0/preregister_dmc_phase0.md`

This is not a paper result. If valid, it only supports authoring a Phase 1 Decode Microkernel Consolidation implementation gate. It does not establish a speedup, latency win, or publishable positive method.

## (a) COLM Area Chair Meta-Review

Scores: novelty 7/10, rigor 7/10, clarity 8/10.

The Phase 0 packet makes a clean pivot from the killed boundary-fusion hypothesis to a narrower measurement question: whether fixed decode traces contain enough launch density and candidate-kernel concentration to justify implementing consolidation. The preregistered decision rule is concrete, and the checker returns PASS with 8 admitted rows, required role coverage, and no checker reasons. The strongest evidence is large median launch density in primary and same-family rows, 1158.59 launches/token, plus cross-family 558.28 launches/token, all above preregistered thresholds.

The contribution is still prospective. Novelty is plausible as a systems-method direction, but no method has been built and no communication or latency improvement is shown. Rigor is good for a Phase 0 audit, but the evidence is fragile at the margin: primary and same-family median top-3 time fractions are only about 0.611 versus a 0.60 threshold, and same-family support falls to exactly the preregistered minimum of 2 rows after one empty `KERNEL_SUMMARY` exclusion.

Fixable issue: Phase 1 should preregister stronger uncertainty and sensitivity checks, especially for top-kernel concentration and excluded-row handling, before any speedup language appears.

## (b) MLSys Systems Review

The packet is a credible implementation-go/no-go artifact, not a systems result. It uses fixed profiler artifacts and client logs, records command metadata, hashes inputs, preserves environment metadata, and avoids new GPU inference as required. The checker PASS is reproducible from the packet with the repo-local GPU venv. The row-level metrics expose the systems target clearly: repeated BF16 GEMV, MoE, and selective-scan-related kernels dominate enough launch/time mass to motivate consolidation work.

Engineering rigor is above average for a Phase 0 gate because the packet separates input admissibility, density, concentration, and class-support conditions. The artifact manifest and sanitized SQLite SHA checks make accidental input drift less likely. The stdout summary is concise and consistent with `metrics.json`.

Main systems concern: this packet measures kernel launch density and concentration, not launch overhead, fusion feasibility, scheduler overhead, memory traffic, or achievable latency savings. It also admits one fewer same-family row due to an empty `KERNEL_SUMMARY`, leaving same-family coverage at the minimum. That is valid under the preregistration, but too thin for design lock-in.

Fixable issue: Phase 1 should include a minimal roofline/overhead accounting plan, a no-op consolidation control, and paired latency confidence intervals before claiming engineering value.

## (c) Adversarial Review

I would accept this only as permission to implement Phase 1. I would reject any attempt to present it as a paper result. The packet passes its preregistered checker, but several features invite overclaiming.

First, the primary and same-family top-3 time fractions barely clear the 0.60 threshold, at about 0.611 and 0.611. A small parsing, classification, or trace perturbation could change the decision. Second, the same-family role loses one of three rows because `KERNEL_SUMMARY` is empty, yet the pass condition still permits 2 rows. This is preregistered, but it weakens robustness. Third, candidate classes are substring-based. `gemv`, `moe`/`expert`, and `selective_scan` are reasonable frozen labels, but they do not prove that the kernels are jointly fusible, schedulable, or part of a single consolidation opportunity.

There is no p-value problem because the gate uses thresholds rather than hypothesis tests, but there is still a multiple-comparisons risk if future memos selectively emphasize the strongest role or top kernel family. The cross-family result is useful falsification coverage, not proof of generality.

Fixable issue: Phase 1 must preregister the exact implementation target and negative controls before inspecting latency results. Required language: this Phase 0 PASS supports only implementation, not speedup, not cross-model communication, and not ICLR readiness.
