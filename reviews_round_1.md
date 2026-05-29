# Reviews Round 1

Date: 2026-05-29

Scope: Efficient Reasoning @ COLM 2026 workshop-hardening review of the committed mechanism/regime draft. Six subagent reviewers were used in this batch: two quantization/algorithm reviewers, two systems/efficiency reviewers, and two methodology/statistics reviewers.

## Score Summary

| Committee | Reviewers | Scores | Mean ordinal score |
|---|---:|---|---:|
| A: Quantization/algorithms | 2 | Weak reject; Weak reject / borderline | 2.25 / 5 |
| B: Systems/efficiency | 2 | Weak reject; Weak reject / borderline | 2.25 / 5 |
| C: ML methodology/statistics | 2 | Weak reject; Borderline | 2.50 / 5 |
| Overall | 6 | 3x weak reject, 3x weak reject/borderline/borderline | 2.33 / 5 |

Ordinal mapping: reject=1, weak reject=2, borderline=3, weak accept=4, accept=5. Mixed scores are averaged between adjacent labels.

## Meta-Review: Consensus Blocking Issues

1. **Novelty versus ParoQuant.** Reviewers A1/A2 flagged that the strongest positive rows are ParoQuant-style rotation, which is prior work. The paper must be read as a mechanism/failure-taxonomy/regime paper, not a new quantizer paper.
2. **Protocol validation language.** A1/A2/C1/C2 flagged that the calibration protocol is derived and evaluated descriptively on the same four models. It must be a checklist or provisional protocol, not a validated deployment rule.
3. **Systems evidence is analytical only.** B1/B2 flagged no profiler, kernel, latency, throughput, or measured memory traffic. The residual-correction cost model should remain a bound for future methods, not a systems result.
4. **Small-N and ratio-statistics risk.** C1/C2 flagged 8--12 trace intervention samples, wide/heavy-tailed CIs, no-gap estimand shifts, and descriptive paired margins.
5. **Scope and fairness caveats.** Reviewers flagged ParoQuant-style implementation fidelity, DecDEC proxy scope, Quamba2 internal-vs-block-output scope, MATH-500 drift-only replication, and overgeneralization from Granite composition.

No reviewer raised a desk-reject concern after Phase 0 fixes.

## Committee A: Quantization / Algorithms

### Reviewer A1

**Summary.** Strong diagnostic/regime paper showing long-decode channel identity drift and that rotation is the strongest baseline. Not convinced the contribution is sufficiently distinct from “ParoQuant works; channel-set variants failed.”

**Strengths.** Concrete 53--67% drift measurement; careful baseline-not-our-method framing; useful Table 1 negative taxonomy with random controls; careful Quamba2 surface scoping; residual-correction failure as mechanism constraint.

**Weaknesses.** Novelty versus ParoQuant remains main problem; ParoQuant-style implementation is underspecified; Quamba2 contrast is delicate; many killed methods are brief; recovery metric is fragile with extreme negative CIs; protocol is descriptive not validated; small-model scope; Granite-only ParoQuant+M11b composition should not be generalized.

**Score.** Weak reject. **Confidence.** High.

### Reviewer A2

**Summary.** Empirical failure taxonomy is useful, but strongest positive numbers come from prior-work ParoQuant-style rotation, and the regime protocol is descriptive rather than validated.

**Strengths.** Clear drift measurement; good controls; stronger no-new-method framing; fairer Quamba2 treatment; useful negative composition result.

**Weaknesses.** Thin novelty versus ParoQuant; protocol is not deployably validated; DriftRot follow-ups look like negative ablations; DecDEC proxy comparison must remain scoped; Quamba2 contrast must avoid overreach; title risks implying a rotation-method contribution.

**Score.** Weak reject / borderline. **Confidence.** Medium-high.

## Committee B: Systems / Efficiency

### Reviewer B1

**Summary.** Mechanism story credible, but systems evidence is not strong enough for systems/efficiency acceptance. The cost envelope is useful but not an evaluated system.

**Strengths.** Does not claim a deployed kernel; concrete HBM/MAC/energy accounting; K-RES failure is systems-relevant; regime framing helps practitioners avoid bad channel-protection work.

**Weaknesses.** No latency, throughput, VRAM, batch-size, occupancy, profiler, or measured traffic; pJ constants are device-agnostic; Table 9 per-projection versus aggregate relationship could be clearer; residual-correction envelope is for a failed method.

**Score.** Weak reject. **Confidence.** High.

### Reviewer B2

**Summary.** Reviewable but borderline for systems track. No measured systems result or deployable efficiency method; only analytical envelope for failed residual-correction branch.

**Strengths.** Avoids overclaiming ParoQuant; formula and cost accounting clear; explicitly analytical; K-RES negative result useful; memory-bound conclusion plausible.

**Weaknesses.** No Nsight/NVML/tokens-sec/latency/VRAM/bandwidth/energy; pJ model too coarse and not tied to hardware/cache/batching; aggregate module/expert count under-justified; hot working set implications underspecified; no roofline plot.

**Score.** Weak reject / borderline. **Confidence.** Medium-high.

## Committee C: ML Methodology / Statistics

### Reviewer C1

**Summary.** Mechanism plausible and negative-result discipline careful, but support is thin: 12 deterministic traces, wide/negative CIs, and same-four-model descriptive protocol.

**Strengths.** Clear recovery metric; matched/random controls; honest ParoQuant baseline framing; wide/negative CIs reported; limitations present.

**Weaknesses.** Small trace counts; recovery ratio unstable near small static gaps; MATH-500 supports drift but not recovery; same models derive and evaluate protocol; adaptive method search not fully addressed by Holm; median narrative can overread wide CIs; small-model scope.

**Score.** Weak reject. **Confidence.** 4/5.

### Reviewer C2

**Summary.** Statistically stronger than earlier drafts but not fully ready. Needs tighter small-N, no-gap, multiple-testing, and descriptive-vs-validated framing.

**Strengths.** Recovery definition explicit; protocol caveat present; negative CIs visible; BCa/Holm mentioned; no-gap traces disclosed.

**Weaknesses.** No-gap estimand changes between early gates and later positive-gap packets; BCa with n=8--12 may be unstable; Holm family is under-specified; some “beats” statements are descriptive; MATH-500 supports drift not recovery; protocol “chooses” remedies too strongly.

**Score.** Borderline. **Confidence.** Medium-high.

## Fixes Selected for Round 1

Safe edits now applied in the draft:

- Changed the contribution bullet from “selects” to “provisionally suggests” for the descriptive calibration checklist.
- In `sections/decision_rule.tex`, changed “chooses” to “provisionally suggests” and “deployment guidance” to “deployment checklist.”
- Added that local confirmation packets remain required.
- Added explicit ParoQuant-style implementation caveat: local baseline implementation using the reported scaled pairwise-rotation design, not official code.
- Added that MATH-500 replication supports drift generalization, not intervention recovery generalization.
- Changed several “beats” statements to “descriptive median margin/higher median” language.
- Scoped ParoQuant+M11b sub-additivity to the tested Granite composition packet.
- Strengthened no-gap estimand caveat: early gates and later positive-gap packets are not identical population quantities.
- Added device-agnostic/cache/batching caveat to the systems cost paragraph.
- Added an aggregate row to the systems-cost table to connect per-projection and full Granite top-8x32 estimates.

## Remaining Honest-Limitation Issues

- No new quantizer beyond ParoQuant-style baseline passes.
- No measured kernel/profiler/latency/throughput evidence.
- 8--12 trace intervention packets remain small with wide CIs.
- Protocol thresholds remain descriptive and not held-out validated.
- MATH-500 replication is drift-only, not recovery-generalization evidence.
