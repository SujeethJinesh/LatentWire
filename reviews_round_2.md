# Reviews Round 2

Date: 2026-05-29

Scope: post-Round-1 hardening draft. Three subagents reviewed the current TeX/PDF sources with distinct committee lenses. No reviewer edited files.

## Score Summary

| Committee | Lens | Score | Confidence | Blocking issue after Round 1? |
|---|---|---|---|---|
| A | Quantization / algorithms | Borderline | Medium-high | No hard blocker; novelty versus ParoQuant remains the main acceptance risk. |
| B | Systems / efficiency | Weak accept | Medium-high | No blocker for workshop framing; missing profiler/kernel blocks stronger systems claims. |
| C | ML methodology / statistics | Weak accept | Medium-high | No blocker for scoped workshop framing; small-N/descriptive protocol remain limitations. |

Mean score on a 5-point reject-to-accept scale: 3.33/5.

## Committee A: Quantization / Algorithms

**Summary.** The no-new-method framing is now mostly acceptable for a mechanism/regime workshop paper. Novelty is not "rotation helps" or "dynamic outliers exist"; it is the audited long-decode protected-channel drift measurement, matched-control failure taxonomy, and practical conclusion that original-basis channel protection should be validated against rotation.

**Strengths.**

- Scope correction is clear: ParoQuant is described as a baseline, not our method, in the introduction, method/results framing, Table 1 caption, and limitations.
- Prior-rotation boundary is stronger. The paper positions itself against ParoQuant/QuaRot/SpinQuant by measuring channel identity drift and testing whether original-basis protection still works.
- Quamba2 comparison is scoped to measured block-output top-K protection and does not deny internal SSM persistence.
- Table 1 is useful because negative rows and heavy-tailed CIs are visible rather than hidden.
- Round 1 fixes landed where needed: MATH-500 is drift-only; paired margins are descriptive; the protocol provisionally suggests and requires confirmation.

**Weaknesses.**

- Novelty remains thin for a quantization/algorithms reviewer because the strongest positive rows are ParoQuant-style rotation, which is prior work.
- The calibration protocol is descriptive, not validated on held-out models.
- The basis-removal mechanism is plausible but not isolated with official ParoQuant/QuaRot/SpinQuant ablations.
- Local ParoQuant-style implementation fidelity is still a caveat.
- Small-N and ratio instability remain visible, especially the ParoQuant+M11b wide CI.

**Score:** Borderline.

## Committee B: Systems / Efficiency

**Summary.** The paper is an honest mechanism-and-cost paper, not a kernel or runtime paper. The absence of profiler/kernel results is acceptable only because the draft explicitly says residual correction failed the quality gate and does not claim measured speedup.

**Strengths.**

- Sections 4.1--4.2 present a systems-relevant failure mode for static protected-channel assumptions.
- Table 1 separates hard switching, EMA, coupling, reactive selection, budgeted EMA, rotation, composition, and DriftRot variants.
- ParoQuant-style rotation is positioned as a strong baseline rather than the paper's method.
- Section 5 and Appendix systems-cost table give concrete HBM/MAC/energy estimates for residual-column correction.
- Limitations state no claims for FP8, MXFP4/NVFP4, W4A4, QAT, full Qwen3 scale-up, or full runtime implementation.

**Weaknesses.**

- No measured systems performance: no vLLM/TensorRT/FlashInfer integration, custom kernel, Nsight trace, latency/throughput/VRAM table.
- Analytical envelope is coarse and device-agnostic; kernel launch, gather/scatter efficiency, batching, cache residency, tensor-core use, and scheduler interactions are not measured.
- Efficient-reasoning contribution is indirect because the paper rejects static channel protection more than it introduces a new runtime.
- Release reproducibility is partly FAST_VERIFY replay rather than full GPU rerun.

**Score:** Weak accept.

## Committee C: Methodology / Statistics

**Summary.** Statistical concerns are substantially addressed for a workshop submission. The paper is careful that budgeted EMA is model-dependent, rotation is a baseline, and the protocol is descriptive.

**Strengths.**

- Matched-control taxonomy is valuable and concrete.
- BCa CIs, no-gap fractions, and included trace counts are now visible.
- Section 5 explicitly states that no-gap handling changes the estimand across early and later packets.
- MATH-500 improves the drift claim without being overread as intervention recovery.

**Weaknesses.**

- Small N remains the main limitation; many intervention rows have 8--12 traces or fewer after filtering.
- Recovery ratio is fragile because it conditions on recoverable static gap.
- BCa/Holm details were not fully auditable from prose alone.
- The abstract's MATH-500 mention can still be misread unless readers catch that it is drift replication only.
- The protocol remains descriptive because thresholds are derived and evaluated on the same four-model study.

**Score:** Weak accept.

## Round 2 Meta-Review

The review distribution improved from Round 1. No committee reports a remaining workshop-level blocker after the Round 1 fixes. The consensus residual risks are:

1. Novelty versus ParoQuant and lack of a new positive quantizer.
2. Analytical-only systems contribution.
3. Small-N ratio-metric inference and no-gap estimand shifts.
4. Descriptive rather than held-out-validated calibration checklist.
5. Local ParoQuant-style implementation fidelity.

These are honest limitations rather than fixable paper errors without new experiments. The only safe Round 2 prose change applied was to make the E15 Holm family more auditable by pointing to the exact 33-test artifact and fields in the Reproducibility Statement.

