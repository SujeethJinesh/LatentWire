# Decode Microkernel Consolidation Reviewer Pack

- status: Phase 1 replay gate passed; not camera-ready
- current decision: positive replay evidence, serving evidence still required
- current paper readiness: not COLM-ready

## Paper Link

- Draft TeX: `experimental/decode_microkernel/paper/decode_microkernel_colm2026.tex`
- Draft PDF: `experimental/decode_microkernel/paper/decode_microkernel_colm2026.pdf`

## Current Claim

Decode Microkernel Consolidation packs trace-derived repeated
gating/routing/state-update replay operations into a single Triton kernel. The
only supported claim is replay-level: GPU-side replay latency falls while saved
outputs remain numerically equivalent. There is no serving-speedup claim yet.

## Strongest Evidence

| Gate | Result | Decision |
|---|---:|---|
| Phase 0 trace screen | 8 admitted rows, dense launch-heavy decode kernels across primary, same-family, and cross-family roles | PASS |
| Phase 1 replay gate | 9 paired rows, 3 per role, 100 warmups and 1000 measured CUDA-event samples per row | PASS_DMC_PHASE1_CONSOLIDATED_REPLAY |
| Primary median replay latency reduction | 0.9650953811812909, 95% CI low 0.9618806519742926 | clears 0.08 threshold |
| Same-family median replay latency reduction | 0.9657258621103347, 95% CI low 0.9656092336667471 | clears 0.08 threshold |
| Cross-family median replay latency reduction | 0.9655504614632344, 95% CI low 0.9654552838317912 | clears 0.05 threshold |
| Minimum launch reduction | 0.9916666666666667 | clears 0.25 threshold |
| Max absolute / relative error | 7.152557373046875e-07 / 8.915998250813573e-07 | clears 1e-2 thresholds |

## Artifact Paths

- Phase 1 preregistration: `experimental/decode_microkernel/phase1/preregister_dmc_phase1.md`
- Phase 1 result packet: `experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z`
- Checker output: `experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z/checker_result.json`
- Metrics: `experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z/metrics.json`
- Replay schedule: `experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z/replay_schedule.json`
- Committee review: `experimental/decode_microkernel/paper/committee_reviews/20260508_phase1_packet.md`
- Phase 2 preregistration: `experimental/decode_microkernel/phase2/preregister_dmc_phase2.md`

## Reviewer Risks

- Phase 1 is trace-derived replay evidence, not vLLM serving evidence.
- The 0.965 replay latency reduction must not be framed as end-to-end speedup.
- Bootstrap intervals over 3 rows per role are not paper-strength uncertainty.
- Existing serving systems may already eliminate part of the measured launch
  overhead.
- Phase 2 must show generated-token equivalence and serving latency gains on
  frozen prompt slices before camera-ready claims are allowed.

## Next Exact Gate

Run `decode_microkernel_phase2` from `swarm/queue.yml`. The Phase 2 checker must
return `PASS_DMC_PHASE2_SERVING_GAIN` before any paper-level serving
acceleration claim is allowed.
