# ThoughtFlow-FP8 Progress

## Status

- Phase 0: partial quick pass.
- Phase 1: quick forensics pass complete.
- Phase 2: synthetic retention simulation complete; real traces still required
- Phase 4: anchor/phase retention reference plus Triton interpreter correctness
  scaffold added, but not phase-complete
- Current viability: pivot/proceed only with narrowed framing.
- Current risk: high field crowding; ThinKV already occupies much of the
  thought-adaptive quantization/eviction space, and DeepSeek V4 raises the
  production compressed-attention systems bar.

## Deliverables

### Phase 0

- Created `experimental/thoughtflow_fp8/.venv` (`Python 3.9.13`).
- `phase0/setup_complete.md`: present.
- Full repo/dataset cloning: not done in this quick Mac-only pass.

### Phase 1

- `phase1/lit_review.md`: present.
- `phase1/competitive_matrix.md`: present.
- `phase1/longflow_failure_hypothesis.md`: present.
- `phase1/v4_differentiation.md`: present.

## Phase 2 Result

`phase2/phase_eviction_analysis.md` and `.json` record a synthetic Mac-local
simulation. At the same 0.368 keep rate, ThoughtFlow preserves phase markers
with 1.000 recall, while LongFlow-like, ThinKV-like, and R-KV-like proxies retain
0.143, 0.286, and 0.286 respectively. Anchor recall stays 1.000.

Status: **ALIVE, but not reviewer-pack-ready**. The result is synthetic policy
evidence, not accuracy evidence and not a GPU systems result.

## Next Gate

Rerun the same policy simulator on real cached/current-model reasoning traces.
If real traces do not show protected-token retention gains at matched keep rate,
drop or pivot the branch.

## Macbook Kernel Correctness Scaffold

Added an anchor/phase-protected int8 quantization primitive:

- CPU reference: `phase2/reference/anchor_phase_quant.py`
- CPU reference test: `phase2/tests/test_anchor_phase_quant_reference.py`
- Triton interpreter wrapper: `phase4/kernel/anchor_phase_quant_triton.py`
- Triton interpreter test: `phase4/tests/test_anchor_phase_quant_triton_interpret.py`

Run locally:

```bash
./venv_arm64/bin/python -m pytest experimental/thoughtflow_fp8/phase2/tests
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest experimental/thoughtflow_fp8/phase4/tests -rs
```

Current Mac status: CPU reference test passes. Triton interpreter tests are
collected but skip because `triton` is not installable/importable in
`./venv_arm64` on this machine. This does not change the branch decision:
ThoughtFlow-FP8 still needs the Phase 2 trace/telemetry simulation before any
reviewer pack or GPU work.

## Log

- Initial scaffold created. No setup, downloads, literature audit, experiments,
  or tests have been verified.
- 2026-05-05: Quick Phase 0/1 forensics completed without SSH, global installs,
  GPU, large model downloads, or edits outside `experimental/thoughtflow_fp8/`.
  Primary sources found for LongFlow, Pitfalls, DeepSeek V4/SGLang, ThinKV,
  R-KV/R-KVHash, RaaS, LazyEviction, ForesightKV, and PM-KVQ. LongFlow official
  reviews were accessible through the OpenReview API and identify concrete
  weaknesses: no production E2E speedup, weak Pareto evidence versus R-KV,
  numerical/approximation concerns, and limited efficiency scaling evidence.
  Recommendation: pivot/proceed with a narrowed retrofit + bias-controlled
  retention framing; do not proceed as a generic LongFlow+FP8+phase kernel.
