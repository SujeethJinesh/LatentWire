# SinkAware Progress

## Status

- Phase 0 setup: partial Mac-only source-audit setup
- Phase 1 literature and code audit: quick-kill audit recorded
- Phase 2: exact static sink-prior gate failed; approximate low-rank revival
  and per-head softmax/output gates completed with a new paired layer-head
  caveat
- Phase 4: fixed sink-token decomposition reference plus Triton interpreter
  correctness scaffold added, but not phase-complete
- Last updated: 2026-05-05

## Phase 0 Checklist

- [x] Create `experimental/sinkaware/.venv` (`Python 3.9.13`)
- [ ] Install `requirements.txt`
- [x] Create local ignored directories for external repos and artifacts
- [x] Record partial setup verification in `phase0/setup_partial.md`
- [ ] Record full setup verification in `phase0/setup_complete.md`

Phase 0 is not complete until all checklist items are verified locally and the
deliverable exists.

## Phase 1 Checklist

- [x] Audit FlashInfer sink/static-mask handling
- [x] Audit FlashAttention sink/static-mask handling
- [x] Audit StreamingLLM sink handling
- [x] Audit DeepSeek DSA / FlashMLA handling of early positions
- [x] Audit NSA / block-sparse attention handling of early positions
- [ ] Audit BLASST and Block-Sparse Flash Attention if public code is available
- [x] Audit GPT-OSS reference attention pattern handling
- [x] Record file paths, line numbers, and comparison table in `phase1/lit_review.md`

Phase 1 is not complete until the audit is source-backed and the kill criterion
has been explicitly checked.

## Current Assessment

Quick-kill criterion was not triggered for fixed-position BOS/sink KV tokens:
the audited sources did not show an existing `output += sink_bias_precomputed`
path that skips fixed sink-token score computation.

Main risk is now sharper: FlashInfer, FlashMLA, and GPT-OSS already implement
learned/per-head attention sink terms in the softmax denominator. Broad
"sink-aware attention kernel" novelty is therefore occupied. The only remaining
wedge is fixed early-position/BOS K/V decomposition or precomputation.

## Phase 2 Result

`phase2/decomposition_decision.md` records the decision: **KILL as an exact
static sink-prior kernel**. The counterexample test shows fixed sink K/V cannot
be reused exactly while skipping per-query `QK_sink`, because the sink softmax
logits remain query-dependent.

What remains possible is a pivot to approximate/learned/low-rank sink priors or
a small fused path that still computes `QK_sink`. The original exact static-prior
claim should not proceed to GPU work.

## Approximate Revival Gate

`phase2/sink_predictability_probe.md` checks the weaker approximate question:
can fixed-sink logits be predicted from a tiny low-rank query representation
under favorable query geometry?

Result: **REVIVE only as approximate low-rank/clustered query prior**. Static
R2 remains near zero across synthetic cases, so the exact static-prior branch is
still dead. Rank-4 query features recover low-rank synthetic sink logits
(`R2=0.999`), and rank-8 features recover clustered synthetic sink logits
(`R2=0.976`), while random queries remain poor (`rank8 R2=0.102`).

Next gate: use real Q/K tensors or attention telemetry. Synthetic geometry is
only a reason to keep the approximate branch alive, not a reviewer-pack result.

`phase2/real_query_sink_probe.md` and
`phase2/real_qk_sink_logit_probe.md` then move the branch to real
distilgpt2 traces. Hidden-query features predict attention sink mass and
Q/K sink logits better than position-only structure. The QK-logit probe reaches
rank-8 hidden+position `R2=0.712` versus position-only `R2=0.153`.

`phase2/qk_sink_cost_model.md` keeps the systems claim narrow: rank-2 is the
only current compromise that has nontrivial QK-logit predictability while
staying below exact four-sink QK cost (`0.531x` estimated multiply-adds,
`R2=0.420`). Rank-8 is more accurate but too expensive under this simple model.

`phase2/real_qk_sink_softmax_output_probe.md` runs the next Mac-local quality
gate. It keeps all non-sink QK scores exact and replaces only fixed sink-token
logits with per-head predictors on held-out distilgpt2 trace tokens. Mean
across layers still favors rank-2 over position-only (`output rel-L2=0.141`
versus `0.170`, `sink-mass MAE=0.055` versus `0.076`). The new paired
layer-head readout is weaker: rank-2's output rel-L2 improvement over
position-only is `+0.0297 +/- 0.0378` across 72 layer-head cells, with only
20/72 output wins. Rank-8 is more accurate (`output rel-L2=0.107`) but remains
too expensive under the current simple cost model.

Current status: **WEAKLY ALIVE for a narrow GPU/interpreter gate as an
approximate low-rank SinkAware branch**, not as exact static-prior reuse. The
rank-2 aggregate improvement is real in the saved probe, but per-head wins are
concentrated enough that seed/split repeats or head-selective gating should be
cleared before any strong paper claim. The next exact gate remains a correctness
prototype comparing exact attention, exact fixed-sink decomposition that still
computes `QK_sink`, and rank-2 approximate sink-logit prediction; native speed
claims require NVIDIA hardware.

## Head-Selective Rank-2 Gate

`phase2/head_selective_sink_gate.md` tested the obvious Mac-local rescue for
the fragile per-head result: fit predictors on a train split, select rank-2
heads on validation when rank-2 beats position-only on output rel-L2, then
evaluate the mixed policy on a held-out split. The rule selected 19/72 heads
but failed held-out: validation-selected rank-2 had output rel-L2 `0.2035`,
worse than position-only `0.1724` and all-rank2 `0.1419`.

Decision: simple validation head selection is **weakened/ruled out** as a
pre-GPU rescue. The approximate branch remains alive only as all-rank2 plus a
future stability mechanism; no paper should claim head-selective robustness.

## Macbook Kernel Correctness Scaffold

Added a scalar fixed sink-token decomposition primitive:

- CPU reference: `phase2/reference/sink_decomposition.py`
- CPU reference test: `phase2/tests/test_sink_decomposition_reference.py`
- Triton interpreter wrapper: `phase4/kernel/sink_decomposition_triton.py`
- Triton interpreter test: `phase4/tests/test_sink_decomposition_triton_interpret.py`

Run locally:

```bash
./venv_arm64/bin/python -m pytest experimental/sinkaware/phase2/tests
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest experimental/sinkaware/phase4/tests -rs
```

Current Mac status: CPU reference test passes. Triton interpreter execution
tests are collected but skip because `triton` is not importable in
`./venv_arm64` on this machine. The scaffold now also has a non-skipped
readiness test that reports `ready=false`, `reason="triton is not importable"`,
`TRITON_INTERPRET=1`, and `torch_cuda_available=false`. The scaffold checks
exact softmax composition for synthetic scalar values only; it does not yet
prove a full attention kernel or a GPU systems win.

## Approximate Attention Reference Gate

Added a Phase 3 reference for the live approximate branch:

- CPU reference: `phase3/reference/approx_sink_attention.py`
- tests: `phase3/tests/test_approx_sink_attention_reference.py`
- note: `phase3/approx_sink_attention_reference.md`

This reference keeps all non-sink tail logits exact, replaces only fixed-sink
logits with a predictor, and then runs the normal softmax denominator. Tests
verify that exact sink logits reproduce exact attention. This is the operator a
future Triton/CUDA kernel must match before native timing is meaningful.

## Approximate Triton Interpreter Scaffold

Added a Phase 4 Triton-interpreter scaffold for the live approximate operator:

- wrapper: `phase4/kernel/approx_sink_attention_triton.py`
- tests: `phase4/tests/test_approx_sink_attention_triton_interpret.py`
- note: `phase4/approx_sink_attention_triton_gate.md`

The current `./venv_arm64` does not have `triton`, so the tests use
`pytest.importorskip("triton")` and skip locally. The correctness contract is
now ready for a Macbook `TRITON_INTERPRET=1` run once Triton is installed in the
repo-local venv; native GPU timing should still wait until that interpreter
gate passes.
