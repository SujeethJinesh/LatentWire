# SinkAware Progress

## Status

- Phase 0 setup: partial Mac-only source-audit setup
- Phase 1 literature and code audit: quick-kill audit recorded
- Phase 2: exact static sink-prior gate failed; approximate low-rank revival
  probe completed
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

Current Mac status: CPU reference test passes. Triton interpreter tests are
collected but skip because `triton` is not installable/importable in
`./venv_arm64` on this machine. The scaffold checks exact softmax composition
for synthetic scalar values only; it does not yet prove a full attention kernel
or a GPU systems win.
