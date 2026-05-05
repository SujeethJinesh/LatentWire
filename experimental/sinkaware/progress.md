# SinkAware Progress

## Status

- Phase 0 setup: partial Mac-only source-audit setup
- Phase 1 literature and code audit: quick-kill audit recorded
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

Next gate: CPU-only Phase 2 reference decomposition on synthetic tensors. If the
fixed sink-token contribution cannot be reused without recomputing `QK_sink`, or
if the idea reduces to learned denominator-only sinks, kill the branch. No GPU or
large model work until that gate passes.
