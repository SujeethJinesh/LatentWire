# SinkAware Progress

## Status

- Phase 0 setup: not started
- Phase 1 literature and code audit: not started
- Last updated: 2026-05-05

## Phase 0 Checklist

- [ ] Create `experimental/sinkaware/.venv`
- [ ] Install `requirements.txt`
- [ ] Create local ignored directories for external repos and artifacts
- [ ] Record setup verification in `phase0/setup_complete.md`

Phase 0 is not complete until all checklist items are verified locally and the
deliverable exists.

## Phase 1 Checklist

- [ ] Audit FlashInfer sink/static-mask handling
- [ ] Audit FlashAttention 3 sink/static-mask handling
- [ ] Audit StreamingLLM sink handling
- [ ] Audit DeepSeek DSA / FlashMLA handling of early positions
- [ ] Audit NSA / block-sparse attention handling of early positions
- [ ] Audit BLASST and Block-Sparse Flash Attention if public code is available
- [ ] Audit GPT-OSS-20B reference attention pattern handling
- [ ] Record file paths, line numbers, and comparison table in `phase1/lit_review.md`

Phase 1 is not complete until the audit is source-backed and the kill criterion
has been explicitly checked.

## Current Assessment

Viability is unverified. The wedge is plausible only if existing kernels still
compute scores or use generic masks for fixed sink positions rather than a
dedicated precomputed-prior path.

Main risk: the claimed novelty may already exist inside a generic sparse or mask
kernel path, or sink mass may depend too strongly on query content to be treated
as static.

Next gate: complete Phase 0 setup, then perform the Phase 1 source audit before
any math, implementation, or GPU benchmarking work.

