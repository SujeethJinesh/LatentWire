# HybridKernel Progress

## Status

- Current phase: Phase 0 setup
- Phase 0: started
- Phase 1: not started
- Last updated: 2026-05-05

This scaffold was created from the experimental brief only. No environment,
external repositories, model configs, literature claims, or tests have been
verified yet.

## Phase 0 Checklist

- [x] Create `experimental/hybridkernel/.venv` (`Python 3.9.13`)
- [ ] Install `experimental/hybridkernel/requirements.txt`
- [ ] Record Python, PyTorch, and Triton import checks
- [ ] Create local `external/` reference clone area if needed
- [ ] Fetch or document model configs without downloading full weights
- [ ] Write `phase0/setup_complete.md`

## Phase 1 Checklist

- [ ] Audit Mamba-3 paper and kernel code
- [ ] Audit Bamba v2 paper and reference implementation
- [ ] Audit Granite-4.0 architecture report
- [ ] Audit Nemotron-H and Nemotron-3 Nano reports/code
- [ ] Audit Apriel-H1-15B-Thinker report
- [ ] Audit Qwen3-Next-80B-A3B report
- [ ] Audit Hymba
- [ ] Audit vLLM hybrid model support and related PRs
- [ ] Audit FlashInfer Mamba integration
- [ ] Search for fused attention/SSM boundary kernels
- [ ] Write `phase1/lit_review.md`

## Viability Notes

The project remains viable only if Phase 1 finds no existing fused boundary
kernel and Phase 2 estimates a meaningful transition overhead. The current
scaffold does not establish either condition.

## Risks

- Existing production or paper implementation may already fuse the same boundary.
- The transition overhead may be too small on modern GPUs to support a systems
  contribution.
- Mac-only phases can validate setup, references, and semantics, but cannot
  provide GPU performance evidence.

## Next Gate

Complete Phase 0 locally, then run the Phase 1 literature/code audit before
writing any kernel code.
