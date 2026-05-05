# HybridKernel Progress

## Status

- Current phase: Phase 2 architecture map and runtime boundary audit complete;
  Phase 3/4 scaffolds present but not phase-complete
- Phase 0: partial Mac setup complete for audit
- Phase 1: quick source-backed audit complete, deeper code audit still pending
- Phase 3/4: interpreter-mode boundary kernel scaffold added for correctness
  gates, but not phase-complete
- Last updated: 2026-05-05

This scaffold now has a local environment check, small public config fetches,
and a quick primary-source audit. No external repositories were cloned, no model
weights were downloaded, no SSH/GPU work was run, and no global installs were
performed.

## Phase 0 Checklist

- [x] Create `experimental/hybridkernel/.venv` (`Python 3.9.13`)
- [ ] Install `experimental/hybridkernel/requirements.txt`
- [x] Record Python, PyTorch, and Triton import checks
- [x] Create local `phase0/configs/` area for config-only artifacts
- [x] Fetch or document model configs without downloading full weights
- [x] Write `phase0/setup_complete.md`

Phase 0 remains partial because the requirements stack is not installed and
some target configs are gated or unavailable at the public paths tried. It is
complete enough for source audit and Granite-only architecture mapping.

## Phase 1 Checklist

- [x] Audit Mamba-3 paper/repo entry points
- [x] Audit Bamba v2 docs/blog at source level
- [x] Audit Granite-4.0 architecture docs and local configs
- [x] Audit Nemotron-H report and vLLM hybrid serving docs
- [ ] Audit Apriel-H1-15B-Thinker report
- [x] Audit Qwen3-Next-80B-A3B public model-card/blog/config
- [ ] Audit Hymba
- [x] Audit vLLM hybrid model support docs/RFC/blog
- [x] Audit FlashInfer public README surface
- [x] Search for fused attention/SSM boundary kernels
- [x] Write `phase1/lit_review.md`

Phase 1 is a quick audit, not a final line-by-line source audit. The current
finding is that vLLM already narrows the systems story through hybrid state
layout and transfer work, but no fused attention-to-SSM layer-boundary compute
kernel was found in this pass.

## Macbook Kernel Correctness Scaffold

Added a minimal attention/SSM boundary blend primitive:

- CPU reference: `phase3/reference/boundary.py`
- CPU reference test: `phase3/tests/test_boundary_reference.py`
- Triton interpreter wrapper: `phase4/kernel/boundary_triton.py`
- Triton interpreter test: `phase4/tests/test_boundary_triton_interpret.py`

Run locally:

```bash
./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase3/tests
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase4/tests -rs
```

Current Mac status: CPU reference test passes. Triton interpreter tests are
collected but skip because `triton` is not installable/importable in
`./venv_arm64` on this machine. This is a correctness scaffold, not a GPU
performance result and not COLM_v3 evidence.

## Viability Notes

The project remains viable only if a deeper Phase 1 source audit finds no
existing fused boundary kernel and Phase 2 estimates a meaningful transition
overhead. The quick audit supports a cautious Granite-focused proceed, but not a
COLM_v3 systems claim.

## Risks

- Existing production or paper implementation may already fuse the same boundary.
- The transition overhead may be too small on modern GPUs to support a systems
  contribution.
- Mac-only phases can validate setup, references, and semantics, but cannot
  provide GPU performance evidence.

## Phase 2 Result

`phase2/architecture_map.md` and `phase2/architecture_map.json` were generated
from the local Granite 4.0 H Tiny/Small and Qwen3-Next configs. The activation
stream upper-bound estimate clears the >=3% theoretical gate for all three
configs:

- Granite 4.0 H Small: 8 boundaries, 131072 boundary bytes/token, 12.0%
  recovered activation-stream fraction at 60% recovery.
- Granite 4.0 H Tiny: 8 boundaries, 49152 boundary bytes/token, 12.0%
  recovered activation-stream fraction at 60% recovery.
- Qwen3-Next-80B-A3B: 23 inferred boundaries, 188416 boundary bytes/token,
  28.7% recovered activation-stream fraction at 60% recovery.

Status: **ALIVE as a systems spinout**, but this is an upper-bound map, not
native latency evidence.

## Next Gate

`phase2/runtime_boundary_audit.md` records the deeper runtime audit decision:
**ALIVE BUT WEAKENED**. vLLM's current hybrid SSM disaggregated-serving path
already handles HMA shared tensors, dual descriptor views, DS conv layout, and
3-descriptor conv transfer without staging buffers or reshuffling. The
architecture-map bytes are therefore not enough to justify more Mac-only
implementation.

The next gate is native profiler evidence, not more local code: look for a real
attention/SSM boundary conversion, launch, or materialization overhead of at
least 3% end-to-end before implementing a fused boundary kernel.
