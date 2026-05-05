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

`phase2/pre_gpu_threshold_model.md` quantifies the pre-GPU threshold. At 60%
recovery, Granite 4.0 H Tiny/Small would need roughly 25% of boundary traffic
to be genuinely avoidable to clear a 3% proxy gain. Qwen3-Next would need
roughly 10.4%, but its linear-attention/Gated-DeltaNet style is less directly
matched to the Granite Mamba2 boundary-fusion idea.

Current status: **WEAKLY ALIVE**. Do not build more Mac-only kernels. The only
useful pre-NVIDIA work is profiler/runbook preparation and source-line audit of
whether a distinct boundary conversion/materialization exists.

## Profiler Analysis Gate

Added a pre-registered native-profiler parser:

- input template: `phase2/profiler_metrics_template.json`
- parser: `phase2/analyze_profiler_metrics.py`
- output: `phase2/profiler_analysis_gate.md`
- tests: `phase2/tests/test_analyze_profiler_metrics.py`

The parser computes avoidable boundary share and recoverable-gain upper bound
from repeated native summaries. Promotion requires at least three repeated runs
whose recoverable-gain upper bound clears 3%. If native summaries show less
than 1% recoverable gain, the branch should be killed or shelved. Current
output is **PENDING native profiler data**, so no speed claim is allowed.

The NVIDIA runbook now includes the exact parser input fields and the command
for turning reduced Nsight summaries into `profiler_analysis_gate.json`. It was
also tightened to profile the vLLM server process rather than only the HTTP
client driver. This makes the next GPU run reviewable: the user needs to fill
`total_step_ms`, `attention_ssm_boundary_ms`, `matched_non_boundary_ms`,
`recoverable_fraction`, and server-side trace metadata per repeated run.

## Native Artifact Review Gate

Added a reviewer-facing artifact verifier:

- checker: `phase2/check_profiler_run_artifacts.py`
- tests: `phase2/tests/test_check_profiler_run_artifacts.py`

The checker validates a future native run directory for environment metadata,
architecture-map metadata, server-side profiling scope, Nsight Systems and
Nsight Compute artifacts, profiling logs, the pre-registered readout questions,
and at least three valid metric rows for one model. It now rejects a
client-only profile scope, which closes a concrete reproducibility risk in the
previous runbook. Future GPU evidence must be both analytically reduced and
artifact-complete before the paper can cite it.

Current status remains **PENDING native profiler data**. The verifier is a gate
for admissible evidence, not a positive result.

## 2026-05-05 Local Validation Rerun

Ran the project-owned Phase 2/3/4 tests in `./venv_arm64`: 11 passed and 2
Triton interpreter tests skipped because `triton` is not importable in the
Mac-local venv. Reran the pre-GPU threshold model; Granite 4.0 H Tiny/Small
still require roughly 25% genuinely avoidable boundary traffic at 60% recovery,
and Qwen3-Next still requires 10.4%. This confirms the current decision: no more
Mac implementation, only native profiler preparation.
