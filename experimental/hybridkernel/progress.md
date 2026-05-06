# HybridKernel Progress

## Status

- Current phase: Phase 2 architecture map and runtime boundary audit complete;
  Phase 4 Triton interpreter and opt-in CPU-backend correctness now pass locally
- Phase 0: partial Mac setup complete for audit
- Phase 1: quick source-backed audit complete, deeper code audit still pending
- Phase 3/4: boundary kernel correctness gates pass under `TRITON_INTERPRET=1`
  and under an opt-in `TRITON_CPU_BACKEND=1` run with `TRITON_INTERPRET` unset
- Last updated: 2026-05-06

This scaffold now has a local environment check, small public config fetches,
and a quick primary-source audit. No external repositories were cloned, no model
weights were downloaded, no SSH/GPU work was run, and no global installs were
performed.

## Phase 0 Checklist

- [x] Create `experimental/hybridkernel/.venv` (`Python 3.9.13`)
- [ ] Install `experimental/hybridkernel/requirements.txt`
- [x] Record Python, PyTorch, CUDA, MPS, Triton import, and Triton package-index
  checks
- [x] Create local `phase0/configs/` area for config-only artifacts
- [x] Fetch or document model configs without downloading full weights
- [x] Write `phase0/setup_complete.md`

Phase 0 remains partial because the per-project requirements stack is not
installed and some target configs are gated or unavailable at the public paths
tried. The package-index Triton route is still unavailable on this Mac ARM64
`./venv_arm64` surface, but the experimental `triton-cpu` repository now builds
from source in the repo-local venv. Phase 0 is complete enough for source audit,
Granite-only architecture mapping, native handoff preparation, and Mac-local
Triton interpreter correctness checks.

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

Current Mac status: CPU reference and Triton interpreter tests pass under the
repo-local `triton-cpu` source install with `TRITON_INTERPRET=1` and
`TRITON_CPU_BACKEND=1`. An opt-in Triton CPU-backend run also matches the CPU
reference with `TRITON_CPU_BACKEND=1` and `TRITON_INTERPRET` unset when this
Mac's existing Homebrew GCC runtime paths are exposed through `LIBRARY_PATH`
and `DYLD_LIBRARY_PATH`. These are kernel-logic correctness checks only, not a
GPU performance result and not COLM_v3 evidence.

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
client-only profile scope. It also requires separate server-side scope fields
for Nsight Systems and Nsight Compute, plus three distinct repeated `run_id`
values, which closes concrete reproducibility risks from profiling only the
request client or copying one trace into multiple rows. Future GPU evidence
must be both analytically reduced and artifact-complete before the paper can
cite it.

Current status remains **PENDING native profiler data**. The verifier is a gate
for admissible evidence, not a positive result.

## Native Packet Checklist And Local Stop Decision

Added `phase2/native_run_packet_checklist.md` as the concise handoff artifact
for the NVIDIA host. It lists the exact packet files, required server-side
profile-scope JSON, valid metric-row conditions, and final analyzer/verifier
commands needed before any result is cited.

Decision: **STOP local Mac work until native profiler data exists.** The Mac
side now has the runbook, fixed-request driver, parser, verifier, tests, and
packet checklist. Further local kernels or paper claims would not improve the
decision surface. The next work must produce a native packet that passes
`check_profiler_run_artifacts.py`.

## 2026-05-05 Local Validation Rerun

At that time, the project-owned Phase 2/3/4 tests in `./venv_arm64` reported
11 passed and 2 Triton interpreter dependency skips because `triton` was not
importable in the Mac-local venv. Reran the pre-GPU threshold model; Granite 4.0 H Tiny/Small
still require roughly 25% genuinely avoidable boundary traffic at 60% recovery,
and Qwen3-Next still requires 10.4%. This confirms the current decision: no more
Mac implementation, only native profiler preparation.

## 2026-05-06 Native Packet Checker Tightening

Tightened the reviewer-facing artifact verifier so future native packets must
include `profiler_analysis_gate.json` and `.md` generated from the exact
`profiler_metrics.json` being checked. The verifier now recomputes the gate and
rejects stale or copied analysis outputs whose status, decision, summary, row
count, or Markdown status no longer match the metrics.

Added `phase2/tests/fixtures/synthetic_profiler_run_packet/` as a Mac-local
schema fixture for the packet layout. It contains placeholder Nsight files and
synthetic rows only, so it is explicitly not profiler evidence. The runbook,
packet checklist, and paper scaffold now document this stricter admissibility
gate.

Status remains **PENDING native profiler data**. HybridKernel is still blocked
on a user-operated NVIDIA/vLLM packet that passes the checker and then clears or
fails the 3% native profiler-analysis gate.

## 2026-05-06 Native Artifact Payload Tightening

Hardened `phase2/check_profiler_run_artifacts.py` so future NVIDIA packets
cannot pass the artifact gate with filename-only Nsight stand-ins. The checker
now rejects tiny native profiler exports and files whose payload contains
placeholder or skeleton markers; the default minimum matched artifact size is
1024 bytes. The synthetic fixture remains usable only for schema-only tests
when native artifact validation is explicitly disabled.

Status remains **PENDING native profiler data**. This is handoff hardening, not
profiler evidence and not a performance claim.

## 2026-05-06 Local Triton Preflight Blocker

Added `phase0/preflight_environment.py` and recorded the current local artifacts
at `phase0/local_preflight.json` and `phase0/local_preflight.md`. The preflight
uses `./venv_arm64` and only queries the active environment/index; it does not
install packages.

Original local readout before the later source build:

- PyTorch `2.6.0` imports in `./venv_arm64`.
- CUDA is unavailable (`cuda_available=false`, `cuda_device_count=0`).
- MPS is built and available.
- Triton is not importable.
- `pip index versions triton`, `triton-cpu`, and `triton-nightly` all return no
  matching distributions from this Mac ARM64 environment.

Decision at that moment: **BLOCKED_TRITON_UNAVAILABLE** for local Phase 4
completion. This was a package-index blocker, not performance evidence.

## 2026-05-06 Triton CPU Source Install

Checked the official Triton installation documentation and the experimental
`triton-lang/triton-cpu` repository. PyPI/index installs remain unavailable for
this Mac arm64 venv, but source installation now works after cloning
`triton-cpu`, initializing the `third_party/sleef` submodule, using the venv
`ninja`/`cmake`, and setting `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` to the venv
certifi bundle.

Local readout:

- installed package: `triton==3.7.0+git270e696d`
- source revision: `triton-cpu` `270e696`, `sleef` submodule `93f04d8`
- preflight status: `PASS`
- Phase 4 kernel-correctness suite: `9 passed`
- owned Phase 0--4 project suite: `103 passed, 2 warnings`

Decision: **LOCAL TRITON INTERPRETER CORRECTNESS UNBLOCKED**. This does not
change the HybridKernel performance decision: the next exact gate remains a
user-operated NVIDIA/vLLM packet that passes the native artifact checker and
the 3% profiler-analysis gate.

## 2026-05-06 Triton CPU Backend Non-Interpreter Gate

Added a public CPU-backend wrapper around the existing boundary Triton kernel
and an opt-in pytest gate:

- wrapper: `phase4/kernel/boundary_triton.py`
- test: `phase4/tests/test_boundary_triton_cpu_backend.py`

The default Phase 3/4 command still passes and skips the opt-in CPU-backend
gate:

```bash
TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1 ./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase3/tests experimental/hybridkernel/phase4/tests -rs
```

Local result: `3 passed, 1 skipped`.

The non-interpreter CPU-backend gate was then run in a fresh process with
`TRITON_INTERPRET` unset:

```bash
TRITON_CPU_BACKEND=1 HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 LIBRARY_PATH=/opt/homebrew/Cellar/gcc/14.1.0_2/lib/gcc/current/gcc/aarch64-apple-darwin23/14:/opt/homebrew/Cellar/gcc/14.1.0_2/lib/gcc/current DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/14.1.0_2/lib/gcc/current ./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py -rs
```

Local result: `1 passed`.

The first direct non-interpreter attempt without the Homebrew GCC runtime paths
failed during Triton CPU backend shared-object linking with `ld: library 'gcc'
not found`. With those existing local paths exposed, the boundary kernel
compiled and matched `phase3/reference/boundary.py` at `rtol=1e-6, atol=1e-6`.

Decision: **CPU-BACKEND CORRECTNESS ONLY UNBLOCKED ON THIS MAC**. This is not
a speed claim, not CUDA evidence, and not a reason to add speculative kernels.
The next exact gate remains the native NVIDIA/vLLM profiler packet with
server-side Nsight Systems and Nsight Compute evidence.
