# HybridKernel Progress

## Status

- Current phase: Phase 2 architecture map and runtime boundary audit complete;
  Phase 4 Triton interpreter correctness passes locally
- Phase 0: historical per-project `.venv` setup was partial; active
  reproducibility now uses repo-root `./venv_arm64` plus the source-built
  `triton-cpu` readout
- Phase 1: source/control audit is sufficient for Mac-local handoff; any deeper
  source audit is deferred until native profiling shows a real boundary signal
- Phase 3/4: boundary kernel correctness gates pass under `TRITON_INTERPRET=1`;
  non-interpreter `TRITON_CPU_BACKEND=1` passes when Homebrew GCC libraries are
  exposed through `LIBRARY_PATH`/`DYLD_LIBRARY_PATH`
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

The original per-project Phase 0 setup remains a historical partial snapshot:
the per-project requirements stack was not installed and some target configs
were gated or unavailable at the public paths tried. The active reproducibility
surface is now repo-root `./venv_arm64`; the package-index Triton route is still
unavailable on this Mac ARM64 surface, but the experimental `triton-cpu`
repository builds from source in the repo-local venv. This is complete enough
for source audit, Granite-only architecture mapping, native handoff preparation,
and Mac-local Triton interpreter correctness checks.

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

Phase 1 is a source-backed reviewer-risk audit, not proof of absence. The
current finding is that vLLM already narrows the systems story through hybrid
state layout and transfer work, but no fused attention-to-SSM layer-boundary
compute kernel was found in this pass. This is complete enough for Mac-local
handoff; further source audit should wait until native profiling shows a real
boundary signal worth implementing.

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
`TRITON_CPU_BACKEND=1`. The opt-in non-interpreter CPU backend also passes when
the Homebrew GCC library directories are provided explicitly:

```bash
HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 \
TRITON_CPU_BACKEND=1 \
TRITON_HOME="$PWD/.debug/triton_home" \
LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current/gcc/aarch64-apple-darwin23/14:/opt/homebrew/opt/gcc/lib/gcc/current${LIBRARY_PATH:+:$LIBRARY_PATH}" \
DYLD_LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py -q -rs
```

## 2026-05-06 Native Packet Control-Replay Hardening

The profiler packet checker now rejects metric rows for models that do not have
matching `profiler_driver.py` client replay logs. Same-family and cross-family
control rows therefore need their own replay evidence with the same
prefill/decode/request shape before a GPU packet can promote.

Decision: **GPU GATE STILL BLOCKS THE PAPER, BUT CONTROL ROWS MUST BE
REPLAY-BACKED**. The next exact gate remains the NVIDIA Nsight packet.

This is a kernel-logic correctness check only, not a GPU performance result and
not COLM_v3 evidence.

## 2026-05-06 Batch Replay and Triton CPU-Backend Check

After COLM-style review, the native artifact checker now treats
`prefill_tokens` as the per-sample prompt length for batch replay and requires
uniform per-sample prompt counts in the client log. Batch-size 8 replay packets
therefore no longer get rejected because their total prompt token count is
larger than the per-row prefill field.

The opt-in Triton CPU-backend correctness gate also passes locally:

```bash
HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 TRITON_CPU_BACKEND=1 \
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py -q -rs
```

Decision: **MAC-SIDE KERNEL AND PACKET REPLAY CHECKS ARE SATURATED**. The next
exact gate remains a user-operated NVIDIA/vLLM Nsight packet.

## Viability Notes

The project remains viable only if native NVIDIA/vLLM profiling finds
separable boundary-local overhead after existing hybrid serving machinery. The
Mac source/control audit is complete enough for that handoff, but it is not a
COLM systems claim and does not prove that no existing fused boundary kernel
exists.

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

Status: **WEAKLY ALIVE as a profiler-gated handoff**, but this is an
upper-bound map, not native latency evidence.

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

Current status: **WEAKLY ALIVE**. Do not build more Mac-only kernels. The
remaining gate is native NVIDIA/vLLM profiling with server-side Nsight traces,
then validation through `check_profiler_run_artifacts.py` and reduction through
`analyze_profiler_metrics.py`.

The Mac-only implementation lane is now marked as killed in
`KILLED_mac_only_kernel_iteration/`. This does not kill HybridKernel; it kills
additional local kernels before native profiling.

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

The verifier also supports `--packet-mode no_boundary_signal_kill`, which keeps
server-side Nsight Systems, client replay, reduced rows, and readout decisions
mandatory while making Nsight Compute optional when no suspicious boundary
kernel exists. This creates a clean negative path rather than forcing a fake NCU
target.

2026-05-06 hardening: metric-row artifact provenance is now path-checked.
Every `nsys_artifact` and `ncu_artifact` field must be a relative path that
stays inside the run packet, uses a valid Nsight export extension, and resolves
to a reviewable native artifact when native artifacts are required. Missing,
external, and wrong-extension row references are covered by tests.
The COLM shell now states this row-level provenance rule and the explicit
`no_boundary_signal_kill` path for reviewable negative Nsight Systems runs.

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

## 2026-05-06 Reviewer-Risk Hardening

Added three Mac-only review artifacts:

- `phase1/source_line_audit_table.md`: exact audited source/doc surfaces and
  what each does or does not rule out.
- `phase2/control_feasibility_matrix.md`: planned native controls and which
  are still GPU-only placeholders.
- `phase2/mac_reproducibility_command.md`: one stable owned-test command and
  the current CPU-backend linker caveat.

Also expanded the Triton interpreter test over 1D, 2D, 3D, block-tail,
non-contiguous, fp16, and shape-mismatch cases. This hardens the toy
correctness preflight but does not change the decision: native profiling is
still the only path to a real systems result.

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

## 2026-05-06 Triton CPU Backend Non-Interpreter Caveat

Added a public CPU-backend wrapper around the existing boundary Triton kernel
and an opt-in pytest gate:

- wrapper: `phase4/kernel/boundary_triton.py`
- test: `phase4/tests/test_boundary_triton_cpu_backend.py`

The default Phase 3/4 command still passes and skips the opt-in CPU-backend
gate:

```bash
TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1 ./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase3/tests experimental/hybridkernel/phase4/tests -rs
```

Local result after the later interpreter-test expansion: the stable owned
command passes and leaves the opt-in CPU-backend gate skipped unless requested.

Fresh non-interpreter CPU-backend attempts remain environment-fragile on this
Mac. With `/usr/bin/gcc`, Triton CPU backend shared-object linking fails with
`ld: library 'gcc' not found`. With `CC=/opt/homebrew/bin/gcc-14`, the build
reaches a different Darwin linker error, `ld: library not found for -lSystem`.

```bash
env -u TRITON_INTERPRET HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 \
  TRITON_CPU_BACKEND=1 CC=/opt/homebrew/bin/gcc-14 \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py -rs
```

Decision: **CPU-BACKEND NON-INTERPRETER EXECUTION IS OPTIONAL AND NOT A STABLE
PAPER GATE**. The stable Mac gate is `TRITON_INTERPRET=1` correctness. This is
not a speed claim, not CUDA evidence, and not a reason to add speculative
kernels. The next exact gate remains the native NVIDIA/vLLM profiler packet
with server-side Nsight Systems and Nsight Compute evidence.

## 2026-05-06 Profiler Gate Config Hardening

Addressed a reviewer-found Mac-local correctness risk in the native profiler
parser/checker. The analysis parser no longer defaults missing
`matched_non_boundary_ms` to zero or missing `recoverable_fraction` to 0.60 for
native rows. Every measured row must explicitly record matched control time,
recoverable fraction, dtype, CUDA graph state, batch/request shape, and control
segment. The summary is now grouped by the full model/config key rather than by
model alone, and reports mean, median, IQR, and deterministic bootstrap 95% CI
for recoverable-gain upper bound.

The native artifact checker now requires at least three repeated rows and three
distinct `run_id` values for the same model/config group, preventing mixed
batch-shape, dtype, graph-state, or control-condition rows from satisfying the
repeat gate. The vLLM runbook now includes the official dynamic server-capture
shape with `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`--capture-range=cudaProfilerApi`, `--capture-range-end=repeat`, and
`--profiler-config.profiler cuda`, while preserving the static server-side
capture fallback.

Decision: **MAC GATE HARDENED; STILL PENDING NATIVE PROFILER DATA**. This closes
another pre-GPU reviewer-risk item. It does not create speed evidence or change
the next exact gate: a native NVIDIA/vLLM packet with server-side Nsight traces
and repeated same-config metric rows.

## 2026-05-06 Dynamic Profiling Driver Bracket

Closed a final Mac-local handoff gap in the native profiler runbook. The
dynamic Nsight path used `--capture-range=cudaProfilerApi`, but the fixed
request driver did not previously call vLLM's server profiling endpoints. The
driver now supports `--profile-bracket`, which POSTs `/start_profile` before
the fixed replay and `/stop_profile` afterward. The stop call is issued in a
`finally` block so request-level failures still close the dynamic capture.

Added tests for dry-run endpoint reporting, normal start/completion/stop order,
and stop-after-request-error behavior:

```bash
./venv_arm64/bin/python -m pytest experimental/hybridkernel/phase2/tests/test_profiler_driver.py -q
```

Decision: **NATIVE RUNBOOK IS READY FOR USER-OPERATED GPU PROFILING**. This is
still not speed evidence. It only reduces the chance that the first NVIDIA run
captures server startup or the HTTP client instead of the vLLM serving window.

## 2026-05-06 Parser/Checklist Reviewer Hardening

Addressed the remaining Mac-side reviewer risk in the native metric-row gate.
The parser now rejects string or numeric placeholders for `cuda_graph_enabled`,
empty `control_model_or_segment` labels, empty dtype strings, and non-positive
batch-shape fields. The runbook and native packet checklist now explicitly list
the required dtype, CUDA graph state, batch shape, request count, and matched
control label fields. Added `paper/reviewer_pack.md` for a concise handoff.

Decision: **LOCAL HANDOFF HARDENED; NO FURTHER MAC BENCHMARK IS MEANINGFUL**.
The next exact gate remains a user-operated NVIDIA/vLLM server-side profiler
packet that passes the artifact checker and the 3% recoverable-gain analysis.

## 2026-05-06 Final Metric Row Reproducibility Tightening

Closed one last Mac-feasible native-packet loophole: the profiler analyzer no
longer synthesizes missing `run_id` values. Every native metric row must now
explicitly record a non-empty `model` and `run_id`, and batch-shape fields must
be JSON positive integers rather than floats, strings, booleans, or placeholders.

Decision: **STOP MAC ITERATION AFTER TESTS PASS**. Further local work cannot
replace the native NVIDIA/vLLM trace evidence required for benchmarks,
ablations, correctness under CUDA, or a real systems contribution.

## 2026-05-06 Environment Capture Handoff Hardening

Closed a final reproducibility loophole in the native packet admissibility
checker. `metadata/environment.txt` must now include `nvidia-smi`, `nsys`,
`ncu`, and `python` capture lines. Missing any of those markers is a hard
checker failure rather than a warning. The NVIDIA runbook and shared native
handoff map now state the same requirement, so a returned packet cannot be
treated as reviewable profiler evidence unless the basic GPU/profiler/Python
environment is recorded.

Decision: **MAC HANDOFF IS SATURATED**. This improves traceability for the
first NVIDIA run, but it does not change the scientific gate: only a native
NVIDIA/vLLM profiler packet can promote or kill HybridKernel.

## 2026-05-06 Native Artifact TODO-Marker Regression

Fixed a final artifact-admissibility edge case in the native packet checker.
Profiler artifact bytes are lowercased before placeholder detection, so the
`TODO_NATIVE_PROFILE_FILL` marker must also be compared in lowercase. Added a
regression test showing that a large fake Nsight export containing the uppercase
TODO marker is rejected as placeholder evidence.

Decision: **CHECKER HARDENING ONLY**. This closes a fake-artifact loophole and
raises the owned Mac suite to 141 passing tests, but it still does not provide
native profiler evidence.

## 2026-05-06 vLLM Command Scope Hardening

Tightened the native profiler packet checker so `profile_scope.json` must record
a vLLM serving command, not merely any server-side CUDA command. A packet whose
`vllm_command` omits `vllm` now fails instead of passing with a warning. Added a
regression test to prevent non-vLLM native traces from satisfying the
HybridKernel handoff gate.

Decision: **ADMISSIBILITY HARDENING ONLY**. The branch still requires native
NVIDIA/vLLM Nsight evidence before any systems claim.

## 2026-05-06 Analysis Sidecar Row Cross-Check

Closed another native-packet admissibility loophole: the artifact checker now
compares `profiler_analysis_gate.json["rows"]` against rows recomputed from the
exact `profiler_metrics.json`, rather than checking only status, decision,
summary, and row count. Added a regression test that mutates one saved analysis
row and expects checker failure.

Decision: **SIDECAR REPRODUCIBILITY HARDENED**. A returned packet now has to
show that its analysis sidecar was generated from the exact metrics file, but
the scientific gate remains native NVIDIA/vLLM profiling.

## 2026-05-06 Server/Client Log Contract Hardening

Closed the last checker/runbook mismatch found by subagent review. The native
artifact checker now requires both Nsight server profiler logs and client replay logs
under `logs/`, rather than accepting a single generic `.log` file. The
synthetic packet fixture and HybridKernel paper/reviewer pack now describe the
same contract.

Decision: **INCOMPLETE-LOG PACKETS REJECTED**. This is admissibility hardening
only. HybridKernel still cannot improve further on Mac hardware until a native
NVIDIA/vLLM packet passes the checker and the 3% recoverable-gain gate.

## 2026-05-06 Profiler-Log Name Tightening

Closed the follow-up reviewer loophole in the log contract. A generic
`server_warmup.log` plus a client replay log no longer satisfies the native
packet checker. At least one server log must be an Nsight profiler log named
with `nsys` or `ncu`, matching the GPU runbook commands, and the packet must
still include a client replay log.

Decision: **WARMUP-ONLY LOG PACKETS REJECTED**. This is still handoff
hardening, not profiling evidence. The next exact gate remains a native
NVIDIA/vLLM server-side Nsight packet.

## 2026-05-06 Final Client Replay and Map Wording Hardening

Closed the remaining client-log contract loopholes in the native packet
checker. Client replay logs must now be valid `profiler_driver.py` JSON with a
non-empty top-level `model`, explicit `dry_run: false`, non-empty requests, and
all request statuses equal to `ok`. Missing `dry_run`, dry-run logs,
nested-only model labels, and failed requests are all rejected. The phase-2
architecture map now also states the current branch correctly: source/control
audits and integration mapping are complete enough for Mac-local work, and the
next gate is native NVIDIA/vLLM profiling.

Decision: **MAC-SIDE HYBRIDKERNEL WORK IS SATURATED**. This is packet
admissibility and stale-wording cleanup only. The scientific gate remains a
native NVIDIA/vLLM server-side Nsight packet that passes the checker and clears
the repeated 3% recoverable-gain gate.

## 2026-05-06 Generator and Literature-Audit Gate Alignment

Closed a reproducibility issue in the architecture-map generator. The checked-in
`architecture_map.md` already said HybridKernel is alive only for native
NVIDIA/vLLM profiling, but rerunning `phase2/build_architecture_map.py` would
have regenerated older wording about deeper Mac source audit work. The generator
now emits the current native-profiling gate. The Phase 1 FlashInfer note also
defers deeper source audit until native profiling shows a real boundary signal.

Decision: **REGENERATION NO LONGER REINTRODUCES STALE MAC-WORK LANGUAGE**.
HybridKernel remains blocked on native NVIDIA/vLLM profiling.

## 2026-05-06 Phase 1 Gate Wording Alignment

Updated the Phase 1 literature review recommendation so it no longer presents
architecture mapping or deeper Mac source audit as the next local gate. The
Phase 2 architecture map is complete, source/control audit is sufficient for
handoff, and deeper vLLM/FlashInfer source audit is deferred until native
profiling shows a real boundary signal.

Decision: **PHASE 1 NOTES MATCH CURRENT HANDOFF**. The next exact gate remains a
native NVIDIA/vLLM packet passing the artifact checker and 3% recoverable-gain
analysis.

## 2026-05-06 Camera-Ready Review Cleanup

Applied the latest COLM-style reviewer pass to the paper and handoff docs. The
paper now includes an explicit limitations section stating that no native
NVIDIA/vLLM profile exists yet, records the stable HybridKernel Mac test result
next to Triton interpreter correctness, and removes the historical workshop
scaffold from the submitted artifact list. The README and status block now
describe repo-root `./venv_arm64` plus source-built `triton-cpu` as the active
reproducibility surface; the old per-project `.venv` setup is historical.

Decision: **NO ADDITIONAL MAC KERNEL OR BENCHMARK WORK REMAINS**. Remaining
Mac-local work is limited to doc/schema alignment and packet-readiness tests;
the next evidence gate is native server-side Nsight evidence or kill/shelve.

## 2026-05-06 Control-Role Promotion Hardening

Tightened the native profiler reducer and artifact checker after COLM-style
review. Reduced rows now carry explicit `row_role`, and the analyzer will not
emit a prototype-promotion status unless the same metric packet includes
matched same-family control and cross-family falsification rows on the same
request/runtime shape. Duplicate run IDs also cannot clear the repeated-run
gate. The paper now describes bootstrap intervals over repeated reduced rows,
not paired trace intervals, and records the stable owned Mac suite without
depending on a stale exact test count.

Decision: **PRIMARY-ONLY GPU ROWS ARE AUDIT-ONLY EVEN IF THEY CLEAR 3%**. The
next exact gate remains the user-operated 5090 Nsight packet with three distinct
repeats plus the required controls.

## 2026-05-06 Artifact-Identity and Timing Hardening

Tightened the native profiler reducer and checker again after review. The
reducer now rejects impossible rows where boundary-local or matched-control
time exceeds total step time, validates optional `time_window_ms` intervals, and
ties promotion controls to the clearing request/runtime shape instead of global
packet roles. The artifact checker now rejects repeated same-model/config rows
that reuse the same `nsys_artifact`, `ncu_artifact`, or time-window interval.
The runbook and native packet checklist now require top-level `model` in
`profile_scope.json`.

Decision: **DUPLICATED TRACE EXPORTS CANNOT PROMOTE HYBRIDKERNEL**. The next
exact gate remains a native 5090/vLLM packet with distinct repeats and matched
controls.

## 2026-05-06 Native Packet Provenance Tightening

After COLM-style review, the native GPU packet contract is stricter before any
5090 run is interpreted:

- `architecture_map.json` now records exact model IDs for generated native
  packets, so the checker can match `profile_scope.json` and metric rows against
  the copied real architecture map;
- metric rows must include `nsys_artifact_sha256`, `ncu_artifact_sha256`,
  `recoverable_fraction_basis`, and `reduction_command`;
- the checker recomputes artifact SHA-256 digests and rejects mismatches;
- client replay prompt/decode/request counts must match metric `batch_shape`
  for models present in the client logs;
- promotion now requires three same-shape same-family controls and three
  same-shape cross-family falsification rows, with both control families below
  the 3% recoverable-gain gate.

Decision: **NEXT GPU PACKET MUST BE HASHED, SHAPE-MATCHED, AND
CONTROL-FALSIFIED**. Mac-side packet mechanics are tighter; the blocker is still
native NVIDIA/vLLM profiling.

## 2026-05-07 Native Skeleton Control-Row Alignment

Updated `phase2/create_native_run_packet.py` so new GPU handoff skeletons create
the full nine-row metric shape required by the current promotion gate: three
primary hybrid repeats, three same-family controls, and three cross-family
falsification rows. The old skeleton created only primary rows, which was
audit-only even though the checker would later reject or weaken such a packet.
The README, native checklist, and top-level native handoff now state the same
nine-row minimum.

Decision: **GPU OPERATORS NOW START FROM THE CONTROL-FALSIFIED PACKET SHAPE**.
The blocker remains the user-operated NVIDIA/vLLM Nsight run.

## 2026-05-07 Native Control Matrix

Added `phase2/native_control_matrix.json` and made
`phase2/create_native_run_packet.py` copy it into each new GPU packet under
`metadata/native_control_matrix.json`. The skeleton rows now use fixed control
labels instead of placeholder control models: Granite primary boundary windows,
Granite same-model non-boundary windows, and the mapped Qwen3-Next
cross-family falsification row. The native checklist and runbook now say that a
missing control makes the packet audit-only; operators should not substitute an
unmapped model during a short GPU run.

Decision: **HYBRIDKERNEL CONTROL ROLES ARE NOW PREDECLARED BEFORE GPU TIME**.
The blocker remains native server-side Nsight evidence.

## 2026-05-07 Cross-Role Artifact-Reuse Guard

After another COLM-style packet review, the native artifact checker now rejects
reuse of the same `nsys_artifact` or `ncu_artifact` across any non-pending
metric rows, including across `primary_hybrid`, `same_family_control`, and
`cross_family_falsification` roles. The earlier guard only checked repeated
rows within one model/config group, which left a loophole where a copied
primary trace could masquerade as a control or falsification row. The runbook
workload matrix now points directly to `native_control_matrix.json` as the
single authority for row roles.

Decision: **NATIVE CONTROLS MUST BE INDEPENDENT PROFILER ARTIFACTS, NOT COPIED
ROWS**. The blocker remains the user-operated NVIDIA/vLLM Nsight packet.

## 2026-05-07 Multi-Model Profile Scope Guard

The native run-packet skeleton now writes `model_scopes` into
`metadata/profile_scope.json`, covering the Granite primary/same-family control
rows and the Qwen3-Next cross-family falsification row. The artifact checker
now rejects multi-model `profiler_metrics.json` packets unless
`profile_scope.json` explicitly covers every metric model with a vLLM command.
This prevents a Granite-only server profile from accidentally being interpreted
as evidence for a copied or separately served cross-family row.

Decision: **EVERY NATIVE METRIC MODEL NOW NEEDS AN EXPLICIT PROFILE SCOPE**.
The blocker remains the user-operated NVIDIA/vLLM Nsight packet.

## 2026-05-07 Native Matrix Enforcement

Fixed a promotion accounting bug: same-family controls are now counted by
`row_role == same_family_control` rather than by a brittle
`control_family.startswith("same_family")` prefix. This matches the
predeclared `same_model_non_boundary_segment_control` label in
`native_control_matrix.json`. The artifact checker also validates native metric
rows against the copied control matrix, rejecting off-matrix models or control
families before analysis.

Decision: **HYBRIDKERNEL PROMOTION ROWS NOW HAVE TO MATCH THE PREDECLARED
CONTROL MATRIX**. The blocker remains the user-operated NVIDIA/vLLM Nsight
packet.

## 2026-05-07 Distinct Control Repeat Guard

After another packet false-positive audit, `phase2/analyze_profiler_metrics.py`
now requires the same-family controls and cross-family falsification rows to
have at least three distinct `run_id` values before a clearing primary group can
promote. The previous analysis counted control rows but did not separately
count distinct control repeats, so copied/cloned control rows could satisfy the
row count while failing the intended three-repeat rule. A new analyzer test
locks this regression.

Decision: **HYBRIDKERNEL CONTROLS NOW NEED DISTINCT REPEATS, NOT JUST THREE
ROWS**. The blocker remains the user-operated NVIDIA/vLLM Nsight packet.
