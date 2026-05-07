# HANDOFF: Experimental Project State For GPU-Side Codex

Last updated: 2026-05-07

Repo: `/Users/sujeethjinesh/Desktop/LatentWire`

Branch at handoff creation: `codex/sinkaware-per-head-readiness`

Latest pushed commit before this handoff file: `aeb25cb6eea9346c0e091d2393ba8e4fb4273750`

## Executive Summary

The action surface is deliberately narrow.

| Project | Current state | Completion | GPU blocked? | Next action |
|---|---:|---:|---|---|
| HybridKernel | only live positive-method branch | ~70% if GPU gate passes; 0% as evidence without GPU | yes | run the native NVIDIA/vLLM/Nsight full-matrix profiler packet |
| ThoughtFlow-FP8 | alive as falsification-methodology paper only | ~90-93% | no | paper copyedit/human review; no new experiments |
| SSQ-LR | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |
| HORN | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |
| HBSM | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |

The positive-method path is blocked on GPU access because HybridKernel's Mac
work is saturated. The next discriminative fact is the native profiler packet.

## Non-Negotiable Operating Rules

- Do not SSH from this repo or run commands through SSH. If a remote/GPU machine
  is needed, work from a local checkout on that GPU machine, or write commands
  for the user to run.
- Use a repo-local virtual environment. On the Mac, prefer `./venv_arm64`.
  On a Linux GPU node, create a repo-local GPU venv such as `./.venv_gpu` or
  `./venv_x86_64`; do not install into the global interpreter.
- Use `.debug/` for scratch artifacts that should not be checked in.
- Commit and push at the end of a work session.
- Do not reopen SSQ-LR, HORN, or HBSM without a new preregistration written
  before any new rows are inspected.
- Do not add GPU numbers to any paper unless the corresponding artifact checker
  passes.

Primary standing instructions are in:

- `AGENTS.md`
- `experimental/README.md`
- `experimental/project_status_20260506.md`
- `experimental/native_gpu_handoff_20260506.md`

## Current Truth Sources

Read these first in any new Codex session:

1. `AGENTS.md`
2. `experimental/README.md`
3. `experimental/project_status_20260506.md`
4. `experimental/native_gpu_handoff_20260506.md`
5. `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
6. `experimental/hybridkernel/phase2/native_run_packet_checklist.md`
7. `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
8. `experimental/KILLED_ssq_lr_cross_model_transfer/README.md`
9. `experimental/KILLED_horn_directional_noise_propagation/README.md`
10. `experimental/KILLED_hbsm_sensitivity_heterogeneity/README.md`

## HybridKernel: Live Positive-Method Branch

### Status

HybridKernel is the only live positive-method branch. It is not a result yet.
The project asks whether attention/SSM boundaries in hybrid models create a
separable conversion, materialization, launch, or locality overhead that could
support a boundary-fusion systems paper.

Current completion: approximately 70% if the GPU gate passes, but 0% as
evidence without native GPU profiling.

### Key Files

Paper/reviewer context:

- `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`
- `experimental/hybridkernel/paper/hybridkernel_colm2026.tex`
- `experimental/hybridkernel/paper/reviewer_pack.md`
- `experimental/hybridkernel/README.md`
- `experimental/hybridkernel/progress.md`

GPU run documents:

- `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
- `experimental/hybridkernel/phase2/native_run_packet_checklist.md`
- `experimental/hybridkernel/phase2/native_control_matrix.json`
- `experimental/hybridkernel/phase2/reduction_worksheet_template.tsv`
- `experimental/hybridkernel/phase2/cross_family_control_replacement_template.json`

Run/validation code:

- `experimental/hybridkernel/phase2/create_native_run_packet.py`
- `experimental/hybridkernel/phase2/profiler_driver.py`
- `experimental/hybridkernel/phase2/analyze_profiler_metrics.py`
- `experimental/hybridkernel/phase2/check_profiler_run_artifacts.py`
- `experimental/hybridkernel/phase2/check_quality_smoke_artifacts.py`

Tests:

- `experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py`
- `experimental/hybridkernel/phase2/tests/test_profiler_driver.py`
- `experimental/hybridkernel/phase2/tests/test_analyze_profiler_metrics.py`
- `experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py`

### Pre-GPU Local Preflight

On the local checkout before spending GPU minutes:

```bash
cd /path/to/LatentWire
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py \
  experimental/hybridkernel/phase2/tests/test_profiler_driver.py \
  experimental/hybridkernel/phase2/tests/test_analyze_profiler_metrics.py \
  experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py \
  -q
```

Expected status from the previous hardening pass: focused HybridKernel and
related synthetic-gate tests passed. A full repo suite had unrelated top-level
LatentWire ICLR evidence-bundle failures; do not fix those while working this
experimental handoff unless the user asks.

### GPU Node Setup Summary

Follow the runbook exactly. These are the core anchors, not a substitute for
the runbook:

```bash
cd /path/to/LatentWire
python3 -m venv ./.venv_gpu
source ./.venv_gpu/bin/activate
python -m pip install --upgrade pip

export HWK_ROOT="$PWD/experimental/hybridkernel"
export GRANITE_MODEL=ibm-granite/granite-4.0-h-tiny
export QWEN_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
export PREREGISTERED_CROSS_FAMILY_MODEL=

python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model "${GRANITE_MODEL:?set GRANITE_MODEL before creating the packet}"
```

The create-packet command prints a `run_dir`. Export it:

```bash
export HWK_RUN=/path/printed/by/create_native_run_packet
```

Then fill the packet according to:

- `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
- `experimental/hybridkernel/phase2/native_run_packet_checklist.md`

### Required GPU Packet

Promotion requires a complete packet directory, not screenshots or pasted logs.
Minimum admissible packet:

- `metadata/environment.txt` and `metadata/environment.json`
- `metadata/profile_scope.json`
- `metadata/model_provenance.json`
- copied `metadata/native_control_matrix.json`
- `metadata/reduction_input_manifest.json`
- filled reduction worksheet or equivalent cited source file
- row-specific client replay logs
- server-side Nsight Systems logs and artifacts
- server-side Nsight Compute logs and artifacts unless using explicit
  no-boundary-signal kill mode
- `profiler_metrics.json`
- `profiler_analysis_gate.json`
- `profiler_analysis_gate.md`
- `artifact_check.json`

`profiler_metrics.json` must include at least nine valid reduced rows:

- three primary HybridKernel rows
- three same-shape same-family controls
- three same-shape cross-family falsification rows

Rows must have distinct run IDs/artifacts/time windows and must be tied back to
Nsight artifacts via SHA-256 hashes.

### GPU Validation Commands

After trace reduction on the GPU host:

```bash
python experimental/hybridkernel/phase2/analyze_profiler_metrics.py \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python experimental/hybridkernel/phase2/check_profiler_run_artifacts.py \
  --run-dir "$HWK_RUN" \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

Do not cite or interpret the packet until both commands pass.

### HybridKernel Decision Rule

Promote only if:

- repeated primary recoverable gain clears the preregistered `>=3%` gate;
- bootstrap/interval readout is above zero;
- same-family controls stay below the 3% gate;
- cross-family falsification rows stay below the 3% gate;
- the full artifact checker passes with `--require-full-matrix`.

Kill or shelve if:

- repeated native summaries show less than 1% recoverable gain;
- controls reproduce the same signal;
- the packet cannot be made artifact-complete;
- Qwen/cross-family substitution is done post-hoc rather than through the
  checked-in replacement template before profiling.

If the gate promotes, the next phase is prototype-kernel investigation. Any
prototype speed table must also pass:

```bash
./venv_arm64/bin/python -m experimental.hybridkernel.phase2.check_quality_smoke_artifacts \
  "$QUALITY_SMOKE_JSON" --repo-root "$PWD"
```

## ThoughtFlow-FP8: Alive Paper-Only Falsification Branch

### Status

ThoughtFlow-FP8 is alive only as a falsification-methodology paper. It is not a
positive sparse-cache or FP8 method. Completion is approximately 90-93%.

Do not run a fifth signal. Do not broaden it with SSQ-LR/HORN/HBSM kills.

### Key Files

- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`
- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
- `experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/README.md`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/manifest.json`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/falsification_table.md`
- `experimental/KILLED_thoughtflow_fp8_positive_method/README.md`

### Current Claim

ThoughtFlow contributes a repo-local registered falsification ladder for
training-free sparse-KV retention signals on reasoning traces. RDU/PSI/VWAC
successor signals failed reproduction or fresh-surface checks. The paper is
valuable as methodology, not as a positive method.

### Current Next Work

Mac-only:

- copyedit;
- venue framing;
- final human review;
- keep historical phase-doc supersession clear;
- no new experiments.

Owned test command:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

The PDF was rebuilt in the previous session after C2C/provenance clarification.

## Killed Branches

Killed means stopped with preserved audit artifacts, not deleted.

### SSQ-LR

Marker:

- `experimental/KILLED_ssq_lr_cross_model_transfer/README.md`

Source project:

- `experimental/ssq_lr/`

Primary stop manifest:

- `experimental/ssq_lr/phase2/s3_transfer_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/ssq_lr/paper/reviewer_pack.md`

Reason killed:

- frozen `mixed_int3_mxfp4_low_error_25pct` recipe on layers `0,30` failed
  no-retuning transfer to Granite 350M;
- layer-0 mixed25/INT3 rescue diagnostics also failed two-model S3.

Do not GPU-promote. Revival requires new preregistration and a fresh surface.

### HORN

Marker:

- `experimental/KILLED_horn_directional_noise_propagation/README.md`

Source project:

- `experimental/horn/`

Primary stop manifest:

- `experimental/horn/phase2/h2_noise_replay_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/horn/paper/reviewer_pack.md`

Reason killed:

- H2 directional drift ratio `1.037`;
- signed selected-direction lower bound `0.324`;
- support `0.5`;
- paired units `6/6`;
- hook-off max delta `0.0`.

Do not GPU-promote. Revival requires a new preregistered full H2/H3 scope and a
concrete reason the near-null Granite Tiny scouts should reverse.

### HBSM

Marker:

- `experimental/KILLED_hbsm_sensitivity_heterogeneity/README.md`

Source project:

- `experimental/hbsm/`

Primary stop manifest:

- `experimental/hbsm/phase2/b1_prompt2_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/hbsm/paper/reviewer_pack.md`

Reason killed:

- two-prompt B1 Fisher p `1.0`;
- boundary top-decile count `0`;
- non-boundary top-decile count `1`;
- cheap-predictor Spearman `-0.667`;
- one-prompt smoke also went wrong direction with Spearman `-0.476`.

Do not GPU-promote. Revival requires a new preregistered narrower mechanism
hypothesis and a fresh surface.

## Reviewer Upload Folder

A reviewer upload folder was created outside the repo at:

- `/Users/sujeethjinesh/Desktop/reviewer_upload_20260507`

It contains the 10 selected reviewer files. It is not tracked in git and will
not exist on a GPU node unless copied separately.

## Useful Test Commands

Focused experimental docs/code smoke used in the previous session:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m pytest \
  experimental/ssq_lr/phase2/tests/test_ssq_lr_synthetic_s1_gate.py \
  experimental/horn/phase2/tests/test_horn_synthetic_h1_gate.py \
  experimental/hbsm/phase2/tests/test_hbsm_synthetic_b1_gate.py \
  experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py \
  -q -x
```

ThoughtFlow saved artifact tests:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests/test_saved_falsification_artifacts.py \
  -q -x
```

Full repo suite from the previous session produced `1516 passed, 4 failed`.
The four failures were unrelated top-level LatentWire ICLR evidence-bundle
tests, outside the current five experimental projects. Do not burn time on them
while executing the GPU handoff unless the user asks.

## Final Checklist For The Next Codex Instance

1. Start by reading `AGENTS.md`, this `HANDOFF.md`, and
   `experimental/native_gpu_handoff_20260506.md`.
2. Confirm `git status --short` is clean.
3. If on the GPU node, run the HybridKernel runbook. Do not improvise packet
   structure.
4. Save the entire `$HWK_RUN` directory.
5. Run `analyze_profiler_metrics.py` and `check_profiler_run_artifacts.py
   --require-full-matrix`.
6. If HybridKernel passes, begin prototype-kernel planning. If it fails, write
   a clean kill/shelve note and preserve the full packet.
7. If not on GPU, polish ThoughtFlow only.
8. Do not reopen SSQ-LR/HORN/HBSM without a new preregistration.

