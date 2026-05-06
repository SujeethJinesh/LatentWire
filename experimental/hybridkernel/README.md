# HybridKernel

Bounded experiment for a fused attention-to-SSM boundary kernel in hybrid LLMs.

This folder is self-contained inside the main LatentWire checkout. Keep local
dependencies, scratch clones, and generated artifacts under
`experimental/hybridkernel/` unless a shared repo asset is explicitly needed.

## Current Handoff Status

Local Mac work is saturated. The project has the config audit, literature/source
audit, architecture map, pre-GPU threshold model, fixed-request driver,
profiler-analysis parser, artifact verifier, synthetic packet fixture, tests,
native packet checklist, and a reproducible local environment preflight. Do not
add more local kernels, scaffolds, or paper claims until native NVIDIA profiler
data exists.

The current Mac ARM64 preflight is recorded in
`phase0/local_preflight.json` and `phase0/local_preflight.md`: PyTorch is
available with MPS, CUDA is unavailable, and `triton==3.7.0+git270e696d` is
importable from a repo-local `triton-cpu` source build. `pip index versions`
still finds no matching wheel distributions for `triton`, `triton-cpu`, or
`triton-nightly` in `./venv_arm64`; the working path is the source-install
readout in `../triton_cpu_source_install_20260506.md`. Local Phase 4 Triton
interpreter correctness is unblocked. The non-interpreter
`TRITON_CPU_BACKEND=1` path also passes on this Mac when Homebrew GCC libraries
are made explicit in `LIBRARY_PATH` and `DYLD_LIBRARY_PATH`. Treat CPU-backend
execution as optional diagnostics, not a required paper gate or GPU performance
result.

The next exact gate is a user-operated NVIDIA/vLLM packet that passes
`phase2/check_profiler_run_artifacts.py` and is then reduced by
`phase2/analyze_profiler_metrics.py`. The checker requires the saved
`profiler_analysis_gate.json`/`.md` outputs to match the metric rows being
checked. Promotion requires repeated server-side Nsight evidence of separable
attention/SSM boundary overhead clearing the 3% recoverable-gain gate. If that
evidence is absent, kill or shelve this branch.

The native artifact checker rejects generated skeleton TODOs, client-only
profile scopes, stale analyzer outputs, duplicated run IDs, and tiny or
placeholder Nsight exports. Matching profiler filenames alone are not
admissible evidence.

## Completion Estimate And Roadmap

Estimated workshop-paper completion: **70%**.

What is already complete:

- Mac-local source/runtime audit, architecture map, threshold model, fixed
  request driver, profiler packet generator, analyzer, artifact checker, toy
  Triton interpreter correctness, COLM-style draft, and reviewer pack.

What remains:

- **20%**: native NVIDIA/vLLM profiler packet with server-side Nsight Systems
  and Nsight Compute artifacts.
- **5%**: reduce repeated same-model/config traces into
  `profiler_metrics.json` and run the analyzer/checker.
- **5%**: update the paper with a promote, shelve, or kill decision.

Do not add more Mac kernels or paper claims before the native packet exists.
The consumed Mac-only implementation lane is explicitly marked in
`KILLED_mac_only_kernel_iteration/`.

### Optional Triton CPU Backend Check

Use this only for Mac-local correctness diagnostics:

```bash
HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 \
TRITON_CPU_BACKEND=1 \
TRITON_HOME="$PWD/.debug/triton_home" \
LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current/gcc/aarch64-apple-darwin23/14:/opt/homebrew/opt/gcc/lib/gcc/current${LIBRARY_PATH:+:$LIBRARY_PATH}" \
DYLD_LIBRARY_PATH="/opt/homebrew/opt/gcc/lib/gcc/current${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py -q -rs
```

## GPU-Node Quickstart

Use this only on a local NVIDIA Linux host. Do not SSH from this repo. Copy or
checkout the repository on the GPU node, then run from the repository root.

### 1. Environment

Required system tools:

- `nvidia-smi`
- `nsys`
- `ncu`
- Python 3.10 or newer
- vLLM installed from a pinned release or commit

Fast setup:

```bash
cd /path/to/LatentWire
python3 -m venv .venv_gpu
source .venv_gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experimental/hybridkernel/requirements.txt
export VLLM_VERSION=...  # fill exact release used on the GPU node
python -m pip install "vllm==${VLLM_VERSION:?set VLLM_VERSION}"

nvidia-smi
nsys --version
ncu --version
python - <<'PY'
import importlib.metadata as m
for name in ["vllm", "torch", "triton", "transformers"]:
    try:
        print(f"{name}=={m.version(name)}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
PY
```

If the GPU node already has a working vLLM environment, use that environment
instead, but record `python -m pip freeze` in the run packet.

### 2. Create The Result Packet

```bash
export HWK_ROOT="$PWD/experimental/hybridkernel"
export MODEL=ibm-granite/granite-4.0-h-tiny
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PACKET_JSON=$(python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model "$MODEL")
echo "$PACKET_JSON"
export HWK_RUN=$(python -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])' <<< "$PACKET_JSON")
```

Expected packet path:

```text
experimental/hybridkernel/phase2/profiler_runs/YYYYMMDDTHHMMSSZ_granite_boundary
```

Return the whole `$HWK_RUN` directory after validation. Do not return
screenshots or notebook snippets.

### 3. Record Metadata

```bash
{
  date -u
  hostname
  nvidia-smi
  nsys --version
  ncu --version
  python -VV
  python -m pip freeze
} | tee "$HWK_RUN/metadata/environment.txt"
```

If the exact vLLM command differs from the one in
`$HWK_RUN/metadata/profile_scope.json`, edit that JSON before validation.

### 4. Run The Smallest Admissible Profile

Start the vLLM server under Nsight Systems:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --force-overwrite=true \
  --stats=true \
  --output="$HWK_RUN/nsys/granite_tiny_b1_decode64_dynamic" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --disable-log-requests \
    --profiler-config.profiler cuda \
  2>&1 | tee "$HWK_RUN/logs/nsys_server_dynamic_b1.log"
```

In a second terminal on the same GPU node, replay fixed requests:

```bash
cd /path/to/LatentWire
source .venv_gpu/bin/activate
export HWK_ROOT="$PWD/experimental/hybridkernel"
export HWK_RUN=/path/printed/by/create_native_run_packet
export MODEL=ibm-granite/granite-4.0-h-tiny

python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 16 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  --profile-bracket \
  | tee "$HWK_RUN/logs/client_b1_profile_bracket.log"
```

If `--profiler-config.profiler cuda` is unsupported, use the static capture
path in `phase2/nvidia_vllm_profiler_runbook.md` and record the change in
`$HWK_RUN/metadata/command_notes.md`.

After Nsight Systems identifies suspicious boundary kernels, run targeted
Nsight Compute on the server process, save real `.ncu-rep` files under
`$HWK_RUN/ncu/`, and save server logs under `$HWK_RUN/logs/`.

### 5. Fill Metrics And Validate

Replace the skeleton `profiler_metrics.json` with at least three independent
rows for one model/config. Each row needs the full checklist schema from
`phase2/native_run_packet_checklist.md`: distinct `run_id`, positive
`total_step_ms`, non-negative boundary/control times, dtype, CUDA graph state,
batch shape, request count, control label, row role, control family, boundary
direction, kernel names, boundary indices, time window, reduction notes, and
relative in-packet Nsight artifact paths. Use `ncu_artifact:
"not_run_no_boundary_signal"` only when validating an explicit negative packet
with `--packet-mode no_boundary_signal_kill`.

Then run:

```bash
python "$HWK_ROOT/phase2/analyze_profiler_metrics.py" \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  | tee "$HWK_RUN/artifact_check.json"
```

Expected final packet:

- `metadata/environment.txt`
- `metadata/profile_scope.json`
- `logs/nsys_server*.log`
- `logs/client*.log`
- `nsys/*.nsys-rep`, `*.sqlite`, or `*.qdrep`
- `ncu/*.ncu-rep` unless `--packet-mode no_boundary_signal_kill` is used after
  a filled readout records no suspicious boundary kernel
- `readout.md`
- `profiler_metrics.json`
- `profiler_analysis_gate.json`
- `profiler_analysis_gate.md`
- `artifact_check.json`

### 6. Decision Rule

Promote only if the checker passes, repeated same-model/config rows clear the
3% recoverable-gain gate, and the packet includes three same-shape same-family
control rows plus three same-shape cross-family falsification rows that do not
reproduce the signal. Kill or shelve if no separable boundary overhead appears,
the mean recoverable gain is below 1%, or the signal is explained by existing
vLLM hybrid SSM layout/transfer machinery.

## Local Setup

Use the repo-local ARM64 virtual environment when available:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
source ./venv_arm64/bin/activate
python -m pytest experimental/hybridkernel/phase2/tests
```

If that environment is unavailable, create a per-project virtual environment:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
python3 -m venv experimental/hybridkernel/.venv
source experimental/hybridkernel/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experimental/hybridkernel/requirements.txt
```

Use Mac CPU/MPS for setup, literature audit, architecture mapping, and reference
tests. Do not run remote GPU work from this repo; write a local runbook for any
future 5090/H100 gate.

Record the local dependency surface before handoff:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
PIP_CACHE_DIR=.debug/pip_cache ./venv_arm64/bin/python \
  experimental/hybridkernel/phase0/preflight_environment.py \
  --check-pip-index \
  --output-json experimental/hybridkernel/phase0/local_preflight.json \
  --output-md experimental/hybridkernel/phase0/local_preflight.md
```

This command does not install packages. It only records torch/CUDA/MPS/Triton
availability and whether Triton candidates are visible from the active pip
index.

## Scope

Primary question: do hybrid attention/SSM layer transitions create enough
decode overhead to justify a fused boundary kernel?

Initial targets:

- Granite-4.0-H-Tiny and Granite-4.0-H-Small
- Nemotron-H-8B-Base
- Nemotron-3-Nano-30B-A3B
- Apriel-H1-15B-Thinker
- Qwen3-Next-80B-A3B config only

## Gates

Phase 0-4 are complete enough for a Mac-local pre-GPU handoff:

- Phase 0: active reproducibility uses repo-root `./venv_arm64`; the original
  per-project `.venv` snapshot is historical.
- Phase 1: literature/source audit is sufficient for handoff; deeper source
  audit waits for a native profiler signal.
- Phase 2: architecture map and threshold model define the native gate.
- Phase 3: CPU reference implementation matches canonical behavior.
- Phase 4: `TRITON_INTERPRET=1` kernel skeleton matches a CPU reference using
  the repo-local `triton-cpu` source install. Non-interpreter CPU backend
  execution remains optional diagnostics on this Mac.

Phase 0-4 Mac-side deliverables are now review-ready as a pre-GPU handoff, not
as performance evidence. Do not proceed with implementation until the native
packet gate clears.

Stable local reproducibility command:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests -rs
```
