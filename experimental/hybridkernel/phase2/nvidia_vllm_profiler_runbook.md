# HybridKernel NVIDIA/vLLM Profiler Runbook

- date: 2026-05-07
- status: current pre-GPU runbook; no NVIDIA run has been executed from this repo
- scope: local instructions for a user-operated NVIDIA host; no SSH required

## Gate

HybridKernel is **weakly alive** only as a profiler-driven systems branch. The
Mac-local evidence says the activation stream is large enough to inspect, but
the runtime audit found that vLLM already implements important hybrid SSM
state-layout and transfer machinery. The next gate is therefore not another
kernel scaffold. It is a native profiler trace that answers:

> Do attention/SSM layer boundaries create a distinct conversion,
> materialization, launch, or locality overhead of at least 3% end-to-end, or a
> larger localized overhead with a credible route to a 3% end-to-end gain?

Kill the branch if this run finds no separable boundary overhead after matching
for sequence length, batch shape, CUDA graph behavior, quantization, and model
family. Treat the first native matrix as a prototype/no-boundary decision, not
as a final paper speed claim: a paper-level throughput claim still needs the
broader pure/mostly-Transformer and mostly-SSM controls listed in
`control_feasibility_matrix.md`.

## Operator Preflight Before GPU Rental

Run this on the Mac/local checkout before spending GPU minutes. It exercises the
packet skeleton, fixed-request driver, reducer, and artifact checker without
claiming native performance evidence:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py \
  experimental/hybridkernel/phase2/tests/test_profiler_driver.py \
  experimental/hybridkernel/phase2/tests/test_analyze_profiler_metrics.py \
  experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py \
  -q
```

## Source Context

| Source | Why it matters for the run |
|---|---|
| vLLM profiling docs: <https://docs.vllm.ai/en/stable/contributing/profiling/> | vLLM recommends Nsight Systems for developer profiling and documents flags including `--trace-fork-before-exec=true` and `--cuda-graph-trace=node`. |
| vLLM hybrid SSM disaggregated serving blog: <https://vllm.ai/blog/hybrid-ssm-disagg> | vLLM describes HMA shared tensors, dual descriptor views, DS conv layout, and no extra buffers/reshuffling for hybrid SSM transfer, so HybridKernel must not claim those wins. |
| NVIDIA Nsight Systems get-started page: <https://developer.nvidia.com/nsight-systems/get-started> | Confirms Nsight Systems CLI is the correct timeline tool for CUDA launch/kernel sequencing. |
| NVIDIA Nsight Compute CLI docs: <https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html> | Confirms `ncu` is the non-interactive per-kernel profiler for hardware counters. |

## Required Machine

Run on a local NVIDIA Linux host with:

- one or more recent NVIDIA GPUs with enough VRAM for the chosen model;
- recent NVIDIA driver, CUDA runtime, Nsight Systems, and Nsight Compute;
- Python virtual environment local to this checkout or to a copied benchmark
  checkout;
- vLLM installed from a pinned commit or release, recorded in the run log;
- no SSH invocation from this repository.

Recommended first GPU target: Granite 4.0 H Tiny or Small if supported by the
local vLLM build. Use Qwen3-Next only as a secondary probe because its
linear-attention/Gated-DeltaNet boundary is less directly matched to the
Granite Mamba2 boundary-fusion hypothesis.

## Artifact Layout

Create all generated artifacts under the local HybridKernel tree on the NVIDIA
host:

```bash
export HWK_ROOT=/path/to/LatentWire/experimental/hybridkernel
export GRANITE_MODEL=ibm-granite/granite-4.0-h-tiny
export QWEN_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
export PREREGISTERED_CROSS_FAMILY_MODEL=
python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model "${GRANITE_MODEL:?set GRANITE_MODEL before creating the packet}"
```

The command prints a `run_dir`. Export that exact path as `HWK_RUN` before
running the profiling commands below. The generated skeleton is deliberately
not admissible evidence: the artifact checker rejects the
`TODO_NATIVE_PROFILE_FILL` sentinels until real native profiler metadata,
readout entries, and metric rows replace them.

The skeleton also copies `phase2/native_control_matrix.json` into
`$HWK_RUN/metadata/native_control_matrix.json`. Treat that file as the
row-role authority: primary and same-family rows use the Granite serving path,
and cross-family falsification uses the mapped Qwen3-Next row if it can be run.
If the cross-family model is unavailable on the GPU node, record the miss and
keep the packet audit-only; do not substitute an unmapped model and promote.

Do not replace Nsight exports with empty files, copied README files, or text
placeholders that only satisfy the expected filename extension. The final
checker requires reviewable profiler payloads, with default minimum artifact
size of 1024 bytes and no skeleton placeholder markers.
It also fails the packet if `metadata/environment.txt` omits `nvidia-smi`,
`nsys`, `ncu`, or `python` capture lines, or if the parseable
`metadata/environment.json` does not include installed `vllm`, `torch`,
`triton`, and `transformers` versions.

The generated packet also includes `metadata/model_provenance.json`. Fill it
before profiling with exact model and tokenizer revisions for every served
model in the metric rows, including the vLLM served model string, cache source,
immutable-revision attestations, `local_files_only`, and `trust_remote_code`
values. The checker rejects a packet whose model provenance does not cover the
profiler metrics and client replay logs, or whose model/tokenizer revision
attestations are missing or false.
Mutable aliases such as `main`, `master`, `HEAD`, `latest`, and `refs/heads/*`
are rejected even when the immutable attestation is set.
Use this schema:

```json
{
  "provenance_version": "hybridkernel_model_provenance_v1",
  "models": [
    {
      "model_id": "ibm-granite/granite-4.0-h-tiny",
      "served_model_id": "ibm-granite/granite-4.0-h-tiny",
      "model_revision": "<resolved git commit or immutable snapshot>",
      "tokenizer_revision": "<resolved git commit or immutable snapshot>",
      "model_revision_is_immutable": true,
      "tokenizer_revision_is_immutable": true,
      "cache_source": "<HF cache path, mounted volume, or download source>",
      "snapshot_manifest_path": "metadata/granite_snapshot_manifest.json",
      "snapshot_manifest_sha256": "sha256:<64 lowercase hex chars>",
      "local_files_only": false,
      "trust_remote_code": true
    }
  ]
}
```

Record immutable metadata before profiling:

```bash
{
  date -u
  hostname
  nvidia-smi
  nsys --version
  ncu --version
  python -VV
  python -m pip freeze
  python - <<'PY'
import importlib.metadata as m
for name in ["vllm", "torch", "triton", "transformers"]:
    try:
        print(f"{name}=={m.version(name)}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
PY
} | tee "$HWK_RUN/metadata/environment.txt"

python - <<'PY'
import datetime as dt, importlib.metadata as m, json, pathlib, platform, subprocess, sys, os

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

payload = {
    "environment_version": "hybridkernel_environment_v1",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": platform.node(),
    "nvidia_smi": run(["nvidia-smi"]),
    "nsys_version": run(["nsys", "--version"]),
    "ncu_version": run(["ncu", "--version"]),
    "python_version": sys.version.replace("\n", " "),
    "packages": {name: m.version(name) for name in ["vllm", "torch", "triton", "transformers"]},
}
out = pathlib.Path(os.environ["HWK_RUN"]) / "metadata/environment.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
PY
```

For each served model, write a lightweight snapshot manifest with config and
tokenizer metadata hashes before profiling and record its SHA in
`metadata/model_provenance.json`:

```bash
python - <<'PY'
import hashlib, json, pathlib, os
snapshot_dir = pathlib.Path(os.environ["MODEL_SNAPSHOT_DIR"])
rows = []
for pattern in ["config*.json", "tokenizer*.json", "generation_config.json", "special_tokens_map.json"]:
    for path in sorted(snapshot_dir.glob(pattern)):
        if path.is_file():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"path": str(path.relative_to(snapshot_dir)), "sha256": "sha256:" + h})
if not rows:
    raise SystemExit("no config/tokenizer metadata files found")
out = pathlib.Path(os.environ["HWK_RUN"]) / "metadata/granite_snapshot_manifest.json"
out.write_text(json.dumps({"files": rows}, indent=2) + "\n")
print("sha256:" + hashlib.sha256(out.read_bytes()).hexdigest())
PY
```

Run this GPU-node preflight before the first Nsight trace. Save both stdout and
stderr. If any line fails, stop and record the failure as an audit-only packet
rather than spending the trace budget.

```bash
{
  echo "## tokenizer/model cache"
  python - <<'PY'
from transformers import AutoTokenizer
import os
for env_name in ["GRANITE_MODEL", "QWEN_MODEL"]:
    model = os.environ[env_name]
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    ids = tok.encode("HybridKernel token-count preflight.", add_special_tokens=False)
    print(env_name, model, "tokenizer_ok", len(ids))
PY
  echo "## vllm import and profiler flag"
  python - <<'PY'
import importlib.metadata as m
print("vllm", m.version("vllm"))
PY
  python -m vllm.entrypoints.openai.api_server --help | grep -E 'profiler-config|disable-log-requests|max-model-len' || true
  echo "## vram"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
} | tee "$HWK_RUN/metadata/gpu_node_preflight.txt"
```

If Qwen3-Next cannot load on the selected 5090, record that in
`metadata/command_notes.md` and keep the packet audit-only. A Granite-only
packet can inform a kill/prototype decision, but it cannot satisfy the frozen
full-matrix promotion gate.
If a smaller cross-family hybrid control is available, fill
`experimental/hybridkernel/phase2/cross_family_control_replacement_template.json`
before any profiling starts and copy the filled row into
`metadata/native_control_matrix.json`. A replacement chosen after seeing
Granite rows is not admissible, and a packet without a preregistered
cross-family hybrid row remains audit-only.

The packet generator already creates
`metadata/reduction_input_manifest.json` with one row per expected metric row.
Do not overwrite it with a one-row TODO manifest. Before reducing any timeline
windows, fill every generated row so each `profiler_metrics.json` row is
traceable to exact Nsight exports, source time windows, commands, reducer source
paths, and reducer source hashes. If a row is reduced manually, record the
manual worksheet path inside the packet as `reduction_source_path` and the
worksheet SHA-256 as `reduction_script_sha256`; do not leave analyst-selected
windows unmanifested.

Use a row worksheet for every metric row before editing `profiler_metrics.json`.
The worksheet can be a Markdown file or TSV, but it must be cited by SHA-256 in
`metadata/reduction_input_manifest.json`. Minimum columns:

```text
run_id	row_role	model	control_segment	nsys_sqlite	nsys_export_sha256	window_start_ms	window_end_ms	total_step_ms	boundary_ms	matched_non_boundary_ms	ncu_rep	ncu_sha256	ncu_kernel_regex	ncu_launch_skip	ncu_launch_count	reducer_command	notes
```

A TSV worksheet template is checked in at
`experimental/hybridkernel/phase2/reduction_worksheet_template.tsv`. Copy it
into the run packet, fill one row per independent metric row, and cite its
SHA-256 in `metadata/reduction_input_manifest.json`.

Do not reuse a worksheet row across repeats. `total_step_ms`, `boundary_ms`,
and `matched_non_boundary_ms` must come from the same server-side trace window
and must cite the exact time interval used to compute them.

Record what process the profiler actually observes. A client-only profile is
not admissible evidence for HybridKernel because the CUDA work lives in the
vLLM server process:

```bash
cat > "$HWK_RUN/metadata/profile_scope.json" <<JSON
{
  "profiled_process": "vllm_server",
  "nsys_profiled_process": "vllm_server",
  "ncu_profiled_process": "vllm_server",
  "trace_scope": "server-side CUDA kernels under fixed request replay",
  "nsys_trace_scope": "server-side CUDA kernels under fixed request replay",
  "ncu_trace_scope": "server-side CUDA kernels under suspicious-kernel replay",
  "request_driver_process": "profiler_driver_http_client",
  "model": "${GRANITE_MODEL:?set GRANITE_MODEL}",
  "vllm_command": "python -m vllm.entrypoints.openai.api_server --model ${GRANITE_MODEL:?set GRANITE_MODEL} --dtype bfloat16 --max-model-len 2048 --disable-log-requests",
  "model_scopes": [
    {
      "row_roles": ["primary_hybrid", "same_family_control"],
      "model": "${GRANITE_MODEL:?set GRANITE_MODEL}",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model ${GRANITE_MODEL:?set GRANITE_MODEL} --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    },
    {
      "row_roles": ["cross_family_falsification"],
      "model": "${QWEN_MODEL:?set QWEN_MODEL}",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model ${QWEN_MODEL:?set QWEN_MODEL} --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    }
  ]
}
JSON
```

If you cannot run the cross-family model, leave the metric rows absent and
record the missing control in `readout.md`; the packet is then audit-only. Do
not keep a Qwen metric row without a matching `model_scopes` entry and client
replay log. If you preregister a smaller cross-family replacement, update this
`profile_scope.json` block before profiling: replace the cross-family
`model_scopes` entry, use the replacement vLLM command, and make sure the
client JSON logs and metric `model` fields use the replacement model ID.

## Workload Matrix

Keep the first run small and discriminative. The authority for row roles is
`experimental/hybridkernel/phase2/native_control_matrix.json`; copy it into
`$HWK_RUN/metadata/native_control_matrix.json` before profiling. A packet can
promote only if all three row roles below have three distinct native repeats on
the same request/runtime shape and cite distinct Nsight artifacts.

| Row role | Model family | Purpose | Prompt/decode shape | Promotion rule |
|---|---|---|---|---|
| `primary_hybrid` | Granite 4.0 H Tiny, or Granite 4.0 H Small if VRAM allows | measure Granite attention/SSM boundary windows | prefill 128, decode 64, requests 16, batch 1 unless explicitly changed everywhere | only these rows may clear the 3% positive gate |
| `same_family_control` | same Granite model | same-model non-boundary SSM/attention-internal windows | exactly same dtype, graph mode, prompt/decode/request shape | must stay below the 3% gate |
| `cross_family_falsification` | Qwen3-Next hybrid family if available, or a separately preregistered feasible hybrid replacement | check whether the same boundary signal appears in a different hybrid family | exactly same dtype, graph mode, prompt/decode/request shape | must stay below the 3% gate |

If either control family is unavailable, record it as missing in the packet and
keep the conclusion limited to a profiling audit. Do not substitute an unmapped
model and call it a promotion control. In particular, if Qwen3-Next cannot fit
or run on the node, the first packet is audit-only even if Granite primary rows
look large.

### Copy-Paste Promotion Matrix

Before renting the node, verify that every model below can be served on the
chosen hardware. On a 5090, Granite Tiny/Small can produce an audit packet; if
the cross-family model cannot fit, the full promotion gate requires a larger
node or a separately preregistered feasible cross-family model. Do not call a
Granite-only packet promotable.

Use this matrix to allocate unique run IDs and artifact/log names. Each run ID
must correspond to a distinct server trace, client JSON log, reduction window,
and metric row.

| Run IDs | Role | Model | Control segment |
|---|---|---|---|
| `granite_primary_r1`, `granite_primary_r2`, `granite_primary_r3` | `primary_hybrid` | `$GRANITE_MODEL` | boundary windows |
| `granite_same_family_r1`, `granite_same_family_r2`, `granite_same_family_r3` | `same_family_control` | `$GRANITE_MODEL` | non-boundary same-type windows |
| `cross_family_r1`, `cross_family_r2`, `cross_family_r3` | `cross_family_falsification` | `$QWEN_MODEL` or `$PREREGISTERED_CROSS_FAMILY_MODEL` | matched boundary windows from that model's architecture map |

For each row in the table, start a fresh profiled server for that model and use
the run ID in every output path:

```bash
export RUN_ID=granite_primary_r1
export MODEL="$GRANITE_MODEL"
export SEED=1

nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --force-overwrite=true \
  --stats=true \
  --output="$HWK_RUN/nsys/${RUN_ID}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --disable-log-requests \
    --profiler-config.profiler cuda \
	  2>&1 | tee "$HWK_RUN/logs/nsys_server_${RUN_ID}.log"
```

In a second terminal, use the same `RUN_ID`, `MODEL`, and seed. First run a
short warmup request stream before the profiled replay; the warmup log must not
be reduced into the metric row:

```bash
python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "${RUN_ID}_warmup" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 16 \
  --requests 2 \
  --seed "$SEED" \
  --tokenizer "$MODEL" \
  --require-token-counts \
  > "$HWK_RUN/logs/client_${RUN_ID}_warmup.log" \
  2> "$HWK_RUN/logs/client_${RUN_ID}_warmup.stderr.log"
```

Then bracket and replay the fixed profiled request stream:

```bash
python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "$RUN_ID" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 16 \
  --seed "$SEED" \
  --tokenizer "$MODEL" \
  --require-token-counts \
  --profile-bracket \
  > "$HWK_RUN/logs/client_${RUN_ID}.log" \
	  2> "$HWK_RUN/logs/client_${RUN_ID}.stderr.log"
```

After the profiled replay exits successfully, stop the profiled server cleanly
before moving to the next row:

```bash
pkill -TERM -f "vllm.entrypoints.openai.api_server.*${MODEL}" || true
sleep 10
pgrep -af "vllm.entrypoints.openai.api_server" || true
```

If the server does not stop on `TERM`, record the full command and reason in
`$HWK_RUN/metadata/command_notes.md` before using a stronger signal. Verify that
the corresponding server log contains this `RUN_ID` or model ID, the client log
contains JSON with `run_id="$RUN_ID"` and every request status `ok`, and the
Nsight export exists and is non-empty:

```bash
ls -lh "$HWK_RUN/nsys/${RUN_ID}".* "$HWK_RUN/logs/nsys_server_${RUN_ID}.log" \
  "$HWK_RUN/logs/client_${RUN_ID}.log"
shasum -a 256 "$HWK_RUN/nsys/${RUN_ID}".* \
  "$HWK_RUN/logs/nsys_server_${RUN_ID}.log" \
  "$HWK_RUN/logs/client_${RUN_ID}.log" \
  >> "$HWK_RUN/metadata/artifact_hashes.sha256"
```

Only after these checks pass should you repeat with `RUN_ID=granite_primary_r2`
and seed `2`, then
`RUN_ID=granite_primary_r3` and seed `3`. Repeat the same pattern for
`granite_same_family_*` and `cross_family_*`, changing only `RUN_ID`, `MODEL`,
and seed. For a replacement cross-family model, use the preregistered
replacement model ID consistently in `MODEL`, `profile_scope.json`,
`native_control_matrix.json`, and client logs. Keep `batch-size`,
`prefill-tokens`, `decode-tokens`, and `requests` fixed across all nine rows
unless a new packet is started.

## Warmup And Determinism

Use fixed prompts, fixed output lengths, and at least three seeds when the
runner exposes seed control. Run warmup before profiling so setup, compilation,
weight loading, and CUDA graph capture do not dominate the trace.

```bash
export MODEL=ibm-granite/granite-4.0-h-tiny
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --disable-log-requests \
  2>&1 | tee "$HWK_RUN/logs/server_warmup.log"
```

If the local vLLM version uses a different serving command, record the exact
replacement in `$HWK_RUN/metadata/command_notes.md`.
After the warmup request succeeds, stop this warmup server cleanly before
starting the Nsight Systems server below. Verify the port is free and record the
stop/restart note in `$HWK_RUN/metadata/command_notes.md`; do not leave the
warmup server running while launching the profiled server.

## Nsight Systems Timeline Pass

Goal: identify whether attention/SSM boundaries show distinct kernels, gaps,
CUDA graph nodes, memory copies, synchronization, or host scheduling stalls.

Run the vLLM server under `nsys`; then replay fixed requests from a second
local terminal. Do **not** profile only `profiler_driver.py`, because that
would trace the HTTP client rather than the CUDA-serving process.

vLLM's profiling documentation also supports a dynamic capture path using the
server-side profiler API. Prefer this path when the installed vLLM build
supports it, because it lets the fixed request driver start and stop the capture
after warmup instead of tracing server startup. Keep
`VLLM_WORKER_MULTIPROC_METHOD=spawn` set in the server environment.

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

If the vLLM build does not expose `--profiler-config.profiler cuda`, use the
static server-side capture below and record the reason in
`$HWK_RUN/metadata/command_notes.md`.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --stats=true \
  --output="$HWK_RUN/nsys/granite_tiny_b1_decode64" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --disable-log-requests \
  2>&1 | tee "$HWK_RUN/logs/nsys_server_b1.log"
```

In a second local terminal on the same NVIDIA host:

```bash
python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "single_process_b1" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 16 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  > "$HWK_RUN/logs/client_single_process_b1.log" \
  2> "$HWK_RUN/logs/client_single_process_b1.stderr.log"
cat "$HWK_RUN/logs/client_single_process_b1.log"
```

For the dynamic Nsight capture path above, bracket the replay with vLLM's
server-side profiling endpoints. This is the preferred command when
`--profiler-config.profiler cuda` is accepted by the server:

```bash
python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "single_process_b1_profile_bracket" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 16 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  --profile-bracket \
  > "$HWK_RUN/logs/client_single_process_b1_profile_bracket.log" \
  2> "$HWK_RUN/logs/client_single_process_b1_profile_bracket.stderr.log"
cat "$HWK_RUN/logs/client_single_process_b1_profile_bracket.log"
```

The bracketed driver POSTs `/start_profile` before the fixed request replay
and `/stop_profile` afterward, so `--capture-range=cudaProfilerApi` captures
the serving process during the benchmark window rather than server startup.
When `--require-token-counts` is enabled, the driver synthesizes prompts
through a local tokenizer decode/encode roundtrip and rejects the run before
the profile window if any prompt cannot be proven to have exactly
`--prefill-tokens` tokens.

For every dynamic or static Nsight Systems row, perform the row lifecycle in this
order: start a fresh server, run warmup outside the reduced window, run the
profiled replay, stop the server, verify exported artifacts, append SHA-256
hashes, and only then start the next repeat. The checker can reject many
placeholder, dry-run, failed-request, and hash-mismatched artifacts, but it
cannot infer lifecycle mistakes if a startup-heavy trace is reduced as the
benchmark window. Record the reduction window and lifecycle note in
`$HWK_RUN/metadata/reduction_input_manifest.jsonl`.

`profiler_driver.py` is tracked in this repository and can be sanity-checked on
Mac with:

```bash
./venv_arm64/bin/python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "mac_dry_run" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 2 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  --profile-bracket \
  --dry-run
```

Do not interpret ad hoc manual API calls as benchmark evidence, and do not
submit a run where `metadata/profile_scope.json` says the profiled process was
only the HTTP client.

Do not add batch 8 rows to the first promotable matrix. If memory allows, run
batch 8 only as a separate audit packet with its own copied
`native_control_matrix.json`; do not mix batch sizes inside one promotion
decision.

## Boundary Annotation Pass

The timeline must be mapped back to layer types. Produce a local layer map from
the model config and save it next to the trace:

```bash
python "$HWK_ROOT/phase2/build_architecture_map.py" \
  > "$HWK_RUN/metadata/architecture_map_stdout.txt"
cp "$HWK_ROOT/phase2/architecture_map.json" "$HWK_RUN/metadata/"
```

For each attention-to-SSM and SSM-to-attention transition, annotate:

- preceding layer type and following layer type;
- kernels immediately before and after the boundary;
- any standalone conversion, copy, transpose, reshape, norm, or residual
  materialization kernel between them;
- idle gap between adjacent GPU kernels on the main stream;
- whether CUDA graph capture changes the visible launch structure.

## Nsight Compute Counter Pass

Goal: inspect only the suspicious kernels found by Nsight Systems. Do not start
with broad `ncu` capture across the full server. As with Nsight Systems, the
profiled process must be the vLLM server or a single-process vLLM benchmark,
not only `profiler_driver.py`.

Template:

```bash
ncu \
  --force-overwrite \
  --target-processes all \
  --set speedOfLight \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name '<SUSPICIOUS_KERNEL_REGEX>' \
  --launch-skip <N> \
  --launch-count <M> \
  --export "$HWK_RUN/ncu/suspicious_boundary_kernel" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --disable-log-requests \
  2>&1 | tee "$HWK_RUN/logs/ncu_server_suspicious_boundary_kernel.log"
```

In a second local terminal, replay the fixed request stream:

```bash
python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
  --run-id "ncu_suspicious_boundary_kernel" \
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 4 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  > "$HWK_RUN/logs/client_ncu_suspicious_boundary_kernel.log" \
	  2> "$HWK_RUN/logs/client_ncu_suspicious_boundary_kernel.stderr.log"
```

After the replay exits, stop the NCU-profiled server, verify the exported
`$HWK_RUN/ncu/suspicious_boundary_kernel.*` file, append its hash to
`$HWK_RUN/metadata/artifact_hashes.sha256`, and record the exact
`--kernel-name`, `--launch-skip`, `--launch-count`, source Nsight Systems
artifact, and time window in the reduction input manifest. Do not reuse one
long-running NCU server across multiple metric rows.

Fill `ncu_launch_selection` for every non-pending metric row from the exact
Nsight Systems timeline used to choose the NCU slice. The checker requires the
kernel regex, launch-skip/count values, source Nsight Systems artifact, matching
source time window, and a short derivation note; manual NCU launch selection
without that provenance is not admissible evidence.

Capture comparable kernels in non-boundary same-type regions. A boundary kernel
is interesting only if it has excess bytes, time, launch overhead, or stalls
relative to matched same-type regions.

## Decision Readout

Write `$HWK_RUN/readout.md` with this table:

| Question | Evidence | Decision |
|---|---|---|
| Distinct boundary conversion/materialization kernel? | kernel names and timestamps | yes/no |
| Boundary idle or launch gap? | median and paired deltas | yes/no |
| Extra DRAM/L2 traffic near boundary? | NCU bytes vs matched controls | yes/no |
| End-to-end impact estimate clears 3%? | all primary rows >= 3% and primary bootstrap CI low > 0 | yes/no |
| Same-family controls available? | model/control rows | yes/no |
| Cross-family falsification attempted? | model/control rows | yes/no |

Use matched comparisons across repeated fixed-request runs. Report median,
interquartile range, and bootstrap intervals over repeated reduced rows. The
analyzer exposes both all-row and primary-row bootstrap intervals; promotion
requires the primary interval low end to be above zero and every primary repeat
to clear the 3% recoverable-gain upper-bound gate. Do not report a single trace
screenshot as a positive result.

## Parser Input

After reducing the Nsight traces, fill the generated
`$HWK_RUN/profiler_metrics.json` skeleton created by
`create_native_run_packet.py`. Do not copy
`phase2/profiler_metrics_template.json` for the live GPU run; that file is a
minimal parser template, while the generated packet already contains the
predeclared primary, same-family control, and cross-family falsification row
shape.

Required fields:

| Field | Meaning |
|---|---|
| `model` | exact served model string |
| `run_id` | repeated-run identifier |
| `total_step_ms` | matched request-step wall or profiler time used as denominator |
| `attention_ssm_boundary_ms` | boundary-local cost from annotated Nsight trace |
| `matched_non_boundary_ms` | same-shape local control cost outside attention/SSM boundaries |
| `recoverable_fraction` | conservative fraction of avoidable boundary cost a fused operator could recover; capped at `0.60` by the current gate |
| `recoverable_fraction_basis` | non-placeholder justification for the recoverable-fraction value; cite the observed kernel/traffic rows or use a conservative assumption |
| `dtype` | exact served dtype, for example `bfloat16`; must be non-empty |
| `cuda_graph_enabled` | JSON boolean, not a string, recording whether CUDA graphs were enabled |
| `batch_shape.batch_size` | positive integer batch size used by the fixed replay |
| `batch_shape.prefill_tokens` | positive integer per-sample prompt/prefill token count |
| `batch_shape.decode_tokens` | positive integer decode token count |
| `batch_shape.requests` | positive integer number of fixed replay requests |
| `control_model_or_segment` | non-empty matched control segment/model label used for the non-boundary comparison |
| `row_role` | `primary_hybrid`, `same_family_control`, or `cross_family_falsification` |
| `control_family` | non-empty label for the matched control family, e.g. `same_family_matched_segment` |
| `boundary_direction` | attention/SSM boundary direction or `mixed_attention_ssm` for an aggregate row |
| `nsys_artifact` | exact relative path to the Nsight Systems artifact used for the row |
| `nsys_artifact_sha256` | `sha256:<64 lowercase hex chars>` digest of the `nsys_artifact` file |
| `ncu_artifact` | exact relative path to the Nsight Compute artifact used for the row |
| `ncu_artifact_sha256` | `sha256:<64 lowercase hex chars>` digest of the `ncu_artifact` file, or `not_run_no_boundary_signal` only in no-boundary-signal kill packets |
| `kernel_names` | non-empty list of kernel names reduced into the row |
| `boundary_indices` | list of architecture-map boundary indices represented by the row |
| `control_window_ids` | stable non-boundary window IDs; required and non-empty for same-family controls, empty for primary and cross-family boundary rows |
| `time_window_ms` | object with numeric `start` and `end` trace-window boundaries |
| `ncu_launch_selection` | object recording `kernel_regex`, `launch_skip`, positive `launch_count`, `source_nsys_artifact`, matching `source_time_window_ms`, and derivation notes for the NCU launch slice |
| `reduction_command` | exact command or script invocation used to reduce the Nsight artifacts into this row |
| `reduction_notes` | short non-placeholder explanation of the trace reduction |

Also update `metadata/reduction_input_manifest.json` for every reduced row.
The manifest should include the source Nsight Systems artifact, source time
window, source Nsight Compute artifact when present, reducer command, reducer
script or worksheet SHA-256, and row role. This does not replace
`profiler_metrics.json`; it makes the human reduction path auditable.

Use distinct `run_id` values, `nsys_artifact` paths, `ncu_artifact` paths, and
`time_window_ms` intervals for independent repeated traces. Duplicating one
trace into three rows is not admissible evidence and will fail the artifact
verifier.
Promotion also requires the primary bootstrap CI low end to be above zero, all
three primary rows to clear 3%, and at least three same-shape same-family control rows and
three same-shape cross-family falsification rows, with those controls staying
below the 3% recoverable-gain gate. A packet where controls preserve the same
3% signal remains audit-only.
Every row named in `profiler_metrics.json`, including same-family and
cross-family controls, must have a matching `profiler_driver.py` client replay
JSON log under `logs/` with the same model, `run_id`, batch size, uniform
per-sample prefill token count, decode-token count, and request count. Every
replay request must record `batch_size`, `prompt_sha256`, `payload_sha256`,
`prompt_token_counts`, `prompt_token_count_total`, `requested_decode_tokens`,
`expected_completion_tokens_total` when available, and
`response_usage.completion_tokens`; completion tokens must equal
`batch_size * requested_decode_tokens` so early-EOS runs cannot satisfy a
fixed-decode gate and batched vLLM usage accounting is interpreted correctly.
Metric rows without replay evidence now fail the packet checker.

Then run:

```bash
python "$HWK_ROOT/phase2/analyze_profiler_metrics.py" \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"
```

The generated Markdown sidecar is the paper-facing gate. It computes avoidable
boundary share and recoverable-gain upper bound:

```text
max(boundary_ms - matched_non_boundary_ms, 0)
------------------------------------------------ * recoverable_fraction
                 total_step_ms
```

Promotion requires at least three repeated primary rows whose minimum
recoverable-gain upper bound is at least 3%, whose primary bootstrap CI low end
is above zero, and whose same-family/cross-family control matrix passes
`check_profiler_run_artifacts.py --require-full-matrix`. If the mean
recoverable-gain upper bound is below 1%, shelve the branch unless a new
profiler anomaly appears.

## Artifact Completeness Check

Before treating the native run as reviewer-facing evidence, run the local
artifact verifier. The shorter packet checklist in
`phase2/native_run_packet_checklist.md` summarizes the exact directory contents
that should be sent back for review.

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

If Nsight Systems shows no plausible boundary-local signal and there is no
meaningful kernel for Nsight Compute, validate a clean kill packet instead of
fabricating an NCU row:

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --packet-mode no_boundary_signal_kill \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

This mode still requires server-side Nsight Systems evidence, fixed client
replay, reduced metric rows, and filled readout decisions. It only makes the
`.ncu-rep` artifact optional when the readout records no suspicious boundary
kernel.
For compute triage, a primary-only no-boundary packet may be used to stop
spending GPU time, but it is audit-only. A reviewer-facing clean kill must keep
`--require-full-matrix` and include the same-family and cross-family rows, so
the negative result is not just a failed single trace.

The verifier checks that the run directory contains:

- immutable environment metadata;
- architecture-map metadata copied beside the trace;
- Nsight Systems and Nsight Compute artifacts;
- server-side Nsight Systems and Nsight Compute profile scope in
  `metadata/profile_scope.json`;
- separate Nsight server profiler logs (`nsys_server*` or `ncu_server*`) and
  client replay logs. Server logs must contain real Nsight/vLLM/CUDA evidence
  markers, and client replay logs must be valid `profiler_driver.py` JSON with
  a non-empty top-level `model`, a top-level `run_id`, `dry_run: false`, and non-empty `requests`
  rows whose `status` fields are all `ok`, whose `batch_size` matches
  `prompt_token_counts`, whose prompt counts are uniform within each fixed
  batch, and whose `response_usage.completion_tokens` equals
  `batch_size * requested_decode_tokens`;
- `readout.md` with the pre-registered decision questions;
- `profiler_metrics.json` with at least three repeated valid rows for one
  model and at least three distinct repeated `run_id` values. Repeated rows
  for the same model/config must also point to distinct `nsys_artifact`,
  `ncu_artifact`, and `time_window_ms` intervals.
- `profiler_analysis_gate.json` and `.md` generated from that exact
  `profiler_metrics.json`.

The verifier recomputes the analysis gate from `profiler_metrics.json` and
rejects stale or copied `profiler_analysis_gate.json`/`.md` outputs whose
status, decision, summary, or row count no longer match the metric rows.

A `PASS` means the artifact bundle is complete enough for human review. It does
not mean HybridKernel is promoted, and it does not authorize any speedup claim.
Promotion still depends on the profiler-analysis gate and the controls below.

A synthetic, non-evidence packet fixture exists at
`phase2/tests/fixtures/synthetic_profiler_run_packet/` to show the required
directory shape and keep the checker covered on Mac. It contains placeholder
Nsight files and must not be cited as profiler data.

## Post-Promotion Benchmark Tables

If the profiler gate promotes, the paper still needs final benchmark tables
before any systems claim:

- stock vLLM versus the prototype on the promoted model/config;
- 1K, 4K, 16K, and 32K prompt lengths with 256 decode tokens;
- batch 1 as the primary setting and batch 8 as an audit packet if memory allows;
- latency breakdown from Nsight Systems and HBM counters from Nsight Compute;
- a quality-invariance smoke table on
  `experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl`
  (`sha256:48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`).
  Run stock vLLM and the prototype with greedy decoding, temperature 0, and
  256 max new tokens. The table must report normalized exact-answer changes,
  output-token count drift, and any available continuation logprob/NLL drift.
  Pass criterion for a prototype claim: zero normalized exact-answer
  regressions on this 12-prompt smoke set and mean output-length drift within
  10%. Save the table as `quality_smoke.json` and validate it with
  `./venv_arm64/bin/python -m experimental.hybridkernel.phase2.check_quality_smoke_artifacts
  path/to/quality_smoke.json --repo-root "$PWD"`. This is only a smoke guard,
  not a benchmark-accuracy claim.

Do not add these tables unless the strict profiler packet promotes first.

## Promotion Criteria

Promote HybridKernel to implementation only if all are true:

- a boundary-local overhead is visible in Nsight Systems and attributable to
  attention/SSM transitions rather than warmup, graph capture, batching, or
  unrelated kernels;
- Nsight Compute shows avoidable memory traffic or stalls on the same boundary
  region;
- the estimated end-to-end gain is at least 3%, or the localized gain is large
  enough that a concrete fused-kernel design plausibly clears 3%;
- the result survives at least three distinct repeated runs, three same-shape
  same-family control rows, and three same-shape cross-family falsification
  rows;
- the readout separates boundary-local runtime/cache effects from generic
  CUDA graph, warmup, batching, and unrelated-kernel effects.
- `check_profiler_run_artifacts.py` passes for the exact run directory being
  cited.

Kill or pause if any are true:

- no separable boundary overhead appears;
- overhead is below 3% and has no credible route to 3%;
- the only apparent gain is already covered by vLLM HMA/NIXL state-transfer
  machinery;
- the signal disappears under CUDA graph capture, batching, or same-family
  controls.
