# HybridKernel NVIDIA/vLLM Profiler Runbook

- date: 2026-05-05
- status: pre-GPU runbook; no NVIDIA run has been executed from this repo
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
family.

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
python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model ibm-granite/granite-4.0-h-tiny
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
`nsys`, `ncu`, or `python` capture lines.

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
```

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
  "model": "$MODEL",
  "vllm_command": "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --max-model-len 2048 --disable-log-requests",
  "model_scopes": [
    {
      "row_role": "primary_hybrid,same_family_control",
      "model": "$MODEL",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    },
    {
      "row_role": "cross_family_falsification",
      "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    }
  ]
}
JSON
```

If you cannot run the cross-family model, leave the metric rows absent and
record the missing control in `readout.md`; the packet is then audit-only. Do
not keep a Qwen metric row without a matching `model_scopes` entry and client
replay log.

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
| `cross_family_falsification` | Qwen3-Next hybrid family if available | check whether the same boundary signal appears in a different hybrid family | exactly same dtype, graph mode, prompt/decode/request shape | must stay below the 3% gate |

If either control family is unavailable, record it as missing in the packet and
keep the conclusion limited to a profiling audit. Do not substitute an unmapped
model and call it a promotion control.

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
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 16 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  > "$HWK_RUN/logs/client_b1.log" \
  2> "$HWK_RUN/logs/client_b1.stderr.log"
cat "$HWK_RUN/logs/client_b1.log"
```

For the dynamic Nsight capture path above, bracket the replay with vLLM's
server-side profiling endpoints. This is the preferred command when
`--profiler-config.profiler cuda` is accepted by the server:

```bash
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
  > "$HWK_RUN/logs/client_b1_profile_bracket.log" \
  2> "$HWK_RUN/logs/client_b1_profile_bracket.stderr.log"
cat "$HWK_RUN/logs/client_b1_profile_bracket.log"
```

The bracketed driver POSTs `/start_profile` before the fixed request replay
and `/stop_profile` afterward, so `--capture-range=cudaProfilerApi` captures
the serving process during the benchmark window rather than server startup.

`profiler_driver.py` is tracked in this repository and can be sanity-checked on
Mac with:

```bash
./venv_arm64/bin/python "$HWK_ROOT/phase2/profiler_driver.py" \
  --model "$MODEL" \
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

Repeat for batch 8 if memory allows.

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
  --batch-size 1 \
  --prefill-tokens 128 \
  --decode-tokens 64 \
  --requests 4 \
  --seed 1 \
  --tokenizer "$MODEL" \
  --require-token-counts \
  2>&1 | tee "$HWK_RUN/logs/client_ncu_suspicious_boundary_kernel.log"
```

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
| End-to-end impact estimate clears 3%? | formula and confidence interval | yes/no |
| Same-family controls available? | model/control rows | yes/no |
| Cross-family falsification attempted? | model/control rows | yes/no |

Use matched comparisons across repeated fixed-request runs. Report median,
interquartile range, and bootstrap intervals over repeated reduced rows. Do not
report a single trace screenshot as a positive result.

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
| `recoverable_fraction` | conservative fraction of avoidable boundary cost a fused operator could recover |
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
| `time_window_ms` | object with numeric `start` and `end` trace-window boundaries |
| `reduction_command` | exact command or script invocation used to reduce the Nsight artifacts into this row |
| `reduction_notes` | short non-placeholder explanation of the trace reduction |

Use distinct `run_id` values, `nsys_artifact` paths, `ncu_artifact` paths, and
`time_window_ms` intervals for independent repeated traces. Duplicating one
trace into three rows is not admissible evidence and will fail the artifact
verifier.
Promotion also requires at least three same-shape same-family control rows and
three same-shape cross-family falsification rows, with those controls staying
below the 3% recoverable-gain gate. A packet where controls preserve the same
3% signal remains audit-only.
Every model named in `profiler_metrics.json`, including same-family and
cross-family controls, must have a matching `profiler_driver.py` client replay
JSON log under `logs/` with the same batch size, uniform per-sample prefill
token count, decode-token count, and request count. Every replay request must record
`batch_size`, `prompt_token_counts`, `prompt_token_count_total`,
`requested_decode_tokens`, `expected_completion_tokens_total` when available,
and `response_usage.completion_tokens`; completion tokens must equal
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

Promotion requires at least three repeated runs where the minimum recoverable
gain upper bound is at least 3%. If the mean recoverable-gain upper bound is
below 1%, shelve the branch unless a new profiler anomaly appears.

## Artifact Completeness Check

Before treating the native run as reviewer-facing evidence, run the local
artifact verifier. The shorter packet checklist in
`phase2/native_run_packet_checklist.md` summarizes the exact directory contents
that should be sent back for review.

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  | tee "$HWK_RUN/artifact_check.json"
```

If Nsight Systems shows no plausible boundary-local signal and there is no
meaningful kernel for Nsight Compute, validate a clean kill packet instead of
fabricating an NCU row:

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --packet-mode no_boundary_signal_kill \
  | tee "$HWK_RUN/artifact_check.json"
```

This mode still requires server-side Nsight Systems evidence, fixed client
replay, reduced metric rows, and filled readout decisions. It only makes the
`.ncu-rep` artifact optional when the readout records no suspicious boundary
kernel.

The verifier checks that the run directory contains:

- immutable environment metadata;
- architecture-map metadata copied beside the trace;
- Nsight Systems and Nsight Compute artifacts;
- server-side Nsight Systems and Nsight Compute profile scope in
  `metadata/profile_scope.json`;
- separate Nsight server profiler logs (`nsys_server*` or `ncu_server*`) and
  client replay logs. Server logs must contain real Nsight/vLLM/CUDA evidence
  markers, and client replay logs must be valid `profiler_driver.py` JSON with
  a non-empty top-level `model`, `dry_run: false`, and non-empty `requests`
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
