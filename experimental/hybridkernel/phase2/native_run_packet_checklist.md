# HybridKernel Native Run Packet Checklist

Status: **required before any native profiler result is cited.**

This checklist is for the user-operated NVIDIA host. It is not a benchmark
claim, and it is not needed for Mac-local development. Its purpose is to make a
native run packet reviewable before anyone interprets profiler numbers.

## Packet Directory

Use one directory per independent native run bundle:

```bash
export HWK_ROOT=/path/to/LatentWire/experimental/hybridkernel
python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model ibm-granite/granite-4.0-h-tiny
```

The command prints the created `run_dir`; export that path as `HWK_RUN` for
the remaining runbook commands. The generated packet is a skeleton only. It
contains `TODO_NATIVE_PROFILE_FILL` sentinels that the artifact checker rejects
until real native metadata, readout evidence, and metric rows replace them.

Do not send partial screenshots, notebook snippets, or client-only logs as
evidence. Send the whole `$HWK_RUN` directory.

## Required Files

The packet is incomplete unless all of these exist:

| Path | Required content |
|---|---|
| `metadata/environment.txt` | timestamp, hostname, `nvidia-smi`, `nsys --version`, `ncu --version`, Python version, package freeze, vLLM/Torch/Triton/Transformers versions |
| `metadata/profile_scope.json` | server-side scope for both Nsight Systems and Nsight Compute |
| `metadata/architecture_map.json` | copied HybridKernel architecture map used for boundary annotation |
| `logs/*.log` or `logs/*.txt` | server profiler logs and client replay logs |
| `nsys/*.nsys-rep`, `nsys/*.sqlite`, or `nsys/*.qdrep` | server-side Nsight Systems timeline artifacts, not placeholder files |
| `ncu/*.ncu-rep` | server-side Nsight Compute artifacts for suspicious and matched control kernels, not placeholder files |
| `readout.md` | completed decision table from the runbook |
| `profiler_metrics.json` | at least three valid rows for one model with distinct repeated `run_id` values |
| `profiler_analysis_gate.json` and `.md` | output from `analyze_profiler_metrics.py` for this exact `profiler_metrics.json` |

After the checker command runs, the returned packet should also include
`artifact_check.json`. The checker validates the files above and then produces
that final self-report.

## Required Scope JSON

`metadata/profile_scope.json` must make it clear that both profilers observed
server-side CUDA work:

```json
{
  "profiled_process": "vllm_server",
  "nsys_profiled_process": "vllm_server",
  "ncu_profiled_process": "vllm_server",
  "trace_scope": "server-side CUDA kernels under fixed request replay",
  "nsys_trace_scope": "server-side CUDA kernels under fixed request replay",
  "ncu_trace_scope": "server-side CUDA kernels under suspicious-kernel replay",
  "request_driver_process": "profiler_driver_http_client",
  "vllm_command": "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
}
```

Accepted values for profiler process fields are `vllm_server` or
`single_process_vllm_benchmark`. `http_client`, `profiler_driver`, `curl`, or
manual API calls are not admissible profiler scopes.

The checker also rejects tiny or placeholder profiler exports. By default each
matched Nsight artifact must be at least 1024 bytes and must not contain
skeleton placeholder markers. Use the default threshold for submitted run
packets.

## Required Metric Rows

Each valid row in `profiler_metrics.json` must represent one independent
reduced native trace:

- `model`: exact served model string;
- `run_id`: distinct repeated-run identifier;
- `total_step_ms`: positive denominator from the matched request-step window;
- `attention_ssm_boundary_ms`: non-negative boundary-local measured cost;
- `matched_non_boundary_ms`: non-negative same-shape control cost;
- `recoverable_fraction`: value in `[0, 1]`.

Do not duplicate one trace into multiple rows. Do not mix different model
families and call them repeated runs for the same gate.

## Final Local Commands

Run these on the NVIDIA host after reducing traces:

```bash
python "$HWK_ROOT/phase2/analyze_profiler_metrics.py" \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  | tee "$HWK_RUN/artifact_check.json"
```

If the artifact checker fails, stop and fix the packet before interpreting
results.

For a Mac-local example of the required directory shape, inspect
`phase2/tests/fixtures/synthetic_profiler_run_packet/`. It is a synthetic
checker fixture with placeholder Nsight files, not native profiler evidence;
the default checker now rejects it unless native artifact validation is
explicitly disabled for schema-only tests.

## Decision

Local Mac work is now saturated. Do not add more Mac kernels, scaffolds, or
paper claims until a native packet passes the artifact checker. The next action
is to run the NVIDIA/vLLM profiling packet and then decide promote, pause, or
kill from the profiler-analysis gate.
