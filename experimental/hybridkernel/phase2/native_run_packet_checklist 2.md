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
Do not use `--allow-missing-native-artifacts` for submitted packets; that flag
exists only for schema fixtures and is rejected with `--require-full-matrix`.

## Required Files

The packet is incomplete unless all of these exist:

| Path | Required content |
|---|---|
| `metadata/environment.txt` | timestamp, hostname, `nvidia-smi`, `nsys --version`, `ncu --version`, Python version, package freeze, vLLM/Torch/Triton/Transformers versions |
| `metadata/environment.json` | parseable environment record with `environment_version: "hybridkernel_environment_v1"`, host/GPU/profiler/Python fields, and installed `vllm`, `torch`, `triton`, and `transformers` versions |
| `metadata/profile_scope.json` | server-side scope for both Nsight Systems and Nsight Compute |
| `metadata/architecture_map.json` | copied HybridKernel architecture map used for boundary annotation |
| `metadata/model_provenance.json` | exact model and tokenizer provenance for every served metric model: model ID, served model ID, resolved model revision, tokenizer revision, immutable-revision attestations, cache source, snapshot manifest path/SHA, `local_files_only`, and `trust_remote_code` |
| `metadata/native_control_matrix.json` | copied control matrix fixing primary, same-family, and cross-family row roles before profiling |
| `metadata/reduction_input_manifest.json` | row-level reduction audit trail with `manifest_version: "hybridkernel_reduction_inputs_v1"` tying each metric row to source Nsight exports, time windows, commands, and reducer script or worksheet path plus SHA-256 digests |
| `metadata/reduction_worksheet.tsv` or equivalent cited source file | filled manual/scripted reduction worksheet cited by `reduction_source_path` in the manifest; template markers are rejected |
| `logs/*.log` or `logs/*.txt` | Nsight server profiler logs (`nsys_server*` or `ncu_server*`) and client replay logs. Server logs must contain real Nsight/vLLM/CUDA evidence markers; client logs must be valid `profiler_driver.py` JSON with a non-empty top-level `model`, a top-level `run_id` matching one metric row, `dry_run: false`, `token_counts_required: true`, a non-empty `token_count_source`, and non-empty `requests` rows whose `status` fields are all `ok`, whose `prompt_sha256` and `payload_sha256` fields are filled, and whose prompt/decode token counts are positive. |
| `nsys/*.nsys-rep`, `nsys/*.sqlite`, or `nsys/*.qdrep` | server-side Nsight Systems timeline artifacts, not placeholder files |
| `ncu/*.ncu-rep` | server-side Nsight Compute artifacts for suspicious and matched control kernels, not placeholder files. Required for boundary-evidence packets; optional only with explicit `--packet-mode no_boundary_signal_kill` and row `ncu_artifact: "not_run_no_boundary_signal"`. |
| `readout.md` | completed decision table from the runbook |
| `profiler_metrics.json` | at least nine valid rows: three primary repeats, three same-shape same-family controls, and three same-shape cross-family falsification rows |
| `profiler_analysis_gate.json` and `.md` | output from `analyze_profiler_metrics.py` for this exact `profiler_metrics.json` |

After the checker command runs, the returned packet should also include
`artifact_check.json`. The checker validates the files above and then produces
that final self-report.

`metadata/model_provenance.json` must use this shape. The checker rejects
mutable revision aliases such as `main`, `master`, `HEAD`, `latest`, and
`refs/heads/*`.

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
  "model": "$MODEL",
  "vllm_command": "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --max-model-len 2048 --disable-log-requests",
  "model_scopes": [
    {
      "row_roles": ["primary_hybrid", "same_family_control"],
      "model": "$MODEL",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    },
    {
      "row_roles": ["cross_family_falsification"],
      "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
      "vllm_command": "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
    }
  ]
}
```

Accepted values for profiler process fields are `vllm_server` or
`single_process_vllm_benchmark`. `http_client`, `profiler_driver`, `curl`, or
manual API calls are not admissible profiler scopes.
If `profiler_metrics.json` contains more than one model, `model_scopes` must
cover every metric model with the actual vLLM command used for that model. If
Qwen3-Next is replaced by a preregistered feasible cross-family hybrid, update
the cross-family `model_scopes` entry, row IDs, metric `model` values,
client replay logs, and `metadata/native_control_matrix.json` before profiling;
do not leave the Qwen placeholder in a replacement packet.

The checker also rejects tiny or placeholder profiler exports. By default each
matched Nsight artifact must be at least 1024 bytes and must not contain
skeleton placeholder markers. Use the default threshold for submitted run
packets.

The checker also rejects empty, placeholder, dry-run, or failed logs. Client
logs must keep the exact non-dry-run JSON printed by
`phase2/profiler_driver.py`, including the row-specific top-level `run_id`;
server logs must keep raw profiler/server stdout with Nsight, vLLM, CUDA, or
model-loading markers. Do not replace the logs with screenshots, shell
comments, or manually written summaries.

## Required Metric Rows

Each valid row in `profiler_metrics.json` must represent one independent
reduced native trace:

- `model`: exact served model string;
- `run_id`: distinct repeated-run identifier;
- `total_step_ms`: positive denominator from the matched request-step window;
- `attention_ssm_boundary_ms`: non-negative boundary-local measured cost;
- `matched_non_boundary_ms`: non-negative same-shape control cost;
- `recoverable_fraction`: value in `[0, 0.60]`; a larger recovery assumption
  requires a new preregistered gate rather than an ad-hoc row edit.
- `recoverable_fraction_basis`: non-placeholder justification for the chosen
  recoverable fraction;
- `dtype`: non-empty served dtype string;
- `cuda_graph_enabled`: JSON boolean, not a string placeholder;
- `batch_shape.batch_size`: positive integer batch size;
- `batch_shape.prefill_tokens`: positive integer per-sample prefill token count;
- `batch_shape.decode_tokens`: positive integer decode token count;
- `batch_shape.requests`: positive integer replay request count;
- `control_model_or_segment`: non-empty matched control label.
- `row_role`: `primary_hybrid`, `same_family_control`, or
  `cross_family_falsification`;
- `control_family`: non-placeholder control family label;
- `boundary_direction`: non-placeholder boundary direction label;
- `nsys_artifact`: relative path inside the packet to a `.nsys-rep`, `.sqlite`,
  or `.qdrep` file;
- `nsys_artifact_sha256`: `sha256:<64 lowercase hex chars>` digest of the
  referenced Nsight Systems file;
- `ncu_artifact`: relative path inside the packet to a `.ncu-rep` file, or
  `not_run_no_boundary_signal` only when using `--packet-mode
  no_boundary_signal_kill`;
- `ncu_artifact_sha256`: `sha256:<64 lowercase hex chars>` digest of the
  referenced Nsight Compute file, or `not_run_no_boundary_signal` in the
  no-boundary-signal kill mode;
- `kernel_names`: non-empty list of kernel names used in the reduction;
- `boundary_indices`: integer boundary IDs, non-empty for `primary_hybrid`;
  also non-empty for `cross_family_falsification` rows because those are
  boundary rows in the mapped cross-family hybrid. Same-family controls use
  empty `boundary_indices`.
- `control_window_ids`: stable non-boundary window IDs, non-empty for
  `same_family_control` rows and empty for primary/cross-family boundary rows;
- `time_window_ms`: object with numeric `start` and `end`, with `end > start`;
- `ncu_launch_selection`: object recording `kernel_regex`, `launch_skip`,
  positive `launch_count`, `source_nsys_artifact`, matching
  `source_time_window_ms`, and derivation notes for the NCU launch slice;
- `reduction_command`: exact command or script invocation used to reduce the
  native artifacts into this row;
- `reduction_notes`: non-placeholder notes explaining how the row was reduced.

The row must also be represented in
`metadata/reduction_input_manifest.json`, including the source Nsight Systems
artifact, source time window, Nsight Compute artifact when present, reducer
command, reducer script or worksheet path, reducer script or worksheet
SHA-256, and row role. This manifest does not replace
`profiler_metrics.json`; it prevents analyst-selected timeline windows from
being unauditable. Copy
`phase2/reduction_worksheet_template.tsv` into the run packet, fill one row per
metric row before editing `profiler_metrics.json`, and cite the filled
worksheet SHA-256 from `metadata/reduction_input_manifest.json`.

Do not duplicate one trace into multiple rows. Every non-pending metric row
must cite its own `nsys_artifact`, and every boundary-evidence row must cite its
own `ncu_artifact`; the checker rejects artifact reuse across primary,
same-family control, and cross-family falsification roles. Repeated rows for
the same model/config must also have distinct `time_window_ms` intervals. Do
not mix different model families and call them repeated runs for the same gate.
Use `metadata/native_control_matrix.json` as the row-role authority. If the
cross-family falsification model is unavailable, record that fact in the
readout and treat the packet as audit-only rather than substituting an unmapped
model. A substitute cross-family hybrid is admissible only if
`phase2/cross_family_control_replacement_template.json` is filled before
profiling, copied into the packet metadata, and its row is added to
`metadata/native_control_matrix.json` before any metric reduction. A replacement
chosen after seeing profiler output is not promotion evidence.
Promotion requires at least three same-shape same-family control rows and three
same-shape cross-family falsification rows, and both control families must stay
below the 3% recoverable-gain gate. Controls that reproduce the same signal do
not promote the branch.

The artifact checker resolves `nsys_artifact` and `ncu_artifact` against the
run directory. Missing files, absolute paths, `..` escapes, wrong extensions,
placeholder profiler exports, UTF-8/plain-text fake profiler files, and SHA-256
mismatches are rejected. It also checks the client replay `model`, `run_id`,
batch/per-sample prefill/decode/request shape against metric rows for models
present in the client logs, requires per-request prompt and payload SHA-256s,
requires uniform prompt counts within each fixed batch, and requires replay
`response_usage.completion_tokens` to equal `batch_size * requested_decode_tokens`,
and records `expected_completion_tokens_total` when emitted by
`profiler_driver.py`.

## Final Local Commands

Run these on the NVIDIA host after reducing traces:

```bash
python "$HWK_ROOT/phase2/analyze_profiler_metrics.py" \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

If the Nsight Systems pass shows no suspicious boundary kernel to target with
Nsight Compute, run the checker in explicit negative mode instead:

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --packet-mode no_boundary_signal_kill \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

A primary-only no-boundary packet is allowed only as compute triage to avoid
profiling unnecessary Nsight Compute launches. It is audit-only and cannot be
used as the final paper-facing kill without the full matrix above.

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
