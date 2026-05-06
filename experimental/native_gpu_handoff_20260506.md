# Native GPU Handoff Map

- date: 2026-05-06
- scope: what remains after Mac/Triton saturation
- rule: do not add GPU numbers to any paper until the relevant gate below
  passes its local verifier or decision checklist

## Priority Order

1. **SinkAware rank-2 native timing**: highest expected value because the
   Mac-local downstream controls are positive and the rank/cost frontier
   identifies a concrete implementation candidate.
2. **HybridKernel profiler packet**: useful only if native vLLM profiling finds
   separable attention/SSM boundary overhead.
3. **ThoughtFlow-FP8**: no GPU work for the current branch. Reopen only after a
   new preregistered utility signal exists and passes a fresh/larger frozen
   sparse-cache surface.

## HybridKernel

Runbook: `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`

Checklist: `experimental/hybridkernel/phase2/native_run_packet_checklist.md`

Minimum admissible packet:

- whole packet directory returned, not screenshots;
- server-side Nsight Systems and Nsight Compute artifacts;
- immutable environment capture including `nvidia-smi`, `nsys`, `ncu`, and
  `python` lines;
- at least three distinct same-model/same-config metric rows;
- explicit `run_id`, dtype, CUDA graph state, batch shape, request count, and
  matched control label in every row;
- local analyzer and artifact checker both pass.

Decision commands on the NVIDIA host after trace reduction:

```bash
python experimental/hybridkernel/phase2/analyze_profiler_metrics.py \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python experimental/hybridkernel/phase2/check_profiler_run_artifacts.py \
  --run-dir "$HWK_RUN" \
  | tee "$HWK_RUN/artifact_check.json"
```

Promote only if the repeated same-config recoverable-gain upper bound clears
3%. Kill or shelve if repeated native summaries show less than 1% recoverable
gain.

## SinkAware

Runbook: `experimental/sinkaware/phase2/gpu_gate_runbook.md`

Validator: `experimental/sinkaware/phase2/check_native_gpu_packet.py`

Native rows to measure:

- exact attention;
- exact fixed-sink decomposition;
- rank-2 sink-logit predictor;
- position-only predictor.

Required native outputs:

- `metadata.json`;
- `quality_drift.csv`;
- `quality_drift_by_head.csv`;
- `latency.csv`;
- `ncu_summary.csv`;
- `decision.md`.

Before citing any returned native packet, run:

```bash
./venv_arm64/bin/python experimental/sinkaware/phase2/check_native_gpu_packet.py \
  "$SINKAWARE_GPU_PACKET" \
  | tee "$SINKAWARE_GPU_PACKET/artifact_check.json"
```

Promote only if rank-2 preserves the Mac-local downstream-control behavior and
shows at least a 3% native speed or memory-traffic improvement over exact
attention. Kill if rank-2 is slower, unstable, or indistinguishable from
position-only.

## ThoughtFlow-FP8

Current manifest:
`experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`

Do not run GPU experiments for the current `rdu_topk` branch. The only valid
next step is a new preregistration artifact for a genuinely different utility
family, followed by one one-shot evaluation on a fresh/larger frozen
sparse-cache surface.

## Local Readiness Recheck

Before interpreting any new native packet, rerun the owned Mac suite:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/tests \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests \
  experimental/sinkaware/phase2/tests \
  experimental/sinkaware/phase3/tests \
  experimental/sinkaware/phase4/tests \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Any paper update after GPU work should cite only rows that pass the project
gate. Do not convert Mac interpreter correctness into throughput, latency, HBM,
energy, or CUDA claims.
