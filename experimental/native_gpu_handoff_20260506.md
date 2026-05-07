# Native GPU Handoff Map

- date: 2026-05-06
- scope: what remains after Mac/Triton saturation
- rule: do not add GPU numbers to any paper until the relevant gate below
  passes its local verifier or decision checklist

## Priority Order

1. **HybridKernel profiler packet**: useful only if native vLLM profiling finds
   separable attention/SSM boundary overhead.
2. **SSQ-LR**: no GPU validation until a real Mac/shared trace packet clears
   S1--S3 without per-model retuning.
3. **HORN**: no GPU validation until real H1a/H1/H3 boundary controls show
   cross-model directional asymmetry.
4. **HBSM**: no GPU validation until real B1/B2/B3 sensitivity and predictor
   gates survive the KL-style and activation/outlier baselines.
5. **ThoughtFlow-FP8**: no GPU work for the current branch set (`rdu_topk`,
   `psi_topk`, or `vwac_topk`). Reopen only after a new preregistered utility
   signal exists and passes a fresh/larger frozen sparse-cache surface.

## HybridKernel

Runbook: `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`

Checklist: `experimental/hybridkernel/phase2/native_run_packet_checklist.md`

Minimum admissible packet:

- whole packet directory returned, not screenshots;
- server-side Nsight Systems and Nsight Compute artifacts;
- immutable environment capture including `nvidia-smi`, `nsys`, `ncu`, and
  `python` lines;
- copied `metadata/native_control_matrix.json` from
  `experimental/hybridkernel/phase2/native_control_matrix.json`;
- at least three distinct primary metric rows;
- at least three same-shape same-family control rows;
- at least three same-shape cross-family falsification rows;
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

Promote only if the repeated primary recoverable-gain upper bound clears 3% and
the same packet includes three same-family and three cross-family controls on
the same request/runtime shape that stay below the 3% gate. A primary-only
packet is audit-only. Kill or shelve if repeated native summaries show less
than 1% recoverable gain.

## SSQ-LR / HORN / HBSM

These branches do not have a GPU handoff yet. Their next admissible artifacts
are real shared trace packets built from saved Mac/GPU tensors or sensitivity
rows and validated with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir> \
  --mode real --project <ssq_lr|horn|hbsm>
```

Resource-limited trace packets are allowed only as hook/schema diagnostics and
must carry a `RESOURCE_LIMITED_NOT_PROMOTABLE` decision. Full GPU validation is
blocked until the Mac/shared trace gates identify a surviving recipe or
mechanism.

Before collecting those tensors or sensitivity rows, use the deterministic
trace plan:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

Current artifact:
`experimental/shared/results/hybrid_trace_plan_20260507/`. It is a row-level
capture checklist only, with no model or GPU evidence.

## ThoughtFlow-FP8

Current manifest:
`experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`

Do not run GPU experiments for the current branch set: the consumed `rdu_topk`,
`psi_topk`, or `vwac_topk` branches.
The only valid next step is a new preregistration artifact for a genuinely
different utility family, followed by one one-shot evaluation on a fresh/larger
frozen sparse-cache surface.

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
  experimental/ssq_lr/phase2/tests \
  experimental/horn/phase2/tests \
  experimental/hbsm/phase2/tests \
  experimental/shared/tests \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Any paper update after GPU work should cite only rows that pass the project
gate. Do not convert Mac interpreter correctness into throughput, latency, HBM,
energy, or CUDA claims.
