# Native GPU Handoff Map

- date: 2026-05-07
- scope: what remains after Mac/Triton saturation
- rule: do not add GPU numbers to any paper until the relevant gate below
  passes its local verifier or decision checklist

## Priority Order

1. **HybridKernel profiler packet**: useful only if native vLLM profiling finds
   separable attention/SSM boundary overhead.
2. **SSQ-LR**: current recipe failed 12-prompt no-retuning transfer to Granite
   350M; no GPU validation until a newly preregistered recipe clears S1--S3
   without per-model retuning.
3. **HORN**: no GPU validation until a newly preregistered reopening produces
   real H1a/H1/H2/H3 boundary controls with cross-model directional asymmetry
   and noise-propagation sensitivity.
4. **HBSM**: no GPU validation until a newly preregistered B1/B2/B3 mechanism
   hypothesis survives the KL-style and activation/outlier baselines.
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
- filled `metadata/model_provenance.json` covering every served metric model
  and tokenizer revision;
- at least three distinct primary metric rows;
- at least three same-shape same-family control rows;
- at least three same-shape cross-family falsification rows;
- a completed `metadata/reduction_input_manifest.json` tying each metric row to
  source Nsight artifacts, time windows, reducer command, and reducer script
  or worksheet path plus SHA-256;
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
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

Promote only if the repeated primary recoverable-gain upper bound clears 3% and
the same packet includes three same-family and three cross-family controls on
the same request/runtime shape that stay below the 3% gate. A primary-only
packet is audit-only. Kill or shelve if repeated native summaries show less
than 1% recoverable gain.

The fixed-request client commands in the runbook must use
`--require-token-counts`; the driver then synthesizes tokenizer-roundtrip
prompts and fails before profiling if the exact prefill length cannot be
proven.

## SSQ-LR / HORN / HBSM

These branches do not have a GPU handoff yet. Under the current evidence,
admissible Mac work is limited to revalidating existing stop packets,
documentation/tests/runbook hygiene, or writing a new preregistration before
any new rows are collected. A newly preregistered reopening would build real
shared trace packets from saved Mac/GPU tensors or sensitivity rows and validate
them with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir> \
  --mode real --project <ssq_lr|horn|hbsm>
```

Resource-limited trace packets are allowed only as hook/schema diagnostics and
must carry a `RESOURCE_LIMITED_NOT_PROMOTABLE` decision. Full GPU validation is
blocked until the Mac/shared trace gates identify a surviving recipe or
mechanism.
As of 2026-05-07, SSQ-LR has no surviving GPU handoff recipe: the frozen
`0,30` mixed INT3/MXFP4 recipe fails Granite 350M transfer, and layer-0 rescue
diagnostics fail two-model S3 because Granite Tiny and Granite 350M prefer
different frozen recipes.

For a newly preregistered reopening only, use the deterministic trace plan
before collecting tensors or sensitivity rows:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

Current artifact:
`experimental/shared/results/hybrid_trace_plan_20260507/`. It is a row-level
capture checklist only, with no model or GPU evidence.

Then generate fill-in capture manifests:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

Current artifact:
`experimental/shared/results/hybrid_capture_manifests_20260507/`. Fill every
`TO_FILL_BEFORE_CAPTURE` field from a real capture before building packets.
Templates are deliberately rejected by the builder.

For a newly preregistered local/shared reopening only, build SSQ-LR and HORN
candidate packets from saved tensors. This is not a GPU handoff and cannot
create native performance evidence:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project ssq_lr \
  --tensor-packet "$SSQ_LR_TENSOR_PACKET" \
  --output-dir experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<YYYYMMDD>_<model_slug>

./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project horn \
  --tensor-packet "$HORN_TENSOR_PACKET" \
  --output-dir experimental/horn/phase2/results/horn_gate_h1_<YYYYMMDD>_<model_slug>
```

For a newly preregistered local/shared reopening only, build HBSM from saved
sensitivity rows. This is not a GPU handoff and cannot create native
performance evidence:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project hbsm \
  --row-packet "$HBSM_ROW_PACKET" \
  --output-dir experimental/hbsm/phase2/results/hbsm_gate_b1_<YYYYMMDD>_<model_slug>
```

Only after a real HBSM B1 packet establishes sensitivity heterogeneity should
the B2 cheap-predictor rank-correlation gate be run. Do not promote a B2-style
claim from the synthetic B1 rehearsal.

Do not collect or reduce GPU artifacts for SSQ-LR, HORN, or HBSM from this
handoff document. A future GPU run requires a separate preregistered runbook
after the local S1--S3, H1--H3, or B1--B3 gates pass.

## ThoughtFlow-FP8

Current manifest:
`experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`

Do not run GPU experiments for the current branch set: the consumed `rdu_topk`,
`psi_topk`, or `vwac_topk` branches.
The only valid next step is a new preregistration artifact for a genuinely
different utility family, followed by one one-shot evaluation on a fresh/larger
frozen sparse-cache surface.

## Local Readiness Recheck

Latest recorded result:
`experimental/local_readiness_recheck_20260507.md`.

```text
294 passed, 1 skipped, 2 warnings in 7.18s
```

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
