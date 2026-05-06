# Experimental Project Status

Date: 2026-05-06

This ledger is the sprint control plane for current COLM/ICLR experimental
branches. It separates live gates, watched branches, and killed branches so we
do not keep spending time on saturated ideas.

## Active Decision Surface

| Rank | Project | Readiness | Current story | Exact blocking gap | Next experiment |
|---:|---|---:|---|---|---|
| 1 | HybridKernel | 70% if GPU gate passes; 0% as local-only result | Boundary-fusion may recover avoidable attention to SSM overhead in hybrid models, but Mac work is saturated. | User-operated NVIDIA/vLLM Nsight packet with three clean repeats and at least 3% recoverable gain. | Run `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`; verify with `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`. |
| 2 | SinkKV | 20% positive method | Sink positions may deserve precision protection during KV quantization; deterministic no-download probe passed only as a policy sanity check. | Real cached Q/K/V gate must show quality recovery at matched memory on at least two model/length surfaces. | Run first real activation-dump gate using `experimental/sinkkv/phase2/preregister_sink_protected_kv_20260506.md`. |
| 3 | SSQ-LR | 10% positive method | Test whether recurrent SSM state in hybrid reasoners can go below FP16 with a stable quantization recipe. | Mac Gate S1 must show state-distribution heterogeneity worth quantizing specially. | Run `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md` Gate S1 on the smallest available hybrid state dumps. |
| 4 | HORN | 10% control branch | Test whether attention-to-SSM and SSM-to-attention boundaries have asymmetric outlier/noise propagation. | Mac Gate H1 must show directional magnitude or kurtosis asymmetry; otherwise HORN stays a control inside SSQ-LR/HBSM. | Run `experimental/horn/phase2/preregister_horn_20260506.md` Gate H1 once shared dumps exist. |
| 5 | HBSM | 10% wounded branch | KL-Lens-like layer sensitivity is crowded; remaining wedge is frontier hybrid mechanism plus cheaper predictor. | Gate B1 must replicate sensitivity heterogeneity on current hybrid reasoners, then B2 must show a cheaper predictor. | Run `experimental/hbsm/phase2/preregister_hbsm_20260506.md` only after shared dumps are available. |
| 6 | ThoughtFlow-FP8 | 85% falsification paper; 0% positive method | The reusable contribution is the preregistered falsification ladder for sparse-cache signals. | Paper framing, not experiments; no fifth signal unless a new preregistration and fresh surface exist. | Reframe `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex` as a falsification-methodology note. |

## New Shared Infrastructure

Shared Mac-local utilities live in `experimental/shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.

These are reference utilities for hypothesis gates only. They do not support
native GPU throughput, HBM, or production-packing claims.

## SinkKV First Result

`experimental/sinkkv/phase2/sinkkv_deterministic_probe.py` generated:

- packet: `experimental/sinkkv/phase2/results/sinkkv_deterministic_probe/`
- decision: `SYNTHETIC_PASS_REAL_DUMPS_NEXT`
- uniform MXFP4 output rel-L2: `0.097249`
- sink-protected matched-budget output rel-L2: `0.068215`
- recent-protected matched-budget output rel-L2: `0.112774`

Interpretation: the policy and byte accounting are plausible enough to justify
the first real cached Q/K/V dump. It is synthetic-only and is not GPU speed,
benchmark accuracy, or evidence that query-dependent `QK_sink` can be skipped.

## Killed Branches

Top-level killed markers:

| Marker | Why killed | Salvage value |
|---|---|---|
| `KILLED_sinkaware_static_prior` | Exact static sink reuse would ignore query-dependent sink logits. | Sink position statistics motivate SinkKV protection instead of sink-logit skipping. |
| `KILLED_sinkaware_systems_framing` | Four sink tokens are too small a compute wedge for an attention-kernel speed paper. | Rank-2 sink predictors remain diagnostic/attention-theory evidence. |
| `KILLED_thoughtflow_fp8_positive_method` | RDU/PSI/VWAC sparse-cache signals failed reproduction or fresh-surface gates. | Falsification methodology and artifact discipline are reusable. |
| `KILLED_anchorspec` | Early-exit/speculation lane is too crowded and not supported by current evidence. | Sink-mass telemetry can remain a feature in future diagnostics. |
| `KILLED_phasequant` | Depends on fragile phase classification from the stopped ThoughtFlow branch. | Phase labels can be used descriptively, not as a live method. |
| `KILLED_moe_phase_routing` | Too crowded and not backed by current routing-specific evidence. | Benchmark/routing notes can inform later literature review. |

## Next Exact Gates

1. HybridKernel: run the native NVIDIA/vLLM profiler packet; this is the only
   current branch that can resolve with one GPU experiment.
2. SinkKV: run the first real cached Q/K/V gate with the frozen policy from the
   deterministic probe; promote only if the real gate recovers at least half of
   the uniform-FP4 quality gap at matched memory.
3. SSQ-LR: dump or reuse hybrid SSM state packets and test state-distribution
   heterogeneity before any quantization recipe is tuned.
4. HORN: use the same dumps to measure directional boundary outlier asymmetry;
   keep it as a control unless H1 passes.
5. HBSM: only run after shared dumps exist; kill if cheap predictors do not
   correlate with forward sensitivity.
6. ThoughtFlow: rewrite the existing paper around falsification methodology; no
   new experiments in the current branch.
