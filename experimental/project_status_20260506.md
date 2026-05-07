# Experimental Project Status

Date: 2026-05-06

This ledger is the sprint control plane for the relevant COLM/ICLR branches:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. It separates live gates,
watched branches, and killed branches so we do not keep spending time on
saturated ideas.

## Active Decision Surface

| Rank | Project | Readiness | Current story | Exact blocking gap | Next experiment |
|---:|---|---:|---|---|---|
| 1 | HybridKernel | 70% if GPU gate passes; 0% as local-only result | Boundary-fusion may recover avoidable attention to SSM overhead in hybrid models, but Mac work is saturated. The packet checker now handles batch>1 replay as per-sample prefill plus aggregate completion tokens, and the opt-in Triton CPU-backend correctness gate passes locally. | User-operated NVIDIA/vLLM Nsight packet with three distinct repeats, at least 3% recoverable gain, three same-shape same-family control rows, three same-shape cross-family falsification rows that stay below 3%, fixed-length replay with `ignore_eos`, and matching client replay logs for every metric model. | Run `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`; verify with `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`. |
| 2 | SSQ-LR | 15% positive method | Test whether recurrent SSM state in hybrid reasoners can go below FP16 with a stable quantization recipe. The current packet is a 288-row synthetic real-schema rehearsal, not model evidence; it exercises the real SSQ-LR checker through `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`. | Mac Gate S1 must show state-distribution heterogeneity on real hybrid SSM state dumps. The checker now requires every prompt/layer pair to cover prefill_end, 2k_or_end, 8k_or_end, and final_minus_128 buckets, SSM/Mamba layer-kind and recurrent-state tensor-kind labels, decision-grade summary fields, bootstrap-style prompt-level lower bounds, Holm-corrected two-sample distribution tests with a 1.25x effect-size floor, full 64-hex SHA provenance, and at least 12 prompts unless explicitly resource-limited and non-promotable. | Run `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md` Gate S1 on the smallest available hybrid state dumps. |
| 3 | HORN | 15% control branch | Test whether attention-to-SSM and SSM-to-attention boundaries have asymmetric outlier/noise propagation. The current packet is a 72-row synthetic real-schema H1a rehearsal, not model evidence; a single-model real screen is labeled H1a, not H1 promotion. | Mac Gate H1 must show consistent directional magnitude or kurtosis asymmetry across real boundary dumps; otherwise HORN stays a control inside SSQ-LR/HBSM. The checker now requires both boundary directions, decision-grade summary fields, per-prompt non-boundary controls paired through `matched_boundary_direction` and below the selected H1 threshold, and permuted controls paired by prompt, boundary, layer, norm positions, metric values, and an actual flipped `direction` label whose selected-direction effect is erased. | Run `experimental/horn/phase2/preregister_horn_20260506.md` H1a once shared dumps exist; promote to H1 only after cross-model consistency. |
| 4 | HBSM | 15% wounded branch | KL-Lens-like layer sensitivity is crowded; remaining wedge is frontier hybrid mechanism plus cheaper predictor. The current packet is a 504-row synthetic real-schema rehearsal, not model evidence; it exercises prompt-to-layer aggregation, B1 controls, summary recomputation, and per-prompt measured-drift top-decile derivation through `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`. | Gate B1 must replicate sensitivity heterogeneity on current hybrid reasoners, then B2 must show a cheaper predictor. The checker now scores only primary `boundary_only` rows after aggregating prompt rows to `(model_id, layer)`, derives top deciles from measured `kl_or_nll_drift`, rejects every prompt-row supplied flag mismatch, requires prompt-level coverage with boundary and non-boundary layers, exact aggregated top-decile cardinality, a non-enriched same-count random baseline, KL-style and activation/outlier comparator controls, finite metrics, and near-zero perturbation-off controls. | Run `experimental/hbsm/phase2/preregister_hbsm_20260506.md` after shared dumps exist. |
| 5 | ThoughtFlow-FP8 | 92% falsification paper; 0% positive method | The reusable contribution is the preregistered falsification ladder for sparse-cache signals. The draft now has protocol, RDU demotion, claim-boundary, related-work citation tables, and a tracked diagnostic packet locking stale-positive and negative conclusions with repo-root input hashes and clean-path provenance. | Paper polish only; no fifth signal unless a new preregistration and fresh surface exist. | Human review of `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`. |

## New Shared Infrastructure

Shared Mac-local utilities live in `experimental/shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `hybrid_architecture_maps.py`: config-derived explicit boundary maps for
  future real trace packets.
- `hybrid_model_eligibility.py`: metadata-only HF size/cache preflight for
  live hybrid targets.
- `hybrid_trace_plan.py`: deterministic SSQ-LR/HORN/HBSM capture plan from the
  frozen 12-prompt manifest and shared architecture maps. The current plan is
  `shared/results/hybrid_trace_plan_20260507/` with 5,184 SSQ-LR rows, 1,008
  HORN rows, and 1,554 HBSM rows. It is trace-plan-only and cannot promote any
  gate.
- `hybrid_trace_packet_builder.py`: converts future saved tensors into strict
  SSQ-LR/HORN real packets and resolves hook names sanitized by tensor-packet
  storage. Resource-limited input metadata now forces a
  `RESOURCE_LIMITED_NOT_PROMOTABLE_...` packet decision, even if the recomputed
  smoke-gate status would otherwise pass.
- `hybrid_gate_evaluators.py`: recomputes SSQ-LR S1, HORN H1, and HBSM B1
  decision fields from raw rows so summaries cannot be hand-filled. SSQ-LR now
  uses prompt-level bootstrap-style lower bounds plus Holm-corrected two-sample
  distribution tests, HORN gates on per-prompt non-boundary controls and
  actual-label-flipped permuted controls, and HBSM derives measured top-decile
  membership from primary-row drift after aggregation.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.
- `check_gate_packet.py`: generic synthetic-packet validator plus strict
  `--mode real --project ...` contracts for SSQ-LR, HORN, and HBSM, including a
  `SCHEMA_REHEARSAL_NOT_PROMOTABLE` path that exercises real schemas without
  allowing synthetic rows to promote a gate.
- `hybrid_trace_packet_runbook.md`: required schema for the first real
  SSQ-LR/HORN/HBSM trace packet.
- `prompts/hybrid_reasoning_smoke_12_20260506.jsonl`: frozen 12-prompt Mac
  gate smoke manifest, SHA-256
  `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

These are reference utilities for hypothesis gates only. They do not support
native GPU throughput, HBM, or production-packing claims.

## Synthetic Gate Packets

Synthetic packets now exist for the three Mac-gated hybrid-quantization
branches. These validate scripts, metrics, artifact shape, and pass/fail logic;
they do not promote any branch.

| Project | Packet | Decision | Key readout |
|---|---|---|---|
| SSQ-LR | `experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1/` | `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1` | 288 real-schema rows across 12 prompts, 6 layers, and 4 S1 buckets |
| HORN | `experimental/horn/phase2/results/horn_synthetic_h1/` | `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A` | 72 real-schema rows; selected ratio `4.044`, non-boundary control `1.042`, permuted control `0.247` |
| HBSM | `experimental/hbsm/phase2/results/hbsm_synthetic_b1/` | `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1` | 504 real-schema rows, 480 primary prompt rows, and 40 scoring layers after aggregation |

All three packets pass `experimental.shared.check_gate_packet`.

Each active project also has a reviewer-pack handoff under
`experimental/<project>/paper/reviewer_pack.md`. For SSQ-LR, HORN, and HBSM
these packs explicitly state that the current drafts are preregistration shells,
not method papers, until real S1--S3, H1--H3, or B1--B3 evidence exists.

The checker now has a stricter real-packet mode:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/results/<packet> --mode real --project <ssq_lr|horn|hbsm>
```

Real packets must include provenance, `summary.md`, matching `row_count`,
project-specific row schemas, required controls, admissible coverage,
`prompt_ids_hash`, `architecture_map_hash`, `trace_plan_hash`, and
decision-grade `summary.json` aggregates. Non-rehearsal real packets now verify `model_id` and
`architecture_map_hash` against
`shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`, not
just hash syntax, and require a hash for the trace-plan JSONL that drove row
capture. Synthetic-only real-schema rehearsals must set `schema_rehearsal:
true` and use a `SCHEMA_REHEARSAL_NOT_PROMOTABLE` decision. Resource-limited
real packets must use a
`RESOURCE_LIMITED_NOT_PROMOTABLE` decision and cannot promote a gate. The
stricter checks reject underspecified SSQ-LR packets without complete
prompt/layer S1 bucket matrices or effect-sized distribution shifts, HORN
packets without both boundary directions plus prompt-paired non-boundary and
metric-reused permuted controls, and HBSM packets without primary-row prompt
coverage, measured-drift top-decile agreement, true top-decile cardinality,
KL/outlier comparators, a true random baseline, finite sensitivity rows, and a
no-op perturbation control.

## Config-Only Architecture Maps

Config-only maps now exist for the local Granite and Qwen hybrid configs:

| Artifact | Use | Claim boundary |
|---|---|---|
| `experimental/shared/results/hybrid_architecture_maps_20260506/` | Provides explicit layer kinds, boundary IDs, direction counts, and config hashes for SSQ-LR/HORN/HBSM real trace packets. | Config provenance only; no activations, SSM state, quality, or GPU evidence. |
| `experimental/shared/results/hybrid_trace_plan_20260507/` | Enumerates exact SSQ-LR/HORN/HBSM trace rows to capture from frozen prompts and architecture maps. | Trace-plan-only; no activations, SSM state, sensitivity, quality, or GPU evidence. |

## HybridKernel Packet Hardening

The native profiler packet now requires per-row reduction provenance:
`row_role`, `control_family`, `boundary_direction`, `nsys_artifact`,
`nsys_artifact_sha256`, `ncu_artifact`, `ncu_artifact_sha256`, `kernel_names`,
`boundary_indices`, `time_window_ms`, `recoverable_fraction_basis`,
`reduction_command`, and `reduction_notes`. The checker also cross-checks model identity across
`profile_scope.json`, client replay logs, metric rows, and the architecture map.
It now has an explicit `no_boundary_signal_kill` packet mode so a clean
Nsight-Systems negative run can be reviewable without inventing an Nsight
Compute target.
Per-row `nsys_artifact` and `ncu_artifact` fields must resolve to reviewable
files inside the run packet with valid Nsight extensions. This prevents a
reduced metric row from citing a missing or external artifact.
The profiler reducer now refuses prototype promotion unless the same metric
packet includes at least three matched same-family control rows and three
cross-family falsification rows on the same request/runtime shape, and those
controls stay below the 3% recoverable-gain gate. Same-family controls may be
matched segments or same-family control models. A primary-only packet that
clears 3%, or a packet whose controls reproduce the same signal, remains
audit-only. The reducer rejects impossible local timings, and the artifact
checker rejects repeated-row packets that reuse the same Nsight artifacts, lack
token-counted client replay JSON, mismatch replay prompt/decode/request shape,
reuse time windows, or cite artifacts whose SHA-256 digest does not match.
The optional Triton CPU-backend correctness test passes on this Mac under
`HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 TRITON_CPU_BACKEND=1`, but it remains a
correctness-only diagnostic.

## Hybrid Model Eligibility

Metadata-only HF preflight found public live targets but no repo-local cached
weights:

| Model | Safetensors GB | Local weights | Decision |
|---|---:|---|---|
| `ibm-granite/granite-4.0-h-tiny` | 12.93 | no | `BLOCKED_NOT_CACHED` |
| `ibm-granite/granite-4.0-h-small` | 59.99 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| `ibm-granite/granite-4.0-h-small-FP8` | 31.19 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151.49 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |

Artifact: `experimental/shared/results/hybrid_model_eligibility_20260506/`.
Large rows remain GPU-sized even though the immediate blocker is the missing
repo-local weight cache.

## Killed Branches

Top-level killed markers:

| Marker | Why killed | Salvage value |
|---|---|---|
| `KILLED_sinkaware_static_prior` | Exact static sink reuse would ignore query-dependent sink logits. | Sink position statistics remain reusable for precision-protection diagnostics, not sink-logit skipping. |
| `KILLED_sinkaware_systems_framing` | Four sink tokens are too small a compute wedge for an attention-kernel speed paper. | Rank-2 sink predictors remain diagnostic/attention-theory evidence. |
| `KILLED_thoughtflow_fp8_positive_method` | RDU/PSI/VWAC sparse-cache signals failed reproduction or fresh-surface gates. | Falsification methodology and artifact discipline are reusable. |
| `KILLED_anchorspec` | Early-exit/speculation lane is too crowded and not supported by current evidence. | Sink-mass telemetry can remain a feature in future diagnostics. |
| `KILLED_phasequant` | Depends on fragile phase classification from the stopped ThoughtFlow branch. | Phase labels can be used descriptively, not as a live method. |
| `KILLED_moe_phase_routing` | Too crowded and not backed by current routing-specific evidence. | Benchmark/routing notes can inform later literature review. |

## Next Exact Gates

1. HybridKernel: run the native NVIDIA/vLLM profiler packet; this is the only
   current branch that can resolve with one GPU experiment.
2. SSQ-LR: dump or reuse hybrid SSM state packets and test state-distribution
   heterogeneity before any quantization recipe is tuned.
3. HORN: use the same dumps to measure directional boundary outlier asymmetry;
   keep it as a control unless H1 passes.
4. HBSM: only run after shared dumps exist; kill if cheap predictors do not
   correlate with forward sensitivity.
5. ThoughtFlow: rewrite the existing paper around falsification methodology; no
   new experiments in the current branch.
