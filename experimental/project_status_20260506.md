# Experimental Project Status

Date: 2026-05-06

This ledger is the sprint control plane for the relevant COLM/ICLR branches:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. It separates live gates,
watched branches, and killed branches so we do not keep spending time on
saturated ideas.

## Active Decision Surface

| Rank | Project | Readiness | Current story | Exact blocking gap | Next experiment |
|---:|---|---:|---|---|---|
| 1 | HybridKernel | 70% if GPU gate passes; 0% as local-only result | Boundary-fusion may recover avoidable attention to SSM overhead in hybrid models, but Mac work is saturated. | User-operated NVIDIA/vLLM Nsight packet with three distinct repeats, at least 3% recoverable gain, same-family control rows, and cross-family falsification rows. | Run `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`; verify with `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`. |
| 2 | SSQ-LR | 15% positive method | Test whether recurrent SSM state in hybrid reasoners can go below FP16 with a stable quantization recipe. Synthetic S1 packet and explicit architecture-map packet validate artifact mechanics/provenance only. | Mac Gate S1 must show state-distribution heterogeneity on real hybrid SSM state dumps. The checker now requires every prompt/layer pair to cover prefill_end, 2k_or_end, 8k_or_end, and final_minus_128 buckets, decision-grade summary fields, and at least 12 prompts unless the packet is explicitly resource-limited and non-promotable. | Run `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md` Gate S1 on the smallest available hybrid state dumps. |
| 3 | HORN | 15% control branch | Test whether attention-to-SSM and SSM-to-attention boundaries have asymmetric outlier/noise propagation. Synthetic H1 packet and explicit boundary maps validate artifact mechanics/provenance only. | Mac Gate H1 must show directional magnitude or kurtosis asymmetry on real boundary dumps; otherwise HORN stays a control inside SSQ-LR/HBSM. The checker now requires both boundary directions, decision-grade summary fields, and permuted controls paired by prompt, boundary, layer, and norm positions. | Run `experimental/horn/phase2/preregister_horn_20260506.md` Gate H1 once shared dumps exist. |
| 4 | HBSM | 15% wounded branch | KL-Lens-like layer sensitivity is crowded; remaining wedge is frontier hybrid mechanism plus cheaper predictor. Synthetic B1/B2 packet and fixed boundary flags validate artifact mechanics/provenance only. | Gate B1 must replicate sensitivity heterogeneity on current hybrid reasoners, then B2 must show a cheaper predictor. The checker now requires true/false boundary flags, finite metrics, decision-grade summary fields, matched random/top-decile counts, and a near-zero perturbation-off row. | Run `experimental/hbsm/phase2/preregister_hbsm_20260506.md` after shared dumps exist. |
| 5 | ThoughtFlow-FP8 | 90% falsification paper; 0% positive method | The reusable contribution is the preregistered falsification ladder for sparse-cache signals. The draft now has protocol, RDU demotion, claim-boundary, related-work citation tables, and saved-artifact tests locking the current negative conclusions. | Paper polish only; no fifth signal unless a new preregistration and fresh surface exist. | Human review of `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`. |

## New Shared Infrastructure

Shared Mac-local utilities live in `experimental/shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `hybrid_architecture_maps.py`: config-derived explicit boundary maps for
  future real trace packets.
- `hybrid_model_eligibility.py`: metadata-only HF size/cache preflight for
  live hybrid targets.
- `hybrid_trace_packet_builder.py`: converts future saved tensors into strict
  SSQ-LR/HORN real packets.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.
- `check_gate_packet.py`: generic synthetic-packet validator plus strict
  `--mode real --project ...` contracts for SSQ-LR, HORN, and HBSM.
- `hybrid_trace_packet_runbook.md`: required schema for the first real
  SSQ-LR/HORN/HBSM trace packet.

These are reference utilities for hypothesis gates only. They do not support
native GPU throughput, HBM, or production-packing claims.

## Synthetic Gate Packets

Synthetic packets now exist for the three Mac-gated hybrid-quantization
branches. These validate scripts, metrics, artifact shape, and pass/fail logic;
they do not promote any branch.

| Project | Packet | Decision | Key readout |
|---|---|---|---|
| SSQ-LR | `experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1/` | `SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT` | late/early max-abs ratio `8.461` |
| HORN | `experimental/horn/phase2/results/horn_synthetic_h1/` | `SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT` | directional max ratio `3.775` |
| HBSM | `experimental/hbsm/phase2/results/hbsm_synthetic_b1/` | `SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT` | kurtosis-vs-sensitivity Spearman rho `0.657` |

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
`prompt_ids_hash`, `architecture_map_hash`, and decision-grade `summary.json`
aggregates. Resource-limited real packets must use a
`RESOURCE_LIMITED_NOT_PROMOTABLE` decision and cannot promote a gate. The
stricter checks reject underspecified SSQ-LR packets without complete
prompt/layer S1 bucket matrices, HORN packets without both boundary directions
and prompt-paired permuted controls, and HBSM packets without finite
sensitivity rows plus a no-op perturbation control.

## Config-Only Architecture Maps

Config-only maps now exist for the local Granite and Qwen hybrid configs:

| Artifact | Use | Claim boundary |
|---|---|---|
| `experimental/shared/results/hybrid_architecture_maps_20260506/` | Provides explicit layer kinds, boundary IDs, direction counts, and config hashes for SSQ-LR/HORN/HBSM real trace packets. | Config provenance only; no activations, SSM state, quality, or GPU evidence. |

## HybridKernel Packet Hardening

The native profiler packet now requires per-row reduction provenance:
`row_role`, `control_family`, `boundary_direction`, `nsys_artifact`,
`ncu_artifact`, `kernel_names`, `boundary_indices`, `time_window_ms`, and
`reduction_notes`. The checker also cross-checks model identity across
`profile_scope.json`, client replay logs, metric rows, and the architecture map.
It now has an explicit `no_boundary_signal_kill` packet mode so a clean
Nsight-Systems negative run can be reviewable without inventing an Nsight
Compute target.
Per-row `nsys_artifact` and `ncu_artifact` fields must resolve to reviewable
files inside the run packet with valid Nsight extensions. This prevents a
reduced metric row from citing a missing or external artifact.
The profiler reducer now refuses prototype promotion unless the same metric
packet includes the required same-family control and cross-family falsification
row roles; a primary-only packet that clears 3% remains audit-only.
The optional Triton CPU-backend correctness test passes on this Mac when
Homebrew GCC library paths are exported, but it remains a correctness-only
diagnostic.

## Hybrid Model Eligibility

Metadata-only HF preflight found public live targets but no repo-local cached
weights:

| Model | Safetensors GB | Local weights | Decision |
|---|---:|---|---|
| `ibm-granite/granite-4.0-h-tiny` | 12.93 | no | `BLOCKED_NOT_CACHED` |
| `ibm-granite/granite-4.0-h-small` | 59.99 | no | `BLOCKED_NOT_CACHED` |
| `ibm-granite/granite-4.0-h-small-FP8` | 31.19 | no | `BLOCKED_NOT_CACHED` |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151.49 | no | `BLOCKED_NOT_CACHED` |

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
