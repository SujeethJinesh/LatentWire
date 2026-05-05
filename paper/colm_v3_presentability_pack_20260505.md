# COLM_v3 Presentability Pack

- date: 2026-05-05
- purpose: one-stop human-review pointer pack for the main COLM_v3 paper and
  three experimental systems spinouts

## Main COLM_v3 Paper

| Artifact | Path | Status |
|---|---|---|
| PDF for human review | `colm_final/paper/latentwire_colm2026.pdf` | rebuilt 2026-05-05; ready for human review |
| TeX source | `colm_final/paper/latentwire_colm2026.tex` | integrated paper source |
| completion audit | `paper/latentwire_colm_v3_completion_audit_20260505.md` | current readiness and remaining work |
| reviewer response memo | `paper/latentwire_colm_v3_reviewer_response_20260505.md` | overclaim fixes and reviewer-risk response |
| review packet | `paper/latentwire_colm_v3_review_packet_20260505.md` | evidence/tables/claims packet |
| aggregate pkt-src CI | `results/source_private_colm_acceptance_baselines_20260502/aggregate_source_index_ci.md` | reviewer-requested clustered source-index audit |
| camera-ready figure script | `scripts/build_latentwire_colm_v3_camera_ready_figures.py` | regenerates the main source-index-aware accuracy figure |
| side-project review | `paper/experimental_projects_workshop_review_20260505.md` | ranking, remaining benchmarks, and completion gates |

## Current COLM_v3 Decision

COLM_v3 is ready for human review as a disciplined workshop paper, not as a
claim that LatentWire beats dense cache-transfer systems. The defensible claim
is that LatentWire gives a practical byte-scale, content-private packet protocol
and evaluation framework with strict destructive controls, narrow packet utility,
and explicit systems/byte accounting.

Remaining before submission:

1. Human copyedit.
2. Page-budget and table-width check.
3. Final double-blind compliance pass.
4. Confirm workshop-specific formatting once the target workshop is selected.

## Side Project Paper Links

| Project | Paper artifact | Current status |
|---|---|---|
| HybridKernel | `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf` | weakly alive as profiler-driven systems branch |
| SinkAware | `experimental/sinkaware/paper/sinkaware_colm2026.pdf` | alive as approximate low-rank fixed-sink prior |
| SinkAware reviewer pack | `experimental/sinkaware/paper/reviewer_pack.md` | presentable narrow review packet |
| ThoughtFlow-FP8 | `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf` | mixed; held-out policy sweep ties R-KV-like within 0.03 NLL |

## Side Project Data Links

| Project | Key data / runbook | Current readout |
|---|---|---|
| HybridKernel | `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md` | native NVIDIA/vLLM profiling is the only useful next gate |
| HybridKernel | `experimental/hybridkernel/phase2/profiler_driver.py` | Mac dry-run validates fixed-request driver plumbing |
| HybridKernel | `experimental/hybridkernel/phase2/profiler_analysis_gate.md` | pre-registered native profiler promote/kill parser; pending GPU data |
| SinkAware | `experimental/sinkaware/phase2/real_qk_sink_softmax_output_probe.md` | rank-2 improves output drift vs position-only on distilgpt2 traces |
| SinkAware | `experimental/sinkaware/phase3/approx_sink_attention_reference.md` | CPU reference for approximate sink-logit attention operator |
| SinkAware | `experimental/sinkaware/phase2/gpu_gate_runbook.md` | native benchmark plan for exact vs approximate sink handling |
| ThoughtFlow-FP8 | `experimental/thoughtflow_fp8/phase2/perplexity_impact_proxy.md` | ThoughtFlow-saliency-recent NLL 3.434 nearly ties but still loses to R-KV-like 3.419 |
| ThoughtFlow-FP8 | `experimental/thoughtflow_fp8/phase2/policy_sweep.md` | train-selected policy ties R-KV-like on held-out traces: 3.480 vs 3.482 NLL |

## Next Exact Gates

| Project | Gate | Promotion threshold |
|---|---|---|
| COLM_v3 | human review pass | no unsupported claims, readable tables, consistent abstract/intro/limitations |
| HybridKernel | run native vLLM/Nsight boundary profiling | distinct boundary overhead with credible >=3% path |
| SinkAware | native GPU gate from the runbook | rank-2 preserves output quality while beating exact sink QK/decomposition cost |
| ThoughtFlow-FP8 | sharper hidden/KV saliency policy | beats R-KV-like retained-prefix proxy on matched-budget continuation NLL |

## Camera-Ready Reviewer Fixes Now Integrated

- Source-index now appears in the main accuracy figure and caption.
- The main text reports aggregate packet-minus-source-index clustered CIs.
- The same-budget structured text baseline is defined separately from explicit
  answer-label/source-index baselines.
- The method section includes a reconstruction-oriented algorithm table.
- The validation-incomplete Qwen2.5-1.5B row is removed from the main claim.
- The score-sketch diagnostic is reported only as a validation boundary.

## Claim Guardrails

Allowed:

- COLM_v3: byte-scale packet protocol and destructive-control evaluation
  framework with narrow controlled utility.
- HybridKernel: profiler-ready hypothesis, no speed claim.
- SinkAware: approximate low-rank sink-logit branch alive for one native GPU
  gate, no exact static-reuse claim.
- ThoughtFlow-FP8: negative/mixed evidence plus a partial policy improvement,
  no paper-ready positive method.

Not allowed:

- LatentWire beats C2C/KVComm on raw accuracy, latency, HBM, energy, or
  throughput.
- Side projects have native systems wins before NVIDIA profiling.
- ThoughtFlow-FP8 is revived as a positive method while R-KV-like still wins the
  matched-budget proxy.
