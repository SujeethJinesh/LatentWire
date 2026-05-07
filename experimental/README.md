# Experimental Project Control Plane

Date: 2026-05-07

This folder currently tracks five relevant COLM/ICLR branches only:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. The control objective is
to finish every Mac-local artifact that can be finished before NVIDIA GPU time,
then move only surviving branches to the 5090 gate.

The current sprint ledger is `project_status_20260506.md`.

## Active Branches

| Project | Current status | Best local evidence | Blocking gap |
|---|---|---|---|
| `hybridkernel/` | Mac-saturated GPU handoff | Architecture/runtime audit, threshold model, exact-token fixed-request vLLM driver, profiler packet verifier, batch-aware client replay checker, Triton interpreter and opt-in CPU-backend toy-kernel tests | User-operated NVIDIA/vLLM Nsight packet with three distinct repeats, same-family control, cross-family falsification, and at least 3% recoverable boundary overhead |
| `ssq_lr/` | Mac gate scaffolded | Non-promoting 288-row synthetic S1 real-schema rehearsal passes the real SSQ-LR checker | Real hybrid SSM state dumps showing distribution heterogeneity with complete prompt/layer bucket coverage |
| `horn/` | Mac gate scaffolded | Non-promoting 72-row synthetic H1a real-schema rehearsal passes the real HORN checker; evaluator now reports single-model screens as H1a only, pairs non-boundary controls through `matched_boundary_direction`, and uses prompt-cluster bootstrap lower bounds | Real attention-to-SSM / SSM-to-attention boundary dumps showing asymmetry with per-prompt non-boundary controls, prompt-cluster IDs, and actual-label-flipped permuted controls across enough models for H1 promotion |
| `hbsm/` | Mac gate scaffolded; novelty is narrow | Non-promoting 720-row synthetic B1 real-schema rehearsal validates prompt-to-layer aggregation, required controls, and per-prompt measured-drift top-decile derivation | Real layer sensitivity packet with every primary prompt-row top-decile flag matching measured drift plus random/KL/outlier controls on current hybrid reasoners |
| `thoughtflow_fp8/` | Positive method stopped; falsification paper active | Preregistered sparse-cache signal ladder, oracle/headroom diagnostics, fresh-surface failures, provenance-locked diagnostic packet with upstream input hashes and clean-path generation guard | Paper-only camera-ready polish |

## Shared Infrastructure

Shared Mac-local utilities live in `shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `hybrid_architecture_maps.py`: config-derived explicit boundary maps and
  negative-control rows for real trace packet provenance.
- `hybrid_model_eligibility.py`: metadata-only HF size/cache preflight for the
  live hybrid targets.
- `hybrid_local_capture_preflight.py`: local environment/cache/dependency
  preflight for the first real SSQ-LR/HORN/HBSM captures. Current artifact:
  `shared/results/hybrid_local_capture_preflight_20260507/`, decision
  `LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE` because `mamba_ssm` is absent and
  the active hybrid weights are not fully cached locally.
- `hybrid_trace_plan.py`: deterministic SSQ-LR/HORN/HBSM row plan from the
  frozen prompt manifest and architecture maps.
- `hybrid_trace_capture_manifest.py`: fill-in metadata templates for real
  tensor/sensitivity captures.
- `hybrid_trace_packet_builder.py`: converts future saved tensors into strict
  SSQ-LR/HORN real packets and resolves hook names sanitized by tensor-packet
  storage.
- `hybrid_gate_evaluators.py`: recomputes SSQ-LR S1, HORN H1, and HBSM B1
  decision aggregates from raw packet rows.
- `followup_gate_contracts.py`: executable S2/S3, H2/H3, and B2/B3 contracts.
  These now require SSQ-LR's full baseline set, HORN H2 three-seed paired
  direction units, and HORN H3 pure controls below the preregistered null
  threshold before any follow-up packet can pass.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.
- `check_gate_packet.py`: generic result-packet validator with strict real
  SSQ-LR/HORN/HBSM packet contracts plus a non-promoting real-schema rehearsal
  path for synthetic-only scaffolds.
- `hybrid_trace_packet_runbook.md`: schema for the first real shared trace
  packet used by SSQ-LR, HORN, and HBSM.
- `prompts/hybrid_reasoning_smoke_12_20260506.jsonl`: frozen 12-prompt
  Mac-gate smoke surface, SHA-256
  `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

These utilities support Mac-local hypothesis gates. They do not support native
GPU throughput, HBM, latency, energy, or production-packing claims.

Current config-only architecture packet:
`shared/results/hybrid_architecture_maps_20260506/`.

Current metadata-only model eligibility packet:
`shared/results/hybrid_model_eligibility_20260506/`.

Current local capture preflight packet:
`shared/results/hybrid_local_capture_preflight_20260507/`.

## Next Exact Gates

1. **HybridKernel**: run the 5090 profiler packet in
   `hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`, then verify with
   `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`.
2. **SSQ-LR**: rerun `python -m experimental.shared.hybrid_local_capture_preflight`
   after installing local capture dependencies or moving to a GPU node, then
   produce the first real hybrid SSM state packet and run Gate S1 from
   `ssq_lr/phase2/preregister_ssq_lr_20260506.md`.
3. **HORN**: use the same local-capture preflight, then run Gate H1 on the
   same real trace packet once boundary activations are available.
4. **HBSM**: use the same local-capture preflight, then run Gate B1 after the
   shared trace packet exists; only after real
   B1 sensitivity heterogeneity is established, run the B2 cheap-predictor
   rank-correlation gate.
5. **ThoughtFlow-FP8**: continue paper reframing and citation/table polish; do
   not run a new signal without a fresh preregistered surface.

## Cost Discipline

Cheap gates come first. Failure at a Macbook phase means stop or fold the result
into a stronger branch before spending GPU time. HybridKernel is the exception:
its Mac work is saturated, so the next discriminative bit is the native GPU
profiler packet.

For SSQ-LR, HORN, and HBSM, the first live trace packet builders/checkers are
S1, H1a/H1, and B1 respectively. The S2/S3, H2/H3, and B2/B3 follow-up
contracts are now executable through `shared/followup_gate_contracts.py`, but
they have no model packets yet and cannot be cited as current evidence.

## Killed Marker Convention

`KILLED_*` folders mark consumed sub-branches and dead framings. They do not
mean every artifact in the source project is useless; each marker README records
what was tried, why it died, and what remains salvageable.
