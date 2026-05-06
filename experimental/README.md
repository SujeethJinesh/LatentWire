# Experimental Project Control Plane

Date: 2026-05-06

This folder currently tracks five relevant COLM/ICLR branches only:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. The control objective is
to finish every Mac-local artifact that can be finished before NVIDIA GPU time,
then move only surviving branches to the 5090 gate.

The current sprint ledger is `project_status_20260506.md`.

## Active Branches

| Project | Current status | Best local evidence | Blocking gap |
|---|---|---|---|
| `hybridkernel/` | Mac-saturated GPU handoff | Architecture/runtime audit, threshold model, fixed-request vLLM driver, profiler packet verifier, Triton interpreter toy-kernel tests | User-operated NVIDIA/vLLM Nsight packet with three clean repeats and at least 3% recoverable boundary overhead |
| `ssq_lr/` | Mac gate scaffolded | Synthetic S1 packet validates metrics, decision logic, and artifact schema | Real hybrid SSM state dumps showing distribution heterogeneity |
| `horn/` | Mac gate scaffolded | Synthetic H1 packet validates directional boundary metrics and artifact schema | Real attention-to-SSM / SSM-to-attention boundary dumps showing asymmetry |
| `hbsm/` | Mac gate scaffolded; novelty is narrow | Synthetic B1/B2 packet validates sensitivity-ranking and cheap-predictor mechanics | Real layer sensitivity packet on current hybrid reasoners |
| `thoughtflow_fp8/` | Positive method stopped; falsification paper active | Preregistered sparse-cache signal ladder, oracle/headroom diagnostics, fresh-surface failures | Paper-only camera-ready polish |

## Shared Infrastructure

Shared Mac-local utilities live in `shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `hybrid_architecture_maps.py`: config-derived explicit boundary maps and
  negative-control rows for real trace packet provenance.
- `hybrid_model_eligibility.py`: metadata-only HF size/cache preflight for the
  live hybrid targets.
- `hybrid_trace_packet_builder.py`: converts future saved tensors into strict
  SSQ-LR/HORN real packets.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.
- `check_gate_packet.py`: generic result-packet validator.
- `hybrid_trace_packet_runbook.md`: schema for the first real shared trace
  packet used by SSQ-LR, HORN, and HBSM.

These utilities support Mac-local hypothesis gates. They do not support native
GPU throughput, HBM, latency, energy, or production-packing claims.

Current config-only architecture packet:
`shared/results/hybrid_architecture_maps_20260506/`.

Current metadata-only model eligibility packet:
`shared/results/hybrid_model_eligibility_20260506/`.

## Next Exact Gates

1. **HybridKernel**: run the 5090 profiler packet in
   `hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`, then verify with
   `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`.
2. **SSQ-LR**: produce the first real hybrid SSM state packet and run Gate S1
   from `ssq_lr/phase2/preregister_ssq_lr_20260506.md`.
3. **HORN**: run Gate H1 on the same real trace packet once boundary
   activations are available.
4. **HBSM**: run Gate B1 after the shared trace packet exists; continue only if
   the cheaper predictor has nontrivial rank correlation with forward
   sensitivity.
5. **ThoughtFlow-FP8**: continue paper reframing and citation/table polish; do
   not run a new signal without a fresh preregistered surface.

## Cost Discipline

Cheap gates come first. Failure at a Macbook phase means stop or fold the result
into a stronger branch before spending GPU time. HybridKernel is the exception:
its Mac work is saturated, so the next discriminative bit is the native GPU
profiler packet.

## Killed Marker Convention

`KILLED_*` folders mark consumed sub-branches and dead framings. They do not
mean every artifact in the source project is useless; each marker README records
what was tried, why it died, and what remains salvageable.
