# Experimental Project Control Plane

Date: 2026-05-07

This folder currently tracks five relevant COLM/ICLR branches only:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. The control objective is
to finish every Mac-local artifact that can be finished before NVIDIA GPU time,
then move only surviving branches to the 5090 gate.
For SSQ-LR, HORN, and HBSM, the current decision is stronger: they are killed
as active COLM positive-method branches under their current preregistered
hypotheses. "Finish" now means preserving stop artifacts, documentation, and
validator checks only unless a new preregistration explicitly reopens a branch.

Current live surface:

- `hybridkernel/`: alive positive-method branch, GPU-gated.
- `thoughtflow_fp8/`: alive paper-only falsification-methodology branch.
- `KILLED_ssq_lr_cross_model_transfer/`: SSQ-LR stopped by S3 transfer failure.
- `KILLED_horn_directional_noise_propagation/`: HORN stopped by weak H2.
- `KILLED_hbsm_sensitivity_heterogeneity/`: HBSM stopped by B1 failure.

The current sprint ledger is `project_status_20260506.md`.
The latest full local readiness recheck is
`local_readiness_recheck_20260507.md` (`303 passed, 1 skipped, 2 warnings`).

## Current Branch Ledger

| Project | Current status | Best local evidence | Blocking gap |
|---|---|---|---|
| `hybridkernel/` | Mac-saturated GPU handoff | Architecture/runtime audit, threshold model, exact-token fixed-request vLLM driver, profiler packet verifier, batch-aware client replay checker with prompt/payload hashes, immutable model-provenance attestation, CUDA-graph-matched control matrix checks, mandatory reduction-input manifest checker, quality-smoke artifact checker, Triton interpreter and opt-in CPU-backend toy-kernel tests | User-operated NVIDIA/vLLM Nsight packet with three distinct repeats, same-family control, cross-family falsification, row-level reduction provenance, and at least 3% recoverable boundary overhead |
| `ssq_lr/` | **KILLED as active COLM branch**; diagnostic scaffold only | Non-promoting 288-row synthetic S1 rehearsal passes the real checker; held-out S1b and S2 scouts were useful, but the frozen `0,30` recipe fails S3 transfer to Granite 350M. Layer-0 rescue diagnostics also fail two-model S3 because Granite Tiny and Granite 350M prefer different frozen recipes. Kill marker: `KILLED_ssq_lr_cross_model_transfer/` | No GPU work. Reopen only with a new preregistered recipe/layer rule on a fresh surface |
| `horn/` | **KILLED as active COLM branch**; negative/control scaffold only | H1a real screen is weak (`1.06` with low `1.06`), and H2 fails under the signed-direction contract: directional drift ratio `1.037`, lower `0.324`, support `0.5`, paired units `6/6`, hook-off max delta `0.0`. Kill marker: `KILLED_horn_directional_noise_propagation/` | No GPU work. Reopen only with new preregistered full H2/H3 scope and a concrete reversal rationale |
| `hbsm/` | **KILLED as active COLM branch**; negative/control scaffold only | B1 scouts fail and point the wrong way: one-prompt Spearman `-0.476`; two-prompt Fisher p `1.0`, boundary top-decile count `0`, cheap-predictor Spearman `-0.667`. Kill marker: `KILLED_hbsm_sensitivity_heterogeneity/` | No GPU work. Reopen only with a new preregistered narrower mechanism hypothesis |
| `thoughtflow_fp8/` | Positive method stopped; falsification paper active | Preregistered sparse-cache signal ladder, oracle/headroom diagnostics, fresh-surface failures, provenance-locked diagnostic packet with local-workspace input hashes, clean-path generation guard, and `.debug` rebuild instructions | Paper-only camera-ready polish |

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
  `LOCAL_CAPTURE_READY_NOT_EVIDENCE` because Granite Tiny weights are cached
  locally and its native `transformers` hybrid model class is available.
  Granite Small and Qwen3-Next remain GPU-sized or uncached. Missing
  `mamba_ssm` and `vllm` packages are optional fast-path gaps here, not hard
  blockers for local Transformers captures.
- `hybrid_transformers_smoke_probe.py`: resource-limited local execution probe
  for cached hybrid models. Current artifact:
  `shared/results/hybrid_transformers_smoke_probe_20260507/`, decision
  `RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`; it loaded Granite Tiny,
  ran an 8-token CPU forward, and observed 36 recurrent-state cache layers and
  4 attention-cache layers. This is execution plumbing evidence only, not
  SSQ-LR/HORN/HBSM gate evidence.
- `hybrid_manifest_local_capture_runner.py`: manifest-driven resource-limited
  local capture runner. Current artifact:
  `shared/results/hybrid_manifest_local_capture_20260507/`, decision
  `RESOURCE_LIMITED_CAPTURE_PACKETS_WRITTEN_NOT_PROMOTABLE`; it produced
  checker-passing non-promoting Granite Tiny SSQ-LR and HORN packets from one
  shared model load. The older combined packet's SSQ-LR row reused one final
  cache state and failed S1 with ratio `1.0`; the current dedicated SSQ-LR
  bucket-truncated packet is
  `shared/results/ssq_lr_local_bucket_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`, selected ratio
  `3.29`. The current multilayer SSQ-LR smoke packet is
  `shared/results/ssq_lr_local_multilayer_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`; it covers one
  prompt and four layers, with only layer 0 passing (`3.29x` max-abs ratio)
  while layers 1-3 stay below the preregistered S1 requirement. Only the compact
  multilayer readout is tracked; regenerate the full tensor packet before
  rerunning its checker. The metrics-only all-recurrent-layer scout is
  `shared/results/ssq_lr_all_layer_scout_20260507/`, decision
  `RESOURCE_LIMITED_ALL_LAYER_SCOUT_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`;
  it scanned 36 recurrent layers from one 8-token Granite Tiny prompt, with
  only layers `0`, `12`, `18`, and `30` clearing the local magnitude screen
  against a 9-layer preregistered requirement. The prompt-repeat tensor packet
  is `shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`; it repeats
  those four layers across all 12 frozen prompts and passes the real checker,
  but remains non-promoting because the layer subset was selected post-hoc.
  The held-out S1b tensor packet is
  `shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`; it freezes
  layers `0`, `12`, `30` as primary and layer `18` as near-miss/control on a
  fresh 12-prompt surface, passes the checker through an explicit held-out
  `trace_plan_config_path`, and justified the completed S2/S3 checks but not
  GPU promotion.
  The S2 replay scouts are
  `shared/results/ssq_lr_s2_state_replay_scout_20260507/` and
  `shared/results/ssq_lr_s2_state_replay_scout_block256_20260507/`; both are
  informational non-promoting failures of the official S2 contract because
  honest scale-byte accounting stays below `4x` state-memory reduction.
  The INT3 follow-up scouts are
  `shared/results/ssq_lr_s2_state_replay_scout_int3_block256_20260507/`,
  `shared/results/ssq_lr_s2_state_replay_scout_int3_block256_12p_20260507/`,
  and `shared/results/ssq_lr_s2_state_replay_scout_int3_block64_12p_20260507/`.
  INT3 block-256 passes on 4 prompts (`5.224x`, zero argmax delta) but both
  12-prompt held-out INT3 scouts fail quality, while MXFP4 remains
  quality-stable and below `4x`. The mixed-block S2b scout is
  `shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_20260507/`, decision
  `RESOURCE_LIMITED_S2_SCOUT_NOT_PROMOTABLE_PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`;
  it selects `mixed_int3_mxfp4_low_error_25pct` at `4.192x` counted memory
  reduction with zero BF16-argmax delta and `0.03956` selected NLL-delta CI
  high, but the longer-window replay
  `shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_20260507/`
  fails with selected accuracy CI high `0.0667` and selected NLL-delta CI high
  `0.0764`. Layer-localization then gave the strongest pre-transfer SSQ-LR
  candidate, now failed by S3:
  `shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layers0_30_20260507/`
  passes with selected recipe `int3_primary_state_block_scaled`, `5.224x`
  counted memory reduction, zero selected accuracy delta, and `0.04294`
  selected NLL-delta CI high. Layer `12` fails and must be excluded. The
  stricter prefilter
  `shared/results/ssq_lr_s3_prefilter_granite_tiny_layers0_30_20260507/`
  then weakens pure INT3 on `0,30` and selects
  `mixed_int3_mxfp4_low_error_25pct` at `4.192x`, zero selected accuracy drift,
  and `0.05044` selected NLL-delta CI high. The S3 transfer prefilter
  `shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507/`
  freezes that mixed recipe and was the cache-only blocker. The blocker is now
  quality transfer: `shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507/`
  fails on Granite 350M, and the validator-clean local two-model S3 packets
  for layer-0 mixed25 and INT3 both fail because the source and transfer models
  prefer different frozen recipes.
  HORN uses 12 prompts over all 8 planned Granite Tiny boundaries and failed
  H1a with hook-captured right-layer input tensors and ratio `1.06`. The H2
  noisy-continuation scout is
  `shared/results/horn_h2_noise_replay_scout_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
  with raw gate status `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`; it is
  contract-valid but weak (`1.037` directional drift ratio, signed
  selected-direction lower `0.324`, support `0.5`), so HORN is demoted as a
  standalone branch. These are
  plumbing/demotion packets, not promotable gate evidence.
- `ssq_lr_all_layer_scout.py`: metrics-only all-recurrent-layer scout that
  avoids committing duplicated tensor packets. Current artifact:
  `shared/results/ssq_lr_all_layer_scout_20260507/`. It is not a
  tensor-provenance gate packet and cannot promote S1.
- `horn_h2_noise_replay_scout.py`: resource-limited HORN H2
  noise-propagation scout from the failed Granite Tiny H1a packet. Current
  artifact: `shared/results/horn_h2_noise_replay_scout_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
  with raw gate status `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`; it
  passes `followup_gate_contracts --gate horn_h2` but the observed directional
  drift ratio is `1.037`, so it demotes HORN rather than reviving it.
- `hbsm_local_sensitivity_runner.py`: manifest-driven resource-limited HBSM B1
  forward-sensitivity runner. Current artifact:
  `shared/results/hbsm_local_sensitivity_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`; it
  wrote a checker-passing 56-row Granite Tiny packet from one 8-token prompt,
  8 layers, MXFP4 E2M1 output perturbation hooks, and all required B1
  controls. The top observed drift was layer 5 (`0.0273` symmetric KL), the
  packet failed B1 (`fisher_p=0.375`), and this is plumbing evidence only.
  The prompt-repeat scout is
  `shared/results/hbsm_prompt2_sensitivity_20260507/`, also
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`;
  it covers two prompts and the same 8 layers, passes the real checker, and
  weakens B1 further (`fisher_p=1.0`, boundary top-decile count `0`,
  cheap-predictor Spearman `-0.667`).
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

Current local Transformers execution-smoke packet:
`shared/results/hybrid_transformers_smoke_probe_20260507/`.

Current resource-limited local capture packet:
`shared/results/hybrid_manifest_local_capture_20260507/`.

## Next Exact Gates

1. **HybridKernel**: run the 5090 profiler packet in
   `hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`, then verify with
   `check_profiler_run_artifacts.py` and `analyze_profiler_metrics.py`.
2. **SSQ-LR**: killed as an active COLM branch. Preserve
   `KILLED_ssq_lr_cross_model_transfer/`; no GPU work without a new
   preregistration.
3. **HORN**: killed as an active COLM branch. Preserve
   `KILLED_horn_directional_noise_propagation/`; no GPU work without a new
   preregistration.
4. **HBSM**: killed as an active COLM branch. Preserve
   `KILLED_hbsm_sensitivity_heterogeneity/`; no GPU work without a new
   preregistration.
5. **ThoughtFlow-FP8**: continue paper reframing and citation/table polish; do
   not run a new signal without a fresh preregistered surface.

## Cost Discipline

Cheap gates come first. Failure at a Macbook phase means stop or fold the result
into a stronger branch before spending GPU time. HybridKernel is the exception:
its Mac work is saturated, so the next discriminative bit is the native GPU
profiler packet.

For SSQ-LR, HORN, and HBSM, the strict real-packet builders/checkers are
S1, H1a/H1, and B1 respectively. The S2/S3, H2/H3, and B2/B3 follow-up
contracts are now executable through `shared/followup_gate_contracts.py`.
Current SSQ-LR S2b and HORN H2 model packets are non-promoting Mac failures;
HBSM B2/B3 still have no model packet evidence.

## Killed Marker Convention

`KILLED_*` folders mark consumed sub-branches and dead framings. They do not
mean every artifact in the source project is useless; each marker README records
what was tried, why it died, and what remains salvageable.
