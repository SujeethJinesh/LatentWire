# Experimental Project Status

Date: 2026-05-07

This ledger is the sprint control plane for the relevant COLM/ICLR branches:
HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8. It separates live gates,
watched branches, and killed branches so we do not keep spending time on
saturated ideas.

## Active Decision Surface

| Rank | Project | Readiness | Current story | Exact blocking gap | Next experiment |
|---:|---|---:|---|---|---|
| 1 | HybridKernel | 70% if GPU gate passes; 0% as local-only result | Boundary-fusion may recover avoidable attention to SSM overhead in hybrid models, but Mac work is saturated. The packet checker now handles batch>1 replay as per-sample prefill plus aggregate completion tokens, enforces the copied native control matrix's request shape/control segment/boundary direction, and the driver synthesizes exact tokenizer-roundtrip prompts when token counts are required, failing before profiling if exact prefill length cannot be proven. The analyzer also requires same-family controls from the same primary model, and the GPU gate command uses `--require-full-matrix` so primary-only packets are audit-only. The artifact checker now requires three distinct rows and run IDs for each primary, same-family-control, and cross-family-falsification role before a full-matrix packet can pass. It also requires `metadata/reduction_input_manifest.json` and a filled reduction worksheet so every metric row is tied to its source Nsight artifacts, SHA-256 digests, time window, reduction command, reducer script/worksheet digest, and reduction notes. If Qwen3-Next is infeasible, a smaller cross-family hybrid can only be used through the checked-in preregistration replacement template before profiling. | User-operated NVIDIA/vLLM Nsight packet with three distinct repeats, at least 3% recoverable gain, three same-model same-shape same-family control rows, three same-shape cross-family falsification rows that stay below 3%, fixed-length replay with `ignore_eos`, matching client replay logs for every metric model, a completed row-level reduction manifest/worksheet, and a clean no-boundary negative if using `no_boundary_signal_kill`. | Run `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`; verify with `check_profiler_run_artifacts.py --require-full-matrix` and `analyze_profiler_metrics.py`. |
| 2 | SSQ-LR | Stopped current recipe; diagnostic only unless newly preregistered | Test whether recurrent SSM state in hybrid reasoners can go below FP16 with a stable quantization recipe. The synthetic S1 rehearsal is schema-only. Granite Tiny is cached locally and the manifest local runner has corrected bucket-truncated recurrent-state capture plus arbitrary layer selection. The fresh held-out S1b prompt split reproduced the frozen heterogeneity hypothesis with layers `0`, `12`, and `30` as primary and layer `18` as near-miss/control: `experimental/shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507/` is a checker-passing 192-row tensor-provenance packet with selected S1 ratio `2.459`, CI low `1.861`, and Holm p-min `2.78e-05`. Uniform S2 recipes split the blocker, and the stricter prefilter replay `experimental/shared/results/ssq_lr_s3_prefilter_granite_tiny_layers0_30_20260507/` selected `mixed_int3_mxfp4_low_error_25pct` on layers `0,30` with memory `4.192x`, selected accuracy CI high `0.0`, and selected NLL-delta CI high `0.05044`. After downloading the second complete local hybrid model (`ibm-granite/granite-4.0-h-350m`, revision `3b17b717b8f2f5d305b0a92c1491e239aeda19c8`), the same frozen `0,30` recipe failed the 12-prompt transfer replay: `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507/` falls back to INT8 at `1.984x`. Layer diagnostics show layer `0` passes on 350M and layer `30` fails, but local two-model S3 packets for layer-0 mixed25 and INT3 both fail because the recipe that passes one Granite model does not pass the other. | S3 now fails on quality/transfer, not just cache completeness. The checker remains ready, and the new local transfer prefilter builder can combine real source and transfer S2 rows while preserving one frozen recipe hash, one source S2 hash, and `retuned=false` rows. The exact blocker is a new preregistered recipe or layer-selection rule that transfers without retuning; the current `0,30` and layer-0 rescue attempts are post-hoc diagnostics and cannot promote. | Do not spend GPU on SSQ-LR under the current recipe. Either preregister a new bounded recipe/layer rule and rerun Mac S2/S3, or keep SSQ-LR as diagnostic evidence for the hybrid-quantization appendix. |
| 3 | HORN | 0% active; 12% reusable scaffold | Test whether attention-to-SSM and SSM-to-attention boundaries have asymmetric outlier/noise propagation. The current packet is a 72-row synthetic real-schema H1a rehearsal, not model evidence; a single-model real screen is labeled H1a, not H1 promotion. Granite Tiny is cached locally, HORN plans/templates preserve `prompt_cluster_id`, and the manifest local runner writes a checker-passing 288-row resource-limited H1a packet from 12 real local prompts over all 8 planned Granite Tiny boundaries using right-layer forward-pre-hook tensors: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`, selected ratio `1.06` with cluster-bootstrap low `1.06`. The H2 scout `experimental/shared/results/horn_h2_noise_replay_scout_20260507/` is contract-valid but also fails under the signed-direction support rule: `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`, 20 rows, 2 prompts, 3 seeds, paired units `6/6`, hook-off max delta `0.0`, H1-selected direction preserved in aggregate, directional drift ratio `1.037`, signed selected-direction lower bound `0.324`, selected-direction support `0.5`, and demotion recommendation `DEMOTE_HORN_STANDALONE_WEAK_H2`. This proves hook/noise plumbing and rules against a large short-surface directional effect; it does not promote H2 or support a precision-allocation claim. | HORN lacks both a strong H1a magnitude asymmetry and a strong H2 noise-propagation asymmetry. The checker now requires both boundary directions, prompt-cluster IDs, decision-grade summary fields including a deterministic prompt-cluster-bootstrap lower bound, per-boundary non-boundary controls paired through `matched_boundary_direction` and below the selected H1 threshold, trace-plan file-hash pinning, saved tensor artifact provenance, and permuted controls paired by prompt, boundary, layer, norm positions, metric/source/hash values, and an actual flipped `direction` label whose selected-direction effect is erased. The H2/H3 contract checker now rejects one-seed noisy replay, unpaired direction units, missing hook-off controls, signed selected-direction support below `0.75`, and pure-architecture controls that fail to fold below the preregistered 1.2 null threshold. | Do not spend GPU on HORN standalone. Keep it as negative/control evidence for SSQ-LR/HBSM unless a deliberately reopened full H2/H3 run on more prompts/models overturns this weak scout. |
| 4 | HBSM | 0% active; 12% reusable scaffold | KL-Lens-like layer sensitivity is crowded; remaining wedge was frontier hybrid mechanism plus cheaper predictor. The synthetic B1 rehearsal validates schema/control logic only. `experimental/shared/results/hbsm_local_sensitivity_20260507/` is the first real-model row packet: 56 rows, one 8-token Granite Tiny prompt, 8 layers, MXFP4 E2M1 layer-output perturbation hooks, all required B1 controls, checker-passing decision `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`; it failed with layer 5 as top drift (`0.0273`), `fisher_p=0.375`, and cheap-predictor Spearman `-0.476`. The new prompt-repeat scout `experimental/shared/results/hbsm_prompt2_sensitivity_20260507/` is also checker-passing and fails harder: 64 rows, two prompts, 8 layers, `fisher_p=1.0`, boundary top-decile count `0`, non-boundary top-decile count `1`, and cheap-predictor Spearman `-0.667`. B2/B3 have executable follow-up contracts, but no follow-up model evidence. | Gate B1 must replicate sensitivity heterogeneity on current hybrid reasoners, then B2 must show a cheaper predictor. The checker now scores only primary `boundary_only` rows after aggregating prompt rows to `(model_id, layer)`, derives top deciles from measured `kl_or_nll_drift`, rejects every prompt-row supplied flag mismatch, requires prompt-level coverage with boundary and non-boundary layers, exact aggregated top-decile cardinality, trace-plan file-hash pinning, layer-aligned KL/outlier comparator controls over the same scoring set, a non-enriched same-count random baseline, finite metrics, and near-zero perturbation-off controls. The B2/B3 contract checker rejects predictor shopping and missing matched-noise controls before later evidence can be cited. | Do not GPU-promote HBSM or run a long 12-prompt sweep under the current hypothesis. Continue only if a narrower mechanism hypothesis is preregistered; otherwise fold HBSM into negative/control evidence. |
| 5 | ThoughtFlow-FP8 | 93% falsification paper; 0% positive method | The reusable contribution is the preregistered falsification ladder for sparse-cache signals. The draft now has protocol, RDU demotion, explicit lessons learned, claim-boundary, related-work citation tables, and a tracked diagnostic packet locking stale-positive and negative conclusions with repo-root input hashes and clean-path provenance. | Paper polish only; no fifth signal unless a new preregistration and fresh surface exist. | Human review of `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`. |

## New Shared Infrastructure

Shared Mac-local utilities live in `experimental/shared/`:

- `fp4_simulator.py`: deterministic MXFP4-style and low-bit simulation.
- `activation_dumper.py`: tensor-packet read/write helpers.
- `boundary_inspector.py`: attention/SSM boundary identification.
- `hybrid_architecture_maps.py`: config-derived explicit boundary maps for
  future real trace packets. The maps now include `canonical_model_id` and
  `model_id_aliases` so packet builders can preserve served HF IDs while
  checking rows against the canonical architecture-map ID.
- `hybrid_model_eligibility.py`: metadata-only HF size/cache preflight for
  live hybrid targets.
- `hybrid_local_capture_preflight.py`: local dependency/cache/size preflight
  for the first real SSQ-LR/HORN/HBSM captures. Current artifact:
  `shared/results/hybrid_local_capture_preflight_20260507/`, decision
  `LOCAL_CAPTURE_READY_NOT_EVIDENCE`; Granite Tiny is cached locally and has a
  native `transformers` model class, while Granite Small and Qwen3-Next remain
  GPU-sized or uncached. Missing `mamba_ssm` and `vllm` are optional fast-path
  gaps here, not hard blockers for local Transformers capture. This is
  preflight-only and cannot promote any gate.
- `hybrid_transformers_smoke_probe.py`: resource-limited local execution probe
  for cached hybrid models. Current artifact:
  `shared/results/hybrid_transformers_smoke_probe_20260507/`, decision
  `RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`; it loaded Granite Tiny,
  ran an 8-token CPU forward, and observed 36 recurrent-state cache layers and
  4 attention-cache layers. This is execution plumbing evidence only and cannot
  promote SSQ-LR/HORN/HBSM gates.
- `hybrid_manifest_local_capture_runner.py`: manifest-driven resource-limited
  local capture runner. Current artifact:
  `shared/results/hybrid_manifest_local_capture_20260507/`, decision
  `RESOURCE_LIMITED_CAPTURE_PACKETS_WRITTEN_NOT_PROMOTABLE`; it produced
  checker-passing non-promoting Granite Tiny SSQ-LR and HORN packets from one
  shared model load. The older SSQ-LR row reused the final cache state across
  buckets and failed S1 with ratio `1.0`; the corrected dedicated packet is
  `shared/results/ssq_lr_local_bucket_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`, selected ratio
  `3.29`, using bucket-specific 2/4/6/8-token recurrent-state replays. The
  current multilayer packet is
  `shared/results/ssq_lr_local_multilayer_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`; it covers one
  prompt and four layers, and only layer 0 clears the S1 magnitude screen.
  Only its compact readout is tracked; the full checker-passing tensor packet
  is regenerated on demand to avoid committing duplicated smoke tensors. HORN
  uses 12 prompts over all 8 planned Granite Tiny boundaries and failed H1a
  with hook-captured right-layer input tensors and ratio `1.06`. These are
  saved-tensor plumbing packets, not promotable gate evidence.
- `ssq_lr_all_layer_scout.py`: metrics-only SSQ-LR all-recurrent-layer scout
  that writes compact JSON/Markdown rows instead of tensor packets. Current
  artifact: `shared/results/ssq_lr_all_layer_scout_20260507/`, decision
  `RESOURCE_LIMITED_ALL_LAYER_SCOUT_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`;
  it scanned 36 recurrent layers from one 8-token Granite Tiny prompt and only
  4 layers cleared the local S1 magnitude screen, below the 9-layer
  preregistered requirement. This is a decision scout only, not promotable
  tensor-provenance evidence.
- `hybrid_manifest_local_capture_runner.py --ssq-layers`: arbitrary selected
  layer capture for SSQ-LR prompt-repeat probes. Current artifact:
  `shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`; it produced a
  checker-passing 192-row tensor-provenance packet over 12 prompts and layers
  `0`, `12`, `18`, `30`. This is stronger than the metrics-only scout but
  remains non-promoting because the layer subset was selected post-hoc.
- `prompts/hybrid_reasoning_s1b_holdout_12_20260507.jsonl`: fresh held-out
  12-prompt SSQ-LR S1b surface. The companion trace plan is
  `shared/results/hybrid_trace_plan_s1b_holdout_20260507/`, with SSQ-LR hash
  `sha256:8fefebf0a704f1f5d3575c83ed9991e0bf09210ee3387fd3f979332afb8892e2`.
  The held-out tensor packet is
  `shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`; it passes the
  checker through the explicit held-out `trace_plan_config_path`, covers 12
  prompts, layers `0`, `12`, `18`, `30`, and all four S1 buckets, and preserves
  layer `18` as the failed near-miss/control.
- `ssq_lr_s2_state_replay_scout.py`: resource-limited SSQ-LR S2 continuation
  replay scout over cached recurrent states. Current artifacts:
  `shared/results/ssq_lr_s2_state_replay_scout_20260507/` and
  `shared/results/ssq_lr_s2_state_replay_scout_block256_20260507/`, both
  `RESOURCE_LIMITED_S2_SCOUT_NOT_PROMOTABLE_FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`.
  They show calibrated INT8/FP8/MXFP4 state perturbations can preserve short
  BF16-argmax fidelity while random same-L2 and shuffled-scale controls break,
  but the official S2 gate stays failed because honest scale-byte accounting
  remains below `4x` state-memory reduction.
- `ssq_lr_s2_state_replay_scout.py --block-size {64,256}` with the INT3
  candidate: current artifacts
  `shared/results/ssq_lr_s2_state_replay_scout_int3_block256_20260507/`,
  `shared/results/ssq_lr_s2_state_replay_scout_int3_block256_12p_20260507/`,
  and `shared/results/ssq_lr_s2_state_replay_scout_int3_block64_12p_20260507/`.
  The 4-prompt INT3 block-256 scout is contract-passing but resource-limited;
  both 12-prompt held-out INT3 scouts are contract-valid S2 failures. INT3
  clears byte accounting but fails quality on the larger held-out surface.
- `ssq_lr_s3_transfer_prefilter.py`: freezes the then-surviving pre-transfer
  SSQ-LR S2b recipe into the S3 schema and inventories local hybrid
  transfer-model cache completeness before GPU or large downloads. Current artifact:
  `shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507/`,
  decision `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER`; the packet freezes
  `mixed_int3_mxfp4_low_error_25pct` on layers `0,30`, is validator-clean, has
  `retuned=false` for every row, and records one complete local transfer
  candidate (`ibm-granite/granite-4.0-h-tiny`) plus config-only caches for
  Granite Small, Granite Small FP8, and Qwen3-Next. It is not promotable because
  S3 requires at least two complete transfer models.
- `ssq_lr_s3_local_transfer_prefilter.py`: combines real source and transfer
  S2 replay rows into the stricter S3 schema without retuning. Current
  artifacts
  `shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507/`
  and
  `shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507/`
  are validator-clean S3 failures. The first freezes the source-passing layer-0
  mixed25 recipe and fails on Granite 350M; the second freezes the
  transfer-passing layer-0 INT3 diagnostic and fails on Granite Tiny. Both have
  two models, 12 prompts per model, one frozen recipe hash, one source S2 hash,
  and `retuned=false` rows. This closes the prior cache-only blocker and
  demotes the current SSQ-LR recipe before GPU.
- `horn_h2_noise_replay_scout.py`: resource-limited HORN H2 noisy-continuation
  replay from the failed Granite Tiny H1a packet. Current artifact:
  `shared/results/horn_h2_noise_replay_scout_20260507/`, decision
  `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`; it passes the H2 follow-up
  contract with hook-off max delta `0.0` and complete paired units, but the
  directional drift ratio is only `1.037`, so HORN is demoted as a standalone
  branch. This is a demotion scout only, not H2 promotion or GPU evidence.
- `hbsm_local_sensitivity_runner.py`: manifest-driven resource-limited HBSM B1
  sensitivity runner. Current artifact:
  `shared/results/hbsm_local_sensitivity_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`;
  it produced a checker-passing 56-row Granite Tiny B1 packet from one
  8-token prompt and 8 layer-output MXFP4 E2M1 perturbation replays. The packet
  validates row-packet provenance and controls, but the smoke surface fails B1.
  The prompt-repeat artifact
  `shared/results/hbsm_prompt2_sensitivity_20260507/` uses the same runner with
  two prompts. It also passes the real checker and fails B1, with no boundary
  layer in the measured top decile and a negative cheap-predictor readout.
- `hybrid_trace_plan.py`: deterministic SSQ-LR/HORN/HBSM capture plan from the
  frozen 12-prompt manifest and shared architecture maps. The current plan is
  `shared/results/hybrid_trace_plan_20260507/` with 5,184 SSQ-LR rows, 1,404
  HORN rows, and 2,304 layer-aligned HBSM rows. SSQ-LR recurrent-state labels are now derived
  from the architecture map rather than hard-coded to Granite/Mamba2, and HORN
  rows carry prompt-cluster IDs derived from the prompt manifest task/cluster
  metadata. It is trace-plan-only and cannot promote any gate.
- `hybrid_trace_capture_manifest.py`: expands the frozen trace plans into
  per-project/per-model metadata templates for real SSQ-LR and HORN tensor
  packets plus HBSM sensitivity row-packet templates. The current artifact is
  `shared/results/hybrid_capture_manifests_20260507/` with 5,184 SSQ-LR
  template entries, 1,404 HORN template entries, and 2,304 layer-aligned HBSM template
  entries. It is capture-template-only and cannot promote any gate.
- `hybrid_trace_packet_builder.py`: converts future saved tensors into strict
  SSQ-LR/HORN real packets and resolves hook names sanitized by tensor-packet
  storage. Tensor packets now include `tensor_manifest.json`, reject sanitized
  filename collisions, and carry original tensor names, storage names, SHA-256,
  dtype, and shape into every SSQ-LR/HORN row. Built SSQ-LR/HORN packets copy
  tensor manifests and `.pt` files into `tensors/`; built HBSM packets copy the
  source sensitivity row packet into `evidence/` with a SHA-256 manifest.
  Resource-limited input metadata now forces a
  `RESOURCE_LIMITED_NOT_PROMOTABLE_...` packet decision, even if the recomputed
  smoke-gate status would otherwise pass. The builder rejects unfilled capture
  templates marked `_template_only: true` or containing `TO_FILL_BEFORE_CAPTURE`
  markers, canonicalizes registered served/HF model IDs, and supports HORN
  `tensor_alias_of` so permuted-direction rows reuse observed boundary tensors;
  real tensor packets fail fast if a referenced tensor is missing from the
  manifest.
- `hybrid_gate_evaluators.py`: recomputes SSQ-LR S1, HORN H1, and HBSM B1
  decision fields from raw rows so summaries cannot be hand-filled. SSQ-LR now
  uses prompt-level bootstrap-style lower bounds plus Holm-corrected two-sample
  distribution tests with per-layer effect-size floors, HORN gates on
  prompt-cluster bootstrap lower bounds, per-prompt non-boundary controls, and
  actual-label-flipped permuted controls, and HBSM derives measured top-decile
  membership from primary-row drift after aggregation while aligning comparator
  Spearman baselines to the same scoring layer set.
- `sensitivity_metrics.py`: rel-L2, KL, kurtosis, and rank-correlation metrics.
- `check_gate_packet.py`: generic synthetic-packet validator plus strict
  `--mode real --project ...` contracts for SSQ-LR, HORN, and HBSM, including a
  `SCHEMA_REHEARSAL_NOT_PROMOTABLE` path that exercises real schemas without
  allowing synthetic rows to promote a gate. Non-rehearsal packets now reject
  unregistered `model_revision`/`tokenizer_revision` strings, decisions that do
  not equal the recomputed S1/H1/B1 gate status, unknown controls, and rows
  whose `config.json` omits `trace_plan_path` or falls outside that cited plan.
  Promotable real packets must cite trace-plan rows whose file SHA-256 equals
  the registered shared `trace_plan_hash`; caller-created subset plans are only
  accepted for explicit `RESOURCE_LIMITED_NOT_PROMOTABLE` packets.
  Non-rehearsal SSQ-LR/HORN packets must also include copied tensor artifacts;
  the checker reloads them and recomputes row metrics from saved bytes.
  Non-rehearsal HBSM packets must include a copied source sensitivity row packet
  with a matching SHA-256 manifest.
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
| HBSM | `experimental/hbsm/phase2/results/hbsm_synthetic_b1/` | `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1` | 720 real-schema rows, 480 primary prompt rows, 240 layer-aligned control rows, and 40 scoring layers after aggregation |

All three packets pass `experimental.shared.check_gate_packet`.

Each active project also has a reviewer-pack handoff under
`experimental/<project>/paper/reviewer_pack.md`. For SSQ-LR, HORN, and HBSM
these packs explicitly state that the current drafts are preregistration shells,
not method papers, until real S1--S3, H1--H3, or B1--B3 evidence exists.

The checker now has a stricter real-packet mode:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/<packet> --mode real --project <ssq_lr|horn|hbsm>
```

Real packets must include provenance, `summary.md`, matching `row_count`,
project-specific row schemas, required controls, admissible coverage,
`prompt_ids_hash`, `architecture_map_hash`, `trace_plan_hash`, and
decision-grade `summary.json` aggregates. Non-rehearsal real packets now verify
`model_id` or a registered model alias and `architecture_map_hash` against
`shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`, not
just hash syntax; verify `model_revision` and `tokenizer_revision` against the
registered eligibility snapshot SHA; verify `trace_plan_hash` against the
project hash in `shared/results/hybrid_trace_plan_20260507/config.json`; and
require `trace_plan_path` so off-plan rows cannot bypass frozen plan coverage.
Synthetic-only real-schema rehearsals must set `schema_rehearsal:
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
- `followup_gate_contracts.py`: strict second-stage contracts for SSQ-LR S2/S3,
  HORN H2/H3, and HBSM B2/B3. The contracts now require SSQ-LR INT8/FP8/MXFP4
  and noise/scale baselines, HORN H2 three-seed paired directional units, and
  HORN H3 pure controls that fold below the preregistered 1.2 null threshold or
  overlap 1.0 by CI.

## Config-Only Architecture Maps

Config-only maps now exist for the local Granite and Qwen hybrid configs:

| Artifact | Use | Claim boundary |
|---|---|---|
| `experimental/shared/results/hybrid_architecture_maps_20260506/` | Provides explicit layer kinds, boundary IDs, direction counts, and config hashes for SSQ-LR/HORN/HBSM real trace packets. | Config provenance only; no activations, SSM state, quality, or GPU evidence. |
| `experimental/shared/results/hybrid_trace_plan_20260507/` | Enumerates exact SSQ-LR/HORN/HBSM trace rows to capture from frozen prompts and architecture maps. | Trace-plan-only; no activations, SSM state, sensitivity, quality, or GPU evidence. |
| `experimental/shared/results/hybrid_capture_manifests_20260507/` | Provides per-model fill-in templates for tensor/sensitivity capture metadata before packet building. | Capture-template-only; no tensors, model outputs, sensitivity metrics, quality, or GPU evidence. |

## HybridKernel Packet Hardening

The native profiler packet now requires per-row reduction provenance:
`row_role`, `control_family`, `boundary_direction`, `nsys_artifact`,
`nsys_artifact_sha256`, `ncu_artifact`, `ncu_artifact_sha256`, `kernel_names`,
`boundary_indices`, `time_window_ms`, `recoverable_fraction_basis`,
`reduction_command`, and `reduction_notes`. The checker also cross-checks model identity across
`profile_scope.json`, client replay logs, metric rows, and the architecture map.
New packets also copy
`experimental/hybridkernel/phase2/native_control_matrix.json` into
`metadata/native_control_matrix.json`, fixing the primary, same-family, and
cross-family row roles before the GPU run starts.
It now has an explicit `no_boundary_signal_kill` packet mode so a clean
Nsight-Systems negative run can be reviewable without inventing an Nsight
Compute target; that mode requires the analysis to be a clean kill and each
row to state explicit no-boundary-signal evidence before NCU can be skipped.
Per-row `nsys_artifact` and `ncu_artifact` fields must resolve to reviewable
files inside the run packet with valid Nsight extensions. This prevents a
reduced metric row from citing a missing or external artifact.
The checker also rejects reuse of the same Nsight Systems or Nsight Compute
artifact across any non-pending metric rows, including across primary,
same-family control, and cross-family falsification roles.
It now also rejects metric rows outside the copied `native_control_matrix.json`,
including mismatched model, control family, control segment, boundary direction,
or request shape, and multi-model metric packets whose `profile_scope.json`
lacks per-model `model_scopes`.
The profiler reducer now refuses prototype promotion unless the same metric
packet includes at least three matched same-family control rows and three
cross-family falsification rows on the same request/runtime shape, and those
controls stay below the 3% recoverable-gain gate. The three same-family and
three cross-family rows must also have distinct `run_id` values, so copied
control rows cannot satisfy the three-repeat rule. Same-family controls may be
matched segments or same-family control models. A primary-only packet that
clears 3%, or a packet whose controls reproduce the same signal, remains
audit-only. The reducer rejects impossible local timings, and the artifact
checker rejects repeated-row packets that reuse the same Nsight artifacts, lack
token-counted client replay JSON, mismatch replay prompt/decode/request shape,
reuse time windows, or cite artifacts whose SHA-256 digest does not match.
The fixed-request profiler driver now synthesizes exact tokenizer-roundtrip
prompts when `--require-token-counts` is set, so approximate whitespace prompts
cannot silently define the GPU prefill shape. The optional Triton CPU-backend
correctness test passes on this Mac under
`HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 TRITON_CPU_BACKEND=1`, but it remains a
correctness-only diagnostic.

## Hybrid Model Eligibility

Metadata-only HF preflight found public live targets and one repo-local cached
Mac candidate:

| Model | Safetensors GB | Local weights | Decision |
|---|---:|---|---|
| `ibm-granite/granite-4.0-h-tiny` | 12.93 | yes | `POSSIBLE_LOCAL_CACHE_CHECK_REQUIRED` |
| `ibm-granite/granite-4.0-h-small` | 59.99 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| `ibm-granite/granite-4.0-h-small-FP8` | 31.19 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151.49 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |

Artifact: `experimental/shared/results/hybrid_model_eligibility_20260506/`.
Large rows remain GPU-sized or uncached. Granite Tiny is the only current
Mac-local execution candidate and remains non-promoting until a full gate packet
exists.

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
   current branch that can resolve with one GPU experiment. The packet must
   include the mandatory row-level reduction input manifest.
2. SSQ-LR: S1b clears on a held-out prompt split. The current Mac S2 candidate
   was narrowed to `mixed_int3_mxfp4_low_error_25pct` on layers `0,30`, but
   that frozen recipe now fails 12-prompt no-retuning transfer to Granite 350M.
   Layer-0 rescue diagnostics also fail as S3 packets because Tiny and 350M
   prefer different recipes. Do not GPU-promote SSQ-LR unless a new
   preregistered recipe/layer rule clears Mac S2/S3 first.
3. HORN: demote as a standalone branch. The H2 scout is contract-valid but
   fails with directional drift ratio `1.037`; do not spend GPU on HORN unless
   a future full H2/H3 reopening has a new reason and preregistered scope.
4. HBSM: do not scale B1 under the current hypothesis. Continue only if a
   narrower mechanism hypothesis is preregistered; otherwise keep HBSM as
   negative/control evidence.
5. ThoughtFlow: continue paper-only copyedit and human review around the
   falsification-methodology contribution; no new experiments in the current
   branch.
