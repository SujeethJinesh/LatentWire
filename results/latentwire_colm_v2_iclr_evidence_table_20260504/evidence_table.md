# LatentWire COLM_v2 / ICLR Evidence Table

- created UTC: `2026-05-04T22:51:18.773447+00:00`
- source triage: `results/iclr_colm_v2_live_branch_triage_20260504/live_branch_triage.json`
- COLM_v2 readiness: `scoped_positive_ready_for_writeup_if_claims_are_narrow`
- ICLR readiness: `blocked_by_lack_of_broad_or_learned_positive_receiver`

## Current Story

LatentWire_v2 can currently support a scoped COLM_v2 story: byte-scale, source-private packets plus strict destructive controls. The ICLR story is still blocked because cross-family conditional PQ, deterministic public-basis conditioning, scalar integrity thresholds, and HellaSwag learned/source-conditioned resonance receivers have not produced a positive row beyond packet/source-choice/target-cache controls.

## Exact Submission Gap

ICLR needs a positive learned or broader-benchmark receiver that passes strict destructive controls. COLM_v2 can be prepared around the conditional-PQ shared-schema method, the fixed-byte HellaSwag packet row, and the target-resonance capacity-versus-held-out-failure analysis with explicit limitations.

## Current Technical Contributions

- `source_private_low_rate_packets`: alive_for_colm_v2; broader or learned receiver evidence for ICLR
- `strict_destructive_controls`: strong; paper integration and compact tables
- `systems_byte_accounting`: mac_local_ready; native dense-KV/C2C measurements before throughput or energy claims
- `sparse_resonance_packets`: framing_alive_method_not_yet_positive; new mechanism beyond deterministic PQ, PCA atoms, chunk encoders, query resamplers, and source-to-prefix decoders
- `target_self_resonance_capacity_probe`: capacity_alive; held-out/source-private receiver that beats slots-only, zero-source, wrong-source, source-choice, and candidate-roll controls

## COLM_v2 Core Rows

These are the rows that can anchor the narrow workshop version without claiming broad latent language.

| Branch | Status | Score | Baseline | Delta | CI95 low | Bytes | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Conditional PQ shared-schema packet | `promote_for_colm_v2_only` | 16 | 16 | 0.658 | 0.658 | 7 | Use as the COLM_v2 positive method boundary; not enough for ICLR generality. |
| HellaSwag fixed hybrid candidate packet | `promote_for_colm_v2_systems_baseline` | 0.532464 | 0.526688 | 0.005776 | 0.002888 | 4 | Useful COLM_v2 systems/privacy row, but not a learned latent receiver. |

## COLM_v2 Supporting / Guardrail Rows

These rows explain headroom, saturation, and why the paper keeps strong claim boundaries.

| Branch | Status | Score | Baseline | Delta | CI95 low | Bytes | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Conditional PQ scalar integrity threshold | `ruled_out_simple_integrity_threshold` | 0.425781 | 0.457031 | -0.03125 | -0.097656 | 7 | Do not continue scalar threshold integrity on this public-zscore conditional-PQ receiver. |
| Target self-resonance oracle soft-prefix capacity | `capacity_alive_not_source_private_method` | 0.9375 | 0.625 | 0.3125 |  |  | Use only as capacity/headroom evidence; it optimizes on eval rows. |
| HellaSwag complementarity-frontier selector diagnostic | `headroom_alive_selector_blocked` | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not train another HellaSwag selector on the same packet fields; require a new information path. |
| HellaSwag multi-signal source packet frontier | `ruled_out_cached_policy_packet` | 0.455729 | 0.467448 | -0.011719 | -0.02347 | 5 | Do not continue cached Qwen policy-prediction packets on this HellaSwag slice. |

## ICLR Blocker Rows

These rows prevent an ICLR-scale claim until a new source-causal interface clears the same controls.

| Branch | Status | Score | Baseline | Delta | CI95 low | Bytes | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Conditional PQ public-zscore held-out-family decoder | `ruled_out_as_cross_family_rescue` | 0.453125 | 0.441406 | 0.011719 | -0.085938 | 7 | Do not widen public-zscore to n500/remap. |
| Conditional PQ public-SVD whitening held-out-family decoder | `ruled_out_as_cross_family_rescue` | 0.375 | 0.613281 | -0.238281 | -0.32041 | 7 | Do not continue deterministic public-basis whitening as the rescue path. |
| Conditional PQ corruption-to-noop receiver | `ruled_out_as_cross_family_rescue` | 0.25 | 0.25 | 0 | 0 | 7 | Keep as diagnostic; do not promote this receiver family. |
| Conditional PQ scalar integrity threshold | `ruled_out_simple_integrity_threshold` | 0.425781 | 0.457031 | -0.03125 | -0.097656 | 7 | Do not continue scalar threshold integrity on this public-zscore conditional-PQ receiver. |
| ARC sparse resonance PCA/target-aligned soft-prefix packets | `ruled_out_current_basis_receiver` | 0.25 | 0.625 | -0.375 |  |  | Do not widen plain PCA/target-aligned soft-prefix packets. |
| HellaSwag protected top-2/rival packet | `ruled_out_shallow_pair_decoder` | 0.46224 | 0.467448 | -0.005208 | -0.013021 | 5 | Do not rerun generic protected top-2/rival switchers. |
| HellaSwag official-train receiver-calibrated selector | `weakened_near_miss` | 0.466146 | 0.467448 | -0.001302 | -0.007812 | 5 | Do not continue shallow linear score-feature selectors without new evidence. |
| HellaSwag harm-controlled complementarity buckets | `ruled_out_no_safe_bucket` | 0.467448 | 0.467448 | 0 | 0 | 6 | Low-harm bucket receiver is saturated. |
| HellaSwag top1/top2 ambiguity buckets | `ruled_out_no_safe_bucket` | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not continue rank/score-bin top2 buckets without a new packet field. |
| HellaSwag denoising syndrome packet | `ruled_out_shallow_syndrome_decoder` | 0.463542 | 0.467448 | -0.003906 | -0.010417 | 4 | Do not promote ridge-denoising syndrome decoder. |
| HellaSwag sparse/common-basis top2 ambiguity code | `weakened_common_basis_atom` | 0.500977 | 0.501953 | -0.000977 | -0.00293 | 4 | Do not claim sparse atom causality from this branch. |
| Target self-resonance oracle soft-prefix capacity | `capacity_alive_not_source_private_method` | 0.9375 | 0.625 | 0.3125 |  |  | Use only as capacity/headroom evidence; it optimizes on eval rows. |
| Target self-resonance held-out learned prefix encoders | `ruled_out_current_target_native_encoder_family` | 0.6875 | 0.6875 | 0 |  |  | Do not run more chunk/distill/query-resampler variants without a new information path. |
| Source-conditioned target-native resonance receivers | `ruled_out_current_source_conditioned_receiver_family` | 0.375 | 0.375 | 0 |  |  | Diagnose complementarity/gating before implementing another source-to-prefix decoder. |
| HellaSwag complementarity-frontier selector diagnostic | `headroom_alive_selector_blocked` | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not train another HellaSwag selector on the same packet fields; require a new information path. |
| HellaSwag multi-signal source packet frontier | `ruled_out_cached_policy_packet` | 0.455729 | 0.467448 | -0.011719 | -0.02347 | 5 | Do not continue cached Qwen policy-prediction packets on this HellaSwag slice. |

## Literature And Novelty Boundaries

| Work | Role | Boundary for LatentWire | Source |
|---|---|---|---|
| Cache-to-Cache (C2C) | closest dense KV-cache communication baseline | C2C projects and fuses source KV cache into a target KV cache with a learned gate; LatentWire must not claim the same contribution unless it reports a different low-rate, source-private packet regime and utility-per-byte controls. | https://openreview.net/forum?id=LeatkxrBCi |
| DroidSpeak | cross-LLM KV-cache sharing / serving baseline | DroidSpeak reuses KV caches across distributed nodes running different LLMs with the same architecture; LatentWire must separate semantic packet transfer from compatible-cache reuse and report source exposure. | https://arxiv.org/abs/2411.02820 |
| KVCOMM | multi-agent KV-cache communication baseline | KVCOMM is training-free cross-context KV-cache reuse for multi-agent inference; LatentWire should avoid claiming general multi-agent cache communication and instead stress source-private packet transfer under destructive controls. | https://arxiv.org/abs/2510.12872 |
| RelayCaching | collaborative decoding KV-cache reuse baseline | RelayCaching accelerates collaboration by reusing decoding KV caches; LatentWire differs only if it transmits compact source evidence rather than reusing produced cache states. | https://arxiv.org/abs/2603.13289 |
| BLIP-2 / Q-Former | learned query bottleneck precedent | A lightweight query transformer between frozen encoders and LMs is established; LatentWire novelty cannot be the existence of a query bottleneck. | https://arxiv.org/abs/2301.12597 |
| Flamingo / Perceiver Resampler | fixed latent-token resampler and gated cross-attention precedent | Fixed latent resampling into a frozen LM is established in multimodal models; LatentWire must distinguish by source-private model-to-model transfer and destructive controls. | https://arxiv.org/abs/2204.14198 |
| Perceiver IO | general latent query architecture | Flexible latent queries for structured inputs/outputs are prior art; use it as architectural motivation, not a novelty claim. | https://arxiv.org/abs/2107.14795 |
| TurboQuant | strong online vector/KV quantization baseline | TurboQuant optimizes vector/KV distortion under low bit widths; LatentWire should compare against it as a source-state byte floor and avoid unmeasured throughput claims. | https://arxiv.org/abs/2504.19874 |
| KVQuant | low-bit KV-cache compression baseline | KVQuant compresses a model's own cache for long-context serving; LatentWire moves compact source evidence to another model and must report this access model difference. | https://arxiv.org/abs/2401.18079 |

## Paper Decision

- single highest priority: Backport the live ICLR triage into COLM_v2 tables/figures now, while the next ICLR method branch is redesigned around a qualitatively new source-causal interface rather than another deterministic PQ transform, scalar integrity gate, source-score selector, or target-native soft-prefix decoder.
- COLM_v2 claim: LatentWire_v2 demonstrates byte-scale source-private packet transfer with strict destructive controls, plus explicit negative gates that prevent overclaiming cross-family latent communication.
- ICLR claim not yet supported: Sparse Resonance Packets as a broad learned cross-model communication method.

## Next Exact Gate

- name: `new_interface_or_colm_v2_integration_gate`
- primary path: Stop HellaSwag cached-selector work and stop conditional-PQ held-out-family rescues that only add deterministic public transforms, no-op weight sweeps, or scalar integrity thresholds. The next ICLR branch needs a qualitatively new source-causal interface, such as a learned bottleneck/resampler preflight, or a benchmark where source quality and complementarity are easier to separate from packet artifacts.
- fallback path: If the next method branch is not implementable Mac-locally in one turn, switch to COLM_v2 table and figure integration using conditional-PQ, fixed-byte HellaSwag, target-resonance capacity, complementarity-frontier saturation, multi-signal packet failure, and scalar-integrity failure.
- pass bar: A learned or rule-based packet receiver must improve over source-index/rank/score, same-byte text, wrong-source, candidate-roll, and target-derived controls with a positive paired CI95 low on a frozen slice.

## Claim Boundaries

- Do not claim broad cross-family latent communication yet.
- Do not claim sparse resonance packets as a positive method from the current PCA or sparse-atom gates.
- Do not claim C2C/KV throughput, HBM, PCIe, NVLink, or energy wins without native measurements.
- Frame C2C/KVComm/TurboQuant/KIVI as high-bandwidth or KV-compression baselines with different exposure regimes.
