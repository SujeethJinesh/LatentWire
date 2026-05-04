# LatentWire COLM_v2 Review Packet

- created UTC: `2026-05-04T23:15:34.883619+00:00`
- COLM_v2 readiness: `scoped_positive_ready_for_writeup_if_claims_are_narrow`
- ICLR readiness: `blocked_by_lack_of_broad_or_learned_positive_receiver`

## Current Story

LatentWire_v2 can currently support a scoped COLM_v2 story: byte-scale, source-private packets plus strict destructive controls. The previously caveated OpenBookQA train-only receiver row is now weakened by same-source-choice wrong-row hardening. The ICLR story is still blocked because cross-family conditional PQ, deterministic public-basis conditioning, scalar integrity thresholds, ARC atom packets, OpenBookQA receiver fusion, and HellaSwag learned/source-conditioned resonance receivers have not produced a broad positive row beyond packet/source-choice/target-cache controls.

## Exact Submission Gaps

- COLM_v2: review-facing packaging: final tables, figures, baseline matrix, claim audit, and artifact manifest
- ICLR: ICLR needs a positive learned or broader-benchmark receiver that passes strict destructive controls with per-seed stability and source-choice separation. COLM_v2 can be prepared around the conditional-PQ shared-schema method, the fixed-byte HellaSwag packet row, OpenBookQA hardening as a negative diagnostic, and the target-resonance capacity-versus-held-out-failure analysis with explicit limitations.

## Current Technical Contributions

| contribution | status | needs_work | colm_v2_role |
| --- | --- | --- | --- |
| source_private_low_rate_packets | alive_for_colm_v2 | broader or learned receiver evidence for ICLR | narrow positive-method evidence |
| strict_destructive_controls | strong | paper integration and compact tables | core evaluation contribution |
| systems_byte_accounting | mac_local_ready | native dense-KV/C2C measurements before throughput or energy claims | systems framing contribution |
| sparse_resonance_packets | framing_alive_method_not_yet_positive | new mechanism beyond deterministic PQ, PCA/behavior atoms, BatchTopK-style atom banks, chunk encoders, query resamplers, and source-to-prefix decoders | narrow positive-method evidence |
| train_only_packet_target_receiver | openbookqa_weakened_by_source_choice_control | a packet that carries row-specific source evidence beyond same-source-choice wrong-row packets | narrow positive-method evidence |
| target_self_resonance_capacity_probe | capacity_alive | held-out/source-private receiver that beats slots-only, zero-source, wrong-source, source-choice, and candidate-roll controls | supporting paper contribution |

## Main Results

Rows that can appear in the scoped workshop story, with guarded wording.

| section | branch | status | score | baseline | delta | ci95_low | record_bytes | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| colm_v2_core | Conditional PQ shared-schema packet | promote_for_colm_v2_only | 16 | 16 | 0.658 | 0.658 | 7 | Use as the COLM_v2 positive method boundary; not enough for ICLR generality. |
| colm_v2_core | HellaSwag fixed hybrid candidate packet | promote_for_colm_v2_systems_baseline | 0.532464 | 0.526688 | 0.005776 | 0.002888 | 4 | Useful COLM_v2 systems/privacy row, but not a learned latent receiver. |
| colm_v2_supporting | Conditional PQ scalar integrity threshold | ruled_out_simple_integrity_threshold | 0.425781 | 0.457031 | -0.03125 | -0.097656 | 7 | Do not continue scalar threshold integrity on this public-zscore conditional-PQ receiver. |
| colm_v2_supporting | OpenBookQA train-only packet+target receiver | weakened_openbookqa_source_choice_control | 0.424 | 0.422 | 0.002 | 0.00795 | 3 | Do not promote as a positive second-benchmark row; the same-source-choice wrong-row control nearly matches the receiver, so it is a source-choice artifact diagnostic. |
| colm_v2_supporting | Target self-resonance oracle soft-prefix capacity | capacity_alive_not_source_private_method | 0.9375 | 0.625 | 0.3125 |  |  | Use only as capacity/headroom evidence; it optimizes on eval rows. |
| colm_v2_supporting | HellaSwag complementarity-frontier selector diagnostic | headroom_alive_selector_blocked | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not train another HellaSwag selector on the same packet fields; require a new information path. |
| colm_v2_supporting | HellaSwag multi-signal source packet frontier | ruled_out_cached_policy_packet | 0.455729 | 0.467448 | -0.011719 | -0.02347 | 5 | Do not continue cached Qwen policy-prediction packets on this HellaSwag slice. |

## Strict Controls

OpenBookQA hardening is included because it is the cleanest warning that source-choice artifacts can mimic packet gains.

| benchmark | seed | condition | receiver_accuracy | matched_minus_condition | base_accuracy | target_public_accuracy | override_rate | help_count | harm_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenBookQA | 47 | candidate_derangement | 0.292 | 0.132 | 0.21 | 0.372 | 0.29 | 66 | 25 |
| OpenBookQA | 47 | candidate_roll_packet | 0.292 | 0.132 | 0.21 | 0.372 | 0.29 | 66 | 25 |
| OpenBookQA | 47 | label_permutation | 0.424 | 0 | 0.378 | 0.372 | 0.28 | 58 | 35 |
| OpenBookQA | 47 | matched_source_private_packet | 0.424 | 0 | 0.378 | 0.372 | 0.28 | 58 | 35 |
| OpenBookQA | 47 | random_same_byte_packet | 0.37 | 0.054 | 0.228 | 0.372 | 0.842 | 135 | 64 |
| OpenBookQA | 47 | same_byte_structured_text | 0.378 | 0.046 | 0.35 | 0.372 | 0.196 | 32 | 18 |
| OpenBookQA | 47 | same_source_choice_wrong_row_packet | 0.422 | 0.002 | 0.378 | 0.372 | 0.414 | 76 | 54 |
| OpenBookQA | 47 | shuffled_source_packet | 0.364 | 0.06 | 0.25 | 0.372 | 0.8 | 127 | 70 |
| OpenBookQA | 47 | source_label_copy | 0.372 | 0.052 | 0.378 | 0.372 | 0.96 | 115 | 118 |
| OpenBookQA | 47 | target_derived_sidecar | 0.352 | 0.072 | 0.276 | 0.372 | 0.312 | 61 | 23 |
| OpenBookQA | 47 | target_public_ridge | 0.372 | 0.052 | 0.372 | 0.372 | 0.836 | 0 | 0 |
| OpenBookQA | 47 | zero_source | 0.372 | 0.052 | 0.276 | 0.372 | 0.95 | 126 | 78 |

## Systems Boundary

Paper-ready systems boundary artifact: LatentWire cached-source rows count the fixed-byte source-private packet object; paired end-to-end rows disclose source scoring separately. KV/cache rows are byte floors or pending native serving baselines. The artifact supports byte/exposure accounting, not a native C2C/KVComm/TurboQuant/QJL/vLLM/SGLang win.

| row_group | method | communicated_object | raw_bytes | framed_bytes | cacheline_bytes | batch64_bytes | source_private | source_kv_exposed | native_measured | claim_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LatentWire packet | LatentWire ARC-Challenge test packet (cached source) | cached-source task-level candidate evidence packet | 8 | 11 | 64 | 704 | true | false | false | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire packet | LatentWire ARC-Challenge test packet (source scoring disclosed) | same packet with source scoring disclosed separately | 8 | 11 | 64 | 704 | true | false | false | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire packet | LatentWire OpenBookQA test packet (cached source) | cached-source task-level candidate evidence packet | 3 | 6 | 64 | 384 | true | false | false | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire packet | LatentWire OpenBookQA test packet (source scoring disclosed) | same packet with source scoring disclosed separately | 3 | 6 | 64 | 384 | true | false | false | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire packet | LatentWire HellaSwag validation_first1024 packet (cached source) | cached-source task-level candidate evidence packet | 2 | 5 | 64 | 320 | true | false | false | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire packet | LatentWire HellaSwag validation_first1024 packet (source scoring disclosed) | same packet with source scoring disclosed separately | 2 | 5 | 64 | 320 | true | false | false | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire packet | LatentWire HellaSwag validation_full_compaction packet (cached source) | cached-source task-level candidate evidence packet | 1 | 4 | 64 | 256 | true | false | false | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire packet | LatentWire HellaSwag validation_full_compaction packet (source scoring disclosed) | same packet with source scoring disclosed separately | 1 | 4 | 64 | 256 | true | false | false | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| local control | Same-byte structured text control (ARC) | text-form control with same packet budget | 8 | 11 | 64 | 704 | false | false | true | negative/control row only; cannot support source-private claim |
| state relay floor | Four-choice fp16 score-vector relay floor | source score vector, fp16 | 8 | 8 | 64 | 512 | false | false | false | source-state exposure floor only; not a source-private packet |
| state relay floor | Four-choice fp16 logit-vector relay floor | source logit vector, fp16 | 8 | 8 | 64 | 512 | false | false | false | source-state exposure floor only; not a source-private packet |
| state relay floor | One hidden-vector fp16 relay floor | one source hidden vector, fp16 | 1792 | 1792 | 1792 | 114688 | false | false | false | state-exposure lower bound only; not a baseline win |
| KV/source-state floor | 1-bit/KV-element accounting floor | one-token K+V state at 1 bit/element | 768 | 768 | 768 | 49152 | false | true | false | mathematical state-size lower bound only |
| KV/source-state floor | KIVI 2-bit KV floor | one-token K+V state at 2 bits/element | 1536 | 1536 | 1536 | 98304 | false | true | false | KV-cache compression comparator only |
| KV communication floor | Q-KVComm optimistic 6x floor | compressed source KV cache representation | 2048 | 2048 | 2048 | 131072 | false | true | false | compressed-KV communication boundary only |
| KV/source-state floor | KVQuant 3-bit proxy floor | one-token K+V state at 3 bits/element | 2304 | 2304 | 2304 | 147456 | false | true | false | sub-4-bit KV comparator only |
| KV/source-state floor | TurboQuant 3.5-bit KV floor | one-token K+V state at 3.5 bits/element | 2688 | 2688 | 2688 | 172032 | false | true | false | KV/vector quantization comparator only |
| KV communication floor | KVComm 30% fp16 KV floor | selected source KV layers, fp16 | 3686.4 | 3686.4 | 3712 | 235930 | false | true | false | selective-KV communication boundary only |
| KV communication floor | C2C one-token fp16 KV floor | projected/fused source KV cache | 12288 | 12288 | 12288 | 786432 | false | true | false | closest cache-transfer baseline; native run still required |
| KV communication floor | KVCOMM cross-context fp16 KV floor | aligned/reused source KV cache | 12288 | 12288 | 12288 | 786432 | false | true | false | systems neighbor only; native run still required |
| serving substrate | vLLM/PagedAttention one-token KV floor | paged KV-cache serving substrate | 12288 | 12288 | 12288 | 786432 | true | false | false | native TTFT/TPOT/goodput/HBM target, not closed on Mac |
| serving substrate | SGLang/RadixAttention one-token KV floor | KV-cache reuse serving substrate | 12288 | 12288 | 12288 | 786432 | true | false | false | native TTFT/TPOT/goodput/HBM target, not closed on Mac |

## Baseline Matrix

| category | baseline | what_transfers | source_private | byte_regime | included_in_current_eval | latentwire_distinction | still_needed | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_cache_transfer | Cache-to-Cache (C2C) | projected source KV cache fused into target KV cache | no, dense source cache state crosses the interface | high bandwidth KV/cache state | systems-boundary comparator only | LatentWire transmits tiny source-private packets and evaluates utility per byte; it does not claim raw-accuracy or native serving wins over C2C. | direct native C2C run on matching models/tasks for any stronger claim | https://openreview.net/forum?id=LeatkxrBCi |
| dense_cache_transfer | Latent Space Communication via K-V Cache Alignment | aligned prefix K/V cache from source model into target model decoding | no, aligned cache state crosses the boundary | dense K/V cache state | related-work boundary | K/V cache alignment is a direct dense latent-communication competitor; LatentWire must claim only low-rate packet transfer unless direct cache alignment rows are run. | direct comparison if claiming broad latent communication novelty | https://arxiv.org/abs/2601.06123 |
| kv_cache_serving | DroidSpeak | compatible KV caches across distributed LLM nodes | no, cache state is reused | KV/cache state | related-work and systems-boundary comparator | DroidSpeak is cache reuse; LatentWire studies source-private task packets with wrong-row and source-choice controls. | native serving comparison only after GPU setup | https://arxiv.org/abs/2411.02820 |
| kv_cache_serving | KVCOMM | cross-context KV-cache information for multi-agent inference | no, cache information crosses the agent boundary | KV/cache state | related-work and systems-boundary comparator | KVCOMM is a high-bandwidth cache communication regime; LatentWire is a low-byte packet protocol with destructive source-private controls. | native baseline if claiming runtime superiority | https://arxiv.org/abs/2510.12872 |
| kv_cache_serving | RelayCaching | decoding-time KV caches for collaborative generation | no, produced cache state is reused | KV/cache state | related-work and systems-boundary comparator | RelayCaching accelerates cache reuse; LatentWire evaluates whether compact source evidence causes target utility without exposing cache/text. | native serving baseline for latency or throughput claims | https://arxiv.org/abs/2603.13289 |
| kv_cache_reuse | CacheGen / LMCache / CacheBlend-style reuse | compressed, streamed, or blended reusable prompt/cache state | no, cache or prompt-derived state is reused | compressed cache/prompt state | systems-boundary comparator | Cache-reuse systems optimize serving reuse for existing context; LatentWire sends small task-level source evidence packets across model boundaries. | native vLLM/SGLang/LMCache run for throughput claims | https://arxiv.org/abs/2310.07240 |
| activation_communication | Communicating Activations Between Language Model Agents | activation vectors between language-model agents | limited; activation vectors are exposed | dense activations | related-work boundary | LatentWire emphasizes source-private low-byte packets, explicit packet bytes, and destructive controls rather than generic activation exchange. | cite and distinguish in related-work matrix | https://arxiv.org/abs/2501.14082 |
| activation_communication | CIPHER / Let Models Speak Ciphers | embedding-level messages between debating LLM agents | partial; embedding messages are exposed to peers | embedding vectors or continuous messages | related-work boundary | CIPHER pressures broad latent-message novelty; LatentWire's narrower claim is byte-counted source-private packets with destructive controls. | avoid claiming generic embedding communication novelty | https://arxiv.org/abs/2310.06272 |
| latent_translation | Direct Semantic Communication via Vector Translation | translated semantic vectors | partial; dense vectors may expose source state | dense vector communication | related-work boundary | LatentWire requires low-byte source-private packet accounting and source-choice destructive controls. | verify final citation metadata before camera-ready | https://arxiv.org/abs/2511.03945 |
| latent_translation | InterLat | intermediate latent representations | partial; dense latent state may cross | latent vector state | related-work boundary | LatentWire claims only packetized, byte-counted, source-private transfer under destructive controls. | verify final citation metadata before camera-ready | https://arxiv.org/abs/2511.09149 |
| soft_prompting | Prefix tuning | task-specific continuous prefix parameters | not a source-to-target communication protocol | learned prompt parameters | conceptual baseline | Prefix tuning adapts one model; LatentWire tests row-specific source evidence packets with wrong-row and source-choice controls. | baseline wording in related work | https://aclanthology.org/2021.acl-long.353/ |
| context_compression | Gist tokens | learned summary tokens for prompt compression | no, compression is of visible context | soft/visible compressed context | conceptual baseline | Gist compresses a model's context; LatentWire transfers hidden source evidence between models with source-private controls. | same-byte visible text baseline remains the local control | https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html |
| context_compression | LLMLingua / LongLLMLingua | compressed visible prompt text or retained prompt tokens | no, visible prompt content is compressed and exposed | compressed text/token context | same-byte visible text is the local control | LLMLingua compresses visible context; LatentWire tests whether opaque source-private packets add utility beyond same-byte visible text. | optional stronger visible-text compression baseline for ICLR | https://arxiv.org/abs/2310.05736; https://arxiv.org/abs/2310.06839 |
| query_bottleneck | BLIP-2 / Q-Former | query bottleneck between vision encoder and language model | not a source-to-target LLM communication protocol | learned query token bottleneck | architectural precedent | LatentWire's novelty is not query bottlenecks; it is packetized source-private model-to-model transfer plus destructive controls. | cite as precedent, not baseline claim | https://arxiv.org/abs/2301.12597 |
| query_bottleneck | Flamingo / Perceiver Resampler | fixed latent resampler tokens into LM cross-attention | not source-private model-to-model communication | latent token bottleneck | architectural precedent | LatentWire differs by source-private packets, byte accounting, and controls against source-choice artifacts. | cite as precedent, not novelty | https://arxiv.org/abs/2204.14198 |
| query_bottleneck | Perceiver IO | latent query array for generic input/output mapping | not a source-to-target LLM communication protocol | latent array | architectural precedent | Latent arrays are prior art; LatentWire's claim is communication protocol evaluation under strict controls. | cite as motivation only | https://arxiv.org/abs/2107.14795 |
| common_basis | Sparse Crosscoders | shared and private features across activation spaces | feature analysis, not packet transfer | sparse feature coordinates if packetized | related-work boundary | Common bases alone are not novel; LatentWire must show sparse features cause source-private downstream utility under atom-shuffle and wrong-row controls. | future SRP method should compare directly if using crosscoders | https://transformer-circuits.pub/2024/crosscoders/ |
| common_basis | Universal Sparse Autoencoders | cross-model sparse concept basis | feature analysis, not packet transfer | sparse coordinates if packetized | related-work boundary | LatentWire uses common-basis work only as a possible packet representation; novelty requires byte-counted causal packet utility. | future SRP baseline if PCA succeeds | https://arxiv.org/abs/2502.03714 |
| common_basis | Transcoders / sparse feature circuits | sparse feature-to-feature computation approximating model internals | analysis method, not direct packet transfer | sparse feature coordinates if packetized | related-work boundary | Transcoders can define interpretable atoms, but LatentWire must prove that packetized atoms causally improve target behavior under strict controls. | strong future baseline for behavior-transcoder packet branch | https://arxiv.org/abs/2406.11944 |
| common_basis | SAEBench and SAE non-canonicity checks | benchmarking and reliability checks for sparse feature dictionaries | analysis/evaluation method | not a communication protocol | related-work boundary | SAE quality and feature stability are prerequisites if Sparse Resonance Packets use sparse atoms; they are not themselves the communication result. | use if upgrading from PCA/SVD to SAE/crosscoder packets | https://proceedings.mlr.press/v267/karvonen25a.html |
| quantization | TurboQuant | low-bit quantized vectors/KV state | no, quantized source state still exposes state | low-bit dense vector/KV compression | systems byte floor comparator | TurboQuant reduces dense state cost; LatentWire sends task-level packets and must avoid unmeasured throughput claims. | native comparison after NVIDIA hardware | https://arxiv.org/abs/2504.19874 |
| quantization | KVQuant | low-bit KV cache for long-context serving | no, cache is still exposed/reused | low-bit KV cache | systems byte floor comparator | KVQuant compresses one model's cache; LatentWire transfers compact source evidence across models. | native comparison after NVIDIA hardware | https://arxiv.org/abs/2401.18079 |
| quantization | KIVI / QJL low-bit KV sketches | low-bit KV/cache values or sketch-corrected quantized vectors | no, dense state is compressed rather than hidden | low-bit dense cache/vector state | systems byte floor comparator | These methods shrink dense state; LatentWire transfers task-level packets and should compare byte floors without claiming native speed. | native low-bit cache comparison for ICLR systems claims | https://arxiv.org/abs/2402.02750; https://arxiv.org/abs/2504.19874 |
| kv_cache_compression | H2O | retained heavy-hitter KV cache tokens | no, same-model cache retention | pruned KV cache | related-work boundary | H2O reduces one model's cache memory; LatentWire sends source-private cross-model evidence packets. | long-context reviewer baseline, not direct COLM_v2 competitor | https://arxiv.org/abs/2306.14048 |
| kv_cache_compression | SnapKV | selected KV cache tokens before generation | no, same-model cache compression | pruned KV cache | related-work boundary | SnapKV is same-model long-context compression; LatentWire is source-to-target packet communication. | systems appendix comparator if adding long-context tasks | https://arxiv.org/abs/2404.14469 |
| kv_cache_compression | Quest | query-aware sparse KV pages for long-context inference | no, same-model cache sparsity | sparse KV page loading | related-work boundary | Quest optimizes attention cache loading; LatentWire optimizes byte-limited model-to-model evidence transfer. | native systems appendix if reviewers demand long-context baselines | https://arxiv.org/abs/2406.10774 |
| kv_cache_compression | KVzip | query-agnostic compressed KV cache with reconstruction objective | no, same-model cache compression | compressed KV cache | related-work boundary | KVzip is cache storage compression; LatentWire is packetized cross-model communication with source-private controls. | appendix context if broad systems framing expands | https://arxiv.org/abs/2505.23416 |
| serving_substrate | vLLM / PagedAttention | paged KV-cache serving substrate | not a communication method | serving memory-management baseline | native-system blocker only | vLLM is a required runtime baseline for TTFT/TPOT/goodput/HBM once GPU measurements are available. | native NVIDIA run before any latency/HBM claim | https://dl.acm.org/doi/10.1145/3600006.3613165 |
| serving_substrate | SGLang / RadixAttention | structured LM serving runtime with KV reuse | not a communication method | serving memory-management baseline | native-system blocker only | SGLang is a second serving substrate for native claims; Mac byte accounting cannot replace it. | native NVIDIA run before any latency/HBM claim | https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html |
| local_control | source-index / source-label copy | the source's preferred answer index or label | yes but artifact-prone | 1-2 bytes | yes | LatentWire must beat this to show more than answer-choice copying. | keep in all packet tables | internal destructive control |
| local_control | source-rank / source-score quantization | source ranking or quantized score vector | yes but may reveal source choice behavior | few bytes | yes | LatentWire must exceed score/rank packet controls to claim richer communication. | include in claim audit | internal destructive control |
| local_control | same-byte visible text | human-readable text within the same byte budget | no, visible text crosses the boundary | same byte budget as packet | yes | LatentWire packets must justify why opaque source-private bytes are useful relative to visible text at the same byte budget. | include in strict-control table | internal destructive control |
| local_control | wrong-row / shuffled / target-derived packet | noncausal or target-only packet with matched format | yes | same packet budget | yes | Passing these controls is the main evidence that the packet is row-specific source communication rather than receiver artifact. | keep as hard gate for ICLR | internal destructive control |

## Negative Results And Saturation

| branch | status | score | baseline | delta | ci95_low | record_bytes | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Conditional PQ public-zscore held-out-family decoder | ruled_out_as_cross_family_rescue | 0.453125 | 0.441406 | 0.011719 | -0.085938 | 7 | Do not widen public-zscore to n500/remap. |
| Conditional PQ public-SVD whitening held-out-family decoder | ruled_out_as_cross_family_rescue | 0.375 | 0.613281 | -0.238281 | -0.32041 | 7 | Do not continue deterministic public-basis whitening as the rescue path. |
| Conditional PQ corruption-to-noop receiver | ruled_out_as_cross_family_rescue | 0.25 | 0.25 | 0 | 0 | 7 | Keep as diagnostic; do not promote this receiver family. |
| Conditional PQ scalar integrity threshold | ruled_out_simple_integrity_threshold | 0.425781 | 0.457031 | -0.03125 | -0.097656 | 7 | Do not continue scalar threshold integrity on this public-zscore conditional-PQ receiver. |
| ARC sparse resonance PCA/target-aligned soft-prefix packets | ruled_out_current_basis_receiver | 0.25 | 0.625 | -0.375 |  |  | Do not widen plain PCA/target-aligned soft-prefix packets. |
| ARC behavior-supervised atom SRP decoder | ruled_out_current_behavior_atom_basis | 0.4375 | 0.4375 | 0 | -0.3125 | 7 | Keep the strict atom harness, but do not scale the current behavior/PCA/BatchTopK-style atom basis without a new source-causality mechanism. |
| OpenBookQA train-only packet+target receiver | weakened_openbookqa_source_choice_control | 0.424 | 0.422 | 0.002 | 0.00795 | 3 | Do not promote as a positive second-benchmark row; the same-source-choice wrong-row control nearly matches the receiver, so it is a source-choice artifact diagnostic. |
| HellaSwag protected top-2/rival packet | ruled_out_shallow_pair_decoder | 0.46224 | 0.467448 | -0.005208 | -0.013021 | 5 | Do not rerun generic protected top-2/rival switchers. |
| HellaSwag official-train receiver-calibrated selector | weakened_near_miss | 0.466146 | 0.467448 | -0.001302 | -0.007812 | 5 | Do not continue shallow linear score-feature selectors without new evidence. |
| HellaSwag harm-controlled complementarity buckets | ruled_out_no_safe_bucket | 0.467448 | 0.467448 | 0 | 0 | 6 | Low-harm bucket receiver is saturated. |
| HellaSwag top1/top2 ambiguity buckets | ruled_out_no_safe_bucket | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not continue rank/score-bin top2 buckets without a new packet field. |
| HellaSwag denoising syndrome packet | ruled_out_shallow_syndrome_decoder | 0.463542 | 0.467448 | -0.003906 | -0.010417 | 4 | Do not promote ridge-denoising syndrome decoder. |
| HellaSwag sparse/common-basis top2 ambiguity code | weakened_common_basis_atom | 0.500977 | 0.501953 | -0.000977 | -0.00293 | 4 | Do not claim sparse atom causality from this branch. |
| Target self-resonance oracle soft-prefix capacity | capacity_alive_not_source_private_method | 0.9375 | 0.625 | 0.3125 |  |  | Use only as capacity/headroom evidence; it optimizes on eval rows. |
| Target self-resonance held-out learned prefix encoders | ruled_out_current_target_native_encoder_family | 0.6875 | 0.6875 | 0 |  |  | Do not run more chunk/distill/query-resampler variants without a new information path. |
| Source-conditioned target-native resonance receivers | ruled_out_current_source_conditioned_receiver_family | 0.375 | 0.375 | 0 |  |  | Diagnose complementarity/gating before implementing another source-to-prefix decoder. |
| HellaSwag complementarity-frontier selector diagnostic | headroom_alive_selector_blocked | 0.467448 | 0.467448 | 0 | 0 | 4 | Do not train another HellaSwag selector on the same packet fields; require a new information path. |
| HellaSwag multi-signal source packet frontier | ruled_out_cached_policy_packet | 0.455729 | 0.467448 | -0.011719 | -0.02347 | 5 | Do not continue cached Qwen policy-prediction packets on this HellaSwag slice. |

## Reviewer Claim Audit

| claim | support_level | safe_wording | evidence | reviewer_risk |
| --- | --- | --- | --- | --- |
| LatentWire_v2 provides a source-private packet-transfer evaluation framework. | supported_for_colm_v2 | We introduce a source-private, byte-accounted evaluation framework with matched, wrong-row, shuffled, target-derived, and same-byte controls. | evidence_table, live_branch_triage, OpenBookQA hardening artifact | low if framed as framework plus strict controls |
| Tiny packets can show narrow utility under strict controls. | supported_narrowly | On selected source-private rows, fixed-byte or conditional packet methods show positive utility with paired uncertainty, but the effect is not broad. | conditional PQ status and HellaSwag fixed hybrid validation | medium; emphasize narrow scope and controls |
| Many apparent wins collapse under source-choice or wrong-row controls. | strongly_supported | OpenBookQA hardening shows that a matched receiver's apparent advantage is nearly matched by the same-source-choice wrong-row control. | OpenBookQA receiver headroom gate | low; this is an honest negative result |
| LatentWire is more byte-efficient than dense cache transfer regimes. | supported_as_byte_accounting_only | The communicated packet object is far smaller than dense source-state or KV/cache floors, but native runtime and task-utility superiority are unmeasured. | systems boundary artifact | medium; avoid latency/energy/HBM claims |
| LatentWire beats C2C. | not_supported | LatentWire studies a different low-rate source-private point in the design space and uses C2C as a high-bandwidth baseline/contrast. | no direct native C2C run | high; do not claim |
| Sparse Resonance Packets are a broad positive ICLR method. | not_supported_yet | Sparse Resonance Packets remain the ICLR direction; current evidence motivates the next tokenwise/source-causal gate but does not establish broad utility. | live branch triage blocker rows | high; keep out of COLM_v2 headline |

## Figure Data

- `figure_data_evidence_ladder.csv`: evidence ladder for core, guardrail, and blocker rows.
- `figure_data_protocol_controls.csv`: protocol/control schematic data.
- `figure_data_systems_boundary.csv`: byte/exposure rows for systems boundary plots.

## Reproducibility Manifest

| key | path | sha256 |
| --- | --- | --- |
| evidence_table | results/latentwire_colm_v2_iclr_evidence_table_20260504/evidence_table.json | a566e42bf445e9bba6859993251b58c927242d3890eea14d4d6d91c8678bd703 |
| live_branch_triage | results/iclr_colm_v2_live_branch_triage_20260504/live_branch_triage.json | abea8804bedb3f7056b6b3c0b681039b3ed38aa94882dfba47d8321ec3d840c5 |
| conditional_pq_status | results/source_private_conditional_pq_iclr_colm_v2_status_20260504/conditional_pq_iclr_colm_v2_status.json | 3d6d2f760c7695eb91169cb7a3093429a39c12f96542eba25ce35e327f4afdd7 |
| hellaswag_fixed_hybrid | results/source_private_hellaswag_fixed_hybrid_full_validation_gate_20260503_validation0_10042/hellaswag_fixed_hybrid_full_validation_gate.json | 0fd7d311cec8c8aac0de60de1193246d9cfb86642d407355bd815cb15d82456d |
| openbookqa_receiver_headroom | results/source_private_openbookqa_receiver_headroom_gate_20260502/openbookqa_receiver_headroom_gate.json | 7b451ba0e16e3793511d0d1e659c2e550ee196745b67134a420b4e7ea7616e8c |
| systems_boundary | results/source_private_systems_boundary_figure_table_split_20260504/systems_boundary_figure_data.json | 81a776e985428f6945c22ccc3b7921016afb4f0a2e32d4b4a7af4b04cea9ee49 |

## Next Exact Gate

- name: `arc_n32_tokenwise_source_evidence_preflight`
- primary path: Materialize a tiny ARC n32 tokenwise source-evidence cache and run a target-loss connector preflight on source-unique repair rows. Existing ARC/HellaSwag caches are mean-pooled and the OpenBookQA hardening now shows score/choice receiver fusion is not source-causal enough.
- fallback path: If local model loading is not feasible on the Mac, run a target-side behavior-transcoder feasibility probe from available target traces, then only packetize source atoms after target atoms causally steer margins.
- pass bar: A learned or rule-based packet receiver must improve over source-index/rank/score, same-byte text, wrong-source, same-source-choice wrong-row, candidate-roll, and target-derived controls with a positive paired CI95 low on a frozen slice.

## Claim Boundaries

- Do not claim broad cross-family latent communication yet.
- Do not claim sparse resonance packets as a positive method from the current PCA or sparse-atom gates.
- Do not claim C2C/KV throughput, HBM, PCIe, NVLink, or energy wins without native measurements.
- Frame C2C/KVComm/TurboQuant/KIVI as high-bandwidth or KV-compression baselines with different exposure regimes.
