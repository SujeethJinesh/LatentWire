# LatentWire COLM v3 Review Packet

- created_utc: `2026-05-05T06:15:29.952112+00:00`
- main_claim: LatentWire provides a practical protocol and evaluation framework for source-private candidate-transfer packets, with controlled evidence of narrow fixed-byte packet utility, explicit utility-per-byte accounting, and destructive controls that expose shortcut claims.
- next_exact_gate: human copyedit, page-budget review, final PDF/table placement, and consistency check between paper, review packet, and artifact manifest

## Readiness

| colm_v3 | workshop_blocker | iclr |
| --- | --- | --- |
| reviewer_hardened_draft_pending_human_review | human copyedit, page-budget review, and final PDF/table placement; no new speculative experiment is required unless review exposes a missing claim-supporting row | still blocked by lack of broad source-causal positive method |

## Contribution Status

| contribution | status | evidence | still_needs_work |
| --- | --- | --- | --- |
| source-private packet protocol | supported_for_colm_v3 | packet rows and source-private interface definition | paper prose must avoid broad latent-language claims |
| strict destructive controls | supported_for_colm_v3 | wrong-row, same-source-choice, source-index/rank/score, same-byte/text controls | compress into one main table plus appendix |
| narrow low-byte packet utility | supported_but_narrow | main_results.csv | state that source-index remains a strong boundary |
| systems byte and exposure accounting | supported_as_accounting | systems_measured_vs_estimated.csv | no native GPU/HBM/energy claim until NVIDIA runbook is executed |
| broad positive latent communication method | not_supported_for_colm_v3 | negative_results.csv and ICLR triage | keep as ICLR future method target |
| source_private_low_rate_packets | alive_for_colm_v2 |  |  |
| strict_destructive_controls | strong |  |  |
| systems_byte_accounting | mac_local_ready |  |  |
| sparse_resonance_packets | framing_alive_method_not_yet_positive |  |  |
| train_only_packet_target_receiver | openbookqa_weakened_by_source_choice_control |  |  |
| target_self_resonance_capacity_probe | capacity_alive |  |  |

## Reviewer Claim Audit

| claim | support_level | evidence_artifact | controls_passed | required_wording |
| --- | --- | --- | --- | --- |
| LatentWire defines a source-private candidate-transfer packet protocol and strict evaluation framework. | supported | COLM_v2 review packet plus COLM_v3 review packet | source-private interface, wrong-row/source-choice controls where available | safe as a protocol/evaluation contribution |
| Low-byte packets show narrow same-family utility on ARC-style rows. | supported_but_narrow | main_results.csv; strict_controls.csv; systems_measured_vs_estimated.csv | target-only and same-byte/text controls on the reported rows; source-index remains a hard boundary | narrow source-private candidate-transfer utility, not broad latent communication |
| The current packet beats source-index communication or selected-candidate codes. | not_supported | main_results.csv; source-index audit | packet-source lower bounds remain negative or zero | do not claim; source-index is the main boundary |
| Many apparent wins collapse into source-choice, source-rank, or target-cache artifacts. | supported | negative_results.csv; source-choice controls; reviewer feedback | same-source-choice wrong-row, source-index/rank/score, and destructive controls where available | use as a reviewer-strengthening result, not as the headline alone |
| LatentWire beats C2C or dense KV/cache transfer. | not_supported | systems boundary table only | none; no native matched C2C row | do not claim; compare as byte/exposure boundary only |
| LatentWire has native GPU latency, HBM, energy, or throughput wins. | not_supported | NVIDIA native benchmark runbook | not run | future work until native measurements exist |
| LatentWire solves broad latent model-to-model communication or cross-family transfer. | not_supported | negative_results.csv; cross-family failure rows | cross-family falsification weakened the broad claim | do not claim; present as an open ICLR goal |

## Table And Figure Inventory

| artifact | status | source | next_action |
| --- | --- | --- | --- |
| unified abstract and introduction | draft_integrated | colm_final/paper/latentwire_colm2026.tex | human copyedit and page-budget review |
| method/protocol definition | draft_integrated | COLM_v1 method intuition plus COLM_v2 packet protocol | verify notation consistency after copyedit |
| source-private threat model | draft_integrated | COLM_v2 controls and systems boundary notes | check against reviewer claim audit |
| strict-control table | draft_integrated | strict_controls.csv | validate table placement in PDF |
| main positive result table | draft_integrated_source_index_bounded | main_results.csv | keep ARC as narrow same-family positive evidence |
| uncertainty summary table | draft_integrated | source-index audit lower bounds | verify table placement in final PDF |
| utility-per-byte / packet-byte table | data_ready | systems_measured_vs_estimated.csv | separate raw, framed, cacheline, and batch64 bytes |
| systems boundary table | draft_integrated | systems_measured_vs_estimated.csv | validate measured-vs-estimated labels in PDF |
| baseline/related-work matrix | draft_integrated | baseline_matrix.csv | check for overflow and page-budget pressure |
| benchmark breadth audit | review_packet_integrated | benchmark_breadth.csv | use as reviewer-pack support; do not turn diagnostics into headline claims |
| latest model breadth audit | review_packet_integrated | latest_model_breadth.csv | treat as source-packet emitter smoke, not ARC/OBQA benchmark evidence |
| negative-results / failure-boundary table | data_ready | negative_results.csv | use to define claim boundaries |
| claim audit table | draft_integrated | claim_audit.csv | keep appendix or move to internal audit depending on page limit |
| reproducibility checklist | partial | artifact_manifest.csv and input_manifest | convert to workshop checklist before submission |
| NVIDIA native benchmark runbook | generated_future_work | nvidia_native_runbook.md | run only on native NVIDIA hardware later |
| ten-reviewer COLM stress panel | recorded | colm_final/audits/colm_v3_10_reviewer_panel_20260505.md | use for human copyedit and final reviewer-risk pass |

## Benchmark Breadth Audit

| benchmark | split | examples | packet_bytes | packet_accuracy | target_accuracy | same_byte_text_accuracy | paper_use | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenBookQA | test | 500 | 3 | 0.3784 | 0.276 | 0.35 | main_or_secondary_test_row | headline-safe only when paired with source-index and source-choice boundary wording |
| ARC-Challenge | test | 1172 | 12 | 0.343686 | 0.265358 | 0.31058 | main_or_secondary_test_row | headline-safe only when paired with source-index and source-choice boundary wording |
| CommonsenseQA | validation | 1221 | 2 | 0.438329 | 0.205569 | 0.424242 | diagnostic_not_headline | reviewer-pack diagnostic |
| HellaSwag | hellaswag_validation1024 | 1024 | 2 | 0.461133 | 0.233398 | 0.385742 | reviewer_pack_breadth_not_full_validation_headline | passes validation1024 seed gate; full-validation terminal tail blocks benchmark-complete claim |
| SciQ | validation/test materialized | 1000/1000 | 12 |  |  |  | future_gate_not_evidence | source-private split/control contract is frozen, but no positive packet row exists yet |

## Latest Model Breadth Audit

| model | params | architecture | device | local_rung | passes | max_n | paper_use | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-0.8B | 0.8B | qwen3_5 conditional generation | cpu | CPU n160 passed on seeds 29/31 | 4 | 160 | reviewer_pack_emitter_breadth | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |
| Qwen/Qwen3.5-2B | 2B | qwen3_5 conditional generation | cpu | CPU n16/n64/n160 passed | 3 | 160 | reviewer_pack_emitter_breadth | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |
| Qwen/Qwen3.5-4B | 4B | qwen3_5 conditional generation | cpu | CPU n16/n64 passed; n160 only if needed | 2 | 64 | reviewer_pack_emitter_breadth | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |
| allenai/OLMo-2-0425-1B-Instruct | 1B | dense | mps | cross-family local n16 |  |  | negative_or_unverified | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |
| ibm-granite/granite-3.3-2b-instruct | 2B | dense | cpu | CPU copied-helper n160 and trace-no-hint n160 passed | 2 | 160 | reviewer_pack_emitter_breadth | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |
| google/gemma-4-E2B-it | 2.3B effective | gemma4 conditional generation | mps | MPS trace-no-hint n500 passed; n160 seed repeat passed | 6 | 500 | reviewer_pack_emitter_breadth | source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence |

## Systems Measured Vs Estimated

| method | raw_bytes | framed_bytes | cacheline_bytes | batch64_bytes | measured_vs_estimated | claim_allowed |
| --- | --- | --- | --- | --- | --- | --- |
| LatentWire ARC-Challenge test packet (cached source) | 8 | 11 | 64 | 704 | measured_packet_object_bytes | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire ARC-Challenge test packet (source scoring disclosed) | 8 | 11 | 64 | 704 | local_partial_measurement_or_missing_phase_trace | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire OpenBookQA test packet (cached source) | 3 | 6 | 64 | 384 | measured_packet_object_bytes | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire OpenBookQA test packet (source scoring disclosed) | 3 | 6 | 64 | 384 | local_partial_measurement_or_missing_phase_trace | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire HellaSwag validation_first1024 packet (cached source) | 2 | 5 | 64 | 320 | measured_packet_object_bytes | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire HellaSwag validation_first1024 packet (source scoring disclosed) | 2 | 5 | 64 | 320 | local_partial_measurement_or_missing_phase_trace | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire HellaSwag validation_full_compaction packet (cached source) | 1 | 4 | 64 | 256 | measured_packet_object_bytes | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire HellaSwag validation_full_compaction packet (source scoring disclosed) | 1 | 4 | 64 | 256 | local_partial_measurement_or_missing_phase_trace | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| Same-byte structured text control (ARC) | 8 | 11 | 64 | 704 | measured_local_control_or_accounting_row | negative/control row only; cannot support source-private claim |
| Four-choice fp16 score-vector relay floor | 8 | 8 | 64 | 512 | analytical_or_literature_byte_floor | source-state exposure floor only; not a source-private packet |
| Four-choice fp16 logit-vector relay floor | 8 | 8 | 64 | 512 | analytical_or_literature_byte_floor | source-state exposure floor only; not a source-private packet |
| One hidden-vector fp16 relay floor | 1792 | 1792 | 1792 | 114688 | analytical_or_literature_byte_floor | state-exposure lower bound only; not a baseline win |
| 1-bit/KV-element accounting floor | 768 | 768 | 768 | 49152 | analytical_or_literature_byte_floor | mathematical state-size lower bound only |
| KIVI 2-bit KV floor | 1536 | 1536 | 1536 | 98304 | analytical_or_literature_byte_floor | KV-cache compression comparator only |
| Q-KVComm optimistic 6x floor | 2048 | 2048 | 2048 | 131072 | analytical_or_literature_byte_floor | compressed-KV communication boundary only |
| KVQuant 3-bit proxy floor | 2304 | 2304 | 2304 | 147456 | analytical_or_literature_byte_floor | sub-4-bit KV comparator only |
| TurboQuant 3.5-bit KV floor | 2688 | 2688 | 2688 | 172032 | analytical_or_literature_byte_floor | KV/vector quantization comparator only |
| KVComm 30% fp16 KV floor | 3686.4 | 3686.4 | 3712 | 235930 | analytical_or_literature_byte_floor | selective-KV communication boundary only |
| C2C one-token fp16 KV floor | 12288 | 12288 | 12288 | 786432 | analytical_or_literature_byte_floor | closest cache-transfer baseline; native run still required |
| KVCOMM cross-context fp16 KV floor | 12288 | 12288 | 12288 | 786432 | analytical_or_literature_byte_floor | systems neighbor only; native run still required |
| vLLM/PagedAttention one-token KV floor | 12288 | 12288 | 12288 | 786432 | analytical_or_literature_byte_floor | native TTFT/TPOT/goodput/HBM target, not closed on Mac |
| SGLang/RadixAttention one-token KV floor | 12288 | 12288 | 12288 | 786432 | analytical_or_literature_byte_floor | native TTFT/TPOT/goodput/HBM target, not closed on Mac |

## Experimental Side-Branch Scope

| experiment | colm_v3_scope | highest_value_gate | novelty_risk | status |
| --- | --- | --- | --- | --- |
| HybridKernel | separate systems spinout; exclude from COLM_v3 claims unless Phase 1 confirms novelty, Phase 2 shows at least 3% theoretical benefit, and native GPU profiling confirms overhead | vLLM hybrid SSM/disaggregated serving source audit | boundary fusion may already be covered by vLLM/vendor hybrid serving optimizations | phase1_quick_audit_complete_phase2_architecture_map_pending |
| SinkAware | separate systems spinout only; mention in COLM_v3 only as future work unless Phase 1-4 produce source-backed novelty plus a reference artifact | CPU-only fixed sink-token decomposition | learned/per-head sink denominator handling is already occupied by FlashInfer/FlashMLA/GPT-OSS-style paths | phase1_quick_audit_complete_narrow_phase2_or_kill |
| ThoughtFlow-FP8 | separate systems spinout candidate after Phase 1, not current COLM_v3 evidence | Mac-only trace simulation for protected reasoning-token classes | field is crowded by LongFlow, ThinKV, R-KV/R-KVHash, RaaS, LazyEviction, ForesightKV, and PM-KVQ | phase1_forensics_complete_high_upside_phase2_pending |

## Submission Checklist

| item | status | blocker |
| --- | --- | --- |
| Main claim agrees across abstract, intro, results, limitations. | reviewer_hardened_pending_human_review | requires human copyedit and page-budget review |
| Every table and figure maps to a claim in the claim audit. | draft_integrated | verify final PDF table placement |
| Systems claims separate measured packet bytes from analytical KV/cache floors. | ready | native GPU claims remain forbidden |
| Related work distinguishes dense KV/cache transfer, compression, and packet controls. | draft_integrated_compressed | page-budget review may require moving matrix to appendix |
| Limitations explicitly cover source-choice artifacts and cross-family failures. | draft_integrated | human copyedit |
| Ten-reviewer stress panel is recorded and actioned. | ready | remaining panel risks are claim-boundary risks, not missing paper sections |
| Experimental side projects are scoped away from COLM_v3 claims. | ready | only future-work wording should remain |

## Input Manifest

| key | path | sha256 |
| --- | --- | --- |
| benchmark_selection_gate | results/source_private_benchmark_selection_gate_20260502/benchmark_selection_gate.json | b27a543034f603517f54d362e556e7d86234a230832575b54f59ce3fb98378bb |
| colm_v2_review_packet | results/latentwire_colm_v2_review_packet_20260504/review_packet.json | 367ce9562a207b6c813b45951ebd25a395ab5e97a043a7c964cf693e371efe65 |
| colm_v3_readiness | paper/latentwire_colm_v3_readiness_20260505.md | e10d3b00a18af57dad663b881c5ddda39b60ce38d88b19190e1199eed6902a70 |
| colm_v3_reviewer_panel | colm_final/audits/colm_v3_10_reviewer_panel_20260505.md | d3e5ded1f81d5d2f705da14f7cc4a6c05b5ea66685cd8c8bfbb4b6de8741d04e |
| colm_v3_tex | colm_final/paper/latentwire_colm2026.tex | cccf7b457962fb6ae17f52a6027f99ab7cbb9f1c83ef4562c768a789c69e4a5e |
| cpu_systems_frontier | results/source_private_cpu_systems_frontier_20260429/cpu_systems_frontier.json | 1bb52d20072193b61289f3a1592949d6b62361ef2ad747be3ba5d6be1c90eb38 |
| experiment_ledger | paper/experiment_ledger_20260421.md | f8ff74b198bd5dc38ced80de422cab1477915f4084d390d2701736858ce6cc69 |
| experimental_status | experimental/status_20260505.md | 726b5fe331cae97ffa046cec22d8a7c6e7ef54d587d62866c70c71824f86c694 |
| hellaswag_seed_stability | results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed/arc_challenge_seed_stability.json | 37f0fca016e6c5b17c6a23d9d5b09e17154921cf51d85bed67a9c8f300d548bd |
| latest_model_matrix | results/source_private_latest_model_matrix_20260428/latest_model_matrix.json | 09694bfd13bb060e146274808994f189b1966e881e3b9c723d91aea513666498 |
| reviewer_feedback | paper/reviewer_feedback.md | 282ee615ffe970797c695908214088af55b0ae80656ea5c4829aa01c2492198e |
| sciq_bridge_contract | results/source_private_sciq_bridge_contract_20260501/sciq_bridge_contract.json | c5de332f794ac4a7f1942dfb26c956ff90da0ce06c9d665132ba45e2fe7b10ef |
| systems_boundary | results/source_private_systems_boundary_figure_table_split_20260504/systems_boundary_figure_data.json | 81a776e985428f6945c22ccc3b7921016afb4f0a2e32d4b4a7af4b04cea9ee49 |
