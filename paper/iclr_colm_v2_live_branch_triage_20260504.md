# ICLR / COLM_v2 Live Branch Triage

- created UTC: `2026-05-04T23:04:12.079413+00:00`
- COLM_v2 readiness: `scoped_positive_ready_for_writeup_if_claims_are_narrow`
- ICLR readiness: `blocked_by_lack_of_broad_or_learned_positive_receiver`

## Current Story

LatentWire_v2 can currently support a scoped COLM_v2 story: byte-scale, source-private packets plus strict destructive controls. The previously caveated OpenBookQA train-only receiver row is now weakened by same-source-choice wrong-row hardening. The ICLR story is still blocked because cross-family conditional PQ, deterministic public-basis conditioning, scalar integrity thresholds, ARC atom packets, OpenBookQA receiver fusion, and HellaSwag learned/source-conditioned resonance receivers have not produced a broad positive row beyond packet/source-choice/target-cache controls.

## Exact Submission Gap

ICLR needs a positive learned or broader-benchmark receiver that passes strict destructive controls with per-seed stability and source-choice separation. COLM_v2 can be prepared around the conditional-PQ shared-schema method, the fixed-byte HellaSwag packet row, OpenBookQA hardening as a negative diagnostic, and the target-resonance capacity-versus-held-out-failure analysis with explicit limitations.

## Current Technical Contributions

- `source_private_low_rate_packets`: alive_for_colm_v2; needs broader or learned receiver evidence for ICLR.
- `strict_destructive_controls`: strong; needs paper integration and compact tables.
- `systems_byte_accounting`: mac_local_ready; needs native dense-KV/C2C measurements before throughput or energy claims.
- `sparse_resonance_packets`: framing_alive_method_not_yet_positive; needs new mechanism beyond deterministic PQ, PCA/behavior atoms, BatchTopK-style atom banks, chunk encoders, query resamplers, and source-to-prefix decoders.
- `train_only_packet_target_receiver`: openbookqa_weakened_by_source_choice_control; needs a packet that carries row-specific source evidence beyond same-source-choice wrong-row packets.
- `target_self_resonance_capacity_probe`: capacity_alive; needs held-out/source-private receiver that beats slots-only, zero-source, wrong-source, source-choice, and candidate-roll controls.

## Branch Table

| Branch | Status | Score | Baseline | Delta | CI95 Low | Bytes | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Conditional PQ shared-schema packet | `promote_for_colm_v2_only` | `16` | `16` | `0.658` | `0.658` | `7` | Use as the COLM_v2 positive method boundary; not enough for ICLR generality. |
| Conditional PQ public-zscore held-out-family decoder | `ruled_out_as_cross_family_rescue` | `0.453125` | `0.441406` | `0.011719` | `-0.085938` | `7` | Do not widen public-zscore to n500/remap. |
| Conditional PQ public-SVD whitening held-out-family decoder | `ruled_out_as_cross_family_rescue` | `0.375` | `0.613281` | `-0.238281` | `-0.32041` | `7` | Do not continue deterministic public-basis whitening as the rescue path. |
| Conditional PQ corruption-to-noop receiver | `ruled_out_as_cross_family_rescue` | `0.25` | `0.25` | `0.0` | `0.0` | `7` | Keep as diagnostic; do not promote this receiver family. |
| Conditional PQ scalar integrity threshold | `ruled_out_simple_integrity_threshold` | `0.425781` | `0.457031` | `-0.03125` | `-0.097656` | `7` | Do not continue scalar threshold integrity on this public-zscore conditional-PQ receiver. |
| ARC sparse resonance PCA/target-aligned soft-prefix packets | `ruled_out_current_basis_receiver` | `0.25` | `0.625` | `-0.375` | `None` | `None` | Do not widen plain PCA/target-aligned soft-prefix packets. |
| ARC behavior-supervised atom SRP decoder | `ruled_out_current_behavior_atom_basis` | `0.4375` | `0.4375` | `0.0` | `-0.3125` | `7` | Keep the strict atom harness, but do not scale the current behavior/PCA/BatchTopK-style atom basis without a new source-causality mechanism. |
| OpenBookQA train-only packet+target receiver | `weakened_openbookqa_source_choice_control` | `0.424` | `0.422` | `0.002` | `0.00795` | `3` | Do not promote as a positive second-benchmark row; the same-source-choice wrong-row control nearly matches the receiver, so it is a source-choice artifact diagnostic. |
| HellaSwag fixed hybrid candidate packet | `promote_for_colm_v2_systems_baseline` | `0.532464` | `0.526688` | `0.005776` | `0.002888` | `4` | Useful COLM_v2 systems/privacy row, but not a learned latent receiver. |
| HellaSwag protected top-2/rival packet | `ruled_out_shallow_pair_decoder` | `0.46224` | `0.467448` | `-0.005208` | `-0.013021` | `5` | Do not rerun generic protected top-2/rival switchers. |
| HellaSwag official-train receiver-calibrated selector | `weakened_near_miss` | `0.466146` | `0.467448` | `-0.001302` | `-0.007812` | `5` | Do not continue shallow linear score-feature selectors without new evidence. |
| HellaSwag harm-controlled complementarity buckets | `ruled_out_no_safe_bucket` | `0.467448` | `0.467448` | `0.0` | `0.0` | `6` | Low-harm bucket receiver is saturated. |
| HellaSwag top1/top2 ambiguity buckets | `ruled_out_no_safe_bucket` | `0.467448` | `0.467448` | `0.0` | `0.0` | `4` | Do not continue rank/score-bin top2 buckets without a new packet field. |
| HellaSwag denoising syndrome packet | `ruled_out_shallow_syndrome_decoder` | `0.463542` | `0.467448` | `-0.003906` | `-0.010417` | `4` | Do not promote ridge-denoising syndrome decoder. |
| HellaSwag sparse/common-basis top2 ambiguity code | `weakened_common_basis_atom` | `0.500977` | `0.501953` | `-0.000977` | `-0.00293` | `4` | Do not claim sparse atom causality from this branch. |
| Target self-resonance oracle soft-prefix capacity | `capacity_alive_not_source_private_method` | `0.9375` | `0.625` | `0.3125` | `None` | `None` | Use only as capacity/headroom evidence; it optimizes on eval rows. |
| Target self-resonance held-out learned prefix encoders | `ruled_out_current_target_native_encoder_family` | `0.6875` | `0.6875` | `0.0` | `None` | `None` | Do not run more chunk/distill/query-resampler variants without a new information path. |
| Source-conditioned target-native resonance receivers | `ruled_out_current_source_conditioned_receiver_family` | `0.375` | `0.375` | `0.0` | `None` | `None` | Diagnose complementarity/gating before implementing another source-to-prefix decoder. |
| HellaSwag complementarity-frontier selector diagnostic | `headroom_alive_selector_blocked` | `0.467448` | `0.467448` | `0.0` | `0.0` | `4` | Do not train another HellaSwag selector on the same packet fields; require a new information path. |
| HellaSwag multi-signal source packet frontier | `ruled_out_cached_policy_packet` | `0.455729` | `0.467448` | `-0.011719` | `-0.02347` | `5` | Do not continue cached Qwen policy-prediction packets on this HellaSwag slice. |

## Evidence Notes

- `Conditional PQ shared-schema packet`: 16/16 disjoint n500 rows pass; 4/4 budget-2 rows pass; cross-family grid is 0/28.
- `Conditional PQ public-zscore held-out-family decoder`: 0/2 pass; best source-minus-control 0.011719 against label_shuffled_encoder.
- `Conditional PQ public-SVD whitening held-out-family decoder`: 0/2 pass; best source-minus-control -0.238281 against permuted_codes. Worst row -0.500000.
- `Conditional PQ corruption-to-noop receiver`: 0/6 pass; best source-minus-control 0.000000 against label_shuffled_encoder.
- `Conditional PQ scalar integrity threshold`: Selected negative_min_l2 threshold; source 0.425781 vs best control label_shuffled_encoder at 0.457031; source/max-corrupt accept 0.773438/1.000000.
- `ARC sparse resonance PCA/target-aligned soft-prefix packets`: 0/4 pass; best matched-minus-control -0.375000 with best control source_free_prefix.
- `ARC behavior-supervised atom SRP decoder`: 0/5 pass; best matched 0.437500 vs required control top_atom_knockout at 0.437500; fired 8 rows, helps/harms 3/0.
- `OpenBookQA train-only packet+target receiver`: Held-out test n=500; receiver candidate pass False; default receiver 0.424000 vs best baseline/control 0.422000; aggregate seed-row CI low 0.000400; strict per-seed CI 2/5.
- `HellaSwag fixed hybrid candidate packet`: Full cached validation pass over 10042 rows; positive on 10/10 slices.
- `HellaSwag protected top-2/rival packet`: Oracle 0.678385, selected decoder harms 6 vs helps 2.
- `HellaSwag official-train receiver-calibrated selector`: Oracle 0.766927; eval-label diagnostic delta is only 0.002604.
- `HellaSwag harm-controlled complementarity buckets`: Selected scheme no_op with 0 eligible buckets and 0 overrides.
- `HellaSwag top1/top2 ambiguity buckets`: Source top1/top2 oracle 0.675781, but selected bucket overrides 0 rows.
- `HellaSwag denoising syndrome packet`: Target/hybrid oracle 0.604167; denoising helps 2 and harms 5.
- `HellaSwag sparse/common-basis top2 ambiguity code`: Best destructive control target_derived_source_pair_ambiguity_control at 0.501953; atom code helps 0 and harms 1.
- `Target self-resonance oracle soft-prefix capacity`: 3/3 tiny oracle rows pass; best optimized agreement 0.937500 beats shuffled_optimized_prefix at 0.625000.
- `Target self-resonance held-out learned prefix encoders`: 1/5 pass; best agreement delta 0.000000 against slots_only_encoder; worst agreement delta -0.125000.
- `Source-conditioned target-native resonance receivers`: 0/5 pass; best accuracy delta 0.000000 against zero_source_hidden; source top1/top2 oracle reaches 1.000000.
- `HellaSwag complementarity-frontier selector diagnostic`: Fixed+source top1/top2 oracle 0.694010; source top1/top2 covers 174 fixed-hybrid errors, but selected frontier makes 0 overrides.
- `HellaSwag multi-signal source packet frontier`: Selector accuracy 0.455729 vs fixed 0.467448; overrides 30 rows; best destructive control field_shuffle_multisignal_control at 0.430990.

## Promoted

- Conditional PQ shared-schema packet as COLM_v2 positive method.
- HellaSwag fixed hybrid candidate packet as a systems/privacy packet row.

## Weakened Or Ruled Out

- Conditional PQ public-zscore and corruption-to-noop as held-out-family rescues.
- Conditional PQ public-SVD whitening as a held-out-family rescue.
- ARC sparse resonance PCA/target-aligned soft-prefix basis as implemented.
- ARC behavior-supervised, packet-innovation, event-triggered, and corruption-no-op atom decoders as implemented.
- OpenBookQA train-only packet+target receiver as a positive row after same-source-choice wrong-row hardening.
- HellaSwag protected-rival, top2 bucket, linear receiver, harm bucket, and denoising syndrome switchers.
- Sparse/common-basis top2 atom causality in the current HellaSwag implementation.
- Target self-resonance chunk/distill/query-resampler encoders as reusable target-native receivers.
- Source-conditioned source-hidden/codebook/refinement target-native receivers as currently implemented.
- HellaSwag complementarity-frontier selector with current top1/top2 packet fields.
- HellaSwag cached hidden/score/vote policy-prediction packets as a repair frontier.
- Conditional PQ scalar integrity thresholds on the public-zscore held-out-family receiver.
- ARC row-level Qwen2.5 token hidden pool plus ridge repair readout: matched
  `0.3125`, source-index control `0.4375`, and matched tied zero-source,
  wrong-row, source-row-shuffle, same-source-choice wrong-row, atom-shuffle,
  and coefficient-shuffle controls.
- ARC candidate-local Qwen2.5 hidden repair readout: public-innovation matched
  `0.3750`, raw-hidden matched `0.3125`, source-index/rank/score controls
  `0.4375`, and same-byte visible text `0.3750`.

## Next Exact Gate

- name: `svamp32_c2c_teacher_sparse_packet_distillation_preflight`
- primary path: Return to the frozen SVAMP32 C2C teacher surface where
  target-alone is `8/32`, C2C teacher is `16/32`, and C2C-only
  target-complementary wins are `10`. Treat dense C2C as a teacher, distill its
  useful behavioral delta into sparse source-private packets, and require
  recovery of C2C-only wins beyond source-destroying controls.
- fallback path: If the C2C-teacher proxy also fails, defer broad C2C
  superiority claims until NVIDIA-backed native KV/C2C comparison or choose a
  benchmark with clearer dense-transfer headroom.
- pass bar: A learned or rule-based packet receiver must improve over source-index/rank/score, same-byte text, wrong-source, same-source-choice wrong-row, candidate-roll, and target-derived controls with a positive paired CI95 low on a frozen slice.
- required controls: `target_only`, `answer_masked_source`, `constrained_wrong_row_source`, `same_source_choice_wrong_row`, `candidate_roll_or_deranged_public_basis`, `permuted_codes`, `random_same_byte`, `opaque_slot_or_deranged_basis`, `source_index_rank_score_comparators_when_meaningful`, `same_byte_visible_text`

## Claim Boundaries

- Do not claim broad cross-family latent communication yet.
- Do not claim sparse resonance packets as a positive method from the current PCA or sparse-atom gates.
- Do not claim C2C/KV throughput, HBM, PCIe, NVLink, or energy wins without native measurements.
- Frame C2C/KVComm/TurboQuant/KIVI as high-bandwidth or KV-compression baselines with different exposure regimes.
