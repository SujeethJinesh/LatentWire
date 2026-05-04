# ICLR / COLM_v2 Live Branch Triage

- created UTC: `2026-05-04T22:02:15.345502+00:00`
- COLM_v2 readiness: `scoped_positive_ready_for_writeup_if_claims_are_narrow`
- ICLR readiness: `blocked_by_lack_of_broad_or_learned_positive_receiver`

## Current Story

LatentWire_v2 can currently support a scoped COLM_v2 story: byte-scale, source-private packets plus strict destructive controls. The ICLR story is still blocked because cross-family conditional PQ, deterministic public-basis conditioning, and HellaSwag learned/source-conditioned resonance receivers have not produced a positive row beyond packet/source-choice/target-cache controls.

## Exact Submission Gap

ICLR needs a positive learned or broader-benchmark receiver that passes strict destructive controls. COLM_v2 can be prepared around the conditional-PQ shared-schema method, the fixed-byte HellaSwag packet row, and the target-resonance capacity-versus-held-out-failure analysis with explicit limitations.

## Current Technical Contributions

- `source_private_low_rate_packets`: alive_for_colm_v2; needs broader or learned receiver evidence for ICLR.
- `strict_destructive_controls`: strong; needs paper integration and compact tables.
- `systems_byte_accounting`: mac_local_ready; needs native dense-KV/C2C measurements before throughput or energy claims.
- `sparse_resonance_packets`: framing_alive_method_not_yet_positive; needs new mechanism beyond deterministic PQ, PCA atoms, chunk encoders, query resamplers, and source-to-prefix decoders.
- `target_self_resonance_capacity_probe`: capacity_alive; needs held-out/source-private receiver that beats slots-only, zero-source, wrong-source, source-choice, and candidate-roll controls.

## Branch Table

| Branch | Status | Score | Baseline | Delta | CI95 Low | Bytes | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Conditional PQ shared-schema packet | `promote_for_colm_v2_only` | `16` | `16` | `0.658` | `0.658` | `7` | Use as the COLM_v2 positive method boundary; not enough for ICLR generality. |
| Conditional PQ public-zscore held-out-family decoder | `ruled_out_as_cross_family_rescue` | `0.453125` | `0.441406` | `0.011719` | `-0.085938` | `7` | Do not widen public-zscore to n500/remap. |
| Conditional PQ public-SVD whitening held-out-family decoder | `ruled_out_as_cross_family_rescue` | `0.375` | `0.613281` | `-0.238281` | `-0.32041` | `7` | Do not continue deterministic public-basis whitening as the rescue path. |
| Conditional PQ corruption-to-noop receiver | `ruled_out_as_cross_family_rescue` | `0.25` | `0.25` | `0.0` | `0.0` | `7` | Keep as diagnostic; do not promote this receiver family. |
| ARC sparse resonance PCA/target-aligned soft-prefix packets | `ruled_out_current_basis_receiver` | `0.25` | `0.625` | `-0.375` | `None` | `None` | Do not widen plain PCA/target-aligned soft-prefix packets. |
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

## Evidence Notes

- `Conditional PQ shared-schema packet`: 16/16 disjoint n500 rows pass; 4/4 budget-2 rows pass; cross-family grid is 0/28.
- `Conditional PQ public-zscore held-out-family decoder`: 0/2 pass; best source-minus-control 0.011719 against label_shuffled_encoder.
- `Conditional PQ public-SVD whitening held-out-family decoder`: 0/2 pass; best source-minus-control -0.238281 against permuted_codes. Worst row -0.500000.
- `Conditional PQ corruption-to-noop receiver`: 0/6 pass; best source-minus-control 0.000000 against label_shuffled_encoder.
- `ARC sparse resonance PCA/target-aligned soft-prefix packets`: 0/4 pass; best matched-minus-control -0.375000 with best control source_free_prefix.
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

## Promoted

- Conditional PQ shared-schema packet as COLM_v2 positive method.
- HellaSwag fixed hybrid candidate packet as a systems/privacy packet row.

## Weakened Or Ruled Out

- Conditional PQ public-zscore and corruption-to-noop as held-out-family rescues.
- Conditional PQ public-SVD whitening as a held-out-family rescue.
- ARC sparse resonance PCA/target-aligned soft-prefix basis as implemented.
- HellaSwag protected-rival, top2 bucket, linear receiver, harm bucket, and denoising syndrome switchers.
- Sparse/common-basis top2 atom causality in the current HellaSwag implementation.
- Target self-resonance chunk/distill/query-resampler encoders as reusable target-native receivers.
- Source-conditioned source-hidden/codebook/refinement target-native receivers as currently implemented.

## Next Exact Gate

- name: `colm_v2_table_refresh_then_complementarity_frontier_gate`
- primary path: Backport this live triage into COLM_v2 tables/figures, then run a small complementarity-frontier diagnostic that isolates rows where the target is wrong and source top1/top2 could help, measuring whether any source-private packet field beats source-choice and wrong-row controls before another decoder is trained.
- fallback path: If the frontier diagnostic shows no separable source-causal signal, stop HellaSwag receiver work and pivot to an alternate benchmark/method where the source has measurable complementarity beyond rank/score shortcuts.
- pass bar: A learned or rule-based packet receiver must improve over source-index/rank/score, same-byte text, wrong-source, candidate-roll, and target-derived controls with a positive paired CI95 low on a frozen slice.
- required controls: `target_only`, `answer_masked_source`, `constrained_wrong_row_source`, `same_source_choice_wrong_row`, `candidate_roll_or_deranged_public_basis`, `permuted_codes`, `random_same_byte`, `opaque_slot_or_deranged_basis`, `source_index_rank_score_comparators_when_meaningful`, `same_byte_visible_text`

## Claim Boundaries

- Do not claim broad cross-family latent communication yet.
- Do not claim sparse resonance packets as a positive method from the current PCA or sparse-atom gates.
- Do not claim C2C/KV throughput, HBM, PCIe, NVLink, or energy wins without native measurements.
- Frame C2C/KVComm/TurboQuant/KIVI as high-bandwidth or KV-compression baselines with different exposure regimes.
