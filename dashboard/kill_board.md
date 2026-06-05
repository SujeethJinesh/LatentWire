# Kill Board

| method_id | paper | source_path | delta_vs_best_baseline | ci95_high_vs_best_baseline | note |
| --- | --- | --- | --- | --- | --- |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget2.jsonl | -0.4 | -0.27 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget4.jsonl | -0.3090909090909091 | -0.21818181818181817 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget6.jsonl | -0.42342342342342343 | -0.32432432432432434 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget2.jsonl | -0.3125 | -0.1875 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget4.jsonl | -0.4036697247706422 | -0.3211009174311927 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget6.jsonl | -0.3883495145631068 | -0.30097087378640774 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget2.jsonl | -0.2653061224489796 | -0.14285714285714285 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget4.jsonl | -0.14 | -0.01 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_103/predictions_budget2.jsonl | -0.25688073394495414 | -0.13761467889908258 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_103/predictions_budget4.jsonl | -0.23157894736842105 | -0.11578947368421053 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_107/predictions_budget4.jsonl | -0.2708333333333333 | -0.15625 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_wz_bins_deployable | latentwire | results/source_private_wyner_ziv_packet_gate_20260429/remap_107/predictions_budget6.jsonl | -0.1619047619047619 | -0.0380952380952381 | dev/gate row-filtered cache screen; no Mac PASSED status |
| L_SCORECOMP_fresh_wz_high_entropy | latentwire | results/mac_continue/fresh_mmlu_pro/fresh_mmlu_pro_rows.jsonl | -0.043478260869565216 | 0.2608695652173913 | fresh high-entropy Mac bounded negative |
| L_ORACLE_POWERED_LADDER | latentwire | results/mac_continue/latentwire_oracle_ladder/oracle_ladder_rows.jsonl | -0.013927576601671309 | 0.022284122562674095 | powered dev/gate ladder classifies current score-packet path as not-deployable |
| L_Q1_receiver_query_packet | latentwire | results/mac_continue/latentwire_query_packet/query_packet_rows.jsonl | -0.022284122562674095 | 0.011142061281337047 | final query-conditioned two-way score-packet shot killed; controls do not collapse |
| C2C_candidate_pool_delta_packet | latentwire | results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.json | -0.218750 |  | matched 3/32 is dominated by coeff-sign-flip control 10/32 |
| C2C_teacher_delta_packet | latentwire | results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.json | 0.000000 |  | matched 14/32 ties target-only and zero-delta; source-necessary clean count is 0 |
| C2C_generated_answer_packet | latentwire | results/svamp32_c2c_generated_answer_packet_audit_20260505/generated_answer_packet_audit.json | 0.000000 |  | answer value/index equals same-byte visible-answer control; not source-private |
| KVCOMM_existing_cache_smoke_controls | latentwire | results/dense_baseline_mcqa_smoke_20260505/kvcomm_damage_diagnostic_n16.json | 0.000000 | 0.000000 | matched predictions have 1.0 agreement with zero-source controls across inspected smokes |
| C_F_m10_random_control | channel_set | experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z/checker_result.json | -0.761487 |  | random-bin control beats M10; C-F cannot promote before denominator cleanup |

## Consolidated Audit Kills And Parks

| method_id | verdict | evidence | action |
| --- | --- | --- | --- |
| `L_PC2_tool_augmented_source_ceiling` | `KILLED_BY_EQUAL_BYTE_VISIBLE_TOOL_CONTROL` | private tool result is valuable, but the visible equal-byte tool-result control ties it | do not frame as no-text latent communication |
| `L_PC5_private_verifier_receiver_candidates` | `ORACLE_ONLY` | strict verifier/source scores are computed from answer correctness | park until a gold-blind verifier cache clears the nondegenerate prompt floor |
| `L_C2_control_trained_lcf_lite_proxy` | `ORACLE_ONLY` | strict fuser prediction is the equation-derived `tool_answer` | park until gold-free source/receiver cache features exist |
| `C_A1_cvar_evt_clip_grid` | `BLOCKED_CONFIRM_CONTAMINATED` | cached Granite tight-clip gate source is confirmation-contaminated | replace with fresh non-confirm same-row three-model packet before GPU handoff |
## MPS-First Parked/Negative State

| priority | probe | status | achieved_n | interpretation |
| ---: | --- | --- | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | `PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER` | 768 | No eligible existing artifact measures I(source_signal;Y|receiver_state) for a complementary source-private signal. |
| 2 | `L_PC2_tool_augmented_source_ceiling` | `PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER` | 0 | No dev/gate cache found where the source privately ran calculator/code-exec and receiver did not see the tool result. |
| 3 | `L_PC5_private_verifier_receiver_candidates` | `PARKED_POOL_TOO_WEAK` | 36 | candidate pool is nondegenerate for only 36 prompts, below the 80-prompt rerank gate; no gain verdict emitted |
| 4 | `L_C2_control_trained_lcf_lite_proxy` | `PARKED_NEEDS_CONTROL_TRAINED_FUSER_PRELAUNCH` | 0 | C2C fusers are local, but no reviewed byte-limited control-trained LCF-lite runner with wrong-row/zero-source objective exists. |
| 5 | `C_U1_drift_as_signal_router` | `PARKED_NEEDS_DRIFT_FEATURE_CACHE` | 171 | Stage-1 rows contain recovery/static_gap but not drift trajectory features paired with difficulty/uplift labels across >=2 models. |
| 6 | `C_W1_fixed_library_warmup_selector` | `PARKED_NEEDS_WARMUP_POLICY_CACHE` | 0 | Existing dashboard states no parseable warmup-policy cache exists. |
| 7 | `C_S1_clean_survival_stablecore_denominator` | `PARKED_NEEDS_IDENTICAL_ROW_DENOMINATOR` | 0 | Available C-F/survival-like evidence is contaminated or lacks a fresh identical-row random/static denominator. |
| 8 | `C_Y5_channel_set_defense_bundle` | `PARKED_NEEDS_NATIVE_PAIRING` | 9 | Defense bundle inputs exist, but claim-bearing C_A1 native paired ParoQuant-vs-tight-clip rows are still missing. |
## MPS-First Strict Rerun

| priority | probe | status | achieved_n | floor | interpretation |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | `MAC_FLOOR_POSITIVE_CEILING_ONLY` | 512 | 500 | see summary |
| 2 | `L_PC2_tool_augmented_source_ceiling` | `MAC_FLOOR_KILLED_BY_EQUAL_BYTE_VISIBLE_TOOL_CONTROL` | 500 | 500 | The private calculator ceiling is large versus a question-only heuristic but exactly tied by an equal-byte visible tool-result control, so it is not a source-private packet win. |
| 3 | `L_PC5_private_verifier_receiver_candidates` | `MAC_FLOOR_POSITIVE_ORACLE_VERIFIER_CEILING` | 500 | 500 | Easy arithmetic oracle-verifier ceiling meets the candidate-pool floor; it is a sanity ceiling, not a deployable L-A2 method. |
| 4 | `L_C2_control_trained_lcf_lite_proxy` | `MAC_FLOOR_POSITIVE_ORACLE_FUSER_CEILING` | 500 | 500 | Oracle source-feature fuser clears the sanity floor; deployable status remains blocked because this uses gold SVAMP equations as source features. |
| 5 | `C_U1_drift_as_signal_router` | `MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED` | 500 | 500 | Floor-sized KL trajectory rows exist, but they lack paired difficulty/policy-uplift labels required by the C_U1 mandatory gate. |
| 6 | `C_W1_fixed_library_warmup_selector` | `MAC_FLOOR_COMPUTED_SCHEMA_BLOCKED` | 500 | 500 | Warmup KL rows can be materialized, but no cached per-policy outcome matrix exists for ParoQuant/C_A1/survival/reject on the same rows. |
| 7 | `C_S1_clean_survival_stablecore_denominator` | `PARKED_LOGGED_LOCAL_FLOOR_BLOCKED` | 198 | 500 | Only non-confirm per-trace rows below the 500-row floor are available locally; generating the missing identical-row denominator requires native replay/backfill, not a Mac-only cached computation. |
| 8 | `C_Y5_channel_set_defense_bundle` | `PARKED_LOGGED_NEEDS_NATIVE_PAIRING` | 6 |  | Defense inputs exist, but the three-model same-row native pairing packet is still missing. |
