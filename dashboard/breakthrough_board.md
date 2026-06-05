# Breakthrough Board

No Mac paper-positive rows.

| priority | method_id | paper | status | evidence | next gate |
| --- | --- | --- | --- | --- | --- |
| 1 | C_A1_cvar_evt_clip_grid | channel_set | BLOCKED_CONFIRM_CONTAMINATED | cached Granite tight-clip gate source references confirmation path | fresh non-confirm same-row Granite/DeepSeek/Falcon ParoQuant-vs-tight-clip manifest |
| 2 | L_A2_verifier_rerank | latentwire | PARKED_NEEDS_CACHE | no generated-solution score cache exists | materialize dev/gate candidate pools with source/target/verifier scores |
| 3 | C2C_qwen25_05b_to_qwen3_06b_replay_anchor | latentwire | CPU_SCREENED | SVAMP replay tractable, but mechanism trace oracle failed | official MMLU-Redux matched reproduction packet before any claim |
| 4 | CACHEWIRE_oracle_syndrome_bound | latentwire | CPU_SCREENED_ORACLE_ONLY | 1-byte oracle syndrome reaches 14-15/32 but deployable predictors fail | collect pre-answer teacher/KV deltas and require source-necessary clean rows |

## Consolidated Audit Leads

| method | status | evidence | next gate |
| --- | --- | --- | --- |
| `L_PC1_cross_family_specialist_ceiling` | `PENDING_FRESH_DEPLOYABLE_OR_KILL` | cached strict slice `+0.042969`, CI `[+0.015625, +0.068359]` | fresh two-pair/two-task non-confirm matrix with source-index and equal-byte text controls |
| `L_PC5_private_verifier_receiver_candidates` | `ORACLE_ONLY` | strict `+0.666` rerank gain uses `cand == answer` verifier scoring | gold-blind verifier/source/target score cache with at least `80` nondegenerate prompts |
| `L_C2_control_trained_lcf_lite_proxy` | `ORACLE_ONLY` | strict `+0.832` fuser gain uses SVAMP equation-derived `tool_answer` | gold-free source/receiver feature cache with wrong-row, zero-source, source-index, and text controls |

Updated by `scripts/overnight_cpu_screening.py`; Mac status remains screening-only.
## MPS-First Iteration

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

## Cheap Exhaustion Scan

No new Mac paper-positive rows. The non-confirm cache inventory parks C_U1, C_W1, CE1, C_C1, C_A2, and C_D1 for missing same-row inputs or underpowered single-model defense evidence. C_A1 remains the highest-priority Channel-Set materialization target, but promotion is disabled until the fresh Granite/DeepSeek/Falcon same-row manifest exists.
