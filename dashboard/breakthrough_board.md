# Breakthrough Board

No Mac paper-positive rows.

| priority | method_id | paper | status | evidence | next gate |
| --- | --- | --- | --- | --- | --- |
| 1 | C_A1_cvar_evt_clip_grid | channel_set | CPU_SCREENED | 6 gate rows, 5 positive median, DeepSeek sentinel negative and Falcon weak positive | native W4A16/ParoQuant backfill with DeepSeek/Falcon regression sentinels |
| 2 | L_A2_verifier_rerank | latentwire | PARKED_NEEDS_CACHE | no generated-solution score cache exists | materialize dev/gate candidate pools with source/target/verifier scores |
| 3 | C2C_qwen25_05b_to_qwen3_06b_replay_anchor | latentwire | CPU_SCREENED | SVAMP replay tractable, but mechanism trace oracle failed | official MMLU-Redux matched reproduction packet before any claim |
| 4 | CACHEWIRE_oracle_syndrome_bound | latentwire | CPU_SCREENED_ORACLE_ONLY | 1-byte oracle syndrome reaches 14-15/32 but deployable predictors fail | collect pre-answer teacher/KV deltas and require source-necessary clean rows |

Updated by `scripts/overnight_cpu_screening.py`; Mac status remains screening-only.
## MPS-First Iteration

| priority | probe | status | achieved_n | interpretation |
| ---: | --- | --- | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | `PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER` | 768 | No eligible existing artifact measures I(source_signal;Y|receiver_state) for a complementary source-private signal. |
| 2 | `L_PC2_tool_augmented_source_ceiling` | `PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER` | 0 | No dev/gate cache found where the source privately ran calculator/code-exec and receiver did not see the tool result. |
| 3 | `L_PC5_private_verifier_receiver_candidates` | `PARKED_POOL_TOO_WEAK` | 36 | candidate pool is nondegenerate for only 36 prompts, below the 80-prompt rerank gate; no gain verdict emitted |
| 4 | `L_C2_control_trained_lcf_lite_proxy` | `PARKED_NEEDS_CONTROL_TRAINED_FUSER_PRELAUNCH` | 0 | C2C fusers are local, but no reviewed byte-limited control-trained LCF-lite runner with wrong-row/zero-source objective exists. |
