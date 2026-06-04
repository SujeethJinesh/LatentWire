# Breakthrough Board

No Mac paper-positive rows.

| priority | method_id | paper | status | evidence | next gate |
| --- | --- | --- | --- | --- | --- |
| 1 | C_A1_cvar_evt_clip_grid | channel_set | CPU_SCREENED | 6 gate rows, 5 positive median, DeepSeek sentinel negative and Falcon weak positive | native W4A16/ParoQuant backfill with DeepSeek/Falcon regression sentinels |
| 2 | L_A2_verifier_rerank | latentwire | PARKED_NEEDS_CACHE | no generated-solution score cache exists | materialize dev/gate candidate pools with source/target/verifier scores |
| 3 | C2C_qwen25_05b_to_qwen3_06b_replay_anchor | latentwire | CPU_SCREENED | SVAMP replay tractable, but mechanism trace oracle failed | official MMLU-Redux matched reproduction packet before any claim |
| 4 | CACHEWIRE_oracle_syndrome_bound | latentwire | CPU_SCREENED_ORACLE_ONLY | 1-byte oracle syndrome reaches 14-15/32 but deployable predictors fail | collect pre-answer teacher/KV deltas and require source-necessary clean rows |

Updated by `scripts/overnight_cpu_screening.py`; Mac status remains screening-only.
