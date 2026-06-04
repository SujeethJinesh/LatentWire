# Overnight CPU-Only Screening Report

- run_id: `20260603_cpu_only_screening`
- created_utc: `2026-06-04T05:36:48+00:00`
- locality: Mac CPU/MPS cached artifacts only; no SSH, CUDA, 30B, long decode, or confirmation run.
- paper readiness: not ICLR-ready; all verdicts are screening or backfill decisions, not Mac PASSED claims.

## Verdicts

| exp | verdict | decision | key evidence |
| --- | --- | --- | --- |
| EXP1 CacheWire cache-level oracle | `RECEIVER_LIMITED_CURRENT_TRACE_ORACLE` | oracle bound alive, deployable trace killed | dense teacher `16/32` vs target `8/32`; 1-byte oracle syndrome `14-15/32`; deployable trace matched `13/32` vs target/zero `14/32` and label-shuffle `15/32` |
| EXP2 C2C tractability | `TRACTABLE_REPLAY_AVAILABLE_MMLU_REDUX_NOT_REPRODUCED` | tractable anchor, not reproduced MMLU-Redux | local repo/checkpoint present `True/True`; SVAMP C2C replay `0.5000` vs target `0.2500`; ARC/OBQA constrained MCQA `0.6875`/`0.4375` |
| EXP3 destructive controls | `CONTROL_HARNESS_READY_EXISTING_KVCOMM_COLLAPSES` | kill existing KVComm and C2C packet smokes as method evidence | KVComm matched predictions have zero-source agreement `1.0`; teacher-delta ties zero/target; candidate-delta matched `3/32` vs control `10/32`; answer packet equals answer-text leak |
| EXP4 L-A2 ceiling | `NOT_RUN_NO_GENERATED_SOLUTION_SCORE_CACHE` | parked until cache exists | only `3` generated-text smoke rows; no source/target/verifier scores |
| EXP5 Channel-Set offline | `C_A1_BACKFILL_FIRST_C_F_CONTROL_CONTAMINATED` | C_A1 backfill first, C_F cleanup | C_A1 gate rows `6` with `5` positive medians; cached DeepSeek/Falcon checker medians `0.5180`/`0.3898`; C-F M10 random-control delta `-0.761487` |

## Prioritized GPU / Backfill Queue

1. `channel_set_c_a1_tail_cvar_grid`: user-run native W4A16/ParoQuant replay on exact cached C_A1 gate IDs with DeepSeek/Falcon sentinels; promotion disabled.
2. `channel_set_c_a1_paroquant_parity_card`: parity card for the same rows before any claim.
3. `latentwire_l_a2_generated_solution_rerank_cache`: generate dev/gate candidate pools plus source/target/verifier scores; no confirm access.
4. `channel_set_c_f_hazard_control_shard`: identical-row denominator for static/EMA/random matched controls before any foreground job.
5. `channel_set_ce13_warmup_policy_cache_materialization`: only after a tiny dev/gate warmup-policy cache exists.

No foreground GPU job is authorized by this report.

## Raw Artifacts

- `results/overnight/20260603_cpu_only_screening/exp1/summary.json`
- `results/overnight/20260603_cpu_only_screening/exp1/raw_rows.jsonl`
- `results/overnight/20260603_cpu_only_screening/exp2/summary.json`
- `results/overnight/20260603_cpu_only_screening/exp2/raw_rows.jsonl`
- `results/overnight/20260603_cpu_only_screening/exp3/summary.json`
- `results/overnight/20260603_cpu_only_screening/exp3/raw_rows.jsonl`
- `results/overnight/20260603_cpu_only_screening/exp4/summary.json`
- `results/overnight/20260603_cpu_only_screening/exp4/raw_rows.jsonl`
- `results/overnight/20260603_cpu_only_screening/exp5/summary.json`
- `results/overnight/20260603_cpu_only_screening/exp5/raw_rows.jsonl`
- `results/overnight/20260603_cpu_only_screening/overnight_summary.json`

## Next Exact Gate

Run the C_A1 native replay/parity backfill locally on the GPU node from the queue packet. Do not run a foreground confirmation until native rows beat matched controls and DeepSeek/Falcon sentinels do not regress.
