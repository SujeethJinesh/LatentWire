# Iteration Log

MPS-first queue execution log. Statuses are screening/parked states only unless a summary explicitly records a powered verdict.

| priority | probe | paper | status | achieved_n | MDE | verdict | wall_clock_seconds | next command |
| ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | latentwire | `PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER` | 768 |  | `PARKED` | 0.000 | `venv_arm64/bin/python scripts/run_mps_first_iteration.py --probe L_PC1_cross_family_specialist_ceiling after authoring a reviewed CPU/MPS scorer that uses Qwen2.5-Math-1.5B or DeepSeek-R1-Distill-Qwen-1.5B as source and Qwen3-0.6B as receiver on frozen dev/gate rows, writes source/receiver label scores, computes conditional MI, and enforces --no-confirm.` |
| 2 | `L_PC2_tool_augmented_source_ceiling` | latentwire | `PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER` | 0 |  | `PARKED` | 0.002 | `venv_arm64/bin/python scripts/build_tool_private_source_ceiling.py --dataset data/gsm8k_100.jsonl --source-tool calculator --source-model Qwen/Qwen2.5-Math-1.5B-Instruct --receiver-model Qwen/Qwen3-0.6B --split dev,gate --min-gate-n 400 --mde-target 0.05 --no-confirm` |
| 3 | `L_PC5_private_verifier_receiver_candidates` | latentwire | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `extend the v3 verifier scorer only after a nondegenerate candidate pool exists; do not run if the subset gate fails` |
| 4 | `L_C2_control_trained_lcf_lite_proxy` | latentwire | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `author tiny CPU/MPS fuser fixture plus planted-signal test; no GPU and no confirm` |
| 5 | `C_U1_drift_as_signal_router` | channel_set | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `build cached CPU screen over existing Channel-Set drift artifacts with identical-row split manifests` |
| 6 | `C_W1_fixed_library_warmup_selector` | channel_set | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `materialize a dev/gate warmup-policy cache audit before any native replay` |
| 7 | `C_S1_clean_survival_stablecore_denominator` | channel_set | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `author a fresh preregistered denominator audit; do not queue old C-F artifacts` |
| 8 | `C_Y5_channel_set_defense_bundle` | channel_set | `NOT_RUN` | 0 |  | `NOT_RUN` | 0.000 | `build an audit packet from current dashboards and runbooks; no experiments` |
