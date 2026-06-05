# Iteration Log

Strict MPS-first rerun. Mac results are screening/ceiling states only; no Mac row is promotion-eligible.

| priority | probe | paper | status | achieved_n | floor | MDE | verdict | wall_clock_seconds | next command |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | `L_PC1_cross_family_specialist_ceiling` | latentwire | `MAC_FLOOR_POSITIVE_CEILING_ONLY` | 512 | 500 | 0.02734375 | `POSITIVE_CEILING_ONLY` | 0.228 | `re-run strict cached floor: venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_PC1_cross_family_specialist_ceiling` |
| 2 | `L_PC2_tool_augmented_source_ceiling` | latentwire | `MAC_FLOOR_KILLED_BY_EQUAL_BYTE_VISIBLE_TOOL_CONTROL` | 500 | 500 | 0.038000000000000034 | `NEGATIVE_SCREEN_CONTROL_DOMINATED` | 0.583 | `venv_arm64/bin/python scripts/run_mps_first_strict_iteration.py --probe L_PC2_tool_augmented_source_ceiling` |
| 3 | `L_PC5_private_verifier_receiver_candidates` | latentwire | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `extend the v3 verifier scorer only after a nondegenerate candidate pool exists; do not run if the subset gate fails` |
| 4 | `L_C2_control_trained_lcf_lite_proxy` | latentwire | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `author tiny CPU/MPS fuser fixture plus planted-signal test; no GPU and no confirm` |
| 5 | `C_U1_drift_as_signal_router` | channel_set | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `build cached CPU screen over existing Channel-Set drift artifacts with identical-row split manifests` |
| 6 | `C_W1_fixed_library_warmup_selector` | channel_set | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `materialize a dev/gate warmup-policy cache audit before any native replay` |
| 7 | `C_S1_clean_survival_stablecore_denominator` | channel_set | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `author a fresh preregistered denominator audit; do not queue old C-F artifacts` |
| 8 | `C_Y5_channel_set_defense_bundle` | channel_set | `NOT_RERUN` | 0 |  |  | `NOT_RERUN` | 0.000 | `build an audit packet from current dashboards and runbooks; no experiments` |
