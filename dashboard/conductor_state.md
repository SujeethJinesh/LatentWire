# Conductor State

- stage: `stage1_row_safe_screen_complete`
- branch: `codex-campaign`
- cache_parse_files_seen: `8725`
- cache_parse_usable_files: `47`
- stage1_rows_consumed: `183653`
- stage1_confirm_rows_consumed: `0`
- mac_status_counts: `{'AMBIGUOUS': 100, 'CPU_SCREENED': 57, 'KILLED': 12, 'PARKED_NEEDS_GPU': 2}`

## Readiness

Row-level parsing and filtering ran on parseable LatentWire prediction caches and Channel-Set per-trace recovery caches. Hard cache families were quarantined instead of read whole. No Mac result is marked `PASSED`; positives, if any, are only `PROVISIONAL_PROMOTE_TO_GPU`.

## Held-Out Answer

The parse coverage confirms many prior artifacts are named `test`, `validation`, `holdout`, or have no clean dev/gate/confirm naming. Treat cached screens as kill/screen evidence only. Final confirmation requires either quarantined confirm rows with a runner-specific row parser or fresh data generation.

## Top Gate Rows

| method_id | paper | status | source_path | matched_accuracy | best_baseline_accuracy | delta_vs_best_baseline | ci95_low_vs_best_baseline | regime | median_recovery | cvar25_recovery | worst_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_bge_validation/predictions.jsonl | 0.3582089552238806 | 0.31343283582089554 | 0.04477611940298507 | -0.11940298507462686 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_hashed_validation/predictions.jsonl | 0.3137254901960784 | 0.3333333333333333 | -0.0196078431372549 | -0.19607843137254902 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/predictions.jsonl | 0.36220472440944884 | 0.24803149606299213 | 0.1141732283464567 | 0.03543307086614173 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/predictions.jsonl | 0.463768115942029 | 0.3333333333333333 | 0.13043478260869565 | -0.043478260869565216 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl | 0.31223628691983124 | 0.2869198312236287 | 0.02531645569620253 | -0.06751054852320675 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl | 0.47761194029850745 | 0.26865671641791045 | 0.208955223880597 | 0.05970149253731343 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/predictions.jsonl | 0.4852941176470588 | 0.2647058823529412 | 0.22058823529411764 | 0.13725490196078433 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation512_12b/predictions.jsonl | 0.48672566371681414 | 0.24778761061946902 | 0.23893805309734514 | 0.12389380530973451 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/predictions.jsonl | 0.40350877192982454 | 0.2894736842105263 | 0.11403508771929824 | 0.0 |  |  |  |  |
| latentwire_cached_fixed_packet_baseline | latentwire | CPU_SCREENED | results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl | 0.31313131313131315 | 0.30303030303030304 | 0.010101010101010102 | -0.1111111111111111 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget2.jsonl | 0.21 | 0.61 | -0.4 | -0.54 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget4.jsonl | 0.2636363636363636 | 0.5727272727272728 | -0.3090909090909091 | -0.4 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget6.jsonl | 0.21621621621621623 | 0.6396396396396397 | -0.42342342342342343 | -0.5225225225225225 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget2.jsonl | 0.125 | 0.4375 | -0.3125 | -0.45535714285714285 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget4.jsonl | 0.14678899082568808 | 0.5504587155963303 | -0.4036697247706422 | -0.4954128440366973 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget6.jsonl | 0.1941747572815534 | 0.5825242718446602 | -0.3883495145631068 | -0.47572815533980584 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget2.jsonl | 0.19387755102040816 | 0.45918367346938777 | -0.2653061224489796 | -0.3877551020408163 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget4.jsonl | 0.31 | 0.45 | -0.14 | -0.28 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | CPU_SCREENED | results/source_private_wyner_ziv_packet_gate_20260429/remap_101/predictions_budget6.jsonl | 0.42168674698795183 | 0.4457831325301205 | -0.024096385542168676 | -0.18072289156626506 |  |  |  |  |
| L_SCORECOMP_wz_bins_deployable | latentwire | KILLED | results/source_private_wyner_ziv_packet_gate_20260429/remap_103/predictions_budget2.jsonl | 0.14678899082568808 | 0.4036697247706422 | -0.25688073394495414 | -0.3669724770642202 |  |  |  |  |
