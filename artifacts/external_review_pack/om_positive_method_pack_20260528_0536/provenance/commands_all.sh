#!/usr/bin/env bash
set -euo pipefail

# 00_four_model_drift
experimental/outlier_migrate/phase1/run_om_phase1.py

# 01_static_top1_static_union
experimental/outlier_migrate/phase3/run_phase3_intervention.py --run-id om_phase3_20260509T212000Z --batch-size 4 --dtype bfloat16 --device auto --reuse-prequant-run-dir experimental/outlier_migrate/phase3/results/om_phase3_20260509T180200Z

# 02_m2_position_switching
experimental/outlier_migrate/phase9/run_om_phase9_m2_position_conditional.py --run-id om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --model-id ibm-granite/granite-4.0-h-small --batch-size 1 --trace-count 12 --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_memfix_20260514T125800Z

# 03_m10_position_bins
experimental/outlier_migrate/phase9/run_om_phase9_m10_position_binned_scales.py --run-id om_phase9_m10_granite_small_vac12_20260515T085800Z --trace-count 12 --batch-size 1 --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z

# 04_m11_ema_top1
experimental/outlier_migrate/phase9/run_om_phase9_m11_ema_drift.py --run-id om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1 --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z

# 05_m11b_granite
experimental/outlier_migrate/phase9/run_om_phase9_m11b_budget_scaling.py --run-id om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1

# 06_m11b_nemotron
experimental/outlier_migrate/phase9/run_om_phase9_m11b_nemotron_static1pct_salvage.py --run-id om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z --batch-size 1

# 07_m11b_deepseek
run_om_phase9_m11b_budget_scaling.py --run-id om_v2_m11b_deepseek_20260527T1210Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/deepseek_r1_distill_qwen_1_5b

# 08_m11b_falcon
run_om_phase9_m11b_budget_scaling.py --run-id om_v2_m11b_falcon_20260527T1438Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/falcon_h1_0_5b

# 09_paroquant_granite
experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py --run-id om_paroquant_granite_small_20260520T1555Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --batch-size 1

# 10_v1_paroquant_nemotron
run_om_paroquant_baseline.py --run-id om_v1_paroquant_nemotron_20260528T0318Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z --batch-size 1 --dtype bfloat16

# 11_e3_paroquant_plus_m11b
experimental/outlier_migrate/phase9/run_om_stage1_e3_paroquant_m11b_composition.py --run-id om_stage1_e3_granite_20260526T034442Z --device auto --dtype bfloat16 --cap-hours 15

# 12_m18_activation_k
experimental/outlier_migrate/phase9/run_om_phase9_m18_joint_kv_activation.py --run-id om_phase9_m18_granite_small_vac12_20260516T193500Z --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1

# 13_m26_stable_core
experimental/outlier_migrate/phase9/run_om_phase9_m26_stable_core.py --run-id om_phase9_m26_granite_small_vac12_20260518T203000Z --stable-core-source experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitudes.jsonl.gz --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z --batch-size 1

# 14_decdec_proxy
experimental/outlier_migrate/phase9/run_om_phase9_decdec_baseline.py --run-id om_phase9_decdec_granite_small_vac12_20260517T141500Z --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1

# 15_kl_fft_diagnostics
--run-id om_phase9_kl_granite_small_dense_20260519T085400Z --m11-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --decdec-run-dir experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z --m26-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z --position-mode dense_grid --batch-size 1

# 16_e2_cross_prompt
experimental/outlier_migrate/phase9/run_om_stage1_e2_cross_prompt_replication.py --run-id om_stage1_e2_mathonly_20260526T1202Z --models deepseek_r1_distill,falcon_h1,granite_small --math-only --math-count 30 --cap-hours 15 --device auto --dtype bfloat16

# 17_mpred
experimental/outlier_migrate/phase9/run_om_phase9_mpred.py --model-key granite --run-id om_phase9_mpred_granite_reduced_20260527T201251Z --batch-size 2 --device auto --dtype bfloat16 --new-regimes mpred_top10_alpha_0_95 mpred_top5_alpha_0_95 mpred_random_alpha_top5 random_walk_top5
