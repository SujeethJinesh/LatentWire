# Next 6 Commands

1. `venv_arm64/bin/python scripts/check_review_packet.py review_packet.zip`
2. `sed -n '1,260p' dashboard/cheap_exhaustion_report.md`
3. `sed -n '1,260p' dashboard/c_a1_gpu_backfill_runbook.md`
4. `sed -n '1,220p' queues/gpu_backfill.yaml`
5. `sed -n '1,80p' queues/gpu_foreground.yaml`
6. `local_runner enqueue channel_set_c_a1_pair_materialization --models granite,deepseek,falcon --split dev,gate --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl --policies paroquant_baseline,tight_clip_c_a1 --scale-clip-min 0.5 --scale-clip-max 2.0 --require-same-row --write-access-manifest --fail-on-confirm --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}`
