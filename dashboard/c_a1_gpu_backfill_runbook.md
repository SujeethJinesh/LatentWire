# C_A1 GPU Backfill Runbook

This is a user-run GPU backfill packet, not a foreground confirmation job. Do not run it through SSH from Codex.

## Preconditions

- Acquire `.gpu.lock`.
- Start `nvidia-smi` monitoring.
- Use branch `codex-campaign` at or after the L_Q1 closeout commit.
- Do not run Stage-3, WZ/L-B1 reruns, sidecars, or confirm-claim promotion.
- Include at least one regression sentinel beyond Granite: DeepSeek and/or Falcon.

## Granite Tail/CVaR Replay

```bash
python experimental/outlier_migrate/phase9/run_om_driftrot_clip_subset.py \
  --run-id <new_write_once_c_a1_granite_gate_replay> \
  --candidate-id clip_tight \
  --split-name diagnostic \
  --prompt-indices 7,9 \
  --base-run-dir experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --batch-size 1 \
  --dtype bfloat16
```

## Regression Sentinel Baselines

Use DeepSeek and/or Falcon ParoQuant parity runs as sentinel denominators:

```bash
python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py \
  --run-id <new_write_once_c_a1_deepseek_sentinel> \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --batch-size 1 \
  --dtype bfloat16

python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py \
  --run-id <new_write_once_c_a1_falcon_sentinel> \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --batch-size 1 \
  --dtype bfloat16
```

## Promotion Rule

Write `dashboard/gpu_backfill_report.md`. Mark C_A1 `PROVISIONAL_PROMOTE_TO_GPU` only if:

- CVaR/worst-trace improves without median regression.
- DeepSeek/Falcon sentinel does not regress in median or tail.
- matched controls fail.
- no-gap denominator audit passes.

A Granite-only win that regresses DeepSeek/Falcon is a scoped negative, not a hero method.
