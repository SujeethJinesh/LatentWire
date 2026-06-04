# C_A1 GPU Backfill Runbook

This is a user-run GPU backfill packet, not a foreground confirmation job. Do not run it through SSH from Codex.

## Preconditions

- Acquire `.gpu.lock`.
- Start `nvidia-smi` monitoring.
- Use branch `codex-campaign` at or after the L_Q1 closeout commit.
- Do not run Stage-3, WZ/L-B1 reruns, sidecars, or confirm-claim promotion.
- Include both regression sentinels beyond Granite when possible: DeepSeek and Falcon.
- Keep promotion disabled. This packet is a native W4A16/ParoQuant replay/parity test, not a paper confirmation.
- Write `dashboard/gpu_backfill_report.md` with row IDs, no-gap denominator counts, parity-card status, sentinel medians/tails, and raw artifact paths.

## Exact Cached Gate IDs

Replay the cached C_A1 gate packet rather than searching new rows. The current cached gate evidence is summarized in `dashboard/stage1_triage.md`:

- aggregate gate rows: `6`
- positive median rows: `5`
- nonpositive median rows: `1`
- median range: `[-1.1159, 1.5666]`
- source rows include:
  - `experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z/per_trace_metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z/per_trace_metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z/per_trace_metrics.json`

The existing subset helper currently exposes the Granite cached packet as prompt indices `7,9`; do not widen beyond the exact cached gate IDs without a new preregistration.

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

## Regression Sentinel Baselines And Parity Card

Use DeepSeek and Falcon ParoQuant parity runs as sentinel denominators:

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

Promotion is disabled for this backfill packet. The report may recommend a later foreground job only if:

- CVaR/worst-trace improves without median regression.
- DeepSeek/Falcon sentinel does not regress in median or tail.
- matched controls fail.
- no-gap denominator audit passes.
- the parity card shows native W4A16/ParoQuant rows reproduce the cached direction on the exact gate IDs.

A Granite-only win that regresses DeepSeek/Falcon is a scoped negative, not a hero method.
