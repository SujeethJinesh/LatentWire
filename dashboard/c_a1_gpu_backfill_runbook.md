# C_A1 GPU Backfill Runbook

This is a user-run GPU backfill packet, not a foreground confirmation job. Do not run it through SSH from Codex.

## Preconditions

- Acquire `.gpu.lock`.
- Start `nvidia-smi` monitoring.
- Use branch `codex-campaign` at or after the L_Q1 closeout commit.
- Do not run Stage-3, WZ/L-B1 reruns, sidecars, or confirm-claim promotion.
- Include both regression sentinels beyond Granite: DeepSeek and Falcon.
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

Exact cached row identifiers for this packet:

| model | cached row IDs | source packet | requirement |
| --- | --- | --- | --- |
| Granite | prompt indices `7,9` (`opencompass_AIME2025_I_7`, `opencompass_AIME2025_I_9`) | current subset helper over the Granite cached gate packet | run ParoQuant baseline and tight-clip C_A1 on the same two rows |
| DeepSeek | prompt indices `0,1,2,3,4,5,6,7,8,9,10,11` (`opencompass_AIME2025_I_0`..`I_11`) | `om_driftrot_deepseek_clip_tight_20260528T2318Z` | run ParoQuant baseline and tight-clip C_A1 on the same twelve rows |
| Falcon | prompt indices `0,1,2,3,4,5,6,7,8,9,10,11` (`opencompass_AIME2025_I_0`..`I_11`) | `om_driftrot_falcon_clip_tight_20260528T2343Z` | run ParoQuant baseline and tight-clip C_A1 on the same twelve rows |

Abort the packet if the local runner cannot materialize both ParoQuant and tight-clip rows on the same prompt IDs for a model. A ParoQuant-only DeepSeek/Falcon denominator is not a sentinel and must not be reported as a C_A1 replay.

## Paired Native Replay Matrix

Every model needs two write-once rows: `paroquant_baseline` and `tight_clip_c_a1`. Do not compare a newly generated C_A1 row against a stale denominator with different prompt IDs.

### Granite

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

### DeepSeek

Run the ParoQuant denominator and the tight-clip C_A1 replay on exactly the same cached DeepSeek IDs:

```bash
python experimental/outlier_migrate/phase9/run_om_v1_paroquant_deepseek.py \
  --run-id <new_write_once_c_a1_deepseek_paroquant_paired> \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --batch-size 1 \
  --dtype bfloat16

python <model_specific_tight_clip_subset_runner_for_deepseek> \
  --run-id <new_write_once_c_a1_deepseek_tight_clip_paired> \
  --candidate-id clip_tight \
  --split-name diagnostic \
  --prompt-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --base-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --batch-size 1 \
  --dtype bfloat16
```

### Falcon

Run the ParoQuant denominator and the tight-clip C_A1 replay on exactly the same cached Falcon IDs:

```bash
python experimental/outlier_migrate/phase9/run_om_v1_paroquant_falcon.py \
  --run-id <new_write_once_c_a1_falcon_paroquant_paired> \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --batch-size 1 \
  --dtype bfloat16

python <model_specific_tight_clip_subset_runner_for_falcon> \
  --run-id <new_write_once_c_a1_falcon_tight_clip_paired> \
  --candidate-id clip_tight \
  --split-name diagnostic \
  --prompt-indices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --base-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --batch-size 1 \
  --dtype bfloat16
```

The existing `run_om_driftrot_clip_subset.py` is Granite-bound through its checker module. Do not use it for DeepSeek/Falcon unless the local runner first installs an explicit model-specific wrapper that patches the checker in the same style as `run_om_v1_paroquant_deepseek.py` and `run_om_v1_paroquant_falcon.py`, and the wrapper records the expected model ID in `model_provenance.json`.

## Parity Card And No-Gap Audit

For each model, the report must include:

- paired row IDs used by both regimes.
- no-gap denominator counts.
- median delta and CVaR/worst-trace delta for tight-clip C_A1 versus ParoQuant.
- paired bootstrap lower bounds.
- whether the DeepSeek and Falcon sentinel medians or tails regress.

## Promotion Rule

Promotion is disabled for this backfill packet. The report may recommend a later foreground job only if:

- CVaR/worst-trace improves without median regression.
- DeepSeek/Falcon sentinel does not regress in median or tail.
- matched controls fail.
- no-gap denominator audit passes.
- the parity card shows native W4A16/ParoQuant rows reproduce the cached direction on the exact gate IDs.

A Granite-only win that regresses DeepSeek/Falcon is a scoped negative, not a hero method.
