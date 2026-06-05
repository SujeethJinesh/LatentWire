# Confirm Path Audit

AUDIT_STATUS: COMPLETE
C_A1_GATE_STATUS: INVALID_CONFIRM_CONTAMINATED

## Summary

- Embedded confirm-looking `source_path` hits: `18`
- Contaminated screening sources: `16`
- Historical/text/non-claim contexts: `2`
- Raw `*_confirm*` paths remain excluded from review packets.

## C_A1 Gate Verdict

C_A1 cached gate evidence is **invalid for GPU handoff selection** because at least one gate/source row references `om_driftrot_granite_clip_tight_confirmation_20260528T1735Z`.

Required action: rebuild the C_A1 backfill packet on fresh non-confirm dev/gate row IDs before any GPU spend. Do not replay the contaminated cached Granite gate IDs.

## Contaminated Hits

| file | line | verdict | source_path |
| --- | ---: | --- | --- |
| `dashboard/leaderboard.csv` | 58 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `dashboard/leaderboard.csv` | 59 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `dashboard/leaderboard.csv` | 60 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `dashboard/leaderboard.csv` | 61 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `dashboard/leaderboard.csv` | 62 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 21 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 22 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 23 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 24 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 25 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/mps_first/C_U1_drift_as_signal_router/raw_rows.jsonl` | 26 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/stage1/leaderboard.csv` | 58 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/stage1/leaderboard.csv` | 59 | `CONTAMINATED_CHANNEL_SET_SCREEN_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/stage1/leaderboard.csv` | 60 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/stage1/leaderboard.csv` | 61 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |
| `results/stage1/leaderboard.csv` | 62 | `CONTAMINATED_C_A1_GATE_SOURCE` | `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json` |

## Historical / Non-Claim Hits

| file | line | verdict | source_path |
| --- | ---: | --- | --- |
| `dashboard/confirm_path_audit.md` | 8 | `HISTORICAL_TEXT_OR_NON_CLAIM_CONTEXT` | `- Embedded confirm-looking `source_path` hits: `17`` |
| `dashboard/leaderboard.csv` | 178 | `HISTORICAL_TEXT_OR_NON_CLAIM_CONTEXT` | `results/mac_continue/latentwire_one_way_confirm/confirm_rows.jsonl` |
