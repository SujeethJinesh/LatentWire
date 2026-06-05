# Cheap Exhaustion Report

- created_utc: `2026-06-05T18:04:49+00:00`
- git_head: `1c6a9546c176ed0321311bbce76b6860663f95fd`
- wall_clock_seconds: `0.550`
- files_seen: `1294`
- files_scanned: `820`
- confirm_path_excluded: `42`
- embedded_confirm_excluded: `35`
- gpu_foreground_empty: `True`

## Verdicts

| method | verdict | promotion_allowed | blocker |
|---|---:|---:|---|
| `C_U1_drift_as_signal_router` | `PARKED` | `False` | No non-confirm cache has row-level drift/KL trajectory features paired with row-level recovery/difficulty/uplift labels. |
| `C_W1_fixed_library_warmup_selector` | `PARKED` | `False` | No non-confirm cache exposes same-row outcomes for the fixed library ParoQuant/C_A1/survival/reject policies. |
| `CE1_codrift_givens_pairing` | `PARKED` | `False` | No non-confirm cache contains co-drift/Givens pairing fields joined to outcome labels. |
| `C_C1_budget_router` | `PARKED` | `False` | No non-confirm cache contains budget-conditioned same-row outcomes across usable policies. |
| `C_A1_cvar_evt_clip_grid` | `PARKED_NEEDS_NATIVE_PAIRING` | `False` | DeepSeek/Falcon have non-confirm 12-row baseline-vs-tight-clip pairs; Granite tight-clip candidates are embedded-confirm or diagnostic residual subsets, so the three-model C_A1 manifest is incomplete. |
| `C_A2_horizon_rotation` | `PARKED_TESTS_ONLY` | `False` | No explicit C_A2 orthogonality plus full-precision-equivalence test artifact was found in the non-confirm scan. |
| `C_D1_osc_decdec_vs_drift_defense` | `CACHE_PRESENT_UNDERPOWERED_SINGLE_MODEL` | `False` | Cached DecDEC/KL evidence is Granite-only and 12-row/8-included with wide CIs; usable as a defense note, not a positive method. |

## C_A1 Manifest Status

- manifest_complete: `False`
- granite: same_row_count=`0`, baseline_runs=`2`, tight_clip_runs=`0`
- deepseek: same_row_count=`12`, baseline_runs=`1`, tight_clip_runs=`1`
- falcon: same_row_count=`12`, baseline_runs=`1`, tight_clip_runs=`1`
- exact_gpu_row_materialization_command: `local_runner enqueue channel_set_c_a1_pair_materialization --models granite,deepseek,falcon --split dev,gate --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl --policies paroquant_baseline,tight_clip_c_a1 --scale-clip-min 0.5 --scale-clip-max 2.0 --require-same-row --write-access-manifest --fail-on-confirm --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}`
- excluded Granite tight-clip files with embedded confirm:
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/command_metadata.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/config.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/per_trace_metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/prompt_manifest.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/traces_used.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/command_metadata.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/config.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/per_trace_metrics.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/prompt_manifest.json`
  - `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/traces_used.json`

## LatentWire CPU-Safe Status

- `L_C2_control_trained_lcf_lite_proxy`: verdict=`PARKED`, status=`PARKED_NEEDS_CONTROL_TRAINED_FUSER_PRELAUNCH`, promotion_allowed=`False`, achieved_n=`0`. C2C fusers are local, but no reviewed byte-limited control-trained LCF-lite runner with wrong-row/zero-source objective exists.
- `L_PC1_cross_family_specialist_ceiling`: verdict=`PARKED`, status=`PARKED_NEEDS_RECEIVER_CONDITIONED_CEILING_RUNNER`, promotion_allowed=`False`, achieved_n=`768`. No eligible existing artifact measures I(source_signal;Y|receiver_state) for a complementary source-private signal.
- `L_PC2_tool_augmented_source_ceiling`: verdict=`PARKED`, status=`PARKED_NEEDS_TOOL_PRIVATE_CACHE_AND_RUNNER`, promotion_allowed=`False`, achieved_n=`0`. No dev/gate cache found where the source privately ran calculator/code-exec and receiver did not see the tool result.
- `L_PC5_private_verifier_receiver_candidates`: verdict=`PARKED`, status=`PARKED_POOL_TOO_WEAK`, promotion_allowed=`False`, achieved_n=`36`. candidate pool is nondegenerate for only 36 prompts, below the 80-prompt rerank gate; no gain verdict emitted

## Required Native Cache Requirements

- `C_U1_drift_as_signal_router`: For each Granite/DeepSeek/Falcon row: prompt_id, split, receiver confidence, KL/drift trajectory by position/layer, warmup activation features, static_gap/recoverable_gap, and per-policy uplift labels.
- `C_W1_fixed_library_warmup_selector`: Same prompt_id rows for paroquant, tight_clip_c_a1, survival_core, and reject/no_answer policies with dev/gate split and row-safe outcomes.
- `CE1_codrift_givens_pairing`: Per-row paired-channel/Givens metadata, co-drift score, row_id, static_gap, and policy recovery for the paired intervention and its controls.
- `C_C1_budget_router`: Per-row budget, policy_id, byte/compute cost, ParoQuant/static/tight-clip outcomes, reject option, and no-gap denominator fields.
- `C_A2_horizon_rotation`: Tests must pass before even a 2-bin C_A2 screen; no GPU queue item should be emitted from this pass.
- `C_D1_osc_decdec_vs_drift_defense`: Same-row OSC/DecDEC stress surfaces joined to drift/KL and outcome labels across at least two models.

## Next 6 Commands

1. `venv_arm64/bin/python scripts/check_review_packet.py review_packet.zip`
2. `sed -n '1,260p' dashboard/cheap_exhaustion_report.md`
3. `sed -n '1,260p' dashboard/c_a1_gpu_backfill_runbook.md`
4. `sed -n '1,220p' queues/gpu_backfill.yaml`
5. `sed -n '1,80p' queues/gpu_foreground.yaml`
6. `local_runner enqueue channel_set_c_a1_pair_materialization --models granite,deepseek,falcon --split dev,gate --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl --policies paroquant_baseline,tight_clip_c_a1 --scale-clip-min 0.5 --scale-clip-max 2.0 --require-same-row --write-access-manifest --fail-on-confirm --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}`
