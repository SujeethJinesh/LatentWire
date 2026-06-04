# Stage-1 Triage

Triage-only pass from commit `1231a08a`; no new experiment data was generated. Confirm rows consumed: `0`. Parsed confirm rows counted but excluded: `46273`.

## Non-Killed Row Buckets
- status counts: `{'AMBIGUOUS': 100, 'CPU_SCREENED': 57, 'KILLED': 12, 'PARKED_NEEDS_GPU': 2}`
- AMBIGUOUS clusters: missing-metric/dev-only-no-gate-decision=77, underpower/gate-n-le-1=23
- PROVISIONAL_PROMOTE_TO_GPU rows: `0`

## L_SCORECOMP_deployable_WZ
- status: `KILLED`
- result paths: `results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget2.jsonl`, `results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget4.jsonl`, `results/source_private_wyner_ziv_cross_family_gate_20260429/core_to_holdout/predictions_budget6.jsonl`, `results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget2.jsonl`, `results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget4.jsonl`, `results/source_private_wyner_ziv_cross_family_gate_20260429/holdout_to_core/predictions_budget6.jsonl`, ... +9 more
- n_dev: `4581`
- n_gate: `1550`
- n_confirm: `0`
- strongest baseline: qjl_constrained_shuffled_source (accuracy=0.6396)
- method delta vs baseline: gate delta min=-0.4234, max=-0.0241; CI-low max=-0.1807; CI-high max=0.1205
- control outcomes: wrong-row/derangement controls are not present in the parsed WZ rows; equal-byte score/label baselines dominate all killed gate rows
- leakage outcomes: row confirm guard passed; deployability remains constrained because these cached rows are old WZ packet artifacts, not fresh high-entropy source-minus-target WZ
- next smallest falsifier: fresh high-entropy small-model WZ on Mac, with delta_beyond_score positive vs all equal-byte baselines before any confirm access
- eligibility: `kill`

## L_SCORECOMP_equal_byte_baselines
- status: `CPU_SCREENED`
- result paths: `results/source_private_arc_challenge_fixed_packet_gate_20260501_bge_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_hashed_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl`, ... +4 more
- n_dev: `3585`
- n_gate: `1275`
- n_confirm: `0`
- strongest baseline: target_only (accuracy=0.3333)
- method delta vs baseline: gate delta min=-0.0196, max=0.2389; CI-low max=0.1373; CI-high max=0.3628
- control outcomes: baseline-only rows; fixed packet sometimes beats target/shuffle, but it is source-copy context, not a candidate-method promotion
- leakage outcomes: row confirm guard passed; baseline rows are prior test/validation artifacts and cannot confirm a paper claim
- next smallest falsifier: reuse as locked baselines in fresh high-entropy LatentWire screens
- eligibility: `Mac_continue`
- gate condition accuracy: answer_only_text_forbidden_oracle=1.0000, candidate_derangement=0.1875, label_permutation=0.3929, matched_source_private_packet=0.3929, random_same_byte_packet=0.2376, same_byte_structured_text=0.3702, shuffled_source_packet=0.2557, target_derived_sidecar=0.2471, target_only=0.2455, zero_source=0.2455

## L_B1_damage_avoidance
- status: `KILLED`
- result paths: `results/source_private_arc_challenge_fixed_packet_gate_20260501_bge_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_hashed_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl`, `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl`, ... +4 more
- n_dev: `3585`
- n_gate: `1275`
- n_confirm: `0`
- strongest baseline: source-index/source-selected metadata
- method delta vs baseline: cached fixed packet has repair but no damage avoidance on gate
- control outcomes: matched packet equals source-selected answer on nearly all gate rows; this is source-copy behavior
- leakage outcomes: AURC, risk@coverage, confidence leakage, packet-source-top1 MI, wrong-row, and derangement are missing-row-fields in current cache
- next smallest falsifier: fresh Mac small-model L-B1 with explicit confidence/risk fields and wrong-row/derangement controls
- eligibility: `Mac_continue`
- dev cached L-B1 proxy: n=3585; target_acc=0.2575; source_acc=0.3632; matched_acc=0.3621; matched_equals_source=0.9983; damage=599/599 (1.0000); repair=975/978 (0.9969)
- gate cached L-B1 proxy: n=1275; target_acc=0.2455; source_acc=0.3945; matched_acc=0.3929; matched_equals_source=0.9969; damage=192/192 (1.0000); repair=380/382 (0.9948)

## L_A1_MMLU_ceiling
- status: `AMBIGUOUS`
- result paths: none
- n_dev: `0`
- n_gate: `0`
- n_confirm: `0`
- strongest baseline: not available
- method delta vs baseline: no cached MMLU-Pro ceiling rows in this Stage-1 surface
- control outcomes: not run
- leakage outcomes: no confirm leakage; no rows consumed
- next smallest falsifier: Mac MMLU-Pro info-ceiling probe with target-only, source-index, and equal-byte baselines
- eligibility: `Mac_continue`

## L_A2_rerank
- status: `AMBIGUOUS`
- result paths: none
- n_dev: `0`
- n_gate: `0`
- n_confirm: `0`
- strongest baseline: not available
- method delta vs baseline: no generated-solution candidate pools found in bounded cache pass
- control outcomes: not run
- leakage outcomes: no confirm leakage; no rows consumed
- next smallest falsifier: materialize a tiny dev/gate candidate-pool shard on Mac before any confirmation plan
- eligibility: `Mac_continue`

## C_A1
- status: `CPU_SCREENED`
- result paths: `experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/per_trace_metrics.json`, ... +3 more
- n_dev: `43`
- n_gate: `11`
- n_confirm: `0`
- strongest baseline: local ParoQuant cached parity / static recovery denominator
- method delta vs baseline: gate rows=6, positive_median=5, nonpositive_median=1, min_median=-1.1159, max_median=1.5666
- control outcomes: offline per-trace recovery only; no native W4A16 forward confirmation
- leakage outcomes: confirm rows excluded; native quantized-forward leakage cannot be assessed on Mac
- next smallest falsifier: GPU backfill parity card only if the same gate packet is replayed with real quantized forwards
- eligibility: `GPU_backfill`

## C_F
- status: `CPU_SCREENED`
- result paths: `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_20260528T2052Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_fixed_20260528T2116Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z/per_trace_metrics.json`, ... +6 more
- n_dev: `228`
- n_gate: `69`
- n_confirm: `0`
- strongest baseline: static top-K / random-matched controls
- method delta vs baseline: gate rows=36, positive_median=13, nonpositive_median=23, min_median=-14537.3834, max_median=5.9396
- control outcomes: mixed gate medians; some random/control regimes are positive, so there is no foreground confirmation target
- leakage outcomes: confirm rows excluded; forward-pass evidence is absent
- next smallest falsifier: tighten offline paired baseline and hazard denominator before GPU foreground
- eligibility: `GPU_backfill`

## CE13
- status: `AMBIGUOUS`
- result paths: none
- n_dev: `0`
- n_gate: `0`
- n_confirm: `0`
- strongest baseline: not available
- method delta vs baseline: no parseable warmup-policy cache found
- control outcomes: not run
- leakage outcomes: no offline evidence justifying foreground GPU
- next smallest falsifier: find or build a tiny dev/gate warmup-policy cache before any GPU forward
- eligibility: `Mac_continue`

## C_D1
- status: `KILLED`
- result paths: `experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z/per_trace_metrics.json`
- n_dev: `15`
- n_gate: `3`
- n_confirm: `0`
- strongest baseline: static_top10 / random_reactive_top1
- method delta vs baseline: gate rows=3, positive_median=0, nonpositive_median=3, min_median=-68.5624, max_median=-0.3606
- control outcomes: all parsed gate medians are negative and underpowered
- leakage outcomes: confirm rows excluded; no positive offline headroom
- next smallest falsifier: do not allocate GPU unless a new preregistered OSC surface appears
- eligibility: `kill`

## CE21
- status: `CPU_SCREENED`
- result paths: `experimental/outlier_migrate/phase9/results/om_driftrot_deepseek_clip_tight_20260528T2318Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_loose_confirmation_20260528T1735Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_finalpos8_20260528T2251Z/per_trace_metrics.json`, `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_holdout_20260528T2144Z/per_trace_metrics.json`, ... +16 more
- n_dev: `0`
- n_gate: `0`
- n_confirm: `0`
- strongest baseline: denominator audit only
- method delta vs baseline: no method delta; denominator audit rows only
- control outcomes: no-gap rows were counted, not promoted
- leakage outcomes: confirm rows excluded
- next smallest falsifier: carry as audit support for future Channel-Set screens
- eligibility: `Mac_continue`

## C_A2_tests
- status: `AMBIGUOUS`
- result paths: none
- n_dev: `0`
- n_gate: `0`
- n_confirm: `0`
- strongest baseline: not available
- method delta vs baseline: tests-only row; no method screen
- control outcomes: orthogonality/full-precision/KV-cache-basis tests not proven by current leaderboard
- leakage outcomes: no offline evidence justifying foreground GPU
- next smallest falsifier: run local correctness tests or keep parked; no GPU foreground
- eligibility: `Mac_continue`
