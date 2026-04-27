# SVAMP70 Exact-ID Overlap Audit

- date: `2026-04-27`
- status: `exact_id_overlap_audit_complete`
- git commit: `c3814de3d335d4996d9a05446363521db692f27b`

## Decision

No branch has canonical live+holdout reusable clean IDs under target preservation; canonical live positives cluster on three generated-source semantic IDs, while canonical holdout has only a trace-router single ID and semantic/likelihood receivers fail.

## canonical_live

- clean source-only count: `6`
- clean source-only IDs: `14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `bd9d8da923981d69`, `ce08a3a269bf0151`
- recovered IDs never recovered by audited branches: none
- recovered IDs not in this surface clean set: none

### Branch Recoveries

- `semantic_predicate_strict_harm20`: IDs `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`; status `semantic_predicate_decoder_fails_smoke`; artifact `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/semantic_predicate_decoder.json`
- `source_trace_router`: IDs `2de1549556000830`; status `source_trace_router_fails_gate`; artifact `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json`
- `source_trace_router_after_sketch_kill`: IDs `2de1549556000830`; status `source_trace_router_fails_gate`; artifact `results/qwen25math_svamp70_source_likelihood_sketch_20260427/source_trace_router_after_sketch_kill.json`
- `source_likelihood_noharm_gate`: IDs none; status `source_likelihood_sketch_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_likelihood_noharm_gate.json`
- `source_predicate_router_penalty025_run0`: IDs none; status `source_sidecar_cv_router_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_predicate_router_penalty025.json`
- `source_predicate_router_penalty025_run1`: IDs `ce08a3a269bf0151`; status `source_sidecar_cv_router_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_predicate_router_penalty025.json`
- `source_predicate_router_penalty025_run2`: IDs `2de1549556000830`, `4d780f825bb8541c`, `ce08a3a269bf0151`; status `source_sidecar_cv_router_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_predicate_router_penalty025.json`
- `source_predicate_router_penalty025_run3`: IDs `2de1549556000830`, `4d780f825bb8541c`, `ce08a3a269bf0151`; status `source_sidecar_cv_router_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_predicate_router_penalty025.json`
- `qwen3_target_likelihood_accept_all_source_top_diagnostic`: IDs `14bfbfc94f2c2e7b`, `2de1549556000830`, `4d780f825bb8541c`, `41cce6c6e6bb0058`, `ce08a3a269bf0151`, `bd9d8da923981d69`; status `fails_live_prune`; artifact `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl`

### Recovered ID Counts

- `14bfbfc94f2c2e7b`: `1` via `qwen3_target_likelihood_accept_all_source_top_diagnostic`
- `2de1549556000830`: `6` via `semantic_predicate_strict_harm20`, `source_trace_router`, `source_trace_router_after_sketch_kill`, `source_predicate_router_penalty025_run2`, `source_predicate_router_penalty025_run3`, `qwen3_target_likelihood_accept_all_source_top_diagnostic`
- `41cce6c6e6bb0058`: `2` via `semantic_predicate_strict_harm20`, `qwen3_target_likelihood_accept_all_source_top_diagnostic`
- `4d780f825bb8541c`: `4` via `semantic_predicate_strict_harm20`, `source_predicate_router_penalty025_run2`, `source_predicate_router_penalty025_run3`, `qwen3_target_likelihood_accept_all_source_top_diagnostic`
- `bd9d8da923981d69`: `1` via `qwen3_target_likelihood_accept_all_source_top_diagnostic`
- `ce08a3a269bf0151`: `4` via `source_predicate_router_penalty025_run1`, `source_predicate_router_penalty025_run2`, `source_predicate_router_penalty025_run3`, `qwen3_target_likelihood_accept_all_source_top_diagnostic`

## canonical_holdout

- clean source-only count: `2`
- clean source-only IDs: `ab1e71e8928661d0`, `daea537474de16ac`
- recovered IDs never recovered by audited branches: `ab1e71e8928661d0`
- recovered IDs not in this surface clean set: none

### Branch Recoveries

- `semantic_predicate_strict_harm20`: IDs none; status `semantic_predicate_decoder_fails_smoke`; artifact `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/semantic_predicate_decoder.json`
- `source_trace_router`: IDs `daea537474de16ac`; status `source_trace_router_fails_gate`; artifact `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json`
- `source_trace_router_after_sketch_kill`: IDs `daea537474de16ac`; status `source_trace_router_fails_gate`; artifact `results/qwen25math_svamp70_source_likelihood_sketch_20260427/source_trace_router_after_sketch_kill.json`
- `source_likelihood_noharm_gate`: IDs none; status `source_likelihood_sketch_fails_gate`; artifact `results/noharm_source_predicate_pruning_20260427/source_likelihood_noharm_gate.json`

### Recovered ID Counts

- `daea537474de16ac`: `2` via `source_trace_router`, `source_trace_router_after_sketch_kill`

## adjacent_chal171_240

- clean source-only count: `1`
- clean source-only IDs: `4157958051c69d70`
- recovered IDs never recovered by audited branches: none
- recovered IDs not in this surface clean set: none

### Branch Recoveries

- `candidate_syndrome_bits4_probe`: IDs `4157958051c69d70`; status `candidate_syndrome_decoder_fails_smoke`; artifact `results/noharm_source_predicate_pruning_20260427/candidate_syndrome_bits4_probe.json`

### Recovered ID Counts

- `4157958051c69d70`: `1` via `candidate_syndrome_bits4_probe`

## adjacent_chal241_310

- clean source-only count: `4`
- clean source-only IDs: `0ee313c160b638a9`, `561daa750422c0e4`, `cd5623c80cf95da9`, `e90d2681e386fb04`
- recovered IDs never recovered by audited branches: none
- recovered IDs not in this surface clean set: none

### Branch Recoveries

- `candidate_syndrome_bits4_probe`: IDs `0ee313c160b638a9`, `561daa750422c0e4`, `cd5623c80cf95da9`, `e90d2681e386fb04`; status `candidate_syndrome_decoder_fails_smoke`; artifact `results/noharm_source_predicate_pruning_20260427/candidate_syndrome_bits4_probe.json`

### Recovered ID Counts

- `0ee313c160b638a9`: `1` via `candidate_syndrome_bits4_probe`
- `561daa750422c0e4`: `1` via `candidate_syndrome_bits4_probe`
- `cd5623c80cf95da9`: `1` via `candidate_syndrome_bits4_probe`
- `e90d2681e386fb04`: `1` via `candidate_syndrome_bits4_probe`

## adjacent_chal311_380

- clean source-only count: `2`
- clean source-only IDs: `3078d2d3fbb94c95`, `63e908e9f42b0637`
- recovered IDs never recovered by audited branches: none
- recovered IDs not in this surface clean set: none

### Branch Recoveries


### Recovered ID Counts


## Next Gate

Do not run another threshold sweep on current canonical SVAMP70 artifacts. If MPS remains blocked, the next CPU-only move is method design/harness work for true condition-specific receiver controls or source-surface discovery with stronger source prompts/models once compute clears.
