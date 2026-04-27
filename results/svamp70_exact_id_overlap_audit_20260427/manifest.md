# SVAMP70 Exact-ID Overlap Audit

- date: `2026-04-27`
- status: `exact_id_overlap_audit_complete`
- git commit at run: `c3814de3d335d4996d9a05446363521db692f27b`
- scale-up rung: smoke / branch selection

## Inputs

- `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`
- `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`
- `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/semantic_predicate_decoder.json`
- `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/source_trace_router_after_sketch_kill.json`
- `results/noharm_source_predicate_pruning_20260427/source_likelihood_noharm_gate.json`
- `results/noharm_source_predicate_pruning_20260427/source_predicate_router_penalty025.json`
- `results/noharm_source_predicate_pruning_20260427/candidate_syndrome_bits4_probe.json`
- `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl`

## Output Hashes

- `exact_id_overlap_audit.json`:
  `358cb6b6db2a76dcea074df91e8e755d03d8114649cce78e019ed4f5626c4f5c`
- `exact_id_overlap_audit.md`:
  `92b688053c8948331b7df070538f645dfaa2746a456ada6c53347cc665bd9ec0`

## Decision

No branch has reusable canonical live+holdout clean IDs under target
preservation. Stop threshold sweeps on this surface; use the exact IDs as a
future falsification surface after either a fair receiver-control harness or a
stronger source interface exists.
