# Qwen2.5-Math SVAMP70 Source Likelihood Sketch Manifest

Date: 2026-04-27

Status: `harness_ready_run_blocked_by_mps_pid_31103`

## Branch

Source likelihood sketch: source model scores target/text/source candidate
predictions as continuations of the source prompt; the gate transmits only a
top-label plus quantized confidence margin and tests strict source-destroying
controls.

## Planned Artifacts

- `live_sketch.jsonl`
- `live_sketch.md`
- `holdout_sketch.jsonl`
- `holdout_sketch.md`
- `sketch_gate.json`
- `sketch_gate.md`
- `sketch_gate_predictions.jsonl`

## Inputs

- live eval:
  `results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl`
- live candidates:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl`
- live target set:
  `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`
- holdout eval:
  `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl`
- holdout candidates:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl`
- holdout target set:
  `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`

## Resume

Confirm the stuck MPS process is gone:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Then run the three commands in
`paper/svamp70_source_likelihood_sketch_20260427.md`.
