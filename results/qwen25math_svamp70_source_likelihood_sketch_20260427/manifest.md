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

The full collection commands now use `--resume`; the memo also includes a
two-example `.debug/` smoke using `--limit 2` before the full live/holdout run.

## CPU Fallback Smoke

While PID `31103` continued blocking MPS, a CPU-only two-example collector
smoke was run as a tooling check:

- JSONL:
  `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.jsonl`
- JSONL sha256:
  `863254ecc5110eab3e62efb65ddb31e9472be42513bce6ce1ab44842e1057e9d`
- markdown:
  `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.md`
- markdown sha256:
  `cd12db13419021f248c311776e9c3b148d60faa69297c31b6a8d272fc863d0f9`
- rows: `2`
- elapsed: `96.06s`
- git commit:
  `154430a33d0d649e30b877d7b4d38015a229ac9a`
