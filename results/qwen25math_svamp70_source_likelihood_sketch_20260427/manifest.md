# Qwen2.5-Math SVAMP70 Source Likelihood Sketch Manifest

Date: 2026-04-27

Status: `source_likelihood_and_post_sketch_syndrome_killed`

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

## Kill Results

The live branch is killed on this surface. Durable readout:

- `paper/source_likelihood_sketch_kill_20260427.md`

Key artifacts:

- `live_normpred_sketch_cpu.jsonl`
  - sha256: `b08b0f5eab854dd7da9f4099238e5f300a2c2508a66341f58ae40d0adaa91254`
- `holdout_normpred_sketch_cpu.jsonl`
  - sha256: `932e5449d5174c1f484f1535575e5b06763bc797fd83b8fec804471d300d5f15`
- `normpred_cpu_sketch_gate.json`
  - sha256: `431844070c012be613b98d64c1effa1b88a14a542d37cc8c2bdf6320942fca55`
- `normpred_sumlogprob_cpu_sketch_gate.json`
  - sha256: `51893cebe1a2cbcb1e91788fbf98d8b721fbbdf858ead6787e63e9efe0ff2e3e`
- `normpred_answer_template_cpu_sketch_gate.json`
  - sha256: `b14d22ceda15c5d83104becbe0645d078c5d47a3c9872bb98d4c0217ec3a92b1`
- `normpred_answer_template_sumlogprob_cpu_sketch_gate.json`
  - sha256: `ad33fd03dac92f5f85c6fcfc24562b6a19c6ba065fb0f5e6c98ed1c2cee9724f`
- `source_trace_router_after_sketch_kill.json`
  - sha256: `e4e5600e139efbf7bc068ff2117e172cba9f87055e9477f51839a90175c54c03`

## Post-Kill Syndrome Bounds

Focused readout:

- `paper/svamp70_syndrome_bounds_after_sketch_kill_20260427.md`

Decision: do not implement a richer learned source-syndrome predictor on this
exact SVAMP70 live/holdout surface. The bounds themselves fail either
target-self preservation or holdout controls.

Artifacts:

- `svamp70_live_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256: `662534e42454526872e760d3ca622daa25b04c84a82bf8600111533770b0d857`
- `svamp70_live_c2c_syndrome_bound_after_sketch_kill.md`
  - sha256: `1278aa33c3f89601b027eb130dcb6ffbc8cd3bc3545c6510f3e7cd8182c92614`
- `svamp70_holdout_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256: `a1125332f34e4585cb3efbc5d6e5b4ad7a4059695d03347881f1fde0000a9d29`
- `svamp70_holdout_c2c_syndrome_bound_after_sketch_kill.md`
  - sha256: `1044c8ec73575892d77f83f0705ade14328b8ec7525311e074f5184447c64a9a`
- `svamp70_live_source_syndrome_bound_after_sketch_kill.json`
  - sha256: `3369b5590c7ea36c732ca8b03904bffe4af18d8067e2c4624e4798a6ce9fcb0a`
- `svamp70_live_source_syndrome_bound_after_sketch_kill.md`
  - sha256: `d9b5ad34397d5d9fffa0f27829b7039573213e84af3b15e6d55e3141a054de35`
- `svamp70_holdout_source_syndrome_bound_after_sketch_kill.json`
  - sha256: `e6fd0ea71815835dc9a9026b8e3d75751c22830fe40b905f2ae1c533b672001f`
- `svamp70_holdout_source_syndrome_bound_after_sketch_kill.md`
  - sha256: `e9aa9e145fa86f3bc37a87500b0b6bdbccbbcd653899a4f4b222413185f7682f`
