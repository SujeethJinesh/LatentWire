# Frozen Candidate Score Sidecar Collection

- date: `2026-04-27`
- status: `frozen_candidate_score_sidecar_collected`
- scorer model: `Qwen/Qwen2.5-Math-1.5B`
- candidate pool: `target_side_only`
- continuation template: `Answer: {text}`
- git commit: `2deef3b445219036016ac4092617a1208f0fb70a`
- target set: `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`
- target set sha256: `fc5eb4ca577ca33e01a9b23f427ff5d45346e1d2435392a7ce01ca25631fc729`
- eval file: `results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl`
- eval file sha256: `bc9178d043bc05f2d1d1dd4aa2c6ec1ed024643b1a2886dfd243c4b1eca3e131`
- output JSONL: `results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl`
- output JSONL sha256: `3734e4884c87bc14d3bc74317a47c195bbac85253927ae799e3eaa717cf2e771`
- rows: `70`
- sidecar bits: `32`
- ordered IDs sha256: `0292230b41840995d6c178c72b571f4f4441e631a6e7f1535a03106717010506`
- resume: `True`
- skipped existing: `0`
- device: `cpu`
- dtype: `float32`

## Command

```bash
scripts/collect_svamp_frozen_candidate_score_sidecar.py --scorer-model Qwen/Qwen2.5-Math-1.5B --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl --continuation-template 'Answer: {text}' --device cpu --dtype float32 --sidecar-bits 32 --scorer-use-chat-template --scorer-enable-thinking false --resume --date 2026-04-27 --output-jsonl results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl --output-md results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.md
```

The JSONL contains model-scored sidecar preferences over target-side
candidate values only. It intentionally omits gold answers, correctness
labels, and source-only candidate values.
