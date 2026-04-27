# SVAMP70 Perceiver Answer-Likelihood CPU Smoke

- date: `2026-04-26`
- git commit before run: `7a5bceb45fcbeb4099d161fa8f3dda070a7362e9`
- status: `answer_likelihood_controls_fail`
- scale-up rung: CPU micro smoke
- branch: SVAMP70 Perceiver answer-teacher contrastive checkpoint

## Purpose

Follow up the SVAMP70 teacher-forced Perceiver failure with gold-answer
continuation likelihood to check for hidden matched-source signal.

## Inputs

- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- device/dtype: `cpu` / `float32`
- checkpoint:
  `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - sha256: `a7221d6d0ee81b99573bf1893b66570ec682f22faee1ffcc6bf7e9fc1f36df6a`
- eval file:
  `results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/svamp70_clean_eval_4.jsonl`
  - sha256: `648c97fbc9f05f5f602a23346b311de0f19c84648f2a48d6f2707037e9f966e3`
- ordered example count: `4`
- gate: fixed `0.15`
- seed/control salt: `1`

## Commands

Matched run:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/svamp70_clean_eval_4.jsonl \
  --task-type generation --device cpu --dtype float32 --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis --kv-transport k_only \
  --gate-mode fixed --fixed-gate 0.15 --methods rotalign \
  --prediction-output results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --source-use-chat-template --target-use-chat-template \
  --source-enable-thinking false --target-enable-thinking false \
  --random-salt 1
```

Controls: `zero_source`, `shuffled_source_salt1`, `target_only`, and
`slots_only` were run with the same command plus the appropriate control switch.

Analysis:

```bash
./venv_arm64/bin/python scripts/analyze_answer_likelihood_controls.py \
  --live results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --control zero_source=results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/zero_source.jsonl \
  --control shuffled_source_salt1=results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/shuffled_source_salt1.jsonl \
  --control target_only=results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/target_only.jsonl \
  --control slots_only=results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/slots_only.jsonl \
  --min-mean-delta 0.05 --min-best-control-wins 3 \
  --output-json results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json \
  --output-md results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.md
```

## Result

| row | n | finite | correct | mean answer logprob |
|---|---:|---:|---:|---:|
| matched | 4 | 4 | 0 | -7.261671 |
| zero_source | 4 | 4 | 0 | -7.262390 |
| shuffled_source_salt1 | 4 | 4 | 0 | -7.220054 |
| target_only | 4 | 4 | 0 | -7.232674 |
| slots_only | 4 | 4 | 0 | -7.241025 |

Best-control wins/losses/ties are `0/4/0`, with mean matched-minus-best-control
delta `-0.112360`. Decision: fail.

## Artifact Hashes

- `answer_likelihood_controls.json`: `6b8778bde08cba1be3af04f8529a0c5e54d54507c4a9212e4977052b0e16f856`
- `answer_likelihood_controls.md`: `5c8345dadac9e548c8afeec9ac0a38827f1838ffb2b4576b637a8735a69c8c39`
- `matched.jsonl`: `b6882e8083561de5837e155c448a7e995b32368675c65bb9ccd2bc99e84a26cc`
- `zero_source.jsonl`: `82b2825eccf64bba6a9e8ae2f284cfce446140a0c8a99297a1d17c76cfd73fed`
- `shuffled_source_salt1.jsonl`: `22ef312c45a3d43715c9b8f01c5f756d92baecd78d8d1e6bbd61adf9850158a3`
- `target_only.jsonl`: `079fe7a3e8538b2b6e88ff752805e7cce257dcc9f0bf107a8922f7b71abc71ca`
- `slots_only.jsonl`: `abba4936581429941fc1a9735c8034ca0dc51b58e068b95f6c0c1bc05b917e68`

## Next

Do not revive the SVAMP70 Perceiver answer-teacher checkpoint. The answer
likelihood diagnostic agrees with the teacher-forced pre-gate: source signal is
control-explained.
