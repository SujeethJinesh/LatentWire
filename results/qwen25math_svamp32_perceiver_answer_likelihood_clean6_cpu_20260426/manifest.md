# Qwen2.5-Math SVAMP32 Perceiver Answer-Likelihood Clean6 CPU Gate

- date: `2026-04-26`
- git commit before run: `7a5bceb45fcbeb4099d161fa8f3dda070a7362e9`
- status: `answer_likelihood_controls_fail`
- scale-up rung: strict-small clean-ID CPU gate
- branch: Qwen2.5-Math SVAMP32 Perceiver C2C-residual checkpoint

## Purpose

Expand the 4-ID answer-likelihood pass to all six clean C2C-headroom IDs. This
is the decisive source-control gate for the checkpoint.

## Inputs

- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- device/dtype: `cpu` / `float32`
- checkpoint:
  `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - sha256: `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
- eval file:
  `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/svamp32_clean_eval_6.jsonl`
  - sha256: `09b9209f7941931819bb73c8797a737b5a126953cda5c35d06968c9e666308e4`
- ordered example count: `6`
- gate: fixed `0.15`
- seed/control salt: `1`

## Commands

Matched run:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/svamp32_clean_eval_6.jsonl \
  --task-type generation --device cpu --dtype float32 --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis --kv-transport k_only \
  --gate-mode fixed --fixed-gate 0.15 --methods rotalign \
  --prediction-output results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/matched.jsonl \
  --source-use-chat-template --target-use-chat-template \
  --source-enable-thinking false --target-enable-thinking false \
  --random-salt 1
```

Controls: `zero_source`, `shuffled_source_salt1`, `target_only`, and
`slots_only` were run with the same command plus the appropriate control switch.

Analysis:

```bash
./venv_arm64/bin/python scripts/analyze_answer_likelihood_controls.py \
  --live results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/matched.jsonl \
  --control zero_source=results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/zero_source.jsonl \
  --control shuffled_source_salt1=results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/shuffled_source_salt1.jsonl \
  --control target_only=results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/target_only.jsonl \
  --control slots_only=results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/slots_only.jsonl \
  --min-mean-delta 0.05 --min-best-control-wins 4 \
  --output-json results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/answer_likelihood_controls.json \
  --output-md results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/answer_likelihood_controls.md
```

## Result

| row | n | finite | correct | mean answer logprob |
|---|---:|---:|---:|---:|
| matched | 6 | 6 | 0 | -8.195434 |
| zero_source | 6 | 6 | 0 | -8.387585 |
| shuffled_source_salt1 | 6 | 6 | 0 | -8.190414 |
| target_only | 6 | 6 | 0 | -8.192871 |
| slots_only | 6 | 6 | 0 | -8.191226 |

Matched beats zero-source strongly, but fails the mean-delta rule against
shuffled-source, target-only, and slots-only controls. Best-control
wins/losses/ties are `4/2/0`, but mean matched-minus-best-control is
`-0.090384`. Decision: fail.

## Artifact Hashes

- `answer_likelihood_controls.json`: `ad731dfa93c08bfb6cd27999a53a11c2f273599722d2d1224ef8df55f94cb0bd`
- `answer_likelihood_controls.md`: `25ecdb05bfa1865aa08c3dead09aea727b185774280a9dea35b1c0380f79ef0a`
- `matched.jsonl`: `1e8c17b9c487c400adbb7fdfcf682913bbe9ece669824b1e4bd4a55725f9d553`
- `zero_source.jsonl`: `6926ce3f649435197d310391f2878c0f5b630587fe9074e7380522cda3e27bc3`
- `shuffled_source_salt1.jsonl`: `bb061fa24653322134635e772c9d65d47f1d621b413857ce83650df831db3fd9`
- `target_only.jsonl`: `207a36df7189487dd39938bd9dfef89e92ee2109d2a5052179259ab1149d41d9`
- `slots_only.jsonl`: `ee095224514f08e8367ce0f9cb75ae482096bb14ed981aaa3cf55c3fbb275134`

## Decision

Kill this checkpoint as a strict positive method candidate. The 4-ID pass was
a useful partial mechanism clue, but it does not survive the full clean-ID gate
because target/slot/shuffle controls match or beat the live row on mean answer
likelihood.

## Next

Do not tune another Perceiver memory checkpoint on this exact surface. The next
gate should reset toward stronger source-surface/interface discovery after the
stuck MPS process is cleared.
