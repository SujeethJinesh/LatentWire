# Qwen2.5-Math SVAMP32 Perceiver Answer-Likelihood 4-ID CPU Smoke

- date: `2026-04-26`
- git commit before run: `7a5bceb45fcbeb4099d161fa8f3dda070a7362e9`
- status: `answer_likelihood_controls_pass`
- scale-up rung: CPU micro smoke
- branch: Qwen2.5-Math SVAMP32 Perceiver C2C-residual checkpoint

## Purpose

Check whether the previously teacher-forced-negative Qwen2.5-Math Perceiver
checkpoint contains hidden gold-answer likelihood signal on a tiny clean-ID
slice before deciding whether to expand.

## Inputs

- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- device/dtype: `cpu` / `float32`
- checkpoint:
  `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - sha256: `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
- eval file:
  `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/svamp32_clean_eval_4.jsonl`
  - sha256: `3483101298677785e5d2827b0f5ff680db3b750d57b0b7153922f78a9ba6882d`
- ordered example count: `4`
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
  --eval-file results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/svamp32_clean_eval_4.jsonl \
  --task-type generation --device cpu --dtype float32 --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis --kv-transport k_only \
  --gate-mode fixed --fixed-gate 0.15 --methods rotalign \
  --prediction-output results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --source-use-chat-template --target-use-chat-template \
  --source-enable-thinking false --target-enable-thinking false \
  --random-salt 1
```

Controls: `zero_source`, `shuffled_source_salt1`, `target_only`, and
`slots_only` were run with the same command plus the appropriate control switch.

Analysis:

```bash
./venv_arm64/bin/python scripts/analyze_answer_likelihood_controls.py \
  --live results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --control zero_source=results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/zero_source.jsonl \
  --control shuffled_source_salt1=results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/shuffled_source_salt1.jsonl \
  --control target_only=results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/target_only.jsonl \
  --control slots_only=results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/slots_only.jsonl \
  --min-mean-delta 0.05 --min-best-control-wins 3 \
  --output-json results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json \
  --output-md results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.md
```

## Result

| row | n | finite | correct | mean answer logprob |
|---|---:|---:|---:|---:|
| matched | 4 | 4 | 0 | -7.989116 |
| zero_source | 4 | 4 | 0 | -8.250677 |
| shuffled_source_salt1 | 4 | 4 | 0 | -8.131923 |
| target_only | 4 | 4 | 0 | -8.162249 |
| slots_only | 4 | 4 | 0 | -8.118848 |

Matched beats all controls by the predefined micro-smoke rule:
best-control wins/losses/ties `3/1/0`, mean live-best delta `+0.080362`.
Decision: pass as a micro clue only; required immediate expansion to all clean
IDs.

## Artifact Hashes

- `answer_likelihood_controls.json`: `a834edac89d7721a2c54968c3007bca22d24e18eec248d99af2b2ffde1ddcfc9`
- `answer_likelihood_controls.md`: `739e3c1096181437315006763948038c6340dc9df1bbc38768db41bb70e05ac8`
- `matched.jsonl`: `9e98f44ad5ec88caf2b593e153166df0d678cd561900e6eef165f0251c439b13`
- `zero_source.jsonl`: `7cb026eab6f96a7218f9e4aa8e8c7b8dc72e3f709755c885f0895df75e29f1c1`
- `shuffled_source_salt1.jsonl`: `1ace4a6cdf524c2697e3c84addd76833205eef001213e8d804ec6b35088a1dd9`
- `target_only.jsonl`: `9d56b5f104a6d0a6b8a27e09d031916d2a20ee087ccc2bdce228d59cb3bd9c4d`
- `slots_only.jsonl`: `58570231a65162fde516e4971a0f1dc93144c22cdc3ef236bc8b91bf9336efe2`

## Next

The immediate clean6 expansion is the decisive gate. This 4-ID result is a
useful mechanism clue, not a positive method claim.
