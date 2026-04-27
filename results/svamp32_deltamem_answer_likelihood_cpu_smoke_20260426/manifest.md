# SVAMP32 Delta-Memory Answer-Likelihood CPU Smoke

- date: `2026-04-26`
- git commit before run: `7a5bceb45fcbeb4099d161fa8f3dda070a7362e9`
- status: `answer_likelihood_controls_fail`
- scale-up rung: CPU micro smoke
- branch: target-conditioned delta-memory query codec

## Purpose

Test whether the target-conditioned delta-memory checkpoint has a matched-source
gold-answer likelihood advantage once `target_only` and `slots_only` controls
are available.

## Inputs

- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- device/dtype: `cpu` / `float32`
- checkpoint:
  `.debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `29ff93c6d7291fb9a4e00ac35a7ffa519c4d71c8bd4a38062c0d748baecf4ebb`
- eval file:
  `results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/svamp_eval_4.jsonl`
  - sha256: `d92c33af27c4e84a5ae83f882e910d9ce3c27c902e1bb0ec310092380c160a6a`
- ordered example count: `4`
- gate: fixed `0.15`
- seed/control salt: `1`

## Commands

Matched run:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/svamp_eval_4.jsonl \
  --task-type generation --device cpu --dtype float32 --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis --kv-transport k_only \
  --gate-mode fixed --fixed-gate 0.15 --methods rotalign \
  --prediction-output results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --source-use-chat-template --target-use-chat-template \
  --source-enable-thinking false --target-enable-thinking false \
  --random-salt 1
```

Control runs used the same command with one switch and a changed output path:

- `--source-kv-control zero` -> `zero_source.jsonl`
- `--source-prompt-control shuffle_examples` -> `shuffled_source_salt1.jsonl`
- `--innovation-memory-control target_only` -> `target_only.jsonl`
- `--innovation-memory-control slots_only` -> `slots_only.jsonl`

Analysis:

```bash
./venv_arm64/bin/python scripts/analyze_answer_likelihood_controls.py \
  --live results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --control zero_source=results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/zero_source.jsonl \
  --control shuffled_source_salt1=results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/shuffled_source_salt1.jsonl \
  --control target_only=results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/target_only.jsonl \
  --control slots_only=results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/slots_only.jsonl \
  --min-mean-delta 0.05 --min-best-control-wins 3 \
  --output-json results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json \
  --output-md results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.md
```

## Result

| row | n | finite | correct | mean answer logprob |
|---|---:|---:|---:|---:|
| matched | 4 | 4 | 0 | -7.673776 |
| zero_source | 4 | 4 | 0 | -7.568559 |
| shuffled_source_salt1 | 4 | 4 | 0 | -7.683792 |
| target_only | 4 | 4 | 0 | -8.072149 |
| slots_only | 4 | 4 | 0 | -8.071200 |

Matched beats `target_only` and `slots_only`, but loses to `zero_source` on
mean likelihood and has best-control wins/losses/ties `0/4/0`. Decision: fail.

## Artifact Hashes

- `answer_likelihood_controls.json`: `a0d39d36ac0cd3b4bc1c4a25d211e2b48554f5c716722188147b4e8c20122615`
- `answer_likelihood_controls.md`: `4b91a9b98eac5503f7b656e9f786042e0476c44e4f360740f79686c0bf38860c`
- `matched.jsonl`: `22fe34a11b77a637d857ed289acffa3c239ba293ccc11f560dde172b3777e839`
- `zero_source.jsonl`: `49929582487209d08950c9e113ec1e011c946473d18e287cc757caa6b17a3993`
- `shuffled_source_salt1.jsonl`: `203b387c4b29bfb7f9cb1df8e1ce99f3ed3fe60f2052bfed8bca2233225c50db`
- `target_only.jsonl`: `ad680cb8da9e63a0dbbd25b5a11d2aaf3c37b71dace491c6399224f716ba75f6`
- `slots_only.jsonl`: `7000db1500f4cc342320181f9a22733de8a2e890c50b3487aa0feaf3e5279922`

## Next

Do not scale this checkpoint. The target-conditioned delta-memory variant is
weakened/killed as a live source-communication row because the matched-source
advantage is not robust to the zero-source control.
