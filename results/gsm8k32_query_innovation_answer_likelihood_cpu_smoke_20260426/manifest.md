# GSM8K32 Query-Innovation Answer-Likelihood CPU Smoke

- date: `2026-04-26`
- git commit before run: `7749b73429d4476d1549690f68e7996282618968`
- status: `answer_likelihood_controls_fail`
- scale-up rung: CPU micro smoke
- live branch: `dynalign_query_innovation_resampler_replace`, gate `0.15`

## Purpose

Test whether the existing finite query-innovation/source-memory checkpoint has
any matched-source advantage under the new gold-answer continuation
log-likelihood diagnostic before spending more MPS time.

## Inputs

- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- device: `cpu`
- dtype: `float32`
- checkpoint:
  `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
  - sha256: `b1f0cfa62c67ffcbdbce631c6cfd80df3240e132e252b0775aef355940a557b8`
- eval file:
  `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/gsm8k_eval_4.jsonl`
  - sha256: `ff0028583cd88ebab02b5f9985d73b436959ef098f8d86a339869d9969e8f5e2`
- ordered example count: `4`
- seed/control salt: `1` for shuffled source

## Commands

Matched:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
  --eval-file results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/gsm8k_eval_4.jsonl \
  --task-type generation \
  --methods rotalign \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --kv-transport k_only \
  --fixed-gate 0.15 \
  --gate-mode fixed \
  --prediction-output results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/matched.jsonl
```

Controls used the same command with:

- `--source-kv-control zero`, output `zero_source.jsonl`
- `--source-prompt-control shuffle_examples --random-salt 1`, output
  `shuffled_source_salt1.jsonl`
- `--innovation-memory-control slots_only`, output `slots_only.jsonl`

`--innovation-memory-control target_only` is unavailable for this checkpoint:

```text
ValueError: --innovation-memory-control target_only requires a target-conditioned query-innovation checkpoint
```

Analysis:

```bash
./venv_arm64/bin/python scripts/analyze_answer_likelihood_controls.py \
  --live results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/matched.jsonl \
  --control zero_source=results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/zero_source.jsonl \
  --control shuffled_source_salt1=results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/shuffled_source_salt1.jsonl \
  --control slots_only=results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/slots_only.jsonl \
  --unavailable-control target_only \
  --min-mean-delta 0.05 \
  --min-best-control-wins 3 \
  --output-json results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json \
  --output-md results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.md
```

## Result

| row | n | finite | correct | mean answer logprob |
|---|---:|---:|---:|---:|
| matched | 4 | 4 | 0 | -7.025400 |
| zero_source | 4 | 4 | 0 | -6.925437 |
| shuffled_source_salt1 | 4 | 4 | 0 | -7.048394 |
| slots_only | 4 | 4 | 0 | -7.025400 |

Paired matched-minus-control deltas:

- zero source: `-0.099963`, wins/losses/ties `1/3/0`
- shuffled source: `+0.022994`, wins/losses/ties `3/1/0`
- slots only: `0.000000`, wins/losses/ties `0/0/4`
- best runnable control: mean delta `-0.115530`, wins/losses/ties `0/4/0`

Pass rule:

- matched must beat every runnable control by at least `0.05` nats/token mean
  answer logprob
- matched must beat the best runnable control on at least `3/4` examples

Decision: fail. Matched source ties slots-only exactly and loses to zero-source
on mean answer likelihood. The current finite query-innovation checkpoint is
killed as a live source-communication row.

## Artifact Hashes

- `answer_likelihood_controls.json`
  - sha256: `4848427ad10a3092169424f63b408afbf95a463c8137a46fdfdf866a155723a3`
- `answer_likelihood_controls.md`
  - sha256: `54872322ab10b95c13f28206b0fb78c17a830d57e301fdb0be7cde3bdbc862db`
- `matched.jsonl`
  - sha256: `24763c66683c4797c0e15374ccd43edfe9737b60eb14b84e6aa027196314d178`
- `matched.jsonl.meta.json`
  - sha256: `ead739c845a5949aed2862405e39dcffa4b4cedac2f962e4d01042e02b200a74`
- `zero_source.jsonl`
  - sha256: `629628885da9aedb53a65f477a119bcf07aaea842dbfff6a608dfb6151933faa`
- `zero_source.jsonl.meta.json`
  - sha256: `34814170673ea797fce90c59773ac41664803668cef11a5da007dfd2500ca1d2`
- `shuffled_source_salt1.jsonl`
  - sha256: `faadfe8ad6e2814741214e11358f9ce825e9467c774c8b2f9c4136d0539249d8`
- `shuffled_source_salt1.jsonl.meta.json`
  - sha256: `f839c12abd62dc9702468a817f9519872575d4708aa69f9f97bda22c908a062e`
- `slots_only.jsonl`
  - sha256: `24763c66683c4797c0e15374ccd43edfe9737b60eb14b84e6aa027196314d178`
- `slots_only.jsonl.meta.json`
  - sha256: `229b25000d2cfa019d63cb9b432272162a2a1398661db048ee963cd68879ad4c`

## Next

Do not scale this checkpoint to GSM32/GSM70 or cross-family. The next
highest-value branch is a target-conditioned query-innovation/source-memory
connector that can run `target_only` and `slots_only` controls from the first
gate, but implementation/testing is blocked until the orphaned MPS calibration
process PID `31103` is cleared.
