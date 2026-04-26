# Qwen2.5-Math -> Qwen3 SVAMP16 Surface Scout

- date: `2026-04-26`
- start commit: `e01c5403`
- scale-up rung: source-surface smoke
- source: `Qwen/Qwen2.5-Math-1.5B`
- target: `Qwen/Qwen3-0.6B`
- eval IDs: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized 16-ID eval sha256: `db10a237015409dd8056af0925669ae083557365e5059d150b9150b757d25980`
- device: `mps`
- max new tokens: `64`

## Why This Scout

The previous DeepSeek-R1-Distill-Qwen scout was weak and lacked a registered
C2C comparator. This pair is higher value because the repo has a published C2C
artifact registered for `Qwen/Qwen2.5-Math-1.5B -> Qwen/Qwen3-0.6B`, and the
source is math-specialized.

## Tooling Fix

The first no-chat run exposed a materializer validation bug. When
`--no-use-chat-template` is selected, `latent_bridge/evaluate.py` records
`source_enable_thinking=auto` and `target_enable_thinking=auto`; the materializer
incorrectly expected explicit `false`. This caused generated source/target/text
rows to be rejected even though the predictions and sidecars were otherwise
valid.

Fixed in:

- `scripts/materialize_generation_baselines.py`
- `tests/test_materialize_generation_baselines.py`

Focused test:

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_generation_baselines.py -q
```

Result: `5 passed`.

## No-Chat Probe

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_qwen3_svamp16_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 16 \
  --device mps \
  --max-new-tokens 64 \
  --no-use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result after validation fix:

| Row | Correct | Accuracy |
|---|---:|---:|
| target-alone | 0/16 | 0.000 |
| source-alone | 1/16 | 0.062 |
| text relay | 5/16 | 0.312 |
| C2C | 5/16 | 0.312 |

Decision: do not use this no-chat surface for claims. The target floor is a
prompt-template artifact risk.

## Chat-Template Probe

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 16 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

| Row | Correct | Accuracy | ID parity | Numeric coverage |
|---|---:|---:|---:|---:|
| target-alone | 2/16 | 0.125 | true | 16/16 |
| source-alone | 4/16 | 0.250 | true | 12/16 |
| text relay | 4/16 | 0.250 | true | 16/16 |
| C2C | 5/16 | 0.312 | true | 16/16 |

Pairwise against target:

| Row | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| C2C | 4 | 1 | 1 | 6/16 |
| source-alone | 3 | 1 | 1 | 5/16 |
| text relay | 3 | 1 | 1 | 5/16 |

## Decision

The chat-template version is live enough to scale one rung to SVAMP32 baselines.
It is not a method result, but it gives a fairer surface than the no-chat probe:

- C2C is available and beats target/text on the 16-ID smoke
- C2C adds four target-missed IDs
- text relay is close, so any method must beat text and preserve target-correct
  examples
- source numeric coverage is only `12/16`, so source extraction quality must be
  monitored before using source-final sidecars

## Artifact Hashes

- no-chat manifest:
  - `results/surface_scout_qwen25math_qwen3_svamp16_20260426/manifest.json`
  - sha256: `bda00aadd8b2ac109a5fb522fcff409045acf64ff644e7c724f6470e3ede0bcc`
- chat manifest:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/manifest.json`
  - sha256: `834a60d9a2a4762f26ac4110e3e0503f73c9235256f3607fa4510b241a727060`
- chat manifest markdown:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/manifest.md`
  - sha256: `d2ec23fb6aa8323a5eba7b33516950b5baf3d6f15277510ba0d89c80ef160d2e`
- chat C2C predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/c2c_generate.jsonl`
  - sha256: `0d00f1b1a6cbb569384de21a3ded03eb9e0edd1cd8e39af5281c64cb3afb410b`
- chat source predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/source_alone.jsonl`
  - sha256: `5089218e785da201fa5c5e9245039f3af6f18107ed795381ee934261b39f364b`
- chat target predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/target_alone.jsonl`
  - sha256: `528216fe004c5691eab530443c265a8f554e5bd4868632670cac46336747427e`
- chat text predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/text_to_text.jsonl`
  - sha256: `9df5856ea3faf362d437fa3bb4c804b3c75c105b4a10ac24709bce06516f0f9a`

## Next Gate

Run the same chat-template materialization on the full frozen SVAMP32 exact-ID
slice. Promote only if C2C remains above target/text with target-complementary
headroom; otherwise prune this pair before any connector training.
