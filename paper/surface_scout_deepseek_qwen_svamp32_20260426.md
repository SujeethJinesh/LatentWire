# DeepSeek -> Qwen SVAMP32 Surface Scout

- date: `2026-04-26`
- start commit: `533640d95c52d4a371b2878b54b22ff91b566f00`
- scale-up rung: source-surface smoke
- source: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- target: `Qwen/Qwen3-0.6B`
- eval IDs: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized eval: `results/surface_scout_deepseek_qwen_svamp32_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- materialized eval sha256: `90093683765363baf86c5402d66d7b4641ef22e4957c0de80ff6768da3968b64`
- device: `mps`
- max new tokens: `64`

## Why This Scout

The historical MD/result audit demotes raw RotAlign and raw GSM dynalign as
paper methods. The best older real row, `dynalign_module_replace_residrank16`,
has source-dependent seed-0 lift but fails seed stability. The next branch
therefore needs source-surface discovery before another connector is trained.

This scout asks whether a locally available stronger reasoning source creates
a useful frozen SVAMP32 decision surface against the same Qwen3 target.

## Command

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_deepseek_qwen_svamp32_20260426 \
  --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

The first invocation ran `source`, `target`, and `t2t`; the second invocation
added `c2c` while reusing validated existing predictions.

## Results

| Row | Correct | Accuracy | ID parity | Numeric coverage |
|---|---:|---:|---:|---:|
| target-alone | 8/32 | 0.250 | true | 32/32 |
| source-alone | 5/32 | 0.156 | true | 32/32 |
| text relay | 5/32 | 0.156 | true | 32/32 |
| C2C | failed | - | - | - |

Pairwise against target:

| Row | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| source-alone | 1 | 4 | 4 | 9/32 |
| text relay | 2 | 5 | 3 | 10/32 |

The C2C row failed before generation because no published C2C artifact is
registered for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -> Qwen/Qwen3-0.6B`.

## Decision

This is not a good next method surface.

- target-alone is stronger than both source-alone and text relay
- text relay adds only two target-missed IDs
- the target/text oracle is only `10/32`
- the registered C2C baseline is unavailable for this pair

The pair can stay as a future target for generic cross-model interfaces, but it
should not receive connector or source-control compute in the current ICLR loop.

## Artifact Hashes

- `results/surface_scout_deepseek_qwen_svamp32_20260426/manifest.json`
  - sha256: `015c0033bbaa7374899bba9297eb142a48630af99a5d0a10ebdf65885295d59a`
- `results/surface_scout_deepseek_qwen_svamp32_20260426/manifest.md`
  - sha256: `433088e6dcaf1f642d06e6456736a646ec593c641c373548f49b222dbefb7b0a`
- `results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl`
  - sha256: `6ca901e94a10275967e6451f3e727f6791f698747ed53ba72a25b21cc3ab445d`
- `results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl`
  - sha256: `202336cb3f516afff6633e39f3ecb069a39456f1ff894b47373f93e819e77304`
- `results/surface_scout_deepseek_qwen_svamp32_20260426/text_to_text.jsonl`
  - sha256: `2a84e04fb897cb637589885911e2c79f74dcf0a02ef77c9a2bb4c15a43c99d95`
- `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/c2c.log`
  - sha256: `c82af7c0faf58e4d1316b10cd245f315aca6ecc46af3327ccd0cde323d81b30e`

## Next Gate

Do not train a connector on this pair. Continue source-surface discovery using
only pairs where target/text/C2C expose enough target-complementary headroom, or
return to a same-family Qwen sidecar gate with full source-destroying controls.
