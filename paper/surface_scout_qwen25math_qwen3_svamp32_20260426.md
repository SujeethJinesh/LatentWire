# Qwen2.5-Math -> Qwen3 SVAMP32 Surface Confirmation

- date: `2026-04-26`
- start commit: `cd4f156aba9f0e43b9ca4a9f7e29b256edf73bf9`
- scale-up rung: strict small surface confirmation
- source: `Qwen/Qwen2.5-Math-1.5B`
- target: `Qwen/Qwen3-0.6B`
- eval IDs: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- materialized eval sha256: `90093683765363baf86c5402d66d7b4641ef22e4957c0de80ff6768da3968b64`
- device: `mps`
- max new tokens: `64`
- prompting: chat templates enabled, thinking disabled

## Command

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

## Results

| Row | Correct | Accuracy | ID parity | Numeric coverage |
|---|---:|---:|---:|---:|
| target-alone | 8/32 | 0.250 | true | 32/32 |
| source-alone | 6/32 | 0.188 | true | 26/32 |
| text relay | 8/32 | 0.250 | true | 32/32 |
| C2C | 15/32 | 0.469 | true | 32/32 |

Pairwise against target:

| Row | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| C2C | 9 | 2 | 6 | 17/32 |
| source-alone | 5 | 7 | 1 | 13/32 |
| text relay | 3 | 3 | 5 | 11/32 |

## Decision

Promote this as the next strict-small decision surface, not as a method result.
C2C creates substantial headroom over both target-alone and text relay, while
the target/text oracle remains much lower than the target/C2C oracle. This is
exactly the kind of surface needed for the next source-derived method gate.

Constraints for the next method:

- must beat target-alone and text relay
- must explain a useful subset of the `9` C2C-only target-missed IDs
- must preserve the `2` target-only IDs against C2C
- must report source-destroying controls, target-only/slots-only controls, and
  exact-ID parity
- source-final sidecars need care because source-alone numeric coverage is only
  `26/32`

## Artifact Hashes

- manifest JSON:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/manifest.json`
  - sha256: `27395a3e79ac5b02243e2b814a8167d65a838ec1e8faf21feaa06e9a22031b3d`
- manifest markdown:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/manifest.md`
  - sha256: `291dc38614f4431d8cb37d7c53bc13ff95e84de826407c821fb0520437f756af`
- C2C predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
  - sha256: `e7389fffe0dc73e1bf583106d130c098e9679ca7dfa35bc693c47b54a509542e`
- source predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl`
  - sha256: `d3781fbe8d6b9f5bd12db298bc8d22ba194e0ce97e9eb4e8047a2cc8bac31e68`
- target predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl`
  - sha256: `202336cb3f516afff6633e39f3ecb069a39456f1ff894b47373f93e819e77304`
- text relay predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl`
  - sha256: `1c019d45aa5234baeb8d5bcec073b9a221fb86f8ef1060486487bbe55810c6ea`

## Next Gate

Build a C2C-headroom target set for this pair and run the cheapest deployable
source-derived sidecar that can be evaluated with zero-source, shuffled-source,
target-only, and slots-only controls. Do not scale to larger slices until a
method recovers source-necessary C2C-only IDs on this surface.
