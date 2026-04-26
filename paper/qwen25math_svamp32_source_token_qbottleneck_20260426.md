# Qwen2.5-Math SVAMP32 Source-Token Query-Bottleneck Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-derived positive method plus
  medium, seed-repeat, uncertainty, source-control, systems, and cross-family
  gates
- current story: the Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom surface is
  still the best strict same-family decision surface, but source-derived
  readouts are not recovering its clean C2C-only IDs
- blocker: all-layer source-token query bottlenecks fail to predict useful C2C
  residue signatures

## Gate

This run tested the non-duplicative version of the older source-token
query-bottleneck probe:

- source model: `Qwen/Qwen2.5-Math-1.5B`
- target/teacher surface: `Qwen/Qwen3-0.6B` SVAMP32 chat C2C headroom
- target set:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
- feature layers: `all`
- query count: `4`
- hidden dim: `16`
- outer folds: `8`
- seed: `0`
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  target-only, and slots-only

Promotion rule:

- matched correct at least `10/32`
- at least `2/6` clean source-necessary recoveries
- destructive controls recover `0/6` clean IDs
- numeric coverage and ordered ID parity preserved

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 8/32 | 0/6 | 0 |
| zero-source | 8/32 | 0/6 | 0 |
| shuffled-source | 7/32 | 0/6 | 0 |
| label-shuffled | 7/32 | 0/6 | 0 |
| same-norm-noise | 8/32 | 0/6 | 0 |
| target-only | 8/32 | 0/6 | 0 |
| slots-only | 7/32 | 1/6 | 0 |

Other checks:

- teacher numeric coverage: `32/32`
- candidate numeric coverage: target `32/32`, C2C `32/32`
- candidate-pool clean gold coverage: `6/6`
- clean source-necessary IDs: `0`
- control clean union: `de1bf4d142544e5b`

## Decision

Kill source-token query-bottleneck residue prediction on the current
Qwen2.5-Math SVAMP32 C2C-headroom surface. This was not just the older
Qwen2.5-0.5B result: it used the stronger Math source and the newer clean
C2C-headroom target set, and still recovered `0/6` clean IDs.

Together with the sparse-anchor sidecar smoke, this weakens shallow
source-readout interfaces on this surface. The next branch needs a materially
different signal path, not more all-layer token readout tuning.

## Next Gate

Stop tuning:

- random sparse-anchor projection seed/top-k/byte-budget variants
- all-layer source-token query bottlenecks
- pooled source-hidden ridge/query readouts

Next highest-value branches:

1. fold-local token/span sparse dictionaries with boundary-only and
   same-norm-noise controls; or
2. an output-aware dynalign-derived method with a target-safe selector/repair
   objective, evaluated under strict source controls; or
3. a real SAE/shared-code adapter only if it is not the already-dead first-pass
   SAE proxy and is tested on this exact clean headroom surface.

Promotion still requires `>=2/6` clean source-necessary IDs and zero clean
destructive-control recovery before any scale-up.

## Artifacts

- JSON:
  `results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.json`
  - sha256:
    `abf23bb105ee05d98717d731414de15d4543419b3e96ffc571a54c54983c83d0`
- Markdown:
  `results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.md`
  - sha256:
    `5ec913320d7f3acf8c75202286910db5b8f7fba8e57130193c88ad5e490975f7`
- Log:
  `.debug/qwen25math_svamp32_source_token_qbottleneck_20260426/logs/probe.log`
  - sha256:
    `bd5f031b76e9c838437b2c968e8409302c05b35f57d141759872c0f5015ab7bf`

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_learned_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --query-count 4 \
  --hidden-dim 16 \
  --epochs 80 \
  --outer-folds 8 \
  --seed 0 \
  --shuffle-offset 1 \
  --min-correct 10 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 26 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers all \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --date 2026-04-26 \
  --output-json results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.json \
  --output-md results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.md
```
