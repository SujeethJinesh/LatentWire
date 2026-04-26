# Qwen2.5-Math -> Qwen3 Source-Contrastive Sidecar

- date: `2026-04-26`
- status: strict-small gate cleared
- readiness: not ICLR-ready; needs medium/seed confirmation and fair systems
  accounting
- commit at run time: `b3c589361812037390e1b2d5829db2537257572e`

## Start Status

- current paper story: C2C exposes strong Qwen2.5-Math -> Qwen3 headroom, but
  C2C-only target sets are hostile to deployable source-derived methods.
- exact blocker: find a source-derived decision surface where the source can
  add clean wins without target/cache leakage.
- live branch: source-contrastive sidecar stack.
- scale-up rung: strict small gate.

## Target Set

The source-contrastive target set asks whether source-alone has useful target
complementary answers after excluding target and text relay.

- target correct: `8/32`
- source correct: `6/32`
- text relay correct: `8/32`
- source-only IDs over target: `5`
- clean source-only IDs after text exclusion: `4`
- target-or-source oracle: `13/32`
- source numeric coverage: `26/32`

Clean source-only IDs:

- `14bfbfc94f2c2e7b`
- `2de1549556000830`
- `41cce6c6e6bb0058`
- `4d780f825bb8541c`

## Method

The cleared row is a stack:

1. Use target-alone as fallback.
2. Use text-relay agreement as a preservation guard: if target and text-relay
   numeric predictions agree, keep target.
3. Otherwise use a compact source numeric residue sidecar to select a candidate
   from the target/source/text candidate pool.

This is source-derived because the switching signal is the source numeric
residue. It is not a pure lower-byte replacement for text relay because text
relay is used as a guard; future systems accounting must report text-relay
tokens/latency plus the 1-byte sidecar.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --baseline t2t=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl,method=text_to_text \
  --min-source-only 2 \
  --output-json .debug/qwen25math_svamp32_source_contrastive_20260426/source_contrastive_target_set.json \
  --output-md .debug/qwen25math_svamp32_source_contrastive_20260426/source_contrastive_target_set.md

PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json .debug/qwen25math_svamp32_source_contrastive_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --preserve-on-agreement-label t2t \
  --min-correct 9 \
  --min-target-self 0 \
  --min-clean-source-necessary 2 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 26 \
  --output-json .debug/qwen25math_svamp32_source_contrastive_20260426/source_only_sidecar_router_t2t_guard.json \
  --output-md .debug/qwen25math_svamp32_source_contrastive_20260426/source_only_sidecar_router_t2t_guard.md
```

## Result

| Moduli | Bytes | Status | Matched | Clean Necessary | Control Clean Union |
|---|---:|---|---:|---:|---:|
| 2,3 | 1 | fails | 9/32 | 1/4 | 0 |
| 2,3,5 | 1 | clears | 10/32 | 2/4 | 0 |
| 2,3,5,7 | 1 | clears | 11/32 | 3/4 | 0 |
| 97 | 1 | clears | 11/32 | 3/4 | 0 |

Best row:

- matched: `11/32`
- target-only: `8/32`
- text relay: `8/32`
- source-alone: `6/32`
- clean source-necessary IDs: `3/4`
- control clean union: `0/4`
- zero-source: `8/32`
- shuffled-source: `8/32`
- label-shuffle: `8/32`
- same-norm noise: `8/32`
- slots-only: `6/32`

## Decision

Promote this branch one rung to medium source-surface confirmation. Do not call
it ICLR-ready yet:

- single frozen 32-example slice only
- no seed/stability evidence
- no paired uncertainty
- no C2C comparison on the same source-contrastive decision surface
- text-relay guard means systems claims must account for generated text tokens

Next exact gate:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math_qwen3_svamp70_source_surface_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Then build the SVAMP70 source-contrastive set and rerun the guarded sidecar
with full controls.

## Artifacts

- target set JSON:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json`
  - sha256: `088f0e1651f95ea04a89ec0931276a943ff104a355fe69f434182f68e778ea96`
- target set markdown:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.md`
  - sha256: `22cea2201a3b60491c6e38b30e3173582d14c51f583730c92ee3092a67568919`
- guarded sidecar JSON:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `c5434aeead9e55f5494ca583533fe863f36ee719e8a5bb75ae6fdb2f6f373306`
- guarded sidecar markdown:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_only_sidecar_router_t2t_guard.md`
  - sha256: `8cb94c6b0ba5d07c428cebcbf54e2b6a0b9b21e2475a9a9413601acdb46e9831`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py
```

