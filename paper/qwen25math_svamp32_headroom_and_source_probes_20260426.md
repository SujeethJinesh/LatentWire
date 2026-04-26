# Qwen2.5-Math -> Qwen3 SVAMP32 Headroom And Source Probes

- date: `2026-04-26`
- status: strict-small surface alive, first deployable source probes fail
- readiness: not ICLR-ready

## Start Status

- current paper story: Qwen2.5-Math -> Qwen3 with chat templates exposes the
  strongest current same-family decision surface: target `8/32`, source `6/32`,
  text relay `8/32`, and C2C `15/32`.
- exact blocker: recover C2C-only target-complementary IDs with a deployable
  source-derived message rather than C2C answer/cache leakage.
- live branch: C2C-headroom surface discovery plus source-derived sidecar
  probes.
- scale-up rung: strict small gate.

## Headroom Target Set

The C2C innovation readout found `9` C2C-only wins over target-alone. Source
alone explains `3` of those IDs and text relay explains none, leaving `6` clean
C2C-headroom targets:

- `1d50b408c8f5cd2c`
- `3e8a5691f5443495`
- `47464cc0b064f172`
- `575d7e83d84c1e67`
- `6e9745b37ab6fc45`
- `de1bf4d142544e5b`

Target-only over C2C is `2` IDs, so any promoted method must preserve target
floor behavior while recovering at least two clean C2C-only IDs under controls.

Commands:

```bash
./venv_arm64/bin/python scripts/analyze_c2c_teacher_innovation.py \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --source source=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,method=source_alone \
  --source t2t=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl,method=text_to_text \
  --min-teacher-only 5 \
  --require-exact-artifacts \
  --output-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_teacher_innovation.json \
  --output-md results/qwen25math_svamp32_c2c_headroom_20260426/c2c_teacher_innovation.md

./venv_arm64/bin/python scripts/build_c2c_headroom_target_set.py \
  --probe-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_teacher_innovation.json \
  --output-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json \
  --output-md results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.md \
  --source-label source \
  --source-label t2t \
  --min-teacher-only 5 \
  --min-clean-teacher-only 2
```

## Source-Only Numeric Sidecar

The source-only numeric sidecar/router fails the gate. Best rows reach matched
`8/32`, but recover `0/6` clean source-necessary IDs and controls explain up to
`3` clean C2C targets. Source numeric coverage is only `26/32`.

Decision: kill raw source-generated numeric residue sidecars on this Qwen-Math
surface.

## Source-Hidden Ridge Probes

Last-layer and all-layer ridge probes over source hidden summaries also fail:

| Feature Set | Matched | Clean Source-Necessary | Control Clean Union |
|---|---:|---:|---:|
| last layer | 8/32 | 0/6 | 1 |
| all layers | 8/32 | 0/6 | 2 |

Decision: kill summary-level source-hidden residue readout on this surface
unless a materially new token/layer-local objective changes the hypothesis.

## Hypothesis Update

Strengthened:

- Qwen2.5-Math -> Qwen3 has real C2C target-complementary headroom.
- The clean decision set is now explicit and reusable.

Weakened or killed:

- source-only generated numeric residue sidecars
- source-hidden summary ridge residue readout
- further summary-feature tuning without a new mechanism

Next exact gate:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --shuffle-offset 1 \
  --min-correct 9 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 26 \
  --device mps \
  --max-new-tokens 1 \
  --residual-projection-dim 16 \
  --output-json results/qwen25math_svamp32_c2c_mechanism_probe_20260426/prefill_projection16_probe.json \
  --output-md results/qwen25math_svamp32_c2c_mechanism_probe_20260426/prefill_projection16_probe.md
```

This is not deployable evidence by itself. It is a diagnostic for whether the
C2C prefill mechanism carries residue information that a later source-derived
objective could imitate.

