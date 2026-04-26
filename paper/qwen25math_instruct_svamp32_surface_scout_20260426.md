# Qwen2.5-Math-Instruct SVAMP32 Source-Surface Scout - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-derived positive method plus
  medium, seed-repeat, uncertainty, source-control, systems, and cross-family
  gates
- current story: C2C still exposes the strongest same-family headroom, but
  source-only signal remains too sparse for deployable sidecars on repeated
  scouts
- blocker: stronger source prompting/modeling did not produce enough clean
  source-only IDs to justify C2C or sidecar spend

## Gate

This cycle tested whether the cached `Qwen/Qwen2.5-Math-1.5B-Instruct` source
creates a better strict-small SVAMP32 source surface against
`Qwen/Qwen3-0.6B`.

Promotion rule before C2C or sidecar spend:

- source-only over target: at least `4/32`
- clean source-only after text-relay exclusion: at least `2/32`
- exact ordered ID parity and numeric coverage intact

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

## Evidence

| Row | Correct | Numeric coverage | Exact ID parity |
|---|---:|---:|---:|
| source-alone | 3/32 | 32/32 | true |
| target-alone | 8/32 | 32/32 | true |
| text relay | 4/32 | 32/32 | true |

Pairwise source versus target:

- source-only over target: `2`
- target-only over source: `7`
- target/source oracle: `10/32`
- clean source-only after text exclusion: `2`

The two clean source-only IDs are:

- `14bfbfc94f2c2e7b`
- `b1200c32546a34a5`

## Decision

Reject `Qwen2.5-Math-1.5B-Instruct -> Qwen3-0.6B` as the next SVAMP32
source-surface branch. It is worse than the non-instruct Math source on this
same exact-ID slice and fails the source-only threshold before C2C, connector,
or sidecar spend.

This weakens "stronger source prompting/model variant fixes the surface" as
the next branch. It does not kill the original non-instruct Qwen2.5-Math
SVAMP32/SVAMP70 surface, but it argues against spending compute on adjacent
prompt/source variants without a new mechanism.

## Next Gate

Do not rerun the all-layer source-token query-bottleneck recommendation on the
current SVAMP32 syndrome surface; the 2026-04-26 memo already killed that
family with `0/6` clean IDs.

The next highest-value branch is a materially different source interface:
implement or run the smallest real-model smoke for a sequence-aligned sparse /
anchor sidecar inspired by the quotient/GPA toy results, or another
rate-capped source-derived interface with explicit zero-source, shuffled-source,
target-only, and slots-only controls.

## Artifacts

- generation manifest:
  `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/manifest.json`
  - sha256:
    `5a032574d92589f092ea6fc0270adfbfbaa3faa7a3cd90a59d4957eeeb1dc297`
- generation readout:
  `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/manifest.md`
  - sha256:
    `0d1e4e3b1d61a152cb7df0df7f8ad7c763afd3fbcfda7a0af299cebd80a71c44`
- source target set:
  `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.json`
  - sha256:
    `d2249cfff9c498c19f6374cb669a14bd7ea066640cc5a18aed01eeea21181312`
- source target set readout:
  `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.md`
  - sha256:
    `6d8e0aa576bcc7ee92edefb91fdab15127007c26392dc0471a97e97fc2eaaec8`
- predictions:
  - source-alone:
    `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone.jsonl`
    - sha256:
      `dc610e04de1bb74bdb577b3ce39f9f82f64fe07bac58905c55503a69db0c3955`
  - target-alone:
    `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/target_alone.jsonl`
    - sha256:
      `202336cb3f516afff6633e39f3ecb069a39456f1ff894b47373f93e819e77304`
  - text relay:
    `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/text_to_text.jsonl`
    - sha256:
      `6b5a27770ac6f312d8e9c937bc838f933ab6daca973046e39e1248f3e1565580`
