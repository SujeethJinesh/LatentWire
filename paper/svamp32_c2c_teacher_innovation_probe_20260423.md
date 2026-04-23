# SVAMP32 C2C Teacher Innovation Probe

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The live evidence has moved from direct source-alone transfer to
a stricter C2C-teacher decision surface: SVAMP32 C2C creates target-complementary
headroom, but no current internal method has yet shown source-specific recovery
under matched/zero/shuffled-source controls.

## Current Story

The strongest live clue is not GSM8K raw source transfer and not SVAMP
source-alone. The strongest surface is SVAMP32 C2C as a teacher/competitor row:
target-alone is `8/32`, C2C is `16/32`, and C2C contributes `10` target-missed
teacher-only wins on the same frozen ordered IDs.

The publishable method story would be: learn a compact, source-conditioned
innovation channel that recovers C2C-style target-complementary wins while
target-only, zero-source, and shuffled-source controls do not.

## Blocking Gap

The remaining submission blocker is source-specificity. Existing artifacts can
be exact-ID compared against the SVAMP32 C2C teacher row, but they do not yet
provide a clean positive result:

- Process repair recovers `3/10` C2C-only IDs, but `target_self_repair` also
  recovers all `3/10`, so this is target-side repair, not communication.
- Dynalign salts recover only `0/10`, `1/10`, and `2/10` C2C-only IDs. Salt 1
  has `1` teacher-only hit not recovered by the provided target-side controls;
  salt 2 has `2` hits with one control overlap. This is a weak hint, not a
  claim, because the old dynalign rows do not include zero-source or shuffled-
  source controls on the same SVAMP32 frozen IDs.

## Probe Added

Added `scripts/analyze_c2c_teacher_innovation.py`, which measures teacher-only
recovery on frozen IDs:

- target row defines baseline correct IDs
- teacher row defines `teacher_correct - target_correct`
- source/control/candidate rows are aligned to the exact target ID order
- candidate wins are split into teacher-only recovery, non-teacher wins, and
  losses vs target
- controls report how much of each candidate's teacher-only recovery they
  retain
- markdown and JSON artifacts record row summaries, control overlap, and
  per-example provenance

Focused tests were added in `tests/test_analyze_c2c_teacher_innovation.py`.

## Command

```bash
./venv_arm64/bin/python scripts/analyze_c2c_teacher_innovation.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source t2t=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --control target_self_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=target_self_repair \
  --control selected_route_no_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=selected_route_no_repair \
  --candidate process_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=process_repair_selected_route \
  --candidate dynalign_salt0=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt0_telemetry.jsonl,method=rotalign_kv \
  --candidate dynalign_salt1=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt1_telemetry.jsonl,method=rotalign_kv \
  --candidate dynalign_salt2=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt2_telemetry.jsonl,method=rotalign_kv \
  --min-teacher-only 5 \
  --require-controls \
  --output-json results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json \
  --output-md results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.md
```

## Evidence

Teacher-only IDs: `10`.

| Row | Correct | Wins vs target | C2C-only recovered | Losses vs target |
|---|---:|---:|---:|---:|
| source-alone | 5/32 | 3 | 1 | 6 |
| text-to-text | 2/32 | 1 | 0 | 7 |
| target self-repair control | 14/32 | 6 | 3 | 0 |
| selected-route no-repair control | 10/32 | 2 | 1 | 0 |
| process repair | 15/32 | 7 | 3 | 0 |
| dynalign salt 0 | 4/32 | 1 | 0 | 5 |
| dynalign salt 1 | 9/32 | 3 | 1 | 2 |
| dynalign salt 2 | 8/32 | 4 | 2 | 4 |

Artifact readouts:

- `results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.md`
- `results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json`

## Interpretation

Alive:

- C2C-teacher innovation transfer remains alive as a method branch because the
  teacher surface has `10/32` target-complementary wins and old dynalign has a
  tiny amount of non-target-control-overlapped C2C-only recovery.

Saturated:

- Process repair is saturated for this paper story unless it can be made
  source-conditioned. Its current C2C-only recovery is fully explained by
  target-side repair.

Blocked:

- Dynalign cannot be promoted because the old salt rows lack matched
  zero-source and shuffled-source controls on the same SVAMP32 gate.

Weakened:

- Direct source-alone SVAMP transfer is weakened by only `3/32` source-only
  wins and `6` losses vs target.

Promoted:

- The next method should train or run a source-control-contrastive innovation
  connector against C2C-only IDs, with controls built into the objective and
  evaluation rather than added later.

## Next Exact Gate

Run a controlled C2C-teacher innovation gate on the frozen SVAMP32 IDs:

- matched-source connector row
- zero-source connector row
- deterministic shuffled-source connector row
- target-only repair/control row
- byte, latency, and accept-rate telemetry

Promotion threshold:

- recover at least `4/10` C2C-only teacher wins
- overall at least `11/32`
- lose no more than `1` target-correct ID
- zero/shuffle controls recover at most `1` of the same C2C-only wins

If this fails, keep C2C as teacher/headroom evidence only and pivot to a
learned Q-former/Perceiver-style innovation bottleneck with explicit
matched-vs-control contrastive loss.
