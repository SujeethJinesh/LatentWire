# SVAMP70 Perceiver Answer-Teacher Contrastive Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a new source-necessary objective or method branch plus
  larger-slice, seed-repeat, source-control, and cross-family gates
- current story: moving from SVAMP32 to the headroom-richer SVAMP70 C2C surface
  did not rescue the Perceiver answer-teacher connector
- blocker: target-only, slots-only, and shuffled-source controls still explain
  every positive clean margin

## Gate

After the SVAMP32 Perceiver answer-teacher checkpoint failed the teacher-forced
pre-gate, test the same branch on the stronger SVAMP70 C2C-vs-process-repair
surface:

- target rows: `21/70`
- C2C source rows: `31/70`
- process-repair baseline rows: `38/70`
- C2C source-only IDs: `18`
- clean C2C source-only IDs after excluding process-repair: `10`

Run only the teacher-forced clean-ID diagnostic first. Do not score all
preservation rows or run generation unless matched source beats every control
on clean IDs.

## Artifacts

- results manifest:
  - `results/svamp70_perceiver_answer_teacher_contrastive_20260426/manifest.md`
- target set:
  - `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json`
  - sha256: `e2f3a4da9848519f009260cb681f378dc2767d1f3ba0bc67ce3bff94747287c5`
- checkpoint:
  - `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - sha256: `a7221d6d0ee81b99573bf1893b66570ec682f22faee1ffcc6bf7e9fc1f36df6a`
  - not tracked, size `1.8G`
- calibration log:
  - `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
  - sha256: `5701b20b25d2dffa1bacbfd5ff65e9b6cbbbd8f8bb591754636c3ab6bffba73f`
- teacher-forced gate:
  - `results/svamp70_perceiver_answer_teacher_contrastive_20260426/teacher_forced_gate015_clean_only.json`
  - sha256: `d57eae6555fbce396e077d63eebcb99b75cfd7bd50bf69363dec488e92b99960`

## Evidence

Calibration:

- prompts: `70`
- dynamic mixture samples: `2948`
- answer-teacher injected prompts: `10`
- answer-teacher injected samples: `446`
- average K alignment cosine: `0.949`
- average V alignment cosine: `0.718`

Teacher-forced clean-only gate at fixed gate `0.15`:

- status: `no_teacher_forced_source_signal`
- clean IDs scored: `10`
- matched-positive clean IDs: `4/10`
- matched-only clean IDs: `0/10`
- control-leak clean IDs: `4/10`
- mean matched margin: `-3.4050`
- mean best-control margin: `-2.6892`
- mean matched-control delta: `-0.7158`

The four positive matched margins were all explained by controls:

- `575d7e83d84c1e67`: slots-only beats matched
- `9325c4efa96bdbca`: shuffled-source beats matched
- `aee922049c757331`: target-only beats matched
- `e3ab8666238a289e`: target-only beats matched

## Decision

Kill the current Perceiver answer-teacher plus contrastive delta-memory branch.
It fails for the same reason on SVAMP32 and SVAMP70: apparent clean-ID signal
is not source-necessary.

Do not run generation, seeds, or larger slices for this checkpoint.

## Next Gate

The next branch should change the objective, not just the surface:

- add an explicit target-only/slots-only penalty before answer-teacher
  supervision, or
- train against token/layer-level C2C residual behavior with matched-vs-control
  separation, or
- pivot away from learned-query delta memory to a source-only sidecar/router
  that cannot access target-only memory during source-signal formation.

Any next run must first pass a teacher-forced matched-only clean-ID gate before
generation.
