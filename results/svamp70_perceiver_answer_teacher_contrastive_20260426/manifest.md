# SVAMP70 Perceiver Answer-Teacher Contrastive Manifest

- date: `2026-04-26`
- scale-up rung: medium/headroom surface pre-gate
- status: `fails_pre_generation_gate`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `data/svamp_eval_70.jsonl`
- target set: `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json`

## Checkpoint

- path: `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- size: `1.8G`
- sha256: `a7221d6d0ee81b99573bf1893b66570ec682f22faee1ffcc6bf7e9fc1f36df6a`
- tracked: no, checkpoint is too large for git
- calibration log: `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
- calibration log sha256:
  `5701b20b25d2dffa1bacbfd5ff65e9b6cbbbd8f8bb591754636c3ab6bffba73f`

## Calibration

Key settings:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- connector mode: `perceiver_queries`
- rank: `16`
- bridge bank size: `16`
- answer-teacher weight: `0.8`
- process-repair preserve weight: `8`
- source-control weight: `0.5`
- source-control mode: `zero_and_shuffle`
- conditional delta memory: enabled
- value loss weight: `0.0`

Calibration readout:

- prompts: `70`
- dynamic mixture samples: `2948`
- answer-teacher injected prompts: `10`
- answer-teacher injected samples: `446`
- average K alignment cosine: `0.949`
- average V alignment cosine: `0.718`

## Teacher-Forced Gate

- artifact: `teacher_forced_gate015_clean_only.json`
- sha256: `d57eae6555fbce396e077d63eebcb99b75cfd7bd50bf69363dec488e92b99960`
- markdown: `teacher_forced_gate015_clean_only.md`
- markdown sha256:
  `1ab97d6b31c2993af1a6830875940c5d6c4d1b8bccb94e44d1da760602c2ceca`
- log: `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/logs/teacher_forced_gate015_clean_only.log`
- log sha256:
  `af12fb6374851daf550f26ce6fe63d0d6d4fcc072a40d962c98abe323892be3c`

Readout:

- status: `no_teacher_forced_source_signal`
- clean IDs scored: `10`
- matched-positive clean IDs: `4/10`
- matched-only clean IDs: `0/10`
- control-leak clean IDs: `4/10`
- mean matched margin: `-3.4050`
- mean best-control margin: `-2.6892`
- mean matched-control delta: `-0.7158`

## Decision

Do not run generation for this checkpoint. Even on the headroom-richer SVAMP70
C2C-vs-process-repair surface, the Perceiver answer-teacher connector does not
produce matched-only clean margins. Target-only, slots-only, or shuffled-source
controls explain every positive clean margin.

This kills the current Perceiver answer-teacher plus contrastive delta-memory
branch until a new objective explicitly penalizes target-only recovery before
answer-teacher supervision dominates.
