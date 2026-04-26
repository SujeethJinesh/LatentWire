# SVAMP32 Anti-Memory Perceiver Manifest

- date: `2026-04-26`
- scale-up rung: strict small teacher-forced pre-gate
- status: `fails_pre_generation_gate`
- code commit: `8b3d7924`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target set: `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`

## Checkpoint

- path: `.debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt`
- sha256: `6a3932946c6fcb580a1b136e1e5d710e555884a73c94def8ef1485fc613692ad`
- tracked: no, checkpoint is too large for git
- calibration log: `.debug/svamp32_anti_memory_perceiver_20260426/logs/calibrate_w080_ctrl050_am050_seed1.log`
- calibration log sha256: `4e1d345a5c3c0ac671c3e244c446b49708f92302d88b8c6ea82b2d2928080318`

## Calibration

Key settings:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- connector mode: `perceiver_queries`
- rank: `16`
- bridge bank size: `16`
- answer-teacher weight: `0.8`
- target-self preserve weight: `16`
- source-control weight: `0.5`
- source-control mode: `zero_and_shuffle`
- anti-memory-control weight: `0.5`
- anti-memory-control mode: `target_and_slots`
- anti-memory contrastive margin: `0.001`
- conditional delta memory: enabled
- value loss weight: `0.0`

Calibration readout:

- prompts: `32`
- dynamic mixture samples: `1411`
- answer-teacher injected prompts: `6`
- answer-teacher injected samples: `277`
- average K alignment cosine: `0.951`
- average V alignment cosine: `0.734`

## Teacher-Forced Gates

| Gate | Status | Matched Positive Clean | Matched-Only Clean | Control Leak Clean | Mean Matched-Control Delta | JSON SHA256 |
|---:|---|---:|---:|---:|---:|---|
| `0.125` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.8898 | `0c45517529bc18ed73953d239543c40f70dfffe79582ea472ca0b3767496ff0d` |
| `0.150` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.8921 | `09b0a4ff220a5be7760eeebcf82ee53f9c9c8baf213aa66f473ac39dc6754f94` |
| `0.200` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -0.8660 | `53b0f3d718455ff26d711ecc51bd603032e16d24258eb12d5ba90947a9bfa1c2` |

Markdown hashes:

- `teacher_forced_gate0125.md`: `a5e1c02731e61b593c1c3f75bcef128a98d135490006ab1ac7639defb6ba2a36`
- `teacher_forced_gate015.md`: `e2045e6802bb791e1515f4d085a052c84f69013ab1211b86c3ab0cb3f097c744`
- `teacher_forced_gate020.md`: `164293409a6f1fd8adf185de2e22f74099cb36d16e3a75d129b0971d5bff2f26`

## Decision

Do not run generation for this checkpoint. Adding target-only and slots-only
training controls did not create source-necessary teacher-forced signal. The
two positive clean IDs are still explained by zero-source or slots-only
controls.

This weakens objective-level anti-memory penalties as a rescue for the current
Perceiver answer-teacher plus delta-memory branch. The next gate should move to
a source-only sidecar/router whose transmitted signal cannot access target-only
or slot memory during source-signal formation.
