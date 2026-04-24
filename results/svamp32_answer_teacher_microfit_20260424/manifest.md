# SVAMP32 Answer-Teacher Microfit Results - 2026-04-24

## Summary

- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`
- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- decision: no greedy generation sweep; kill this calibration-proxy answer-teacher microfit

## Tracked Artifacts

- `answer_teacher_w090_gate015_clean_self.json`
- sha256: `e9db6ffed6ba5c42a9b983e48154fde3eac98248c56b05c57900cd9870266f71`
- `answer_teacher_w090_gate015_clean_self.md`
- sha256: `324e123639f812030b3e5e3f8c1ab81127468010f0438c3ee5752ca166a1a6e2`

## Scratch Artifacts

- `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`
- sha256: `437b7eecf8f0b3704eb8e6260cefcd9d45ead2a31d02855c33655c06dd2de8fc`
- `.debug/svamp32_answer_teacher_microfit_20260424/logs/calibrate_answer_teacher_w090_r16_q8_seed1.log`
- sha256: `8cbfe57de7c83d86fbae9c46e134f08110938794a6b2f60606456cb9b4091d88`
- `.debug/svamp32_answer_teacher_microfit_20260424/logs/diagnostic_answer_teacher_w090_gate015_clean_self.log`
- sha256: `ada211e52f4d0b3189a5a5ce2d9487536367419d0db5705942fdb0c9302461a1`

## Reproduction Notes

See `paper/svamp32_answer_teacher_microfit_20260424.md` for full commands.
The run used exact-ID SVAMP32, Qwen2.5-0.5B-Instruct as source, Qwen3-0.6B as
target, `8` Perceiver connector queries, rank `16`, answer-token teacher weight
`0.90`, K/V residual loss, target-self zero-residual preservation, and
zero/shuffle source-control training.
