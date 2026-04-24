# SVAMP32 Teacher-Forced Connector Diagnostic Results - 2026-04-24

## Summary

- status: `no_teacher_forced_source_signal`
- checkpoint: `.debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt`
- fixed gate: `0.15`
- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- decision: kill this checkpoint as a source-necessary positive row

## Tracked Artifacts

- `perceiver_queries_gate015_answer_margin_clean.json`
- sha256: `3e67de34ca7121cc803bc10bad78b1b3aab4e2857efd8654eab6655132f693a9`
- `perceiver_queries_gate015_answer_margin_clean.md`
- sha256: `098cd43ddddc9e269f357699260880cacab3cc4925851035db90323107ccb48d`
- `perceiver_queries_gate015_answer_margin_clean_self.json`
- sha256: `47443a71295606330e26911777ed4b496f390506538f943695e2e1d6df746c0c`
- `perceiver_queries_gate015_answer_margin_clean_self.md`
- sha256: `6bf0367b38a34621508ebb8c4e40209462ac7107c432914d650a0ea584be6903`

## Scratch Artifacts

- `.debug/svamp32_teacher_forced_connector_diagnostic_20260424/logs/perceiver_queries_gate015_answer_margin_clean.log`
- `.debug/svamp32_teacher_forced_connector_diagnostic_20260424/logs/perceiver_queries_gate015_answer_margin_clean_self.log`
- sha256: `ebecf85e36ff89b93ba15b947eaf889ecb30af1ac728c43b01ae691c23d182b0`

## Reproduction Notes

See `paper/svamp32_teacher_forced_connector_diagnostic_20260424.md` for full
commands. The diagnostic used exact-ID SVAMP32, Qwen2.5-0.5B-Instruct as
source, Qwen3-0.6B as target, K-only transport, attention-selected positions,
the previous 8-query Perceiver connector checkpoint, and controls `matched`,
`zero_source`, `shuffled_source`, `target_only`, and `slots_only`.
