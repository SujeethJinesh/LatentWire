# SVAMP32 Source Margin Audit Results - 2026-04-24

## Summary

- status: `no_source_margin_signal`
- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- source/text final clean correct: `0/6`
- source-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `0/6`
- decision: kill source-final-answer and source-margin informativeness as a
  justification for more same-pair Qwen2.5-0.5B to Qwen3-0.6B connector tuning

## Tracked Artifacts

- `source_margin_clean_self.json`
- sha256: `16ab06a97024d61cbb6efb3b1cfbebacc9f542bab25862e1671b5e4ec7a919ff`
- `source_margin_clean_self.md`
- sha256: `9bb5a859eff08ff0727ab6fd8d57bc37b38dbd8dae0f072b725e6fb609f9b91b`

## Scratch Artifacts

- `.debug/svamp32_source_margin_audit_20260424/logs/source_margin_clean_self.log`
- sha256: `ce606e110ab7f35096ea799d52bedeeed070d8590d0b3db1af789f11b5ab03c3`

## Reproduction Notes

See `paper/svamp32_source_margin_audit_20260424.md` for the full command. The
audit used exact-ID SVAMP32, Qwen2.5-0.5B-Instruct as source, Qwen3-0.6B as
target, source `brief_analysis` prompts, chat templates for both models, and
gold-vs-target-wrong numeric continuation margins.
