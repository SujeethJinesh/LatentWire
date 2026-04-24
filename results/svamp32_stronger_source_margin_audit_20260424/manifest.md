# SVAMP32 Stronger-Source Margin Audit Results - 2026-04-24

## Summary

- status: `no_source_margin_signal`
- frozen target: `Qwen/Qwen3-0.6B`
- stronger sources tested:
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
- gate: pass only if source/text final answers or positive source-margin
  advantage expose at least `2/6` clean residual IDs without relying on target
  priors
- outcome: strongest row remains below threshold
  - 1.5B source-final clean correct: `0/6`
  - 1.5B positive source-margin advantage: `1/6`
  - 7B source-final clean correct: `0/6`
  - 7B text-relay clean correct: `1/6`
  - 7B positive source-margin advantage: `1/6`

## Tracked Artifacts

- `qwen25_15b_to_qwen3_06b_source_margin_clean_self_with_sourcegen.json`
- sha256: `2c0b067317e6e47235a167a50f9a609aff02e2c4105c30c3c630f63bf895fa58`
- `qwen25_15b_to_qwen3_06b_source_margin_clean_self_with_sourcegen.md`
- sha256: `68706d5e4054ec939cda353b263761cb236e517beee27c765473659c3b1fbc4b`
- `qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.json`
- sha256: `59bffba1dd9bd06155bab537a888bdc4bac92eaf963258aebc19363cfe806e97`
- `qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.md`
- sha256: `fbf44984dd795c8a5e9033f648d783f5f6044a73d85af4e9e1b7621a3ed18d65`
- `svamp32_clean_self_eval.jsonl`
- sha256: `07784ad26e52e51a1c6080b71294543bf420854b67e4851f5fe6a6dcf0e30995`
- `svamp32_clean_self_eval.jsonl.meta.json`
- sha256: `eb5be5095c5d4e055e2d6e2e2414d4bd0340e9f3b4410ae5656dc8450c87008a`
- `qwen25_7b_source_alone_clean_self.jsonl`
- sha256: `0163af711efba78a106a54a85ab10ebab22d9bb5612c4b4f4df40302554a3774`
- `qwen25_7b_source_alone_clean_self.jsonl.meta.json`
- sha256: `3d88284defe19772679c020c6a49af65d3e968473111dfaf6f583e2a3ee5281a`
- `qwen25_7b_text_to_text_clean_self.jsonl`
- sha256: `1261e702f8e9e089ffe1cdf2f43282dfba8c93ddb105479575e726a37dd0f860`
- `qwen25_7b_text_to_text_clean_self.jsonl.meta.json`
- sha256: `b731b61b1bd74d738a37724b291d4fe411fd5c3920a4f4da5e1722ec54ed7221`

## Related Full-32 Stronger-Source Artifact

- `results/svamp32_stronger_source_baselines_20260424/source_alone.jsonl`
- sha256: `24c00515dec342c44b52267b5a9d269f6ee92b2f7ba0676bb30db4ccd535a228`
- `results/svamp32_stronger_source_baselines_20260424/source_alone.jsonl.meta.json`
- sha256: `7f1dd73da08070732a6cacb73ecfd2b2a808a37033ac973d4efbd80bd58b25ce`
- `results/svamp32_stronger_source_baselines_20260424/manifest.json`
- sha256: `582eaaa66ea3e83293bc4dcc046025978696a0e6a9b196fe6fcdd4a683bb9fb7`
- `results/svamp32_stronger_source_baselines_20260424/manifest.md`
- sha256: `cf823985424605d3aca5edfc693f676e57f4ea1d8b9fc11a17c9988c47024466`

## Scratch Logs

- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_15b_to_qwen3_06b_source_margin_clean_self.log`
- sha256: `37415863d312369f614960cf2e88172db7e89c12bf2e005c8888dbc983d9d7de`
- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_15b_to_qwen3_06b_source_margin_clean_self_with_sourcegen.log`
- sha256: `347c7841f09ad4272001f0f392c97929c67c47a5562eb13c2c211d871549aaf0`
- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_7b_to_qwen3_06b_source_margin_clean_self.log`
- sha256: `0b3fe591dc66f3b254cd41cd107f6f4c68a026bea413ac8e11298611040306fe`
- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_7b_source_alone_clean_self.log`
- sha256: `b2d8d9e2d259a7f3ef611eae0886a827153bc938fcfad0582f4d930ad0a19593`
- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_7b_text_to_text_clean_self.log`
- sha256: `9c6b7303f7081954025f11b8bc3069c929f5b770a537f561a3b697cea9843306`
- `.debug/svamp32_stronger_source_margin_audit_20260424/logs/qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.log`
- sha256: `fcdd5e6000ccc8fa117f640f87f596decf6c3e71a15a1f97be075f7356ef521d`

## Reproduction Notes

See `paper/svamp32_stronger_source_margin_audit_20260424.md` for commands and
interpretation. The 7B audit uses `--dtype float16`; the 1.5B audit uses the
default `float32` margin scoring.
