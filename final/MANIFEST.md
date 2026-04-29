# Final Folder Manifest

- created: `2026-04-28`
- purpose: consolidated reproducibility and finalization folder
- source commit at creation: `8687e3748d3a1841231f29a2dd1419df3dcdbfd7`
- latest-model generalization scout added: `2026-04-28`

## Included

- upload-ready anonymous PDF/source/artifact files;
- final ICLR manuscript source and figures;
- source-private experiment scripts;
- focused tests;
- relevant result directories and manifests;
- final paper/review/upload memos;
- references and citation manifests;
- top-level repo guidance and environment files;
- `MANIFEST.sha256` with checksums for all files in this folder.
- latest-model/MoE generalization scout files for Qwen3.5/Qwen3.6 planning.
- Qwen3.5-0.8B CPU n16/n64 latest-small confirmation artifacts and non-Qwen
  cross-family falsification matrix.
- Qwen3.5-0.8B CPU n160 latest-small row and Granite/OLMo non-Qwen packet rows.
- Qwen3.5-0.8B n160 seed-repeat artifacts and Granite copied-helper n160 /
  trace-no-hint n64 prompt-contract diagnostics.
- Qwen3.5-2B n16 latest-small cross-size smoke and Qwen3.6 MoE/FP8 runbook.
- Qwen3.5-2B n64 latest-small cross-size confirmation artifacts.
- Qwen3.5-2B n160 latest-small cross-size confirmation artifacts.
- Source-private codebook-remap gate artifacts showing three remapped
  diagnostic codebooks across `500` examples and low-rate budgets.
- Qwen3.5-4B n16/n64 latest-small capacity-scaling artifacts.
- Gemma 4 E2B n16/n64 strict-prompt cross-family artifacts.
- Gemma 4 E2B MPS n160 seed-stability strict-prompt artifacts.
- Gemma 4 E2B MPS n500 large-slice strict-prompt artifact.
- Gemma 4 E2B MPS n160 raw-log/no-trace source-destroying ablation.
- Granite 3.3 2B n160 strict-prompt cross-family artifact.
- Granite 3.3 2B n160 strict-prompt seed-repeat and raw-log/no-trace ablation.
- Qwen3 target-model decoder core n64 CPU ablation.
- Qwen3 target-model decoder held-out n64 CPU ablation.
- Source-private systems summary artifact with deterministic rate rows,
  model-produced packet rows, target-decoder rows, and `183.2x-186.7x`
  full-log compression headline.
- Systems/novelty/future-method reference memo for C2C, KVComm, activation
  communication, prompt compression, tool-agent handoff, source coding,
  quantization, JEPA, Q-Former, and diffusion-inspired successor branches.
- Learned syndrome packet smoke script, test, seed29/seed30 artifacts, and
  competitor-threat reference memo for the next method-contribution branch.
- Real-feature tool-trace learned syndrome script, tests, seed-pair artifacts,
  and memo showing a common 6-byte pass on two seed pairs.
- OpenAI/vLLM-compatible endpoint runner and tests for Qwen3.6 MoE/FP8 gates.

## Notes

The original repo layout is preserved outside `final/`. This folder is a
staging copy for finalization and handoff, so existing commands in paper memos
still refer to repository-root paths such as `scripts/`, `tests/`, `paper/`,
and `results/`.

For anonymous submission, prefer the files in `final/upload/`.
