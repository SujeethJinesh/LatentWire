# Qwen2.5-Math -> Qwen3 SVAMP32 Source Cross-Attention Logprob Manifest

- date: `2026-04-26`
- status: `source_cross_attention_first_rung_fails_gate`
- git base commit: `a960d68256b8fd51e7108ce62cc6309a2367e89f`
- rung: strict-small teacher-forced pre-generation smoke

## Result

- clean source-communication candidate IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.383649`
- target-preservation IDs scored: `8`
- target-preservation matched-positive count: `5/8`

Decision: fail the first-rung token-local cross-attention gate. This exact
tiny prefix-emitting cross-attention implementation should not be scaled by
epochs or hidden width without a new reason; controls are already dominating on
the clean IDs.

## Files

- `smoke.json`
- `smoke.md`
- `sha256.txt`
