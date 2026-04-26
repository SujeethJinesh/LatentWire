# Qwen2.5-Math -> Qwen3 SVAMP70 Source Cross-Attention Logprob Manifest

- date: `2026-04-26`
- status: `source_cross_attention_top_surface_fails_gate`
- git base commit: `e5a7be224d2ce53bfca747a5106add17a4f74b03`
- rung: medium-surface teacher-forced smoke on `svamp70_live`

## Result

- clean source-communication candidate IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `3/6`
- clean control leaks: `3/6`
- mean matched-minus-best-control clean margin: `-0.443233`
- target-preservation IDs scored: `22`
- target-preservation matched-positive count: `13/22`

Decision: fail the top-surface rescue. The tiny prefix-emitting
cross-attention connector remains source-control dominated even on the
highest-ranked source-complementary surface.

## Files

- `live_smoke.json`
- `live_smoke.md`
- `sha256.txt`
