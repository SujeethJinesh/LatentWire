# Target-Decoder N64 CPU Gate

- date: `2026-04-28`
- gate: `source_private_target_decoder_n64_cpu_20260428`
- status: passed CPU n64 on core and held-out surfaces; MPS n160 backend-blocked before generation
- scale rung: target-decoder smoke to small confirmation

## Readiness

This gate addresses the reviewer objection that the target-side decoder is only
a hand-coded lookup. It does not replace the deterministic protocol decoder in
the main result, but it strengthens the claim that a target LLM can use the
same compact source packet and public candidate handles.

## MPS Backend Failure

The intended Qwen3 target-decoder n160 MPS run failed before prediction with an
Apple MPS matmul shape error. This is logged as a backend compatibility issue,
not method evidence.

## CPU Fallback Result

`Qwen/Qwen3-0.6B` target decoder on CPU, core seed29, n64:

- matched source packet: `42/64 = 0.656`
- target-only: `16/64 = 0.250`
- best source-destroying / structured 2-byte control: `16/64 = 0.250`
- matched-minus-target: `+0.406`
- matched-minus-best-control: `+0.406`
- valid matched predictions: `1.000`
- exact-ID parity: `true`
- p50 matched latency: `2182 ms`

Controls:

- shuffled packet: `0.250`
- random same-byte packet: `0.250`
- structured JSON 2-byte: `0.250`
- structured free-text 2-byte: `0.250`

Held-out seed30 n64:

- matched source packet: `46/64 = 0.719`
- target-only: `16/64 = 0.250`
- best source-destroying / structured 2-byte control: `17/64 = 0.266`
- matched-minus-target: `+0.469`
- matched-minus-best-control: `+0.453`
- valid matched predictions: `1.000`
- exact-ID parity: `true`
- p50 matched latency: `2237 ms`

## Interpretation

This is the strongest target-decoder evidence so far. It upgrades target-side
model decoding from tiny smoke to paired local n64 confirmation: the target LLM
can use the compact source-private packet above target prior on both core and
held-out surfaces, while source-destroying and same-byte structured controls
remain at the target floor.

The result is still not large enough to make the learned target decoder the main
claim. It should be presented as an ablation that reduces the hand-coded decoder
objection.

## Artifacts

- `results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n64_cpu/summary.json`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n64_cpu/target_predictions.jsonl`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n64_cpu/manifest.md`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n64_cpu/summary.json`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n64_cpu/target_predictions.jsonl`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n64_cpu/manifest.md`

## Next Gate

Stop widening target-decoder rows locally. The next highest-value technical
branch is a learned Wyner-Ziv/syndrome packet smoke on the same source-private
candidate-pool contract, unless MoE/FP8 endpoint access becomes available.
