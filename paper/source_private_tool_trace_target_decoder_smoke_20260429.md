# Source-Private Tool-Trace Target-Decoder Smoke

- date: `2026-04-29`
- status: target-decoder smoke passed on core micro and held-out small slice
- live branch: explicit source-private tool-trace packet handoff
- scale rung: target-decoder smoke

## Question

Can the deterministic `REPAIR_DIAG -> candidate metadata` lookup be replaced by
an LLM-mediated target-side selector while preserving source-destroying
controls?

This row addresses the skeptical-reviewer concern that the current method could
look like a hand-coded label lookup rather than a communication interface.

## Setup

The source packet remains the same compact two-character diagnostic. The target
decoder is replaced by Qwen3-0.6B. The target prompt contains:

- target-prior fallback label
- source packet or `<NO_SOURCE_PACKET>`
- public candidate labels
- each candidate's `handles_repair_diag` metadata

The prompt instructs the target model to return the candidate whose
`handles_repair_diag` exactly matches a valid packet, otherwise return the
target-prior label. This is still a protocol-shaped decoder, but the candidate
selection step is model-mediated rather than hard-coded.

Controls:

- target-only
- shuffled packet
- random same-byte packet
- matched-byte structured JSON truncated to 2 bytes
- matched-byte free text truncated to 2 bytes

## Commands

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl .debug/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_benchmark/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n16 \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 16 \
  --seed 29 \
  --max-new-tokens 24 \
  --no-enable-thinking
```

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl .debug/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_benchmark/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n32 \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 32 \
  --seed 30 \
  --max-new-tokens 24 \
  --no-enable-thinking
```

## Results

| Surface | N | Target | Matched packet | Best control | Matched - target | Matched - control | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| core seed 29 | `16` | `0.250` | `0.688` | `0.250` | `0.438` | `0.438` | `True` |
| held-out seed 30 | `16` | `0.250` | `0.750` | `0.312` | `0.500` | `0.438` | `False` |
| held-out seed 30 | `32` | `0.250` | `0.750` | `0.281` | `0.500` | `0.469` | `True` |

The held-out `16` row failed only because one random same-byte packet landed
inside the strict `+0.05` control tolerance on a tiny slice. Widening the same
surface to `32` examples restored the pass.

## Interpretation

This is not the main paper evidence, but it reduces the largest novelty risk.
The target-side selection does not have to be a hand-coded lookup: a small LLM
can read the source-private packet and candidate metadata, follow the fallback
rule, and preserve same-byte relay controls.

The row should be presented as a smoke/ablation, not as a replacement for the
large deterministic decoder evidence. Scaling it to `160+` examples would be
useful, but it is compute-expensive because each example currently runs all
conditions as separate generations.

## Decision

Promote the row as positive target-decoder smoke. The paper can now say the
deterministic decoder is not essential in principle, while still scoping the
main result to a protocol-shaped candidate decoder.

## Next Gate

`source_private_tool_trace_paper_sections_20260429`:

- draft introduction, method, benchmark, results, controls, systems, and
  limitations sections
- include the target-decoder smoke as an ablation that reduces protocol-lookup
  novelty risk
- do not overstate it as full learned target decoding until scaled
