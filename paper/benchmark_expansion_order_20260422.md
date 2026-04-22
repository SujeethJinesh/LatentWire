# Benchmark Expansion Order

Date: `2026-04-22`

## Why This Note Exists

Reviewer feedback and the latest reference pass both say the same thing:

- do not widen benchmarks casually from a tiny same-pair smoke
- keep same-pair and cross-family stories separated
- report uncertainty and budget, not only raw accuracy

This note freezes the expansion order and the comparison contract for the next
phase.

## Expansion Order

1. Larger frozen same-pair GSM8K campaign
   - purpose: replace the oracle-saturated GSM8K32 smoke as the main nearby
     variant ranker
   - requirements: paired deltas vs `target_alone`, seed repeats, diagnostics

2. One matched cross-family pair
   - purpose: falsify the “this is just same-family calibration” hypothesis
   - preferred pairs:
     - `Qwen2.5-3B-Instruct ↔ Llama-3.2-3B-Instruct`
     - `Qwen2.5-1.5B-Instruct ↔ Llama-3.2-1B-Instruct`
     - `Gemma-2-2B-it ↔ Phi-3.5-mini-instruct`

3. `RULER`
   - purpose: cheapest long-context falsifier with controlled tasks

4. `SCBench`
   - purpose: closest benchmark to the KV-cache / shared-context medium

5. `LongBench v2`
   - purpose: first realistic broad reasoning expansion

Only after that:

6. `LongBench`
7. `∞Bench`
8. multimodal extensions

## Fair Comparison Contract

- Freeze example IDs and ordering.
- Keep prompts, decoding, parsing, and normalization identical.
- Report accuracy with uncertainty:
  - paired delta vs `target_alone`
  - bootstrap interval
  - seed-repeat spread
- Report efficiency with every main row:
  - bytes
  - latency
  - TTFT when relevant
  - transport depth / layers / heads / tokens
- Separate same-family and cross-family tables.
- Keep no-communication, text relay, and strongest public comparator visible.
- Reject any row that fails ID parity, extraction coverage, or non-empty-output
  checks.

## Comparator Stack

Keep these visible once the benchmark blocks open up:

- `C2C`
- `KVComm`
- latent-space KV-alignment comparators when runnable on the exact pair
- same-model or no-communication compression controls when bytes are part of
  the claim

For appendix-scale cross-family expansion after the small-model phase:

- `Qwen2.5-7B-Instruct ↔ Mistral-7B-Instruct-v0.3`
- `Qwen2.5-7B-Instruct ↔ Llama-3.1-8B-Instruct`
- `Llama-3.1-8B-Instruct ↔ Gemma-2-9B-it`

## Current Read

The current live same-pair row is still:

- `dynalign_module_replace_residrank16 = 0.1250`

The first larger frozen same-pair read now says:

- `target_alone = 0.0571` (`4/70`)
- `dynalign_module_replace_residrank16 = 0.1143` (`8/70`)
- `C2C = 0.1286` (`9/70`)

So GSM8K32 should remain only a smoke gate, while the larger frozen slice
becomes the real decision surface for nearby variants.
