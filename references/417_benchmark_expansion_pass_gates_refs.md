# Benchmark Expansion Pass-Gates Refs (2026-04-21)

Purpose: keep benchmark widening interpretable after the frozen `GSM8K32`
contract, with explicit pass gates before each new block.

## Strongest Sources

1. GSM8K
   Link: https://arxiv.org/abs/2110.14168
   Why it matters: keep `GSM8K32` as the frozen same-pair smoke gate.

2. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: current external same-pair bar on the exact frozen slice.

3. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: direct cache-style communication baseline for the later
   long-context block.

4. RULER
   Link: https://arxiv.org/abs/2404.06654
   Why it matters: next exact long-context follow-up once the same-pair row is
   stable.

5. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: structured KV-cache-centric follow-up after `RULER`.

6. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Why it matters: broader realistic long-context block after `SCBench`.

7. LongBench Pro
   Link: https://arxiv.org/abs/2601.02872
   Why it matters: later realism expansion, not a replacement for `RULER` or
   `SCBench`.

## Exact Block Structure

1. same-pair / same-family communication
2. cross-family text communication
3. long-context
4. multimodal

## Exact Pass Gates

1. `GSM8K32`
   Gate: beat `target self-repair` on the exact frozen IDs while keeping prompt
   hash, parser, tokenizer revision, seed, and answer extraction fixed.

2. `RULER-32`
   Gate: preserve retrieval hit rate, trace consistency, numeric extraction
   coverage, and per-length accuracy.

3. `SCBench`
   Gate: add bytes, latency, reuse rate, and KV loading/retrieval telemetry.

4. `LongBench v2`
   Gate: require stable per-task and per-length-bucket gains before widening
   further.

## Current Read

- Do not widen competitor tables before one new branch survives the same frozen
  GSM8K32 contract.
- Keep `ReasonBridge`, `LatentMAS`, and multimodal rows outside the main
  same-pair table unless they are ported into the exact same contract.
- The live widening order remains `RULER -> SCBench -> LongBench v2`, with
  multimodal still separate and last.
