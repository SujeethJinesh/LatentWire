# Competitor Benchmark Stack Refs (2026-04-22)

Purpose: freeze the nearest public comparison stack and the order for widening
once a same-pair row survives the frozen GSM8K32 contract.

## Closest Communication Competitors

1. Cache-to-Cache (C2C)
   Link: https://arxiv.org/abs/2510.03215
   Code: https://github.com/thu-nics/C2C
   Why it matters: direct KV-cache projection and fusion; closest same-medium
   baseline for the current real lane.

2. KVComm: Selective KV Sharing
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: selective layer/head/value sharing with explicit importance
   scoring; strongest cheap efficiency-side comparator.

3. KVCOMM: Online Cross-context KV-cache Communication
   Link: https://arxiv.org/abs/2510.12872
   Code: https://github.com/FastMAS/KVCOMM
   Why it matters: anchor reuse and offset approximation are close to the
   preserve-anchor story.

4. Latent Space Communication via K-V Cache Alignment
   Link: https://arxiv.org/abs/2601.06123
   Why it matters: closest shared-space / latent alignment comparator.

5. Q-KVComm
   Link: https://arxiv.org/abs/2512.17914
   Why it matters: explicit quantized heterogeneous KV communication baseline.

## Benchmark Expansion Order

1. RULER
   Link: https://arxiv.org/abs/2404.06654
   Code: https://github.com/NVIDIA/RULER
   Why it matters: cheapest long-context falsifier after the frozen GSM8K32
   contract.

2. One matched cross-family pair
   Why it matters: architecture/tokenizer mismatch test before broad benchmark
   spend.

3. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: closest benchmark to the KV-cache lifecycle / shared-context
   medium.

4. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Code: https://github.com/THUDM/LongBench
   Site: https://longbench2.github.io/
   Why it matters: first broad realistic reasoning expansion.

5. LongBench Pro
   Link: https://arxiv.org/abs/2601.02872
   Why it matters: broader realism and bilingual stress after LongBench v2.

6. MMLongBench
   Repo: https://github.com/EdinburghNLP/MMLongBench
   Why it matters: optional multimodal extension only after the text-only story
   is stable.

## Fair Comparison Contract

- Freeze example IDs and ordering.
- Keep prompts, decoding, parsing, and normalization identical.
- Report accuracy and efficiency together: bytes sent, KV layers/heads shared,
  latency, and TTFT when relevant.
- Separate same-family and cross-family tables.
- Keep no-communication, text-relay, and upper-bound/oracle-style comparators
  visible.
- For learned repair branches, report fit-set size and guarantee zero eval
  leakage.
- Reject rows that fail exact ID parity, coverage, or non-empty-output checks.

## Local Bootstrap

- `references/external/C2C`
- `references/external/RULER`
- `references/external/LongBench`

These are local inspection clones for harness/bootstrap work; the paper story
should still stay frozen on GSM8K32 until a branch survives the same contract.

## Current Read

- The next widening order should be `RULER -> matched cross-family pair ->
  SCBench -> LongBench v2`.
- Do not spend serious benchmark time before a new same-pair row either beats
  `dynalign_module_replace_residrank16 = 0.1250` or ties it with a clear
  bytes/latency win.
