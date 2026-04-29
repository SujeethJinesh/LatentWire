# Source-Private Systems Caveat Frontier

- date: `2026-04-29`
- status: passed Mac-local systems-caveat gate
- result root: `results/source_private_systems_caveat_frontier_20260429/`
- scale rung: endpoint-proxy systems confirmation / reviewer overclaim control

## Current Readiness

The paper is stronger as a scoped source-private communication paper, but still
not ready for a broad systems or cross-family latent-transfer claim. This gate
locks the safest systems story: a 2-byte source-private packet occupies an
extreme-rate point that passes local endpoint controls, while text and KV/cache
baselines occupy different and usually higher-rate regimes.

## What This Adds

I added `scripts/build_source_private_systems_caveat_frontier.py`, which
aggregates the paper-ready `n=160` Mac-local endpoint rows, paired uncertainty
summaries, the derived KV/cache byte-floor table, and an explicit terse-prompt
failure case into one reviewer-facing artifact.

This is intentionally a caveat frontier, not a throughput benchmark. It records
both the positive systems evidence and the non-claims reviewers will care about.

## Result

- pass gate: `true`
- passing endpoint rows: `2/2`
- packet payload: `2` bytes
- minimum packet minus target accuracy: `+0.425`
- minimum packet minus best source-destroying control accuracy: `+0.425`
- minimum paired bootstrap CI95 lower bound versus target: `+0.350`
- query-aware text payload: `14` bytes, `7.0x` larger than the packet
- full hidden-log relay: `183.25x-186.75x` larger than the packet
- full hidden-log p50 TTFT delta: `+164.3 ms` to `+183.5 ms` versus packet
- minimum QJL-style 1-bit cache payload lower-bound: `10752.0x` packet bytes
- terse prompt stress: fails, as intended, with packet accuracy equal to target

## Endpoint Rows

| Surface | Packet | Target | Best destructive control | Query-aware bytes | Full-log bytes | Full-log p50 TTFT delta |
|---|---:|---:|---:|---:|---:|---:|
| core `n=160` label-strict | 0.675 | 0.250 | 0.250 | 14.0 | 366.5 | +164.3 ms |
| holdout `n=160` label-strict | 0.688 | 0.250 | 0.250 | 14.0 | 373.5 | +183.5 ms |
| core `n=16` terse prompt stress | 0.250 | 0.250 | 0.250 | 14.0 | 366.5 | +283.4 ms |

## Interpretation

This strengthens the systems contribution but narrows the claim. The packet is
excellent at the far-left byte point and passes local endpoint controls, but
query-aware/structured text can match or exceed accuracy once it is allowed more
bytes. The paper should claim an extreme-rate source-private communication
frontier, not general text-compression superiority.

Similarly, TurboQuant, QJL, KIVI/KVQuant, KVComm, C2C, PagedAttention, and
DistServe are relevant systems neighbors, but they solve different native
problems: vector/KV compression, cache sharing, or serving scheduling. The
artifact therefore compares byte floors and metric conventions while explicitly
avoiding a false claim that LatentWire is a better KV compression kernel.

## Non-Claims To Preserve In The Paper

- No claim of beating TurboQuant, QJL, KIVI, KVQuant, C2C, or KVComm on their
  native same-model/KV tasks.
- No production GPU serving throughput claim until vLLM/OpenAI-compatible
  server rows report TTFT, TPOT, throughput, memory, and concurrency.
- No broad cross-family latent-transfer claim; the current cross-family
  appendix is negative-boundary evidence.
- No prompt-contract-free receiver claim; the terse prompt stress shows the
  public receiver contract is part of the method.

## Next Gate

The highest-priority reviewer-risk gate is now anti-lookup receiver stress:
`n=160` core + holdout, label-blind candidates, per-example/remapped diagnostic
codes, Qwen/learned target decoder, full source-destroying controls,
matched/higher-byte text comparators, paired CIs, validity, bytes, and local
TTFT.
