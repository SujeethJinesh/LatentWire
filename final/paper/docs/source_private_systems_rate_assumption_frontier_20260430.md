# Source-Private Systems Rate And Assumption Frontier

- date: `2026-04-30`
- gate: `source_private_systems_rate_assumption_frontier`
- artifact: `results/source_private_systems_rate_assumption_frontier_20260430/`
- status: pass as a systems/positioning artifact, not as a new native KV benchmark

## Question

Can the paper state a defensible systems contribution without implying an
unfair win over KV compression, prompt compression, or cache-to-cache methods
that solve different native problems?

## Method

I added an assumption-aware frontier builder:

```bash
./venv_arm64/bin/python scripts/build_source_private_systems_rate_assumption_frontier.py \
  --output-dir results/source_private_systems_rate_assumption_frontier_20260430
```

The table merges the endpoint systems caveat frontier, deterministic rate
frontier, KV byte-floor table, and semantic-anchor medium confirmation. Each
row records the communicated object, source-private assumption, receiver side
information, whether text or KV is exposed, source-destroying controls, byte
unit, prompt-token delta, KV byte floor, latency scope, and allowed claim.

## Result

The gate passes. Headline numbers:

- endpoint packet rows passing: `2/2`
- semantic-anchor medium pass rows: `18/18`
- minimum endpoint packet lift over target: `+0.425`
- minimum semantic-anchor packet lift over target: `+0.500`
- same-byte structured text max accuracy: `0.250`
- query-aware text oracle minimum payload: `14` bytes, `7.0x` the 2-byte packet
- full hidden-log relay byte ratio: `183.2x-186.8x`
- minimum full-log TTFT delta over packet: `+164.27 ms`
- minimum KV byte-floor ratio over packet: `10752.0x`
- under-specified receiver contract accuracy: `0.250`

## Interpretation

This strengthens technical contribution 3: the paper can now claim an
assumption-aware rate frontier for source-private task communication. LatentWire
packets occupy a 2-8 byte regime where same-byte visible text does not help,
higher-rate text relays catch up only by exposing private text, and KV/cache
methods require internal-state transport and much larger byte floors.

The artifact deliberately prevents overclaiming. C2C, KVComm, TurboQuant, QJL,
KIVI, KVQuant, LLMLingua, and gist-token rows are listed as related-work or
accounting contrasts unless we run them on their native tasks.

## What This Does Not Prove

- It does not show that LatentWire beats KV compression on native KV/cache
  tasks.
- It does not show production GPU serving throughput.
- It does not solve the learned latent receiver objection.
- It does not remove the need for cross-family model evidence.

## Next Gate

Run the learned/frozen embedding receiver ablation proposed by the method scout:
deterministic anchor-relative or k-means anchor features, no candidate feature
leakage, `train=768`, `eval=512`, budgets `4/8`, and the same
zero/shuffled/random/answer/text controls. Pass requires at least `+0.15` over
target and best destructive controls while preserving exact ID parity.
