# Source-Private Rate Frontier

- date: `2026-04-29`
- rung: systems/rate frontier on frozen deterministic core and holdout surfaces
- status: pass as a far-left byte-rate systems contribution

## Purpose

This gate strengthens the systems side of the paper without requiring remote
GPU access. It aggregates the existing reviewer-risk deterministic sweeps into a
rate frontier over packet, matched-byte text, structured JSON/free-text relay,
full diagnostic text, and full hidden-log relay.

It does not claim endpoint TTFT. It claims a reproducible byte-rate frontier
under the same deterministic target decoder and frozen IDs.

## Artifact

- `results/source_private_rate_frontier_20260429/`
- `results/source_private_rate_frontier_20260429/rate_frontier.json`
- `results/source_private_rate_frontier_20260429/rate_frontier.md`

## Result

| Surface | Target | Packet oracle bytes | JSON oracle bytes | Free-text oracle bytes | Full log bytes | Packet vs JSON | Packet vs full log | Matched-byte text at packet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core seed29 | 0.250 | 2.0 | 21.0 | 17.0 | 366.4 | 10.5x | 183.2x | 0.250 |
| holdout seed30 | 0.250 | 2.0 | 21.0 | 17.0 | 373.5 | 10.5x | 186.7x | 0.250 |

Headline:

- pass gate: `true`
- packet oracle bytes max: `2.0`
- JSON/free-text oracle bytes min: `21.0` / `17.0`
- packet vs JSON oracle compression min: `10.5x`
- packet vs full hidden-log compression min: `183.2x`
- matched-byte text at packet accuracy max: `0.250`

## Interpretation

This is now the cleanest systems contribution:

- At the same 2-byte rate, hidden-log truncation, JSON relay, and free-text
  relay stay at target-only accuracy.
- Structured relays do become oracle, but only at materially higher byte rates:
  17-21 bytes on these surfaces.
- Full hidden-log relay is oracle but 183-187x larger than the packet.

The paper should claim the far-left byte frontier, not that text relay is
universally weak. Text catches up at higher rates, and that is exactly why the
rate curve is reviewer-useful.

## Caveat

This artifact reports local Python decode timing from deterministic artifacts.
It is not a TTFT or serving-throughput benchmark. A later endpoint run should
measure TTFT, prefill/decode split, source generation time, memory, and
throughput under a fixed server stack.

## Exact Command

```bash
./venv_arm64/bin/python scripts/build_source_private_rate_frontier.py \
  --output-dir results/source_private_rate_frontier_20260429
```
