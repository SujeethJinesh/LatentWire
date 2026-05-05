# Aggregate Packet-vs-Source-Index CI

This artifact consumes frozen prediction rows from the COLM acceptance-baseline audit. It uses a two-stage seed/item cluster bootstrap and does not rerun models.

| Benchmark | Seeds | Items/seed | Mean pkt-src | CI95 low | CI95 high |
|---|---:|---:|---:|---:|---:|
| ARC-Challenge | 10 | 1172 | -0.0016 | -0.0030 | -0.0003 |
| OpenBookQA | 5 | 500 | +0.0004 | -0.0012 | +0.0020 |

Interpretation: neither benchmark supports packet superiority over explicit source-index communication. This strengthens the workshop claim boundary rather than the positive-method claim.
