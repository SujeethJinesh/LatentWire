# SinkAware Approximate Revival Gate

Status: **REVIVE only as approximate low-rank/clustered query prior; exact static prior remains killed.**

The exact static sink-prior branch remains dead: fixed sink keys still need query-dependent `QK_sink`.
This gate only tests whether an approximate low-rank or clustered-query prior might be worth a later real-query probe.

| Query case | Static R2 | Rank-1 query R2 | Rank-2 query R2 | Rank-4 query R2 | Rank-8 query R2 |
|---|---:|---:|---:|---:|---:|
| clustered | -0.006 | 0.227 | 0.518 | 0.816 | 0.976 |
| low_rank | -0.008 | 0.388 | 0.774 | 0.999 | 0.999 |
| random | -0.009 | 0.007 | 0.015 | 0.036 | 0.102 |

## Decision

Static priors are not revived. They do not explain query-dependent sink logits.
The only branch still alive is approximate: exploit a low-dimensional or clustered query manifold, or fuse exact `QK_sink` computation more cheaply without pretending it can be skipped.
The next gate must use real Q/K tensors or attention telemetry; synthetic geometry alone is not enough for a reviewer pack.
