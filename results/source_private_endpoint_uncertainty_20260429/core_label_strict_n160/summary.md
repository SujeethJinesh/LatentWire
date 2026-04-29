# Source-Private Endpoint Uncertainty

- pass gate: `True`
- bootstrap samples: `5000`
- min packet vs target CI95 low: `0.350`
- min packet vs best-control CI95 low: `0.350`
- min strict packet vs target CI95 low: `0.338`

| Surface | N | Packet | Strict packet | Target | Best control | Valid | Packet-target CI | Packet-control CI | Strict packet-target CI | Full-log TTFT delta ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `core_seed29_qwen3_n160_cpu_label_strict_controls` | 160 | 0.675 | 0.662 | 0.250 | 0.250 | 1.000 | [0.350, 0.500] | [0.350, 0.500] | [0.338, 0.487] | 164.3 |

## Rate/Quality Comparators

| Surface | Query-aware accuracy | Packet-query CI | Query bytes / packet bytes | Structured JSON accuracy | Free-text accuracy | Full-log accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `core_seed29_qwen3_n160_cpu_label_strict_controls` | 0.694 | [-0.069, 0.031] | 14.0/2.0 | 0.575 | 0.713 | 0.463 |

Pass rule: All endpoint rows must pass their strict gate, packet valid rate must be >=0.95, paired bootstrap lower bounds versus target and best source-destroying control must be positive, strict-label packet-vs-target lower bound must be positive, and full-log p50 TTFT must be slower than the packet. Query-aware/structured relays are reported as rate-quality comparators, not required accuracy losses.
