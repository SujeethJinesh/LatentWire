# Source-Private Endpoint Uncertainty

- pass gate: `True`
- bootstrap samples: `5000`
- min packet vs target CI95 low: `0.297`
- min packet vs best-control CI95 low: `0.297`
- min strict packet vs target CI95 low: `0.281`

| Surface | N | Packet | Strict packet | Target | Best control | Valid | Packet-target CI | Packet-control CI | Strict packet-target CI | Full-log TTFT delta ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `core_seed29_qwen3_n64_cpu_label_strict_controls` | 64 | 0.703 | 0.672 | 0.250 | 0.250 | 1.000 | [0.328, 0.578] | [0.328, 0.578] | [0.297, 0.547] | 217.2 |
| `holdout_seed30_qwen3_n64_cpu_label_strict_controls` | 64 | 0.672 | 0.656 | 0.250 | 0.250 | 1.000 | [0.297, 0.547] | [0.297, 0.547] | [0.281, 0.531] | 192.7 |

## Rate/Quality Comparators

| Surface | Query-aware accuracy | Packet-query CI | Query bytes / packet bytes | Structured JSON accuracy | Free-text accuracy | Full-log accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `core_seed29_qwen3_n64_cpu_label_strict_controls` | 0.703 | [-0.078, 0.078] | 14.0/2.0 | 0.594 | 0.719 | 0.484 |
| `holdout_seed30_qwen3_n64_cpu_label_strict_controls` | 0.703 | [-0.109, 0.047] | 14.0/2.0 | 0.609 | 0.719 | 0.531 |

Pass rule: All endpoint rows must pass their strict gate, packet valid rate must be >=0.95, paired bootstrap lower bounds versus target and best source-destroying control must be positive, strict-label packet-vs-target lower bound must be positive, and full-log p50 TTFT must be slower than the packet. Query-aware/structured relays are reported as rate-quality comparators, not required accuracy losses.
