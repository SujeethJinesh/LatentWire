# Source-Private Target-Decoder Paired Uncertainty

- pass gate: `True`
- rows: `2`
- pass rows: `2`
- min matched-target: `0.750`
- min matched-best-control: `0.734`
- min CI95 low vs target: `0.641`
- min CI95 low vs best control: `0.625`
- min valid prediction rate: `1.000`
- max p50 latency ms: `1119.061`

## Rows

| Surface | N | Matched | Target | Best control | Matched-target | Matched-control | CI low target | CI low control | Valid | p50 ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_seed29_n64_binary_logprob_cpu | 64 | 1.000 | 0.250 | 0.250 | 0.750 | 0.750 | 0.641 | 0.641 | 1.000 | 1087.6 | `True` |
| qwen3_seed31_n64_binary_logprob_cpu | 64 | 1.000 | 0.250 | 0.266 | 0.750 | 0.734 | 0.641 | 0.625 | 1.000 | 1119.1 | `True` |

## Interpretation

This gate asks whether a frozen target LLM can consume the compact source packet beyond target priors and same-byte/source-destroyed controls. It is a reviewer-defense receiver gate, not a systems-speed claim.

## Pass Rule

Every target-decoder run must pass its point gate, preserve exact-ID parity, have matched valid prediction rate >=0.95, and have paired CI95 lower bounds above +0.10 versus both target-only and the strongest source-destroying or matched-byte text control.
