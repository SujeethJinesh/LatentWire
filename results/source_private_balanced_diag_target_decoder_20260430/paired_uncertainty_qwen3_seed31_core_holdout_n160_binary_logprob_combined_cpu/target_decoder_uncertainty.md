# Source-Private Target-Decoder Paired Uncertainty

- pass gate: `True`
- rows: `2`
- pass rows: `2`
- min matched-target: `0.750`
- min matched-best-control: `0.750`
- min CI95 low vs target: `0.681`
- min CI95 low vs best control: `0.681`
- min valid prediction rate: `1.000`
- max p50 latency ms: `1674.072`

## Rows

| Surface | N | Matched | Target | Best control | Matched-target | Matched-control | CI low target | CI low control | Valid | p50 ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_seed31_core_n160_binary_logprob_combined_cpu | 160 | 1.000 | 0.250 | 0.250 | 0.750 | 0.750 | 0.681 | 0.681 | 1.000 | 1651.6 | `True` |
| qwen3_seed31_holdout_n160_binary_logprob_combined_cpu | 160 | 1.000 | 0.250 | 0.250 | 0.750 | 0.750 | 0.681 | 0.681 | 1.000 | 1674.1 | `True` |

## Interpretation

This gate asks whether a frozen target LLM can consume the compact source packet beyond target priors and same-byte/source-destroyed controls. It is a reviewer-defense receiver gate, not a systems-speed claim.

## Pass Rule

Every target-decoder run must pass its point gate, preserve exact-ID parity, have matched valid prediction rate >=0.95, and have paired CI95 lower bounds above +0.10 versus both target-only and the strongest source-destroying or matched-byte text control.
