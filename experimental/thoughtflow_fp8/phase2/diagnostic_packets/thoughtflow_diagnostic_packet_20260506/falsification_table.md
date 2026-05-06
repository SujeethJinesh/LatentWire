# ThoughtFlow Diagnostic Provenance Packet

Status: diagnostic only; not a positive-method packet.

| Artifact | Role | SHA-256 | Readout |
|---|---|---|---|
| `frozen_sparse_cache_probe.json` | stale_positive_first_surface | `sha256:f79c286c24c68c0f38d65a3d628db59084c3119c7b3e26ffe0ac8702be851ec9` | ALIVE on frozen sparse-cache probe; rdu_topk clears the preregistered promotion rule with margins 0.160 vs R-KV-like and 0.121 vs ThinKV-like. |
| `rdu_robustness_diagnostic.json` | stale_positive_robustness_probe | `sha256:623b962fd3c98068a59764b31a49a471dbe3b8ca33a4190ed68ac1354b4e4e26` | PROMOTED on cached full gate; deterministic trace splits keep positive margins and paired means, but split CIs are not uniformly below zero. |
| `rdu_no_retune_reproduction_check.json` | historical_positive_same_surface | `sha256:9a6420cf4d2bd18764f5e0e60fcd5964378a27996111b82485b71fcb668accd9` | REPRODUCED on measured no-retuning rerun; rdu_topk remains best compressed and clears the preregistered rule. |
| `rdu_alt_surface_reproduction_check.json` | same_family_falsification | `sha256:1302ce31ffa666aba02ed0e03fab8bdb01019e37fdd01ad77f234a6a0a1843fd` | NOT REPRODUCED on alternate measured no-retuning surface; inspect measured decision details. |
| `rdu_independent_trace_reproduction_check.json` | cross_family_falsification | `sha256:5a568e1c5178ccb55443184aa7d1c8ad521c30f8930ad3c10a11f27722f8c6a1` | NOT REPRODUCED on independent saved-trace no-retuning surface; inspect same-family/cross-family decision details. |
| `psi_fresh_sparse_cache_check.json` | fresh_successor_kill | `sha256:d0a08035650374a93babeb87845c1540411e12c72d54abfb568a431e345f26d3` | KILLED on one-shot fresh sparse-cache surface; psi_topk fails the preregistered promotion rule. |
| `vwac_fresh_sparse_cache_check.json` | fresh_successor_kill | `sha256:80a359545dc074f02f699e06a8228bb894d39548422bd47a49ad68ca9999ae0d` | KILLED on one-shot fresh sparse-cache surface; vwac_topk fails the preregistered promotion rule. |

## Interpretation

The same-surface RDU row is historical. It is paired here with the
original stale-positive surface, the cached robustness diagnostic,
the alternate-surface same-family failure, the independent-surface
cross-family failure, and the fresh PSI/VWAC successor kills. Together
these artifacts lock the current branch as diagnostic-only.
