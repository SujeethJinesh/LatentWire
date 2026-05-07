# ThoughtFlow Diagnostic Provenance Packet

Status: diagnostic only; not a positive-method packet.

| Artifact | Role | SHA-256 | Readout |
|---|---|---|---|
| `frozen_sparse_cache_probe.json` | stale_positive_first_surface | `sha256:f79c286c24c68c0f38d65a3d628db59084c3119c7b3e26ffe0ac8702be851ec9` | SUPERSEDED historical readout; later gates demote or kill this row |
| `rdu_robustness_diagnostic.json` | stale_positive_robustness_probe | `sha256:623b962fd3c98068a59764b31a49a471dbe3b8ca33a4190ed68ac1354b4e4e26` | SUPERSEDED historical readout; later gates demote or kill this row |
| `rdu_no_retune_reproduction_check.json` | historical_positive_same_surface | `sha256:9a6420cf4d2bd18764f5e0e60fcd5964378a27996111b82485b71fcb668accd9` | SUPERSEDED historical readout; later gates demote or kill this row |
| `rdu_alt_surface_reproduction_check.json` | same_family_falsification | `sha256:1302ce31ffa666aba02ed0e03fab8bdb01019e37fdd01ad77f234a6a0a1843fd` | NOT REPRODUCED on alternate measured no-retuning surface; inspect measured decision details. |
| `rdu_independent_trace_reproduction_check.json` | cross_family_falsification | `sha256:5a568e1c5178ccb55443184aa7d1c8ad521c30f8930ad3c10a11f27722f8c6a1` | NOT REPRODUCED on independent saved-trace no-retuning surface; inspect same-family/cross-family decision details. |
| `psi_fresh_sparse_cache_check.json` | fresh_successor_kill | `sha256:d0a08035650374a93babeb87845c1540411e12c72d54abfb568a431e345f26d3` | KILLED on one-shot fresh sparse-cache surface; psi_topk fails the preregistered promotion rule. |
| `vwac_fresh_sparse_cache_check.json` | fresh_successor_kill | `sha256:80a359545dc074f02f699e06a8228bb894d39548422bd47a49ad68ca9999ae0d` | KILLED on one-shot fresh sparse-cache surface; vwac_topk fails the preregistered promotion rule. |

## Preregistrations

| File | Role | SHA-256 |
|---|---|---|
| `preregister_recurrence_distance_utility_20260506.md` | one_shot_method_preregistration | `sha256:676d5239dddf00668368161df31f90b9a2a91ee2f3b47da378bd65887fd4d832` |
| `preregister_prefix_surprisal_utility_20260506.md` | fresh_successor_preregistration | `sha256:cd8a6a108636bfaf427d3e71752b140f559c7438eb51930abe1a8aa82feeeec5` |
| `preregister_value_weighted_attention_contribution_20260506.md` | fresh_successor_preregistration | `sha256:a701495b06ecbb9d0133f106539a59f4932e7c81a6e25888d6165e36714c2204` |

## Interpretation

The same-surface RDU row is historical. It is paired here with the
original stale-positive surface, the cached robustness diagnostic,
the alternate-surface same-family failure, the independent-surface
cross-family failure, and the fresh PSI/VWAC successor kills. Together
these artifacts lock the current branch as diagnostic-only.
