# LatentWire Terminal Negative Sanitized

This dashboard is safe for future screening sessions: it cites aggregate metrics, run IDs, and split-level hashes/paths only. It does not expose raw held-out rows.

## Scope

- terminal branches: `L_SCORECOMP_wz_bins_deployable`, `L_B1_damage_avoidance_trust`, `L_Q1_receiver_query_packet`, and deterministic C2C/KVComm packet smokes.
- claim type: bounded negative / diagnostic failure localization, not a positive communication method.
- locality: existing aggregate artifacts only; no confirm-row payloads are included here.

## Aggregate Evidence

| branch | artifact | n | aggregate result | decision |
| --- | --- | ---: | --- | --- |
| one-way deployable WZ / score packet | `results/mac_continue/latentwire_one_way_confirm/summary.json` | 381 | deployable delta vs source-index-confidence `-0.023622`, CI `[-0.060367, 0.013123]`; full source-only oracle delta `-0.031496`, CI `[-0.062992, 0.002625]`; source+target-at-encoder upper bound delta `0.110236`, CI `[0.081365, 0.141732]` | `BOUNDED_NEGATIVE` |
| powered dev/gate score ladder | `results/mac_continue/latentwire_oracle_ladder/summary.json` | 1500 scored dev/gate rows; 359 gate rows | deployable WZ/current packet delta vs best score baseline `-0.013928`, CI `[-0.050139, 0.022284]`; source+target-at-encoder upper bound delta `0.125348`, CI `[0.089136, 0.161560]`; information drops from `2.098527` bits to `0.281933` bits after receiver conditioning | `not-deployable` |
| L_Q1 receiver-query packet | `results/mac_continue/latentwire_query_packet/summary.json` | 359 gate rows | L_Q1 delta vs source-index-confidence `-0.022284`, CI `[-0.055710, 0.011142]`; controls did not collapse; query-only/reply-only ablations explain the signal | `KILLED` |
| KVComm/C2C packet smokes | `registry/L_C2_c2c_kv_lcf_anchor.yaml` | smoke/diagnostic rows only | KVComm matched zero-source agreement `1.0`; answer packets leak answer text; teacher/candidate delta packets tie or lose to deterministic controls | `KILLED_AS_METHOD_EVIDENCE`; baseline anchor only |

## Safe Usage

- Use this file for paper drafting and reviewer-facing aggregate claims.
- Do not import raw rows from any path containing `confirm`.
- Do not queue `L_SCORECOMP`, `L_B1`, or `L_Q1` again without a new mechanism-level falsifier and a fresh preregistered card.
- Treat L_A2 and CacheWire separately: their v2 measurements were underpowered/broken, not terminal negatives.
