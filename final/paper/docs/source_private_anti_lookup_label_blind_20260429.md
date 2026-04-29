# Source-Private Anti-Lookup Label-Blind Smoke

- date: `2026-04-29`
- status: passed as an anti-lookup collapse control
- result root: `results/source_private_anti_lookup_label_blind_20260429/`
- scale rung: micro smoke anti-lookup stress

## Current Readiness

This gate strengthens the paper's honesty, not its headline accuracy. The
current positive endpoint method is a source-private packet decoded with public
target-side side information. It is not protocol-free semantic transfer.

The key reviewer objection was that the endpoint result might be just a visible
diagnostic-code lookup. The label-blind stress removes both the public
`handles_repair_diag` table and original candidate labels from the target
prompt, replacing candidates with opaque visible IDs such as `Option A`.
The parser is also hardened: under `candidate_view=label_blind`, generated
repair keys like `G0` are not mapped through hidden candidate metadata.

## Result

- pass gate: `true` as a collapse-control artifact
- rows: `2`
- surfaces: core `n=8`, holdout `n=8`
- exact-ID parity: `true`
- matched packet valid rate: `1.000`
- max opaque-payload accuracy minus target: `0.000`
- positive diagnostic-table comparator lift: minimum `+0.425`

## Rows

| Surface | Target | Matched packet | Max opaque payload | Opaque-target | Positive diagnostic-table lift |
|---|---:|---:|---:|---:|---:|
| core `n=8` label-blind | 0.250 | 0.250 | 0.250 | 0.000 | +0.425 |
| holdout `n=8` label-blind | 0.250 | 0.250 | 0.250 | 0.000 | +0.438 |

Opaque payloads include the 2-byte packet, matched-byte text, random same-byte
packet, deranged diagnostic table, query-aware diagnostic text, JSON diagnostic
text, free-text diagnostic text, and full hidden log. All collapse to target
accuracy when the public key-to-candidate table is removed.

## Interpretation

This weakens a leakage concern but narrows the claim. The positive endpoint
result is not caused by the model hallucinating answer labels from opaque
diagnostic strings, because the same strings fail when the public diagnostic
table and original labels are hidden. At the same time, the result confirms that
the current method requires target-side public side information. The paper
should present this as source-private side-information communication, not
protocol-free latent semantics.

## Next Gate

Run the same label-blind stress at `n=160` core + holdout with paired
uncertainty. If the goal is a deeper learned contribution, implement the shared
sparse crosscoder packet with atom knockout rather than further tuning the
diagnostic-table protocol.
