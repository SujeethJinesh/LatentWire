# Source-Private Candidate-Embedding Receiver

- date: `2026-04-29`
- artifact: `results/source_private_candidate_embedding_receiver_20260429/gated_budget4_seed29_30/`
- script: `scripts/run_source_private_candidate_embedding_receiver.py`
- test: `tests/test_run_source_private_candidate_embedding_receiver.py`
- scale rung: strict small learned-receiver smoke

## Purpose

The endpoint packet evidence is strong, but it still uses an explicit public
packet contract. This gate tests a less hand-designed target-side receiver: the
source encoder emits a rate-capped bit packet, and a learned candidate scorer
decodes it using public candidate features plus packet/candidate interactions.
A calibrated target-preservation margin gate chooses the target prior unless
the packet gives enough evidence to switch.

## Result

The 4-byte learned candidate-embedding receiver passes on the all-family
train/eval `768/512` seed `29 -> 30` surface:

| Budget | Pass | Matched | Target | Best destructive control | Delta target | Delta control | Full diagnostic oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 bytes | `True` | 0.748 | 0.250 | 0.262 | +0.498 | +0.486 | 0.998 |

Destructive controls remain near the target prior:

- zero-source: `0.250`
- shuffled-source: `0.250`
- answer-masked source: `0.221`
- random same-byte: `0.262`
- target-derived sidecar: `0.250`
- answer-only text bytes: `0.242`
- structured hidden-log prefix at the same byte budget: `0.203`
- wrong-projection source: `0.232`

The calibrated preservation threshold is `0.625476`; on calibration it kept
the best control at `0.271` while preserving matched accuracy `0.849`.

## Interpretation

This is a useful technical addition because it targets the hand-designed-decoder
critique directly. The receiver is still simple and trained on the same
synthetic source-private repair surface, but it is no longer just a deterministic
diagnostic-code lookup. It learns when to trust a compact source packet and when
to preserve the target prior.

This should be framed as a learned receiver smoke, not yet a headline method.
Before promotion, it needs seed repeats, held-out-family splits, paired
uncertainty, and comparison against the existing scalar WZ and endpoint packet
rows. The immediate value is that it gives the paper a credible path to a
third technical contribution: target-preserving learned decoding under
source-destroying controls.

## Next Gate

Run a 3-seed repeat at the 4-byte budget, then a held-out-family split. Keep the
same pass rule: matched packet must beat target and every destructive control by
at least `15` points, all destructive controls must stay within `target + 0.05`,
and the full diagnostic oracle must stay above `0.95`.
