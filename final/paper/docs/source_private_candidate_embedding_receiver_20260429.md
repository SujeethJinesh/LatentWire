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

## Initial Result

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

## Mechanism Readout

This is a useful technical addition because it targets the hand-designed-decoder
critique directly. The receiver is still simple and trained on the same
synthetic source-private repair surface, but it is no longer just a deterministic
diagnostic-code lookup. It learns when to trust a compact source packet and when
to preserve the target prior.

## Seed and Held-Out Follow-Up

Artifacts:

- `results/source_private_candidate_embedding_receiver_20260429/multiseed_and_heldout_summary.json`
- `results/source_private_candidate_embedding_receiver_20260429/multiseed_and_heldout_summary.md`

Same-distribution seed repeats show that `4` bytes is promising but unstable:

| Budget | Pass count | Matched mean | Matched min | Max destructive | Min delta control |
|---:|---:|---:|---:|---:|---:|
| 4 bytes | 2/3 | 0.589 | 0.328 | 0.281 | 0.049 |
| 8 bytes | 3/3 | 0.749 | 0.514 | 0.283 | 0.230 |

The `8`-byte learned receiver is therefore the first seed-stable version of
this contribution on the all-family `768/512` surface. It still is not
cross-family stable: training on core families and evaluating on holdout
families at `8` bytes gives matched accuracy `0.453`, target-only `0.250`,
best destructive control `0.311`, and full diagnostic oracle `0.809`, so the
held-out gate fails. A no-candidate-feature invariant ablation is worse at
`n=256`: matched `0.332`, target `0.250`, best destructive `0.309`, oracle
`0.742`.

## Interpretation

This should be framed as a learned receiver contribution with a clear boundary:
it is same-distribution seed-stable at `8` bytes, but not yet a general
cross-family receiver. The failure is useful because it separates the learned
decoder claim from the stronger endpoint packet claim. The next method should
use family-invariant anchor-relative/codebook features or fold-heldout
calibration rather than relying on raw candidate coordinate features.

## Next Gate

Promote only after a held-out-family receiver clears the same pass rule:
matched packet must beat target and every destructive control by at least `15`
points, all destructive controls must stay within `target + 0.05`, and the full
diagnostic oracle must stay above `0.95`. The highest-value next receiver is an
anchor-relative or fold-heldout-calibrated packet decoder at `8` bytes.
