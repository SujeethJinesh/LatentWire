# Source-Private Candidate-Local Threshold Frontier, 2026-05-01

## Status

- paper readiness: improves COLM and ICLR reviewer evidence, but does not close
  native systems or broader-benchmark gaps.
- artifact:
  `results/source_private_candidate_local_threshold_frontier_20260501/`
- code:
  `scripts/build_source_private_candidate_local_threshold_frontier.py`
- test:
  `tests/test_build_source_private_candidate_local_threshold_frontier.py`
- references:
  `references/554_candidate_local_threshold_frontier_refs_20260501.md`

## Question

Does the live candidate-local residual receiver pass because of a lucky fixed
threshold, or does it have a robust clean decision band where matched packets
help and destructive controls remain near target?

This diagnostic replays stored `predictions_budget8.jsonl` per-candidate
scores across thresholds without rerunning the model. A threshold row is clean
only when every replayed direction satisfies:

- matched accuracy is at least `0.15` above target;
- matched accuracy is at least `0.10` above the best destructive control;
- every strict destructive control is within `target + 0.03`.

This is intentionally stricter about control separation but lighter than the
original pass gate because it does not recompute bootstrap intervals or
knockout reductions.

## Result

The live candidate-local residual receiver has a clean all-row threshold band:

- live clean threshold range: `0.45-0.48`;
- at threshold `0.48`: `9/9` rows clean;
- minimum matched accuracy at `0.48`: `0.500`;
- maximum best destructive-control accuracy at `0.48`: `0.260`;
- minimum matched-control gap at `0.48`: `0.240`.

By contrast:

- RR anchor-coordinate dot product has no all-row clean threshold;
- public random-rotation sign sketch has no all-row clean threshold;
- at low threshold, random-rotation sign recovers matched signal but destructive
  controls rise too;
- at high threshold, random-rotation sign controls become clean but
  holdout-to-core collapses to target.

Representative replay rows:

| method | threshold | clean rows | min matched | max best control | min matched-control |
|---|---:|---:|---:|---:|---:|
| live candidate-local residual norm | `0.00` | `0/9` | `0.625` | `0.625` | `0.250` |
| live candidate-local residual norm | `0.30` | `6/9` | `0.500` | `0.375` | `0.230` |
| live candidate-local residual norm | `0.48` | `9/9` | `0.500` | `0.260` | `0.240` |
| live candidate-local residual norm | `0.60` | `6/9` | `0.250` | `0.258` | `-0.008` |
| RR anchor-coordinate dot | `0.48` | `6/9` | `0.250` | `0.264` | `-0.014` |
| public random-rotation sign | `0.48` | `2/3` | `0.250` | `0.250` | `0.000` |

## Interpretation

The live method is not merely threshold luck. It has a narrow but real clean
operating band where source-private packets move the target decision while fake
source packets remain at target. The lower threshold rows explain why
thresholding is necessary: if the receiver accepts too many packet decisions,
some destructive controls become useful. The higher threshold rows explain why
the band is not arbitrary: if the receiver becomes too conservative, the hard
cross-family direction falls back to target.

The diagnostic also explains why public random bases and RR are not promoted:
they either have no threshold that works for all directions, or they trade
matched signal for control leakage.

## Lay Explanation

The experiment asks: if we turn the model's confidence knob up and down, is
there a setting where the real private hint helps but fake hints do not? For
the live residual packet, yes. For the randomized sign sketch and RR variants,
no.

## Next Gate

Use this threshold frontier in the ICLR evidence bundle and paper figure/table.
The next method gate should be a true candidate-conditioned residual code with
control regularization, not another public random-basis scorer.
