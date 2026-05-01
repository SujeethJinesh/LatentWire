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
- low thresholds recover matched signal but leak destructive controls;
- high thresholds clean controls but collapse the hard direction to target.

## Interpretation

The live method is not merely threshold luck. It has a narrow but real clean
operating band where source-private packets move the target decision while fake
source packets remain at target. The lower threshold rows explain why
thresholding is necessary; the higher threshold rows explain why the band is
not arbitrary.

## Lay Explanation

The experiment asks: if we turn the model's confidence knob up and down, is
there a setting where the real private hint helps but fake hints do not? For
the live residual packet, yes. For the randomized sign sketch and RR variants,
no.

## Next Gate

Use this threshold frontier in the ICLR evidence bundle and paper figure/table.
The next method gate should be a true candidate-conditioned residual code with
control regularization, not another public random-basis scorer.
