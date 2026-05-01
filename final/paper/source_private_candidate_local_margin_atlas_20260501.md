# Source-Private Candidate-Local Margin Atlas, 2026-05-01

## Status

- paper readiness: strengthens COLM and ICLR reviewer evidence, but does not
  close the native NVIDIA/vLLM systems gap or the broader benchmark gap.
- artifact:
  `results/source_private_candidate_local_margin_atlas_20260501/`
- code:
  `scripts/build_source_private_candidate_local_margin_atlas.py`
- test:
  `tests/test_build_source_private_candidate_local_margin_atlas.py`
- references:
  `references/555_candidate_local_margin_atlas_refs_20260501.md`

## Question

Does the live candidate-local residual packet win with a real score margin, or
is it only winning by fragile argmax flips? Do common-basis and random-basis
alternatives show the same control-separated margin behavior?

The atlas reads stored `predictions_budget8.jsonl` rows and computes, for each
example, the gold-candidate margin, winner margin, prior margin, answer rank,
and the threshold-`0.48` decision. It then aggregates those margins across the
live method, RR anchor coordinates, Procrustes common-basis transfer, and the
public random-rotation sign sketch.

## Result

The margin atlas passes its gate.

- live matched positive-margin rate: `0.750`;
- live best destructive-control positive-margin rate: `0.375`;
- live matched p50 margin: `0.243`;
- live oracle positive-margin rate: `0.875`;
- Procrustes matched/control positive-margin rates: `0.750` / `0.750`;
- RR matched positive-margin rate: `0.688`;
- public random-rotation sign matched positive-margin rate: `0.750`.

The important reviewer-facing detail is not that live is perfect. It is that
the live matched packet has a much healthier margin profile than the strict
destructive controls, while Procrustes has essentially the same margin profile
for matched packets and the permuted-teacher destructive control. That makes
Procrustes a useful common-basis falsification row rather than a substitute for
the source-private packet.

## Lay Explanation

The experiment asks whether the real private hint moves the right answer ahead
of the wrong answers by a meaningful amount. It is like checking whether a race
was won by several steps or by a photo finish. The live packet usually gives
the right answer a real lead; fake packets do not. Procrustes also gives a
lead, but it gives the same lead when the teacher is deliberately mismatched,
so it fails the privacy/control test.

## Interpretation

Promoted:

- source-private candidate-local residual packet as the current live positive
  method;
- margin atlas as a paper figure/table that explains confidence, headroom, and
  control failure modes;
- Procrustes/common-basis rows as useful falsification rows.

Weakened:

- public random-basis sign sketch as a headline method, because its matched
  margin can look healthy while threshold and destructive-control gates still
  fail;
- generic common-basis transfer as a novelty claim, because Procrustes leaks
  through the permuted-teacher control.

Still alive:

- candidate-conditioned residual code with control regularization. The margin
  atlas gives the exact score surface that the next learned packet should
  optimize: improve matched margins while keeping all source-destroying
  controls near target.

## Next Gate

Use this atlas in the top-level ICLR evidence bundle and paper figure/table.
The next method branch should train or select a candidate-conditioned residual
code against the margin objective and destructive controls, instead of trying
another generic public random projection.
