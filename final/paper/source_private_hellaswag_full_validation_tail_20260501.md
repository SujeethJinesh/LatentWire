# HellaSwag Full-Validation Terminal-Tail Stress

## Status

The dense bagged hidden-innovation branch does not clear the full HellaSwag
validation set under the current strict gate. The method remains positive on
the terminal tail in aggregate, but the terminal slice fails the predeclared
jackknife robustness requirement.

Current paper story: the defensible HellaSwag headline is still the `2B` raw /
`5B` framed source-private hidden-innovation packet over validation `0:9216`.
The full-validation claim is explicitly blocked.

## Why This Gate Was Run

The previous artifact passed validation rows `0:9216`. This run tested the
remaining short terminal tail, rows `9216:10042`, with the same frozen method,
train samples, split seeds, packet bytes, and control ladder.

In lay terms: the tiny private hint still helped overall on the final examples,
but one robustness split was too uncertain. That means the result is promising
but not strong enough to call a full validation pass.

## Validation[9216:10042] Terminal Tail

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=false`.

- terminal rows: `826`
- selected accuracy: `0.539952`
- best label-copy: `0.498789`
- score-only bagged: `0.497579`
- delta vs best label-copy: `+0.041162`
- paired CI95 low vs best label-copy: `+0.012107`
- delta vs score-only: `+0.042373`
- wrong-example hidden: `0.484262`
- candidate-roll hidden: `0.433414`
- jackknife: `2/3`
- jackknife min delta vs best label-copy: `+0.020581`
- jackknife min CI95 low vs best label-copy: `-0.012167`
- packet: `2B` raw / `5B` framed

## Validation[0:10042] Aggregate

Artifact:
`results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_10042/hellaswag_hidden_innovation_multi_slice_stress.json`

Result: `pass_gate=false`.

- slices passing: `9/10`
- total eval rows: `10042`
- weighted selected accuracy: `0.526688`
- weighted best label-copy: `0.485162`
- weighted score-only: `0.480880`
- min delta vs best label-copy among passing slices: `+0.034180`
- min CI95 low vs best label-copy among passing slices: `+0.011719`
- contiguous validation prefix: `true`
- source-private packet: `true`

## Interpretation

Promoted:

1. The terminal tail is not a collapse: the method beats best label-copy and
   score-only controls overall, with a positive paired confidence interval.
2. The corrupted-hidden controls remain below the selected packet on the final
   slice.
3. The evaluation harness now records the exact full-validation failure instead
   of silently excluding the short tail.

Weakened:

1. HellaSwag is not a strict full-validation pass.
2. The current dense packet is not yet robust enough to survive every
   jackknife subbag on the short terminal slice.
3. The paper cannot claim that the HellaSwag method is fully saturated or
   benchmark-complete.

## Reviewer-Safe Claim

Safe claim: LatentWire passes a strict contiguous HellaSwag validation `0:9216`
stress with a fixed-byte source-private packet, and the terminal tail shows a
positive but insufficiently robust signal.

Do not claim: LatentWire passes the full HellaSwag validation set under the
strict jackknife gate.

## Four-Sample Stabilization Follow-Up

Follow-up artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042_4sample_scout/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=false`.

- train sample seeds: `4`
- selected accuracy: `0.530266`
- best label-copy: `0.498789`
- score-only bagged: `0.497579`
- delta vs best label-copy: `+0.031477`
- paired CI95 low vs best label-copy: `+0.002996`
- jackknife: `2/4`
- jackknife min delta vs best label-copy: `+0.015738`
- jackknife min CI95 low vs best label-copy: `-0.006689`

Interpretation: adding a fourth train-sample seed preserves a positive overall
terminal-tail margin, but weakens robustness and does not rescue the full
validation claim. The next method branch needs a new mechanism rather than a
larger version of the same dense bagging rule.

## Next Gate

The next highest-value branch is one of:

1. A predeclared terminal-tail stability rule that repeats the short-tail
   uncertainty estimate without changing the method, then accepts only if every
   recorded robustness split clears the same sign and confidence criteria.
2. A stronger sparse top-2 trust-or-switch repair packet trained on HellaSwag
   train rows and evaluated on the same frozen slices, keeping the label-copy,
   score-only, zero-hidden, wrong-hidden, candidate-roll, and same-byte text
   controls.

If neither clears, preserve HellaSwag `0:9216` as a strong but bounded result
and prioritize strict cross-family falsification plus native systems rows.
