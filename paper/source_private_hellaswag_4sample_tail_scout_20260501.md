# HellaSwag Four-Sample Terminal-Tail Scout

## Status

The four-train-sample dense hidden-innovation scout does not rescue the
HellaSwag terminal tail. The current defensible paper claim remains the
contiguous validation `0:9216` pass; full validation `0:10042` is still
blocked.

Current paper story: LatentWire has a `2B` raw / `5B` framed source-private
hidden-innovation packet that beats label-copy and score-only controls on
large frozen HellaSwag slices, but the terminal tail exposes jackknife
fragility that is not solved by simply adding another train-sample bag.

## Why This Gate Was Run

The prior terminal-tail run used three train-sample seeds and failed because
one leave-one-train-sample jackknife split had a non-positive paired confidence
bound. This run tested the cheapest stabilization hypothesis: add one more
independent train-sample seed while keeping the same packet budget, frozen
terminal-tail evaluation rows, and strict control ladder.

In lay terms: we asked whether one extra independent training view makes the
small private hint reliable on the final examples. It did not.

## Artifact

Primary artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042_4sample_scout/hellaswag_hidden_innovation_eval_slice_stress.json`

The run used explicit eval-cache reuse via `--eval-score-cache` and
`--eval-hidden-cache`; both eval cache metadata entries report cache hits
against the prior terminal-tail caches.

## Result

Result: `pass_gate=false`.

- terminal rows: `826`
- train sample seeds: `4`
- component models: `12`
- selected accuracy: `0.530266`
- best label-copy: `0.498789`
- score-only bagged: `0.497579`
- delta vs best label-copy: `+0.031477`
- paired CI95 low vs best label-copy: `+0.002996`
- delta vs score-only: `+0.032688`
- paired CI95 low vs score-only: `+0.012682`
- zero-hidden control: `0.497579`
- wrong-example hidden: `0.489104`
- candidate-roll hidden: `0.437046`
- jackknife: `2/4`
- jackknife min delta vs best label-copy: `+0.015738`
- jackknife min CI95 low vs best label-copy: `-0.006689`
- packet: `2B` raw / `5B` framed

## Comparison To The Three-Sample Tail

The three-sample terminal-tail run was stronger overall:

- selected accuracy: `0.539952`
- delta vs best label-copy: `+0.041162`
- paired CI95 low vs best label-copy: `+0.012107`
- jackknife: `2/3`
- jackknife min delta vs best label-copy: `+0.020581`
- jackknife min CI95 low vs best label-copy: `-0.012167`

Adding a fourth train sample preserved an overall positive margin, but lowered
selected accuracy, lowered the overall confidence margin, and left only `2/4`
jackknife rows passing. This weakens the "more bagging fixes the tail" branch.

## Decision

Promoted:

1. The terminal tail still contains real positive signal over score-only,
   zero-hidden, and label-copy controls.
2. Explicit eval-cache reuse is now supported, making future frozen-slice
   reruns cheaper and less error-prone.

Weakened:

1. Simple train-sample bag enlargement is not enough to claim full HellaSwag
   validation.
2. The terminal-tail failure is not just a one-missing-seed artifact.
3. A threshold fallback/margin guard is not a promising rescue unless it is
   reformulated with a new hidden-private decision signal; the local scratch
   guard effectively kept the current selected packet and did not improve the
   terminal-tail result.

Ruled out for now:

1. Claiming a full HellaSwag validation pass.
2. Spending more Mac cycles on larger versions of this same dense bagging
   branch without a new mechanism.

## ICLR And COLM Implication

For ICLR, the next exact gate should be a genuinely new method branch or a
native systems/cross-family gate, not another small dense-bag tweak. The best
method branch is a hidden-private top-2 trust-or-switch packet with the same
strict controls and a predeclared full-validation aggregation rule.

For COLM, this result is useful negative evidence: the paper can honestly show
that LatentWire passes `0:9216`, remains positive on the terminal tail, and
rejects an easy overclaim when robustness fails.
