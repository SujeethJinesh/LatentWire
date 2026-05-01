# HellaSwag Train Source-Score Repair Probe

Date: 2026-05-01

## Status

This branch is weakened, not promoted. A train-only source-score repair decoder
does not beat source-label copy on the frozen HellaSwag first-1024 validation
slice.

## Plain-Language Goal

The previous HellaSwag probes showed that the source model often had the right
answer in its top two guesses, but the simple confidence margin did not tell us
when to switch from the first guess to the second. This probe asked whether
that switch rule could be learned from source scores on HellaSwag training
examples.

In simple terms: we let the source model practice on 512 training questions,
learned what its wrong-answer score patterns looked like, then tested whether
that learned pattern could fix its validation mistakes. It could not.

## Protocol

Artifact:
`results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/`

Script:
`scripts/build_source_private_hellaswag_train_source_score_repair_probe.py`

Train rows scored:
`512` deterministic official HellaSwag train rows selected with seed `1729`.

Internal train split:
`384` fit rows and `128` dev rows. The repair policy is selected only on this
internal train/dev split.

Frozen evaluation:
`1024` official HellaSwag validation rows using the existing source-score cache.

Source model:
Qwen2.5-0.5B-Instruct choice log-likelihood scoring on Mac CPU.

## Readout

| Condition | Accuracy | Correct |
|---|---:|---:|
| source-label copy | `0.462` | `473/1024` |
| trained choice-bias label-copy control | `0.459` | `470/1024` |
| selected train-source-score repair | `0.447` | `458/1024` |

Selected policy:
`top2_margin_8bin`

Additional diagnostics:

- selected internal dev accuracy: `0.461`
- selected minus source-label copy: `-0.015`
- selected minus trained choice-bias label-copy: `-0.012`
- source top-2 oracle: `0.716`
- source top-4 oracle: `1.000`
- train source scoring latency on Mac CPU: `412.72s`
- raw payload: `2B`
- framed record: `5B`
- pass gate: `false`

## Interpretation

The source-score shape alone is not stable enough across train and validation
to identify the source model's top-choice mistakes. The top-2 oracle remains
large, so the target is not impossible; this specific signal is too weak.

This prunes the following branch:

- top-2 margin thresholding
- train-only score-shape bins
- trained option-bias label-copy
- full-rank score-shape bins at this small scored-train scale

It keeps alive a narrower next branch:

- source hidden-summary repair, where the packet carries a learned hidden-state
  or residual-code signal rather than only score shape
- a larger train-source-score cache only if we first define a better feature
  family than margin/entropy/rank bins

## Reviewer Boundary

Do not claim HellaSwag positive-method success from this row. The useful paper
role is reviewer-facing rigor: we tried the most obvious train-only score
calibration and it failed against the strict source-label-copy control.

## Next Gate

Move to a source-hidden-summary repair packet on a smaller cheap slice:

1. collect source hidden summaries for a bounded train/eval HellaSwag slice
2. train a repair code only on train labels
3. evaluate on frozen validation rows
4. require `>=494/1024` if using first-1024, or the corresponding `+0.02`
   paired lift over source-label copy on a smaller pre-registered slice
5. keep trained label-copy, same-byte text, shuffled source, candidate
   derangement, and metadata controls

If hidden summaries also fail, HellaSwag should stay a diagnostic and ICLR
should emphasize ARC/OpenBookQA plus train-donor generalization and native
systems rows.
