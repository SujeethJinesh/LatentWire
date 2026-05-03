# HellaSwag Hidden-Innovation Rank/Score-Channel Controls

Date: 2026-05-03

## Status

This is a strengthened positive-method control on the current best HellaSwag
branch, not an ICLR-ready final result. It hardens the first frozen validation
slice against the reviewer concern that the hidden-innovation packet may simply
copy source rank, candidate index, or source score metadata.

## Artifact

- script:
  `scripts/build_source_private_hellaswag_hidden_innovation_bagged_gate.py`
- result:
  `results/source_private_hellaswag_hidden_innovation_bagged_gate_20260503_rank_score_channel_controls_qwen05_train512_validation1024/`
- JSON:
  `hellaswag_hidden_innovation_bagged_gate.json`
- tests:
  `tests/test_build_source_private_hellaswag_hidden_innovation_bagged_gate.py`
  and
  `tests/test_build_source_private_hellaswag_hidden_innovation_multi_slice_stress.py`

## What changed

The bagged HellaSwag gate now trains and reports:

- a source-rank/index-only control bank, using rank position and candidate id
  but no source score magnitudes and no hidden innovation;
- a score-channel-roll hidden control, which keeps the hidden residuals but
  rolls the source score/rank feature channel across candidates;
- paired uncertainty versus the source-rank/index-only control;
- jackknife minima for the new source-rank/index-only control;
- stricter pass rules requiring the selected hidden-innovation receiver to
  beat source-rank/index-only, score-only, best label-copy, and zero-hidden,
  while wrong-row, candidate-roll, and score-channel-roll controls remain
  below label-copy.

The eval-slice and multi-slice aggregators were also updated so future
multi-slice claims require these strict rank/score-channel controls on every
included slice.

## Main result

On the frozen HellaSwag validation first-1024 slice:

| Row | Accuracy |
| --- | ---: |
| selected hidden-innovation packet | `0.512695` |
| best label-copy | `0.463867` |
| source-label copy | `0.461914` |
| source-rank/index-only bagged control | `0.461914` |
| score-only bagged control | `0.461914` |
| zero-hidden control | `0.461914` |
| wrong-example hidden control | `0.437500` |
| candidate-roll hidden control | `0.389648` |
| score-channel-roll hidden control | `0.252930` |

Selected minus controls:

- versus best label-copy: `+0.048828`, CI95 low `+0.026367`;
- versus source-rank/index-only: `+0.050781`, CI95 low `+0.033203`;
- versus score-only: `+0.050781`, CI95 low `+0.031250`;
- versus zero-hidden: `+0.050781`;
- versus score-channel-roll hidden: `+0.259766`.

Jackknife over the three train-sample bags also passes:

- pass count: `3/3`;
- min delta versus best label-copy: `+0.032227`;
- min CI95 low versus best label-copy: `+0.010742`;
- min delta versus source-rank/index-only: `+0.033203`;
- min CI95 low versus source-rank/index-only: `+0.014648`;
- max score-channel-roll hidden control: `0.270508`.

## Interpretation

The new source-rank/index-only control matching the score-only/zero-hidden
accuracy means candidate rank and candidate id alone are not enough to explain
the positive row. The score-channel-roll hidden control collapses far below
label-copy, which suggests the receiver uses aligned score/rank metadata and
hidden innovation together; hidden residuals alone, or score metadata attached
to the wrong candidate, do not preserve the win.

This is stronger than the previous HellaSwag memo, but it is still only the
first frozen slice under the new controls. The old multi-slice artifacts remain
valuable evidence for slice robustness under the older control set, but they
should not be cited as strict rank/score-channel-control evidence until rerun.

## Lay explanation

This test asks whether the small source message helps because it carries useful
hidden evidence, or whether it only smuggles in simple answer-choice hints like
"the source liked choice B." The stricter controls show that simple rank/choice
hints do not explain the gain on this slice. When the score hints are attached
to the wrong answer choices, performance falls sharply, so the useful signal is
not just generic hidden noise.

## Decision

Promote HellaSwag hidden-innovation to the strongest current positive branch,
but do not call it ICLR-ready. The next exact gate is to rerun the strict
rank/score-channel controls on the held-out HellaSwag validation slices
`1024:2048` through the terminal tail and rebuild the multi-slice aggregate.

If the strict multi-slice gate passes, the next paper-critical gate is
receiver-family/cross-family separation. If it fails, pivot immediately to the
ARC sparse common-feature innovation packet recommended by the literature and
code-scout subagents.
