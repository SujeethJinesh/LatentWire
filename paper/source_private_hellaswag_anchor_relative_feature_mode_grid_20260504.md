# HellaSwag Anchor-Relative Feature-Mode Grid

Date: 2026-05-04

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked by a learned positive method that survives strict held-out
  controls and by native systems rows.
- Current story: dense HellaSwag hidden-innovation packets remain the strongest
  hard-surface branch, but shallow common-basis compression still fails.
- Exact gap: a source-conditioned common-basis or syndrome packet must beat
  packet-only/label-copy/score-only controls with paired uncertainty and
  destructive source controls.

## Lay Explanation

The dense hidden packet can use a source model's hidden-state clue to improve a
HellaSwag answer. This experiment asked whether we can translate that clue into
a more shared coordinate language by measuring it against a bank of anchor
vectors. We tried sharper anchor features: keep only the closest anchors, turn
similarities into RBF/kernel scores, or concatenate cosine and RBF views.

All of those versions lost the useful clue. They mostly collapsed back to the
source-label or score-only baseline.

## Code Change

Extended:

`scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py`

New opt-in anchor feature modes:

- `cosine` (old default);
- `rbf`;
- `cosine_rbf`;
- `topk_cosine`;
- `topk_rbf`.

The default remains `cosine` for reproducibility of earlier artifacts.

New tests:

`tests/test_build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py`

The tests check that top-k RBF features are sparse/bounded, invalid modes raise,
and the mocked full gate records the selected feature mode.

## Gate

Frozen evaluation slice:

`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/`

New artifacts:

- `results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260504_qwen05_train512_validation1024_2048_topk_rbf/`
- `results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260504_qwen05_train512_validation1024_2048_topk_cosine/`
- `results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260504_qwen05_train512_validation1024_2048_cosine_rbf/`

All runs use `128` train-only anchors, `3` train-sample seeds, `3` split seeds,
`9` component models, and the same `2B` raw / `5B` framed packet contract.

## Results

| Feature mode | Selected acc. | Best label-copy | Score-only | Zero-hidden | Delta vs label | CI95 low vs label | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| `topk_rbf` | 0.409180 | 0.414062 | 0.409180 | 0.409180 | -0.004883 | -0.022461 | false |
| `topk_cosine` | 0.411133 | 0.414062 | 0.409180 | 0.409180 | -0.002930 | -0.020508 | false |
| `cosine_rbf` | 0.412109 | 0.414062 | 0.409180 | 0.409180 | -0.001953 | -0.020508 | false |

All jackknife summaries have `0/3` passing subbags.

## Interpretation

This demotes the obvious top-k/RBF anchor-relative repair. The previous
anchor-relative common-basis result was not merely hurt by diffuse cosine
features; sharpening the anchor chart still removes the useful dense hidden
innovation signal.

The useful source signal appears to live in task-specific dense residual
directions that the tested anchor bank does not preserve. The next branch should
therefore stop retuning shallow anchor features and instead test a conditional
source-syndrome receiver:

- source sends small candidate-pair innovation/parity bits or sparse residual
  atom bits;
- target uses its own candidate scores as decoder side information;
- receiver applies one to three bounded denoising/update steps;
- wrong-source, zero-syndrome, random-syndrome, candidate-roll, same-byte text,
  source-rank, and source-score quantization controls must be present.

## Decision

Demote top-k/RBF anchor-relative common-basis features as a current positive
method branch. Keep the result as a reviewer-facing negative ablation against
the claim that anchor coordinates alone solve cross-model latent communication.
