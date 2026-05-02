# ARC Public-Innovation Soft-Prefix Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: LatentWire has a rigorous fixed-byte/source-private transfer
  scaffold, destructive controls, public-basis diagnostics, and systems
  byte/exposure accounting.
- Exact gap: the current learned connector still does not show source
  necessity against target-only, target-cache-only, zero-source, shuffled,
  noise, train-mean, label-shuffled, candidate-deranged, and same-byte visible
  text controls.

## What Changed

Added train-only public-side-information innovation modes to
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`:

- `hf_choice_hidden_public_innovation_candidate_pool`
- `hf_choice_hidden_public_innovation_candidate_pool_residual`
- `hf_choice_hidden_score_public_innovation_candidate_pool`
- `hf_choice_hidden_score_public_innovation_candidate_pool_residual`

The source packet now computes source candidate hidden features, predicts those
features from public question-choice candidate hashes using only the fit rows,
and passes the residual candidate pool to the existing query soft-prefix
connector. The hidden-only innovation path does not read cached source selected
choices. The hidden+score path explicitly includes the cached source selected
choice as a separate score factor.

Lay explanation: the run asks whether the source model knows something beyond
what the target can already guess from the question and answer choices. We
subtract the public guess first, then send only the leftover source signal.

## Evidence

All runs use the same ARC n8 CPU `label_and_choice` smoke surface with 4 fit
rows and 4 eval rows.

| Mode | Matched | Best Control | Margin Delta | Pass |
|---|---:|---|---:|---|
| hidden public innovation residual | `1/4` | shuffled-source `2/4` | `-0.900` | `False` |
| hidden+score public innovation residual, ridge 10 | `2/4` | target-cache-only `2/4` | `+0.051` | `False` |
| hidden+score public innovation residual, ridge 10, target-conditioned query | `1/4` | target-only `1/4` | `-1.007` | `False` |
| hidden+score public innovation residual, ridge 1000 | `1/4` | target-cache-only `2/4` | `-0.863` | `False` |

Artifacts:

- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_public_innovation_candidate_pool_residual_n8_cpu_label_choice/`
- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n8_cpu_label_choice/`
- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n8_cpu_label_choice_target_query/`
- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n8_cpu_label_choice_ridge1000/`

## Interpretation

This is not a positive-method result. The best row is the ridge-10
hidden+score public-innovation packet, which ties the target-cache-only learned
prefix at `2/4` and improves mean margin by `0.051`, but it does not separate
accuracy from the target-cache baseline or same-norm noise. Hidden-only
innovation is prediction-free but loses to shuffled-source. Target-conditioned
queries and stronger ridge regularization both worsen the matched row.

The fit diagnostic is also important: ridge 10 explains `0.9985` of fit hidden
variance from only 16 fit candidates, so the public predictor is probably
overfitting the tiny fit surface. Raising ridge to 1000 lowers fit explained
variance to `0.6295`, but matched accuracy drops to `1/4`.

## Decision

Rule out train-only public-ridge innovation as a standalone soft-prefix
positive method on this Mac-local gate. Keep the conditional-innovation
framing, but move away from answer-CE-only soft-prefix training. The next exact
gate should be a source-control-contrastive conditional innovation receiver:
cache source/public candidate features once, train the receiver to improve
matched packets while penalizing zero-source, shuffled-source, same-norm-noise,
and candidate-rolled packets, then run the same n8 decision surface before any
n32 widening.

This branch is still useful for the paper because it sharpens the uniqueness
boundary: LatentWire is not just prefix tuning or raw hidden transfer; it is
trying to prove source-private conditional packets under destructive controls.
That proof is not yet achieved.
