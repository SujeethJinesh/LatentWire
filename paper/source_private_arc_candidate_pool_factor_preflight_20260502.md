# ARC Candidate-Pool Factor Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, public-basis/residual
  gates, and systems byte/exposure accounting remain the defensible core.
- Exact gap: a positive source-necessary learned connector is still missing.

## Gate

Updated script:
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

New source feature modes:

- `cached_choice_score_pool`
- `cached_choice_score_pool_residual`
- `hf_choice_hidden_candidate_pool`
- `hf_choice_hidden_candidate_pool_residual`
- `hf_choice_hidden_score_candidate_pool`
- `hf_choice_hidden_score_candidate_pool_residual`

These modes create a candidate-level source pool rather than a selected-vector
or raw-token pool. The hidden modes regroup answer-key-forbidden source hidden
means by answer choice, optionally row-center them, and feed the candidate set
to the learned query soft-prefix connector. The score modes use the existing
cached source selected-choice pattern as a one-hot candidate feature. The
combined mode concatenates hidden residuals with that cached source-selection
feature.

## Artifacts

- score-only:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_cached_score_pool_residual_n8_cpu_label_choice/`
- hidden-only:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_candidate_pool_residual_n8_cpu_label_choice/`
- hidden+score:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_candidate_pool_residual_n8_cpu_label_choice/`

All runs use the same ARC n8 CPU smoke surface, one target-loss epoch,
`label_and_choice` continuations, prefix length `2`, hidden dim `8`, and
source pool size `4`.

## Result

| Mode | Pass | Matched | Best Control | Target Only | Zero Source | Label Shuffled | Matched Minus Best-Control Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached score residual | `False` | `1/4` | `2/4` zero-source | `1/4` | `2/4` | `0/4` | `-1.537` |
| hidden candidate residual | `False` | `1/4` | `1/4` target-only | `1/4` | `1/4` | `0/4` | `-0.184` |
| hidden+score candidate residual | `False` | `0/4` | `2/4` target-cache-only | `1/4` | `1/4` | `1/4` | `-1.664` |

The source-label-copy audit upper bound remains `3/4` on this slice, which
confirms that the cached source selection contains useful answer-side signal,
but the learned query soft-prefix connector does not convert it into a
source-necessary target gain under controls.

## Decision

Rule out shallow candidate-level query soft-prefix pooling on this Mac-local
n8 gate. Candidate hidden residuals are less bad than raw token pooling and
score-combined pooling, but they only tie target-only and lose on margin. The
cached source-choice score feature is mostly a label-copy stress test, and
adding it to hidden residuals hurts rather than helps.

The next live branch should be conditional/syndrome-style packetization:

- learn or construct a compact candidate residual packet relative to the
  target's own public candidate state;
- compare hidden-only, score-only, and score+hidden factors at matched rate;
- include an explicit target-derived packet and shuffled-source packet control;
- only run larger n64/seed gates if n8 matched clears zero-source and
  label-shuffled controls.

## Lay Explanation

This run stopped giving the receiver one chosen-answer clue or a bag of raw
tokens. Instead, it gave one small feature per answer choice and let the
receiver learn which answer-choice features to attend to. It still did not
work. The source-choice signal clearly has answer information, but the current
soft-prefix translator cannot use it in a way that beats simple controls.
