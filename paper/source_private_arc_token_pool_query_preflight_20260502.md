# ARC Token-Pool Query Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, public-basis/residual
  gates, and systems byte/exposure accounting are the defensible core.
- Exact gap: a positive source-necessary learned connector is still missing.

## Gate

Updated script:
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

New source feature modes:

- `hf_choice_token_hidden_pool`
- `hf_choice_token_hidden_pool_residual`

The token-pool mode extracts answer-key-forbidden source hidden states from all
candidate continuations, packs them into a fixed-size per-row token pool, and
feeds the pool to a learned query soft-prefix connector. The residual mode
row-centers token features before normalization. The target model remains
frozen, and the same source-destroying controls are retained.

Artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_token_pool_residual_n8_cpu_label_choice/`

## Result

Qwen source token-pool residual ARC n8 CPU smoke, `label_and_choice`
continuation:

- pass gate: `False`
- fit/eval rows: `4 / 4`
- matched query soft-prefix accuracy: `0.000`
- target-only accuracy: `0.250`
- target-cache-only prefix accuracy: `0.250`
- slots-only/static prefix accuracy: `0.250`
- zero-source accuracy: `0.250`
- shuffled-source accuracy: `0.000`
- same-norm noise accuracy: `0.000`
- train-mean source accuracy: `0.250`
- label-shuffled accuracy: `0.500`
- same-byte visible text accuracy: `0.250`
- source-label-copy audit upper bound: `0.750`
- matched mean margin: `-0.689839`
- best pass-control margin: `-0.124259`
- matched minus best-control margin: `-0.565581`
- source token pool size: `8`
- token count min/mean/max before fixed pooling: `12 / 26.875 / 34`
- suffix fallback choice count: `4`
- runtime: about `77.6s`
- peak RSS: about `7.1 GiB`

## Decision

Rule out this shallow token-pool residual query connector on the current ARC
n8 Mac-local surface. It is worse than the selected residual vector gate and
does not beat target-only, zero-source, train-mean source, or label-shuffled
controls. This does not kill learned query connectors in general, but it
weakens the hypothesis that simply exposing more source hidden tokens is
sufficient.

The next exact branch should be narrower and more diagnostic:

- candidate-level all-choice residual query pooling, not raw token pooling;
- explicit source-score-only versus hidden-residual versus score+hidden
  ablation;
- train-only selection of continuation mode and connector capacity;
- n64 and three seeds only if the n8 smoke beats label-shuffled and zero-source
  controls;
- native NVIDIA for any larger target-forward loop.

## Lay Explanation

The previous run gave the receiver one vector describing the source model's
chosen answer. This run gave it a small pile of hidden-state tokens from all
answer choices and let the receiver learn how to attend to them. It still did
not help. A deliberately wrong-label training control did better, which means
this connector is not yet learning useful model-to-model communication.
