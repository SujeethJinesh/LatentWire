# Learned Anchor-Relative Receiver Ablation

- date: `2026-04-30`
- gate: `source_private_candidate_embedding_receiver_20260430`
- status: negative / prune this adjacent receiver variant for now

## Question

Can a less hand-symbolic receiver replace the semantic-anchor lexicon by
building a public learned anchor basis from training candidate embeddings, then
communicating an 8-byte source packet decoded against target-side candidate
side information?

## Implementation

I added `--packet-feature-mode learned_anchor_relative` to
`scripts/run_source_private_candidate_embedding_receiver.py`. The mode builds a
deterministic spherical-k-means anchor basis from public training candidates:

1. hash candidate text into the existing embedding surface,
2. initialize public anchors by deterministic farthest-first selection,
3. run fixed-iteration spherical k-means,
4. encode candidates by cosine similarity to those public anchors,
5. evaluate with the existing target-preserving receiver and destructive
   controls.

This is deliberately not allowed to see answer labels from the evaluation
examples.

## Commands

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py \
  --output-dir results/source_private_candidate_embedding_receiver_20260430/heldout_core_to_holdout_learned_anchor_relative_code_similarity_budget8_seed29_30 \
  --train-examples 768 --eval-examples 512 \
  --train-family-set core --eval-family-set holdout \
  --feature-dim 512 --candidate-feature-dims 0 \
  --receiver-kind code_similarity \
  --packet-feature-mode learned_anchor_relative --anchor-count 128 \
  --budgets 8 --train-seed 29 --eval-seed 30 --ridge 1e-2

PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py \
  --output-dir results/source_private_candidate_embedding_receiver_20260430/heldout_core_to_holdout_learned_anchor_relative_ridge_budget8_seed29_30 \
  --train-examples 768 --eval-examples 512 \
  --train-family-set core --eval-family-set holdout \
  --feature-dim 512 --candidate-feature-dims 0 \
  --receiver-kind ridge \
  --packet-feature-mode learned_anchor_relative --anchor-count 128 \
  --budgets 8 --train-seed 29 --eval-seed 30 --ridge 1e-2
```

## Results

| Receiver | Pass | Matched | Target | Best destructive | Delta target | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| code similarity | `False` | 0.250 | 0.250 | 0.268 | 0.000 | 0.723 |
| ridge | `False` | 0.250 | 0.250 | 0.336 | 0.000 | 0.516 |

Exact ID parity holds in both runs.

## Interpretation

This ablation weakens the hypothesis that public anchor geometry alone explains
or replaces the semantic-anchor receiver. The code-similarity run has mostly
clean controls but insufficient candidate-map headroom. The ridge run has lower
oracle headroom and visible control leakage (`answer_only` reaches `0.336`).

Decision: do not spend another adjacent cycle on this embedding receiver family
without a genuinely new signal, such as frozen LLM embeddings, activation
features, or a contrastive source-control objective. The live positive story
remains semantic-anchor packets plus strict systems/rate assumptions, with the
learned/frozen latent receiver still open.
