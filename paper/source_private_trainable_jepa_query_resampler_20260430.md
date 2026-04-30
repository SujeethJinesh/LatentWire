# Source-Private Trainable JEPA Query Resampler

- date: `2026-04-30`
- gate: `source_private_trainable_jepa_query_resampler`
- artifact: `results/source_private_trainable_jepa_query_resampler_semantic_anchor_smoke_20260430/`
- status: implemented; smoke negative because source-control leakage explains
  the strongest gain

## Question

Does training query/key/value packet-attention factors end-to-end improve over
the random-feature JEPA/Q-Former-style resampler without violating
source-destroying controls?

## Implementation

I added `--receiver-mode jepa_query_resampler_trainable` to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`.

The mode keeps the same inference contract as the random-feature
`jepa_query_resampler`:

- candidate features generate query vectors;
- decoded packet atoms provide source-private keys and values;
- masked attention produces query contexts;
- a learned output head scores each candidate.

The difference is that query factors, atom keys, atom values, output head, and
bias are trained with CPU Torch using matched-source positives and shuffled
source negatives, then converted back to NumPy arrays for deterministic scoring.

## Command

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_trainable_jepa_query_resampler_semantic_anchor_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 128 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler_trainable \
  --contrastive-negative-sources 2 \
  --jepa-query-count 8 --jepa-hidden-dim 16 \
  --jepa-train-epochs 40 --jepa-lr 0.01 --jepa-weight-decay 0.001 \
  --ridge 0.001 --top-k 8 --min-score 0.0 --min-decision-score 0.20
```

## Results

- pass rows: `0/6`
- best learned accuracy: `0.625`
- best learned-target lift: `+0.375`
- query effective rank: `119-121`
- query entropy: about `1.32`
- context variance: about `0.003`

Rows:

| Direction | Budget | Learned | Target | Best control | Best control name | Oracle | Read |
|---|---:|---:|---:|---:|---|---:|---|
| core -> holdout | 4 | 0.250 | 0.250 | 0.250 | zero_source | 0.375 | no lift |
| core -> holdout | 8 | 0.250 | 0.250 | 0.250 | zero_source | 0.625 | no lift |
| holdout -> core | 4 | 0.625 | 0.250 | 0.375 | shuffled_source | 0.250 | leaked/asymmetric |
| holdout -> core | 8 | 0.500 | 0.250 | 0.625 | shuffled_source | 0.375 | control explains gain |
| same-family | 4 | 0.375 | 0.250 | 0.250 | zero_source | 0.375 | weak partial signal |
| same-family | 8 | 0.375 | 0.250 | 0.250 | zero_source | 0.500 | weak partial signal |

## Interpretation

Training the resampler factors changes the behavior and recovers stronger
matched signal than the first random-feature JEPA-Q smoke in one direction, but
it is not source-safe. The strongest held-out row is contaminated by
shuffled-source improvement, and the 8-byte row is completely explained by the
shuffled-source control. Core -> holdout remains dead.

This is useful because it narrows the blocker:

- the query-resampler architecture can learn something nontrivial;
- the current objective does not sufficiently penalize source-control leakage;
- oracle/headroom is still weak in the successful direction;
- threshold tuning would be misleading.

## Decision

Do not promote. Keep as a learned-connector baseline and prune this exact
objective unless the next variant adds explicit control regularization:

- train on zero-source, shuffled-source, random-same-byte, and atom-deranged
  negatives, not only shuffled-source;
- add a target-preservation/control-matching penalty that pulls destructive
  controls back to target-only scores;
- only continue if both held-out directions beat target while all controls stay
  within target `+0.03`.
