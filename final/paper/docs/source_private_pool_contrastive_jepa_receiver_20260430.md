# Source-Private Whole-Pool Contrastive JEPA Receiver

- date: `2026-04-30`
- gate: `source_private_pool_contrastive_jepa_receiver`
- artifact: `results/source_private_pool_contrastive_jepa_query_resampler_semantic_anchor_smoke_20260430/`
- status: implemented; strict smoke negative

## Question

Does training the query-resampler receiver on the full candidate pool solve the
failure mode of independent candidate-row training?

The previous trainable JEPA-Q variants scored candidate/source rows
independently. That is not the real deployment problem: the target sees a
candidate pool and must rank one candidate given a source-private packet. This
gate trains that exact decision surface with a whole-pool softmax loss.

## Implementation

I added `--receiver-mode jepa_query_resampler_pool_contrastive` to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`.

The receiver keeps the same packet-attention inference contract as the prior
JEPA-Q variants:

- candidate features generate query vectors;
- decoded packet atoms provide source-private keys and values;
- candidate-conditioned packet attention produces contexts;
- a learned output head scores the candidates.

The training objective changes:

- matched source packets train a whole-pool cross-entropy target on the answer
  candidate;
- shuffled-source packets train a target-prior-preservation target;
- atom-deranged and random same-byte packets also train target-prior
  preservation;
- a rank margin penalizes non-target candidates exceeding the intended target.

This directly tests whether the receiver can learn a source-safe decision rule
instead of relying on row-wise classification artifacts.

## Command

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_pool_contrastive_jepa_query_resampler_semantic_anchor_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 128 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler_pool_contrastive \
  --contrastive-negative-sources 2 \
  --jepa-query-count 8 --jepa-hidden-dim 16 \
  --jepa-train-epochs 40 --jepa-lr 0.01 --jepa-weight-decay 0.001 \
  --ridge 0.001 --top-k 8 --min-score 0.0 --min-decision-score 0.20
```

## Results

- pass rows: `0/6`
- max learned accuracy: `0.250`
- max learned-target lift: `+0.000`
- controls stay at target under the default gate
- query effective rank: `128`
- query entropy: about `1.3246`
- context variance: about `0.0050-0.0053`

Rows:

| Direction | Budget | Learned | Target | Best control | Oracle | CI95 low vs target | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| core -> holdout | 4 | 0.125 | 0.250 | 0.250 | 0.375 | -0.21875 | no |
| core -> holdout | 8 | 0.125 | 0.250 | 0.250 | 0.375 | -0.203125 | no |
| holdout -> core | 4 | 0.125 | 0.250 | 0.250 | 0.250 | -0.203125 | no |
| holdout -> core | 8 | 0.250 | 0.250 | 0.250 | 0.250 | -0.125 | no |
| same-family | 4 | 0.250 | 0.250 | 0.250 | 0.250 | -0.078125 | no |
| same-family | 8 | 0.1875 | 0.250 | 0.250 | 0.250 | -0.125 | no |

## Interpretation

Whole-pool training fixes the objective mismatch but does not recover a usable
source signal on the current semantic-anchor/JQK feature path. Like the
control-regularized row-wise variant, it keeps source-destroying controls flat
by suppressing the matched packet signal.

This is a clean falsification of the simplest "train on the actual candidate
pool" rescue. The learned receiver blocker is no longer just a row-wise loss
artifact.

## Decision

Do not promote. Prune the current JEPA-Q receiver family on this feature path:

- random-feature query resampler: partial/asymmetric;
- trainable row-wise query resampler: leaked through shuffled source;
- control-regularized row-wise query resampler: collapsed to target;
- whole-pool contrastive query resampler: collapsed below or at target.

The next high-yield method branches are:

1. stronger frozen LLM/activation features under the same whole-pool objective;
2. compression-native packets such as rotation-sign or product-codebook packets;
3. a less hand-coded source-private task surface using private retrieval/tool
   traces rather than ontology atoms.
