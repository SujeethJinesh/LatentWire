# Source-Private Control-Regularized JEPA Query Resampler

- date: `2026-04-30`
- gate: `source_private_control_regularized_jepa_query_resampler`
- primary artifact: `results/source_private_control_regularized_jepa_query_resampler_semantic_anchor_smoke_20260430/`
- threshold diagnostic artifact: `results/source_private_control_regularized_jepa_query_resampler_semantic_anchor_smoke_lowthreshold_20260430/`
- status: implemented; strict smoke negative

## Question

Can a trainable JEPA/Q-Former-style query resampler keep the asymmetric source
signal from the previous trainable receiver while explicitly suppressing
destructive controls?

The prior trainable JEPA-Q smoke recovered holdout -> core signal, but
shuffled-source controls explained the strongest rows. This gate makes those
controls part of the training objective rather than treating them only as
post-hoc diagnostics.

## Implementation

I added `--receiver-mode jepa_query_resampler_control_regularized` to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`.

The receiver keeps the same inference contract as the trainable JEPA-Q
receiver:

- candidate features generate query vectors;
- decoded packet atoms provide source-private keys and values;
- masked attention produces candidate-conditioned query contexts;
- a learned output head scores each candidate.

The new objective adds explicit destructive-control negatives:

- shuffled source packets, controlled by `--contrastive-negative-sources`;
- atom-deranged packets;
- random same-byte atom packets.

Control negatives are upweighted and a pairwise margin penalizes negative
control scores that approach matched-source positive scores. The trained
Torch factors are still exported to NumPy for deterministic artifact scoring.

## Commands

Primary strict smoke:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_control_regularized_jepa_query_resampler_semantic_anchor_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 128 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler_control_regularized \
  --contrastive-negative-sources 2 \
  --jepa-query-count 8 --jepa-hidden-dim 16 \
  --jepa-train-epochs 40 --jepa-lr 0.01 --jepa-weight-decay 0.001 \
  --ridge 0.001 --top-k 8 --min-score 0.0 --min-decision-score 0.20
```

Low-threshold diagnostic:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_control_regularized_jepa_query_resampler_semantic_anchor_smoke_lowthreshold_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 128 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler_control_regularized \
  --contrastive-negative-sources 2 \
  --jepa-query-count 8 --jepa-hidden-dim 16 \
  --jepa-train-epochs 40 --jepa-lr 0.01 --jepa-weight-decay 0.001 \
  --ridge 0.001 --top-k 8 --min-score 0.0 --min-decision-score 0.0
```

## Results

Primary strict smoke:

- pass rows: `0/6`
- max learned accuracy: `0.250`
- max learned-target lift: `+0.000`
- all default-threshold source-destroying controls stay at target
- query effective rank: `128`
- query entropy: about `1.322-1.324`
- context variance: about `0.0067-0.0068`

Rows:

| Direction | Budget | Learned | Target | Best control | Oracle | Pass |
|---|---:|---:|---:|---:|---:|---|
| core -> holdout | 4 | 0.250 | 0.250 | 0.250 | 0.250 | no |
| core -> holdout | 8 | 0.250 | 0.250 | 0.250 | 0.375 | no |
| holdout -> core | 4 | 0.250 | 0.250 | 0.250 | 0.250 | no |
| holdout -> core | 8 | 0.250 | 0.250 | 0.250 | 0.250 | no |
| same-family | 4 | 0.250 | 0.250 | 0.250 | 0.250 | no |
| same-family | 8 | 0.250 | 0.250 | 0.250 | 0.3125 | no |

The low-threshold diagnostic also fails. It does not recover matched-source
accuracy, and it exposes the same control risk that motivated this gate:
best controls rise to `0.500` in core -> holdout at 4 bytes and `0.375` in
same-family at 4 bytes.

## Interpretation

This variant fixes the symptom by suppressing decisions, not by learning a
source-safe communication interface. The non-collapsed rank/entropy telemetry
shows the connector is not numerically dead, but the usable source signal is
gone under a reviewer-safe decision threshold. When the threshold is relaxed,
destructive controls can again become useful.

## Decision

Do not promote. This prunes the current semantic-anchor JEPA-Q objective family:

- random-feature JEPA-Q was partial but asymmetric;
- plain trainable JEPA-Q learned signal but leaked through shuffled controls;
- control-regularized trainable JEPA-Q cleaned controls only by collapsing to
  target-only behavior.

The next learned-receiver gate should not keep tuning this same objective. The
highest-value next method branch is a stronger feature source or architecture:

1. frozen activation/LLM hidden-state features with the same control objective;
2. a proper contrastive candidate-pool objective that trains on whole
   candidate sets rather than independent candidate rows;
3. a source-private packet task with less hand-coded atom ontology and a
   real-ish private tool/retrieval trace.
