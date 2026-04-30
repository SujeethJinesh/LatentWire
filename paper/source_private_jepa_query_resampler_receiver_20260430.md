# Source-Private JEPA Query Resampler Receiver

- date: `2026-04-30`
- gate: `source_private_jepa_query_resampler_receiver`
- status: implemented; smoke negative / partial signal only

## Question

Can a less hand-designed receiver replace the semantic-anchor packet decoder by
using candidate-conditioned learned queries over decoded source-private packet
atoms?

## Implementation

I added `--receiver-mode jepa_query_resampler` to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`.

The receiver is deliberately different from the pruned bilinear/low-rank BGE
family:

- candidate text features generate `K` query vectors;
- decoded packet atoms provide source-private keys and values;
- masked softmax attention over active packet atoms produces `K` contexts;
- a learned output head scores candidate/source compatibility;
- the same zero-source, shuffled-source, random same-byte, answer-only,
  answer-masked, target-derived, atom-derangement, and knockout controls are
  reused.

The artifact also records query count, hidden dimension, query entropy, context
variance, and effective rank so a target-only collapse cannot be mistaken for a
safe receiver.

## Commands

Diagnostic-head corrected smoke:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_jepa_query_resampler_semantic_anchor_smoke2_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 96 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler \
  --contrastive-negative-sources 2 \
  --jepa-query-count 4 --jepa-hidden-dim 8 \
  --ridge 0.01 --top-k 8 --min-score 0.0 --min-decision-score 0.30
```

Wider query smoke:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_jepa_query_resampler_semantic_anchor_k8_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 64 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 128 --text-feature-mode semantic_anchor \
  --receiver-mode jepa_query_resampler \
  --contrastive-negative-sources 2 \
  --jepa-query-count 8 --jepa-hidden-dim 16 \
  --ridge 0.001 --top-k 8 --min-score 0.0 --min-decision-score 0.20
```

## Results

`K=4, D=8`:

- pass rows: `0/6`
- best learned accuracy: `0.375`
- best learned-target lift: `+0.125`
- exact transformed overlap: `0`
- query rank: `32`
- query entropy: about `1.327`
- context variance: about `0.024-0.027`

`K=8, D=16`:

- pass rows: `0/6`
- best learned accuracy: `0.625`
- best learned-target lift: `+0.375`
- core -> holdout reaches `0.500` at both `4/8` bytes with clean controls, but
  oracle is only `0.500/0.750`
- holdout -> core remains weak: `0.125` at `4` bytes and `0.250` at `8` bytes
- same-family reaches `0.625`, but atom derangement rises to `0.3125` at
  `8` bytes and oracle remains below promotion
- query rank: `128`
- query entropy: about `1.332`
- context variance: about `0.006-0.007`

## Interpretation

The first JEPA/Q-Former-style query resampler is not a paper claim. It is a
useful negative/partial result:

- it is not collapsed by rank or entropy;
- it recovers real source signal in core -> holdout and same-family rows;
- it fails bidirectional held-out transfer;
- oracle/headroom is too weak, meaning the receiver does not yet align packet
  atom contexts with candidate semantics reliably;
- one same-family control is too high under atom derangement.

This weakens the current random-feature query-resampler formulation. The next
credible method branch should train query/key/value factors end-to-end or use a
stronger frozen feature source; it should not tune thresholds on this smoke.

## Decision

Do not promote `jepa_query_resampler` as implemented. Keep it as a harness and
diagnostic contribution because it gives the paper a cleaner learned-connector
baseline than bilinear maps.

Next method gate:

- implement trained query/key/value factors with source-control negatives, or
- switch to a stronger activation/frozen-LLM feature source and rerun the same
  query-resampler controls.
