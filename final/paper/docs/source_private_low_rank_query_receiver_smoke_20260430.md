# Source-Private Low-Rank Query Receiver Smoke

- date: `2026-04-30`
- gate: `source_private_low_rank_query_receiver_smoke`
- status: negative / useful capacity frontier

## Question

Does a low-rank query-style bottleneck over frozen BGE features preserve the
matched source-private signal while reducing the leakage risk seen in the full
bilinear contrastive receiver?

## Implementation

`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py` now
supports:

- `--receiver-mode contrastive_low_rank_query`
- `--contrastive-rank`

The implementation first fits the source-control contrastive bilinear receiver
with shuffled-source negatives, then truncates the learned bilinear map by SVD
to a fixed rank. This gives a cheap Mac-local proxy for a query bottleneck: rank
controls how many candidate-feature/source-atom interaction directions can be
used.

## Command

```bash
for rank in 1 2 4 8; do
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
    scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
    --output-dir results/source_private_low_rank_query_frozen_receiver_bge_rank${rank}_smoke_20260430 \
    --budgets 2 4 --train-examples 64 --eval-examples 32 --seed 47 \
    --candidate-atom-view heldout_synonym \
    --calibration-atom-view synonym_stress \
    --candidate-calibration all_public --calibration-examples 64 \
    --feature-dim 768 --text-feature-mode hf_mid_last_mean \
    --receiver-mode contrastive_low_rank_query \
    --contrastive-negative-sources 2 \
    --contrastive-rank ${rank} \
    --feature-model BAAI/bge-small-en --feature-device cpu \
    --feature-dtype float32 --feature-max-length 96 --local-files-only \
    --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.30
done
```

## Results

| Rank | Pass | Pass rows | Best learned | Best lift | Readout |
|---:|---:|---:|---:|---:|---|
| 1 | `False` | 0/6 | 0.250 | +0.000 | too little capacity |
| 2 | `False` | 0/6 | 0.375 | +0.125 | weak signal |
| 4 | `False` | 0/6 | 0.375 | +0.125 | weak signal, oracle improves |
| 8 | `False` | 1/6 | 0.625 | +0.375 | one holdout-to-core row passes, not bidirectional |

Rank-8 details at 4 bytes:

- core -> holdout: `0.375` vs target/control `0.250`, oracle `0.875`
- holdout -> core: `0.625` vs target/control `0.250`, oracle `0.875`, pass
- same-family: `0.500` vs target/control `0.250`, oracle `0.875`, still fails

## Interpretation

The low-rank bottleneck makes the capacity/leakage tradeoff clearer but does
not become a headline method. Very low ranks keep controls flat but erase useful
signal. Rank 8 recovers some useful signal with clean controls, but the effect
is not bidirectional and remains below the semantic-anchor receiver.

Decision: do not promote simple SVD-truncated bilinear low-rank receivers. The
next method should be a real learned query-resampler / information-bottleneck
connector trained directly with source-control negatives, not post-hoc SVD
truncation.

## Next Gate

Build a small trainable receiver with explicit low-rank factors or query vectors
optimized end-to-end under the same controls:

- objective: candidate ranking plus shuffled-source/random same-byte negatives
- rate axis: query count or factor rank `{2,4,8,16}`
- first strict pass: `n=64`, bidirectional held-out synonym, controls <= target
  + `0.03`, matched >= target + `0.25`
