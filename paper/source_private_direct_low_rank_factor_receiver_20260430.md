# Source-Private Direct Low-Rank Factor Receiver

- date: `2026-04-30`
- gate: `source_private_direct_low_rank_factor_receiver`
- status: negative / pruned for the current BGE frozen-feature setup

## Question

The previous low-rank query receiver SVD-truncated a full bilinear map after
training. Does directly training the low-rank factors preserve useful source
signal while avoiding shuffled-source leakage?

## Implementation

`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py` now
supports:

- `--receiver-mode contrastive_low_rank_factor`
- `--contrastive-rank`
- `--low-rank-factor-epochs`
- `--low-rank-factor-lr`
- `--low-rank-factor-loss`
- `--low-rank-factor-seed`

The receiver scores:

```text
score(candidate, packet) = (feature(candidate) @ U) dot (atom_vector(packet) @ V) + bias
```

It is trained in NumPy with matched-source positives and shuffled-source
negative packets.

## Command

```bash
for rank in 4 8 16; do
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
    scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
    --output-dir results/source_private_direct_low_rank_factor_bge_rank${rank}_smoke_20260430 \
    --budgets 2 4 --train-examples 64 --eval-examples 32 --seed 47 \
    --candidate-atom-view heldout_synonym \
    --calibration-atom-view synonym_stress \
    --candidate-calibration all_public --calibration-examples 64 \
    --feature-dim 768 --text-feature-mode hf_mid_last_mean \
    --receiver-mode contrastive_low_rank_factor \
    --contrastive-negative-sources 2 \
    --contrastive-rank ${rank} \
    --low-rank-factor-epochs 220 \
    --low-rank-factor-lr 0.02 \
    --low-rank-factor-loss bce \
    --low-rank-factor-seed 947 \
    --feature-model BAAI/bge-small-en --feature-device cpu \
    --feature-dtype float32 --feature-max-length 96 --local-files-only \
    --ridge 0.0005 --top-k 8 --min-score 0.05 --min-decision-score 0.30
done
```

## Results

| Rank | Pass | Best learned | Best lift | Controls |
|---:|---:|---:|---:|---|
| 4 | `False` | 0.250 | +0.000 | clean |
| 8 | `False` | 0.250 | +0.000 | clean |
| 16 | `False` | 0.250 | +0.000 | clean |

All evaluated 4-byte rows stay at target-only accuracy (`0.250`) with oracle
rows also at `0.250`. The effective factor ranks are recorded correctly, but
the trained factors do not recover source signal from frozen BGE features.

## Interpretation

This is a clean prune for the current feature/training setup. Direct low-rank
factors solve the leakage problem by collapsing to target-only behavior. They
do not rescue the frozen BGE receiver and should not be tuned further unless a
new feature source or objective changes the hypothesis.

The next method gate should be a real query-resampler / target-preserving
connector, not another bilinear or low-rank map over the same BGE features.
