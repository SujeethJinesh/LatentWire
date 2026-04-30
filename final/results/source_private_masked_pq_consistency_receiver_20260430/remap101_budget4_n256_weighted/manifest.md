# Source-Private Masked-PQ Consistency Receiver Manifest

## Command

```bash
scripts/run_source_private_masked_pq_consistency_receiver.py --output-dir results/source_private_masked_pq_consistency_receiver_20260430/remap101_budget4_n256_weighted --train-examples 512 --eval-examples 256 --train-seed 29 --eval-seed 30 --train-family-set all --eval-family-set all --candidates 4 --feature-dim 512 --budget-bytes 4 --ridge 0.01 --receiver-ridge 0.01 --candidate-view slot --no-fit-intercept --remap-slot-seed 101 --seed 29 --mask-rounds 2 --random-rounds 2 --matched-weight 8 --mask-weight 4 --control-weight 1 --target-only-weight 1 --no-require-pass
```

## Outcome

- source packet pass: `True`
- pass gate: `False`
- learned matched accuracy: `0.582`
- deterministic L2 matched accuracy: `0.582`

## Artifacts

- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `receiver_weights.json`
- `manifest.json`
- `manifest.md`
