# Source-Private Product-Codebook Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_product_codebook_target_decoder_smoke.py --output-dir results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n16_distance_no_explicit_prior_cpu --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --train-examples 512 --eval-examples 16 --train-seed 29 --eval-seed 30 --train-family-set all --eval-family-set all --candidates 4 --feature-dim 512 --budget-bytes 4 --ridge 0.01 --candidate-view slot --no-fit-intercept --remap-slot-seed 101  --candidate-metadata-mode distance --seed 29 --max-new-tokens 8 --no-enable-thinking  --progress-jsonl .debug/source_private_product_codebook_target_decoder_20260430/remap101_budget4_n16_distance_no_explicit_prior_cpu_progress.jsonl --partial-predictions-jsonl results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n16_distance_no_explicit_prior_cpu/target_predictions.partial.jsonl --progress-every 4 --no-require-pass
```

## Outcome

- pass gate: `False`
- strict CI pass gate: `False`
- examples: `16`
- matched accuracy: `0.312`
- target-only accuracy: `0.312`
- best control accuracy: `0.312`

## Artifacts

- `target_predictions.jsonl`
- `target_predictions.partial.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
