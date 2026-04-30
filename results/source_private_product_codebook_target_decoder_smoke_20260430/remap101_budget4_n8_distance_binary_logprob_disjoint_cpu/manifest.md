# Source-Private Product-Codebook Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_product_codebook_target_decoder_smoke.py --output-dir results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n8_distance_binary_logprob_disjoint_cpu --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --train-examples 512 --eval-examples 8 --train-seed 29 --eval-seed 30 --train-start-index 0 --eval-start-index 10000 --train-family-set all --eval-family-set all --candidates 4 --feature-dim 512 --budget-bytes 4 --ridge 0.01 --candidate-view slot --no-fit-intercept --remap-slot-seed 101  --candidate-metadata-mode distance --decode-mode candidate_binary_logprob --binary-fallback-threshold 0.0 --seed 29 --max-new-tokens 24 --no-enable-thinking  --progress-jsonl results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n8_distance_binary_logprob_disjoint_cpu/progress.jsonl --partial-predictions-jsonl results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n8_distance_binary_logprob_disjoint_cpu/target_predictions.partial.jsonl --progress-every 2 --no-require-pass
```

## Outcome

- pass gate: `False`
- strict CI pass gate: `False`
- examples: `8`
- matched accuracy: `0.500`
- target-only accuracy: `0.250`
- best control accuracy: `0.500`
- train/eval ID overlap count: `0`

## Artifacts

- `target_predictions.jsonl`
- `target_predictions.partial.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
