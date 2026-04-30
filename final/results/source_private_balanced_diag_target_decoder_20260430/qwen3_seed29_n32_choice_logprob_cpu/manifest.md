# Source-Private Tool-Trace Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --benchmark-jsonl results/source_private_diag_only_public_ablation_20260430/direct_diag_n500_seed29/benchmark.jsonl --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_logprob_cpu --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 32 --seed 29 --max-new-tokens 1 --no-enable-thinking --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte --prompt-mode choice_alias --decode-mode choice_logprob --progress-jsonl .debug/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_logprob_cpu_progress.jsonl --partial-predictions-jsonl results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_logprob_cpu/target_predictions.partial.jsonl --progress-every 4
```

## Outcome

- pass gate: `False`
- examples: `32`
- matched accuracy: `0.250`
- target-only accuracy: `0.250`
- best control accuracy: `0.250`

## Artifacts

- `target_predictions.jsonl`
- `target_predictions.partial.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
