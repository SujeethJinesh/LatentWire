# Tokenizer Stress Split

- Source tokenizer: `source_stress`
- Target tokenizer: `target_stress`
- Examples: `12`
- Remap capacity: `12`
- Overall boundary F1: `0.9463`
- Overall byte-span remap coverage: `0.9354`
- Token-ID exact reconstruction proxy: `0.0833`

## Category Metrics

| category | n | boundary_f1 | frag_delta | remap_coverage | token_id_exact | multibyte_span_rate |
|---|---:|---:|---:|---:|---:|---:|
| overall | 12 | 0.9463 | 0.0448 | 0.9354 | 0.0833 | 0.0608 |
| decimals | 5 | 0.9428 | 0.0436 | 0.9550 | 0.0000 | 0.0650 |
| math_units | 3 | 0.9403 | 0.0600 | 0.9825 | 0.0000 | 0.1083 |
| multi_byte_span | 4 | 0.9457 | 0.0608 | 1.0000 | 0.0000 | 0.1376 |
| punctuation | 10 | 0.9445 | 0.0422 | 0.9225 | 0.1000 | 0.0442 |
| unicode | 5 | 0.9384 | 0.0681 | 0.9024 | 0.0000 | 0.1291 |
| variables | 8 | 0.9415 | 0.0365 | 0.9312 | 0.1250 | 0.0427 |

## Interpretability Notes

- `source_target_boundary_f1` measures byte-boundary agreement between source and target tokenizations.
- `fragmentation_delta` is target tokens per byte minus source tokens per byte; positive values mean target-side fragmentation is worse.
- `byte_span_remap_coverage` is the byte share covered by a bounded source-token-to-target-piece remap table, with single-character spans treated as trivially coverable.
- `token_id_exact_reconstruction` is the brittle token-ID transfer proxy: source token IDs decoded under the target vocabulary.
