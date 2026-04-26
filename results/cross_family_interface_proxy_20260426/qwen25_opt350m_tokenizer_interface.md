# Real Tokenizer Interface Pair Sweep

- Input: `data/gsm8k_gate_search_30.jsonl`
- Calibration examples / pair: `15`
- Remap capacity: `24`
- Pair count: `1`

| Pair | Src frag | Tgt frag | Frag delta | Shared decoded | Boundary F1 | Remap coverage | Src toks | Tgt toks | Remap table |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen25_to_opt350m | 0.2414 | 0.2362 | -0.0052 | 0.9047 | 0.9434 | 0.0550 | 85.87 | 84.07 | 10 |

## Read

- `mean_shared_decoded_token_rate` near `1.0` means the two tokenizers expose almost the same surface pieces on this slice.
- `mean_boundary_f1` and `mean_fragmentation_delta` capture whether token boundaries diverge even when decoded surfaces overlap.
- `mean_byte_span_remap_coverage` measures how much of the byte stream is covered by a bounded source-token to target-piece remap table learned on a calibration subset.
