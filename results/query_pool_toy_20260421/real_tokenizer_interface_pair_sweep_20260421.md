# Real Tokenizer Interface Pair Sweep

- Input: `data/gsm8k_gate_search_30.jsonl`
- Calibration examples / pair: `15`
- Remap capacity: `24`
- Pair count: `3`

| Pair | Src frag | Tgt frag | Frag delta | Shared decoded | Boundary F1 | Remap coverage | Src toks | Tgt toks | Remap table |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen25_to_qwen3 | 0.2414 | 0.2414 | 0.0000 | 1.0000 | 1.0000 | 0.0550 | 85.87 | 85.87 | 0 |
| qwen25_to_mistral | 0.2414 | 0.2643 | 0.0229 | 0.8174 | 0.9496 | 0.0947 | 85.87 | 93.73 | 24 |
| qwen25_to_phi3 | 0.2414 | 0.2639 | 0.0225 | 0.7972 | 0.9347 | 0.0706 | 85.87 | 93.60 | 24 |

## Read

- `mean_shared_decoded_token_rate` near `1.0` means the two tokenizers expose almost the same surface pieces on this slice.
- `mean_boundary_f1` and `mean_fragmentation_delta` capture whether token boundaries diverge even when decoded surfaces overlap.
- `mean_byte_span_remap_coverage` measures how much of the byte stream is covered by a bounded source-token to target-piece remap table learned on a calibration subset.
