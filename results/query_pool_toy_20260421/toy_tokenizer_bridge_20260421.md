# Toy Tokenizer Bridge

- Seed: `0`
- Examples: `192`
- Source tokenizer: `source_a`
- Target tokenizer: `target_b`

| Method | Exact recon | Digit acc | Operator acc | Fragmentation | Bytes/example |
|---|---:|---:|---:|---:|---:|
| token_id | 0.0000 | 0.1285 | 0.0077 | 0.8271 | 26.75 |
| vocab_remap | 0.0677 | 0.2575 | 0.2988 | 1.2717 | 20.03 |
| byte_span_canonical | 1.0000 | 1.0000 | 1.0000 | 0.8779 | 68.30 |
| byte_span_noisy_bytes | 0.9010 | 0.9916 | 0.9953 | 0.8790 | 69.30 |
| byte_span_noisy_spans | 0.9010 | 0.9946 | 0.9897 | 0.8779 | 69.30 |
