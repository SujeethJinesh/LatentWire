# Toy Tokenizer Frontier Bridge

- Seed: `0`
- Train examples: `96`
- Test examples: `96`
- Source tokenizer: `source_frontier`
- Target tokenizer: `target_frontier`

| Method | Exact recon | Decoded boundary F1 | Source-target boundary F1 | Bytes/example | Learned remap coverage |
|---|---:|---:|---:|---:|---:|
| token_id | 0.0000 | 0.3777 | 0.7952 | 19.23 | 0.0000 |
| frontier_regroup | 1.0000 | 1.0000 | 0.7952 | 14.26 | 0.0000 |
| learned_remap | 1.0000 | 1.0000 | 0.7952 | 11.74 | 0.3211 |
