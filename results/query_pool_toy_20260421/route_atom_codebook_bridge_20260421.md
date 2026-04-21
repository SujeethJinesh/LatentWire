# Toy Route Atom Codebook Bridge

- Seed: `13`
- Train examples: `96`
- Test examples: `64`
- Route families: `4`
- Atoms per family: `4`
- Protected atoms: `3`

| Method | Accuracy | MSE | Atom recovery | Codebook entropy | Codebook perplexity | Bytes proxy | Compute proxy | Help vs raw | Harm vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ridge | 0.7812 | 0.0437 | 0.0000 | 2.7332 | 15.3819 | 1088.0000 | 24576.0000 | 0.0000 | 0.0000 |
| uniform_codebook_quantization | 0.6875 | 0.1659 | 0.3478 | 2.1250 | 8.3725 | 280.0000 | 26112.0000 | 0.0000 | 0.0938 |
| learned_shared_codebook | 0.8438 | 0.7225 | 0.9231 | 2.4725 | 11.8526 | 280.0000 | 26112.0000 | 0.0625 | 0.0000 |
| route_conditioned_codebook | 0.7656 | 0.5927 | 0.8659 | 2.4854 | 12.0059 | 328.0000 | 12288.0000 | 0.0000 | 0.0156 |
| protected_outlier_atoms | 0.8438 | 0.2069 | 0.9231 | 2.4725 | 11.8526 | 376.0000 | 26880.0000 | 0.0625 | 0.0000 |
| oracle | 1.0000 | 0.0000 | 1.0000 | 2.7332 | 15.3819 | 512.0000 | 1536.0000 | 0.2188 | 0.0000 |
