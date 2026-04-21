# Toy Process Repair Bridge

- Seed: `0`
- Examples: `192`
- Pool size: `5`
- Chain length: `4`
- Repair threshold: `0.5500`

| Method | Accuracy | Oracle accuracy | Repair application rate | False repair rate | Mean selected severity |
|---|---:|---:|---:|---:|---:|
| rerank_only | 0.5469 | 1.0000 | 0.0000 | 0.0000 | 1.1823 |
| process_aware_repair | 1.0000 | 1.0000 | 0.5417 | 0.0885 | 1.1823 |
| oracle | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Severity Subgroups

| Severity level | Count | Rerank-only accuracy | Repair accuracy | Oracle accuracy | Repair application rate | False repair rate |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 0.9375 | 1.0000 | 1.0000 | 0.2188 | 0.1562 |
| 1 | 64 | 0.5000 | 1.0000 | 1.0000 | 0.5781 | 0.0781 |
| 2 | 64 | 0.2031 | 1.0000 | 1.0000 | 0.8281 | 0.0312 |
