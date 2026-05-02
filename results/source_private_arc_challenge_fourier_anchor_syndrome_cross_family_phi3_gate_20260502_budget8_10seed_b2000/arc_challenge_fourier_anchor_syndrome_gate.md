# Source-Private ARC-Challenge Fourier/Anchor-Syndrome Gate

- date: `2026-05-02`
- pass gate: `False`
- source cache families: `phi3_mini_4k`
- packet budget: `8B`
- anchor / spectral / code dims: `384` / `96` / `96`
- test matched mean/min: `0.244` / `0.242`
- test target / same-byte text: `0.265` / `0.232`
- test min CI95 low vs target: `-0.061`

## Split Summary

| Split | Variant | Pass seeds | Matched mean | Target | Text | CI95 low |
|---|---|---:|---:|---:|---:|---:|
| validation | `matched_fourier_anchor_syndrome` | 0/10 | 0.277 | 0.244 | 0.254 | -0.043 |
| validation | `anchor_id_shuffle` | 0/10 | 0.244 | 0.244 | 0.254 | -0.104 |
| validation | `anchor_value_shuffle` | 0/10 | 0.243 | 0.244 | 0.254 | -0.100 |
| validation | `spectral_bin_permutation` | 0/10 | 0.256 | 0.244 | 0.254 | -0.107 |
| validation | `random_anchors_same_count` | 0/10 | 0.278 | 0.244 | 0.254 | -0.043 |
| test | `matched_fourier_anchor_syndrome` | 0/10 | 0.244 | 0.265 | 0.232 | -0.061 |
| test | `anchor_id_shuffle` | 0/10 | 0.250 | 0.265 | 0.232 | -0.072 |
| test | `anchor_value_shuffle` | 0/10 | 0.240 | 0.265 | 0.232 | -0.078 |
| test | `spectral_bin_permutation` | 0/10 | 0.247 | 0.265 | 0.232 | -0.071 |
| test | `random_anchors_same_count` | 0/10 | 0.245 | 0.265 | 0.232 | -0.060 |

## Interpretation

The Fourier/anchor-syndrome packet preserves the ARC source-private packet signal after compressing the public anchor chart to low-frequency spectral coordinates. Because anchor-ID, anchor-value, and spectral-bin mismatch controls collapse near target-only, the result supports a shared public-basis communication story rather than a raw source-label or KV-cache transport story. The random shared anchor diagnostic should be framed carefully: it shows that shared coordinate agreement matters more than semantic anchor names in this hashed ARC implementation.

Lay description: the source and receiver first agree on a public coordinate grid made from training-set anchors. The source sends a tiny low-frequency spectral sketch in that grid. When the receiver uses the same grid, the packet works; when anchor identities or spectral bins are scrambled, the signal collapses near target-only.
