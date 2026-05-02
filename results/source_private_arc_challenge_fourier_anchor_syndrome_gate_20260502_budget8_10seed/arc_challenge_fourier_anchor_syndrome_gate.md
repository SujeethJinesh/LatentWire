# Source-Private ARC-Challenge Fourier/Anchor-Syndrome Gate

- date: `2026-05-02`
- pass gate: `True`
- packet budget: `8B`
- anchor / spectral / code dims: `384` / `96` / `96`
- test matched mean/min: `0.344` / `0.342`
- test target / same-byte text: `0.265` / `0.300`
- test min CI95 low vs target: `0.038`

## Split Summary

| Split | Variant | Pass seeds | Matched mean | Target | Text | CI95 low |
|---|---|---:|---:|---:|---:|---:|
| validation | `matched_fourier_anchor_syndrome` | 10/10 | 0.385 | 0.244 | 0.334 | 0.060 |
| validation | `anchor_id_shuffle` | 0/10 | 0.244 | 0.244 | 0.334 | -0.094 |
| validation | `anchor_value_shuffle` | 0/10 | 0.256 | 0.244 | 0.334 | -0.097 |
| validation | `spectral_bin_permutation` | 0/10 | 0.259 | 0.244 | 0.334 | -0.087 |
| validation | `random_anchors_same_count` | 10/10 | 0.388 | 0.244 | 0.334 | 0.064 |
| test | `matched_fourier_anchor_syndrome` | 10/10 | 0.344 | 0.265 | 0.300 | 0.038 |
| test | `anchor_id_shuffle` | 0/10 | 0.252 | 0.265 | 0.300 | -0.065 |
| test | `anchor_value_shuffle` | 0/10 | 0.242 | 0.265 | 0.300 | -0.070 |
| test | `spectral_bin_permutation` | 0/10 | 0.263 | 0.265 | 0.300 | -0.064 |
| test | `random_anchors_same_count` | 10/10 | 0.344 | 0.265 | 0.300 | 0.037 |

## Interpretation

The Fourier/anchor-syndrome packet preserves the ARC source-private packet signal after compressing the public anchor chart to low-frequency spectral coordinates. Because anchor-ID, anchor-value, and spectral-bin mismatch controls collapse near target-only, the result supports a shared public-basis communication story rather than a raw source-label or KV-cache transport story. The random shared anchor diagnostic should be framed carefully: it shows that shared coordinate agreement matters more than semantic anchor names in this hashed ARC implementation.

Lay description: the source and receiver first agree on a public coordinate grid made from training-set anchors. The source sends a tiny low-frequency spectral sketch in that grid. When the receiver uses the same grid, the packet works; when anchor identities or spectral bins are scrambled, the signal collapses near target-only.
