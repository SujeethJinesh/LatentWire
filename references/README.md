# References

This folder contains the papers requested for `latent_bridge`, numbered to match the list you provided.

- Files `01_` through `41_` correspond to items 1 through 41 in your reading list.
- Most entries are saved as PDFs.
- `38_scaling_monosemanticity.html` is saved as HTML because the source link is a web article on Transformer Circuits rather than a direct PDF.
- `download_manifest.json` records the source URL, output path, and download status for each item.

The search-only items were resolved to concrete paper links before download, so the folder should be complete.

## Math Grounding Addendum

Files `42_` through `54_` were added after the first live validation run. They cover representation similarity, CCA/CKA diagnostics, structured random/Fourier/Hadamard transforms, product quantization, incoherence processing, and randomized linear algebra. These are intended to ground the next RotAlign-KV ablations:

- `42_` through `44_`: CKA/SVCCA/PWCCA layer-pairing and representation-similarity diagnostics.
- `45_` through `48_`: random orthogonal, Hadamard/Fourier, and butterfly transform families.
- `49_` through `51_`, `53_`, `54_`: rotation/scale/product/incoherence quantization methods.
- `52_`: randomized low-rank methods for reduced-rank and subspace alignment ablations.

See `math_grounding_manifest.json` for source URLs and the reason each paper was added.
