# Qwen-to-Phi Top-2/Rival Codebook Gate

- pass gate: `False`
- method accuracy: `0.460938`
- fixed hybrid accuracy: `0.467448`
- source top-1 accuracy: `0.411458`
- source top-1/top-2 oracle accuracy: `0.675781`
- best destructive: `source_row_shuffle_codebook` (0.468750)

## Interpretation

The protected top-2/rival codebook does not pass the larger cached HellaSwag gate; the top-2 oracle remains headroom rather than a learned receiver.
