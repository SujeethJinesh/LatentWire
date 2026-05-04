# HellaSwag Qwen-To-Phi Quantized Score Packet Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- best quantized score packet: `quantized_score_packet_rotated_uniform_zscore_2B`
- best quantized accuracy: `0.468750`
- best quantized delta: `0.001302`
- best quantized CI95 low: `-0.003906`
- raw source-score fusion accuracy: `0.391927`
- source top1/top2 oracle accuracy: `0.675781`
- best destructive: `zero_source_packet_uniform_zscore_1B_control` (`0.467448`)

## Budget Rows

| Raw bytes | Framed bytes | Best method | Accuracy | Delta | CI95 low | Overrides |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 4 | `quantized_score_packet_uniform_zscore_1B` | 0.464844 | -0.002604 | -0.010417 | 11 |
| 2 | 5 | `quantized_score_packet_rotated_uniform_zscore_2B` | 0.468750 | 0.001302 | -0.003906 | 8 |
| 4 | 7 | `quantized_score_packet_rotated_uniform_zscore_4B` | 0.468750 | 0.001302 | -0.003906 | 8 |
| 8 | 11 | `quantized_score_packet_rotated_uniform_zscore_8B` | 0.468750 | 0.001302 | -0.003906 | 8 |

## Slice Rows

| Slice | Rows | Method acc. | Fixed hybrid acc. | Delta | CI95 low | Helps | Harms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 384 | 0.494792 | 0.486979 | 0.007812 | 0.000000 | 3 | 0 |
| 1536 | 384 | 0.442708 | 0.447917 | -0.005208 | -0.013021 | 0 | 2 |

## Interpretation

This gate closes the reviewer-requested calibrated source-score quantization baseline. It tests whether equal-byte score packets recover the large source top1/top2 oracle headroom before we spend more effort on heavier target-native latent receivers.

## Lay Explanation

Instead of sending Qwen's full private scores, we squashed its four answer scores into tiny 1-8 byte messages and let Phi combine the reconstructed scores with its own scores. If this works, then a simple score message is enough; if it fails, the problem needs richer latent evidence.
