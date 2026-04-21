# Toy Protected-Basis Quant Bridge

This toy compares four transport schemes under a near-matched byte band.
The calibration set identifies salient channels; the test set measures how each scheme
preserves reconstruction and a downstream linear boundary.

| Method | Accuracy | MSE | Cosine | Outlier mass | Bytes estimate | Help vs uniform | Harm vs uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_low_bit | 0.9740 | 0.2130 | 0.9873 | 0.8803 | 20.0 | 0.0000 | 0.0000 |
| protected_salient_channels | 0.9792 | 0.4933 | 0.9668 | 0.4258 | 22.0 | 0.0052 | 0.0000 |
| incoherent_preprocess | 0.9948 | 0.0524 | 0.9966 | 0.8897 | 20.0 | 0.0208 | 0.0000 |
| mixed_bit_allocation | 1.0000 | 0.0405 | 0.9962 | 0.8899 | 20.0 | 0.0260 | 0.0000 |

Interpretation:

Protected salient channels and mixed-bit allocation should recover the highest-variance
coordinates under almost the same byte band, while incoherent preprocessing is the basis
fix that should reduce quantization error by spreading coordinate outliers before low-bit
transport. The task is intentionally small so that future bridge claims can be tied back
to a specific transport choice, a specific byte budget, and a specific salient-channel
selection rule.
