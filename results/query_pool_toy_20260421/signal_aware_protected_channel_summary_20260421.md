# Signal-Aware Protected-Channel Toy Summary

Date: 2026-04-21

Run:

`query_pool_signal_aware_protected_channel_vs_topk.json`

Question: can a protected-channel residual codebook become more interpretable
and less gauge-brittle if the protected basis is chosen from supervised
between-class signal rather than fixed coordinates or PCA variance?

| Scenario | Residual | Fixed protected | Gauge-aware protected | Signal-aware protected | Best protected delta vs residual |
|---|---:|---:|---:|---:|---:|
| aligned | 0.5417 | 0.5938 | 0.5469 | 0.5469 | +0.0521 |
| rotated | 0.6406 | 0.5729 | 0.5677 | 0.5573 | -0.0677 |
| outlier | 0.5677 | 0.6198 | 0.5677 | 0.5104 | +0.0521 |
| slot_permuted | 0.5417 | 0.4948 | 0.5260 | 0.5781 | +0.0365 |

Signal telemetry:

| Scenario | Gauge energy frac. | Signal task energy | Signal/variance alignment | Signal query energy | Signal slot energy |
|---|---:|---:|---:|---:|---:|
| aligned | 0.3040 | 0.6784 | 0.6457 | 0.3934 | 0.3050 |
| rotated | 0.2986 | 0.7267 | 0.4080 | 0.3773 | 0.2487 |
| outlier | 0.2986 | 0.6673 | 0.3048 | 0.4730 | 0.2019 |
| slot_permuted | 0.3097 | 0.7066 | 0.5032 | 0.5016 | 0.2807 |

Interpretation:

The signal-aware basis is useful as a diagnostic, but not yet a uniformly
better toy method. It repairs the slot-permuted case and exposes a clean
supervised signal-energy readout, but it underperforms the fixed protected
variant on aligned/outlier and remains weak under rotation. This supports a
more specific blocker: the protected subspace must be task-aware and
orientation-aware, not merely high-variance or fixed-coordinate.

Next toy ablations:

- Combine signal-aware basis selection with an orthogonal Procrustes alignment
  before residual coding.
- Sweep protected-channel count and report task energy captured per channel.
- Add a learned soft mask over channels and compare it to fixed, PCA, and
  supervised bases.
- Add a permutation-aware slot-matching loss so the method does not depend on
  stable slot identities.
