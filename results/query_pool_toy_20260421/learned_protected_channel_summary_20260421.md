# Learned Protected-Channel Toy Summary

Date: 2026-04-21

Run:

`query_pool_learned_protected_channel_vs_topk.json`

Question: can a learned soft channel mask beat fixed, PCA/gauge-aware, and
supervised signal-aware protected residual channels while staying interpretable?

| Scenario | Residual | Fixed protected | Gauge-aware protected | Signal-aware protected | Learned protected | Best method |
|---|---:|---:|---:|---:|---:|---|
| aligned | 0.5417 | 0.5938 | 0.5469 | 0.5469 | 0.5781 | fixed protected |
| rotated | 0.6406 | 0.5729 | 0.5677 | 0.5573 | 0.6302 | residual |
| outlier | 0.5677 | 0.6198 | 0.5677 | 0.5104 | 0.6562 | learned protected |
| slot_permuted | 0.5417 | 0.4948 | 0.5260 | 0.5781 | 0.5573 | signal-aware protected |

Learned-mask telemetry:

| Scenario | Mask entropy | Effective channels | Top mass | Signal alignment | Task acc |
|---|---:|---:|---:|---:|---:|
| aligned | 2.5919 | 13.3560 | 0.1695 | 0.3580 | 0.5781 |
| rotated | 2.3957 | 10.9758 | 0.1768 | 0.5168 | 0.6302 |
| outlier | 2.8159 | 16.7099 | 0.1755 | 0.5670 | 0.6562 |
| slot_permuted | 2.2956 | 9.9317 | 0.2725 | 0.6500 | 0.5573 |

Interpretation:

The learned soft mask is useful but not sufficient as a standalone method. It
wins the outlier scenario, nearly recovers the rotated residual baseline, and
keeps interpretable non-collapsed masks. It still loses to fixed protected
channels on aligned data and to signal-aware channels under slot permutation.
This suggests the next toy branch should combine learned masks with explicit
signal alignment and orientation/permutation constraints rather than replacing
the structured basis entirely.

Next ablations:

- Add a signal-regularized learned mask with a penalty toward supervised
  between-class signal importance.
- Add an orthogonal/Procrustes pre-alignment before learning the mask.
- Sweep protected-channel count and report task-energy capture per channel.
- Add a permutation-aware slot loss so protected routing is less dependent on
  stable slot identities.

