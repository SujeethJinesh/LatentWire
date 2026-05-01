# HellaSwag Hidden-Innovation Bagged Gate References

## Primary Sources

- HellaSwag benchmark: https://arxiv.org/abs/1905.07830
- Relative representations / anchor-basis latent communication: https://arxiv.org/abs/2209.15430
- Sparse autoencoders for interpretable language-model features: https://arxiv.org/abs/2309.08600
- Universal sparse autoencoders for cross-model concept alignment: https://arxiv.org/abs/2502.03714
- SPARC concept-aligned sparse autoencoders: https://arxiv.org/abs/2507.06265
- C2C cache-to-cache communication: https://openreview.net/forum?id=LeatkxrBCi
- KVCOMM selective KV sharing: https://arxiv.org/abs/2510.12872
- Q-KVComm: https://arxiv.org/abs/2512.17914
- QJL 1-bit quantized Johnson-Lindenstrauss transform: https://arxiv.org/abs/2406.03482
- TurboQuant online vector quantization: https://arxiv.org/abs/2504.19874

## Novelty Boundary

This bagged gate is not claiming to invent sparse autoencoders, relative
representations, model stitching, quantization, or cache compression. Those
works already establish important pieces: common/relative coordinate systems,
sparse cross-model concept spaces, and low-bit residual-preserving sketches.

The local novelty is the communication boundary and the reviewer gate: a
predeclared model bank uses source hidden innovation internally, then emits
only a fixed `2B` raw / `5B` framed source-private candidate packet. The
artifact tests that the gain survives two fresh train-row support samples
(`2027`, `2039`), beats label-copy and score-only bagged controls, and
collapses under zero-hidden, wrong-example, and candidate-roll hidden controls.
The current strengthened artifact also requires all `2-of-3` train-sample
jackknife subbags to pass, so the positive result is not carried by only one
support draw.

## Method Implication

The result supports a next ICLR branch that combines stability selection with a
more principled common-basis learner. A reviewer-safe framing is:

1. Single dense denoiser: positive on cached split, failed fresh support.
2. Bagged dense denoiser: rescues three-sample and jackknife stress without
   extra bytes.
3. Next gate: freeze the method and widen to a larger/full validation slice.
4. Next method branch: replace dense hidden residuals with sparse/common-basis features
   and test whether the same support-sampled bank improves robustness further.
