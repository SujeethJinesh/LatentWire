# HellaSwag Qwen-To-Phi Oracle Switch References

Date: 2026-05-04

This memo supports the failed Qwen-to-Phi oracle-switch decomposition gate.
The safe claim is that selective deferral and Fourier score-contrast packets
were tested and weakened on the current cached surface; the broader
decoder-side-information/rate-distortion framing remains alive.

## Primary Sources

- HellaSwag: Zellers et al., 2019, https://arxiv.org/abs/1905.07830.
  This is the multiple-choice commonsense benchmark used for the cached
  Qwen-to-Phi gate.

- Reject option / selective prediction: Chow, "On Optimum Recognition Error
  and Reject Tradeoff", 1970,
  https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff.
  This motivates reporting help/harm/coverage instead of only unconditional
  accuracy when a receiver may switch away from the default packet.

- Selective classification: Geifman and El-Yaniv, 2017,
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks;
  SelectiveNet, Geifman and El-Yaniv, 2019,
  https://proceedings.mlr.press/v97/geifman19a.html.
  These sources motivate the risk/coverage framing, but LatentWire's gate is
  stricter because it must improve total held-out accuracy versus fixed hybrid,
  not merely covered-set risk.

- Learning to defer: Madras et al., 2018,
  https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer.
  This is the closest decision-systems framing: the learned router chooses
  between decision makers. Our negative result says the current packet/Phi
  features do not provide a safe defer-to-Phi controller.

- Calibration: Guo et al., 2017,
  https://proceedings.mlr.press/v70/guo17a.html; Hendrycks and Gimpel, 2017,
  https://arxiv.org/abs/1610.02136. These sources motivate margin/confidence
  baselines and why raw max-probability is not enough for a reviewer-proof
  switcher.

- Slepian-Wolf distributed source coding:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources.
  Wyner-Ziv side-information coding:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf.
  These motivate the "receiver has side information" framing; the gate is an
  empirical packet test, not a new source-coding theorem.

- Mixture-of-experts routing: Shazeer et al., 2017,
  https://research.google/pubs/pub45929; Switch Transformer, 2022,
  https://www.jmlr.org/beta/papers/v23/21-0998.html.
  These motivate sparse routing/frontier control, but LatentWire should not
  claim MoE novelty.

- Prefix tuning: Li and Liang, 2021,
  https://aclanthology.org/2021.acl-long.353/.
  Gist tokens: Mu et al., 2023, https://arxiv.org/abs/2304.08467.
  These are related compact conditioning methods. LatentWire differs because
  the packet is per-example, source-private, discrete/byte-scale, and does not
  expose source prompt text or learned prompt vectors.

- C2C: https://openreview.net/forum?id=LeatkxrBCi and
  https://arxiv.org/abs/2510.03215. KVComm:
  https://openreview.net/forum?id=F7rUng23nw and
  https://arxiv.org/abs/2510.03346. These are close internal-state
  communication competitors because they transmit/fuse KV/cache state. The
  current gate should be framed as an extreme-rate source-private packet row,
  not as a native-speed replacement for cache transfer.

- FlashAttention: https://arxiv.org/abs/2205.14135 and
  https://arxiv.org/abs/2307.08691. vLLM/PagedAttention:
  https://arxiv.org/abs/2309.06180. SGLang:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.
  These define the systems comparison family for future NVIDIA rows.

- QJL: https://arxiv.org/abs/2406.03482. KIVI:
  https://icml.cc/virtual/2024/poster/34318. KVQuant:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html.
  TurboQuant: https://openreview.net/forum?id=tO3ASKZlok.
  These support the quantization lesson: low-rate signs/residual bits matter
  only if they improve the downstream decision frontier, not if they merely
  reconstruct source scores.

## Boundary For Paper Claims

Safe:

- "We tested selective deferral and Fourier/Helmert score-contrast packets on
  a cached Qwen-to-Phi HellaSwag surface and found that the current byte-scale
  packet family cannot exploit the measured oracle gap."
- "Qwen source score top-2 contains much larger diagnostic headroom, motivating
  protected top-rival/frontier packets."

Unsafe:

- "LatentWire solves learned deferral."
- "Fourier score packets are a positive cross-model latent method."
- "LatentWire beats C2C/KVComm/TurboQuant systems baselines without native GPU
  measurements."
