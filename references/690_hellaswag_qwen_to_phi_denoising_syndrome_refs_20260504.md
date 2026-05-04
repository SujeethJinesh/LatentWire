# HellaSwag Qwen-To-Phi Denoising Syndrome References

Date: 2026-05-04

## Purpose

This memo records the literature boundaries for the failed Qwen-to-Phi
denoising syndrome packet gate. The result should be framed as a negative
method probe, not a new positive ICLR method.

## Primary-Source Boundaries

| Area | Sources | Boundary |
|---|---|---|
| HellaSwag benchmark | https://arxiv.org/abs/1905.07830 | HellaSwag is the MCQ surface; do not overgeneralize to open-ended reasoning without more benchmarks. |
| Rate-distortion | Shannon 1959, https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf | Use rate-distortion as an empirical bits-vs-accuracy framing, not as new theory. |
| Decoder side information | Slepian-Wolf, https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources; Wyner-Ziv, https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf | Phi scores are decoder side information; the packet is a correction clue. Do not claim first side-information coding. |
| Syndrome coding | DISCUS reference listing, https://www.scirp.org/reference/referencespapers?referenceid=994209 | "Syndrome packet" is an analogy and design principle. Do not claim formal DISCUS optimality. |
| Denoising and diffusion | DDPM https://arxiv.org/abs/2006.11239; D3PM https://arxiv.org/abs/2107.03006; DiT https://arxiv.org/abs/2212.09748 | Our receiver is a tiny belief-vector denoiser, not a generative diffusion transformer contribution. |
| Quantized residuals and rotations | QJL https://arxiv.org/abs/2406.03482; KIVI https://arxiv.org/abs/2402.02750; KVQuant https://arxiv.org/abs/2401.18079; TurboQuant https://arxiv.org/abs/2504.19874; Google TurboQuant blog https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ | These are KV/vector compression baselines. Borrow residual/quantization intuition only; do not claim to beat them without native rows. |
| Prefix/gist tokens | Prefix tuning https://aclanthology.org/2021.acl-long.353/; Gist tokens https://arxiv.org/abs/2304.08467 | LatentWire packets are not learned prompt tokens or prompt-compression artifacts. |
| Cross-model KV/hidden communication | DroidSpeak https://arxiv.org/abs/2411.02820; C2C https://arxiv.org/abs/2510.03215; KVComm https://arxiv.org/abs/2510.03346; Interlat https://arxiv.org/abs/2511.09149 | LatentWire is not first latent communication. Safe boundary is byte-scale source-private packet evidence rather than KV/hidden transfer. |

## Claim After This Gate

Safe:

- A hand-coded `1B/4B` denoising syndrome packet was tested under strict
  source-code controls and did not beat fixed hybrid.
- The target-or-hybrid oracle remains large, so the receiver surface still has
  method headroom.
- Source-row shuffle, code permutation, candidate-roll, random-code, zero-byte,
  target-derived-code, and label-permutation controls are implemented.

Unsafe:

- Claiming solved cross-model latent reasoning.
- Claiming a positive learned receiver from this gate.
- Claiming novelty over formal syndrome/rate-distortion theory.
- Claiming systems wins over C2C, KVComm, QJL, TurboQuant, vLLM, or SGLang.
