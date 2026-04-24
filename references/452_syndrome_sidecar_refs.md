# Latent Syndrome Sidecar References

Date: `2026-04-24`

## Why This Memo Exists

The current SVAMP32 blocker is not C2C headroom; it is converting that
headroom into an auditable source-necessary method. The syndrome-sidecar branch
uses target candidate pools as decoder side information and transmits a compact
source/C2C-derived check rather than dense KV or final text.

## Primary Sources

- [Cache-to-Cache Communication](https://arxiv.org/abs/2510.03215)
  Closest direct LLM-to-LLM semantic communication baseline; motivates
  distilling cache-level signal rather than source final answers.
- [KVComm](https://arxiv.org/abs/2510.03346)
  Selective KV sharing for efficient LLM communication; relevant to keeping
  the sidecar low-rate and source-specific.
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
  Learned latent-query bottleneck for arbitrary inputs and outputs; relevant if
  the syndrome head later becomes a small query module over source states.
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Frozen-backbone query bottleneck; useful design precedent for training a
  small interface while freezing both models.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Perceiver-style resampling plus gated integration; useful if future syndrome
  evidence needs low-interference target integration.
- [Slepian-Wolf Coding](https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf)
  Lossless distributed source coding with decoder side information; the
  conceptual analogue for transmitting only missing residual information.
- [Wyner-Ziv Coding](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf)
  Lossy source coding with side information available at the decoder; the
  direct information-theoretic framing for target-candidate side information.
- [AWQ](https://arxiv.org/abs/2306.00978)
  Selective protection of salient activation/weight channels; motivates
  protecting only a compact source check rather than dense state transport.
- [KIVI](https://arxiv.org/abs/2402.02750)
  Asymmetric KV-cache quantization; supports separating key/value or
  candidate/check precision rather than uniform compression.

## Practical Read

The first useful gate is not training. It is a bound: verify that target-side
candidate pools contain enough clean residual gold answers, and that compact
C2C-derived residues select them while zero/shuffle/target-only/slots-only
controls fail. If this holds, train a source-latent syndrome predictor next.
