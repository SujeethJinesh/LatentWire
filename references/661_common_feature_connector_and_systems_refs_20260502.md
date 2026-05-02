# Common-Feature Connector and Systems References

Date: 2026-05-02

## Purpose

This memo records the external prior-work boundary for the next branch after the
ARC Phi-3 failure decomposition. The safe next method is not another raw
Procrustes/PCA alignment; it is a fixed-byte, source-private, target-conditioned
communication packet built from source-only sparse innovations.

## Common-Feature and Sparse-Dictionary Priors

- Universal Sparse Autoencoders: https://arxiv.org/abs/2502.03714
  USAE motivates a shared sparse concept space across models. LatentWire would
  differ by transmitting a per-example fixed-byte packet and testing causal
  receiver behavior, not just reconstructing activations.
- Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language
  Models: https://arxiv.org/abs/2410.06981
  This supports the hypothesis that some LLM features can align across models.
  It is not itself a task-packet protocol.
- Cross-Architecture Model Diffing with Crosscoders:
  https://arxiv.org/abs/2602.11729
  Cross-architecture crosscoders motivate shared/private feature separation for
  non-identical model families.
- Delta-Crosscoder: https://arxiv.org/abs/2603.04426
  Delta-crosscoders motivate isolating behavior-causing latent directions, but
  remain a model-diffing tool rather than a fixed-byte communication method.
- SAEBench: https://arxiv.org/abs/2503.09532
  Use as a caution that sparse-feature quality needs explicit evaluation, not
  just reconstruction loss.

## Connector and Prefix Baselines

- Relative Representations: https://openreview.net/forum?id=SrC-nwieGJ
  This is the closest common-coordinate precedent; LatentWire should not claim
  public anchors or relative coordinates as novel by themselves.
- Model Stitching: https://arxiv.org/abs/2506.06609
  Model stitching transfers intermediate representations through learned maps.
  It is an alignment baseline, not a source-private packet protocol.
- Prefix-Tuning: https://arxiv.org/abs/2101.00190
  Continuous prefixes are a baseline for frozen-LM conditioning.
- Gist Tokens: https://arxiv.org/abs/2304.08467
  Gist tokens motivate prompt-compression and soft-token controls.
- Perceiver IO: https://arxiv.org/abs/2107.14795
  Perceiver-style latent resamplers motivate tokenwise query bottlenecks.
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
  Q-Former is a strong precedent for query tokens extracting target-consumable
  information from a different representation stream.
- Consistency Models: https://arxiv.org/abs/2303.01469
  Consistency objectives motivate one-step denoising/repair controls for noisy
  packets, but should be lightweight on Mac.

## Quantization and Systems Baselines

- QJL: https://arxiv.org/abs/2406.03482
  QJL is a 1-bit Johnson-Lindenstrauss/sign-sketch KV-cache comparator and a
  possible codec component.
- TurboQuant: https://arxiv.org/abs/2504.19874
  TurboQuant motivates rotation plus scalar/residual quantization. Treat it as
  a vector/KV quantization baseline, not a solved semantic communication method.
- C2C: https://arxiv.org/abs/2510.03215
  C2C is the closest high-bandwidth source-KV communication baseline.
- KVComm: https://arxiv.org/abs/2510.03346
  KVComm and related KVCOMM work are source-state/KV exposure baselines.
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  Native systems rows should use vLLM TTFT/TPOT/goodput/GPU-memory metrics.
- SGLang/RadixAttention: https://arxiv.org/abs/2312.07104
  Use SGLang as the second native serving/runtime baseline.

## Safe Novelty Framing

The target contribution should be framed as a fixed-byte, source-private,
task-conditioned communication protocol that uses sparse/common features only
as a means of constructing and testing a causal packet. Prior work aligns,
interprets, compresses, or conditions models; the novelty must be the
source-private packet contract, strict destructive controls, and matched
systems/exposure accounting.
