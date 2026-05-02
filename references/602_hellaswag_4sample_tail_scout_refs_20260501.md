# HellaSwag Four-Sample Terminal-Tail Scout References

This memo supports
`paper/source_private_hellaswag_4sample_tail_scout_20260501.md`. The local
result is a negative stabilization scout: adding a fourth train-sample seed
keeps an overall positive terminal-tail margin, but does not clear the strict
jackknife gate.

## Local Claim

The four-sample scout weakens the simple-bagging rescue branch. LatentWire
should continue to claim the contiguous HellaSwag validation `0:9216` pass and
the positive-but-insufficient terminal-tail diagnostic, not a full-validation
pass.

## Primary Related Work Boundaries

- HellaSwag is the commonsense completion benchmark used for this frozen-slice
  gate. Source: https://arxiv.org/abs/1905.07830
- Selective prediction and calibration are relevant to future trust-or-switch
  packets, but this four-sample scout is not a new selective-classification
  method. Source: https://arxiv.org/abs/1705.08500
- Prefix tuning, prompt tuning, P-Tuning, and adapters learn persistent
  conditioning parameters or model-side modules. LatentWire's current packet is
  per-example, fixed-byte, and source-private rather than a learned soft prompt
  or adapter. Sources: https://arxiv.org/abs/2101.00190,
  https://arxiv.org/abs/2104.08691, https://arxiv.org/abs/2110.07602, and
  https://arxiv.org/abs/1902.00751
- Sparse autoencoders, universal sparse autoencoders, sparse crosscoders, and
  relative representations remain the relevant common-basis literature. The
  four-sample scout does not establish a universal latent coordinate system.
  Sources: https://arxiv.org/abs/2309.08600,
  https://arxiv.org/abs/2502.03714,
  https://transformer-circuits.pub/2024/crosscoders/index.html, and
  https://arxiv.org/abs/2209.15430
- C2C, KVComm, KVCOMM, and Q-KVComm are close non-text communication
  competitors because they transmit or reuse source-side cache/state. LatentWire
  differs by exposing no source KV cache or raw hidden state, but native
  matched comparisons remain pending. Sources: https://arxiv.org/abs/2510.03215,
  https://openreview.net/forum?id=F7rUng23nw,
  https://arxiv.org/abs/2510.12872, and https://arxiv.org/abs/2512.17914
- QJL, TurboQuant, KIVI, and KVQuant define systems-side source-state and
  KV-cache compression floors. They are comparison rows for a systems table,
  not substitutes for the current source-private packet. Sources:
  https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, and https://arxiv.org/abs/2401.18079
- Diffusion transformers and latent-reasoning methods motivate future
  iterative repair or hidden-state denoising branches. This scout remains a
  fixed-byte classification repair diagnostic. Sources:
  https://arxiv.org/abs/2212.09748, https://arxiv.org/abs/2502.12134, and
  https://arxiv.org/abs/2412.06769

## Citation Use

Use this memo to make the reviewer-facing boundary explicit:

> Adding a fourth train-sample bag does not solve the HellaSwag terminal-tail
> robustness failure. The current contribution is therefore a strong bounded
> source-private packet result plus strict negative evidence, not a full
> benchmark-complete latent communication result.

## Next Branch

The next Mac-local method branch should not be another dense bagging sweep. It
should add a new mechanism: a hidden-private top-2 trust-or-switch packet or an
anchor/common-basis repair that has a plausible reason to change terminal-tail
failure modes while preserving the current packet/exposure accounting.
