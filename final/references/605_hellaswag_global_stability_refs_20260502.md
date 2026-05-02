# HellaSwag Global Stability References

This memo supports
`paper/source_private_hellaswag_global_stability_20260502.md`.

## Local Claim

The candidate-wise hidden-innovation packet clears full frozen HellaSwag
validation with `10042` rows, `3/3` leave-one-train-sample subbags, and `10/10`
contiguous slice passes. The safe novelty claim is an extreme-rate,
source-private task-evidence packet, not first latent communication, not prompt
tuning, not KV/cache compression, and not native systems superiority.

## Primary Related Work Boundaries

- HellaSwag defines the frozen commonsense completion benchmark for this gate.
  Source: https://arxiv.org/abs/1905.07830
- Prefix tuning and prompt tuning learn continuous task-specific vectors or soft
  prompt tokens for frozen models. LatentWire sends a per-example discrete
  source-private packet and does not insert learned prefix vectors into the
  receiver. Sources: https://aclanthology.org/2021.acl-long.353/ and
  https://arxiv.org/abs/2104.08691
- LoRA and adapters are parameter-efficient adaptation methods. LatentWire does
  not adapt receiver weights. Sources: https://arxiv.org/abs/2106.09685 and
  https://arxiv.org/abs/1902.00751
- Selective classification, SelectiveNet, and learning-to-defer cover
  accept/reject/defer decision theory. The top-2 switch-only branch failed, so
  LatentWire should frame selection only as part of the destructive-control
  evaluation ladder. Sources: https://arxiv.org/abs/1705.08500,
  https://arxiv.org/abs/1901.09192, and https://arxiv.org/abs/1711.06664
- C2C and KVComm are direct inter-model communication baselines that transmit or
  fuse source KV/cache information. LatentWire differs by transmitting only a
  `2B` raw / `5B` framed task packet, but native systems comparisons remain
  pending. Sources: https://arxiv.org/abs/2510.03215 and
  https://arxiv.org/abs/2510.03346
- QJL, TurboQuant, KIVI, and KVQuant define quantized-vector or KV-cache
  systems floors. They are mandatory comparisons for bytes/HBM claims, but they
  are not the same object as a fixed-byte candidate packet. Sources:
  https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, and https://arxiv.org/abs/2401.18079
- Sparse autoencoder universality, sparse crosscoders, and relative
  representations remain the common-basis literature for future branches.
  Sources: https://arxiv.org/abs/2410.06981,
  https://transformer-circuits.pub/2025/crosscoder-diffing/index.html, and
  https://arxiv.org/abs/2209.15430
- vLLM and SGLang define native serving baselines for future systems rows.
  Current evidence supports byte/exposure accounting only. Sources:
  https://arxiv.org/abs/2309.06180 and https://arxiv.org/abs/2312.07104

## Citation Use

Use this memo to defend the promoted contribution:

> A fixed-byte source-private hidden-innovation packet improves full frozen
> HellaSwag validation accuracy over label-copy and score-only controls while
> destructive hidden-corruption controls collapse, suggesting the packet carries
> matched source hidden evidence rather than a source label, score shortcut, or
> exposed model state.

Use this memo to avoid overclaiming:

> LatentWire is not a replacement for C2C/KVComm or KV-cache compression. The
> current contribution is a stricter privacy/rate point; native serving
> superiority remains an open systems gate.

## Next Method Branch

Run the strict cross-family falsification pair next. If the dense packet fails
cross-family, the highest-value method branch is a sparse residual dictionary or
relative-residual anchor packet that creates a more stable common basis while
preserving the `2B` raw / `5B` framed contract.
