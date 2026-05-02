# HellaSwag Switch Decomposition References

This memo supports
`paper/source_private_hellaswag_switch_decomposition_20260502.md`.

## Local Claim

The switch-only branch is not a paper contribution. The source top-2 oracle has
substantial headroom, but train-only score/hidden switch models do not capture
that headroom on either validation-first1024 or the terminal tail. The
candidate-wise hidden-innovation packet remains the live HellaSwag method.

## Primary Related Work Boundaries

- HellaSwag is the frozen commonsense completion benchmark used for this gate.
  Source: https://arxiv.org/abs/1905.07830
- Selective classification, SelectiveNet, and learning-to-defer already cover
  confidence-based accept/defer/switch decisions. Because the current switch
  gate fails, LatentWire should not claim novelty as selective prediction.
  Sources: https://arxiv.org/abs/1705.08500,
  https://arxiv.org/abs/1901.09192, and https://arxiv.org/abs/1711.06664
- Prefix tuning, prompt tuning, P-Tuning, and adapters learn persistent soft
  prompts or model modules. The LatentWire switch gate transmits only a fixed
  decision packet and does not insert learned receiver-side prefix tokens or
  change receiver weights. Sources: https://arxiv.org/abs/2101.00190,
  https://arxiv.org/abs/2104.08691, https://arxiv.org/abs/2110.07602, and
  https://arxiv.org/abs/1902.00751
- Sparse autoencoders, sparse crosscoders, and relative representations remain
  the common-basis literature. The switch decomposition does not solve shared
  latent-coordinate alignment. Sources: https://arxiv.org/abs/2309.08600,
  https://transformer-circuits.pub/2025/crosscoder-diffing/index.html, and
  https://arxiv.org/abs/2209.15430
- C2C and KVComm are close non-text communication baselines because they
  communicate or reuse model-internal state. LatentWire differs by transmitting
  a tiny packet, but native systems comparisons are still pending. Sources:
  https://arxiv.org/abs/2510.03215 and https://arxiv.org/abs/2510.03346
- QJL, TurboQuant, KIVI, and KVQuant define systems-side compression baselines
  for latent/KV state. They are not the same method as the fixed-byte switch
  gate, but they set a systems comparison floor. Sources:
  https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, and https://arxiv.org/abs/2401.18079
- Diffusion transformers and latent-reasoning methods motivate future iterative
  denoising or latent-space reasoning branches, but this gate is a fixed
  supervised switch diagnostic. Sources: https://arxiv.org/abs/2212.09748,
  https://arxiv.org/abs/2502.12134, and https://arxiv.org/abs/2412.06769

## Citation Use

Use this memo to justify cutting the switch branch:

> The source top-2 oracle reveals substantial recoverable headroom, but a
> train-only hidden+score switch captures none of it reliably; therefore the
> paper should frame dense/hybrid candidate-wise hidden-innovation packets as
> the live method and retain switch-only rows only as diagnostic evidence.

## Next Method Branch

Run candidate-wise hidden-innovation stability under stricter no-eval-leak
selection, then a cross-family falsification pair, before widening benchmarks or
claiming systems superiority.
