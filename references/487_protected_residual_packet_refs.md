# Protected Residual Packet References

- date: `2026-04-29`
- blocker: defend the packet method against quantization and compression
  reviewers who may argue the channel is an ad hoc byte code rather than a
  principled low-rate codec.

## Quantization And Sketching Sources

1. **TurboQuant**
   - source: https://arxiv.org/abs/2504.19874
   - blocker helped: reviewers will expect a modern compression baseline with
     rotations and residual corrections.
   - mechanism/design idea: random rotation plus scalar quantization plus a
     residual/sketch correction suggests a protected scalar head with a sign
     tail for source-private packets.
   - next experiment: implemented as the protected rotated residual packet gate.
   - role: inspiration and systems baseline pressure.

2. **QJL**
   - source: https://arxiv.org/abs/2406.03482
   - blocker helped: random projection/sign sketches are a natural same-byte
     comparator to learned source packets.
   - mechanism/design idea: Johnson-Lindenstrauss projection followed by
     low-bit signs gives a strong nonlearned residual comparator.
   - next experiment: included in the protected residual gate as the QJL
     residual baseline.
   - role: baseline and ablation.

3. **QuIP#**
   - source: https://arxiv.org/abs/2402.04396
   - blocker helped: motivates incoherence/rotation before low-bit coding.
   - mechanism/design idea: transform coordinates before quantization so
     low-bit scalar codes are less dominated by outlier axes.
   - next experiment: keep as future Hadamard/orthogonal-rotation ablation; the
     current gate uses random normalized rotations.
   - role: inspiration and ablation.

4. **SpinQuant**
   - source: https://arxiv.org/abs/2405.16406
   - blocker helped: random rotations may be weaker than learned rotations.
   - mechanism/design idea: learn rotations that preserve task loss under
     quantization.
   - next experiment: only worth implementing if protected residual becomes a
     live branch; current strict gate failed.
   - role: future-method inspiration.

5. **SmoothQuant**
   - source: https://proceedings.mlr.press/v202/xiao23c.html
   - blocker helped: outlier dimensions can dominate quantization error.
   - mechanism/design idea: move scale/outlier burden into a protected part of
     the representation rather than uniformly quantizing every coordinate.
   - next experiment: report protected-coordinate selection and source-control
     behavior; current gate does this at packet level.
   - role: systems inspiration.

6. **LLM.int8()**
   - source: https://arxiv.org/abs/2208.07339
   - blocker helped: mixed-precision baselines need protected high-precision
     channels.
   - mechanism/design idea: preserve salient/outlier features at higher
     precision and compress the rest.
   - next experiment: protected scalar head plus low-bit residual is the packet
     analogue.
   - role: baseline inspiration.

7. **KIVI**
   - source: https://arxiv.org/abs/2402.02750
   - blocker helped: KV-cache methods distinguish key/value quantization axes,
     so a packet paper needs a clear byte/latency comparison boundary.
   - mechanism/design idea: asymmetric treatment of components maps to a
     protected head and residual tail.
   - next experiment: cite as KV/cache compression baseline; do not claim packet
     superiority over cache quantization without endpoint rows.
   - role: systems baseline/framing.

8. **SnapKV**
   - source: https://arxiv.org/abs/2404.14469
   - blocker helped: pruning only important cache states is a strong systems
     competitor.
   - mechanism/design idea: retain task-relevant coordinates/atoms rather than
     uniformly sending all features.
   - next experiment: protected-coordinate selection is the analogous ablation
     for source-private packets.
   - role: baseline and ablation inspiration.

9. **H2O**
   - source: https://arxiv.org/abs/2306.14048
   - blocker helped: heavy-hitter retention provides a simple importance-based
     cache baseline.
   - mechanism/design idea: choose high-utility packet coordinates by
     calibration separation.
   - next experiment: current protected residual gate uses calibration-ranked
     coordinates and compares against random/QJL controls.
   - role: baseline inspiration.

10. **CacheGen / CacheBlend**
    - sources: https://arxiv.org/abs/2310.07240 and
      https://arxiv.org/abs/2405.16444
    - blocker helped: byte savings alone do not equal serving systems gains.
    - mechanism/design idea: cache compression/reuse work reports transfer,
      prefill, and fusion costs.
    - next experiment: endpoint TTFT/prefill telemetry remains required before
      claiming serving-latency superiority.
    - role: systems framing.

## Experiment Implication

The protected residual gate is useful but not promotable yet. It shows that a
principled protected-head/sign-tail codec can preserve source-control positivity
and improve the 2-byte scalar WZ frontier on some remaps, but it misses the
strict latency and high-budget scalar-preservation bar. The paper should cite it
as a compression-native comparator and near-miss, while keeping scalar WZ and
canonical RASP as the stronger current packet methods.
