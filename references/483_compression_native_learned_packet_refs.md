# Compression-Native Learned Packet References

Date: 2026-04-29

## Blocker Addressed

The learned syndrome packet was positive against no-source and source-destroying controls, but reviewers can argue that the method is only a weak quantizer/sketch. This memo records primary sources used to design matched-byte compression baselines and to reframe the live method as a learned source-private packet decoded with target-side side information.

## Sources

### TurboQuant

Primary source: https://arxiv.org/abs/2504.19874

- Blocker: learned packets must beat strong vector-quantization baselines, not only text/random controls.
- Mechanism: random rotation, scalar quantization, and a low-bit residual/sign component for inner-product preservation.
- Experiment impact: motivated `scalar_quantized_source` as the first matched-byte baseline and candidate method.
- Role: baseline and systems framing.

### QJL

Primary source: https://arxiv.org/abs/2406.03482

- Blocker: sign sketches may preserve enough target-side similarity at very low byte budgets.
- Mechanism: Johnson-Lindenstrauss projection plus quantized/sign representations.
- Experiment impact: motivated `raw_source_sign_sketch` and random-hyperplane comparisons.
- Role: baseline and theory support.

### RaBitQ

Primary source: https://arxiv.org/abs/2405.12497

- Blocker: binary randomized vector quantization is a strong nearest-neighbor baseline.
- Mechanism: randomized binary quantization with distance estimation for high-dimensional search.
- Experiment impact: supports Hamming/sign-sketch baselines and nearest-candidate decode controls.
- Role: baseline and theory support.

### KVQuant

Primary source: https://arxiv.org/abs/2401.18079

- Blocker: KV/cache compression methods are natural systems competitors for model-to-model state transfer.
- Mechanism: low-bit KV-cache quantization with outlier-aware and channel-aware treatment.
- Experiment impact: not directly implemented in the tool-trace gate, but required for future real-cache packet baselines.
- Role: systems baseline.

### KIVI

Primary source: https://arxiv.org/abs/2402.02750

- Blocker: uniform quantization is not enough when competing against modern KV compression.
- Mechanism: asymmetric key/value quantization with keys and values handled differently.
- Experiment impact: future cache-transfer rows should compare against asymmetric KV compression rather than generic byte counts only.
- Role: systems baseline.

### Distributed Indirect Source Coding With Decoder Side Information

Primary source: https://arxiv.org/abs/2405.13483

- Blocker: decoder-side-information coding is a known information-theoretic setup; the novelty must be in the LLM/tool-trace method and controls.
- Mechanism: encoder sends a compact message about a correlated source while the decoder has side information.
- Experiment impact: strengthens the framing of target candidates/caches as decoder side information.
- Role: theory and framing.

### Distributed Deep JSCC With Decoder-Only Side Information

Primary source: https://arxiv.org/abs/2310.04311

- Blocker: learned encoder-decoder side-information systems exist outside LLMs.
- Mechanism: neural joint source-channel coding uses receiver-only side information during decoding.
- Experiment impact: supports the learned packet framing but does not change the immediate CPU gate.
- Role: theory and framing.

## Design Consequence

The next ICLR-strength path should not claim that random-hyperplane syndromes are the best transport. The stronger candidate is a learned source-to-target posterior vector sent through an aggressively quantized low-dimensional packet, with raw source sign sketch and same-byte text as controls.
