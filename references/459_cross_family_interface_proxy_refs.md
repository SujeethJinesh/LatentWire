# Cross-Family Interface Proxy References

Focused memo for the tokenizer/interface proxy gate after OPT-350m proved too
weak as a real benchmark surface.

## Sources

1. Singh et al., "Cross-Tokenizer LLM Distillation through a Byte-Level
   Interface." arXiv:2604.07466.
   <https://arxiv.org/abs/2604.07466>
   - Problem: tokenizer mismatch makes source/target token-level alignment
     brittle.
   - Mechanism: distill through a shared byte-level interface rather than a
     token-vocabulary map.
   - Experiment impact: keep byte-level sidecars and byte-span remap as the
     most defensible cross-tokenizer interface baseline.
   - Role: baseline and inspiration.

2. Zhao et al., "DWA-KD: Dual-Space Weighting and Time-Warped Alignment for
   Cross-Tokenizer Knowledge Distillation." arXiv:2602.21669.
   <https://arxiv.org/abs/2602.21669>
   - Problem: token sequences with different tokenizers need sequence-level
     alignment, not just local token overlap.
   - Mechanism: Soft-DTW on embeddings and hidden states.
   - Experiment impact: supports sequence-aligned sidecar features and suggests
     adding a Soft-DTW-style ablation if the real sidecar is implemented.
   - Role: inspiration and ablation.

3. Yao et al., "TokAlign: Efficient Vocabulary Adaptation via Token Alignment."
   ACL 2025 / arXiv:2506.03523.
   <https://arxiv.org/abs/2506.03523>
   <https://aclanthology.org/2025.acl-long.207.pdf>
   - Problem: vocabulary/interface mismatch creates transfer loss.
   - Mechanism: align tokens through co-occurrence statistics.
   - Experiment impact: use as a fair token-alignment comparator only if we
     build an explicit token-level vocabulary bridge.
   - Role: baseline and ablation.

4. Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
   Distortion Rate." arXiv:2504.19874.
   <https://arxiv.org/abs/2504.19874>
   - Problem: transported sidecars/KV caches are too large for a systems claim.
   - Mechanism: data-oblivious vector quantization with residual correction.
   - Experiment impact: if a sidecar row works, add a polar/vector-quantized
     sidecar ablation with equal accuracy and bytes/latency accounting.
   - Role: systems inspiration and compression ablation.

5. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
   Cache." arXiv:2402.02750 / ICML 2024.
   <https://arxiv.org/abs/2402.02750>
   - Problem: cache transport can lose the systems argument if it ships raw
     high-dimensional tensors.
   - Mechanism: asymmetric 2-bit KV quantization, with different treatment of
     key and value cache distributions.
   - Experiment impact: use K/V-specific byte budgets and report whether keys
     alone preserve the effect.
   - Role: systems baseline.

6. Liu et al., "SpinQuant: LLM Quantization with Learned Rotations." ICLR
   2025.
   <https://openreview.net/forum?id=ogO6DGE6FZ>
   - Problem: naive quantization and basis mismatch can amplify outliers.
   - Mechanism: learned rotations improve quantization robustness.
   - Experiment impact: motivates learned/gauge-fixed rotations before sidecar
     quantization, but only after source controls pass.
   - Role: compression inspiration.

7. Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
   Language Models." arXiv:2510.03215 / ICLR 2026.
   <https://arxiv.org/abs/2510.03215>
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Problem: direct cache communication is the strongest comparator for this
     project.
   - Mechanism: project source KV caches into target cache space.
   - Experiment impact: every promoted row must compare against C2C or clearly
     beat it on bytes/latency at comparable accuracy.
   - Role: baseline.

## Design Rule

Do not spend more large compute on a cross-family proxy until the target and
text-relay baselines have nonzero headroom on the exact-ID slice. Once a real
sidecar row passes, immediately run byte/latency ablations against C2C, KIVI,
and a TurboQuant-inspired vector-quantized sidecar.

