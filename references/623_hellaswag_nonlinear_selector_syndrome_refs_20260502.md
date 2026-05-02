# References: HellaSwag Nonlinear Selector/Syndrome Gate

This memo supports
`results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502/`.
The artifact is a negative train-only method gate for fixed-byte nonlinear
source-private selector/syndrome packets.

## Claim Boundary

Safe claim: bounded nonlinear random Fourier feature benefit prediction over
the current compact packet/syndrome and Qwen receiver-side score features does
not recover a meaningful fraction of measured TinyLlama/Qwen HellaSwag oracle
headroom.

Unsafe claim: this rules out learned query connectors, C2C-style KV fusion,
KVComm-style selective KV sharing, sparse dictionaries, future source-private
task syndromes, or NVIDIA-native joint connectors.

## Primary Sources

1. Random Fourier features
   - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
   - Relevance: this gate uses random Fourier features as a bounded nonlinear
     RBF-kernel approximation before ridge benefit prediction. Boundary: RFF
     is not the technical contribution; it is a cheap falsification of the
     hypothesis that the previous selector failed only because it was linear.

2. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: C2C is the closest direct model-to-model communication prior
     because it projects and fuses source KV cache into the receiver. Boundary:
     LatentWire transmits no source KV/cache and cannot claim C2C-quality or
     native-latency wins without native baselines.

3. KVComm and KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: these are selective/on-line KV-sharing competitors for
     inter-LLM communication. Boundary: this gate sends only a task-level
     discrete packet, not KV pairs, hidden states, cache offsets, or raw
     attention state.

4. BLIP-2 / Q-Former, Perceiver IO, and Flamingo
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - Relevance: learned query/resampler bottlenecks are the right nonlinear
     connector prior if we leave selectors. Boundary: continuous query states
     or target-side cross-attention are not a fixed-byte packet method unless
     quantized to a discrete source-private syndrome.

5. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Relevance: any target-injected continuous prefix, soft prompt, or virtual
     token interface is prior art. Boundary: this gate uses no target-injected
     continuous vector or virtual token sequence.

6. Sparse autoencoders, universal SAE, and sparse crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2502.03714
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: shared sparse dictionaries are a plausible next common-basis
     method. Boundary: transmitting sparse activations would become latent
     vector communication; LatentWire's safer novelty is private dictionary
     computation used to emit a tiny task-level syndrome.

7. QJL, TurboQuant, and vector quantization
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/1711.00937
   - Relevance: these define strong vector/KV compression baselines and
     quantization-inspired byte floors. Boundary: LatentWire packets are
     decision/syndrome codes optimized for task distortion, not activation
     reconstruction, inner-product preservation, or KV-cache compression.

## Reviewer Risk

Reviewers will not accept the eval-only `+0.002191` scout as a contribution.
The correct use is a negative ablation: HellaSwag complementarity exists, but
the current packet plus train-only receiver-side nonlinear selectors cannot
recover it. The next positive-method branch must either change source
information, train a true joint connector, or cut the receiver-improvement
claim from the ICLR story.
