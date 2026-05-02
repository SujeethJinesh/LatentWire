# References: HellaSwag Switch Observability Gate

This memo supports
`results/source_private_hellaswag_switch_observability_gate_20260502/`.
The artifact is a diagnostic decision gate, not a promoted communication
method.

## Claim Boundary

Safe claim: on the current compact HellaSwag packet/Qwen-score surface, simple
linear and RFF nonlinear probes have weak validation AUC for identifying
helpful Qwen-over-source switches, and even validation-oracle thresholds
capture almost none of the available oracle headroom.

Unsafe claim: this rules out cross-model latent communication, Q-Former-style
connectors, C2C/KV fusion, KVComm-style sharing, SAE/crosscoder shared
dictionaries, or future source-private codes trained with a stronger target
objective.

## Primary Sources

1. HellaSwag
   - https://arxiv.org/abs/1905.07830
   - Relevance: benchmark under study. Boundary: this gate is a switch
     observability diagnostic on HellaSwag validation, not a new HellaSwag
     state-of-the-art claim.

2. Random Fourier features
   - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
   - Relevance: RFF probes test whether a cheap nonlinear kernel view exposes
     switch labels better than linear probes. Boundary: RFF is prior art and is
     used only as a diagnostic falsification tool.

3. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: C2C motivates direct model-to-model communication and is a
     required competitor for any future joint connector. Boundary: this gate
     transmits no source KV/cache and makes no native latency claim.

4. KVComm and KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: these are selective/on-line KV-sharing competitors for
     inter-LLM communication. Boundary: this gate sends no KV pairs, hidden
     states, cache offsets, or raw attention state.

5. BLIP-2 / Q-Former, Perceiver IO, and Flamingo
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - Relevance: these motivate the next materially different connector branch:
     fixed-query learned bottlenecks. Boundary: a continuous query connector is
     prior art unless LatentWire contributes a distinct fixed-byte task
     syndrome or a systems result at matched baselines.

6. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Relevance: target-injected virtual tokens or soft prompts are mandatory
     baselines if the next branch emits continuous vectors. Boundary:
     LatentWire's current packet story is different because it sends only
     discrete task evidence.

7. Sparse autoencoders, universal SAE, and sparse crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2502.03714
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: shared dictionaries remain a plausible common-language route.
     Boundary: transmitting sparse activations or reconstructed hidden vectors
     would become latent-vector communication, not the current fixed-byte
     packet method.

8. QJL, TurboQuant, and vector quantization
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/1711.00937
   - Relevance: these define strong vector/KV compression byte floors and
     systems baselines. Boundary: the HellaSwag switch gate is a
     task-observability diagnostic, not vector reconstruction or KV-cache
     quantization.

## Reviewer Risk

The correct conclusion is a cut decision. Reviewers will not accept more
threshold-tuned HellaSwag selectors after this gate: the best diagnostic AUC is
only `0.561172`, and a validation-oracle threshold gives only `+0.000199`
against packet-only. The ICLR story should pivot to a materially different
source representation or a true connector, while keeping HellaSwag as a
systems/headroom/negative-ablation benchmark.
