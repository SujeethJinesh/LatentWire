# HellaSwag Qwen-To-Phi Receiver-Calibrated Gate References

Date: 2026-05-04

## Gate Boundary

This memo supports the failed official-train receiver-calibrated Qwen-to-Phi
gate. The gate is not claiming novelty in selective classification,
learning-to-defer, soft prompting, KV sharing, or latent communication broadly.
The narrow LatentWire boundary is: learn a receiver-calibrated, fixed-byte,
source-private packet accept/defer rule where the receiver's own scores act as
decoder-side information.

## Selective Prediction, Deferral, And Risk Control

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks",
  NeurIPS 2017. https://arxiv.org/abs/1705.08500
- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated
  Reject Option", ICML 2019. https://arxiv.org/abs/1901.09192
- Madras et al., "Predict Responsibly: Improving Fairness and Accuracy by
  Learning to Defer", NeurIPS 2018. https://arxiv.org/abs/1711.06664
- Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control", arXiv 2021. https://arxiv.org/abs/2110.01052
- Angelopoulos et al., "Conformal Risk Control", ICLR 2024.
  https://arxiv.org/abs/2208.02814

Boundary: the receiver-calibrated gate is a selective accept/defer problem
over Qwen packet candidates and Phi's local evidence. These works supply
calibration and risk-control machinery; the LatentWire contribution must be
the source-private packet/receiver-side-information setting and the empirical
evidence that it beats packet-only.

## Side-Information Source Coding

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources", IEEE
  Transactions on Information Theory 1973.
  https://doi.org/10.1109/TIT.1973.1055037
- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder", IEEE Transactions on Information Theory 1976.
  https://doi.org/10.1109/TIT.1976.1055508
- Whang et al., "Neural Distributed Source Coding", arXiv 2021.
  https://arxiv.org/abs/2106.02797
- Yang et al., "Learned Wyner-Ziv Compressors Recover Binning", arXiv 2023.
  https://arxiv.org/abs/2305.04380
- Yilmaz et al., "Distributed Deep Joint Source-Channel Coding with Decoder-Only
  Side Information", arXiv 2023. https://arxiv.org/abs/2310.04311

Boundary: Phi's local score simplex is decoder-side information. The failed
gate says the present linear receiver cannot exploit that side information
strongly enough, despite large oracle headroom.

## Model-To-Model And Multi-Agent Communication Baselines

- Li et al., "CAMEL: Communicative Agents for Mind Exploration of Large Scale
  Language Model Society", arXiv 2023. https://arxiv.org/abs/2303.17760
- Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent
  Conversation", arXiv 2023. https://arxiv.org/abs/2308.08155
- Du et al., "Improving Factuality and Reasoning in Language Models through
  Multiagent Debate", arXiv 2023. https://arxiv.org/abs/2305.14325
- Hong et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative
  Framework", arXiv 2023. https://arxiv.org/abs/2308.00352

Boundary: these communicate through text or agent protocols. LatentWire is a
fixed-byte candidate packet with explicit source-private controls.

## Prefix, Gist, And Latent-Token Compression

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
  ACL 2021. https://arxiv.org/abs/2101.00190
- Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning",
  EMNLP 2021. https://arxiv.org/abs/2104.08691
- Mu et al., "Learning to Compress Prompts with Gist Tokens", NeurIPS 2023.
  https://arxiv.org/abs/2304.08467
- Ge et al., "In-Context Autoencoder for Context Compression in a Large
  Language Model", ICLR 2024. https://arxiv.org/abs/2307.06945

Boundary: soft-prefix/gist methods compress prompts for a target model. The
current gate is discrete, source-private, and receiver-calibrated. A future
resonance branch must start with target self-compression before claiming
cross-model latent transfer.

## KV/Hidden Compression And Systems Context

- KIVI: "A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024.
  https://arxiv.org/abs/2402.02750
- KVQuant: "KVQuant: Towards 10 Million Context Length LLM Inference with KV
  Cache Quantization", NeurIPS 2024. https://arxiv.org/abs/2401.18079
- QJL: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead", arXiv 2024. https://arxiv.org/abs/2406.03482
- TurboQuant: "TurboQuant: Taming LLMs with Ternary Quantization", arXiv 2025.
  https://arxiv.org/abs/2504.19874

Boundary: these compress continuous model state for serving. LatentWire's
packet is task-level and byte-scale, but no systems superiority should be
claimed until quality is positive and native NVIDIA serving rows exist.

## Diffusion And Iterative Refinement

- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces",
  NeurIPS 2021. https://arxiv.org/abs/2107.03006
- Li et al., "Diffusion-LM Improves Controllable Text Generation", NeurIPS 2022.
  https://arxiv.org/abs/2205.14217
- Ye et al., "Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion
  Language Models", arXiv 2024. https://arxiv.org/abs/2402.07754
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback", NeurIPS
  2023. https://arxiv.org/abs/2303.17651

Boundary: iterative refinement is relevant only if the receiver has a calibrated
stop/accept rule. Fixed-depth refinement over the current features risks
over-refining and should not be promoted without harm control.

## Paper Implication

The receiver-calibrated branch is alive but not solved. The next experiment
should move from a linear utility selector to explicit harm-controlled
accept/defer buckets or to target self-compression/resonance. The paper should
avoid broad latent-communication claims until a positive row survives the
official-train receiver gate.
