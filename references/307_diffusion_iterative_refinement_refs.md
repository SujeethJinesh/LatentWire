# Diffusion / Iterative Refinement Memo for LatentWire

Primary sources worth stealing from:

- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent](https://arxiv.org/abs/2302.09057)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [Language Rectified Flow: Advancing Diffusion Language Generation with Probabilistic Flows](https://arxiv.org/abs/2403.16995)
- [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://arxiv.org/abs/2409.02908)
- [LatentLM: Multimodal Latent Language Modeling with Next-Token Diffusion](https://arxiv.org/abs/2412.08635)
- [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)
- [Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052)
- [FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models](https://arxiv.org/abs/2509.20624)
- [Transition Matching: Scalable and Flexible Generative Modeling](https://arxiv.org/abs/2506.23589)
- [Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](https://arxiv.org/abs/2510.03206)
- [Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models](https://arxiv.org/abs/2511.08577)
- [Evo: Autoregressive-Diffusion Large Language Models with Evolving Balance](https://arxiv.org/abs/2603.06617)
- [VDLM: Variable Diffusion LMs via Robust Latent-to-Text Rendering](https://arxiv.org/abs/2602.15870)

Math ideas to steal:

- Consistency across refinement steps: constrain the same latent state to map to the same denoised output under different step budgets.
- Rectified / flow-matching updates: replace noisy jumpy transitions with an explicit velocity or transport field over latent states.
- Belief-state refinement: keep uncertain positions as mixtures instead of hard commits, then progressively finalize.
- Budget-conditioned sampling: make the number of refinement steps a control input, not a fixed constant.
- Confidence-adaptive updates: allocate more compute only to hard tokens / unstable slots / high-entropy heads.
- Blockwise bidirectional refinement: let a latent block revise itself jointly before committing outward.
- Joint discrete-continuous transitions: move between token-space and latent-space with an explicit renderer / decoder.
- KL- or entropy-based stopping: terminate refinement when successive states stop changing meaningfully.

LatentWire ablations to run:

- `topk` vs `query_pool` vs `route_atom` vs `preconditioned_query_pool` under the same byte budget.
- Fixed-step refinement vs confidence-adaptive refinement on the same examples.
- 1 / 2 / 4 / 8 latent refinement steps with matched total compute.
- Hard commit vs mixture-based belief states for translated K/V slots.
- Identity / no-op refinement as a control to separate “extra compute” from actual transport.
- Token-level vs head-level vs block-level refinement, keeping the interface constant.
- Entropy-gated routing and step skipping for uncertain slots.
- Preconditioning on values, queries, or both to test whether scale conditioning is the real gain.

Interpretability telemetry to log:

- Step-wise KL / JS divergence between successive latent states.
- Per-step entropy, top-1 margin, and dead-slot / dead-head rates.
- Update norm ratio, cosine drift, and sign flip rate across refinement iterations.
- Budget utilization per example, per head, and per slot.
- Step histogram for when the system actually commits versus keeps refining.
- Calibration curves and ECE for route confidence.
- Reconstruction fidelity alongside task accuracy so gains are not just compression artifacts.
- Stability under seeds, perturbations, and schedule changes.

Practical read:

- The strongest cross-over for LatentWire is not “more steps by default”; it is “controlled refinement with explicit uncertainty accounting.”
- The best next positive-method candidate is a learned query-conditioned interface that can behave like `query_pool` when stable, and like a route-atom / selective refinement mode when the latent state is uncertain.
- The next paper-facing claim should be about interpretable compression plus selective refinement, not raw diffusion-style generation.
