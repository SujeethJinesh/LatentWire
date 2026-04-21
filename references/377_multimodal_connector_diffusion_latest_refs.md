# Multimodal Connector / Diffusion Latest References for LatentWire

Web check: 2026-04-21. This memo prioritizes 2025-2026 primary sources, with one 2024 anchor where it still defines the cleanest Perceiver-style resampler baseline.

## Sources

### Connector bottlenecks / Q-Former- or Perceiver-style resamplers

- [VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs](https://arxiv.org/abs/2510.16598) (2025-10-18). Learnable scorer decoupled from the backbone, with differentiable Top-K and curriculum annealing. Strong evidence that connector quality is largely a token-selection problem, not just a projector-capacity problem.
- [FlashVLM: Text-Guided Visual Token Selection for Large Multimodal Models](https://arxiv.org/abs/2512.20561) (2025-12-23). Uses explicit cross-modal similarity in LM space plus diversity-preserving background retention. Good reference for query-conditioned selection without relying on unstable deep attention maps.
- [InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression](https://arxiv.org/abs/2503.21307) (2025-03-27). Introduces a local/global-query projector (`PVTC`) and layer-wise compression/expansion (`LVTC`). Closest recent resampler variant to compare against Q-Former/Perceiver-style connectors.
- [Look-Back: Implicit Visual Re-focusing in MLLM Reasoning](https://arxiv.org/abs/2507.03019) (2025-07-02). Shows later reasoning can re-focus on visual inputs without explicit reinjection. Useful for recurrent connector schedules instead of one-shot visual handoff.
- [Large Language Models Facilitate Vision Reflection in Image Classification](https://arxiv.org/abs/2508.06525) (2025-08-02). Finds the connector maps visual features into compact textual concepts and that a few text-like tokens can often replace many raw vision tokens. Important if LatentWire should communicate concepts rather than dense features.
- [PaLM2-VAdapter: Progressively Aligned Language Model Makes a Strong Vision-language Adapter](https://arxiv.org/abs/2402.10896) (2024-02-16, anchor baseline). Still the cleanest explicit statement that Perceiver resamplers can converge slowly and scale poorly without direct supervision.

### Latent diffusion / refinement interfaces

- [Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space](https://arxiv.org/abs/2510.12603) (2025-10-14). Represents each reasoning step as latent text plus selected latent vision, trained with a progressive multi-stage schedule. Probably the most direct multimodal latent-reasoning reference for LatentWire.
- [Multimodal Latent Language Modeling with Next-Token Diffusion](https://arxiv.org/abs/2412.08635) (2024-12-11). Uses a VAE latent interface plus next-token diffusion to unify discrete and continuous modalities. Relevant if the bridge should emit latent vectors rather than pseudo-tokens.
- [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](https://arxiv.org/abs/2512.15713) (2025-12-17; rev. 2026-03-31). Converts AR VLMs into diffusion VLMs with block decoding and KV-cache reuse. Suggests LatentWire can add iterative repair without replacing its backbone.
- [X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation](https://arxiv.org/abs/2503.06134) (2025-03-08, ICCV 2025). Distills multimodal understanding into a lightweight `AlignNet` bridge with attention supervision. Good bridge-only training recipe.
- [Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](https://arxiv.org/abs/2506.07986) (2025-06-09, ICCV 2025). Identifies token-imbalance and missing timestep-aware weighting as core alignment failures in MM-DiTs; proposes `TACA`. Useful if bridge strength should depend on timestep or modality ratio.
- [Reasoning with Latent Tokens in Diffusion Language Models](https://arxiv.org/abs/2602.03769) (2026-02-03). Shows latent-token count is a controllable compute-quality dial and can be ported back into AR models. Useful if LatentWire wants variable-bandwidth communication.

### Training objectives / alignment / cross-model communication

- [Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions](https://arxiv.org/abs/2602.09483) (2026-02-10). Distills vision-instruction interactions and response-token transitions, not just next-token logits. Strong argument against response-only KD for bridge training.
- [Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization](https://arxiv.org/abs/2506.04039) (2025-06-04, EMNLP 2025). Makes preference optimization explicitly modality-aware so language priors do not swamp image evidence.
- [TARS: MinMax Token-Adaptive Preference Strategy for Hallucination Reduction in MLLMs](https://arxiv.org/abs/2507.21584) (2025-07-29). Uses token-adaptive adversarial preference training plus spectral alignment. Useful if LatentWire needs robustness to visually-agnostic or stale bridge tokens.
- [The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems](https://arxiv.org/abs/2602.15382) (2026-02-17). Reuses a VLM visual pathway as a universal latent port and reduces alignment complexity from pairwise `O(N^2)` to hub-and-spoke `O(N)`. Very relevant for heterogeneous cross-model communication.
- [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215) (2025-10-03). Projects and fuses source KV cache into the target model instead of serializing to text.
- [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://arxiv.org/abs/2510.03346) (2025-10-02). Shares only informative KV layers, which is a good control against over-crediting high-bandwidth latent channels.

## Why It Matters For LatentWire

- The recent connector literature is converging on a simple point: the bottleneck is usually `which` visual or latent units cross the boundary, not just how many. Query-conditioned selection and local/global query resampling look stronger than static MLP projection.
- Several 2025 papers imply the connector already behaves like a concept bottleneck. If a few distilled concept-like tokens preserve performance, LatentWire should test semantic bottlenecks directly instead of assuming dense hidden-state transfer is necessary.
- The best latent-interface papers treat communication as an iterative state that can be revised, not a one-shot projection. That maps well to LatentWire's diffusion/refinement direction.
- The training signal is moving away from plain next-token KL. Attention distillation, token-interaction distillation, modality-aware preference optimization, and token-adaptive robustness objectives all look more aligned with connector quality.
- Cross-model communication work now has a credible hub-and-spoke story: shared codec or shared KV medium can reduce pairwise translator complexity and may generalize better than bespoke source-target adapters.

## Concrete Ablations / Diagnostics

- Compare `static projector` vs `local/global query resampler` vs `query-conditioned selector` vs `explicit LM-space similarity selector`, holding transferred bytes fixed.
- Run token-budget sweeps at aggressive retention rates, e.g. `10%`, `20%`, `30%`, `50%`, and log both task accuracy and whether the bridge collapses to a few concept tokens.
- Test `one-shot handoff` vs `recurrent look-back` vs `latent diffusion repair`, with matched wall-clock and matched target-side compute.
- Compare bridge payload types: `dense latent slots`, `few concept tokens`, `projected KV slices`, and `mixed latent+token` payloads.
- Add a `block-decoded refinement` ablation with KV reuse to separate the value of iterative repair from the value of a larger bridge.
- Train the same connector with `next-token KL`, `attention distillation`, `token-interaction distillation`, and `entity-centric / token-adaptive preference optimization`.
- Evaluate `pairwise translator` vs `shared hub codec` vs `visual-port universal codec`, especially on held-out sender/receiver pairs.
- Log diagnostics that expose real communication quality: source-target cosine / CKA, retained-token maps, attention mass on reinjected latents, bridge entropy, and hallucination under modality-drop tests.
- Include hard controls for fake wins: `zero-byte target-cache` control, shuffled-source control, random-background-token control, and frozen-target vs unfrozen-target controls.
- Report byte-normalized and latency-normalized performance against text handoff, latent handoff, and KV handoff so gains are not explained away by extra hidden bandwidth.
