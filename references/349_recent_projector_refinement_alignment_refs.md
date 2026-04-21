# Recent Latent / Projector / Refinement / Alignment References for LatentWire

Web check: 2026-04-21. This memo prioritizes 2024-2026 primary sources that map onto LatentWire’s cross-model communication problem, especially where the interface is latent rather than textual.

## Latent communication

- [Communicating Activations Between Language Model Agents](https://arxiv.org/abs/2501.14082). Core idea: agents can exchange intermediate activations directly, avoiding text serialization overhead and information loss. `LatentWire ablation:` replace the current bridge with activation handoff at one or two intermediate layers, then compare against text handoff and a learned projection of the same activations. `Telemetry:` exchanged bytes, layer index, activation cosine, compute per exchange, task score, communication efficiency. `Claim risks:` gains may be pair-specific and disappear when sender/receiver architecture, tokenization, or layer choice changes.

- [Enabling Agents to Communicate Entirely in Latent Space](https://arxiv.org/abs/2511.09149). Core idea: use the last hidden states themselves as the message and optionally compress them further in latent space, yielding fully latent inter-agent communication. `LatentWire ablation:` add a fully latent bottleneck between source and target, then sweep compression ratio and compare with text, activation, and KV-cache transport. `Telemetry:` latent compression ratio, reconstruction loss, entropy, answer accuracy, wall-clock speedup, message-length-normalized quality. `Claim risks:` a latent-only win can come from extra test-time compute or from the specific compression head rather than a general communication primitive.

## Multimodal projector interfaces

- [Spatial-Aware Efficient Projector for MLLMs via Multi-Layer Feature Aggregation](https://arxiv.org/abs/2410.10319). Core idea: projector quality depends not just on token count but on preserving spatial structure while compressing visual features. `LatentWire ablation:` replace the current projector with a spatial-aware aggregator and compare against flat MLP, resampler, and pooling-based bridges under equal visual-token budgets. `Telemetry:` visual-token count, spatial overlap / grounding accuracy, downstream VQA score, projector parameter count, convergence speed. `Claim risks:` spatial gains may only matter on localization-heavy tasks and may not transfer to reasoning-heavy prompts.

- [TokenPacker: Efficient Visual Projector for Multimodal LLM](https://arxiv.org/abs/2407.02392). Core idea: coarse-to-fine injection can compress visual tokens while retaining fine detail through region-to-point updates. `LatentWire ablation:` test coarse-only, coarse-to-fine, and no-compression projector variants, holding the backbone and token budget fixed. `Telemetry:` compression ratio, region coverage, fine-detail recall, hallucination rate, benchmark score, latency. `Claim risks:` improvement may depend on high-resolution inputs and shrink when the vision encoder already produces compact features.

- [Libra: Building Decoupled Vision System on Large Language Models](https://arxiv.org/abs/2405.10140). Core idea: decouple inner-modal modeling from cross-modal interaction with a routed visual expert plus a bridge module, instead of forcing one projector to do everything. `LatentWire ablation:` compare a single bridge, a decoupled vision expert plus bridge, and a frozen-backbone variant to see whether the interface should own modality-specific modeling or only alignment. `Telemetry:` routed-expert usage, cross-modal attention mass, visual-language alignment score, hallucination rate, language retention. `Claim risks:` gains may reflect extra capacity rather than a better interface, so matched-parameter controls matter.

## Diffusion / refinement transformers

- [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072). Core idea: expert-aware diffusion transformers use structured compression plus deep text-video fusion to improve long-horizon generative coherence. `LatentWire ablation:` add an expert/refinement stage after bridge transport and compare one-pass decoding vs iterative latent refinement before emission. `Telemetry:` refinement steps, latent drift, prompt alignment, coherence score, latency, final-token entropy. `Claim risks:` better outputs may come from more compute and better data curation rather than the refinement mechanism itself.

- [DiffTF++: 3D-aware Diffusion Transformer for Large-Vocabulary 3D Generation](https://arxiv.org/abs/2405.08055). Core idea: a diffusion transformer plus explicit refinement can remove artifacts and recover finer structure after coarse generation. `LatentWire ablation:` treat the bridge as a coarse proposal and add a second refinement pass over latent residuals, then compare against single-pass transport and direct decoding. `Telemetry:` proposal-vs-refinement loss, artifact rate, residual norm, edit distance to final state, downstream task fidelity. `Claim risks:` refinement can mask a weak initial bridge, so a good final score does not prove the bridge itself is high quality.

## Crosscoders / model diffing

- [Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](https://arxiv.org/abs/2504.02922). Core idea: standard crosscoder sparsity can misattribute features as model-specific; latent scaling plus BatchTopK makes shared vs chat-specific concepts more reliable. `LatentWire ablation:` train crosscoders on source/target activations, then compare L1 vs BatchTopK losses and with/without latent-scaling diagnostics before using the dictionary as the bridge. `Telemetry:` shared-latent fraction, model-specific latent rate, causal intervention effect, feature density, reconstruction error, false-specificity rate. `Claim risks:` interpretability can be inflated by the sparsity objective, so causal tests are needed before claiming a genuine communication channel.

- [Sparse Crosscoders for diffing MoEs and Dense models](https://arxiv.org/abs/2603.05805). Core idea: crosscoders can jointly model multiple activation spaces and expose how MoEs and dense models organize shared versus unique features. `LatentWire ablation:` use a multi-model crosscoder as the bridge dictionary and compare it to separate per-model SAEs and a direct linear map. `Telemetry:` shared-feature fraction, per-model decoder norms, feature density, fractional variance explained, transfer accuracy. `Claim risks:` a shared dictionary can force artificial consensus, especially if the models differ mainly in routing or capacity rather than semantics.

## Activation / representation alignment

- [Model Stitching by Functional Latent Alignment](https://arxiv.org/abs/2505.20142). Core idea: stitching should optimize functional latent alignment, not just a local feature match at the stitch point. `LatentWire ablation:` compare direct matching, task-loss stitching, and FuLA-style alignment when connecting source and target hidden spaces. `Telemetry:` stitch residual, penultimate-layer mismatch, task accuracy, cue leakage rate, alignment stability. `Claim risks:` strong stitching can overfit task cues, so a higher score may not mean the representations are broadly aligned.

- [Towards a Learning Theory of Representation Alignment](https://arxiv.org/abs/2502.14047). Core idea: representation alignment can be studied through metric, probabilistic, and spectral views, with stitching linked to kernel alignment. `LatentWire ablation:` log a kernel-alignment baseline for every bridge and test whether higher alignment predicts better transfer across tasks or model pairs. `Telemetry:` kernel alignment, CKA/SVCCA, transfer accuracy, robustness to prompt shifts, bridge norm. `Claim risks:` alignment statistics are descriptive, not sufficient; two spaces can look aligned yet still fail under causal intervention.

## Highest-priority LatentWire ablations

1. Activation handoff vs text handoff vs learned projection, under equal byte and latency budgets.
2. Fully latent inter-agent communication with a compression sweep, to separate bandwidth gains from representation gains.
3. Spatial-aware projector vs flat projector, using grounding-heavy multimodal prompts.
4. Coarse proposal plus refinement vs single-pass latent decoding, to see whether second-pass correction is doing real work.
5. Shared crosscoder dictionary vs per-model SAE plus a direct linear map, with causal intervention checks on the learned features.
