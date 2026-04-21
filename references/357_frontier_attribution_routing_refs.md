# Frontier Attribution and Routing References for LatentWire

Web check: 2026-04-21. This memo is for protected-frontier selection after pruning, not for a generic interpretability survey. The goal is to borrow mechanisms that help us decide which frontier nodes, features, or routes should survive when compute is tight.

## Primary sources

- **[Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models](https://arxiv.org/abs/2410.06981)**. `Core idea:` sparse autoencoders turn dense residual streams into sparse, more interpretable feature bases that can be matched across models. `LatentWire ablation:` train an SAE on the frontier state and select the protected frontier by top SAE feature mass instead of raw activation magnitude. `Telemetry to log:` feature sparsity, feature overlap across models, reconstruction error, protected-feature hit rate, repair help/harm. `Claim risk:` a sparse basis can make the frontier look cleaner without proving the kept features are the right causal ones.

- **[Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html)**. `Core idea:` crosscoders read and write to multiple layers, giving one feature set that can track persistent signals across depth. `LatentWire ablation:` use a crosscoder-style frontier selector to keep features that persist across layers, then compare against per-layer pruning and scalar saliency. `Telemetry to log:` cross-layer feature persistence, dead-feature rate, frontier coverage, reconstruction loss, repair delta. `Claim risk:` cross-layer persistence may be a good compression objective without being the best selection rule for repair.

- **[Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)**. `Core idea:` attribution graphs built from transcoders provide a feature-level causal graph and a practical route to sparse circuit discovery. `LatentWire ablation:` replace raw frontier scoring with graph-edge attribution on the replacement model, then keep only frontier nodes on high-attribution paths. `Telemetry to log:` edge attribution mass, graph sparsity, path overlap, frontier retention rate, post-prune accuracy. `Claim risk:` graph readability can outrun causal faithfulness if the replacement model is only an approximation.

- **[AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/abs/2403.00745)**. `Core idea:` attribution patching is a fast surrogate for node patch effect and comes with diagnostics for cancellation and saturation. `LatentWire ablation:` score frontier nodes with AtP*-style attribution patching, then compare to exact ablation on a small calibration slice before pruning. `Telemetry to log:` attribution-vs-ablation rank correlation, cancellation rate, saturation rate, frontier precision/recall, compute saved. `Claim risk:` fast attribution can still mis-rank nodes when effects cancel or saturate, so the memo should not overclaim faithfulness.

- **[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)**. `Core idea:` causal tracing localizes decisive activations for factual recall and connects localization to model editing. `LatentWire ablation:` use causal-tracing heatmaps to choose which frontier positions to protect, then test whether protecting the traced zone is better than protecting the top-magnitude zone. `Telemetry to log:` traced layer/position peak, protection overlap, recovery accuracy, false-protect rate, false-drop rate. `Claim risk:` causal peaks can be useful for localization yet still be the wrong basis for selecting a surviving frontier.

- **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)**. `Core idea:` sparse routing with top-1 expert selection makes conditional compute practical, but only if routing is stable. `LatentWire ablation:` turn protected-frontier selection into top-k sparse routing over candidate repair nodes, and compare load-balanced vs greedy vs random routing. `Telemetry to log:` load entropy, expert utilization, route collapse rate, spare capacity, repair success, latency. `Claim risk:` better routing balance may improve efficiency while masking whether the frontier itself is meaningfully better.

- **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)**. `Core idea:` conditional computation can scale sparse experts if routing and sharding are handled carefully. `LatentWire ablation:` test sharded frontier experts and compare a static protected-frontier budget against a learned router with the same byte cap. `Telemetry to log:` shard utilization, dispatch imbalance, token-to-expert entropy, end-to-end bytes, held-out accuracy. `Claim risk:` the benefit may come from scale engineering rather than from a better frontier selection rule.

- **[Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)**. `Core idea:` dynamically bias router scores to prevent expert collapse without a destabilizing auxiliary loss. `LatentWire ablation:` apply bias-corrected protected-frontier routing so frequently selected nodes do not monopolize the repair budget. `Telemetry to log:` selection histogram, collapse rate, bias drift, repair diversity, accuracy per byte. `Claim risk:` preventing collapse can improve coverage but may not improve task fidelity if the wrong nodes remain evenly distributed.

- **[Efficient Deweather Mixture-of-Experts with Uncertainty-aware Feature-wise Linear Modulation](https://arxiv.org/abs/2312.16610)**. `Core idea:` uncertainty-aware routing assigns inputs to experts with calibrated weights. `LatentWire ablation:` protect frontier nodes only when the router uncertainty is low, and fall back to a broader frontier when uncertainty is high. `Telemetry to log:` router entropy, calibration error, protect/skip threshold, missed-help rate, false-protect rate. `Claim risk:` uncertainty-aware selection may mostly be a confidence filter and not a new causal selector.

- **[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** and **[The (Un)Reliability of Saliency Methods](https://arxiv.org/abs/1711.00867)**. `Core idea:` SHAP-style perturbation attribution is model-agnostic, but saliency methods are brittle under input shifts and reference choices. `LatentWire ablation:` compare perturbation-based frontier saliency to learned-attribution and patching-based scores, then check whether the protected frontier changes under harmless prompt shifts. `Telemetry to log:` saliency stability, input-invariance score, perturbation cost, selection agreement, repair help/harm. `Claim risk:` perturbation saliency can be expensive and unstable, so it should be treated as a calibration baseline rather than a final selection rule.

## Highest-priority LatentWire ablations

1. **SAE / crosscoder frontier selector.** Select protected nodes by sparse feature mass and persistence across layers, then compare to scalar magnitude and patch-based selection.
2. **Attribution-patched frontier ranking.** Use AtP* or causal-tracing-style scores to pick the protected frontier, then validate against exact ablation on a calibration slice.
3. **Sparse routed frontier.** Treat protected-node choice as MoE routing with load balancing and uncertainty-aware fallback, and measure whether balance helps beyond accuracy.
4. **Prompt-shift robustness check.** Perturb prompts in harmless ways and verify that the protected frontier remains stable if the attribution is trustworthy.
5. **Graph-path protection.** Keep only frontier nodes that lie on high-attribution paths in the replacement-model graph, then compare to top-k flat scoring.

## Telemetry contract

If a frontier-selection ablation is meant to be paper-worthy, log at least:

- `frontier_id`
- `protected_node_ids`
- `selection_score`
- `selection_method`
- `route_entropy`
- `attribution_rank_corr`
- `patch_recovery`
- `repair_help_rate`
- `repair_harm_rate`
- `false_protect_rate`
- `false_drop_rate`
- `bytes_saved`
- `latency_ms`
- `accuracy_per_byte`
- `accuracy_per_token`
