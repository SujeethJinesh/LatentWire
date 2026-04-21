# 363 Recent Routed Projector, Refinement, and Latent Reasoning References

Date: 2026-04-21

Scope: 2025-2026 methods relevant to routed projector banks, adaptive test-time compute, iterative latent refinement, latent reasoning, mixture-of-depth/adaptive compute, and cross-model or inter-agent latent communication. This memo is intentionally paper-facing: every source is mapped to concrete LatentWire ablations and interpretable telemetry.

## Core Read

The strongest additive direction is not a single larger bridge. The recent literature converges on a stacked interface:

1. **Route the interface.** Projector/adapter banks outperform one monolithic connector when the input regimes differ.
2. **Allocate compute conditionally.** Dynamic layer routing, recursive latent thoughts, and confidence-guided refinement all argue against uniform compute for every token/example.
3. **Repair in latent space.** One-shot latent transfer is brittle; bounded refinement and stop rules are necessary to avoid overthinking.
4. **Make communication auditable.** Latent MAS papers claim large token/latency savings, but for LatentWire the reviewer-facing evidence must include help/harm, route identity, selected atoms, stop reason, byte budget, and patch-effect correlation.

## Primary Sources

- **[ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c33cd281f8cd784626a57de340e43fe4-Abstract-Conference.html)**. `Core idea:` replace a single linear multimodal projector with a MoE connector whose experts are initialized from different alignment tasks. `LatentWire use:` a routed projector bank can specialize by source layer, target layer, task type, tokenizer stress, or confidence rather than forcing one bridge to solve every transfer mode. `Telemetry:` selected expert, gate entropy, expert collapse rate, per-expert help/harm, and expert-task mutual information.

- **[Dynamic Multi-Expert Projectors with Stabilized Routing for Multilingual Speech Recognition](https://arxiv.org/abs/2601.19451)**. `Core idea:` SMEAR-MoE stabilizes multi-expert projector routing for frozen encoder-to-LLM ASR and reports meaningful expert specialization across languages. `LatentWire use:` use dense-gradient or smoothing-style router training to avoid bridge-bank collapse. `Telemetry:` expert load balance, route stability across paraphrases, source-family clustering, and target-family clustering.

- **[From Specific-MLLMs to Omni-MLLMs: A Survey](https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2025.findings-acl.453.pdf)**. `Core idea:` multimodal systems use multi-branch projectors, shared projectors, Q-Former/Perceiver-style compression, and discrete codebook alignment. `LatentWire use:` adapt the same design space to model-to-model interfaces: multi-branch bridge, shared bridge, cross-attention resampler, and discrete route-atom codebook. `Telemetry:` connector type, compression ratio, boundary F1, recovered atom IDs, and whether compressed atoms preserve verifier-localized evidence.

- **[Latent Collaboration in Multi-Agent Systems / LatentMAS](https://arxiv.org/abs/2511.20639)** and **[LatentMAS code](https://github.com/Gen-Verse/LatentMAS)**. `Core idea:` same-model agents exchange latent working memory instead of text, with reported accuracy, token, and latency gains over text-MAS baselines. `LatentWire use:` direct competitor lane and same-model upper bound, but not a substitute for cross-model communication. `Telemetry:` matched examples, text-vs-latent cost, latent steps, trace hashes, route help/harm, and parse failures.

- **[Interlat: Enabling Agents to Communicate Entirely in Latent Space](https://arxiv.org/abs/2511.09149)**. `Core idea:` learned latent communication between agents can outperform CoT/text baselines and can work across heterogeneous models, with optional latent compression. `LatentWire use:` test whether learned compression helps only same-family exchange or survives tokenizer/model mismatch. `Telemetry:` compression ratio, hidden-state reconstruction, answer delta, heterogeneous-pair delta, and latent-message norm drift.

- **[Latent-Space Communication for Multi-Agent Collaboration / Vision Wormhole preprint](https://openreview.net/pdf/a44471ae083f20ebfaa700cc00f28a90230efec9.pdf)**. `Core idea:` heterogeneous MAS needs a modular hub-and-spoke latent interface instead of O(N^2) pairwise translators; the paper frames text as a bandwidth and quantization bottleneck. `LatentWire use:` evaluate hub dictionary/codec vs pairwise source-target bridge. `Telemetry:` per-family adapter count, cross-family generalization, hub residual, pairwise residual, and maintenance cost proxy.

- **[Latent-DARM](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/ICLR26_latentDARM.pdf)**. `Core idea:` a discrete diffusion language model can act as a high-level planner and an autoregressive model as executor, connected by a learned latent projector instead of text. `LatentWire use:` planner-executor split is a lateral template for source-reasoner to target-executor communication. `Telemetry:` text-interface vs latent-interface accuracy, projection loss, executor conditioning length, fluency/parse failures, and plan-following errors.

- **[Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)**. `Core idea:` recurrent latent-depth computation can increase reasoning capacity without emitting more visible tokens. `LatentWire use:` test bridge refinement as hidden recurrent compute rather than more text. `Telemetry:` recurrent steps, accuracy per step, stop reason, latent drift, and over-refinement harm.

- **[Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts](https://arxiv.org/abs/2510.07358)**. `Core idea:` iterate over selected reasoning-relevant layers and optionally adapt depth per input. `LatentWire use:` restrict refinement/repair to route-critical layers instead of all layers. `Telemetry:` selected layer window, per-layer patch correlation, recurrent-depth sweep, and layer-window stability.

- **[Learning to Ponder: Adaptive Reasoning in Latent Space](https://arxiv.org/abs/2509.24238)**. `Core idea:` a small controller adaptively decides whether to halt or apply latent steering/ponder steps. `LatentWire use:` add a target-side halt/repair controller after bridge import. `Telemetry:` controller confidence, applied ponder steps, easy-vs-hard split, false-halt rate, and overthink rate.

- **[CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute](https://arxiv.org/abs/2602.08948)**. `Core idea:` a lightweight confidence controller chooses halt, re-examine, or alternative refinement with far fewer tokens than parallel decoding. `LatentWire use:` use confidence dynamics as a stop rule for latent repair and as an interpretable guardrail against unbounded refinement. `Telemetry:` confidence trace, halt precision, re-examine count, alternative-branch count, repair gain per token, and introduced harm.

- **[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)**. `Core idea:` VAE-encoded latent thought blocks plus diffusion denoising enable holistic iterative refinement of reasoning. `LatentWire use:` denoise route atoms or bridge residuals instead of only applying linear correction. `Telemetry:` denoise steps, residual norm decay, corrected atoms, diversity, parseability, and stop condition.

- **[Dr.LLM: Dynamic Layer Routing in LLMs](https://arxiv.org/abs/2510.12773)**. `Core idea:` lightweight per-layer routers skip, execute, or repeat layers under a compute budget. `LatentWire use:` route target-side repair compute by example difficulty and source-route uncertainty. `Telemetry:` skipped/repeated layers, compute budget, layer decisions, route confidence, and accuracy/latency Pareto curves.

- **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)**. `Core idea:` token-level top-k routing allocates layer compute nonuniformly under a static compute budget. This is 2024, but it is the key adaptive-depth precursor to Dr.LLM/recursive-depth methods. `LatentWire use:` allocate target-side repair to high-fragility route atoms/tokens under fixed compute. `Telemetry:` selected tokens, selected layers, fragility scores, and compute-normalized gain.

- **[Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275)**. `Core idea:` VQ-VAE latent tokens can compress or abstract early reasoning steps while retaining later textual reasoning. `LatentWire use:` compare continuous route atoms against discrete latent codebook atoms and hybrid text/latent summaries. `Telemetry:` codebook usage, latent-token length, text-token length, reconstruction, and exact-answer impact.

- **[PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space](https://arxiv.org/abs/2509.23184)**. `Core idea:` introduce latent thought states before token prediction during pretraining. `LatentWire use:` test whether target-side latent thought slots are better receivers than standard token embeddings or KV insertion points. `Telemetry:` receiver slot type, slot count, slot drift, and downstream repair gain.

- **[RELISH: LLM REgression with a Latent Iterative State Head](https://arxiv.org/abs/2604.01206)**. `Core idea:` frozen LLM representations can feed a lightweight iterative latent state head through cross-attention. `LatentWire use:` add a tiny cross-attention state head as an interpretable verifier/repair module over imported route atoms. `Telemetry:` state iterations, cross-attended atom IDs, state convergence, and regression/verifier calibration.

## Concrete Ablations To Add

1. **Routed projector bank vs monolithic bridge.** Compare one bridge, static bank, and query/layer/confidence-routed bank under matched bytes and training examples. Log expert ID, gate entropy, load balance, help/harm, and collapse rate.

2. **Stabilized router training.** Add SMEAR-style dense-gradient or smoothed routing to the projector bank. Compare to hard top-1 routing and random expert assignment. Log route stability across seeds/paraphrases and expert specialization.

3. **Multi-branch vs shared projector.** Copy Omni-MLLM connector variants: source-specific projectors, target-specific projectors, shared projector, and cross-attention resampler. Log parameter count, bytes, atom recovery, and boundary F1.

4. **Hub dictionary vs pairwise bridge.** Compare O(N^2) pairwise maps against a hub latent dictionary/codebook with per-family adapters. Log adapter count, hub residual, cross-family transfer, and held-out model-family performance.

5. **LatentMAS matched competitor ladder.** Run single-agent, text-MAS, LatentMAS, target-alone, and LatentWire strict-route rows on the same examples. Log accuracy, output tokens, latent steps, latency, trace hashes, and parser failures.

6. **Interlat-style learned latent compression.** Compare uncompressed latent route, learned compressed route, discrete codebook route, and byte-span summary. Log compression ratio, route reconstruction, downstream answer delta, and mismatch sensitivity.

7. **Planner-executor latent interface.** Treat source model as planner and target model as executor; compare decoded text plan, latent projected plan, and latent projected plan plus target repair. Log plan-following errors and projection residuals.

8. **Adaptive latent repair controller.** After bridge import, run 0/1/2/4 repair steps with a confidence-based halt controller. Compare fixed-step refinement, confidence-gated refinement, and oracle-stop. Log halt precision, overthink rate, repair gain, and introduced harm.

9. **Recursive layer-window repair.** Inspired by ETD/recurrent-depth, repeat only the target layer window with highest patch correlation instead of all layers. Log selected layers, recurrent-depth sweep, accuracy per step, and latency.

10. **Dynamic layer route for target repair.** Add Dr.LLM/MoD-style skip/execute/repeat decisions for target-side correction. Compare fixed-depth repair against dynamic repair under equal compute. Log skipped/repeated layers and compute-normalized accuracy.

11. **Latent diffusion residual denoising.** Denoise bridge residuals or route atoms for a small step budget. Compare one-pass bridge, linear correction, diffusion-style residual repair, and noisy refinement. Log residual norm decay, atom corrections, and parseability.

12. **Discrete latent route atoms.** Quantize route atoms through a small VQ/codebook and compare continuous atoms, discrete atoms, and hybrid text+latent atoms. Log codebook perplexity, dead codes, reconstruction, byte cost, and help/harm.

## Interpretability Contract

Every ablation above should emit these fields where applicable:

- `method`, `source_model`, `target_model`, `dataset`, `example_id`, `seed`
- `projector_id`, `gate_entropy`, `expert_load`, `route_stability`, `expert_help`, `expert_harm`
- `hub_adapter_id`, `pairwise_adapter_id`, `hub_residual`, `pairwise_residual`
- `latent_steps`, `repair_steps`, `stop_reason`, `halt_confidence`, `overthink_flag`
- `selected_layers`, `skipped_layers`, `repeated_layers`, `compute_budget`, `actual_compute`
- `route_atoms`, `route_atom_scores`, `discrete_codes`, `codebook_usage`, `dead_code_rate`
- `compression_ratio`, `bytes`, `tokens_in`, `tokens_out`, `latency_ms`, `tokens_per_second`
- `projection_residual`, `denoise_residual_decay`, `patch_corr`, `quant_error_corr`
- `answer`, `gold`, `correct`, `baseline_correct`, `route_help`, `route_harm`, `parse_failure`

## Paper Guidance

- Do not expand the headline method until a stacked row beats target-alone/text/LatentMAS controls on matched examples. The credible story is a **bounded, routed, interpretable latent interface**, not unconstrained latent messaging.
- The most promising stack is: tokenizer/byte-span input normalization, routed projector bank, protected route-atom/mixed-bit allocation, confidence-gated target repair, and LatentMAS/KVPress/text-MAS controls.
- Keep all claims compute-normalized. The ablations must report accuracy per byte, per generated token, per latent step, and per wall-clock second.
- The strongest symmetry hypothesis to probe next is whether different model pairs share a small number of reusable bridge modes. If yes, routed banks and hub dictionaries are the scalable route; if no, pairwise bridges remain necessary and the paper should emphasize selector/repair rather than universal transport.
