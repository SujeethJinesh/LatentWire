# 364 Hub, Router, Tokenizer, and Verifier References

Date: 2026-04-21

Scope: 2025-2026 sources for shared hub dictionaries / universal latent spaces, dynamic routing regularization, verifier and stop-policy test-time compute, and tokenizer or vocabulary adaptation. This memo is for the active LatentWire paper loop and deliberately turns each source into ablations and telemetry fields so we do not repeat old experiments without recording what failed.

## Paper Status

LatentWire is shaping up as a positive-method paper about a **bounded, routed, interpretable latent interface** for cross-model reasoning. The current story should not be "one universal projector solves cross-model communication." The evidence points to a stacked method:

1. Normalize or expose tokenizer mismatch using byte/span and vocabulary telemetry.
2. Route through a small bank or hub dictionary rather than a single monolithic map.
3. Regularize routing so the bank does not collapse to one bridge.
4. Repair only when verifier or confidence signals justify extra compute.
5. Report accuracy per byte, per latent step, per route atom, and per wall-clock second against text, LatentMAS, and target-alone controls.

The next blocker is to prove whether reusable hub atoms exist across held-out model pairs. If they do, the scalable paper contribution is hub-and-spoke latent communication. If they do not, the paper should pivot to reliable routed pairwise adapters plus verifier-governed repair.

## Shared Hub Dictionaries And Universal Latent Spaces

- **[The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems](https://arxiv.org/abs/2602.15382)**. `Core idea:` heterogeneous agents communicate through a Universal Visual Codec with a hub-and-spoke topology, reducing pairwise alignment from O(N^2) to O(N). `LatentWire use:` add a hub dictionary baseline that maps each model family into shared route atoms before target reconstruction. `Ablation:` pairwise source-target projector vs source-to-hub plus hub-to-target adapters vs hub plus pairwise residual. `Telemetry:` hub atom IDs, hub residual, pairwise residual, held-out target-family delta, adapter count, route help/harm, bytes, latency, and parse failure rate.

- **[Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes](https://arxiv.org/abs/2603.04426)**. `Core idea:` crosscoders learn shared dictionaries of interpretable latent directions between related models and improve narrow-change isolation with delta losses and paired activations. `LatentWire use:` train a crosscoder-style dictionary over paired source/target activations and use sparse dictionary coordinates as route atoms instead of raw hidden vectors. `Ablation:` raw Procrustes, ridge bridge, shared sparse dictionary, and delta-weighted shared dictionary. `Telemetry:` dictionary sparsity, shared-vs-exclusive feature fraction, atom activation overlap, atom causal patch score, dead atom rate, reconstruction error, and downstream answer delta.

- **[Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages](https://arxiv.org/abs/2501.06346)**. `Core idea:` sparse autoencoder features expose morphosyntactic concepts shared across languages, and causal interventions verify their role. `LatentWire use:` treat shared features as an existence proof for cross-context hub atoms, but require causal validation on reasoning examples. `Ablation:` shared-feature route atoms vs token-span route atoms vs random sparse atoms. `Telemetry:` atom-language/model mutual information, causal ablation effect, transfer-to-held-out-family effect, atom selectivity, and false-shared atom rate.

- **[Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html)**. `Core idea:` crosscoders decompose two activation streams into shared and model-specific features. `LatentWire use:` use this older but central baseline to avoid overclaiming that every bridge coordinate is shared. `Ablation:` shared-only atoms, target-exclusive repair atoms, and shared-plus-exclusive residual. `Telemetry:` shared atom coverage, exclusive residual norm, shared atom patch correlation, target-exclusive rescue rate, and whether shared-only hurts on tokenizer-stress cases.

- **[Latent-DARM: Bridging Discrete Diffusion And Autoregressive Models For Reasoning](https://arxiv.org/abs/2603.09184)**. `Core idea:` a learned latent projector connects a diffusion planner and autoregressive executor. `LatentWire use:` frame source model as planner and target model as executor; compare text plan, latent plan, and hub-code plan. `Ablation:` decoded plan, direct latent projection, hub dictionary code, and hub code plus target repair. `Telemetry:` plan-following score, projection residual, hub-code entropy, executor conditioning length, correct, route help/harm, and latency.

## Dynamic Routing Regularization

- **[Latent Prototype Routing: Achieving Near-Perfect Load Balancing in Mixture-of-Experts](https://arxiv.org/abs/2506.21328)**. `Core idea:` view routing as latent clustering and reduce expert load imbalance without sacrificing performance. `LatentWire use:` use prototype routes for bridge-bank selection so each expert owns a stable region of activation/task space. `Ablation:` hard top-1 route, soft top-k route, prototype-balanced route, and oracle route. `Telemetry:` route Gini, min/max route load, route entropy, prototype distance, expert collapse rate, per-expert accuracy, and seed stability.

- **[Optimizing MoE Routers: Design, Implementation, and Evaluation in Transformer Models](https://arxiv.org/abs/2506.16419)**. `Core idea:` router architecture changes trade off speed, expressiveness, entropy, and expert utilization, including structured sparse MLP-Hadamard routing. `LatentWire use:` run bridge-bank routers beyond confidence routing, especially linear, MLP, attention, hash, and Hadamard variants. `Ablation:` confidence router vs feature router vs learned linear router vs MLP-Hadamard router under equal training examples. `Telemetry:` route latency, route entropy, expert utilization, route-feature sparsity, route mismatch, and compute-normalized accuracy.

- **[Multilingual Routing in Mixture-of-Experts](https://arxiv.org/abs/2510.04694)**. `Core idea:` MoE routing is language-specific in early/late layers but more aligned in middle layers, and routing similarity can correlate with performance. `LatentWire use:` test whether bridge route agreement is layer-window dependent and whether middle-layer hub atoms generalize best. `Ablation:` early-only, middle-only, late-only, and all-layer route atoms; pairwise route maps vs shared middle-layer hub. `Telemetry:` route agreement by layer, source-target route alignment, held-out-pair delta, language/tokenizer stress split, and layer-window stability.

- **[MixLLM: Dynamic Routing in Mixed Large Language Models](https://aclanthology.org/2025.naacl-long.545/)**. `Core idea:` route user queries among multiple LLMs based on quality, cost, and latency. `LatentWire use:` include a query-level "do not transfer" or "text fallback" policy so latent transfer only fires when it has expected utility. `Ablation:` always-transfer, confidence-gated transfer, query-router transfer, and oracle transfer. `Telemetry:` transfer decision, fallback reason, expected utility, actual route help/harm, cost, latency, and target-alone correctness.

- **[SMEAR-MoE / Dynamic Multi-Expert Projectors](https://arxiv.org/abs/2601.19451)**. `Core idea:` stabilize multi-expert projector routing for frozen encoder-to-LLM interfaces with useful expert specialization. `LatentWire use:` this is the closest modern analogue to a routed bridge bank. `Ablation:` monolithic bridge, hard routed projector bank, stabilized/dense-gradient route bank, and random bank. `Telemetry:` expert load, gate entropy, specialization by dataset/task/model-family, route stability across paraphrases, and expert help/harm.

## Verifier And Stop-Policy Test-Time Compute

- **[CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute](https://arxiv.org/abs/2602.08948)**. `Core idea:` a small controller consumes confidence traces and chooses halt, re-examine, or alternative refinement, saving tokens versus massive parallel sampling. `LatentWire use:` stop target-side latent repair when confidence dynamics suggest no further gain. `Ablation:` fixed 0/1/2/4 repair steps, verifier-harm stop, confidence-trace stop, and oracle stop. `Telemetry:` confidence trace, stop reason, false halt, overthink flag, repair gain, introduced harm, tokens saved, latency saved, and calibration ECE.

- **[Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](https://arxiv.org/abs/2505.11730)**. `Core idea:` verifier call frequency is a tunable granularity parameter, not a fixed final-only or step-only decision. `LatentWire use:` verify route atoms at chunk, step, or final answer granularity to find the lowest-cost useful verifier. `Ablation:` final-only verifier, every-repair-step verifier, route-atom verifier, and adaptive granularity verifier. `Telemetry:` verifier calls, verifier token/latency cost, granularity, accepted/rejected atoms, corrected atoms, accuracy per verifier call, and compute-normalized gain.

- **[PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier](https://arxiv.org/abs/2506.10406)**. `Core idea:` one model alternates policy and verifier roles and revises selectively only when its verification detects error. `LatentWire use:` use the target model as a cheap generative verifier for imported latent atoms before running repair. `Ablation:` external verifier, target self-verifier, source self-verifier, and no verifier. `Telemetry:` verifier verdict, revision decision, verifier-answer agreement, self-verifier false positive, self-verifier false negative, and route harm avoided.

- **[S*: Test Time Scaling for Code Generation](https://arxiv.org/abs/2502.14382)**. `Core idea:` hybrid test-time scaling improves code generation by combining generation, selection, and verification. `LatentWire use:` for executable or symbolic subsets, external deterministic checks can isolate whether latent transfer helps reasoning or merely changes surface text. `Ablation:` latent transfer alone, latent transfer plus self-verifier, latent transfer plus deterministic checker, and checker-guided repair. `Telemetry:` candidate count, verifier type, pass/fail trace, selection accuracy, repair attempts, and accuracy per generated token.

## Tokenizer And Vocabulary Adaptation

- **[Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)**. `Core idea:` byte-level probabilities form a common interface for distillation across mismatched tokenizers, though improvements remain uneven. `LatentWire use:` add a byte-level supervision branch for cross-tokenizer pairs and report when byte common-ground fixes vocabulary mismatch. `Ablation:` token-ID overlap route, byte-span route, byte-distilled route head, and hybrid token+byte route. `Telemetry:` byte coverage, token overlap, byte reconstruction loss, token-fragmentation ratio, byte route help/harm, and benchmark-specific failures.

- **[Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)**. `Core idea:` approximate likelihood matching enables distillation across fundamentally different tokenizers and can transfer models to byte-level tokenization. `LatentWire use:` use ALM-style likelihood matching as a supervision objective for target-native receiver distributions instead of only hidden-state MSE. `Ablation:` hidden MSE bridge, logit KL bridge, ALM/byte likelihood bridge, and bridge plus verifier repair. `Telemetry:` target NLL, byte likelihood, answer accuracy, tokenizer divergence, bridge residual, and route help/harm by tokenizer mismatch.

- **[Tokenizer-Aware Cross-Lingual Adaptation of Decoder-Only LLMs through Embedding Relearning and Swapping](https://aclanthology.org/2026.eacl-long.357/)**. `Core idea:` customized tokenizers plus embedding relearning on fixed model weights can improve low-resource cross-lingual adaptation and reduce forgetting. `LatentWire use:` test whether lightweight receiver embedding relearning helps a target model consume source-derived route atoms or byte spans. `Ablation:` frozen target embeddings, learned adapter only, embedding relearning only, and adapter plus embedding relearning. `Telemetry:` embedding drift, original-task retention, tokenizer fragmentation, source-span coverage, target parse failure, and held-out task accuracy.

- **[AdaptBPE: From General Purpose to Specialized Tokenizers](https://arxiv.org/abs/2601.21665)**. `Core idea:` post-training vocabulary adaptation replaces low-utility tokens with corpus-relevant ones while keeping vocabulary size fixed. `LatentWire use:` build a small bridge-domain vocabulary and test whether route atoms align better when rare reasoning spans are less fragmented. `Ablation:` original tokenizer, byte-span normalization, adapted BPE vocabulary, and adapted BPE plus byte fallback. `Telemetry:` tokens per character, rare-span fragmentation, replaced token list, vocab coverage, answer accuracy, bytes, and KV-cache size.

- **[Length-MAX Tokenizer for Language Models](https://arxiv.org/abs/2511.20849)**. `Core idea:` optimize vocabulary for fewer tokens per character, reducing inference latency and KV memory while improving downstream metrics in reported settings. `LatentWire use:` treat tokenization length as a communication budget variable, not a preprocessing nuisance. `Ablation:` original tokenizer vs length-optimized segmentation for source summaries/route atoms under matched character input. `Telemetry:` tokens per character, KV bytes, route atom count, segmentation boundary F1, latency, and exact-answer impact.

- **[Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)**. `Core idea:` dynamic byte patches allocate compute by byte-level entropy and can avoid fixed vocabulary brittleness. This is late 2024, but it is the main bridge to 2025-2026 tokenizer work. `LatentWire use:` implement an entropy/fragmentation-triggered byte fallback for hard examples rather than fully replacing tokenizers. `Ablation:` no fallback, always-byte route, entropy-triggered byte route, and tokenizer-fragmentation-triggered byte route. `Telemetry:` byte entropy, patch length, fallback trigger, fallback precision, bytes transferred, and help/harm.

## Concrete LatentWire Ablations

1. **Hub dictionary vs pairwise bridge.** Train pairwise maps and a shared hub dictionary with per-family adapters. Hold out one model pair. Report held-out transfer, hub residual, pairwise residual, route help/harm, and adapter count.

2. **Shared sparse route atoms.** Replace dense bridge payloads with sparse dictionary coordinates from crosscoder/SAE-style atoms. Compare shared-only, shared-plus-exclusive, and raw hidden transfer. Report atom sparsity, causal patch effect, atom overlap, and dead atom rate.

3. **Prototype-balanced bridge routing.** Add LPR-style balanced routing to the routed projector bank. Compare hard route, soft route, prototype-balanced route, and oracle route. Report route Gini, entropy, expert load, expert specialization, and accuracy per expert.

4. **Router architecture sweep.** Run linear, MLP, attention, hash, and Hadamard-style route heads for bridge-bank selection. Report route latency, route stability, utilization, and compute-normalized accuracy.

5. **Layer-window hub atoms.** Test early/middle/late/all-layer atoms, motivated by multilingual MoE route alignment. Report route agreement by layer, selected layer windows, patch correlation, and held-out-pair generalization.

6. **Transfer-or-fallback policy.** Add a query router that chooses latent transfer, text fallback, or target-alone. Report decision accuracy, false transfer, false fallback, route help/harm, cost, and latency.

7. **Verifier granularity sweep.** Verify at final answer, repair step, route atom, or adaptive granularity. Report verifier call count, accepted/rejected atoms, verifier latency, and accuracy per verifier call.

8. **Stop-policy repair stack.** Combine routed projector bank plus confidence/verifier stop rule. Compare fixed-depth repair, verifier-harm stop, confidence-trace stop, and oracle stop. Report overthink rate, false halt, introduced harm, and tokens saved.

9. **Byte common-ground route.** Add byte-level route supervision for cross-tokenizer pairs. Compare token-overlap, byte-span, byte-distilled, and hybrid token+byte routes. Report token overlap, byte coverage, fragmentation, byte loss, and answer delta.

10. **Vocabulary adaptation stress test.** Adapt a small BPE vocabulary or receiver embedding layer for the bridge-domain corpus. Compare original tokenizer, adapted tokenizer, byte fallback, and adapted tokenizer plus byte fallback. Report tokens per character, rare-span fragmentation, original-task retention, target parse failure, and accuracy.

11. **Entropy-triggered byte fallback.** Route only high-entropy or high-fragmentation spans through byte fallback. Compare always-byte, never-byte, entropy-triggered, and fragmentation-triggered policies. Report fallback precision, bytes transferred, latency, and help/harm.

12. **External-check subset.** On code/math examples with deterministic checks, compare self-verifier vs external verifier for stopping repair. Report checker pass rate, self-verifier disagreement, selection accuracy, and repair attempts.

## Required Telemetry

Every new row should emit the following fields where applicable:

- `method`, `source_model`, `target_model`, `dataset`, `example_id`, `seed`, `commit`
- `bridge_type`, `hub_dictionary_id`, `hub_atom_ids`, `hub_residual`, `pairwise_residual`
- `shared_atom_count`, `exclusive_atom_count`, `atom_sparsity`, `dead_atom_rate`, `atom_patch_corr`
- `router_type`, `projector_id`, `route_entropy`, `route_gini`, `expert_load`, `route_stability`, `route_latency_ms`
- `selected_layers`, `layer_window`, `layer_route_agreement`, `patch_corr_by_layer`
- `transfer_decision`, `fallback_reason`, `expected_utility`, `route_help`, `route_harm`
- `verifier_type`, `verifier_granularity`, `verifier_calls`, `verifier_latency_ms`, `accepted_atoms`, `rejected_atoms`
- `repair_steps`, `stop_reason`, `halt_confidence`, `false_halt`, `overthink_flag`, `introduced_harm`
- `tokenizer_pair`, `token_overlap`, `byte_coverage`, `tokens_per_char`, `fragmentation_ratio`, `byte_entropy`
- `vocab_variant`, `embedding_drift`, `original_task_retention`, `parse_failure`
- `bytes`, `tokens_in`, `tokens_out`, `kv_bytes`, `latency_ms`, `correct`, `baseline_correct`

## Decision Rules For The Next Loop

- Promote hub dictionaries only if held-out model-pair performance beats pairwise bridge or matches it with fewer adapters.
- Promote routed banks only if route balance improves without lowering compute-normalized accuracy.
- Promote verifier repair only if it lowers over-refinement and introduced harm versus fixed repair.
- Promote tokenizer adaptation only if improvements remain after controlling for token count, byte count, and latency.
- Mark failures explicitly in the experiment ledger: no reruns without changing the hypothesis, data split, route objective, or telemetry.
