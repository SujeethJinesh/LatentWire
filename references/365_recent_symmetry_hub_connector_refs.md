# 365 Recent Symmetry, Hub, Connector, And Stop-Policy References

Date: 2026-04-21

Scope: 2025-2026 papers that add new angles beyond `363` and `364` for LatentWire. The emphasis here is on symmetry-aware alignment, shared sparse codebooks, routed connector banks, tokenizer/vocabulary adaptation, quantization geometry, and verifier-driven stop/retry rules. These sources are only useful if they change an ablation, a telemetry field, or a stop rule.

## Paper Status

LatentWire is still in the evidence-gathering phase, but the story is sharpening. The strongest current hypothesis is not a universal adapter; it is a **symmetry-aware, routed, and auditable latent interface** with three properties:

1. Use a shared reference space only if it is stable under held-out model pairs.
2. Route through a small bank or sparse codebook rather than a single bridge.
3. Stop repair early when verifier or confidence signals say the marginal gain is gone.

The main question this memo addresses is whether the shared space should be expressed as:

- a gauge-fixed / Procrustes-style aligned basis,
- a universal sparse dictionary,
- a routed connector bank,
- or a tokenizer-normalized byte/common-ground interface.

## New Sources Worth Pulling Into The Paper

- **[Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205)**. `Core idea:` build a shared orthogonal universe across `M >= 3` models, then apply geometry correction when strict isometry is too rigid. `Why it matters:` this is the cleanest recent symmetry-aware alignment result for the exact multi-model setting LatentWire wants. `New angle vs 363/364:` not just pairwise transport; it is a multi-way shared reference space with a post-hoc correction stage. `LatentWire use:` try a shared hub basis first, then a correction map, rather than learning a different bridge for every pair. `Telemetry:` basis condition number, singular-value spectrum, shared-space residual, pairwise residual, held-out-pair delta, and route help/harm.

- **[Transformers learn factored representations](https://arxiv.org/abs/2602.02385)**. `Core idea:` transformers prefer factored, orthogonal subspaces when the underlying latent factors are conditionally independent. `Why it matters:` it gives a principled reason to look for reusable low-dimensional bridge modes instead of one dense universal map. `LatentWire use:` probe whether route atoms cluster into a small number of orthogonal factors and whether a hub dictionary can expose them. `Telemetry:` factor count, orthogonality score, factor reuse across model pairs, atom overlap, and factor-specific patch correlation.

- **[Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment](https://arxiv.org/abs/2502.03714)**. `Core idea:` a single overcomplete sparse autoencoder can reconstruct and interpret activations from multiple models. `Why it matters:` the shared-space claim is stronger than generic CCA because it is explicitly interpretable. `LatentWire use:` test whether a universal sparse codebook can serve as the communication substrate for route atoms. `Telemetry:` code usage, dead-code rate, reconstruction error, shared-vs-private atom split, and downstream answer delta.

- **[SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability](https://arxiv.org/abs/2507.06265)**. `Core idea:` enforce identical top-k latent dimensions across streams with cross-reconstruction loss. `Why it matters:` this is a concrete recipe for aligned sparse latents without manual neuron matching. `LatentWire use:` compare dense bridge payloads to aligned sparse concept payloads under matched byte budgets. `Telemetry:` Jaccard overlap of active codes, cross-reconstruction error, concept purity, and mismatch sensitivity.

- **[LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery](https://arxiv.org/abs/2602.07311)**. `Core idea:` shared sparse codes plus OT matching can produce a single interpretable latent dictionary across modalities. `Why it matters:` it adds an OT-based alignment view to the universal-codebook story. `LatentWire use:` test whether a small OT-matched sparse hub improves cross-family transfer more than pure Procrustes or pure SAE. `Telemetry:` OT cost, sparse-code alignment, concept clustering, dictionary interpretability, and held-out-family generalization.

- **[Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes](https://arxiv.org/abs/2603.04426)**. `Core idea:` delta-weighted shared dictionaries isolate narrow behavioral changes better than vanilla crosscoders. `Why it matters:` LatentWire’s bridge failures may be narrow and asymmetric, not globally shared. `LatentWire use:` add a delta-aware branch that only learns directions that change between source and target model families. `Telemetry:` delta magnitude, shared-vs-exclusive features, dead-direction rate, and causal patch effect.

- **[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)**. `Core idea:` align source vocabulary to target vocabulary with a one-to-one token mapping and embedding relearning. `Why it matters:` vocabulary mismatch may be a first-order communication bottleneck rather than just preprocessing noise. `LatentWire use:` compare byte/span normalization against token-aligned vocab adaptation before latent transfer. `Telemetry:` tokens-per-character, vocab overlap, fragmentation ratio, embedding drift, and parse failure.

- **[DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation](https://arxiv.org/abs/2602.21669)**. `Core idea:` combine dual-space weighting with Soft-DTW alignment across tokenizer spaces. `Why it matters:` it gives a better alignment objective than plain hidden MSE when token boundaries differ. `LatentWire use:` use it as a cross-tokenizer supervision branch for receiver-side alignment and latent repair. `Telemetry:` dual-space KL, DTW cost, alignment lag, byte coverage, and answer delta under tokenizer mismatch.

- **[HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning](https://arxiv.org/abs/2601.21626)**. `Core idea:` rotate/compress weights to condition the Hessian before quantization. `Why it matters:` the math says compression is safest after the space has been rotated into a better basis. `LatentWire use:` test whether the bridge should be basis-conditioned before sparsification, truncation, or byte-level compression. `Telemetry:` Hessian top eigenvalues, quantization error, rotation norm, compute-normalized accuracy, and route harm under bitwidth changes.

- **[SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs](https://arxiv.org/abs/2506.05413)**. `Core idea:` scaling plus rotation suppresses activation outliers and improves low-bit robustness. `Why it matters:` this is a practical warning that the geometry of the interface matters as much as the payload. `LatentWire use:` use a rotated/scaled transport basis before route compression. `Telemetry:` outlier mass, activation kurtosis, transport error, byte budget, and downstream recovery.

- **[CoDE-Stop: Early Stopping for Large Reasoning Models via Confidence Dynamics](https://arxiv.org/abs/2604.04930)**. `Core idea:` confidence trajectories can decide when to stop reasoning early. `Why it matters:` this is the clearest recent stop-policy signal for avoiding overthinking. `LatentWire use:` stop target-side repair when confidence is flat or falling, even if the latent bridge can still produce edits. `Telemetry:` confidence trace, halt precision, false-halt rate, overthink flag, and repair gain per token.

- **[Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models](https://arxiv.org/abs/2507.15512)**. `Core idea:` verifier-guided scaling at the step level can balance accuracy and compute more effectively than final-only checks. `Why it matters:` LatentWire should probably verify route atoms or repair steps, not just final answers. `LatentWire use:` route-atom-level or step-level verification before deciding whether to continue repair. `Telemetry:` verifier call count, accepted/rejected atoms, step-level stop reason, compute-normalized accuracy, and latency.

- **[Load Balancing Mixture of Experts with Similarity Preserving Routers](https://arxiv.org/abs/2506.14038)** and **[ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](https://arxiv.org/abs/2511.10971)**. `Core idea:` stabilize router decisions by preserving similarity structure or aligning routing with an eigenbasis. `Why it matters:` router collapse is a known failure mode, and LatentWire already sees confidence-only routing collapse. `LatentWire use:` compare feature routing, similarity-preserving routing, and eigenbasis-style routing against confidence routing. `Telemetry:` gate entropy, Gini load, collapse rate, route stability across paraphrases, and expert help/harm.

## Concrete Ablations To Add

1. **Gauge-fixed shared hub vs pairwise maps.** Compare pairwise Procrustes, multi-way shared basis, and shared basis plus residual correction. Keep the same byte budget and the same held-out pair split.

2. **Universal sparse codebook vs dense bridge.** Compare dense hidden transfer, sparse SAE/USAE code transfer, and sparse code transfer plus private residuals.

3. **Routed connector bank vs single bridge.** Compare monolithic bridge, routed projector bank, and routed bank with similarity-preserving or eigenbasis-style routing.

4. **Tokenizer normalization stack.** Compare raw tokenizer, byte-span fallback, TokAlign-style vocabulary adaptation, and DWA-KD-style cross-tokenizer alignment.

5. **Compression-order test.** Compare compression before alignment vs alignment before compression under the same quantization bitwidth and route budget.

6. **Verifier granularity sweep.** Compare final-only verification, step-level verification, and route-atom verification, each with the same stop budget.

7. **Confidence stop vs gain-based stop.** Compare CoDE-Stop style confidence dynamics, fixed-depth repair, and gain-per-token stop rules.

8. **Orthogonality probe.** Measure whether route atoms occupy factored / orthogonal subspaces, and whether the shared basis is actually reusable across held-out pairs.

## Telemetry Fields That Matter

Every new run should emit these fields where applicable:

- `method`, `source_model`, `target_model`, `dataset`, `example_id`, `seed`, `commit`
- `basis_type`, `shared_basis_id`, `basis_condition_number`, `singular_spectrum`, `gauge_fix_residual`
- `hub_atom_ids`, `shared_atom_count`, `exclusive_atom_count`, `atom_sparsity`, `dead_atom_rate`, `atom_overlap`
- `router_type`, `route_entropy`, `route_gini`, `route_stability`, `collapse_rate`, `expert_help`, `expert_harm`
- `tokenizer_pair`, `tokens_per_char`, `fragmentation_ratio`, `byte_coverage`, `vocab_overlap`, `embedding_drift`
- `quant_bitwidth`, `rotation_norm`, `hessian_top_eig`, `quant_error`, `outlier_mass`
- `verifier_type`, `verifier_granularity`, `verifier_calls`, `halt_confidence`, `false_halt`, `overthink_flag`, `stop_reason`
- `repair_steps`, `route_help`, `route_harm`, `projection_residual`, `answer`, `gold`, `correct`, `baseline_correct`
- `latency_ms`, `tokens_in`, `tokens_out`, `bytes`, `kv_bytes`

## Stop And Retry Rules

- Do not rerun the same failure mode unless at least one of these changes: basis construction, router objective, tokenizer/vocab handling, quantization order, verifier granularity, or stop policy.
- Cap retries at `2` per hypothesis family unless a new telemetry field exposes a concrete failure mechanism.
- Stop expanding the method if compute-normalized accuracy does not improve on `2` consecutive seeds and `1` held-out pair.
- Mark a branch as dead if route entropy collapses, expert load collapses, or route harm exceeds route help on matched controls.
- Do not claim a shared hub unless it improves held-out pair transfer or matches pairwise performance with fewer adapters and lower maintenance cost.

## Practical Read

The most actionable next move is to combine a shared sparse basis with stabilized routing, then verify whether that basis survives held-out pairs and tokenizer mismatch. If it does not, the paper should pivot to a reliable pairwise/routed interface with strict verifier-gated repair rather than continuing to chase a universal bridge.
