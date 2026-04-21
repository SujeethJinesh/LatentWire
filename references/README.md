# References

This folder contains the papers requested for `latent_bridge`, numbered to match the list you provided.

- Files `01_` through `41_` correspond to items 1 through 41 in your reading list.
- Most entries are saved as PDFs.
- `38_scaling_monosemanticity.html` is saved as HTML because the source link is a web article on Transformer Circuits rather than a direct PDF.
- `download_manifest.json` records the source URL, output path, and download status for each item.

The search-only items were resolved to concrete paper links before download, so the folder should be complete.

## Math Grounding Addendum

Files `42_` through `75_` were added after the initial live validation runs. They cover representation similarity, CCA/CKA diagnostics, structured random/Fourier/Hadamard transforms, product quantization, incoherence processing, randomized linear algebra, control-theoretic gating, source coding, predictive-coding style residual transmission, model stitching, and representation-alignment theory. These are intended to ground the next RotAlign-KV ablations:

- `42_` through `44_`: CKA/SVCCA/PWCCA layer-pairing and representation-similarity diagnostics.
- `45_` through `48_`: random orthogonal, Hadamard/Fourier, and butterfly transform families.
- `49_` through `51_`, `53_`, `54_`: rotation/scale/product/incoherence quantization methods.
- `52_`: randomized low-rank methods for reduced-rank and subspace alignment ablations.
- `55_` through `57_`: Kalman, adaptive-computation, and James-Stein style shrinkage/gating references.
- `58_` through `60_`: xKV, MoH, and causal head gating for head-aware transmission.
- `61_` through `66_`: saliency-guided cache compression, rotation repair, and low-rank KV projection papers.
- `67_` through `71_`: source coding, adaptive routing, event-triggered control, rate-distortion, and predictive-coding controls.
- `72_` through `75_`: model stitching, representation-alignment theory, Fourier-domain compression, and multiresolution attention references.

Files `76_` through `85_` extend the KV-specific compression and selection literature around the new `k_only` branch:

- `76_`: layer-wise asymmetric KV quantization.
- `77_` through `82_`: retrieval-head, head-level, key-token, heavy-hitter, and benchmark papers for selective KV retention.
- `83_`: task-aware adaptive KV budgeting.
- `84_` and `85_`: redundancy-aware reasoning compression and key/value-asymmetric quantization arguments.

Files `86_` through `89_` extend the current overnight branch around K/V asymmetry,
attention-fidelity preservation, and token-level KV selection:

- `86_`: stronger evidence that keys and values should be handled asymmetrically.
- `87_` and `88_`: attention-fidelity and lossless-periodic KV compression ideas.
- `89_`: dynamic token-level KV selection for selective key transport.

Files `90_` through `93_` extend the same branch toward query-aware sparsity,
attention-space preservation, and query-centric fusion:

- `90_`: query-aware sparsity for selective key-position retention.
- `91_`: attention-score-weighted KV merging as a reliability-weighted fusion baseline.
- `92_`: low-rank attention-space compression for attention-preservation objectives.
- `93_`: query-centric cache fusion as a direct source-conditioned fusion reference.

Files `94_` through `99_` deepen the current selective-routing branch around
query-aware eviction, task-aware head differentiation, and retrieval-head
interpretability:

- `94_`: self-attention-guided KV eviction as a query-aware token-retention baseline.
- `95_`: task-aware semantic differentiation of attention heads for task-conditioned selector ablations.
- `96_`: query-agnostic KV compression with context reconstruction as a stronger blind-selector baseline.
- `97_` and `98_`: retrieval-head mechanistic papers linking head subsets to long-context factuality and reasoning.
- `99_`: an L2-norm KV compression baseline that grounds simple norm-based sparse selectors.

Files `100_` through `103_` extend the same branch toward future-query priors,
QK-geometry preservation, variable per-head budgets, and offline head
reordering:

- `100_`: future-query expected-attention priors for stronger fixed-selector baselines.
- `101_`: QK-geometry filters for attention-logit-preserving sparse selection.
- `102_`: variable per-head compression rates instead of one flat transport budget.
- `103_`: offline-calibrated head reordering for stronger retrieval-head-only transport.

Files `104_` through `106_` extend the next likely branch around dynamic
budgeting, principal-key concentration, and mechanistic head localization:

- `104_`: dynamic KV budgets for task-adaptive selector and head-budget ablations.
- `105_`: principal-key attention as a direct sparse-key concentration reference.
- `106_`: scalable component localization for causal or attribution-based head selection.

Files `107_` through `113_` deepen the next head-routing branch around
head-wise offloading, fine-grained retrieval, asymmetric K/V importance, and
OpenReview-only head-budget papers that are currently stored as link notes:

- `107_`: head-wise offloading as a direct head-specific transport systems baseline.
- `108_`: fine-grained KV retrieval for stronger query-aware token and head routing.
- `109_`: asymmetric K/V quantization, reinforcing the current `K-only` paper direction.
- `110_` through `113_`: link notes for hierarchical sharing, head-specific retention,
  adaptive budgeting, and head-level key pruning where direct shell PDF download
  returned HTML landing pages on this machine.

Files `114_` through `119_` extend the current paper-tightening loop around
calibration-derived head priors, attention-causality cautions, and stronger
dynamic budgeting alternatives:

- `114_`: layer-wise dynamic budget allocation as a direct next-step comparator for fixed head priors.
- `115_`: classic evidence that attention-head importance is sharply nonuniform.
- `116_`: a cautionary reference against overclaiming live attention as explanation or mechanism.
- `117_`: adaptive probabilistic memory retention as a stronger query-blind retention baseline.
- `118_`: heterogeneous group-attention experts for dynamic token-wise KV optimization.
- `119_`: evolutionary KV compression as a broader search-based budgeting reference.

Files `120_` and `121_` extend the next head-budget and transfer loop around
retrieval-vs-streaming head structure and sink-token priors:

- `120_`: retrieval heads vs streaming heads as a direct head-budgeting baseline.
- `121_`: attention sinks and streaming retention as a stronger blind prior / sink-token control.

Files `122_` through `125_` extend the next method loop around more stable head
scoring, low-rank attention structure, and mechanistic retrieval-head evidence:

- `122_`: entropy-aware head scoring as a stronger calibrated-head-prior feature.
- `123_`: causal head scoring as a reviewer-facing mechanism check for head budgets.
- `124_`: low-rank attention geometry as a direct compression and prior-factorization reference.
- `125_`: retrieval heads as a concrete mechanism for why sparse key import can help reasoning.

Files `126_` and `127_` add explicit math grounding for the next prior and
budgeting ideas:

- `126_`: nonlinear covariance shrinkage as a direct template for stabilizing noisy head priors.
- `127_`: rate-distortion theory for model compression as a principled framing for sparse head budgets.

Files `128_` through `130_` extend the current fixed-prior branch around
semantic retrieval heads, layer-wise KV sharing, and explicit attention-prior
reasoning:

- `128_`: semantic retrieval heads as a direct mechanism for pruning tokens before generation.

Files `191_` through `194_` extend the last live positive-method lane around
retrieval-head transport, QK geometry, and heterogeneous cache behavior:

- `191_`: query-focused retrieval heads as the clearest recent argument for retrieval-aware transport descriptors.
- `192_`: QK-geometry filtering as the strongest direct compression/transport inspiration.
- `193_`: query-key alignment as a meaningful structural objective rather than a cosmetic score.
- `194_`: heterogeneous KV compression via dynamic retrieval as the closest recent systems-style reminder that head mismatch is dynamic, not just static.
- `129_`: layer-wise dissimilar KV sharing as a close systems analogue for pair-conditioned transfer.
- `130_`: length-aware attention priors as a direct reference for cached-prior test-time reasoning.

Files `131_` through `133_` extend the next loop around query-conditioned
retrieval heads, dynamic head behavior, and attention-logit preservation:

- `131_`: query-focused retrieval heads as a direct target for query-aware sparse routing.
- `132_`: dynamic retrieval heads as a warning that fixed head identities may be brittle across queries.
- `133_`: attention-logit interpolation as a direct reference for preserving useful attention geometry under sparse transport.

Files `134_` through `136_` extend the next symmetry-aware loop around
permutation, query-aware reranking, and cross-task head transfer:

- `134_`: learnable permutation as a direct reference for head or channel reordering before sparse routing.
- `135_`: query-focused memory-aware reranking as a stronger live query-conditioned comparator.
- `136_`: cross-lingual head contribution analysis as evidence that head importance can vary systematically across tasks and settings.

Files `137_` through `141_` deepen the next stabilization loop around
symmetry, QK geometry, OT matching, and statistical prior repair:

- `137_`: gauge symmetry as a direct explanation for brittle fixed head identities and a motivation for explicit head matching.
- `138_`: key/query distribution matching as a direct QK-geometry reference for attention-logit-preserving routing.
- `139_`: QK-geometry filters as a direct sparse-selection reference for preserving useful attention structure.
- `140_`: graph optimal transport as a principled head-matching objective across non-identical spaces.
- `141_`: linear covariance shrinkage as a lighter-weight statistical stabilizer for noisy head priors.

Files `142_` through `146_` extend the next branch around expected-attention
priors, attention-fidelity objectives, OT fusion, causal induction structure,
and Wasserstein alignment:

- `142_`: expected-attention priors as a stronger query-blind baseline for sparse routing.
- `143_`: attention-fidelity guarantees as a direct reference for QK-geometry-preserving compression and ranking.
- `144_`: transformer fusion with OT as a direct symmetry-aware matching baseline across model spaces.
- `145_`: selective induction heads as a causal-structure reference for context-dependent head importance.
- `146_`: Wasserstein Procrustes as a canonical unsupervised alignment reference for cross-space matching.

File `147_` extends that same matching branch with a cheaper, more practical
OT-style alignment variant:

- `147_`: quantized Wasserstein Procrustes as a direct reference for lighter-weight OT alignment under finite precision and tighter compute budgets.

File `148_` adds a stronger query-blind retention / eviction control:

- `148_`: self-attention-guided KV eviction as a direct query-blind sparse-retention baseline against which expected-attention or structured routing branches should be compared.

Files `149_` through `151_` extend the current loop around K/V asymmetry,
task-conditioned budgeting, and low-rank head geometry:

- `149_`: low-dimensional attention selection as a direct reference for why thin keys and full values can outperform symmetric KV treatment.
- `150_`: answer-first KV budgeting as a direct reference for task-conditioned cache allocation under chain-of-thought reasoning.
- `151_`: low-rank key-value attention as a direct reference for head-space redundancy and shared subspace geometry.

Files `152_` through `155_` extend the next symmetry-aware matching loop:

- `152_`: learnable permutation for structured transformer sparsity as the closest direct reference for permutation-aware head remapping.
- `153_`: PermLLM as a Sinkhorn-style soft-permutation reference for sparse channel / head ordering.
- `154_`: Task-KV as a direct task-aware head-importance and KV-budgeting reference.
- `155_`: KVLinC as the cleanest lightweight linear-correction reference to pair with matched sparse transport.

Files `156_` through `158_` extend the heavier symmetry / transport path:

- `156_`: maximal gauge symmetry as the clearest direct mathematical explanation for why head identity can remain non-canonical even after naive matching.
- `157_`: Transport and Merge as a direct OT / transport reference for cross-architecture alignment under component mismatch.
- `158_`: FlashSinkhorn as a practical entropic-OT reference for making soft transport cheaper enough to prototype locally.

Files `159_` through `162_` tighten the same lane around gauge fixing,
orthogonal alignment, and OT-based transport:

- `159_`: complete gauge-symmetry characterization as a stronger mathematical reference for why naive head identity can stay non-canonical.
- `160_`: Procrustes bounds as a cleaner direct citation for orthogonal alignment under representation mismatch.
- `161_`: transformers-as-OT as a principled geometric framing for soft transport rather than heuristic head matching.
- `162_`: OT alignment for contextual embeddings as a practical reference for alignment objectives under contextual, not static, representations.

File `163_` adds a newer cross-model KV reuse system angle:

- `163_`: activated-LoRA cross-model KV reuse as a recent systems reference for combining reusable transport paths with lightweight adaptation rather than treating transport as a fully fixed map.

Files `164_` and `165_` extend that same systems-and-adaptation lane:

- `164_`: LRAgent as a direct multi-LoRA KV-sharing reference for separating reusable shared-cache structure from adapter-specific deltas.
- `165_`: ForkKV as a copy-on-write KV reuse reference for splitting stable shared transport from branch-specific cache growth.

Files `166_` and `167_` extend the runtime-correction lane:

- `166_`: Decomposing LLM Self-Correction as a fresh direct reference for why runtime correction behavior can matter even when static alignment looks plausible.
- `167_`: LLM Layers Immediately Correct Each Other as a canonical-link note for layer-to-layer corrective dynamics; OpenReview blocked direct shell download from this machine, so this is stored as a markdown note rather than a PDF.

Files `168_` through `170_` extend the correction-and-transfer lane:

- `168_`: YOCO++ as a direct residual-KV correction reference, useful for motivating “transport plus residual” rather than transport-only.
- `169_`: KV-CAR as a learned KV autoencoding / reuse reference, useful for the idea that compact learned correction can matter more than another static map.
- `170_`: Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts as a direct theory-and-evidence citation for why lightweight cross-model correction should be possible at all.

Files `171_` through `175_` extend the transport-and-canonicalization lane:

- `171_`: KVCOMM Online Cross-context KV-cache Communication as a recent runtime KV-communication reference, useful for contrasting selective sharing against heterogeneous transport mismatch.
- `172_`: Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking as a direct retrieval-head routing reference for transport decisions tied to live query structure.
- `173_`: PRAC as a recent principal-subspace compression reference, useful for low-rank canonical subspace ideas before cross-model transport.
- `174_`: Complete Characterization of Gauge Symmetries in Transformer Architectures as a source-note reference for transformer gauge freedom and non-identifiable coordinates; stored as markdown because OpenReview returned HTML from this machine during download.
- `175_`: Scalpel as a recent attention-manifold transport reference, useful for dynamic transport-plus-correction ideas beyond static head matching.

Files `176_` through `178_` extend the transport-plus-correction lane:

- `176_`: Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers as a direct transport-plus-reconstruction reference for repairing imperfect cache maps with structured fusion.
- `177_`: SkipV1Former as a residual/skip-style KV correction reference, useful for cheap correction paths layered on top of transport.
- `178_`: The Residual Stream Is All You Need as a strong reconstruction-side reference, useful if the paper pivots from full KV transport toward smaller transported state plus correction.

Files `179_` and `180_` extend the transport-geometry lane:

- `179_`: Probabilistic Geometric Alignment via Bayesian Latent Transport for Domain-Adaptive Foundation Models as a fresh latent-transport reference for uncertainty-aware geometry matching.
- `180_`: Constrained Gaussian Wasserstein Optimal Transport with Commutative Covariance Matrices as a direct Bures/Wasserstein-style geometry reference for richer transport costs.

Files `183_` and `185_` extend the gauge-and-correction lane:

- `183_`: GaugeKV as a direct gauge-canonicalization note for the hypothesis that some of the remaining cross-model KV mismatch is a coordinate / symmetry problem rather than just a weak transport score.
- `185_`: KVLinC as the cleanest recent transport-plus-lightweight-correction reference for the current “transport first, then small repair” method lane.

Files `186_` through `190_` extend the retrieval-and-repair lane:

- `186_`: RESA as a direct sparse-attention residual-repair reference, useful for motivating transport-plus-residual rather than transport-only.
- `187_`: ContextKeeper as a head-specific retention reference, useful for retrieval-head-aware transport and budgeting.
- `188_`: REAL as a retrieval-vs-logic attention-behavior reference, useful for refining the head-behavior template space beyond simple mean attention.
- `189_`: FreeKV as a retrieval-focused KV reuse/compression reference, useful for the next retrieval-template transport pivot.
- `190_`: RACC as a retrieval-augmented KV compression reference, useful for framing accuracy-vs-bytes under retrieval-aware transport rules.

Files `195_` through `197_` extend the query-conditioned transport backlog:

- `195_`: A2ATS as a direct query-aware KV reduction reference, useful for transport costs that try to preserve attention behavior under the live query.
- `196_`: HybridKV as a recent static-vs-dynamic head heterogeneity reference, useful for motivating query-conditioned transport rather than one static head map.
- `197_`: TokAlign as a tokenizer-alignment reference for the longer-term blocker stack when broader heterogeneous pairs make vocabulary mismatch unavoidable.

Files `198_` and `199_` extend the same runtime-query backlog:

- `198_`: Retrieval Heads are Dynamic as direct evidence that averaged calibration-time retrieval templates are too static, supporting live query-conditioned transport.
- `199_`: Query-Focused and Memory-Aware Reranker for Long Context Processing as a recent query-relevance routing reference, useful for prompt- or query-conditioned template-bank transport ideas.

Files `200_` through `203_` extend the query-conditioned and bridge backlog:

- `200_`: Causal Head Gating as a direct query-conditioned head-control reference for lightweight runtime gate calibration on top of a frozen transport.
- `201_`: CompressKV as a semantic retrieval-head reference for turning live query relevance into transport budgets.
- `202_`: LangBridge as a multimodal bridge / projector reference for mapping foreign states into a target-native basis.
- `203_`: Beyond Next-Token Alignment as a token-interaction distillation reference for richer bridge objectives than plain state matching.

Files `204_` and `205_` extend the projector-and-anchor backlog:

- `204_`: MASSV as a lightweight projector plus self-distillation reference for bridge-style adaptation without full end-to-end retraining.
- `205_`: AttAnchor as an attention-anchor reference for injecting live query structure into a bridge or routing policy.

Files `206_` through `208_` extend the bridge-and-geometry backlog:

- `206_`: AdapterTune as a clean zero-init low-rank adapter reference for tiny bridge modules on top of a frozen transport path.
- `207_`: Complete Characterization of Gauge Symmetries in Transformer Architectures as the strongest recent symmetry note for why head identities and bases are not canonical.
- `208_`: ViSpec as a lightweight adaptor / speculative-bridge reference from multimodal decoding that supports the current query-conditioned bridge lane.

File `209_` records the prompt/control backlog:

- `209_`: Qwen3 hybrid thinking and prompt controls as the official-source note for the next fairness ablation around `enable_thinking=False` and shared prompt serialization.

Files `210_` through `213_` extend the query-conditioned bridge backlog:

- `210_`: Task-KV as an instruction/task-conditioned KV reuse reference, useful for making the bridge or transport path depend on the live query rather than one static descriptor.
- `211_`: Activated LoRA as a runtime-activated low-rank adapter reference, useful for query-conditioned bridge modules that should only fire when needed.
- `212_`: Expected Attention as a direct query-conditioned attention-estimation reference, useful for transport or bridge costs that preserve live retrieval behavior rather than mean templates.
- `213_`: MoRA as an on-the-fly low-rank adaptation reference, useful for the next dynamic bridge/projector attempt beyond static `bridge_ridge`.

Files `214_` through `217_` extend the routed-projector backlog:

- `214_`: LORAUTER as a lightweight adapter-routing reference for bridge-bank variants that should switch correction subspaces by query or task regime.
- `215_`: LRAgent as a runtime-activated low-rank adaptation reference for keeping the base translator fixed while activating only a small bridge when needed.
- `216_`: QMoP as the cleanest current mixture-of-projectors reference for selecting among a few tiny bridge experts from live query structure.
- `217_`: SEMI as a shared-projector / shared-interface reference, useful for transferring multimodal bridge ideas into a lightweight cross-model KV interface.

Files `218_` through `220_` extend the richer bridge-supervision backlog:

- `218_`: Activated LoRA as a direct runtime adapter-routing reference for query-conditioned bridge modules.
- `219_`: Cross-Tokenizer Distillation via Approximate Likelihood Matching as a higher-level supervision reference when plain latent regression stops improving.
- `220_`: Attention Editing as a direct reference for training a bridge to preserve attention behavior rather than only KV coordinates.

Files `221_` through `225_` extend the next distillation-and-benchmark pivot:

- `221_`: CAB as the cleanest attention-bridge distillation reference for replacing latent regression with attention-behavior supervision.
- `222_`: EM-KD as a token-affinity distillation reference for richer interaction targets.
- `223_`: SCBench as a KV-centric benchmark framing for bytes, latency, and lifecycle artifacts.
- `224_`: Quest as a fast query-aware KV-selection comparator.
- `225_`: DuoAttention as a retrieval-vs-streaming head baseline for head-routing diagnostics.

Files `236_` through `240_` extend the symmetry-and-routed-bridge backlog:

- `236_`: Basis Decomposition for Attention as a shared-basis canonicalization reference beyond plain grouped canonical transport.
- `237_`: MOSA as a lightweight mixture-of-adapters reference for routed bridge variants.
- `238_`: MedBridge as a query-encoder / mixture-of-experts projector reference for query-conditioned bridge banks.
- `239_`: Share Your Attention as a matrix-dictionary / shared-basis reference for stronger canonicalization before transport.
- `240_`: Dynamic Multi-Expert Projectors as a modern routed-projector reference for small query-conditioned bridge mixtures.

File `241_` extends the routed-bridge backlog:

- `241_`: Libra as a decoupled routed-bridge reference for using a small expert/bridge module instead of one monolithic projector.

File `242_` extends the shared-latent bridge backlog:

- `242_`: Latent Space Communication via K-V Cache Alignment as a direct shared-latent-interface reference for future learned bridge/projector branches.

Files `243_` through `245_` extend the stronger-teacher bridge backlog:

- `243_`: DWA-KD as a weighted prediction-level distillation reference when uniform local bridge losses stop helping.
- `244_`: X2I as an attention-distillation-across-architectures reference for supervising a small bridge by behavior rather than coordinates.
- `245_`: CTPD as a reserve higher-level teacher reference if tokenizer-aware or output-side supervision becomes necessary.

Files `246_` and `247_` extend the bridge-interpretability backlog:

- `246_`: UniCrossAdapter as a modular-adapter reference if one monolithic bridge keeps saturating.
- `247_`: AtP* as a paired-flip / component-localization reference for reviewer-facing interpretability artifacts.

Files `248_` through `250_` extend the modular-bridge backlog:

- `248_`: AsymLoRA as a shared-base plus asymmetric-residual adapter reference.
- `249_`: CREMA as a modular-fusion reference for replacing one monolithic bridge with a few specialized modules.
- `250_`: MOSA as a mixture-of-simple-adapters reference when one more complex bridge keeps saturating.

Files `357_` through `359_` are local synthesis memos for the current paper loop:

- `357_`: Frontier attribution and routing references for protected-frontier selection, including SAE/crosscoder selectors, attribution patching, causal tracing, sparse routing, uncertainty-aware fallback, and saliency robustness checks.
- `358_`: Recent lateral method references for routed projectors, diffusion-style latent refinement, KV/cache controls, quantization geometry, tokenizer adaptation, and transport initialization.
- `359_`: Competitor benchmark references for direct cross-model communication peers, same-model cache controls, prompt/context compression, KV quantization, adapters, and feature-dictionary baselines.

`75_transformers_with_multiresolution_attention_heads.md` is a canonical-link note rather than a PDF because OpenReview blocked direct shell download from this machine on 2026-04-16.

See `math_grounding_manifest.json` for source URLs and the reason each paper was added. See `research_memo_manifest.json` for local synthesis memos that collect multiple source links for the active paper loop.
