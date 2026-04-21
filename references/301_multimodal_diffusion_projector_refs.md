# Multimodal / Diffusion Projector References for LatentWire

Scope: current primary-source inspiration for cross-model reasoning and KV communication, focused on small projector/adaptor interfaces, latent bottlenecks, query/resampler pooling, diffusion-transformer conditioning, and cheap diagnostic ablations.

## Short diagnosis

LatentWire's recent negative branches suggest that the current bridge is too local: it tries to repair transported KV with static or prompt-local objectives, but it does not have an explicit interface that can decide what target-native information should be exposed to the decoder. Multimodal LLMs faced a similar problem when connecting frozen vision encoders to frozen LLMs. The strongest pattern is not "make the projector larger"; it is to give the interface a small set of trainable latent/query slots, train it with staged alignment objectives, and gate its influence so the frozen target model is not overwritten.

For LatentWire, that points to cheap tests before larger training:

- replace one-to-one transported token repair with a small fixed-size latent/query pool;
- supervise the pool in target-native embedding/logit/readout geometry;
- use gated target-layer injection instead of unconditional module replacement;
- measure whether the bridge preserves reasoning-relevant attention routes rather than only next-token mass.

## Primary sources

| Area | Source | Link | Transferable mechanism |
|---|---|---|---|
| Query bottleneck | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | https://arxiv.org/abs/2301.12597 | Frozen encoder and frozen LLM are bridged by a lightweight Querying Transformer trained in stages. This is the closest analogue for a learned cross-model "communication layer" that extracts a target-sized latent summary before generation. |
| Instruction-aware query bottleneck | InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning | https://arxiv.org/abs/2305.06500 | Makes the query transformer instruction-aware, so selected features depend on the downstream prompt. LatentWire analogue: query slots should be conditioned on target prompt/query states, not just source positions. |
| Perceiver latent bottleneck | Perceiver IO: A General Architecture for Structured Inputs & Outputs | https://arxiv.org/abs/2107.14795 | Uses a latent array and output queries to decouple input size from output semantics. LatentWire analogue: compress source KV into a fixed latent bank, then query it from target-layer states. |
| Gated cross-attention bridge | Flamingo: a Visual Language Model for Few-Shot Learning | https://arxiv.org/abs/2204.14198 | Interleaves frozen LM layers with gated cross-attention to external features. This is a safer alternative to direct KV replacement: start the gate near zero and learn/use only when helpful. |
| Minimal projector | LLaVA: Visual Instruction Tuning | https://arxiv.org/abs/2304.08485 | A simple projection layer can connect strong frozen-ish modules when trained with instruction data. LatentWire analogue: test tiny target-space projector baselines before adding complex routers. |
| One-layer alignment | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models | https://arxiv.org/abs/2304.10592 | Shows a single projection layer can unlock behavior, but also that short-caption alignment can cause repetitive/fragmented outputs unless followed by richer instruction tuning. LatentWire analogue: exact token/logit teachers may be too narrow; add richer reasoning-style calibration targets. |
| Projector locality | Honeybee: Locality-enhanced Projector for Multimodal LLM | https://arxiv.org/abs/2312.06742 | Projector quality depends on flexible token count and local-context preservation. LatentWire analogue: preserve local source span neighborhoods during compression instead of only per-position top-k transport. |
| Coarse-to-fine resampling | TokenPacker: Efficient Visual Projector for Multimodal LLM | https://arxiv.org/abs/2407.02392 | Compresses redundant visual tokens with coarse point queries updated by fine region keys/values. LatentWire analogue: use coarse latent route tokens updated by high-confidence source span neighborhoods. |
| Token-level alignment | SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs | https://arxiv.org/abs/2408.11813 | Aligns adapter outputs to the LLM embedding space with token-level contrastive supervision, improving interpretability without inference cost. LatentWire analogue: contrast transported bridge states against target vocabulary/readout neighborhoods, not only next-token IDs. |
| Any-to-any adapters | NExT-GPT: Any-to-Any Multimodal LLM | https://arxiv.org/abs/2309.05519 | Connects encoders, an LLM, and diffusion decoders by tuning a small number of projection-layer parameters plus modality-switching instruction tuning. LatentWire analogue: keep source/target frozen, add small interface projections, and train on "when to communicate" examples. |
| Pairwise objective | Sigmoid Loss for Language Image Pre-Training | https://arxiv.org/abs/2303.15343 | Pairwise sigmoid alignment avoids dependence on global batch softmax normalization and works at smaller batch sizes. LatentWire analogue: pairwise preference/contrastive losses are more robust than exact target token likelihood on tiny calibration slices. |
| General cross-modal contrast | CLIP: Learning Transferable Visual Models From Natural Language Supervision | https://arxiv.org/abs/2103.00020 | Contrastive alignment was more efficient than exact caption prediction for representation transfer. LatentWire analogue: prefer route/readout contrastive objectives over exact answer-token reconstruction when tokenizers differ. |
| Diffusion transformer conditioning | DiT: Scalable Diffusion Models with Transformers | https://arxiv.org/abs/2212.09748 | Adaptive LayerNorm / conditioning injects external signal through modulation rather than token replacement. LatentWire analogue: test FiLM/adaLN-style K/V modulation from source latents instead of hard replacing target KV. |
| Rich latent autoencoders | Diffusion Transformers with Representation Autoencoders | https://arxiv.org/abs/2510.11690 | Argues that semantically rich high-dimensional latents improve transformer diffusion, but require a suitable lightweight head. LatentWire analogue: the latent bridge may need a wide shallow target-native head instead of low-rank correction alone. |
| Multimodal diffusion reasoning | LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models | https://hf.co/papers/2602.14147 | Uses post-training objectives for unified multimodal diffusion reasoning. LatentWire analogue: corruption/denoising objectives over target hidden states could train robustness to imperfect transported states. |
| Internal tool bridge | Transformers meet Neural Algorithmic Reasoners | https://arxiv.org/pdf/2406.09308 | Interleaves pretrained transformers with gated cross-attention to a separate reasoning module, initially closing the gate to preserve LM knowledge. LatentWire analogue: treat the source model as an internal reasoning tool, not as a KV donor that must always be trusted. |

## High-priority cheap diagnostic ablations

### 1. Query-pool transport bridge

Inspired by BLIP-2, InstructBLIP, Perceiver IO, Flamingo, and TokenPacker.

Implementation sketch:

- collect transported source K/V for the aligned prompt positions as usual;
- pool them into `m` learned or deterministic query slots, e.g. `m in {4, 8, 16}`;
- update query slots with cross-attention over source-span K/V;
- inject the slots into target layers with gated cross-attention or as a small additive K/V prefix;
- initialize the gate at zero or use a fixed small gate for the first diagnostic.

Cheap variants:

- deterministic query slots from top-attention source positions, no new training;
- learned query slots with only a ridge/least-squares readout head;
- target-query-conditioned slots where target hidden states query source K/V.

Interpretable telemetry:

- per-slot attention entropy over source positions;
- source span coverage and overlap with current dynalign positions;
- gate magnitude by layer and prompt;
- paired flips versus target-alone and versus dynalign prefdist.

Expected positive signal:

- higher GSM10 than target-alone without increasing bytes too much;
- flips concentrated on prompts where slots attend to reasoning spans, not boilerplate.

### 2. Gated adaLN / FiLM modulation instead of K/V replacement

Inspired by DiT conditioning and Flamingo-style gated fusion.

Implementation sketch:

- compute a compact source summary from transported K/V or query-pool slots;
- map it to per-layer scale/shift/gate parameters for target K and V projections, attention output, or hidden states;
- use `h_target + gate * f(norm(h_target), source_summary)` rather than replacing module outputs;
- start with fixed small gates or learned scalar gates per layer.

Cheap variants:

- scalar per-layer gate only;
- diagonal scale/shift only;
- low-rank FiLM: `A(source_summary) B(h_target)`;
- apply only to mid layers where prior layer-localization suggests bridge sensitivity.

Interpretable telemetry:

- gate-by-layer heatmap;
- norm ratio of modulation to target hidden;
- answer flips versus gate magnitude;
- whether successful prompts use fewer/later layers.

Expected positive signal:

- less degradation than module replacement;
- successful cases where source information helps but target LM keeps its native decoding geometry.

### 3. Token-level target-space contrastive projector

Inspired by SEA, CLIP, SigLIP, LLaVA, and MiniGPT-4.

Implementation sketch:

- project transported bridge states into target embedding/readout space;
- for each aligned span, choose positives from target hidden/readout neighbors and negatives from nearby-but-wrong vocabulary rows or shuffled prompt spans;
- train with pairwise sigmoid/InfoNCE-style loss rather than exact next-token likelihood;
- keep the inference path unchanged initially and evaluate whether the fitted projector improves bridge state geometry.

Cheap variants:

- no new generation path: fit projector and report alignment metrics only;
- plug projector before existing prefdist branch;
- compare pairwise sigmoid versus softmax contrastive on the same calibration slice.

Interpretable telemetry:

- top-k target embedding retrieval accuracy for aligned source spans;
- margin between gold/teacher positives and hard negatives;
- relation between retrieval margin and downstream answer flips;
- whether failures are vocabulary-alignment failures or later decoding failures.

Expected positive signal:

- retrieval margins improve before generation accuracy does;
- prompts with better margins are the same prompts that flip positively.

### 4. Coarse-to-fine span packer

Inspired by TokenPacker and Honeybee.

Implementation sketch:

- build coarse route tokens from low-resolution prompt segments;
- update each coarse token from fine-grained source positions in its local region;
- keep the packed token count fixed, e.g. 8 or 16;
- feed packed tokens as a K/V prefix or cross-attention memory.

Cheap variants:

- region pooling by prompt position buckets;
- confidence-weighted pooling using existing alignment confidence;
- attention-weighted pooling using source or target attention telemetry;
- compare local buckets versus global top-k.

Interpretable telemetry:

- which regions survive packing;
- local versus global contribution to answer flips;
- whether math operators/numbers are retained;
- packed-token cosine similarity to target readout states.

Expected positive signal:

- improved bytes/accuracy frontier relative to transporting many raw positions;
- fewer failures from missing numbers or operators.

### 5. Denoising bridge objective

Inspired by diffusion transformer training and representation autoencoder work.

Implementation sketch:

- corrupt target hidden/readout states with transported-source residual noise;
- train a tiny bridge head to denoise toward clean target hidden/readout states;
- use corruption levels matching observed transport error norms;
- evaluate hidden reconstruction and then plug into generation.

Cheap variants:

- offline diagnostic only: no generation, just target-hidden denoising error;
- denoise only selected layers;
- denoise K and V separately;
- compare Gaussian noise versus real source-transport residuals.

Interpretable telemetry:

- denoising error by layer and token type;
- whether real residuals are harder than Gaussian residuals;
- relation between denoising improvement and downstream bridge success;
- residual spectrum before and after denoising.

Expected positive signal:

- real-residual denoising improves target-space geometry on held-out prompts;
- if generation still fails, the blocker is injection/readout rather than alignment.

## Older ideas worth re-adding to the paper plan

- **Bridge as interface, not correction.** The paper should frame the positive method as a communication interface with explicit bottleneck/query/gate structure, not as residual repair of transported KV.
- **Staged alignment.** Multimodal systems usually separate projector alignment from instruction tuning. LatentWire should similarly separate target-space geometry alignment, route/gate calibration, and downstream reasoning evaluation.
- **Conditional use.** The source model should be an optional internal tool. Always-on replacement is too brittle; gated or query-conditioned injection gives a clearer path to positive deltas.
- **Tokenization as supervision, not only preprocessing.** SEA/CLIP/SigLIP-style objectives suggest using token/readout neighborhoods as the training target while keeping generation tokenizer-native.
- **Latent capacity sweep.** Run explicit bottleneck sizes. If performance is non-monotonic across 4/8/16/32 slots, that is paper-useful evidence about cross-model communication bandwidth.

## Recommended immediate run order

1. **Offline token-level contrastive projector diagnostic.** Cheapest and most interpretable. Success metric: held-out target-space retrieval margin improves over dynalign prefdist.
2. **Deterministic query-pool prefix diagnostic.** No heavy training required. Success metric: slot attention covers reasoning spans and does not collapse to boilerplate.
3. **Fixed-gate FiLM/adaLN diagnostic.** Tests whether modulation is safer than replacement. Success metric: less degradation than module replacement at equal bytes.
4. **Coarse-to-fine span packer.** Tests whether bandwidth and locality are the missing pieces. Success metric: better accuracy/bytes frontier and number/operator retention.
5. **Real-residual denoising diagnostic.** Tests whether transport errors are learnably removable. Success metric: denoising generalizes to held-out prompts and predicts answer flips.

## Decision criterion for the next positive-method branch

The next branch should graduate from diagnostic to full ablation only if it clears at least two of:

- improves target-space retrieval/readout margin on held-out prompts;
- produces positive paired flips on GSM10/GSM30;
- reduces degradation relative to direct module replacement;
- has interpretable gates/slots that focus on reasoning-bearing spans;
- improves the bytes/accuracy frontier over raw dynalign transport.

If none of these pass, the blocker is probably not "more projector capacity"; it is the injection protocol or tokenizer/readout mismatch, and the next pivot should be target-native contrastive alignment plus gated target-layer modulation.
