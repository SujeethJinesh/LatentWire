# Multimodal Query Interface References for LatentWire

## Bottom line

The useful 2025-2026 multimodal literature is converging on the same structural idea from several angles:

1. do not pass dense tokens through the whole model;
2. put a small, explicit query or bottleneck interface in the middle;
3. make that interface query-conditioned, diversity-aware, or gated;
4. and then measure whether the bottleneck actually carries task-relevant information instead of just reducing FLOPs.

For LatentWire, that means the next ablations should treat the bridge as a **query interface family**, not just as a transport map. The bridge should be tested as:

- a fixed query bank,
- a hierarchical query bank,
- an explicit bottleneck-token pool,
- a query-conditioned pruning/gating module,
- a modulation mask over aligned latents,
- and a latent rollout interface that compresses the communication path into a short state sequence.

These are modern descendants of the Perceiver / Q-Former / resampler family; the point here is not the ancestor class itself, but which descendants actually change interface geometry.

## Recent primary sources

| Paper | Date | Link | What to steal |
|---|---:|---|---|
| [HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](https://arxiv.org/abs/2503.08585) | 2025-03-11 | https://arxiv.org/abs/2503.08585 | Hierarchical query banks, dual-stream memory, task-aware query routing. |
| [Introducing Visual Perception Token into Multimodal Large Language Model](https://arxiv.org/abs/2502.17425) | 2025-02-24 | https://arxiv.org/abs/2502.17425 | Explicit control tokens that trigger selective re-perception and re-encoding. |
| [LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts](https://arxiv.org/abs/2504.04653) | 2025-04-07 | https://arxiv.org/abs/2504.04653 | Compact learnable queries, similarity-based token reduction, text-conditioned routing. |
| [DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models](https://arxiv.org/abs/2503.02175) | 2025-03-04 | https://arxiv.org/abs/2503.02175 | Max-min diversity selection rather than naive top-k. |
| [TokenCarve: Information-Preserving Visual Token Compression in Multimodal Large Language Models](https://arxiv.org/abs/2503.10501) | 2025-03-13 | https://arxiv.org/abs/2503.10501 | Information-preservation criteria and two-stage prune/merge compression. |
| [TAMP: Token-Adaptive Layerwise Pruning in Multimodal Large Language Models](https://arxiv.org/abs/2504.09897) | 2025-04-14 | https://arxiv.org/abs/2504.09897 | Layerwise sparsity schedules and attention-score-guided token activation. |
| [MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs](https://arxiv.org/abs/2506.01850) | 2025-06-02 | https://arxiv.org/abs/2506.01850 | Cross-attention-generated modulation masks over aligned tokens. |
| [Gated Recursive Fusion: A Stateful Approach to Scalable Multimodal Transformers](https://arxiv.org/abs/2507.02985) | 2025-07-01 | https://arxiv.org/abs/2507.02985 | Symmetric cross-attention plus GRU-like gating on a shared state. |
| [Script: Graph-Structured and Query-Conditioned Semantic Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2512.01949) | 2025-12-01 | https://arxiv.org/abs/2512.01949 | Query-conditioned semantic pruning with an explicit redundancy graph. |
| [Bottleneck Tokens for Unified Multimodal Retrieval](https://arxiv.org/abs/2604.11095) | 2026-04-13 | https://arxiv.org/abs/2604.11095 | Small fixed-capacity bottleneck tokens and a condensation mask that forces information through them. |
| [Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions](https://arxiv.org/abs/2602.09483) | 2026-02-10 | https://arxiv.org/abs/2602.09483 | Token-interaction supervision instead of static logit matching. |
| [PLUME: Latent Reasoning Based Universal Multimodal Embedding](https://arxiv.org/abs/2604.02073) | 2026-04-02 | https://arxiv.org/abs/2604.02073 | Short continuous latent rollouts, semantic-anchor-guided transitions, and an explicit-to-latent curriculum. |

## Interface math worth reusing

### 1) Query-token pooling

The standard resampler / Q-Former move is still the right abstraction:

```math
z_j = \sum_{i=1}^{n} \operatorname{softmax}\!\left(\frac{q_j^\top k_i}{\sqrt{d}}\right) v_i
```

where `m << n` latent queries `q_j` compress a dense source sequence. For LatentWire, this suggests a small number of **bridge queries** or **route atoms** instead of one monolithic transport map.

### 2) Explicit bottleneck tokens

The strongest modern interface families do not hide the bottleneck inside the last token. They make it explicit:

```math
X \mapsto [X, B_1, \dots, B_m]
```

and force prediction or retrieval to flow through `B`. This is useful for LatentWire because it makes the bridge capacity measurable, and it gives a direct handle on interpretability.

### 3) Condensation masks

The most interesting 2026 retrieval work adds a mask that blocks direct paths and forces supervision through the bottleneck. In LatentWire terms, the key idea is:

```math
\text{output} \leftarrow f(\text{input}, B)
```

with the direct `input -> output` shortcut reduced or severed during training. That is the cleanest way to test whether the bridge is doing real communication.

### 4) Diversity-aware pruning

The pruning papers are all variants of:

```math
\max_{S:\ |S|=k} \;\; \lambda \cdot \text{relevance}(S, q) + (1-\lambda)\cdot \text{diversity}(S)
```

with `q` a query or instruction signal. For LatentWire, this is the right formalism for query-conditioned prefix selection and slot reduction.

### 5) Gated modulation

The adapter/gating papers all implement a soft information valve:

```math
h_{\text{out}} = g \odot h_{\text{base}} + (1-g) \odot h_{\text{bridge}}
```

with `g` predicted from query-conditioned features. The important question is whether the gate is just suppressing noise or actually preserving a distinct route family.

### 6) Latent rollout

The 2026 latent-embedding work suggests replacing verbose external reasoning with a short continuous latent chain:

```math
s_{t+1} = \phi(s_t, q)
```

This is a good fit for LatentWire if the current bridge is really a short iterative communication protocol rather than a one-shot transport.

## Six concrete LatentWire ablations

### 1) Fixed query bank vs hierarchical query bank

Test a single flat bank of `m` bridge queries against a two-level hierarchy like `m = m_scene + m_entity`.

What to log:

- query entropy;
- dead-query rate;
- effective rank of the query-to-slot attention matrix;
- per-layer CKA between query slots and source hidden states;
- exact-match accuracy and paired delta vs target-alone.

Why it matters:

- If hierarchical queries help, the bottleneck is not just size; it is compositional structure.

### 2) Explicit bottleneck tokens vs implicit last-token pooling

Insert `m` explicit bottleneck tokens and force the bridge to communicate through them, then compare to the current implicit pooling / prefix-state path.

What to log:

- bottleneck usage rate;
- direct-path leakage score;
- mutual information proxy between bottleneck states and output logits;
- accuracy under direct-path masking;
- latency / bytes per example.

Why it matters:

- This tests whether the current bridge is failing because the model is using a hidden shortcut.

### 3) Query-conditioned pruning vs attention-only pruning vs diversity pruning

Compare:

- top-k by attention mass,
- top-k by query similarity,
- max-min diversity selection,
- and a mixed relevance + diversity score.

What to log:

- retained token relevance mass;
- pairwise cosine diversity of retained tokens;
- selected-span coverage;
- compression ratio;
- accuracy / McNemar against the current transport.

Why it matters:

- It isolates whether the bridge needs relevance, diversity, or both.

### 4) Gated modulation vs additive adapter vs identity

Add a simple query-conditioned gate on the bridge path and compare it to the current additive adapter and a no-op baseline.

What to log:

- mean gate value;
- gate sparsity;
- gate variance across heads and layers;
- attribution concentration on a small subset of heads;
- output confidence calibration.

Why it matters:

- If the gate helps but the additive adapter does not, the interface needs a valve, not more capacity.

### 5) Condensation mask vs unrestricted bridge

Train or evaluate with a mask that blocks direct source-to-output paths and forces the bottleneck to carry the signal.

What to log:

- direct-path leakage;
- bridge dependency ratio;
- accuracy drop under mask removal;
- teacher-student token-interaction correlation;
- prefix-length efficiency.

Why it matters:

- This is the cleanest test of whether the bridge is genuinely communicating.

### 6) Short latent rollout depth sweep

Test a latent chain of depth `0, 1, 2, 4, 8` against the current one-shot bridge.

What to log:

- latent-step count;
- entropy drop per step;
- representation drift;
- stability under head permutation / gauge perturbation;
- exact-match and calibration.

Why it matters:

- If a short latent chain beats one-shot transport, the interface should be treated as iterative reasoning, not pure compression.

## Interpretability metrics to keep standard

These should be logged for every bridge ablation:

- route entropy;
- selected-slot effective rank;
- dead-slot / dead-query rate;
- direct-path leakage;
- bottleneck usage rate;
- pairwise token diversity;
- per-head gate sparsity;
- token-interaction correlation;
- accuracy delta under mask removal;
- latency / bytes / FLOPs.

## Practical read

The 2025-2026 pattern is consistent:

- query interfaces work when they are explicit and task-conditioned;
- pruning works when it preserves diversity and relevance together;
- adapters work when they modulate, not just add;
- and bottlenecks work when the path through them is forced, not optional.

For LatentWire, the next ablation priority should be:

1. explicit bottleneck tokens,
2. query-conditioned pruning with diversity,
3. hierarchical query banks,
4. gated modulation,
5. condensation masks,
6. short latent rollout.

If those all tie the current floor, then the issue is probably not interface shape alone. The next move would be a module transplant, not another transport tweak.
