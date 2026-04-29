# Competitor Threats And Learned-Syndrome References

- date: `2026-04-29`
- role: primary-source novelty and baseline memo for the learned syndrome packet branch
- blocker: reviewers may see the current and learned packet methods as already covered by cache/activation communication, latent-agent communication, prompt compression, or classic source coding.

## Highest-Risk Competitor Families

### C2C / Cache-To-Cache

- Source: [C2C: Cache-to-Cache](https://arxiv.org/abs/2510.03215), [OpenReview](https://openreview.net/forum?id=LeatkxrBCi)
- Blocker helped: direct threat to any broad "beyond text" communication claim.
- Mechanism/design idea: project and fuse source KV cache into a target KV cache with gating.
- Next experiment change: compare as a high-rate/cache-access baseline, not as a same-byte source-private packet baseline.
- Role: baseline and novelty threat.

### KVComm

- Source: [KVComm](https://arxiv.org/abs/2510.03346)
- Blocker helped: cache-state communication can look more general than diagnostic packets.
- Mechanism/design idea: selectively share informative KV layers/pairs.
- Next experiment change: report model-internal access, bytes, latency, and whether source-destroying controls are run.
- Role: baseline.

### Online Cross-Context KV-Cache Communication

- Source: [KVCOMM: Online Cross-context KV-cache Communication](https://openreview.net/forum?id=yGOytgjurF)
- Blocker helped: multi-agent cache reuse and offset correction can be confused with source-private communication.
- Mechanism/design idea: reuse overlapping context caches with anchor-based offset alignment.
- Next experiment change: distinguish shared-context speedup from accuracy gains from private source evidence.
- Role: systems baseline.

### Communicating Activations Between Language Model Agents

- Source: [Communicating Activations Between LM Agents](https://arxiv.org/abs/2501.14082), [OpenReview](https://openreview.net/forum?id=W6RPXUUFic)
- Blocker helped: activation-level agent communication threatens broad latent-transfer novelty.
- Mechanism/design idea: inject hidden activations from one agent into another.
- Next experiment change: keep activation communication as a high-dimensional, model-access baseline with zero/shuffle/same-norm controls if implemented.
- Role: baseline and novelty threat.

### CIPHER

- Source: [CIPHER: Let Models Speak Ciphers](https://openreview.net/forum?id=sehRvaIPQQ)
- Blocker helped: embedding/cipher-style communication could make symbolic packet communication seem less novel.
- Mechanism/design idea: agents communicate richer beliefs through non-natural-language representations.
- Next experiment change: emphasize source-private evidence, explicit rate cap, and source-destroying controls.
- Role: related communication baseline.

### LatentMAS

- Source: [LatentMAS](https://arxiv.org/abs/2511.20639)
- Blocker helped: same-model or multi-agent latent working memory claims overlap with "agent communication."
- Mechanism/design idea: agents collaborate through latent memory with lower token/latency cost than text.
- Next experiment change: treat as a broad latent-collaboration comparison, not as a source-private residual coding baseline.
- Role: baseline/framing.

### Interlat

- Source: [Interlat: Enabling Agents to Communicate Entirely in Latent Space](https://arxiv.org/abs/2511.09149)
- Blocker helped: learned compressed latent communication threatens future learned-connector novelty.
- Mechanism/design idea: latent-space agent communication rather than natural-language messages.
- Next experiment change: distinguish learned syndrome packets as task/rate/source-private residual codes decoded with target candidates.
- Role: novelty threat.

### Direct Semantic Communication Via Vector Translation

- Source: [Direct Semantic Communication via Vector Translation](https://arxiv.org/abs/2511.03945)
- Blocker helped: vector translation between LLM spaces can look like a more general cross-model communication solution.
- Mechanism/design idea: translate continuous vectors between model representation spaces.
- Next experiment change: avoid broad semantic-vector claims unless a latent branch clears the same controls.
- Role: inspiration and baseline if implemented.

### LLMLingua-2

- Source: [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/)
- Blocker helped: learned prompt compression can be a strong compact text baseline.
- Mechanism/design idea: task-agnostic prompt compression through token classification.
- Next experiment change: include matched-byte structured text and larger-byte compressed diagnostic/full-log rows.
- Role: text-compression baseline.

### LongLLMLingua

- Source: [LongLLMLingua](https://arxiv.org/abs/2310.06839)
- Blocker helped: question-aware long-context compression can solve high-byte trace relay efficiently.
- Mechanism/design idea: reorder and compress visible long context for downstream question answering.
- Next experiment change: report rate frontier so the packet claim is limited to the far-left private-evidence point.
- Role: systems/text baseline.

### Chain-of-Agents

- Source: [Chain-of-Agents](https://openreview.net/forum?id=LuCLf4BJsr), [arXiv](https://arxiv.org/abs/2406.02818)
- Blocker helped: natural-language multi-agent handoff is the practical baseline reviewers will know.
- Mechanism/design idea: worker agents pass summaries to a manager.
- Next experiment change: compare full trace/structured summary rows separately from low-byte packet rows.
- Role: practical baseline.

### Distributed Indirect Source Coding With Decoder Side Information

- Source: [Distributed Indirect Source Coding with Decoder Side Information](https://arxiv.org/abs/2405.13483)
- Blocker helped: the theory frame is prior work; novelty cannot be the abstract coding problem itself.
- Mechanism/design idea: source observes indirect evidence and decoder uses side information to recover a task variable.
- Next experiment change: map variables explicitly and present learned syndrome packets as an ML method/evaluation contribution.
- Role: theory support.

### DeepJSCC With Decoder Side Information

- Source: [DeepJSCC-WZ](https://arxiv.org/abs/2310.04311)
- Blocker helped: learned source coding with decoder-only side information is established outside LLMs.
- Mechanism/design idea: neural encoder sends a channel code decoded with side information unavailable to the encoder.
- Next experiment change: use as support for learned syndrome packet design while avoiding optimality claims.
- Role: theory and method inspiration.

## Baseline Table Required For The Paper

| Family | Medium | Rate | Model internals? | Source-private controls? | Our distinction |
|---|---|---:|---:|---:|---|
| C2C / KVComm | KV cache | high | yes | not central | cache-state communication baseline |
| Activation communication | activations | high | yes | not central | high-dimensional latent baseline |
| LatentMAS / Interlat | latent memory | medium/high | often yes | not central | broad latent-agent communication |
| LLMLingua / LongLLMLingua | compressed text | variable | no | no | visible-context compression baseline |
| Chain-of-Agents | natural-language summaries | high | no | no | practical multi-agent handoff |
| Source coding with side information | abstract code | explicit | no | by formulation | theory lineage |
| LatentWire learned syndrome | binary syndrome packet | 1-4 bytes | no for packet transport | yes | source-private residual code decoded with target candidates |

## Immediate Experimental Implication

The learned syndrome branch is distinct only if it preserves the existing
source-private control contract:

- matched learned syndrome beats target-only and matched-byte text by at least
  `15` points;
- zero/shuffled/random/answer-only/answer-masked/target-derived controls stay
  at target floor;
- the result is reported as a rate frontier at `1/2/4/8` bytes;
- a full text/log oracle is included to avoid claiming text is impossible.
