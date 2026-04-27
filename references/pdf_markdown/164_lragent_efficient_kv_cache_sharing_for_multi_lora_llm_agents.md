# references/164_lragent_efficient_kv_cache_sharing_for_multi_lora_llm_agents.pdf

<!-- page 1 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Hyesung Jeon 1 Hyeongju Ha 1 Jae-Joon Kim 1
Abstract
Role specialization in multi-LLM agent systems
is often realized via multi-LoRA, where agents
share a pretrained backbone and differ only
through lightweight adapters. Despite sharing
base model weights, each agent independently
builds and stores its own KV cache for the same
long, tool-augmented trajectories, incurring sub-
stantial memory and compute overhead. Ex-
isting KV cache sharing methods largely over-
look this multi-LoRA setting. We observe that,
across agents, cache differences are dominated
by adapter outputs, while activations from the
shared pretrained backbone remain highly simi-
lar. Based on this observation, we propose LRA-
gent, a KV cache sharing framework for multi-
LoRA agents that decomposes the cache into
a shared base component from the pretrained
weights and an adapter-dependent component
from LoRA weights. LRAgent reduces memory
overhead by sharing the base component and stor-
ing the adapter component in its inherent low-
rank form, and further reduces compute overhead,
enabled by shared-A multi-LoRA architectures,
by also sharing the low-rank cache and avoid-
ing redundant computations for contexts already
processed by other agents. To efficiently recon-
struct adapter contributions at runtime, we in-
troduce Flash-LoRA-Attention, a kernel that re-
orders attention computation to avoid materializ-
ing the low-rank cache to full dimension. LRA-
gent achieves throughput and time-to-first-token
latency close to fully shared caching, while pre-
serving accuracy near the non-shared caching
baseline across agentic question-answering bench-
marks. Code available at https://github.
com/hjeon2k/LRAgent.
1Department of Electrical and Computer Engineering, Seoul
National University, Seoul, Korea. Correspondence to: Jae-Joon
Kim<kimjaejoon@snu.ac.kr>.
Preprint. February 3, 2026.
1. Introduction
Recently, LLMs have been widely adopted in agent systems
due to their long-context understanding ability (Dubey et al.,
2024; Jiang et al., 2023; Yang et al., 2025a), reasoning abil-
ity (Wei et al., 2022; Yao et al., 2023a; Snell et al., 2024),
and external tool interaction capabilities (Yao et al., 2023b;
Shen et al., 2024; Qin et al., 2024). In particular, multi-LLM
agent systems have gained increasing attention for their
ability to assign specialized roles to multiple agents that
collaboratively decompose and solve complex tasks (Talebi-
rad & Nadiri, 2023; Wu et al., 2024; Rasal, 2024; Zhang
et al., 2025). These agents retrieve information from ex-
ternal tools, augment it with generated outputs, and pass
the accumulated context, referred to as trajectories, to other
agents for subsequent steps. To improve accuracy, a com-
mon approach is to fine-tune a pretrained model separately
for each agent role. These fine-tuned models are typically
trained on pre-generated trajectories that reflect role-specific
behavior and tool usage patterns (Shinn et al., 2023; Bo
et al., 2024; Liu et al., 2025a; Bai et al., 2025; Fu et al.,
2025a). Parameter-efficient fine-tuning (PEFT) methods,
such as low-rank adaptation (LoRA) (Hu et al., 2022), fur-
ther enhance scalability by reducing the number of trainable
parameters from the full model to a pair of low-rank matri-
ces. As a result, multi-LoRA architectures enable agents to
share the large pretrained backbone during inference while
retraining lightweight, role-specific adapters (Wang et al.,
2023; Xia et al., 2024). This design has proven effective
in practice, consistently outperforming single-model agents
and non-fine-tuned baselines in agentic tasks (Qiao et al.,
2024; Yu et al., 2024; Liu et al., 2025b; Li et al., 2025).
Due to the long trajectories in LLM agent systems, KV
cache overhead and compute overhead become more se-
vere in multi-agent systems than in single-agent settings,
because each agent maintains its own KV cache and redun-
dant prefills occur even though a large portion of the context
is shared. This redundancy increases both memory usage
and inference latency. To mitigate the memory issue, recent
work has explored KV cache sharing across agents. How-
ever, existing approaches either require architectural modifi-
cations and additional training for cache fusion (Woo et al.,
2025; Fu et al., 2025b), focus mainly on handling positional
misalignment caused by agent-specific prefixes (Yang et al.,
2025b; Pan et al., 2025; Ye et al., 2025), or rely on selective
1
arXiv:2602.01053v1  [cs.LG]  1 Feb 2026

<!-- page 2 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
recomputation of certain tokens or layers (Yao et al., 2025;
Liu et al., 2026). Furthermore, these works focus primarily
on memory reduction but still incur redundant computation
to build a hidden state for a context that has already been
processed by other agents. Importantly, KV cache sharing
schemes that explicitly exploit the multi-LoRA architecture
remain largely unexplored.
In this work, we make a key observation that, for the same
context, cache discrepancies across agents are dominated by
task-specific LoRA-induced outputs, while activations pro-
duced by the shared pretrained backbone remain highly simi-
lar. Motivated by this observation, we propose LRAGENT, a
KV cache sharing framework tailored to multi-LoRA agent
systems. We decompose the cache into two parts: a shared
component computed from the pretrained weights, which
we call the base cache, and an agent-dependent component
induced by the LoRA weights, which we call the adapter
outputs. The next key property we exploit is that the adapter
output naturally admits a low-rank representation. Specifi-
cally, we store the intermediate activations produced right
after the LoRA down-projection, which have a small rank
dimension. We refer to these activations as the LR cache.
At runtime, we reconstruct the full-dimension adapter con-
tribution from LR cache by multiplying it with the LoRA
up-projection matrix only when needed. As a result, we com-
press multiple KV caches into a single shared base cache
with lightweight LR caches. Based on this concept, we in-
troduce two cache sharing schemes.BaseSharedshares the
base cache across all agents while maintaining a separate LR
cache per agent, substantially reducing KV cache memory.
Furthermore, motivated by recent multi-LoRA variants that
share the down-projection matrix across tasks (Tian et al.,
2024; Yang et al., 2025c), we extend our idea toBaseLR-
Shared, which also shares the LR cache as well as the base
cache by aligning agents to use a common down-projection.
This extension further reduces both memory usage and the
amount of computation for previously seen contexts. To min-
imize the runtime overhead of reconstructing adapter contri-
bution using LR cache, we design Flash-LoRA-Attention,
which reorders attention computation to avoid materializing
low-rank caches to full dimension and implements this strat-
egy efficiently on top of FlashAttention (Dao et al., 2022;
2023). Overall, our approach enables KV cache sharing
tailored to the multi-LoRA architecture, achieving memory
and inference efficiency close to fully shared KV caching
while preserving accuracy near the non-shared KV baseline.
2. Background
2.1. Multi-LoRA Architecture
LoRALoRA (Hu et al., 2022) is a PEFT method that
adapts model weights by adding a pair of low-rank matrices
to the frozen pretrained base weights. Formally, LoRA
parameterizes the weight update as:
W=W 0 + ∆W,∆W=AB,(1)
where W0 ∈R din×dout is the pretrained base weight, A∈
Rdin×r is the down-projection matrix, and B∈R r×dout
is the up-projection matrix, with rank r≪min(d in, dout).
Since only A and B are optimized instead of W0, it signifi-
cantly reduces the number of trainable parameters, leading
to lower memory usage and computation compared to full
fine-tuning. In particular, applying LoRA to the query and
value projections yields the best accuracy for a given num-
ber of parameters and is therefore widely used in practice.
Unless otherwise stated, we apply LoRA to the query and
value projections in all experiments.
Multi-LoRAA typical multi-LoRA system augments a
pretrained base weight with multiple task-specific low-
rank weights (Xia et al., 2024). For each task index
i∈ {0,1, . . . , N−1} , where N is the number of tasks,
and given LoRA weights Ai ∈R din×r and Bi ∈R r×dout,
the task-specific weight used for inference is:
Wi =W 0 + ∆Wi,∆W i =A iBi.(2)
Given an input activation tensor for task i, Xi ∈R l×din,
with sequence length l, the output Yi ∈R l×dout and the
adapter output∆Y i are computed as:
Yi =X iWi =X iW0 + (XiAi)Bi,
∆Yi =X i∆Wi = (XiAi)Bi, (3)
where XiAi ∈R l×r is the intermediate activation produced
by the down-projection.
Multi-LoRA with Shared- A Recent works report that
task-specific differences in multi-LoRA systems are driven
primarily by the up-projection matrices Bi, while the down-
projection matrices Ai encode highly similar intrinsic infor-
mation across tasks and datasets (Tian et al., 2024; Yang
et al., 2025c). Building on this observation, sharing a down-
projection matrix A can improve accuracy compared to con-
ventional multi-LoRA systems, sinceA is trained to be more
generalizable across tasks. For example, HydraLoRA (Tian
et al., 2024) introduces a trainable router that produces
token-wise mixture weights over multiple up-projections for
subtasks within a dataset. In our setting, agent roles are pre-
defined and do not require sequence-dependent routing. We
therefore adopt the shared-A design and simplify the router
to a static, sequence-wise assignment. This yields the same
overall architecture as conventional multi-LoRA, except
that the down-projections share weights, and we also find
that it improves accuracy. Based on this, we design cache-
sharing strategies that support both standard multi-LoRA
and the shared-A variant, with the latter further leveraging
the efficiency benefits of our approach. Overall, our method
reduces memory usage and latency while preserving each
agent’s role-specific behavior.
2

<!-- page 3 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
2.2. Multi-LLM Agent KV Cache Sharing
Multi-LLM AgentMulti-LLM agent systems can be im-
plemented either with a single model that plays different
roles via role-specific system prompts, which can be viewed
as a form of in-context learning, or with multiple models,
often by fine-tuning the same backbone model for different
roles (Talebirad & Nadiri, 2023; Wu et al., 2024; Shinn et al.,
2023; Liu et al., 2025a; Fu et al., 2025a). These systems
commonly use three agent roles: a planning agent for reason-
ing, an action agent for tool use, and a reflection agent for re-
vising the answer. Methods that incorporate fine-tuning for
role specialization often synthesize agent-trajectory datasets
using instruction-following models (Touvron et al., 2023;
OpenAI et al., 2024; DeepSeek-AI et al., 2024), and then
fine-tune with PEFT methods such as LoRA. These ap-
proaches have been shown to outperform both single-model
agents and non-fine-tuned baselines (Qiao et al., 2024; Yu
et al., 2024; Liu et al., 2025b), leading to broad adoption
in practice. In this work, we follow AutoAct (Qiao et al.,
2024) and fine-tune LoRA adapters for the plan, action, and
reflection agents using the provided agent trajectory dataset.
KV Cache SharingMeanwhile, due to long trajectories
from multi-step reasoning and multiple retrieval of large con-
texts from external tools, memory and compute overhead
becomes much more pronounced in multi-agent settings
than in single-agent scenarios. This is because each agent
maintains its own KV cache even though much of the con-
text is shared. Moreover, the same context is processed
independently by multiple agents, leading to substantial
computation. Together, these effects introduce memory and
compute redundancy, increasing memory usage and latency.
To mitigate the memory issue, recent works have explored
KV cache sharing for context that is shared across agents.
Most approaches either introduce a new model architecture
that necessitates additional training to enable cache shar-
ing, or handle positional misalignment so that precomputed
KV caches for overlapping context chunks can be reused
within a single model. ICaRus (Woo et al., 2025) proposes
a decoder architecture that fine-tunes only the query pro-
jections for downstream tasks, enabling agents to share an
encoder-generated cache and reconstruct task-specific ad-
ditive caches at runtime. Cache2Cache (Fu et al., 2025b)
adds a trainable linear layer that project and fuse a source
model’s KV cache into a target model. KVLink (Yang et al.,
2025b), KVFlow (Pan et al., 2025), and KVComm (Ye
et al., 2025) address positional misalignment from agent-
specific prefixes by aligning cache offsets and adjusting
positional embedding at runtime, allowing reuse of over-
lapping context despite divergent prefixes within a single
model. Other works rely on selective recomputation. For
instance, CacheBlend (Yao et al., 2025) reuses KV caches
by recomputing a small subset of tokens that are critical
for accuracy. DroidSpeak (Liu et al., 2026) enables KV
Table 1.Average cosine similarity across agent pairs for the base
cache, full cache, and adapter output, on the same context.
Model Full cache Base cacheAdapter output
LLaMA-3.1-8B-Instruct0.9576 0.9726 0.0538
Ministral-8B-Instruct0.9200 0.9530 0.0225
cache reuse across fine-tuned LLMs that share the same
backbone by selectively recomputing KV cache of a pre-
defined set of critical layers while reusing the KV cache
for the remaining layers, reporting higher accuracy than
token-wise recomputation methods such as CacheBlend.
In addition, DroidSpeak introduces a hidden state cache
to skip recomputation for the initial few non-recomputed
layers and directly feed to the first recomputed layer. How-
ever, although DroidSpeak is the closest to our work, it still
processes hidden states for already seen context, similar to
other approaches, so it reduces only the key and value pro-
jections for cache updates while leaving most computation
unchanged. This highlights that fully reusing KV caches to
eliminate computation for redundant context across models
remains challenging. Moreover, KV cache sharing tailored
to multi-LoRA systems remains largely overlooked despite
their wide deployment in practice. To the best of our knowl-
edge, this is the first work that explicitly tailors KV cache
sharing to multi-LoRA agent settings and our work is com-
plementary to prior cache-sharing methods.
3. Methodology
3.1. Cache Similarity Across the Agents
We first demonstrate that, for the same context, differ-
ences in cache values mainly arise from the adapter out-
put, which is small in magnitude (see Appendix A.1) but
contains critical agent-specific information. Applying Equa-
tion 3 to the value projections, where the input Xi corre-
sponds to the hidden states obtained by running agent i
on a given context, Figure 1 and Table 1 show that the
base cache Ybase,i =X iW0 remains highly similar across
agents on the same context. In contrast, the adapter output
∆Yi =X i∆Wi acts as a largely decorrelated perturbation,
with cosine similarity close to zero. With these small, decor-
related adapter output, the expected cosine similarity of the
full cache Yi =Y base,i + ∆Yi is lower than that of the base
cache, empirically by about 3% on average, where concrete
derivation is provided in Appendix A.2. This motivates shar-
ing only the base cache, rather than the full cache, to better
preserve the small but critical agent-specific contributions
from the adapters. Otherwise, the discrepancy accumulates
over iterative agent executions and can lead to a notice-
able accuracy drop. We also note that the cosine similarity
of the key cache remains above 0.98 on average (see Ap-
pendix A.3), suggesting that value cache management is the
key factor for preserving accuracy in multi-agent inference.
3

<!-- page 4 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Figure 1.(Left)Relationship between the full cache, base cache, and adapter output.(Right)Layer-wise pairwise cosine similarity of the
base and full caches, measured on the same context across three agent pairs using 128 samples of 2k tokens from the HotpotQA dataset.
Figure 2.Agent iteration and cache accumulation for Non-Shared,
BaseShared, and BaseLRShared. T0 denotes the system
prompt shared across agents, and Ti denotes trajectory context
blocks, formed by concatenating model-generated tokens and
retrieved context from external sources. BaseShared shares
only the base cache and maintains a separate LR cache per agent,
whereasBaseLRSharedshares both the base and LR caches.
3.2. BaseShared and BaseLRShared
In this section, we present KV cache sharing schemes
tailored to the multi-LoRA architecture. Our key idea
is to decouple the value cache into a shared component
(base cache) produced by the pretrained weights, and an
adapter-dependent component. We reuse the base cache
across agents without recomputation, and store the adapter-
dependent component in a compact low-rank form (LR
cache) that is expanded to full-dimension form on de-
mand at runtime. We first introduce BaseShared, which
primarily reduces memory usage, and then extend it to
BaseLRShared, which leverages shared-A multi-LoRA
architecture and further reduces both computation and mem-
ory usage. Figure 2 illustrates the agent execution flow
and the corresponding cache management in our methods,
which we discuss in detail in the following paragraphs.
BaseSharedWe first decouple the value cache into a base
component Ybase and an adapter-dependent component∆Yi.
Here, we share a single base cache across agents even
though the layer inputs Xi are not exactly identical, mo-
tivated by the observations in Section 3.1, which show that
the base cache XiW0 remains highly similar across agents
on the same context and that the remaining differences are
dominated by the adapter contribution. Concretely, for each
newly appended context, the agent that processes the con-
text first materializes the base cache and stores it in memory.
When another agent later processes the same context, it
reuses the stored base cache without recomputation of value
projections as illustrated in Figure 2(b). We refer to this
scheme asBaseShared.
For the adapter output, naively storing it in full-dimension
form would largely negate the benefit of sharing, since keep-
ing full-dimension cache per agent in addition to the base
cache increases the total cache size to 1 + 1/N times that
of the default non-shared scheme. Instead, we store the
adapter output in its inherent low-rank form as the interme-
diate output Ylr,i =X iAi, which we call the LR cache, and
reconstruct the required contribution at runtime via the up-
projection as Ylr,iBi. In addition, since key cache similarity
is sufficiently high across agents, we fully share the key
cache. While the same idea can also be applied to the key
projection when LoRA is used, LoRA is typically applied to
the query and value projections for the best accuracy, so we
focus on the value projection in our main implementation.
Further accuracy and efficiency analysis for LoRA applied
to the key projection is presented in Appendix D.1.
Figure 3 illustrates a forward pass that agent j processes
a context segment of length Lc given a prefilled context
of length Lp by agent i. Here, the LR cache is accu-
mulated over the sequence, and later expanded into the
full-dimension adapter contribution via the up-projection
Bi, then added to the base cache. In terms of mem-
ory, BaseShared maintains a single shared base cache
along with lightweight per-agent LR caches. Because
each LR cache is smaller than the full cache by a factor
of r/dout ≪1 , the total KV cache size is reduced to
1/N+r/d out ≃1/N of the non-shared scheme. How-
ever, in terms of computation, since only the base cache is
shared, switching agents requires constructing the LR cache
4

<!-- page 5 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Figure 3.Diagram of base and LR cache computation with an initial context of length Lp prefilled by agent i, followed by an additional
context of length Lc processed by agent j, under (a) BaseShared and (b) BaseLRShared. BaseShared maintains per-agent
LR caches and computes the LR cache using hidden states for all context tokens not yet processed by the current agent, whereas
BaseLRShared shares a single LR cache and uses hidden states only for newly appended tokens. Both methods first compute the base
cache from the pretrained weights ( 1⃝, 4⃝). They then compute the LR cache via the LoRA down-projection ( 2⃝, 5⃝), and later expand it
to the full dimension via the LoRA up-projection over the full sequence ( 3⃝, 6⃝). Efficient LR cache expansion is described in Section 3.3.
Table 2.Average cosine similarity of the LR cache across agent
pairs for the same context in shared-A multi-LoRA architectures.
Model(Plan, Action) (Action, Reflect) (Reflect, Plan)
LLaMA-3.1-8B-Instruct0.9486 0.9634 0.9607
Ministral-8B-Instruct0.9473 0.9526 0.9498
for the entire accumulated context that the new agent has
not yet processed; we refer to this as the ‘LR prefill’ process,
as illustrated in Figure 2(b). For example, in Figure 3(a), the
hidden states span a sequence of length at leastLp +L c, and
the LR cache for agent j must be newly computed via the
down-projection Aj. At this step, computation of the key
and value projection that produces the shared base cache
for context segment Lp can be skipped, but the majority
of computation (e.g., proceeding MLP layers after the At-
tention layers) is still required. As a result, the amount of
computation remains similar to the non-shared setting, scal-
ing as O(N L2dmodel), where L is the total trajectory length
and dmodel is the model hidden dimension. We note that
this is consistent with prior KV cache sharing methods in-
cluding DroidSpeak, since selective recomputation requires
hidden states for the contexts that were previously processed
only by other agents. In summary, BaseShared serves
as arobust and memory-efficient solutionapplicable to
standard multi-LoRA systems. It significantly reduces KV
cache memory usage compared to the non-shared baseline
while preservinghigher accuracythan existing prior KV
cache sharing methods. As such, BaseShared is particu-
larly well-suited for conventional multi-LoRA agents when
memory efficiency is a primary concern.
BaseLRSharedBuilding on this foundation, we next in-
troduce BaseLRShared to achievecomputational ac-
celerationas well as memory savings. By leveraging the
shared-A multi-LoRA architecture, BaseLRShared fur-
ther eliminates the redundant prefill computation, enabling
substantially higher throughput and lower latency than prior
approaches. As discussed in Section 2.1, task-specific dif-
ferences in multi-LoRA systems mainly arise from the up-
projection matrices Bi, making it effective to share A for
improving both parameter efficiency and accuracy (Tian
et al., 2024; Yang et al., 2025c). In this setting, we fur-
ther observe that the LR cache AXi produced by the shared
down-projection can also be shared across agents. As shown
in Table 2, the LR cache in shared-A multi-LoRA exhibits
high cosine similarity across agents, analogous to the base
cache, which motivates sharing the LR cache as well. We
therefore maintain a single base cache and a single LR cache
for the entire system, and construct agent-specific outputs
via their task-specific up-projectionsB i.
Here, the memory usage is reduced by a factor of 1/N+
r/(N dout)≃1/N relative to the non-shared implemen-
tation. Moreover, because both the base cache and the
LR cache are available for all previously seen tokens in
BaseLRShared, an agent does not require recomputation
over context processed by previous agents to construct either
cache or the hidden states. For example, in Figure 3(b), the
base and LR caches for the Lp tokens that are already avail-
able, so the forward pass only needs to compute activations
for newly appendedLc contexts. Therefore, across N agents
over the length-L trajectory, BaseLRShared avoids the
N separate prefills required by the non-shared implemen-
tation. As a result, the overall computational complexity
becomes even comparable to full cache sharing, scaling as
O(L2dmodel), considering that the LR cache expansion cost
is small (discussed in the Section 3.3).
3.3. Flash-LoRA-Attention
Naive runtime expansion of the LR cache introduces non-
trivial computational overhead compared to fully shared
caching because it scales with both the accumulated se-
quence length L and the full output dimension dout. Unlike
5

<!-- page 6 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Algorithm 1Flash-LoRA-Attention Forward (head-wise)
Require:L=L p +L c,Q∈R Lc×dhead
Require:K∈R L×dhead, base cacheV base ∈R L×dhead
Require:LR cacheV lr ∈R L×r, LoRAB∈R r×dhead
Require:Key, Value block sizeB c,T c =⌈L/B c⌉
Require:Query block sizeB r,T r =⌈L c/Br⌉
Ensure:Attn output blockO∈R Lc×dhead
1:Q, O→T r blocksQ i,O i (i= 0, . . . T r −1).
2:K, V base →T c blocksK j, Vbase,j (j= 0, . . . T c −1).
3:V lr →T c blocksV lr,j (j= 0, . . . T c −1).
4:for0≤i < T r do
5: Initialize Oi ←0∈R Br×dhead, Olr,i ←0∈R Br×r
6:Initializem i ← −∞ ∈R Br,ℓ i ←0∈R Br
7:LoadQ i to ShrMem
8:for0≤j < T c do
9:LoadK j,V base,j,V lr,j to ShrMem
10:S←Mask(Q iK ⊤
j /√dhead)
11:m new
i ←max
 
mi,rowmax(S)

12:α←exp(m i −m new
i )
13:P i ←exp(S−m new
i )
14:ℓ i ←α⊙ℓ i + rowsum(Pi)
15:O i ←α⊙O i +P iVbase,j
16:O lr,i ←α⊙O lr,i +P iVlr,j
17:m i ←m new
i
18:end for
19:O i ←O i +O lr,iB
20:WriteO i ←O i/ℓi
21:end for
prior low-rank cache compression methods (Yuan et al.,
2023; Tomar et al., 2025; Chang et al., 2025) that treat the
expansion of the compressed caches as an inevitable over-
head, our approach explicitly reduces it via reordered com-
putation with the attention weight. Specifically, we reorder
the matrix multiplications so that the attention-weighted
multiplication is performed directly on the LR cache first,
and the up-projection is applied afterward, making the over-
head scale with the small rank dimension rather than dout.
Concretely, suppose the base cache Vbase ∈R L×dout and
a LR cache Vlr ∈R L×r for the value projection. Given
the up-projection matrix B, the adapter contribution is
VlrB∈R L×dout. A straightforward implementation is to
first reconstruct the adapter contribution and then applies
attention with attention weights P to produce the attention
outputO:
O=P(V base +V lrB).
This expands the LR cache to the full-dimension dout for
all L tokens, so the computation overhead scales with both
Landd out. Instead, we exploit associativity and compute
O=P V base + (P Vlr)B,
so the length- L multiplication is performed in the low-
rank space, and the multiplication by B is applied after-
ward. For instance, in a decoding step where the accumu-
lated context length was L, the naive implementation adds
Θ(L r dout) computation to form VlrB already before the
attention computation. With reordering, we compute the
low-rank intermediate P Vlr ∈R 1×r in Θ(L r) and apply
the up-projection in Θ(r dout), for a total of Θ(L r+r d out)
computation. As a result, since L is the dominant term that
grows over agent trajectories, our approach reduces the com-
putation of LR cache expansion by approximately a factor of
r/dout ≪1 . Algorithm 1 realizes this idea by augmenting
FlashAttention (Dao et al., 2022; 2023). Here, dhead is used
instead of dout to show the head-wise computation explicit.
A generalized analysis of the computation overhead, along
with further algorithmic details, is provided in Appendix B.
4. Experiments
4.1. Implementation Detail
Agent SetupWe evaluate the accuracy and efficiency of
our cache sharing schemes in a multi-hop agent execution
framework, following AutoAct (Qiao et al., 2024) which
is adapted from FastChat Engine. We fine-tune three role-
specific agents,plan,action, andreflect. The plan agent
performs reasoning and selects which external tool to in-
voke based on the reasoning. The action agent produces
tool-specific arguments and executes the selected API calls,
including web search (Google Developers, 2025), Wikipedia
lookup in the ReAct-style tool-use setting (Yao et al., 2023b).
The reflect agent reviews the accumulated trajectory, in-
cluding tool outputs and reasoning, and decides whether to
terminate with a final answer or to continue the interaction.
Models and DatasetsWe evaluate on LLaMA-3.1-8B-
Instruct and Ministral-8B-Instruct. We fine-tune and eval-
uate on a split of 2.5k HotpotQA (Yang et al., 2018) and
2.0k ScienceQA (Lu et al., 2022) datasets, which require
external knowledge and multi-step reasoning to answer. For
training, we use the synthetic and filtered agent trajectories
released by AutoAct, generated with LLaMA-2-70B-Chat.
For evaluation, we run multi-hop inference on the test set
with three difficulty levels of HotpotQA and ScienceQA
questions, using 20 iterations for each level. Agent prompts
and trajectory templates are provided in the Appendix C.1.
Training SettingsFor shared- A multi-LoRA, we simplify
the dynamic token-wise routers used in prior work (Tian
et al., 2024) into a static assignment that selects agent-
specific LoRA weights for each predefined role. Under
this setting, we observe an accuracy gain over naive multi-
LoRA, both with and without cache sharing methods (see
Appendix C.2). We therefore use the shared-A weights for
our main accuracy evaluations. Since shared- A is imple-
mented by duplicating the same A weights across agents,
6

<!-- page 7 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Table 3.Benchmark accuracy (%) of the default non-sharing scheme and cache sharing schemes on HotpotQA and ScienceQA at each
level. For each model, the tiny value in the Avg. column denotes the difference from the correspondingNon-Sharedbaseline.
HotpotQA ScienceQA
Model Method Easy Medium Hard Avg. 1-4 5-8 9-12 Avg.
LLaMA-3.1-8B-Instruct
Non-Shared42.80 41.95 31.90 38.880.0070.63 60.54 76.75 69.310.00
FullShared41.15 39.15 28.90 36.40-2.4868.00 55.67 72.00 65.22-4.08
DroidSpeak40.60 39.55 30.15 36.77-2.1268.79 59.25 74.42 67.49-1.82
BaseShared42.70 41.95 31.15 38.60-0.2870.38 60.75 76.58 69.24-0.07
BaseLRShared42.40 40.80 30.55 37.92-0.9770.42 60.25 76.71 69.13-0.18
Ministral-8B-Instruct
Non-Shared41.30 37.75 28.75 35.930.0071.50 63.83 70.92 68.750.00
FullShared39.60 33.95 24.80 32.78-3.1568.83 57.33 64.25 63.47-5.28
DroidSpeak40.85 35.65 26.95 34.48-1.4568.88 59.54 69.96 66.13-2.63
BaseShared40.95 37.60 29.00 35.85-0.0870.75 63.25 70.25 68.08-0.67
BaseLRShared41.10 37.70 27.15 35.32-0.6269.71 62.08 70.17 67.32-1.43
it does not change the multi-LoRA architecture, and there-
fore does not affect efficiency comparisons. Unless stated
otherwise, we use rank r= 8 on query and value projec-
tions. Additional training hyperparameters and loss curves
are reported in the Appendix C.3. Additional accuracy and
efficiency results for LoRA applied to the query, key, value,
and output projections with the same number of trainable
parameters are provided in Appendix D.1. Our method still
performs better than the LoRA variations.
BaselinesWe compare against the default Non-Shared
baseline, and several cache sharing methods. These include
a fully shared baseline, denoted as FullShared, which
shares the entire KV cache across agents without any re-
computation, and DroidSpeak. For DroidSpeak, we follow
the official configuration and recompute the top 33% most
sensitive layers at runtime, selected by probing HotpotQA
accuracy, matching the Pareto-optimal setting reported in
(Liu et al., 2026). The selected layers are listed in the
Appendix C.4. We note that prefix-aware positional em-
bedding matching methods such as KVLink, KVFlow, and
KVComm reduce to FullShared in our setup because
the prefixes are identical across agents.
Efficiency EvaluationWe observe that latency in agent
systems depends on both the cache sharing method’s effi-
ciency and the system accuracy, since lower-accuracy meth-
ods tend to take more steps and accumulate longer contexts.
To measure the intrinsic efficiency of each cache sharing
scheme, we evaluate latency under a controlled trace of
sequence lengths, similar to the evaluation protocol of KV-
Comm (Ye et al., 2025). We construct this trace by varying
the amount of context retrieved from external tools from 1k
to 64k tokens, and we feed the same trace to all schemes.
This yields total sequence lengths ranging from 2k to 66k
tokens. The detailed construction of the emulated trace and
the latency analysis on the HotpotQA benchmark are de-
scribed in Appendix C.5 and Appendix D.2, respectively,
where our schemes achieve the best results. We conducted
Figure 4.System throughput (tokens per second) ofBaseShared
andBaseLRShared, with Flash-LoRA-Attention (FLA).
all experiments on a single NVIDIA A6000 48GB GPU.
4.2. Benchmark Accuracy
We demonstrate that both of our cache sharing schemes
preserve accuracy more effectively than prior baselines,
as shown in Table 3. BaseShared stays close to
Non-Shared, with an average accuracy drop of at most
0.7%. BaseLRShared also maintains strong accuracy,
with an average drop of at most 1.5%. In contrast,
FullShared and DroidSpeak exhibit larger average
drops, up to 5.3% and 2.6%, respectively. Overall, these
results indicate that decoupling the cache into a shared base
component and a low-rank component is a key factor for
robust KV cache sharing, compared with selective recompu-
tation such as DroidSpeak. Additionally, we provide further
analysis of out-of-function incident ratios where the system
fails to produce an answer, rank ablations, and deviation of
scores in Appendix D.3, D.4, and D.5.
4.3. System Efficiency
We report system throughput and time-to-first-token (TTFT)
in Table 4 and Table 5, respectively. We define system
throughput as the total processed sequence length divided
by the end-to-end latency. TTFT is the sum of prefill la-
tencies across agent steps, since each step incurs a new
7

<!-- page 8 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Table 4.System throughput (tokens per second) under each total sequence length of the traces.OOMindicates out-of-memory.
Model Method 1.9k 3.0k 5.0k 9.1k 17.3k 33.7k 66.4k
LLaMA-3.1-8B-Instruct
Non-Shared176.2 262.4 401.3 592.1 669.2 683.2OOM
FullShared193.7 293.4 468.5 808.1 1246.1 1697.6 1826.5
DroidSpeak182.4 263.5 412.3 633.5 844.2 931.0 813.1
BaseShared169.0 257.0 392.3 617.3 862.8 969.6 823.2
BaseLRShared188.7 279.4 463.8 785.7 1239.4 1678.1 1790.6
Ministral-8B-Instruct
Non-Shared158.9 231.0 361.5 541.2 610.7 638.4OOM
FullShared169.9 251.4 420.3 711.2 1163.9 1538.6 1768.3
DroidSpeak160.9 251.4 360.3 570.2 785.1 856.0OOM
BaseShared157.0 227.4 364.0 552.1 796.5 885.0 870.5
BaseLRShared164.1 247.9 410.9 703.2 1159.2 1521.9 1757.0
Table 5.TTFT (second) under each total sequence length of the traces.OOMindicates out-of-memory. Lower is better.
Model Method 1.9k 3.0k 5.0k 9.1k 17.3k 33.7k 66.4k
LLaMA-3.1-8B-Instruct
Non-Shared1.94 2.55 3.72 6.79 16.38 38.87OOM
FullShared1.13 1.28 1.63 2.40 4.19 9.13 23.28
DroidSpeak1.62 2.17 3.22 5.55 11.15 25.43 67.80
BaseShared1.62 2.12 3.06 5.26 10.51 23.91 67.80
BaseLRShared1.13 1.28 1.64 2.43 4.24 9.19 23.35
Ministral-8B-Instruct
Non-Shared2.02 2.65 3.85 6.84 17.67 41.67OOM
FullShared1.19 1.35 1.69 2.50 4.37 9.30 20.84
DroidSpeak1.65 2.22 3.28 5.69 11.53 26.71OOM
BaseShared1.66 2.19 3.17 5.43 10.85 25.57 59.62
BaseLRShared1.20 1.35 1.71 2.53 4.42 9.38 20.98
Figure 5.Memory usage (GB) of cache sharing methods on total
sequence length of 66.4k on Ministral-8B-Instruct.
model generation in multi-hop scenarios. We first demon-
strate the impact of Flash-LoRA-Attention in Figure 4. It
yields up to a 1.24× gain in throughput for BaseShared
and up to a 1.35 × gain for BaseLRShared. This
shows that reducing LR cache expansion overhead en-
ables substantial speedups. With Flash-LoRA-Attention
enabled, BaseShared achieves up to a 1.42 × gain and
BaseLRShared achieves up to a 2.46× gain in through-
put, approaching the upper bound of full cache sharing
with FullShared. DroidSpeak reaches a similar through-
put gain to BaseShared, up to 1.36 ×, since both meth-
ods compute hidden states for tokens that have not been
processed by the current agent. For TTFT, BaseShared
and BaseLRShared provide up to 1.63× and 4.44× re-
ductions, respectively, both exceeding DroidSpeak which
achieves up to a 1.56 × reduction. BaseLRShared
achieves TTFT reductions close to those of FullShared.
Additionally, BaseShared and BaseLRShared re-
duce KV cache memory by nearly 1/3 compared to
Non-Shared baseline, as shown in Figure 5, which
is comparable to other cache-sharing baselines and
only marginally higher than FullShared within 1GB.
This is because the LR caches in BaseShared and
BaseLRShared, as well as the hidden-state cache in
DroidSpeak, are negligible in size. However, under group-
query attention (GQA), the hidden state cache used by
DroidSpeak is larger than the KV cache for a layer, which
led to near out-of-memory (OOM) behavior in some cases
of our experiments. We provide a detailed memory analysis
in Appendix D.6.
5. Conclusion
In this work, we introduce LRAGENT, a KV cache sharing
framework for multi-LoRA agent systems that decouples
the value cache into a shared base cache and an adapter-
dependent LR cache. BaseShared reduces KV memory
by sharing the base cache, and BaseLRShared further
reduces computation by sharing the LR cache under shared-
A multi-LoRA variants while preserving role-specific be-
haviors. We validate that these methods preserve accuracy
close to the non-shared baseline. Flash-LoRA-Attention pro-
vides substantial efficiency gains by avoiding full-dimension
materialization of the LR cache, enabling throughput and
TTFT improvements close to fully shared caching. Overall,
LRAGENTconsistently outperforms prior cache sharing
baselines in both accuracy and efficiency.
8

<!-- page 9 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Software and Data
We provide a file upload that reproduces the main results
in this paper, including training, evaluation, and latency
measurements under the same experimental settings. De-
tailed descriptions and step-by-step guidelines, such as en-
vironment setup and commands to run each experiment, are
included in the uploaded file.
Impact Statement
This paper presents work whose goal is to advance the field
of deep learning and large language models. There are many
potential societal consequences of our work, none of which
we feel must be specifically highlighted here.
References
Bai, T., Yang, L., Wong, Z. H., Sun, F., Zhuang, X., Peng,
J., et al. Efficient pretraining data selection for language
models via multi-actor collaboration. InProceedings of
the 63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pp. 9465–
9491, Vienna, Austria, jul 2025. Association for Com-
putational Linguistics. doi: 10.18653/v1/2025.acl-long.
466. URL https://aclanthology.org/2025.
acl-long.466/.
Bo, X., Chen, X., Dai, Q., Feng, X., Li, R., Wang, L.,
et al. Reflective multi-agent collaboration based on large
language models. InAdvances in Neural Information Pro-
cessing Systems, volume 37, pp. 138595–138631, 2024.
doi: 10.52202/079017-4397.
Chang, C.-C., Lin, C.-Y ., Akhauri, Y ., Lin, W.-C., Wu,
K.-C., Ceze, L., et al. xkv: Cross-layer svd for kv-
cache compression.arXiv preprint arXiv:2503.18893,
2025. doi: 10.48550/arXiv.2503.18893. URL https:
//arxiv.org/abs/2503.18893.
Dao, T., Fu, D. Y ., Ermon, S., Rudra, A., and Re, C. Flashat-
tention: Fast and memory-efficient exact attention with
io-awareness. InAdvances in Neural Information Pro-
cessing Systems, volume 35, pp. 16344–16359, 2022. doi:
10.5555/3600270.3601459. URL https://github.
com/Dao-AILab/flash-attention. Paper:
arXiv:2205.14135, code repository linked in url field.
Dao, T., Haziza, D., Massa, F., and Sizov, G. Flash-
decoding for long-context inference. PyTorch Blog, Oc-
tober 2023. URL https://pytorch.org/blog/
flash-decoding/. Updated Nov 16, 2024.
DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu,
B., et al. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437, 2024. doi: 10.48550/arXiv.2412.
19437. URL https://arxiv.org/abs/2412.
19437.
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
A., et al. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783, 2024. doi: 10.48550/arXiv.2407.
21783.
Fu, D., He, K., Wang, Y ., Hong, W., Gongque, Z.,
Zeng, W., et al. Agentrefine: Enhancing agent gen-
eralization through refinement tuning.arXiv preprint
arXiv:2501.01702, 2025a.
Fu, T., Min, Z., Zhang, H., Yan, J., Dai, G., Ouyang, W.,
et al. Cache-to-Cache: Direct semantic communica-
tion between large language models.arXiv preprint
arXiv:2510.03215, 2025b. doi: 10.48550/arXiv.2510.
03215. URL https://arxiv.org/abs/2510.
03215.
Google Developers. Custom Search JSON API: Introduc-
tion. Programmable Search Engine Documentation, 2025.
URL https://developers.google.com/
custom-search/v1/introduction?hl=ko.
Last updated 2025-08-31. Accessed 2026-01-06.
Hu, E. J., Shen, Y ., Wallis, P., Allen-Zhu, Z., Li, Y ., Wang,
S., et al. LoRA: Low-rank adaptation of large lan-
guage models. InInternational Conference on Learn-
ing Representations, 2022. doi: 10.48550/arXiv.2106.
09685. URL https://openreview.net/forum?
id=nZeVKeeFYf9.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., de las Casas, D., et al. Mistral 7b.arXiv
preprint arXiv:2310.06825, 2023. doi: 10.48550/arXiv.
2310.06825.
Li, B., Wang, Y ., Ma, H., Chen, L., Xiao, J., and Wang,
S. Mobilora: Accelerating lora-based llm inference on
mobile devices via context-aware kv cache optimiza-
tion. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 23400–23410, Vienna, Austria,
July 2025. Association for Computational Linguistics.
doi: 10.18653/v1/2025.acl-long.1140. URL https://
aclanthology.org/2025.acl-long.1140/.
Liu, S., Chen, T., Liang, Z., Lyu, X., and Amato, C. Llm
collaboration with multi-agent reinforcement learning.
arXiv preprint arXiv:2508.04652, 2025a.
Liu, Y ., Lin, K. Q., Chen, C. W., and Shou, M. Z. Video-
mind: A chain-of-LoRA agent for long video reason-
ing.arXiv preprint arXiv:2503.13444, 2025b. doi:
10.48550/arXiv.2503.13444. URL https://arxiv.
org/abs/2503.13444.
9

<!-- page 10 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Liu, Y ., Huang, Y ., Yao, J., Feng, S., Gu, Z., Du, K., et al.
DroidSpeak: KV cache sharing for cross-LLM commu-
nication and multi-LLM serving. InProceedings of the
23rd USENIX Symposium on Networked Systems Design
and Implementation (NSDI ’26). USENIX Association,
2026. doi: 10.48550/arXiv.2411.02820. URL https:
//arxiv.org/abs/2411.02820. Accepted for
NSDI 2026.
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu,
S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to
explain: Multimodal reasoning via thought chains for
science question answering. InAdvances in Neural Infor-
mation Processing Systems, volume 35, pp. 2507–2521,
2022.
OpenAI, Hurst, A., Lerer, A., Goucher, A. P., Perelman, A.,
Ramesh, A., et al. Gpt-4o system card.arXiv preprint
arXiv:2410.21276, 2024. doi: 10.48550/arXiv.2410.
21276. URL https://arxiv.org/abs/2410.
21276.
Pan, Z., Patel, A. D., Shen, Y ., Hu, Z., Guan, Y ., Li,
W.-L., et al. KVFlow: Efficient prefix caching for
accelerating LLM-based multi-agent workflows. In
Advances in Neural Information Processing Systems
(NeurIPS 2025), 2025. URL https://arxiv.org/
abs/2507.07400.
Qiao, S., Zhang, N., Fang, R., Luo, Y ., Zhou, W., Jiang, Y .,
et al. AutoAct: Automatic agent learning from scratch
for QA via self-planning. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 3003–3021,
Bangkok, Thailand, August 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.acl-long.
165. URL https://aclanthology.org/2024.
acl-long.165/.
Qin, Y ., Liang, S., Ye, Y ., Zhu, K., Yan, L., Lu, Y ., et al.
Toolllm: Facilitating large language models to master
16000+ real-world apis. InThe Twelfth International
Conference on Learning Representations, 2024.
Rasal, S. Llm harmony: Multi-agent communication for
problem solving.arXiv preprint arXiv:2401.01312, 2024.
Shen, W., Li, C., Chen, H., Yan, M., Quan, X., Chen, H.,
et al. Small llms are weak tool learners: A multi-llm agent.
InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, pp. 16658–
16680, 2024.
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and
Yao, S. Reflexion: Language agents with verbal rein-
forcement learning. InAdvances in Neural Information
Processing Systems, volume 36, pp. 8634–8652, 2023.
Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm test-
time compute optimally can be more effective than scal-
ing model parameters.arXiv preprint arXiv:2408.03314,
2024. doi: 10.48550/arXiv.2408.03314.
Talebirad, Y . and Nadiri, A. Multi-agent collaboration:
Harnessing the power of intelligent llm agents.arXiv
preprint arXiv:2306.03314, 2023.
Tian, C., Shi, Z., Guo, Z., Li, L., and Xu, C. Hydralora:
An asymmetric lora architecture for efficient fine-tuning.
InAdvances in Neural Information Processing Systems,
volume 37, pp. 9565–9584, 2024. doi: 10.5555/3737916.
3738220. URL https://proceedings.neurips.
cc/.
Tomar, A., Hooper, C., Lee, M., Xi, H., Tiwari, R., Kang,
W., et al. Xquant: Breaking the memory wall for llm
inference with kv cache rematerialization.arXiv preprint
arXiv:2508.10395, 2025. doi: 10.48550/arXiv.2508.
10395. URL https://arxiv.org/abs/2508.
10395.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y ., et al. Llama 2: Open foundation and fine-
tuned chat models.arXiv preprint arXiv:2307.09288,
2023. doi: 10.48550/arXiv.2307.09288. URL https:
//arxiv.org/abs/2307.09288.
Wang, Y ., Lin, Y ., Zeng, X., and Zhang, G. Multi-
lora: Democratizing lora for better multi-task learn-
ing.arXiv preprint arXiv:2311.11501, 2023. doi:
10.48550/arXiv.2311.11501. URL https://arxiv.
org/abs/2311.11501.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
E., et al. Chain-of-thought prompting elicits reasoning in
large language models. InAdvances in Neural Informa-
tion Processing Systems, volume 35, pp. 24824–24837,
2022.
Woo, S., Kim, H., Kil, J., Kim, M., Kim, J., Seo, A., et al.
Icarus: Identical cache reuse for efficient multi-model
inference. OpenReview, ICLR 2026 Conference Sub-
mission, 2025. URL https://openreview.net/
forum?id=qrMo6R7lOS. Accessed 2026-01-02.
Wu, Q., Bansal, G., Zhang, J., Wu, Y ., Li, B., Zhu, E., et al.
Autogen: Enabling next-gen llm applications via multi-
agent conversations. InFirst Conference on Language
Modeling, jul 2024. URL https://openreview.
net/forum?id=BAakY1hNKS.
Xia, Y ., Fu, F., Zhang, W., Jiang, J., and Cui, B. Efficient
multi-task LLM quantization and serving for multiple
LoRA adapters. InAdvances in Neural Information Pro-
cessing Systems, volume 37, pp. 63686–63714, 2024. doi:
10.5555/3737916.3739950.
10

<!-- page 11 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
Yang, A., Yu, B., Li, C., Liu, D., Huang, F., Huang, H.,
et al. Qwen2.5-1m technical report.arXiv preprint
arXiv:2501.15383, 2025a. doi: 10.48550/arXiv.2501.
15383.
Yang, J., Hou, B., Wei, W., Bao, Y ., and Chang, S.
KVLink: Accelerating large language models via effi-
cient KV cache reuse.arXiv preprint arXiv:2502.16002,
2025b. doi: 10.48550/arXiv.2502.16002. URL https:
//arxiv.org/abs/2502.16002.
Yang, Y ., Muhtar, D., Shen, Y ., Zhan, Y ., Liu, J.,
Wang, Y ., et al. Mtl-lora: Low-rank adaptation
for multi-task learning. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39, pp.
22010–22018, 2025c. doi: 10.1609/aaai.v39i20.
35509. URL https://ojs.aaai.org/index.
php/AAAI/article/view/35509.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W., Salakhut-
dinov, R., and Manning, C. D. Hotpotqa: A dataset
for diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, pp. 2369–
2380, Brussels, Belgium, 2018. Association for Compu-
tational Linguistics. doi: 10.18653/v1/D18-1259. URL
https://aclanthology.org/D18-1259/.
Yao, J., Li, H., Liu, Y ., Ray, S., Cheng, Y ., Zhang, Q.,
et al. CacheBlend: Fast large language model serving
for RAG with cached knowledge fusion. InProceed-
ings of the Twentieth European Conference on Com-
puter Systems (EuroSys ’25), pp. 94–109, New York,
NY , USA, 2025. Association for Computing Machin-
ery. doi: 10.1145/3689031.3696098. URL https:
//arxiv.org/abs/2405.16444.
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y .,
et al. Tree of thoughts: Deliberate problem solving with
large language models. InAdvances in Neural Informa-
tion Processing Systems, volume 36, pp. 11809–11822,
2023a.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., et al. React: Synergizing reasoning and acting in
language models. InThe Eleventh International Confer-
ence on Learning Representations, 2023b.
Ye, H., Gao, Z., Ma, M., Wang, Q., Fu, Y ., Chung, M.-
Y ., et al. KVCOMM: Online cross-context KV-cache
communication for efficient LLM-based multi-agent sys-
tems. InAdvances in Neural Information Processing Sys-
tems (NeurIPS 2025), 2025. doi: 10.48550/arXiv.2510.
12872. URL https://arxiv.org/abs/2510.
12872. Accepted for publication in NeurIPS 2025.
Yu, X., Luo, T., Wei, Y ., Lei, F., Huang, Y ., Peng, H.,
et al. Neeko: Leveraging dynamic LoRA for efficient
multi-character role-playing agent. InProceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pp. 12540–12557, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
697. URL https://aclanthology.org/2024.
emnlp-main.697/.
Yuan, Z., Shang, Y ., Song, Y ., Wu, Q., Yan, Y ., Sun, G., et al.
Asvd: Activation-aware singular value decomposition
for compressing large language models.arXiv preprint
arXiv:2312.05821, 2023. doi: 10.48550/arXiv.2312.
05821. URL https://arxiv.org/abs/2312.
05821.
Zhang, C., Goh, X. D., Li, D., Zhang, H., and Liu, Y . Plan-
ning with multi-constraints via collaborative language
agents. InProceedings of the 31st International Confer-
ence on Computational Linguistics, pp. 10054–10082,
Abu Dhabi, UAE, jan 2025. Association for Computa-
tional Linguistics. URL https://aclanthology.
org/2025.coling-main.672/.
11

<!-- page 12 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
A. Base Cache and Adapter Output
A.1. Cache L1 norm
As discussed in Section 3.1, the output contributions from the pretrained base weights and the low-rank adapters differs in
magnitude. We analyze the average output magnitude of the value projection in a multi-LoRA system with three agents,
where the pretrained base-weight contribution is treated as the base cache and the LoRA contribution is treated as the
adapter output. As shown in Figure 6, the base cache and adapter output magnitudes follow similar trends across layers, but
the adapter outputs are much smaller, by factors of 27.3 and 14.77 for LLaMA-3.1-8B-Instruct and Ministral-8B-Instruct,
respectively.
Given that the base cache remains highly similar across agents while the adapter outputs are largely decorrelated, the adapter
output can be viewed as a small but non-trivial, approximately random perturbation to the base cache. This motivates sharing
the base cache rather than the full cache.
Figure 6.L1 norm of the base cache and adapter output across model layers.
12

<!-- page 13 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
A.2. Cosine Similarity Bound
As shown in Section 3.1, the base cache remains highly similar across agents for the same context, while the adapter outputs
are largely decorrelated across agent pairs. This implies that the full cache can be viewed as the base cache perturbed by an
approximately random adapter component. In this section, we formalize the resulting effect on similarity: we show that,
under the approximate orthogonality assumptions on adapter outputs consistent with our empirical observations, the cosine
similarity of the base cache is higher than that of the full cache.
Based on the empirical observations on the similarity of the base cache and adapter contributions in Tables 1 and 6, we
derive the following bound showing that the full cache exhibits lower cosine similarity.
Following the multi-LoRA convention in Section 3.1, the base cache and adapter output are defined as:
Ybase,i :=X iW0,∆Y i :=X i∆Wi, Y i :=Y base,i + ∆Yi.(4)
Assuming that the adapter outputs are approximately orthogonal to the base cache and decorrelated across agents:
Y ⊤
base,i∆Yi = 0, Y ⊤
base,j∆Yj = 0, Y ⊤
base,i∆Yj = ∆Y ⊤
i Ybase,j = ∆Y ⊤
i ∆Yj = 0,(i̸=j).(5)
This satisfies the following:
Y ⊤
i Yj = (Ybase,i + ∆Yi)⊤(Ybase,j + ∆Yj) =Y ⊤
base,iYbase,j.(6)
Furthermore, we have
∥Yi∥2 =∥Y base,i + ∆Yi∥2 =∥Y base,i∥2 +∥∆Y i∥2 ≥ ∥Ybase,i∥2,(7)
and similarly∥Y j∥ ≥ ∥Y base,j∥. Therefore,
cos(Yi, Yj) = Y ⊤
i Yj
∥Yi∥ ∥Yj∥ =
Y ⊤
base,iYbase,j
∥Yi∥ ∥Yj∥ ≤
Y ⊤
base,iYbase,j
∥Ybase,i∥ ∥Ybase,j∥ = cos(Ybase,i, Ybase,j),(8)
and taking expectation over contexts yields
E

cos(Ybase,i, Ybase,j)

≥E

cos(Yi, Yj)

,(9)
which states that the base cache cosine similarity is higher than the full cache cosine similarity.
Table 6.Cosine similarity between base cache and adapter outputs across agent pairs.
Contribution∆Y plan ∆Yaction ∆Yreflect
Ybase,plan 0.0033 0.0050 -0.0003
Ybase,action 0.0188 0.0985 0.0006
Ybase,reflect -0.0076 0.0037 0.0488
13

<!-- page 14 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
A.3. Key Cache Cosine Similarity
As noted in Section 3.1, in typical multi-LoRA settings the key cache remains highly similar across agents. In particular, the
minimum key cache similarity already exceeds the average base-cache similarity reported in Table 1.
Figure 7 reports pairwise cosine similarity of the key cache for each model. The average similarity is 0.9922 for LLaMA-
3.1-8B-Instruct and 0.9840 for Ministral-8B-Instruct, and even the minimum similarity across agent pairs is higher than the
corresponding average base-cache similarity: 0.9726 and 0.9530, respectively. This indicates that the primary cross-agent
differences come from the value cache, mainly through the adapter-induced component. Therefore, we simply share the
entire key cache across agents in all of our schemes.
Figure 7.K cache Cosine similarity of each agent pairs across the model layers.
14

<!-- page 15 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
B. Flash-LoRA-Attention
In this section, we provide a detailed description of Flash-LoRA-Attention, which is introduced in Section 3.3, and analyze
its computational overhead in a general form, following the notation in Section 3.2 and Figure 3.
Setup.We follow the notation in Algorithm 1 and describe the forward pass for a single attention head. Let L=L p +L c,
where Lp is the accumulated context length and Lc is the context length of the current prefill/decoding step. We denote
the query and key as Q∈R Lc×dhead and K∈R L×dhead, and the base value cache as Vbase ∈R L×dhead. For the LoRA
update on the value projection, we save the LR cache Vlr ∈R L×r and keep the up-projection matrix B∈R r×dhead, where
r≪d head. With attention weightP= softmax
 
QK ⊤/√dhead

∈R Lc×L, the output is
O=P
 
Vbase +V lrB

=O base +O lr, O base =P V base, O lr =P V lrB(10)
Matrix Multiplication Reordering Based on Associativity.A straightforward implementation materializes the full-
dimension adapter contributionV lrB∈R L×dhead for allLtokens and then applies attention:
Olr =P(V lrB).(11)
This expands the LR cache to the head dimension over the entire trajectory, so the computation grows with bothL and dhead.
Instead, we exploit associativity and reorder the computation as
Olr =
 
P Vlr

B,(12)
so the length-L accumulation is performed in rank r, and the head-dimension multiplication by B is applied only once per
query block.
Compute Overhead.We report multiply-add counts up to constant factors and omit the shared base-attention cost for
Obase. Without reordering, we first formV lrBand then multiply byP:
W/o reorder:L r d head| {z }
VlrB
+L c L dhead| {z }
P(V lrB)
=O
 
L r dhead +L cL dhead

.(13)
With reordering, we first accumulateM=P V lr ∈R Lc×r and then applyB:
Reorder:L c L r|{z}
P Vlr
+L c r dhead| {z }
(P Vlr)B
=O
 
LcL r+L cr dhead

.(14)
Since r≪d head, reordering replaces the dominant LcL dhead-scaled expansion with an LcL r-scaled low-rank accumula-
tion, and thed head-dependent multiplication appears only once after the accumulation.
Flash-LoRA-Attention Kernel Implementation.Algorithm 1 implements Eq. (12) by extending FlashAttention with
one additional low-rank accumulator. For each query block Qi ∈R Br×dhead, the kernel streams over key/value blocks
(Kj, Vbase,j) and computes S= Mask(Q iK ⊤
j /√dhead), maintaining the online-softmax statistics (mi, ℓi). Using the same
block-wise weights, it accumulates both Oi ←O i +P iVbase,j and the low-rank intermediate Olr,i ←O lr,i +P iVlr,j, where
Olr,i ∈R Br×r remains in rank r throughout the streaming pass. After all blocks are processed, the kernel applies a single
post multiplication Oi ←O i +O lr,iB and then normalizes by ℓi. This preserves FlashAttention’s memory-efficient I/O
pattern while ensuring that the LR cache expansion computation scales primarily with the rank, directly reducing the runtime
overhead in bothBaseSharedandBaseLRShared.
15

<!-- page 16 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
C. Implementation Details
C.1. Agent Prompts and Trajectory Templates
We present an example agent prompt and trajectory from our multi-LoRA agent system, based on the AutoAct implementa-
tion (Qiao et al., 2024). As shown in Figure 8, a trajectory accumulates a predefined system prompt, the user question, and
multiple rounds of agent-generated tokens interleaved with rule-based context inserts, such as tool outputs retrieved from
external sources. The example is from HotpotQA, and ScienceQA follows the similar template except for an explanation on
the additional image caption lookup tool.
The system prompt, which specifies thethought,action, andobservationformat, is identical for all agents and thus fully
shared. As a result, prefix positional alignment based KV cache sharing methods (Yang et al., 2025b; Pan et al., 2025; Ye
et al., 2025) reduce toFullSharedin our setup.
The highlighted parts indicate agent-generated outputs, where agents execute in a predefined order (plan-plan-action). The
plan agent first produces reasoning and selects a tool, then the action agent generates the tool arguments. If the selected tool
is either Web search API, Wikipedia lookup, or image caption lookup that retrieves a predefined image caption, the retrieved
context is appended to the trajectory. If the selected tool isFinish, the action agent outputs a final answer and the reflect
agent is invoked to decide whether the answer is sufficient or whether another information retrieval round is needed. The
reflection step is divided into two stages, and it can override an incorrectFinishdecision and return control to the plan agent
when the retrieved evidence is insufficient. The total number of agent iterations is limited to 45 per question.
We train the agents using filtered trajectories generated by a single LLaMA-2-70B-Chat model, provided by AutoAct.
Figure 8.Agent prompts and an example of an accumulated trajectory on HotpotQA.
16

<!-- page 17 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
C.2. Shared-AMulti-LoRA Architecture
We observe that simply sharing the LoRA down-projection matrix A yields higher accuracy than conventional multi-LoRA
training with independent (Ai, Bi) pairs, consistent with prior findings (Tian et al., 2024; Yang et al., 2025c). Table 7 reports
HotpotQA accuracy with and without shared-A. We note that we use the same training conditions and hyperparameters
listed in Appendix C.3, which yield the best results in both settings. Across all methods, the shared- A variant improves the
original (Non-Shared) accuracy and also benefits cache sharing schemes such as FullShared, BaseShared, and
BaseLRShared. In particular, BaseLRShared degrades when A is not shared, since sharing an LR cache computed
with differentAmatrices and expanding it with a mismatchedBintroduces large errors.
In our implementation, we duplicate the shared A across agents, which is the same implementation with conventional
multi-LoRA architecture and therefore inference efficiency is unchanged. Since shared-A improves accuracy in all settings
without introducing change of model structure or inference overheads, we conduct our main experiments using shared-A
multi-LoRA trained weights. We note that sharing A also reduces the number of trainable parameters by 33%, providing a
efficiency benefit in training.
Table 7.Accuracy (%) comparison between non-shared-Aand shared-Amulti-LoRA variants on HotpotQA easy benchmark.
Model Architecture Non-Shared FullShared BaseShared BaseLRShared
LLaMA-3.1-8B-Instruct Non-sharedA42.05 40.30 41.85 36.25
Shared-A42.80+0.7541.15+0.8542.70+0.8542.40+6.15
Ministral-8B-Instruct Non-sharedA41.10 37.40 40.80 36.95
Shared-A41.30+0.2039.60+2.2040.95+0.1541.10+4.15
17

<!-- page 18 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
C.3. Hyperparameter and Loss Curve
We report the training hyperparameters and loss curves for multi-LoRA agents in Table 8 and Figure 9, respectively. Most
hyperparameters, including the optimizer, scheduler, and weight decay, follow AutoAct (Qiao et al., 2024), and we perform
a grid search over learning rates and the number of training epochs. The sum of training time across all agents is 3.9 hours
for HotpotQA and 3.4 hours for ScienceQA on a single 48GB NVIDIA A6000 GPU. We also note that the HotpotQA and
ScienceQA test sets consist of 300 and 360 questions, respectively, and we run 20 iterations for each accuracy evaluation.
Table 8.Hyperparameter settings for multi-LoRA training.
Hyperparameter LLaMA-3.1-8B-Instruct Ministral-8B-Instruct
Optimizer AdamW
Batch Size 32
LR Scheduler cosine
Max Sequence Length 32786
Epochs 10
Warmup Ratio 0.05
Weight Decay 0
Rank 8
LoRA Dropout 0.05
LoRA Scale 16
Learning Rate
Plan: 5e-5 Plan: 5e-5
Action: 6e-5 Action: 9e-5
Reflect: 6e-5 Reflect: 9e-5
Figure 9.Train loss and L2 norm of the gradient for each agent types.
18

<!-- page 19 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
C.4. DroidSpeak Recomputation Layer Selection
We implement DroidSpeak as a baseline, following the methodology described in (Liu et al., 2026). In the original
implementation, KV caches are selectively recomputed for a set of critical layers, identified by probing the benchmark
accuracy drop when the KV cache is directly reused in each layer while the remaining layers are recomputed. DroidSpeak
provides a Pareto-optimal configuration that balances accuracy degradation and the inference efficiency gains from cache
sharing, corresponding to recomputing 33% of the model layers. Following this guideline, we probe critical layers on
HotpotQA and select 11 layers for LLaMA-3.1-8B-Instruct (32 layers total) and 12 layers for Ministral-8B-Instruct (36
layers total). In addition, since the first layer typically does not require recomputation in Ministral-8B-Instruct, we enable
hidden state caching that can be passed directly from the previous model to the current model to eliminate computation
for these layers. We note that recent models often use group-query attention (GQA), where the hidden state dimension
is typically four times larger than the output dimension of the key or value projections, so the hidden state cache can be
roughly twice as large as the KV cache of a single layer. This additional memory overhead is discussed in Appendix D.6.
Table 9.Selected layers for recomputation in DroidSpeak.
Model Selected layers
LLaMA-3.1-8B-Instruct 0, 2, 16, 19, 20, 22, 23, 24, 26, 30, 31
Ministral-8B-Instruct 1, 4, 5, 12, 14, 15, 17, 21, 22, 25, 29, 31
C.5. Emulated Trace for Efficiency Analysis
In the experiments on HotpotQA or ScienceQA benchmarks, methods with lower accuracy tend to produce longer trajectories
and thus accumulate longer contexts. Therefore, to enable fair efficiency comparisons of only the cache sharing method
itself under the same context length, we construct a fixed trace of context lengths and an agent schedule. This trace is based
on profiled trajectories, including the concatenated context length at each agent step and the number of steps per iteration.
On average, each iteration consists of 17 steps, comprising five plan-plan-action cycles and two reflect steps at the end of the
iteration.
We vary the retrieved context lengthLctx from 0.25k to 16k, which results in total trajectory lengths ranging from 1.9k to
66.4k, as reported in Table 4 and Table 5. We note that the detailed agent trajectory templates are described in Appendix C.1.
Table 10.Agent iteration trace with prefill and generation lengths.
Agent Type Step Prefill Generation
plan 1 512 32
plan 2 8 8
action 3 8 8
plan 4L ctx 32
plan 5 8 8
action 6 8 8
plan 7L ctx 32
plan 8 8 8
action 9 8 8
plan 10L ctx 32
plan 11 8 8
action 12 8 8
plan 13L ctx 32
plan 14 8 8
action 15 8 8
reflect 16 32 32
reflect 17 8 8
19

<!-- page 20 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
D. Experiments
D.1. Ablation on LoRA Application
Although LoRA is most commonly applied to the query and value projections, which we denote as the qv setting, we also
evaluate an alternative configuration with the same parameter budget by applying LoRA to the query, key, value, and output
projections with rank r= 4 , which we denote as qkvo. Following Section 4.3, we measure HotpotQA accuracy along with
system throughput and TTFT under the same emulated trace, where the results are presented in Table 11, 12, and 13.
We find that the non-shared baseline inqkvo achieves lower accuracy than in qv, and this degradation carries over to all
cache sharing methods. Nevertheless, our methods still achieve the best accuracy among the cache sharing approaches,
indicating that decoupling the base cache and the LR cache remains effective when LoRA is applied to the key projection.
In terms of system throughput, the qkvo setting is inherently less favorable because it introduces additional LoRA
computation paths on multiple projections. Moreover, in our schemes, adapter contribution reconstruction from key LR
cache must be performed in the head dimension before applying rotary positional embeddings (RoPE), which limits the same
associativity-based reordering we exploit for the value cache. Concretely, letting the post-RoPE query beQ and the pre-RoPE
key be K ′ =K ′
base +K ′
lrB with K= RoPE(K ′), the attention score is QK ⊤ =Q
 
RoPE(K ′)
⊤
=Q

RoPE(K ′
base) +
RoPE(K ′
lrB)
⊤
, so the low-rank reordering fromQ(B ⊤K ′⊤
lr )to(QB ⊤)K ′⊤
lr is not directly applicable becauseRoPE(·)
applies a position-dependent rotation on the head dimension. As a result, both BaseShared and BaseLRShared under
qkvo achieve lower throughput than their qv counterparts reported in Table 4. Still, BaseLRShared remains more
efficient than DroidSpeak. On the other hand, because the adapter output reconstruction primarily affects the generation stage
rather than prefill, TTFT under qkvo increases marginally overall compared to Table 5. Similarly with the previous results,
BaseLRSharedremains close toFullSharedin TTFT, whileBaseSharedremains comparable to DroidSpeak.
Overall, our approach maintains the strongest accuracy among cache sharing baselines, and BaseLRShared retains a
clear efficiency advantage, demonstrating the generality and scalability of our design across LoRA configurations, while qv
setting used in this paper is favorable across the as mentioned in Section 4.1.
Table 11.LLaMA-3.1-8B-Instruct average HotpotQA benchmark accuracy (%) under theqkvoscheme.
Method Non-Shared FullShared DroidSpeak BaseShared BaseLRShared
Accuracy (%) 38.43 34.88 36.28 38.12 37.57
Table 12.LLaMA-3.1-8B-Instruct system throughput (tokens per second) under each total sequence length of the traces in qkvo setting.
Method 1.9k 3.0k 5.0k 9.1k 17.3k 33.7k 66.4k
Non-Shared123.1 184.4 297.1 467.1 525.0 556.0OOM
FullShared155.8 215.1 357.2 626.3 1080.6 1544.9 1699.4
DroidSpeak145.9 211.8 326.2 519.6 739.1 887.2 792.9
BaseShared127.2 195.6 325.7 484.9 679.3 755.6 673.2
BaseLRShared150.3 219.7 361.3 560.3 806.7 998.8 1051.9
Table 13.LLaMA-3.1-8B-Instruct TTFT (second) under each total sequence length of the traces inqkvosetting.
Method 1.9k 3.0k 5.0k 9.1k 17.3k 33.7k 66.4k
Non-Shared4.79 5.00 4.94 10.52 20.73 50.04OOM
FullShared1.23 1.40 1.81 2.70 4.82 9.57 24.05
DroidSpeak1.77 2.29 3.37 5.84 11.54 25.43 67.80
BaseShared1.78 2.34 3.29 5.59 10.99 26.20 70.57
BaseLRShared1.27 1.46 1.92 2.89 4.93 10.30 25.30
20

<!-- page 21 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
D.2. Latency on HotpotQA Benchmark
We report the end-to-end system latency on the HotpotQA benchmark. In real-world scenarios, additional latency other
than model inference arises from function calls such as web search and Wikipedia retrieval. We therefore split latency into
model latency, which includes only the inference (prefill and generation), and end-to-end (E2E) latency, which additionally
includes function call latency and the latency of data processing the retrieved context. We also report time-to-first-token
(TTFT), defined as the sum of model prefill latencies across agent steps, consistent with Table 5 in Section 4.3.
As shown in Table 14 and Table 15, methods with lower accuracy, such asFullShared and DroidSpeak, tend to produce
longer sequences and incur higher latency, sometimes even exceeding the Non-Shared baseline. This occurs despite
their strong efficiency which they have demonstrated in trace-based emulations. These results highlight that overall latency
depends not only on the cache sharing efficiency, but also on accuracy. When cache sharing degrades generation quality,
the agent is more likely to take additional steps to re-reason and retrieve more external context, which increases sequence
length and, in turn, increases E2E latency. Overall, FullShared achieves a low TTFT but incurs substantial E2E latency
overhead compared to Non-Shared on LLaMA-3.1-8B-Instruct. DroidSpeak and BaseShared exhibit end-to-end
latency similar to Non-Shared. BaseLRShared achieves the best efficiency while preserving strong accuracy, making
it empirically optimal for agentic systems.
Table 14.End-to-end (E2E) latency and its breakdown. The lowest latency is highlighted in bold.
Model Latency Level Non-Shared FullShared DroidSpeak BaseShared BaseLRShared
LLaMA-3.1-8B-Instruct
Model Latency (s)
Hard 5.90 7.39 5.75 5.975.70
Medium 5.60 6.70 5.30 5.355.13
Easy 5.04 6.22 4.69 4.764.54
Avg. 5.51 6.77 5.24 5.365.12
E2E Latency (s)
Hard 13.63 19.73 13.75 13.7313.59
Medium 12.10 18.48 11.85 12.0511.73
Easy 11.10 14.87 10.64 10.8910.54
Avg. 12.28 17.69 12.08 12.2311.96
TTFT (s)
Hard 1.02 0.81 1.00 1.070.77
Medium 1.01 0.77 0.98 1.020.66
Easy 1.050.720.97 1.03 0.88
Avg. 1.030.770.98 1.04 0.77
Ministral-8B-Instruct
Model Latency (s)
Hard 6.64 7.23 6.82 6.796.39
Medium 6.36 6.43 6.23 6.396.00
Easy 6.86 6.05 6.11 5.865.77
Avg. 6.62 6.57 6.38 6.356.05
E2E Latency (s)
Hard 14.65 14.36 14.98 15.0613.91
Medium 12.66 14.30 12.41 12.2511.45
Easy 15.22 12.85 12.98 12.9212.79
Avg. 14.18 13.83 13.46 13.4112.72
TTFT (s)
Hard 1.26 1.30 1.44 1.331.26
Medium 1.171.071.26 1.17 1.12
Easy 1.22 1.12 1.24 1.201.08
Avg. 1.22 1.16 1.31 1.231.15
Table 15.Average of total sequence length (tokens) accumulated during multi-agent execution.
Model Level Non-Shared FullShared DroidSpeak BaseShared BaseLRShared
LLaMA-3.1-8B-Instruct
Hard 1099 1584 1223 1039 1302
Medium 1092 1475 1108 996 1229
Easy 1088 1483 1134 1077 1154
Avg. 1093 1514 1155 1038 1228
Ministral-8B-Instruct
Hard 1120 1460 1355 1202 1468
Medium 1019 1398 1347 1144 1406
Easy 1154 1393 1414 1142 1363
Avg. 1098 1417 1372 1162 1412
21

<!-- page 22 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
D.3. Out-of-Function Ratio
We define cases where the agent system fails to produce an answer before reaching the maximum number of iterations (e.g.,
45) as out-of-function (OOF). In the benchmark accuracy evaluation, these cases are counted as incorrect. However, from a
user-experience perspective, returning no answer can be qualitatively different from returning an incorrect answer, and may
be considered a more severe failure. We therefore report OOF incidents in addition to benchmark accuracy.
As shown in Table 16, which reports the OOF ratio and its difference from the Non-Shared baseline, methods with
lower accuracy generally exhibit higher OOF ratios. Consistent with the accuracy results where BaseShared and
BaseLRShared achieve the best accuracy among cache sharing methods, our schemes also yield lower OOF ratios in
most cases, except for Ministral-8B-Instruct on ScienceQA.
Table 16.Out-of-function (OOF) rate (%) for each benchmark and difficulty level. The underlying value in the Avg. column denotes the
difference from the correspondingNon-Sharedbaseline. Lower is better.
HotpotQA ScienceQA
Model Method Easy Medium Hard Avg. 1-4 5-8 9-12 Avg.
LLaMA-3.1-8B-Instruct
Non-Shared1.50 1.80 1.65 1.650.000.00 0.13 0.21 0.110.00
FullShared2.05 2.05 3.25 2.45+0.800.29 2.46 0.54 1.10+0.99
DroidSpeak2.10 3.10 5.15 3.45+1.800.58 1.29 1.71 1.19+1.08
BaseShared1.35 1.50 2.65 1.83+0.180.21 1.83 0.13 0.72+0.61
BaseLRShared1.25 1.80 2.25 1.77+0.120.29 2.50 0.25 1.01+0.90
Ministral-8B-Instruct
Non-Shared3.90 5.15 5.80 4.950.000.38 0.17 0.83 0.460.00
FullShared7.15 7.65 10.95 8.58+3.634.17 2.96 4.92 4.01+3.56
DroidSpeak7.05 9.50 8.20 8.25+3.301.79 2.17 3.33 2.43+1.97
BaseShared6.45 6.35 9.50 7.43+2.482.75 1.92 3.25 2.64+2.18
BaseLRShared4.45 6.55 7.60 6.20+1.251.83 1.67 3.38 2.29+1.83
D.4. Rank Ablations
We report the accuracy of the Non-Shared baseline and our schemes, BaseShared and BaseLRShared, under ranks
from 4 to 32, as shown in Table 17. Since the agent-trajectory training data are relatively small and easy to adapt, we observe
marginal accuracy differences across ranks, and ranks above 8 do not yield noticeable accuracy improvements. Therefore, to
minimize both training and inference overhead, we use rankr= 8in all experiments.
Table 17.Rank ablation on HotpotQA average accuracy (%).
Rank 4 8 16 32
Non-Shared 38.25 38.88 39.00 38.93
BaseShared 37.88 38.60 38.43 38.50
BaseLRShared 37.32 37.92 37.95 37.97
22

<!-- page 23 -->

LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
D.5. Accuracy Score Deviation
We report the standard deviation of the average accuracy in Table 3 of Section 4.2 for each baseline and our methods
in Table 18. We note that all experiments use a single random seed (42), but accuracy can still vary due to subtle non-
deterministic characteristics in external tool usage. For each benchmark level, we run 20 iterations. The accuracy gaps
between methods are larger than the observed deviations and therefore we see that the comparisons remain reliable.
Table 18.Standard deviation of average benchmark accuracy (%) with 20 iterations.
Dataset Non-Shared FullShared DroidSpeak BaseShared BaseLRShared
HotpotQA 0.16 0.32 0.21 0.19 0.24
ScienceQA 0.20 0.45 0.25 0.26 0.31
D.6. Memory Usage
We report the memory usage of each method across traces with diverse trajectory lengths on Ministral-8B-Instruct. We note
that the pretrained model weights consume 14.95 GB of memory, and the three LoRA weights add 0.11 GB of memory.
Beyond these components, KV cache memory becomes severe in long-context scenarios where retrieved contexts accumulate.
Since KV cache sharing methods typically maintain a single shared KV cache for three agents and recompute and overwrite
it when needed, their memory usage is similar within 1 GB difference overall.
In particular, FullShared has the lowest memory usage because it directly reuses the KV cache without additional
components. DroidSpeak additionally maintains a hidden state cache. Since the first layer typically does not require
recomputation, its hidden states can be transferred directly from the previous model to the current model, eliminating
computation for these layers. However, this cache becomes an overhead in modern group-query attention (GQA) models.
The hidden state dimension is often about four times larger than the key or value projection dimension, so the hidden state
cache can be roughly twice as large as the KV cache of a single layer. We note that the OOM observed in Section 4.3 mainly
arises from memory fragmentation, despite the gap between the GPU’s peak capacity and the actual allocated usage. For
BaseShared and BaseLRShared, there exists an additional LR cache, which is three times larger in BaseShared
than in BaseLRShared, but it remains negligible due to its small dimension relative to the base cache. As a result, our
schemes achieve memory usage close toFullSharedas well as other cache sharing methods.
Table 19.Memory usage (GB) for each total sequence length trace.
Total Seq. Len. Non-Shared FullShared DroidSpeak BaseShared BaseLRShared
1.9k 15.72 15.25 15.26 15.26 15.25
3.0k 16.10 15.38 15.40 15.40 15.39
5.0k 16.87 15.65 15.68 15.67 15.65
9.1k 18.40 16.18 16.25 16.23 16.19
17.3k 21.46 17.24 17.37 17.34 17.27
33.7k 27.59 19.36 19.62 19.56 19.43
66.4k 39.84 23.61 24.12 23.99 23.74
132.0k 64.34 32.11 33.12 32.87 32.37
23
