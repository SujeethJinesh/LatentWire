# references/150_crystal_kv_efficient_kv_cache_management_for_chain_of_thought_llms_via_answer_first_principle.pdf

<!-- page 1 -->

Crystal-KV: Efficient KV Cache Management for
Chain-of-Thought LLMs via Answer-First Principle
Zihan Wang
School of Computer Science and
Technology, University of Science and
Technology of China
Hefei, China
wangzh196@mail.ustc.edu.cn
Cheng Tang
School of Computer Science and
Technology, University of Science and
Technology of China
Hefei, China
sisyphustc@mail.ustc.edu.cn
Lei Gong
School of Computer Science and
Technology, University of Science and
Technology of China
Hefei, China
leigong0203@ustc.edu.cn
Cheng Li
University of Science and Technology
of China; Institute of Artificial
Intelligence, Hefei Comprehensive
National Science Center
Hefei, China
chengli7@ustc.edu.cn
Chao Wang
School of Computer Science and
Technology, University of Science and
Technology of China
Hefei, China
cswang@ustc.edu.cn
Teng Wang
Suzhou Institute for Advanced
Research, University of Science and
Technology of China
Suzhou, China
wangt635@ustc.edu.cn
Wenqi Lou
Suzhou Institute for Advanced
Research, University of Science and
Technology of China
Suzhou, China
louwenqi@ustc.edu.cn
Xuehai Zhou
School of Computer Science and
Technology, University of Science and
Technology of China
Hefei, China
xhzhou@ustc.edu.cn
Abstract
Chain-of-Thought (CoT) reasoning in large language models (LLMs)
significantly improves accuracy on complex tasks, yet incurs ex-
cessive memory overhead due to the long think-stage sequences
stored in the Key-Value (KV) cache. Unlike traditional generation
tasks where all tokens are uniformly important, CoT emphasizes
the final answer, rendering conventional KV compression strate-
gies ineffective. In this paper, we present Crystal-KV, an efficient
KV cache management framework tailored for CoT reasoning. Our
key insight is the answer-first principle. By mapping answer pref-
erences into think-stage attention map, we distinguish between
SlipKV, which mainly maintains the reasoning flow but may occa-
sionally introduce misleading context, and CrystalKV, which truly
contributes to the correctness of the final answer. Next, we propose
an attention-based Least Recently Frequently Used algorithm. It
precisely identifies when a SlipKV entry’s utility expires and evicts
it, retaining CrystalKV without disrupting reasoning flow. Finally,
we introduce an adaptive cache budget allocation algorithm. Based
on the dynamic proportion of CrystalKV, it estimates the impor-
tance of each layer/head and adjusts the KV cache budget during
inference, amplifying critical components to improve budget uti-
lization. Results show that Crystal-KV achieves state-of-the-art KV
cache compression, significantly improves throughput, and enables
faster response time, while maintaining, or even improving, answer
accuracy for CoT reasoning. The Code will be open-sourced
at github.com/xxx.
Keywords
KV Cache Compression, Efficient CoT Reasoning, LLM Inference
1 Introduction
Chain-of-Thought (CoT) reasoning has been widely adopted in large
language models (LLMs) to solve complex tasks such as mathemat-
ics and programming, with notable success in models like ChatGPT
5 (Thinking) [21], DeepSeek R1 [12], Qwen3 [30], and Gemini 2.5 [7].
The key of CoT is to insert a think stage before generating the final
answer [26]. As shown in Fig. 1, the user first submits a prompt.
The LLM then enters a think stage (i.e., the CoT stage), where it
generates a massive number of tokens for knowledge expansion
and logical deduction. Finally, the LLM outputs the final answer.
Importantly, the intermediate think stage is often hidden or useless to
users, while what users care about is the final answer. Despite CoT’s
potential for accuracy, it incurs substantial memory overhead, as
massive think-stage tokens correspond to a large Key-Value (KV)
cache [11]. For example, DeepSeek-R1-Distill-Qwen-14B may con-
sume 8K tokens to solve a complex coding problem, requiring 10
GB of memory for KV storage, hundreds of times larger than the
prompt and answer stage KV. This dramatically increases memory
usage and computational cost, while also slowing down response
time for users. Therefore, compressing the think-stage KV cache is
essential to enable efficient deployment of CoT reasoning.
The key to KV compression is dynamically evicting redundant
KV cache entries during inference while preserving accuracy [4, 29].
Although many compression methods exist, most are limited to nor-
mal long-content generation (LCG) tasks, such as dialogue systems,
and fail to generalize to reasoning tasks with CoT. The fundamental
reason lies in the generation objective shift: from producingALL
output tokens to focusingONLYon the final answer. In LCG tasks,
arXiv:2601.16986v1  [cs.CL]  5 Jan 2026

<!-- page 2 -->

What can I help you today?
Write all numbers from 1 to 100 in a
row with no spaces. What digit is at
position 2025? <User Prompt>
<prompt>(User Prompt)</prompt>
<think>Ok, let me think step by step…
We are asked to find …
Let’s count how many digits…
First, one-digit numbers…
Then, three-digit numbers…
So the total length of the…
Maybe the question is…
Let me read question again…
Wait,…
So I assume question meant 1 to…
…
</think>
<answer>Answer is 1.</answer>
LLM Models
with CoT
Wow! Amazing!
Let me try…
GPU
Memory
②
①
③
Huge Hidden/Useless Think Token
LLM Models
with CoT
USER
Massive
Think
-
KV
Model Thinking (Reasoning)...
Ugh,Too slow!!!
What’s up bro?
USER
USER
Okay, I just
need this answer.
An Example of Using a CoT-Reasoning Model
Answer
-
First
Principle
Memory’s
On Fire
Step ①~③are CoT Reasoning Process, non-CoT reasoning without ②
Figure 1: The Workflow of Chain-of-Thought Reasoning
since every token is relevant to the user, these compression meth-
ods fairly maintain the quality of each token generation. As a result,
by approximating attention scores in the near future, they greedily
evict KV entries that are lowly attended by the next few tokens. In
contrast, in reasoning tasks with CoT, users only care about the final
answer. Therefore,uniform token treatmentandanswer-first
prioritizationare inherently conflicting optimization goals. Even
worse, the shortsighted greed induced by the uniform treatment
significantly harms the quality of answer-stage tokens, which are
generated at the end of the sequence. While some work explores
KV compression for CoT [4, 15], they overlook the decisive role of
the final answer and treat the think stage as a special case of LCG.
Our key insight is the answer-first principle: retain only those
KV entries that contribute to the final correct answer during the
think stage, without disrupting the reasoning flow. However, three
challenges arise: (i) Temporal lag: Since the answer stage occurs
entirely after the think stage, it is infeasible to reliably assess a
think-stage KV entry’s importance to the final answer based on
attention score approximation. This raises a fundamental question:
Do the think-stage KV entries that contribute to the correct answer
exhibit any unified attention pattern during the think stage?If so, it
could serve as the basis for early identification of critical KV entries
before the answer stage begins. (ii) Dynamism: CoT reasoning is a
dynamic logical deduction, where important think-stage KV entries
may emerge at any time. This requires not only maintaining the
evolving reasoning flow, but also accurately retaining critical KV
entries in real time. (iii) The combined effects of temporal lag and
dynamism push KV-cache budget allocation into a dilemma: either
wasting cache space or missing critical KV entries.
To address these challenges, we propose Crystal-KV, an efficient
KV cache management framework for CoT reasoning in LLM in-
ference. (i) By mapping answer preferences into the think-stage
attention map, we reveal two unified attention patterns. The KV
entries truly contributing to the correct answer are intermittently,
rather than continuously, attended until the end of reasoning. We
refer to these as CrystalKV. Meanwhile, others exhibit a streaming
attention pattern. Though helpful in maintaining reasoning flow,
these tokens may occasionally introduce misleading information
for the final answer, and are thus referred to as SlipKV. Excitingly,
our experiments show that focusing more attention on CrystalKV
during answer stage can even turn previously wrong answers into
correct ones. (ii) Based on unified attention patterns, we propose
an attention-based Least Recently Frequently Used (LRFU) algo-
rithm. It precisely identifies when a SlipKV entry’s utility expires
and performs eviction, ensuring CrystalKV is accurately retained
without disrupting reasoning flow. (iii) We introduce an adaptive
cache budget allocation algorithm. Based on the dynamic propor-
tion of CrystalKV, it estimates the importance of each layer/head
and adjusts the KV cache budget during inference, amplifying the
role of critical components to improve cache space utilization.
Results show that Crystal-KV substantially reduces memory
usage by an average of 90.89%, improves throughput by 7.57× on
average, and achieves up to 1.24× speedup in user-level response
latency, all while maintaining lossless compression. Even more
compelling, under complex long-sequence reasoning tasks, Crystal-
KV achieves 105% of FullKV accuracy using only 10% of the KV
budget, whereas existing methods attain only 30%–67% accuracy.
The main contributions are as follows:
• By mapping answer preferences into think-stage attention
map, we distinguish unified attention patterns of CrystalKV
and SlipKV, laying foundation for answer-first principle.
• We propose an attention-based LRFU algorithm that retains
CrystalKV while maintaining the reasoning flow.
• We introduce an adaptive cache budget allocation algorithm
that amplifies the influence of critical layers and heads, im-
proving cache space utilization.
• Results show that Crystal-KV achieves state-of-the-art KV
cache compression and faster response time, while maintain-
ing, or even improving, answer accuracy in reasoning tasks
with Chain-of-Thought.
2 Background and Motivation
2.1 Chain-of-Thought
Chain-of-Thought (CoT) has emerged as a mainstream technique
for enhancing the reasoning capabilities of large language models
(LLMs) by explicitly modeling the human-like intermediate thinking
process, enabling LLMs to solve complex tasks such as mathematics
and programming [12, 21, 34]. Although CoT still relies on tradi-
tional autoregressive token generation for knowledge expansion
and logical deduction, it introduces two key distinctions. First, it
generates massive think tokens (often tens of times larger than the
prompt and answer), leading to substantial KV-cache overhead that
becomes the major performance bottleneck [ 2, 10, 14, 25, 31, 33].
Second, unlike normal generation tasks where all outputs are ex-
posed to the user, the think stage remains hidden or irrelevant, and
users ultimately care only about the final answer.
In summary, these two distinctions respectively highlight the ne-
cessity of think-stage KV compression and the answer-first priority
in CoT reasoning.
2.2 KV Cache Compression
During LLM autoregressive generation, each new token must at-
tend to all previously generated tokens, and models store their

<!-- page 3 -->

?
? ?
? ? ?
? ? ? ?
Think Query
Think2Think Attention MAP
②
①Answer2Think Attention MAP
Attention MAP
Answer Query
Top 30%
Think KV Cache
Case1 Case2 Case3 Case5 Case6
 Intermittently
Pattern in
Think2Think
Streaming
Pattern In
Think2Think
Answer only focus
part of
Think tokens
①
Top/Bottom 30%
②: Top/Bottom 30% Answer-Focused Think-KV Patterns in Think2Think Attention Map
②
Case7 Case8
Answer Focused
ThinkKV
①
Patterns during
Think Stage?
Think Token
Answer Token (User Focus)
Case4
Bottom 30% Key/Value
②
②
②
Changed 1-3
Figure 2: Distinguishing CrystalKV and SlipKV by Projecting Answer Preferences onto Think-Stage Attention Map
intermediate states as Key-Value (KV) cache to avoid recomputa-
tion [6, 16, 17, 19, 28, 37]. As output sequences grow, the KV cache
expands proportionally, introducing substantial memory and com-
pute overhead, which makes KV-cache compression (i.e., evicting
redundant KV entries) essential [1, 22, 24, 32, 36]. Attention scores
serve as the gold standard for evaluating the importance of KV
entries [4, 15, 29]. Formally, for head ℎ in layer 𝑡, the 𝑖-th token
𝑥𝑖 is projected into 𝑞𝑖, 𝑘𝑖, 𝑣𝑖, where (𝑘𝑖, 𝑣𝑖 ) are appended to the ex-
isting KV cache (𝐾𝑡,ℎ, 𝑉𝑡,ℎ ). The next token 𝑦 is then generated by
attending over the entire cached history:
AttnScore=softmax(𝑞 𝑖𝐾 ⊤
𝑡,ℎ ), 𝑦=AttnScore𝑉 𝑡,ℎ .
Here, AttnScore∈𝑅 1×𝑖 denotes the attention weights over previous
𝑖 cached tokens. A higher value of AttnScore[0, 𝑠] (for0 ≤𝑠≤𝑖 )
indicates that the 𝑠-th token contributes more to the generation of
𝑦, and is therefore more critical for preserving output quality.
By estimating near-future attention scores, existing KV compres-
sion methods fairly preserve the quality of each token generation.
For instance, StreamingLLM [29] assumes that upcoming tokens
tend to focus on the initial and most recent tokens. H2O [35] and
SnapKV [20] approximate the attention distribution of next few
tokens using score accumulation and observation windows. How-
ever, CoT-based reasoning tasks emphasize answer prioritization.
The shortsighted greed induced by the uniform treatment harms
the quality of answer-stage tokens generated at the end. More-
over, several recent works have attempted KV compression for CoT.
Raas [15] identifies attention handoff patterns to facilitate high-
quality think tokens. R-KV [ 4] observes high similarity among
think-stage KV entries and performs deduplication. Unfortunately,
all of them inherit a fundamental flaw of token-uniform principle,
treating CoT as a special case of normal long-context generation.
In summary, these limitations motivate a comprehensive rethink-
ing of KV cache management for CoT reasoning, guided by the
answer-first principle.
2.3 Budget Allocation
KV cache budget allocation faces a trade-off between wasting mem-
ory and missing important KV entries. Pyramid [ 5] introduces a
layer-wise strategy: shallow layers exhibit dispersed attention and
require larger budgets, while deeper layers are more focused and
need less. Ada-KV [9] observes a similar observation across atten-
tion heads and proposes head-wise allocation. However, both rely
on attention distributions of the next few tokens and are tailored
for LCG compression, making them incompatible with CoT-specific
methods like RaaS and R-KV.
In summary, these limitations motivate a rethinking of how to
amplify the role of key layers and heads during the think-stage KV
compression, thereby improving cache space utilization.
3 Key Insight
We now revisit the question posed in Sec. 1:Do the think-stage
KV entries that contribute to the correct answer exhibit any unified
attention pattern during the think stage?As shown in Fig. 2 (left), the
think-stage attention map is split into two submaps ① and ②. The
right panel presents attention maps from math and code reasoning
tasks. Following the answer-first principle, we first examine the
(AnswerQuery,ThinkKV) submap. As shown in the rows of right
panel ①, answer tokens sparsely yet consistently attend to a specific
subset of think-stage KV entries, implying that only a fraction
contributes to answer generation. We then rank all think-stage KV
entries by attention scores from AnswerQuery and project top and
bottom 30% to (ThinkQuery,ThinkKV) submap. Top-30% entries
show a sink-to-bottom pattern, intermittent rather than continuous
T6T3
Compression
(per Head)
×
T1 T2 T3 T4 T5 T6 T7
New
KV Cache
Old Think-Stage
KV Cache
Time Stamp
T4T2
T1 T3 T5 T6 T7 T8 T9
Filtering CrystalKV New
T1 T5 T7 T8 T9
Filtering CrystalKV
T1 T8 TK Tm Tn
GPU’s HBM Final Hold
Memory’s
At Ease
GPU
Memory
√
Budget Size (<< n)
LLM Infer
Init as
PotentialKV
Attn Based
LRFU
Section 3
SlipKV
CrystalKV
Principle
Answer
-
First
Section 5.2
Maintain Reasoning Flow
& Filter CrystalKV
Layer/Head Wise
Adaptive Budget Allocation
Section 5.3
Layer3
Head_0 Head_1
…
LayerN
Head_0 Head_1
Layer2
Head_0 Head_1
Layer3
Head_0 Head_1
Layer1
Head_0 Head_1
High
 Utilization
Amplified Vital
… Until
Think-Stage End
×
Tx
CrystalKV
Correct
Answer
SlipKV
Tx Reasoning
Flow
PotentialKV
Tx
Initial
State
Figure 3: Overview of Crystal-KV

<!-- page 4 -->

attention that persists until the end of the think stage, referred to as
CrystalKV. Bottom-30% entries show a streaming pattern, receiving
strong local attention before gradually fading, which we call SlipKV.
Functionally, CrystalKV underpins answer correctness, whereas
SlipKV supports the maintenance of the ongoing reasoning process.
This two-step analysis reveals a unified attention pattern that
distinguishes CrystalKV and SlipKV, serving as the foundation of
the answer-first principle.
4 Overview
Fig. 3 presents the overview of Crystal-KV, a KV cache management
framework tailored for think-stage KV compression. First, by distill-
ing two unified attention patterns, we distinguish CrystalKV, which
contributes to answer correctness and SlipKV, which maintains the
reasoning flow, forming the foundation of the answer-first principle
(see Sec. 3). Then, during the think stage, an attention-based LRFU
policy accurately retains CrystalKV while maintaining ongoing
reasoning. Instead of explicitly identifying and locking CrystalKV,
we adopt an evolutionary perspective and treat all newly arrived
KV entries as PotentialKV. When a PotentialKV transitions into
SlipKV, it is evicted, indicating its utility in supporting the reason-
ing flow expires. Therefore, only CrystalKV with persistent and
answer-oriented contribution remains in the cache by the end of
the think stage (see Sec. 5.2). Finally, the adaptive budget allocator
dynamically adjusts per-layer and per-head cache sizes, amplifying
the impact of critical layer/head to improve overall cache utilization
(see Sec. 5.3). In practice, Crystal-KV achieves high answer accuracy
with extremely low memory usage and fast response time (Sec. 6).
5 Implementation Details
5.1 Beyond Local Perspective
In Fig. 2, the attention pattern is remarkably clear from a global
perspective after the think stage ends. However, during the think
stage, we only have access to local attention views, making it non-
trivial to distinguish CrystalKV from SlipKV.
The key challenge arises from the local perspective: within a
short time window after a new KV arrives, the boundary between
CrystalKV and SlipKV is ambiguous. As shown in Fig. 2, some
SlipKV may have long utility spans for maintaining reasoning
flow. If H2O is used, these SlipKV may be mistakenly treated as
CrystalKV, occupying valuable cache space. If StreamingLLM is
used, these SlipKV may be evicted too early, disrupting reasoning
process. Moreover, CrystalKV tends to receive intermittent atten-
tion. If SnapKV is used, it may be misclassified as SlipKV and evicted
during attention gaps.
These limitations motivate us to distinguish CrystalKV and
SlipKV from a temporal and evolutionary perspective. Specifically,
we initialize all newly arrived KV entries as PotentialKV. During
think stage, each PotentialKV may evolve into CrystalKV (and be
retained) or SlipKV (and be evicted). However, due to the presence
of attention gaps, the transition from PotentialKV to CrystalKV is
not intuitive. Shifting our focus, however, we identify a qualitative
criterion for detecting the transition from PotentialKV to SlipKV:
at the current time step, if a PotentialKV is attended infrequently
in its past history and remains unattended for a sufficiently long
recent period, it can be safely classified as SlipKV and evicted. By
Algorithm 1Attention-Based LRFU Compression (per Head)
Inputs: B: KV budget, p: top-𝑝 threshold for hit mask, 𝜆: CRF decay rate, t:
current time step, K/V: key/value cache (PotentialKV), Qlast: last query
state,CRF: CRF score for all KV,𝜏: last update time for all KV.
Outputs:K ′/V′: compressed key/value cache.
1:if𝐿𝑒𝑛(K)<Bthen
2:returnK,V
/* Step 1: 0-1 Hit KV Mask via Top-𝑝Attention */
3:s←AttnScores(Q last,K)
4:M ℎ𝑖𝑡 ←GetHitMask(s,p)
/* Step 2: Update and Record Hit KV CRF */
5:CRF 𝑡𝑒𝑚𝑝 [Mℎ𝑖𝑡 ]←𝜆 𝑡−𝜏[M ℎ𝑖𝑡 ] ·CRF[M ℎ𝑖𝑡 ]+1
6:𝜏[M ℎ𝑖𝑡 ]=t
7:CRF[M ℎ𝑖𝑡 ]←CRF 𝑡𝑒𝑚𝑝 [Mℎ𝑖𝑡 ]
/* Step 3: Update Miss KV CRF without Record */
8:M 𝑚𝑖𝑠𝑠 ←1−M ℎ𝑖𝑡
9:CRF 𝑡𝑒𝑚𝑝 [M𝑚𝑖𝑠𝑠 ]←𝜆 𝑡−𝜏[M 𝑚𝑖𝑠𝑠 ] ·CRF[M 𝑚𝑖𝑠𝑠 ]
/* Step 4: SlipKV Identification and Eviction */
10:Retain←TopK(CRF 𝑡𝑒𝑚𝑝 ,B).index
11:CRF←Gather(CRF,Retain),𝜏←Gather(𝜏,Retain)
12:K ′ ←Gather(K,Retain),V ′ ←Gather(V,Retain)
13:returnK ′,V ′
prioritizing the identification and eviction of SlipKV, the cache is
left with only CrystalKV, ensuring correct answers. Fortunately,
when attempting to quantify this criterion, we find that it closely
aligns with the classical cache replacement policy: Least Recently
Frequently Used (LRFU) [18].
5.2 Attention Based LRFU
Along the temporal dimension, LRFU integrates both access (hit)
frequency and recency to assess the importance of each entry using
the Combined Recency and Frequency (CRF) score. Given current
time 𝑡, for an entry 𝑒𝑖 with historical hit timestamps 𝑡1, 𝑡2, . . . , 𝑡𝑘
and decay factor𝜆∈ [0,1], the CRF score is defined as:
𝐶𝑅𝐹(𝑒 𝑖 )=
𝑘∑︁
𝑗=1
𝜆𝑡−𝑡 𝑗 .
Intuitively, when an entry 𝑒𝑖 is hit at time 𝑡 𝑗 , it receives a reward of
𝜆𝑡 𝑗 −𝑡 𝑗 = 1, which then decays over time to 𝜆𝑡−𝑡 𝑗 at a later time 𝑡.
When the cache is full, entries with smaller CRF scores are evicted.
The CRF-based eviction policy precisely models the transition
from PotentialKV to SlipKV. However, we cannot observe hits as in
CPU instructions directly. Thus, we perform Top-𝑝 nucleus sam-
pling based on attention scores, selecting the smallest set of KV
whose cumulative attention scores exceed a threshold 𝑝 as hit KV
entries. To reduce memory overhead, we further implement incre-
mental CRF tracking. For each 𝑒𝑖, we only store its last hit time 𝑡𝑖,
and the CRF score is updated at any time𝑡as:
𝐶𝑅𝐹 𝑖 (𝑡)=𝜆 𝑡−𝑡 𝑖 ·𝐶𝑅𝐹 𝑖 (𝑡𝑖 ) +1 {hit at𝑡}
Algorithm 1 outlines the KV cache compression per head. First,
we apply Top-𝑝 nucleus sampling on attention scores to derive a
boolean matrix Mhit, indicating which entries are hits (lines 3–4).
For hits, we update and record both CRF and last-hit time (lines 5-7).

<!-- page 5 -->

Algorithm 2Adaptive Budget Allocation (Layer & Head)
Inputs: B𝑡𝑜𝑡𝑎𝑙 : total KV budget for LLM, L: layer num of LLM, H: head num
per layer, CRF𝑖,𝑗 : aggregated CRF scores of head j(<H ) in layer i(<L ),
B𝑖,𝑗 : current KV budget for head j(<H) in layer i(<L)
Outputs:B ′
𝑖,𝑗 : new KV budget for head j(<H) in layer i(<L)
/* Step 1: Budget Utilization per Layer */
1:𝜂 𝑖 ← Íℎ
𝑗=1 CRF𝑖,𝑗 / Íℎ
𝑗=1 B𝑖,𝑗
/* Step 2: Layer-wise Budget Allocation */
2:B ′
𝑖 ←B 𝑡𝑜𝑡𝑎𝑙 · (𝜂𝑖 / Í𝑙
𝑖=1 𝜂𝑖 )
/* Step 3: Budget Utilization per Head */
3:𝜂 𝑖,𝑗 ←CRF 𝑖,𝑗 /B 𝑖,𝑗
/* Step 4: Head-wise Budget Allocation */
4:B ′
𝑖,𝑗 ←B ′
𝑖 · (𝜂𝑖,𝑗 / Íℎ
𝑗=1 𝜂𝑖,𝑗 )
5:returnB ′
𝑖,𝑗
For misses, we only update CRF (lines 8-9). Finally, entries with the
lowest CRF are classified as SlipKV and evicted accordingly (lines
10–13). Throughout the think stage, it gradually compresses the
KV cache by dropping SlipKV, ensuring only CrystalKV remains
before entering the answer stage.
Notably, when𝜆= 0, the algorithm degenerates to Least Recently
Used (LRU) [3], focusing only on recency. When 𝜆= 1, it becomes
Least Frequently Used (LFU) [ 27], focusing purely on frequency.
Therefore, we constrain 𝛼<𝜆<𝛽 . 𝛼<𝜆 prevents excessive
reward decay of CrystalKV during attention gaps, avoiding its
misclassification as SlipKV. 𝜆<𝛽 ensures that long-lived SlipKV
entries are not over-preserved, preventing occupation of valuable
cache space. The choice of𝛼and𝛽is further discussed in Sec. 6.4.
5.3 Adaptive Budget Allocation
Under a uniform and fixed budget allocation, we observe that
after Algorithm 1, certain layers/heads end with relatively high
CRF scores, indicating that the tight budget forced them to evict
CrystalKV entries important for the final answer. Conversely, some
layers or heads show consistently low CRF scores, suggesting little
or no CrystalKV in those regions. This reveals that different lay-
ers/heads contribute heterogeneously to the reasoning process. To
capture the relative importance of each layer/head under a limited
budget, we define a metric of cache utilization:
𝜂=
Í CRFÍ Budget
A higher 𝜂 indicates more potential CrystalKV entries under a
tighter budget, implying the layer/head is more important and
should be allocated more budget.
Algorithm 2 outlines the complete adaptive budget allocation
process. First, it computes layer-wise utilization by aggregating
CRF scores and budgets across all heads (line 1). Then, it normalizes
utilization across layers and adjusts their budget proportionally
(line 2). The same procedure is applied to heads within each layer
(lines 3–4). This algorithm is periodically triggered. By amplify-
ing the impact of critical layers/heads, adaptive budget allocation
significantly improves cache space efficiency.
6 Experiment
6.1 Experiment Setup
Models and Datasets. We evaluate three DeepSeek-R1–distilled
open-source models (Llama-8B, Qwen-14B, and Qwen-32B) [12],
chosen for their strong CoT reasoning performance. Our exper-
iments span two challenging reasoning domains: programming
using the CodeForces benchmark (10K competitive programming
problems up to 2025) [23] and mathematics using MATH-500 (ad-
vanced competition-level math problems) [13].
Baselines. We compare Crystal-KV with five representative KV
compression baselines: R-KV, RaaS, SnapKV, StreamingLLM, and
H2O. SnapKV, StreamingLLM, and H2O are long-context genera-
tion (LCG) compress methods that follow the token-uniform prin-
ciple, while R-KV and RaaS are CoT-oriented compressors but still
treat CoT as a special case of LCG. Since all these baselines adopt
static, uniform, and fixed budget allocation, we introduce Crystal-
KV.Lite, a variant of our method where dynamic budget allocation
is disabled, to ensure a fair comparison. Moreover, we include Ada-
SnapKV, which augments SnapKV with dynamic budget realloca-
tion, enabling a direct comparison with the full-featured Crystal-KV.
It is worth noting that due to inherent limitations in their com-
pression strategies, other baselines cannot be adapted to support
dynamic reallocation. Finally, we include FullKV, which retains the
full cache and serves as the gold standard.
Evaluation Settings. Due to the limited ability of LLMs on chal-
lenging coding tasks, we restrict our CodeForces evaluation to prob-
lems with difficulty ratings below 1500 [8, 12], to avoid accuracy
saturation or complete failure, ensuring headroom for performance
improvements. We use the recommended sampling temperature
(0.6) and top-p (0.95) [ 12]. For each problem, we generate 𝑘= 8
answers, and report the average correctness scores, computed as
1
𝑘
Í𝑘
𝑖=1 𝑝𝑖, where 𝑝𝑖 denotes the correctness of the 𝑖-th answer. This
procedure yields more reliable accuracy estimates. All experiments
are conducted on a workstation equipped with three NVIDIA RTX
PRO 6000 Blackwell GPUs.
6.2 Accuracy Comparison
Fig. 4 reports the accuracy of all methods across different tasks and
KV-budgets. Budget Ratio is defined relative to the average per-
question KV usage of FullKV. We can distill four key observations:
(i) Under the same KV budget, Crystal-KV consistently outperforms
StreamingLLM, H2O, R-KV, Fixed-SnapKV, Ada-SnapKV, and RaaS.
On code tasks, the average accuracy gains are 18.98%, 14.36%, 12.46%,
9.04%, 7.91%, and 7.30%. On math tasks, the average accuracy gains
are 23.87%, 23.05%, 7.97%, 11.39%, 9.13%, and 14.65%. (ii) Focusing
more attention on CrystalKV during the answer stage can turn
previously wrong answers into correct ones. On code tasks, a peak
appears where Crystal-KV even surpasses FullKV. Before this peak,
the budget is too small to retain all CrystalKV, so increasing budget
improves accuracy. After this point, the KV budget exceeds the total
amount of CrystalKV, causing SlipKV entries to be stored as well,
so the accuracy gradually converges back to the FullKV level. (iii)
Adaptive Budget Allocation further improves performance under
very small budgets by enhancing space utilization and amplifying
critical layers and heads. (iv) Accuracy gains on math are smaller
than on code under the same budget. This is because math tasks

<!-- page 6 -->

Figure 4: Accuracy Comparison on Math and Code Tasks across Three Reasoning LLMs
Table I: Comparison of Memory Consumption and Inference Throughput
Gen.Len. Method Budget HBM Saving (%) Batch Throughput (tok/s) Tokens Gen. Dec.Time (s)
8K-levelFullKV - - 1 53.14 9,350 175.95- - 51 (max) 458.67 476,850 (max) 1039.63
Crystal-KVFixed-1024 89.05 1 59.58 9,350 156.93Fixed-1024 89.05 379 (max)2297.37 3,543,650 (max)1542.48Ratio-10%-93590.00 412 (max)2478.39 3,852,200 (max)1554.31
16K-levelFullKV - - 1 41.99 18,700 445.34- - 25 (max) 174.95 467,500 (max) 2672.22
Crystal-KVFixed-1024 94.52 1 56.90 18,700 328.65Fixed-1024 94.52 379 (max)2141.44 7,087,300 (max)3309.59Ratio-10%-187090.00 228 (max)1332.69 4,263,600 (max)3199.25
contain more dispersed key information, requiring larger CrystalKV
budgets to match the performance level.
6.3 Efficient Memory Saving and Computation
Under the FullKV strategy, GPU memory usage grows linearly with
the reasoning length, making long-sequence reasoning prone to
Out of Memory (OOM), and the cost of computation increases. In
contrast, Crystal-KV enforces a fixed memory budget, significantly
reducing both memory usage and computation cost.
Table I compares Crystal-KV with the FullKV baseline on long-
sequence reasoning using DeepSeek-R1-Distill-Llama-8B with a
total HBM capacity of 288 GB. The average reasoning lengths are
9,350 and 18,700 tokens (8K and 16K levels), consistent with practi-
cal CodeForces usage. We evaluate both fixed and ratio-based bud-
gets to accommodate both strict and flexible memory constraints.
Existing KV-compression baselines are not considered here because
they cannot attain FullKV-level accuracy under strict budgets in
long-sequence reasoning. At 8K level, Crystal-KV accelerates single-
batch inference (response time) by 6.44 token/s over FullKV. Given
the total GPU HBM capacity, FullKV can support at most 51 parallel
batches. In contrast, Crystal-KV can support 379 parallel batches
under a fixed budget of 1024 KV entries per head, and 412 batches
under a 10% ratioed budget (a setting that already achieves loss-
less compression). The corresponding throughput is improved by
5.01× and 5.40× over FullKV, and maximum total number of tokens
that can be processed concurrently reaches 7.43× and 8.07× that of
FullKV. When the reasoning problem becomes more difficult and
the sequence length increases to the 16K level, the advantage of
Crystal-KV becomes even more pronounced. The overall through-
put is boosted to 12.24× and 7.62×, and the maximum total number
of tokens increases to 15.16× and 9.12×. This indicates that, in sce-
narios requiring much longer reasoning chains (i.e., more complex
reasoning tasks), Crystal-KV has substantial potential to handle
problems that are infeasible for FullKV, owing to its ability to handle
much longer reasoning chains.
In summary, Crystal-KV substantially reduces memory usage by
an average of 90.89%, improves throughput by 7.57× on average,

<!-- page 7 -->

λ （Decay Rate）
Top-p (threshold)
Accuracy Heatmap of (𝛌𝛌, top-p) on Code/Math Performance (32B)
0.9
0.8
0.7
0.6
0.5
Math
0.9
0.8
0.7
0.6
0.5
Code
Normalization Accuracy (0-1)
0.0 0.25 0.5 0.75 1.0
Figure 5: Effect of Decay Rate and Sampling Threshold
and achieves up to 1.24× speedup in user-level response latency,
all while maintaining lossless compression.
6.4 More Discussions
Fig. 5 shows accuracy heatmaps of Crystal-KV on Codeforces and
MATH500 under different sampling thresholds (𝑡𝑜𝑝-𝑝∈ { 0.5,0 .6,
0.7,0 .8,0 .9) and decay factors (𝜆∈ { 0.0, 0.1, . . . , 1.0}). Both tasks
exhibit unified patterns. For 𝜆, accuracy peaks at [0.6, 0.7] for code
and [0.5, 0.6] for math. When 𝜆 is too small, historical rewards for
CrystalKV decay excessively during attention gaps, causing misclas-
sification as SlipKV. When𝜆 is too large, overemphasis on historical
hits delays the eviction of long-lived SlipKV and occupies space
needed for CrystalKV. Furthermore, accuracy for 𝜆∈ [ 0.8, 1.0]
exceeds that for [0, 0.2], suggesting that most CrystalKV entries un-
dergo long attention gaps while SlipKV entries have short lifetimes.
For 𝑡𝑜𝑝-𝑝 , higher thresholds (e.g., 0.9) better preserve CrystalKV
and prevent the loss of critical context, whereas lower thresholds
(e.g., 0.5) produce volatile attention patterns and make 𝜆 more sen-
sitive. Thus, we recommend 𝑡𝑜𝑝 -𝑝= 0.9for robust performance
and set 𝜆 within a moderate range, approximately [0.5, 0.7], corre-
sponding to the(𝛼, 𝛽)bounds discussed in Sec. 5.2.
7 Conclusion
In this work, we present Crystal-KV, an efficient CoT KV-cache
management framework built on an answer-first principle that
prioritizes think-KV essential to the final answer. This perspective
redefines KV management for reasoning-oriented LLMs during long
think stage. Experiments show that Crystal-KV delivers substantial
memory savings and latency improvements while maintaining or
even improving answer accuracy across diverse reasoning tasks.
These findings establish answer-centric cache management as a
promising paradigm for efficient large-scale reasoning. In future
work, we will formally model CrystalKV and SlipKV to quantify the
capability boundary of KV-cache compression for CoT reasoning.
References
[1] Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J Nair, Ilya Solov-
eychik, and Purushotham Kamath. 2024. Keyformer: Kv cache reduction through
key tokens selection for efficient generative inference.2024 Proceedings of Ma-
chine Learning and Systems (MLsys)(2024).
[2] Daman Arora and Andrea Zanette. 2025. Training language models to reason
efficiently.arXiv preprint arXiv:2502.04463(2025).
[3] Laszlo A. Belady. 1966. A study of replacement algorithms for a virtual-storage
computer.IBM Systems journal5, 2 (1966), 78–101.
[4] Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng
Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar,
Abedelkadir Asi, and Junjie Hu. 2025. R-KV: Redundancy-aware KV Cache
Compression for Training-Free Reasoning Models Acceleration. InAdvances in
Neural Information Processing Systems (NeurIPS 2025).
[5] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming
Lu, Wayne Xiong, Yue Dong, Junjie Hu, et al . 2024. Pyramidkv: Dynamic kv
cache compression based on pyramidal information funneling.arXiv preprint
arXiv:2406.02069(2024).
[6] Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping
Chen, Yi Zheng, Baoxing Huai, and Gang Chen. 2025. IMPRESS: An Importance-
Informed Multi-Tier Prefix KV Storage System for Large Language Model Infer-
ence. In23rd USENIX Conference on File and Storage Technologies (FAST 25).
[7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen
Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multi-
modality, long context, and next generation agentic capabilities.arXiv preprint
arXiv:2507.06261(2025).
[8] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. https://huggingface.co/deepseek-ai/DeepSeek-R1.
[9] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S. Kevin Zhou. 2025. Ada-KV:
Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM
Inference. InAdvances in Neural Information Processing Systems (NeurIPS 25).
[10] Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. 2025. Deep think
with confidence.arXiv preprint arXiv:2508.15260(2025).
[11] Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo
Deng, Xingkun Yang, Zhou Yu, and Pengfei Zuo. 2024. Cost-Efficient large
language model serving for multi-turn conversations with CachedAttention. In
2024 USENIX Annual Technical Conference (USENIX ATC 24). 111–126.
[12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu,
Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1 incen-
tivizes reasoning in llms through reinforcement learning.Nature645, 8081 (2025),
633–638.
[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric
Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem
solving with the math dataset.arXiv preprint arXiv:2103.03874(2021).
[14] Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le,
Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A Saurous.
2023. Training chain-of-thought via latent-variable inference. InAdvances in
Neural Information Processing Systems (NeurIPS).
[15] Junhao Hu, Wenrui Huang, Weidong Wang, Zhenwen Li, Tiancheng Hu, Zhixia
Liu, Xusheng Chen, Tao Xie, and Yizhou Shan. 2025. RaaS: Reasoning-Aware
Attention Sparsity for Efficient LLM Reasoning. InFindings of the Association for
Computational Linguistics (ACL 2025).
[16] Minsu Kim, Seongmin Hong, RyeoWook Ko, Soongyu Choi, Hunjong Lee, Junsoo
Kim, Joo-Young Kim, and Jongse Park. 2025. Oaken: Fast and Efficient LLM
Serving with Online-Offline Hybrid KV Cache Quantization. InProceedings of
the 52nd Annual International Symposium on Computer Architecture (ISCA 25).
[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles (SOSP 23).
[18] Donghee Lee, Jongmoo Choi, Jong-Hun Kim, Sam H Noh, Sang Lyul Min, Yookun
Cho, and Chong Sang Kim. 2001. LRFU: A spectrum of policies that subsumes
the least recently used and least frequently used policies.IEEE transactions on
Computers50, 12 (2001), 1352–1361.
[19] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. InfiniGen:
Efficient generative inference of large language models with dynamic KV cache
management. In18th USENIX Symposium on Operating Systems Design and Im-
plementation (OSDI 24). 155–172.
[20] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli,
Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm
knows what you are looking for before generation.Advances in Neural Informa-
tion Processing Systems (NeurIPS)37 (2024), 22947–22970.
[21] OpenAI. 2025.Introducing GPT-5. August 7 2025.
[22] Xiurui Pan, Endian Li, Qiao Li, Shengwen Liang, Yizhou Shan, Ke Zhou, Yingwei
Luo, Xiaolin Wang, and Jie Zhang. 2025. InstAttention: In-Storage Attention

<!-- page 8 -->

Offloading for Cost-Effective Long-Context LLM Inference. In2025 IEEE Interna-
tional Symposium on High Performance Computer Architecture (HPCA 25). IEEE,
1510–1525.
[23] Guilherme Penedo, Anton Lozhkov, Hynek Kydlíček, Loubna Ben Allal, Edward
Beeching, Agustín Piqueres Lajarín, Quentin Gallouédec, Nathan Habib, Lewis
Tunstall, and Leandro von Werra. 2025. CodeForces. https://huggingface.co/
datasets/open-r1/codeforces.
[24] Derrick Quinn, E Ezgi Yücel, Jinkwon Kim, José F Martínez, and Mohammad
Alian. 2025. LongSight: Compute-Enabled Memory to Accelerate Large-Context
LLMs via Sparse Attention. InProceedings of the 58th IEEE/ACM International
Symposium on Microarchitecture (Micro 25). 34–48.
[25] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi
Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, et al. 2025. Stop
Overthinking: A Survey on Efficient Reasoning for Large Language Models.arXiv
preprint arXiv:2503.16419(2025).
[26] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning
in large language models.Advances in neural information processing systems35
(2022), 24824–24837.
[27] Wikipedia contributors. 2025. Least frequently used.Wikipedia, The Free Ency-
clopedia. https://en.wikipedia.org/wiki/Least_frequently_used.
[28] Tianhua Xia and Sai Qian Zhang. 2025. Kelle: Co-design KV Caching and eDRAM
for Efficient LLM Serving in Edge Computing. InProceedings of the 58th IEEE/ACM
International Symposium on Microarchitecture (Micro 25). 18–33.
[29] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024.
Efficient Streaming Language Models with Attention Sinks. InProceedings of the
12th International Conference on Learning Representations (ICLR 24).
[30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[31] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei
Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2025. Dynamic early exit in
reasoning models.arXiv preprint arXiv:2504.15895(2025).
[32] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast large language
model serving for RAG with cached knowledge fusion. InProceedings of the
Twentieth European Conference on Computer Systems (EuroSys 25). 94–109.
[33] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue.
2025. Demystifying long chain-of-thought reasoning in llms.arXiv preprint
arXiv:2502.03373(2025).
[34] Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. 2024.
Chain of preference optimization: Improving chain-of-thought reasoning in llms.
Advances in Neural Information Processing Systems (NeurIPS)37 (2024), 333–356.
[35] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi
Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. 2023. H2o:
Heavy-hitter oracle for efficient generative inference of large language models.
Advances in Neural Information Processing Systems (NeurIPS 23)36 (2023), 34661–
34710.
[36] Youpeng Zhao, Di Wu, and Jun Wang. 2024. Alisa: Accelerating large language
model inference via sparsity-aware kv caching. In2024 ACM/IEEE 51st Annual
International Symposium on Computer Architecture (ISCA 24). IEEE, 1005–1017.
[37] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu,
Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating prefill and decoding for
goodput-optimized large language model serving. In18th USENIX Symposium on
Operating Systems Design and Implementation (OSDI 24). 193–210.
