# references/93_qcfuse_query_centric_cache_fusion_for_efficient_rag_inference.pdf

<!-- page 1 -->

QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference
Jianxin Yan
Zhejiang University
Hangzhou, China
22521296@zju.edu.cn
Zeheng Qian
The University of Sydney
Sydney, Australia
zqia0047@uni.sydney.edu.au
Wangze Ni
Zhejiang University
Hangzhou, China
niwangze@zju.edu.cn
Zhitao Shen
Ant Group
Shanghai, Cina
zhitao.szt@antgroup.com
Zhiping Wang
Ant Group
Shanghai, China
laoman.wzp@antgroup.com
Haoyang Li
PolyU
Hong Kong, China
haoyang-comp.li@polyu.edu.hk
Jia Zhu
Zhejiang Normal University
Jinhua, China
jiazhu@zjnu.edu.cn
Lei Chen
HKUST(GZ) & HKUST
Guangzhou, China
leichen@cse.ust.hk
Kui Ren
Zhejiang University
Hangzhou, China
kuiren@zju.edu.cn
ABSTRACT
Cache fusion accelerates generation process of LLMs equipped
with RAG through KV caching and selective token recomputation,
thereby reducing computational costs and improving efficiency.
However, existing methods primarily rely on local perspectives
for token selection and lack global awareness from the user query.
Utilizing this global awareness is challenging due to the high cost
of obtaining context-aware query representations and the strict
pipeline constraints required for efficient attention analysis. Thus,
this demonstration introduces QCFuse, an innovative KV cache
fusion system centered on the user query. QCFuse leverages se-
mantic summary anchors to enhance query representations and
selectively recomputes query-related tokens to improve accuracy,
updating tokens based on the attention distribution of the most
critical Transformer layer to preserve the high efficiency of the
pipeline structure. Evaluations on real-world datasets demonstrate
that QCFuse significantly improves the response efficiency of LLMs
by 40% while maintaining equivalent accuracy compared to cur-
rent methods. Additionally, in certain scenarios, QCFuse achieves
an attention denoising effect that yields higher response accuracy,
demonstrating substantial potential in the optimization of LLM
inference.
PVLDB Reference Format:
Jianxin Yan, Zeheng Qian, Wangze Ni, Zhitao Shen, Zhiping Wang,
Haoyang Li, Jia Zhu, Lei Chen, and Kui Ren. QCFuse: Query-Centric Cache
Fusion for Efficient RAG Inference. PVLDB, 14(1): XXX-XXX, 2020.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at
https://github.com/uYanJX/QCFuse-Demo.
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX
Figure 1: Comparison among Full Computation, Full Reuse,
and Cache Fusion.
1 INTRODUCTION
LLMs equipped with RAG are standard for enterprise knowledge-
base question answering and content generation, as they mitigate
hallucinations and support real-time knowledge updates. In high-
concurrency production environments, however, RAG generation
remains heavily bottlenecked. Although context chunks retrieved
for different queries can overlap by over 70%, strict prefix-matching
policies prevent traditional prefix caching from reusing them. LLMs
are thus forced to fully prefill redundant contexts. As a result, time
to first token (TTFT) grows quadratically with context length, wast-
ing immense computational resources.
As illustrated in Figure 1, cache fusion has emerged as a pri-
mary optimization strategy. These approaches merge historical KV
caches and selectively recompute a subset of tokens. This reduces
costs while preserving accuracy comparable to full computation.
Prior work determines token recomputation targets in various ways:
CacheBlend [5] uses numerical deviations of KV tokens, whereas
EPIC [1] statically recomputes a fixed ratio. Both methods effec-
tively reduce computational overhead.
A central limitation of existing methods is their lack of global
awareness regarding the user query. By relying primarily on local
cues, such as static positional heuristics or first-layer KV deviations,
they overlook the query’s role as the primary driver of the genera-
tion process. Operating on intermediate representations without
considering the original request leads to suboptimal resource allo-
cation. Computational budget is often spent on irrelevant tokens
arXiv:2604.08585v1  [cs.DB]  30 Mar 2026

<!-- page 2 -->

Figure 2: Architecture of the QCFuse System.
while critical ones are ignored, causing significant accuracy drops
under aggressive acceleration.
Using the query’s attention distribution over context tokens as
a selection criterion is an intuitive alternative. Tokens with high
query attention typically exert the greatest influence on genera-
tion quality. Realizing this within cache fusion systems, however,
presents two primary challenges.
• The first challenge concerns obtaining context-aware query
representations at minimal cost. A naive query-only forward-
ing process, which omits the contextual KV cache, yields un-
grounded representations and unreliable attention distributions.
Conversely, injecting the complete context KV cache during
query forwarding disrupts the pipelined execution of cache fu-
sion systems. To handle very long contexts, systems store full KV
caches on SSDs and process them layer by layer. While the GPU
recomputes one layer, the pipeline prefetches the next layer’s
KV from the SSD. Requiring the full context KV to be materi-
alized beforehand forces the system to wait for all data to load,
reducing the pipeline to sequential execution and eliminating effi-
ciency gains. Acquiring context-enhanced query representations
without disrupting this pipeline is thus essential.
• The second challenge involves efficient attention analysis
within these pipeline constraints. Computing query attention
across all Transformer layers, as done in ProphetKV [3], blocks
the pipeline due to cross-layer dependencies. Relying solely on
the final layer, as in FusionRAG [ 2], provides an incomplete
semantic view. The system must instead identify a single pipeline-
friendly layer whose attention distribution serves as a reliable
proxy for global token importance.
We address these challenges through two main technical con-
tributions.First, we propose anchor-based lightweight query
probing.By analyzing token key-norm magnitudes, we extract
anchor tokens from each precomputed context chunk to serve
as compressed semantic summaries. These tokens are injected as
lightweight prefixes during query forwarding to produce context-
enhanced query representations. Unlike previous context-free for-
warding methods, this approach maintains pipeline efficiency.
Second, we achieve semantic localization via critical-layer
attention profiling.Empirical findings suggest that middle layers
offer superior semantic localization. We therefore analyze query
attention at a single critical middle layer. This mechanism avoids
the pipeline stalls associated with cross-layer dependencies and the
incomplete views of last-layer approaches, successfully balancing
accuracy and system efficiency.
Building on these mechanisms, this demonstration paper presents
QCFuse, an efficient query-centric cache fusion system implemented
on SGLang. The system features a high-performance sparse-attention
Triton kernel for discrete token recomputation. Evaluations across
multiple LLMs and multi-hop QA benchmarks demonstrate up to a
2× speedup of TTFT over full computation and a 40% latency reduc-
tion compared to existing cache fusion baselines, with matching or
improved generation quality. We demonstrate how QCFuse inte-
grates seamlessly into enterprise knowledge assistants to deliver
near-real-time answers over massive document collections.
2 RELATED WORK
KV and Prefix Caching.KV and prefix caching [ 6] accelerates
large language model inference by reusing historical Key/Value
matrices, avoiding redundant computation and reducing latency.
Frameworks such as SGLang [ 6] implement this via prefix-tree
matching. While highly effective for constant prefixes, these mech-
anisms struggle with the dynamic context assemblies typical of
RAG applications, where retrieval chunks are frequently reordered
or inserted into the middle of the prompt.
Location-independent Caching and Selective Recomputation.
To enable cache reuse in dynamic contexts, fusion methods con-
catenate discrete KV chunks and selectively recompute tokens that
experience semantic shifts. CacheBlend [5] identifies these tokens
using first-layer KV deviations, EPIC [ 1] defaults to a static ra-
tio at the sequence start, and KVShare [4] scales KV deviation by
initial attention weights. More recently, ProphetKV [3] and Fusion-
RAG [2] proposed query-guided token selection, yet both encounter
pipeline synchronization issues. ProphetKV evaluates all layers con-
currently, ignoring stage-wise scheduling constraints. FusionRAG
computes an initial query pass without context, relying on last-
layer attention that provides insufficient semantic grounding. We
emulate these two query-centric baselines in our evaluations as
QCAll and QCLast.
2

<!-- page 3 -->

3 SYSTEM
3.1 System Workflow
As shown in Figure 2, the workflow of QCFuse consists of four
highly optimized phases:
Phase 1: Offline Pre-computation and Anchor Extraction.
Before query processing, the system pre-computes the KV cache
for all context chunks in the RAG database. During this stage, the
full pre-computed KV cache is stored persistently on the SSD. Con-
currently, a minor fraction of tokens with the highest key-norm
values are copied and extracted to serve as a compressed semantic
summary. Due to their minimal storage footprint, these anchor KV
tokens are stored directly in the CPU memory to minimize latency.
This offline procedure completely avoids any time overhead during
the online generation process.
Phase 2: RAG Retrieval and Context-aware Query Probing.
When a user query enters the scheduler of the SGLang, the system
performs a standard RAG retrieval process to fetch relevant chunks.
Instead of conducting a context-free query forwarding process, the
system utilizes the CPU-resident KV token anchors corresponding
to the retrieved chunks. These anchors are injected into the GPU
as lightweight prefixes alongside the query, endowing the initial
query representations with profound contextual grounding without
initiating massive data transfers from the SSD.
Phase 3: Critical-layer Attention Analysis.Following the query
forwarding, the system exclusively loads the key (K) cache of the
most critical middle layer from the SSD. It then performs an atten-
tion analysis between the query (Q) cache of the user request and
the K cache of this specific critical layer. The resulting attention
weights dictate the Top-𝑁 context tokens to which the query re-
lates most strongly, thereby yielding the essential indices for token
reconstruction.
Phase 4: Pipelined Cache Reconstruction and Response Gen-
eration.Guided by these Top- 𝑁 indices, the GPU initiates discrete
token recomputation, which adheres to a strictly pipelined archi-
tecture: while the GPU reconstructs the selected tokens for layer 𝑖,
the pipeline simultaneously prefetches the KV cache for layer 𝑖+ 1
from the SSD. Ultimately, the updated and contextually enriched
matrix set of KV tokens is fed into the native decoding engine of
the SGLang framework for response generation with exceptionally
low latency.
3.2 System Implementation
To implement QCFuse, we modify the process of the SGLang frame-
work at a minimal scale for the following key components:
Location-independent Cache Indexing Module.This module
extends the native RadixCache of the SGLang framework to create a
hash indexing table for context chunks. The KV tokens for each con-
text chunk are independently cached. This configuration supports
precise content-based hash searches while remaining intrinsically
compatible with existing prefix caching logic.
Query-related Token Selector.This selector is executed during
Phase 3. By forwarding the query with anchor prefixes and ex-
clusively analyzing the attention of the critical middle layer, this
module rapidly estimates the attention distribution of the query
across all contextual tokens. After averaging and ranking the multi-
head attention weights, the Top-𝑁 tokens are formally selected as
0.1 0.2 0.3 0.4 0.5 0.6
0.3
0.35
0.4
0.45
better
TTFT (s)
ROUGE-L
Averaged Performance Comparison
QCFuse QCAll QCLast
CacheBlend EPIC KVShare
Random Full Reuse Full Comp
Figure 3: Average ROUGE-L vs. TTFT of existing methods un-
der different recomputation ratios (0.1–0.5), averaged across
models and datasets.
the corresponding set for recomputation. The recomputation ratio
can be dynamically configured based on functional requirements
for accuracy and performance.
Location-aware Sparse Attention Kernel.This kernel is a cus-
tomized location-aware attention operator developed via Triton,
which is completely compatible with the operator invocation in-
terface of the SGLang framework. The kernel supports receiving
a table of discrete token indices alongside their corresponding ab-
solute locations to construct attention masks that faithfully follow
causal constraints. This robust design guarantees absolute semantic
correctness during the token recomputation process.
3.3 Core Technique and Evaluation
We evaluate QCFuse on one A100 GPU (80GB) with tested models
including Llama3.1-8B, Qwen3-8B, and Mistral-v0.3-7B. We apply
three question-answering datasets: Musique, 2WikiMQA, and Hot-
potQA, aiming to simulate real RAG features.
Performance of the Query-related Global Assessing Strategy.
Our testing indicates that QCFuse achieves a ROUGE-L score 2.3
to 3.5 points higher than CacheBlend [5]. At a 40% recomputation
ratio, QCFuse matches the accuracy of full computation. On the Hot-
potQA dataset, QCFuse is 0.8 points better than full computation
because it removes attention interactions with irrelevant tokens.
This directly proves the attention denoising effect and improves
the overall accuracy. Furthermore, QCFuse achieves accuracy com-
parable to QCAll with lower latency, while delivering substantially
higher accuracy than QCLast.
Effect of Sparse Attention Kernels.Our sparse attention kernel
only accelerates the partial computation phase. After applying this
kernel across all tested schemes to ensure a fair comparison, as
shown in Figure 3, QCFuse is two times faster than full computa-
tion. When QCFuse reaches the same accuracy as the baseline, it
reduces the delay by an extra 40%, which perfectly meets the strict
requirements of fast RAG tasks.
3

<!-- page 4 -->

Figure 4: Detailed Interface of QCFuse for the Demonstration of KV Recomputation.
4 DEMONSTRATION
The demonstration system of QCFuse is developed using a front-
backend separation architecture based on React and FastAPI. The
frontend utilizes React-based charting libraries for dynamic visu-
alizations, while the backend serves via the SGLang framework
integrated with the QCFuse extension to facilitate real-time com-
parisons with baseline solutions. The demonstration interactively
comprises two consecutive scenarios that progressively highlight
the core advantages of the system, allowing users to input cus-
tomized queries, adjust parameters, and verify results.
Scenario 1: Conversational Interaction between User and
LLMs.This scenario showcases a standard chat interface of LLMs.
However, before entering a query, the user can review all the context
chunks stored in the RAG database, along with their summaries,
hash index values, and other related information displayed on the
left side of the interface. If no relevant chunks are available, the
user can upload custom context files to the system. These files
can be in standard contextual formats (e.g., .txt, .pdf, .csv) and
are automatically pre-computed in real time. Subsequently, the
user can select a preferred LLM and cache mode to execute the
generation process of RAG. This process supports different LLMs
and various cache modes, including QCFuse and other baselines
such as CacheBlend, thereby facilitating a direct comparison of user
experience between QCFuse and alternative methods.
Scenario 2: Token Retrieval and Update.Figure 4 demonstrates
the core capabilities of QCFuse: independent chunk-level KV caching
and location-independent token reuse. After receiving the response
of the LLM on the chat interface, the user can click the "View De-
tails" button at the bottom-right corner of each response bubble to
inspect the details of the generation process.
• Query-related context chunks retrieved from the RAG database
are labeled with a content summary, relevance scores, and other
information. These are exhibited on the left side of the interface
as the base materials for the retrieval of KV tokens.
• After clicking the "Process" button, the user can observe changes
in partially displayed KV tokens in the middle of the interface.
The query-related KV tokens (i.e., those with high attention
scores) are first uniformly highlighted in orange based on the
critical layer. Subsequently, during the recomputation process,
they are marked in green layer by layer as they are updated.
• The basic parameters (e.g., the number of computed tokens and
the time cost) of the currently selected recomputation method
and full computation are displayed on the right side of the inter-
face, along with the process timeline of the system and the final
answer generated by the LLM. The user can also click the "View
Detailed Metrics" button to monitor other system parameters,
such as cache storage and memory usage.
5 CONCLUSION
We innovatively develop QCFuse, a query-centric KV cache fusion
system that selectively recomputes query-related tokens. This sys-
tem optimizes the generation process of LLMs equipped with RAG,
achieving higher accuracy and superior efficiency.
REFERENCES
[1] Junhao Hu, Wenrui Huang, Weidong Wang, Haoying Wang, Tiancheng Hu, Qin
Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2024. EPIC: Efficient
Position-Independent Caching for Serving Large Language Models. InInterna-
tional Conference on Machine Learning. https://api.semanticscholar.org/CorpusID:
273502907
[2] Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxin Zhang, Jianwei Dong, Yuening
Zhu, Chen Lin, Jin Tang, Yaochen Han, Zhiyuan Ai, Xianglin Chen, Yongwei Wu,
and Cong Jiang. 2026. From Prefix Cache to Fusion RAG Cache: Accelerating
LLM Inference in Retrieval-Augmented Generation.ArXivabs/2601.12904 (2026).
https://api.semanticscholar.org/CorpusID:284911305
[3] Shihao Wang, Jiahao Chen, Yanqi Pan, Hao Huang, Yichen Hao, Xiangyu Zou,
Wen Xia, Wentao Zhang, Chong Qiu, and Pengfei Wang. 2026. ProphetKV: User-
Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-
Augmented Generation.ArXivabs/2602.02579 (2026). https://api.semanticscholar.
org/CorpusID:285275140
[4] Huan Yang, Renji Zhang, Ming-Yi Huang, Weijun Wang, Yin Tang, Yuanchun
Li, Yunxin Liu, and Deyu Zhang. 2025. KVShare: An LLM Service System with
Efficient and Effective Multi-Tenant KV Cache Reuse. https://api.semanticscholar.
org/CorpusID:277244216
[5] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2024. CacheBlend: Fast Large Lan-
guage Model Serving for RAG with Cached Knowledge Fusion.Proceedings
of the Twentieth European Conference on Computer Systems(2024). https:
//api.semanticscholar.org/CorpusID:270062853
[6] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao
Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett,
and Ying Sheng. 2023. SGLang: Efficient Execution of Structured Language
Model Programs.Advances in Neural Information Processing Systems 37(2023).
https://api.semanticscholar.org/CorpusID:266174771
4
