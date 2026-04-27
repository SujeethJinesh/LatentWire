# references/128_compresskv_semantic_retrieval_heads_know_what_tokens_are_not_important_before_generation.pdf

<!-- page 1 -->

CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important
Before Generation
Xiaolin Lin1, Jingcun Wang1, Olga Kondrateva1, Yiyu Shi2, Bing Li3, Grace Li Zhang1
1Technical University of Darmstadt 2University of Notre Dame 3University of Siegen
xiaolin.lin@tu-darmstadt.de, jingcun.wang@tu-darmstadt.de, olga.kondrateva@tu-darmstadt.de,
yshi4@nd.edu, bing.li@uni-siegen.de, grace.zhang@tu-darmstadt.de
Abstract
Recent advances in large language models (LLMs) have sig-
nificantly boosted long-context processing. However, the in-
creasing key-value (KV) cache size poses critical challenges
to memory and execution efficiency. Most KV cache com-
pression methods rely on heuristic token eviction using all
attention heads in Grouped Query Attention (GQA)-based
LLMs. This method ignores the different functionalities of
attention heads, leading to the eviction of critical tokens and
thus degrades the performance of LLMs.
To address the issue above, instead of using all the attention
heads in GQA-based LLMs to determine important tokens
as in the previous work, we first identify the attention heads
in each layer that are not only capable of retrieving the initial
and final tokens of a prompt, but also capable of retrieving im-
portant tokens within the text and attending to their surround-
ing semantic context. Afterwards, we exploit such heads to
determine the important tokens and retain their corresponding
KV cache pairs. Furthermore, we analyze the cache eviction
error of each layer individually and introduce a layer-adaptive
KV cache allocation strategy. Experimental results demon-
strate the proposed CompressKV consistently outperforms
state-of-the-art approaches under various memory budgets on
LongBench and Needle-in-a-Haystack benchmarks. Notably,
it retains over 97% of full-cache performance using only 3%
of KV cache on LongBench’s question-answering tasks and
achieves 90% of accuracy with just 0.07% of KV storage on
Needle-in-a-Haystack benchmark. Our code is publicly avail-
able at: https://github.com/TUDa-HW AI/CompressKV .git.
Introduction
Recent advances in large language models (LLMs) (Achiam
et al. 2024; Anthropic 2024; Dubey et al. 2024; Hui et al.
2025; Wang et al. 2025) have boosted their long-context pro-
cessing capabilities. However, with the increasing length of
texts, the resulting key-value (KV) cache size grows linearly.
The large KV cache leads to slow inference due to the atten-
tion calculation across past KV cache. In addition, the large
KV cache requires substantial memory storage, which cre-
ates a major bottleneck in the deployment of long-context
LLMs. Therefore, effective compression of KV cache is
essential for optimizing the computational efficiency and
model scalability.
State-of-the-art KV cache compression focuses on quan-
tization, low-rank approximation, and KV cache eviction
(Liu et al. 2024; Kang et al. 2024; Ge et al. 2024; Xiao
et al. 2024; Li et al. 2024; Cai et al. 2025; Yang et al. 2024;
Qin et al. 2025). Among such techniques, KV cache eviction
strategy where KV pairs corresponding to those unimportant
tokens are eliminated and the remaining KV pairs are kept
has started to draw more and more attention.
There are different criteria to determine unimpor-
tant tokens for KV cache compression. For example,
StreamingLLM (Xiao et al. 2024) retain the first and last to-
kens and neglects potentially important tokens in the middle
of the prompt. SnapKV (Li et al. 2024) clusters recent atten-
tion scores within an observation window at the end of the
prompt, either per head or per head group, to identify and
retain the important tokens receiving the highest attention
values. CAKE (Qin et al. 2025) extends SnapKV’s method
by adding the attention variance in an observation window
to the eviction score, enabling it to capture tokens whose
importance fluctuates over time.
The criteria described above are effective in many sce-
narios in compressing KV cache. However, they treat all
heads equally without examining their distinct functionali-
ties, so that they use the sum of the attention scores across
all the attention heads to make decisions on KV cache evic-
tion. In fact, attention heads exhibit different functional-
ities. For example, in Grouped Query Attention (GQA)-
based LLMs (Ainslie et al. 2023), some attention heads,
called Streaming Heads, exclusively focus on the beginning
and the end of a prompt (Xiao et al. 2024, 2025)). When
the attention heads within a GQA group are dominated by
Streaming Heads, those heads have the largest influence on
KV cache eviction, resulting in only the initial and last to-
kens’ KV pairs being retained. This leads to the eviction of
crucial tokens in the middle of a prompt and thus degrades
the performance of LLMs.
Besides eliminating KV pairs for those unimportant to-
kens, state-of-the-art research also allocates specified mem-
ory budgets to layers. For example, (Xiao et al. 2024;
Li et al. 2024) allocates each layer to a fixed number
of KV pairs without considering layer difference. (Yang
et al. 2024; Cai et al. 2025; Qin et al. 2025) allocates
KV cache budget across layers based on attention distri-
butions or layer-wise statistics such as attention entropy or
variance, which often require additional online computation
cost. Moreover, since attention distributions can vary signif-
arXiv:2508.02401v1  [cs.CL]  4 Aug 2025

<!-- page 2 -->

icantly across different models, limiting their generalization
ability and effectiveness.
In this paper, we observe that certain attention heads are
capable of retrieving important tokens within the text and
attending to their surrounding semantic context. We refer to
these heads as Semantic Retrieval Heads. Motivated by this
observation, we identify such Semantic Retrieval Heads in
each layer and use them to determine the crucial tokens and
share a unified set of crucial token indices across all heads
within that layer. This approach can substantially address the
dominance of Streaming Heads in KV cache evictions, so
that it can enhance the performance of GQA-based models.
Furthermore, we analyze the cache eviction error of each
layer individually and introduce a layer-adaptive KV cache
allocation strategy. Our contributions are as follows:
(1) We identify which attention heads are Semantic Re-
trieval Heads capturing both copy-and-paste and semantic
information. Such heads are used to determine unimpor-
tant tokens for KV cache eviction. Our experimental results
demonstrate Semantic Retrieval Heads know what tokens
are unimportant before generation.
(2) We estimate each layer’s compression impact by com-
puting the Frobenius norm of the difference between its
attention-block outputs with the compressed cache and those
with the full cache, during the decoding stage. Cache bud-
gets are then proportionally assigned across layers, priori-
tizing layers with higher errors. Importantly, this analysis
is performed offline and does not introduce any additional
overhead during online inference.
(3) CompressKV is validated on multiple LLMs using
LongBench and Needle-in-a-Haystack (NIAH). On Long-
Bench, CompressKV maintains over 99% of full-cache per-
formance with only 19% of KV entries and retains 97% of
question-answering accuracy using just 3% of the cache. On
Needle-in-a-Haystack retrieval benchmark, it achieves 90%
of the baseline accuracy with only 0.07% of KV storage.
Background and Related Work
KV-Cache Basics
The motivation of KV cache is to reduce the signification
computation cost of attention evaluation. To explain this,
consider the case of a single attention head. This atten-
tion head can be evaluated with weight matrices, denoted
as WQ, WK, WV ∈ Rd×d, and a prompt, denoted as
X ∈ Rl×d, where where l is the sequence length and d
the hidden dimension. The attention evaluation includes two
phases, i.e., prefilling phase and decoding phase.
Prefilling Phase: in this phase, the query Q, key K, and
value V are evaluated with the entire input embeddings as
follows
Q = XWQ, K = XWK, V = XWV (1)
With K, V and Q, the output of the attention can be evalu-
ated as follows
O = Softmax
 
Q K⊤
V (2)
The key K and the value V are then stored in cache mem-
ory, which is also called KV cache. Decoding Phase: In this
phase, the previously stored KV cache is used to generate
new tokens and the newly generated KV pair is then ap-
pended to the previously stored KV cache to refresh KV
cache. Specifically, at a decoding step t, given a new token
embedding xt ∈ R1×d, we first evaluate the newly gener-
ated KV pairs with this new token as follows
kt = xt WK, vt = xt WV. (3)
Afterwards, we use such new KV pairs to update the
cache via
K ← Concat

K, kt

, V ← Concat

V, vt

. (4)
In GQA-based LLMs, query heads in a layer are partitioned
into multiple groups. Multiple query heads within the same
group share the same KV cache. The shared key and value
are evaluated once per group and reused to produce the out-
put of each head in the group. Although KV caching re-
moves the need to recompute keys and values at every step,
the cache itself grows linearly with prompt sequence length,
becoming especially problematic for long-text tasks.
KV Cache Compression To alleviate the burden of KV
cache storage, various KV cache compression methods,
e.g., quantization (Liu et al. 2024), low-rank approxima-
tions (Kang et al. 2024), and KV cache eviction strategy
have been proposed. In particular, KV cache eviction re-
duces cache size by removing KV cache pairs of unim-
portant tokens without retraining. There are different evic-
tion strategies. For example, StreamingLLM (Xiao et al.
2024) focuses solely on retaining the first and last tokens,
which only addresses the Streaming Head scenario and ne-
glects potentially important tokens in the middle of the se-
quence. To overcome this limitation, more advanced meth-
ods have been proposed(Liu et al. 2023; Zhang et al. 2023;
Li et al. 2024; Han et al. 2024; Oren et al. 2024). A repre-
sentative example is SnapKV (Li et al. 2024), which clus-
ters recent attention scores, either per head or per head
group to identify important token and retain the KV cache
pairs of such tokens. Besides, recent approaches, including
PyramidKV (Cai et al. 2025), D2O (Wan et al. 2025), and
CAKE (Qin et al. 2025), dynamically allocate cache budgets
based on attention statistics or modeled attention dynamics
of all the layers in an LLM. Their selection strategies for im-
portant tokens are an extended version of SnapKV’s eviction
strategy.
The KV cache eviction approaches above have two major
limitations. First, they treat all attention heads equally, ig-
noring their functional heterogeneity; Recent work (Olsson
et al. 2022; Kwon et al. 2022; Zheng et al. 2024; Ren et al.
2024; Wu et al. 2025; Todd et al. 2024; Yin and Steinhardt
2025) has shown that different attention heads have distinct
roles. For example, some attention heads, called Streaming
Heads in the state-of-the-art research, always focus on the
beginning and the end of a prompt. For example, in Fig-
ure 1(a), head 0 is such a Streaming Head since the atten-
tion scores of the initial token and the last tokens are larger
than the remaining tokens. On the contrary, some attention
heads, called Retrieval heads in Wu et al. (2025), exhibit
copy-and-paste behaviors for long-context scenarios. For ex-
ample, in Figure 1(b), head 1 is such a retrieval head since

<!-- page 3 -->

 <|BOS|> Once upon a  ... ...  reading drafts of this.
Attention Score
0  The best thing to do in San Francisco is to eat a  sandwich
ContextContext Needle Sentence
What is the best thing to do in San Francisco?
Question
High attention zone High attention zone  SandwichA: Eat a   ...
(a) Attention distribution of head 0 (Streaming Head(SH))
 SandwichA: Eat a   ...
 <|BOS|> Once upon a  ... ...  reading drafts of this.
Attention Score
0  The best thing to do in San Francisco is to eat a  sandwich
ContextContext Needle Sentence
What is the best thing to do in San Francisco?
Question
Highest attention score
(b) Attention distribution of head 1 (Retrieval Head(RH))
Darker = Higher
Lighter = LowerTop-N
Head 1(SH)
Avg
Head 2(RH)
Sum      Pooling
Head 0(SH)
(c) Critical tokens were evicted
Key Information in the ContextInitial tokens Last tokens
Figure 1: Motivation. (a) The attention score distribution of a streaming head (SH). (b) The attention score distribution of a
retrieval head (RH). (c) Streaming attention heads in a GQA group dominate the token eviction, indicating only initial and final
tokens are remained. The critical tokens are evicted.
the attention scores of the correct answer “sandwich” are
larger. In GQA-based LLMs, Streaming Heads tend to have
larger effect than the other heads for KV cache eviction,
which indicates only KV cache pairs corresponding to ini-
tial and last tokens are retained. This leads to the eviction of
crucial tokens in the middle of a prompt and thus degrades
the performance of LLMs. Figure 1(c) illustrates such an ex-
ample, where Streaming Heads including head0 and head1
dominate token eviction for KV cache compression.
Second, the layer budget allocation in the previous work
typically relies on attention distributions or layer-wise statis-
tics such as attention entropy or variance, which often re-
quire additional online computation. Moreover, since at-
tention distributions can vary significantly across different
models, directly adopting a fixed allocation strategy accord-
ing to attention distributions may not yield optimal results.
CompressKV
CompressKV includes three key components: (1) Identifi-
cation of the attention heads that are capable of retriev-
ing important tokens within the text and attending to their
surrounding semantic context. (2) Important token selec-
tion driven by such identified heads. (3) Error-aware layer-
adaptive cache allocation. In the following subsections, we
will first explain our observations and insights into identifi-
cation of attention heads with specified functionalities. Af-
terwards, we will take advantage of such heads to select to-
kens for KV cache eviction. Furthermore, different cache
budgets will be allocated to different layers.
Observations and Insights
To avoid that streaming attention heads dominate the KV
cache eviction as illustrated in Figure 1(c), intuitively, re-
trieval heads instead of all the attention heads can be used
to identify important tokens for KV cache eviction. How-
ever, the state-of-the-art research on identification of Re-
trieval Heads consider only those attention heads, the highest
attention score of which aligns exactly with the correct token
answer during generation, as retrieval attention heads. Such
retrieval attention heads exhibits copy-and-paste behaviors.
However, such an identification might lose some attention
heads that are capable of retrieving important tokens within
the text and attending to their surrounding semantic context.
Figure 2(a) illustrates an example to explain the draw-
back of the state-of-the-art identification technique of re-
trieval heads. Head 0 is not considered as Retrieval Head
since its highest attention score does not falls on the “sand-
wich” token in the needle sentence when generating “sand-
wich”. Head 1 is considered as the Retrieval Head. However,
sum of the attention scores surrounding “sandwich” in head
0 is still large, which indicate that it is still capable of re-
trieving important tokens within the text and attending to
their surrounding semantic context.
In long-context scenarios, the attention distribution is par-
ticularly sparse, with a substantial amount of attention often
allocated to initial tokens and trailing tokens. As a result, tra-
ditional identification methods of Retrieval Heads that rely
on top-1 or top-k matches exhibit extremely low hit rates,
causing most retrieval scores to be zero. Moreover, these
metrics capture only copy-and-paste behaviors and ignore
deeper semantic dependencies. For example, as shown in
Figure 2(a), when generating “sandwich,” the model attends
not only to “sandwich” itself but also to related tokens like
“eat” or “a thing.” Under a strict top-1/top-k criterion, such
attentions may not be credited. Accordingly, the identifica-
tion of retrieval attention heads is not effective.
To address the issue above, we propose a new standard
to identify the heads that capture not only copy-and-paste
behaviors and but also deeper semantic dependencies. We
call such attention heads as Semantic Retrieval Heads. We
use such heads to identify important tokens for KV cache
eviction.
Semantic Retrieval Head Identification Standards
Instead of requiring exact top-k hits in the traditional Re-
trieval Head identification, we aggregate a head’s attention
scores over the entire answer span inserted into a long con-
text whenever the model generates a correct answer token as
the score of this head. This evaluation is expressed with the

<!-- page 4 -->

...  reading drafts of this. ...  reading drafts of this.
Context
(b) Semantic retrieval head identification
Q: What is the the best thing
to do in San Francisco?
Attention distribution of head 0
 <|BOS|> Once upon a  ...Attention Score
0
Context Needle Sentence
 The best thing to do in San Francisco is to eat a  sandwich
A: Eat a   Sandwich ...
Larger enough
(a) Traditional retrieval head identification
Q: What is the the best thing
to do in San Francisco?
Attention distribution of head 1
 <|BOS|> Once upon a  ...
Attention Score
0  The best thing to do in San Francisco is to eat a  sandwich
Context Needle Sentence
A: Eat a   Sandwich ...
Larger enough
Successfully CapturedUnable to recognize
...  reading drafts of this.
Context
...  reading drafts of this.
Q: What is the the best thing
to do in San Francisco?
Attention distribution of head 1
 <|BOS|> Once upon a  ...Attention Score
0  The best thing to do in San Francisco is to eat a  sandwich
Context Needle Sentence
A: Eat a   Sandwich ...
Top 1 score
Context
Q: What is the the best thing
to do in San Francisco?
Attention distribution of head 0
 <|BOS|> Once upon a  ...
Attention Score
0  The best thing to do in San Francisco is to eat a  sandwich
Context Needle Sentence
A: Eat a   Sandwich ...
Not the top-1 score
Context
Figure 2: Illustration of Semantic Retrieval Head identification versus traditional Retrieval Head selection. Semantic Retrieval
Heads capture attention over the entire answer span, addressing the limitations of traditional methods that rely solely on copy-
and-paste behavior.
following equation as follows
SemanticRetrievalScore(h) =
NX
t=1
I

yt ∈ A
 X
j∈A
ah
t,j (5)
where yt is the generated token at step t, A is the answer
span, and ah
t,j is head h’s attention weight on thej-th token
of A. The higher the score of a head is, the more capable of
capturing semantic information this head is.
Figure 2(b) illustrates the concept of this new identifi-
cation standard. By summing over the entire span, we can
capture attention heads that contribute semantically rele-
vant context even when they never achieve top-1 atten-
tion on a single token, dramatically reducing the fraction
of zero-score heads. Aggregation over multiple tokens en-
ables the method to recognize heads that attend to semantic
cues—such as “eat” or “a thing” around “sandwich”—rather
than only pure copy-and-paste patterns. For example, head 0
in Figure 2 is considered as Semantic Retrieval Head in our
new standard although it is not considered as Retrieval Head
in the traditional identification methods. For a visual com-
parison between Semantic Retrieval Heads and traditional
Retrieval Heads, please refer to Appendix C
Token Selection Driven by Semantic Retrieval
Heads
In GQA-based LLMs, for each layer, we will select top top-
k Semantic Retrieval Heads with high scores defined with
equation (5) as the criterion for selecting important tokens
for KV cache eviction. All the attention heads within this
layer share a common set of selected token indices deter-
mined by these top Semantic Retrieval Heads. This con-
cept is illustrated in Figure 3, where a layer has two groups.
In this example, head2 and head3 are top 2 Semantic Re-
trieval Heads. The attention score matrices of such heads
are compressed by summing over the observation window
and pooling across the token dimension. Afterwards, such
compressed vectors are averaged. The tokens with the top
N highest attention scores will be selected and their corre-
sponding KV cache pairs will be retained. The KV cache
pairs for the remaining tokens will be evicted to compress
KV cache.
Error-Aware Layer-Adaptive Cache Allocation
To maximize memory efficiency under strict budget con-
straints, we propose an error-aware and layer-adaptive cache
allocation strategy. Instead of relying on attention statistics
as in the previous methods, this approach quantifies the com-
pression error caused by KV cache compression, using full-
cache outputs as the reference. We specifically focus on the
extreme compression setting, where only a small fraction of
tokens are retained in each layer’s KV cache. For each layer
l and decoding step t, let Ol
full,t and Ol
comp,t denote the at-
tention outputs using the full and compressed KV caches,
respectively:
Ol
full,t = Wl
O Attention
 
Ql
t, Kl
full, Vl
full

(6)
Ol
comp,t = Wl
O Attention
 
Ql
t, Kl
comp, Vl
comp

(7)
where W(l)
O is the output projection matrix of layer l, Ql
t is
the query, Kl is the key, and Vl is the value representation
at layer l. To evaluate the error incurred by compressing KV
cache per layer, the error score for layer l is computed and
normalized as:
e(l) =
TX
t=1


Ol
comp,t − Ol
full,t


F


Ol
full,t



F
+ ϵ
, ˜e(l) = e(l)
P
k e(k) (8)
where T is the total number of decoding steps,| · |F denotes
the Frobenius norm and ϵ is a small positive constant (e.g.,
10−6) to prevent division by zero.
Given the normalized per-layer error scores ˜ eand total
cache budget Btotal, we first assign a minimum allocation
m and a maximum allocation M to each layer to avoid a
layer either has no memory budget or a large memory bud-
get. The remaining budget is distributed in proportion to the
error scores. More details can be found in Appendix B.

<!-- page 5 -->

CompressKV
Sum      Pooling
Head 1(SH)Head 0(SH) Head 2(SRH) Head 4(SH)Head 3(SRH)
Top-N
Head 5(SH)
Group 0 Group 1Avg
Sum      Pooling
Last tokens
Initial tokens
Key Information in the Context
SH: Streaming HeadSRH: Semantic Retrieval Head
Figure 3: Illustration of the token selection driven by Semantic Retrieval Heads.
Experiments
Baselines and Backbone LLMs We com-
pare CompressKV with four representative work:
StreamingLLM (Xiao et al. 2024), SnapKV (Li et al.
2024), PyramidKV (Cai et al. 2025), CAKE (Qin et al.
2025)). All methods are evaluated on state-of-the-art open-
source LLMs, including Llama-3.1-8B-Instruct (Dubey
et al. 2024) and Mistral-7B-Instruct-v0.3 (Jiang et al. 2024).
Evaluations are conducted in a generative setting using
greedy decoding to ensure fair comparison across tasks.
Evaluating Tasks To evaluate CompressKV’s perfor-
mance under different memory budgets, we adopt two com-
prehensive benchmarks and one masking-based ablation
analysis: (1) LongBench (Bai et al. 2024), which evalu-
ates long-context understanding across 16 datasets; see Ap-
pendix A for more details. (2) Needle-in-a-Haystack (Kam-
radt 2023), which measures the retrieval of a target answer
hidden in extended text; and (3) a masking-based ablation
study of different head types, in which we selectively disable
each type to quantify its contribution to overall performance.
Implementation Details Our experiments evaluate Com-
pressKV and baseline methods under total memory budgets
ranging from 128 to 2048 tokens for each layer. The KV
cache budget is distributed equally across layers for base-
line methods: StreamingLLM and SnapKV , while methods
such as PyramidKV , CAKE, and CompressKV distributes
the cache differently across layers but keeps total memory
usage fixed. To ensure a fair comparison, tokens are evicted
only during the prefilling phase. For CompressKV , we se-
lect the top four Semantic Retrieval Heads in each layer
to identify and preserve the most important tokens. Using
the LongBench benchmark, we derive each layer’s normal-
ized error scores by simulating minimal-size KV compres-
sion and computing the Frobenius-norm reconstruction er-
ror of its attention-block outputs. During budget allocation,
we impose per-layer bounds [m, M] with m = 32 and
M = 3 × Bper-layer —and distribute the remaining KV pairs
proportionally to the normalized errors.
Evaluation on LongBench Benchmark
Table 1 demonstrates performance comparison under two
KV cache regimes—low (256) and high (2048)—with full
results across additional budgets in Appendix D. Com-
pressKV consistently ranks the top performers across var-
ious tasks. The advantage of CompressKV is particularly
pronounced in low-memory scenarios. CompressKV im-
proves accuracy by nearly 2 percentage points over SnapKV
and outperforms CAKE by 0.7 points; even in the 2048
cache budget setting scenario, where CAKE falls be-
hind SnapKV on Llama-3.1-8B-Instruct, CompressKV still
maintains superior accuracy. By leveraging a small number
of Semantic Retrieval Heads to accurately identify semanti-
cally important tokens, combined with an effective adaptive
layer budget allocation strategy, CompressKV achieves the
best overall performance.
As illustrated in Figure 4, we benchmark CompressKV on
LongBench across KV cache sizes from 128 to 2048, pre-
senting results for both Llama-3.1-8B-Instruct and Mistral-
7B-Instruct-v0.3. The evaluation metric is the average score
across all LongBench datasets. SnapKV outperforms the
legacy method StreamingLLM. Despite its methodologi-
cal similarities to SnapKV , PyramidKV underperforms in
many scenarios, possibly due to its limited adaptability.
CAKE achieves better results than previous baseline meth-
ods in most cases by dynamically allocating memory to each
layer and incorporating additional computations of variance
and entropy scores. CompressKV consistently surpasses all
aforementioned methods across all cache budgets, with the
performance gap being particularly notable under small KV
cache sizes where memory constraints are more severe.
Figure 4: Average performance on 16 LongBench datasets
under different KV cache budget settings compared with
various baseline methods.

<!-- page 6 -->

Method KV Size Single-doc QA Multi-doc QA Summarization Few-shot Learning Synthetic Code Avg.
Llama-3.1-8B-Instruct
FullKV Full 43.41 44.44 29.22 69.48 52.75 60.06 49.08
StreamingLLM
2048
37.02 33.10 25.76 56.57 38.74 44.51 38.99
SnapKV 42.95 44.01 27.29 69.02 52.75 60.09 48.47
PyramidKV 42.85 44.19 26.93 69.15 53.03 59.01 48.34
CAKE 42.56 43.87 27.45 68.67 52.84 59.45 48.26
CompressKV 43.43 44.17 27.88 69.11 52.75 60.02 48.71
StreamingLLM
256
26.52 29.73 21.16 47.60 47.06 36.83 33.92
SnapKV 38.84 43.57 23.41 63.40 52.63 55.21 45.21
PyramidKV 37.28 43.41 23.04 62.40 52.38 53.29 44.36
CAKE 41.01 43.30 24.38 66.02 52.82 55.56 46.30
CompressKV 41.84 43.75 24.26 66.52 52.82 56.29 46.71
Mistral-7B-Instruct-v0.3
FullKV Full 41.16 38.99 29.50 70.70 52.00 60.03 47.82
StreamingLLM
2048
34.17 28.72 25.85 53.99 38.50 39.47 36.51
SnapKV 41.21 38.65 26.66 70.18 51.50 59.87 47.05
PyramidKV 40.54 38.69 26.70 70.39 51.50 58.83 46.85
CAKE 41.18 38.32 27.83 70.24 51.50 59.96 47.22
CompressKV 41.28 39.52 27.93 70.58 51.50 59.97 47.55
StreamingLLM
256
25.26 26.40 20.76 49.37 34.50 32.58 31.22
SnapKV 35.20 37.08 22.35 67.72 51.00 55.59 43.76
PyramidKV 34.73 36.80 21.89 67.66 49.75 53.10 43.06
CAKE 38.29 37.73 24.03 67.81 50.00 56.06 44.73
CompressKV 39.34 38.48 23.56 69.99 50.50 55.89 45.43
Table 1: Performance comparison of CompressKV with StreamingLLM, SnapKV , PyramidKV , CAKE, and FullKV on Long-
Bench for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. CompressKV generally outperforms other KV cache compres-
sion methods across various KV cache sizes and LLMs.
Evaluation on Needle In A Haystack
In the Mistral-7B-Instruct-v0.3, both CompressKV and
CAKE achieve lossless compression under a 256 KV cache
budget for 32K long-context inputs, as shown in Figure 5.
Notably, CompressKV attains performance comparable to
other methods even under 128K long-context inputs in
Llama3.1-8B-Instruct, as shown in Figure 6. Remarkably,
CompressKV reaches 90% of the original accuracy using
only 256 KV cache entries (0.07% of the full capacity). To-
gether with the LongBench evaluation, these results demon-
strate that CompressKV effectively maintains general LLM
performance across diverse long-context tasks while achiev-
ing efficient KV cache compression. For more results, please
refer to the Appendix E.
Masking-Based Ablation of Different Head Types
To isolate the contribution of Semantic Retrieval Heads, we
perform targeted ablation by masking the top 20 of these
heads and comparing against traditional Retrieval Heads, as
shown in Figure 7. Even masking a small subset of Semantic
Retrieval Heads causes a sharp drop in retrieval accuracy and
a significant rise in hallucinations, underscoring their essen-
tial role in preserving factual consistency and their ability to
retrieve and localize textual information. For more results,
please refer to the Appendix F.
Evaluation of Latency and Peak Memory
We evaluate the end-to-end generation latency and peak
memory usage on Llama-3.1-8B-Instruct, implemented with
FlashAttention-2 (Dao 2024), running on a single NVIDIA
A100 GPU. The evaluation spans context lengths from 4K
to 128K tokens with a fixed generation length of 1024
tokens. We compare our proposed CompressKV method
against a full cache baseline and four KV cache evic-
tion methods—StreamingLLM, SnapKV , PyramidKV , and
CAKE—each constrained by a KV cache budget of 1024.
As illustrated in Figure 8, the end-to-end generation la-
tency increases with longer context lengths for all meth-
ods. However, all KV cache eviction strategies—including
CompressKV—significantly reduce latency compared to the
full cache baseline, especially as the context length grows.
CAKE exhibits slightly higher latency than the other meth-
ods, likely due to the additional computations required for
entropy and variance estimation. Figure 8 shows that, under
a fixed KV budget, all eviction methods (including Com-
pressKV) incur similar peak memory, whereas the full-cache
baseline uses substantially more—especially at longer con-
texts.

<!-- page 7 -->

Mistral-7B-Instruct-v0.3
Figure 5: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 256. All methods are evalu-
ated under identical settings.
Llama-3.1-8B-Instruct
Figure 6: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 256. All methods are evaluated
under identical settings.
Ablation Studies
To understand the contributions of each component in our
CompressKV framework, we conduct a series of ablation
studies on the LongBench benchmark using Mistral-7B-
Instruct-v0.3 with a fixed KV cache budget of 256.
Ablation Study on the Number of Selected Heads per
Layer. To quantify how many Semantic Retrieval Heads
are needed per layer, we vary the selection from 2 up to
24 heads and measure average accuracy on LongBench (Ta-
ble 2). Moving from 2 to 4 heads yields the largest gain
(+0.63 percentage points), while increasing beyond 4 offers
no further improvement (Top-6: -0.17; Top-12: 0.00). Se-
lecting 24 heads slightly degrades performance. This indi-
cates that a small subset of around four heads is sufficient to
capture the majority of semantic retrieval capacity.
Ablation Study on Token Selection and Layer-Wise
Cache Allocation. We conduct an ablation study to evalu-
Figure 7: Ablation analysis on masking different head types
in Mistral-7B-Instruct-v0.3.
Figure 8: Comprehensive evaluation of LLaMA-3.1-8B-
Instruct on a single NVIDIA A100 GPU. Both the KV cache
budget and generation length are fixed at 1024 tokens.
Heads per Layer Mean Accuracy (%) ∆ vs. Top-4 (%)
Top-2 44.33 –0.63
Top-4 44.96 0.00
Top-6 44.79 –0.17
Top-12 44.96 0.00
Top-24 44.30 –0.66
Table 2: Ablation study on the number of Semantic Retrieval
Heads per layer; ∆ denotes the change relative to selecting
four heads.
ate the individual contribution of Semantic Retrieval Head
driven token selection and layer-aware budget allocation
methods on LongBench. Results on Mistral-7B-Instruct-
v0.3 are shown in Table 3. Introducing the proposed selec-
tion mechanism over the SnapKV baseline yields a clear
gain, and incorporating our layer-aware allocation further
improves accuracy, confirming that both components are
complementary.
Method Acc. (%)
SnapKV 43.76
+ SRH Selection 44.96
+ SRH Selection + Layer Alloc 45.43
Table 3: Ablation on token selection strategy (SRH = Se-
mantic Retrieval Heads) and layer-aware cache allocation
Conclusion
In this work, we have proposed CompressKV , a novel KV-
cache compression framework for GQA-based LLMs that
(1) identifies Semantic Retrieval Heads, which not only fo-
cus on initial and terminal tokens but also retrieve semanti-
cally important tokens and their contexts—and (2) allocates
a layer-adaptive cache budget by measuring each layer’s of-
fline cache-eviction error. Extensive experiments on Long-

<!-- page 8 -->

Bench and Needle-in-a-Haystack across multiple model ar-
chitectures and cache budgets confirm CompressKV’s con-
sistently superior performance under diverse memory con-
straints.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya,
I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman,
S.; Anadkat, S.; et al. 2024. GPT-4 Technical Report.
arXiv:2303.08774.
Ainslie, J.; Lee-Thorp, J.; de Jong, M.; Zemlyanskiy, Y .; Le-
bron, F.; and Sanghai, S. 2023. GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Check-
points. In The 2023 Conference on Empirical Methods in
Natural Language Processing.
Anthropic. 2024. The Claude 3 Model Family: Opus, Son-
net, Haiku. Technical report, Anthropic. Accessed: 2024-
07-09.
Bai, Y .; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du,
Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y .; Tang, J.; and Li,
J. 2024. LongBench: A Bilingual, Multitask Benchmark for
Long Context Understanding. In Proceedings of the 62nd
Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers).
Cai, Z.; Zhang, Y .; Gao, B.; Liu, Y .; Li, Y .; Liu, T.; Lu, K.;
Xiong, W.; Dong, Y .; Hu, J.; and Xiao, W. 2025. Pyra-
midKV: Dynamic KV Cache Compression based on Pyra-
midal Information Funneling. arXiv:2406.02069.
Dao, T. 2024. FlashAttention-2: Faster Attention with Better
Parallelism and Work Partitioning. In The Twelfth Interna-
tional Conference on Learning Representations.
Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle,
A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.;
Fan, A.; et al. 2024. The Llama 3 Herd of Models.
arXiv:2407.21783.
Ge, S.; Zhang, Y .; Liu, L.; Zhang, M.; Han, J.; and Gao,
J. 2024. Model Tells You What to Discard: Adaptive KV
Cache Compression for LLMs. In The Thirteenth Interna-
tional Conference on Learning Representations.
Han, C.; Wang, Q.; Peng, H.; Xiong, W.; Chen, Y .; Ji, H.;
and Wang, S. 2024. LM-Infinite: Zero-Shot Extreme Length
Generalization for Large Language Models. In Proceedings
of the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), 3991–4008.
Hui, B.; Yang, J.; Cui, Z.; Yang, J.; Liu, D.; Zhang, L.; Liu,
T.; Zhang, J.; Yu, B.; Lu, K.; et al. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115.
Jiang, D.; Liu, Y .; Liu, S.; Zhao, J.; Zhang, H.; Gao, Z.;
Zhang, X.; Li, J.; and Xiong, H. 2024. From CLIP to
DINO: Visual Encoders Shout in Multi-modal Large Lan-
guage Models. arXiv:2310.08825.
Kamradt, G. 2023. NeedleInAHaystack. https://github.com/
gkamradt/LLMTest NeedleInAHaystack. Accessed: 2025-
07-13.
Kang, H.; Zhang, Q.; Kundu, S.; Jeong, G.; Liu, Z.; Krishna,
T.; and Zhao, T. 2024. GEAR: An Efficient KV Cache Com-
pression Recipe for Near-Lossless Generative Inference of
LLM. arXiv:2403.05527.
Kwon, W.; Kim, S.; Mahoney, M. W.; Hassoun, J.; Keutzer,
K.; and Gholami, A. 2022. A Fast Post-Training Pruning
Framework for Transformers. In Oh, A. H.; Agarwal, A.;
Belgrave, D.; and Cho, K., eds., Advances in Neural Infor-
mation Processing Systems.
Li, Y .; Huang, Y .; Yang, B.; Venkitesh, B.; Locatelli, A.; Ye,
H.; Cai, T.; Lewis, P.; and Chen, D. 2024. SnapKV: LLM
Knows What You are Looking for Before Generation. In
The Thirty-eighth Annual Conference on Neural Information
Processing Systems.
Liu, Z.; Desai, A.; Liao, F.; Wang, W.; Xie, V .; Xu, Z.; Kyril-
lidis, A.; and Shrivastava, A. 2023. Scissorhands: Exploit-
ing the Persistence of Importance Hypothesis for LLM KV
Cache Compression at Test Time. In Thirty-seventh Confer-
ence on Neural Information Processing Systems.
Liu, Z.; Yuan, J.; Jin, H.; Zhong, S.; Xu, Z.; Braverman, V .;
Chen, B.; and Hu, X. 2024. KIVI: A Tuning-Free Asym-
metric 2bit Quantization for KV Cache. In Forty-first Inter-
national Conference on Machine Learning.
Olsson, C.; Elhage, N.; Nanda, N.; Joseph, N.; DasSarma,
N.; Henighan, T.; Mann, B.; Askell, A.; Bai, Y .; Chen, A.;
Conerly, T.; Drain, D.; Ganguli, D.; Hatfield-Dodds, Z.; Her-
nandez, D.; Johnston, S.; Jones, A.; Kernion, J.; Lovitt, L.;
Ndousse, K.; Amodei, D.; Brown, T.; Clark, J.; Kaplan, J.;
McCandlish, S.; and Olah, C. 2022. In-context Learning and
Induction Heads. arXiv:2209.11895.
Oren, M.; Hassid, M.; Yarden, N.; Adi, Y .; and Schwartz, R.
2024. Transformers are Multi-State RNNs. In Proceedings
of the 2024 Conference on Empirical Methods in Natural
Language Processing, 18724–18741.
Qin, Z.; Cao, Y .; Lin, M.; Hu, W.; Fan, S.; Cheng, K.; Lin,
W.; and Li, J. 2025. CAKE: Cascading and Adaptive KV
Cache Eviction with Layer Preferences. In The Thirteenth
International Conference on Learning Representations.
Ren, J.; Guo, Q.; Yan, H.; Liu, D.; Zhang, Q.; Qiu, X.; and
Lin, D. 2024. Identifying Semantic Induction Heads to Un-
derstand In-Context Learning. In Findings of the Associa-
tion for Computational Linguistics: ACL 2024.
Todd, E.; Li, M.; Sharma, A. S.; Mueller, A.; Wallace, B. C.;
and Bau, D. 2024. Function Vectors in Large Language
Models. In The Twelfth International Conference on Learn-
ing Representations.
Wan, Z.; Wu, X.; Zhang, Y .; Xin, Y .; Tao, C.; Zhu, Z.;
Wang, X.; Luo, S.; Xiong, J.; Wang, L.; and Zhang, M.
2025. D2O: Dynamic Discriminative Operations for Effi-
cient Long-Context Inference of Large Language Models. In
The Thirteenth International Conference on Learning Rep-
resentations.
Wang, J.; Chen, Y .-G.; Lin, I.-C.; Li, B.; and Zhang, G. L.
2025. Bsis Sharing: Cross-Layer Parameter Sharing for
Large Language Model Compression. In International Con-
ference on Learning Representations.

<!-- page 9 -->

Wu, W.; Wang, Y .; Xiao, G.; Peng, H.; and Fu, Y . 2025. Re-
trieval Head Mechanistically Explains Long-Context Factu-
ality. In The Thirteenth International Conference on Learn-
ing Representations.
Xiao, G.; Tang, J.; Zuo, J.; junxian guo; Yang, S.; Tang,
H.; Fu, Y .; and Han, S. 2025. DuoAttention: Efficient
Long-Context LLM Inference with Retrieval and Stream-
ing Heads. In The Thirteenth International Conference on
Learning Representations.
Xiao, G.; Tian, Y .; Chen, B.; Han, S.; and Lewis, M. 2024.
Efficient Streaming Language Models with Attention Sinks.
In The Twelfth International Conference on Learning Rep-
resentations.
Yang, D.; Han, X.; Gao, Y .; Hu, Y .; Zhang, S.; and Zhao, H.
2024. PyramidInfer: Pyramid KV Cache Compression for
High-throughput LLM Inference. In Findings of the Associ-
ation for Computational Linguistics ACL 2024, 3258–3270.
Yin, K.; and Steinhardt, J. 2025. Which Attention Heads
Matter for In-Context Learning? arXiv:2502.14010.
Zhang, Z.; Sheng, Y .; Zhou, T.; Chen, T.; Zheng, L.; Cai, R.;
Song, Z.; Tian, Y .; Re, C.; Barrett, C.; Wang, Z.; and Chen,
B. 2023. H2O: Heavy-Hitter Oracle for Efficient Genera-
tive Inference of Large Language Models. In Thirty-seventh
Conference on Neural Information Processing Systems.
Zheng, Z.; Wang, Y .; Huang, Y .; Song, S.; Yang, M.; Tang,
B.; Xiong, F.; and Li, Z. 2024. Attention Heads of Large
Language Models: A Survey. arXiv:2409.03752.

<!-- page 10 -->

A Dataset Details
Table 4 presents the LongBench benchmark used in
our experiments, which consists of 14 English sub-
tasks and 2 code-completion subtasks organized into
six categories—single-document QA, multi-document QA,
summarization, few-shot learning, synthetic tasks, and code
completion. Each subtask contains 150–500 samples with
input lengths ranging from 1,235 to 18,409 words. Evalu-
ation metrics include F1, Rouge-L, classification accuracy,
and edit similarity.
B More Implementation Details
In this section, we provide additional details of our experi-
mental setup and a comprehensive description of the error-
aware, layer-adaptive cache allocation algorithm used by
CompressKV . To ensure a fair comparison across all KV
cache compression methods, we use identical hyperparame-
ters: an observation window of 8 tokens, a 1D pooling kernel
of size 5, and average-pooling to aggregate attention scores.
Detailed Description of Error-Aware
Layer-Adaptive Cache Allocation
Using the LongBench benchmark, we simulate an extreme
compression scenario by restricting each layer’s KV cache
size to 32 tokens (approximately 0.3% of full capacity). Un-
like completely skipping an attention block (binary on/off),
retaining a small subset of tokens allows us to explicitly
quantify the direct impact of KV cache compression on the
attention outputs. This approach effectively captures fine-
grained compression errors without incurring multiple for-
ward computations that would otherwise be necessary for
evaluating the complete removal of attention blocks.
Formally, for each dataset d ∈ D, transformer layer l,
and decoding step t, we compute the per-layer compression-
induced reconstruction error as follows:
e(l)
d =
TX
t=1
∥O(l)
comp,t − O(l)
full,t∥F
∥O(l)
full,t∥F + ϵ
(9)
where T denotes the total decoding steps, ∥ · ∥ F represents
the Frobenius norm, and ϵ = 10−6 ensures numerical stabil-
ity. Next, we perform an L1 normalization of the per-layer
errors within each dataset:
ˆe(l)
d = e(l)
dX
k
e(k)
d
. (10)
Then, we average these normalized per-layer errors across
all datasets:
¯e(l) = 1
|D|
X
d∈D
ˆe(l)
d . (11)
Finally, we apply another L1-normalization across layers to
obtain the final importance scores:
˜e(l) = ¯e(l)
P
k ¯e(k) . (12)
Averaging normalized errors across all datasets ensures
both generalizability and fairness: by averaging errors from
diverse datasets, we capture consistent trends in layer im-
portance rather than overfitting to any single task or do-
main. Compared with budget allocation methods that rely
solely on attention-score distributions, our error-aware ap-
proach explicitly quantifies the impact of compression on
the model’s final attention outputs, resulting in a more pre-
cise and effective allocation strategy. These normalized,
dataset-averaged error scores ˜e(l) guide our error-aware,
layer-adaptive cache allocation as detailed in Algorithm 1
below.
To safeguard against extreme cases, we impose per-layer
bounds [m, M], where the minimum allocation m = 32
ensures that each layer receives at least a small, baseline
cache allocation, preventing any single layer from becom-
ing completely inactive under extreme conditions. The up-
per bound M = 3 × Bper-layer prevents excessive cache allo-
cation to any individual layer, ensuring a balanced distribu-
tion of cache resources and maintaining overall model per-
formance. Additionally, we plot the performance of both the
Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct models
under a per-layer KV cache budget of 256 tokens as bar
charts (see Figures 9 and 10), illustrating the distinct allo-
cation characteristics of each model.
Algorithm 1: Error-aware Layer-adaptive Cache Allocation
Require: Scores ˜ e, total budget Btotal, per-layer bounds
[m, M]
Ensure: Allocations B
1: Bi ← m, ∀i
2: R ← Btotal −P
i Bi
3: Bi ← clip(Bi + round(˜ei · R), m, M), ∀i
4: ∆ ← Btotal −P
i Bi
5: while ∆ ̸= 0 do
6: if ∆ > 0 then
7: L ← { i | Bi < M }
8: if L = ∅ then
9: Break
10: end if
11: j ← arg maxi∈L ˜ei, Bj ← Bj + 1, ∆ ← ∆ − 1
12: else
13: L ← { i | Bi > m}
14: if L = ∅ then
15: Break
16: end if
17: j ← arg mini∈L ˜ei, Bj ← Bj − 1, ∆ ← ∆ + 1
18: end if
19: end while
20: return B
C Head visualization
In Figures 11 and 12, we present a comparison between
traditional Retrieval Heads and Semantic Retrieval Heads
identified using Mistral-7B-Instruct-v0.3 and Llama-3.1-
8B-Instruct. All scores are L1-normalized across the at-
tention head importance distributions. Unlike traditional

<!-- page 11 -->

Dataset Source Task Type Avg Len Metric Language # Samples
NarrativeQA Literature, Film Single-Document QA 18,409 F1 English 200
Qasper Science Single-Document QA 3,619 F1 English 200
MultiFieldQA-en Multi-field Single-Document QA 4,559 F1 English 150
HotpotQA Wikipedia Multi-Document QA 9,151 F1 English 200
2WikiMultihopQA Wikipedia Multi-Document QA 4,887 F1 English 200
MuSiQue Wikipedia Multi-Document QA 11,214 F1 English 200
GovReport Government report Summarization 8,734 Rouge-L English 200
QMSum Meeting Summarization 10,614 Rouge-L English 200
MultiNews News Summarization 2,113 Rouge-L English 200
TREC Web question Few-shot Learning 5,177 Accuracy (CLS) English 200
TriviaQA Wikipedia, Web Few-shot Learning 8,209 F1 English 200
SAMSum Dialogue Few-shot Learning 6,258 Rouge-L English 200
PassageCount Wikipedia Synthetic Task 11,141 Accuracy (EM) English 200
PassageRetrieval-en Wikipedia Synthetic Task 9,289 Accuracy (EM) English 200
LCC Github Code Completion 1,235 Edit Sim Python/C#/Java 500
RepoBench-P Github repository Code Completion 4,206 Edit Sim Python/Java 500
Table 4: An overview of the dataset statistics in LongBench.
methods that require exact top- k attention hits, our ap-
proach aggregates scores over entire answer spans, captur-
ing heads that contribute semantically relevant context even
when they never achieve top-1 attention for individual to-
kens, thus significantly reducing zero-score heads. For in-
stance, as shown in Figure 11, layers 0 and 1 of the Mis-
tral model have zero scores for all heads using the tradi-
tional method, whereas our approach successfully identifies
heads of lower yet meaningful importance. Likewise, Fig-
ure 12 shows that Llama layer 4 head 16 and layer 26 head
3—missed by the standard criterion—are successfully iden-
tified by our Semantic Retrieval Heads (similar behavior is
observed for Mistral’s layer 7 head 18). These examples
highlight our method’s superior ability to detect Semantic
Retrieval Heads—patterns that traditional approaches miss.
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
Layer Index
0
100
200
300Allocated T okens
Mistral-7B-Instruct-v0.3
Figure 9: Per-layer KV cache allocation for Mistral-7B-
Instruct-v0.3 under a total budget of 256 tokens per layer.
D Comprehensive Results on the LongBench
Dataset
In table 5, we provide the detailed results of Figure 4
in the main paper. Across every KV cache budget, Com-
pressKV outperforms all baseline methods—an advantage
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
Layer Index
0
100
200
300Allocated T okens
Llama-3.1-8B-Instruct
Figure 10: Per-layer KV cache allocation for Llama-3.1-8B-
Instruct under a total budget of 256 tokens per layer.
Figure 11: Head visualization for Mistral-7B-Instruct-v0.3.
Left: Traditional Retrieval Heads. Right: Semantic Retrieval
Heads identified.

<!-- page 12 -->

Llama-3.1-8B-Instruct
Figure 12: Head visualization for Llama-3.1-8B-Instruct.
Left: Traditional Retrieval Heads. Right: Semantic Retrieval
Heads identified.
that becomes especially pronounced under tight memory
constraints (i.e., smaller cache sizes).
E Detailed Results for Needle-in-a-Haystack
Evaluation
This section provides detailed results for the Needle-in-
a-Haystack evaluation referenced in the main paper. Fig-
ures 13–17 present the performance of the Mistral-7B-
Instruct-v0.3 model under KV cache budgets ranging from
128 to 2048. Figures 18–22 present the corresponding re-
sults for the Llama-3.1-8B-Instruct model under the same
cache budgets. CompressKV consistently achieves the high-
est accuracy across all settings, demonstrating its superiority
over competing compression strategies.
Mistral-7B-Instruct-v0.3
Figure 13: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 128.
Mistral-7B-Instruct-v0.3
Figure 14: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 256.
Mistral-7B-Instruct-v0.3
Figure 15: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 512.
Mistral-7B-Instruct-v0.3
Figure 16: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 1024.

<!-- page 13 -->

Method KV Size Single-doc QAMulti-doc QA SummarizationFew-shot LearningSynthetic Code Avg.
Llama-3.1-8B-Instruct
FullKV Full 43.41 44.44 29.22 69.48 52.75 60.06 49.08
StreamingLLM
2048
37.02 33.10 25.76 56.57 38.74 44.51 38.99
SnapKV 42.95 44.01 27.29 69.02 52.75 60.09 48.47
PyramidKV 42.85 44.19 26.93 69.15 53.03 59.01 48.34
CAKE 42.56 43.87 27.45 68.67 52.84 59.45 48.26
CompressKV 43.43 44.17 27.88 69.11 52.75 60.02 48.71
StreamingLLM
1024
31.90 30.83 24.58 53.81 44.39 39.57 36.96
SnapKV 42.82 43.90 26.21 67.91 52.81 58.53 47.82
PyramidKV 42.80 43.86 25.74 68.28 52.79 57.39 47.65
CAKE 42.48 43.82 26.57 68.57 52.84 58.76 47.97
CompressKV 42.96 44.22 26.63 68.72 52.75 59.38 48.24
StreamingLLM
512
29.07 30.11 23.16 50.51 47.10 38.31 35.59
SnapKV 41.03 44.02 24.70 66.09 52.52 57.38 46.71
PyramidKV 41.07 43.95 24.58 66.09 52.79 55.58 46.49
CAKE 41.86 43.38 25.47 67.91 52.92 57.12 47.25
CompressKV 42.78 44.29 25.36 68.67 53.04 57.56 47.78
StreamingLLM
256
26.52 29.73 21.16 47.60 47.06 36.83 33.92
SnapKV 38.84 43.57 23.41 63.40 52.63 55.21 45.21
PyramidKV 37.28 43.41 23.04 62.40 52.38 53.29 44.36
CAKE 41.01 43.30 24.38 66.02 52.82 55.56 46.30
CompressKV 41.84 43.75 24.26 66.52 52.82 56.29 46.71
StreamingLLM
128
25.51 29.46 19.25 43.94 45.23 35.79 32.28
SnapKV 34.84 42.90 21.62 60.40 48.15 52.86 42.58
PyramidKV 33.96 42.74 21.53 59.32 50.25 49.62 42.02
CAKE 39.46 42.47 23.08 63.79 52.67 52.83 44.84
CompressKV 39.10 43.67 22.68 64.16 52.64 53.70 45.10
Mistral-7B-Instruct-v0.3
FullKV Full 41.16 38.99 29.50 70.70 52.00 60.03 47.82
StreamingLLM
2048
34.17 28.72 25.85 53.99 38.50 39.47 36.51
SnapKV 41.21 38.65 26.66 70.18 51.50 59.87 47.05
PyramidKV 40.54 38.69 26.70 70.39 51.50 58.83 46.85
CAKE 41.18 38.32 27.83 70.24 51.50 59.96 47.22
CompressKV 41.28 39.52 27.93 70.58 51.50 59.97 47.55
StreamingLLM
1024
30.54 27.33 24.92 53.62 36.94 36.26 34.73
SnapKV 39.65 38.58 25.39 70.32 51.75 59.22 46.49
PyramidKV 39.42 37.96 25.05 70.18 51.25 57.54 45.96
CAKE 39.76 38.36 26.82 69.96 51.50 59.40 46.66
CompressKV 40.48 39.08 26.70 70.47 51.25 59.35 46.96
StreamingLLM
512
25.96 26.68 23.40 51.71 35.63 33.92 32.65
SnapKV 38.87 37.74 23.66 69.26 51.00 57.74 45.38
PyramidKV 37.57 37.32 23.63 68.85 51.00 56.47 44.82
CAKE 39.73 38.73 25.32 69.18 51.50 57.53 46.06
CompressKV 40.41 38.45 25.10 70.10 51.50 58.53 46.39
StreamingLLM
256
25.26 26.40 20.76 49.37 34.50 32.58 31.22
SnapKV 35.20 37.08 22.35 67.72 51.00 55.59 43.76
PyramidKV 34.73 36.80 21.89 67.66 49.75 53.10 43.06
CAKE 38.29 37.73 24.03 67.81 50.00 56.06 44.73
CompressKV 39.34 38.48 23.56 69.99 50.50 55.89 45.43
StreamingLLM
128
23.47 25.96 18.82 46.08 36.12 31.16 29.85
SnapKV 32.40 36.51 20.54 63.20 45.50 51.85 40.79
PyramidKV 31.91 35.32 20.75 62.48 47.50 49.13 40.29
CAKE 35.88 37.69 22.69 65.09 49.75 52.55 43.04
CompressKV 37.47 37.61 21.96 67.41 49.75 52.01 43.56
Table 5: Details Performance comparison of CompressKV with StreamingLLM, SnapKV , PyramidKV , CAKE, and FullKV
on LongBench for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. CompressKV generally outperforms other KV cache
compression methods across various KV cache sizes, from 128 to 2048 per layer.

<!-- page 14 -->

Mistral-7B-Instruct-v0.3
Figure 17: Needle-in-a-Haystack test results on Mistral-7B-
Instruct-v0.3 with KV cache = 2048.
Llama-3.1-8B-Instruct
Figure 18: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 128.
Llama-3.1-8B-Instruct
Figure 19: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 256.
Llama-3.1-8B-Instruct
Figure 20: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 512.
Llama-3.1-8B-Instruct
Figure 21: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 1024.
Llama-3.1-8B-Instruct
Figure 22: Needle-in-a-Haystack test results on Llama-3.1-
8B-Instruct with KV cache = 2048.

<!-- page 15 -->

F Comprehensive Masking-Based Ablation
of Different Head Types
We extend the masking analysis from the main paper by
evaluating the effect of masking the top 10, 20, and 30 Se-
mantic Retrieval Heads and the traditional Retrieval Heads
in both Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct,
shown in Figure 23. Our experiments demonstrate that
masking the top 30 traditional Retrieval Heads in Mistral-
7B-Instruct-v0.3 results in only a ≈ 12% drop in accu-
racy, whereas masking the top 30 Semantic Retrieval Heads
causes a ≈ 74% degradation. Similarly, in Llama-3.1-8B-
Instruct, masking Semantic Retrieval Heads yields a sub-
stantially larger accuracy loss compared to masking tradi-
tional Retrieval Heads. These findings underscore the criti-
cal role of Semantic Retrieval Heads in overall model per-
formance and validate the superiority of our identification
method over conventional head-selection approaches.
Mistral-7B-Instruct-v0.3
Llama-3.1-8B-Instruct
Figure 23: Ablation on the Needle-in-a-Haystack re-
trieval task for Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-
Instruct. The left column masks the top-k retrieval heads,
and the right column masks the top-k semantic retrieval
heads. Lower scores indicate heads with the greatest impact
on model performance—masking them causes the most se-
vere drop in accuracy.
