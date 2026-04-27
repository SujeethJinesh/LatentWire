# references/114_lava_layer_wise_dynamic_budget_allocation_for_kv_cache_compression.pdf

<!-- page 1 -->

Findings of the Association for Computational Linguistics: EMNLP 2025, pages 13672–13692
November 4-9, 2025 ©2025 Association for Computational Linguistics
LA Va: Layer-wise KV Cache Eviction with Dynamic Budget Allocation
Yiqun Shen12 Song Yuan3 Zhengze Zhang12 Xiaoliang Wang1
Daxin Jiang3 Cam-Tu Nguyen12*
1State Key Laboratory for Novel Software Technology, Nanjing University
2School of Artificial Intelligence, Nanjing University 3Stepfun
yiqunshen@smail.nju.edu.cn
ncamtu@nju.edu.cn
Abstract
KV Cache is commonly used to accelerate
LLM inference with long contexts, yet its high
memory demand drives the need for cache com-
pression. Existing compression methods, how-
ever, are largely heuristic and lack dynamic
budget allocation. To address this limitation,
we introduce a unified framework for cache
compression by minimizing information loss
in Transformer residual streams. Building on
it, we analyze the layer attention output loss
and derive a new metric to compare cache en-
tries across heads, enabling layer-wise com-
pression with dynamic head budgets. Addition-
ally, by contrasting cross-layer information, we
also achieve dynamic layer budgets. LA Va is
the first unified strategy for cache eviction and
dynamic budget allocation that, unlike prior
methods, does not rely on training or the com-
bination of multiple strategies. Experiments
with benchmarks (LongBench, Needle-In-A-
Haystack, Ruler, and InfiniteBench) demon-
strate its superiority. Moreover, our experi-
ments reveal a new insight: dynamic layer bud-
gets are crucial for generation tasks (e.g., code
completion), while dynamic head budgets play
a key role in extraction tasks (e.g., extractive
QA). As a fully dynamic compression method,
LA Va consistently maintains top performance
across task types. Our code is available at
https://github.com/MGDDestiny/Lava.
1 Introduction
Large language models (LLMs) have shown re-
markable capability in handling long-text scenarios,
enabling advancements in tasks such as question
answering (Kamalloo et al., 2023), code genera-
tion (Guo et al., 2023), and multi-turn dialogues
(Chiang et al., 2023). To further enhance external
knowledge integration, state-of-the-art models like
Claude 3.5 (Anthropic and et al.), GPT-4 (OpenAI
and et al., 2024), and Qwen2.5 Max (Qwen and
*Corresponding author
et al., 2025) have extended their context lengths
beyond 128K tokens. However, supporting such
long contexts comes with increased computational
challenges. One common approach to accelerating
LLM inference is caching Key and Value vectors
(KV Cache), but its high memory demand necessi-
tates efficient cache compression techniques.
While existing compression methods have
shown promise, they are largely heuristic, relying
on statistical measures such as accumulated atten-
tion scores (Zhang et al., 2023; Li et al., 2024).
These metrics are derived from empirical observa-
tions rather than a theoretical foundation. Addi-
tionally, although dynamic head allocation (Feng
et al., 2024) and dynamic layer allocation (Qin
et al., 2025) have been explored, no method, to our
knowledge, fully adapts head and layer budgets.
To address this gap, we propose a unified frame-
work for cache compression, which is formulated
through the lens of minimizing information loss
in Transformer residual streams (see Figure 1, and
Sec. 3). Many existing methods can be formulated
within our framework. Specifically, context com-
pression methods (Qin et al., 2024a,b) aim to mini-
mize global information loss at the logits layer. KV
Cache compression methods (Zhang et al., 2023;
Cai et al., 2024; Qin et al., 2025) primarily focus
on local information loss at the head or layer levels.
Our framework provides a principled approach
to designing new algorithms. This paper introduces
a novel method based on Layer Attention Output
Loss, which measures the impact of compression
on the information retained in each layer after
multi-head attention. The layer-wise loss function
provides a balanced perspective on both local infor-
mation within layers and global information flow
across layers. Within each layer, the loss function
guides the design of a scoring mechanism to assess
token importance across heads, allowing for simul-
taneous head budget allocation and cache eviction.
Across layers, it enables dynamic layer budget al-
13672

<!-- page 2 -->

Current Step
(the -th Residual Stream)
Expected Last Step
(Last Residual Stream)
Unembedding
Layer
Next Token
Head1
Head2
Head3
MHA
Logit Loss
Layer Attention
Output Loss
Head Attention
Output Loss
Head Attention
 Loss
FFN
Layer Output
Loss
Layer L
Layer 1
Layer 2
Figure 1: Information flow in decoder-only LLMs. The decoding process can be seen as operating on the current
residual stream. Each residual stream (red lines) corresponds to one token, and is considered as a communication
channel. Attention heads copy information from past residual streams to the current one (green lines) .
location by comparing information between layers.
Our method is theoretically grounded, and signifi-
cantly simpler than CAKE, the only training-free
method with dynamic layer budgets.
Extensive experiments were conducted using var-
ious LLM series on the LongBench and Needle in
a Haystack benchmarks. The results consistently
demonstrate LA Va’s strong ability to preserve the
model’s long-text comprehension under various
memory constraints. Additionally, compared to
a full cache implementation of FlashAttention-2,
LA Va significantly reduces memory consumption
while simultaneously reducing latency (9×faster
decoding for 128K-token sequences). Our empiri-
cal findings highlight that dynamic layer budgets
are essential for generation tasks, while dynamic
head budgets are crucial for text extraction tasks.
Achieving dynamic budget allocation at both the
head and layer levels is key to optimizing perfor-
mance across different tasks.
Our Contributions: 1) We introduce aprincipled
framework for KV Cache eviction by analyzing
the information flow through Transformer residual
streams, accounting for information loss at var-
ious points during decoding. 2) Building on this
framework and the notion of information loss at the
layer-wise attention output, we propose LA Va—a
unified method that simultaneously performs KV
cache eviction and dynamic budget allocation. To
the best of our knowledge, LA Va is the first training-
free method to achieve dynamic budget allocation
without relying on multiple combined metrics, mak-
ing it simple for practical purposes. 3) Evaluations
on LongBench, Needle in a Haystack, Ruler and
InfiniteBench demonstrate that our simple method
outperforms strong baselines . 4) Experiments
reveal new insights into the role of dynamic budget
allocation across different tasks, offering guidance
for the adaptive selection of strategies.
2 The Information Flow of LLM
Decoding Process with KV Cache
KV Cache is initialized at prefilling stage, which
basically computes the Key and Value for tokens
in the initial prompts in the standard way (Vaswani,
2017). In the following, we assume that there ex-
ists a KV Cache of (N−1) previous tokens and
demonstrate how decoding is performed at step-N.
Notations The LLM has L layers, each has H
heads. The model and head dimensions ared and
dh = d/H;Kl,V l are the KV Cache for the l-th
layer up to the current time step (theN-th token),
which are of [H, (N−1),d h] sizes. The full nota-
tion Table 3 is in Appendix A.
Decoding Process According to (Ferrando and
V oita, 2024), LLM decoding can be viewed as op-
erating on the current (N-th) residual stream, as
illustrated in Figure 1. Specifically, suppose that
xN
l is the current input for layerl, we first calculate
13673

<!-- page 3 -->

the correspondingQN
l ,K N
l ,V N
l as follows:
QN
l =xN
l W Q
l ;KN
l =xN
l W K
l ;V N
l =xN
l W V
l
whereQN
l ,K N
l ,V N
l are of size (H×1×dh), con-
tainingH head-wise caches. The layer-wise KV
Cache is then updated as follows:
Kl =Cat[Kl,K N
l ],V l =Cat[Vl,V N
l ]
whereKl,V l are tensors of size (H×N×dh), and
Cat indicates the concatenation operation.We then
calculate the attention scores of step-N for layer-l:
AN
l =Cath∈[H]
(
AN
l,h
)
where AN
l,h = Softmax (
QN
l,h(Kl,h)T
√dh
). Here,
AN
l,h[i] indicates how much the token at step- N
attends to the token-i (i≤N). Layer-l attention
output is calculated as follows:
yN
l =Cath∈[H](AN
l,hVl,h)W O
l ∈R1×d
The layer output xN
l+1 is calculated as xN
l+1 =
yN
l +FFN (yN
l ), which is then passed as the input
the next layer l + 1. In the last layer, we exploit
an un-embedding layer (W M∈Rd×|V|) to get the
probability vectorpN for next token sampling.
3 A Principled Framework for KV Cache
Eviction based on Information Loss
Given the KV Cache, compression can be seen
as masking entries in the KV tensors so that the
attention heads cannot copy masked information to
the later residual streams. Formally, one can define
the attention maskIl,hfor layer-l and head-h:
Il,h[i] =
{
1 ifKl,h[i] andVl,h[i] are retained
0 evictKl,h[i] andVl,h[i]
The goal is to find a KV Cache eviction policy so
that to minimize the information loss for the logits
at the last layer (pN) for all subsequent residual
streams (fromN toNe; see Figure 1). LetP denote
this logit loss, andB be the memory constraint. The
unified problem for budget allocation and cache
eviction can be defined as follows:
min
I,B
P(x1...N
1 ,I,B) (1)
st.
∑
i∈[N ]
Il,h[i] =Bl,h;
∑
h∈[H]
Bl,h=Bl;
∑
l∈[L]
Bl = B
Il,h[k] = 1,∀l,h ; and∀k∈[N−w,N ]
Here,Bl,hrepresents the budget for layer-l and
head-h,Bl denotes the total budget for layer-l. The
final constraint ensures that the most recent tokens
within a window of sizew are retained for all heads,
aligning with the common practice in the literature.
As computing the loss over future, unseen tokens
is impractical. To address this, we approximate the
loss by considering only residual streams up to the
current step N. Considering the current step- N,
one can defineP as the cross-entropy loss between
pN and ˆpN, which is the logit obtained with the at-
tention mask (Qin et al., 2024a). Additionaly, since
the search space for the mask matrix is combina-
torial, we instead search for a scoring function s,
wheresl,h[i] assigns an importance score to tokeni
at layerl and headh. This scoring function allows
us to greedily choose the least important entries to
be maskedI = Select(s,B). All in all, we have
the following (surrogate) optimization problem:
min
B,s∈F
P(x1...N
1 ,s, B) (2)
whereF denotes the space of all scoring functions.
The scoring function can be parameterized by a
network ϕ, which is then found through offline
training. This is the common approach employed in
context compression methods (Qin et al., 2024a,b).
The aforementioned approach to minimizing
Global Logit Loss can be impractical for online
inference when the scoring function is computa-
tionally expensive. A more feasible alternative is
to focus on local information and apply localized
KV Cache eviction. For instance, Head Attention
Loss can be used for head-wise eviction, a strategy
adopted by most existing methods (Zhang et al.,
2023; Li et al., 2024; Qin et al., 2025). In this case,
the scoring functions are lightweight, relying on
simple statistical features, like head-wise attention
weights. Table 1 summarizes how existing meth-
ods can be formalized within our framework, with
further details provided in Appendix B.
4 LA Va: Layer-wise Cache Eviction with
Dynamic Budget Allocation
4.1 Layer Attention Output Loss and the
Scoring Function
The aforementioned framework provides a prin-
cipled approach to designing new algorithms for
KV Cache eviction. This section demonstrates the
design of our novel algorithm based on Layer At-
tention Output Loss (see Figure 1). Specifically, we
13674

<!-- page 4 -->

Methods Budgets Scoring Function Loss
Bl,h Bl
SnapKV (Li et al., 2024) Bl/H B/L Recent attention scores
Head Attention
sl,h[i] = 1
w
∑N
j=N−wAj
l,h[i],∀i<N −w
CAKE (Qin et al., 2025) Bl/H Dynamic Recent attention scores + attention shifts
sl,h[i] = γV ARN
j=N−w([Aj
l,h[i]))
+ 1
w
∑N
j=N−wAj
l,h[i],∀i<N −w
AdaKV (Feng et al., 2024) Dynamic Fixed Recent attention scores (like SnapKV) Layer Attention
OutputLA Va (Ours) Dynamic Dynamic Recent attention scores×value norm
sl,h[i] =
maxk∥Vl,h[k]∥1
w
∑N
j=N−wAj
l,h[i]
Table 1: Summary of representative methods for KV Cache compression. LA Va is the only method to support
dynamic head (Bl,h) and layer (Bl) budgets. For the full table and more comparison, please refer to Appendix B.
.
show how our scoring function is designed based
on analyzing the upper bound of the loss and how
we can exploit the scoring function for layer-wise
cache eviction with dynamic budget allocation.
Lemma 1. Based on the Lp norm, the layer at-
tention output loss due to the attention maskI is
measured for layer-l at the current (N-th) residual
stream as follows:
P(x1...N
1 ,I,B) =∥yN
l −ˆyN
l ∥p (3)
=
Cath
[(
AN
l,h−
AN
l,h⊙Il,h
∥AN
l,h⊙Il,h∥1
)
Vl,h
]
W O
l

p
where⊙indicates element-wise multiplication and
ˆyN
l indicates the layer attention output obtained by
masking the KV Cache withI (equivalently, after
KV Cache eviction).
We then develop a new upper bound for theL1
norm and provide the result in Theorem 1. The
proof of these are both provided in Appendix C.
Theorem 1. TheL1 norm of the layer attention
output loss can be bounded by:
(4)
∥yN
l −ˆyN
l ∥1
≤2 ˆC
∑
h∈[H]
∑
i∈[N ]
AN
l,h[i] ¯Vl,h(1−Il,h[i])
where ˆC = ∥W O
l
T∥1 is a constant indepen-
dent of any head or token within layer- l; ¯Vl,h =
maxk∈[N ]∥Vl,h[k]∥1 is a head-dependent value.
Given a fixed budgetBl, we consider a greedy
algorithm that iteratively evicts one cache entry at a
time until the cache budget is met. We evict the en-
tries with the smallest scores, given by the scoring
functionsl,h[i] =AN
l,h[i] ¯Vl,hto minimize the upper
bound. Notably, this function incorporates a head-
dependent value ¯Vl,h, which should not be ignored
when comparing KV Cache entries across different
heads. This is different from AdaKV (Feng et al.,
2024), which considers the layer attention output
loss yet does not take into account the values. This
also provides a theoretical justification for the intro-
duction of values into the scoring, which has been
exploited heuristically in V ATP (Guo et al., 2024).
4It is noted that we derive our metric through a de-
tailed reasoning process, independently from V ATP.
The process is key to understanding the approxima-
tions we introduce, which enable future improve-
ments. Moreover, recognizing that the metric is
inherently grounded in a layer-wise perspective
enables the design of dynamic budget allocation
strategies, as demonstrated below. Empirical com-
parison to V ATP is given in Table 5.
The scoring function sl,h[i] = AN
l,h[i] ¯Vl,h de-
scribed earlier is based solely on analyzing the cur-
rent residual stream (theN-th decoding step). To
improve the performance for KV Cache eviction,
we can incorporate information from all past resid-
ual streams similarly to H2O (Zhang et al., 2023).
However, doing so introduces more computational
overhead. Inspired by SnapKV (Li et al., 2024),
we instead incorporate information from recentw
residual streams, yielding a new scoring function.
Definition 1. Layer-wise Attention and Value
(LAVa) score for the token-i at layer-l, head-h is
defined as follows:
sl,h[i] = maxk∈[N ]∥Vl,h[k]∥1
w
N∑
j=N−w
Aj
l,h[i] (5)
Based on this scoring function, we develop the
layer-wise KV Cache eviction as outlined in Al-
gorithm 1. Notably, we only evict entries outside
13675

<!-- page 5 -->

Algorithm 1 LayerEvict: Layer-wise KV Cache
Eviction based on LA Va Score
1: Input: BudgetBl, KV CacheKl,V l
2: Output: Compressed KV Cache ˆKl, ˆVl
3: sl = [ ]
4: forh = 1 toH do
5: Calculatesl,h[i],∀i /∈[N−w,N ] based
on Eq. 5
6: sl.extend(sl,h)
7: end for
8: function EVICT (Bl,s l,K l,V l)
9: Sl←Bl largest entries based onsl
10: Il,h[k] = 0,∀(h,k ) /∈Sl
11: forh = 1 toH do
12: ˆKl,h=Kl,h⊙Il,h
13: ˆVl,h=Vl,h⊙Il,h
14: end for
15: Return ˆKl, ˆVl
16: end function
17: Return EVICT (Bl,s l,K l,V l)
the recent window [N−w,N ], effectively retain-
ing the most recent tokens as specified by the final
constraint in the optimization problem (Eq. 1).
Dynamic Head Budget. Our eviction method op-
erates across attention heads within layer-l. Specif-
ically, we flatten the LA Va scores from all heads
in the layer into a one-dimensional arraysl (Algo-
rithm 1, lines 3–6). We then compare and rankBl
cache entries across all heads for layer-wise evic-
tion, effectively obtaining dynamic head budget
while performing eviction.
4.2 Layer Budget Allocation
Recently, CAKE (Qin et al., 2025) and PyramidKV
(Cai et al., 2024) have demonstrated the potential
of allocating different budgets across layers. Pyra-
midKV , however, is suboptimal as it assigns a fixed
allocation pattern regardless of the input. In con-
trast, CAKE is prompt-dependent allocation (dy-
namic) but combines different scores for cache evic-
tion and budget allocation, which requires tuning
three hyperparameters, hindering its practical ap-
plication. Below, we describe our hyperparameter-
free algorithm based on the LaVa score.
Our key idea is that layers with greater uncer-
tainty in determining which cache entry to evict
should be allocated a larger budget. Specifically,
based on the LA Va score,the probability of evict-
ing token-k at layer-l and head-h is obtained by
Algorithm 2 LA Va: Dynamic Budget Allocation
and Cache Eviction based on LA Va Score
1: Input: Total Budget B, KV CacheK,V Num-
ber of LayersL
2: Output: Compressed KV Cache ˆK, ˆV
3: s = [ ],e = [ ], ˆK =K, ˆV =V
4: forl = 1 toL do
5: Calculatesl based on Eq. 5
6: Calculateel based on Eq. 6, 7
7: s.append(sl)
8: e.append(el)
9: for ˜l = 1 tol do
10: B˜l = e˜l∑
l el
B
11: ˆK˜l, ˆV˜l = EVICT (B˜l,s ˜l, ˆK˜l, ˆV˜l)
12: end for
13: end for
14: Return ˆK, ˆV
normalizing the LA Va scoring values:
ˆsl,h[i] = sl,h[i]∑
k,hsl,h[k] (6)
The uncertainty for layer-l is then measured by the
normalized entropy as follows:
el =
−∑
h,i(ˆsl,h[i] log ˆsl,h[i])
H×N (7)
With such a measure, we can first initialize all
KV Cache through prefilling, followed by cache
compression. Unfortunately, this approach results
in a high memory peak after prefilling (and before
compression). To address this, the common prac-
tice is that we perform prefilling and cache eviction
layer by layer. For dynamic layer budget allocation,
we draw inspiration from CAKE: after prefilling
layer-l, the lower layers (< l) are recompressed.
As a result, a lower layer is compressed multiple
times using the same LA Va scores, but the budget is
adjusted, becoming smaller over time as the mem-
ory is shared with more layers being prefilled. The
complete algorithm is outlined in Algorithm 2.
4.3 LLMs with GQA
Group Query Attention (GQA) (Ainslie et al.,
2023) is the technique most modern LLMs adopt
due to its balance between performance loss and
memory efficiency. In GQA, the KV Cache is
compressed by sharing a single KV Cache among
all heads within a group. When applying LA Va
scores to GQA, we take a conservative approach:
13676

<!-- page 6 -->

the group-wise score for a token is determined as
the maximum of its head-wise scores within the
corresponding group. In other words, we tend to
retain the entry as long as it is important for at least
one head within the group.
5 Experiments
5.1 Experimental Settings
Backbone LLMs. We evaluate three series of
LLMs: Mistral-7B-Instruct-v0.2 (Jiang et al.,
2023), Qwen2.5-7/14/32B-Instruct (Qwen and
et al., 2025), all with a context length of 32k and
Llama3-8B-Instruct with 8k context length. These
models are widely adopted for their moderate pa-
rameter sizes and strong performance all utilizing
GQA (Ainslie et al., 2023).
Evaluation Benchmarks. To validate the effec-
tiveness of our algorithm, we perform evaluation
LongBench (Bai et al., 2024), a bilingual, multi-
task benchmark for long-context understanding. It
comprises 21 datasets across six task categories
in both English and Chinese, with an average
length of 6,711 words (English) and 13,386 char-
acters (Chinese). LongBench covers key long-
text application areas, including single-document
QA, multi-document QA, summarization, few-shot
learning, synthetic tasks, and code completion.
We also conduct experiments on Needle In A
Haystack (Cai et al., 2024; Liu et al., 2024; Fu
et al., 2024), Ruler (Hsieh et al., 2024) and In-
finiteBench (Zhang et al., 2024), of which the re-
sults are given in Appendix D.
Baseline Methods. We compare our meth-
ods against several baselines: PyramidKV ,
SnapKV , Ada-SnapKV , Ada-PyramidKV , and
CAKE. Among these, PyramidKV and CAKE al-
low different layer budgets. AdaKV is derived
from the layer attention output loss but relies solely
on attention for its scoring function and does not
incorporate dynamic layer budget allocation. Ada-
SnapKV employs the same scoring function and
uniform layer allocation as SnapKV but allows dy-
namic head budgets. Ada-PyramidKV follows the
same approach but assigns fixed, varying budgets
across layers like PyramidKV .
Pooling operators, such as max pooling or aver-
age pooling, can be applied to token score vectors
to smooth score variations across adjacent tokens
(Li et al., 2024; Cai et al., 2024; Qin et al., 2025).
This strategy is also employed in the implemen-
tation of LA Va and all the baselines. For pooling
operation, for all methods, we adopt maxpool func-
tion and set kernel size as 7. More information
is given in Appendix B, and for implementation
details, please refer to Appendix D.
5.2 Main Results
Table 2 presents the results of Mistral-7B with dif-
ferent eviction policies on LongBench, revealing
several key observations. First, LAVa outperforms
all baselines across different budgets, with a more
pronounced advantage at smaller budgets. Sec-
ond, among methods requiring no hyperparameter
tuning (SnapKV , Ada-SnapKV , and LA Va), LA Va
achieves the best performance, significantly sur-
passing others. For instance, at B = 128HL, LA Va
achieves an average score of 36.74, compared
to Ada-SnapKV’s 35.82. And finally, LAVa and
CAKE excel in code-related tasks. On RepoBench-
P with a 128HL budget, LA Va (48.92) and CAKE
(48.53) outperform Ada-SnapKV (46.85) by a sig-
nificant margin. This is interesting given that
Ada-SnapKV surpasses CAKE on average over
20 datasets. Similar trends are observed with the
Qwen series and presented in Appendix D.
To further investigate the last observation, we cat-
egorize the 20 LongBench datasets into two types:
extraction tasks, which require extracting answers
from the context (e.g., QA tasks evaluated with F1
or Accuracy), and generation tasks (e.g., summa-
rization and code completion). For each category,
we then compute the average scores obtained with
Qwen and Mistral under varying cache budgets and
eviction policies. Figure 2 highlights several key
findings: 1) Extraction tasks are generally less af-
fected by compression, as LLM performance with
a compressed cache remains closer to that with
a full cache; 2) The performance gap among dif-
ferent eviction policies is greater on generation
tasks.; 3) CAKE and LAVa outperform Ada-SnapKV
and methods with fixed-layer budgets on genera-
tion tasks, though CAKE performs significantly
worse than Ada-SnapKV on extraction tasks with
Mistral-7B. This suggests the importance of (dy-
namic) layer budget allocation for generation tasks.
LA Va, however, consistently achieves top perfor-
mance across both task types and language models.
5.3 Evaluation of Latency and Memory Peak
We evaluate LA Va’s efficiency during LLM infer-
ence by analyzing peak memory usage and de-
coding latency on Mistral-7B-Instruct-v0.2, imple-
13677

<!-- page 7 -->

Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Full Cache26.77 32.34 49.63 48.42 43.43 27.89 18.61 30.85 32.92 24.54 15.04 27.20 71.00 86.23 43.41 39.00 2.81 86.56 89.75 55.29 52.5545.07
B= 128HLPyramidKV20.01 19.23 43.81 32.37 35.62 22.34 14.38 17.53 18.95 21.91 11.07 20.87 47.0085.34 40.2119.25 2.86 65.60 59.49 49.52 45.6734.51SnapKV 20.99 19.6545.0432.02 36.48 22.19 14.04 17.68 18.83 21.36 10.91 20.29 45.00 84.10 40.01 19.75 3.06 64.48 60.50 49.84 45.2734.42Ada-PyramidKV20.2120.80 43.82 33.6537.21 22.99 14.9318.0619.4122.02 11.16 20.9752.00 83.93 39.97 20.00 2.8172.7372.89 51.00 46.6236.22Ada-SnapKV20.61 20.56 44.0334.0336.3923.6616.1517.82 19.21 21.7311.25 20.35 50.00 84.32 39.82 19.75 3.87 69.11 70.52 50.21 46.8535.82CAKE 21.01 20.16 44.08 32.52 36.1623.8915.32 17.67 18.8222.6210.9321.03 47.00 85.14 39.9021.25 3.02 63.65 65.9651.8148.5335.06LA Va (Ours)19.5721.1144.2933.9138.2923.5915.3218.5619.3322.3211.42 21.07 53.5085.2040.1621.752.8869.8774.75 51.94 48.9236.74
B= 256HLPyramidKV20.79 22.74 45.90 35.72 38.63 24.02 15.97 18.99 21.61 22.34 11.02 22.24 58.00 84.06 40.52 22.75 2.96 74.70 83.83 51.85 48.8638.23SnapKV 21.39 22.15 46.50 34.7739.6825.01 14.86 19.11 21.6123.04 11.46 22.67 57.00 85.04 40.81 23.25 3.18 76.49 83.60 51.99 49.4238.49Ada-PyramidKV22.6123.8447.65 36.5639.33 24.8617.2219.65 21.22 22.5411.82 22.2964.00 84.93 40.36 24.50 3.40 77.3985.83 52.48 49.4339.43Ada-SnapKV21.63 23.55 47.5137.42 38.89 23.65 16.06 19.3421.98 23.2111.49 22.3964.0086.3340.5425.25 2.2377.4485.42 52.31 49.6239.40CAKE 21.37 23.40 46.84 35.02 38.10 24.50 14.81 19.40 21.59 22.77 11.3222.68 55.0085.4641.9224.75 2.96 75.6686.46 54.2951.3838.84LA Va (Ours) 22.70 24.67 48.62 37.81 39.68 25.9616.7720.2621.92 22.4811.88 22.91 65.0085.2441.2826.752.8876.76 85.7554.1751.7740.12
B= 512HLPyramidKV23.57 24.84 48.74 39.54 38.90 25.22 17.40 20.42 23.04 23.24 11.91 24.19 66.50 86.07 41.06 28.00 3.29 87.29 88.83 53.77 50.4241.15SnapKV 23.6728.0849.40 40.2540.14 25.58 16.97 20.4923.75 23.6912.03 24.31 65.00 86.29 41.98 28.50 3.22 85.79 88.67 53.99 51.0241.48Ada-PyramidKV24.37 27.30 48.01 40.88 39.75 25.9618.5820.90 23.59 23.33 12.07 24.0467.5086.44 42.5831.50 3.38 85.8889.67 54.15 51.3041.89Ada-SnapKV24.63 27.48 48.9041.28 39.8426.33 18.2620.91 23.59 23.5112.27 24.3267.5086.38 42.3432.50 2.9887.6589.17 54.39 51.0342.11CAKE 22.76 27.5449.4741.27 38.17 25.85 17.26 20.6023.7223.65 11.9524.50 66.00 86.0142.56 29.50 3.45 86.79 88.7556.4052.3741.76LA Va (Ours) 25.0127.84 48.9742.14 40.95 26.8818.3321.1223.59 23.5912.28 24.51 68.5086.3442.4833.502.9087.2389.8355.8352.8542.59
B= 1024HLPyramidKV 25.6228.96 48.35 42.18 40.89 26.6519.69 21.96 25.10 23.57 12.58 25.42 68.5086.3041.92 35.50 2.98 86.7789.50 55.26 51.0342.79SnapKV 24.80 30.1749.1343.2341.16 26.92 17.89 22.58 25.75 23.64 12.88 25.85 67.5086.25 42.56 36.00 2.88 88.10 88.92 55.23 51.3843.00Ada-PyramidKV24.98 29.92 47.97 41.43 40.83 26.98 19.42 22.45 25.46 23.58 12.94 25.61 68.5086.30 42.8435.50 2.89 88.18 89.25 54.51 51.3242.90Ada-SnapKV24.84 29.9949.2142.55 41.0027.3919.2323.2325.8924.1813.13 25.8569.00 86.2342.8436.25 2.9089.02 89.7555.38 51.9343.34CAKE 25.1530.34 49.00 43.08 40.86 26.7019.9323.07 25.82 23.7213.1626.0568.0086.2542.70 36.00 2.9188.60 88.7556.7553.2643.36LA Va (Ours)25.5931.2148.2743.43 41.9227.38 19.4823.48 26.0623.8613.3826.0070.0086.22 42.4338.002.73 87.01 88.7557.31 53.2843.65
Table 2: Final comparison based on Mistral-7B-Instruct-v0.2 among 21 datasets of LongBench. (Note: The best
result is highlighted in bold, and the second is in underline. Due to the negligible numerical values obtained from
the passage count dataset, its results were excluded from the computation of the average scores.)
Figure 2: Results of generation and extraction tasks.
mented with FlashAttention-2 (Dao, 2023). Our
comparison includes Full Cache, SnapKV , Ada-
SnapKV and CAKE, all using allocation budget
1024HL. We set input at varying lengths while
keeping the output length fixed at 128.
Decoding Latency. By analyzing the decoding
latency in Figure 3, we observe that our scor-
ing function and dynamic budget allocation intro-
duce negligible decoding cost, achieving over a 9×
speedup compared to Full Cache at a 128K context
length. Notably, our method is easier to deploy
than PyramidKV , Ada-PyramidKV , and CAKE, as
these baselines require parameter tuning.
Figure 3: Peak memory usage and decoding latency in
A800 80GB based on Mistral-7B-Instruct-v0.2.
Peak Memory Usage. The peak memory usage
of all methods generally increases with context
length due to prefilling. Our method effectively
maintains peak memory at a reasonable level, par-
ticularly compared to Full Cache, which encoun-
ters OOM issues at higher context lengths. CAKE
and LA Va, both employing dynamic layer budgets,
generally have slightly higher peak memory usage.
Compared to CAKE, LA Va requires additional stor-
age for the norms of head-wise value vectors, but
this extra memory overhead remains minimal.
Theoretical Analysis. We provide the theoretical
analysis of time complexity and memory usage in
Appendix D. The time complexity and peak mem-
ory usage of SnapKV is O(HN (Nd h +wdh +
logBl,h)) andO(HNd h +LHB l,hdh), while that
of LA Va isO(HN (Nd h +wdh +dh +logBl)
andO(HNd h +LHB l,hdh +LHB l,hdh). Setting
13678

<!-- page 8 -->

Figure 4: Ablation study on LongBench.
context lengthN as 10,000, head budget Bl,has
1024, the extra computation of LA Va compared to
SnapKV is 0.01% and the extra memory usage is
0.6%, which is consistent with Figure 3.
5.4 Further Analysis
Dynamic Budget Allocation To examine the im-
pact of dynamic budget allocation, we introduce
two modifications: LA Va (-layer dynamic), which
enforces a uniform layer budget of B/L, and LA Va
(-head dynamic), which fixes the head budget at
Bl/H after dynamically determining the layer bud-
getBl, performing head-wise cache eviction with-
out cross-head comparisons. Results in Figure 4
demonstrate that dynamic budget allocation at both
the head and layer levels is essential for perfor-
mance. Furthermore, it reinforces the finding that
dynamic layer budgets are essential for generation
tasks, whereas dynamic head budgets play a crucial
role in text extraction tasks. Detailed results are
provided in Appendix D, where we also analyze the
influence of different layer allocation approaches.
Analysis of LA Va Score. To validate the effec-
tiveness of LA Va score, we replace our dynamic
layer budgets with fixed ones with PyramidKV
or Uniform allocation. For different total bud-
gets, we then compare LA Va-Pyramid with Ada-
PyramidKV and LA Va-Uniform with AdaKV on
LongBench. For each comparison, we count the
number of tasks in LongBench where one method
outperforms the other. Figure 5 presents the final
winning rates. The results show that our scoring
function yields a significantly higher number of
wins in most cases, validating its effectiveness.
6 Related Work
Recently, various KV Cache compression methods
have been proposed, leveraging different policies
such as recency (Xiao et al., 2024), accumulated
attention scores (Zhang et al., 2023), last-token
attention scores (Oren et al., 2024), and recent at-
tention scores (Li et al., 2024; Dai et al., 2024).
Figure 5: LaVa score vs AdaKV score on LongBench.
While most approaches assume a uniform budget,
recent efforts have been made for dynamic bud-
get allocation across layers (Qin et al., 2025) and
heads (Feng et al., 2024). Some methods aim at
layer-dependent budgets but fix the patterns across
all samples (Cai et al., 2024; Yang et al., 2024). In
general, KV Cache eviction and budget allocation
are typically treated as separate problems, requir-
ing a combination of independent strategies. In
contrast, we develop a principled framework based
on information loss in the residual stream and pro-
pose a unified method for both cache compression
and dynamic budget allocation.
Closely related to LA Va is (Feng et al., 2025,
2024), which aims at minimizing the layer output
perturbation. However, this study only applies the
derived metric locally for head budget allocation.
In contrast, we propose a metric for layer-wise
cache eviction with dynamic layer budgets.
7 Conclusion
This paper provided a comprehensive of current
KV Cache compression into a unified framework,
grounded in the principle of minimizing informa-
tion loss in Transformer residual streams. By
analyzing the Layer Attention Output Loss , we
proposed LA Va, a novel layer-wise compression
method that enables fully dynamic head and layer
budget allocation. Our experiments demonstrate
that dynamic layer budgets are crucial for gener-
ation tasks , whereas dynamic head budgets are
important for extraction tasks. As a fully dynamic
compression method, LA Va consistently maintains
top performance across task types and LLM archi-
tectures, while achieving the same speedup of 9×
with 128K context length compared to full cache.
Future directions include exploring new com-
pression algorithms based on our framework, as
well as extending our framework for model com-
pression. By advancing efficient methods for
LLMs, our work contributes to making LLM more
accessible and scalable for diverse applications.
13679

<!-- page 9 -->

Limitations
There are several limitations to our work. While
we propose a unified framework with multiple opti-
mization opportunities, our theoretical analysis and
experiments focus on only one direction. Although
LA Va’s simplicity is a key advantage, other ap-
proaches should be explored to further close the per-
formance gap with a full-cache setup, particularly
for generation tasks. Additionally, further research
is needed to better understand why dynamic layer
budget is crucial for generation tasks. Lastly, apart
from FlashAttention-2 (Dao, 2023), our method
has not yet been integrated into other widely used
inference frameworks, such as vLLM (Kwon et al.,
2023). We believe that such integration is essential
for broader adoption and real-world deployment of
our algorithm.
Acknowledgment
This work was partially supported by NSFC
62172204.
References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebron, and Sumit Sanghai.
2023. GQA: Training generalized multi-query trans-
former models from multi-head checkpoints. In The
2023 Conference on Empirical Methods in Natural
Language Processing.
Anthropic and et al. The claude 3 model family: Opus,
sonnet, haiku.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu
Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao
Chang, Junjie Hu, et al. 2024. Pyramidkv: Dynamic
kv cache compression based on pyramidal informa-
tion funneling. arXiv preprint arXiv:2406.02069.
Guanzheng Chen, Xin Li, Michael Shieh, and Lidong
Bing. 2025. LongPO: Long context self-evolution of
large language models through short-to-long prefer-
ence optimization. In The Thirteenth International
Conference on Learning Representations.
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion
Stoica, and Eric P. Xing. 2023. Vicuna: An open-
source chatbot impressing gpt-4 with 90%* chatgpt
quality.
Jincheng Dai, Zhuowei Huang, Haiyun Jiang, Chen
Chen, Deng Cai, Wei Bi, and Shuming Shi. 2024.
Corm: Cache optimization with recent message
for large language model inference. Preprint,
arXiv:2404.15949.
Tri Dao. 2023. Flashattention-2: Faster attention with
better parallelism and work partitioning. arXiv
preprint arXiv:2307.08691.
Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and
S Kevin Zhou. 2024. Ada-kv: Optimizing kv cache
eviction by adaptive budget allocation for efficient
llm inference. arXiv preprint arXiv:2407.11550.
Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and
S Kevin Zhou. 2025. Identify critical kv cache in
llm inference from an output perturbation perspec-
tive. Preprint, arXiv:2502.03805.
Javier Ferrando and Elena V oita. 2024. Information flow
routes: Automatically interpreting language models
at scale. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 17432–17445, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Han-
naneh Hajishirzi, Yoon Kim, and Hao Peng. 2024.
Data engineering for scaling language models to
128k context. In Proceedings of the 41st Interna-
tional Conference on Machine Learning, ICML’24.
JMLR.org.
Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Ju-
lian McAuley. 2023. Longcoder: A long-range pre-
trained language model for code completion. In In-
ternational Conference on Machine Learning.
Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe.
2024. Attention score is not all you need for token im-
portance indicator in KV cache reduction: Value also
matters. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 21158–21166, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Roger A Horn and Charles R Johnson. 2012. Matrix
analysis. Cambridge university press.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, and Boris Gins-
burg. 2024. RULER: What’s the real context size of
your long-context language models? In First Confer-
ence on Language Modeling.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models. arXiv preprint
arXiv:2106.09685.
13680

<!-- page 10 -->

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b. Preprint,
arXiv:2310.06825.
Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and
Davood Rafiei. 2023. Evaluating open-domain ques-
tion answering in the era of large language models.
In Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 5591–5606, Toronto, Canada.
Association for Computational Linguistics.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In Proceedings of the 29th
Symposium on Operating Systems Principles, SOSP
’23, page 611–626, New York, NY , USA. Association
for Computing Machinery.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. SnapKV:
LLM knows what you are looking for before gener-
ation. In The Thirty-eighth Annual Conference on
Neural Information Processing Systems.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics, 12:157–173.
OpenAI and et al. 2024. Gpt-4 technical report.
Preprint, arXiv:2303.08774.
Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi,
and Roy Schwartz. 2024. Transformers are multi-
state rnns. arXiv preprint arXiv:2401.06104.
Guanghui Qin, Corby Rosset, Ethan Chau, Nikhil Rao,
and Benjamin Van Durme. 2024a. Dodo: Dynamic
contextual compression for decoder-only lms. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 9961–9975.
Guanghui Qin, Corby Rosset, Ethan C. Chau, Nikhil
Rao, and Benjamin Van Durme. 2024b. Nugget 2d:
Dynamic contextual compression for scaling decoder-
only language models.
Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan
Fan, Ke Cheng, Weiyao Lin, and Jianguo Li. 2025.
CAKE: Cascading and adaptive KV cache eviction
with layer preferences. In The Thirteenth Interna-
tional Conference on Learning Representations.
Qwen and et al. 2025. Qwen2.5 technical report.
Preprint, arXiv:2412.15115.
A Vaswani. 2017. Attention is all you need. Advances
in Neural Information Processing Systems.
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2025. Retrieval head mechanis-
tically explains long-context factuality. In The Thir-
teenth International Conference on Learning Repre-
sentations.
Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, junxian
guo, Shang Yang, Haotian Tang, Yao Fu, and Song
Han. 2025. Duoattention: Efficient long-context
LLM inference with retrieval and streaming heads. In
The Thirteenth International Conference on Learning
Representations.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2024. Efficient streaming lan-
guage models with attention sinks. In The Twelfth
International Conference on Learning Representa-
tions.
Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin
Zhang, and Hai Zhao. 2024. PyramidInfer: Pyramid
KV cache compression for high-throughput LLM
inference. In Findings of the Association for Com-
putational Linguistics: ACL 2024, pages 3258–3270,
Bangkok, Thailand. Association for Computational
Linguistics.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang
Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai,
Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024.
∞Bench: Extending long context evaluation beyond
100K tokens. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 15262–
15277, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, et al. 2023.
H2o: Heavy-hitter oracle for efficient generative in-
ference of large language models. Advances in Neu-
ral Information Processing Systems, pages 34661–
34710.
A Extension of The Information Flow of
LLM Decoding Process with KV Cache
KV Cache is initialized at prefilling stage, which
basically computes the Key and Value for tokens
in the initial prompts in the standard way (Vaswani,
2017). In the following, we assume that there ex-
ists a KV Cache of (N−1) previous tokens and
demonstrate how decoding is performed at step-N.
Notation Table The LLM hasL layers, each has
H heads. The model and head dimensions are d
anddh =d/H;Kl,V l are the KV Cache for thel-
th layer up to the current time step (theN-th token),
which are of [H, (N−1),d h] sizes. The notations
for the theoretical analysis are listed in Table 3.
13681

<!-- page 11 -->

Notation Explanation Notation Explanation
N Current token length ANl,h[i] Attention weight of positioniat layerl, headhand stepN
Ne Expected token length yNl Attention output of layerl and stepN
L Total number of layers ˆyNl Modified attention output of layerl and stepNafter eviction
H Total number of heads per layer p Logits after last layer for next token
l Layer index,l∈[L] ˆp Modified logits after last layer for next token after eviction
h Head index,h∈[H] P Information loss function of Transformer residual streams
d The model embedding dimension w Sliding window size
dh The head embedding dimensiondh =d/H Bl,h Budget for headhof layerl
xNl The input hidden states of stepNand layerl Bl Budget for layerl
QNl The query vector of stepNand layerl B Fixed total budget for KV Cache,B=∑
l∈[L]Bl
KNl The key vector of stepNand layerl sl,h[i] Score of positioniat layerl and headh
VNl The value vector of stepNand layerl el The uncertainty of layerl for dynamic layer budget allocation
Kl,h Key cache of layerl and headh Il,h Attention mask for the headhof layerl,Il,h∈[1,0]N
Vl,h Value cache of layerl and headh I Attention maskI∈[1,0]L×H×N
Table 3: Notation table.
Decoding Process According to (Ferrando and
V oita, 2024), the decoding process of large lan-
guage models (LLMs) can be viewed as a series
of operations on the current residual stream, as il-
lustrated in Figure 1. In each layer, information is
read from the residual stream, updated, and then
written back. Specifically, supposing that xN
l is
the current input for layerl, we first calculate the
correspondingQN
l ,K N
l ,V N
l as follows:
QN
l =xN
l W Q
l ;KN
l =xN
l W K
l ;V N
l =xN
l W V
l
whereQN
l ,K N
l ,V N
l are of size (H×1×dh), con-
tainingH head-wise caches. The layer-wise KV
Cache is then updated as follows:
Kl =Cat[Kl,K N
l ],V l =Cat[Vl,V N
l ]
whereKl,V l are tensors of size (H×N×dh), and
Cat indicates the concatenation operation.We then
calculate the attention scores of step-N for layer-l:
AN
l =Cath∈[H]
(
AN
l,h
)
where AN
l,h = Softmax (
QN
l,hKl,h√dh
). Here,
textbfAN
l,h[i] indicates how much the token at step-
N (theN-th token) attends to thei-th token (i< =
N). Layer-l attention output is calculated as fol-
lows:
yN
l =Cath∈[H](AN
l,hVl,h)W O
l ∈R1×d
whereW O
l ∈Rd×d. The layer outputxN
l+1, which
is also the input for the layer-(l + 1), is calculated
asxN
l+1 =yN
l +FFN (yN
l ).
In the last layer, we exploits an un-embedding
layer (W M∈Rd×|V|) to get the probability vector
p for next token sampling:
pN =
(
yN
L +FFN (yN
L
)
W M (8)
Head-wise vs Layer-wise Cache Current query
matrix and KV Cache on headh of layerl are :
QN
l,h=QN
l [:,d h∗h :dh∗(h + 1)]∈R1×dh (9)
Kl,h=Kl[:,d h∗h :dh∗(h + 1)], (10)
Vl,h=Vl[:,d h∗h :dh∗(h + 1)]∈RN×dh
(11)
Henc, the layer-wise KV Cache can be treated as
concatenation of head-wise elements where we just
change the order of dimensions:
Kl =Cath∈[H][Kl,h]∈RH×N×dh, (12)
Vl =Cath∈[H][Vl,h]∈RH×N×dh (13)
And the same to the query matrix:
QN
l =Cath∈[H][QN
l,h]∈RH×1×dh (14)
B Extension of A Principled Framework
for KV Cache Eviction based on
Information Loss
The unified problem for budget allocation and
cache eviction can be defined as follows:
min
I,B
P(x1...N
1 ,I,B) (15)
st.
∑
i∈[N ]
Il,h[i] =Bl,h;
∑
h∈[H]
Bl,h=Bl;
∑
l∈[L]
Bl = B
Il,h[k] = 1,∀l,h ; and∀k∈[N−w,N ]
13682

<!-- page 12 -->

The optimization problem in Eq. 15 is infeasible to
solve for several reasons. We can instead search for
a scoring functions, wheresl,h[i] assigns an impor-
tance score to token i at layerl and headh. This
scoring function allows us to greedily choose the
least important entries to be masked until the bud-
get is metI =Select(s,B). Bringing everything
together, we arrive at the following (surrogate) op-
timization problem:
min
B,s∈F
P(x1...N
1 ,s, B) (16)
Current various kv cache eviction methods can
be adapted into our framework, just defining sev-
eral significant functions and parameters (includ-
ingP,I,B ands) and introducing additional con-
straints, which will result in suboptimal perfor-
mance. In addition, they adopt many heuristic
techniques based on observations to simplify the
problem. The full summarization of how existeing
methods can be formalized within our framework
is presented in Table 4.
H2O. (Zhang et al., 2023) Allocation budgetsB
are all fixed before generation. The budgets of all
layers are the same and the budgets of all heads are
also the same.
Bl,h= B
HL (17)
H2O uses head attention loss and adopt accumu-
lated attention scores as score function.
sl,h[i] =
N∑
j=i+1
Aj
l,h[i],Il,h=Select(sl,h,Bl,h)
(18)
H2O claimed that the accumulated attention score
can preserve the future attention pattern better. This
technique is heuristic and based on observations
of experiments in several methods like H2O and
SnapKV (Li et al., 2024), but it is valid and actually
can improve the performance, mitigating the im-
pact of absolutism of only current attention scores
(Oren et al., 2024).
TOV A. (Oren et al., 2024) The difference be-
tween TOV A and H2O is that TOV A uses current
attention scores as score function.
sl,h[i] =AN
l,h[i],Il,h=Select(sl,h,Bl,h) (19)
SnapKV . (Li et al., 2024) The difference between
SnapKV and H2O is that SnapKV uses recent atten-
tion scores as score function, which means SnapKV
only utilizes tokens within sliding window to cal-
culate accumulated attention scores. We set sliding
window size asw:
sl,h[i] =
N∑
j=N−w
Aj
l,h[i]
Il,h=Select(sl,h,Bl,h) (20)
SnapKV claims that the accumulated attention
scores of the recent sliding window is enough to
represent the significance of tokens. Furthermore,
SnapKV adopts pooling operation to preserve the
completeness of the information. In our view, bet-
ter protecting the coherence of the text is the reason
for the effectiveness of pooling operation.
PyramidKV . (Cai et al., 2024) The difference
between PyramidKV and SnapKV is that consider-
ing the different significance of layers in the long-
context setting, PyramidKV set the budgets of lay-
ers in a descending order like a pyramid. It uses a
hyper-parameterβto control the shape of pyramid.
BL−1 = B
β∗L,B0 = 2∗B
L −BL−1
Bl =B0−BL−1−B0
L−1 ∗l (21)
And the budgets of heads in one layer are the same:
Bl,h=Bl
H .
Hence, compared with SnapKV , PyramidKV
consider about different budgets of layers in a
heuristic way.
CAKE. (Qin et al., 2025) Allocation budgetsB
are generated through the online prefilling stage.
All heads of one layer have the same budget. So
CAKE do not consider the level of head (such as
using mean information across heads).
Considering spatial and temporal information,
CAKE allocates different budgets to different lay-
ers. And not adopting the fixed pattern like Pyra-
midKV , CAKE claims that for different samples,
the allocation pattern also needs to be adapted. It
defines functions of spatial and temporal informa-
tion for one layerl, the spatial information function
H is formed as entropy of attention scores (larger
values means more even distribution) and the tem-
poral information functionV (larger values means
more distribution shift) is formed as variance of
attention scores (A(n) means the attention scores
13683

<!-- page 13 -->

Methods Budgets Scoring Function Loss
Bl,h Bl
H2O (Zhang et al., 2023) Bl/H B/L Accumulated attention scores
Head Attention
sl,h[i] = ∑N
j=i+1Aj
l,h[i]
SnapKV (Li et al., 2024) Bl/H B/L Recent attention scores
sl,h[i] = 1
w
∑N
j=N−wAj
l,h[i],∀i<N −w
TOV A (Oren et al., 2024) Bl/H B/L Last-token attention scores
sl,h[i] = AN
l,h[i]
CAKE (Qin et al., 2025) Bl/H Dynamic Recent attention scores + attention shifts
sl,h[i] = γV ARN
j=N−w([Aj
l,h[i]))
+ 1
w
∑N
j=N−wAj
l,h[i],∀i<N −w
V ATP (Guo et al., 2024) Bl/H B/L Recent attention scores + value vectors Head Attention
Outputsl,h[i] =
∥Vl,h[i]∥1
w
∑N
j=N−wAj
l,h[i]
Dodo (Qin et al., 2024a) Dynamic B/L Neural Network (LoRA) Logits
DuoAttention (Xiao et al., 2025) w or full - Head classifier (retrieval vs non-retrieval)
Layer Attention
Output
AdaKV (Feng et al., 2024) Dynamic Fixed Recent attention scores
LA Va (Ours) Dynamic Dynamic Recent attention scores + value vectors
sl,h[i] =
maxk∥Vl,h[k]∥1
w
∑N
j=N−wAj
l,h[i]
Table 4: Comparison between different methods; Dodo and DuoAttention require training; The layer cache budget
Bl of AdaKV is based on the method it is integrated with.
distribution in the n-th step of prefilling stage):
Hl =−
N∑
j=1
Aj
l log(Aj
l ),
Vl =
N∑
j=1
V AR([At
l[j]]t∈[j,N]) (22)
Then CAKE uses these two functions to determine
the budget of layers, whereγ1 andγ2 are two hyper-
parameters to control the influence of two func-
tions:
Pl =H
1
γ1
l V
1
γ2
l ,Bl = Pl
∑l∈[L]Pl
B,Bl,h=Bl
H
(23)
CAKE also uses head attention loss function as
optimization objective but it also introduces tempo-
ral information into score function of SnapKV . It
adopts variance to represent the distribution shift
of attention scores for the same token. Let γbe a
hyper-parameter to control the influence of tempo-
ral information, andw as the sliding window size,
CaKE score is:
sl,h[i] =
N∑
j=N−w
Aj
l,h[i] +γV AR([At
l,h[i]]t∈[i,N])
Il,h=Select(sl,h,Bl,h) (24)
AdaKV . (Feng et al., 2024) The algorithm of
AdaKV is based on other methods. It adopts
layer attention output loss function but not con-
duct real training. Deriving the upper bound of
output loss (as shown in Eq. 25 where C =
maxh∈[H]∥W O
l,h
TV T
l,h∥1), AdaKV obtains the in-
sight that allocating different budgets to heads of
one layer based on the score function just consider-
ing about information within attention scores can
preserve the performance of model further.
∥yl−ˆyl∥1≤2C
∑
h∈[H]
(
∑
i∈[N ]
AN
l,h[i](1−Il,h[i]))
(25)
We set ˆsl as the topk results of allsl,h,h∈[H], the
budget of one headh can be calculated by:
Bl,h=Num(ˆsl,h), ˆsl,Il =Select(sl,h,Bl,h)
(26)
AdaKV combines this insight with SnapKV and
PyramidKV for better results. So the score func-
tion of AdaKV is the same as Eq. 20. However, the
bound of AdaKV ignores the influence of value in-
formation and just use the max information, which
will make the bound too loose. Our framework
about output loss is motivated by this research and
we conduct some modification and further studies.
For the details and how to derive upper bound of
output loss, refer to Section 4.
DuoAttention. (Xiao et al., 2025) DuoAttention
uses layer attention output loss function as op-
timization objective. Unlike H2O and TOV A, at-
tention maskI of DuoAttention is constraint to
13684

<!-- page 14 -->

a pattern combined with sink and recent tokens
based on allocation budgetsB, which means score
functions id for tokens are not needed. Here sink
tokens means several initial tokens in prompt de-
fined by StreamingLLM (Xiao et al., 2024).
Il,h[i] =
{
1 if positionk is sink or recent,k∈[N]
0 otherwise, evictKl,h[k] andVl,h[k]
(27)
DuoAttention adopts real optimization method and
needs training based on 2-norm of output loss func-
tion. The optimization result is to determine the
allocation budgetsB. In detail, it determines which
head was allocated with full budget and which head
was allocated with a compressed budget. So be-
sidesI andB, DuoAttention introduces a param-
eterαto be optimized and finally determines the
different functions of heads, including Retrieval
Heads (Wu et al., 2025) and Streaming Heads. We
define ˆw as the numbers of sink and recent tokens.
Bl,h=
{
n if headh of layerl is Retrieval Head
ˆw otherwise, Streaming Head
(28)
Dodo. (Qin et al., 2024a) Dodo uses logit loss
function as optimization objective. But not adopt-
ing a predefined rule for attention maskI, Dodo
uses a score functionϕimplemented by LoRA (Hu
et al., 2021) adapters to determine the attention
mask for tokens, which is trained along with log-
its loss. Logits loss is defined by loss of future
expected tokens which are not pratical. So Dodo
converts the expected tokens into past tokens and
the loss function can be formalized as:
P(I,B) =
∑
i∈[N ]
CE(p, ˆp)i (29)
The score functionϕis trained via this loss func-
tion and finally determines which tokens will be
preserved. The cache budgetB for all heads and
layers are the same. Besides, Dodo merges the in-
formation within tokens evicted into the preserved
tokens similar to KV Cache merging methods.
V ATP. (Guo et al., 2024) The difference between
LA Va and V ATP is shown in Table 4 and explained
as follows: (1) V ATP directly multiplies each to-
ken’s value norm with attention scores. In contrast,
LA Va calculates the maximum value norm, which
serves as scaling factors for heads; (2) V ATP has
fixed head and layer budgets, while LA Va is totally
Budgets 128 512 1024
SnapKV 34.42 41.48 43.00
+V ATP 35.34 41.93 43.32
LA Va 36.74 42.59 43.65
-layer dynamic 36.20 42.11 43.35
Table 5: Comparison between V ATP and LA Va.
dynamic. The deeper difference, however, lies in
how the two scores are developed. V ATP comes
with an intuition of "Value also matters" but lacks
theoretical analysis. We independently derive from
layer attention output with a complete reasoning
process: starting from layer attention output, de-
riving the upper bound, getting an approximate
score in greedy solution, smoothing it out based on
multiple residual stream.
This reasoning is very important. As we start
from the layer point of view, we can see that such
scores can be used to compare entries across heads
for layer-wise KV Cache eviction. And we ar-
gue that doing so could reduce the information
loss at layer attention output. The reasoning pro-
cess shows what approximation we make and gives
room for future improvement.
To validate our elaboration, we compares three
configurations: (1) V ATP integrated with SnapKV ,
(2) standard LA Va, and (3) LA Va without dynamic
layer budgeting based on Mistral-7B-Instruct-v0.2
in LongBench. The results in Table 5 demonstrate
that while V ATP shows improvement over baseline
SnapKV , it consistently underperforms compared
to both LA Va and LA Va (-layer dynamic). From
the computational perspective, V ATP incurs similar
overhead to LA Va(refer to Appendix D, yet delivers
suboptimal performance. This verifies our claim
that intuition and a theoretical analysis help you
get to a more optimal solution.
C Extension of LA Va: Layer-wise Cache
Eviction with Dynamic Budget
Allocation
Details of Lemma 1. We define and derive the
Layer Attention Output Loss in this lemma.
Lemma 1. Based on the Lp norm, the layer at-
tention output loss due to the attention maskI is
measured for layer-l at the current (N-th) decoding
13685

<!-- page 15 -->

step as follows:
P(x1...N
1 ,I,B) =∥yN
l −ˆyN
l ∥p (30)
=
Cath
[
(AN
l,h−
AN
l,h⊙Il,h
∥AN
l,h⊙Il,h∥1
)Vl,h
]
W O
l

p
where⊙indicates element-wise multiplication and
ˆyN
l =Cath( ˆAN
l,hVl,h)W O
l
As we mentioned above:
yN
l =Cath∈[H](AN
l,hVl,h)W O
l
ˆyN
l =Cath∈[H( ˆAN
l,hVl,h)W O
l
(31)
And based on the definition of attention maskI, the
attention weights after eviction can be calculated
as:
ˆAN
l,h=Softmax (
−inf⊙(1−Il,h) +QN
l,hKT
l,h√dh
)
(32)
Hence, Lemma 31 is equal to (Temporarily ignor-
ing the superscriptN):
ˆAl,h= Al,h⊙Il,h
∥Al,h⊙Il,h∥1
(33)
This theorem has been proved by AdaKV (Feng
et al., 2024), so we will not elaborate further here.
Proof of Theorem 1. Then we drive the upper
bound of Layer Attention Output Loss and give
this theorem.
Theorem 1. TheL1 norm of layer attention out-
put loss can be bounded by:
(34)∥yl−ˆyl∥1
≤2 ˆC
h∈[H]∑ ¯Vl,h(
k∈[N ]∑
AN
l,h[k](1−Il,h[k]))
where ¯Vl,h = maxk∈[N ]∥Vl,h[k]∥1 and ˆC =
∥W O
l
T∥1 is a constant, which is independent of
any head or token within layer-l.
Proof. First we need to introduce a lemma:
Lemma 2. Given a vectorx∈R1×m and a matrix
W∈Rm×n, we can get the relationship between
matrix norm and vector norm:
∥xW∥p≤∥x∥p∥W T∥p (35)
∥xW∥p and∥x∥p are vector p-norm, ∥W T∥p is
matrix p-norm which is calculated by the largest
sum of column absolute value.
This lemma is derived from Horn and Johnson
(2012). Then we can obtain (Temporarily ignoring
the superscriptN):
∥yl−ˆyl∥1
≤∥Cath[(Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
)Vl,h]∥1∥W O
l
T
∥1
(36)
We set∥W O
l
T∥1 as ˆC because it is the constant
model parameter. Then we know that and set:
Gl,h= (Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
)Vl,h∈R1×dh
(37)
Thus∥Cath∈[H][Gl,h]∥1 is the vector 1-norm of a
vector∈R1×(dh∗H). According to the definition of
vector 1-norm, we can transform cat operation to
sum and continue derivation based on Theorem 2:
∥yl−ˆyl∥1
≤ˆC∥Cath∈[H][(Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
)Vl,h]∥1
= ˆC
∑
h∈[H]
∥(Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
)Vl,h∥1
≤ˆC
∑
h∈[H]
(∥Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
∥1∥V T
l,h∥1)
(38)
Next we will prove that∥Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
∥1=
2 ∑i∈[N ]
ifIl,h[i]=0Al,h[i].
Let∥Al,h⊙Il,h∥1= ∑
i∈[N ]Il,h[i]Al,h[i] =
∑i∈[N ]
ifIl,h[i]=1Al,h[i] asF∈(0, 1]:
13686

<!-- page 16 -->

∥Al,h−Al,h⊙Il,h
∥Al,h⊙Il,h∥1
∥1 =∥F−Il,h
F ⊙Al,h∥1
=
∑
i∈[N ]
|(F−Il,h[i])Al,h[i]
F |
=
i∈[N ]∑
ifIl,h[i]=0
Al,h[i] +
i∈[N ]∑
ifIl,h[i]=1
(1−F)Al,h[i]
F
=
i∈[N ]∑
ifIl,h[i]=0
Al,h[i] +
∑i∈[N ]
ifIl,h[i]=1Al,h[i]
F
−
i∈[N ]∑
ifIl,h[i]=1
Al,h[i]
=
i∈[N ]∑
ifIl,h[i]=0
Al,h[i] + 1−
i∈[N ]∑
ifIl,h[i]=1
Al,h[i]
= 2
i∈[N ]∑
ifIl,h[i]=0
Al,h[i]
(39)
Then based on the definition of matrix 1-norm
and∥V T
l,h∥1∈Rdh×N, we can calculate this as the
largest sum of row absolute value ofVl,h∈RN×dh,
which is equals to the largest vector 1-norm of V
value of previous tokens, formalized as:
¯Vl,h=∥V T
l,h∥1=maxk∈[N ]∥Vl,h[k]∥1 (40)
Now we can obtain:
(41)∥yl−ˆyl∥1
≤2 ˆC
∑
h∈[H]
(
i∈[N ]∑
ifIl,h[i]=0
AN
l,h[i]∥V T
l,h∥1)
= 2 ˆC
∑
h∈[H]
(
∑
i∈[N ]
AN
l,h[i] ¯Vl,h(1−Il,h[i]))
Here the proof is done.
Potential Future Work. Building on our frame-
work, multiple research directions can be further
explored. One possible question is whether the
Layer Output Loss, which takes into account the
FFN layer, should be considered. The interaction
between the FFN layer and the layer attention out-
put determines what information a layer writes to
the residual stream (Ferrando and V oita, 2024). In
other words, certain tokens in past residual streams
may play a crucial role in activating the layer’s
knowledge within the FFN. Accounting for these
interactions could reduce performance loss, yet the
challenge lies in how to do so efficiently.
Another potential avenue is formulating the prob-
lem as an online reinforcement learning (RL) task,
where the objective is to optimize the policy (i.e.,
the scoring function) to maximize the expected re-
ward. Here, the expected reward can be cast as min-
imizing the expected loss in future residual streams,
not just the past ones. This direction is potential for
the cache-offload and retrieval problem, where we
need to decide which parts of the cache to offload
to CPU or retrieve from CPU while maintaining
the communication cost.
Additionally, this framework could be extended
to model pruning, not just masking tokens but also
selectively masking model parameters to minimize
information flow while preserving efficiency. Also,
the value of algorithms like LA Va and SnapKV ,
CAKE lies in their potential to serve as compo-
nents of larger solutions tailored for long output
and multi-turn contexts. These static methods can
be integrated with merging techniques , cache re-
trieval and offloading, quantization methods, or
dynamic approaches. This is also a promising di-
rection for our future work.
D Extension of Experiments
Implementation Details. For SnapKV and Ada-
SnapKV , no additional hyperparameters are re-
quired. However, for PyramidKV , we must adjust
the parameterβto control the shape of the cache
budget pyramid. We setβto (5, 10, 20) and select
the best-performing result, the same approach to
Ada-PyramidKV . For CAKE, three parameters re-
quire tuning:γ1 andγ2 for layer budget allocation,
andγ3 for the scoring function, as explained in Ap-
pendix B. Based on recommendations from (Qin
et al., 2025), we set 1/γ1 to (0.2, 0.3, 0.5, 1, 2),
1/γ2 to (0.2, 0.3, 0.5, 1, 2), andγ3 to (0, 5, 10, 200).
We then evaluate different combinations and select
the one that yields the best overall performance.
Pooling operators, such as max pooling or aver-
age pooling, can be applied to token score vectors
to smooth score variations across adjacent tokens
(Li et al., 2024; Cai et al., 2024; Qin et al., 2025).
This strategy is also employed in the implemen-
tation of LA Va and all the baselines. For pooling
operation, for all methods, we adopt maxpool func-
tion and set kernel size as 7.
Results of LA Va in LongBench. The results of
Qwen2.5-7B-Instruct are listed in Table 6. The re-
sults of Qwen2.5-14B-Instruct and Qwen2.5-32B-
Instruct are in Table 7. The results of Llama3-8B-
13687

<!-- page 17 -->

Instruct are in Table 8. From all these results, we
can obtain the similar conclusion like Mistral in
main text. LA Va outperforms all baselines across
different budgets, even in models with larger pa-
rameter size.
Results of LA Va in Needle In A Haystack. The
results of Needle In A Haystack are shown in Ta-
ble 9. The conclusion is consistent with that of
LongBench. Our method shows superior overall
performance, demonstrating its robust in preserv-
ing the model’s retrieval capacity.
Results of LA Va in Ruler and InfiniteBench.
The results of Ruler and InfiniteBench are shown
in Table 11 and Table 12. we set the cache budget
as 5%-10% of the task context length, i.e. 1024
and 10000. We use Mistral-7B-Instruct-v0.2 as
the backbone of Ruler. For InfiniteBench, we
change the backbone into Mistral-7B-LongPO-
128K (Chen et al., 2025), which is fine-tuned based
on Mistral-7B-Instruct-v0.2, because the task con-
text length of InfiniteBench is much longer than the
original maximum model length 32K. The results
reconfirm the effectiveness of LA Va.
Results of Dynamic Budget Allocation. The de-
tailed results of ablation study based on Mistral-7B-
Instruct-v0.2 in LongBench are listed in Table 10.
It demonstrates that dynamic budget allocation at
both the head and layer levels is essential for strong
performance, with a more pronounced performance
drop when head-wise allocation is removed under
constrained budgets. This is expected, as LA Va’s
strength lies in its ability to compare cache entries
across heads.
Analysis of Different Layer Allocation. To vali-
date the effectiveness of our layer budget allocation,
we modify LA Va to incorporate two alternative
strategies: LA Va-Uniform, which is equivalent
to LA Va (-layer), andLA Va-Pyramid, which re-
tains LA Va’s head budget allocation and layer-wise
cache eviction but adopts Pyramid for layer allo-
cation. The results in Table 13 indicate that our
method outperforms these alternatives. Notably,
LA Va-Pyramid requires finetuning, whereas the
other methods do not. Moreover, LA Va-Pyramid
fails to outperform LA Va-Uniform at higher bud-
gets, aligning with the observed comparison be-
tween Ada-SnapKV and Ada-Pyramid. This un-
derscores the limitation of heuristic-based designs,
which may not always yield optimal results.
Analysis of Time Complexity. Our study builds
upon the SnapKV framework with a batch size of 1,
consistent with prior works like CAKE and AdaKV .
We start with the analysis for SnapKV (the most
computationally efficient method among baselines)
in computation for one layer as a reference.
• For layer l, SnapKV needs to calculate the
layer’s original KV Cache with the time com-
plexity ofO(HN 2dh), ignoring the IO opera-
tions. Generally, this is done with FlashAtten-
tion, which avoids saving the large attention
matrix of sizeO(N 2). The computation cost
in practice is high due to IO operations and
recomputation (to avoid saving the attention
matrix), but we ignore it for simplicity.
• As Flash attention does not save the attention
matrix, for calculating the scores to evict KV
Cache, SnapKV needs to recompute the atten-
tion scores for the recent window of size w
in the second pass. The time complexity is
O(HNwd h).
• The top-Bl,h selection for head-wise cache
eviction with a min-heap takesO(NlogB l,h),
and for H heads, it takes O(HNlogB l,h),
whereBl,hH =Bl,B lL = B.
To summarize, SnapKV requires:
•O(HN 2dh) for original cache for one layer;
•O(HNwd h) for recomputing the recent atten-
tion scores;
•O(HNlogB l,h) for cache eviction.
In contrast, LA Va requires the computation for one
layer as follows:
•O(HN 2dh) for the original cache of one
layer, same as SnapKV;
•O(HNwd h) for recomputing the recent atten-
tion scores, same as SnapKV;
•O(HNd h) for computing the value norms for
each token;
•O(HNlogB l) for layer-wise cache eviction
because the eviction of LA Va is operated in
all cache of one layer.
13688

<!-- page 18 -->

Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Full Cache29.05 43.34 52.52 62.27 57.59 47.05 30.24 29.25 31.78 23.64 15.96 23.96 72.50 88.82 45.61 42.75 8.50 100.00 96.50 59.61 67.1248.96
B= 128HLPyramidKV21.96 26.41 42.53 52.77 49.33 42.17 23.48 17.88 16.80 19.29 11.24 14.3042.50 83.78 41.15 22.398.50 95.50 63.50 48.53 51.3937.88SnapKV 25.2427.66 43.90 53.53 51.00 42.12 24.59 18.56 18.04 19.85 11.32 15.55 41.00 83.18 40.6824.889.00 98.0081.50 49.44 52.5839.60Ada-PyramidKV23.08 27.53 42.07 53.17 50.73 42.03 23.31 18.03 17.48 19.65 11.21 14.7142.50 83.90 41.25 22.819.0094.00 76.00 49.17 52.6938.78Ada-SnapKV25.20 28.45 45.00 54.3751.0844.0224.66 18.8118.2620.09 11.5016.2542.50 84.06 41.00 22.499.0096.5087.5049.92 54.3240.24CAKE 24.4330.1545.0354.86 50.65 42.4125.9118.89 18.2120.6611.60 15.84 42.0084.5441.9526.248.50 95.50 81.5051.6055.0940.26LA Va (Ours)23.2928.8746.80 56.10 52.6542.9625.0919.2518.2420.5211.80 16.28 43.00 84.56 42.1823.958.50 96.0085.0053.45 56.0740.69
B= 256HLPyramidKV24.82 31.13 46.92 56.06 53.07 42.31 25.06 19.54 19.27 20.47 12.01 16.55 50.00 84.88 42.04 25.398.5096.00 85.50 52.03 55.8241.30SnapKV 26.61 23.77 49.1558.3756.0344.18 25.6820.9620.84 20.99 12.19 18.52 48.5086.31 43.06 29.898.5097.5095.0054.26 59.4243.32Ada-PyramidKV25.97 31.01 47.31 56.43 54.17 43.03 25.23 19.41 19.60 21.09 11.87 17.0754.5086.04 42.69 27.288.5097.00 90.00 52.78 56.5542.26Ada-SnapKV26.5234.5050.0158.2855.61 43.60 26.1420.8921.3020.9412.5118.5952.50 85.50 42.97 28.438.50 98.0093.50 53.94 59.3043.41CAKE 26.59 33.9549.80 58.25 54.8944.4226.47 20.3521.2321.9412.35 18.53 47.50 85.4143.51 32.33 8.5097.5094.0055.5661.1343.53LA Va (Ours) 27.04 35.1949.3659.7455.35 44.1327.2520.88 21.1521.5112.77 18.9649.0086.7343.4230.358.50 98.0093.0056.19 62.1943.84
B= 512HLPyramidKV28.02 35.7450.8458.11 55.26 44.72 25.85 20.94 21.83 21.34 12.33 18.9559.50 86.13 43.04 32.838.5099.0096.0055.65 59.4244.48SnapKV 28.2728.2250.6960.2756.1844.69 27.28 21.98 23.79 21.8913.2020.64 59.50 84.10 43.68 35.528.50 100.0094.00 56.66 62.6945.32Ada-PyramidKV27.31 37.36 49.62 58.57 55.40 44.66 26.74 21.35 22.39 21.12 12.42 19.3262.00 86.2943.78 33.338.5099.0095.50 55.78 60.9944.83Ada-SnapKV28.03 38.51 50.0660.5455.50 45.0628.8122.0423.9822.4913.0520.8062.0085.83 44.37 37.108.50 100.0094.00 56.44 62.7145.71CAKE 28.1739.0950.22 60.00 54.8945.21 26.3122.20 23.65 21.98 13.04 20.57 57.50 85.6044.6137.238.5099.50 94.0058.2763.9545.45LA Va (Ours)27.2139.08 50.47 60.0955.6345.2527.7522.9123.8322.8113.0520.8458.5086.1545.02 37.43 8.50 100.0093.5058.0264.5745.74
B= 1024HLPyramidKV28.06 40.11 51.83 60.2257.5545.38 29.31 22.42 24.35 22.04 13.12 21.12 68.00 85.27 44.18 36.998.50 100.00 96.5058.29 62.5646.47SnapKV 29.0142.0251.8661.2256.82 45.04 28.9523.9726.26 22.76 13.6622.5068.50 86.8545.52 42.50 8.50 100.00 96.5057.94 65.5947.43Ada-PyramidKV28.52 40.5051.8760.27 56.4245.8029.18 23.01 24.45 22.10 13.31 21.2569.00 86.41 45.10 37.798.50 100.00 96.5057.16 63.3146.69Ada-SnapKV29.6142.3051.79 60.29 56.3845.75 29.30 23.64 26.21 22.8013.8522.3969.0088.0945.36 41.758.50 100.0096.00 58.15 65.7747.47CAKE 29.70 41.08 51.85 60.6457.34 45.0230.4823.82 25.9222.9513.6922.45 67.50 86.63 45.2242.008.50 100.00 96.5059.4965.9947.47LA Va (Ours) 29.7941.68 51.8460.79 57.04 45.2730.0123.99 26.3622.9013.81 22.4269.5087.4245.46 41.008.50 100.00 96.50 59.97 66.2447.64
Table 6: Final comparison based on Qwen2.5-7B-Instruct among 21 datasets of LongBench. (Note: The best result
is highlighted in bold, and the second is in underline. )
Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Qwen2.5-14B-Instruct
Full Cache29.33 45.19 53.59 62.79 62.59 57.69 38.47 29.87 29.74 23.53 14.75 21.90 77.50 90.23 47.27 50.00 9.23 98.67 98.25 62.60 51.1350.21
Qwen2.5-14B-Instruct, B=128hPyramidKV19.67 22.26 39.57 50.04 50.75 49.47 30.31 16.67 16.10 19.43 10.53 13.51 42.00 82.29 40.90 27.0012.12 82.50 56.67 54.52 41.3837.03SnapKV 21.04 25.50 42.11 49.89 54.31 51.8733.60 17.78 17.12 19.95 10.75 14.53 43.50 85.95 41.81 26.75 10.50 89.58 65.00 55.42 43.4239.07Ada-PyramidKV20.85 24.83 40.88 51.78 54.65 52.34 29.78 16.83 16.67 19.59 10.32 13.9046.5080.76 40.58 25.75 11.18 87.75 63.75 53.72 43.4937.90Ada-SnapKV22.16 25.5842.8052.2255.10 53.21 33.5017.9817.6920.2510.86 14.81 45.50 85.6242.49 27.00 9.0591.33 68.1756.2643.3939.76CAKE 22.2026.13 42.10 50.83 54.7553.25 31.77 17.73 17.56 19.98 10.8415.4444.0087.51 42.65 28.50 13.9686.5078.8354.9243.9040.16LA Va (Ours) 22.24 26.52 43.09 52.39 55.97 53.43 33.68 18.23 17.94 20.57 10.9815.1046.0086.79 42.2027.17 10.5392.0073.0055.7444.6340.39
Qwen2.5-14B-Instruct, B=512hPyramidKV26.18 38.19 48.71 59.8160.7455.26 36.82 20.55 21.21 21.27 11.86 18.43 68.50 89.21 45.38 44.258.5998.3396.75 59.71 48.7146.59SnapKV 26.9939.34 48.84 59.34 60.20 54.86 37.4721.43 22.25 21.95 11.93 19.34 66.50 88.78 45.95 45.25 8.2298.2598.5861.12 49.4246.95Ada-PyramidKV26.78 40.2549.7160.4060.6455.6937.72 20.75 21.49 21.54 11.67 18.6070.0088.59 45.70 44.508.77 98.3396.75 60.23 48.8547.00Ada-SnapKV26.0341.5649.4260.8859.9955.6338.34 21.33 22.4922.0911.96 19.3269.5089.0146.3546.757.72 98.1798.5062.2149.9247.48CAKE 25.39 39.92 48.62 60.30 60.42 55.1938.3721.4022.56 21.7212.31 19.57 70.00 89.0346.1946.25 6.68 98.17 98.25 60.90 49.3147.17LA Va (Ours)26.2340.65 48.93 59.45 60.34 55.36 37.5021.53 22.57 22.1311.9119.48 67.00 88.6846.50 46.757.98 97.75 97.7561.8550.3847.18
Qwen2.5-32B-Instruct
Full Cache OOM
Qwen2.5-32B-Instruct, B=128hPyramidKV21.32 27.86 43.55 56.05 55.74 53.85 32.25 16.74 17.08 18.88 10.71 15.76 48.00 54.4140.6929.50 11.17 94.00 73.09 48.04 35.3638.29SnapKV 21.72 28.31 42.83 56.03 54.43 55.52 30.78 16.94 16.92 19.04 10.53 15.69 48.5058.30 39.64 27.5012.00 93.75 74.37 47.15 35.8238.37Ada-PyramidKV21.1929.6745.61 58.04 57.3055.6532.96 17.45 17.37 19.3010.89 16.0251.5056.24 40.2430.2512.0097.00 82.6748.1435.9439.78Ada-SnapKV21.79 28.64 45.49 56.5657.1256.14 32.5417.6617.63 19.31 10.66 16.1249.5060.0740.03 27.5012.00 96.0485.1347.9636.2939.72CAKE 21.28 28.40 43.30 55.71 55.93 54.89 32.86 17.04 17.0019.44 10.5016.18 46.50 56.3540.3831.88 12.5094.79 82.92 46.63 36.0539.07LA Va (Ours) 22.29 30.1245.5057.06 56.5958.51 33.7217.50 17.4219.97 11.09 16.2948.50 57.21 40.23 28.17 10.0097.4284.0948.1236.6839.83
Qwen2.5-32B-Instruct, B=512hPyramidKV26.00 37.40 48.67 61.17 60.60 60.44 34.75 19.37 20.84 20.61 11.64 18.48 66.00 55.11 42.71 39.0011.5699.75 98.54 50.28 38.1243.86SnapKV 25.71 40.23 48.81 62.94 61.16 60.60 34.8520.64 22.69 21.27 11.61 20.04 66.5077.7744.01 41.86 11.19100.0099.03 52.2039.1545.82Ada-PyramidKV26.41 38.9750.14 61.50 61.5061.86 37.5519.67 21.49 20.71 11.23 18.68 67.50 60.81 43.40 39.75 11.0899.7599.6250.60 38.2744.79Ada-SnapKV 27.5139.44 49.2163.0961.7061.60 37.23 20.35 22.6921.7211.7420.45 69.00 77.87 44.1942.0411.56 100.0098.24 52.2239.1446.24CAKE 25.3240.24 49.6663.2859.75 61.42 37.11 20.4422.73 21.22 11.67 20.28 66.50 77.31 43.9244.5811.19100.0098.7852.3638.9946.04LA Va (Ours)26.5641.18 50.8062.4961.9060.8337.2521.44 23.16 22.02 11.8620.3068.50 77.69 43.9742.2311.50100.0098.5352.24 38.8646.35
Table 7: Final comparison based on Qwen2.5-14B-Instruct and Qwen2.5-32B-Instruct among 21 datasets of
LongBench. (Note: The best result is highlighted in bold, and the second is in underline.)
13689

<!-- page 19 -->

Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Full Cache21.23 43.25 44.68 57.47 48.12 38.55 24.72 27.46 29.97 22.19 0.18 27.32 73.00 90.50 42.04 23.50 7.00 70.50 93.00 60.76 48.8544.71
B= 128HLPyramidKV18.36 30.37 42.49 49.8347.5935.11 23.20 18.65 18.62 20.95 0.07 20.27 49.50 88.6738.5221.003.7568.50 91.00 58.54 50.2939.76SnapKV 18.8830.10 41.3350.6847.16 34.47 23.91 18.73 18.43 20.92 0.08 19.63 46.50 87.98 38.25 20.00 3.50 68.00 89.50 59.16 52.9739.51Ada-PyramidKV16.6232.27 43.4150.49 47.0735.70 24.09 19.27 19.6321.14 0.0620.96 58.5089.10 38.4121.00 4.0068.5093.5059.27 51.7240.73Ada-SnapKV17.78 30.31 42.60 49.81 46.94 34.97 23.7219.1619.2121.280.09 20.3755.5089.68 38.96 21.00 4.00 69.00 93.50 60.3253.5240.58CAKE 18.40 30.92 42.41 49.50 47.04 35.0824.03 18.77 18.77 20.95 0.07 19.74 47.0089.34 38.0320.754.0068.5092.00 58.97 53.3339.88LA Va (Ours)17.2431.5443.26 49.96 46.6935.1724.0918.99 19.0221.19 0.1220.52 52.50 89.22 38.45 20.004.00 69.00 93.5059.7453.5840.38
B= 256HLPyramidKV18.34 36.26 43.69 52.91 47.19 36.4425.0320.28 20.5921.570.14 22.50 61.50 88.7739.26 21.755.0070.0093.00 60.36 49.7441.71SnapKV 19.37 35.25 42.82 52.53 46.5736.8824.68 20.10 20.75 21.15 0.16 22.39 61.00 89.82 38.94 22.005.0070.00 94.0061.16 52.5741.84Ada-PyramidKV18.7137.2944.5953.6347.26 36.39 24.3320.9221.06 21.32 0.1423.22 65.5089.81 39.0722.754.5069.50 92.50 61.00 50.2742.18Ada-SnapKV18.65 36.6645.14 52.6247.27 36.2924.84 19.9121.4121.14 0.15 22.7065.5089.72 39.1522.255.5069.50 93.0062.11 51.9142.26CAKE 19.41 35.81 43.14 51.81 47.1936.76 24.76 20.30 20.81 21.02 0.12 22.50 60.5089.94 39.09 21.755.0069.5093.50 61.4552.6541.84LA Va (Ours)19.1637.0445.1553.5847.8736.57 24.6120.8421.4121.41 0.1523.1465.0090.36 39.9322.005.50 70.0093.0062.70 53.6942.65
B= 512HLPyramidKV19.63 40.70 43.99 55.36 47.30 37.8124.9022.4322.78 21.44 0.16 24.38 68.5090.2340.1823.75 7.00 70.5093.00 61.83 49.5143.26SnapKV 19.67 40.65 45.65 54.6847.57 37.38 24.37 21.39 22.6521.78 0.17 24.44 68.00 90.19 40.32 23.257.0070.0093.0062.59 50.5143.25Ada-PyramidKV19.46 41.4246.19 55.86 47.7237.90 24.54 21.98 22.9121.840.1224.8569.5090.2340.6023.506.5070.50 93.5061.64 49.7743.52Ada-SnapKV19.38 41.03 45.0555.53 47.30 37.75 24.61 21.7323.10 21.59 0.15 24.7070.50 90.2340.4923.756.5070.50 93.5062.2151.6443.55CAKE 19.83 41.6945.76 54.77 47.23 37.3325.0821.51 22.56 21.50 0.17 24.34 69.00 90.19 40.47 23.007.0070.0093.0062.6450.5543.37LA Va (Ours)19.3541.52 45.53 55.49 47.2537.9624.3622.0123.1921.53 0.1725.38 70.5090.2141.30 23.75 7.00 70.50 93.5062.0451.5743.70
B= 1024HLPyramidKV19.73 41.62 43.51 56.7548.7337.23 24.32 22.76 24.72 21.91 0.15 25.97 71.00 90.23 40.9923.756.0071.0093.00 60.66 49.3143.66SnapKV 19.31 41.67 43.92 56.25 47.90 37.7724.50 22.55 24.76 21.72 0.25 25.9172.00 90.24 41.4323.506.50 71.0093.00 61.5950.5343.80Ada-PyramidKV20.23 41.47 44.82 56.3948.48 37.62 24.3523.59 25.1422.24 0.17 26.05 71.00 90.23 41.3423.756.0071.0093.00 61.01 48.6543.82Ada-SnapKV19.9942.0745.8657.16 48.29 38.01 24.4023.60 25.3722.19 0.1626.1072.0090.31 41.5023.506.50 71.00 93.5061.6250.3044.16CAKE 19.29 41.67 43.72 56.38 47.7038.10 24.8322.72 24.68 21.90 0.2126.2572.00 90.2441.5923.506.0071.0093.0061.46 50.1743.81LA Va (Ours) 20.93 42.74 46.43 57.2648.3538.09 24.44 23.2625.2622.330.1726.25 72.50 90.39 42.0523.506.0071.0093.0061.9350.2344.30
Table 8: Final comparison Based on Llama3-8B-Instruct among 21 datasets of LongBench. (Note: The best result is
highlighted in bold, and the second is in underline.)
Methods Mistral-7B Qwen2.5-7B
Full Cache 99.88 99.66
B = 128HL
PyramidKV 91.44 91.10
SnapKV 91.25 93.28
Ada-PyramidKV 92.08 92.70
Ada-SnapKV 92.12 94.30
CAKE 92.79 94.61
LA Va (Ours) 93.35 95.57
B = 1024HL
PyramidKV 97.88 99.56
SnapKV 97.95 99.48
Ada-PyramidKV 98.58 99.58
Ada-SnapKV 98.54 99.53
CAKE 98.32 99.55
LA Va (Ours) 98.95 99.59
Table 9: Average scores of Mistral-7B-Instruct-v0.2 and
Qwen2.5-7B-Instruct in Needle In A HayStack.
For one layerl, the difference of time complexity
between LA Va and SnapKV isO(HN (dh+logH).
In a long context, N is very large, and thus
O(HN (dh +logH) is much smaller than the dom-
inant factorO(HN 2dh). Based on the setting of
Mistral-7B-Instruct-v0.2, we havedh = 128 and
H = 32, the extra computation of LA Va com-
pared to SnapKV isHN (dh +logH) divided by
HN 2dh, which is approximately 0.01% when
N = 10, 000. The computation time increases
with the increase of the number of layers and batch
size for both SnapKV and LA Va, but the ratio of
the extra computation time for LA Va is still 0.01%.
A similar analysis can be achieved to see that all
the other methods have similar latency, aligning
with the latency results in Figure 3.
Analysis of Memory Usage. We analyze the dif-
ference between SnapKV and LA Va/CAKE, which
are dynamic layer budget methods.
• For SnapKV , the cache size increases from
O(HKd h) in the first layer to the last layer,
where it reaches the peak of O(LHB l,hdh).
The memory peaks when the latest (full)
layer cache O(HNd h) is not pruned, and
the current retained cache reaches the size of
O(LHB l,hdh). In sum, the peak memory is
O(HNd h +LHB l,hdh).
• For LA Va and CAKE, the cache size is always
O(LHB l,hdh) from the first layer to the last
layer, yet it is distributed among prefilled lay-
ers. The memory peak, however, is similar to
SnapKV , which isO(HNd h +LHB l,hdh),
except that for LA Va/CAKE, we need to store
the layer scores. As we save only the top
scores for each layer, the size for scores is
O(LHB l,h). Given that the total cache size
13690

<!-- page 20 -->

Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Full Cache26.77 32.34 49.63 48.42 43.43 27.89 18.61 30.85 32.92 24.54 15.04 27.20 71.00 86.23 43.41 39.00 2.81 86.56 89.75 55.29 52.5545.07
B= 128HLLA Va (Ours)19.57 21.11 44.29 33.9138.2923.5915.32 18.56 19.33 22.32 11.42 21.07 53.5085.2040.16 21.752.8869.87 74.75 51.94 48.9236.74−layer 20.3221.18 45.17 35.0037.3723.6215.09 18.20 19.21 22.04 11.35 20.99 48.5085.3239.33 20.75 3.42 67.93 73.75 51.28 47.5236.20−head 20.3320.27 44.06 32.23 36.64 22.84 14.19 18.15 18.88 21.51 11.09 20.89 45.00 84.29 39.57 20.25 3.21 65.23 64.25 51.88 47.5134.95
B= 256HLLA Va (Ours) 22.7024.6748.62 37.81 39.68 25.96 16.77 20.26 21.9222.4811.88 22.91 65.0085.24 41.2826.752.88 76.76 85.7554.17 51.7740.12−layer 21.7824.7447.82 37.47 39.06 25.53 16.21 19.94 21.8623.2211.8122.9162.0085.37 41.5325.25 2.7778.53 87.6752.78 49.8539.77−head 21.34 22.77 47.43 35.87 37.71 25.50 15.47 19.43 21.55 23.06 12.08 22.86 58.00 84.88 41.69 22.25 3.11 74.77 84.18 53.89 51.1938.80
B= 512HLLA Va (Ours) 25.0127.8448.97 42.14 40.95 26.8818.3321.1223.5923.5912.2824.51 68.50 86.34 42.48 33.502.90 87.2389.83 55.83 52.8542.59−layer 24.4327.9848.72 41.00 40.23 26.1718.5020.7424.0023.4012.6824.20 66.50 86.04 42.26 32.75 2.8487.8989.33 54.11 51.2242.11−head 23.59 27.70 48.61 40.61 40.22 25.79 17.87 20.68 23.91 23.39 12.38 24.28 66.50 86.09 41.95 28.50 2.97 86.88 89.17 55.73 52.5341.82
B= 1024HLLA Va (Ours)25.5931.2148.27 43.4341.9227.3819.48 23.4826.0623.86 13.38 26.00 70.0086.22 42.4338.002.73 87.01 88.7557.31 53.2843.65−layer25.7630.3849.54 43.5441.08 27.03 18.83 22.73 25.79 23.69 13.13 25.88 69.5086.30 43.1037.25 2.71 87.5689.2555.04 51.6743.35−head 25.76 29.61 49.31 42.77 40.8227.6318.59 22.6426.2923.77 12.70 25.82 68.00 85.82 41.77 35.00 2.6389.06 89.25 57.3153.2243.26
Table 10: Ablation study based on Mistral-7B-Instruct-v0.2 among 21 datasets of LongBench. (Note: The best
result is highlighted in bold. )
Context Length 4K 8K 16K
PyramidKV 72.55 62.02 55.42
SnapKV 70.71 61.52 55.61
Ada-PyramidKV 70.80 60.83 54.95
Ada-SnapKV 71.14 60.31 55.05
CAKE 72.41 61.55 55.84
LA Va (Ours) 75.39 62.61 56.70
Table 11: Results of Mistral-7B-Instruct-v0.2 in Ruler.
Tasks En Sum En MC En Dia
PyramidKV 25.3 67.2 6.5
SnapKV 25.1 67.2 7.0
Ada-PyramidKV 24.9 67.2 7.0
Ada-SnapKV 24.6 66.8 7.0
CAKE 24.8 67.8 6.6
LA Va (Ours) 25.4 66.8 9.5
Table 12: Results of Mistral-7B-LongPO-128K in In-
finiteBench.
isO(LHB l,hdh), it is sufficient to just keep a
total ofLHK scores for comparison. Again,
the extra factor is dominated byO(HNd h +
LHB l,hdh). The extra memory usage of
LA Va is 0.6% of SnapKV peak memory
whenL = H = 32,B l,h= 1024,d h = 128,
andN = 10, 000. This is small, but not as
negligible as in time complexity, consistent
with Figure 3. However, dynamic layer bud-
get is important for tasks like summarization
or code generation, as shown in Figure 2.
Analysis of Layer Attention Output Loss. To
validate the effectiveness of LA Va in minimizing
layer attention output loss, we compare LA Va with
AdaKV , which also aims to minimize layer atten-
tion output loss and its scoring function is the same
with SnapKV . We set the cache budget as 128 to
make the difference clear and calculate the loss
in the first and the last layer. The backbone is
Mistral-7B-Instruct-v0.2. The results in Table 14
are consistent with the evaluation of other bench-
marks, proving that the upper bound of LA Va is
tighter compared to that of AdaKV .
13691

<!-- page 21 -->

Single-Doc. QA Multi-Doc. QA Summarization Few-shot Learning Synthetic Code
NrtvQAQasperMF-enMF-zhHotpotQA2WikiMQAMusiqueDureaderGovReportQMSumVCSUMMultiNewsTRECTriviaQASAMSumLSHTPCountPR-enPR-zhLccRepoBench-PAvg
Full Cache26.77 32.34 49.63 48.42 43.43 27.89 18.61 30.85 32.92 24.54 15.04 27.20 71.00 86.23 43.41 39.00 2.81 86.56 89.75 55.29 52.5545.07
B= 128HLLA Va-Pyramid19.91 20.36 44.3235.0637.68 23.5815.4017.9919.6122.09 10.87 21.05 52.00 84.45 40.09 20.25 2.8972.32 76.9251.81 46.8136.63LA Va-Uniform 20.32 21.18 45.1735.00 37.3723.6215.09 18.20 19.21 22.04 11.35 20.99 48.5085.3239.33 20.75 3.42 67.93 73.75 51.28 47.5236.20LA Va (Ours)19.57 21.11 44.29 33.9138.2923.59 15.3218.5619.3322.32 11.42 21.07 53.5085.2040.16 21.752.88 69.87 74.7551.94 48.9236.74
B= 256HLLA Va-Pyramid21.22 23.96 47.86 37.12 38.92 24.94 16.70 19.11 21.43 22.44 11.20 22.77 62.50 85.17 41.34 23.75 3.3479.0786.58 52.25 49.7039.40LA Va-Uniform21.7824.7447.82 37.47 39.06 25.53 16.21 19.94 21.8623.2211.8122.9162.00 85.3741.5325.25 2.77 78.5387.6752.78 49.8539.77LA Va (Ours) 22.7024.6748.62 37.81 39.68 25.96 16.77 20.26 21.9222.4811.88 22.91 65.0085.24 41.2826.752.88 76.76 85.7554.17 51.7740.12
B= 512HLLA Va-Pyramid24.59 27.33 48.36 40.24 39.75 26.18 18.26 20.82 23.39 23.38 12.35 24.08 67.0086.66 42.5532.00 2.93 86.13 89.62 53.46 51.5341.88LA Va-Uniform24.4327.9848.72 41.00 40.23 26.1718.5020.7424.0023.4012.6824.20 66.50 86.04 42.26 32.75 2.8487.8989.33 54.11 51.2242.11LA Va (Ours) 25.0127.8448.97 42.14 40.95 26.8818.3321.1223.5923.5912.2824.51 68.5086.34 42.4833.502.90 87.2389.83 55.83 52.8542.59
B= 1024HLLA Va-Pyramid24.88 29.51 49.01 42.57 41.16 27.20 19.40 22.61 25.5824.0013.08 25.71 68.50 86.1943.1937.00 2.6787.73 90.2554.72 51.5343.19LA Va-Uniform 25.7630.3849.54 43.5441.08 27.03 18.83 22.73 25.79 23.69 13.13 25.88 69.5086.3043.10 37.25 2.71 87.56 89.25 55.04 51.6743.35LA Va (Ours)25.5931.2148.27 43.4341.92 27.38 19.48 23.48 26.0623.8613.38 26.00 70.0086.22 42.4338.002.73 87.01 88.7557.31 53.2843.65
Table 13: Layer allocation comparison based on Mistral-7B-Instruct-v0.2 among 21 datasets of LongBench. (Note:
The best result is highlighted in bold. )
Tasks Qasper HotpotQA Gov Report TriviaQA Passage Retrieval ZH LCC
Layer 0
AdaKV 1.77 1.63 1.82 2.64 1.59 1.91
LA Va 1.61 1.59 1.73 2.61 1.40 1.86
Layer 31
AdaKV 134.69 133.33 107.94 121.53 93.50 149.25
LA Va 132.97 130.02 106.06 121.31 90.50 147.16
Table 14: Results of Layer Attention Output Loss.
13692
