# references/108_fier_fine_grained_and_efficient_kv_cache_retrieval_for_long_context_llm_inference.pdf

<!-- page 1 -->

FIER: Fine-Grained and Efficient KV Cache Retrieval for Long-context
LLM Inference
Dongwei Wang1, Zijie Liu2, Song Wang3, Yuxin Ren1, Jianing Deng4,
Jingtong Hu4,Tianlong Chen 2,Huanrui Yang 1†,
1The University of Arizona, 2The University of North Carolina at Chapel Hill,
3The University of Virginia, 4University of Pittsburgh
{dongweiw, yuxinr, huanruiyang}@arizona.edu
{jesseliu, tianlong}@cs.unc.edu,sw3wv@virginia.edu
{jthu, JID70}@pitt.edu
Abstract
The Key-Value (KV) cache reading latency in-
creases significantly with context lengths, hin-
dering the efficiency of long-context LLM in-
ference. To address this, previous works pro-
pose retaining a small fraction of KV cache
based on token importance. For example, KV
eviction uses static heuristics to retain tokens,
while KV retrieval dynamically selects query-
relevant tokens for more adaptive cache man-
agement. However, we observe that important
tokens are often sparsely distributed across
the long context. This sparsity makes exist-
ing page-level KV retrieval inaccurate, as each
page may include irrelevant tokens and miss
critical ones. In this work, we propose Fier, a
Fine-Grained and Efficient KV cache Retrieval
method. Fier uses 1-bit quantized keys to es-
timate the importance of each token, resulting
in efficient and precise retrieval. Experiments
show that Fier matches full KV performance
using only 11% of the cache budget across var-
ious long-context tasks, reducing decoding la-
tency by 1.2×to 1.5×. Code is available at
here.
1 Introduction
KV caching is a memory-for-computation acceler-
ation technique for LLM inference, enabling faster
decoding by reusing intermediate hidden states
(Waddington et al., 2013). However, during in-
ference, each decoded token must attend to the
full KV cache, causing cache reading latency to
grow significantly with context length. For exam-
ple, in LLaMA 7B (Touvron et al., 2023), a 32k-
token KV cache requires 16GB of memory and
over 11ms to read—accounting for more than half
of the total inference time (Tang et al., 2024).
To address this issue, previous works have
proposed to selectively retain only a subset of
KV cache entries based on token importance.
†Corresponding author
Page 4Page 2
(a) Full KV (b) KV Eviction
(c) Existing KV Retrieval (d) Fine-grained Retrieval (ours)
Full KV Query
Keep fixed positions
for all queries
Page 1 Page 3
Min-max product
Page 2 Page 3
not recallable
Quantize to 1-bit
Quant attention
Important/evicted tokens:
Self attention
Page-level
retrieval
Token-level
retrieval
×
×
×
/
× ×
Acc: 100%    O(L)
×
Acc: 1%    O(K)
Acc: 77%    O(K) Acc: 100%    O(K)
Precise token retrieval
Figure 1: Comparison of KV eviction (b), KV retrieval
(c) and Fier (d). While existing retrieval methods suf-
fer from coarse granularity, Fier achieves higher accu-
racy through fine-grainedtoken-levelretrieval, and pre-
serves selection efficiency by quantization.
Among them, one line of work—known as KV
eviction (Fig. 1(b))—focuses on retaining fixed-
position tokens that typically receive higher atten-
tion weights, such as the initial tokens (due to the
attention sink phenomenon) and the most recent
tokens (Xiao et al., 2023; Zhang et al., 2023; Li
et al., 2024; Liu et al., 2023). However, these
approaches overlook the dynamic nature of KV
criticality, i.e., tokens that are evicted earlier may
become important in the future. Their inability
to recall evicted tokens often results in degraded
performance in multi-round QA or long conversa-
tion applications. Motivated by this, another line
of work—KV retrieval (Fig. 1(c))—has been pro-
posed to dynamically recall tokens that are most
relevant to the current query during generation
(Tang et al., 2024; Chen et al., 2024a).
Despite achieving better performance, KV re-
trieval requires frequent estimation of the to-
ken importance for every new query, introduc-
ing additional computational overhead. To miti-
gate this, existing methods perform page-level re-
trieval, where all tokens that belong to a certain
1
arXiv:2508.08256v2  [cs.DB]  16 Sep 2025

<!-- page 2 -->

page of the KV cache are retrieved (or not re-
trieved) simultaneously to avoid computing full at-
tention scores, leading to a coarser granularity of
retrieval.
However, we observe that in long-context tasks,
information relevant to the current query may be
scattered throughout the entire input, i.e., impor-
tant tokens are often sparsely distributed across
the KV cache (shown in Fig. 2). Consequently,
page-level retrieval inevitably leads to imprecise
selection:retrieved pages may include irrelevant
tokens, while evicted pages may exclude critical
ones, thereby affecting the model performance.
In this paper, we aim to address the retrieval
inaccuracy while preserving selection efficiency.
Specifically, we find that quantizing the key cache
to as low as 1-bit has minimal impact on the ac-
curacy of Top-kselection in token importance es-
timation (shown in Fig. 3). Despite quantization
truncates large values, important tokens are still
preserved in the Top-kafter computing quantized
attention. Based on this insight, we propose Fi ne-
Grained and Efficient KV cache Retrieval (Fier), a
1-bit quantization-based KV retrieval method. As
shown in Fig. 1(d), Fier enables more accurate re-
covery of important tokens and reduces the selec-
tion cost. This leads to improved model perfor-
mance under the same cache budget.
We evaluate Fier across PG19 (Rae et al., 2019),
LongBench (Bai et al., 2023), and the passkey re-
trieval benchmarks (Peng et al., 2023). The re-
sults demonstrate the effectiveness of Fier in both
generative and retrieval-focused settings. Experi-
ments show that Fier achieves performance com-
parable to using the full KV cache while requir-
ing only 11% of the cache budget, and consis-
tently outperforms existing KV eviction and re-
trieval baselines. Additionally, Fier achieves 1.2×
to 1.5×decoding speedup across different context
lengths on a single RTX 4090 GPU. In summary,
we make the following contributions in this work:
• We observe the sparse distribution of impor-
tant tokens within the KV cache in long-
context scenarios, highlighting the necessity
of fine-grained retrieval.
• We propose Fier, a KV retrieval method built
on 1-bit key quantization, which enables effi-
cient and accurate token-level retrieval.
• We conduct comprehensive evaluations of
Fier across diverse long-context tasks and
model architectures, demonstrating its supe-
rior performance and efficiency.
2 Related Work
Long-Context LLMs.Large Language Models
(LLMs) have transformed the landscape of natu-
ral language processing, largely due to their strong
ability to deal with long context. Their context
length capacity has increased dramatically—from
4k to 128k (Grattafiori et al., 2024), 1M (Yang
et al., 2025), and even 10M (Team et al., 2024)
tokens. This expansion unlocks a range of ad-
vanced capabilities, including o1 long-range rea-
soning (Guo et al., 2025; OpenAI, 2024), in-
context learning (Li et al., 2025), and multimodal
intelligence (Weng et al., 2024). Fier aims to
improve the inference efficiency of long-context
LLMs by exploiting the sparsity of the KV cache.
KV Cache Eviction.Previous work identified the
sparsity of attention matrices, showing that retain-
ing only a small fraction of tokens is sufficient for
the performance. For example, Xiao et al. (2023)
propose to retain the first few tokens based on the
“attention sink” phenomenon. H2O (Zhang et al.,
2023) retains a limited set of KV cache by select-
ing tokens with the highest cumulative attention
scores. SnapKV (Li et al., 2024) selects clustered
historical tokens along with a localized window of
recent tokens. However, these approaches ignore
the fact that tokens evicted can become important
in the future. Fier addresses this via query-specific
KV retrieval, enabling dynamic reassessment of
token importance at each decoding step.
KV Cache Retrieval.KV retrieval methods, in-
cluding our proposed Fier, dynamically select to-
kens relevant to the current query. However, ex-
isting approaches like Quest (Tang et al., 2024),
ArkVale (Chen et al., 2024a), PQCache (Zhang
et al., 2025), SparQattn (Ribar et al., 2023) and
MagicPIG (Chen et al., 2024b) retrieve at the page
or cluster level or compute partial tokens’ critical-
ity for efficiency, overlooking the sparse distribu-
tion of important tokens. In this paper, we pro-
pose a fine-grained, token-level retrieval strategy
that maintains efficiency while improving accu-
racy. This design better captures critical informa-
tion missed by existing methods.
KV Cache Quantization.Another related line of
work is KV quantization (Liu et al., 2024; Dong
et al., 2024), which compresses the cache by per-
forming self-attention in low-bit space. The objec-
2

<!-- page 3 -->

tive of KV quantization is to minimize global re-
construction error. In contrast, Fier achieves cache
compression by retaining only a subset of the full
KV and adopts a relaxed quantization objective fo-
cused on preserving Top-ranked tokens, enabling
the use of extremely low bit-widths.
3 Methodology
3.1 Preliminaries
In an autoregressive LLM, the inference process
typically consists of two stages: the prefill stage
and the decoding stage.
Prefill Stage.Given an input contextX∈
Rlprompt×d, the model computes the initial key
and value representations, denoted asK 0 ∈
Rlprompt×d andV 0 ∈R lprompt×d, which together
form the initial KV cache.
Decoding Stage.At each step, a new query
q∈R 1×d is generated and the corresponding key
kand valuevare appended to the initial cache
(K0,V 0)to form the current KV cache:
K← −Concat(K0,k),V← −Concat(V0,v).
The attention output is then computed as:
s=softmax(qK T ),o=sV.
The major overhead of decoding comes from com-
putation of attention scores, which reflects the im-
portance of KV tokenk j to the query. At each
step, the current query must attend to the entire
KV cache. This cost becomes higher in long-
context tasks. To address this, previous works
have demonstrated attention sparsity, observing
that initial tokens (attention sink) and recent to-
kens (locality) tend to receive higher attention
scores. Based on this, they retain a fixed subset
of tokens, denoted as(K ′,V ′)∈R n×d at these
positions for all queries, wherenis cache budget.
However, subsequent studies show that token
criticality varies across different queries. As a re-
sult, query-specific token selection is necessary,
where token importance needs to be recomputed
for each query. To improve importance estimation
efficiency, existing works tend to perform selec-
tion at a coarser, page-level granularity. For ex-
ample, Quest (Tang et al., 2024) partitions the key
cacheK∈R l×d into l
L pages (Lis typically 16
or 32). For each pageP, it extracts maximum and
minimum vectorsk max
P andk min
P ∈R 1×d, per-
forms point-wise multiplication withq,
αmax =q⊙k max
P ,(1)
αmin =q⊙k min
P ,(2)
and takes the maximum across both the hidden di-
mension and the two vectors to obtain the page im-
portance score.
sP = max
i=1,...,d
 
max
 
αmax
i , α min
i

.(3)
Pages with the highest importance scores are then
selected for self-attention computation.
During the selection, Quest loads only 2Kvec-
tors per page. AssumingKis stored infloat16,
this results in a key cache load ratio of:
2×16×l/L
l×16 = 2
L .(4)
It is clear that larger page sizes reduce importance
estimation costs, but lead to coarser granularity
by grouping more tokens together. There exists
a trade-off between efficiency and accuracy.
3.2 Fine-grained and Efficient KV Retrieval
To improve upon existing page-level KV retrieval
methods, we make the following two observations
on the token importance estimation process.
OB1: Important Token Sparsity Makes Page
Retrieval Inaccurate.To understand the trade-off
of page granularity, we visualize both the highest-
attended tokens and those selected under page-
level partitioning by mapping them back to the
original context. As shown in Fig. 2, queriesQ1
andQ2attend to different regions of the con-
text, which aligns with prior findings on the dy-
namic nature of token criticality. Moreover, tokens
with high attention scores are sparsely distributed
across the context, and we observe that pages 7,
16, and 54 each contain a mixture of both impor-
tant and unimportant tokens. This overlap makes
inaccurate retrieval, which pinpoints the necessity
of fine-grained retrieval strategies that identify im-
portant information at the token level, rather than
relying on coarse-grained page grouping.
Motivated by this, we aim to design a retrieval
strategy that operates at fine-grained token level
while incurring minimal additional computation
overhead. Notably, quantization enables low-bit
computation, achieving high efficiency while still
allowing every token to participate in the critical-
ity estimation. We begin by validating this insight
through the following observation.
3

<!-- page 4 -->

Q1: How many Monday night games did ABC show in 1976?
Key positions
Q2: Why did Clark throw over to picher Worrell?
slot 1 (attention sink)
slot 2 slot 3
slot 4 slot 5
Query positions
Attention value
?U[ GXK MO\KT G JUI[SKTZ GTJ _U[ TKKJ ZU GTY]KX G W[KYZOUT HGYKJ UT ZNOY JUI[SKTZȘȘ (KZ]KKT  GTJ  '() GMXKKJ ZU VG_
 SORROUT GTT[GRR_
LUX ZNK XOMNZY ZU HXUGJIGYZ  3UTJG_ TOMNZ MGSKY OT ȘȘ 5T 0[TK   3UTJG_ 4OMNZ (GYKHGRRȘȘ (_  '() UTR_ ZKRK\OYKJ  3UTJG_ 4OMNZ
(GYKHGRR MGSKYȘȘ IUTZXGYZ ZU ZNK  MGSKY ZU ZNGZ ]KXK YINKJ[RKJ OT ȘȘ  3UTJG_ TOMNZ MGSKY H[Z UTR_ [YKJ KOMNZ UL ZNUYK YRUZY ȘȘ 2OZZRK YW[OHHKX
ZU ZNK XOMNZ YOJK =UXXKRR XGIOTM ZU IU\KX GTJ ZNK ZNXU] JUKYT
Z MKZ NOS ȘȘ SU\KJ [V ZNK YZGXZ ZOSK LUX ZNK KGXR_ ]KKQY UL 3UTJG_ 4OMNZ ,UUZHGRR ȘȘ VRG_H_
VRG_ GTTU[TIKX LUX 3UTJG_ 4OMNZ (GYKHGRR ȘȘ )RGXQ ZNXK] U\KX ZU VOZINKX =UXXKRR ]NU ]GY X[TTOTM U\KX ZU IU\KX LOXYZ HGYK OT ZOSK ZU HKGZ ZNK YVKKJ_ 5XZG GTJ
JOJ ?KZ LOXYZ HGYK [SVOXK *UT *KTQOTMKX YZORR IGRRKJ 5XZG YGLK GZ LOXYZ ȘȘ ]NOIN 0GIQ )RGXQ SOYYKJ LUX GT KXXUX QKKVOTM (GRHUTO
Y GZHGZ GRO\K ȘȘ :NK OTLGSU[Y
GTJ IUTZXU\KXYOGR RKGJULL YOTMRK H_ 5XZG GTJ ZNK 0GIQ )RGXQ KXXUX K\KTZ[GRR_ RKJ ZU ȘȘ
Context
S1
S2
S3
S5
[                              ]
page 7
[                                ]
page 54
[                              ] page 16
S4
Figure 2: High-scoring tokens selected by two different queries in LLaMA (Grattafiori et al., 2024) are mapped to
their corresponding text. Important tokens are query-dependent and sparsely distributed across the context, causing
pages to contain a mix of important and unimportant tokens, which leads to inaccuracy in page-level retrieval.
OB2: Quantization has Minimal Impact on
Top-kSelection, even at 1-bit.Quantizing KV
cache values to a lower precision significantly re-
duces the cost of data movement and attention
computation. LetK∈R l×d denote the original
key cache, wherek i ∈R 1×d is the key vector cor-
responding to thei-th token, quantization converts
eachk i to its dequantized counterpart ˜ki as
kQ
i =
 ki −z K
i
sK
i

, ˜ki =k Q
i ⊙s K
i +z K
i ,(5)
wheres K
i ,z K
i ∈R 1×d are the per-channel cali-
brated scaling and bias vectors specific to thei-
th key vector. Previous KV quantization methods
Top ones remain
Figure 3: Averaged full/quantized attention scores
along the sequence. Despite distribution distortion
caused by low-bit quantization, the Top-ktokens are
largely preserved, indicating that token criticality re-
mains identifiable under extreme quantization.
(Zhang et al., 2024) aim to optimize these fac-
tors through the calibration process to minimize
the impact of quantization on the computed atten-
tion score. This objective can be formulated as an
ℓ2 loss:
min
sK ,zK
lX
i=1

qk⊤
i −q ˜k⊤
i
2
.(6)
However, quantization introduces perturbations
on the values, drawing computation results away
from intended. The impact of quantization is more
severe if outliers exist in the distribution. Previ-
ous methods like Kivi (Liu et al., 2024), equipped
with advanced channel-wise grouping and rescal-
ing scheme, cannot quantize the KV cache below
2 bit while retaining model performance.
Meanwhile, we observe that quantizing the key
cache to low bits has significantly less impact on
the token importance estimation than retaining the
attention scores. Here we explore an extreme case
by quantizingKto just 1-bit. Using the full-
precision attention scores as ground truth, we eval-
uate whether Top-ktoken selection can still be ac-
curately recovered under such an aggressive quan-
tization setting. Specifically, we feed a long in-
put context (14k tokens) into LlaMA during the
prefill stage, and compute attention scores under
both full-precision and 1-bit quantizedKusing
the same query. As shown in Fig. 3, despite the
fact that low-bit quantization truncates large val-
ues and distorts the overall distribution, the Top-k
tokens remain largely unchanged. This suggests
that token criticality is still well captured, even un-
4

<!-- page 5 -->

der extreme quantization.
To understand the reason, we analyze the quan-
tization objective implied by the goal of token im-
portance estimation. For importance estimation,
we aim to maintain the ranking of the Top-kto-
kens rather than preserving all attention scores
precisely. Assumemis the minimum margin be-
tween the attention scores of Top-kand non-Top-
ktokens in the full-precision setting. To preserve
this ranking after quantization, it is sufficient to
ensure that the attention score of each token devi-
ates from its full-precision counterpart by at most
m/2. This leads to the following hinge objective:
min
sK ,zK
lX
i=1
max

0, m
2 −

qk⊤
i −q ˜k⊤
i

.(7)
Compared to theℓ 2 loss, the hinge loss imposes
a relaxed objective that prioritizes preserving the
relative ranking of Top-ktokens (Fig. 4). More
importantly, outlier tokens that lead to large atten-
tion scores enjoy larger margins under the hinge
loss, making their quantization errors less impact-
ful. This enables quantization with 1-bit and larger
group sizesgwhile still maintaining Top-kaccu-
racy.
Easy to minimizeHard to minimize
Figure 4: Intuition of Fier. Our relaxed quantization
objective ignores errors smaller thanm/2, allowing the
use of extremely low bit-widths while preserving Top-
kranking accuracy.
3.3 Fier Workflow
Motivated by previous observations, we propose
Fier, a token-level retrieval method based on 1-bit
linear quantization. Fier compresses the key cache
into 1-bit using a simple round-to-nearest (RTN)
quantizer. Given a queryq, approximate atten-
tion scores are computed efficiently using quan-
tized keys. Based on these scores, Fier selects
the Top-ktokens and performs full-precision self-
attention over the selected subset. We summarize
the workflow of Fier in Algorithm 1.
3.4 Theoretical Analysis of Efficiency
Beyond retrieval accuracy, the efficiency of the se-
lection and decoding stage is also critical for prac-
Algorithm 1Fier: Token-Level KV Retrieval via
1-Bit RTN Quantization
1:Input:Queryq, full-precision(K,V), group sizeg
2:Output:Attention outputo
3: // Step 1: QuantizeKto 1-bit
4: PartitionKinto groups of sizegalong each channel
5: For each group, compute the scaling factors(s, z)and
broadcast them to constructs K ,z K ∈R l×d.
6:K Q =
j
K−zK
sK
k
,K Q ∈ {−1,1} l×d # binary
7: ˜K=K Q ⊙s K +z K
8: // Step 2: Compute Approximate Attention Scores
9: ˜s=q· ˜K⊤
10: // Step 3: Select Top-kTokens
11:S q =Top-k( ˜s)
12: // Step 4: Compute Real Attention on Selected Tokens
13:K ′ =K[S q],V ′ =V[S q]
14:Return:o=softmax(qK ′⊤)V′
tical deployment. We measure the efficiency using
the key cache access ratio (CAR), defined as the
fraction of the key cache accessed throughout the
pipeline. Specifically, we compare both retrieval-
and eviction-based methods under the same top-k
setting. In the decoding phase, all methods (Fier,
Quest, and H2O) accessktokens, so their CAR is
identical. The key difference lies in the selection
phase. ForKstored infloat16, we quantize it
to 1-bit with group sizeg. Note that in addition to
the 1-bitK Q, each group also needs to store a pair
of(s, z)infloat16. Therefore, the CAR of Fier
will be calculated as:
l×1 + (l/g)×2×16
l×16 = 1 + 32/g
16 ,(8)
which decreases with a larger group size. Recall
that Quest has a load ratio of2/L. For fairness,
we setg= 32, which matches the load ratio of
1/8with page sizeL= 16as implemented in the
Quest baseline. In contrast, H2O maintains a dy-
namic pool of sizek/lduring the selection stage.
Therefore, whenk=l/10, the CAR of the three
methods can be summarized in Tab. 1. While re-
trieval does introduce additional overhead during
the selection compared to eviction, its goal is to
improve performance with only a modest CAR in-
crease.
Table 1: CAR comparison of different methods (As-
sumingk=l/10).
Method Sel. (%) Dec. (%) Total CAR (%)
H2O 10.0 10.0 20.0
Quest 12.5 10.0 22.5
Fier 12.5 10.0 22.5
5

<!-- page 6 -->

Figure 5: Language modeling evaluation. We measure output perplexity by prompting the model with input lengths
ranging from 0 to 32k tokens. Fier achieves performance comparable to full KV and significantly surpasses Quest.
4 Experiments
4.1 Setting
Datasets.We evaluate Fier on the language mod-
eling task PG19 (Rae et al., 2019). To assess
its performance on long-context QA, we further
conduct experiments on LongBench (Bai et al.,
2023) using six representative datasets: Narra-
tiveQA, HotpotQA, Qasper, TriviaQA, GovRe-
port, and MultifieldQA. The detailed information
about the six datasets is in Appendix A. We eval-
uate Fier on the passkey retrieval task (Peng et al.,
2023) to assess its ability to model long-range de-
pendencies. We also compare responses from Fier
and Quest enabled chatbots in Appendix E.
Models.We apply our method to three
open-sourced models: LLaMA-3-8B-Instruct
(Grattafiori et al., 2024), LongChat-v1.5-7B-32k
(Li et al., 2023), and Mistral-7B-Instruct (Jiang
et al., 2023). Following the same setup as in Quest,
neither Fier nor the baseline methods are applied
to the first two layers of the model. We evalu-
ate the performance of each method under varying
KV cache budgets.
Baselines.We thoroughly compare the perfor-
mance of Fier and Quest (Tang et al., 2024) across
various benchmarks. Note that in all performance
evaluation, we set the page size of Quest to 16
and the grouping size of Fier to 32 for a fair
comparison. We also compare Fier with four
KV eviction baselines: H2O (Zhang et al., 2023),
StreamingLLM (Xiao et al., 2023), SnapKV (Li
et al., 2024) and TOV A (Oren et al., 2024). The
results in the experiments are either taken from the
original paper or obtained by running open-source
code. More implementation details can be found
in Appendix B.
4.2 Insight Verification
More Accurate Retrieval of Important Tokens.
In Fig. 6, we visualize the positions of Top-64to-
kens selected by Quest with different page sizes
Full KV
Quest, page_size=16, Recall rate=44%
Quest, page_size=8, Recall rate=67%
Fier, 1bit w/ g32, Recall rate=91%
Figure 6: Fier’s token-level retrieval preserves more
Top-ktokens compared to Quest’s page-level ap-
proach, resulting in higher recall and better alignment
with full attention.
and by Fier-1bit-g32, all mapped back onto the full
KV cache. We then compute the recall rate, de-
fined as the overlap between the retrieved tokens
and those selected using the full attention score.
The experiment is conducted on LLaMA.
We observe that Quest with either small or large
page sizes tends to retain unimportant tokens and
evict important ones, due to its coarse-grained
page-level retrieval. In contrast, Fier performs
token-level retrieval through low-bit quantized at-
tention computation, resulting in a significantly
higher recall rate and better alignment with the full
attention distribution.
4.3 Performance Evaluation
4.3.1 PG19 Results
We begin by evaluating language modeling per-
plexity on PG19, a benchmark consisting of 100
books with an average length of 70k tokens. We
evaluate three different models by feeding them
texts of varying lengths and compare the results
against both Full KV and Quest (Fig. 5). Note that
both Fier and Quest are evaluated under the same
KV cache budget of 4096 tokens. Fier achieves
performance close to that of Full KV and signifi-
cantly outperforms Quest on both the LLaMA and
Mistral models.
6

<!-- page 7 -->

Figure 7: LongBench evaluation on LaMA-3-8B-Instruct. Fier outperforms all baselines across six long-context
datasets and matches full KV performance with just 1k cache budget.
Table 2: LongBench evaluation on LongChat and Mistral. Consistent performance gains of Fier on two models.
LLMs Method Multifield en NarrativeQA GovReport Avg.
512 1024 2048 4096 512 1024 2048 4096 512 1024 2048 4096
LongChat-7B
Full KV 43.2 20.88 30.89 31.66
SLM 21.17 21.29 26.55 34.82 10.69 12.46 17.55 18.94 16.85 21.88 22.77 26.96 20.99
H2O 21.15 25.07 30.28 37.75 10.67 12.96 14.75 19.31 19.73 22.69 26.15 27.55 22.34
SnapKV 36.74 37.93 40.26 42.2119.21 19.3219.2820.6822.57 23.45 26.3 28.55 28.04
Quest 38.0541.95 44.0342.41 16.51 18.76 19.37 20.12 27.54 30.1231.2731.21 30.11
Fier 39.05 39.85 42.4 42.54 17.23 17.83 19.96 19.51 30.18 30.89 30.85 31.67 30.16
Mistral-7B
Full KV 52.92 28.49 34.81 38.74
SLM 29.91 31.16 35.75 44.12 24.21 24.79 25.91 28.9 22.09 24.6 27.57 31.19 29.18
H2O 47.39 48.43 49.03 49.95 23.04 27.79 28.6 30.2 24.24 26.15 27.19 30.04 34.34
SnapKV 53.05 52.64 52.9253.4425.57 28.0930.2729.76 25.83 28.28 30.91 32.74 36.96
Quest 48.07 50.67 53.7 51.76 20.25 25.71 28.31 27.28 31.42 32.57 33.07 33.52 36.36
Fier 53.97 54.67 54.37 53.32 26.75 28.75 29.11 31.11 34.47 34.53 34.65 34.9 39.22
4.3.2 Longbench Results
We evaluate on the LongBench benchmark us-
ing LLaMA-3-8B-Instruct across a diverse set
of long-context tasks, including single-document
QA (NarrativeQA, Qasper, MultiFieldQA), multi-
document QA (HotpotQA), summarization (Gov-
Report), and few-shot learning (TriviaQA). We
also compare Fier with H2O, StreamingLLM,
SnapKV , and Quest under varying KV cache bud-
get settings. In addition, we perform evaluations
using the Mistral-7B and LongChat-7B models to
verify the generality of our method across differ-
ent model architectures.
As shown in Fig. 7, Fier consistently achieves
superior performance compared to all baselines
across six long-context datasets under various KV
cache budgets. Overall, Fier surpasses all base-
lines at cache budgets of 512, 1024, and 2048 to-
kens, and achieves comparable performance to full
KV using only 1k tokens, which is only 11% of the
full cache. This suggests that Fier benefits from
accurately retrieving critical tokens, enabling effi-
cient use of limited cache resources without com-
promising model quality. Similar results are ob-
served on the other two models. As shown in
Tab. 2, Fier shows consistent improvements on
both single-document and multi-document tasks.
4.3.3 Passkey Retrieval
We further evaluate Fier’s ability to handle long-
distance dependencies. Specifically, we employ
the passkey retrieval task (Peng et al., 2023) as
our benchmark. This task assesses whether the
model can retrieve a simple passkey from a large
amount of irrelevant content. Following the setup
in Tang et al. (2024), we use a context length of
10k tokens for evaluation with both LongChat-
7B and a smaller model, LLaMA3-1B. KV evic-
tion methods perform poorly due to their inability
to recall discarded tokens, while Quest provides
noticeable improvements over them. Fier, how-
ever, achieves even higher retrieval accuracy, per-
forming well under extremely low budgets of just
7

<!-- page 8 -->

1.22× 1.31× 1.52×
Figure 8: Decoding latency of 256 tokens on LLaMA-2-7B (Touvron et al., 2023) under varying prefill context
lengths. Fier achieves increasing speedup over full KV by restricting attention to a small subset of the cache. At
32k context length, it delivers over 1.5× acceleration.
Table 3: Passkey retrieval accuracy under 10k context
length. KV eviction methods struggle to recall dis-
carded tokens, while Quest improves retrieval perfor-
mance. Fier achieves the highest accuracy, even with
extremely low budgets (32 and 64), effectively enhanc-
ing smaller models.
Longchat-7B Cache Budget
Method32 64 128 256 512
H20 0% 1% 1% 1% 3%
StreamingLLM 1% 1% 1% 3% 5%
TOV A 0% 1% 1% 3% 8%
Quest 65% 99% 99% 99% 100%
Fier (ours) 87% 99% 99% 100% 100%
LlaMA3-1B Cache Budget
Method32 64 128 256 512
H20 0% 1% 0% 1% 2%
StreamingLLM 0% 0% 1% 2% 4%
TOV A 0% 1% 1% 3% 6%
Quest 36% 60% 92% 99% 99%
Fier (ours) 63% 87% 97% 100% 100%
32 and 64 tokens, especially improving the long-
context processing capability of smaller models
(shown in Tab. 3).
4.4 Ablation Study
Token Granularity vs. Quantized Attention.
To understand whether Fier’s performance gain
primarily stems from its fine-grained token-level
selection (as opposed to page-level) or from the
use of quantized attention as an importance met-
ric, we conduct an ablation study on LLaMA-3-
8B-Instruct, isolating these two factors. Specif-
ically, we reduce the page size in Quest to ap-
proximate finer granularity and compare perfor-
mance. In addition, we apply the averaged quan-
tized attention score as the page-level importance
metric under the same page size, and evaluate
its effect on Quest. As shown in Tab. 4, using
smaller page sizes in Quest leads to improved per-
Table 4: Ablation study. Fier benefits from both token
granularity and quantized attention. Larger group sizes
yield better efficiency but may reduce accuracy.
Method Load R. HotpotQA
512 1024 2048
Quest-p32 0.063 7.16 8.98 11.39
Quest-p16 0.125 11.78 14.03 14.33
Quest-p16-w/quant 0.125 13.54 14.77 15.26
Quest-p8 0.25 15.16 16.6617.3
Fier-g256 0.07 12.55 14.51 16.73
Fier-g128 0.08 13.96 15.53 17.04
Fier-g32 0.125 15.46 17.0 16.95
formance. However, it also increases the cache
load ratio. Additionally, incorporating quantized
attention for scoring further enhances its effective-
ness. Notably, Fier can be viewed as quantized
attention with a page size of 1, achieving the best
overall results. These results suggest that Fier ben-
efits from both the token-level granularity and the
use of quantized attention as a lightweight yet ef-
fective importance estimator.
Fier w/ Different Group Sizes.We also inves-
tigate how the group size used during key quan-
tization affects Fier’s performance. We find that
as the group size increases, the cache load ratio
decreases, but this comes at the cost of reduced
performance. Nevertheless, Fier consistently out-
performs Quest under the same cache load ratio.
4.5 Efficiency Profiling
Inference Efficiency.In Fig. 8, we present the
decoding latency of 256 tokens on LLaMA-2-7B
under different prefill context lengths. To ensure
a fair comparison with Full KV , we include both
the time spent on computing quantized attention
and the time required to recall the selected Top-
ktokens. We implement the group-wise quanti-
zation kernel using Triton, and employ the Top-k
8

<!-- page 9 -->

CUDA operator to efficiently perform Top-ktoken
recall. Fier’s efficiency gain is mainly attributed
to the speedup in the self-attention computation,
as it restricts attention to only a small subset of
the KV cache. This acceleration becomes more
pronounced as the context length increases; for
instance, at a context length of 32k tokens, Fier
achieves over 1.5× decoding speedup. We also
provide a detailed latency breakdown comparison
of Fier and Quest in Appendix D.
5 Conclusion
We present Fier, a fine-grained and efficient KV
cache retrieval algorithm that selects important to-
kens using 1-bit quantization. By involving all to-
kens in the computation, Fier enables token-level
criticality estimation, leading to improved recall
rate and model performance. Extensive experi-
ments across various tasks and model architectures
show that Fier consistently outperforms existing
methods. Notably, Fier matches full cache per-
formance using only 11% of the KV budget and
achieves a 1.2×–1.5×decoding speedup.
Limitations
Model Scale.Due to limited computational re-
sources, our experiments are restricted to models
up to 8B parameters. Evaluating Fier on larger
models (e.g., 13B, 70B) may reveal further in-
sights into its scalability and effectiveness.
System Optimization.Our current implementa-
tion uses Triton to develop low-bit operators for
quantized attention. While Triton offers flexibil-
ity and ease of development, it does not match the
low-level optimization potential of custom CUDA
kernels, potentially limiting the achievable infer-
ence speedup.
Compatibility with GQA.Fier is not yet in-
tegrated with grouped-query attention (GQA)
(Ainslie et al., 2023). This is because token prun-
ing and grouped-query attention are orthogonal in
principle: GQA reduces the number of KV heads,
while token pruning reduces the number of tokens.
Exploring their compatibility remains an impor-
tant direction for future work.
Acknowledgments
This work was based upon High Performance
Computing (HPC) resources supported by the
University of Arizona TRIF, UITS, and Research,
Innovation, and Impact (RII) and maintained
by the UArizona Research Technologies depart-
ment and the computational resource supported
by TetraMem Inc. Partial support for this work
was provided by NSF grants CNS-2122320, CNS-
2133267, and CNS-2328972. Additional sup-
port was received through the Amazon Research
Award, Cisco Faculty Award, UNC Accelerating
AI Awards, NAIRR Pilot Award, OpenAI Re-
searcher Access Award, and the Gemma Aca-
demic Program GCP Credit Award.
References
Joshua Ainslie, James Lee-Thorp, Michiel De Jong,
Yury Zemlyanskiy, Federico Lebr ´on, and Sumit
Sanghai. 2023. Gqa: Training generalized multi-
query transformer models from multi-head check-
points.arXiv preprint arXiv:2305.13245.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, and 1 others. 2023.
Longbench: A bilingual, multitask benchmark
for long context understanding.arXiv preprint
arXiv:2308.14508.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu,
Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong,
Baobao Chang, Junjie Hu, and Xiao Wen. 2024.
Pyramidkv: Dynamic kv cache compression based
on pyramidal information funneling.arXiv preprint
arXiv:2406.02069.
Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu,
Size Zheng, Xiuhong Li, Xuechao Wei, Shengen
Yan, Meng Li, and Yun Liang. 2024a. Arkvale: Ef-
ficient generative llm inference with recallable key-
value eviction.Advances in Neural Information
Processing Systems, 37:113134–113155.
Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang
Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian,
Matthijs Douze, Leon Bottou, Zhihao Jia, and 1 oth-
ers. 2024b. Magicpig: Lsh sampling for efficient
llm generation.arXiv preprint arXiv:2410.16179.
Tri Dao. 2023. Flashattention-2: Faster attention with
better parallelism and work partitioning.arXiv
preprint arXiv:2307.08691.
Shichen Dong, Wen Cheng, Jiayu Qin, and Wei Wang.
2024. Qaq: Quality adaptive quantization for llm kv
cache.arXiv preprint arXiv:2403.04643.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, and 1 others. 2024. The llama 3
herd of models.arXiv preprint arXiv:2407.21783.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong
9

<!-- page 10 -->

Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
hardt. 2020. Measuring massive multitask language
understanding.arXiv preprint arXiv:2009.03300.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna
Lengyel, Guillaume Lample, Lucile Saulnier, and
1 others. 2023. Mistral 7B.arXiv preprint
arXiv:2310.06825.
Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang
Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo,
Da Chen, Dong Li, and 1 others. 2025. Minimax-01:
Scaling foundation models with lightning attention.
arXiv preprint arXiv:2501.08313.
Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lian-
min Zheng, Joseph Gonzalez, Ion Stoica, Xuezhe
Ma, and Hao Zhang. 2023. How long can con-
text length of open-source llms truly promise? In
NeurIPS 2023 Workshop on Instruction Tuning and
Instruction F ollowing.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. Snapkv:
Llm knows what you are looking for before gener-
ation.Advances in Neural Information Processing
Systems, 37:22947–22970.
Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao
Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyril-
lidis, and Anshumali Shrivastava. 2023. Scis-
sorhands: Exploiting the persistence of importance
hypothesis for llm kv cache compression at test
time.Advances in Neural Information Processing
Systems, 36:52342–52364.
Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong,
Zhaozhuo Xu, Vladimir Braverman, Beidi Chen,
and Xia Hu. 2024. Kivi: A tuning-free asymmet-
ric 2bit quantization for kv cache.arXiv preprint
arXiv:2402.02750.
OpenAI. 2024. Learning to reason with large language
models.https://openai.com/index/
learning-to-reason-with-llms/. Ac-
cessed: 2025-05-10.
Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi,
and Roy Schwartz. 2024. Transformers are multi-
state rnns.arXiv preprint arXiv:2401.06104.
Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and En-
rico Shippole. 2023. Yarn: Efficient context window
extension of large language models.arXiv preprint
arXiv:2309.00071.
Jack W Rae, Anna Potapenko, Siddhant M Jayakumar,
and Timothy P Lillicrap. 2019. Compressive trans-
formers for long-range sequence modelling.arXiv
preprint arXiv:1911.05507.
Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley,
Charlie Blake, Carlo Luschi, and Douglas Orr. 2023.
Sparq attention: Bandwidth-efficient llm inference.
arXiv preprint arXiv:2312.04985.
Jiaming Tang, Yilong Zhao, Kan Zhu, Guangx-
uan Xiao, Baris Kasikci, and Song Han. 2024.
Quest: Query-aware sparsity for efficient long-
context llm inference, 2024.URL https://arxiv.
org/abs/2406.10774.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, and 1
others. 2024. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth ´ee Lacroix,
Baptiste Rozi `ere, Naman Goyal, Eric Hambro,
Faisal Azhar, and 1 others. 2023. Llama: Open
and efficient foundation language models.arXiv
preprint arXiv:2302.13971.
Daniel Waddington, Juan Colmenares, Jilong Kuang,
and Fengguang Song. 2013. Kv-cache: A scalable
high-performance web-object cache for manycore.
In2013 IEEE/ACM 6th International Conference on
Utility and Cloud Computing, pages 123–130. IEEE.
Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun
Chang, and Bohan Zhuang. 2024. Longvlm: Effi-
cient long video understanding via large language
models. InEuropean Conference on Computer Vi-
sion, pages 453–470. Springer.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2023. Efficient stream-
ing language models with attention sinks.arXiv
preprint arXiv:2309.17453.
An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoyan Huang, Jiandong Jiang, Jian-
hong Tu, Jianwei Zhang, Jingren Zhou, and 1 others.
2025. Qwen2. 5-1m technical report.arXiv preprint
arXiv:2501.15383.
June Yong Yang, Byeongwook Kim, Jeongin Bae,
Beomseok Kwon, Gunho Park, Eunho Yang,
Se Jung Kwon, and Dongsoo Lee. 2024. No to-
ken left behind: Reliable kv cache compression
via importance-aware mixed precision quantization.
arXiv preprint arXiv:2402.18096.
Hailin Zhang, Xiaodong Ji, Yilin Chen, Fangcheng Fu,
Xupeng Miao, Xiaonan Nie, Weipeng Chen, and Bin
Cui. 2025. Pqcache: Product quantization-based kv-
cache for long context llm inference.Proceedings of
the ACM on Management of Data, 3(3):1–30.
10

<!-- page 11 -->

Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, and Anshumali
Shrivastava. 2024. Kv cache is 1 bit per channel:
Efficient large language model inference with cou-
pled quantization.Advances in Neural Information
Processing Systems, 37:3304–3331.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher R´e, Clark Barrett, and 1 oth-
ers. 2023. H2o: Heavy-hitter oracle for efficient
generative inference of large language models.Ad-
vances in Neural Information Processing Systems,
36:34661–34710.
A Dataset Details
We use a subset of LongBench (Bai et al., 2023)
for long-context QA evaluation. Tab. 5 shows the
statistics and evaluation metrics used in the exper-
iments.
Table 5: Dataset Statistics and Evaluation Metrics
Dataset Avg len Metric #data
NarrativeQA 18,409 F1 200
Qasper 3,619 F1 200
MultiFieldQA-en 4,559 F1 150
HotpotQA 9,151 F1 200
GovReport 8,734 Rouge-L 200
TriviaQA 8,209 F1 200
B Implementation Details
For experiments on LongBench (Bai et al., 2023)
and PG19 (Rae et al., 2019), we use three models:
LLaMA-3-8B-Instruct (Grattafiori et al., 2024),
LongChat-v1.5-7B-32k (Li et al., 2023), and
Mistral-7B-Instruct (Jiang et al., 2023), with the
maximum input length uniformly set to 32k tokens
to ensure a fair comparison. All inference runs
are conducted on NVIDIA A6000 GPUs. Dur-
ing the self-attention phase, we utilize FlashAtten-
tion (Dao, 2023) for acceleration, except for H2O
(Zhang et al., 2023), which requires computation
of historical attention scores and therefore cannot
benefit from FlashAttention.
Except for H2O (Zhang et al., 2023) and Quest
(Tang et al., 2024), which are run using their
publicly released implementations, all other base-
lines are run using the unified KV Cache-Factory
framework (Cai et al., 2024). Experiments on
observations and decoding latency are conducted
separately on NVIDIA RTX 4090 GPUs.
C Comparison of Fier and MiKV
Different from the methods discussed in the main
text, MiKV (Yang et al., 2024) adopts a hybrid
strategy: during decoding, it stores top tokens in
high precision while retaining the remaining to-
kens in a low-bit format. The main distinction be-
tween Fier and MiKV is that MiKV maintains the
entire KV cache in mixed precision, whereas Fier
only preserves top tokens during decoding. To en-
sure fairness, we evaluated Fier on the MMLU
dataset (Hendrycks et al., 2020) using the same
cache size as reported for MiKV . Note that the
cache size of Fier accounts for the CAR in the
selection phase, which is not explicitly stated in
the MiKV . As shown in Tab. 6, Fier demonstrates
better performance compared to MiKV under the
same cache budget.
Table 6: Comparison of Fier and MiKV on MMLU.
Cache Size (%) Method Acc. (%)
25 MiKV 43.9
25 Fier 45.12
20 MiKV 42.7
20 Fier 43.75
D Latency breakdown comparison with
Quest
We provide a detailed latency breakdown between
the pipelines of Fier and Quest in Tab. 7. The pri-
mary difference lies in the importance estimation
stage: Fier uses an extreme 1-bit quantized atten-
tion, while Quest adopts page-level attention. In
the top-krecall stage: Quest retrieves pages, while
Fier directly retrieves top-ktokens. The final top-
kself-attention is identical for both methods. In
Tab. 7, we compare the average per-layer latency
for each token, as well as the end-to-end genera-
tion latency. With the samek, Fier has higher la-
tency than Quest due to its fine-grained top-ksort-
ing in the recall stage. However, Fier achieves bet-
ter performance. By reducingk, Fier can achieve
lower latency than Quest while still maintaining
better performance. Overall, Fier offers a better
trade-off between latency and performance com-
pared to Quest.
E Comparison of Chatbot Responses
To better illustrate the practical differences be-
tween the two retrieval methods, we deploy
a LLaMA-3-8B-Instruct chatbot using Fier and
Quest, respectively. Given the same long con-
text and user question, we present the correspond-
ing responses from each chatbot for a qualitative
comparison (Fig. 9). In the first example, which
asks about the orders in which Mufti-e-Azam-e-
Hind received Khilafat, the Fier-enabled chatbot
11

<!-- page 12 -->

Table 7: Time breakdown of Fier and Quest with different Top-k.
Stage Fier (k= 2048) Fier (k= 512) Quest (k= 2048)
Importance Estimation 16µs 11µs 14µs
Top-kRecall 41µs 25µs 23µs
Top-kSelf-Attn 68µs 45µs 68µs
End-to-End Latency 20.5ms16.7ms18.3ms
GovQA Perf.34.4233.98 33.7
Ghousul Waqt, Mufti-e-Azam-e-Hind (radi Allahu anhu) was born on Monday, 22nd of Zil Hijjah 1310 AH (18 July 1892) in Bareilly Shareef, India. ... His father,
Imam-e-Ahle Sunnat, A‘la  Hazrat, Imam Ahmed Raza Khan Al Qaderi (radi Allahu anhu) was also born in this city…… . Hazrat Makhdoom Shah Abul Hussain
Ahmadi Noori (radi Allahu anhu) named the child "Abul Barkaat Muhiy’yuddeen  Jilani". He was later named "Mustapha Raza Khan".... When Sayyiduna Abul
Hussain Ahmadi Noori (radi Allahu anhu) visited Bareilly Shareef, he saw the six-month-old child and said, “This  child will be of great assistance to the Deen ...
He is a Wali.”He  then placed his blessed finger in the child’s  mouth and made him a Mureed, blessing him with I'jaazat and Khilafat at the same time.Not only
did he receive Khilafat in the Qaderi Silsila (Order), but also in the Chishti, Nakshbandi, Suharwardi, and Madaari Orders.He also received Khilafat from his father,
A'la Hazrat, in the Qaderi Silsila....He received most of his education from his family, especially from A'la Hazrat, and became proficient in Tafseer, Hadith, Fiqh,
Sarf, Nahw, Tajweed, etc. ... He wrote his first Fatawa at age 13 on the topic of "Raza’at " (milk relations).
Context
Question: Which orders did Mufti-e-Azam-e-Hind receive Khilafat from?
Ground Truth: “Mufti-e-Azam-e-Hind received Khilafat in the Qaderi, Chishti,
Nakshbandi, Suharwardi, and Macari Orders.”
“Qaderi, Chishti, Nakshbandi, Suharwardi, and Madaari. He also
received Khilafat from his father, A'la Hazrat.”
“Sayyiduna Ghousul Azam, Sheikh Abdul Qari Mslihud’ deen and
Hazrat Muhaddith-e-Azam-e-Hind.
Fier
Quest
\section{Introduction}In recent years, vehicular technology has attracted significant attention from the automotive and telecommunication industries, leading
to…… . V2X supported by the sixth generation (6G) is envisioned to be a key enabler of future connected autonomous vehicles \cite{9779322}. Modernized
vehicles are augmented with various types of sensors divided into exteroceptive and proprioceptive…… Existing methods for GPS spoofing detection include GPS
signal analysis and message encryption \cite{9845684}, but these require either ground truth or costly infrastructure…… In this work, we propose a method to
jointly detect GPS spoofing and jamming attacks in the V2X network. A coupled generalized dynamic Bayesian network (C-GDBN) is employed to learn the
interaction between RF signals received by the RSU from multiple vehicles and their corresponding trajectories. ...The main contributions of this paper can be
summarized as follows: …… \subsection{Joint Prediction and Perception} RSU starts predicting the RF signals it expects to receive using a Modified Markov Jump
Particle Filter (M-MJPF). To leverage cross-signal correlations encoded in C-GDBN, we introduce an Interactive M-MJPF (IM-MJPF). This filter:……
Context
Question: What is the name of the generative interactive model used in the method?
Ground Truth: "The generative interactive model used in the method is called
the Coupled Generalized Dynamic Bayesian Network (C-GDBN)."
“Coupled Generalized Dynamic Bayesian Network (C-GDBN).”
“IM-MJPF. Modified Markov Jump. Interactive M-MJPF.”
Fier
Quest
Figure 9: Chatbot responses from Fier and Quest. Fier provides more complete and accurate answers in both
examples.
correctly identifies all five orders, while the Quest-
enabled chatbot only retrieves a single name, miss-
ing key information. A consistent trend is ob-
served in the second example, which involves a
scholarly article. When asked about the generative
model adopted in the paper, the Fier-based chat-
bot accurately identifies the overall framework,
whereas the Quest-based chatbot focuses narrowly
on a sub-module mentioned in a later section.
12
