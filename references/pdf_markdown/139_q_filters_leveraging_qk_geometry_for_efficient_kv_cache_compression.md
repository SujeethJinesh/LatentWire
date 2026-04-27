# references/139_q_filters_leveraging_qk_geometry_for_efficient_kv_cache_compression.pdf

<!-- page 1 -->

Q-Filters: Leveraging Query-Key Geometry
for Efficient Key-Value Cache Compression
Nathan Godey 1 2 Alessio Devoto 3 Yu Zhao4 Simone Scardapane 3
Pasquale Minervini 4 5 Éric de la Clergerie 2 Benoît Sagot 2
/githubgithub.com/NathanGodey/qfilters
Abstract
Autoregressive language models rely on a Key-
Value (KV) Cache, which avoids re-computing
past hidden states during generation, making it
faster. As model sizes and context lengths grow,
the KV Cache becomes a significant memory bot-
tleneck, which calls for compression methods that
limit its size during generation. In this paper, we
discover surprising properties of Query (Q) and
Key (K) vectors that allow us to efficiently ap-
proximate attention scores without computing the
attention maps. We propose Q-Filters, a training-
free KV Cache compression method that filters
out less crucial Key-Value pairs based on a sin-
gle context-agnostic projection. Contrarily to
many alternatives, Q-Filters is compatible with
FlashAttention, as it does not require direct ac-
cess to attention weights. Experimental results in
long-context settings demonstrate that Q-Filters
is competitive with attention-based compression
methods such as SnapKV in retrieval tasks while
consistently outperforming efficient compression
schemes such as Streaming-LLM in generation
setups. Notably, Q-Filters achieves a 99% accu-
racy in the needle-in-a-haystack task with a ×32
compression level while reducing the generation
perplexity drop by up to 65% in text generation
compared to Streaming-LLM.
1. Introduction
The performance of Large Language Models (LLMs) as
autoregressive text-generation systems relies on the effec-
tiveness of the Transformer architecture (Vaswani et al.,
2017). Recently, long-context models such as Gemini-Pro-
1.5 (Reid et al., 2024), Claude-3 (Anthropic, 2024), GPT-
4 (Achiam et al., 2023), and Llama3.1 (Dubey et al., 2024)
1Sorbonne Université, Paris, France 2Inria, Paris,
France 3Sapienza University of Rome 4University of Ed-
inburgh 5Miniml.AI. Correspondence to: Nathan Godey
<nathan.godey@inria.fr>.
5 6 7 8 9 10 11
Time T o First T oken (s)
30
35
40
45
50
55
60Ruler average score
SnapKV
Q-Filters (ours)
Streaming-LLM
Expected attention
K-norm
Figure 1: Accuracy vs Time to First Token (TTFT) tradeoff
for Llama-3.1-70B-Instruct, measured on the Ruler dataset
with ×8 compression. The TTFT is measured using 2 A100
GPUs on 8192-tokens sequences.
have demonstrated the ability to process hundreds of thou-
sands of tokens. However, processing such long sequences
comes with significant challenges, as it may lead to higher
decoding latency and memory saturation. As the context
length grows, each inference step involves storing an in-
creasingly large context from GPU memory in the form of
the KV Cache, creating a memory bottleneck that hinders
efficient inference (Fu, 2024). To address this issue, KV
Cache compression methods aim to reduce the size of this
past-context representations storage by removing or merging
Key-Value pairs, thereby alleviating memory bottlenecks.
While KV Cache compression techniques have gained popu-
larity, many approaches require fine-tuning or re-training the
underlying models (Nawrot et al., 2024; Ainslie et al., 2023;
DeepSeek-AI et al., 2024), which limits their applicability
in real-world deployment scenarios. Training-free methods
have also been proposed, but they often rely on access to
attention weights to evaluate the importance of Key-Value
pairs (Xiao et al., 2024; Li et al., 2024), making them incom-
patible with the widely adopted efficient attention algorithm
FlashAttention (Dao, 2024). These methods usually require
a partial re-computation of the attention matrices, which
1
arXiv:2503.02812v1  [cs.CL]  4 Mar 2025

<!-- page 2 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
(a) Layer 14, Head 5 (ϵ = −1)
 (b) Layer 31, Head 14 (ϵ = +1)
 (c) SVD absolute average coefficients
Figure 2: Left and center: distributions of the projections of Qh and K h on uh for Llama-3.1-8B. Right: estimates ofEi(⟨Qh
i , vm⟩)
 where vm are the right vectors from the SVD of a set of Qh representations from different Llama models,
averaged over all layers and heads.
leads to a time and memory overhead. Hence, these algo-
rithms are often used to compress prompts before generating
answers and are not ideally suited for memory-constrained
generation.
In this work, we propose Q-Filters, a training-free KV Cache
compression method that uses the geometric properties of
Query-Key to filter out the less important Key-Value pairs.
Our approach achieves competitive results across synthetic
tasks and pure generation cases while maintaining compati-
bility with FlashAttention and, thus, better time efficiency.
Analysing the properties of queries (Q) and Keys (K) dis-
tributions, we find that a single direction, spanned by the
principal eigenvector of Q, encodes an input selection pro-
cess for each head. Identifying this direction allows us to
efficiently estimate which inputs are mostly ignored by a
given head and can thus be discarded with minimal per-
formance loss. Interestingly, we find that this direction is
context-agnostic, i.e., the directions we identify in different
contexts are highly consistent. Leveraging this property,
we calculate lightweight projections, which we refer to as
Q-Filters, based on a small held-out calibration dataset only
once for every model, incurring minimal computational
overhead. At inference time, we use Q-Filters to project
Keys in the pre-computed direction to estimate the impor-
tance of Key-Value pairs without accessing attention scores,
and we prune the KV Cache accordingly. This makes our
method faster than most KV Cache compression alternatives
that use attention scores to estimate the importance of the
KV pairs.
Additionally, our method is training-free, requiring only a
very short initial calibration, and we show it can be easily
applied to a variety of decoder-only language models. We
validate our method on a wide set of tasks, ranging from
language modelling to in-context learning and long-context
tasks, achieving competitive performance even with 32x
compression ratios.
2. Background
2.1. Key-Value Cache
We first introduce the relevant notation for our analysis
and the role of the KV Cache in efficient LLM inference.
Consider a transformer model with a hidden dimension dm
and nl layers, processing a sequence of length L. Each
transformer layer processes the input sequence via Multi-
Head Self-Attention (MHA).
In MHA, the model transforms the input features X ∈
RL×dm into three distinct representations for each attention
head h ∈ [1, H]. These representations, known as queries
Qh, Keys K h, and ValuesV h, each belong toRL×dh, where
dH = dm/H represents the dimension per head, and h de-
notes the head index. The second step computes the atten-
tion output Oh for each head using the following equation:
Oh = softmax
 Qh(K h)T
√dH

V h.
In causal modelling, where the model generates text sequen-
tially, we ensure that each token only attends to previous
tokens and itself. This causality constraint means that when
generating the t-th token, its output Oh
t depends only on the
current and previous inputs, as expressed by:
Oh
t = softmax

Qh
t (K h
≤t)T
√dH
!
V h
≤t.
The Key and Value representations K h
≤t, V h
≤t, which com-
bine previous Keys and Values with the current ones
K h
t , V h
t , reuse information from previous generation steps.
By storing these representations in a KV Cache during the
generation process, we can avoid the computational cost
of recalculating them at each step, thereby significantly im-
proving efficiency at the cost of the memory occupied by
stored KV pairs.
2

<!-- page 3 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
(a) Layer 0, Head 21
(b) Layer 13, Head 16
Figure 3: Projection of Qh and K h vectors in the first two
components of the SVD of Qh for different heads in Llama-
3.2-1B. The colour on the K projections represents the
log-average attention at the corresponding index for the
current head. The x-axis and y-axis indicate the results of a
projection of the representations on v1 and v2, respectively.
However, this memory-compute tradeoff introduces a new
challenge: as the context length grows, decoding latency
increases due to the frequent transfers of large KV Cache
states between high-bandwidth memory (HBM) and stream-
ing multiprocessors (SM) (Fu, 2024). For this reason, KV
Cache compression methods have become essential to allow
inference in long contexts.
2.2. Geometry of Multi-Head Attention
In Devoto et al. (2024), the authors examined a relationship
between basic characteristics of the Key representations
and attention score distributions. Notably, they observe a
negative correlation between the average attention weight
given to a position and the L2-norm of the K h
t vector at
that position. Leveraging this observation, they propose to
compress the KV Cache by selecting the KV pairs for which
||K h
t ||2 is the smallest. Using this simple heuristic, they
are able to reach ×2 compression ratios without altering
the retrieval and modelling performance of the models they
study. In their paper, while they relate this approach to the
well-known oulier dimension phenomenon (Kovaleva et al.,
2021), they do not provide a grounded explanation as to the
strength of the observed correlation.
A promising path towards a better explanation of the L2-
norm observation consists in systematically exploring the
geometry of the representations involved in the attention
score computation, namely Qh and K h.
Godey et al. (2024) show that the distributions of Qh
t and
K h
t are anisotropic, i.e. they do not uniformly occupy RdH .
They observe that both distributions “drift away” from the
origin as training progresses. Crucially, this drift occurs
along parallel directions in RdH , so that the dot product
between mean Qh
t and mean K h
t representations tends to
increase in absolute value, and to be either positive or neg-
ative for different heads. In the paper, it is argued that this
drift could be linked to the sparsity of attention patterns, but
the authors do not propose a clear interpretation of this phe-
nomenon from the perspective of the attention mechanism.
In this paper, we bridge the gap between the two aforemen-
tioned observations; namely, we explain the effectiveness
of the L2-norm heuristic introduced in Devoto et al. (2024)
by leveraging the (jointly) anisotropic nature of Query-Key
representations, and we explore a stronger heuristic that
exploits this finding to refine the L2-norm approximation
by projecting Keys onto the drift directions, that we refer to
as Q-Filters.
3. Method
3.1. Exploring the Query-Key Geometry
Motivated by Devoto et al. (2024) and Godey et al. (2024),
we propose to further explore some geometrical properties
of Qh and K h vectors and their implications for unnormal-
ized attention logits Qh(K h)T .
First, we formalize the findings from Godey et al. (2024)
into our theoretical framework. The authors shed light on
the existence of a favored common normalized direction for
both Qh and K h distributions. We denote such direction
as uh ∈ SdH −1 where SdH −1 is the dH-dimensional hyper-
sphere (i.e. SdH −1 = {x ∈ RdH s.t. ||x||2 = 1 }). As a
consequence, the projection of Qh and K h distributions on
uh is usually non-null but can take opposite signs in Qh and
K h. Hence, we use ϵ = ±1 to account for the possible sign
discrepancy and formulate the following Observation 3.1 in
terms of expectation.
Observation 3.1 (Joint anisotropy) . There exist uh ∈
SdH −1 and ϵ = ±1 such that
E
 
⟨Qh
i , uh⟩

> 0 and E
 
⟨K h
j , ϵuh⟩

> 0,
3

<!-- page 4 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Figure 4: Spearman rank correlation between KV com-
pression scoring metrics and the observed attention Sh for
Llama-3.2-1B, for K-norm (top) and Q-Filters (bottom).
where ⟨·, ·⟩ denotes the dot product.
To validate Observation 3.1, we compute the Singular Value
Decomposition (SVD) of a set of Qh representations taken
from various sequences for Llama-3.1-8B. We find that the
first right-vector of the SVD verifies Observation 3.1 for
all tested heads, and we display examples of projection dis-
tributions in Figures 2a and 2b. The intuitive consequence
of this observation regarding attention weights is that, if
a given K h
t has a strong projection along ϵuh, then future
queries Qh
≥t can be expected to have a stronger dot-product
with K h
t in average.
However, it is not clear a priori that this effect is uni-
directional, i.e. that there exists a unique direction uh (up
to a sign) that verifies Observation 3.1. Hence, identify-
ing one such direction may not suffice to characterize the
anisotropy of Qh representations and to derive estimations
of the dot-products used in attention. The uni-directional
nature of the Query-Key anisotropy can be formalized as in
Observation 3.2.
Observation 3.2. Let uh = arg maxu∈SdH −1 E
 
⟨Qh
i , u⟩

and B = ( uh, u2, ..., udH) an orthonormal basis of RdH .
Then for all attention inputs X:
∀m ∈ [2, dH], E
 
⟨Qh
i , um⟩

≈ 0
In Figure 2c, we observe that only the first singular compo-
nent of the SVD ofQh representations carries an anisotropic
behavior, as the projections on all other components have
a null mean. Hence, by taking the SVD right-vector basis
as B, we can show that the first component of the SVD
empirically verifies Observation 3.2. This lets us derive
a basic estimation for the average unnormalized attention
logits ⟨Qh
i , Kh
j ⟩.
Theorem 3.3 (proof in Appendix A) . Under Observa-
tion 3.1 and Observation 3.2, we have:
EQh
i
(⟨Qh
i , Kh
j ⟩) ≈ κh⟨K h
j , uh⟩
where κh is a positive constant.
Intuitively, projecting K h
t along the anisotropic direction
uh gives us an estimate of the attention logits that involve
K h
t up to a positive multiplicative constant κh.
This result provides a justification for the method developed
in Devoto et al. (2024). As a matter of fact, Observation 3.1
implies that Ej
 
cos(K h
j , uh)

should have the same sign
as ϵ. In practice, we observe ϵ = −1 for a vast majority of
heads in trained causal LMs. Hence, we can derive a looser
estimation from Theorem 3.3:
Ei,X(⟨Qh
i , Kh
j ⟩) ≈ −κhEj,X
 
cos(K h
j , uh)
 ||K h
j ||2
This estimation shows that the L2-norm of K h
j vectors is
negatively correlated with the corresponding mean attention
logits and can therefore be used to approximate them. How-
ever, only using the L2-norm to estimate the attention score
as done in Devoto et al. (2024) is suboptimal, as it ignores
the angular component of the ⟨K h
j , uh⟩ product. In practice,
one can approximate uh as defined in Observation 3.2 using
the SVD of concatenated representations Qh extracted by
passing samples through the model. Formally, we collect
a batch of Query activations Qh = {Qh
1 , Qh
2 , ..., Qh
n} by
passing documents sampled from pre-training corpora and
using the right-vectors V as the orthonormal basis B:
Qh = UΣV ⊤, with V = (v1, v2, ..., vdH) (1)
The resulting v1 vectors are, up to a sign, what we refer to as
Q-Filters, as they allow to estimate which Key-Value pairs
are worth storing for each head along generation. Figure 3
also displays information about attention levels for the cor-
responding indices. For a given input X, we measure the
average attention at position t as:
S h
t = 1
L − t + 1
LX
i=t
Ah
it,
4

<!-- page 5 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
where Ah is the attention map for head h. It appears clearly
from Figure 3 that there exists a strong correlation between
the average attention at a given index and the projection of
K h on the v1 component.
We observe that the projection of K h on the v1 component
has a consistent sign for a given head, e.g., it is consistently
positive in Figure 3a and consistently negative in Figure 3b,
while the projection results on v2 have a near-zero expecta-
tion, further validating Observation 3.1 and Observation 3.2.
3.2. Q-Filters
Based on Theorem 3.3, we can design a KV Cache compres-
sion scheme that consists of the following steps:
1. For a given model, retrieve its Q-Filters, which can be
obtained with the following procedure:
(a) Gather Qh representations by passing samples
through the model;
(b) Compute the SVD of the gathered representations
at each layer and head;
(c) Obtain the positive right vector (or Q-Filter) for
each head v+
1 = sgn(1uT
1 )v1.
2. At inference, for each head, discard the K h
t with the
lowest ⟨K h
t , v+
1 ⟩ value.
In the case of Grouped-Query Attention or GQA (Ainslie
et al., 2023), we simply average the Q-Filters for each group
of Query representations.
We bring the attention of the reader to the fact that this
method only requires a single preparation step following
training for a given model. The Q-Filters are entirely
context-agnostic and rely on inherent properties of the Query
and Key latent spaces. In the rest of this article, we use a
subset of the Pile dataset (Gao et al., 2020) to compute the
Q-Filters and discuss the choice of the dataset and of the
number of necessary SVD samples in Section 4.1.
In Figure 4, we observe that the Q-Filters heuristic is notice-
ably more correlated with the attention score Sh for most
heads compared to the L2-norm metric. As such, ordering
the Key-Value pairs using the Q-Filters heuristic should al-
low us to select more relevant pairs than using the method
from Devoto et al. (2024) - that we will call K-norm for the
sake of simplicity.
4. Experiments
We validate our method both on memory-constrained lan-
guage modelling and on long-context retrieval tasks (e.g.
needle-in-a-haystack). Additionally, we test our method on
the Ruler dataset (Hsieh et al., 2024), which is specifically
designed to test the model’s long context modelling abilities.
We test Q-Filters on Llama-3.1-8B, Llama-3.1-70B (Dubey
et al., 2024) and Qwen-2.5-7B (Qwen et al., 2025), but the
method can be easily adapted to any pre-trained decoder-
only LLM. We compare Q-Filters with several KV Cache
compression methods. These include StreamingLLM (Xiao
et al., 2024), which focuses on language modeling by al-
ways retaining the initial tokens of the sequence. We also
compare with SnapKV (Li et al., 2024), which performs
compression based on attention scores from the final portion
of the prompt, making it particularly suitable for compres-
sion of large prompts. Additionally, we compare against
preserving low-L2 norm tokens (Devoto et al., 2024) and
the recent ExpectedAttention (Jegou & Jeblick, 2024).
Language Modelling To evaluate the performance of Q-
Filters in the language modelling setup, we perform gen-
eration on the Pile dataset (Gao et al., 2020). We let the
KV Cache grow up until a certain threshold, after which
we start evicting the KV pairs so that the total size never
exceeds the maximum threshold. We measure performance
by tracking the model perplexity computed on past tokens
in 20 sequences. We report results for a maximum KV
Cache size of 512 pairs in Figure 5. We observe that Q-
Filters consistently achieves the lowest perplexity among
compression schemes, even for very long contexts. This
observation scales to the 70B model, where Q-Filters sig-
nificantly reduces the perplexity gap. This improvement is
more pronounced in the latter portions of the sequences, sug-
gesting better retention of relevant contextual information.
Needle in a Haystack The Needle-in-a-Haystack task em-
beds a key piece of information (the “needle”) within a long
sequence of distractors (the “haystack”), followed by a ques-
tion that requires retrieving the needle. This evaluates the
model’s ability to handle long-range dependencies and tests
how well KV Cache compression retains critical informa-
tion. If important KV pairs are evicted, the model fails to
answer correctly.
We evaluate Q-Filters by placing the needle at depths from
1k to 64k tokens and measuring retrieval accuracy. Simi-
larly to (Devoto et al., 2024), we do not compress key-value
pairs in the first two layers of the models in this experiment.
As shown in Figure 6, Q-Filters outperforms K-Norm (De-
voto et al., 2024), preserving crucial information even in
extremely long contexts.
Ruler Tasks We evaluate the proposed method on the
Ruler dataset (Hsieh et al., 2024), which comprises sev-
eral sub-tasks that test the model long context modelling
abilities, including Multi-hop Tracing, Long Context Ag-
gregation, Long Context Retrieval and Question Answer-
5

<!-- page 6 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Figure 5: Generation performance for a KV Cache size
limited to 512 items for Llama-3.1-8B (top) and Llama-3.1-
70B (bottom).
ing. The dataset offers 3 variants with different sequence
lengths: 4096, 8192, and 16384. We compare the score on
Ruler with several other KV Cache compression methods
and show average results in Figure 7a. We report detailed
per-task results in Table 1 and in Appendix C. We test the
model’s score for several compression factors ranging from
2× to 32 ×. While for some lower compression factors,
we find performance on par with other methods, Q-Filters
achieve the highest score with the strongest compression
factor of 32×, demonstrating the method’s effectiveness at
high compression rates.
4.1. Robustness of the Calibration Dataset
In Figure 8, we analyse how the calibration dataset size
impacts the performance of our Q-Filters computation. Our
experimental results demonstrate that increasing the number
of samples in the calibration dataset leads to an improve-
ment in average perplexity, although the marginal benefits
diminish beyond a certain point, namely around 1k samples.
This suggests that while larger calibration datasets gener-
ally produce more robust Q-Filters, there exists a practical
trade-off balancing computational cost and performance
(a) K-norm (average accuracy: 63%)
(b) Q-filters (average accuracy: 91%)
Figure 6: Needle-in-a-haystack performance for Llama-3.1-
8B using 64x KV Cache compression.
benefits. Based on these empirical findings and computa-
tional efficiency considerations, we standardized our experi-
mental protocol to utilize 3,000 samples for computing the
Q-Filters across all subsequent experiments. Another im-
portant consideration in the development of robust Q-Filters
is the choice of calibration dataset. To investigate this as-
pect, we conducted a systematic analysis using multiple
diverse datasets and model versions in Figure 9. Our experi-
ments revealed that the Q-Filter vectors exhibit remarkable
stability across different calibration datasets, with a high
average cosine similarity between vectors computed from
distinct sources. This finding suggests that our method is
relatively insensitive to the specific choice of calibration
data, provided it maintains sufficient diversity and quality.
Based on these results, we opted to use a carefully curated
subset of the Pile dataset (Gao et al., 2020) for all Q-Filter
computations.
4.2. Q-Filters Estimation Overhead
It could be argued that our method introduces a memory
overhead as we need to store the Q-Filters on-device. Nev-
ertheless, for a model using l layers and nH heads, storing
the Q-Filters requires l × nH × dH parameters. For Llama-
6

<!-- page 7 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Compression method FA-compatibleCWE FWE Multi-Key Multi-Query Multi-Value Single QA VT Average
SnapKV ✗ 88.7 89.0 15.1 29.6 28.8 68.7 42.8 83.2 50.5
Expected Attention ✗ 70.0 79.3 12.0 59.7 37.8 31.2 44.2 96.3 43.2
Streaming-LLM ✓ 53.8 93.4 14.1 16.8 16.7 15.7 62.3 15.8 31.6
K-Norm ✓ 22.9 74.8 8.7 16.6 25.8 55.9 20.6 32.0 31.3
Q-Filters (ours) ✓ 82.5 80.2 22.9 49.1 60.6 71.1 37.6 100 56.1
Table 1: Results on the Ruler-4096 dataset for Llama-3.1-70B-Instruct with an 8× compression ratio. The second column
indicates compatibility with FlashAttention.
(a) Average performance on Ruler (8192)
(b) Average performance on Loogle (Short Depen-
dency QA)
Figure 7: Average score for different long-context bench-
marks using Llama-3.1-8b with different methods and com-
pression ratios
3.2-1B, this is 36k× smaller than the total parameter count
and 196k× smaller in the case of Llama-3.2-405B. Another
source of overhead could be attributed to the initial compu-
tation of the filters that are required for every new model.
We find that passing 20 samples of length 2048 through
the model and performing the SVD on 3k randomly sam-
pled representations for each head is sufficient to obtain
strong performance. In our experiments with Llama-3.2-
70B, computing the filters took less than 3 minutes on two
A100-80GB GPUs. This cost is thus negligible when com-
pared with the cost of inference.
101 102 103
SVD samples
1.934
1.936
1.938
1.940
1.942
1.944
1.946Final perplexity
Figure 8: Perplexity after 1024 tokens for Q-Filters obtained
using different sizes of Qh (Eq. (1)) to calculate the SVD.
4.3. Throughput and Scalability
In this section, we analyze the time and memory overhead
induced by the Q-Filters method. Our approach is more
efficient than many KV Cache compression methods, as it
estimates the relevance of aK h representation without mate-
rializing the attention maps. This property makes it compat-
ible with memory-efficient self-attention implementations
such as FlashAttention (Dao, 2024). During inference, Q-
Filters maintains the same theoretical time complexity as
the K-norm method (Devoto et al., 2024), since computing
a norm and a scalar product require a comparable number
of floating-point operations.
By avoiding the explicit computation of attention scores,
our method achieves lower inference latency compared to
existing approaches. To quantify this efficiency, we mea-
sure the Time to First Token across different methods in
Figure 10. Time to First Token (TTFT) refers to the latency
between submitting a prompt and receiving the first gener-
ated token. This metric is particularly relevant in scenarios
where fast response times are critical, such as interactive
AI applications. Compressing the KV Cache directly im-
pacts TTFT: by reducing the memory footprint of the KV
Cache, it allows a larger portion of the prompt context to fit
within fast-access memory, minimizing memory swapping
overhead. As a result, compression techniques that effi-
ciently manage the KV Cache should significantly reduce
7

<!-- page 8 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Raw modelInstruct model
Instruct modelRaw model
Figure 9: Cosine-similarity between Q-Filters computed
on datasets coming from different domains and languages
and on pre-trained and post-trained models. The scores are
averaged over all layers and heads.
1.33 2 4 8 16 32
Compression factor
102
103
Time to first token (ms)
Q-Filters (ours)
SnapKV
Streaming-LLM
Expected attention
Low L2-norm
Figure 10: First token latency across KV Cache compression
methods of Llama-3.2-8B with a length of 64k prompt.
initial response latency. Notably, our experiments show that
Q-Filters maintain this performance advantage even as the
sequence length increases, suggesting better scalability com-
pared to methods that require explicit attention computation.
5. Limitations
In Appendix B, we run generation experiments on Qwen-
2.5-7B-Instruct (Qwen et al., 2025), and we observe that,
although the results still favour the Q-Filters method, the
gap is less clear compared to the Llama models. Our main
hypothesis for this discrepancy lies in the slightly different
attention mechanism used in Qwen-2.5 suite, which adds
a bias to the QKV projection. Hence, it is likely that the
geometrical observations made in Section 3 are not accurate
in that case. Similarly, initial experiments with Olmo-2
models (OLMo et al., 2025) were unsuccessful, which can
be explained by their use of the QK-normalization technique
(Dehghani et al., 2023). These different tricks would most
likely require an adaptation of our analysis to yield a better
approximation of the attention distributions.
6. Related Works
After the success of long-context models (Reid et al., 2024;
Anthropic, 2024; Achiam et al., 2023), compressing the KV
Cache has become a key research focus to enable processing
of long-context inputs.
Some methods reduce the KV Cache size by modifying the
model architecture. For example, Ainslie et al. (2023) and
Shazeer (2019) reuse the same Keys for multiple queries,
thereby reducing redundancy in storage. Nawrot et al.
(2024) propose a dynamic token-merging strategy, learning
which KV pairs to merge. While these approaches achieve
significant compression, they require training or fine-tuning,
making them less practical in real-world scenarios where
retraining the model from scratch is not feasible. In con-
trast, our method requires only a short, computationally
inexpensive calibration step, avoiding parameter updates
entirely. Recently DeepSeek-AI et al. (2024) introduced a
Multi-Head Latent Attention, a modification to the standard
attention mechanism that performs a low-rank reduction of
the KV Cache during pre-training.
Training-free approaches aim to compress the KV Cache
without modifying the model, typically by approximating
the attention score over long sequences and prioritizing
tokens with higher importance. Among these, Xiao et al.
(2024) focus on language modelling tasks and propose al-
ways retaining the first token(s) (as an attention sink) and the
last n tokens in a sliding window. Also, Zhang et al. (2024)
focuses on generation tasks and introduces a policy that
evicts tokens during generation based on a scoring function
derived from cumulative attention. In contrast, other works
focus on the task of compressing a large prompt provided
by the user. Li et al. (2024) uses attention from the last part
of the prompt to estimate KV pairs importance. With the
same goal, Cai et al. (2024) assigns more cache budget to
lower layers and less to higher layers. Finally, Guo et al.
(2024) proposes to rescale the KV score of other methods
by the L1 norm of the Values.
In contrast, our approach is not tailored to a specific use
case but provides competitive performance across both syn-
thetic tasks and real-world scenarios, including in-context
learning and chat-based interactions. Additionally, many of
these approaches are incompatible with FlashAttention Dao
(2024) due to their reliance on accessing the full attention
weights, which FlashAttention does not expose.
8

<!-- page 9 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
7. Conclusion
We introduced Q-Filters, a novel training-free method for
KV Cache compression. We show that projecting the Key
representations on the main SVD component of the Query
vectors results in an accurate approximation of the attention
scores. Q-Filters is extremely efficient and is compatible
with FlashAttention as it does not require accessing the at-
tention scores. We validated our method on several tasks
(Language modelling, NIAH, Ruler) and models up to 70B
parameters, and showed competitive performance with re-
spect to more costly state-of-the-art KV Cache compression
methods.
8. Impact Statement
This paper introduces Q-Filters, a training-free technique
for compressing the Key-Value cache in large language
models by exploiting the geometry of Query and Key vec-
tors. By discarding less important representations through
a single projection direction, Q-filters substantially reduce
memory usage while preserving performance across long
contexts. Crucially, it remains compatible with memory-
efficient attention mechanisms, facilitating practical adop-
tion in real-world scenarios. This advancement addresses
pressing scalability and latency challenges and offers a fresh
perspective on harnessing geometrical insights to develop
more efficient language modelling strategies.
9. Acknowledgements
This collaboration was made possible by my academic visit
to Prof. Edoardo Ponti’s lab at the University of Edinburgh.
I express my sincere gratitude to Prof. Ponti for this oppor-
tunity and for our exciting discussions.
This work was funded by the last author’s chair in the
PRAIRIE institute funded by the French national agency
ANR as part of the “Investissements d’avenir” programme
under the reference ANR-19-P3IA-0001.
This work was granted access to the HPC resources of
IDRIS under the allocation 2024-AD011013680R2 made
by GENCI.
References
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y .,
Lebron, F., and Sanghai, S. GQA: Training generalized
multi-query transformer models from multi-head check-
points. In The 2023 Conference on Empirical Methods
in Natural Language Processing, 2023. URL https:
//openreview.net/forum?id=hmOwOZWzYE.
Anthropic, A. The claude 3 model family: Opus, sonnet,
haiku. Claude-3 Model Card, 2024.
Cai, Z., Zhang, Y ., Gao, B., Liu, Y ., Liu, T., Lu, K.,
Xiong, W., Dong, Y ., Chang, B., Hu, J., and Xiao, W.
Pyramidkv: Dynamic kv cache compression based on
pyramidal information funneling, 2024. URL https:
//arxiv.org/abs/2406.02069.
Dao, T. FlashAttention-2: Faster attention with better paral-
lelism and work partitioning. In International Conference
on Learning Representations (ICLR), 2024.
DeepSeek-AI, Liu, A., Feng, B., Wang, B., Wang, B., Liu,
B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., Yang,
D., Chen, D., Ji, D., Li, E., Lin, F., Luo, F., Hao, G.,
Chen, G., Li, G., Zhang, H., Xu, H., Yang, H., Zhang,
H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L.,
Liang, J., Guo, J., Ni, J., Li, J., Chen, J., Yuan, J., Qiu, J.,
Song, J., Dong, K., Gao, K., Guan, K., Wang, L., Zhang,
L., Xu, L., Xia, L., Zhao, L., Zhang, L., Li, M., Wang,
M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N.,
Huang, P., Wang, P., Zhang, P., Zhu, Q., Chen, Q., Du,
Q., Chen, R. J., Jin, R. L., Ge, R., Pan, R., Xu, R., Chen,
R., Li, S. S., Lu, S., Zhou, S., Chen, S., Wu, S., Ye, S.,
Ma, S., Wang, S., Zhou, S., Yu, S., Zhou, S., Zheng, S.,
Wang, T., Pei, T., Yuan, T., Sun, T., Xiao, W. L., Zeng,
W., An, W., Liu, W., Liang, W., Gao, W., Zhang, W.,
Li, X. Q., Jin, X., Wang, X., Bi, X., Liu, X., Wang, X.,
Shen, X., Chen, X., Chen, X., Nie, X., Sun, X., Wang,
X., Liu, X., Xie, X., Yu, X., Song, X., Zhou, X., Yang,
X., Lu, X., Su, X., Wu, Y ., Li, Y . K., Wei, Y . X., Zhu,
Y . X., Xu, Y ., Huang, Y ., Li, Y ., Zhao, Y ., Sun, Y ., Li, Y .,
Wang, Y ., Zheng, Y ., Zhang, Y ., Xiong, Y ., Zhao, Y ., He,
Y ., Tang, Y ., Piao, Y ., Dong, Y ., Tan, Y ., Liu, Y ., Wang,
Y ., Guo, Y ., Zhu, Y ., Wang, Y ., Zou, Y ., Zha, Y ., Ma, Y .,
Yan, Y ., You, Y ., Liu, Y ., Ren, Z. Z., Ren, Z., Sha, Z.,
Fu, Z., Huang, Z., Zhang, Z., Xie, Z., Hao, Z., Shao, Z.,
Wen, Z., Xu, Z., Zhang, Z., Li, Z., Wang, Z., Gu, Z., Li,
Z., and Xie, Z. Deepseek-v2: A strong, economical, and
efficient mixture-of-experts language model, 2024. URL
https://arxiv.org/abs/2405.04434.
Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P.,
Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos,
R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschan-
nen, M., Arnab, A., Wang, X., Riquelme Ruiz, C., Min-
derer, M., Puigcerver, J., Evci, U., Kumar, M., Steenkiste,
S. V ., Elsayed, G. F., Mahendran, A., Yu, F., Oliver, A.,
Huot, F., Bastings, J., Collier, M., Gritsenko, A. A.,
Birodkar, V ., Vasconcelos, C. N., Tay, Y ., Mensink,
T., Kolesnikov, A., Pavetic, F., Tran, D., Kipf, T., Lu-
cic, M., Zhai, X., Keysers, D., Harmsen, J. J., and
9

<!-- page 10 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Houlsby, N. Scaling vision transformers to 22 billion
parameters. In Krause, A., Brunskill, E., Cho, K., En-
gelhardt, B., Sabato, S., and Scarlett, J. (eds.), Pro-
ceedings of the 40th International Conference on Ma-
chine Learning, volume 202 of Proceedings of Machine
Learning Research, pp. 7480–7512. PMLR, 23–29 Jul
2023. URL https://proceedings.mlr.press/
v202/dehghani23a.html.
Devoto, A., Zhao, Y ., Scardapane, S., and Minervini,
P. A simple and effective l_2 norm-based strat-
egy for KV cache compression. In Al-Onaizan, Y .,
Bansal, M., and Chen, Y .-N. (eds.), Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing , pp. 18476–18499, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
1027. URL https://aclanthology.org/2024.
emnlp-main.1027.
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A.,
Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A.,
Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravanku-
mar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A.,
Rodriguez, A., Gregerson, A., Spataru, A., Rozière, B.,
Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak, C.,
Bi, C., Marra, C., McConnell, C., Keller, C., Touret, C.,
Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Allonsius,
D., Song, D., Pintz, D., Livshits, D., Esiobu, D., Choud-
hary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hup-
kes, D., Lakomkin, E., AlBadawy, E., Lobanova, E., Di-
nan, E., Smith, E. M., Radenovic, F., Zhang, F., Synnaeve,
G., Lee, G., Anderson, G. L., Nail, G., Mialon, G., Pang,
G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Tou-
vron, H., Zarov, I., Ibarra, I. A., Kloumann, I. M., Misra,
I., Evtimov, I., Copet, J., Lee, J., Geffert, J., Vranes, J.,
Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Bil-
lock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J., Liu,
J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca,
J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V ., Upasani,
K., Plawiak, K., Li, K., Heafield, K., Stone, K., and et al.
The llama 3 herd of models. CoRR, abs/2407.21783,
2024. doi: 10.48550/ARXIV .2407.21783. URL https:
//doi.org/10.48550/arXiv.2407.21783.
Fu, Y . Challenges in deploying long-context transformers:
A theoretical peak performance analysis. arXiv preprint
arXiv:2405.08944, 2024.
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T.,
Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N.,
Presser, S., and Leahy, C. The Pile: An 800gb dataset
of diverse text for language modeling. arXiv preprint
arXiv:2101.00027, 2020.
Godey, N., Clergerie, É., and Sagot, B. Anisotropy is inher-
ent to self-attention in transformers. In Graham, Y . and
Purver, M. (eds.), Proceedings of the 18th Conference of
the European Chapter of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pp. 35–48,
St. Julian’s, Malta, March 2024. Association for Compu-
tational Linguistics. URL https://aclanthology.
org/2024.eacl-long.3.
Guo, Z., Kamigaito, H., and Watanabe, T. Attention score
is not all you need for token importance indicator in
KV cache reduction: Value also matters. In Al-Onaizan,
Y ., Bansal, M., and Chen, Y .-N. (eds.), Proceedings
of the 2024 Conference on Empirical Methods in Nat-
ural Language Processing , pp. 21158–21166, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
1178. URL https://aclanthology.org/2024.
emnlp-main.1178/.
Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D.,
Jia, F., Zhang, Y ., and Ginsburg, B. Ruler: What’s the
real context size of your long-context language models?
arXiv preprint arXiv:2404.06654, 2024.
Jegou, S. and Jeblick, M. Kvpress, 2024. URL https:
//github.com/kvpress.
Kovaleva, O., Kulshreshtha, S., Rogers, A., and Rumshisky,
A. BERT busters: Outlier dimensions that disrupt
transformers. In Zong, C., Xia, F., Li, W., and Nav-
igli, R. (eds.), Findings of the Association for Com-
putational Linguistics: ACL-IJCNLP 2021 , pp. 3392–
3405, Online, August 2021. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2021.findings-acl.
300. URL https://aclanthology.org/2021.
findings-acl.300/.
Li, Y ., Huang, Y ., Yang, B., Venkitesh, B., Locatelli, A.,
Ye, H., Cai, T., Lewis, P., and Chen, D. SnapKV:
LLM knows what you are looking for before genera-
tion. In The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024. URL https:
//openreview.net/forum?id=poE54GOq2l.
Nawrot, P., Ła ´ncucki, A., Chochowski, M., Tarjan,
D., and Ponti, E. Dynamic memory compression:
Retrofitting LLMs for accelerated inference. In Forty-
first International Conference on Machine Learning ,
2024. URL https://openreview.net/forum?
id=tDRYrAkOB7.
OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K.,
Arora, S., Bhagia, A., Gu, Y ., Huang, S., Jordan, M.,
Lambert, N., Schwenk, D., Tafjord, O., Anderson, T.,
Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri,
N., Guerquin, M., Ivison, H., Koh, P. W., Liu, J., Malik,
10

<!-- page 11 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
S., Merrill, W., Miranda, L. J. V ., Morrison, J., Murray,
T., Nam, C., Pyatkin, V ., Rangapur, A., Schmitz, M.,
Skjonsberg, S., Wadden, D., Wilhelm, C., Wilson, M.,
Zettlemoyer, L., Farhadi, A., Smith, N. A., and Hajishirzi,
H. 2 olmo 2 furious, 2025. URL https://arxiv.
org/abs/2501.00656.
Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng,
B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H.,
Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J.,
Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L.,
Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R.,
Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y ., Su,
Y ., Zhang, Y ., Wan, Y ., Liu, Y ., Cui, Z., Zhang, Z., and
Qiu, Z. Qwen2.5 technical report, 2025. URL https:
//arxiv.org/abs/2412.15115.
Reid, M., Savinov, N., Teplyashin, D., Lepikhin, D., Lilli-
crap, T., Alayrac, J.-b., Soricut, R., Lazaridou, A., Firat,
O., Schrittwieser, J., et al. Gemini 1.5: Unlocking multi-
modal understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530, 2024.
Shazeer, N. Fast transformer decoding: One write-head
is all you need. 2019. URL https://arxiv.org/
abs/1911.02150.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I.
Attention is all you need. In Guyon, I., Luxburg, U. V .,
Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S.,
and Garnett, R. (eds.), Advances in Neural Information
Processing Systems, volume 30. Curran Associates, Inc.,
2017. URL https://proceedings.neurips.
cc/paper_files/paper/2017/file/
3f5ee243547dee91fbd053c1c4a845aa-Paper.
pdf.
Xiao, G., Tian, Y ., Chen, B., Han, S., and Lewis, M. Ef-
ficient streaming language models with attention sinks.
In The Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.
net/forum?id=NG7sS51zVF.
Zhang, Z., Sheng, Y ., Zhou, T., Chen, T., Zheng, L., Cai,
R., Song, Z., Tian, Y ., Ré, C., Barrett, C., et al. H2o:
Heavy-hitter oracle for efficient generative inference of
large language models. Advances in Neural Information
Processing Systems, 36, 2024.
11

<!-- page 12 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
A. Proof of Theorem 3.3
We begin the proof by writing ⟨Qh
i , Kh
j ⟩ in the basis B:
EQh
i
(⟨Qh
i , k⟩) = EQh
i
(⟨Qh
i , uh⟩)⟨k, uh⟩
+
dhX
m=2
EQh
i
(⟨Qh
i , um⟩)⟨k, um⟩
Observation 3.2 states that Ei,X(⟨Qh
i , um⟩) ≈ 0, which lets
us do the following approximation:
dhX
m=2
EQh
i
(⟨Qh
i , um⟩)⟨k, um⟩ ≈ 0
By combining Observation 3.1 and Observation 3.2, we also
have that:
EQh
i
(⟨Qh
i , uh⟩) > 0
We conclude the proof by setting κh = EQh
i
(⟨Qh
i , uh⟩).
B. Generation Results
We compute the final perplexity of Llama-3.1-70B in the
memory-constrained setup for various compression factors
and methods.
Figure 11: Final perplexity after 512 tokens for Llama-3.1-
70B in the memory-constrained generation scenario.
We also run a study similar to the one conducted in Figure 5
with Qwen-2.5-7B-Instruct, which we display in Figure 12,
and with Llama-3.2-1B, which we display in Figure 13.
C. Ruler Results
In Figure 14 we report detailed evaluation on the subsets of
the Ruler dataset Hsieh et al. (2024).
D. Generation examples
Using Llama-3.1-8B, we identify interesting cases where
Q-Filters provide the correct next token in a given long
Figure 12: Perplexity of the Qwen-2.5-7B-Instruct model
along generation.
Figure 13: Perplexity of the Llama-3.2-1B model along
generation.
context, while K-norm and Streaming-LLM fail to capture
the relevant information.
E. Implementation Details
For all our experiments, we use the popular Huggingface
models with the recently released KVPress library (Jegou &
Jeblick, 2024).
12

<!-- page 13 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
(a) Variable tracking
 (b) NIAH - single (1)
 (c) NIAH - single (2)
(d) NIAH - single (3)
 (e) NIAH - Multi-key (1)
 (f) NIAH - Multi-key (2)
(g) NIAH - Multi-key (3)
 (h) NIAH - Multi-Value
 (i) NIAH - Multi-Query
(j) Frequent Words Extraction (FWE)
 (k) Common Words Extraction (CWE)
Figure 14: Performance of Llama-3.1-8B-Instruct using several KV Cache compression methods on individual tasks from
the Ruler dataset (with length 8192) as compression ratio evolves. We report prompt compression methods using dotted
lines for comparison.
13

<!-- page 14 -->

Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
Text Sample (Context) Q-Filters K-Norm Streaming-LLM
One of the show’s first longest-running storylines was
the rivalry between a young manicurist Jill Foster Ab-
bott (Brenda Dickson, Jess Walton) and wealthy so-
cialite, Katherine Chancellor (Jeanne Cooper). [...]
After much investigation, it is revealed that Kay is Jill’s
biological...
mother father father
Both extreme right-wing leaders taught and practised
the theology ofChristian Identity, a belief system which
the FBI includes on its watch list as an extremist reli-
gion. [...] Here, the group trained an estimated 1,500
of like-minded Christian...
Identity fundamental fundamental
The Viral Fever
[...] TVF debuted their platform, releasing the final
two episodes of Pitchers on TVFPlay. [...] TVF claims
to have worked with over 150 brands. [...] The show
has been on hold as writer Biswapati Sarkar focuses
on writing web series, including the sequel to TV ...
F _show _show
Table 2: Next-token generation examples for different KV Cache Compression methods, applied to Wikipedia article.
Passages in bold correspond to useful information that is necessary to resolve the ambiguity in the choice of the next token.
14
