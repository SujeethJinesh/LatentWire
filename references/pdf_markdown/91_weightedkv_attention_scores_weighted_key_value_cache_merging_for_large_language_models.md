# references/91_weightedkv_attention_scores_weighted_key_value_cache_merging_for_large_language_models.pdf

<!-- page 1 -->

1
WeightedKV: Attention Scores Weighted Key-Value
Cache Merging for Large Language Models
Jian Yuan1† , Ziwei He 1†, Haoli Bai 2, Jingwen Leng 1, Bo Jiang 1∗
1Shanghai Jiao Tong University 2Huawei Noah’s Ark Lab
{yuanjian, ziwei.he, leng-jw, bjiang }@sjtu.edu.cn
Abstract—Large Language Models (LLMs) use key-value (KV)
cache to reduce redundant computation in autoregressive gen-
eration. However, the KV cache size increases linearly during
generation, leading to excessive memory usage, especially for long
texts. Most KV cache compression methods evict the unimportant
KV pairs to maintain a fixed cache size, which leads to the
permanent loss of tokens during generation. However, singular
value decomposition shows that values do not exhibit a strong
low-rank property as keys do, suggesting that information is
distributed more evenly across values, in contrast to its more
redundant distribution within keys. Therefore, methods that
evict both keys and values risk losing crucial information and
compromise context integrity, ultimately degrading the output
quality. To address this problem, we propose WeightedKV , a
novel, training-free approach that discards the keys of less impor-
tant tokens, while merging their values into neighboring tokens
via a convex combination weighted by their average attention
scores. In this way, the retained keys serve as anchors that guide
the generation process, while the merged values provide a rich
contextual backdrop. We assess our method on four widely used
language modeling datasets, demonstrating superior performance
compared to all baseline methods, particularly with a lower
budget ratio.
Index Terms—Large Language Models, KV Cache Compres-
sion, Efficient Inference
I. I NTRODUCTION
Recently, the decoder-only transformer has emerged as the
leading architecture for large language models (LLMs) [1]–
[3], showcasing exceptional effectiveness across various appli-
cation areas [4]–[6]. Nevertherless, inference is costly due to
the autoregressive nature of LLMs, which generate new tokens
by repeatedly using key-value (KV) pairs of previous ones.
To minimize the redundant calculations, these generated KV
pairs are stored in a cache, known as KV cache [7], for later
use, allowing the models to trade memory for computational
power. However, as sequence length increases, the KV cache
presents significant scalability challenges, leading to memory
and latency bottlenecks for LLMs.
Several KV cache eviction methods have been proposed to
optimize KV cache usage by maintaining a fixed size during
generation. One approach involves static cache management,
which evicts KV pairs of tokens outside predefined scopes,
as seen in StreamingLLM [8] and LM-Infinite [9]. Another
approach involves dynamic management, where less important
KV pairs are removed based on their impact on model perfor-
mance [10]–[12]. While these methods ensure the KV cache
† Equal contribution.
∗ Bo Jiang is the corresponding author.
Fig. 1. Normalized singular values of KV averaged over the first 10 sequences
truncated to length 1k in the PG19 test set.
remains within capacity, they inevitably lead to the permanent
loss of entire KV pairs during generation. These eviction
strategies could discard crucial KV pairs unintentionally, strug-
gling to maintain context and diminish model’s performance
in long sequence generation. To further investigate the effects
of permanent eviction of both keys and values, we perform
singular value decomposition on the hidden states of KV pairs
across different layers and heads from Llama-2-7B [1]. Fig.1
shows the normalized singular values averaged over the first 10
sequences from PG19 [13]. Note that the normalized singular
values for keys rapidly approach zero, whereas those for values
exhibit a much heavier tail. This suggests that information
within keys is more redundant, while it is distributed more
evenly across values. Therefore, while it may be reasonable
to discard the keys of evicted tokens, simply discarding their
values may lead to information loss and degrade the quality
of the generated outputs.
To address these issues, we propose WeightedKV , a novel,
training-free approach that discards the keys of less important
tokens, while merging their values into neighboring tokens
via a convex combination weighted by their average attention
scores. In contrast to eviction-based methods, WeightedKV
does not simply evict KV pairs, but instead focuses on a more
nuanced management of the cache. By maintaining a subset
of keys corresponding to important tokens, we ensure that
critical information remains accessible without overloading
device memory. The convex merging of the values allows
us to create a more compact representation that captures the
diverse contributions of non-retained tokens, ensuring that the
overall context is preserved even in their absence. In this way,
the retained keys serve as anchors that guide the generation
Copyright © 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including
reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any
copyrighted component of this work in other works.
arXiv:2503.01330v1  [cs.CL]  3 Mar 2025

<!-- page 2 -->

process, while the merged values provide a rich contextual
backdrop. It is important to note that the convex combination
of values is derived from a theoretical analysis of how evicting
keys and merging values can approximate an ideal merging that
minimizes perturbations in the attention output. Additionally,
we show that the proposed way of merging values in a
single step has only a minimal impact on the attention score
distribution in future steps in Section IV.
Our experimental setup evaluates the performance of
WeightedKV against some conventional eviction methods [8],
[10], [11] and another value merging method CAM [14] on
four language modeling datasets. WeightedKV demonstrates
the state-of-the-art performance in perplexity for long text
generation compared to all baselines. By optimizing the KV
cache with WeightedKV , we can not only reduce memory con-
sumption but also improve the quality of generated sequences.
Our contributions can be summarized as follows:
• We propose WeightedKV , an efficient, training-free ap-
proach that compresses the KV cache by preserving
the keys of important tokens, and merging the values
of discarded tokens into retained ones, via a convex
combination weighted by their average attention scores.
• We show that WeightedKV is grounded on a theo-
retical derivation of how evicting tokens and merging
values can approximate an ideal merging that minimizes
perturbations in attention output. We also demonstrate
that our value merging incurs negligible perturbation to
subsequent inference compared to full attention.
• We present thorough experimental results confirming the
effectiveness of WeightedKV in the generation stage. It
achieves the lowest perplexity on PG19, OpenWebText2,
ProofPile, and ArXiv, compared to all baselines.
II. R ELATED WORK
There has been an increase in the number of works focusing
on compressing KV cache during generation in the field of
natural language processing, making it a critical area for
enhancing the performance of LLMs with longer contexts.
These methods can be categorized into eviction-based and
merging-based approaches.
a) Eviction-based methods: To limit the size of the KV
cache, LM-Infinite [9] and StreamingLLM [8] utilize static
cache management. They retain predefined initial and recent
regions of a sequence, which typically receive high attention
scores, while evicting KV pairs of tokens outside these regions.
Other methods use more dynamic strategies that remove the
least important KV pairs according to various importance
measures. TOV A [11] identifies the token with the lowest
attention score from the previous step as the least important
and prioritizes it for eviction. Recognizing that some tokens
with high attention scores are crucial for model performance,
H2O [10] determines token importance based on cumulative
attention scores. Similarly, Scissorhands [12] binarizes these
cumulative scores to establish token importance.
b) Merging-based methods: KVMerger [15] is a method
that merges both the key and value states. Motivated by
the observation that key states show considerable token-level
similarity, KVMerger first clusters the key states into several
groups using cosine similarity. Within each group, the key
and value states are then merged according to their cosine
similarity and attention scores, respectively. CaM [14] shares
some similarity with WeightedKV , as it also discards keys
and merges values. However, CaM merges the values of
evicted tokens only probabilistically, with a non-negligible
probability of discarding them and hence losing information.
When a value state is to be merged, it is scaled by a constant
factor of 1/n and added to the value states of the next n
tokens. In contrast, WeightedKV uses a deterministic merging
process that accounts for the importance of different tokens
as measured by their contributions to model performance.
There are also some other merging methods such as Dynamic
Memory Compression (DMC) [16] and Anchor-LLM [17].
Those methods require additional parameter optimization and
are not training-free.
III. P RELIMINARY
A. Background
First, we illustrate the auto-regressive decoding process
during generation. We use Wq, Wk, Wv ∈ Rd×d to denote
the attention modules’ weights in a layer. Here d is the
hidden dimension and we omit the dimensions of batches and
heads for simplicity. Initial KV cache K0, V0 is set as empty.
Suppose at generation step t a new input xt ∈ Rd enters, the
attention gets query, key, and value by
qt = WQxt, kt = WKxt, vt = WV xt.
Then the KV cache at the current step is updated by concate-
nating the new value and key to the cache
Kt = [Kt−1, kt] , Vt = [Vt−1, vt] .
Using the updated KV cache, we can compute the attention
weights At and the output of the attention module as
ot = A⊤
t Vt, where At = softmax
 
q⊤
t Kt

.
As the generation goes on, Kt, Vt grows linearly with t. This
allows the model to reference all previous tokens, but it comes
with significant computing and storage costs as the context
grows longer.
B. Ideal Merging
We propose evicting keys and merging values based on
the previous singular value analysis, demonstrating that this
approach can achieve evicting keys without affecting output
in ideal scenarios, followed by an exploration of practical
approximations. To preserve the order of tokens, we focus on
merging adjacent values.
There is an ideal way to substitute two adjacent values
v1, v2 with ˜ vin Vt = [ v1, v2 · · · , vt] without changing
the output when we want to compress k1, k2 in key cache

<!-- page 3 -->

Fig. 2. Cosine similarity between attention weights with merging values and
without merging values at step 100 on the books from PG19.
Kt = [ k1, k2, · · · , kt] by keeping k2 and discarding k1. For
easy of presentation, we assume the initial two positions are
to be compressed. To draw the formulation of ˜ v, we consider
the equation of outputs before and after the compression
softmax

q⊤
t [k1, k2, k3, · · · , kt]/
√
d
⊤
[v1, v2, v3, · · · , vt]
= softmax

q⊤
t [k2, k3, · · · , kt]/
√
d
⊤
[˜ v, v3, · · · , vt].
The equation gives
˜ v=

1 − eq⊤
t k1/
√
d
Pt
i=1 eq⊤
t ki/
√
d
!
eq⊤
t (k1−k2)/
√
dv1 + v2

− eq⊤
t (k1−k2)/
√
d
Pt
i=1 eq⊤
t ki/
√
d
tX
i=3
eq⊤
t ki/
√
dvi.
We approximate the formula due to its O(td) time com-
plexity, omitting the terms with v3, . . . ,vt and normalizing to
prevent information decay. This yields the approximation
˜ v≈ eq⊤
t k1/
√
d
eq⊤
t k1/
√
d + eq⊤
t k2/
√
d v1 + eq⊤
t k2/
√
d
eq⊤
t k1/
√
d + eq⊤
t k2/
√
d v2,
which is a convex combination of v1, v2. To minimize the
approximation error, the attention weights of the evicted
token might be kept as low as possible. Motivated by this
approximation, we propose WeightedKV to compress the KV
cache by consolidating values and evicting keys.
IV. M ETHODOLOGY
This section demonstrates the use of WeightedKV for
compressing the KV cache. We first explain the selection and
merging of values for a single step, followed by a detailed
illustration of the complete algorithm.
In a single-step merge, we use average attention scores
to weigh and consolidate values. Following the analysis in
section III-B, we first merge the value with the least weight
into its neighboring value. To investigate the perturbation
caused by merging, we merge the token with the lowest
average attention with its adjacent right token for PG-19 test
set samples at generation step 100, then calculate the attention
scores for the next 800 steps. After that, we compute the
attention scores on full attention and determine the cosine
1.0
0.8 0.5
0.5 0.3 0.9
0.4 0.2 0.7 0.6
0.3 0.50.30.1 0.5
0.3 0.4 0.2 0.3 0.9
0.3 0.4 0.2 0.5 0.7
0.2 0.4 0.3 0.4 0.6
V1 V2 V8
Step 1
Step 8
Step 2
V3 V4 V5 V6 V7
Step 3
Step 4
Step 5
Step 6
Step 7
Fig. 3. Compression process on a toy attention map with a maximum cache
size of 4. Numbers in blocks represent average attention scores of tokens,
while the red boxes indicate the values to be merged.
Algorithm 1 WeightedKV
Input: x1, · · · , xt, m
1: a, n are empty lists.
2: for i from 1 to t do
3: Get kt, qt, vt, Kt, Vt, At from xt.
4: a = [a, 0] + At.
5: n = [n, 0] + 1.
6: if Length of Vt exceeds m then
7: a = a/n.
8: j = arg minj∈{0,··· ,m−1} a[j]
9: Replace Vt[j : j + 2] with a[j]Vt[j]+a[j+1]Vt[j+1]
a[j]+a[j+1] .
10: Remove Kt[j], a[j], n[j].
11: end if
12: end for
similarity to the previous scores. Fig. 2 shows the cosine
similarity between the attention scores over the next 800
steps, with standard variance represented as error bars. The
average cosine similarity is extremely close to 1, indicating
that merging values by such a convex combination has a
minimal impact on the attention score of other tokens in the
future. This experimental validation shows that our merging
way has a negligible effect on subsequent inference steps.
The complete compression procedure of WeightedKV is
outlined in Algorithm 1. Our algorithm dynamically updates
the accumulated attention weights and the calculation times for
each key-value pair in lists a, n respectively during generation.
When the cache reaches its maximum capacity m, historical
average attention scores are calculated as a/n, which are
weights for merging values and evicting keys. The value
corresponding to the smallest weight is selected to merge into
its right adjacent value. The algorithm will merge two values
in the value cache by convexly combining them weighted on
historical average attention scores and remove the key and
item corresponding smaller weights from both the key cache
and the storage list. Fig. 3 depicts a compression process
by our algorithm on a toy average attention map with a
maximum cache size of 4. Numbers in blocks indicate the

<!-- page 4 -->

TABLE I
PERPLEXITY ON PG19, O PEN WEBTEXT 2, P ROOF PILE AND ARXIV
PG19 OpenWebText2 ProofPile ArXiv
Context 4k 8k 16k 4k 8k 16k 4k 8k 16k 4k 8k 16k
FullKV 6.84 - - 5.44 - - 2.51 - - 3.04 - -
Cache Size = 1024
StreamingLLM 7.19 7.18 7.20 5.78 5.87 5.33 2.84 2.84 2.85 3.48 3.45 3.49
TOV A 7.00 7.05 7.14 5.62 5.73 5.24 2.64 2.67 2.72 3.22 3.24 3.30
H2O 7.06 7.06 7.24 5.60 5.71 5.24 2.63 2.66 2.76 3.20 3.19 3.34
CaM 7.19 7.18 7.19 5.85 5.89 5.46 2.83 2.84 2.85 3.48 3.48 3.49
WeightedKV 6.98 6.99 7.10 5.60 5.68 5.21 2.63 2.64 2.72 3.20 3.17 3.30
Cache Size = 256
StreamingLLM 7.99 8.00 7.99 6.67 6.66 6.05 3.61 3.61 3.59 4.51 4.47 4.52
TOV A 7.70 7.83 7.97 6.36 6.45 6.11 3.30 3.38 3.48 4.12 4.29 4.69
H2O 7.85 8.14 8.44 6.50 6.64 6.23 3.36 3.60 3.83 4.13 4.38 4.77
CaM 7.98 7.97 7.99 6.70 6.78 6.07 3.62 3.63 3.61 4.53 4.53 4.54
WeightedKV 7.49 7.61 7.87 6.27 6.37 5.97 3.13 3.23 3.39 3.86 3.98 4.30
Fig. 4. Comparison between WeightedKV and its eviction variant.
average attention scores of tokens. At generation step 5, the
cache exceeds its capacity, triggering compression. Since v2
has the lowest average attention score of 0.1, the algorithm will
replace v2, v3 by v2/6 + 5v3/6 given the v3’s score 0.5 and
remove k2 at the same time. By repeating this procedure, after
the compression at step 8, the cache maintains v3, v6, v7, v8
where v3, v6 contains the information of 3 single values
respectively.
V. E XPERIMENTS
We assess WeightedKV’s performance on four datasets that
are commonly used for evaluating long text generation. Then
we verify the efficacy of merging values by an ablation study.
A. Long Context Language Modeling
We compare WeightedKV with five baseline methods. Three
of them are efficient eviction-based methods: StreamingLLM
[8], H2O [10], and TOV A [11]. The remaining two are the ef-
ficient merging-based method CaM [14] and the full attention
method. We report sliding window perplexity with window
sizes ranging from 4k to 16k and strides of half of the windows
on four datasets using Llama-2-7B [1]. We assess perplexity on
the complete test set of PG19 [13], which contains 100 books
with an average length of 70k. We randomly select 10 samples
from the ProofPile [18] test set and ArXiv [19] train set,
averaging 128k tokens for evaluation. For OpenWebText2 [19],
we randomly select 100 samples averaging 18k tokens from its
test set. We retain 4 initial tokens in cache and consider cache
sizes of 1024 and 256, keeping 508 and 124 recent tokens for
each size.
The results in Table I indicate that, with a cache size of
1024, WeightedKV outperforms all other efficient methods
on the PG19 dataset across all context lengths. On the other
three datasets, our approach is tied for best with H2O or
TOV A. WeightedKV outperforms all other methods across all
datasets and context lengths with a smaller cache size 256.
These results demonstrate that WeightedKV is an effective
method for KV cache compression, particularly evident at
smaller cache sizes.
B. Ablation Study
To validate the pivotal role of merging, we consider a pure
evicting method that compresses the KV cache in the same
manner as WeightedKV in all other aspects, but instead of
merging, it evicts values. This modification gives a targeted
analysis of the impact of token merging on model perfor-
mance. In Fig. 4, we label our method as “WeightedKV”
and the modified method as “Eviction”, showing their log
average perplexity on the first sample from the PG19 test set
truncated at 32k. The comparison between WeightedKV , which
merges values and evicts keys (blue line), and the pure evicting
method (orange line) demonstrates that our merging approach
significantly improves perplexity, confirming that merging is
essential to our method.
VI. C ONCLUSION
Our findings underline the importance of rethinking con-
ventional cache eviction strategies. The WeightedKV method
that discards unimportant token keys, while convexly merging
their values into neighboring tokens, weighted by their aver-
age attention scores. Evaluations of our approach on various
datasets demonstrate its superiority over existing methods.

<!-- page 5 -->

REFERENCES
[1] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y . Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al. , “Llama
2: Open foundation and fine-tuned chat models,” arXiv preprint
arXiv:2307.09288, 2023.
[2] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier et al.,
“Mistral 7b,” arXiv preprint arXiv:2310.06825 , 2023.
[3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al. , “Gpt-4
technical report,” arXiv preprint arXiv:2303.08774 , 2023.
[4] T. Zhang, F. Ladhak, E. Durmus, P. Liang, K. McKeown, and T. B.
Hashimoto, “Benchmarking large language models for news summa-
rization,” Transactions of the Association for Computational Linguistics,
vol. 12, pp. 39–57, 2024.
[5] E. Kamalloo, N. Dziri, C. L. Clarke, and D. Rafiei, “Evaluating open-
domain question answering in the era of large language models,” arXiv
preprint arXiv:2305.06984, 2023.
[6] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y . Adi,
J. Liu, T. Remez, J. Rapin et al., “Code llama: Open foundation models
for code,” arXiv preprint arXiv:2308.12950 , 2023.
[7] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Lev-
skaya, J. Heek, K. Xiao, S. Agrawal, and J. Dean, “Efficiently scaling
transformer inference. corr, abs/2211.05102 (2022),” 2022.
[8] G. Xiao, Y . Tian, B. Chen, S. Han, and M. Lewis, “Efficient streaming
language models with attention sinks,”arXiv preprint arXiv:2309.17453,
2023.
[9] C. Han, Q. Wang, H. Peng, W. Xiong, Y . Chen, H. Ji, and S. Wang,
“Lm-infinite: Zero-shot extreme length generalization for large language
models,” in Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), 2024, pp. 3991–4008.
[10] Z. Zhang, Y . Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y . Tian, C. R´e, C. Barrett et al., “H2o: Heavy-hitter oracle for efficient
generative inference of large language models,” Advances in Neural
Information Processing Systems , vol. 36, 2024.
[11] M. Oren, M. Hassid, Y . Adi, and R. Schwartz, “Transformers are multi-
state rnns,” arXiv preprint arXiv:2401.06104 , 2024.
[12] Z. Liu, A. Desai, F. Liao, W. Wang, V . Xie, Z. Xu, A. Kyrillidis, and
A. Shrivastava, “Scissorhands: Exploiting the persistence of importance
hypothesis for llm kv cache compression at test time,” Advances in
Neural Information Processing Systems , vol. 36, 2024.
[13] J. W. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap, “Com-
pressive transformers for long-range sequence modelling,”arXiv preprint
arXiv:1911.05507, 2019.
[14] Y . Zhang, Y . Du, G. Luo, Y . Zhong, Z. Zhang, S. Liu, and R. Ji,
“Cam: Cache merging for memory-efficient llms inference,” in Forty-
first International Conference on Machine Learning , 2024.
[15] Z. Wang, B. Jin, Z. Yu, and M. Zhang, “Model tells you where to
merge: Adaptive kv cache merging for llms on long-context tasks,”arXiv
preprint arXiv:2407.08454, 2024.
[16] P. Nawrot, A. Ła ´ncucki, M. Chochowski, D. Tarjan, and E. Ponti, “Dy-
namic memory compression: Retrofitting llms for accelerated inference,”
in Forty-first International Conference on Machine Learning , 2024.
[17] J. Pang, F. Ye, D. F. Wong, and L. Wang, “Anchor-based large language
models,” arXiv preprint arXiv:2402.07616 , 2024.
[18] Z. Azerbayev, E. Ayers, and B. Piotrowski, “Proof-pile,” 2022. [Online].
Available: https://github.com/zhangir-azerbayev/proof-pile
[19] L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster,
J. Phang, H. He, A. Thite, N. Nabeshima et al. , “The pile: An
800gb dataset of diverse text for language modeling,” arXiv preprint
arXiv:2101.00027, 2020.
