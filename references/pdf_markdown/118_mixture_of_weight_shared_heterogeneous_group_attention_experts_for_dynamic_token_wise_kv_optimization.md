# references/118_mixture_of_weight_shared_heterogeneous_group_attention_experts_for_dynamic_token_wise_kv_optimization.pdf

<!-- page 1 -->

Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22890–22903
November 4-9, 2025 ©2025 Association for Computational Linguistics
Mixture of Weight-shared Heterogeneous Group Attention Experts for
Dynamic Token-wise KV Optimization
Guanghui Song1 Dongping Liao2 Yiren Zhao3
Kejiang Ye1,4 Cheng-zhong Xu2 Xitong Gao1,4*
1 Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
2 State Key Lab of IOTSC, University of Macau
3 Imperial College London 4 Shenzhen University of Advanced Technology
Abstract
Transformer models face scalability challenges
in causal language modeling (CLM) due to in-
efficient memory allocation for growing key-
value (KV) caches, which strains compute
and storage resources. Existing methods like
Grouped Query Attention (GQA) and token-
level KV optimization improve efficiency but
rely on rigid resource allocation, often discard-
ing “low-priority” tokens or statically grouping
them, failing to address the dynamic spectrum
of token importance. We proposemixSGA, a
novel mixture-of-expert (MoE) approach that
dynamically optimizes token-wise computa-
tion and memory allocation. Unlike prior ap-
proaches,mixSGAretains all tokens while adap-
tively routing them to specialized experts with
varying KV group sizes, balancing granularity
and efficiency. Our key novelties include:(1)
a token-wise expert-choice routing mechanism
guided by learned importance scores, enabling
proportional resource allocation without token
discard;(2)weight-sharing across grouped at-
tention projections to minimize parameter over-
head; and(3)an auxiliary loss to ensure one-hot
routing decisions for training-inference consis-
tency in CLMs. Extensive evaluations across
Llama3, TinyLlama, OPT, and Gemma2 model
families showmixSGA’s superiority over static
baselines. On instruction-following and contin-
ued pretraining tasks,mixSGAachieves higher
ROUGE-L and lower perplexity under the same
KV budgets.
1 Introduction
Transformer architectures have emerged as the
backbone of modern deep learning, powering state-
of-the-art advancements across diverse fields such
as natural language processing (Vaswani et al.,
2017), computer vision (Alexey, 2020), reinforce-
ment learning (Parisotto and Salakhutdinov, 2021)
and beyond (Le et al., 2020; Chen et al., 2023).
*Corresponding author,xt.gao@siat.ac.cn.
Token Sequence
7.8
7.9
8.0
8.1
8.2
8.3
Sequence Ppl.
0
2
4
6
∆ Seq. Ppl. (%)
(a) Change for each token.
0.000 0.025 0.050 0.075
∆ Sequence Perplexity (%)
0
10
20
30Frequency (b) Change distribution.
Figure 1:Token importance is dynamic and has a
wide spectrum.We replace each token’s forward pass
with multi-query attention on Llama3.1-8b by averaging
key and value states across heads, and see the sequence
perplexity changes on a sample sequence of WikiText-2.
However, their self-attention mechanism, while ef-
fective, suffers from quadratic computational and
memory costs with respect to the sequence length,
posing scalability challenges (Tay et al., 2022).
Efforts towards addressing these challenges have
largely focused on improving the efficiency of at-
tention mechanisms. One line of methods seek to
improve the design of the attention block. Grouped
Query Attention (GQA) (Ainslie et al., 2023), for
example, reduces computational overhead by clus-
tering keys and values into coarse groups, which
reduces the number of processed KV pairs. Nev-
ertheless, GQA assumes static group sizes and
allocates resources uniformly, disregarding vari-
ations in token importance. Some works have
been devoted to optimize the memory footprint of
the widely adopted KV cache (Waddington et al.,
2013). Token-level approaches, such as Dynam-
icKV (Zhou et al., 2024), introduce flexible KV
cache allocation by prioritizing high-value tokens.
However, these methods often involve rigid re-
source allocation strategies that neglect to fully
exploit the significance of low-priority tokens.
Another promising line of work (Shazeer et al.,
2017; Lepikhin et al., 2020) adopts MoEs to dy-
namically route tokens to a subset of experts, en-
abling efficient resource utilization. While these ap-
proaches achieve computational efficiency, they fre-
quently suffer from imbalanced expert utilization.
22890

<!-- page 2 -->

Moreover, their coarse-grained routing overlooks
token-level variability, highlighting the need for
finer-grained adaptivity in token-level resource al-
location. In many cases, tokens deemed less impor-
tant are outright discarded or receive minimal pro-
cessing, which can lead to degraded performance
for certain tasks.
Our work is motivated by the experimental find-
ings presented in Figure 1, which reveal that token
importance exhibits dynamic behavior and spans
a wide spectrum. This observation naturally in-
spires the high-level idea of tailoring experts’ to-
ken selection based on their importance. While
this approach holds significant promise for effi-
ciently leveraging the potential of token prioritiza-
tion, it faces several critical challenges that must
be addressed. First, current token-choice routing
(TCR) approaches can result in unbalanced expert
utilization, particularly challenging to tune for het-
erogeneous capacities of experts, as tokens may
always prefer high capacity experts, posing the risk
of collapsed routing mechanisms. Second, exist-
ing expert-choice routing (ECR) methods shifts
imbalanced expert utilization to token utilization,
where tokens may be assigned to multiple experts,
while some tokens are ignored. Third, existing
ECRs also introducetraining and inference dispar-
itiesin CLMs, where during the training or prefill
phase, routings are made based on the complete
sequence, whereas the decoding-phase routings are
made based on past context.
To surmount these obstacles, we introduce
mixSGA. Unlike prior work that discards less sig-
nificant tokens, our method retains all tokens while
dynamically allocating computation and memory
resources proportionally to their importance. For
the experts, we propose a weight-sharing mecha-
nism across grouped attentions, allowing the model
to remain lightweight while dynamically scaling
based on token significance. To overcome the rout-
ing disparity between prefill and decode stages, we
propose a layer-wise auxiliary loss that encourages
routing consistency.
Our contributions are summarized as follows:
• ThemixSGAFramework:mixSGAinte-
grates dynamic token-wise routing with KV
attention head grouping, enabling adaptive
computational/memory allocation without dis-
carding tokens. It also uses weight-sharing for
parameter efficiency.
• Autoregressive Expert-Choice Routing: We
propose a novel past-context routing mecha-
nism with an auxiliary loss to ensure prefill-
decode consistency in CLMs. It also enables
flexible tuning of individual expert capacities.
• Broad Empirical Validation: We demon-
strates superior efficiency and performance
over static and dynamic baselines across
OPT, Llama3 and Gemma2 models on diverse
instruction-following and continued pretrain-
ing benchmarks.
2 Related Work
2.1 KV Cache Management
KV cache optimization enhances the memory effi-
ciency of CLMs (Waddington et al., 2013). Recent
methods such as PyramidKV (Cai et al., 2024), Dy-
namicKV (Zhou et al., 2024), H2O (Zhang et al.,
2024b) and NACL (Chen et al., 2024a) aim to re-
duce memory footprint, typically by prioritizing
high-utility tokens based on heuristics, random, or
learned importance, and evict those deemed less
critical to stay within a constrained memory budget.
Furthermore, methods like SnapKV (Li et al., 2024)
and FastGen (Ge et al., 2024) focus on pattern- and
importance-based token selection to optimize KV
cache efficiency during inference.
Despite these improvements, such methods often
rely on predefined grouping mechanisms, which
may not fully capture token-level variability. As a
result, they can overlook the nuanced importance
of individual tokens, limiting their ability to opti-
mize resource utilization effectively. In addition,
inevitably remove tokens from the attention con-
text, which may lead to degraded performance on
fine-grained contextual understanding. By contrast,
mixSGAadaptively allocates smaller KV cache
sizes to less critical tokens without evicting them,
striking a balance between efficiency and preserv-
ing full contextual integrity. It also preserves all
tokens, and instead of hard eviction, adaptively allo-
cates memory and compute resources in proportion
to token importance. This design ensures that even
less critical tokens retain their contextual influence,
offering a more flexible and context-preserving al-
ternative to hard cache eviction.
2.2 MoE Routing Strategies and Challenges
MoEs provide a scalable approach to increase
model capacity without proportionally increasing
computational costs (Shazeer et al., 2017; Lepikhin
et al., 2020). In token-choice routing (TCR) MoE
22891

<!-- page 3 -->

models such as GShard (Lepikhin et al., 2020) and
Switch Transformer (Fedus et al., 2022), each to-
ken independently select an expert or a subset of
experts for resource-efficient computation. As TCR
allows each token to independently select an expert,
it may suffer from imbalanced expert utilization, in-
efficient resource allocation, and potentially unsta-
ble training dynamics. This is especially problem-
atic when experts have heterogeneous capacities,
as tokens may favor experts with higher capaci-
ties, making it challenging to balance expert loads.
Expert-choice routing (ECR) (Zhou et al., 2022) en-
ables experts to select tokens for processing while
explicitly defining their capacities, improving load
balancing and resource utilization. Despite its ad-
vantages over TCR, ECR presents two significant
challenges in the context of CLMs:(1)it requires
access to the entire input sequence to make routing
decisions, which is incompatible with CLMs that
rely solely on past tokens to predict the next to-
ken; and(2)it shifts the issue ofimbalancedexpert
utilization toimbalancedtoken utilization, where
some tokens may remain unprocessed by any ex-
pert, while others may be redundantly processed
by multiple experts.
Our work differentiates itself from existing TCR
and ECR methods by introducing a routing mecha-
nism specifically designed for CLMs. This mecha-
nism evaluates token significance based on partial
sequence context, enables dynamic expert selec-
tion, and ensures prefill-decode routing consistency
in decoder-only architectures. Additionally, our
method accommodates experts with heterogeneous
capacity, delivering fine-grained resource alloca-
tion and improved efficiency on computation and
memory costs.
2.3 Grouped Attention Methods
Grouped Query Attention (GQA) (Ainslie et al.,
2023) reduces the computational and memory costs
by merging keys and values into larger groups, re-
ducing the number of KV pairs processed during
attention. This can lead to inefficiencies when to-
ken importance varies significantly, as structured
merging fails to prioritize tokens critical to the task.
Decoupled-Head Attention (DHA) (Chen et al.,
2024b) adaptively merges attention heads across
layers, while Align Attention (Jin et al., 2024b)
usesℓ0 regularization to convert multi-head atten-
tion into group-based formulations. Mixture-of-
Head Attention (MoH) (Jin et al., 2024a) reformu-
lates attention heads as experts in a MoE, employ-
ing TCR for sparse head activation. Cross-layer
Attention (CLA) (Brandon et al., 2024) merges key
and value projectors across adjacent layers, repre-
senting one approach to cross-structure KV sharing
that reduces parameter count and memory usage.
(Wu et al., 2025) further generalizes CLA to pro-
vide a systematic study of static cross-layer KV
sharing techniques, highlighting their efficiency
benefit. ReAttention (Liu et al., 2025) offers a
training-free approach to token selection based on
attention scores as importance proxies, enabling
infinite context with finite attention scope through
dynamic token prioritization.
While these approaches learn to enhance struc-
tural efficiency, they rely on static group sizes for
attention heads during inference, assuming uniform
token importance and lacking fine-grained, token-
level adaptability. Additionally, these methods do
not support heterogeneous expert configurations
with varying group sizes, limiting their adaptabil-
ity.
In contrast,mixSGAintegrates the strengths of
grouped attention and token-level adaptivity by dy-
namically routing each token to weight-shared ex-
perts with heterogeneous KV configurations, based
on learned token importance. Unlike prior meth-
ods,mixSGAretains all tokens, ensuring no loss of
contextual information, while adaptively allocating
computational and memory resources at both group
and token levels.
3 ThemixSGAMethod
mixSGA, mixture of weight- shared grouped
attention experts, combines dynamic token-wise
expert assignment with token-level KV optimiza-
tion to achieve efficient attention computation and
minimize KV memory. This section elaborates on
the key components, including the routing mech-
anism for expert selection, the mixture of weight-
shared KV grouping experts, and the auxiliary loss
designed to improve prefill/decode consistency.
3.1 Prefill and Training Phase Routing
Token-to-expert mapping score functionGiven
an input sequenceX∈RL×D, whereL is the se-
quence length andDis the embedding dimension,
we define the following token-to-expert mapping
scoring function for all tokens, a trainable linear
layer S:R L×D→RL×Ewith weight ϕ∈RD×E
and bias β∈RE, where E is the number of ex-
22892

<!-- page 4 -->

Expert1cachecachecachecachecachecache
Expert2
 Expert3
KeysValuesQueries
Shared Weights
←Tokens→
←→Expertsone-hot
<latexit sha1_base64="zeEMzy0dvK0QKsCGwcKrde9/yLI=">AAACAXicbVBNS8NAEN34WetX1IvgJVgETyURqR6LXjx4qGA/oAlhs922SzebsDuRlhAv/hUvHhTx6r/w5r9x0+agrQ8GHu/NMDMviDlTYNvfxtLyyuraemmjvLm1vbNr7u23VJRIQpsk4pHsBFhRzgRtAgNOO7GkOAw4bQej69xvP1CpWCTuYRJTL8QDwfqMYNCSbx66IYYhwTy9zXwX6BhkmOJknPlmxa7aU1iLxClIBRVo+OaX24tIElIBhGOluo4dg5diCYxwmpXdRNEYkxEe0K6mAodUeen0g8w60UrP6kdSlwBrqv6eSHGo1CQMdGd+r5r3cvE/r5tA/9JLmYgToILMFvUTbkFk5XFYPSYpAT7RBBPJ9K0WGWKJCejQyjoEZ/7lRdI6qzq1au3uvFK/KuIooSN0jE6Rgy5QHd2gBmoigh7RM3pFb8aT8WK8Gx+z1iWjmDlAf2B8/gDIKZfF</latexit>
Laux
TokenSequenceT1T2T3T4T5T6
T4
T4T1T3 T1T3T2T5T6 T2T5T6
routingmask
argmaxRouter
Training Path
T7
Prefill TokenDecode TokenInference PathShared Path
auxiliary loss
Decode: Token-wise Arg-Max Routing(Context Independent)
T7T1 -T6
Prefill: Progressive Routing(Context dependent)
T7
Figure 2: High-level overview ofmixSGA. During train-
ing, the router learns to compute assignment scores for
each token-expert pair. These scores are utilized to se-
quentially routeρe tokens to theeth expert, ordered by
their computational and memory costs, while leaving
the remaining tokens to be routed to other experts. The
experts consist of a set of key and value projections that
generate state representations at varying levels of granu-
larity. This process ensures a unique routing assignment
for each token, which is subsequently used to encourage
the router to produce one-hot decisions through an aux-
iliary loss. During decoding, each token independently
selects its corresponding expert througharg max.
perts:
S(x) =σ(xϕ+β),(1)
andσ(·)is the sigmoid function. The sigmoid acti-
vation ensures bounded scores within [0,1] , avoid-
ing additional normalization during training.
MoEs with Heterogeneous CapacitiesTo fa-
cilitate downstream KV cache optimization, our
method employs a routing mechanism that dynami-
cally assigns tokens to experts based on predefined
capacity ratios. These ratios regulate token dis-
tribution among experts, aligning with memory
and computational constraints. Assume that we
haveE experts, where with predefined capacity
ratios for each expert ρ={ρ1,ρ2,...,ρE}, rep-
resenting the fraction of tokens it processes. The
capacity ratios lie in the range [0,1] , and are nor-
malized such that the sum of all ratios is 1,i.e.,∑E
e=1ρe = 1. During training, our token-to-expert
routing thus takes the scoring function output S(x)
and greedily assigns tokens to experts progres-
sively. For theeth expert, we assign tokens based
on the top-⌈ρeL⌉scores, and route the remaining
tokens to the next (e+ 1 th) expert. Formally, it
employs the following sparse masking function
me :R L×E→{0,1}L×E, where:
me(x) =1
[
top⌈ρeL⌉
(
S(x)∏e−1
i=1 (1−mi(x))
)]
,
(2)
and 1 denotes the element-wise indicator function,
producing 1 for the top-⌈ρeL⌉scores, and 0 other-
wise. Note me(x) depends on the masks of pre-
ceding experts, ensuring that tokens previously as-
signed to other experts are skipped, thereby guar-
anteeing an exclusive mapping of each token to a
single expert.
3.2 Decode-Phase Routing
The preceding paragraphs outline the train-
ing/prefill phase of our token-wise ECR mecha-
nism, which operates on a sequence of tokens as
input. However, this routing approach cannot be
directly applied to the decoding phase of CLMs,
where tokens are generated iteratively, this means
that we need a different routing strategy for the
decode phase.
A key advantage of eq. (2) is that it ensures ex-
clusive expert mapping for each token, resulting in∑E
e=1me(x) being a one-hot vector for each token.
If we encourage both phases to have the same ex-
pert assignments, we can simply use arg maxS(x)
to determine the expert assignment during decod-
ing. During the decoding phase, expert assign-
ments for the next token are then determined by
simply taking the arg max of the scoring function,
i.e., This approach eliminates the need for a top-
k operation over the entire input sequence, which
is infeasible during decoding. To summarize, the
prefill and decode phases use the following routing
functions:
Tprefill(x) =∑E
e=1me(x),
Tdecode(x) =1[arg max(S(x)) =e].
(3)
3.3 Prefill-Decode Consistency Loss
To align arg maxS(x) with the expert assignment
arg maxTprefill(x), we introduce the following
consistency loss where arg maxTprefill(x) extracts
the expert index assigned to each token:
Laux(x) =L sce(
S(x),arg maxT(x)
)
.(4)
The total training loss for the model combines
the primary language-modeling loss Lmodel with
the auxiliary lossLaux(x(l)) applied across all lay-
ersl∈{1,...,L}, weighted byα:
L=L model +α
L
∑L
l=1Laux(x(l)).(5)
22893

<!-- page 5 -->

3.4 Mixture of Weight-Shared GQAs
KV projectionBuilding on the token-wise ex-
pert assignment described earlier, we extend the
attention mechanism by introducing a mixture of
weight-sharedGQAs. Each expert processes its
assigned tokens independently and maintains KV
caches tailored to its group configuration, achiev-
ing an efficient trade-off between computation
and memory. Assuming a pretrained attention
layer with (wk,w v)∈RD×D,(b k,b v)∈RH×D,
key and value weights and biases, where D is
the embedding dimension, we first define the fol-
lowing key and value projection pj
h :R L×D→
RH×L×(D/H)for thehth head, wherej∈{k,v},
h∈{1,...,H}, and:
P j(x)h =
(
wjx⊤+b j)[ D(h−1)
H +1 : Dh
H
],(6)
Here, the subscript z[a:b] denotes the slice operation
which selects elements from the first dimension of
zranging fromatob.
KV groupingInspired by GQA (Ainslie et al.,
2023), for each expertfj
e , we design the following
mechanism to reduce the number of projected KV
heads fromH toH/2e groups of size 2e by taking
the average of the corresponding grouped heads.
Specifically, for each groupingg∈Ge of experte,
we havef j
e :R H×L×(D/H)→RH/2e×L×(D/H):
fj
e,g(x) = 1/2e
∑
h∈gpj(x)h,(7)
whereGe groups a range of heads by size 2e, For
example, if H= 4 and E= 3 , we have G1 =
{{1},{2},{3},{4}},G2 ={{1,2},{3,4}}, and
G3 ={{1,2,3,4}}. Notably to ensure parameter
efficiency, we share the same key and value weights
across all experts. While for mathematical clarity
we define the mean operation over the projected
heads, one can easily instead aggregate the KV
projection weights before applying the projection
operation to achieve the same effect.
Due to this grouping, the total KV cache size is
thus adjusted based on which expert processes the
token, with the cache size of the eth expert being
H/2e of the original size.
Attention computationBefore computing the
attention, for experte we match the KV head counts
H/2e with the query head countH by repeating the
KV heads 2e times usinghj
e,g :R H/2e×L×(D/H)→
RH×L×(D/H):
hj
e,g(x) =f j
e,g(x)⊗12e.(8)
where⊗denotes the outer product, and 12e is a
vector of ones of size 2e. Finally, the overall result
computed by the MoE is:
hj(x) =∑E
e=1 me(x)⊙hj
e,g(x).(9)
It is noteworthy that since me(x) is sparse and
has token-wise exclusive expert assignment, the
most of thehj
e,g(x) are zeroed out and skipped. In
practice, this is carried out efficiently with scatter
and gather tensor operations.
The attention computation is then performed fol-
lowing the standard scaled dot-product attention
mechanism, whereq(x) is the original query pro-
jection:
a(x) =softmax
(
q(x)hk(x)⊤/
√
D
)
hv(x).
(10)
Expert Allocation for Memory Efficiency
mixSGAcomputes varying KV sizes per token
thanks to its dynamic routing mechanism assign-
ing tokens to experts of different group sizes. For
E= 3 experts, the group sizes are 1,2,4 respec-
tively, and the head counts are thusH,H/2,H/4 .
This means that on average given a ratio ofa:b:c ,
all tokens require (a+b/2 +c/4)/(a+b+c) of
the original KV size. Along with the KV cache,
we also store a single index value for each token to
track expert assignment.
Integration with KV evictionAlthoughmixSGA
dynamically allocates per-token KV sizes, it re-
mains fully compatible with KV eviction such as
H2O (Zhang et al., 2024b) and NACL (Chen et al.,
2024a) to further reduce memory usage.
4 Experiments
4.1 Supervised Fine-tuning
Models and methodsWe evaluatemixSGAon
the following CLMs: OPT-{125m,355m} (Zhang
et al., 2022), Llama3.1-8b, Llama3.2-{1b,3b} (Tou-
vron et al., 2023), and Gemma2-2b (Gemma Team
et al., 2024), covering various model sizes and ar-
chitectures. As a default baseline, we implement a
GQA-variant of the original models which forms
KV head groups of size 2 by initializing the KV
projection matrices with the mean of the group. For
fair comparisons,mixSGAis configured with expert
density ratios which maintain the same active KV
head counts, and thus the same KV size, as GQA.
It keeps the pretrained weights from the original
models, and randomly initializes the newly added
22894

<!-- page 6 -->

routing weights with He initialization (He et al.,
2015) and biases with zeros.
Training and evaluation setupWe fine-tune
the modified models on the Dolly-15k instruction-
following dataset (Conover et al., 2023) with
14,000 training samples, and evaluate their per-
formance on 5 conversational datasets: Dolly (DL,
500 testing samples from Dolly-15k), Self-Instruct
(SI) (Wang et al., 2023), Vicuna (VC) (Chiang et al.,
2023), Super-Natural Instructions (SN) (Wang
et al., 2022), and Unnatural Instruction (UI) (Hon-
ovich et al., 2023). In addition to the ROUGE-L
(R-L) scores, which measure the longest common
sub-sequence between generated and reference an-
swers, we also evaluate all answers to the queries
using DeepSeek-V3 (DeepSeek-AI et al., 2024)
to provide feedback scores ranging from 0 to 10.
The template to generate feedback is provided in
Section A. All hyperparameter configurations are
provided in Section A for reproducibility.
Main ResultsFor supervised fine-tuning tasks,
we initiate our approach by conducting a grid
search on a smaller model (OPT-355M) to deter-
mine the optimal expert density ratios, increment-
ing by 0.1 while maintaining the total KV size
constant at 50% of the original model. Our results
show that allocating tokens as 30% to experts with
a group size of 1, 10% to size 2, and 60% to size 4
optimizes performance across most metrics. This
3:1:6 ratio consistently outperforms other configu-
rations. As shown in Table 1,mixSGAconsistently
outperforms GQA across various benchmarks and
model sizes. These results demonstratemixSGA’s
ability to dynamically allocate resources and im-
prove performance over static GQA baselines.
4.2 Continued Pretraining
Models and methodsWe investigatemixSGA’s
ability in continued pretraining on additional cor-
pus. We used a TinyLlama-1.1B model (Zhang
et al., 2024a), which was pretrained on SlimPa-
jama (Soboleva et al., 2023) and StarCoder (Li
et al., 2023) and adapted its weights to GQA with
group size set to 2, CLA (Brandon et al., 2024),
andmixSGA. Both CLA andmixSGAaligns the
same KV cache size as the GQA baseline.
Training and evaluation setupWe train the
models with each method applied for one epoch
of MiniPile (Kaddour, 2023), which amounts to
1.6 billion tokens. We use a diverse set of bench-
marks to evaluate the resulting models: HellaSwag
(Zellers et al., 2019), PIQA (Bisk et al., 2020),
Winogrande (Sakaguchi et al., 2019), ARC-Easy
(ARC-E), ARC-Challenge (ARC-C) (Clark et al.,
2018), and the perplexity on Wikitext-2 (Merity
et al., 2016). For the first six tasks, higher accu-
racy (%) indicates better performance, while lower
perplexity on Wikitext-2 reflects stronger language
modeling ability. The training and evaluation de-
tails are provided in Section A.
Main ResultsIn our continued pretraining set-
ting, the key challenge is to recover previously
learned capabilities of the model with a fraction of
data drawn from a distribution domain similar to
the original pretraining data. As shown in Table 2,
mixSGAconsistently demonstrates competitive or
superior accuracy on most benchmarks. It attains
37.00% on HellaSwag and 56.30% on Winogrande,
both surpassing GQA (group size = 2) and CLA.
Performance on ARC-C (25.17%) also exceeds that
of the baselines, highlightingmixSGA’s strength
in handling more challenging tasks.mixSGAalso
shows a clear advantage in Wikitext-2 PPL, deliver-
ing the lowest value (20.46) among all models. To
summarize, these results indicate thatmixSGAcan
enable the model to preserve previously acquired
knowledge, as applying it to existing models does
not impact their pretrained weights.
mixSGAcompliments cache eviction betterTo
investigate the compatibility ofmixSGAwith dy-
namic KV cache eviction strategies, we conduct
a set of controlled experiments by integrating
H2O (Zhang et al., 2024b) with both GQA and
mixSGAon Gemma2-2b. These experiments are
designed to evaluate whether the orthogonal bene-
fits of token-level eviction and token-wise KV allo-
cation can be combined effectively. Both GQA and
mixSGAare configured to operate under a shared
KV budget of 50% of the original size, with H2O
applied as a post-processing eviction method to
further compress memory. We vary the H2O keep
ratio from 80% down to 20% to simulate increasing
memory pressure. The results, shown in Table 3,
demonstrate thatmixSGAconsistently outperforms
GQA across all compression levels. This validates
thatmixSGAnot only preserves the contextual co-
herence lost in aggressive token eviction, but also
enhances the effectiveness of cache compression
when used in conjunction with existing methods
like H2O. The results demonstrate that integrat-
ingmixSGAwith cache eviction policies further
22895

<!-- page 7 -->

Architecture Method Expert Dolly Self-Instruct Vicuna SN UN Avg.
Ratios R-L DSv3 R-L DSv3 R-L DSv3 R-L DSv3 R-L R-L
OPT-125m
GQA 0:1:0 17.70 2.19 6.93 2.35 12.57 1.81 8.33 2.28 10.56 11.22
mixSGA 1:1:2 19.80 3.22 6.89 3.22 12.33 2.16 11.21 2.65 13.77 12.80
mixSGA 3:1:6 17.65 3.25 7.75 2.96 10.79 2.48 8.77 2.99 12.83 11.56
OPT-355m
GQA 0:1:0 21.11 3.19 7.88 2.09 10.86 1.75 10.51 2.27 12.77 12.63
mixSGA 1:1:2 17.21 3.36 9.19 3.06 10.55 2.10 11.03 2.72 14.67 12.53
mixSGA 3:1:6 21.43 3.48 8.68 3.57 12.19 2.64 11.34 2.85 14.90 13.71
Llama3.2-1B
GQA 0:1:0 20.09 3.45 7.90 3.17 13.21 2.41 12.43 2.74 14.50 13.63
mixSGA 1:1:2 18.87 4.02 9.01 3.68 10.93 2.97 14.09 3.33 17.70 14.12
mixSGA 3:1:6 20.11 4.05 10.03 3.65 14.41 2.99 15.52 3.24 20.42 16.10
Llama3.2-3B
GQA 0:1:0 23.26 4.19 9.95 3.45 14.93 3.54 15.73 3.68 18.23 16.42
mixSGA 1:1:2 25.49 5.08 11.20 4.66 15.34 4.29 19.46 4.12 24.19 19.14
mixSGA 3:1:6 25.57 5.23 13.13 4.43 14.61 3.86 18.32 4.18 24.24 19.17
Llama3.1-8B
GQA 0:1:0 27.40 4.85 11.60 4.60 15.36 3.43 21.72 4.22 23.75 19.97
mixSGA 1:1:2 26.50 6.40 17.22 6.01 15.06 4.90 32.52 6.43 33.91 25.04
mixSGA 3:1:6 28.47 6.97 17.30 5.93 19.19 4.88 35.81 6.68 34.62 27.08
Gemma2-2B
GQA 0:1:0 25.68 5.64 10.43 3.73 16.53 4.25 20.00 4.27 23.68 19.26
mixSGA 1:1:2 24.79 6.18 16.08 5.37 12.70 5.26 26.01 5.55 27.39 21.39
mixSGA 3:1:6 26.15 6.25 17.36 5.62 14.47 5.40 26.82 5.98 28.71 22.70
Table 1: Supervised fine-tuning of a range of models on the Dolly-15k instruction-following dataset (Conover
et al., 2023). Evaluation includes ROUGE-L (R-L) and DeepSeek-V3 feedback scores (DSv3) on 5 conversational
datasets.mixSGAdemonstrates consistent improvements over GQA baselines with the same KV budgets. The “Avg.
R-L” column shows the average ROUGE-L scores across all datasets.
Method↑HS↑PI↑WG↑AE↑AC↑Avg.↓WT
GQA 36.7070.6255.90 54.92 23.89 48.41 22.66
CLA 35.90 68.82 55.4055.4723.81 47.88 24.62
mixSGA37.0069.5356.3054.8425.17 48.57 20.46
Table 2: Continued pretraining on TinyLlama-1.1B with
MiniPile. (↑: higher is better, ↓: lower is better, HS:
HellaSwag, PI: PIQA, WG: Winogrande, AE: ARC-E,
AC: ARC-C, WT: Wikitext-2.)
KR Method↑HS↑PI↑WG↑AE↑AC↑Avg.↓WT
80% GQA 36.869.58 55.04 53.41 23.81 47.7322.70
mixSGA36.570.02 55.33 53.91 25.77 48.3120.53
60% GQA 36.3 68.6254.0653.17 23.63 47.1622.72
mixSGA36.5 70.1853.5153.66 25.34 47.8420.63
40% GQA 36.167.94 53.27 52.19 24.40 46.7822.80
mixSGA35.869.10 54.14 52.27 24.92 47.2520.98
20% GQA 35.2 63.18 49.96 44.36 21.33 42.8123.56
mixSGA35.5 64.80 50.75 44.51 22.53 43.6222.19
Table 3: Integrating H2O with various KV keep ratios on
Gemma2-2b.mixSGAconsistently outperforms GQA
across most tasks and H2O KV keep ratios (KR).
enhances its applicability in inference tasks while
reducing KV memory footprint.
4.3 Ablation Studies
To comprehensively attribute the impact of each
component inmixSGA, we perform ablation studies
under three key aspects by varying the following:
expert density ratios and expert counts, and the aux-
iliary loss with learned routing mechanism. Exper-
iments in Tables 4 and 5 and Table 6 are conducted
Ratios DL SI VC SN UN Avg.
1:9:2 18.41 7.80 11.49 9.78 13.53 12.20
1:6:2 19.608.4712.1411.53 14.79 13.31
1:1:2 18.879.0110.9314.09 17.70 14.12
Table 4: Effect of different expert group ratios under the
same KV size budget (50%) for Llama3.2-1B. Results
are reported for ROUGE-L across multiple benchmarks.
(DL: Dolly Evaluation, SI: Self-Instruct, VC: Vicuna,
SN: Super-Natural Instructions, UN: Unnatural Instruc-
tions, Avg.: Average ROUGE-L across benchmarks)
on Llama3.2-1b and Gemma2-2B respectively, fol-
lowing the same setup in Section 4.1. We provide
detailed analyses of the results below.
Varying the expert ratiosTable 4 investigates
the effect of varying density ratios among experts
while keeping a fixed KV size budget of50%. We
systematically increase the ratio assigned to the 2nd
expert in a group of size 2, testing configurations
from 1:1:2 to 1:9:2, Our results reveals that evalu-
ation metrics improve as the 2nd expert’s ratio de-
creases, indicating a preference for allocating more
tokens to the 1st and 3rd experts. This suggests the
model prioritizes assigning important tokens to the
1st expert, which retains the original model’s KV
projection weights, while routing less significant
tokens to the smallest (3rd) expert.
Varying the expert countsIn Table 5, we in-
vestigate the influence of employing 2-3 experts
22896

<!-- page 8 -->

Ratios DL SI VC SN UN Avg.
3:1:6 20.11 10.03 14.41 15.52 20.42 16.10
3:4:0 18.11 9.12 15.47 12.78 16.25 14.35
1:1:2 18.87 9.01 10.93 14.09 17.70 14.12
1:2:0 13.94 6.83 14.75 8.80 11.06 11.08
Table 5: Effect of redistributing KV cache across tokens
under fixed KV size (50% of the original model) for
Llama3.2-1B. Results are reported for ROUGE-L fol-
lowing the style in Table 4.
Ratios DL SI VC SN Avg.
mixSGA 26.15 17.36 14.47 26.82 21.20
Random router 24.65 12.56 12.24 20.98 17.68
No auxiliary loss10.07 6.22 4.56 8.54 7.35
Table 6: Ablation study on the effect of auxiliary loss
and learned routing for Gemma2-2B with 3:1:6 expert
ratios under a 50% KV budget. Results report ROUGE-
L scores across benchmarks.
while maintaining a fixed total KV budget of 50%.
Specifically, we compare configurations with 3:1:6,
3:4:0, 1:1:2, 1:2:0 ratios. Here, a value of 0 for the
3rd expert indicates its exclusion from the model.
Remarkably, we observe that introducing a 3rd ex-
pert significantly enhances performance, achieving
an average ROUGE-L score improvement of up
to 3.12 across all benchmarks. Given the variable
information content of individual tokens, this find-
ing highlights the critical role of the 3rd expert in
capturing less crucial tokens within the input se-
quence, allowing the other two experts to focus on
processing more significant ones.
Learned RoutingTo assess the auxiliary loss
and learned routing mechanism, we conduct ex-
periments on Gemma2-2B with a 3:1:6 expert ra-
tio, following Section 4.1. As shown in Table 6,
we found that removing the auxiliary loss leads
to inconsistent routing between prefill and decod-
ing, resulting in near-random expert assignments
(0.3458:0.3306:0.3236 for the 3 experts on Dolly),
as the model never learns to route according to ex-
pert density ratios. This causes a severe average
ROUGE-L drop (21.20 to 7.35). We also found that
replacing the learned router with a router that ran-
domly assigns experts per the 3:1:6 ratio degrades
performance.
Varying KV BudgetsTo evaluate the influence
of varying KV budgets on language modeling abil-
ity, we conducted comparative experiments involv-
ingmixSGA, GQA, and CLA across different KV
budgets using the TinyLlama continued pretraining
30% 40% 50% 60% 70%
KV Size
18
20
22
24
26
28Perplexity
CLA
mixSGA
GQA
Figure 3: Comparing TinyLlama-1.1B continued pre-
training withmixSGAand baselines (GQA and CLA)
across varying KV size ratios. Lower perplexity indi-
cates better language modeling performance.
task as outlined in Section 4.2. FormixSGA, the
configurations were set as follows: 0:0:1 for a 25%
KV budget, 1:1:8 targeting 35%, 3:1:6 for 50%,
and 1:1:0 for 75%. CLA was configured to align
with these KV sizes. Given that the TinyLlama-
1.1B attention module comprises only 4 heads,
GQA could thus only employ a group size of 2
to achieve a50%KV budget.
As illustrated in Figure 3,mixSGAconsistently
achieves superior performance, manifesting in
lower perplexity across most KV budgets compared
to the baselines. Notably, CLA experiences a pro-
nounced increase in perplexity as the KV budget
decreases, particularly below 50%, where its per-
formance deteriorates significantly. This highlights
the challenges faced by static approaches in main-
taining accuracy under constrained KV budgets.
Conversely,mixSGAexhibits enhanced robustness,
with lower perplexity levels across various budgets,
suggesting that its dynamic token routing mech-
anism enables more effective resource allocation.
This adaptability underscores its capability to de-
liver improved language modeling performance,
even under limited KV budgets.
5 Conclusion
This paper introducedmixSGA, a framework that
combines dynamic token-wise expert-choice rout-
ing with attention grouping to optimize KV repre-
sentations. By using weight-shared heterogeneous
attention experts,mixSGAadaptively allocates re-
sources based on token importance. Our exper-
iments with Llama3, OPT, and Gemma2 model
families show thatmixSGAoutperforms baseline
approaches in computational efficiency and task
performance, with improved scalability in resource-
constrained scenarios. The routing mechanism of
mixSGAensures consistency between prefill and de-
22897

<!-- page 9 -->

code phases. Overall,mixSGAoffers a scalable and
efficient solution for dynamic KV optimization.
6 Limitations and Risks
Our method assigns tokens to diverse experts
within each layer to improve efficiency. However,
we only configure the same capacity ratios across
different layers, which may ignore diverse token
importance across lower and deeper layers. There-
fore, our method can be further improved with a
global importance metric, automatically configur-
ing capacity ratios tailored for each layer. This
potentially offers versatility and flexibility for bet-
ter resource utilization. Moreover, mixSGA’s effi-
ciency gains in computational and memory costs
enable resource-constrained researchers to lever-
age advanced LLMs, fostering innovation in fields
like NLP and healthcare while supporting sustain-
able AI through reduced energy consumption. This
democratization of access can accelerate scien-
tific progress and broaden AI’s societal benefits.
However, these efficiencies could lower barriers
for harmful uses, such as generating misinforma-
tion or amplifying biases in routing decisions, and
may introduce vulnerabilities like slowdowns from
adversarial inputs. Future work may incorporate
fairness-aware routing and adversarial robustness
to ensure ethical deployment, aligning technologi-
cal advances with responsible AI practices.
Acknowledgments
This work is supported in part by the
National Key R&D Program of China
(2023YFC3321600), National Natural Sci-
ence Foundation of China (62376263 and
62372443), Guangdong Basic and Applied
Basic Research Foundation (2023B1515130002),
Natural Science Foundation of Guangdong
(2024A1515030209 and 2024A1515011970),
Shenzhen Science and Technology Innovation
Commission (JCYJ20230807140507015 and
JCYJ20220531100804009). This work is also sup-
ported in part by Technology Development Fund
of Macao S.A.R (FDCT) under 0123/2022/AFJ
and FDCT 0081/2022/A2, and was carried out in
part at SICC, which is supported by SKL-IOTSC,
University of Macau.
References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebron, and Sumit Sanghai.
2023. GQA: Training generalized multi-query trans-
former models from multi-head checkpoints. InThe
2023 Conference on Empirical Methods in Natural
Language Processing.
Dosovitskiy Alexey. 2020. An image is worth 16x16
words: Transformers for image recognition at scale.
arXiv preprint arXiv: 2010.11929.
Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng
Gao, and Yejin Choi. 2020. Piqa: Reasoning about
physical commonsense in natural language. InThirty-
Fourth AAAI Conference on Artificial Intelligence.
William Brandon, Mayank Mishra, Aniruddha
Nrusimha, Rameswar Panda, and Jonathan Ragan-
Kelley. 2024. Reducing transformer key-value
cache size with cross-layer attention. InThe Thirty-
eighth Annual Conference on Neural Information
Processing Systems.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu
Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao
Chang, Junjie Hu, and Wen Xiao. 2024. PyramidKV:
Dynamic kv cache compression based on pyramidal
information funneling.Preprint, arXiv:2406.02069.
Hailin Chen, Amrita Saha, Steven Hoi, and Shafiq Joty.
2023. Personalized distillation: Empowering open-
sourced llms with adaptive learning for code gener-
ation. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 6737–6749.
Yilong Chen, Guoxia Wang, Junyuan Shang, Shiyao
Cui, Zhenyu Zhang, Tingwen Liu, Shuohuan Wang,
Yu Sun, Dianhai Yu, and Hua Wu. 2024a. Nacl: A
general and effective kv cache eviction framework for
llms at inference time.Preprint, arXiv:2408.03675.
Yilong Chen, Linhao Zhang, Junyuan Shang, Zhenyu
Zhang, Tingwen Liu, Shuohuan Wang, and Yu Sun.
2024b. DHA: Learning decoupled-head attention
from transformer checkpoints via adaptive heads fu-
sion. InThe Thirty-eighth Annual Conference on
Neural Information Processing Systems.
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion
Stoica, and Eric P. Xing. 2023. Vicuna: An open-
source chatbot impressing gpt-4 with 90%* chatgpt
quality.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind
Tafjord. 2018. Think you have solved question
answering? try arc, the ai2 reasoning challenge.
arXiv:1803.05457v1.
Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie,
Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell,
Matei Zaharia, and Reynold Xin. 2023. Free dolly:
Introducing the world’s first truly open instruction-
tuned llm.
22898

<!-- page 10 -->

DeepSeek-AI et al. 2024. Deepseek-v3 technical report.
Preprint, arXiv:2412.19437.
William Fedus, Barret Zoph, and Noam Shazeer. 2022.
Switch transformers: Scaling to trillion parameter
models with simple and efficient sparsity.Preprint,
arXiv:2101.03961.
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li,
Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, An-
ish Thite, Ben Wang, Kevin Wang, and Andy Zou.
2021. A framework for few-shot language model
evaluation.
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang,
Jiawei Han, and Jianfeng Gao. 2024. Model tells you
what to discard: Adaptive kv cache compression for
llms.Preprint, arXiv:2310.01801.
Gemma Team et al. 2024. Gemma 2: Improving
open language models at a practical size.Preprint,
arXiv:2408.00118.
Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024.
MiniLLM: Knowledge distillation of large language
models. InThe Twelfth International Conference on
Learning Representations.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. 2015. Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification.
Preprint, arXiv:1502.01852.
Or Honovich, Thomas Scialom, Omer Levy, and Timo
Schick. 2023. Unnatural instructions: Tuning lan-
guage models with (almost) no human labor. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 14409–14428, Toronto, Canada.
Association for Computational Linguistics.
Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. 2024a.
Moh: Multi-head attention as mixture-of-head atten-
tion.Preprint, arXiv:2410.11842.
Qingyun Jin, Xiaohui Song, Feng Zhou, and Zengchang
Qin. 2024b. Align attention heads before merging
them: An effective way for converting mha to gqa.
Preprint, arXiv:2412.20677.
Jean Kaddour. 2023. The minipile challenge
for data-efficient language models.Preprint,
arXiv:2304.08442.
Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-
Young Yun. Distillm: Towards streamlined distilla-
tion for large language models. InForty-first Interna-
tional Conference on Machine Learning.
Hang Le, Juan Pino, Changhan Wang, Jiatao Gu, Didier
Schwab, and Laurent Besacier. 2020. Dual-decoder
transformer for joint automatic speech recognition
and multilingual speech translation. InCOLING
2020 (long paper).
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu,
Dehao Chen, Orhan Firat, Yanping Huang, Maxim
Krikun, Noam Shazeer, and Zhifeng Chen. 2020.
Gshard: Scaling giant models with conditional
computation and automatic sharding.Preprint,
arXiv:2006.16668.
R Li, LB Allal, Y Zi, N Muennighoff, D Kocetkov,
C Mou, M Marone, C Akiki, J Li, J Chim, et al. 2023.
Starcoder: May the source be with you!Transactions
on machine learning research.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. SnapKV:
LLM knows what you are looking for before genera-
tion.Preprint, arXiv:2404.14469.
Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Qipeng Guo,
Yuerong Song, Kai Lv, Hang Yan, Linlin Li, Qun
Liu, and Xipeng Qiu. 2025. ReAttention: Training-
free infinite context with finite attention scope. In
The Thirteenth International Conference on Learning
Representations.
Stephen Merity, Caiming Xiong, James Bradbury, and
Richard Socher. 2016. Pointer sentinel mixture mod-
els.Preprint, arXiv:1609.07843.
Emilio Parisotto and Russ Salakhutdinov. 2021. Effi-
cient transformers in reinforcement learning using
actor-learner distillation. InInternational Confer-
ence on Learning Representations.
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhaga-
vatula, and Yejin Choi. 2019. Winogrande: An ad-
versarial winograd schema challenge at scale.arXiv
preprint arXiv:1907.10641.
Noam Shazeer, *Azalia Mirhoseini, *Krzysztof
Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
and Jeff Dean. 2017. Outrageously large neural net-
works: The sparsely-gated mixture-of-experts layer.
InInternational Conference on Learning Representa-
tions.
Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Ja-
cob R Steeves, Joel Hestness, and Nolan Dey. 2023.
SlimPajama: A 627B token cleaned and deduplicated
version of RedPajama. https://cerebras.ai/bl
og/slimpajama-a-627b-token-cleaned-and-d
eduplicated-version-of-redpajama.
Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Met-
zler. 2022. Efficient transformers: A survey.ACM
Comput. Surv., 55(6).
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, et al. 2023. Llama: Open and effi-
cient foundation language models.arXiv preprint
arXiv:2302.13971.
22899

<!-- page 11 -->

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Preprint, arXiv:1706.03762.
Daniel Waddington, Juan Colmenares, Jilong Kuang,
and Fengguang Song. 2013. Kv-cache: A scalable
high-performance web-object cache for manycore.
In2013 IEEE/ACM 6th International Conference on
Utility and Cloud Computing, pages 123–130. IEEE.
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa
Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh
Hajishirzi. 2023. Self-instruct: Aligning language
models with self-generated instructions.Preprint,
arXiv:2212.10560.
Yizhong Wang, Swaroop Mishra, Pegah Alipoormo-
labashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva
Naik, Arjun Ashok, Arut Selvan Dhanasekaran,
Anjana Arunkumar, David Stap, Eshaan Pathak,
Giannis Karamanolakis, Haizhi Lai, Ishan Puro-
hit, Ishani Mondal, Jacob Anderson, Kirby Kuznia,
Krima Doshi, Kuntal Kumar Pal, Maitreya Patel,
Mehrad Moradshahi, Mihir Parmar, Mirali Purohit,
Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma,
Ravsehaj Singh Puri, Rushang Karia, Savan Doshi,
Shailaja Keyur Sampat, Siddhartha Mishra, Sujan
Reddy A, Sumanta Patro, Tanay Dixit, and Xudong
Shen. 2022. Super-NaturalInstructions: Generaliza-
tion via declarative instructions on 1600+ NLP tasks.
InProceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing, pages
5085–5109, Abu Dhabi, United Arab Emirates. As-
sociation for Computational Linguistics.
You Wu, Haoyi Wu, and Kewei Tu. 2025. A systematic
study of cross-layer KV sharing for efficient LLM
inference. InProceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Asso-
ciation for Computational Linguistics: Human Lan-
guage Technologies (Volume 2: Short Papers), pages
396–403. Association for Computational Linguistics.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
Farhadi, and Yejin Choi. 2019. Hellaswag: Can a
machine really finish your sentence? InProceedings
of the 57th Annual Meeting of the Association for
Computational Linguistics.
Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and
Wei Lu. 2024a. TinyLlama: An open-source small
language model.Preprint, arXiv:2401.02385.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel
Artetxe, Moya Chen, Shuohui Chen, Christopher De-
wan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022.
OPT: Open pre-trained transformer language models.
arXiv preprint arXiv:2205.01068.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, et al. 2024b.
H2o: Heavy-hitter oracle for efficient generative in-
ference of large language models.Advances in Neu-
ral Information Processing Systems, 36.
Xiabin Zhou, Wenbin Wang, Minyan Zeng, Jiaxian
Guo, Xuebo Liu, Li Shen, Min Zhang, and Liang
Ding. 2024. DynamicKV: Task-aware adaptive kv
cache compression for long context llms.Preprint,
arXiv:2412.14838.
Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping
Huang, Vincent Y Zhao, Andrew M. Dai, Zhifeng
Chen, Quoc V Le, and James Laudon. 2022. Mixture-
of-experts with expert choice routing. InAdvances
in Neural Information Processing Systems.
22900

<!-- page 12 -->

A Experimental Setup
Our experiments are conducted on open-sourced
datasets. These datasets serve as artifacts for
research purposes, which is aligned with the
goal of our experimental evaluation. Specifically,
Databricks-Dolly-15k dataset uses CC BY-SA 3.0
license. Wikitext-2 is available under the Creative
Commons Attribution-ShareAlike License. The re-
maining datasets in lm-eval-harness are available
under the MIT License.
A.1 Supervised Fine-Tuning Tasks
For the supervised fine-tuning tasks, we apply tem-
plates to both the training and test datasets, fol-
lowing the standard procedure described in (Gu
et al., 2024; Ko et al.). All input text was stan-
dardized to ensure consistency and fairness across
different models. For the DeepSeek-V3 feedback
evaluation (DeepSeek-AI et al., 2024), we use the
template shown in Figure 4, with a temperature
coefficient set to 0.7 to balance the randomness
and diversity of the generated outputs. We first
construct the training data from the Databricks-
Dolly-15k dataset (Conover et al., 2023), wherein
we randomly select 14,000 samples for training
and equally leave 500 samples for validation and
testing, respectively.
As part of our baselines, we modify the origi-
nal pretrained models by integrating them into a
more advanced GQA setup. For models not orig-
inally including GQA results, we apply the GQA
mechanism. In cases where the models already
have GQA results, we replace them with a more
compressed version of GQA, which offers stronger
compression levels. This ensures that our baseline
is consistently adapted for a fair comparison with
the new methods.
We performed full parameter fine-tuning for the
OPT model series (OPT-{125m, 355m}). The
batch size was set to 32, and we used a cosine learn-
ing rate schedule. The learning rate was initially
set to 5e−5and decayed according to the cosine
decay scheduler. The models were trained for 40
epochs to ensure sufficient fine-tuning.
The routing weights were initialized using He
initialization (He et al., 2015). For the learning
rate setup, we initialized the learning rate at 5e−5,
and the learning rate decay followed the same co-
sine schedule. We used a batch size of 32 for both
training and evaluation. We employed gradient ac-
cumulation to simulate a larger batch size without
exceeding memory constraints.
A.2 Continued Pretraining Tasks
In the continued pretraining tasks, we fine-tune the
TinyLlama-1.1b model using the MiniPile dataset
(Kaddour, 2023), which contains 1.6 billion to-
kens. The pretraining weights for the TinyLlama
model were originally trained on a much larger
dataset containing 3 trillion tokens (Zhang et al.,
2024a). To adapt it as our baseline, we reduce
the number of KV heads by half and implement a
deeper GQA configuration. This modification of
pretrained weights degrades the original model’s
performance, as halving the KV heads impacts the
model’s integrity. Therefore, we perform continued
pretraining on the 1.6 billion tokens of MiniPile
to recover the model’s performance and address
this degradation. Note that for all methods, we use
the same hyperparameter settings in continued pre-
training experiments, as illustrated in Table 7 and
Table 8, for fair comparison.
We train the TinyLlama-1.1B model for one
epoch on the MiniPile dataset and then evaluate
its performance using lm-eval-harness(Gao et al.,
2021) framework in a zero-shot setting across sev-
eral benchmarks, including HellaSwag (Zellers
et al., 2019), PIQA (Bisk et al., 2020), Wino-
grande (Sakaguchi et al., 2019), ARC-Easy (ARC-
E), ARC-Challenge (ARC-C) (Clark et al., 2018),
and perplexity on Wikitext-2 (Merity et al., 2016).
These benchmarks present comprehensive assess-
ment of the model’s language modeling abilities
and task-specific performance.
Please act as an impartial judge and
evaluate the quality of the response provided
by an AI assistant to the user question
displayed below. Consider factors such as
helpfulness, relevance, accuracy, depth, and
creativity. While evaluating, focus on
clarity, usefulness, and effort. Please rate
the response on a scale of 1 to 10 by following
this format: ’Rating: [[x.xx]]’, for example:
’Rating: [[5.00]]’.
Figure 4: System prompt template for DeepSeek-V3
feedback evaluation.
B Computational Resources
The experiments were conducted on two types of
server equipped with NVIDIA A100 and V100
GPUs, configured by different model sizes and pre-
cision types.
22901

<!-- page 13 -->

Model Size 1.1B
Max LR 2e-4
LR Scheduler cosine
Optimizer AdamW
β1 0.9
β2 0.95
Warmup Ratio 0.015
Weight Decay 0.1
Gradient Clipping 1.0
Precision Bfloat16
Batch Size (tokens) 256K
Epochs 1
DataSet MiniPile
GPU A100
Table 7: Training Hyperparameters for Continued Pre-
training (TinyLlama-1.1B)
Model Size 1.1B
Hidden Size 2048
Intermediate Size 5632
Max Trained Length 2048
# Layers 22
# Attention Heads 32
# KV Heads 4
RMS Norm eps 1e-5
V ocab Size 32000
Table 8: Model Hyperparameters for TinyLlama 1.1B
For the Llama 8B model, we used servers with 4
NVIDIA A100 GPUs (80GB) and Intel Xeon Gold
6230R processors with 104 CPU cores. We use
bfloat16 (bf16) precision to align with the precision
applied for pretraining and reduce memory burden.
For other Llama and Gemma model series, our
experiments were performed on servers with 4
NVIDIA A100 GPUs (40GB) and the same CPU
and precision configurations.
The OPT models (OPT-{125m, 355m}) were
trained on 4 NVIDIA V100 GPUs (32GB), with In-
tel Xeon Gold 5118 processors with 48 CPU cores,
using float32 (fp32) precision due to the V100’s
lack of hardware support for bfloat16 format.
C Additional Experiments
C.1 Compute and Memory Overheads
Table 9 presents the real compute and memory over-
heads ofmixSGAcompared to GQA. It shows that
mixSGAincurs a marginal increase in FLOPs and
Metrics Method O-125m L3.2-1b G2-2b
#Params GQA 118m 1.22b 2.55b
mixSGA 125m 1.24b 2.61b
#FLOPs GQA 94.8k 113k 237k
mixSGA 94.9k 113k 237k
KV size per GQA 36,864 32,768 106,496
token (bytes)mixSGA 36,867 32,772 106,502
Table 9: Comparison of parameter counts, FLOPs, mem-
ory usage, and time for GQA andmixSGAunder the
same KV size budget (50%). (O-125m: OPT-125m,
L3.1-8b: Llama3.1-8b, L3.2-1b: Llama3.2-1b, G2-2b:
Gemma2-2b)
Method L3.2-1b L3.2-3b L3.1-8b
GQA 59.75 40.16 26.77
mixSGA 57.28 38.65 25.92
∆(%) 4.13 3.75 3.17
Table 10: Decoding throughput (tokens/s) of GQA and
mixSGAunder a 50% KV budget, with batch size 1,
10,000 tokens generated, and averaged over five tri-
als. Higher is better. (L3.2-1b: Llama3.2-1b, L3.2-3b:
Llama3.2-3b, L3.1-8b: Llama3.1-8b)
KV sizes, with slightly higher parameter overheads
due to routing weights. To compare the real in-
ference time ofmixSGAand GQA, we measure
decoding throughput (tokens per second) under a
50% KV budget, using a batch size of 1 and gener-
ating 10,000 tokens over five trials. As shown in Ta-
ble 10,mixSGAachieves throughput performance
of 57.28, 38.65, and 25.92 tokens/s on Llama3.2-
1b, Llama3.2-3b, and Llama3.1-8b, respectively,
compared to GQA’s 59.75, 40.16, and 26.77 to-
kens/s. This results in a modest 3-4% overhead for
mixSGA, reflecting its dynamic routing complexity.
These results, based on a naïve implementation,
suggest significant potential for optimization to fur-
ther reduce this gap.
C.2 Optimal Expert Ratio Analysis
To understand why the 3:1:6 expert ratio consis-
tently yields superior performance across models,
we analyze the allocation of expert ratios under a
fixed 50% KV budget. The ratios (x,y,z ) for three
experts are constrained as follows:



x+ 0.5y+ 0.25z= 0.5,Budget,
x+y+z= 1,Allocation,
0≤x,y,z≤1,Bound.
(11)
22902

<!-- page 14 -->

We perform a grid search on Gemma2-2B, vary-
ing x in 0.05 increments, resulting in six feasi-
ble configurations as shown in Table 11. Notably,
x= 0.3,y= 0.1,z= 0.6 , (i.e., x:y:z= 3:1:6 )
achieves the highest average ROUGE-L score
(21.20) across Dolly (DL), Self-Instruct (SI), Vi-
cuna (VC), and Super-Natural Instructions (SN),
outperforming other configurations. This indicates
that allocating 30% of tokens to the first expert
(group size 1), 10% to the second (group size 2),
and 60% to the third (group size 3) optimizes per-
formance by prioritizing significant tokens for the
first expert while efficiently handling less critical
tokens with the third. These results justify the adop-
tion of the default ratio for our experiments in this
study.
Ratios DL SI VC SN Avg.
1:17:2 20.61 10.18 6.42 14.54 12.94
1:7:2 23.49 10.70 14.10 17.72 16.50
3:11:6 25.36 12.6016.9018.52 18.35
1:2:2 25.80 13.45 15.54 19.04 18.46
1:1:2 24.79 16.08 12.70 26.01 19.90
3:1:6 26.15 17.3614.4726.82 21.20
Table 11: ROUGE-L scores (↑) for Gemma2-2B with
varying expert ratios under a 50% KV budget across
Dolly (DL), Self-Instruct (SI), Vicuna (VC), and Super-
Natural Instructions (SN).
22903
