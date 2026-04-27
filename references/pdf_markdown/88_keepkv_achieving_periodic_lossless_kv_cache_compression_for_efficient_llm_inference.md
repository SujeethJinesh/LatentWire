# references/88_keepkv_achieving_periodic_lossless_kv_cache_compression_for_efficient_llm_inference.pdf

<!-- page 1 -->

KeepKV: Achieving Periodic Lossless KV Cache Compression
for Efficient LLM Inference
Yuxuan Tian1, Zihan Wang1, Yebo Peng1, Aomufei Yuan1, Zhiming Wang1,
Bairen Yi2, Xin Liu2, Yong Cui3, Tong Yang1*
1State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University
2ByteDance
3Tsinghua University
Abstract
Efficient inference of large language models (LLMs) is hin-
dered by an ever-growing key-value (KV) cache, making
KV cache compression a critical research direction. Tradi-
tional methods selectively evict less important KV cache
entries, which leads to information loss and hallucinations.
Recently, merging-based strategies have been explored to re-
tain more information by merging KV pairs that would be
discarded; however, these existing approaches inevitably in-
troduce inconsistencies in attention distributions before and
after merging, causing degraded generation quality. To over-
come this challenge, we propose KeepKV, a novel adaptive
KV cache merging method designed to preserve performance
under strict memory constraints, achieving single-step lossless
compression and providing error bounds for multi-step com-
pression. KeepKV introduces the Electoral V otes mechanism
that records merging history and adaptively adjusts attention
scores. Moreover, it further leverages a novel Zero Inference-
Perturbation Merging method, compensating for attention loss
resulting from cache merging. Extensive experiments on var-
ious benchmarks and LLM architectures demonstrate that
KeepKV substantially reduces memory usage while success-
fully retaining essential context information, achieving over
2× inference throughput improvement and maintaining supe-
rior generation quality even with only 10% KV cache budgets.
1 Introduction
Transformer-based large language models (LLMs) have
demonstrated remarkable capabilities across various appli-
cations (Touvron et al. 2023b; Jiang et al. 2023; OpenAI
et al. 2024; Wan et al. 2024a; Rozi`ere et al. 2024). To accel-
erate inference, LLMs commonly employ a key-value (KV)
cache mechanism, which stores the KV embeddings of pre-
viously processed tokens to avoid redundant computations
(Vaswani et al. 2017; Dai et al. 2019; Rae et al. 2019). How-
ever, as LLMs continue to support increasingly longer context
lengths, the size of the KV cache grows rapidly, becoming
a major bottleneck for inference (Kwon et al. 2023). For
example, in the case of LLaMA-3-70B, a batch size of 128
with an 8K context length requires up to 320GB of KV cache
memory (Grattafiori, Dubey et al. 2024). Consequently, com-
*Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
pressing the KV cache while preserving generation quality
has become a crucial challenge.
Prior works mainly explore two approaches for KV cache
compression: eviction-based and merging-based methods,
both of which are inherently lossy. Eviction-based approaches
selectively retain critical cache entries using heuristics like
attention scores and token positions, permanently discarding
less critical entries (Xiao et al. 2024b; Zhang et al. 2023;
Reid and Zhu 2024; Liu et al. 2024; Li et al. 2024; Yang
et al. 2024a) and thus causing context loss and potential hal-
lucinations (Zhang et al. 2024). In contrast, merging-based
methods aim to integrate rather than discard KV entries to
retain more information. Recent representative studies, such
as CaM (Zhang et al. 2024), D2O (Wan et al. 2024b), and
KVMerger (Wang et al. 2024), have explored strategies like
weighted key-value merging to mitigate context loss. Nev-
ertheless, these methods vary widely in merge candidate
selection and merging weight computation, and lack solid
theoretical foundations. We observe that existing strategies
inevitably induce attention inconsistencies and output per-
turbation. Specifically, the merged KV pair’s attention score
is lower than the sum of the original scores prior to merg-
ing—a phenomenon we term ”Attention Sag” (illustrated in
Figure 1). These issues underline the necessity for an efficient
and theoretically grounded KV cache merging strategy.
In this paper, we propose KeepKV, a novel KV cache
merging method designed to maintain inference consistency
and preserve essential contextual information. To the best of
our knowledge, KeepKV is the first approach to achieve
single-step lossless compression and to provide theoreti-
cal error bounds for multi-step compression. We first con-
duct a comprehensive theoretical analysis of existing evic-
tion and weighted merging methods, grounded in the atten-
tion computation process, to reveal their fundamental limi-
tations.Building on these theoretical insights, we propose a
two-stage innovative design in KeepKV:
First, we propose techniques that achieve lossless compres-
sion for a single step. Specifically, we introduce the Elec-
toral V otes mechanism, which records merging history, en-
abling accurate reconstruction of the original KV embeddings
from compressed representations. Additionally, we present
the Zero Inference-Perturbation Merging (ZIP-Merging) ap-
proach, which automatically adjusts weights to compensate
for any losses caused by merging, maintaining attention con-
arXiv:2504.09936v2  [cs.LG]  27 Nov 2025

<!-- page 2 -->

Eviction Merging (Others) KeepKV
1 1 1 1 1 1 1
1
 3
 2 1
Full(Target)
Permanent Lost
Attention Sag
Electoral
Votes
ZIP-Merging
Weighted Merging
Equivalent
(a) (b) (c) (d)
Figure 1: Illustration of KeepKV vs. Existing Methods. The three middle blocks represent KV subject to eviction/merging.
(a) Eviction methods permanently discard them. (b) Merging methods integrates them into retained KV , but the result is not
equivalent to the full KV , causing ”Attention Sag.” (c) Full KV serves as the ideal baseline. (d) KeepKV uses Electoral V otes as
merging records and applies ZIP-Merging to minimize output disturbance, ensuring consistency and improving performance.
sistency. These designs theoretically guarantee zero output
perturbation at the current iteration despite compression.
Second, we extend KeepKV to multi-step generation by
estimating attention scores based on historical patterns. This
is motivated by our empirical observation of strong locality
in attention scores, also confirmed in prior studies (Dong
et al. 2024; Zhang et al. 2024). Crucially, we provide theo-
retical analyses guaranteeing bounded output perturbation
across multiple steps, thereby ensuring consistent inference
quality under extended generation. Moreover, we offer a theo-
retical interpretation for prevalent similarity-based candidate
selection methods, incorporating it into our design.
By integrating these innovations, KeepKV further enables
periodic lossless KV cache compression by storing a com-
plete KV cache externally and periodically loading com-
pressed representations into memory, thereby maintaining
inference consistency and preserving essential contextual
information. Through theoretical derivation and extensive
experiments, we demonstrate that KeepKV effectively pre-
serves attention stability and output consistency, outperform-
ing state-of-the-art KV cache eviction and merging methods.
The contributions of this paper are summarized as follows:
• We propose KeepKV, a novel adaptive KV cache merging
approach, which introduces the Electoral V otes mecha-
nism and Zero Inference-Perturbation Merging, achiev-
ing single-step lossless compression and providing error
bounds for multi-step compression.
• Extensive experiments across various tasks and models
show that KeepKV maintains better performance under
limited cache, outperforming existing KV cache eviction
and merging methods.
• We are the first to theoretically analyze KV merging from
the perspective of eliminating output perturbation. We
provide guarantees on the perturbation bound of KeepKV
and reveal the theoretical basis for merge candidate selec-
tion and weight design. Hopefully, our study can inspire
future research on KV cache compression.
2 Related Work
KV cache has become a major bottleneck for efficient LLMs
inference. Post-training optimization serves as a key solu-
tion due to its real-time and extensible capabilities.(Shi et al.
2024). Existing methods fall into three categories:quantiza-
tion,eviction, andmerging.
KV Cache Quantization.Quantization methods con-
vert tensor values to lower precision to reduce bit-width.
KVQuant (Hooper et al. 2024) applies Per-Channel Quantiza-
tion for keys and Per-Token Quantization for values. MiKV
(Yang et al. 2024b) introduces mixed-precision KV caching,
where less critical KV are stored at lower precision. Addi-
tionally, GEAR (Kang et al. 2024) leverages low-rank matrix
approximation for quantization residuals to minimize quanti-
zation loss. Our KeepKVreduces the number of cached KV
pairs through merging, which is orthogonal to quantization
methods and can be combined for better efficiency.
KV Cache Eviction.Eviction methods only retain more
important KV entries. StreamingLLM (Xiao et al. 2024b)
and LM-infinite (Han et al. 2024) identifies the importance of
the initial k tokens for generation. H2O (Zhang et al. 2023),
ScissorsHand (Liu et al. 2024) and RoCo (Reid and Zhu
2024) recognize crucial KV based on attention scores, while
SnapKV (Li et al. 2024) utilizes attention within an obser-
vation window. Recent works explore improved budget allo-
cation strategies across layers and heads. Pyramid (Cai et al.
2024; Yang et al. 2024a) allocates more cache to lower layers,
whereas AdaKV (Feng et al. 2024), HeadKV (Fu et al. 2024),
and DuoAttention (Xiao et al. 2024a) focus on inter-head dif-
ferences. However, eviction causes irreversible information
loss, potentially degrading generation quality.
KV Cache Merging.KV cache merging combines less
important KV entries instead of permanently discarding
them. DMC (Nawrot et al. 2024) learns when and how to
merge through training, which limits generalization and in-
troduces extra overhead. In contrast, CaM (Zhang et al. 2024)
adaptively merges evicted value states into others but does
not merge the corresponding keys. Recently, D2O (Wan

<!-- page 3 -->

et al. 2024b) selects merge candidates and assigns merg-
ing weights based on cosine similarity between key states,
while KVMerger (Wang et al. 2024) introduces a clustering-
based method to group merge candidates and computes merg-
ing weights using Gaussian Kernel Weights. However, these
methods fail to maintain attention consistency before and
after merging, leading to output perturbation. We propose
a novel merging approach designed to eliminate output per-
turbation, supported by theoretical analysis and extensive
comparisons.
3 Methodology
3.1 Preliminary: Inference with KV Cache
We first introduce the attention computation process with KV
cache. For simplicity, we consider a single attention head
at one layer. Let the attention module’s weight matrices be
Wq, Wk, Wv ∈R d×d, where d denotes the hidden dimension.
During the prefill stage, given an input prompt tensor XL ∈
RL×d = [x 1, x2, . . . , xL], where L represents the prompt
length, the KV states are computed and stored in the KV
cache as follows:
KL =X LWk = [k1, k2, . . . , kL],
VL =X LWv = [v1, v2, . . . , vL].(1)
In the decoding phase, KV cache are repeatedly utilized,
while the newly computed KV pairs are continuously ap-
pended to it. Specifically, given the input at the t-th gen-
eration step, xt ∈R d, the KV cache update and attention
computation are performed as follows:
Kt = [Kt−1, kt], k t =x tWk
Vt = [Vt−1, vt], v t =x tWv (2)
At =softmax
 qtK T
t√
d

, q t =x tWq
st
i =e
qt ki√
d , o t =
tX
i=1
At
ivi =
Pt
i=1 st
ivi
Pt
i=1 st
i
(3)
KV cache effectively reduces redundant computation, but
at the cost of increased memory consumption. Therefore,
an important challenge is to compress the KV cache while
maintaining model performance.
3.2 Rethinking KV Cache Eviction and Merging
Eviction and merging methods reduce memory usage by de-
creasing the number of stored KV pairs. The core motivation
behind these studies is to minimize the impact of cache com-
pression on the output. A fundamentalsubtaskis to ensure
that the output (ot) remains as close as possible before and
after compression at the current step. However, our analy-
sis shows that existing methods inevitably introduce output
perturbation and can not accomplish this task.
Perturbation in KV Cache Eviction.Eviction methods
discard KV pairs deemed unimportant. Suppose we discard
the pair (ke, ve), and denote the output as o′
t. Based on Equa-
tion 3, we obtain:
o′
t =
Pt
i=1,i̸=e st
ivi
Pt
i=1,i̸=e st
i
= 1
1−A te
 
ot −A t
eve

.(4)
Remark 1.Equation 4 reveals that evicting (ke, ve) causes
o′
t to deviate from ot, with the deviation primarily determined
by At
e. This formally explains why eviction methods generally
prioritize discarding KV pairs with lower attention scores.
Although current methods optimize eviction and cache
allocation strategies (Yang et al. 2024a; Fu et al. 2024) to
minimize output impact, they cannot eliminate the pertur-
bation in Equation 4. Previous studies have indicated that
attention is not always sparse, especially in tasks requiring
full context, as shown in Figure 2. Moreover, evicted KV
may become important later, but irreversible eviction leads
to permanent loss.
Attention Sag in KV Cache Merging.Merging methods
integrate less important KV into others rather than discard-
ing them. Existing studies typically use weighted merging
(Nawrot et al. 2024; Wan et al. 2024b; Wang et al. 2024);
formally, merging(k e, ve)into(k c, vc)is expressed as:
kr =w eke +w ckc, v r =w eve +w cvc.(5)
Here, (kr, vr) are the merged vectors, with weightswe, wc
determined by the merging method. In D2O (Wan et al.
2024b), they depend on the cosine similarity between ke
and kc, while in KVMerger (Wang et al. 2024), they are
computed using Gaussian Kernel values. The weights sat-
isfy the normalization condition we +w c = 1 . However,
this widely used convex combination method also introduces
output perturbations:
Theorem 2.Current weighted merging (convex combina-
tion) methods reduce the merged KV pair’s attention score
compared to the sum of the original scores before merging,
i.e.,A ′t
r < A t
e +A t
c, ultimately leading to∥o ′
t −o t∥>0.
The formal proof is in Appendix A.2. We term this atten-
tion inconsistency from merging asAttention Sagand Figure
2 (c) illustrates this phenomenon. We provide an intuitive
comprehension: existing methods merge multiple vectors
into one, treating it equivalently as any other single vector
in subsequent attention computations. This erases merging
history, making it impossible to distinguish whether a KV
pair is original or has absorbed numerous others.
3.3 Method: KeepKV
Electoral Votes and ZIP-Merging Electoral Votes.To
address Attention Sag, we propose the Electoral V otes mech-
anism, which records the number of merges pi (initialized to
1) each KV pair undergoes. A natural analogy is the Electoral
College system, where electors hold votes proportional to
their state’s population rather than a uniform share. The at-
tention score of each KV is then scaled by its votes to approx-
imate the original multiple KV’s influence before merging.
For example, if a KV pair(kr, vr) has a vote count of pr = 3,
it is equivalent to three identical and independent instances of
(kr, vr) participating in the attention computation. Formally,

<!-- page 4 -->

0
 20
 40
 60
 80
 100
T op-x% tokens
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Attention
(a)
Streaming
H2O
 SnapKV
Pyramid
0
10
20
30
40
50Recall Rate (%) (b)
0
 200
 400
 600
 800
1000
T oken Index
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4Attention Variance
1e−5
Global Variance
Local Variance (c)
0
 10
 20
 30
 40
 50
 60
Generation Step
100
101
102
103
Relative Error
H2O
SnapKV
PyramidInfer
KeepKV (d)
Figure 2: (a) Cumulative distribution of attention scores. Retaining the top-k tokens does not always preserve the majority of
scores. (b) Proportion of to-be-evicted prompt tokens appearing in the top-20% attention scores during generation (compression
rate = 20%). (c) Each token’s variance of its attention scores at each generation step (blue dots) is greater than the average
variance within a sliding window (orange dots).(d) Relative errors for prediction of KeepKV and existing methods.
the outputs before (ot) and after merging (o′
t) are defined as
follows:
ot =
Pt
i=1 pist
ivi
Pt
i=1 pist
i
,
o′
t =
Pt
i=1,i̸=e,c pist
ivi +p rst
rvr
Pt
i=1,i̸=e,c pist
i +p rstr
,
pr =p e +p c.(6)
Zero Inference-Perturbation Merging (ZIP-Merging).
The Electoral V otes mechanism enables the elimination of
output perturbations. We define the merging equations and
theorem as follows:
kr =
(weke +w ckc) ln we+wc
pe+pc
we lns te +w c lns tc
,
vr = weve +w cvc
we +w c
,
we =p est
e, w c =p cst
c.(7)
Theorem 3.The merging method in Equation 7 is
perturbation-free, that is,∥o ′
t −o t∥= 0
Remark 4.The proof is in Appendix A.3. Intuitively, our
method ensures attention consistency by preserving historical
information via Electoral Votes and applying proper scaling
(ZIP-Merging) to(k r, vr)instead of a convex combination.
This theorem confirms that our novel merging approach
can eliminate output perturbations and complete the subtask
introduced at the beginning of this section. However, its appli-
cability remains limited to the current iteration, and extending
it to multi-step generation requires additional design.
Extending to Multi-Step Generation EMA Attention
Scores.For ZIP-Merging to be effective in real-world multi-
step generation, a solid comprehension of attention score
dynamics is essential. Fortunately, empirical observations
show that attention scores exhibit strong locality (Figure
2 (d)), meaning a token’s attention scores evolve smoothly
across adjacent steps, which is also validated by prior studies
(Yang et al. 2024a; Zhang et al. 2024; Dong et al. 2024). From
this, we employ the Exponential Moving Average (EMA)
(Hunter 1986; Busbridge et al. 2023) with bias correction, a
widely used technique in time-series analysis, formulated as
follows:
ˆst = 1
1−α t St, S t =



tP
k=t−w
(1−α)α t−ksk, t=L
αSt−1 + (1−α)s t, t > L
(8)
Note that after the prefill stage, we compute EMA scores
using a recent window of length w rather than the entire se-
quence to obtain a more accurate estimation (Li et al. 2024;
Yang et al. 2024a). We find that this method outperforms
mainstream approaches, such as cumulative attention and slid-
ing window averaging, in predicting attention scores.Building
on this, replacing all score st
i in Equation 7 with our EMA
scores ˆst
i from Equation 8 successfully achieves the exten-
sion to multi-step generation. Consequently, the future output
perturbation becomes estimable and controllable. We present
the following theorem and lemma(proof in Appendix A.4):
Theorem 5.For the t′-th step, let
1−
ˆst′
i
st′
i
 ≤ϵ, ϵ <1 , the
output perturbation satisfies Θt′ < 2ϵ(1+ϵ)γ
(1−ϵ)2 , provided that
∥vi −v j∥ ≤γ,∀i∈[t ′], j∈ {e, c}.
Lemma 6.As the prediction error ϵ decreases and the
merged candidates become increasingly similar, the output
perturbation reduces to zero. That is, when either ϵ= 0 or
(ke, ve) = (kc, vc), we have:Θ t′ = 0.
Similarity-driven merging.Lemma 6 shows that output
perturbation decreases as prediction error ϵ reduces, and
closer merging objects result in lower perturbation. Clearly, if
the merged KV pairs are identical, retaining one pair and set-
ting its Electoral V otes to 2 introduces no error in subsequent
computations. This provides a theoretical justification for
prior merging strategies favoring high-similarity KV pairs
(Wan et al. 2024b; Wang et al. 2024). Following this, we
merge each evicted KV pair with the retained one having the
highest cosine similarity of keys, using a predefined thresh-
old T to determine whether merging should occur, avoiding

<!-- page 5 -->

𝒌𝒄𝒌𝒆
11111
𝒌࢘
111
 2
𝒌࢘
𝒌࢘
𝑞𝑡+1




Attention Score
KV
Cache
Votes 11111
෡࢙𝒆
࢚෡࢙𝒄
࢚
Algorithm 1KeepKV Merging att-th Step
1:Input:Attention scoress t, EMA scoresS t−1, KV cache
2:LetK e denote the to-be-evicted cache
3:LetK c denote the retained cache
4:U=cosineSimilarity(K e, Kc)
5:For eachk e ∈K e, selectk c = Argmaxkc∈Kc(U e), U e,c > T
6: ˆst =updateEMA(S t−1, st)▷Eq. (8)
7:Merge:
8:k r = ln ((pe ˆste+pc ˆstc)/(pe+pc))
pe ˆste ln ˆste+pc ˆstc ln ˆstc
(pe ˆsteke +p c ˆstckc)
9:v r = 1
pe ˆste+pc ˆstc
(pe ˆsteve +p c ˆstcvc)
10:p r =p e +p c
11:Discard(k e, ve),(k c, vc)and insert(k r, vr)into KV cache
12:Output:Updated KV cache
Figure 3: Illustrative example of KeepKV. (0) (ke, ve) is selected for eviction by specific compression method. (1) The retained
KV with the highest cosine similarity, (kc, vc), is selected. (2) EMA attention scores are updated. (3) ZIP-Merging is performed.
(4) Consequently, with the Electoral V otes, the compressed KV can preserve the influence of the original KV in attention
computations.
the overhead of dynamic adjustments like D2O (Wan et al.
2024b). Furthermore, we observe that, during the prefill stage,
reversing the conventional order—by first merging based on
key similarity and then applying the eviction strategy, instead
of merging after eviction as commonly done—can improve
generation quality.
We present the workflow of KeepKV in Figure 3. Notably,
KeepKV imposes no specific constraints on cache allocation
or token selection strategies. It can directly integrate with
common token selection methods by designating the merg-
ing pairs based on their eviction and retention sets, and it
is also compatible with various cache allocation strategies.
Thus, KeepKV demonstrates strong adaptability and can be
combined with a range of mainstream cache compression
methods, significantly enhancing both compression capabil-
ity and generation quality.
4 Experiment
4.1 Experiment Settings
TasksWe evaluate KeepKV on datasets with standard and
extended context lengths, covering question-answering, sum-
marization, and synthetic tasks. Specifically, for question-
answering, we utilize MathQA (Amini et al. 2019), Open-
BookQA (Mihaylov et al. 2018) and other tasks from the
lm-eval-harness framework (Gao et al. 2024). For summa-
rization, we employ the XSUM(Narayan, Cohen, and Lapata
2018) and CNN/DailyMail(Nallapati et al. 2016) tasks pro-
vided by the HELM framework. To assess performance on
long-context tasks, we adopt LongBench(Bai et al. 2024),
which effectively examines the algorithm’s compression ca-
pabilities across diverse subtasks, including single-document
QA, multi-document QA and synthetic tasks.
Models and baselines.Our evaluation is based on sev-
eral representative LLMs, including Llama-2 (Touvron et al.
2023b), Llama-3 (Grattafiori, Dubey et al. 2024), and Mistral
(Jiang et al. 2023). We compare our method against multiple
baseline approaches: representative cache eviction methods
such as Streaming (Xiao et al. 2024b), H2O (Zhang et al.
2023) and PyramidInfer (Yang et al. 2024a), and prominent
cache merging methods including CaM (Zhang et al. 2024)
and D2O (Wan et al. 2024b). For all baselines, we follow
the authors’ released implementations and hyperparameter
settings in their papers.
Implementation.In our main experiments, we set the
merging threshold T to 0.8. For token selection and cache
allocation, we follow the strategy recommended by Pyra-
midInfer (Yang et al. 2024a), which allocates fixed cache
budgets, making it simple and efficient. And it is sufficient
to demonstrate the advantages of our algorithm. In contrast,
D2O (Wan et al. 2024b) applies dynamic allocation based on
extra computation after prefill phase for each sequence. We
implement KeepKV using the Hugging Face Transformers
(Wolf et al. 2020) and experiments are by default conducted
on NVIDIA A100 80GB GPUs.
4.2 Accuracy on KV Cache Compression Ratios
In Figure 4, we benchmark KeepKV on both the lm-eval-
harness and HELM frameworks, comparing the fully cached
KV version against multiple KV cache compression meth-
ods, including our proposed KeepKV. The x-axis repre-
sents the compression ratio, defined as the ratio between
the compressed KV cache budget and the prompt length L.
The results demonstrate that KeepKV consistently outper-
forms all other compression methods across various compres-
sion ratios. Particularly at extremely low compression rates,
KeepKV achieves significantly better performance, highlight-
ing its superior compression capability to retain maximal
information within highly constrained memory budgets while
effectively minimizing output perturbations introduced by
compression.

<!-- page 6 -->

100
 101
 102
KV Cache Budget (%)
0.00
0.05
0.10
0.15
0.20
0.25ROUGE-L
xsum, LLAMA-7B
Full
KeepKV
Pyramid
H2O
Streaming
100
 101
 102
KV Cache Budget (%)
0.00
0.03
0.05
0.08
0.10
0.13ROUGE-2
xsum, LLAMA-7B
Full
KeepKV
Pyramid
H2O
Streaming
100
 101
 102
KV Cache Budget (%)
0.20
0.40
0.60
0.80COVERAGE
cnndm, LLAMA-7B
Full
KeepKV
Pyramid
H2O
Streaming
100
 101
 102
KV Cache Budget (%)
0.00
0.10
0.20
0.30ROUGE-L
xsum, LLAMA-2-7B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.00
0.20
0.40
0.60
0.80
1.00COVERAGE
cnndm, LLAMA-2-7B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.00
0.05
0.10
0.15ROUGE-2
xsum, LLAMA-2-13B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.00
0.20
0.40
0.60
0.80
1.00COVERAGE
cnndm, LLAMA-2-13B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.50
0.60
0.70COVERAGE
xsum, LLAMA-3-8B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.50
0.60
0.70
0.80
0.90COVERAGE
cnndm, LLAMA-3-8B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.22
0.24
0.26
0.28
0.30Accuracy
mathqa-5, LLAMA-7B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.25
0.30
0.35
0.40
0.45Accuracy
openbookqa-5, LLAMA-7B
Full
KeepKV
H2O
CaM
Streaming
100
 101
 102
KV Cache Budget (%)
0.20
0.23
0.25
0.28
0.30Accuracy
mathqa-5, LLAMA-2-7B
Full
KeepKV
H2O
CaM
Streaming
Figure 4: Performance of KeepKV and other methods for LLama backbones on HELM and LM-Eval evaluations.
Methods Batch Size Throughput (tokens/s)
Full cache 2 116.54
H2O 8 317.33
D2O 8 214.8
KeepKV 8 255.99
Table 1: Throughput comparison of KeepKV and other meth-
ods (4k context, 20% compression ratio).
4.3 Accuracy on Long-context Tasks
We evaluate KeepKV on the LongBench across Llama and
Mistral model families,including Llama-2-7B, Llama-2-13B,
Llama-3-8B and Mistral-7B, as shown in Table 2. The evalu-
ation tasks include Single-Document QA, Multi-Document
QA, Summarization, Synthetic, and Code. The results indi-
cate that KeepKV achieves performance closer to the full-
cache baseline on most tasks, maintaining high generation
quality despite limited cache availability. Notably, KeepKV
significantly outperforms eviction-based methods, such as
Local Window, StreamingLLM (Xiao et al. 2024b), and H2O
(Zhang et al. 2023). Furthermore, KeepKV also surpasses
existing KV-cache merging methods, like CaM(Zhang et al.
2024) and D2O(Wan et al. 2024b), underscoring the effective-
ness of our carefully designed merging strategy in enhancing
output accuracy.
4.4 Throughput Analysis
Our experiments demonstrate that KeepKV significantly en-
hances the inference throughput of the model by efficiently
compressing the KV Cache, as illustrated in Table 1. We
conducted experiments on the Llama-2-7B model using an
A100-80G GPU, with tasks derived from the LongBench
evaluation framework. The experimental results indicate that
various compression techniques improve throughput by re-
ducing cache size and increasing batch size. Compared to
the original full-cache method, KeepKV achieved over 2×
increase in throughput. It is noteworthy that, due to the addi-
tional computations, the throughput per request of merging
methods is typically lower than that of classical eviction meth-
ods, such as H2O (Zhang et al. 2023). Nonetheless, KeepKV
achieves higher throughput than the state-of-the-art (SOTA)
merging-based algorithm, D2O (Wan et al. 2024b). This ad-
vantage arises because D2O computes attention distribution
variance for real-time cache allocation, whereas our method
adopts a fixed cache allocation strategy to emphasizing the
generalizability of KeepKV.
4.5 Ablation Study
To evaluate the generalizability of KeepKV, we conducted
ablation experiments combining KeepKV with existing state-
of-the-art eviction methods. Since KeepKV does not impose
specific requirements on cache allocation or eviction/preser-
vation strategies, it can be directly integrated with commonly
used eviction/preservation policies. This only requires setting
the merged parties as the eviction and preservation sets deter-

<!-- page 7 -->

Methods Single-Doc QA Multi-Doc QA Summarization Synthetic Code
NrtvQA Qasper MF-en HotpotQA 2WikiMQA Musique TREC TriviaQA SAMSum PCount PRe Lcc RB-P
Llama-2-7B
Full Model 15.8 9.39 22.09 8.56 10.85 4.3 65.0 89.64 34.16 1.0 8.29 66.77 60.1
Local Window 2.22 9.29 1.83 5.14 7.18 1.02 17.5 4.07 3.17 1.5 2.58 16.31 15.35
StreamingLLM 11.81 5.18 19.26 7.07 10.48 3.71 55.5 87.31 31.84 1.5 4.29 63.79 56.07
H2O 16.54 7.57 20.61 7.68 9.28 4.09 64.087.98 33.62 1.34 9.14 65.34 58.49
CaM 11.79 5.1 19.12 7.2610.483.64 56.0 87.31 31.85 1.5 4.29 63.66 55.98
D2O 16.04 6.54 19.48 8.14 10.12 4.62 63.5 88.3934.1 1.39 7.54 65.859.44
Ours 17.32 7.48 22.2 8.519.724.65 60.588.8733.2 2.23 8.45 65.956.36
Llama-2-13B
Full Model 12.64 8.61 19.82 9.1 10.98 5.8 69.5 87.04 41.89 2.0 6.03 67.08 57.53
Local 4.95 5.11 3.82 7.05 9.87 3.42 19.0 7.83 2.63 1.17 6.51 16.7 14.65
StreamingLLM 5.04 5.75 12.24 9.4 10.47 4.71 57.0 82.48 37.21 1.5 5.04 61.47 50.84
H2O 13.836.41 15.52 9.04 9.55 5.53 66.0 86.08 40.2 2.887.37 64.52 55.46
CaM 5.16 5.95 12.31 9.19 10.52 4.66 57.0 82.48 37.28 2.5 5.25 61.75 50.71
D2O 12.76 6.53 14.87 8.59 10.34 5.75 66.586.5240.52 2.0 6.99 65.2355.84
Ours 12.096.89 17.81 9.49 10.54 5.79 66.882.7241.35 1.757.55 64.8156.29
Llama-3-8B
Full Model 14.34 13.68 21.7 9.42 10.75 6.99 72 90.7 45.13 3.74 6.72 70.54 66.04
Local 2.14 6.69 5.17 6.16 5.0 2.42 34.25 30.5 10.66 2.36 2.0 28.91 24.52
StreamingLLM 10.43 7.84 13.85 9.18 10.44 5.47 61.0 90.37 44.35 2.6 10.5 68.49 63.94
H2O 13.7310.02 17.2 9.31 10.62 6.42 63.3 90.44 45.02 3.29 7.56 68.95 63.84
CaM 10.43 7.83 13.89 9.11 10.37 5.47 61.0 90.37 44.31 3.16 10.5 68.59 64.04
D2O 13.5 8.86 17.21 9.16 10.52 6.35 65.5 90.5244.64 3.44 5.8 68.49 64.84
Ours 12.7610.63 18.57 9.37 10.72 6.53 64.5 90.3345.2 3.54 7.16 69.05 65.68
Mistral-7B
Full Model 22.92 39.74 51.46 43.28 39.46 25.59 74.0 88.64 46.97 4.0 63.5 61.42 58.72
Local 16.89 16.92 21.11 23.33 22.49 10.23 58.5 81.29 36.3 2.17.71 41.1 47.88
StreamingLLM 16.76 17.28 21.41 24.16 22.54 10.72 60.3 82.21 37.43 2.14 7.67 51.19 47.94
H2O 18.06 16.75 22.28 24.77 21.68 8.86 61.0 83.03 30.34 2.15 5.76 56.5 49.88
CaM 16.46 17.26 21.4 25.66 22.5410.72 59.17 82.21 37.33 2.14 7.67 51.01 47.89
D2O 18.5815.92 21.71 26.41 21.68 9.07 61.5 83.12 39.5 2.18 7.3 57.51 50.59
Ours 18.1617.95 22.93 26.56 23.189.42 62 83.47 39.7 2.197.26 58.9 50.71
Table 2: Performance evaluation of KeepKV on various models in LongBench benchmarks (20% compression ratio).
mined by their respective algorithms; it can also be applied
with various cache allocation strategies, simply by modifying
the cache configurations between layers and attention heads.
As shown in Figure 5, we combined KeepKV with existing
mainstream eviction methods, H2O(Zhang et al. 2023) and
PyramidInfer(Yang et al. 2024a), using the HELM evaluation
framework. The results demonstrate that, with KeepKV incor-
porated, the methods outperform the original ones across all
compression ratios. This proves that our algorithm is highly
scalable and versatile, capable of being integrated with vari-
ous eviction schemes to enhance their compression efficiency
and generation quality.
5 Conclusion
In this paper, we conduct a comprehensive analysis of the
impact of KV cache compression on attention computation
and propose KeepKV, which introduces the Electoral V otes
mechanism and Zero Inference-Perturbation Merging to adap-
tively and dynamically merge the KV cache while minimiz-
ing output disturbance. KeepKV effectively preserves more
information within limited memory, significantly mitigating
100
 101
 102
KV Cache Budget (%)
0.00
0.10
0.20ROUGE-L
xsum, LLAMA-7B
Full
H2O
H2O w.KeepKv
(a) Combining with H2O.
100
 101
 102
KV Cache Budget (%)
0.05
0.10ROUGE-2
xsum, LLAMA-7B
Full
Pyramid
Pyramid w.KeepKV (b) Combining with Pyramid.
Figure 5: Accuracy experiments combining KeepKV with
existing eviction methods.
the adverse effects of KV cache compression on generation
quality. Our experiments demonstrate that KeepKV achieves
performance closest to that of the full cache across various
compression ratios. It also excels in both standard and long-
context tasks. We believe KeepKV provides a novel perspec-
tive and a powerful tool for advancing KV cache compression
methods, laying the foundation for efficient LLM inference.

<!-- page 8 -->

Acknowledgements
We are grateful to Chenhong He, Ruijie Miao, Yuhan Wu
and Yanshu Wang from Peking University for their insightful
discussions and helpful suggestions throughout the develop-
ment of this research. We thank ByteDance Ltd. for providing
technical support during the internship period. This work was
supported by the National Key Research and Development
Program of China under Grant No. 2024YFB2906603, by the
Beijing Natural Science Foundation (Grant No. QY25123),
and in part by the National Natural Science Foundation of
China (NSFC) (624B2005).
References
Amini, A.; Gabriel, S.; Lin, P.; Koncel-Kedziorski, R.; Choi,
Y .; and Hajishirzi, H. 2019. MathQA: Towards Interpretable
Math Word Problem Solving with Operation-Based For-
malisms. arXiv:1905.13319.
Bai, Y .; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du,
Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y .; Tang, J.; and Li, J.
2024. LongBench: A Bilingual, Multitask Benchmark for
Long Context Understanding. arXiv:2308.14508.
Busbridge, D.; Ramapuram, J.; Ablin, P.; Likhomanenko, T.;
Dhekane, E. G.; Suau, X.; and Webb, R. 2023. How to Scale
Your EMA. arXiv:2307.13813.
Cai, Z.; Zhang, Y .; Gao, B.; Liu, Y .; Liu, T.; Lu, K.; Xiong,
W.; Dong, Y .; Chang, B.; Hu, J.; and Xiao, W. 2024. Pyra-
midKV: Dynamic KV Cache Compression based on Pyrami-
dal Information Funneling. arXiv:2406.02069.
Chen, Z.; Sadhukhan, R.; Ye, Z.; Zhou, Y .; Zhang, J.; Nolte,
N.; Tian, Y .; Douze, M.; Bottou, L.; Jia, Z.; and Chen, B. 2024.
MagicPIG: LSH Sampling for Efficient LLM Generation.
arXiv:2410.16179.
Dai, Z.; Yang, Z.; Yang, Y .; Carbonell, J. G.; Le, Q. V .; and
Salakhutdinov, R. 2019. Transformer-XL: Attentive Lan-
guage Models beyond a Fixed-Length Context. InAnnual
Meeting of the Association for Computational Linguistics.
Dong, S.; Cheng, W.; Qin, J.; and Wang, W. 2024.
QAQ: Quality Adaptive Quantization for LLM KV Cache.
arXiv:2403.04643.
Feng, Y .; Lv, J.; Cao, Y .; Xie, X.; and Zhou, S. K. 2024. Ada-
KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference. arXiv:2407.11550.
Fu, Y .; Cai, Z.; Asi, A.; Xiong, W.; Dong, Y .; and Xiao, W.
2024. Not All Heads Matter: A Head-Level KV Cache Com-
pression Method with Integrated Retrieval and Reasoning.
arXiv:2410.19258.
Gao, L.; Tow, J.; Abbasi, B.; Biderman, S.; Black, S.; DiPofi,
A.; Foster, C.; Golding, L.; Hsu, J.; Le Noac’h, A.; Li, H.; Mc-
Donell, K.; Muennighoff, N.; Ociepa, C.; Phang, J.; Reynolds,
L.; Schoelkopf, H.; Skowron, A.; Sutawika, L.; Tang, E.;
Thite, A.; Wang, B.; Wang, K.; and Zou, A. 2024. A frame-
work for few-shot language model evaluation.
Grattafiori, A.; Dubey, A.; et al. 2024. The Llama 3 Herd of
Models. arXiv:2407.21783.
Gu, X.; Pang, T.; Du, C.; Liu, Q.; Zhang, F.; Du, C.; Wang,
Y .; and Lin, M. 2025. When Attention Sink Emerges in
Language Models: An Empirical View. arXiv:2410.10781.
Han, C.; Wang, Q.; Peng, H.; Xiong, W.; Chen, Y .; Ji, H.; and
Wang, S. 2024. LM-Infinite: Zero-Shot Extreme Length Gen-
eralization for Large Language Models. arXiv:2308.16137.
Hooper, C.; Kim, S.; Mohammadzadeh, H.; Mahoney, M. W.;
Shao, Y . S.; Keutzer, K.; and Gholami, A. 2024. KVQuant:
Towards 10 Million Context Length LLM Inference with KV
Cache Quantization. arXiv:2401.18079.
Hunter, J. S. 1986. The exponentially weighted moving
average.Journal of quality technology, 18(4): 203–210.
Jiang, A. Q.; Sablayrolles, A.; Mensch, A.; Bamford, C.;
Chaplot, D. S.; de las Casas, D.; Bressand, F.; Lengyel, G.;
Lample, G.; Saulnier, L.; Lavaud, L. R.; Lachaux, M.-A.;
Stock, P.; Scao, T. L.; Lavril, T.; Wang, T.; Lacroix, T.; and
Sayed, W. E. 2023. Mistral 7B. arXiv:2310.06825.
Kang, H.; Zhang, Q.; Kundu, S.; Jeong, G.; Liu, Z.; Krishna,
T.; and Zhao, T. 2024. GEAR: An Efficient KV Cache Com-
pression Recipe for Near-Lossless Generative Inference of
LLM. arXiv:2403.05527.
Kwon, W.; Li, Z.; Zhuang, S.; Sheng, Y .; Zheng, L.; Yu,
C. H.; Gonzalez, J.; Zhang, H.; and Stoica, I. 2023. Efficient
Memory Management for Large Language Model Serving
with PagedAttention. arXiv:2307.08999.
Li, Y .; Huang, Y .; Yang, B.; Venkitesh, B.; Locatelli, A.;
Ye, H.; Cai, T.; Lewis, P.; and Chen, D. 2024. SnapKV:
LLM Knows What You are Looking for Before Generation.
arXiv:2404.14469.
Liu, Z.; Yuan, J.; Jin, H.; Zhong, S.; Xu, Z.; Braverman, V .;
Chen, B.; and Hu, X. 2024. ScissorHands: Exploiting the
Persistence of Importance Hypothesis for LLM KV Cache
Compression at Test Time. arXiv:2402.02750.
Mihaylov, T.; Clark, P.; Khot, T.; and Sabharwal, A. 2018.
Can a Suit of Armor Conduct Electricity? A New Dataset for
Open Book Question Answering. arXiv:1809.02789.
Nallapati, R.; Zhou, B.; dos santos, C. N.; Gulcehre,
C.; and Xiang, B. 2016. Abstractive Text Summa-
rization Using Sequence-to-Sequence RNNs and Beyond.
arXiv:1602.06023.
Narayan, S.; Cohen, S. B.; and Lapata, M. 2018. Don’t
Give Me the Details, Just the Summary! Topic-Aware Con-
volutional Neural Networks for Extreme Summarization.
arXiv:1808.08745.
Nawrot, P.; Ła´ncucki, A.; Chochowski, M.; Tarjan, D.;
and Ponti, E. M. 2024. Dynamic Memory Com-
pression: Retrofitting LLMs for Accelerated Inference.
arXiv:2403.09636.
OpenAI; Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.;
Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.;
Altman, S.; Anadkat, S.; et al. 2024. GPT-4 Technical Report.
arXiv:2303.08774.
Rae, J. W.; Potapenko, A.; Jayakumar, S. M.; and Lillicrap,
T. P. 2019. Compressive Transformers for Long-Range Se-
quence Modelling.ArXiv, abs/1911.05507.
Reid, S.; and Zhu, K. 2024. On the Efficacy of Eviction Pol-
icy for Key-Value Constrained Generative Language Model
Inference. arXiv:2402.06262.

<!-- page 9 -->

Rozi`ere, B.; Gehring, J.; Gloeckle, F.; Sootla, S.; Gat, I.; Tan,
X. E.; Adi, Y .; Liu, J.; Sauvestre, R.; Remez, T.; Rapin, J.;
Kozhevnikov, A.; Evtimov, I.; Bitton, J.; Bhatt, M.; Ferrer,
C. C.; Grattafiori, A.; Xiong, W.; D ´efossez, A.; Copet, J.;
Azhar, F.; Touvron, H.; Martin, L.; Usunier, N.; Scialom,
T.; and Synnaeve, G. 2024. Code Llama: Open Foundation
Models for Code. arXiv:2308.12950.
Shi, L.; Zhang, H.; Yao, Y .; Li, Z.; and Zhao, H. 2024. Keep
the Cost Down: A Review on Methods to Optimize LLM’ s
KV-Cache Consumption. arXiv:2407.18003.
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux,
M.-A.; Lacroix, T.; Rozi `ere, B.; Goyal, N.; Hambro, E.;
Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample,
G. 2023a. LLaMA: Open and Efficient Foundation Language
Models.arXiv preprint arXiv:2302.13971.
Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.;
Babaei, Y .; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale,
S.; et al. 2023b. Llama 2: Open Foundation and Fine-Tuned
Chat Models. arXiv:2307.09288.
Vaswani, A.; Shazeer, N. M.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. At-
tention is All you Need. InNeural Information Processing
Systems.
Wan, Z.; Wang, X.; Liu, C.; Alam, S.; Zheng, Y .; Liu, J.; Qu,
Z.; Yan, S.; Zhu, Y .; Zhang, Q.; Chowdhury, M.; and Zhang,
M. 2024a. Efficient Large Language Models: A Survey.
arXiv:2312.03863.
Wan, Z.; Wu, X.; Zhang, Y .; Xin, Y .; Tao, C.; Zhu, Z.; Wang,
X.; Luo, S.; Xiong, J.; and Zhang, M. 2024b. D2O: Dynamic
Discriminative Operations for Efficient Generative Inference
of Large Language Models. arXiv:2406.13035.
Wang, Z.; Jin, B.; Yu, Z.; and Zhang, M. 2024. Model Tells
You Where to Merge: Adaptive KV Cache Merging for LLMs
on Long-Context Tasks. arXiv:2407.08454.
Wolf, T.; Debut, L.; Sanh, V .; Chaumond, J.; Delangue, C.;
Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; et al.
2020. HuggingFace’s Transformers: State-of-the-art Natural
Language Processing. arXiv:1910.03771.
Xiao, G.; Tang, J.; Zuo, J.; Guo, J.; Yang, S.; Tang, H.;
Fu, Y .; and Han, S. 2024a. DuoAttention: Efficient Long-
Context LLM Inference with Retrieval and Streaming Heads.
arXiv:2410.10819.
Xiao, G.; Tian, Y .; Chen, B.; Han, S.; and Lewis, M. 2024b.
Efficient Streaming Language Models with Attention Sinks.
arXiv:2309.17453.
Yang, D.; Han, X.; Gao, Y .; Hu, Y .; Zhang, S.; and Zhao, H.
2024a. PyramidInfer: Pyramid KV Cache Compression for
High-Throughput LLM Inference. arXiv:2406.02069.
Yang, J. Y .; Kim, B.; Bae, J.; Kwon, B.; Park, G.; Yang, E.;
Kwon, S. J.; and Lee, D. 2024b. No Token Left Behind: Re-
liable KV Cache Compression via Importance-Aware Mixed
Precision Quantization. arXiv:2402.18096.
Zhang, Y .; Du, Y .; Luo, G.; Zhong, Y .; Zhang, Z.; Liu, S.;
and Ji, R. 2024. CaM: Cache Merging for Memory-efficient
LLMs Inference. InForty-first International Conference on
Machine Learning.
Zhang, Z.; Sheng, Y .; Zhou, T.; Chen, T.; Zheng, L.; Cai,
R.; Song, Z.; Tian, Y .; R´e, C.; Barrett, C.; et al. 2023. H2o:
Heavy-hitter oracle for efficient generative inference of large
language models.Advances in Neural Information Process-
ing Systems, 36: 34661–34710.
A Theoretical Analysis
Recently, many studies have analyzed KV cache compression
strategies in LLM inference from a theoretical perspective
(Zhang et al. 2023; Liu et al. 2024; Li et al. 2024; Yang
et al. 2024a; Zhang et al. 2024; Wan et al. 2024b; Wang
et al. 2024; Gu et al. 2025). Overall, the primary objective
of most existing works can be summarized as minimizing
the impact of compression on the output. For instance, exist-
ing eviction-based methods and cache allocation strategies
(Zhang et al. 2023; Liu et al. 2024; Reid and Zhu 2024; Li
et al. 2024; Yang et al. 2024a; Cai et al. 2024; Feng et al.
2024; Fu et al. 2024; Xiao et al. 2024a) all aim to maxi-
mize the retention of essential information within limited
memory by evicting less important tokens or reducing cache
allocation in non-critical heads and layers based on empiri-
cal observations of attention distributions. However, eviction
inevitably leads to irreversible information loss, which has
motivated the development of KV cache merging methods
(Zhang et al. 2024; Wan et al. 2024b; Wang et al. 2024;
Nawrot et al. 2024). Despite this, key challenges such as
the selection of merging candidates and the assignment of
merging weights remain largely unexplored, with a lack of
systematic theoretical foundations. In this work, we introduce
a novel perspective distinct from prior approaches. We formu-
late the problem as eliminating output perturbation and derive
a novel merging method by analyzing the attention computa-
tion process. First, we introduce Electoral V otes mechanism,
making the elimination of output perturbation feasible. Then,
we derive a merging computation formula to eliminate pertur-
bation at the current step. Finally, we extend this framework
to multi-step generation, providing a theoretical guarantee for
output perturbation and offering a reasonable explanation for
mainstream similarity-based merging candidates selection
methods.
Specifically, we first demonstrate the unavoidable output
perturbation caused by KV cache eviction. Next, we dis-
cuss the attention sag issue in existing KV cache merging
methods and provide a formal proof. Then, we present the
derivation process of our KeepKV merging method. Finally,
we provide a theoretical guarantee for the output perturbation
of KeepKV , including proofs for the main theorem and its
associated lemma. The symbolic representation of the atten-
tion computation process remains consistent with the one
introduced in the methodology section.
A.1 Perturbation in KV Cache Eviction
Eviction methods discard KV pairs deemed unimportant. We
denote the first generation step in the decoding phase as the
(L+ 1) -th generation step, where L represents the prompt
length. And for a positive integer n, let [n] :={1,2, ..., n} .
At t-th generation step, let Ke ={e 1e2, ..., em}, m∈[t]

<!-- page 10 -->

denoted the index of to-be-evicted cache. Based on Equation
3, the output after eviction(o ′
t)is:
o′
t =
tX
i=1,i /∈Ke
A
′t
i vi, A
′t
i = st
iPt
i=1,i /∈Ke st
i
.(9)
By transformingo ′
t towardso t, we obtain:
o′
t =
Pt
i=1 st
iPt
i=1 st
i −Pt
j∈Ke st
j
∗
Pt
i=1 st
ivi −Pt
j∈Ke st
jvj
Pt
i=1 st
i
=
Pt
i=1 st
iPt
i=1 st
i −Pt
j∈Ke st
j

ot −
Pt
j∈Ke st
jvj
Pt
i=1 st
i
!
=
Pt
i=1 At
iPt
i=1 At
i −Pt
j∈Ke At
j

ot −
Pt
j∈Ke At
jvj
Pt
i=1 At
i
!
= 1
1− Pt
j∈Ke At
j

ot −
tX
j∈Ke
At
jvj

 .
(10)
Equation 10 indicates that the difference between o′
t
and ot decreases as the attention score of the evicted KV
(At
j, j∈K e) diminishes. When the total score of Ke be-
comes negligible, the output perturbation at the current step
approaches zero. This formally explains why eviction meth-
ods generally prioritize discarding KV pairs with lower atten-
tion scores.
However, existing studies (Chen et al. 2024) have shown
that attention can be relatively dispersed in certain tasks,
meaning that evicting even a small number of tokens can have
a non-negligible impact. Furthermore, as the compression
ratio increases, evicted tokens will account for a significant
portion of the attention scores, exacerbating the degradation
of generation quality.
A.2 Attention Sag in KV Cache Merging
Merging methods integrate less important KV into others
rather than discarding them directly. Specifically, mainstream
studies select, for each KV pair to be evicted, a merging target
among the preserved KVs, allowing many-to-one merges.
Typically, weighted merging rather than direct averaging is
used, with weights satisfying a normalization constraint, i.e.,
the merged vectors are obtained via convex combinations.
Formally, merging the evicted pairs (kj, vj), j∈K e into a
preserved pair (kc, vc) yields a new KV pair(kr, vr), defined
as follows:
kr =w ckc +
X
j∈Ke
wjkj,
vr =w cvc +
X
j∈Ke
wjvj,
s.t.w c +
X
j∈Ke
wj = 1(11)
Let K ′
e =K e ∪ {c}, representing the index of the original
KVs before merging. For instance, the weight wj in D2O
(Wan et al. 2024b) is computed based on the cosine similarity
between key vectors, whereas for KVmerger (Wang et al.
2024), it is calculated based on the Gaussian Kernel value.
Formally, these are represented as follows:
wjD2O = exp(cosθ kj,kc)P
j∈K ′e
exp(cosθ kj,kc) ,
wjKVMerger =
exp

− ||kj −kc||2
2σ2

P
j∈K ′e
exp

− ||kj −kc||2
2σ2
 .(12)
However, the widely adopted convex combination ap-
proach also introduces output disturbances, as stated in the
following theorem:
Theorem 7(Formal version of Theorem 2).The merging
method indicated by Equation 11 causes the attention score
of the merged KV to become less than the sum of attention
scores from the original multiple KVs merged into it, indepen-
dently of the specific weighting scheme. Formally, this implies:
A′t
r <P
j∈K ′e
At
j, ultimately leading to:∥o ′
t −o t∥>0.
Proof. The attention score and output after merging can be
expressed as:
o′
t =
tX
i=1,i /∈K′e
A′t
ivi +A ′t
rvr =
Pt
i=1,i /∈K′e
st
ivi +s t
rvr
Pt
i=1,i /∈K′e
st
i +s tr
.
(13)
First, we compare the denominators of At and A′t, for-
mally provingPt
i=1,i /∈K′e
st
i +s t
r <Pt
i=1 st
i:
tX
i=1,i /∈K′e
st
i+st
r =
tX
i=1
st
i+st
r −
X
i∈K ′e
st
i =e
qt kr√
d −
X
i∈K ′e
e
qt ki√
d
(14)
Substituting Equation 11, and applying the Weighted
AM–GM Inequality, we have:
e
qt kr√
d −
X
i∈K ′e
e
qt ki√
d =e
qt (P
i∈K′e
wi ki )
√
d −
X
i∈K ′e
e
qt ki√
d
=
Y
i∈K ′e
(e
qt ki√
d )
wi
−
X
i∈K ′e
e
qt ki√
d
≤
X
i∈K ′e
wie
qt ki√
d −
X
i∈K ′e
e
qt ki√
d <0
(15)
Thus,
tX
i=1,i /∈K′e
st
i +s t
r =
tX
i=1
st
i + (st
r −
X
i∈K ′e
st
i)<
tX
i=1
st
i (16)

<!-- page 11 -->

Since the sum of the normalized attention scores equals
one, and given thatPt
i=1,i /∈K′e
st
i+st
r <Pt
i=1 st
i, we obtain:
A′t
r = 1−
Pt
i=1, i /∈K′e
st
i
Pt
i=1, i /∈K′e
st
i +s tr
<1−
Pt
i=1, i /∈K′e
st
i
Pt
i=1 st
i
=
X
i∈K ′e
At
i (17)
Similarly, we can derive:
A′t
j = st
j
Pt
i=1,i /∈K′e
st
i +s tr
> st
j
Pt
i=1 st
i
=A t
j, j̸=r
(18)
Finally, the output perturbation can be represented as:
∥o′
t −o t∥=






(
tX
i=1,i /∈K′e
A
′t
i vi +A ′t
rvr)−
tX
i=1
At
ivi






=






tX
i=1,i /∈K′e
(A
′t
i −A t
i)vi + (A′t
rvr −
X
i∈K ′e
At
ivi)






=






tX
i=1,i /∈K′e
(A
′t
i −A t
i)vi +
X
i∈K ′e
(wiA′t
r −A t
i)vi






(19)
In the above expression, all vector coefficients are nonzero.
Moreover, due to the high dimensionality and sparsity of the
KV cache (Wang et al. 2024; Gu et al. 2025), the vectors are
almost linearly independent. In practical inference scenarios,
it is impossible for them to form a zero vector through linear
combination. Consequently, we have:∥o ′
t −o t∥>0.
We term this phenomenon asAttention Sag, indicating
that improper merging methods result in a reduced attention
score for the newly merged vector, while attention scores
of unmerged KVs relatively increase. This leads to output
disturbances and ultimately degrades generation quality.
A.3 KeepKV Merging Method
In the main text, we introduced the concept of merging count
via the Electoral V otes mechanism, aiming for a KV pair with
vote count pi to be equivalent, in attention computation, to
pi independent occurrences of this KV . Moreover, the vote
count of the merged KV equals the sum of vote counts before
merging. Formally, the outputs before (ot) and after merging
(o′
t)can be expressed as follows:
ot =
Pt
i=1 pist
ivi
Pt
i=1 pist
i
,
o′
t =
Pt
i=1, i /∈K′e
pist
ivi +p rst
rvr
Pt
i=1, i /∈K′e
pist
i +p rstr
,
pr =
X
i∈K ′e
pi.(20)
Next, we demonstrate how our new merging approach can
be derived naturally from the objective of eliminating output
disturbances, which consequently serves as a direct proof for
Theorem 3.
Based on Equation 20, setting∥o ′
t −o t∥= 0, we obtain:
tX
i=1
pist
ivi =
tX
i=1, i /∈K′e
pist
ivi +p rst
rvr,
tX
i=1
pist
i =
tX
i=1, i /∈K′e
pist
i +p rst
r.(21)
which implies:
X
i∈K ′e
pist
ivi =p rst
rvr,
X
i∈K ′e
pist
i =p rst
r (22)
Dividing the two expressions above, we obtain the expres-
sion ofv r:
vr =
P
i∈K ′
e
pist
ivi
P
i∈K ′e
pist
i
(23)
Similarly, let kr =C( P
i∈K ′e
pist
iki), substituting this
intoPt
i∈K ′e
pist
i =p rst
r from Equation 22 and solving, we
obtain:
C=
ln
P
i∈K′e
pist
iP
i∈K′e
pi
P
i∈K ′e
pist
i lns t
i
(24)
Finally, we derive the merging expression:
kr =
P
i∈K ′e
pist
iki

ln
 P
i∈K′e
pist
iP
i∈K′e
pi

P
i∈K ′e
pist
i lns t
i
,
vr =
P
i∈K ′e
pist
ivi
P
i∈K ′e
pist
i
, p r =
X
i∈K ′e
pi.(25)
Consequently, merging in this manner eliminates the out-
put disturbance in the t-step, satisfying: ∥o′
t −o t∥= 0 . By
setting the merging candidates K ′
e ={e} ∪ {c} , we obtain
Theorem 3.
A.4 Error Bound Analysis
After extending KeepKV to multi-step generation, for t′-step,
all st
i terms in Equation 25 are replaced with ˆst′
i , which
represents our estimation of future attention score trends
obtained through a certain method. In this case, the merging
expressions become:
kr =
P
i∈K ′e
piˆst′
i ki

ln
 P
i∈K′e
pi ˆst′
iP
i∈K′e
pi

P
i∈K ′e
piˆst′
i ln ˆst′
i
,
vr =
P
i∈K ′e
piˆst′
i vi
P
i∈K ′e
piˆst′
i
, p r =
X
i∈K ′e
pi.(26)

<!-- page 12 -->

and they satisfy:
X
i∈K ′e
pi ˆst′
i vi =p r ˆstrvr,
X
i∈K ′e
pi ˆst
i =p r ˆstr (27)
Then the perturbation at steps ′, can be expressed as:
Θt′ =∥o t′ −o ′
t′∥
=






Pt′
i=1 pist′
i vi
Pt′
i=1 pist′
i
−
Pt′
i=1, i /∈K′e
pist′
i vi +p rst′
r vr
Pt′
i=1, i /∈K′e
pist′
i +p rst′
r






=



Pt′
i=1 pist′
i
P
j∈K ′e
pjst′
j (vj −v i)−p rst′
r (vr −v i)



Pt′
i=1 pist′
i
Pt′
i=1, i /∈K′e
pist′
i +p rst′
r

(28)
Substituting the expression for vr from Equation 26 andP
j∈K ′e
pj ˆst′
j =p r ˆst′
r from Equation 27 into the above:
t′
X
i=1
pist′
i

X
j∈K ′e
pjst′
j (vj −v i)−p rst′
r
 P
k∈K ′e
pkˆst′
k vk
P
k∈K ′e
pkˆst′
k
−v i
!

=
t′
X
i=1
pist′
i
X
j∈K ′e
pjst′
j

1− st′
r
ˆst′
r
· ˆst′
j
st′
j
!
(vj −v i)(29)
Let
1−
ˆst′
i
st′
i
 ≤ϵ, ϵ <1, then1−ϵ≤
ˆst′
i
st′
i
≤1 +ϵ, thus:
1− st′
r
ˆst′
r
ˆst′
j
st′
j
 =

ˆst′
r
st′
r
−
ˆst′
j
st′
j
ˆst′
r
st′
r

≤ 2ϵ
1−ϵ , j∈K ′
e (30)
Let ∥vj −v i∥ ≤γ,∀i∈[t ′], j∈K ′
e, where γ represents
the inherent variation in the input, which cannot be elimi-
nated through algorithmic design. Then, applying the triangle
inequality, we obtain:






t′
X
i=1
pist′
i

X
j∈K ′e
pjst′
j

1− st′
r
ˆst′
r
·
ˆst′
j
st′
j
!
(vj −v i)








≤
t′
X
i=1
pist′
i

X
j∈K ′e
pjst′
j
1− st′
r
ˆst′
r
·
ˆst′
j
st′
j
 · ∥vj −v i∥


≤ 2ϵγ
1−ϵ


t′
X
i=1
pist′
i



X
j∈K ′e
pjst′
j

 (31)
Substituting this inequality into Equation 28, we obtain:
Θt′ ≤ 2ϵγ
1−ϵ
(Pt′
i=1 pist′
i )(P
j∈K ′e
pjst′
j )
(Pt′
i=1 pist′
i )(Pt′
i=1,i /∈K′e
pist′
i +p rst′
r )
= 2ϵγ
1−ϵ
P
j∈K ′e
pjst′
j
Pt′
i=1,i /∈K′e
pist′
i +p rst′
r
< 2ϵγ
1−ϵ
P
j∈K ′e
pjst′
j
prst′
r
(32)
Due to Equation 27, we have
P
j∈K ′e
pj
ˆst′
j
pr
ˆst′
r
= 1, then:
P
j∈K ′e
pjst′
j
prst′
r
≤
1
1−ϵ
P
j∈K ′e
pj ˆst′
j
1
1+ϵ pr ˆst′
r
= 1 +ϵ
1−ϵ (33)
Thus,
Θt′ < 2ϵγ
1−ϵ
P
j∈K ′e
pjst′
j
prst′
r
< 2ϵ(1 +ϵ)γ
(1−ϵ) 2
(34)
Finally, we obtain the following theorem:
Theorem 8.For the t′-th step, let
1−
ˆst′
i
st′
i
 ≤ϵ, ϵ <1 , the
output perturbation satisfies Θt′ < 2ϵ(1+ϵ)γ
(1−ϵ)2 , provided that
∥vj −v i∥ ≤γ,∀i∈[t ′], j∈K ′
e.
Next, we prove the following lemma:
Lemma 9.As the prediction error ϵ decreases and the
merged candidates become increasingly similar, the output
disturbance reduces to zero. That is, when either ϵ= 0 or
(ki, vi) = (kj, vj),∀i, j∈K ′
e, we have:Θ t′ = 0.
Proof. By Theorem 8, it is easy to obtain that when ϵ= 0 ,
Θt′ < 2ϵ(1+ϵ)γ
(1−ϵ)2 = 0. Next, we prove that when (ki, vi) =
(kj, vj),∀i, j∈K ′
e, it also holds that Θt′ = 0. First, we fur-
ther expand (1− st′
r
ˆst′
r
ˆst′
j
st′
j
), j∈K ′
e in Equation 29 by applying
Equation 27:
1− st′
r
ˆst′
r
ˆst′
j
st′
j
= 1− e
qt′ kr√
d
P
i∈K′e
pi
ˆst′
iP
i∈K′e
pi
ˆst′
j
st′
j
, j∈K ′
e (35)
Substituting the expression for kr from Equation 26 into
the above:
1− e
qt′ kr√
d
P
i∈K′e
pi ˆst′
iP
i∈K′e
pi
· ˆst′
j
st′
j
= 1−
Q
i∈K ′e
(st′
i )pi ˆst′
i

ln


P
i∈K′e
pi ˆst′
iP
i∈K′e
pi


P
i∈K′e
pi ˆst′
i ln ˆst′
i
P
i∈K′e
pi ˆst′
iP
i∈K′e
pi
· ˆst′
j
st′
j
(36)

<!-- page 13 -->

When (ki, vi) = (kj, vj),∀i, j∈K ′
e, it follows that ∀i∈
K ′
e, st′
i =s t′
, ˆst′
i = ˆst′
, thereby:
1−
Q
i∈K ′e
st′
i
pi ˆst′
i

ln


P
i∈K′e
pi ˆst′
iP
i∈K′e
pi


P
i∈K′e
pi ˆst′
i ln ˆst′
i
P
i∈K′e
pi ˆst′
iP
i∈K′e
pi
· ˆst′
j
st′
j
= 1−
Q
i∈K ′e
st′ pi ˆst′
ln


P
i∈K′e
pi ˆst′
P
i∈K′e
pi


P
i∈K′e
pi ˆst′ ln ˆst′
P
i∈K′e
pi ˆst′
P
i∈K′e
pi
· ˆst′
st′
= 1− st′
ˆst′ · ˆst′
st′ = 0(37)
Under this condition, it follows that Equation 29 equals 0,
i.e.,
t′
X
i=1
pist′
i

X
j∈K ′e
pjst′
j (vj −v i)−p rst′
r (vr −v i)


=
t′
X
i=1
pist′
i

X
j∈K ′
e
pjst′
j

1− st′
r
ˆst′
r
· ˆst′
j
st′
j
!
(vj −v i)

 = 0
(38)
Finally, Substituting it into Equation 28, we obtain Θt′ =
0.
Remark 10.This lemma provides a theoretical justification
for prior merging strategies favoring high-similarity KV pairs
(Wan et al. 2024b; Wang et al. 2024). Meanwhile, we offer
an intuitive interpretation: if the merged two KV pairs are
identical, i.e., (ke, ve) = (kc, vc), retaining one pair and set-
ting its Electoral Votes to 2 introduces no error in subsequent
computations.
In this section, we have proven Theorem 8 and Lemma
9, which provide guarantees on the output perturbation in
multi-step generation—an aspect that existing methods strug-
gle to achieve. Moreover, our method demonstrates superior
performance across various experimental evaluations. How-
ever, it should be acknowledged that predicting attention
distributions further into the future is inherently challenging,
leading to a significant increase in the estimated perturbation
upper bound. Furthermore, the inherent input differences γ
cannot be ignored, representing a fundamental problem in
KV cache compression—namely, the inability to perfectly
compress the KV cache into a smaller memory without any
loss of information. Nevertheless, our work introduces a new
perspective and analytical approach to studying KV cache
eviction and merging algorithms, which we hope will inspire
future research.
B Implementation Details
B.1 Models and benchmarks
Across all experiments, we utilized pre-trained model weights
from Huggingface (Wolf et al. 2020). Specifically, for Llama
architectures: we used the ’huggyllama/llama-7b’ checkpoint
for Llama-1-7B (Touvron et al. 2023a), ’meta-llama/Llama-
2-7b-hf’ for Llama-2-7B (Touvron et al. 2023b), and ’meta-
llama/Llama-2-13b-hf’ for Llama-2-13B. For Llama-3-8B,
the ’meta-llama/Meta-Llama-3-8B-Instruct’ checkpoint was
used in the HELM evaluations, and ’meta-llama/Meta-Llama-
3-8B’ in LongBench evaluations. Regarding the Mistral archi-
tecture, we employed the ’mistralai/Mistral-7B-Instruct-v0.3’
model (Jiang et al. 2023). For detailed information on the
LongBench benchmark, please refer to its official repository
(Bai et al. 2024).
B.2 Experimental Setup
Our study does not involve model training, thus no data pre-
processing is required. All evaluation datasets are sourced
from the publicly available lm-eval-harness (Gao et al. 2024),
HELM (Narayan, Cohen, and Lapata 2018; Nallapati et al.
2016) and LongBench (Bai et al. 2024) benchmarks, and
we follow their original evaluation metrics. Our implementa-
tion is primarily based on modifications to the open-source
H2O and D2O codebases (Zhang et al. 2023; Wan et al.
2024b), both of which are publicly accessible. All open-
source datasets, evaluation frameworks, and algorithmic im-
plementations used in this work are employed in full compli-
ance with their respective licenses and terms of use.
B.3 Details of Parameter Settings
For cache allocation, we adopt the default configuration pro-
vided by the open-source PyramidInfer (Yang et al. 2024a)
implementation. Specifically, we follow a cosine-based de-
cay strategy for distributing cache across layers. Within each
layer, the ratio between the recent window and the heavy
hitter (i.e., crucial KV entries) varies between 5:1 and 4:1.
Additionally, we retain the first 4 tokens (referred to as ’sink
tokens’ in StreamingLLM (Xiao et al. 2024b)) in the cache
throughout. For algorithmic hyperparameters, we adjust the
merging threshold T within the range of 0.6 to 0.95, and
empirically select 0.8 as the default value based on its stable
performance across benchmarks. The corresponding experi-
mental results are reported below.
Cache Budget 1% 10%
Evict-first 0.051 0.111
Merge-first 0.059 0.115
Table 3: Performance comparison with different merge policy
during the prefill stage (ROUGE-2; LLAMA-7B).
Prefill’s Merge PolicyThe results in Table 3 show that,
during the prefill stage, performing merging before applying
the eviction strategy—rather than merging after eviction as
is commonly done—can improve generation quality. This is
because, according to Lemma 6, merging more similar items

<!-- page 14 -->

induces smaller perturbations. In the prefill stage, there exist
more tokens whose similarity exceeds the threshold; merging
them first allows these tokens to be consolidated rather than
evicted unnecessarily, thereby preserving more information
within the limited-size cache.
Merging Thres XSUM
0.7 0.223
0.8 0.233
0.9 0.214
Table 4: Merging threshold impact (4% compression ratio).
Merging ThresholdWe investigate the impact of different
merging thresholds on performance. The results in Table 4
indicate that setting the merging threshold (i.e., the cosine
similarity between key vectors) to 0.8 yields better perfor-
mance, and its robustness has been further validated across a
broader range of experiments.
C Additional Discussion
C.1 Limitations and Future Work
As analyzed in Section A, the bound on output perturbation
in multi-step generation depends on two quantities: the in-
trinsic difference of the inputs γ, and the prediction error ϵ.
The former is inherent and cannot be influenced or altered,
whereas the latter is inevitably non-negligible in realistic in-
ference scenarios. In practice, accurately predicting future
attention scores is highly challenging. Prior work typically
employs sliding-window averages or weighted averages as
predictors, while more recent studies have begun to explore
machine-learning-based approaches to obtain more accurate
predictions; this appears to be a promising direction for future
research.
Moreover, the merging operation inevitably incurs addi-
tional computational overhead, which can render it less fa-
vorable than purely eviction-based strategies with simpler
computation in certain settings. Our mathematically derived
lossless KV cache merging method, along with many other
related algorithms, further relies on access to attention scores,
which limits its compatibility with widely adopted inference
acceleration techniques such as FlashAttention. Future work
could therefore focus on designing KV cache eviction and
merging algorithms that are amenable to integration with
FlashAttention and related acceleration methods.
C.2 Comparison with CaM in Eliminating Output
Perturbations
A closely related line of work is CaM (Zhang et al. 2024),
which also analyzes the effect of KV cache eviction and
its proposed merging algorithm on output perturbation, and
aims to reduce or even eliminate such perturbation. However,
there are two fundamental differences between CaM and
our approach: (i) how ”output perturbation” is defined and
modeled, and (ii) how the theoretical analysis is connected
to the asynchronous KV cache updates that actually occur in
autoregressive decoding.
Before presenting our analysis, it is important to clarify the
implementation setting shared by both CaM and our method.
In practical autoregressive decoding, KV compression is al-
ways performed inside the decoding loop: the model first runs
the forward pass at step t using the current KV cache and
produces the output, and only afterwards applies eviction or
merging operations to the cache state that will be consumed
at step t+1 and beyond. In other words, cache compression is
asynchronous with respect to the forward computation. Con-
sequently, when we design an algorithm from the perspective
of eliminating output perturbation, our first requirement is
consistency at the current step: if we hypothetically recom-
pute the step-t forward pass on the compressed KV cache, the
resulting output should match that obtained from the original,
uncompressed cache. Intuitively, the attention computation
between the step-t query and the pre-compression KV cache
should yield exactly the same result as the attention between
the same query and the post-compression KV cache. This
ensures that, when we ”look back” at step t in the future,
its behavior is indistinguishable from a world in which step
t had always been computed on the compressed KV cache,
without needing to refer to the original KV entries that are
no longer stored. The overall decoding process thus appears
temporally coherent.
In contrast, when CaM studies the effect of eviction and
merging on the output at step t, it does not adopt such a
”recompute on the compressed cache” viewpoint. Instead, it
directly removes (or rescales) the contribution of the evicted
or merged KV pair in the attention scores and in the attention-
weighted output, while leaving the contributions of all re-
maining tokens unchanged. This effectively treats the atten-
tion scores of the other KV pairs as fixed coefficients that
are not affected by modifying the cache. However, due to
the softmax normalization, evicting or merging any KV pair
inevitably changes the normalization term and thus the at-
tention weights of all remaining KV pairs. Capturing this
global effect of softmax is precisely why our analysis starts
from the full attention expression, rather than from a model
in which the other attention weights are held constant. In
summary, CaM and our method formalize and analyze output
perturbation from different perspectives, and each seeks to
reduce or eliminate perturbation under its own definition, but
our framework is explicitly aligned with the asynchronous
KV-update pattern and with the impact of softmax on all
tokens in the cache.
