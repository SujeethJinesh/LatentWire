# references/145_selective_induction_heads_how_transformers_select_causal_structures_in_context.pdf

<!-- page 1 -->

SELECTIVE INDUCTION HEADS: HOW TRANSFORMERS
SELECT CAUSAL STRUCTURES IN CONTEXT
Francesco D’Angelo, Francesco Croce, Nicolas Flammarion
Theory of Machine Learning Lab, EPFL, Lausanne, Switzerland
Abstract.Transformers have exhibited exceptional capabilities in sequence modeling tasks, lever-
aging self-attention and in-context learning. Critical to this success are induction heads, attention
circuits that enable copying tokens based on their previous occurrences. In this work, we introduce a
novel framework that showcases transformers’ ability to dynamically handle causal structures. Existing
works rely on Markov Chains to study the formation of induction heads, revealing how transformers
capture causal dependencies and learn transition probabilities in-context. However, they rely on a fixed
causal structure that fails to capture the complexity of natural languages, where the relationship be-
tween tokens dynamically changes with context. To this end, our framework varies the causal structure
through interleaved Markov chains with different lags while keeping the transition probabilities fixed.
This setting unveils the formation ofSelective Induction Heads, a new circuit that endows transform-
ers with the ability to select the correct causal structure in-context. We empirically demonstrate that
transformers learn this mechanism to predict the next token by identifying the correct lag and copying
the corresponding token from the past. We provide a detailed construction of a 3-layer transformer to
implement the selective induction head, and a theoretical analysis proving that this mechanism asymp-
totically converges to the maximum likelihood solution. Our findings advance the understanding of how
transformers select causal structures, providing new insights into their functioning and interpretability.
1.Introduction
As autoregressive generative models continue to scale and are increasingly deployed in real-world ap-
plications, the question of how Transformer models [Vaswani et al., 2017] function internally becomes
pressing. Yet the inherent complexity of natural language hinders the ability to fully comprehend how
these models make decisions and work internally. To address this challenge, many recent works have
attempted to formulate synthetic frameworks that simplify the problem and enable theoretical analysis
while still capturing the remarkable properties and phenomena observed in large language models, such
as in-context learning [Brown, 2020, Garg et al., 2022, Bai et al., 2023, Von Oswald et al., 2023, Sander
et al., 2024, Chan et al., 2022]. Mechanistic interpretability [Olsson et al., 2022] emerges as a line of
research focused on reverse-engineering the complex computations performed inside a transformer in
order to understand how a certain output is produced for a given input. This research has uncovered
the formation of induction heads [Olsson et al., 2022] i.e., interpretable circuits embedded within the
transformer’s weights, capable of simple operations such as copying tokens. By examining such cir-
cuits and their combinations, one can understand the algorithms that transformers implement to solve
a given task. For instance, [Nichani et al., 2024, Bietti et al., 2023, Edelman et al., 2024] demonstrated
that induction heads enable transformers to implement in-context bigrams for next-token predictions
in Markov Chains. Such mechanisms are not limited to simplified models: Nguyen [2024] showed that
transformers may rely on N-gram rules even in natural language processing. Yet the process by which
transformers select between such learned rules remains poorly understood. While in-context learning
studies how transformers solve tasks from demonstrations in the prompt,in-context selectionfocuses
on how transformers select the most suitable approach to solve a given task from those encountered
during training, using instances present in the context. For example, Bai et al. [2023] examines how
transformers perform in-context algorithm selection when pre-trained on mixtures of linear and logistic
regression with different noise. Similarly, Yadlowsky et al. [2023] studied the ability of transformers to
perform model selection between different function class families. Our work takes another step toward
understanding this selection process, leveraging Markov chains with different causal structures.
E-mail address:francesco.dangelo@epfl.ch.
Published as a conference paper at ICLR 2025https: // openreview. net/ forum? id= bnJgzAQjWf.
1
arXiv:2509.08184v1  [cs.LG]  9 Sep 2025

<!-- page 2 -->

2
Data:interleaved
Markov Chains
with lag 1 and 2
X1 X2 X3 X4 X5
lagk= 1
lagk= 2
T ask:predict the
next token by se-
lecting in-context
the correct causal
structure
s1 s2 s3 s4 . . . si−2 si−1 si si+1
k= 1?
k= 2?
Input sequencewith true lagk= 2
How do 3-layer attention-only transformers solve this task?
L1: extracts single transition
probabilities for each lag
L2: aggregates transition
probabilities from the past
L3:selective induction head
select the most likely lag
Figure 1.Summary of the framework. Top:We define a new task based on Interleaved Markov Chains of
different lags (k= 1andk= 2in the example).Middle:given a sequence generated from a chain of unknown
lag, the model has to identify the true lag, and use it to predict the distribution of the next token.Bottom:
attention-only transformers can solve this task with 3 layers. The first computes the transition probabilities for
each lag seen during training, the second aggregates these probabilities over the entire past, and finally the third
layer implements the selective induction head, which selects the correct lag.
In-context causal structure selection.Recently, Markov chains have been employed to formulate
interesting sequence-to-sequence tasks that can be solved by transformers with interpretable solutions
[Ildiz et al., 2024, Nichani et al., 2024, Makkuva et al., 2024, Edelman et al., 2024]. In particular,
Nichani et al. [2024] show that transformers trained on Markov Chain sequences learn circuits that
capture the causal structure, i.e., the set of parent tokens for each token in the sequence and estimate
transition probabilities in-context. The existing works relying on Markov chains fail to model the nuanced
relationships typical of natural language. The same word pair can have different causal relationships
depending on the surrounding context. While an effective model should recognize these contextual
dependencies, previous research has overlooked this consideration by adopting fixed causal structures. To
address this limitation, we proposea new synthetic taskdesigned to mimic different causal dependencies
(Sec. 4). We considerInterleaved Markov Chains, with fixed transition probabilities between states
but different underlying causal structures (Fig. 1), and theoretically study how 3-layer attention-only
transformers learn to correctly predict the next token in a sequence.
Selective induction heads.To solve the task at hand—correctly predicting the next token in-context
in a sequence generated within this setup (middle of Fig. 1)—transformers need to learn a circuit that
adapts to the given context to select the correct causal structure among those seen during training. We
call this circuit aselective induction head, as it differs fundamentally from the induction heads introduced
so far in the literature, where the circuit learns either to copy a token from a certain position fixed by the
unique structure of the data or by comparing its semantics. In our task, the transformer (with attention
maps depicted in Fig. 1) needs to learn to aggregate all past information to determine from which past
position the corresponding token should be copied in order to predict the next token.
A transformer construction for in-context selection.To understand and formalize the selective
heads, we provide aninterpretableconstruction of the self-attention layer weights in a 3-layer attention-
only disentangled transformer [Friedman et al., 2023] that implements this mechanism (Sec. 5). We
empirically demonstrate that the constructed transformer matches the performance of both disentangled
and standard transformers trained from scratch (Sec. 6) and that 2-layer attention-only transformers
cannot solve the task. Moreover, we observe that the attention maps of the trained and constructed
transformers present the same patterns, further supporting the validity of our algorithm. Finally, we

<!-- page 3 -->

3
theoretically analyze the predictor implemented by this construction (Sec. 5.3) showing that, in certain
cases, it asymptotically converges to the maximum likelihood solution. Our findings provide valuable
insights into the mechanisms by which transformers perform model selection.
Additional theoretical analyses, omitted proofs (App. A), extra experiments (Apps. C, E, B), and gen-
eralizations of our transformer construction (Apps. F, G, H) are deferred to the appendix.
2.Related Work
Following the initial empirical observations of the emergent in-context learning capabilities of trans-
formers [Brown, 2020], several works have attempted to understand this phenomenon. Xie et al. [2021]
sought to formulate in-context learning as Bayesian inference, while Garg et al. [2022] studied the ability
of transformers to learn simple functions, such as linear models or multilayer perceptrons, in context.
A subsequent line of work [Akyürek et al., 2022, Bai et al., 2023, Von Oswald et al., 2023,, Raventós
et al., 2024] shows that transformer layers might implement gradient descent to solve in-context linear
regression. Ahn et al. [2023] extends this idea to higher-order algorithms. Importantly, Olsson et al.
[2022] postulates that in-context learning is tied to the emergence of induction heads. Bietti et al. [2023],
subsequently extended this idea, showing the development of induction heads to learn bigrams in-context
and showcasing a connection with associative memories. More closely related to our work is the literature
analyzing transformers through the lens of Markov chains. In particular, Nichani et al. [2024] shows how
transformers trained on sequences generated by Markov chains on a graph learn simple circuits to cap-
ture the underlying causal structure and implement the Bayes-optimal solution by estimating transition
probabilities in context. Similarly, Edelman et al. [2024] illustrate the formation of statistical induc-
tion heads that accurately compute posterior probabilities based on bigram statistics. Makkuva et al.
[2024] used Markov chains to study the loss landscape of transformers, while Rajaraman et al. [2024]
show that a constant depth is sufficient to learn k-th order Markov chains. Svete and Cotterell [2024]
demonstrate that transformers with hard or sparse attention can exactly represent any n-gram model.
Hu et al. [2024] highlights the limitations of transformers in learning HMMs compared to RNNs. Nguyen
[2024] recently studied how rules formed out of simple N-gram statistics can approximate transformer
predictions; however, the mechanism through which such rules are selected remains unexplained. On
the problem of in-context selection, Bai et al. [2023] demonstrates that a single transformer can adap-
tively select between different base algorithms—or even qualitatively different tasks like regression and
classification—based on the context provided. Similarly, Yadlowsky et al. [2023] studied the ability of
transformers to perform model selection between different function class families.
3.(Disentangled) Transformer Models
Inthefollowingweintroducethenecessarybackgroundandnotationaboutthemodelsweuselater.
T ransformers.The architecture of decoder-only transformers is built on two fundamental components,
the attention mechanism and the multi-layer perceptron (MLP). Given a finite alphabetS, transformers
map an input sequences=s 1:T = (s1, . . . , sT )∈ S T to a sequence of vectorsz= (z1, . . . , zT )wherez i ∈
Rd. Each element of the input sequencesi is first encoded using its corresponding one-hot vector,esi ∈
{0,1} |S|. These one-hot representations are then mapped tod-dimensional vectors via an embedding
matrixE∈R d×|S|. To incorporate positional information, a positional embedding matrixF∈R d×T
is added. With a slight abuse of notation, letei denote thei-th element of the canonical basis ofRT
such that each input elementsi is mapped to a vectorxi ∈R d viax i =Ee si +F e i. The information of
the different tokens is then mixed by the causal self-attention heads: denoting the key, query and value
matricesK, Q∈R d×dQK,V∈R d×d, and given an inputh∈R d×T, one gets
Attn(h;Q, K) :=A(h;Q, K)h ⊤,withA(h;Q, K) :=Softmax
 
M(h⊤QK ⊤h);α

,
where Softmax(v;α)i := exp (vi/α)P
j exp (vj /α) is applied row-wise andα >0is a temperature parameter. In the
following, we callA=QK ⊤ ∈R d×d theattention matrix,A ∈R T×T theattention, and Attn:R d×T →
Rd×T theattention layer. The causality of the self-attention is enforced by a maskM, to prevent
the model from attending to future tokens i.e.,M(A)ij =A ij ifi≥j,−∞otherwise. For a model
withLlayers and{H l}l∈[L] attentions heads per layer, we denote byQ(l,h), K(l,h), V (l,h) the attention
parameters for thei-th head in thel-th layer,W (l)
1 , W (l)
2 ∈R d×dFF the parameters of the MLP at layer

<!-- page 4 -->

4
l, andW O ∈R |S|×d the parameters of the output linear layer. Then, withh(0) = (x1, . . . , xT )∈R d×T
as computed above, the decoder transformerT(s1:T )can be written forl= 1, . . . , L, as
˜h(l) =h (l−1) +
HlX
h=1
Attn(h(l−1);Q (l,h), K(l,h))V (l,h), h (l) = ˜h(l) +W (l)
2 σ

W (l)⊤
1 ˜h(l)

where the output is given byWOh(L) ∈R |S|×T.
Disentangled T ransformers.To improve the interpretability of the operations implemented by the
models, Friedman et al. [2023] propose a transformer architecture in which each layer’s output is con-
catenated, rather than added, to its input. This construction makes the residual stream explicitly
disentangled, but increases the embedding dimension (constant for standard transformers) with depth.
Additionally, in suchdisentangled transformersthe MLP layers are removed, the attention heads are pa-
rameterized by a single matrix˜A :=QK ⊤ ∈R dℓ×dℓ, and the value matrices are absorbed into the output
layerfWO. Both the token and positional embedding are one-hot encoding, i.e.,EandFare identity
matrices, and we encode the inputsi as[e si , ei]via concatenation rather than addition. Altogether, the
disentangled transformereT(s 1:T )is formalized forl= 1, . . . , Las
ˆh(l,h) =Attn(h (l−1); ˜A(l,h))forh= 1, . . . , H l,andh (l) = [h(l−1), ˆh(l,1), . . . ,ˆh(l,Hl)],
where the output isfWOh(L). Due to the concatenation, the embedding dimension grows over layers as
dl = (1 +H l)·d l−1 withd 0 =|S|+T. Importantly, Nichani et al. [2024] demonstrate that disentangled
transformers are equivalent to standard transformers using only attention layers.
4.Markov Chains and Causal Structure Selection
To address the limitations of existing synthetic settings based on Markov chains and better capture
the complexity of natural language, we propose a novel framework. In this framework, the model must
learn to select the correct causal structure in-context in order to solve the task and generate the input
sequence. In the following, we describe this task in detail and outline its solution.
Interleaved Markov Chains.The framework consists of sequences of lengthTon a finite alphabet
of tokensS, generated byKdistinct sources. LetU={U 1, . . . , UK}be the set of sources andK=
{k1, . . . , kK}a set of positive integers; each sourceUj consists ofk j interleaved and identical irreducible
aperiodic Markov chains [Batu et al., 2004, Minot and Lu, 2014]. All the sources are defined by the same
transition matrixP ⋆ ∈ P |S|×|S|, wherePis the set of row-stochastic matrices. This model is equivalent
to a time-homogeneous Markov chain(X(j)
t )t≥0 of orderk j , whose transition probabilities depend only
on a single statekj steps back:
P(Xt =s t |X t−1 =s t−1, . . . , X1 =s 1) =P(X t =s t |X t−kj =s t−kj) =s ⊤
t−kj P ⋆st .
Here, we callk j ∈ Kthelagparameter, as defined by Berchtold and Raftery [2002], whereK ⊆J1, TK
is the set of possible lags. The lag, represented by the edges in Fig. 1, encodes the causal structure by
explicitly representing the causal relationship between the variables in the Markov chain.
Data.GivenP ⋆ andK, a lag is uniformly sampled fromKfor each sequence. Denoting the maximum
lag byˆk= max(K), the first ˆkelements of each sequence are sampled from the stationary distributionπ
ofP ⋆, ensuring a constant number of independent variables for all sources. The likelihood of a sequence
of lagkisP(X 1, . . . , XT |k) = Qˆk
i=1 π(Xi)QT
j=ˆk+1 P(Xj |X j−k).
T ask.In this setting, the task is to predict the next statesT+1 given an input sequences1:T generated
from one of the sources, sampled at random. However, the identity of the source, and therefore the lag,
is unknown. This task amounts to solving the following minimization problem:
f ⋆ = inf
f
E k∼Unif[1,...,ˆk]
(X1:T )∼P(X1,...,XT |k)
DKL (P(XT+1 |X T−k+1 )∥f(X 1, . . . , XT )),(1)
whereD KL is the Kullback–Leibler divergence. Eq. 1 admits a closed form solution which is the Bayesian
model average (BMA), defined as the average of the transition probabilities for each lag, weighted by
their posterior probabilities:
P(XT+1 |X 1:T ) =
X
k∈K
wk(X1:T )P(XT+1 |X T−k+1 )withw k(X1:T ) = P(X1:T |k)P(k)P
j∈K P(X1:T |j)P(j) .

<!-- page 5 -->

5
Asymptotically, the posterior distribution concentrates around the maximum likelihood estimate (MLE)
[Rousseau and Mengersen, 2011]. Letk ∗ be the lag that maximizes the likelihood for a sequence
(s1, . . . , sT ), i.e.,k ∗ = arg max k∈K P(X1 =s 1, . . . , XT =s T |k). AsT→ ∞, the posterior proba-
bilityw k converges to 1 fork∗ and to 0 for the other lags, i.e.,wk →1[k=k ∗]where1is the indicator
function. Then, BMA reduces to selecting the lag with the highest likelihood:
Q(XT+1 |X 1, . . . , XT ) =
X
k∈K
1[k=k ∗]P(XT+1 |X T−k+1 ).(2)
It is important to note that an interleaved Markov chain of lagkis mathematically equivalent to ak-th
order Markov chain with a specific transition structure. Thus, given a set of ordersKand a sequence
generated according to one such order, one could theoretically solve the task by learning in-context the
corresponding( ˆk+1)-gram transition probabilities [Nichani et al., 2024, Edelman et al., 2024]. However,
such an approach fails to leverage the low-dimensional structure of the problem, resulting in a suboptimal
sample complexity ofO(|S| ˆk+1).
5.How Can Transformer Do In-Context Selection?
We now want to understand which algorithm transformers learn during training. We focus on disen-
tangled transformers as defined in Sec. 3, which allow for a more interpretable analysis of the model
internal computations. The following proposition, which is the main result of the paper, shows how a
disentangled transformer can implement a predictor to solve the in-context selection task.
Proposition 1.LetKbe a contiguous subset of integers, i.e.,K=J ˆk−K+ 1, ˆkKforK=|K|and
ˆk= max(K). For anyT≥ ˆkthere exists a three-layer disentangled transformereTwithKheads in the
second layer such that, defining the normalized transition probabilities˜pi,k :=
X ⊤
i−kP ⋆XiP
l∈K,l<i X ⊤
i−lP ⋆Xi
fori >1:
eT(X 1:T )T =
X
k∈K
˜wk(X1:T )P(XT+1 |X T−k+1 )with˜w k(X1:T ) =
exp

β
(T− ˆk)
PT
i=ˆk+1 ˜pi,k

P
m∈K exp

β
(T− ˆk)
PT
i=ˆk+1 ˜pi,m
 .
(3)
The predictor implemented by the transformer in Eq. 3 resembles BMA but differs in how it aggregates
past information. Instead of using the posterior of each model as in BMA, our method employs weights
proportional to the exponential of the average of normalized transition probabilities. We analyze this
predictor in Sec. 5.3, and discuss its convergence to ML. Proposition 3 illustrates how transformers can
implementselective induction heads, a mechanism that adapts to the input sequence by copying the
token correspondent to the lag that maximizes the average normalized transition probabilities.
The proof of Proposition 1, in Sec. 5.1 below, involves an explicit construction for the weights of the
disentangled transformer implementing the solution in Eq. 3 (an alternative construction for the third
layer is in App. F). Notably, this construction produces attention maps similar to those in standard trans-
formers (see Fig. 1), suggesting our algorithm closely aligns with trained transformer implementations.
Moreover, we discuss in Sec. 5.4 different generalizations, including the construction for non-contiguous
lags. The same construction using a single head implements the same algorithm but results in worse
sample complexity. However, in the specific case where|K|= 2, we provide a different single-head
construction that recovers the sample complexity of the multi-head version.
5.1.Proof of Proposition 1: Construction for Contiguous Lags
To aid intuition, we use a running example with visual illustrations forT= 10,K={1,2,3}. We recall
that each input elementsi is encoded ash(0)
i = [esi , ei]∈ {0,1} |S|+T.
First layer: extraction of transition probabilities.The first attention matrix, ˜A(1), consists of two
blocks: the first block operates on the semantic component of the input tokens, learning the transpose
of the logarithm of the transition matrix1. The second blockA(1) learns the causal relationships induced
1Here,logPdenotes the element-wise logarithm, i.e.,(logP) ij = log(P ij ).

<!-- page 6 -->

6
by each possible lagsi−k →s i fork∈ K:
eA(1) =
 logP ⋆⊤ 0
0 A(1)

A(1)
ij =
(
+λifi−j∈ K
−λifi−j̸∈ K.
A(1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ +λ -λ -λ -λ -λ -λ -λ -λ -λ+λ +λ +λ -λ -λ -λ -λ -λ -λ -λ-λ +λ +λ +λ -λ -λ -λ -λ -λ -λ-λ -λ +λ +λ +λ -λ -λ -λ -λ -λ-λ -λ -λ +λ +λ +λ -λ -λ -λ -λ-λ -λ -λ -λ +λ +λ +λ -λ -λ -λ-λ -λ -λ -λ -λ +λ +λ +λ -λ -λ-λ -λ -λ -λ -λ -λ +λ +λ +λ -λ


We can compute the first layer’s attention as:
[esi , ei]⊤ ˜A(1)[esj , ej] = (logP) sj ,si +λsign(A (1)
ij )
and applying the softmax:
A(1)(h(0)
1:T ; ˜A(1))ij = e(logP) sj ,si +λsign(A(1)
i,j )
Pi
r=1 e(logP) sr ,si +λsign(A(1)
i,r )
= e(logP) sj ,si +λsign(A(1)
ij )
P
r∈K e(logP) sr ,si +λ +P
r̸∈K e(logP) sr ,si −λ .
Forλ→ ∞(in practice, forλlarge enough) and having denoted˜pi,k :=
Psi−k ,siP
r∈K,r<i Psi−r ,si
fori >1,
lim
λ→∞
A(1)(h(0)
1:T ; ˜A(1))ij =



˜pi,i−j ifi−j∈ K
1ifi=j= 1
0elsewhere
.
Therefore, the output at indexiafter the first layer corresponds to a weighted average of the past tokens
h(0)
i−k fork∈ Kwhere the weights are given by the normalized probabilities˜pi,k:
ˆh(1)
i =Attn(h (0)
1:T ; ˜A(1))i =
( Pi
j=1 1[i−j∈ K]
Psj ,siP
r∈K Psr ,si
h(0)
j ifi >1
h(0)
1 ifi= 1
=



P
k∈K,k<i
˜pi,kh(0)
i−k ifi >1
h(0)
1 ifi= 1
With the input vectorsh(0)
i being the concatenation of the one-hot encoding of the state and position
[esi , ei], the first|S|entries of ˆh(1)
i correspond to˜si = P
k∈K,k<i ˜pi,kesi−k fori >1and˜s 1 =e s1.
The remaining entries, due to the one-hot positional encoding, directly copy the normalized transition
probabilities for the transitionsi−k →s i into the|S|+ (i−k)-th element of ˆh(1)
i . To build intuition, we
refer to the example in Eq. 4 where the colors highlight transition probabilities of the same lag:
A(1) =


1 0 0 0 0 0 0 0 0 01 0 0 0 0 0 0 0 0 0˜p3,2 ˜p2,1 0 0 0 0 0 0 0 0˜p4,3 ˜p4,2 ˜p4,1 0 0 0 0 0 0 00 ˜p5,3 ˜p5,2 ˜p5,1 0 0 0 0 0 00 0 ˜p6,3 ˜p6,2 ˜p6,1 0 0 0 0 00 0 0 ˜p7,3 ˜p7,2 ˜p7,1 0 0 0 00 0 0 0 ˜p8,3 ˜p8,2 ˜p8,1 0 0 00 0 0 0 0 ˜p9,3 ˜p9,2 ˜p9,1 0 00 0 0 0 0 0 ˜p10,3˜p10,2˜p10,10


ˆh(1) =


˜s1 ˜s2 ˜s3 ˜s4 ˜s5 ˜s6 ˜s7 ˜s8 ˜s9 ˜s10
1 1 ˜p3,2 ˜p4,3 0 0 0 0 0 00 0 ˜p3,1 ˜p4,2 ˜p5,3 0 0 0 0 00 0 0 ˜p4,1 ˜p5,2 ˜p6,3 0 0 0 00 0 0 0 ˜p5,1 ˜p6,2 ˜p7,3 0 0 00 0 0 0 0 ˜p6,1 ˜p7,2 ˜p8,3 0 00 0 0 0 0 0 ˜p7,1 ˜p8,2 ˜p9,3 00 0 0 0 0 0 0 ˜p8,1 ˜p9,2 ˜p10,3
0 0 0 0 0 0 0 0 ˜p9,1 ˜p10,2
0 0 0 0 0 0 0 0 0 ˜p10,1
0 0 0 0 0 0 0 0 0 0


(4)
The operation of the first layer is now explicit: for each tokenh(0)
i , it extracts the normalized transition
probabilities˜pi,k for each possible lag and stores them in the elementˆh(1)
i,S+T−k. The resulting vector is
subsequentlyconcatenatedtotheresidualstreamtobefedtothesecondlayerh (1)
i = [[esi , ei], ˆh(1)
i ].
Second layer: aggregation of transition probabilities.To predict the next token, the model needs
to determine which lag generated the sequence based on the past transitions. This selection requires
aggregating the normalized transition probabilities from the past, and storing them in the embedding
of the current token. However, since consecutive tokens store transition probabilities in overlapping
positions, the attention needs to learn a convex combination of tokens that avoid mixing information
from different transitions while maximizing the number of˜pstored (to not discard useful information).
For instance, when aggregating the past for the token ati= 10in Eq. 4, summingˆh(1)
9 and ˆh(1)
10 would
mix˜p10,2 and˜p9,1 together. This mixing can be avoided, for example, by only selecting tokens every
3 steps (ˆh(1)
4 , ˆh(1)
7 , ˆh(1)
10 ) copying transitions without blending information. More generally, the attention
A(2,1) should attend to everyK-th token from the current one, which is equivalent to having non-zero
entries along the diagonals at positionsnKforn∈NandnK < T. This structure can be enforced by

<!-- page 7 -->

7
constructing the attention matrix˜A(2,1) with a single non-zero block operating on the tokens’ positional
encoding, as follows:
˜A(2,1) =


0 0
0 A(2,1) 0
0 0

A(2,1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ +λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ +λ -λ -λ-λ -λ -λ -λ -λ +λ -λ -λ +λ -λ-λ -λ -λ +λ -λ -λ +λ -λ -λ +λ


A(2,1) =


0 0 0 0 0 0 0 0 0 00 0 0 0 0 0 0 0 0 00 0 0 0 0 0 0 0 0 00 0 0 1 0 0 0 0 0 00 0 0 0 1 0 0 0 0 00 0 0 0 0 1 0 0 0 00 0 0 1/2 0 0 1/2 0 0 00 0 0 0 1/2 0 0 1/2 0 00 0 0 0 0 1/2 0 0 1/2 00 0 0 1/3 0 0 1/3 0 0 1/3


where the firstˆkrows and columns are empty because the firstˆkelements of the sequence are sampled
independently from the stationary distribution and therefore not informative.
This construction resolves the issue of overlapping transitions, but copying only a subset of tokens implies
losing information from the excludedˆhi. Introducing additional attention heads˜A(2,2), . . . , ˜A(2,H2) with
the same form as ˜A(2,1) above overcomes this limitation. The resulting attentionsA (2,2), . . .A (2,H2)
still follow a diagonal structure asA(2,1) to avoid overlapping transitions, but they are shifted to copy
different tokens. For the given example, we can designA(2,2) as in Eq. 5 to attendˆh(1)
9 and ˆh(1)
6 , and
similarly, constructA (2,3) for ˆh(1)
8 and ˆh(1)
5 when computing the output ati= 10.
A(2,2) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ +λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ +λ -λ -λ-λ -λ -λ -λ -λ +λ -λ -λ +λ -λ


A(2,3) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ-λ -λ -λ +λ -λ -λ +λ -λ -λ -λ-λ -λ -λ -λ +λ -λ -λ +λ -λ -λ


(5)
Each head has a diagonal structure with non-zero entries along the diagonals at positionnK+h−1for
n≥0andh∈ {1, . . . , H 2}, the attention matrices can be formalized as:
A(2,h)
ij =λ
(
+1,ifi≥j > ˆkand(i−j) modK=h−1
−1,otherwise,
where the conditioni≥j≥ ˆkensures that all entries in the first ˆkrows and columns are set to−λ
and imposes a lower triangular structure due to causal masking. The modulo operation instead assigns
each diagonal multiple ofKto+λ(allowing attention) and the remaining diagonals to−λ(masking
attention), whilehdetermines the shift of the the first positive diagonal to ensure the heads do not
overlap. The output of each head in the second layer is given by computing:
[[esi , ei], ˆh(1)
i ]⊤ ˜A(2,h)[[esj , ej], ˆh(1)
j ] =A (2,h)
i,j =λsign(A (2,h)
i,j ),
and applying softmax and in the limit asλ→ ∞, the rows of the attention become uniform for positive
entries and zero otherwise:
A(2,h)
ij =Softmax(A (2,h))ij =
1
h
A(2,h)
ij =λ
i
Pi
m=1 1
h
A(2,h)
im =λ
i .
The outputˆh(2,h)
i of each head is then concatenated into the residual stream. Fig. 2 shows the output for
the10th token 2 for each attention headˆh(2,1)
10 , ˆh(2,2)
10 and ˆh(2,3)
10 to visualize the mechanism of the second
layer. The arrows of different colors represent how each head aggregates transition probabilities by
attending to non-overlapping past tokens and averaging them with uniform weights. When concatenating
the output of the different headsh(2)
i = [h(0)
i , ˆh(1)
i , ˆh(2,1)
i , . . . ,ˆh(2,H2)
i ], we can see how the10th token stores
the transition probabilities of its entire past for each lag.
Third layer: average of transition probabilities and lag selection.To build some intuition,
suppose the current token is at positioni= 10, and we are predicting the 11th token in the sequence.
Given a set of possible lags{1,2,3}, the third attention mechanism must concentrate around one of the
tokens at positions8,9, or10. This ensures that transitions for all possible lags are considered: If the
sequence was generated from the source of lag 1, the token at position10needs to be copied to predict
the transition probabilities for the 11th token. For lag2, the token at position9is copied, and so on.
2To simplify the notation we depict withsand single gray block instead of|S|elements, the entries correspondent to
the semantics and their average after the attention which are not used in the mechanism.

<!-- page 8 -->

8


es6+es9
0000010010˜s6+˜s9
00˜p6,3
˜p6,2
˜p6,1
˜p9,3
˜p9,2
˜p9,1
00


1
2


es5+es8
0000100100˜s5+˜s8
0˜p5,3
˜p5,2
˜p5,1
˜p8,3
˜p8,2
˜p8,1
000


1
2


es1 es2 es3 es4 es5 es6 es7 es8 es9 es10
1 1 1 1 1 1 1 1 1 1˜s1 ˜s2 ˜s3 ˜s4 ˜s5 ˜s6 ˜s7 ˜s8 ˜s9 ˜s10
1 1 ˜p3,2 ˜p4,3
˜p3,1 ˜p4,2 ˜p5,3
˜p4,1 ˜p5,2 ˜p6,3
˜p5,1 ˜p6,2 ˜p7,3
˜p6,1 ˜p7,2 ˜p8,3
˜p7,1 ˜p8,2 ˜p9,3
˜p8,1 ˜p9,2 ˜p10,3
˜p9,1 ˜p10,2
˜p10,1


1
3


P
i=4,7,10esi
0001001001P
i=4,7,10˜si
˜p4,3
˜p4,2
˜p4,1
˜p7,3
˜p7,2
˜p7,1
˜p10,3
˜p10,2
˜p10,1
0


A(2,1)
A(2,1)
A(2,1)
A(2,2)
A(2,2)
A(2,2)
A(2,2)
Figure 2.Visualization of the mechanism of the second attention layer in our construction.The
matrix represents the input of the second layerh(1) whereas the single vectors represent the output for the 10th
token ˆh(2,1)
10 , ˆh(2,2)
10 , ˆh(2,3)
10 . Each of the three attention heads (arrows of different colors) copies non-overlapping
transition probabilities at distance3from each other from the past. By doing this, the output of the second layer
for the current token (10) contains all˜pfor each lag without loss of information.
To determine the correct lag based on the sequence’s history, our construction relies on the sum of past
transitions up to the current token,P
j<i ˜pj,k. Therefore, the third attention is constructed such that
the entries corresponding to the transitions of possible lags are proportional to the respective cumulative
sums. For example, to select which token among the ones in position8,9,10should be copied to predict
11, the third attention must be such thatA(3)
10,10 is proportional exclusively to the sum of transitions of
lag 1, i.e.,P
j≤10 ˜pj,1 whileA (3)
10,9 is exclusively proportional toP
j≤10 ˜pj,2 andA (3)
10,8 ∝P
j≤10 ˜pj,3. Then,
in the limit of the softmax converging to the hardmax, the attention collapses to the entry corresponding
to the larger sum and selects the correspondent lag by copying the associated token. More generally, for
this to apply to all rows, the third attention matrix must be constructed such thatA(3)
i,i ∝P
j≤i ˜pj,1,
whileA (3)
i,i−1 ∝P
j≤i ˜pj,2, andA (3)
i,i−2 toP
j≤i ˜pj,3, with all remaining entries set to zero.
This selection mechanism is implemented through the combination of multiple blocks within the third
attention matrix, ˜A(3), which acts on the concatenated tokensh(2)
i = [h(0)
i , ˆh(1)
i , ˆh(2,1)
i , . . . ,ˆh(2,H2)
i ]and is
structured as follows:
˜A(3) =


0 00 A(3)
0 00 B(3)
0 00 B(3)
0 0 0 . . . 0 0
0 0 0 0 . . . 0 0
0 0 0 0 . . . 0 0
0 0 0 . . . 0 0
... ... ... ... ... ... ...
0 0 0 0 . . . 0 0
0 0 0 0 . . . 0


A(3) =


+λ-λ -λ -λ -λ -λ -λ -λ -λ -λ+λ+λ-λ -λ -λ -λ -λ -λ -λ -λ+λ+λ+λ-λ -λ -λ -λ -λ -λ -λ-λ+λ+λ+λ-λ -λ -λ -λ -λ -λ-λ -λ+λ+λ+λ-λ -λ -λ -λ -λ-λ -λ -λ+λ+λ+λ-λ -λ -λ -λ-λ -λ -λ -λ+λ+λ+λ-λ -λ -λ-λ -λ -λ -λ -λ+λ+λ+λ-λ -λ-λ -λ -λ -λ -λ -λ+λ+λ+λ-λ-λ -λ -λ -λ -λ -λ -λ+λ+λ+λ


B(3) =β


0 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 0


with the general formulation of the two matricesA(3) andB (3) given by:
A(3)
ij =λ
(
+1ifi−j+ 1∈ K
−1ifi−j+ 1̸∈ K , B (3)
ij =β
(
+1,if(i−j) modK=K−1
0,otherwise.
The matrixA (3) acts on the positional embedding of the input, similarly to the matrixA(1) in the first
layer. The difference is that the position of the diagonals is now shifted by one. This shift ensures that
the only non-zero entries after softmax are the ones on the diagonals corresponding tok−1fork∈ K.
The matrixB (3) is instead responsible for the sum of the normalized transitions. Each block operates
on the output of a corresponding head in the second layer. To understand how, consider the following

<!-- page 9 -->

9
tokens in output of the first head in the second layer,
ˆh(2,1)10 = 1/3 · P
i=4,7,10esi 0 0 0 1 0 0 1 0 0 1 P
i=4,7,10˜si ˜p4,3 ˜p4,2 ˜p4,1 ˜p7,3 ˜p7,2 ˜p7,1 ˜p10,3 ˜p10,2 ˜p10,1 0
ˆh(2,1)9 = 1/2 · es6+es9 0 0 0 0 0 1 0 0 1 0 ˜s6+ ˜s9 0 0 ˜p6,3 ˜p6,2 ˜p6,1 ˜p9,3 ˜p9,2 ˜p9,1 0 0
ˆh(2,1)8 = 1/2 · es5+es8 0 0 0 0 1 0 0 1 0 0 ˜s5+ ˜s8 0 ˜p5,3 ˜p5,2 ˜p5,1 ˜p8,3 ˜p8,2 ˜p8,1 0 0 0
| {z }
ˆm(2,1)8
| {z }
ˆp(2,1)8
where we defineˆp(2,h)
i ∈R T as the block ofˆh(2,h)
i which contains the normalized transition probabilities
andˆm(2,h)
i ∈R T contains a copy of the second attention. With the structure of˜A(3), we can see howB(3)
acts on these two blocks such that:h(2)⊤
i ˜A(3)h(2)
j =PK
h=1 p(2,h)⊤
i B(3)m(2,h)
j +e iA(3)ej. This operation
selectively sums the normalized transition probabilities such that the entry of the attention in the third
layer corresponding to the transition with lagkfor the next token contains only the sum of the transitions
with the same lag:h(2)⊤
i ˜A(3)h(2)
i−k+1 ∝P
j≤i ˜pj,k. This process is illustrated in the following:
ˆp(2,1)⊤
10 B(3) ˆm(2,1)
8 = β
3


˜p4,3
˜p4,2
˜p4,1
˜p7,3
˜p7,2
˜p7,1
˜p10,3
˜p10,2
˜p10,1
0


⊤

0 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 00 0 1 0 0 1 0 0 1 01 0 0 1 0 0 1 0 0 10 1 0 0 1 0 0 1 0 0




0000
1/2
00
1/2
00


= β
3


˜p4,3
˜p4,2
˜p4,1
˜p7,3
˜p7,2
˜p7,1
˜p10,3
˜p10,2
˜p10,1
0


⊤

1001001000


= β
3 (˜p4,3 + ˜p7,3 + ˜p10,3). (6)
Notably, this mechanism allows the transformer to average the transition probabilities in the past of
each lag independently. The key idea behind it is thatB(3)m(2,h)
i−k+1 is a boolean vector, such that when
multiplied byˆp(2,h)
i , it sums only the entries that correspond to transitions of lagk. For instance the
productˆp(2,1)
10 B(3) ˆm(2,1)
8 in Eq. 6, it should sum the transitions of lag3stored inˆh(2,1) to giveA10,8 such
that the 8th token can be copied to predict the 11th if the lag of the sequence is3. However, we can notice
how simply taking the inner productˆp(2,1)⊤
10 ˆm(2,h)
8 would lead to the wrong computations summing over
the transitions of lag2instead of3and excluding the transition correspondent to˜p 4,2. This happens
because all the transitions are stored starting from the elementi−1ofˆp(2,1)
i and noti. To account for
this, the matrixB(3) performs a permutation such that the mask is shifted by one position. Along with
permuting the mask, the matrixB(3) also removes the normalization factor (1/2) and adds the missing
entries in the mask due toj < i. To achieve this, each column ofB(3) follows a pattern in which the
entries are spaced at intervals ofK, and the pattern shifts by one position between successive columns
such that all possible sequences are present, allowing to sum over all possible lags. This shift creates a
cyclic arrangement where the columns repeat everyKas the transitions within the vectorˆp(2,h)
i .
The additional blocks containingB(3) act on the outputs of the other heads, performing the same opera-
tion by selectively summing the transitions of the same lag stored in the respective outputs. Considering
all heads and only the non-zero entries after softmax, occurring atj=i−k+1due toA(3), we get
h(2)⊤
i ˜A(3)h(2)
i−k+1 =
KX
h=1
p(2,h)⊤
i B(3)m(2,h)
i−k+1 +λ=
KX
h=1
β
τh,i + 1
τh,iX
n=0
˜pˆk+h+nK,k +λ,
whereτ h,i =⌊ i−ˆk−h
K ⌋. Applying the softmax, taking the limitλ→ ∞results in non-zero entries of the
attention only whereA(3)
i,j = +λsimilarly to the first layer. Moreover, for largeifor whichτh,i + 1≈ i−ˆk
K
and absorbingKinside the temperatureβ, considering the last tokenTwe can see how the attention
weights for the final tokenTrecover the weights˜wk(X1:T )from Eq. 3 in Proposition 1:
A(3)(h(2)
1:T ; ˜A(3))T,j =



exp

β
(T− ˆk)
TP
n=ˆk+1
˜pn,k
!
P
r∈K
exp

β
(T− ˆk)
TP
n=ˆk+1
˜pn,r
! ifT−j+ 1 =kfork∈ K
0elsewhere
(7)
The third attention layer,A(3)(h(2)
1:T ; ˜A(3)), is non-zero only at positionsjwhere the positionk=T−j+1
corresponds to a valid lag in the setK. At these positions, the attention value is precisely the weight˜wk:

<!-- page 10 -->

10
˜wk(X1:T ) =A (3)(h(2)
1:T ; ˜A(3))T,j=T−k+1 fork∈ K. The output of the third attention has the following
form:
ˆh(3)
i =
P
k ˜wkesi−k+1 ,P
k ˜wkei−k⋆+1, P
k ˜wkˆh(1)
i−k⋆+1, P
k ˜wkˆh(2,1)
i−k⋆+1, . . . P
k ˜wkˆh(2,H2)
i−k⋆+1

(8)
which is then concatenated to the residual stream:
h(3)
i =
h
esi , ei, ˆh(1)
i , ˆh(2,1)
i , . . . ˆh(2,H2)
i , ˆh(3)
i
i
.
Output layer:Finally, the output layer fWO ∈R S×P
l dl contains all zero blocks, except for the one
acting on the semantics of the token copied by the third attention (blue block in Eq.8). This block learns
the transition matrixP ⋆:
fWO =
 0S×S 0S×T 0S×d0 0S×2d0 . . . 0S×2d0 P ⋆⊤ 0S×T 0S×d0 0S×2d0 . . . 0S×2d0

.
Therefore, the output of the disentangled transformer for the last token becomes:
eT(X 1:T )T =fWOh(3)
T =
X
k
˜wkP ⋆⊤esT−k+1 =
X
k
˜wkP ⋆
sT−k+1
which concludes the constructive proof of Proposition 1.
5.2.Selective induction head and next token prediction
We have shown in Eq. 8 how the last attention layer computes a weighted average of the tokens at a
distancekfrom the next one, with weights proportional to the average of the normalized transition
probabilities of lagk. However, we observe empirically (see Fig. 5) that trained models learn large values
ofβfor which the softmax converges to the hardmax, effectively concentrating attention on the single
token with the largest average normalized transition probability. This observation motivates our analysis
of the limiting case asβ→ ∞:
A(3)
i,j = ˜wk =1[i−j+ 1 =k ⋆]withk ⋆ =argmax k
iX
n=ˆk+1
˜pn,k.
Here, the transformer selects the causal structure (i.e., the lag) corresponding to the largestPi
n=ˆk+1 ˜pn,k.
Given the current tokeni, the third layer then copies the token from the positioni−k ⋆ + 1, i.e.,
ˆh(3)
i =Pi
j=ˆk 1[i−j+ 1 =k ⋆]h (2)
j =h (2)
i−k⋆+1. After concatenation to the residual stream, the tokens
are of the following form:
h(3)
i =
h
esi , ei, ˆh(1)
i , ˆh(2,1)
i , . . . ˆh(2,H2)
i , esi−k⋆ +1 , ei−k⋆+1, ˆh(1)
i−k⋆+1, ˆh(2,1)
i−k⋆+1, . . . ˆh(2,H2)
i−k⋆+1
i
Theoutput 3ofthedisentangledtransformerbecomesthetransitionprobabilitycorrespondingtok ⋆:
eT(X 1:T )T =fWOh(3)
T =P ⋆
sT−k ⋆ +1 .
We define aselective induction headas a multi-layer attention circuit that generalizes
the standard induction head [Olsson et al., 2022]. It is a mechanism where the position to
copy from is not determined by a fixed rule (e.g., a previous token match), but is instead
selected in-contextbased on an evidence-aggregation score (Pi
n=ˆk+1 ˜pn,k in this case). Unlike
a standard induction head, which implements a static copying pattern, a selective induction
head dynamically chooses which causal rule to apply from a set of learned possibilities. In our
construction, this is achieved by:
(1) Preceding layers computing a score for each causal structure (i.e., each lagk).
(2) The final layer acting as the selector, attending to and copying the token corresponding
to the lag with the highest aggregated score.
3Notice that we omitted the final softmax for simplicity. However, our construction would remain equivalent, with the
only difference that the relevant block of the output layerfWO would learnlogP ⋆⊤ instead ofP ⋆⊤. Since the selective
induction head copies the one-hot embeddinge sT−k ⋆ +1, the model’s logits would become the correct row of the log-
transition matrix, and the subsequent softmax operation would recover the probability distribution.

<!-- page 11 -->

11
5.3.Equiv alence with Maximum Likelihood
The disentangled transformer we propose does not rely on likelihood nor computes the BMA. Due to the
normalization applied by the softmax function, the model infers the lag of the sequence using the sum
of normalized probabilities˜p. For the inference to be accurate, the cumulative sum corresponding to the
correct lagPi
n=ˆk+1 ˜pn,k⋆ must be larger than that of any other lag. This fact is formalized in terms of
expected values in the following claim:
Claim 1.LetKbe a subset of integers andX t an interleaved Markov chain of lagk∈ K, then, for
r∈ Kandi≥ ˆk,
E
h X ⊤
i−rP ⋆XiP
l∈K X ⊤
i−lP ⋆Xi
i
≤E
h X ⊤
i−kP ⋆Xi
P
l∈K X ⊤
i−lP ⋆Xi
i
.
While simpler cases (e.g., two lags, no normalization, or independent lags) are proven in App. A, we leave
the complete proof of Claim 1 for future work. However, we provide empirical validation of this claim
in Fig. 3 (left); more details and additional experiments are reported in App E. Due to ergodicity, the
average 1
T
PT
i=1 ˜pi,r converges to its expected value, and for large enoughTit is higher for the correct
lag. Therefore, applying the exponential and scaling the temperatureβleads to the same result as MLE
in the asymptotic limit, as shown in Fig. 3 (right).
0.020 0.025 0.030 0.035
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]−maxr̸=k
r∈K
E[
X ⊤
i−rP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
0
250
500
750
1000
1250Frequency
0 50 100
Sequence Length
10 4
10 3
10 2
10 1
100
KL Divergence
KL for = {1, 2, 3}
BMA
ML
= 1.0
= 2.0
= 5.0
= 10.0
= 25.0
= 100.0
Figure 3.Leftdifference of the expected normalized transition probabilities for the true lag and the maximum
over all other lags for|S|= 10.Right:the estimator in Eq. 3 matches MLE for largeβ.
5.4.Generalizations and special cases
Single-head transformers.The construction above allows the model to store all the past transition
probabilities in the embedding of the current token by scaling the number of heads with the total number
of lagsK. Since all heads perform the same operation, reducing the number of heads implements an
equivalent algorithm but withworse sample complexity, as some past transitions are discarded. Thus, a
3-layer single-head transformer still solves the task by implementing the algorithm in Proposition 1, but
it only uses T− ˆk−1
K + 1samples to estimate the correct lag.
Non-contiguous lags.In App. G, we provide examples of constructions to handle non-contiguous lags,
where the core approach remains similar to that in Sec. 5.1. Depending on the specific case, the number
of heads needed ranges between the number of lags andˆk−min(K) + 1.
T wo lags.By the construction in Sec. 5.1, handling two contiguous lags requires two attention heads
in the second layer for optimal sample complexity. However, we provide in App. H an alternative
construction for the third layer which enables a single-head model to match the performance of the
two-head model for any two lags, whether contiguous or not (see empirical results below).
6.Experiments and Discussion
We conduct a series of experiments to empirically validate our construction and determine whether
transformers trained via gradient descent learn it.
Setup.We train3-layer disentangled transformers ( ˜T) and3-layer standard transformers (T) with
learned positional and semantic embeddings both withα=
p
dQK using cross-entropy loss. At each

<!-- page 12 -->

12
0 50 100
Sequence Length
10 4
10 3
10 2
10 1
100
KL Divergence
= {1, 2, 3}
BMA
ML
 train 3H
 train 1H
 train 3H
 constr. 1H
 constr. 3H
Figure 4.Performance of our constructed transformers, trained transformers, and theoretical
estimator (BMA, ML).First plot: lags 1,2,3. Second: the model solves the task with non-contiguous lags.
Third: the model is effective with additional lags. Fourth: one head is enough for two lags.
step, we generate a fresh batch (size256) of sequences (length128) via Alg. 1, and train using Adam
optimizer with fixed learning rate0.001and no weight decay. For the standard transformer, we tested
embedding sizes of64and128, andd QK = 32. For the constructions, we fixβ= 100andλ= 500. We
reportD KL between the true and predicted next-token distribution along the sequence. We generate
different tasks with alphabet size|S|= 5(no differences are observed for other sizes) varying the number
and values of lags:K={1,2,3}(our example) andK={1,2,3,4,5}for the case of contiguous lags
(optimal number of heads3and5according to Proposition 1),K={1,3,4}to show non-contiguous lags
(4heads needed, see App. G), andK={1,3}for the special case of two lags.
Main results.We observe in Fig. 4 how the construction with optimal number of heads, indicated
as ˜Tconstr. in the plots, matches the performance of the maximum likelihood. Moreover, both the
disentangled transformers trained from scratch (˜Ttrain) and the standard one (Ttrain) match the
performance of the theoretical construction. Interestingly, we instead observe that when the number of
heads is fixed to1the trained transformers can find solutions which perform better than the construction:
this indicates the existence of more efficient, yet elusive and non-interpretable, ways of aggregating the
transition probabilities, and that gradient descent can find them. Finally, we illustrate how for the simple
case of two lags (last plot in Fig. 4) our construction with single head (detailed in App H) attains the
optimal sample complexity, while the construction in App. G would assume using more heads. Moreover,
in this case, it appears that the trained transformers can obtain performances closer to the BMA rather
than the ML for small sequence lengths.
T rained vs constructed attention maps.In Fig. 5 we report the attention maps for the three lay-
ers for the caseK={3,5,7}for the trained standard transformer (top row), the trained disentangled
transformer (middle row) and our construction (see construction in App. G) (bottom row).We observe
that the attention maps of the first and third layers are nearly identical between the trained and the-
oretical models, with these layers functioning precisely as expected from the theoretical construction.
Notably, the attention entries of the first layer are proportional tologP ⋆, even when the model is
trained from scratch and for both the disentangled and standard cases. For the second layer (aggrega-
tion), the trained transformers converge to a slightly different structure, likely because aggregation is

<!-- page 13 -->

13
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 3
Figure 5.Attention Maps: Trained versus Constructed Transformers.Heatmaps depicting the atten-
tion mechanisms of the trained standard transformer, disentangled transformer, and our constructed model are
presented for training lags 3, 5, and 7 and the true lag that generated the test sequence isk= 3. The first and
second layers demonstrate a high degree of similarity, and the third layer exhibits the selective induction head.
a combinatorial problem with multiple valid implementations. Despite this variability, a clear diagonal
pattern emerges, closely resembling that in our theoretical construction. Furthermore, as demonstrated
in Fig. 4, all models achieve comparable performance on the task. These findings strongly suggest that
the trained transformer finds a solution that aligns closely with our construction. Remarkably, we also
show that standard transformers trained with learned positional and semantic embeddings and attention
parameterized byQ, K, Vproduce attention maps in agreement with our construction. This provides
compelling evidence that our construction is not merely a byproduct of the disentangled transformer’s
architecture but can also be implemented by standard transformers. Moreover, we observe that even
when the embedding dimension (64) is smaller than the sequence length (128), standard transformers
are still capable of matching the optimal performances, therefore finding more efficient ways to store and
use all the transitions in the past.
7.Conclusion
We introduced a novel synthetic task based on interleaved Markov chains to study how attention-only
transformers perform in-context causal structure selection. Our findings demonstrated that a 3-layer
transformercansolvethistaskwithnear-optimalsamplecomplexity, effectivelyshowcasingtheemergence
of selective induction heads, attention circuits that aggregate past information, and select the correct
causal structure. Moreover, we provided a fully interpretable construction of a disentangled transformer
implementing these circuits to solve the task and empirically verified that both disentangled and standard
transformers trained with Adam closely align with this construction. Finally, we theoretically analyze the
algorithm implemented by this construction, showing that, in certain cases, it asymptotically converges to
maximum likelihood. We believe that the fundamental mechanism behind the formation of these simple
circuits is the same as that underlying the emergence of more complex reasoning capabilities recently
observed in larger models. Understanding this mechanism is crucial for enhancing these capabilities by
developing better training strategies and architectures.

<!-- page 14 -->

14
References
[1] K. Ahn, X. Cheng, H. Daneshmand, and S. Sra. Transformers learn to implement preconditioned
gradient descent for in-context learning. InThirty-seventh Conference on Neural Information Pro-
cessing Systems, 2023. URLhttps://openreview.net/forum?id=LziniAXEI9.
[2] E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context
learning? investigations with linear models.arXiv preprint arXiv:2211.15661, 2022.
[3] Y. Bai, F. Chen, H. Wang, C. Xiong, and S. Mei. Transformers as statisticians: Provable in-context
learning with in-context algorithm selection. InThirty-seventh Conference on Neural Information
Processing Systems, 2023. URLhttps://openreview.net/forum?id=liMSqUuVg9.
[4] Y. Bai, F. Chen, H. Wang, C. Xiong, and S. Mei. Transformers as statisticians: Provable in-context
learning with in-context algorithm selection.Advances in neural information processing systems,
36, 2023.
[5] T. Batu, S. Guha, and S. Kannan. Inferring mixtures of markov chains. InLearning Theory: 17th
Annual Conference on Learning Theory, COLT 2004, Banff, Canada, July 1-4, 2004. Proceedings
17, pages 186–199. Springer, 2004.
[6] A. Berchtold and A. Raftery. The mixture transition distribution model for high-order markov
chains and non-gaussian time series.Statistical Science, 17(3):328–356, 2002.
[7] A. Bietti, V. Cabannes, D. Bouchacourt, H. Jegou, and L. Bottou. Birth of a transformer: A
memory viewpoint.Advances in Neural Information Processing Systems, 36, 2023.
[8] T. B. Brown. Language models are few-shot learners.arXiv preprint arXiv:2005.14165, 2020.
[9] S. Chan, A. Santoro, A. Lampinen, J. Wang, A. Singh, P. Richemond, J. McClelland, and F. Hill.
Data distributional properties drive emergent in-context learning in transformers.Advances in
neural information processing systems, 35:18878–18891, 2022.
[10] B. L. Edelman, E. Edelman, S. Goel, E. Malach, and N. Tsilivis. The evolution of statistical
induction heads: In-context learning markov chains.arXiv preprint arXiv:2402.11004, 2024.
[11] D. Friedman, A. Wettig, and D. Chen. Learning transformer programs. InThirty-seventh Confer-
ence on Neural Information Processing Systems, 2023. URLhttps://openreview.net/forum?id=
Pe9WxkN8Ff.
[12] S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? a case
study of simple function classes.Advances in Neural Information Processing Systems, 35:30583–
30598, 2022.
[13] J. Hu, Q. Liu, and C. Jin. On limitation of transformer for learning hmms.arXiv preprint
arXiv:2406.04089, 2024.
[14] M. E. Ildiz, Y. HUANG, Y. Li, A. S. Rawat, and S. Oymak. From self-attention to markov mod-
els: Unveiling the dynamics of generative transformers. InForty-first International Conference on
Machine Learning, 2024. URLhttps://openreview.net/forum?id=72oT4mPLUb.
[15] A. V. Makkuva, M. Bondaschi, A. Girish, A. Nagle, M. Jaggi, H. Kim, and M. Gastpar. Attention
with markov: A framework for principled analysis of transformers via markov chains.arXiv preprint
arXiv:2402.04161, 2024.
[16] A. Minot and Y. M. Lu. Separation of interleaved markov chains. In2014 48th Asilomar Conference
on Signals, Systems and Computers, pages 1757–1761. IEEE, 2014.
[17] T. Nguyen. Understanding transformers via n-gram statistics.arXiv preprint arXiv:2407.12034,
2024.
[18] E.Nichani, A.Damian, andJ.D.Lee. Howtransformerslearncausalstructurewithgradientdescent.
InForty-first International Conference on Machine Learning, 2024. URLhttps://openreview.
net/forum?id=jNM4imlHZv.
[19] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai,
A. Chen, et al. In-context learning and induction heads.arXiv preprint arXiv:2209.11895, 2022.

<!-- page 15 -->

15
[20] N. Rajaraman, M. Bondaschi, K. Ramchandran, M. Gastpar, and A. V. Makkuva. Transformers on
markov data: Constant depth suffices.arXiv preprint arXiv:2407.17686, 2024.
[21] A. Raventós, M. Paul, F. Chen, and S. Ganguli. Pretraining task diversity and the emergence of
non-bayesian in-context learning for regression.Advances in Neural Information Processing Systems,
36, 2024.
[22] J. Rousseau and K. Mengersen. Asymptotic behaviour of the posterior distribution in overfitted
mixture models.Journal of the Royal Statistical Society Series B: Statistical Methodology, 73(5):
689–710, 2011.
[23] M. E. Sander, R. Giryes, T. Suzuki, M. Blondel, and G. Peyré. How do transformers perform
in-context autoregressive learning ? InForty-first International Conference on Machine Learning,
2024. URLhttps://openreview.net/forum?id=kZbTkpnafR.
[24] A. Svete and R. Cotterell. Transformers can representn-gram language models.arXiv preprint
arXiv:2404.14994, 2024.
[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polo-
sukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, editors,Advances in Neural Information Processing Systems,
volume 30. Curran Associates, Inc., 2017. URLhttps://proceedings.neurips.cc/paper_files/
paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.
[26] J. Von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and
M. Vladymyrov. Transformers learn in-context by gradient descent. InInternational Conference on
Machine Learning, pages 35151–35174. PMLR, 2023.
[27] J. Von Oswald, E. Niklasson, M. Schlegel, S. Kobayashi, N. Zucchet, N. Scherrer, N. Miller, M. San-
dler, M. Vladymyrov, R. Pascanu, et al. Uncovering mesa-optimization algorithms in transformers.
arXiv preprint arXiv:2309.05858, 2023.
[28] S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit
bayesian inference.arXiv preprint arXiv:2111.02080, 2021.
[29] S. Yadlowsky, L. Doshi, and N. Tripuraneni. Pretraining data mixtures enable narrow model selec-
tion capabilities in transformer models.arXiv preprint arXiv:2311.00871, 2023.

<!-- page 16 -->

16
Organization of the Appendix.The Appendix is organized as follows. App. A extends the statistical
analysis of the estimator implemented by the transformer in Prop. 1 and includes omitted proofs. App. B
reports additional experiments and discussions about using more than3layers and varying the number of
headsinthesecondlayer. Additionalattentionmapsfordifferenttasksandbothdisentangledandtrained
transformers as well as the construction are provided in App. C. App. E includes several experiments to
validate Claim 1. App. D details the algorithm used to generate the interleaved Markov chains. App. F
discusses an alternative third layer construction using positional embedding. The construction for non-
contiguous lags is presented in App. G. Finally, App. H explains the single-head construction for the
case of two lags.
AppendixA.Statistical Analysis of the transformer estimator
For the transformer estimator to accurately select the correct lag, the following inequalities (claim 1)
must hold for a lagkand a sequence of lengthTgenerated accordingly:
TX
i=1
˜pi,k >
TX
j=1
˜pj,r ∀, r̸=k;and;r∈ K.
These results enable us to recover the MLE estimator in the high-temperature limit and approximate the
BMA at finite temperatures. Assuming the process is ergodic, and by taking the limit of the inequality
above, we require the following condition:
E
h X ⊤
i−rP ⋆XiP
l∈K X ⊤
i−lP ⋆Xi
i
≤E
h X ⊤
i−kP ⋆Xi
P
l∈K X ⊤
i−lP ⋆Xi
i
forr∈ Kandi≥max(K),
as formalized in Claim 1.
We leave the complete proof of this result as future work, but we have fully validated it empirically in
Section E. We provide here the proofs of Claim 1 for three specific cases.
T wo-lag case.In the case of two lags, we can show the following general result for any two distributions,
PandQ, over1, . . . , kfork≥0.
Lemma 1.
kX
i=1
P(i)Q(i)
P(i) +Q(i) ≤
kX
i=1
P(i) 2
P(i) +Q(i) .(9)
Proof.We first show that
kX
i=1
P(i) 2 −Q(i) 2
P(i) +Q(i) =
kX
i=1
(P(i)−Q(i))(P(i) +Q(i))
P(i) +Q(i) =
kX
i=1
(P(i)−Q(i)) = 0.
Then, by using Cauchy-Schwarz inequality, we obtain:
kX
i=1
P(i)Q(i)
P(i) +Q(i) ≤
vuut
kX
i=1
P(i) 2
P(i) +Q(i)
vuut
kX
i=1
Q(i)2
P(i) +Q(i) =
kX
i=1
P(i) 2
P(i) +Q(i) .
□
The result in the two-lag case follows directly. Letµdenote the distribution of the lag-kinterleaved
processX t, (i.e,µ(s i, sj, sk) =P(X i =s i, Xj =s j, Xk =s k)) . For any lag{r}we have
E
h X ⊤
i−rP ⋆Xi
X ⊤
i−rP ⋆Xi +X ⊤
i−kP ⋆Xi
i
=
X
si−k,si−r,si
µ(si−k, si−r, si) Psi−k,si Psi−r,si
Psi−k,si +P si−r,si
=
X
si−k,si−r
µ(si−k, si−r)
X
si
Psi−k,si Psi−r,si
Psi−k,si +P si−r,si
By applying Lemma 1, we directly obtain
E
h X ⊤
i−rP ⋆Xi
X ⊤
i−rP ⋆Xi +X ⊤
i−kP ⋆Xi
i
≤E
h X ⊤
i−kP ⋆Xi
X ⊤
i−rP ⋆Xi +X ⊤
i−kP ⋆Xi
i
which proves Claim 1 in the case of two lags.

<!-- page 17 -->

17
Independent lags.In the case where all lags inKare such that(Xi−l)l∈K are independent, we can prove
Claim 1. Indeed, in this case, the distribution of the observed lags can be factorized asµ((si−l)l∈K) =Q
l∈K µ(si−l). Thus we have
E
h X ⊤
i−rP ⋆Xi −X ⊤
i−kP ⋆Xi
P
l∈K X ⊤
i−lP ⋆Xi
i
=
X
si,(si−l)l∈K
µ((si−l)l∈K) Psi−k,si(Psi−r,si −P si−k,si)P
l∈K Psi−l,si
=
X
si,(si−l)l∈K
µ((si−l)l∈K,l̸=k,r)µ(si−k)µ(si−r) Psi−k,si(Psi−r,si −P si−k,si)P
l∈K Psi−l,si
Then, by observing thata(a−b) +b(b−a) = (a−b) 2 ≥0, the result follows from:
2
X
si−r,si−l
µ(si−k)µ(si−r) Psi−k,si(Psi−r,si −P si−k,si)P
l∈K Psi−l,si
=
X
si−r,si−l
µ(si−k)µ(si−r) Psi−k,si(Psi−r,si −P si−k,si) +P si−r,si(Psi−k,si −P si−r,si)P
l∈K Psi−l,si
=
X
si−r,si−l
µ(si−k)µ(si−r)(Psi−r,si −P si−k,si)2
P
l∈K Psi−l,si
≥0
We observe that similar techniques can be applied to prove Claim 1 in the case of a symmetric Markov
kernelP ⋆.
No normalization case.Our estimator’s score is computed by aggregating the normalized probabilities
˜pi,k, a necessity imposed by mechanisms such as the softmax normalization in the attention layer. If
we were to use the unnormalized probabilities, we could rely on the following result, which simplifies
Claim 1 by excluding the normalization step.
Lemma 2.LetKbe a subset of integers andX t a stationary interleaved Markov chain of lagk∈ K,
then
E[X ⊤
i−rP ⋆Xi]≤E[X ⊤
i−kP ⋆Xi]forr∈ Kandi≥max(K).(10)
Proof.
E[X ⊤
i−rP ⋆Xi] =
X
si−r,si−k,si
µ(si−r, si−k, si)Psi−r,si
=
X
si−r,si−k,si
µ(si−r, si−k)Psi−k,si Psi−r,si
≤
X
si
s X
si−r,si−k
µ(si−r, si−k)P 2si−k,si
s X
si−r,si−k
µ(si−r, si−k)P 2si−r,si
≤
X
si
sX
si−k
(
X
si−r
µ(si−r, si−k))P 2si−k,si
sX
si−r
(
X
si−k
µ(si−r, si−k))P 2si−r,si ,
where the inequality follows from the Cauchy-Schwarz inequality.
Assuming thatµ(s i−r, si−k)is a coupling of the stationary measureπ, we then have:
E[X ⊤
i−rP ⋆Xi]≤
X
si
sX
si−k
π(si−k)P 2si−k,si
sX
si−r
π(si−r)P 2si−r,si
=
X
si,si−k
π(si−k)P 2
si−k,si
=E[X ⊤
i−kP ⋆Xi].
It remains to prove thatµ(si−r, si−k)is a coupling of the stationary measureπ.
First, let’s assume thatrandkare such thatXi−r andX i−k are independent. In this caseµ(si−r, si−k) =
µ(si−r)µ(si−k)and we have both thatP
si−r µ(si−r, si−k)) =µ(s i−k)and P
si−k
µ(si−r, si−k) =µ(s i−r).
Alternatively, ifrandkare such thatX r andX k come from the same Markov Chain withr > k. We
have thus thatXi−r ∼µandX i−k|Xi−r ∼s ⊤
i−rP l for somel≥0andµ(s i−r, si−k) =π(s i−r)s⊤
i−rP lsi−k.

<!-- page 18 -->

18
SincePis a stochastic matrix, summing oversi−k givesP
si−k
µ(si−r, si−k) =µ(s i−r) =π(s i−r). Finally,
by definition of the stationary distributionπ, summing oversi−r givesP
si−r
µ(si−r, si−k) =π(s i−k).□

<!-- page 19 -->

19
AppendixB.Scaling heads and layers
In this section, we investigate how varying the number of heads in the second layer and the number
of layers affects the model’s performance. We train standard transformers with learned positional and
semantic embeddings in the same setup as reported in Section 6. In Figure 6 (left) we consider the task
given byK={1,2,3}and first show the behavior of the model with 2 layers and different combinations
of heads[1,1],[3,1],[1,3],[3,3] 4, the results show that transformers with2layers can’t solve the task.
Second, we show that increasing the number of layers beyond3does not change the performance. In
Figure 6 (right) instead we consider the task defined byK={1,2,3,4,5}and train transformers with
fewer, equal to, or more thanKheads. As predicted by our construction increasing the number of
heads leads to performances that get closer to the maximum likelihood up to having the number of
heads equal to the number of lags in the setK. Beyond this point adding more heads does not improve
performance, this is expected as ML is optimal. Figures 7a,7b,7c illustrates the attention maps for
a3-layer transformer with only1,2,3heads respectively in the second layer, despite the task having
5lags. Remarkably, even with fewer than K heads, the layers remain consistent with our theoretical
construction, displaying analogous patterns: the first layer extracts transition probabilities, the second
aggregates them, and the third implements the selective head. However, in the case of fewer heads, the
second layer appears to find an efficient way to superpose information—a mechanism we could not yet
interpret. Understanding this behaviour in the second layer remains an open question for future work.
0 50 100
Sequence Length
10 4
10 3
10 2
10 1
100
KL Divergence
= {1, 2, 3}
BMA
ML
 2L [1, 1]
 2L [1, 3]
 2L [3, 1]
 2L [3, 3]
 3L [1, 3, 1]
 4L [1, 3, 1, 1]
 5L [1, 3, 1, 1, 1]
0 50 100
Sequence Length
10 4
10 3
10 2
10 1
100
KL Divergence
= {1, 2, 3, 4, 5}
BMA
ML
 3L [1, 1, 1]
 3L [1, 2, 1]
 3L [1, 3, 1]
 3L [1, 4, 1]
 3L [1, 5, 1]
 3L [1, 6, 1]
 3L [1, 7, 1]
Figure 6.(left)Scalingnumberoflayers:wetrainstandardtransformerswithlearnedpositionandsemantic
embeddings. Transformers with2layers can’t solve the task for any combination of heads. Transformers with
more than3layers achieve the same performance as for 3 layers.(right) Scaling number of heads in the
second layer:we train standard transformers with learned position and semantic embeddings increasing the
number of heads in the second layer. As predicted by the construction increasing the number of layers leads to
performance closer to the Maximum Likelihood.
4With this notation we intend the following[#heads layer 1, . . . ,#heads layer L]

<!-- page 20 -->

20
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 2
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 3 order 1
(a)Attention maps forT3L[1,1,1]andK={1,2,3,4,5}:
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 2
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 3 order 1
(b)Attention maps forT3L[1,2,1]andK={1,2,3,4,5}:
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 2
0
5
10
15
20
25
30
35
40
45 Sequence Index
Attention Layer 3 order 1
(c)Attention maps forT3L[1,3,1]andK={1,2,3,4,5}:
Figure 7.Attention maps for different heads in the second layer andK={1,2,3,4,5}:we observe how
even with fewer heads the transformer learns layers which are consistent with the operations in our construction.
In particular the first layer is still extracting the transition probabilities, the second is aggregating them and the
third one implements the selective head.

<!-- page 21 -->

21
AppendixC.Additional Attention Maps plots
As an additional confirmation for our construction, we report here comparison of the attention maps
after softmax for the task introduced in Figure 1. We compare, trained standard 3-layer attention-only
transformer with learned positional encoding and one attention head per layer, trained disentangled
transformer and our construction. The standard transformer was trained in the same setup already
introduced in Section 6. In Figure 8 we train on data withK= 1,2, we observe a remarkable similarity
between the attention maps of our construction and the trained transformer. This further confirms that
the disentangled transformer is a good proxy to study the residual stream and the flow of information
inside the transformer in a more interpretable way. Moreover, it confirms that our construction is realistic
and aligns with what transformers learn in practice by gradient descent. Moreover, In order to showcase
the adaptivity in-context of the selective induction head depending on the true lag of the input sequence,
in Figures 9, 10, 11 we train on lagsK={1,2,3}and test on sequences generated with each one of the
training lags, similarly in Figures 12, 13, 14 we train on lagsK={1,2,3}and test on each. As expected,
the third layer adapts to the input sequence selecting the correct lag and copying the correspondent
token via the selective induction head.
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 2
Figure 8.Attention mapsK={1,2}and true lagk= 2.

<!-- page 22 -->

22
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 1
Figure 9.Attention mapsK={1,2,3}and true lagk= 1.
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 2
Figure 10.Attention mapsK={1,2,3}and true lagk= 2.

<!-- page 23 -->

23
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 3
Figure 11.Attention mapsK={1,2,3}and true lagk= 3.
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 1
Figure 12.Attention mapsK={1,3,4}and true lagk= 1.

<!-- page 24 -->

24
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 3
Figure 13.Attention mapsK={1,3,4}and true lagk= 3.
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained standard transformer
0
5
10
15
20
25
30
35
40
45 Sequence Index
Trained disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 1
0
5
10
15
20
25
30
35
40
45 Sequence Index
0
5
10
15
20
25
30
35
40
45
Attention Layer 2
Construction disentangled transformer
0
5
10
15
20
25
30
35
40
45
Attention Layer 3 order 4
Figure 14.Attention mapsK={1,3,4}and true lagk= 4.

<!-- page 25 -->

25
AppendixD.Task Details
In this section, we illustrate the algorithm we used to generate batches of new samples at each iteration.
Algorithm 1Generate a batch of N Sequences from Interleaved Markov Chains
Require:N(sequences),T(length),S(state space),K(set of lag values),P ∗ (transition matrix)
Ensure:BatchDofNsequences
1:D ← ∅
2:π←stationary distribution ofP ∗
3:fori= 1toNdo
4:k←Uniform(K)▷Randomly select lag for this sequence
5:X 0 ←Sample fromπ ▷Initialize first state
6:S←[X 0]▷Initialize sequence
7:fort= 1toT−1do
8:ift < ˆkthen
9:X t ←Sample fromπ ▷Sample from stationary distribution
10:else
11:X t ←Sample fromP ∗[Xt−k,:]▷Transition based on lagk
12:AppendX t toS
13:AddStoD
returnD
AppendixE.Empirical v alidation of Claim 1
To empirically validate Claim 1 we first sample a set of12lags uniformly between1and30; we then
sample1000different transition matrices and for each matrix and each lag1000sequences of length1000
according to the respective Interleaved Markov chain. For each lag and each set of sequences, we then
compute the expectation in Claim 1 by averaging the last transition in each sampled sequence. We then
compute the following quantity:
E[ X ⊤
i−kP ⋆Xi
P
l∈K X ⊤
i−lP ⋆Xi
]−max
r̸=k
r∈K
E[ X ⊤
i−rP ⋆XiP
l∈K X ⊤
i−lP ⋆Xi
](11)
and report it in the histogram in Fig.3. We can see that all values in the histogram are positive therefore
confirming our claim. Similarly, the results in Fig.15 report the quantity in the claim for each single lag.
As per our claim, the expected normalized transition probabilities of the true lag is always larger than
the same quantity for any other lag. As a further confirmation of the claim, in Fig.17 and Fig.18 we
report the cumulative average of the normalized transition probabilities along the sequence for a single
sequence. We observe that even with few samples (smallt) the cumulative average for the true order is
always larger than the same quantity for the other lags.

<!-- page 26 -->

26
1
2
7
9
10
11
13
15
16
22
26
28
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 1
True Lag
Other Lags
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 2
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 7
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 9
1
2
7
9
10
11
13
15
16
22
26
28
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 10
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 11
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 13
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 15
1
2
7
9
10
11
13
15
16
22
26
28
lag k
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 16
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 22
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 26
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 28
Figure 15.Expected Normalized Transition Probabilities for|S|= 10:The sampled set of lags is
K={1,2,7,9,10,11,13,15,16,22,26,28}, we sampled10different transition matrices and for each lag and each
matrix sampled1000sequences of length1000. The expected normalized transition probability is always larger
for the true lag.
1
2
7
9
10
11
13
15
16
22
26
28
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 1
True Lag
Other Lags
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 2
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 7
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 9
1
2
7
9
10
11
13
15
16
22
26
28
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 10
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 11
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 13
1
2
7
9
10
11
13
15
16
22
26
28
True lag: 15
1
2
7
9
10
11
13
15
16
22
26
28
lag k
0.080
0.085
0.090
0.095
0.100
0.105
0.110
E[
X ⊤
i−kP ⋆Xi∑
l∈K X ⊤
i−lP ⋆Xi
]
True lag: 16
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 22
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 26
1
2
7
9
10
11
13
15
16
22
26
28
lag k
True lag: 28
Figure 16.Expected Normalized Transition Probabilities for|S|= 25:The sampled set of lags is
K={1,2,7,9,10,11,13,15,16,22,26,28}, we sampled10different transition matrices and for each lag and each
matrix sampled1000sequences of length1000. The expected normalized transition probability is always larger
for the true lag.

<!-- page 27 -->

27
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 1
Lag 1 (True Lag)
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 2
Lag 1
Lag 2 (True Lag)
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 7
Lag 1
Lag 2
Lag 7 (True Lag)
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 9
Lag 1
Lag 2
Lag 7
Lag 9 (True Lag)
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 10
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10 (True Lag)
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 11
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11 (True Lag)
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 13
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13 (True Lag)
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 15
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15 (True Lag)
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 16
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16 (True Lag)
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 22
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22 (True Lag)
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 26
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26 (True Lag)
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 28
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28 (True Lag)
Figure 17.Cumulative average of Normalized Transition Probabilities for|S|= 10:The sampled
set of lags isK={1,2,7,9,10,11,13,15,16,22,26,28}, we report one sequence sampled according to one the
transition matrix. The cumulative average of normalized transition probability quickly becomes larger for the
true lag.
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 1
Lag 1 (True Lag)
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 2
Lag 1
Lag 2 (True Lag)
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 7
Lag 1
Lag 2
Lag 7 (True Lag)
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 9
Lag 1
Lag 2
Lag 7
Lag 9 (True Lag)
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 10
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10 (True Lag)
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 11
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11 (True Lag)
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 13
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13 (True Lag)
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 15
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15 (True Lag)
Lag 16
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
1
t
t∑
i=1
˜pi,k
True Lag 16
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16 (True Lag)
Lag 22
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 22
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22 (True Lag)
Lag 26
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 26
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26 (True Lag)
Lag 28
0 200 400 600 800 1000
Sequence index t
0.05
0.06
0.07
0.08
0.09
0.10
0.11
0.12
True Lag 28
Lag 1
Lag 2
Lag 7
Lag 9
Lag 10
Lag 11
Lag 13
Lag 15
Lag 16
Lag 22
Lag 26
Lag 28 (True Lag)
Figure 18.Cumulative average of Normalized Transition Probabilities for|S|= 25:The sampled
set of lags isK={1,2,7,9,10,11,13,15,16,22,26,28}, we report one sequence sampled according to one the
transition matrix. The cumulative average of normalized transition probability quickly becomes larger for the
true lag.

<!-- page 28 -->

28
AppendixF.Alternative Third layer construction using positional
embedding
In this section, we illustrate an alternative but equivalent construction that implements the same pre-
dictor as in Proposition 1. The first and second layers remain identical, the only difference is in the
third layer which implements the selective sum of the normalized transition probabilities. This selection
mechanism is implemented through the combination of multiple blocks within the third attention matrix,
˜A(3), which, in this alternative construction is structured as follows:
˜A(3) =


0 0
0 A(3)
0 0
0 B(3,1)
0 0
0 B(3,H2)
0 0 0 . . . 0 0
0 0 0 0 . . . 0 0
0 0 0 0 . . . 0 0
0 0 0 . . . 0 0
... ... ... ... ... ... ...
0 0 0 0 . . . 0 0
0 0 0 . . . 0 0


(12)
We can notice how, compared to the construction in Section 5, the blocksB(3,1), . . . , B(3,H2) are now
positioned all in the first column. Moreover, they are not parameterized by the same matrix contrary to
the other construction. The matrixA(3) acts on the positional embedding of the input similarly to the
matrixA (1) in the first layer as in the previous construction:
A(3)
ij =λ 1
(
+1ifj−i+ 1∈ K
−1ifj−i+ 1̸∈ K A(3) =


+λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ +λ -λ -λ -λ -λ -λ -λ -λ -λ+λ +λ +λ -λ -λ -λ -λ -λ -λ -λ-λ +λ +λ +λ -λ -λ -λ -λ -λ -λ-λ -λ +λ +λ +λ -λ -λ -λ -λ -λ-λ -λ -λ +λ +λ +λ -λ -λ -λ -λ-λ -λ -λ -λ +λ +λ +λ -λ -λ -λ-λ -λ -λ -λ -λ +λ +λ +λ -λ -λ-λ -λ -λ -λ -λ -λ +λ +λ +λ -λ-λ -λ -λ -λ -λ -λ -λ +λ +λ +λ


This ensures that the only non-zero entries after softmax will be the ones on the diagonals corresponding
tothelagsseenduringtraining. ThematricesB (3,1), . . . , B(3,H2) areagainresponsibleforthesummation;
each matrix operates on the output of a corresponding head in the second layer. To understand how
this selective sum is implemented, let us consider the output of the first head in the second layerh(2) =
[[esi , ei], ˆh(1)
i , ˆh(2,1)
i , . . . ,ˆh(2,H2)
i ]in our example for the tokens 8,9 and 10:
ˆh(2,1)10 = 1/3 · P
i=4,7,10esi 0 0 0 1 0 0 1 0 0 1 P
i=4,7,10˜si ˜p4,3 ˜p4,2 ˜p4,1 ˜p7,3 ˜p7,2 ˜p7,1 ˜p10,3 ˜p10,2 ˜p10,1 0
ˆh(2,1)9 = 1/2 · es6+es9 0 0 0 0 0 1 0 0 1 0 ˜s6+ ˜s9 0 0 ˜p6,3 ˜p6,2 ˜p6,1 ˜p9,3 ˜p9,2 ˜p9,1 0 0
ˆh(2,1)8 = 1/2 · es5+es8 0 0 0 0 1 0 0 1 0 0 ˜s5+ ˜s8 0 ˜p5,3 ˜p5,2 ˜p5,1 ˜p8,3 ˜p8,2 ˜p8,1 0 0 0
| {z }
ˆm(2,1)8
| {z }
ˆp(2,1)8
and defineˆp(2,h)
i ∈R T as the block ofˆh(2,h)
i which contains the normalized transition probabilities. By
the structure in Eq. 12 we can see how, when computing the attention, the matricesB(3,h) act on these
two blocks:
h(2)⊤
i ˜A(3)h(2)
j =
KX
h=1
p(2,h)⊤
i B(3,h)ej +e iA(3)ej
Here we notice how the difference compared to the construction in Section 5 lies in the fact that, due
to the position, we are not using the copy of the attention to construct the boolean vector but directly
the one-hot encoding of the position. Each operation involvingB(3,h) is still selectively summing the
transition probabilities from the corresponding head, but with a slightly different mechanism. Let us
consider the producth(2)⊤
i ˜A(3)h(2)
i−k which will be the only non zero entries after softmax, and show how
it only sums the transitions of lagk. The main idea is thatB (3,h) a boolean matrix such that each
column sums only the entries containing the transitions for one of the lags. To achieve this, each column
in the matrix follows a pattern in which the entries are spaced at intervals ofK, and the pattern shifts
by one position between successive columns. This shift creates a cyclic arrangement across the columns,
which repeats with frequencyK. For each headh, the matrixB(3,h) is structured such that the product
ˆp(2,h)
i B(3,h) results in a vector where each element is the sum of the transitions for a givenk.In particular,

<!-- page 29 -->

29
the first element of the vector corresponds to the sum ofkmin, theK-th element corresponds to the sum
ofk max, and this pattern repeats cyclically for subsequent elements. To give an example, consider the
productˆp(2,1)
10 B(3,1)e8 in Eq. 14, which sums the transitions stored inˆh(2,1). Notice the structure of
B(3,1); the first column aligns with the transitions of lag1inˆp(2,1)
10 . Given that the index ofˆp(2,1)
i is10
and the index ofej is 8, the sum constructsA(3)
(10,8), which is used to copy the 8th token to predict the
11th if the lag of the sequence is3. Hence,ˆp(2,1)
10 B(3,1)e8 has to sum the transitions of lag3:
ˆp(2,1)⊤
10 B(3,1)e8 = β
3


˜p4,3
˜p4,2
˜p4,1
˜p7,3
˜p7,2
˜p7,1
˜p10,3
˜p10,2
˜p10,1
0


⊤

0 1 0 0 1 0 0 1 0 0
0 0 1 0 0 1 0 0 1 0
1 0 0 1 0 0 1 0 0 1
0 1 0 0 1 0 0 1 0 0
0 0 1 0 0 1 0 0 1 0
1 0 0 1 0 0 1 0 0 1
0 1 0 0 1 0 0 1 0 0
0 0 1 0 0 1 0 0 1 0
1 0 0 1 0 0 1 0 0 1
0 1 0 0 1 0 0 1 0 0




0
0
0
0
0
0
0
1
0
0


= β
3


˜p4,1 + ˜p7,1 + ˜p10,1
˜p4,3 + ˜p7,3 + ˜p10,3
˜p4,2 + ˜p7,2 + ˜p10,2
˜p4,1 + ˜p7,1 + ˜p10,1
˜p4,3 + ˜p7,3 + ˜p10,3
˜p4,2 + ˜p7,2 + ˜p10,2
˜p4,1 + ˜p7,1 + ˜p10,1
˜p4,3 + ˜p7,3 + ˜p10,3
˜p4,2 + ˜p7,2 + ˜p10,2
˜p4,1 + ˜p7,1 + ˜p10,1


⊤

0
0
0
0
0
0
0
1
0
0


(13)
= β
3
  ˜p10,3 + ˜p7,3 + ˜p4,3

(14)
We can see how the operation implemented by this different parameterization is the same then in the
other construction. Therefore the overall predictor remains unchanged. The additional matricesB(3,h),
whichactontheoutputsoftheotherheads ˆh(2,h), performthesameoperationbysummingthetransitions
stored in the outputs of the respective heads. The difference in the construction of the matrixB(3,h)
forh̸= 1is that the columns are shifted byhpositions relative toh= 1. Specifically, for eachh, the
columns are shifted byhpositions compared to the matrixB(3,1). In more generality, the matrixB(3,h)
is constructed as follows:
B(3,h)
ij =β
(
+1,if((i−j−h+ 1) modK= 0)
0,otherwise
wherehtakes into account for the shift.
AppendixG.Construction for any set of lags
The construction illustrated in Section 5 considers only contiguous lags, i.e. set of lags that are intervals
of the positive integers. However, both our interleaved Markov chain framework and the Transformer
construction can be extended to any set of lags. The implemented algorithm is the same, but the
structure of the weights in the different layers becomes more complex because the mechanism with which
the transition probabilities are aggregated depends on the relative distance between the lags in the set.
Due to the difficulties in finding a general formulation of the matrices involved for any set of lags as well
as the optimal number of heads which depends now not only on the number of lags but on the relative
distance between them, we limit this section into illustrate two example withT= 10andK={1,3}and
T= 12K={1,3,4}for which we will visualize the matrices and operations involved.
G.1.Example forK={1,3}
First layer:The structure of the first layer remains unchanged from Section 5. The important difference
is that now the diagonals in the matrixA(1) with positive entries are only 2nd and 4th:
eA(1) =
 logP ⋆⊤ 0
0 A(1)

A(1)
ij =
(
+λifj−i∈ K
−λifj−i̸∈ K.
A(1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
+λ -λ +λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ +λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ +λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ +λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ +λ -λ


The output token at indexiafter the first layer still corresponds to a weighted average of the past tokens
h(0)
i−k fork∈ Kwhere the weights are given by the normalized probabilities˜pi,k:

<!-- page 30 -->

30
ˆh(1)
i =Attn(h (0)
1:T ; ˜A(1))i =
( Pi
j=1 1[i−j∈ K]
Psj ,siP
r∈K Psr ,si
h(0)
j ifi >1
h(0)
1 ifi= 1
=



P
k∈K,k<i
˜pi,kh(0)
i−k ifi >1
h(0)
1 ifi= 1
Due to the lack of the entries on the 3rd diagonal, both the attention and the output token will change
accordingly:
A(1) =


1 0 0 0 0 0 0 0 0 0
1/2 1/2 0 0 0 0 0 0 0 0
1/3 1/3 1/3 0 0 0 0 0 0 0˜p4,3 0 ˜p4,1 0 0 0 0 0 0 00 ˜p5,3 0 ˜p5,1 0 0 0 0 0 00 0 ˜p6,3 0 ˜p6,1 0 0 0 0 00 0 0 ˜p7,3 0 ˜p7,1 0 0 0 00 0 0 0 ˜p8,3 0 ˜p8,1 0 0 00 0 0 0 0 ˜p9,3 0 ˜p9,1 0 00 0 0 0 0 0 ˜p10,3 0 ˜p10,10


ˆh(1) =


˜s1 ˜s2 ˜s3 ˜s4 ˜s5 ˜s6 ˜s7 ˜s8 ˜s9 ˜s10
1 1/2 1/3 ˜p4,3 0 0 0 0 0 00 1/2 1/3 0 ˜p5,3 0 0 0 0 00 0 1/3 ˜p4,1 0 ˜p6,3 0 0 0 00 0 0 0 ˜p5,1 0 ˜p7,3 0 0 00 0 0 0 0 ˜p6,1 0 ˜p8,3 0 00 0 0 0 0 0 ˜p7,1 0 ˜p9,3 00 0 0 0 0 0 0 ˜p8,1 0 ˜p10,3
0 0 0 0 0 0 0 0 ˜p9,1 00 0 0 0 0 0 0 0 0 ˜p10,1
0 0 0 0 0 0 0 0 0 0


(15)
Second layer.Similarly to the construction for contiguous lags, the second layer is responsible for
aggregating the normalized transition probabilities such that they are stored in the embedding of the
current vector for its entire history. The second attention needs to learn an effective way of doing a
convex combination of the input tokens such that the overlap is minimized and all the transitions are
stored without mixing them. Consider the token ati= 10in Eq. 15, summing two consecutive tokens
such as ˆh(1)
9 and ˆh(2)
10 ,contrary to the contiguous case in Eq. 4, does not lead to any mixing due to the
absence of transitions of lag2. Therefore,2attention heads are still sufficient to copy all the transitions
in the past as long as they learn to attend two consecutive tokens each.
Therefore, the optimal way to combine past tokens strictly depends on the number of tokens and the
relative distance between them. Hence, finding a general formula for the positions at which the second
attentionA (2) should be attended to minimize overlap, is challenging and beyond the scope of this work.
Similar considerations apply to the optimal number of heads required, which depends on the solution of
the previous problem. However, the task for arbitrary sets of lags, can always be solved by consider the
correspondent contiguous problem withˆk−min(K+ 1)heads. However there are cases in which we can
leverage the structure given by the distance between the lags to use fewer heads. One example is the one
considered in this section withK={1,3}we only need two heads to achieve optimal sample complexity.
The form of the matrix˜A(2,h) remains unchanged:
˜A(2,h) =


0 0
0 A(2,1) 0
0 0


Considering the case illustrated in Eq. 15, in order for the two heads to copy all the tokens without
overlap, it is sufficient to sum two consecutive tokens and skip two. Therefore, the first attention has
the pattern :(0,0,1,1)while the second one(1,1,0,0)as illustrated in the following:
A(2,1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ +λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ +λ +λ -λ -λ
-λ -λ -λ +λ +λ -λ -λ +λ +λ -λ
-λ -λ -λ -λ +λ +λ -λ -λ +λ +λ


A(2,2) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ +λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ +λ +λ -λ -λ


(16)
where the firstˆkrows and columns are empty because the firstˆkelements of the sequence are sampled
independently from the stationary distribution and therefore no transitions are present.
The attention computes the same operation as before:
ˆh(2)
i =Attn(h (1)
1:T ; ˜A(2,h))i =
iX
j=ˆk
1
h
A(2,h)
ij = +λ
i
Pi
m=1 1
h
A(2,h)
im = +λ
i h(1)
j .
The output of each head is then concatenated into the residual stream. The structure of the third layer
for the general case of any set of lags, also needs some modifications to take into account the particular
structure that was enforced in the second layer. We extend the construction introduced in Section F using

<!-- page 31 -->

31
the positional embeddings. First of all, the matrixA(3) remains unchanged compared to the previous
constructions, it has positive values along the diagonals correspondent to the lags shifted by one position
to take into account the fact that we are predicting the next token in the sequence:
A(3)
ij =
(
+λifj−i+ 1∈ K
−λifj−i+ 1̸∈ K A(3) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ -λ +λ -λ -λ -λ -λ -λ -λ -λ-λ +λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ +λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ +λ -λ +λ -λ -λ -λ -λ-λ -λ -λ -λ +λ -λ +λ -λ -λ -λ-λ -λ -λ -λ -λ +λ -λ +λ -λ -λ-λ -λ -λ -λ -λ -λ +λ -λ +λ -λ-λ -λ -λ -λ -λ -λ -λ +λ -λ +λ


(17)
The matrixB (3) is responsible for the sum of the normalized transitions; each block operates on the
output of a corresponding head in the second layer. To understand how, consider the following tokens
in output of the first head in the second layer:
ˆh(2)10 = 1/4 · P
i=5,6,9,10esi 0 0 0 0 1 1 0 0 1 1 P
i=5,6,9,10˜si 0 ˜p5,3 ˜p6,3 ˜p5,1 ˜p6,1 ˜p9,3 ˜p10,3 ˜p9,1 ˜p10,1 0
ˆh(2)9 = 1/4 · P
i=4,5,8,9esi 0 0 0 1 1 0 0 1 1 0 P
i=4,5,8,9˜si ˜p4,3 ˜p5,3 ˜p4,1 ˜p5,1 ˜p8,3 ˜p9,3 ˜p8,1 ˜p9,1 0 0
ˆh(2)8 = 1/3 · P
i=4,7,8esi 0 0 0 1 0 0 1 1 0 0 P
i=4,7,8˜si ˜p4,3 0 ˜p4,1 ˜p7,3 ˜p8,3 ˜p7,1 ˜p8,1 0 0 0
| {z }
ˆp(2)8
(18)
By the structure of˜A(3) we can see how, when computing the attention, the matricesB(3,h) are applied
on the positional encodingej and the result is multiplied byˆp(2,h)
i :
h(2)⊤
i ˜A(3)h(2)⊤
j =
KX
h=1
p(2,h)⊤
i B(3,h)ej +e iA(3)ej (19)
whereB (3,h) is selectively summing the transition probabilities from the corresponding head. As for the
simpler case of contiguous lags, for the sum to be selective it must hold thath(2)⊤
i ˜A(3)h(2)
i−k+1 ∝P
j≤i ˜pj,k,
wherei−k+ 1are the only non-zero entries due toA (3) after applying softmax. As before,B (3,h) is
a boolean matrix such that each column sums only the entries containing the transitions for one of the
lags. To achieve this, the matrix needs to learn the same pattern as in the attention of the second
layerA (2), which was used to sum the vectors and create the current inputs. Each column is shifted
by one position, and they cyclically repeat with frequencyK. In the following example, we consider
ˆp(2,1)
10 B(3,1)e8 in Eq. 20, which sums the transitions stored inˆh(2,1) in the entryA(3)
10,8:
ˆp(2,1)⊤
10 B(3,1)e8 = β
4


0
˜p5,3
˜p6,3
˜p5,1
˜p6,1
˜p9,3
˜p10,3
˜p9,1
˜p10,1
0


⊤

0 1 1 0 0 1 1 0 0 1
0 0 1 1 0 0 1 1 0 0
0 0 0 1 1 0 0 1 1 0
0 0 0 0 1 1 0 0 1 1
0 0 0 0 0 1 1 0 0 1
0 0 0 0 0 0 1 1 0 0
0 0 0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0




0
0
0
0
0
0
0
1
0
0


= β
4


0
˜p5,3
˜p6,3
˜p5,1
˜p6,1
˜p9,3
˜p10,3
˜p9,1
˜p10,1
0


⊤

0
1
1
0
0
1
1
0
0
0


= β
4
  ˜p10,3 + ˜p9,3 + ˜p6,3 + ˜p5,3

(20)
Notice how the matrixB(3,1) has the same pattern asA(2,1) in Eq. 16 but along the columns instead of
the rows. Intuitively, it makes sense since we need to sum the same entries resulting from the sum in the
previous attention. The matrixB(3,2) acting on the second head will have the same pattern but shifted
by two positions in order to have the same pattern asA(2,2).
G.2.Example withK={1,3,4}
The case of two lagsK={1,3}, despite not being contiguou,s does not adequately represent the general
case. Indeed due to the structure, we could always sum two consecutive tokens and therefore recover
optimal performance using2heads. It is helpful to also consider a case where the lags do not form a

<!-- page 32 -->

32
structure that allows for fewer heads in the construction. For example the case of three lagsK={1,3,4}
andT= 12:
First layer:The main structure of the first layer remains unchanged, the diagonals in the matrixA(1)
with positive entries are 2nd, 3rd and 4th:
eA(1) =
 logP ⋆⊤ 0
0 A(1)

A(1)
ij =
(
+λifj−i∈ K
−λifj−i̸∈ K.
A(1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
+λ -λ +λ -λ -λ -λ -λ -λ -λ -λ -λ
+λ +λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ +λ +λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ +λ +λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ +λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ +λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ +λ -λ +λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ +λ -λ +λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ +λ -λ +λ


(21)
The output token at indexiafter the first layer still corresponds to a weighted average of the past tokens
h(0)
i−k fork∈ Kwhere the weights are given by the normalized probabilities˜pi,k:
A(1) =


1 0 0 0 0 0 0 0 0 0 0 01/2 1/2 0 0 0 0 0 0 0 0 0 01/3 1/3 1/3 0 0 0 0 0 0 0 0 01/4 1/4 1/4 1/4 0 0 0 0 0 0 0 0˜p5,4˜p5,3 0 ˜p5,1 0 0 0 0 0 0 0 00 ˜p6,4˜p6,3 0 ˜p6,1 0 0 0 0 0 0 00 0 ˜p7,4˜p7,3 0 ˜p7,1 0 0 0 0 0 00 0 0 ˜p8,4˜p8,3 0 ˜p8,1 0 0 0 0 00 0 0 0 ˜p9,4 ˜p9,3 0 ˜p9,1 0 0 0 00 0 0 0 0 ˜p10,4˜p10,3 0 ˜p10,1 0 0 00 0 0 0 0 0 ˜p11,4˜p11,3 0 ˜p11,1 0 00 0 0 0 0 0 0 ˜p12,4˜p11,3 0 ˜p11,10


ˆh(1) =


˜s1 ˜s2 ˜s3 ˜s4 ˜s5 ˜s6 ˜s7 ˜s8 ˜s9 ˜s10 ˜s11 ˜s121 1/21/31/4˜p5,4 0 0 0 0 00 1/21/31/4˜p5,3˜p6,4 0 0 0 0 0 00 0 1/31/4 0 ˜p6,3˜p7,4 0 0 0 0 00 0 0 1/4˜p5,1 0 ˜p7,3˜p8,4 0 0 0 00 0 0 0 0 ˜p6,1 0 ˜p8,3˜p9,4 0 0 00 0 0 0 0 0 ˜p7,1 0 ˜p9,3˜p10,4 0 00 0 0 0 0 0 0 ˜p8,1 0 ˜p10,3˜p11,4 00 0 0 0 0 0 0 0 ˜p9,1 0 ˜p11,3˜p12,40 0 0 0 0 0 0 0 0 ˜p10,1 0 ˜p12,30 0 0 0 0 0 0 0 0 0 ˜p11,1 00 0 0 0 0 0 0 0 0 0 0 ˜p12,10 0 0 0 0 0 0 0 0 0 0 0


(22)
Second layer, aggregation of transition probabilities:In this case, we can’t use the fact that 2
consecutive tokens can be summed without mixing the information. In fact, summing the last tow tokens
ˆh(1)
11 and ˆh(2)
12 would now result in the mixing of˜p11,3 and˜p12,4. In order to avoid this, the only possibility
is to sum one token every4, similar to the case where we would have4contiguous lags. This solution
is less efficient because summing each4tokens while having the missing transition corresponding to lag
2leaves an empty element in the embedding of the token and adds an additional head, increasing both
the dimension and the number of parameters. This means that even if we only have 3 lags, in order to
not have any overlap, we still need 4 attention heads for our construction to not mix the information.
Each head has the pattern(0,0,0,1)shifted by one position as if the lags would be 1,2,3,4:
A(2,1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ +λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ +λ


A(2,2) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ +λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ +λ


A(2,3) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ +λ -λ -λ


A(2,4) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ +λ -λ -λ -λ



<!-- page 33 -->

33
Third layer.For the third layer we use again the construction with the positional encoding that was
introduced in App F:
A(3)
ij =λ 1
(
+1ifj−i+ 1∈ K
−1ifj−i+ 1̸∈ K A(3) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ +λ -λ +λ -λ -λ -λ -λ -λ -λ -λ -λ-λ +λ +λ -λ +λ -λ -λ -λ -λ -λ -λ -λ-λ -λ +λ +λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ -λ +λ +λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ -λ +λ +λ -λ +λ -λ -λ -λ -λ-λ -λ -λ -λ -λ +λ +λ -λ +λ -λ -λ -λ-λ -λ -λ -λ -λ -λ +λ +λ -λ +λ -λ -λ-λ -λ -λ -λ -λ -λ -λ +λ +λ -λ +λ -λ


For the selective sum, the matricesB(3,1), . . . , B(3,4) have the same form as before but considering now
the fact that even if we only have 3 lags in the set, we still need4heads:
B(3,h)
ij =β
(
+1,if

(i−j−h+ 1) mod ˆk−min(K) + 1 = 0

0,otherwise
The computation related to the matrixB(3,1) in the attention are reported in the following:
ˆp(2,1)⊤
12 B(3,1)e10 = β
2


0
0
0
˜p8,4
˜p8,3
0
˜p8,1
˜p12,4
˜p12,3
0
˜p12,1
0


⊤

0 1 0 0 0 1 0 0 0 1 0 0
0 0 1 0 0 0 1 0 0 0 1 0
0 0 0 1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1 0 0 0
0 1 0 0 0 1 0 0 0 1 0 0
0 0 1 0 0 0 1 0 0 0 1 0
0 0 0 1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1 0 0 0
0 1 0 0 0 1 0 0 0 1 0 0
0 0 1 0 0 0 1 0 0 0 1 0
0 0 0 1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1 0 0 0




0
0
0
0
0
0
0
0
0
1
0
0


= β
2


0
0
0
˜p8,4
˜p8,3
0
˜p8,1
˜p12,4
˜p12,3
0
˜p12,1
0


⊤

0
0
0
0
1
0
0
0
1
0
0
0


= β
2
  ˜p12,3 + ˜p8,3

AppendixH.Construction for two lags and Single head
In the constructions illustrated so far, in order to store all the transitions in the history of the current
token and not lose any information, we had to scale the number of heads at least as the number of lags
in the taskK. This allows to achieve optimal sample complexity. However, driven by experimental
evidence, we observed that scaling the number of heads as the number of lags is not necessary in the
special case of|K|= 2. In this case indeed, there exists a solution, which transformers can learn, that
achieves optimal sample complexity using only one head in the second layer. In the following we will
report the construction that proves the previous statement while illustrating it for the case ofK={1,3}
analogous to Section G.
First layer:The structure of the first layer remains unchanged from Section 5. The important difference
is that now the diagonals in the matrixA(1) with positive entries are only 2nd and 4th:
eA(1) =
 logP ⋆⊤ 0
0 A(1)

A(1)
ij =
(
+λifj−i∈ K
−λifj−i̸∈ K.
A(1) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
+λ -λ +λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ +λ -λ +λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ +λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ -λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ -λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ -λ +λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ +λ -λ +λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ +λ -λ +λ -λ


Remarking that each input elementsi is encoded ash (0)
i = [e si , ei]∈ {0,1} |S|+T, the output token at
indexiafter the first layer still corresponds to a weighted average of the past tokensh(0)
i−k fork∈ K
where the weights are given by the normalized probabilities˜pi,k:
ˆh(1)
i =Attn(h (0)
1:T ; ˜A(1))i =
( Pi
j=1 1[i−j∈ K]
Psj ,siP
r∈K Psr ,si
h(0)
j ifi >1
h(0)
1 ifi= 1
=



P
k∈K,k<i
˜pi,kh(0)
i−k ifi >1
h(0)
1 ifi= 1

<!-- page 34 -->

34
Due to the lack of the entries on the 3rd diagonal, both the attention and the output token will change
accordingly:
A(1) =


1 0 0 0 0 0 0 0 0 0
1/2 1/2 0 0 0 0 0 0 0 0
1/3 1/3 1/3 0 0 0 0 0 0 0˜p4,3 0 ˜p4,1 0 0 0 0 0 0 00 ˜p5,3 0 ˜p5,1 0 0 0 0 0 00 0 ˜p6,3 0 ˜p6,1 0 0 0 0 00 0 0 ˜p7,3 0 ˜p7,1 0 0 0 00 0 0 0 ˜p8,3 0 ˜p8,1 0 0 00 0 0 0 0 ˜p9,3 0 ˜p9,1 0 00 0 0 0 0 0 ˜p10,3 0 ˜p10,10


ˆh(1) =


˜s1 ˜s2 ˜s3 ˜s4 ˜s5 ˜s6 ˜s7 ˜s8 ˜s9 ˜s10
1 1/2 1/3 ˜p4,3 0 0 0 0 0 00 1/2 1/3 0 ˜p5,3 0 0 0 0 00 0 1/3 ˜p4,1 0 ˜p6,3 0 0 0 00 0 0 0 ˜p5,1 0 ˜p7,3 0 0 00 0 0 0 0 ˜p6,1 0 ˜p8,3 0 00 0 0 0 0 0 ˜p7,1 0 ˜p9,3 00 0 0 0 0 0 0 ˜p8,1 0 ˜p10,3
0 0 0 0 0 0 0 0 ˜p9,1 00 0 0 0 0 0 0 0 0 ˜p10,1
0 0 0 0 0 0 0 0 0 0


(23)
Second Layer:the second layer uses only the first head ˜A(2) = ˜A(2,1) compared to the construction
illustrated in Section G and the matrixA(2) =A (2,1) remains identical.
˜A(2) =


0 0
0 A(2) 0
0 0

 A(2) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ -λ -λ -λ -λ
-λ -λ -λ +λ +λ -λ -λ -λ -λ -λ
-λ -λ -λ -λ +λ +λ -λ -λ -λ -λ
-λ -λ -λ -λ -λ +λ +λ -λ -λ -λ
-λ -λ -λ +λ -λ -λ +λ +λ -λ -λ
-λ -λ -λ +λ +λ -λ -λ +λ +λ -λ
-λ -λ -λ -λ +λ +λ -λ -λ +λ +λ


For the case of two lags examined here, we can derive a mathematical expression for the matrixA(2)
which is valid for any set of lags. For convenience we introduce¯k= minK:
A(2)
i,j =



0,ifi < ˆkorj < ˆk,
λ,ifj≤iand

|i−j|mod 2( ˆk− ¯k)

<( ˆk− ¯k),
0,otherwise
(24)
where the first condition ensures that all elements in the firstˆkrows and the firstˆkcolumns of the matrix
are zero. The conditionj≤iensures that only the lower triangular part of the matrix (including the
diagonal). Finally, the condition(|i−j|mod 2d)< dintroduces a periodic pattern within the lower
triangular part of the matrix. The modulo operation creates a repeating cycle of length2(ˆk− ¯k), and
the condition<( ˆk− ¯k)determines whether to place a one or a zero within each cycle segment.
Third layer:the third layer instead has a different structure. As before, there are only two non-
zero blocksA (3) andB (3), but the latter appears in the transpose position compared to the previous
constructions:
˜A(3) =


0 0 0 0
0 A(3) 0 B(3)0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0


A(3) =


-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ-λ -λ -λ -λ -λ -λ -λ -λ -λ -λ+λ -λ +λ -λ -λ -λ -λ -λ -λ -λ-λ +λ -λ +λ -λ -λ -λ -λ -λ -λ-λ -λ +λ -λ +λ -λ -λ -λ -λ -λ-λ -λ -λ +λ -λ +λ -λ -λ -λ -λ-λ -λ -λ -λ +λ -λ +λ -λ -λ -λ-λ -λ -λ -λ -λ +λ -λ +λ -λ -λ-λ -λ -λ -λ -λ -λ +λ -λ +λ -λ-λ -λ -λ -λ -λ -λ -λ +λ -λ +λ


B(3) =β


0 0 0 0 0 0 0 0 0 01 0 0 0 0 0 0 0 0 01 1 0 0 0 0 0 0 0 0−11 1 0 0 0 0 0 0 0−1−11 1 0 0 0 0 0 01 −1−11 1 0 0 0 0 01 1 −1−11 1 0 0 0 0−11 1 −1−11 1 0 0 0−1−11 1 −1−11 1 0 01 −1−11 1 −1−11 1 0


(25)
ThematrixA (3) remainsunchangedandhaspositiveentriesalongthediagonals, correspondingtothelags
shifted by one position. The main difference lies in the matrixB(3), which now includes negative entries in
positions that previously contained zeros. The general formulation of this matrix is the following:
A(3)
ij =
(
+λifj−i+ 1∈ K
−λifj−i+ 1̸∈ K B(3)
i,j =



0,ifj≥i,
+β,if

(i−j−1) mod 2( ˆk− ¯k)

<( ˆk− ¯k),
−β,otherwise.
So far the matrixB(3) has been structured such that it would compute the selective sum of the normalized
transition of the lag of the corresponding entry in the attention:˜A(3)
ij ∝h (2)⊤
i ˜A(3)h(2)
i−k+1 ∝P
j≤i ˜pj,k,
wherei−k+ 1are the only non-zero entries due toA (3) after applying softmax. To understand the
impact of having negative entries, let us consider the previous example for the case ofK={1,3}and

<!-- page 35 -->

35
the output of the second attention for the 8th,9th and 10th token:
ˆh(2)10 = 1/4 · P
i=5,6,9,10esi 0 0 0 0 1 1 0 0 1 1 P
i=5,6,9,10˜si 0 ˜p5,3 ˜p6,3 ˜p5,1 ˜p6,1 ˜p9,3 ˜p10,3 ˜p9,1 ˜p10,1 0
ˆh(2)9 = 1/4 · P
i=4,5,8,9esi 0 0 0 1 1 0 0 1 1 0 P
i=4,5,8,9˜si ˜p4,3 ˜p5,3 ˜p4,1 ˜p5,1 ˜p8,3 ˜p9,3 ˜p8,1 ˜p9,1 0 0
ˆh(2)8 = 1/3 · P
i=4,7,8esi 0 0 0 1 0 0 1 1 0 0 P
i=4,7,8˜si ˜p4,3 0 ˜p4,1 ˜p7,3 ˜p8,3 ˜p7,1 ˜p8,1 0 0 0
| {z }
ˆp(2)8
and defineˆp(2)
i ∈R T as the block of ˆh(2)
i which contains the normalized transition probabilities such
that ˆh(2)
i = [P
j∈Ni esj ,ˆm(2)
i ,P
j∈Ni ˜sj,ˆp(2)
i ]. By the different structure in Eq. 25 we can see how,
when computing the attention for the concatenated tokensh(2)
i = [[e si , ei], ˆh(1)
i , ˆh(2)
i ], the order of the
multiplication has been reversed (ei is now on the left) and the matrixB(3) is applied toˆp(2)
j :
h(2)⊤
i ˜A(3)h(2)
j =e ⊤
i B(3)ˆp(2)
j +e iA(3)ej .(26)
To better understand the implications of the reverse order in the multiplication and the presence of
negative entries, consider the producte⊤
10B(3,1)ˆp(2)
8 in Eq. 27, which sums the transitions stored inˆh(2,1)
which, after softmax, will correspond toA(3)
10,8:
e⊤
10B(3)ˆp(2)⊤
8 = β
3


0
0
0
0
0
0
0
0
0
1


⊤

0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0
−1 1 1 0 0 0 0 0 0 0
−1 −1 1 1 0 0 0 0 0 0
1 −1 −1 1 1 0 0 0 0 0
1 1 −1 −1 1 1 0 0 0 0
−1 1 1 −1 −1 1 1 0 0 0
−1 −1 1 1 −1 −1 1 1 0 0
1 −1 −1 1 1 −1 −1 1 1 0




˜p4,3
0
˜p4,1
˜p7,3
˜p8,3
˜p7,1
˜p8,1
0
0
0


= β
3


1
−1
−1
1
1
−1
−1
1
1
0


⊤

˜p4,3
0
˜p4,1
˜p7,3
˜p8,3
˜p7,1
˜p8,1
0
0
0


= β
3
  ˜p8,3 + ˜p7,3 + ˜p4,3 − ˜p8,1 − ˜p7,1 − ˜p4,1

(27)
where we observe how, the product involvingB(3), is now not only computing the sum of transitions for
the lag3as for the previous constructions to copy the 8th to predict the 11th, it is also subtracting all
the transitions of lag3. To fully understand the implications, we also consider the entry of the attention
correspondent to the other lag in the set,1and the relative producte⊤
10B(3,1)ˆp(2)
10 :
e⊤
10B(4)ˆp(2)⊤
10 = β
3


0
0
0
0
0
0
0
0
0
1


⊤

0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0
−1 1 1 0 0 0 0 0 0 0
−1 −1 1 1 0 0 0 0 0 0
1 −1 −1 1 1 0 0 0 0 0
1 1 −1 −1 1 1 0 0 0 0
−1 1 1 −1 −1 1 1 0 0 0
−1 −1 1 1 −1 −1 1 1 0 0
1 −1 −1 1 1 −1 −1 1 1 0




0
˜p5,3
˜p6,3
˜p5,1
˜p6,1
˜p9,3
˜p10,3
˜p9,1
˜p10,1
0


= β
4


1
−1
−1
1
1
−1
−1
1
1
0


⊤

0
˜p5,3
˜p6,3
˜p5,1
˜p6,1
˜p9,3
˜p10,3
˜p9,1
˜p10,1
0


= β
4
  ˜p10,1 + ˜p9,1 + ˜p6,1 + ˜p5,1 − ˜p10,3 − ˜p9,3 − ˜p6,3 − ˜p5,3

(28)
Therefore, both products contain the sum of the transitions for the respective lags and the negative sum
of the other lag, and notice how they are all computed on different elements of the past. The first one
contains the transitions for the tokens 8,7,4, whereas the second one contains the remaining ones 10,9,6,5.
If we now compute the softmax:
A10,10 =
exp

e⊤
10B(3)ˆp(2)
10 +λ

P9
i=1
i̸=8
exp

e⊤
10B(3)ˆp(2)
j −λ

+ exp

e⊤
10B(3)ˆp(2)
8 +λ

+ exp

e⊤
10B(3)ˆp(2)
10 +λ

=
exp

e⊤
10B(3)ˆp(2)
10

P9
i=1
i̸=8
exp

e⊤
10B(3)ˆp(2)
j −2λ

+ exp

e⊤
10B(3)ˆp(2)
8

+ exp

e⊤
10B(3)ˆp(2)
10


<!-- page 36 -->

36
Considering the limit ofλ→ ∞:
lim
λ→∞
A10,10 =
exp

e⊤
10B(3)ˆp(2)
10

exp

e⊤
10B(3)ˆp(2)
8

+ exp

e⊤
10B(3)ˆp(2)
10

= 1
exp

e⊤
10B(3)ˆp(2)
8 −e ⊤
10B(3)ˆp(2)
10

+ 1
= 1
exp

+ β
3
P
i∈{8,7,4} ˜pi,3 − β
3
P
i∈{8,7,4} ˜pi,1 − β
4
P
i∈{10,9,6,5} ˜pi,1 +P
i∈{10,9,6,5} ˜pi,3

+ 1
which is considering all the possible transitions as in the case of two heads, therefore achieving optimal
sample complexity.
