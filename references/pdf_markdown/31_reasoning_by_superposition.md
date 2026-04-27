# references/31_reasoning_by_superposition.pdf

<!-- page 1 -->

Reasoning by Superposition: A Theoretical
Perspective on Chain of Continuous Thought
Hanlin Zhu∗
UC Berkeley
hanlinzhu@berkeley.edu
Shibo Hao∗
UCSD
s5hao@ucsd.edu
Zhiting Hu
UCSD
zhh019@ucsd.edu
Jiantao Jiao
UC Berkeley
jiantao@berkeley.edu
Stuart Russell
UC Berkeley
russell@cs.berkeley.edu
Yuandong Tian
Meta AI
yuandong@meta.com
Abstract
Large Language Models (LLMs) have demonstrated remarkable performance
in many applications, including challenging reasoning problems via chain-of-
thought (CoT) techniques that generate “thinking tokens” before answering the
questions. While existing theoretical works demonstrate that CoT with discrete
tokens boosts the capability of LLMs, recent work on continuous CoT lacks a
theoretical understanding of why it outperforms discrete counterparts in various
reasoning tasks, such as directed graph reachability, a fundamental graph reasoning
problem that includes many practical domain applications as special cases. In this
paper, we prove that a two-layer transformer withDsteps of continuous CoT can
solve the directed graph reachability problem, whereD is the diameter of the graph,
while the best known result of constant-depth transformers with discrete CoT
requires O(n2) decoding steps where n is the number of vertices (D < n ). In our
construction, each continuous thought vector is a superposition state that encodes
multiple search frontiers simultaneously (i.e.,parallel breadth-first search (BFS)),
while discrete CoT must choose a single path sampled from the superposition state,
which leads to a sequential search that requires many more steps and may be trapped
in local solutions. We also performed extensive experiments to verify that our
theoretical construction aligns well with the empirical solution obtained via training
dynamics. Notably, encoding of multiple search frontiers as a superposition state
automaticallyemergesin training continuous CoT, without explicit supervision to
guide the model to explore multiple paths simultaneously. Our code is available at
https://github.com/Ber666/reasoning-by-superposition.
1 Introductions
Large language models (LLMs) have shown strong performance in many reasoning tasks, especially
when empowered with chain-of-thought (CoT) [Wei et al., 2022] (e.g., hard problems like AIME
and math proving). However, they also struggle with tasks that require more sophisticated reasoning
capability [Kambhampati, 2024], e.g., reasoning and planning problems of increasing scales [Zheng
et al., 2024, Xie et al., 2024], even with CoT [Valmeekam et al., 2024, Zhou et al., 2025].
It remains an open problem how to expand existing discrete CoT to solve more complex reasoning
problems. Recently, Hao et al. [2024] proposes COCONUT(chain-of-continuous-thought) that uses
continuous latent thoughts for reasoning, showing empirical performance boost on synthetic tasks
*Equal contributions.
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
arXiv:2505.12514v3  [cs.LG]  1 Nov 2025

<!-- page 2 -->

such as directed graph reachability (i.e., given a specification of a directed graph and one starting
node, determine which candidate destination node is reachable), as well as strong performance on
real-world math reasoning benchmarks such as GSM8K [Cobbe et al., 2021]. Interestingly, COCONUT
shows preliminary results that continuous latent thought may store multiple candidate search frontiers
simultaneously, before the final answer is reached. This is in sharp contrast with discrete CoT, in
which each discrete thought token has to be sampled (or “realized”) before feeding into LLMs in an
autoregressive manner. However, the expressive power and the mechanism of continuous thought
still remain elusive and lack a deep understanding.
In this work, we explore the mechanism of COCONUTfor the problem ofgraph reachability, i.e.,
whether there exists a path from given start and end nodes in a directed graph. The problem setting is
general [Ye et al., 2024, Hao et al., 2024, Zhou et al., 2025] and includes many important theoretical
problems (e.g., Turing machine halting problem) and practical use cases (e.g., knowledge graph).
Given this setting, we proved that a two-layer transformer with D steps of continuous thought can
solve graph reachability for graphs of n vertices, where D < n is the graph’s diameter (longest
path length between two nodes). In contrast, for graph reachability, the best existing result on
constant-depth transformers with discrete CoT requires O(n2) steps [Merrill and Sabharwal, 2023a].
Intuitively, in our construction, each latent thought vector is a superposition of multiple valid search
traces, and thus can performimplicit parallel searchon the graph in each autoregressive step. The
continuous thoughts can be regarded as “superposition states” in quantum mechanics [Böhm, 2013],
storing multiple search frontiers simultaneously, and thus enabling efficient breadth-first search (BFS).
In contrast, discrete thought tokens can be viewed as “collapsed states” from superpositions. This
forces the model to choose a branch deterministically, yielding either an incorrect greedy search
or a depth-first style search with backtracking, which requires more computation. Unlike previous
theoretical work that constructs positional encodings specifically for a given problem or even for a
given input length, our construction works for widely-used positional encodings in practice, such as
sinusoidal positional encoding [Vaswani et al., 2017] and rotary position embedding [Su et al., 2024].
Moreover, we show that our theoretical construction can be achieved in gradient-based training.
Specifically, a two-layer transformer with continuous CoT outperforms a 12-layer one with discrete
CoT on graph reachability. An inspection of attention patterns and their underlying representation
demonstrates that the continuous thought indeed encodes multiple plausible search frontiers in parallel
in superposition states. Notably, such a superpositional representation automaticallyemergesfrom
training only with the optimal path of graph reachability, without strong supervision that aligns the
latent thought vectors with other plausible search traces.
1.1 Related works
LLM reasoning in text and latent spaces.LLM’s reasoning capability can be significantly boosted
by chain-of-thought (CoT) [Wei et al., 2022], which allows LLMs to explicitly output intermediate
thoughts in text space before predicting the final answer. CoT includes prompt-only methods [Khot
et al., 2022, Zhou et al., 2022] and training with samples containing intermediate thoughts [Yue et al.,
2023, Yu et al., 2023, Wang et al., 2023b, Shao et al., 2024]. Besides text-based CoT, many previous
works also study LLM reasoning in the latent space [Goyal et al., 2023, Wang et al., 2023c, Pfau
et al., 2024, Su et al., 2025] where the intermediate thoughts do not necessarily correspond to textual
tokens. In particular, Hao et al. [2024] proposed to train LLMs to reason in a continuous latent space,
which outperforms discrete CoT on graph reasoning tasks, especially for graphs with high branching
factors. Based on empirical case studies in Hao et al. [2024], continuous thoughts are hypothesized
to encode multiple plausible search frontiers simultaneously. In this work, we formally study the
mechanism and theoretically show that transformers equipped with continuous thoughts benefit from
superposition states during reasoning.
Expressivity of transformers.There is a long line of work studying the expressivity of transform-
ers [Yun et al., 2019, Bhattamishra et al., 2020a,b, Pérez et al., 2021, Likhosherstov et al., 2021,
Yao et al., 2021, Edelman et al., 2022, Akyürek et al., 2022, Merrill and Sabharwal, 2023b]. A
more recent line of work shows CoT can improve the expressivity of transformers [Liu et al., 2022,
Feng et al., 2023, Merrill and Sabharwal, 2023a, Li et al., 2024]. For example, Liu et al. [2022]
studies low-depth transformer expressivity for semi-automata, of which the setting corresponds to
one CoT step. Feng et al. [2023] shows that constant-depth transformers with CoT can solve certain
2

<!-- page 3 -->

P-complete problems. Li et al. [2024] further provides constructions of constant-depth transformer
for each problem in P/poly with CoT. Merrill and Sabharwal [2023a] studies the expressivity with
different lengths of CoT, showing that logarithmic steps of CoT in input length can expand the upper
bound of constant-depth transformer expressivity from TC0 to L, while a linear number of steps can
further expand the upper bound to NC1-complete. While these expressivity results mainly focus on
discrete CoT, theoretical studies on continuous CoT [Hao et al., 2024] are rare, which our work is
focused on. Gozeten et al. [2025] studies the expressivity of one-layer transformers with continuous
CoT on the minimum non-negative sum problem, demonstrating the superposition mechanism in
the arithmetic domain, which complements our results on the graph reachability problem. While
their construction requires an exponentially large embedding dimension, our theoretical construction
requires only a linear embedding dimension in the graph size. Moreover, unlike many previous
works that construct problem-specific (or even length-specific) positional encodings, our construction
applies to practical positional encodings, such as sinusoidal [Vaswani et al., 2017] and RoPE [Su
et al., 2024].
Reasoning as graph problems.Graph problems are essential to understand LLM reasoning
capability since many reasoning problems can be abstracted as computational graphs [Ye et al.,
2024, Zhou et al., 2025] where relational data fed into transformers can be modeled as edges [Wang
et al., 2024a,b, Guo et al., 2025]. Many previous works have shown that pretrained LLMs can deal
with reasoning tasks in graphs, but may still have difficulties with more complicated tasks [Wang
et al., 2023a, Guo et al., 2023, Fatemi et al., 2023, Sanford et al., 2024, Luo et al., 2024, Dai et al.,
2024, Cohen et al., 2025]. Other works study how transformers solve classic and fundamental graph
problems, such as graph reachability [Merrill and Sabharwal, 2023a, 2025], shortest path [Cohen et al.,
2025], etc. For example, Cohen et al. [2025] shows that a two-layer transformer can leverage spectral
decomposition of the line graph to predict the shortest path on small-scale undirected graphs. Merrill
and Sabharwal [2025] shows that a log-depth transformer can solve directed graph reachability, which
constant-depth transformers can not solve. For a constant-depth transformer, Merrill and Sabharwal
[2023a] shows directed graph reachability can be solved withO(n2) CoT steps where n is the number
of vertices, while it remains unclear whether a smaller number of discrete CoT steps can solve the
task. On the contrary, our work shows that a two-layer transformer can solve graph reachability for a
D-diameter graph withDcontinuous thought steps.
2 Preliminaries
Basic notations.For any integer N >0 , we use [N] to denote the set {1,2, . . . , N}. For any finite
set X , let |X | denote the cardinality of X . Let R be the set of real numbers. We use 1{·} to denote
the indicator function. Without further clarification, we use lower-case bold letters (e.g., x, θ) and
upper-case bold letters (e.g., W,U ) to denote vectors and matrices, respectively. In particular, we
use Id to denote a d×d identity matrix, use 0m×n (or 0m) to denote an m×n zero matrix (or an m-
dimensional zero vector), and useei to denote a one-hot vector of which thei-th entry is one and other
entries are all zero, where the dimension ofei can be inferred from the context. We also use∥·∥ ∞ and
∥ · ∥2 to represent L∞ norm and L2 norm of vectors, respectively. For vectors u∈R m and v∈R n,
let u⊗v=uv ⊤ ∈R m×n denote their outer product. Also, for vectors u,v∈R d, let ⟨u,v⟩=u ⊤v
denote their inner product. For any vector x= (x 1, . . . , xd)∈R d, we define the softmax function
SoftMax:R d →R d as SoftMax(x)i = exp(x i)/(Pd
j=1 exp(xj)), and the layer normalization
operator LayerNorm(x) =x/∥x∥ 2. Moreover, for a sequence of vectors (x1,x 2, . . . ,xt)∈R d×t,
we abuse the notationLayerNorm(x 1, . . . ,xt) = (LayerNorm(x1), . . . ,LayerNorm(xt))∈R d×t.
Tokens and embeddings.For a fixed positive integer V >0 , let Voc= [V] denote a vocabulary
of size V . For each token v∈Voc , there is an associated token embedding uv ∈R d where d >0
is the embedding dimension. Assume d= 3d TE +d PE. We refer to the first dTE entries of a
d-dimensional vector as its content, the subsequent dTE entries as its first buffer, the followingdTE
entries as its second buffer, and the finaldPE entries as its effective positional encoding. Formally,
for a vector x= (x 1, x2, . . . , xd)⊤ ∈R d, we define content(x) = (x1, . . . , xdTE)⊤,buffer 1(x) =
(xdTE+1, . . . , x2dTE)⊤,buffer 2(x) = (x2dTE+1, . . . , x3dTE)⊤ and pos(x) = (x3dTE+1, . . . , xd)⊤. Let
˜uv =content(u v)∈R dTE for any v∈Voc . Assume for each uv, only the first dTE entries are
3

<!-- page 4 -->

non-zero. Furthermore, let U= [ ˜u1, ˜u2, . . . ,˜uV ]∈R dTE×V and we assume that U⊤U=I V , i.e.,
the token embeddings are orthonormal.
Algorithm 1Transformer (TF)
Input:Parametersθ= (θ PE,{θ (l,h)
Attn }L−1,Hl−1
l=0,h=0 ,{θ (l)
MLP}L−1
l=0 ), input embeddingsh= (h 1, . . . ,ht)
1:h (0)
i ←h i +PosEncode θPE(i),∀i∈[t]▷adding positional encoding
2:forl= 0toL−1do
3:h (l+0.5) ←h (l) +PHl−1
h=0 Attnθ(l,h)
Attn
(h(l))▷self attention
4:h (l+1) ←LayerNorm

MLPθ(l)
MLP
(h(l+0.5))

▷MLP and layer normalization
5:end for
Output:Embedding of the last layer at the last positionh (L)
t .
Transformer architectures.An L-layer autoregressive transformer receives a sequence of input
embeddings h=h [t]
△
= (h 1,h 2, . . . ,ht)∈R d×t and outputs TFθ(h)∈R d where TFθ(·) is
defined in Algorithm 1. Let WO ∈R V×d be the decoding matrix. A traditional decoder will
sample the next token vt+1 ∼SoftMax(W OTFθ(h)) and append its token embedding in position
t+ 1 , i.e., ht+1 =u vt+1, to autoregressively generate subsequent outputs. When using the chain
of continuous thought [Hao et al., 2024], one skips the sampling step and directly appends the
output of the transformer as the input embedding of the next position, i.e., ht+1 =TF θ(h). The
parameter θ contains positional encodings θPE, L attention layers where each layer l contains Hl
heads {θ(l,h)
Attn }L−1,Hl−1
l=0,h=0 and L MLP layers {θ(l)
MLP}L−1
l=0 . The definitions of attention heads and MLPs
are in Algorithm 2.
Algorithm 2Causal Self-Attention (Attn) and (position-wise) Multilayer Perceptron (MLP)
Input:θ Attn = (Q,K,V,O), θ MLP = ({Wi}LMLP
i=1 ,{σ i(·)}LMLP−1
i=1 ), inputh= (h 1, . . . ,ht)
1:q i ←Qh i,k i ←Kh i,v i ←Vh i,∀i∈[t]▷compute queries, keys, values
2:fori= 1totdo
3:s i ←SoftMax(⟨q i,k 1⟩, . . . ,⟨qi,k i⟩),h Attn
i ←O Pi
j=1 si,jvj
4:h MLP
i ←W LMLP σLMLP−1(· · ·W2σ1(W1hi)· · ·)
5:end for
Output:Attn θAttn(h) = (hAttn
1 , . . . ,hAttn
t )orMLP θMLP(h) = (hMLP
1 , . . . ,hMLP
t )
Positional encodings.Given an input sequence (h1, . . . ,hT ), for each position i∈[T] , there is a
corresponding positional encoding pi =PosEncode θPE(i)∈R d. For each pi, we assume that only
the last dPE entries are non-zero and thus call dPE the effective positional encoding dimension. For
notation convenience, we denote ˜hi =content(h i)∈R dTE , ¯pi =pos(p i)∈R dPE for any i∈[T] .
We use the widely used sinusoidal positional encoding for ¯pi = (¯pi,1, . . . ,¯pi,dPE)as defined below.
Definition 1(Positional Encoding).Let dPE be even. We use positional encoding generated by
sinusoidal functions Vaswani et al. [2017]. Specifically, for any position i≥1 and any index
j∈[d PE/2], we have
¯pi,2j−1 = cos(i·M −2j/dPE),¯p i,2j = sin(i·M −2j/dPE),
whereM >0is a large constant integer, e.g.,M= 10 4 as chosen in Vaswani et al. [2017].
Remark 1.We also discuss theoretical construction with RoPE[Su et al., 2024] (Section B.6).
3 Problem Formulations
Graph reachability.Let G= (V,E) be a directed graph, where V={v 1, v2, . . . , vn} is the set
of vertices and E={e 1, e2, . . . , em} is the set of edges. Each vertex vi ∈ V corresponds to a token
in the vocabulary, and thus we use vi to represent both a vertex and its corresponding token. Let
4

<!-- page 5 -->

nmax >0 denote the maximum possible number of vertices of a graph. Note that nmax <|Voc| .
Each edge ei ∈ E is a tuple, where ei = (si,t i)∈ V × V denotes there is a directed edge from the
source node si to the target node ti. Given a graph (i.e., all the edges of the graph), two candidate
destination nodes c1 and c2, and a root node r, the task is to identify which of the two nodes can be
reached by r. Note that we guarantee one and only one of c1 and c2 is reachable by r, which we
denote byc i∗.
Figure 1:Prompt format of the graph reachability problem.
Input structures.The prompt structure is illustrated in Figure 1. The prompt starts with the BOS
(beginning of sentence) token <s>. The subsequent 3m tokens represent m edges, where each edge
is represented by the source node si, target node ti, and a special edge token <e> that marks the end
of an edge. Then there is a special question token <Q> followed by two candidate destination nodes
c1 and c2. Finally, there is a special reasoning token <R> followed by a root node r. See Table 2 for
the full list of token notations. Let t0 = 3m+ 6 be the length of the prompt, and let (h1,h 2, . . .h t0)
be the input embedding sequence whereh i is the token embedding of thei-th token in the prompt.
Chain of continuous thought.We allow transformers to utilize continuous thoughts. Concretely,
for c= 1,2, . . . , the transformer autoregressively generates ht0+c =TF θ(h1, . . . ,ht0+c−1). For
notation convenience, we use [tc]=h t0+c to represent the continuous thought of step c for c≥0 ,
and thus [t0]=u r. To request the transformer to make the final prediction afterC steps of thoughts,
one simply appends a special answer token <A> at the end of the sequence, i.e., sets hT =u <A> where
T=t 0 +C+ 1 , and gets the prediction sampled from SoftMax(WOTFθ(h[T] )) or using greedy de-
coding arg maxv∈Voc WOTFθ(h[T] ). We denotefTFθ,C,WO(h[t0]) = arg maxv∈Voc WOTFθ(h[T] )
as the output token of greedy decoding after generatingCsteps of continuous thoughts.
Position index.To present the position of each token or continuous thought in the sequence
in a clear way, we use Idx(v) to represent the position of a token in the input sequence (e.g.,
Idx(<s>) = 1,Idx(s i) = 3i−1,Idx(<Q>) = 3m+ 2 ), use Idx(<e>, i) = 3i+ 1 to represent the
position of the i-th <e> token in the prompt, and use Idx([ti]) =t 0 +i to represent the position of
the continuous thought of stepi. See Table 3 for the complete list of position indices.
In the following sections, we demonstrate that the chain of continuous thought can efficiently solve
the graph reachability problem both theoretically (Section 4) and empirically (Section 5).
4 Theoretical Results
In this section, we theoretically prove that a two-layer transformer with continuous thought can
efficiently solve the graph reachability problem. We first introduce a basic building block, the
attention chooser, in our transformer constructions in Section 4.1. Then we present the key result
that continuous thought maintains a superposition state of multiple search traces simultaneously in
Section 4.2. We show our main theorem in Section 4.3 and make further discussion in Section 4.4.
4.1 Attention chooser
We use the attention chooser as a building block in our construction, which will choose the appropriate
positions to attend conditioned on the token in the current position. This allows us to use the same
parameter constructions for various input lengths. The proof is deferred to Section B.1.
Lemma 1(Attention chooser, informal version of Lemma 3).Under sinusoidal positional encoding
as defined in Definition 1, for any token <x>∈Voc and relative position ℓ≥0 , there exists
5

<!-- page 6 -->

a construction of K,Q∈R (2dPE)×d, such that for any input sequence h[T] that satisfying mild
assumptions (see Lemma 3 in Section B.2), it holds that for any position i∈[T] , it will pay almost
all attention to position(i−ℓ)ifh i =u <x>, and pay most attention to position one otherwise.
Proof sketch. We define vector ˜u ¯<x> =P
v∈Voc\{<x>} ˜uv ∈R dTE as the superposition of all token
embeddings in the vocabulary except for <x>. By the property of sinusoidal positional encoding,
there exists a rotation matrix R(ℓ) as in Lemma 4 in Section B.5, s.t. ¯pi+ℓ =R (ℓ)¯pi,∀i≥1 . Then
we can construct the query and key matrices as
Q=

0dPE×dTE 0dPE×2dTE IdPE
ξ¯p1 ⊗ ˜u ¯<x> 0dPE×2dTE 0dPE×dPE

,K=

0dPE×3dTE ηR(ℓ)
0dPE×3dTE ηIdPE

,
whereξ, η >0and thus the query and key vectors can be calculated as
qi =Q(h i +p i) =
 ¯pi
ξ⟨˜u ¯<x>, ˜hi⟩¯p1

,k i =K(h i +p i) =

ηR(ℓ)¯pi
η¯pi

=

η¯pi+ℓ
η¯pi

.
Now for any 1≤j≤i≤T , we have ⟨qi,k j⟩=η

⟨¯pi, ¯pj+ℓ⟩+ξ⟨ ˜u ¯<x>, ˜hi⟩⟨¯p1, ¯pj⟩

. Fix any
i∈[T] . By the property of sinusoidal positional encoding, ⟨¯p1, ¯pi′⟩ is maximized when i′ =i as in
Lemma 5 in Section B.5. If hi =u <x>, it holds that ⟨˜u ¯<x>, ˜hi⟩= 0 and thus ⟨qi,k j⟩ is determined
by ⟨¯pi, ¯pj+ℓ⟩, which is maximized at j=i−ℓ . If hi ̸=u <x>, one can show that ⟨˜u ¯<x>, ˜hi⟩ ≥1 and
thus⟨q i,k j⟩is largely determined by⟨ ¯p1, ¯pj⟩whenξis large, and thus maximized atj= 1.
4.2 Continuous thought maintains superposition states
Recall that h=h [t0] denotes the input sequence as defined in Section 3. We define Vc as the set of
vertices that are reachable fromrwithincsteps. Below, we present our key lemma.
Lemma 2(Continuous thought maintains superposition states).We autoregressively generate
[tc+1]=TF θ(h[t0],[t 1]. . . ,[t c])for anyc≥0. There exists a construction ofθsuch that
[tc]=h t0+c = 1p
|Vc|
X
v∈Vc
uv,(1)
i.e., the c-th continuous thought is the normalized superposition of all vertices that can be reached
fromrwithincsteps.
Lemma 2 precisely characterizes that each continuous thought is a superposition of all reachable
vertices. We provide a proof sketch below and defer the complete proof to Section B.2.
Proof sketch. We prove by induction. For c= 0 , by definition, V0 ={r} and [t0]=u r =
1√
|V0|
P
v∈V0 uv. Now we briefly show how to construct the two-layer transformer such that under
the induction assumption that (1) holds for0, . . . , c, we can obtain that (1) also holds forc+ 1.
First layer attention.The first attention layer contains five attention heads, and each head is an
attention chooser as constructed in Lemma 1. Let hk = (<x>, ℓ) denote the k-th attention head
that attends to position (i−ℓ) when the i-th token is <x> and attends to the first token otherwise.
We construct h0 = (<e>,2), h 1 = (<e>,1), h 2 = (<R>,2), h 3 = (<R>,1), h 4 = (<A>,1) , which is
illustrated in Figure 2. For each head, the value matrix will read the value in the content space, and
the output matrix will copy the value state to a designated space.
Second layer attention.In the second layer, we only need one attention head. Note that after the first
layer, for the i-th edge token <e>, we have buffer1(hIdx(<e>,i)) = ˜usi and buffer2(hIdx(<e>,i)) = ˜uti.
By the induction assumption, the current thought [tc] is a superposition of all vertices in Vc. We
construct the query and key matrices such that [tc] pays large attention to the i-th edge token <e> if
si ∈ V c (roughly speaking, we can view qIdx([tc]) =[t c],k Idx(<e>,i) =u si, and their inner product
is positive iff si ∈ V c), and add the target node ti stored in buffer 2 back to the current thought (see
Figure 3). This is exactly a one-step expansion of the currently explored vertices Vc and thus the
continuous thought at the next step(c+ 1)will correspond toV c+1.
6

<!-- page 7 -->

Figure 2: Illustration of the embedding space and first layer attention mechanism.
Figure 3:Illustration of the second layer attention mechanism for thought generation. We omit the positional
encoding space since they are not used in the second layer.
MLP as filter for signals.Note that after the attention layer, the weight of each node in the current
thought is not uniform, and the current thought might contain noise tokens since the normalized
attention score to each position is non-zero. We use the MLP layer to filter out the noise token and
adjust the weight of each node in Vc+1. Informally, for a superposition state h= P
v∈Voc λvuv, we
want to eliminate noise tokens v where λv < ε , and want to equalize the weights of other tokens.
By setting the first layer parameter as W1 = [u1, . . . ,uV ]⊤, nonlinearity as σ(x) =1{x≥ε} and
second layer as W2 =W ⊤
1 , we have W2(σ(W1h)) =P
v∈Voc 1{λ v ≥ε}u v, where W1 rotates
the basis {uv} to the standard basis {ev}, σ(·) serves as a coordinate-wise filter, and W2 rotates the
basis back. After layer normalization, we can obtain that (1) also holds forc+ 1.
4.3 Measuring the superposition state as prediction
Since the continuous thought [tc] is a superposition of all vertices in Vc, as long as ci∗ can be
reached by r within C steps, the superposition state [tC] will contain ci∗. At the final prediction
step, the answer token <A> will “measure” the superposition state [tC] using c1 and c2 and predict
the token with the larger signal in [tC] as the output (see Figure 7 in Section B.2 for a pictorial
illustration). We formalize our result in the following theorem, and defer the proof to Section B.4.
Theorem 1(Chain of continuous thought solves graph reachability).Fix Tmax >0 . Assume the
length of the input sequence (including the continuous thoughts and the special answer token <A>)
does not exceed Tmax. There exists a construction of a two-layer transformer parameters θ and the
readout matrix WO ∈R |Voc|×d (where d=O(|Voc|) ) that are independent of graphs, such that
for any directed graph G= (V,E) with at most nmax nodes (nmax <|Voc| ), a root node r, two
candidate destination nodesc 1,c 2, for anyCexceeds the diameter of the graph, it holds that
fTFθ,C,WO(h[t0]) =c i∗ .
4.4 Discussions
Role of buffer spaces.The buffer spaces are subspaces of the embedding to store useful information.
For clarity, our construction separates the “content” and the two “buffer” spaces into different
dimensions. In practice, they can be jointly projected into a more compact and lower-dimensional
space. For example, we can construct u= Pk
i=1 R(i)u(i) ∈R d where the column space of each
R(i) ∈R d×d forms a subspace. Different subspaces can be (almost) orthogonal, and for each
subspace, the column vectors ofR (i) can also be almost orthonormal.
7

<!-- page 8 -->

Weights of different nodes in the superposition.In our construction, each superposition state
maintains nodes with uniform weights. In practice, the weights of different nodes can vary due to
factors such as training signals or the model’s internal heuristic on which nodes are more likely to
reach the final answer [Cohen et al., 2025]. In Section 5, we show that in practice, the training signal
could bias the superposition states towards thefrontier nodesthat can be reached with exactly i steps
and theoptimal nodesthat can lead to the destination node.
5 Experiments
In this section, we conduct extensive experiments to validate our theoretical results that COCONUT
outperforms discrete CoT even with many fewer layers (Section 5.2), which is indeed due to superpo-
sition states encoded in continuous thoughts during reasoning (Section 5.3).
5.1 Training Setup
0.5
0.6
0.7
0.8
0.9
1.0Accuracy
Coconut
CoT
CoT*
No CoT
Figure 4:The overall accuracy
of COCONUT, CoT, CoT ∗(12 lay-
ers,n heads = 12) and No CoT.
Model.We adopt a GPT -2 style decoder with two transformer
layers, dmodel=768, nheads=8. We train from scratch using AdamW
(β1=0.9, β2=0.95, weight-decay 10−2) and a constant learning rate
of1×10 −4.
Dataset.We construct a subset of ProsQA [Hao et al., 2024], with
questions whose solutions require 3–4 reasoning hops. Each node
in the graph is injected as a dedicated token into the vocabulary. The
split statistics are summarised in Table 4.
Method.Following Hao et al. [2024], we adopt a multi-stage
training strategy with the supervision from chain-of-thoughts data.
Stage i teaches the model to use i continuous thoughts before pre-
dicting the i-th node in the given chain of thought as the next token.
If the index of the training stage is larger than the CoT solution
length l, the model is trained to output the final prediction after l
continuous thoughts and <A>. We train the model for 25 epochs at
each stage, and the whole training lasts 300 epochs in total. At each stage, we mix the data from
the previous stage randomly with 0.1 probability, which helps improve performance in our early
experiments.
5.2 Overall Results
Extending the findings of Hao et al. [2024], we further show that a 2-layer transformer trained from
scratch can effectively solve ProsQA when using COCONUT. As shown in Figure 4, COCONUT
achieves near-perfect accuracy, while both CoT and No CoT baselines solve only about 75% of the
tasks (random guessing = 50%). Even with a larger model of 12 layers and nheads = 12, marked with
∗in the figure, CoT improves to 83% accuracy but still cannot solve the tasks reliably.
5.3 Visualising Latent Reasoning
We inspect both theattention patternand therepresentationof the continuous thought learned by the
model to validate our theoretical construction.
Layer 1 attention.According to our theoretical construction, the most important function of LAYER
1 attention heads is tocopythe source and target node tokens of an edge onto the corresponding edge
token ⟨e⟩. Figure 5 shows a representative attention map, confirming that the model has instantiated
this copying mechanism in practice.
Layer 2 attention.LAYER2 is responsible fornode expansion: each continuous thought attends to
all outgoing edges from nodes that are currently reachable. To quantify this behavior, we compute,
when generating i-th continuous thought, the aggregated attention score received by each edge token
triplet (s, t, <e>) across all heads. 4 kinds of edges exist: (1)Reachable: their source node is in the
8

<!-- page 9 -->

reachable set at step i; (2)Not Reachable: source nodenotin the reachable set; (3)Frontier: a subset
of reachable edges whose source node is on the current search frontier, i.e., exactly i steps away from
the root node; (4)Optimal: a subset of frontier edges that lead to the optimal reasoning chain.
<s>
5
15
<e>
11
16
<e>
1
3
<e>
14
17
<e>
0
14
<e>
Input T okens (Key)
<s>
5
15
<e>
11
16
<e>
1
3
<e>
14
17
<e>
0
14
<e> Input T okens (Query)
0
1
2
3
4
5
6
7
8
Attention Weight
Figure 5:A representative example of Layer 1 attention
map: the edge token <e> (y-axis) places nearly all its
attention mass on its source and target nodes (x-axis), in
line with the theoretical construction.
Table 1 reports group-wise means and standard
deviations averaged on the test set. The model
sharply concentrates its attention onReachable
edges, as predicted by our theoretical construc-
tion. Interestingly, there is an additional bias to-
ward theFrontiersubset. One possibility is that
the training signal encourages the model to pre-
dict a frontier node at each step, coupled with the
decaying effects for previously explored nodes.
We also find that the optimal edges receive more
attention scores, likely due to the supervision of
multi-stage training from the CoT solution.
Representation of continuous thoughts.To
verify that the continuous thought serves as su-
perposition states for the search, we compute the
inner product between the continuous thought
at step i, [ti], and each node embedding uv.
Similar to edge classification above, we classify
nodes intoReachable,Not Reachable,Frontier,
andOptimal. Figure 6 (top) plots the distri-
bution segregated by the reasoning step i. As
predicted, nodes within i hops exhibit markedly
higher similarity scores than distant nodes. Moreover,Frontiernodes are noticeably closer to [ti]
than other reachable nodes, illustrating that the superposition emphasizes the candidate expansion
fronts. Besides, theOptimalnodes are even closer to [ti], also likely due to the training data always
presenting the optimal path.
Collectively, these analyses confirm that the intendedsuperpositional searchexists in trained models:
Layer 1 establishes the query context, Layer 2 expands the frontier, and the latent vectors encode a
soft, parallel representation of reachable state sets, realizing the theoretical construction in Section 4.
We also show in Appendix C.2 that this search pattern is consistent across multiple experiments with
random seeds.
Figure 6:The histogram of inner product between the i-th continuous thoughts and the node embeddings.
The mean value for each group is shown in the legend. Note thatFrontieris a subset ofReachablenodes, and
Optimalis a subset ofFrontiernodes.
5.4 Exploration Priority
9

<!-- page 10 -->

Table 1:Layer 2 attention scores to different edge groups at step i
(mean±standard deviation).
Step 1 Step 2 Step 3 Step 4
Not Reachable0.04±0.070.03±0.090.08±0.170.12±0.20
Reachable2.12±1.070.71±0.920.38±0.720.29±0.66
–Frontier2.12±1.071.00±0.960.67±0.870.61±0.95
–Optimal2.54±1.031.72±1.131.67±1.202.23±1.35
An interesting fact revealed by the
visualizations in Section 5.3 is that
the model allocates disproportion-
ate attention tooptimaledges and
nodes, reminiscent of a prioritized
search strategy. One hypothesis is
that this behavior is an artifact of
our multi-stage curriculum, which
explicitly guides the model on the
CoT solution at every step. To un-
derstand the effect of this multi-stage guidance, we introduce an alternative supervision method
COCONUT -BFS: at stage i, the supervision token is drawn uniformly at random from the fron-
tier nodes exactly i hops from the root, rather than the i-th node in the CoT solution. All other
hyperparameters are unchanged.
Our experiment results indicate that COCONUT -BFS also achieves near-perfect accuracy on ProsQA,
matching the original COCONUT. Figure 6 compares the inner product distributions of the latent
thoughts for both models. Remarkably, the two supervision methods converge to similar exploration
strategies, even though there is no explicit guidance towards optimal nodes for COCONUT-BFS.
Conversely, the original COCONUT, although trained exclusively on optimal nodes during intermediate
stages, still assigns elevated weight to non-optimal frontier nodes compared to other non-frontier
nodes, implicitly performing a breadth-first expansion before honing in on the solution. We leave
explaining this behavior from the perspective of training dynamics as future work.
6 Conclusions
In this paper, we study how the chain-of-continuous-thought boosts LLM’s reasoning capability by
focusing on the graph reachability problem. We provided a theoretical construction of a two-layer
transformer that can efficiently solve graph reachability for an n-vertex D-diameter directed graph by
D steps of continuous thoughts, while the best existing result on constant-depth transformers with
discrete CoT requires O(n2) steps. Our construction reveals that the superposition states that encode
multiple search traces simultaneously are the key to the strong reasoning capability of COCONUT,
and we conducted thorough experiments to validate that our theoretical construction matches the
solutions obtained by training dynamics. Several interesting future directions include: (1) Deriving
a lower bound on the number of discrete CoT steps in the graph reachability problem to show a
strict separation of expressivity between CoT and COCONUT; (2) Theoretical understanding of the
emergence of exploration behavior with only deterministic search trace demonstration via training
dynamics; (3) Advantages of reasoning in a continuous space in more general settings.
Acknowledgements
This work was partially supported by a gift from Open Philanthropy to the Center for Human-
Compatible AI (CHAI) at UC Berkeley and by NSF Grants IIS-1901252 and CCF-2211209.
References
Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algo-
rithm is in-context learning? investigations with linear models.arXiv preprint arXiv:2211.15661,
2022.
Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers
to recognize formal languages.arXiv preprint arXiv:2009.11264, 2020a.
Satwik Bhattamishra, Arkil Patel, and Navin Goyal. On the computational power of transformers and
its implications in sequence modeling.arXiv preprint arXiv:2006.09286, 2020b.
Arno Böhm.Quantum mechanics: foundations and applications. Springer Science & Business
Media, 2013.
10

<!-- page 11 -->

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems.arXiv preprint arXiv:2110.14168, 2021.
Andrew Cohen, Andrey Gromov, Kaiyu Yang, and Yuandong Tian. Spectral journey: How transform-
ers predict the shortest path.arXiv preprint arXiv:2502.08794, 2025.
Xinnan Dai, Haohao Qu, Yifen Shen, Bohang Zhang, Qihao Wen, Wenqi Fan, Dongsheng Li, Jiliang
Tang, and Caihua Shan. How do large language models understand graph patterns? a benchmark
for graph pattern comprehension.arXiv preprint arXiv:2410.05298, 2024.
Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable
creation in self-attention mechanisms. InInternational Conference on Machine Learning, pages
5793–5831. PMLR, 2022.
Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large
language models.arXiv preprint arXiv:2310.04560, 2023.
Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing
the mystery behind chain of thought: a theoretical perspective.Advances in Neural Information
Processing Systems, 36:70757–70798, 2023.
Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh
Nagarajan. Think before you speak: Training language models with pause tokens.arXiv preprint
arXiv:2310.02226, 2023.
Halil Alperen Gozeten, M Emrullah Ildiz, Xuechen Zhang, Hrayr Harutyunyan, Ankit Singh Rawat,
and Samet Oymak. Continuous chain of thought enables parallel exploration and reasoning.arXiv
preprint arXiv:2505.23648, 2025.
Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, and Shi Han. Gpt4graph: Can large
language models understand graph structured data? an empirical evaluation and benchmarking.
arXiv preprint arXiv:2305.15066, 2023.
Tianyu Guo, Hanlin Zhu, Ruiqi Zhang, Jiantao Jiao, Song Mei, Michael I Jordan, and Stuart Russell.
How do llms perform two-hop reasoning in context?arXiv preprint arXiv:2502.13913, 2025.
Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong
Tian. Training large language models to reason in a continuous latent space.arXiv preprint
arXiv:2412.06769, 2024.
Subbarao Kambhampati. Can large language models reason and plan?Annals of the New York
Academy of Sciences, 1534(1):15–18, 2024.
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish
Sabharwal. Decomposed prompting: A modular approach for solving complex tasks.arXiv
preprint arXiv:2210.02406, 2022.
Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to
solve inherently serial problems.arXiv preprint arXiv:2402.12875, 1, 2024.
Valerii Likhosherstov, Krzysztof Choromanski, and Adrian Weller. On the expressive power of
self-attention matrices.arXiv preprint arXiv:2106.03764, 2021.
Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers
learn shortcuts to automata.arXiv preprint arXiv:2210.10749, 2022.
Zihan Luo, Xiran Song, Hong Huang, Jianxun Lian, Chenhao Zhang, Jinqi Jiang, and Xing Xie.
Graphinstruct: Empowering large language models with graph understanding and reasoning
capability.arXiv preprint arXiv:2403.04483, 2024.
William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought.
arXiv preprint arXiv:2310.07923, 2023a.
11

<!-- page 12 -->

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision
transformers.Transactions of the Association for Computational Linguistics, 11:531–545, 2023b.
William Merrill and Ashish Sabharwal. A little depth goes a long way: The expressive power of
log-depth transformers.arXiv preprint arXiv:2503.03961, 2025.
Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing-complete.Journal of Machine
Learning Research, 22(75):1–35, 2021.
Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in
transformer language models.arXiv preprint arXiv:2404.15758, 2024.
Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow,
Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph
algorithms.Advances in Neural Information Processing Systems, 37:78320–78370, 2024.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
DiJia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, and Qinqing Zheng. Token
assorted: Mixing latent and text tokens for improved language model reasoning.arXiv preprint
arXiv:2502.03275, 2025.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced
transformer with rotary position embedding.Neurocomputing, 568:127063, 2024.
Karthik Valmeekam, Kaya Stechly, and Subbarao Kambhampati. Llms still can’t plan; can lrms? a
preliminary evaluation of openai’s o1 on planbench.arXiv preprint arXiv:2409.13373, 2024.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need.Advances in neural information processing
systems, 30, 2017.
Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokked transformers are implicit reasoners: A
mechanistic journey to the edge of generalization.arXiv preprint arXiv:2405.15071, 2024a.
Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov.
Can language models solve graph problems in natural language?Advances in Neural Information
Processing Systems, 36:30840–30861, 2023a.
Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui.
Math-shepherd: Verify and reinforce llms step-by-step without human annotations.arXiv preprint
arXiv:2312.08935, 2023b.
Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, William Yang Wang, and Alessan-
dro Sordoni. Guiding language model reasoning with planning tokens.arXiv preprint
arXiv:2310.05707, 2023c.
Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan, Wenhu Chen, and William Yang
Wang. Understanding reasoning ability of language models from the perspective of reasoning
paths aggregation. InInternational Conference on Machine Learning, pages 50026–50042. PMLR,
2024b.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and
Yu Su. Travelplanner: A benchmark for real-world planning with language agents.arXiv preprint
arXiv:2402.01622, 2024.
Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. Self-attention networks
can process bounded hierarchical languages.arXiv preprint arXiv:2105.11115, 2021.
12

<!-- page 13 -->

Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of language models: Part 2.1,
grade-school math and the hidden reasoning process. InThe Thirteenth International Conference
on Learning Representations, 2024.
Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo
Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for
large language models.arXiv preprint arXiv:2309.12284, 2023.
Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning.arXiv preprint
arXiv:2309.05653, 2023.
Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar.
Are transformers universal approximators of sequence-to-sequence functions?arXiv preprint
arXiv:1912.10077, 2019.
Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova,
Le Hou, Heng-Tze Cheng, Quoc V Le, Ed H Chi, et al. Natural plan: Benchmarking llms on
natural language planning.arXiv preprint arXiv:2406.04520, 2024.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning
in large language models.arXiv preprint arXiv:2205.10625, 2022.
Yang Zhou, Hongyi Liu, Zhuoming Chen, Yuandong Tian, and Beidi Chen. Gsm-infinite: How
do your llms behave over infinitely increasing context length and reasoning complexity?arXiv
preprint arXiv:2502.05252, 2025.
13

<!-- page 14 -->

A Notation Details
We provide detailed descriptions of different tokens in Table 2, and the position index of different
tokens or continuous thoughts in Table 3.
Tokens Meanings
<s>a special token denoting the beginning of the sentence
si the source node of edgei
ti the target node of edgei
<e>a special token marking the end of an edge
<Q>a special token followed by two candidate nodes
c1,c 2 two candidate destination nodes
<R>a special token marking the start of reasoning
rthe root node
[ti]thei-th continuous thought (represented by ad-dimensional vector)
<A>a special token driving the model to make the final prediction
Table 2: Meaning of each token.
Notations Position indices
Idx(<s>) 1
Idx(si) 3i−1
Idx(ti) 3i
Idx(<e>, i) 3i+ 1
Idx(<Q>) 3m+ 2
Idx(c1) 3m+ 3
Idx(c2) 3m+ 4
Idx(<R>) 3m+ 5
Idx(r) 3m+ 6 =t 0
Idx([ti])t 0 +i
Idx(<A>)t 0 +C+ 1 =T
Table 3: Position indices of different tokens or continuous thoughts in the input sequence.
B Missing Proofs
B.1 Formal version and proof of Lemma 1
Lemma 3(Attention chooser, formal version of Lemma 1).Fix any token <x>∈Voc , integer
ℓ≥0 , and ε∈(0,1) . Under sinusoidal positional encoding as defined in Definition 1, there exists a
construction ofK,Q∈R (2dPE)×d, such that for any input sequence(h 1, . . . ,hT )that satisfies
hi =
X
v∈Voc
λi,vuv,whereλ i,v ≥0∀v∈Voc,
X
v∈Voc
λ2
i,v = 1,∀i∈[T],(2)
and satisfies ⟨u<x>,h i⟩ ∈ {0,1} (i.e., each input embedding is either equal to the embedding of token
<x> or orthogonal to it) and ⟨u<x>,h i⟩= 0 for i≤ℓ (i.e., the first ℓ tokens are not <x>), it holds
that for anyi∈[T],
if⟨h i,u <x>⟩= 1, thens i,i−l >1−ε, otherwises i,1 >1−ε,
where si,j is the attention score from the i-th token to the j-th token as defined in Algorithm 2 with
the input sequence(h 1 +p 1, . . . ,hT +p T ).
Proof. Note that (2) implies that each input embedding is a normalized superposition of token
embeddings in the vocabulary. We aim to construct an attention head such that when the i-th token is
14

<!-- page 15 -->

<x>, it will pay almost all attention to the position i−ℓ , otherwise it will pay almost all attention to
the BOS token <s> (known as the attention sink). Define vector ˜u ¯<x> =P
v∈Voc\{<x>} ˜uv ∈R dTE,
which is the superposition of all token embeddings in the vocabulary except for<x>. We define
Q=

0dPE×dTE 0dPE×dTE 0dPE×dTE IdPE
ξ¯p1 ⊗ ˜u ¯<x> 0dPE×dTE 0dPE×dTE 0dPE×dPE

∈R (2dPE)×d,
K=

0dPE×dTE 0dPE×dTE 0dPE×dTE ηR(ℓ)
0dPE×dTE 0dPE×dTE 0dPE×dTE ηIdPE

∈R (2dPE)×d,
whereξ, η >0will be specified later andR (ℓ) is defined as in Lemma 4. Therefore,
qi =Q(h i +p i) =
 ¯pi
ξ⟨˜u ¯<x>, ˜hi⟩¯p1

,k i =K(h i +p i) =

ηR(ℓ)¯pi
η¯pi

=

η¯pi+ℓ
η¯pi

.
Now for any1≤j≤i≤T, we have
⟨qi,k j⟩=η

⟨¯pi, ¯pj+ℓ⟩+ξ⟨ ˜u ¯<x>, ˜hi⟩⟨¯p1, ¯pj⟩

.
Now we fix i∈[T] . We first consider the case where ⟨hi,u <x>⟩= 1 (which also implies i > ℓ ). By
(2) and the assumption that token embeddings are orthonormal, we have
⟨˜hi, ˜uv⟩=⟨h i,u v⟩= 0,∀v∈Voc\{<x>}=⇒ ⟨ ˜hi, ˜u ¯<x>⟩= 0.
Therefore, we have⟨q i,k j⟩=η⟨ ¯pi, ¯pj+ℓ⟩. By Lemma 5, we have
⟨qi,k i−ℓ⟩=η⟨ ¯pi, ¯p(i−ℓ)+ℓ⟩=η⟨ ¯pi, ¯pi⟩=ηd PE/2
and
⟨qi,k j⟩=η⟨ ¯pi, ¯pj+ℓ⟩ ≤ηd PE/2−ηε T ,∀j̸=i−ℓ.
where εT >0 is defined in Lemma 5. This implies ⟨qi,k j⟩ is maximized when j=i−ℓ with a
non-zero gapηε T , and therefore, we have
si,i−ℓ = exp(⟨qi,k i−ℓ⟩)P
j∈[i] exp(⟨qi,k j⟩) ≥ exp(ηεT )
exp(ηεT ) + (i−1) .(3)
Now we consider the case where ⟨hi,u <x>⟩= 0 . Again, by (2) and the orthonormal assumption of
token embeddings, we have
⟨˜hi, ˜u ¯<x>⟩=
X
v∈Voc\{<x>}
⟨hi,u v⟩=
X
v∈Voc\{<x>}
λi,v ≥
X
v∈Voc\{<x>}
λ2
i,v = 1.
By Lemma 5, we can defineξ= max j=2,...,T
2dPE
⟨¯p1,¯p1⟩−⟨¯p1,¯pj ⟩, and thus
ξ(⟨¯p1, ¯p1⟩ − ⟨¯p1, ¯pj⟩)≥2d PE,∀j∈ {2, . . . , T}.
Note that for anyj∈ {2, . . . , T}, we have
⟨qi,k 1⟩ − ⟨qi,k j⟩
=η

ξ⟨˜u ¯<x>, ˜hi⟩(⟨ ¯p1, ¯p1⟩ − ⟨¯p1, ¯pj⟩)−(⟨ ¯pi, ¯pj+ℓ⟩ − ⟨¯pi, ¯p1+ℓ⟩)

≥η(2d PE −d PE)
=ηdPE.
Therefore,
si,1 = exp(⟨qi,k 1⟩)P
j∈[i] exp(⟨qi,k j⟩) ≥ exp(ηdPE)
exp(ηdPE) + (i−1) .(4)
By choosing a sufficiently large η, the lower bound of (3) and (4) will both exceed 1−ε and thus the
proof is complete.
15

<!-- page 16 -->

B.2 Proof of Lemma 2
Note that in the proof sketch of Lemma 2, we showed how to prove the lemma by induction. Here,
we use an alternative way that only requires a single forward pass of the transformer. Note that
the continuous thoughts are generated in an autoregressive manner, i.e., for an input sequence
h[T] = (h1, . . . ,hT ), we have h(L)
t =TF θ(h[t]), where h(L)
t is defined in Algorithm 1 with the
input sequence h[T] . This means for a sequence of length t, appending additional vectors at the end
of the sequence does not affect the computation of the firstt positions. This means, instead of proving
the following property inductively for eachc= 0,1, . . . , C−1:
[tc+1]=TF θ(h[t0],[t 1], . . . ,[tc])
where for each c, it holds [tc]= 1√
|Vc|
P
v∈Vc uv and h[t0] is defined as in Section 3, we can
instead prove by a single forward pass by setting the input embedding
ht0+c = 1p
|Vc|
X
v∈Vc
uv,∀c∈[C],(5)
and additionally setting hT =h t0+C+1 =u <A>, and prove the hidden embedding in the last layer
with inputh [T] satisfies
h(L)
t0+c = 1p
|Vc+1|
X
v∈Vc+1
uv,∀0≤c≤C.
In Section B.3, we construct the parameter for each component of the two-layer transformer. Finally,
Proposition B.4 precisely provides the result we desire.
B.3 Construction of transformer parameters
Proposition B.1(First layer attention).For any fixed ε∈(0,1/2) , there exists a construction of
key and query matrices for first layer attention heads {Q(0,h),K (0,h)}h=0,...,4 s.t. for any input
embeddings h= (h 1, . . . ,ht) satisfying the format specified in Section B.2, the value of following
terms exceed1−εfor alli∈[t]:
•s (0,0)
i,i−2 ifh i =u <e>, ands (0,0)
i,1 otherwise;s (0,1)
i,i−1 ifh i =u <e>, ands (0,1)
i,1 otherwise;
•s (0,2)
i,i−2 ifh i =u <R>, ands (0,2)
i,1 otherwise;s (0,3)
i,i−1 ifh i =u <R>, ands (3)
i,1 otherwise;
•s (0,4)
i,i−1 ifh i =u <A>, ands (0,4)
i,1 otherwise;
where s(0,h)
i ←SoftMax(⟨q (h)
i ,k (h)
1 ⟩, . . . ,⟨q(h)
i ,k (h)
i ⟩)∈R i and k(h)
i =K (0,h)h(0)
i ,q (h)
i =
Q(0,h)h(0)
i ,h (0)
i =h i +p i.
Proof. Note that each of the attention heads is an attention chooser as defined in Lemma 3. Therefore,
the proposition directly holds by constructing each attention head as described in Lemma 3.
By Proposition B.1, each edge token <e> will pay attention to its corresponding source node and
target node. Also, the reasoning token <R> will pay attention to two candidate destination nodes, and
the answer token <A> will pay attention to the last continuous thought. When no token or position
needs to be paid attention to for some attention heads, it will dump attention to the BOS token <s>,
a phenomenon known as attention sink. Since the attention after softmax cannot exactly be zero,
there will be undesired tokens with noise-level magnitude copied to each position. The subsequent
MLP layer will filter out the noise. We formalize the above statements rigorously in the following
proposition.
Proposition B.2(First layer MLP).When the input sequence h[T] satisfies conditions in Section B.2,
there exists a construction of θ(0,h)
Attn for h∈ {0, . . . ,4} , θ(0)
MLP such that the output of the first layer
h(1) satisfies:
16

<!-- page 17 -->

•h (1)
Idx(<e>,i) = 1√
3[˜u⊤
<e> ˜u⊤
si ˜u⊤
ti 0⊤
dPE]⊤,∀i∈[m]
•h (1)
Idx(<R>) = 1√
3[˜u⊤
<R> 0⊤
dTE (˜uc1 + ˜uc2)⊤ 0⊤
dPE]⊤
•h (1)
Idx(<A>) = 1√
|VC |+1
˜u⊤
<A>
P
v∈VC
˜u⊤
v 0⊤
dTE 0⊤
dPE
⊤
•h (1)
Idx([ti]) = 1√
|Vi|
P
v∈Vi
˜u⊤
v 0⊤
dTE 0⊤
dTE 0⊤
dPE
⊤
,∀i∈[C]
•h (1)
i =h i for otheri∈[T].
Proof. We use the construction in Proposition B.1 for {Q(0,h),K (0,h)}h=0,...,4. For each position,
after attending to the desired tokens, each attention head will copy the attended tokens to one of the
two buffer spaces. Formally, we provide the construction of value and output matrices for each head
below. First, the value matrices are constructed as
V(0,h) = [IdTE − ˜u<s> ⊗ ˜u<s> 0dTE×(d−dTE)]∈R dTE×d, h= 0, . . . ,4.
By construction, we have v(h)
i =V (0,h)h(0)
i =content(h i)·1{i >1}= ˜hi ·1{i >1} for
i∈[T], h∈ {0, . . . ,4} . This is due to the input format specified in Section B.2, which ensures that
only ˜h1 contains ˜u<s>.
Now we construct output matrices such that the h-th attention head copies the content of the attended
token to buffer 1 forh= 0,4, and to buffer 2 forh= 1,2,3. Formally, we construct
O(0,h) =[0dTE×dTE IdTE 0dTE×dTE 0dTE×dPE]⊤ ∈R d×dTE , h= 0,4,
O(0,h) =[0dTE×dTE 0dTE×dTE IdTE 0dTE×dPE]⊤ ∈R d×dTE , h= 1,2,3.
Therefore, it holds that O(0,h)v(h)
i = [0⊤
dTE
˜h⊤
i ·1{i >1}0 ⊤
dTE 0⊤
dPE]⊤ for h= 0,4 and O(0,h)v(h)
i =
[0⊤
dTE 0⊤
dTE
˜h⊤
i ·1{i >1}0 ⊤
dPE]⊤ forh= 1,2,3, and thus we have
Attnθ(0,h)
Attn
(h(0))i =
iX
j=1
s(0,h)
i,j O(0,h)v(h)
j =

0⊤
dTE
iX
j=2
s(0,h)
i,j ˜h⊤
j 0⊤
dTE 0⊤
dPE


⊤
, h= 0,4,
Attnθ(0,h)
Attn
(h(0))i =
iX
j=1
s(0,h)
i,j O(0,h)v(h)
j =

0⊤
dTE 0⊤
dTE
iX
j=2
s(0,h)
i,j ˜h⊤
j 0⊤
dPE


⊤
, h= 1,2,3,
where s(0,h)
i,j is defined as in Proposition B.1. Therefore, the output at position i of the first attention
layer is
h(0.5)
i =h (0)
i +
4X
h=0
Attnθ(0,h)
Attn
(h(0))i =

˜h⊤
i
X
h=0,4
iX
j=2
s(0,h)
i,j ˜h⊤
j
3X
h=1
iX
j=2
s(0,h)
i,j ˜h⊤
j pos(pi)⊤


⊤
.
Next, we construct the parameters of the MLP of the first layer. For notation convenience and by
the input format in Section B.2, we can decompose ˜hi =PV
j=1 λ(0)
i,j ˜uj =Uλ (0)
i where λ(0)
i =
[λ(0)
i,1 , λ(0)
i,2 , . . . , λ(0)
i,V ]⊤ ∈R V . Let
MLPθ(0)
MLP
(h(0.5))i =W (0)
2 σ(0)
ε (W(0)
1 h(0.5)
i )∈R d
which is a two-layer neural network. Let W(0)
1 = [diag(U⊤,U ⊤,U ⊤),0 (3V)×d PE]∈R 3V×d . Then
W(0)
1 h(0.5)
i =


U⊤Uλ(0)
i
U⊤UP
h=0,4
Pi
j=2 s(0,h)
i,j λ(0)
j
U⊤UP3
h=1
Pi
j=2 s(0,h)
i,j λ(0)
j

 =


λ(0)
iP
h=0,4
Pi
j=2 s(0,h)
i,j λ(0)
jP3
h=1
Pi
j=2 s(0,h)
i,j λ(0)
j

 ∈R 3V .
17

<!-- page 18 -->

Let σ(0)
ε (·) be a coordinate-wise non-linearity such that σ(0)
ε (x) =1{x≥ε} for x∈R . We
choose ε= 1
2√n where n is the (maximum possible) number of vertices in the graph. We denote
i(h)
∗ = arg maxj≤i s(0,h)
i,j , i.e., i(h)
∗ is the position that position i pays the most attention within head
h. By Proposition B.1, we can construct key and query matrices such that s(0,h)
i,i(h)
∗
>1−ε/5 for any i
and h. Also, by the input format especially by (5), λ(0)
i,k ∈ {0,1} ∪ {1/
√
i|i∈[n]} , which implies
that any non-zero λ(0)
i,k satisfies λ(0)
i,k ∈[2ε,1] and all non-zero entries of λ(0)
i share the same value
for any fixedi. Then by definition ofσ (0)
ε (·), we haveσ (0)
ε (λ(0)
i ) = λ(0)
i
∥λ(0)
i ∥∞
.
Now we analyzePi
j=2 s(0,h)
i,j λ(0)
j . For any k∈[V] , we considerPi
j=2 s(0,h)
i,j λ(0)
j,k. If i(h)
∗ = 1, then
we have
iX
j=2
s(0,h)
i,j λ(0)
j,k ≤


iX
j=2
s(0,h)
i,j

 ·max
2≤j≤i
λ(0)
j,k < ε
5 ·1 = ε
5 .
Ifi (h)
∗ >1, we have
iX
j=2
s(0,h)
i,j λ(0)
j,k =s (0,h)
i,i(h)
∗
λ(0)
i(h)
∗ ,k +
iX
j=2,j̸=i (h)
∗
s(0,h)
i,j λ(0)
j,k.
Whenλ (0)
i(h)
∗ ,k = 0, we can obtain that
iX
j=2
s(0,h)
i,j λ(0)
j,k =
iX
j=2,j̸=i (h)
∗
s(0,h)
i,j λ(0)
j,k ≤


iX
j=2,j̸=i (h)
∗
s(0,h)
i,j

 ·max
2≤j≤i
λ(0)
j,k < ε
5 .
Whenλ (0)
i(h)
∗ ,k >0, we can obtain that
iX
j=2
s(0,h)
i,j λ(0)
j,k ≥s (0,h)
i,i(h)
∗
λ(0)
i(h)
∗ ,k ≥(1−ε/5)·2ε≥ε.
Therefore, σ(0)
ε (P
h=0,4
Pi
j=2 s(0,h)
i,j λ(0)
j,k) =1
nS
h=0,4

i(h)
∗ >1

∩

λ(0)
i(h)
∗ ,k >0
o
, ∀k∈
[V] . Also, by Proposition B.1, for any i∈[T] and k∈[V] , there is at most one h∈ {0, . . . ,4}
such that i(h)
∗ >1 and λ(0)
i(h)
∗ ,k >0 hold simultaneously. Therefore, σ(0)
ε (P
h=0,4
Pi
j=2 s(0,h)
i,j λ(0)
j,k) =
P
h=0,4 1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗ ,k >0},∀k∈[V], and thus we can write this compactly
σ(0)
ε (
X
h=0,4
iX
j=2
s(0,h)
i,j λ(0)
j ) =
X
h=0,4
1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }.
Similarly, we have
σ(0)
ε (
3X
h=1
iX
j=2
s(0,h)
i,j λ(0)
j ) =
3X
h=1
1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }.
Therefore,
σ(0)
ε (W(0)
1 h(0.5)
i ) =


λ(0)
i /∥λ(0)
i ∥∞P
h=0,4 1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }
P3
h=1 1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }

 ∈R 3V .
18

<!-- page 19 -->

Finally, we setW (0)
2 =W (0)⊤
1 = [diag(U⊤,U ⊤,U ⊤),0 (3V)×d PE]⊤ ∈R d×3V , and thus
MLPθ(0)
MLP
(h(0.5))i =


Uλ(0)
i /∥λ(0)
i ∥∞
UP
h=0,4 1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }
UP3
h=1 1{i (h)
∗ >1} ·1{λ (0)
i(h)
∗
>0 V }
0dPE


∈R d.
Now we derive the output of the MLP for different positionsi.
For i=Idx(<e>, k) = 3k+ 1 where k∈[m] , we have i(0)
∗ =i−2 =Idx(s k) and i(1)
∗ =i−1 =
Idx(tk). Note that i(h)
∗ = 1 for h= 2,3,4 . Also, λ(0)
Idx(sk) =e sk, λ(0)
Idx(tk) =e tk and λ(0)
i =e <e>
are allV-dimensional one-hot vectors. Therefore, we have
MLPθ(0)
MLP
(h(0.5))Idx(<e>,k) = [˜u⊤
<e> ˜u⊤
sk ˜u⊤
tk 0⊤
dPE]⊤.
For i=Idx(<R>) = 3m+ 5 , we have i(2)
∗ =i−2 =Idx(c 1) and i(3)
∗ =i−1 =Idx(c 2). Note that
i(h)
∗ = 1forh= 0,1,4. Also,λ (0)
i−2 =e c1 andλ (0)
i−1 =e c2 ,λ (0)
i =e <R>. Therefore,
MLPθ(0)
MLP
(h(0.5))Idx(<R>) = [˜u⊤
<R> 0⊤
dTE (˜uc1 + ˜uc2)⊤ 0⊤
dPE]⊤.
For i=Idx(<A>) =t 0 +C+ 1 , we have i(4)
∗ =i−1 and i(h)
∗ = 1 for 0≤h≤3 . Note that
λ(0)
i =e <A>, λ(0)
i−1 =λ (0)
Idx([tC]) = 1√
|VC |
P
v∈VC
ev where VC is the set of vertices reachable from
rwithinCsteps. Then we have
MLPθ(0)
MLP
(h(0.5))Idx(<A>) =
"
˜u⊤
<A>
X
v∈VC
˜u⊤
v 0⊤
dTE 0⊤
dPE
#⊤
.
For i=Idx([t c]) =t 0 +c for c∈[C] , we have i(h)
∗ = 1 for all h and λ(0)
i = 1√
|Vc|
P
v∈Vc ev,
and thus
MLPθ(0)
MLP
(h(0.5))Idx([tc]) =
"X
v∈Vc
˜u⊤
v 0⊤
dTE 0⊤
dTE 0⊤
dPE
#⊤
.
For remainingi, we havei (h)
∗ = 1for allhandλ (0)
i is one-hot. So
MLPθ(0)
MLP
(h(0.5))i =
h
˜h⊤
i 0⊤
dTE 0⊤
dTE 0⊤
dPE
i⊤
=h i.
By applying layer normalization, we can obtain the final result.
Note that Proposition B.2 shows that after the first layer, each <e> token will copy its corresponding
source and target token embeddings into its two buffer spaces, respectively. For the second layer,
since it is the last layer, we only need to focus on positions for current thoughtsIdx([t c]) =t 0 +c
and the position for the final prediction Idx(<A>) =T . Since we only need one attention head for the
second attention layer, we will omit the index for heads and only keep the index for layers.
Proposition B.3(Second layer attention).Under the construction of Proposition B.2 and input format
specified in Section B.2, for any fixed ε∈(0,1/2) , there exists a construction of the second-layer
attention parametersθ (1)
Attn = (Q(1),K (1),V (1),O (1))s.t.
h(1.5)
Idx([tc]) =
"
1√
|Vc|
P
v∈Vc
˜uv +
Pm
j=1,sj ∈Vc ˜utj
√
3|{j|sj ∈Vc,j∈[m]}|
0d−dTE
#
+

Uε(c)
0d−dTE

,∀c∈ {0} ∪[C],
and
h(1.5)
Idx(<A>) =


1√
|VC |+1
˜u<A> + 1√
3(˜uc1 + ˜uc2)
1√
|VC |+1
P
v∈VC
˜uv
0dTE+dPE

 +


Uε(C+1)
0dTE
0dTE+dPE

 ,
whereε (c) ∈R V and∥ε (c)∥∞ < εfor allc∈ {0} ∪[C+ 1].
19

<!-- page 20 -->

Proof. In the second attention layer, we aim to construct key and query matrices such that (1) the
current thought [tc] will pay attention to all edges if the source node of the edge is contained in the
superposition (see Figure 3); (2) the answer token <A> pays large attention to the reasoning token
<R>which stores the two candidate destination tokens in its buffer space (see Figure 7).
Figure 7: Illustration of the second layer attention mechanism for final prediction.
First, we construct
Q(1) =[IdTE 0dTE×dTE 0dTE×dTE 0dTE×dPE]∈R dTE×d,
K(1) =[τ ˜u<A> ⊗ ˜u<R> τI dTE 0dTE×dTE 0dTE×dPE]∈R dTE×d,
where the value ofτ >0will be decided later.
We defineq i =Q (1)h(1)
i ,k i =K (1)h(1)
i and let
si,j = exp(⟨qi,k j⟩)P
j′≤i exp(⟨qi,k j′⟩) .
Note that we have T=Idx(<A>) =t 0 +C+ 1 . By the construction of the query and key
matrices, we have qi =content(h (1)
i ),∀i∈[T] , kIdx(<R>) =τ·buffer 1(h(1)
Idx(<R>)) +τ ˜u<A> and
ki =τ·buffer 1(h(1)
i ) for other i. Now we consider attention weights for i=Idx([t c]) =
t0 +c,∀c∈ {0} ∪[C]andi=Idx(<A>) =T.
For i=Idx([t c]) for any fixed c∈ {0} ∪[C] , we have qi = 1√
|Vc|
P
v∈Vc
˜uv. By Proposition B.2,
we have kIdx(<e>,j) = τ√
3 ˜usj for j∈[m] , kIdx(<R>) =τ ˜u<A> and kj = 0 for other j≤i . Define
Ic ={Idx(<e>, j)|s j ∈ V c forj∈[m]}. Therefore,
⟨qi,k j⟩= τp
3|Vc|
1{j∈ I c}.
Then we have
si,j =
exp

τ /
p
3|Vc|

·1{j∈ I c}
|Ic|exp

τ /
p
3|Vc|

+ (i− |I c|)
+ 1{j /∈ I c}
|Ic|exp

τ /
p
3|Vc|

+ (i− |I c|)
.
For i=Idx(<A>) =T , note that qT = 1√
|VC |+1
˜u<A>. Then ⟨qT ,k j⟩ is non-zero only when
j=Idx(<R>), and the inner product is τ√
|VC |+1
. Then we have
sT,j =
exp

τ /
p
|VC|+ 1

·1{j=Idx(<R>)}
exp

τ /
p
|VC|+ 1

+ (T−1)
+ 1{j̸=Idx(<R>)}
exp

τ /
p
|VC|+ 1

+ (T−1)
.
By choosing a large enoughτ, we have that
sIdx([tc]),j =( 1
|Ic| −ε c,1)·1{j∈ I c}+ε c,2 ·1{j /∈ I c},∀c∈ {0} ∪[C],
sT,j =(1−ε C+1,1)·1{j=Idx(<R>)}}+ε C+1,2 ·1{j̸=Idx(<R>)},
20

<!-- page 21 -->

whereε c,1, εc,2 ∈(0, ε/T),∀c∈ {0} ∪[C+ 1].
Now we construct the value and output matrix where
V(1) =[0dTE×dTE 0dTE×dTE IdTE 0dTE×dPE]∈R dTE×d,
O(1) =[IdTE 0dTE×(d−dTE)]⊤ ∈R d×dTE .
Then vi
∆
=V (1)h(1)
i =buffer 2(h(1)
i )∈R dTE reads from buffer space 2, and O(1)vi =
[buffer2(h(1)
i )⊤ 0⊤
d−dTE]⊤ ∈R d writes to the content space. Note that
Attnθ(1)
Attn
(h(1))i =
iX
j=1
si,jO(1)vj =
Pi
j=1 si,jbuffer2(h(1)
j )
0d−dTE

.
Since,h (1.5)
i =h (1)
i +Attn θ(1)
Attn
(h(1))i, we have
h(1.5)
Idx([tc]) =
"
1√
|Vc|
P
v∈Vc
˜uv + ( 1
|Ic| −ε c,1)Pm
j=1,sj ∈Vc
˜utj√
3 +ε c,2
P
j∈[t0+c]\Ic buffer2(h(1)
j )
0d−dTE
#
and
h(1.5)
T =


1√
|VC |+1
˜u<A> + (1−ε C+1,1)
˜uc1+˜uc2√
3 +ε C+1,2
P
j∈[T]\{Idx(<R>)} buffer2(h(1)
j )
1√
|VC |+1
P
v∈VC
˜uv
0dTE+dPE

 .
Combining the fact that buffer2(h(1)
j ) is either zero or equal to the superposition of ˜uv for some
v∈Vocwith norm less than 1, we can obtain the final result.
After the second-layer attention, the current thought h(1.5)
t0+c now contains all vertices in Vc and all
vertices that can be reached within one step from Vc, which is Vc+1 by definition. The subsequent
MLP layer will then equalize the weights of each vertex and eliminate the noise in the continuous
thought.
Proposition B.4(Second layer MLP).Under the construction of Proposition B.3 and the input format
specified by Section B.2, there exists a construction of θ(1)
MLP such that the output of the second-layer
MLP satisfies:
•h (2)
Idx([tc]) = 1√
|Vc+1|
P
v∈Vc+1 uv,∀c∈ {0} ∪[C],
•h (2)
Idx(<A>) = 1√
|VC |+5
(uc1 +u c2 +u <A> +P
v∈VC
uv).
Proof.Similar to Proposition B.2, we let
MLPθ(1)
MLP
(h(1.5))i =W (1)
2 σ(1)
ε (W(1)
1 h(1.5)
i )∈R d
where ε >0 and the (elementwise) non-linearity σ(1)
ε (·) will be specified later. We only consider
h(1.5)
i wherei=t 0 +cfor0≤c≤C+ 1. By Proposition B.3, we can decompose
h(1.5)
t0+c =


Uλ(c)
0dTE
0dTE+dPE

 ∀c∈ {0} ∪[C],h (1.5)
T =


Uη(1)
Uη(2)
0dTE+dPE

 ,
whereλ (c) = [λ(c)
1 , . . . , λ(c)
V ]⊤,η (1) = [η(1)
1 , . . . η(1)
V ]⊤,η (2) = [η(2)
1 , . . . η(2)
V ]⊤ ∈R V . Let
W(1)
1 =

U⊤ 0V×(d TE+dPE)
U⊤ 0V×(d TE+dPE)

∈R (2V)×d ,W (1)
2 =

U U
0(d−dTE)×V 0(d−dTE)×V

∈R d×(2V) ,
21

<!-- page 22 -->

andσ (1)
ε (x) =1{x≥ε}. Then
MLPθ(1)
MLP
(h(1.5))t0+c =W(1)
2 σ(1)
ε (W(1)
1 h(1.5)
t0+c)
=W(1)
2 σ(1)
ε (

U⊤ 0V×(d TE+dPE)
U⊤ 0V×(d TE+dPE)


Uλ(c)
0dTE
0dTE+dPE

)
=

U U
0(d−dTE)×V 0(d−dTE)×V

σ(1)
ε (

U⊤Uλ(c)
0V

)
=

Uσ(1)
ε (λ(c))
0d−dTE

,
and similarly,
MLPθ(1)
MLP
(h(1.5))T =W(1)
2 σ(1)
ε (W(1)
1 h(1.5)
T )
=W(1)
2 σ(1)
ε (

U⊤ 0V×(d TE+dPE)
U⊤ 0V×(d TE+dPE)


Uη(1)
Uη(2)
0dTE+dPE

)
=

U U
0(d−dTE)×V 0(d−dTE)×V

σ(1)
ε (

U⊤Uη(1)
U⊤Uη(2)

)
=
"
U

σ(1)
ε (η(1)) +σ (1)
ε (η(2))

0d−dTE
#
.
Now we choose ε= 1
4n where n=|V| is the number of vertices in the graph. By Proposition B.3, we
can make sure that ∥ε(c)∥∞ < 1
16n for all c∈ {0} ∪[C+ 1] where ε(c) is defined in Proposition B.3.
We define∆c ={t j |s j ∈ V c, j∈[m]} which is the set of vertices that can be reached within one
step from the currently explored vertex set Vc. By definition, we have Vc+1 =V c ∪∆ c. We consider
any fixedc∈ {0} ∪[C]. Forv∈ V c ∪∆ c, it holds that
λ(c)
v ≥min{1/
p
|Vc|,1/(
√
3|∆c|)}+ε (c)
v ≥ 1
2n − 1
16n > 1
4n =ε.
For otherv,
λ(c)
v ≤ε (c)
v < 1
16n < ε.
Therefore,σ (1)
ε (λ(c)
v ) =1{v∈ V c ∪∆ c}=1{v∈ V c+1}, and thus we have
MLPθ(1)
MLP
(h(1.5))t0+c =
P
v∈Vc+1
˜uv
0d−dTE

,∀c∈ {0} ∪[C].
Also, we have η(1)
c1 = 1√
3 +ε (C+1)
c1 , η(1)
c2 = 1√
3 +ε (C+1)
c2 , η(1)
<A> = 1√
|VC |+1
+ε (C+1)
<A> , and η(1)
v < 1
4n
for other v∈Voc . Moreover, η(2)
v = 1√
|VC |+1
·1{v∈ V C}, which implies σ(1)
ε (η(2)
v ) =1{v∈ V C}.
Then
MLPθ(1)
MLP
(h(1.5))T =
˜uc1 + ˜uc2 + ˜u<A> +P
v∈VC
˜uv
0d−dTE

.
Note that by our construction of the input sequence, one and only one of ci is in VC, and thus
∥MLPθ(1)
MLP
(h(1.5))T ∥2 =
p
|VC|+ 5.
By applying layer normalization, we can obtain the final result.
B.4 Proof of the main theorem
Proof of Theorem 1. We use the same construction of θ as in Lemma 2 where the maximum input
length is Tmax. By Lemma 2, we haveht0+c = 1√
|Vc|
P
v∈Vc uv, for 0≤c≤C . Then (h1, . . . ,hT )
22

<!-- page 23 -->

satisfies the input format specified in Section B.2, and by Proposition B.4, we have
TFθ(h1, . . . ,hT ) =h (2)
T = 1p
|VC|+ 5
(uc1 +u c2 +u <A> +
X
v∈VC
uv)
= 1p
|VC|+ 5
(2uci∗ +u c3−i∗ +u <A> +
X
v∈VC \{ci∗ }
uv)
since ci∗ ∈ V C and c3−i∗ /∈ VC by the property the graph reachability problem. We set WO =
[u1,u 2, . . . ,uV ]⊤ and then
WOTFθ(h1, . . . ,hT ) = 1p
|VC|+ 5
(2eci∗ +e c3−i∗ +e <A> +
X
v∈VC \{ci∗ }
ev)
=⇒fTFθ,C,WO(h[t0]) = arg max
v∈Voc
WOTFθ(h1, . . . ,hT ) =c i∗ .
B.5 Properties of sinusoidal positional encodings
Lemma 4.For any integerℓ≥1, there exists a matrixR (ℓ) ∈R dPE×dPE, such that
¯pi+ℓ =R (ℓ)¯pi,∀i≥1.
Proof.For anyj∈[d PE/2], we construction a two-dimensional rotation matrix
R(ℓ,j) =

cos(ℓ·ω j)−sin(ℓ·ω j)
sin(ℓ·ω j) cos(ℓ·ω j)

.
By definition of the rotation matrix, for anyi≥1, we have
R(ℓ,j)

¯pi,2j−1
¯pi,2j

=R (ℓ,j)

cos
 
i·ω j
sin
 
i·ω j

=

cos
 
(i+ℓ)·ω j
sin
 
(i+ℓ)·ω j

=

¯pi+ℓ,2j−1
¯pi+ℓ,2j

.
Then by setting R(ℓ) = diag{R(ℓ,1),R (ℓ,2), . . . ,R(ℓ,dPE/2)} ∈R dPE×dPE we can obtain the desired
result.
Lemma 5.For any integer T≥1 , there exists εT >0 , s.t. ⟨¯pi, ¯pj⟩ ≤d PE/2−ε T for any i, j∈[T]
wherei̸=j. Also,⟨ ¯pi, ¯pi⟩=d PE/2for alli∈[T].
Proof.For anyi, j∈[T], we have
⟨¯pi, ¯pj⟩=
dPEX
k=1
pi,kpj,k
=
dPE/2X
k=1
(cos(i·ω k) cos(j·ω k) + sin(i·ω k) sin(j·ω k))
=
dPE/2X
k=1
cos((i−j)ω k).
Note that for i=j , we can directly obtain that ⟨pi, pj⟩=d PE/2 since cos((i−j)ω k) = cos 0 = 1
for i=j and any k∈[d PE/2]. Also, since (i−j)ω k = i−j
M 2k/dPE is not a multiplier of 2π for
i̸=j, k∈[d PE/2], we havecos((i−j)ω k)<1. Let
εT,k = 1−max
i,j∈[T],i̸=j
cos((i−j)ω k)>0,∀k∈[d PE/2],
and letε T =PdPE/2
k=1 εT,k >0. Therefore, fori̸=j, we have
⟨¯pi, ¯pj⟩=
dPE/2X
k=1
cos((i−j)ω k)≤
dPE/2X
k=1
(1−ε T,k) =d PE/2−ε T .
23

<!-- page 24 -->

B.6 Implementing attention chooser under rotary position embedding
In this section, we discuss how to extend our constructions under sinusoidal positional encoding, a
widely used absolute positional encoding, to the rotary position embedding (RoPE) [Su et al., 2024],
a widely used relative positional encoding, to solve graph reachability. Since in our construction,
the positional encoding only functions in the first attention layer, and the building blocks of the first
attention layers are attention choosers, we mainly focus on how to build the attention chooser block
under RoPE.
Since RoPE is a relative positional encoding, we don’t usepi in computation and thus don’t need the
last dPE entries in the embedding vectors. Also, since the attention chooser only uses the information
in the content space, we omit the two buffer spaces and only keep two dimensions to make our
construction clean and assume d=d TE+2 in this section for the simplicity of the notation. Therefore,
we have uv = [˜u⊤
v ,0,0] ⊤ ∈R d for all v∈Voc in this section, where {˜uv}v∈Voc are orthonormal.
Also, assume the input embedding of attention layers satisfies hi = [˜h⊤
i ,1,0] ⊤ ∈R d. This can be
achieved easily by adding a bias before the attention layer or modifying the (dTE + 1)-th entry of
token embeddings.
Recall the definition of RoPE below:
Definition 2(Rotary position embedding [Su et al., 2024]).Let d be even. For any integer i and any
indexk∈[d/2], we define
R(i,k) =

cos(i·ω k)−sin(i·ω k)
sin(i·ω k) cos(i·ω k)

.
Let
R(i)
TE = diag{R(i,1),R (i,2), . . . ,R(i,dTE/2)} ∈R dTE×dTE
and
R(i) = diag{R(i,1),R (i,2), . . . ,R(i,d/2)} ∈R d×d,
where ω=M −2/d and M >0 is a large constant integer, e.g., M= 10 4 as chosen in Su et al.
[2024]. We make one additional assumption that M≥T where T is the maximum length of the input
sequence. Then the query vectorq i and key vectork i in Algorithm 2 will be calculated as
qi =R (i)Qhi,k i =R (i)Khi.
Now we show the counterpart of Lemma 3 under RoPE below.
Lemma 6(Attention chooser under RoPE).Fix any token <x>∈Voc , integer ℓ≥0 , and ε∈(0,1) .
Under rotary position embedding as defined in Definition 2, there exists a construction ofK,Q∈
R(2d)×d, such that for any input sequence(h 1, . . . ,hT )that satisfies
˜hi =
X
v∈Voc
λi,v ˜uv,whereλ i,v ≥0∀v∈Voc,
X
v∈Voc
λ2
i,v = 1,∀i∈[T],(6)
and satisfies ⟨˜u<x>, ˜hi⟩ ∈ {0,1} (i.e., each input embedding is either equal to the embedding of token
<x> or orthogonal to it) and ⟨˜u<x>, ˜hi⟩= 0 for i≤ℓ (i.e., the first ℓ tokens are not <x>), it holds
that for anyi∈[T],
if⟨ ˜hi, ˜u<x>⟩= 1, thens i,i−l >1−ε, otherwises i,1 >1−ε,
where si,j is the attention score from the i-th token to the j-th token as defined in Algorithm 2 with
the modification defined in Definition 2 with the input sequence(h1, . . . ,hT ).
Proof. Note that (6) implies that each input embedding is a normalized superposition of token
embeddings in the vocabulary (ignoring the last two entries). We aim to construct an attention head
such that when the i-th token is <x>, it will pay almost all attention to the position i−ℓ , otherwise it
will pay almost all attention to the BOS token <s> (known as the attention sink). We define vector
˜u ¯<x> =P
v∈Voc\{<x>} ˜uv ∈R d, which is the superposition of all token embeddings in the vocabulary
except for <x>. Moreover, for any integer i, we define pTE
i = (pi,1, . . . , pi,dTE)⊤ ∈R dTE, and define
p fTE
i = (pi,d−1, pi,d)⊤ ∈R 2, where for anyj∈[d/2], we have
pi,2j−1 = cos
 
i·ω j
, p i,2j = sin
 
i·ω j
,
24

<!-- page 25 -->

andω=M −2/dPE whereMis defined in Definition 2.
Now we construct query and key matrices to be
Q=
 0dTE×dTE pTE
0 ⊗1 2
ξp fTE
−T ⊗ ˜u ¯<x> 02×2

∈R d×d,
K=
"
0dTE×dTE ηR(ℓ)
TE(pTE
0 ⊗1 2)
02×dTE ηp fTE
0 ⊗1 2
#
∈R d×d,
where ξ, η >0 will be specified later and R(ℓ)
TE ∈R dTE×dTE ,R (−T,d/2) ∈R 2×2 are defined as in
Definition 2 and 1m denotes the all-one vector of dimension m. By Lemma 4, one can calculate that
R(j)
TEpTE
i =p TE
i+j,R (j,d/2)p
fTE
i =p
fTE
i+j,for any integersi, j.
Therefore,
qi =R(i)Qhi =R (i)
 pTE
0
ξ⟨˜u ¯<x>, ˜hi⟩p fTE
−T

,
ki =R(i)Khi =R (i)
"
ηR(ℓ)
TEpTE
0
ηp fTE
0
#
=R (i)
ηpTE
ℓ
ηp fTE
0

.
Then for any1≤j≤i≤T, we have
⟨qi,k j⟩=η

⟨R(i)
TEpTE
0 ,R (j)
TEpTE
ℓ ⟩+ξ⟨ ˜u ¯<x>, ˜hi⟩⟨R(i)p
fTE
−T ,R (j)p
fTE
0 ⟩

=η

⟨pTE
i ,p TE
j+ℓ⟩+ξ⟨ ˜u ¯<x>, ˜hi⟩ · ⟨p
fTE
i−j−T ,p
fTE
0 ⟩

.
Now we fix i∈[T] . We first consider the case where ⟨˜hi, ˜u<x>⟩= 1 (which also implies i > ℓ ). By
(6) and the assumption that token embeddings are orthonormal, we have
⟨˜hi, ˜uv⟩= 0,∀v∈Voc\{<x>}=⇒ ⟨ ˜hi, ˜u ¯<x>⟩= 0.
Therefore, we have⟨q i,k j⟩=η⟨p TE
i ,p TE
j+ℓ⟩. By Lemma 5, we have
⟨qi,k i−ℓ⟩=η⟨p TE
i ,p TE
(i−ℓ)+ℓ⟩=η⟨p TE
i ,p TE
i ⟩=ηd TE/2
and
⟨qi,k j⟩=η⟨p TE
i ,p TE
j+ℓ⟩ ≤ηd TE/2−ηε T ,∀j̸=i−ℓ.
where εT >0 . This implies ⟨qi,k j⟩ is maximized when j=i−ℓ with a non-zero gap ηεT , and
therefore, we have
si,i−ℓ = exp(⟨qi,k i−ℓ⟩)P
j∈[i] exp(⟨qi,k j⟩) ≥ exp(ηεT )
exp(ηεT ) + (i−1) .(7)
Now we consider the case where ⟨˜hi, ˜u<x>⟩= 0 . Again, by (6) and the orthonormal assumption of
token embeddings, we have
⟨˜hi, ˜u ¯<x>⟩=
X
v∈Voc\{<x>}
⟨˜hi, ˜uv⟩=
X
v∈Voc\{<x>}
λi,v ≥
X
v∈Voc\{<x>}
λ2
i,v = 1.
Note that for anyj∈ {2, . . . , i},
⟨p
fTE
i−j−T ,p
fTE
0 ⟩= cos((i−j−T)ω d/2)
= cos
 T−(i−j)
M

<cos
 T−(i−1)
M

=⟨p
fTE
i−1−T ,p
fTE
0 ⟩,
25

<!-- page 26 -->

where the inequality holds due toM≥Tand the cosine function decreases strictly in[0,1].
Then, we can defineξ= max 1≤j<i≤T
2dPE
⟨p fTE
i−1−T ,p fTE
0 ⟩−⟨p fTE
i−j−T ,p fTE
0 ⟩, and thus
ξ

⟨p
fTE
i−1−T ,p
fTE
0 ⟩ − ⟨p
fTE
i−j−T ,p
fTE
0 ⟩

≥2d PE,∀1≤j < i≤T.
Note that for anyj∈ {2, . . . , T}, we have
⟨qi,k 1⟩ − ⟨qi,k j⟩
=η

ξ⟨˜u ¯<x>, ˜hi⟩

⟨p
fTE
i−1−T ,p
fTE
0 ⟩ − ⟨p
fTE
i−j−T ,p
fTE
0 ⟩

−(⟨ ¯pi, ¯pj+ℓ⟩ − ⟨¯pi, ¯p1+ℓ⟩)

≥η(2d PE −d PE)
=ηdPE.
Therefore,
si,1 = exp(⟨qi,k 1⟩)P
j∈[i] exp(⟨qi,k j⟩) ≥ exp(ηdPE)
exp(ηdPE) + (i−1) .(8)
By choosing a sufficiently large η, the lower bound of (7) and (8) will both exceed 1−ε and thus the
proof is complete.
C Experiment Details
C.1 Dataset
Table 4: ProsQA statistics. Numbers are averaged over problem instances.
#Problems|V| |E|Sol. Len.
Train 14785 22.8 36.5 3.5
Val 257 22.7 36.3 3.5
Test 419 22.7 36.0 3.5
The statistics of the ProsQA dataset is shown in Table 4.
C.2 Stability of the Representation
Table 5: The inner products betweeni-th continuous thoughts and nodes (3 runs with different random
seeds).
Step 1 Step 2 Step 3 Step 4
Not Reachable−0.37,−0.25,−0.33−0.26,−0.04,−0.14−0.09,−0.01,0.02−0.25,−0.23,−0.27
Reachable3.59,3.62,3.71 1.55,1.42,1.37 0.80,0.77,0.62 0.61,0.66,0.53
–Frontier5.09,5.13,5.38 2.69,2.45,2.63 2.11,1.95,2.01 2.27,2.12,2.29
–Optimal6.41,6.52,6.84 4.78,4.67,5.11 6.00,5.44,6.43 9.48,8.98,9.58
To test whether COCONUTcan always learn the desired superpositional search behavior, we conduct
multiple experiments with 3 random seeds. We report the mean inner products between continuous
thoughts and each node group in Table 5, following the setting in Figure 6. The results are consistent
across multiple runs.
C.3 Computing Resources
Each run of COCONUTtakes about 24 hours on two Nvidia A100 80GB GPUs.
26
