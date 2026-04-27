# references/87_kq_svd_compressing_the_kv_cache_with_provable_guarantees_on_attention_fidelity.pdf

<!-- page 1 -->

KQ-SVD: Compressing the KV Cache with
Provable Guarantees on Attention Fidelity
Damien Lesens Beheshteh T. Rakhshan Guillaume Rabusseau
ENS de Lyon DIRO, Université de Montréal
Mila
DIRO, Université de Montréal
Mila - CIFAR AI Chair
Abstract
The Key–Value (KV) cache is central to the
efficiency of transformer-based large language
models (LLMs), storing previously computed
vectors to accelerate inference. Yet, as sequence
length and batch size grow, the cache becomes
a major memory bottleneck. Prior compression
methods typically apply low-rank decomposi-
tion to keys alone or attempt to jointly embed
queries and keys, but both approaches neglect
that attention fundamentally depends on their
inner products. In this work, we prove that such
strategies are sub-optimal for approximating
the attention matrix. We introduce KQ-SVD,
a simple and computationally efficient method
that directly performs an optimal low-rank de-
composition of the attention matrix via a closed-
form solution. By targeting the true source of re-
dundancy, KQ-SVD preserves attention outputs
with higher fidelity under compression. Exten-
sive evaluations on LLaMA and Mistral models
demonstrate that our approach consistently de-
livers superior projection quality.
1 Introduction
The rise of Large Language Models (LLMs) [Touvron
et al., 2023, Chaplot, 2023, Achiam et al., 2023, Guo
et al., 2025] has expanded AI capabilities beyond ear-
lier models. Transformers [Vaswani et al., 2017] replace
recurrence with self-attention, enabling parallelism and
improved sequence modeling, but their quadratic memory
and computation costs limit long-sequence scalability.
Key-Value (KV) caches are introduced to accelerate au-
toregressive generation by storing intermediate attention
KV vectors, avoiding redundant computation of shared
prefixes for each generated token. Although KV caching
reduces computational overhead, it substantially increases
memory consumption, as the cache size grows linearly
with both sequence length and batch size. This trade-
off motivates the development of KV cache compression
techniques, which are crucial for enabling efficient and
cost-effective deployment of LLMs across diverse hard-
ware platforms [Fu, 2024, Shi et al., 2024]. Variants
like Multi-Query Attention (MQA) [Shazeer, 2019] and
Grouped-Query Attention (GQA) [Ainslie et al., 2023]
reduce KV cache size by sharing or grouping query vec-
tors while maintaining performance comparable to full
Multi-Head Attention (MHA). However, they may intro-
duce accuracy trade-offs and hardware sensitivity, which
can affect performance generalization. Additional ap-
proaches, including sparse [Zhang et al., 2021] and lin-
earized [Katharopoulos et al., 2020] attention, further
reduce computational and memory costs, shaping KV
cache optimization strategies.
Another promising line of research exploits the low-
rank structure of KV caches to reduce memory over-
head. Multi-Head Latent Attention (MLA) [Liu et al.,
2024, Guo et al., 2025] maps tokens into the low-rank
latent space and stores these compressed representations
in place of the original key and value states. However, us-
ing MLA necessitates training the model from the ground
up. In contrast, ASVD [Yuan et al., 2024], LoRC [Zhang
et al., 2024], and Palue [Chang et al., 2025b] apply SVD to
key-value parameter matrices without retraining to build
low-rank projection modules. A key limitation of these
approaches is that they often compress only the keys, ne-
glecting the query-key interaction that underlies attention.
EigenAttention [Saxena et al., 2024] and Zack [Zhang and
Shen, 2024] attempt to address this by incorporating both
queries and keys in low-rank decompositions, yet their
behavior largely resembles that of SVD-based methods
that compress keys alone.
In this work, we address these limitations by introducing
KQ-SVD, a compression method that achieves optimal
low-rank approximation of the attention matrix efficiently
and in closed form. Our method explicitly captures the
arXiv:2512.05916v1  [cs.LG]  5 Dec 2025

<!-- page 2 -->

interactions between queries and keys through their inner
products, preserving the fundamental structure of atten-
tion. Beyond key-query interactions, we also consider
the corresponding interactions between values and the
output projection, enhancing the fidelity of the approxima-
tion. By leveraging the inherent low-rank structure of KV
caches [Yu et al., 2024, Saxena et al., 2024], we formulate
attention matrix approximation as a principled low-rank
decomposition problem. Our theoretical analysis quan-
tifies the error between prior key-only SVD approaches
and our optimal method, and shows that methods incor-
porating both queries and keys can degrade when keys
and queries are rescaled by the same factor, effectively
behaving like key-only SVD methods. Our contributions
can be summarized as follows:
• We introduce KQ-SVD, an optimal low-rank approx-
imation of the attention matrix capturing key-query
interactions.
• We theoretically quantify the advantages of KQ-
SVD over methods based on key low-rank decompo-
sition and SVD on concatenated queries and keys.
• We show that KQ-SVD is compatible with and also
optimal in the Grouped-Query Attention setting.
• We provide extensive empirical evaluations with
LLaMA2-7B, LLaMA2-13B, LLama3-8B and
Mistral-7B models on the C4 dataset demonstrat-
ing significant advantages of KQ-SVD over existing
low rank projection methods.
2 Related Works
Low-rank structure of the KV-cache.Several meth-
ods exploit the inherent low-rank structure of cached
key–value (KV) matrices to reduce memory footprint.
ECKVH [Yu et al., 2024] compresses the cache by group-
ing attention heads, performing singular value decompo-
sition (SVD) within each group, and retaining only the
dominant singular components. EigenAttention [Saxena
et al., 2024] generalizes this idea by constructing low-rank
bases that jointly approximate queries, keys, and values,
effectively lowering the dimensionality of KV representa-
tions. Q-Filters [Godey et al.] introduces a training-free
variant, projecting keys into a low-rank subspace via SVD
to approximate attention scores efficiently with minimal
accuracy loss. Moreover, [Yu et al., 2024] investigates the
intrinsic low-rank nature of KV caches and compresses
KV heads through careful grouping and SVD-based de-
composition. In contrast, Loki [Singhania et al., 2024]
adopts a two-stage strategy: it first estimates approximate
attention scores in a lower-dimensional space to rank and
select the most relevant keys, and then computes exact
attention scores using only the selected keys, reducing
both memory and computational cost.
KV Weights Compression.An alternative approach
targets the KV weight matrices themselves rather than
the cached matrices. LoRC [Zhang et al., 2024] applies
low-rank approximations directly to the key and value
weight matrices, achieving compression at the parame-
ter level. Palu [Chang et al., 2025b] follows a similar
strategy, jointly compressing key and value weight ma-
trices via SVD. ShadowKV [Sun et al., 2024] introduces
a distinct perspective by performing SVD on pre-RoPE
key matrices to reduce their dimensionality, demonstrat-
ing the versatility of low-rank methods in optimizing KV
representations.
Positioning of KQ-SVD.Although prior methods have
made significant strides in compressing KV caches and
their weight matrices, they often treat keys and values
independently or approximate attention only indirectly,
leaving the core query–key interactions underrepresented
and leading to sub-optimal low rank approximation of at-
tention matrices. KQ-SVD addresses these limitations by
formulating a principled, closed-form low-rank approxi-
mation of the full attention matrix.
3 Preliminary
In this section, we introduce our notations and present the
necessary background on Multi-Head Attention (MHA).
3.1 Notations
We use lower case bold letters for vectors (e.g., a,b ),
upper case bold letters for matrices (e.g., A,B ). A+ de-
notes the Moore–Penrose pseudo-inverse of A. Through-
out the paper, the singular value decomposition (SVD)
of a matrix S∈Rm×nis presented by S=UΣV ⊤,
with U∈Rm×n, V∈Rn×nmatrices with orthonormal
columns and Σ∈Rn×na diagonal matrix with positive
diagonal entries{σi}n
i=1. The columns of U and V are
called respectively the left and right singular vectors of
S, and theσi’s the singular values ofS, notedσi(S). The
optimal rank-R approximation of S with respect to the
Frobenius norm can be obtained via the SVD by truncat-
ing it to keep only the firstR singular vectors and singular
values: S≈ˆU ˆΣ ˆV⊤with ˆU∈Rm×R, ˆV∈Rn×Rand
ˆΣ∈RR×R. The column space ofSis notedR(S).
3.2 Background
In transformer architectures, self-attention assigns relative
importance to tokens, enabling the model to selectively
focus on different segments of the input sequence. For a

<!-- page 3 -->

sequence of token embeddings X∈RT×D, multi-head
attention is computed as
MHA(X) = [H 1,...,Hh]WO,
where
Hi = Softmax
(QiK⊤
i√
d
)
Vi,
with WQ
i ,WK
i ,WV
i ∈RD×d,d=D/h , Qi =XW Q
i ,
Ki =XW K
i , Vi =XW V
i , and WO ∈RD×D. In
masked attention, the upper-diagonal entries of the atten-
tion matrix QiK⊤
i are set to−∞to prevent a token from
attending to future positions.
This computation scales quadratically with the sequence
lengthT . In auto-regressive decoding, previously com-
puted key and value vectors are cached to avoid redundant
computation, reducing the per-token cost. Specifically at
timeT , for each head, new key–value pairs are concate-
nated to the caches K and V, followed by an attention
computation:
K←Concat(K,kT ),V←Concat(V,vT ),
hT =Softmax
(qTK⊤
√
d
)
V
where kT =x TWK, vT =x TWV , qT =x TWQ.
While caching mitigates redundant computation, generat-
ing theT -th token still incursO(T) cost, and the memory
footprint of stored keys and values grows linearly with
sequence length. For sufficiently long contexts, this mem-
ory requirement becomes a dominant bottleneck, as the
cumulative size of the KV cache can surpass the model pa-
rameters by several orders of magnitude. In the following
sections, we briefly review the K-SVD method [Chang
et al., 2025b, Yu et al., 2024, Zhang et al., 2024], which
compresses key representations using singular value de-
composition (SVD), and the Eigen approach [Saxena
et al., 2024], which jointly considers keys and queries
by vertically concatenating them and applying SVD to
the resulting matrix.
3.3 Cache compression with SVD
Recent works [Chang et al., 2025b, Zhang et al., 2024,
Chang et al., 2025a] demonstrate that singular value de-
composition (SVD) is a powerful tool for compressing
the KV cache in large language models, as the cache
exhibits low-rank structure. Let K=U KΣKV⊤
K ∈
RT×dbe the SVD of the key matrix, and let ˜K=
ˆUK ˆΣK ˆV⊤
K denote its rank-R truncated version. By the
Eckart–Young–Mirsky theorem,˜K is the best rank-R ap-
proximation of K under the Frobenius norm. In other
words, the optimization problem
min
P∈Rd×d
∥KP−K∥2
F s.t.rank(P)≤R,
is solved by P= ˆVK ˆV⊤
K, leading to the approximation
˜K=K ˆVK ˆV⊤
K = ˆUK ˆΣK ˆV⊤
K. Applying the same
procedure to the value matrix V=U V ΣV V⊤
V , we can
approximate the attention output as
˜H= Softmax(Q ˜K⊤/
√
d)˜V
= Softmax(Q ˆVK ˆV⊤
KK⊤/
√
d)V ˆVV ˆV⊤
V.
This formulation is particularly useful because it allows
to store only the compressed caches K ˆVK and V ˆVV
in memory. These matrices are of size R×Tinstead
ofd×T, resulting in significant memory savings. At
runtime, queries are multiplied by ˆVK, while ˆV⊤
V can
be absorbed into the output projection WO, streamlining
computation.
A key advantage of this approach is that the SVD does
not need to be computed during token generation. In-
stead, it can be performed once in a post-training cali-
bration phase. For each layer l and attention headi, we
only need to determine a basis Vi,l∈Rd×Rsuch that
Ki,l≈Ki,lVi,l(Vi,l)⊤with an analogous construction
for the values. This basis can be learned from a calibra-
tion set of sequences. Specifically, we passns calibration
sequences (e.g., sampled from a high-quality dataset such
as C4 [Raffel et al., 2020]) through the model. The kth
sequence produces caches Kk
i,l and Vk
i,l for every layerl
and headi. These are concatenated to form large cache
matrices Ki,l =
(K1
i,l,K 2
i,l,...,Kns
i,l
)
. This aggregated
cache provides a representative sample of the key vec-
tors that will appear during inference. Performing SVD
on Ki,l then yields the dominant singular vectors, which
form a suitable low-rank basis for compression. The cost
of generating calibration caches and computing the SVDs
is negligible compared to model training, and is offset
by the runtime speedups from cache compression. In the
following, we will refer to this method as K-SVD.
Rank selection.The compression rank R is determined
per layer by examining the singular value spectrum. Let
{σj}j denote the singular values of a matrix M. For a
relative error toleranceϵ, we select the smallest R such
that
∥M−˜M∥2
F≤ϵ∥M∥2
F⇔
∑R
j=1σ2
j
∑d
i=1σ2
j
≥1−ϵ.
Prior studies [Yu et al., 2024, Saxena et al., 2024] have
shown that KV matrices are indeed approximately low-
rank, so substantial compression can be achieved with
small error budgetϵ. The chosen rank may differ for keys
and values depending on their spectra.

<!-- page 4 -->

3.4 Cache compression with Eigen
Other works [Saxena et al., 2024, Zhang and Shen, 2024]
emphasize that queries should also be considered when
compressing key caches. Indeed, by projecting the key
cache we also project the query matrix: Q ˆVK ˆV⊤
KK⊤=
(Q ˆVK ˆV⊤
K)(K ˆVK ˆV⊤
K)⊤as ˆVK ˆV⊤
K is an idempotent
matrix. Hence, it makes sense to compute the low rank
projection by solving
min
S
∥K−KS∥2
F +∥Q−QS∥2
F s.t.rank(S)≤R,
which is equivalent to
min
S

[
K
Q
]
−
[
K
Q
]
S

2
F
s.t.rank(S)≤R,
so that S will approximate queries and keys simultane-
ously. As the second formulation of the optimization prob-
lem shows, S can be computed by performing an SVD
on the combined matrix
[
K
Q
]
. This approach ensures
that the learned projection preserves keys and queries
while reducing dimensionality. We refer to this approach
as Eigen throughout this paper. The calibration process
follows the same procedure as K-SVD: large calibration
caches are formed by using a collection of calibration
sequences. The only difference is that query matrices
from the calibration set are also used in the projection
computation.
4 Methodology
In this section, we introduce our proposed approach KQ-
SVD for KV cache compression. We consider taking
the interaction between queries and keys into account.
The method views the key and query matrices as a single
entity and applies singular value decomposition (SVD) to
KQ⊤. We begin by outlining the motivation behind this
idea, followed by a detailed explanation of the technique.
4.1 Motivation
Existing compression methods based on SVD typically
compress keys or jointly embed Q/K/V. Theorem 1
(proof in Appendix A) inspired by [Wang et al., 2025]
shows why that can fail: perturbations in K are amplified
by the inner products QK⊤and further propagated by
the value multiplication.
Theorem 1.Let X∈RT×dbe a sequence of token
embeddings,K,Q,V∈RT×dand
MHA(X) =
[
Softmax(QiK⊤
i /
√
d)Vi)
]
iWO,
^MHA(X) =
[
Softmax(Qi˜Ki
⊤
/
√
d)˜Vi)
]
iWO,
where ^MHA(X),˜Ki and˜Vi represent the approximation
of MHA(X),Ki and Vi, respectively. The difference
between the actual attention output and the one produced
with approximate keys and values is upper bounded as
∥^MHA(X)−MHA(X)∥2
≤
h∑
i=1
∥ViWO
i∥2√
d
∥QiK⊤
i −Qi˜Ki
⊤
∥2
+∥ViWO
i −˜ViWO
i∥2.(1)
The method proposed by this paper stems from the will to
minimize this upper bound on the output approximation
error. In each attention head, for a key cache K and a
calibration set of query vectors{q(j)}j∈calibration(row
vectors of sized), we want to minimize
∑
j∈calibration
∥q(j)K⊤−q(j)˜K⊤∥2
2,
over low-rank ˜K. This objective is equivalent to a low-
rank approximation of the query–key interaction matrix
and admits an optimal closed-form solution via the singu-
lar value decomposition. Similarly, to compress values we
directly optimize the second term of Eq.(3) over low-rank
˜Vfor each attention head (see Appendix B).
4.2 Proposed Method
In the following, we drop the head index i for ease of
clarity. Our method addresses cache compression by pro-
ducing an optimal solution to
min
˜K
∥QK⊤−Q˜K⊤∥2
F s.t.rank( ˜K)≤R,
with Q=
(
q(j))
j∈calibrationthe collection of query vectors
from the calibration set. This formulation jointly consid-
ers the properties of both K and Q, while taking into
account their interaction through the inner product. The
optimization problem can be restated using a projection
matrix as
min
S
∥KSQ⊤−KQ⊤∥2
F s.t.rank(S)≤R,
where S∈Rd×d. To be able to compress the key cache
during the attention computation, we write the projection
matrix S as a product of two matrices, i.e., S=AB with
A,B∈Rd×R. The minimization problem tackled by
KQ-SVD is thus
min
A,B∈Rd×R
∥KAB⊤Q⊤−KQ⊤∥2
F.(2)
Given A∗and B∗the solutions of this optimization prob-
lem, we store the low-rank matrix KA∗∈RT×R, which

<!-- page 5 -->

allows for compression while maintaining an accurate
reconstruction of the attention matrix. The same strat-
egy also applies to the value–output matrices (see Ap-
pendix B). In the next section, we state the main theorem
which demonstrates that the optimization problem de-
scribed above admits an optimal solution which can be
computed efficiently.
4.3 KQ-SVD: Optimal attention factorization
The following theorem establishes the provably optimal
low-rank factorization of the key–query matrix, which
admits a simple closed-form solution.
Theorem 2.Let K,Q∈RT×dbe key and query cache
matrices. The optimal solution to the low rank attention
approximation problem
min
A,B∈Rd×R
∥KAB⊤Q⊤−KQ⊤∥F,
is given by
A=K + ˆU,B=K T ˆU,
where ˆU∈RT×Ris the matrix having the top R left
singular vectors ofKQ⊤as columns.
Proof. Observe that KAB⊤Q⊤has rank at most R.
Hence,
min
A,B∈Rd×R
∥KAB⊤Q⊤−KQ⊤∥F,
is lower bounded by
min
M∈RT×T
∥M−KQ⊤∥F s.t.rank(M)≤R.
By the Eckart-Young theorem, we know the best rankR
approximation ofKQ⊤is given by its truncated SVD:
KQ⊤≃ˆU ˆΣ ˆV⊤,
where ˆU∈RT×R, ˆΣ∈RR×Rand ˆV∈RT×R. We
will show that KAB⊤Q⊤= ˆU ˆΣ ˆV⊤from which the
optimality ofAandBfollows.
Let K=UΣV be the full SVD of K. Observe that we
have the inclusion of (column) spans:R( ˆU)⊆R(U) =
R(K). Since KK+ is the orthogonal projection onto
R(K), we haveKK + ˆU= ˆU, hence
KAB⊤Q⊤=KK + ˆU ˆU⊤KQ⊤= ˆU ˆU⊤KQ⊤
= ˆU ˆU⊤UΣV⊤= ˆU ˆΣ ˆV⊤.
Therefore A=K + ˆU and B=K ⊤ˆU are optimal solu-
tions tomin A,B∈Rd×R∥KAB⊤Q⊤−KQ⊤∥F .
The Moore-Penrose pseudo inverse ofK can be expressed
through the SVD of K as K+ =V KΣ−1
K U⊤
K. The
singular value decomposition of KQ⊤∈RT×Tcan
be computed efficiently as its rank is at most d. In-
deed, we can first perform an SVD on K=U KΣKV⊤
K
and Q=U QΣQV⊤
Q, then on the d×d matrix
ΣKV⊤
KVQΣQ =U′Σ′V′⊤. The SVD of KQ⊤is then
UKU′Σ′(VQV′)⊤=UΣV ⊤with U=U KU′∈
RT×dand V=V QV′∈RT×d. This way the opti-
mal solution provided by Theorem 2 can be computed
efficiently in timeO(Td 2).
A similar approach is used by KQ-SVD to derive opti-
mal projections for the low rank approximation of the
product of the value cache with the output matrix (see
Appendix B).
5 Theoretical analysis
In this section, we focus on the minimization problem
in Eq. (5); the same reasoning applies to value and out-
put matrices (see Appendix B). We first provide an exact
formula quantifying the accuracy difference between K-
SVD and KQ-SVD. We then exhibit a failure mode of
Eigen method [Saxena et al., 2024] which KQ-SVD cir-
cumvent by design. Finally, we show how KQ-SVD also
obtains optimal low rank approximation in the Grouped
Query Attention (GQA) framework.
5.1 Comparing K-SVD and KQ-SVD
In this section, we characterize the optimality gap between
KQ-SVD and K-SVD (Section 3.3) and derive a closed
form expression of their accuracy difference.
Theorem 3.Let opt = min A,B∈Rd×R∥KAB⊤Q⊤−
KQ⊤∥2
F be the optimal error for the low rank attention
approximation problem (achieved by KQ-SVD), and let
errK-SVD =∥KˆVK ˆVT
KQT−KQT∥2
F be the error of K-
SVD. We have
errK-SVD−opt =
R∑
i=1
σi(KQ⊤)2−∥KˆVK ˆVT
KQT∥2
F≥0,
with equality only if the top R left singular vectors of K
and the topRofKQ ⊤span the same subspace.
Proof. Let K≈ˆUK ˆΣK ˆV⊤
K be the rank R truncated
SVD ofKandKQ ⊤≈ˆU ˆΣ ˆV⊤be the one ofKQ⊤.
By Theorem 2, opt =∥ˆU ˆU⊤KQT −KQT∥2
F =∑d
i=R+1σi(KQT )2.

<!-- page 6 -->

On the one hand we have
∥KQT∥2
F =
d∑
i=1
σi(KQT )2 =
R∑
i=1
σi(KQT )2 + opt.
On the other hand,
∥KQ⊤∥2
F =∥KQ⊤−ˆUK ˆU⊤
KKQ⊤+ ˆUK ˆU⊤
KKQ⊤∥2
F
=∥(I−ˆUK ˆU⊤
K)KQ⊤∥2
F +∥ˆUK ˆU⊤
KKQ⊤∥2
F,
since (I−ˆUK ˆU⊤
K) and ˆUK ˆU⊤
K are projections on or-
thogonal subspaces.
Since K ˆVK ˆV⊤
K = ˆUK ˆU⊤
KK, the left term is ex-
actly errK-SVD and the second term can be reduced to
ˆUK ˆU⊤
KKQ⊤=K ˆVK ˆV⊤
KQ⊤.
Putting everything together, we get
R∑
i=1
σi(KQT )2 + opt = err K-SVD +∥KˆVK ˆV⊤
KQ⊤∥2
F,
which shows the equality in the theorem.
To show thaterr K-SVD−opt≥0, first observe that
errK-SVD−opt =∥ˆU ˆU⊤KQ⊤∥−∥ˆUK ˆU⊤
KKQ⊤∥.
A direct consequence of the Eckart-Young theorem is that,
for any matrix M∈RT×Tand anyR≤T, the solution
of
max
X∈RT×R
∥XX⊤M∥s.t.X⊤X=I
is obtained by setting the columns of X to the top R
left singular vectors of M. Hence, ∥ˆU ˆU⊤KQ⊤∥ ≥
∥ˆUK ˆU⊤
KKQ⊤∥and thus errK-SVD−opt≥0, with equal-
ity only if ˆU ˆU⊤= ˆUK ˆU⊤
K, i.e., when the top R left
singular vectors of K and the top R of KQ⊤span the
same subspace.
It is worth observing that equality between errK-SVD and
opthappensonly whenthe projection onto ˆVK captures
allof the energy (Frobenius norm) in the top R singular
values of KQ⊤. Since the best rank-R approximation in
the Frobenius norm is unique, this holds precisely when
the subspace spanned by the topR left singular vectors of
K coincides with that spanned by the topR left singular
vectors of KQ⊤. In other words, equality holdsprecisely
when these two subspaces match.
5.2 Comparing Eigen and KQ-SVD
We now compare Eigen [Saxena et al., 2024] with KQ-
SVD. Although we do not derive an exact value for the
optimality gap of Eigen, we identify a critical limitation
of Eigen: the method is highly sensitive to unbalance
between the norms of K and Q. Eigen’s performance can
be degraded simply by multiplyingK by a constantβand
dividing Q by the same constant. While this rescaling
leaves the attention computation unchanged and does
not affect KQ-SVD, it causes Eigen method to behave
increasingly like K-SVD as the unbalance between the
two norms grows. Theorem 4 formalizes this intuition.
Theorem 4.Let K,Q∈RT×d, let α=∥Q∥F
∥K∥F
and let
errEigen =∥KˆVEigen ˆV⊤
EigenQ⊤−KQ⊤∥be the error of
Eigen. If there is a non-trivial gap between the Rth and
(R+ 1) th singular values of K, i.e. σR(K)>σR+1(K),
thenlimα→0errEigen = err K-SVD.
Proof. Recall that Eigen approximate KQ⊤with
K ˆVEig ˆV⊤
EigQ⊤where ˆVEig is the matrix with the top
R right singular vectors of
[
K
Q
]
as columns. In the limit
where ∥Q∥F
∥K∥F
tends to 0, the concatenated matrix
[K
Q
]
tends to
[K
0
]
. Since σR(K)> σR+1(K), it follows
from the Davis-Kahan theorem [Davis and Kahan, 1970]
that, asαtends to 0, the space spanned by the topR right
singular vectors of
[K
Q
]
converges to the one of
[K
0
]
,
and thus of K. Hence limα→0ˆVEig ˆV⊤
Eig = ˆVK ˆV⊤
K
anderr Eigen tends toerr K-SVD.
5.3 Handling grouped query attention
Standard multi-head attention (MHA) is powerful
but slow at inference, while multi-query attention
(MQA) [Shazeer, 2019] is much faster but can hurt model
quality and requires retraining. Grouped-query attention
(GQA) [Ainslie et al., 2023], sits in between MHA and
MQA by letting groups of query heads share a key-value
head, balancing accuracy and performance. GQA orga-
nizes attention/query heads into groups of size m. All
query heads within a group share the same set of keys
and values, enabling more efficient computation with-
out significantly compromising model performance. We
have assumed in the presentation of our method that each
key head attends to a single query head. We show in
the following theorem that simply applying KQ-SVD on
the shared key cache and the concatenated query caches
provides the optimal approximation with GQA.
Theorem 5.Given a (shared) key cache matrix K∈
RT×dand m full column rank query cache matrices
Q1,···,Qm∈RT×d, solving
min
A,B1,···,Bm∈Rd×R
m∑
i=1
∥KAB⊤
i Q⊤
i −KQ⊤
i∥2
F,

<!-- page 7 -->

is equivalent to solving
min
A,B∈Rd×R
∥KAB⊤Q⊤−KQ⊤∥2
F,
where Q= [Q⊤
1 Q⊤
2 ···Q⊤
m]⊤∈RmT×dis the matrix
obtained by stacking themquery matrices.
Proof. We first show that the solution matricesBi can be
chosen to be all equal, i.e., that solving
min
A,B1,···,Bm∈Rd×R
m∑
i=1
∥KAB⊤
i Q⊤
i −KQ⊤
i∥2
F,
is equivalent to solving
min
A,B∈Rd×R
m∑
i=1
∥KAB⊤Q⊤
i −KQ⊤
i∥2
F.
Indeed, for any A and anyi, since Q⊤
i is full row rank,
the minimizers of minBi∥KAB⊤
i Q⊤
i−KQ⊤
i∥2
F are the
same as the minimizers of minBi∥KAB⊤
i −K∥2
F , and
are thus independent of Qi. Since this is true for any A,
and in particular for the optimal one, this shows that all
the solutionsBi can be taken to be equal.
The result then directly follows from the fact that the
squared Frobenius norm of a block matrix is equal to the
sum of the squared Frobenius norms of the blocks.
Theorem 5 states that optimal projections can be com-
puted for models using GQA by stacking query matrices
in each group and using KQ-SVD as in the non GQA
case. Computing the optimal projection for a head group
costsO(mTd 2), wherem is the size of the head group,
leading to an amortized cost per query head ofO(Td 2).
6 Experiments
The theoretical results established in the previous sections
require further validation on real cache matrices generated
by state-of-the-art LLMs. In this section, we first compare
KQ-SVD with K-SVD and Eigen, and then demonstrate
the claim of Theorem 4 using real-world cache data.
6.1 Comparing methods
Setup:To validate our theoretical claims, we run experi-
ments on several widely used LLMs. We test two models
without grouped-query attention (GQA)—Llama2-7B and
Llama2-13B [Touvron et al., 2023]—and two models with
GQA—Llama3-8B [Grattafiori et al., 2024] and Mistral-
7B-v0.3 [Jiang et al., 2023]. All experiments use the C4
dataset [Raffel et al., 2020], with projections learned on
the training split and evaluated on the validation split.
Learning projections:Following the methodology of
Saxena et al. [2024], we select 128 training sequences of
2048 tokens each from C4. Each sequence is passed
through the model, storing the key, value, and query
caches for every layer and attention head (queries are
needed by both Eigen and KQ-SVD). For each layer–head
pair, we collect the corresponding caches from all 128
sequences and concatenate them, yielding large matri-
ces K,Q,V∈RThuge×dwithThuge = 262,144. Because
model context length is limited and attention cost scales
quadratically with sequence length, it is not feasible to
build these matrices from a single long sequence. Instead,
we construct them from multiple shorter ones. Once the
large cache matrices are formed, we perform SVD and
apply the formulas presented in previous sections to com-
pute low-rank projections.
Rank selection:All methods are evaluated at the same
rank R, determined individually for each layer. For a
given layer, we analyze the singular value spectra of the
key and value matrices, averaged across heads, and choose
the smallest R that discards no more than an ϵ= 0.1
fraction of the spectral energy. That is, for singular values
{σj}j of K,
∑R
j=1σ2
j
∑d
i=1σ2
j
≥1−ϵ. This is equivalent to
requiring that the relative Frobenius error is at mostϵ.
Evaluation:We evaluate the learned projections on 32
validation sequences of 2048 tokens each. For every se-
quence, we extract the cache matrices (K,Q,V) at each
layer and head. Using these matrices, we can simulate
attention computations directly, since attention depends
only on these three components. Each cache is projected
onto its corresponding low-rank subspace to form approx-
imations, and we then compute the approximate Multi-
Head Attention output using the standard formulas. For
comparison, we also compute the exact (uncompressed)
attention output. This enables us to measure the relative
error of each method across all matrices of the attention
pipeline. Errors are averaged across validation sequences.
Metrics:For a matrix M and its approximation˜M, we re-
port the relative Frobenius norm error errFro =∥M−˜M∥2
F
∥M∥2
F
.
This error is computed for the key, query, and value ma-
trices, for the attention score matrix KQ⊤, and for the
Multi-Head Attention outputMHA(X).
Results:Results are shown in Figure 1. For each model,
the top plot reports the relative error on the attention out-
put across layers, while the bottom plot reports the aver-
aged errors on the intermediate components. We observe
that K-SVD provides the most accurate approximation of
the key matrices (as expected from the optimality of SVD),
but performs poorly on query matrices, leading to weaker
approximations of the attention scores and consequently

<!-- page 8 -->

0 10 20 30
0.1
0.2
0.3
0.4
0.5
0.6
Output error
 per layer
Llama2-7B
0 10 20 30 40
0.1
0.2
0.3
0.4
0.5
0.6
Llama2-13B
0 10 20 30
10 1
100
Llama3-8B
0 10 20 30
10 1
100
Mistral-7B-v0.3
K-SVD
Eigen
KQ-SVD
K Q V
Attn
Output
10 2
10 1
100
Mean
 Relative Error
 Across Layers
K Q V
Attn
Output
10 2
10 1
100
K Q V
Attn
Output
10 2
10 1
100
K Q V
Attn
Output
10 2
10 1
100
K-SVD
Eigen
KQ-SVD
Figure 1: Relative Frobenius approximation output error per layer (top) and mean relative errors on K, Q, V, KQ⊤
and attention layer output across layers (bottom) for Mistral and Llama models.
higher output errors. This effect is more pronounced in
GQA models, where sharing the key matrix across queries
in a group amplifies approximation errors.
In contrast, Eigen and KQ-SVD achieve comparable accu-
racy on keys, queries, and values. The key difference lies
in the attention score matrix: KQ-SVD consistently de-
livers higher accuracy, resulting in lower attention errors
and outperforming all other methods on all models.
6.2 UnbalancedKandQmatrices
We verify experimentally the claims of Section 5.2 and an-
alyze the attention approximation error under unbalanced
KandQmatrices.
Set up: We follow the same experimental setup as in
the previous section using the C4 train/validation split to
learn projections and evaluate their accuracy. The only
difference is that cache matrices are scaled to assess the
effect of unbalance. Key matrices are multiplied byβand
query matrices are divided by β. This is equivalent to
scaling the projection matrices WK,i/ WQ,i byβ, as the
operation permutes with matrix multiplication (this does
not change the output since the query and key matrices
are multiplied before any non-linear activation).
Metrics: For each unbalance ratio, we plot the relative
attention output error for the three methods, averaged
across all layers and validation sequences.
Results: Results are shown in Figure 2. As discussed in
Section 5.2, K-SVD and KQ-SVD are invariant to scaling
K by a factorβand dividing Q by the same factor, which
is confirmed by the constant error observed for these meth-
ods. As predicted theoretically, a higher unbalance ratio
brings Eigen closer to K-SVD; for β= 10, their errors
are nearly indistinguishable. This confirms Theorem 4
1 2 3 4 5 6 7 8 9 10
Unbalance factor
0.275
0.300
0.325
0.350
0.375
Relative Attention
 Computation Output
 Error
KQ-SVD
K-SVD
Eigen
Figure 2: Llama2-7B relative output approximation error
averaged across layers for varying unbalanced factorβ.
and exposes a limitation of Eigen, which underperforms
even under modest query-key unbalance.
7 Conclusion
We introduced KQ-SVD, a novel approach for computing
low-rank projections for KV cache compression. KQ-
SVD is driven by minimizing an upper bound on the
attention output approximation error, resulting in an opti-
mization problem that yields the optimal low-rank approx-
imation of the attention matrix. Crucially, this problem
admits a closed-form solution that can be computed effi-
ciently. We quantify the advantage of KQ-SVD over pre-
vious methods, either through exact error formulas or by
highlighting failure modes of prior approaches. Our tech-
nique is complementary to popular cache compression
methods such as GQA. Experiments validate our theoreti-
cal findings and demonstrate that considering the interac-
tion between queries and keys—as KQ-SVD—provides a
superior alternative to performing SVD on the key cache
or on concatenated keys and queries.

<!-- page 9 -->

Acknowledgment
This research is supported by the Canadian Institute for
Advanced Research (CIFAR AI chair program). This
work was completed while Damien Lesens interned at
Mila. This work made use of compute resources provided
by the Digital Research Alliance of Canada and by Mila
(mila.quebec).
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
Gqa: Training generalized multi-query transformer
models from multi-head checkpoints, 2023. URL
https://arxiv.org/abs/2305.13245.
Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang,
P Winston Michalak, Shahab Asoodeh, and Flavio P
Calmon. Beyond adult and compas: Fairness in multi-
class prediction.arXiv preprint arXiv:2206.07801,
2022.
Chi-Chih Chang, Chien-Yu Lin, Yash Akhauri, Wei-
Cheng Lin, Kai-Chiang Wu, Luis Ceze, and Mo-
hamed S Abdelfattah. xkv: Cross-layer svd for kv-
cache compression.arXiv preprint arXiv:2503.18893,
2025a.
Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-
Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi
Huang, Luis Ceze, Mohamed S Abdelfattah, and Kai-
Chiang Wu. Palu: Kv-cache compression with low-
rank projection. InThe Thirteenth International Con-
ference on Learning Representations, 2025b.
Devendra Singh Chaplot. Albert q. jiang, alexandre
sablayrolles, arthur mensch, chris bamford, devendra
singh chaplot, diego de las casas, florian bressand, gi-
anna lengyel, guillaume lample, lucile saulnier, lélio
renard lavaud, marie-anne lachaux, pierre stock, teven
le scao, thibaut lavril, thomas wang, timothée lacroix,
william el sayed.arXiv preprint arXiv:2310.06825, 3,
2023.
Chandler Davis and William Morton Kahan. The rotation
of eigenvectors by a perturbation. iii.SIAM Journal on
Numerical Analysis, 7(1):1–46, 1970.
Yao Fu. Challenges in deploying long-context transform-
ers: A theoretical peak performance analysis.arXiv
preprint arXiv:2405.08944, 2024.
Nathan Godey, Alessio Devoto, Yu Zhao, Simone Scarda-
pane, Pasquale Minervini, Éric Villemonte de la Clerg-
erie, and Benoît Sagot. Q-filters: Leveraging query-key
geometry for efficient key-value cache compression. In
Sparsity in LLMs (SLLM): Deep Dive into Mixture of
Experts, Quantization, Hardware, and Inference.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing
reasoning capability in llms via reinforcement learning.
arXiv preprint arXiv:2501.12948, 2025.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix, and
William El Sayed. Mistral 7b, 2023. URL https:
//arxiv.org/abs/2310.06825.
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas,
and François Fleuret. Transformers are rnns: Fast
autoregressive transformers with linear attention. In
International conference on machine learning, pages
5156–5165. PMLR, 2020.
Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu,
Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai
Dai, Daya Guo, et al. Deepseek-v2: A strong, econom-
ical, and efficient mixture-of-experts language model.
arXiv preprint arXiv:2405.04434, 2024.
Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga,
Alban Desmaison, Andreas Kopf, Edward Yang,
Zachary DeVito, Martin Raison, Alykhan Tejani,
Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie
Bai, and Soumith Chintala. Pytorch: An imperative
style, high-performance deep learning library. In
Advances in Neural Information Processing Systems
32, pages 8024–8035. Curran Associates, Inc., 2019.
URL http://papers.neurips.cc/paper/
9015-pytorch-an-imperative-style-high-performance-deep-learning-library.
pdf.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J. Liu. Exploring the limits of
transfer learning with a unified text-to-text transformer.
Journal of Machine Learning Research, 21(140):1–67,

<!-- page 10 -->

2020. URL http://jmlr.org/papers/v21/
20-074.html.
Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and
Kaushik Roy. Eigen attention: Attention in low-
rank space for kv cache compression.arXiv preprint
arXiv:2408.05646, 2024.
Noam Shazeer. Fast transformer decoding: One write-
head is all you need.arXiv preprint arXiv:1911.02150,
2019.
Luohe Shi, Hongyi Zhang, Yao Yao, Zuchao Li, and Hai
Zhao. Keep the cost down: A review on methods to
optimize llm’s kv-cache consumption.arXiv preprint
arXiv:2407.18003, 2024.
Prajwal Singhania, Siddharth Singh, Shwai He, Soheil
Feizi, and Abhinav Bhatele. Loki: Low-rank keys for
efficient sparse attention.Advances in Neural Informa-
tion Processing Systems, 37:16692–16723, 2024.
Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng,
Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, and
Beidi Chen. Shadowkv: Kv cache in shadows for high-
throughput long-context llm inference.arXiv preprint
arXiv:2410.21465, 2024.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,
Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288, 2023.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need.
Advances in neural information processing systems, 30,
2017.
Hao Wang, Ligong Han, Kai Xu, and Akash Srivastava.
Squat: Subspace-orthogonal kv cache quantization.
arXiv preprint arXiv:2503.24358, 2025.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chau-
mond, Clement Delangue, Anthony Moi, Pierric Cis-
tac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe
Davison, Sam Shleifer, Patrick von Platen, Clara Ma,
Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao,
Sylvain Gugger, Mariama Drame, Quentin Lhoest, and
Alexander M. Rush. Huggingface’s transformers: State-
of-the-art natural language processing, 2020. URL
https://arxiv.org/abs/1910.03771.
Hao Yu, Zelan Yang, Shen Li, Yong Li, and Jianxin Wu.
Effectively compress kv heads for llm.arXiv preprint
arXiv:2406.07056, 2024.
Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu,
Yan Yan, and Guangyu Sun. Asvd: Activation-aware
singular value decomposition for compressing large lan-
guage models, 2024. URL https://arxiv.org/
abs/2312.05821.
Biao Zhang, Ivan Titov, and Rico Sennrich. Sparse atten-
tion with linear units.arXiv preprint arXiv:2104.07012,
2021.
Rongzhi Zhang, Kuang Wang, Liyuan Liu, Shuohang
Wang, Hao Cheng, Chao Zhang, and Yelong Shen.
Lorc: Low-rank compression for llms kv cache with
a progressive compression strategy.arXiv preprint
arXiv:2410.03111, 2024.
Zeyu Zhang and Haiying Shen. Zack: Zero-overhead
llm inference acceleration via dimensionality com-
pression of the key-value cache.arXiv preprint
arXiv:2408.04107, 2024.

<!-- page 11 -->

KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity
(Supplementary Material)
A Proof of Theorem 1
Theorem.LetX∈RT×dbe a sequence of token embeddings,K,Q,V∈RT×dand
MHA(X) =
[
Softmax(QiK⊤
i /
√
d)Vi)
]
iWO,
^MHA(X) =
[
Softmax(Qi˜Ki
⊤
/
√
d)˜Vi)
]
iWO,
where ^MHA(X),˜Ki and ˜Vi represent the approximation of MHA(X),Ki and Vi, respectively. The difference
between the actual attention output and the one produced with approximate keys and values is upper bounded as
∥^MHA(X)−MHA(X)∥2
≤
h∑
i=1
∥ViWO
i∥2√
d
∥QiK⊤
i −Qi˜Ki
⊤
∥2
+∥ViWO
i −˜ViWO
i∥2.(3)
Proof. Let WO = [WO
1 ;WO
2 ;···;WO
h ]∈RD×D, where WO
i ∈Rd×Dare stacked vertically. By the definition of
multi-head attention (see Section 3.2), we can write
MHA(X) = [H 1,...,Hh]WO =
h∑
i=1
HiWO
i , ^MHA(X) = [˜H1,...,˜Hh]WO =
h∑
i=1
˜HiWO
i .
Therefore,
∥^MHA(X)−MHA(X)∥2 =

h∑
i=1
(˜Hi−Hi)WO
i

2
≤
h∑
i=1
∥(˜Hi−Hi)WO
i∥2.
For each headi∈{1,...,h},
∥(˜Hi−Hi)WO
i∥2 =
(
Softmax( QiK⊤
i√
d )Vi−Softmax(Qi˜K⊤
i√
d )˜Vi
)
WO
i

2
≤
(
Softmax( QiK⊤
i√
d )−Softmax(Qi˜K⊤
i√
d )
)
ViWO
i

2 +
Softmax( Qi˜K⊤
i√
d )(Vi−˜Vi)WO
i

2.
(4)
For the first term in (4), applying the submultiplicative property of the 2-norm gives
(
Softmax( QiK⊤
i√
d )−Softmax(Qi˜K⊤
i√
d )
)
ViWO
i

2≤∥Softmax(QiK⊤
i√
d )−Softmax(Qi˜K⊤
i√
d )∥2∥ViWO
i∥2.
The factor∥ViWO
i∥2 acts as an amplification term, capturing how sensitivity in the value projections may magnify
small perturbations in the attention weights—this term is typically not directly controllable in practice.
Since the Softmax function is 1
2-Lipschitz continuous (see Appendix A.4 [Alghamdi et al., 2022]), we have
∥Softmax(QiK⊤
i√
d )−Softmax(Qi˜K⊤
i√
d )∥2≤1√
d
∥QiK⊤
i −Qi˜K⊤
i∥2.
For the second term in (4), note that∥Softmax(·)∥1 = 1, implying∥Softmax(·)∥2≤1. Therefore,
Softmax( Qi˜K⊤
i√
d )(Vi−˜Vi)WO
i

2≤∥ViWO
i −˜ViWO
i∥2.

<!-- page 12 -->

Combining the two bounds yields
∥(˜Hi−Hi)WO
i∥2≤∥ViWO
i∥2√
d
∥QiK⊤
i −Qi˜K⊤
i∥2 +∥ViWO
i −˜ViWO
i∥2.
Summing over all headsi= 1,...,hgives the desired bound (3), completing the proof.
B Value-Output Projection
In this section, we examine the interaction between the value representations and the output projection. While the main
analysis of this paper focuses on the relationship between keys and queries, the same reasoning naturally extends to
values Vi∈RT×dand the output matrix WO
i ∈Rd×D. To further tighten the upper bound established in Theorem 1,
we aim to minimize the second term in the summation, which leads to the following optimization problem:
min
˜V
VWO−˜VWO2
F s.t.rank( ˜V)≤R,
where for simplicity we drop the subscript. Likewise for the keys and queries case, the optimization problem can be
restated using a projection matrixS∈Rd×dsuch that
min
S
∥VSWO−VWO∥2
F s.t.rank(S)≤R,
where S∈Rd×dwhere we write the projection matrix S as a product of two matrices, i.e., S=AB ⊤with
A,B∈Rd×R. The minimization problem tackled by KQ-SVD for the value and output projections is
min
A,B∈Rd×R
∥VAB⊤WO−VWO∥2
F.(5)
Theorem 2 applies, in a similar fashion to the case of values and outputs.
C Practical settings
Code:The code used for experiments presented in Section 6 is available at https://github.com/
DamienLesens/KQ-SVD. We used Pytorch [Paszke et al., 2019] and the Hugging Face transformers library
[Wolf et al., 2020]. For reproducibility, we fixed random seeds equals to 0.
Hardware:All experiments were conducted on NVIDIA Tesla V100-SXM2-32GB GPUs with CUDA acceleration.
The primary compute nodes were Intel Xeon E5-2698 v4 @ 2.20GHz (503GB RAM).
