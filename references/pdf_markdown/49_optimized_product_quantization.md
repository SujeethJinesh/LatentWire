# references/49_optimized_product_quantization.pdf

<!-- page 1 -->

Optimized Product Quantization for Approximate Nearest Neighbor Search
Tiezheng Ge1∗ Kaiming He2 Qifa Ke3 Jian Sun2
1University of Science and Technology of China 2Microsoft Research Asia 3Microsoft Research Silicon V alley
Abstract
Product quantization is an effective vector quantization
approach to compactly encode high-dimensional vectors
for fast approximate nearest neighbor (ANN) search. The
essence of product quantization is to decompose the orig-
inal high-dimensional space into the Cartesian product of
a ﬁnite number of low-dimensional subspaces that are then
quantized separately. Optimal space decomposition is im-
portant for the performance of ANN search, but still re-
mains unaddressed. In this paper , we optimize produc-
t quantization by minimizing quantization distortions w.r .t.
the space decomposition and the quantization codebooks.
W e present two novel methods for optimization: a non-
parametric method that alternatively solves two smaller
sub-problems, and a parametric method that is guaranteed
to achieve the optimal solution if the input data follows
some Gaussian distribution. W e show by experiments that
our optimized approach substantially improves the accura-
cy of product quantization for ANN search.
1. Introduction
Approximate nearest neighbor (ANN) search is of great
importance for many computer vision problems, such as re-
trieval [17], classiﬁcation [2], and recognition [18]. Re-
cent years have witnessed the increasing interest ( e.g.,
[18, 20, 3, 10, 6]) in encoding high dimensional data in-
to distance-preserving compact codes. With merely tens
of bits per data item, compact encoding not only saves the
cost of data storage and transmission, but more important-
ly, it enables efﬁcient nearest neighbor search on large-scale
datasets, taking only a fraction of a second for each nearest
neighbor query [18, 10]
Hashing [1, 18, 20, 19, 6, 8] has been a popular approach
to compact encoding, where the similarity between two da-
ta points is approximated by the Hamming distance of their
hashed codes. Recently, product quantization (PQ) [10] was
applied to compact encoding, where a data point is vector-
quantized to its nearest codeword in a predeﬁned codebook,
∗This work is done when Tiezheng Ge is an intern at Microsoft Re-
search Asia.
and the distance between two data points is approximated
by the distance between their codewords. PQ achieves a
large effective codebook size with the Cartesian product of
a set of small sub-codebooks. It has been shown to be more
accurate than various hashing-based methods ( c.f . [10, 3]),
largely due to its lower quantization distortions and more
precise distance computation using a set of small lookup ta-
bles. Moreover, PQ is computationally efﬁcient and thus at-
tractive for large-scale applications—the Cartesian product
enables pre-computed distances between codewords to be s-
tored in tables with feasible sizes, and query is merely done
by table lookups using codeword indices. It takes about 20
milliseconds to query against one million data points for the
nearest neighbor by exhaustive search.
To keep the size of the distance lookup table feasible,
PQ decomposes the original v ector space into the Cartesian
product of a ﬁnite number of low-dimensional subspaces.
It has been noticed [10] that the prior knowledge about the
structures of the input data is of particular importance, and
the accuracy of ANN search would become substantially
worse if ignoring such knowledge. The method in [11] op-
timizes a Householder transfo rm under an intuition that the
data components should have balanced variances. It is al-
so observed that a random rotation achieves similar perfor-
mance [11]. But the optimality in terms of quantization er-
ror is unclear. Thus, optimal space decomposition for PQ
remains largely an unaddressed problem.
In this paper, we formulate product quantization as an
optimization problem that minimizes the quantization dis-
tortions by searching for optimal codebooks and space de-
composition. Such an optimization problem is challenging
due to large number of free parameters. We proposed t-
wo solutions. In the ﬁrst solution, we split the problem
into two sub-problems, each having a simple solver. The
space decomposition and the codebooks are then alterna-
tively optimized, by solving for the space decomposition
while ﬁxing the codewords, and vice versa. Such a solution
is non-parametric in that it does not assume any priori in-
formation about the data distribution. Our second solution
is a parametric one in that it assumes the data follows Gaus-
sian distribution. Under such assumption, we show that the
lower bound of the quantization distortion has an analytical
1

<!-- page 2 -->

formulation, which can be effectively optimized by a sim-
ple Eigenvalue Allocation method. Experiments show that
our two solutions outperform the original PQ [10] and other
alternatives like transform coding [3] and iterative quantiza-
tion [6], even when the prior knowledge about the structure
of the input data is used by PQ [10].
Concurrent with our work, a very similar idea is inde-
pendently developed by Norouzi and Fleet [14].
2. Quantization Distortion
In this section, we show that a variety of distance approx-
imation methods, including k-means [13], product quan-
tization [10], and orthogonal hashing [19, 6], can be for-
mulated within the framework of vector quantization [7]
where quantization distortion is used as the objective func-
tion. Quantization distortion is tightly related to the empiri-
cal ANN performance, and thus can be used to measure the
“optimality” of a quantization algorithm for ANN search.
2.1. V ector Quantization
V ector quantization (VQ) [7] maps a vector x ∈RD to
a codeword c in a codebook C = {c(i)} with i in a ﬁnite
index set. The mapping, termed as a quantizer, is denoted
by: x → c(i(x)). In information theory, the function i(·) is
called an encoder, and function c(·) is called a decoder [7].
The quantization distortion E is deﬁned as:
E = 1
n
∑
x
∥x−c(i(x))∥2, (1)
where∥·∥ denotes the l2-norm,n is the total number of da-
ta samples, and the summation is over all the points in the
given sample set. Given a codebook C, a quantizer that min-
imizes the distortion E must satisfy the ﬁrst Lloyd’s condi-
tion [7]: the encoder i(x) should map any x to its nearest
codeword in the codebook C. The distance between two
vectors can be approximated b y the distances between their
codewords, which can be precomputed ofﬂine.
2.2. Codebook Generation
We show that a variety of methods minimize the distor-
tion w.r.t. to the codebook using different constraints.
K-means
If there is no constraint on the codebook, minimizing the
distortion in Eqn.(1) leads to the classical k-means cluster-
ing algorithm [13]. With the encoder i(·) ﬁxed, the code-
word c of a given x is the center of the cluster that x belongs
to—this is the second Lloyd’s condition [7].
Product Quantization [10]
If any codeword c must be taken from the Cartesian
product of a ﬁnite number of sub-codebooks, minimizing
the distortion in Eqn.(1) leads to the product quantization
method [10].
Formally, denote any x ∈ RD as the concatenation of
M subvectors: x =[ x1,... xm,... xM]. For simplicity it is
assumed [10] that the subvectors have common number of
dimensionsD/M. The Cartesian product C = C1×...×C M
is the set in which a codeword c ∈C is formed by concate-
nating the M sub-codewords: c =[ c1,... cm,... cM], with
each cm ∈Cm. We point out that the objective function for
PQ, though not explicitly deﬁned in [10], is essentially:
min
C1,...,CM
∑
x
∥x−c(i(x))∥2, (2)
s.t. c ∈C = C1 ×...×C M.
It is easy to show that x’s nearest codeword c in C is
the concatenation of the M nearest sub-codewords c =
[c1,... cm,... cM] where cm is the nearest sub-codeword of
the subvector xm. So Eqn. (2) can be split into M separate
subproblems, each of which can be solved by k-means in its
corresponding subspace. This is the PQ algorithm.
The beneﬁt of PQ is that it can easily generate a code-
book C with a large number of codewords. If each sub-
codebook has k sub-codewords, then their Cartesian prod-
uctC haskM codewords. This is not possible for classical k-
means whenkM is large. PQ also enables fast distance com-
putation: the distances between any two sub-codewords in a
subspace are precomputed and stored in a k-by-k lookup ta-
ble, and the distance between two codewords in C is simply
the sum of the distances compute from the M subspaces.
Iterative Quantization [6]
If any codeword c must be taken from “the vertexes of
a rotating hyper-cube,” minimizing the distortion leads to a
hashing method called Iterative Quantization (ITQ) [6].
The D-dimensional vectors in {−a,a}
D are the ver-
tices of an axis-aligned D-dimensional hyper-cube. Sup-
pose the data has been zero-centered. The objective func-
tion in ITQ [6] is essentially:
min
R,a
∑
x
∥x−c(i(x))∥2, (3)
s.t. c ∈C = {c|Rc ∈{ −a,a}D},R TR = I,
whereR is an orthogonal matrix and I is an identity matrix.
The beneﬁt of using a rotating hyper-cube as the code-
book is that the squared Euclidean distance between any
two codewords is equivalent to the Hamming distance be-
tween their indices. So ITQ is in the category of binary
hashing methods [1, 20, 19]. Eqn.(3) also indicates that any
orthogonal hashing method is equivalent to a vector quan-
tizer. The length a in (3) does not impact the resulting hash-
ing functions as noticed in [6], but it matters when we com-
pare the distortion with other quantization methods.
2.3. Distortion as the Objective Function
The above methods all optimize the same form of quan-
tization distortion, but subject to different constraints. This

<!-- page 3 -->

0.1 0.3 0.50
0.1
0 15000 300000
0.2
*OYZUXZOUT *OYZUXZOUT
S'6
S'6
1SKGTY
67
67
67
/:7
1SKGTY
67
67
67 /:7
9/,: -/9:
Figure 1: mAP vs. quantization distortion. We show result-
s from ﬁve methods: k-means, ITQ, and three variants of
PQ. The datasets are SIFT1M and GIST1M from [10]. All
methods are given 16 bits for codeword length. The data
consist of the largest 16 principal components (this is to en-
able measuring the ITQ distortion).
implies that distortion is an objective function that can be
evaluated across different quantization methods. We empir-
ically observe that the distortion is tightly correlated to the
ANN search accuracy of different methods.
To show this, we investigate nearest neighbor search for
100 nearest neighbors on two large datasets. We adop-
t the common strategy of linear search using compact
codes [10, 6]: the return results are ordered by their approxi-
mate distances to the query. Here both the query and the da-
ta are quantized, and their distance is approximated by their
codeword distances (for ITQ this is equivalent to ranking by
Hamming distance). We compute the mean Average Preci-
sion (mAP) over ten thousand and one thousand queries on
the ﬁrst and second data set, respectively. The mAP using
different methods and their distortions are shown in Fig. 1.
We compare ﬁve methods: k-means, ITQ, and three variants
of PQ (decomposed into M =2 , 4, or 8 subspaces, denoted
as PQ
M ). For all methods the codeword length is ﬁxed to be
B =1 6 bits, which essentially gives K =2 16 codewords
(though PQ and ITQ need not explicitly store them). We can
see that mAP (from different methods) has a strong corre-
lation with the quantization distortion. We observe similar
behaviors under various ANN me trics, like precision/recall
at the ﬁrst N samples, with various number of ground-truth
nearest neighbors.
Based on the above observations, we use distortion as an
objective function to evaluate the optimality of a quantiza-
tion method.
3. Optimized Product Quantization
Product quantization involves decomposing the D-
dimensional vector space into M subspaces, and comput-
ing a sub-codebook for each subspace. M is determined by
the budget constraint of memory space (to ensure a feasi-
ble lookup table size) and computational costs, and is pre-
determined in practice. We use an orthonormal matrix R
to represent the space decomposition. The D-dimensional
vector space is ﬁrst transformed by R. The dimension-
s of the transformed space are then divided into D/M
chunks. The i-th chunk, consisting of the dimensions of
(i −1) ∗D/M + {1,2,...,D/M }, is then assigned to the
i-th subspace. Note that any reordering of the dimensions
can be represented by an orthonormal matrix. Thus R de-
cides the dimensions of the transformed space assigned to
each subspace. The free parameters of product quantization
thus consist of the sub-codebooks for the subspaces, and
the matrix R. The additional free parameters of R allows
the vector space to rotate, thus relax the constraints on the
codewords. From Fig. 1 we see that relaxing constraints
leads to lower distortions. We optimize product quantiza-
tion over these free parameters:
min
R,C1,...,CM
∑
x
∥x−c(i(x))∥2, (4)
s.t. c ∈C = {c| Rc ∈C1 ×...×C M,R TR = I}
We call the above problem Optimized Product Quantization
(OPQ). The effective codebook C is jointly determined by
R and sub-codebooks {Cm}M
m=1.
Notice that assigning x to its nearest codeword c is e-
quivalent to assigning Rx to the nearest Rc. To apply the
codebook in Eqn.(4) for encoding, we only need to pre-
process the data by Rx, and the remaining step is the same
as that in PQ.
Optimizing the problem in Eqn.(4) was considered “ not
tractable” [11], because of the large number of degrees
of freedom. Previous methods pre-processed the data us-
ing simple heuristics like randomly ordering the dimensions
[10] or randomly rotating the space [11]. The matrix R has
not been considered in any optimization. In the following
we propose a non-parametric iterative solution and a simple
parametric solution to the problem of Eqn.(4).
3.1. A Non-Parametric Solution
Our non-parametric solution does not assume any data
distribution 1. We split the problem in Eqn.(4) into two sim-
pler sub-problems that are optimized in an alternative way.
Step (i): Fix R, optimize the sub-codebooks {Cm}M
m=1.
Recall that assigning x to the nearest c is equivalent to
assigning Rx to the nearest Rc. Denote ˆx = Rx and ˆc =
Rc.S i n c eR is orthonormal, we have∥x−c∥2 = ∥ˆx−ˆc∥2.
WithR ﬁxed, Eqn.(4) then becomes:
min
C1,...,Cm
∑
ˆx
∥ˆx−ˆc(i(ˆx))∥2, (5)
s.t. ˆc ∈C1 ×...×C M.
1We follow the terminology in statistics that a “non-parametric” model
is the one that does not rely on any assumption about the data distribu-
tion, while “parametric” model explicitly assumes certain parameterized
distribution such as Gaussian distribution.

<!-- page 4 -->

Eqn.(5) is the same problem as PQ in Eqn.(2). We can sep-
arately run k-means in each subspace to compute the sub-
codebooks.
Step (ii): Fix the sub-codebooks {Cm}M
m=1, optimize R.
Since∥x−c∥2 = ∥Rx−ˆc∥2, the sub-problem becomes:
min
R
∑
x
∥Rx−ˆc(i(ˆx))∥2, (6)
s.t. R TR = I.
For each ˆx, the codeword ˆc(i(ˆx)) is ﬁxed in the subproblem
and can be derived from the sub-codebooks computed in
Step (i). To ﬁnd ˆc(i(ˆx)), we simply concatenate the M
sub-codewords of the sub-vectors in ˆx. We denote ˆc(i(ˆx))
as y.G i v e nn training samples, we denote X and Y as two
D-by-n matrices whose columns are the samples x and y
respectively. Note Y is ﬁxed in this subproblem. Then we
can rewrite (6) as:
min
R
∥RX −Y∥2
F, (7)
s.t. R TR = I,
where ∥·∥ F is the Frobenius norm. This is the Orthogonal
Procrustes problem [16, 6] and there is a closed-form solu-
tion: ﬁrst apply Singular V alue Decomposition to XY T =
USV T, and then let R = VU T. In [6] this solution was
used to optimized the ITQ problem in Eqn.(3).
Our algorithm alteratively optimizes Step (i) and (ii).
Note that in Step (i) we need to run k-means, which by itself
is an iterative algorithm. However, we notice that after Step
(ii) the updated matrixR would not change the cluster mem-
bership of the previous k-means clustering results, thus we
only need to reﬁne the previous k-means results instead of
restarting k-means. With this strategy, we empirically ﬁnd
that even if we only run one k-means iteration in each Step
(i), our entire algorithm still converges to a good solution.
A pseudo-code is in Algorithm 1.
Note that if we ignore line 3 and line 8 in Algorithm 1, it
is equivalent to PQ (for PQ one might usually put line 2 in
the inner loop). Thus its complexity is comparable to PQ,
except that in each iteration our algorithm updates R and
transforms the data by R. Fig. 2 shows the convergence of
our algorithm. In practice we ﬁnd 100 iterations are good
enough for the purpose of ANN search.
Like many other alternative-optimization algorithms, our
algorithm is locally optimal and the ﬁnal solution depends
on the initialization. In the next subsection we propose a
parametric solution that can be used to initialize our alter-
native optimization procedure.
3.2. A Parametric Solution
We further propose another solution assuming the data
follows a parametric Gaussian distribution. This paramet-
Algorithm 1 Non-Parametric OPQ
Input: training samples {x}, number of subspaces M ,
number of sub-codewords k in each sub-codebook.
Output: the matrix R, sub-codebooks {Cm}M
m=1, M sub-
indices {im}M
m=1 for each x.
1: Initialize R,{Cm}M
m=1,a n d{im}M
m=1.
2: repeat
3: Step(i): project the data: ˆx = Rx.
4: form =1 toM do
5: for j =1 tok: update ˆcm(j) by the sample mean
of {ˆxm |im(ˆxm)= j}.
6: for ∀ˆxm: update im(ˆxm) by the sub-index of the
sub-codeword ˆcm that is nearest to ˆxm.
7: end for
8: Step(ii): solve R by Eqn.(7).
9: until max iteration number reached
/ZKXGZO\KT[SHKX
*OYZUXZOUT
0 100 200 300 400 5003.5
3.7
3.9
4.1
4.3 x 10
4
Figure 2: Convergence of Algorithm 1 in the SIFT1M
dataset[10]. Here we use M =4 andk = 256(32 bits).
ric solution has both practical an d theoretical merits. First,
it is a simpler method and is globally optimal if the data
follows Gaussian distribution. Second, it provides a way to
initialize the non-parametric method. Third, it provides new
theoretical explanations for t wo commonly used criteria in
some other quantization methods.
3.2.1 Distortion Bound of Quantization
We assume each dimension of x ∈R
D is subject to an in-
dependent Gaussian distribution with zero mean. From rate
distortion theory [5] we know that the codeword length b
from any quantizer achieving a distortion of E must satisfy:
b(E) ≥
D∑
d=1
1
2 log2
Dσ2
d
E , (8)
whereσ2
d is the variance at each dimension. In this equation
we have assumed σ2
d is sufﬁciently large (a more general
form is in [5]). Equivalently, the distortion E satisﬁes:
E ≥k− 2
D D|Σ|
1
D , (9)
where k =2 b and we assume all codewords have identical
code length. The matrix Σis the covariance of x,a n d|Σ|

<!-- page 5 -->

is the determinant. Here we have relaxed the independence
assumption and allowed x to follow a Gaussian distribu-
tion N(0,Σ). Eqn.(9) is the distortion lower bound for any
quantizer with k codewords. The following table shows the
values of this bound and the empirical distortion of k-means
(105 samples,k = 256,σ2
d randomly generated in [0.5,1]):
D
 32 64 128
distortion bound
 16.2 38.8 86.7
empirical distortion
 17.1 39.9 88.5
It is reasonable to consider this bound as an approxima-
tion to the k-means distortion. The small gap ( ∼5%)m a y
be due to two reasons: k-means can only achieve a local-
ly optimal solution, and the ﬁxed code-length for all code-
words may not achieve optimal bit rate 2.
3.2.2 Distortion Bound of Product Quantization
Now we study the distortion bound of product quantization
when x ∼N (0,Σ). Suppose x has been decomposed into
a concatenation of M equal-dimensional sub-vectors. Ac-
cordingly, we can decompose ΣintoM ×M sub-matrices:
Σ=
⎛
⎜⎝
Σ
11 ··· Σ1M
..
. . . . ..
.
Σ
M1 ··· ΣMM
⎞
⎟⎠. (10)
Here the diagonal sub-matrices Σmm are the covariance of
the m-th subspace. Notice that the marginal distribution
of xm subjects to D
M -dimensional Gaussian N(0,Σmm).
From (9), the distortion bound of PQ is:
EPQ = k− 2M
D
D
M
M∑
m=1
|Σmm|
M
D , (11)
3.2.3 Minimizing Distortion Bound of PQ
Remember that space decomposition can be parameterized
by an orthonormal matrix R. When applying R to data, the
variable ˆx = Rx is subject to N(0, ˆΣ)with ˆΣ= RΣRT.
We propose to minimize the distortion bound w.r.t.R to op-
timize the space decomposition in product quantization:
min
R
M∑
m=1
|ˆΣmm|
M
D , (12)
s.t. R TR = I,
where ˆΣmm is the diagonal sub-matrix of ˆΣ. The constant
scale in Eqn.(11) has been ignored in this objective func-
tion. Due to the orthonormal constraint, this problem is in-
herently non-convex. Fortunately, the special form of our
objective function can be minimized by a simple algorithm,
as we show next.
2In information theory, it is possible to reduce the average bit rate by
varying the bit-length of codewords, known as entropy encoding [5].
3.2.4 Eigenvalue Allocation
We ﬁrst show that the objective function in Eqn.(12) has a
constant lower bound. Using the inequality of arithmetic
and geometric means (AM-GM inequality) [4], the objec-
tive in Eqn.(12) satisﬁes:
M∑
m=1
|ˆΣmm|
M
D ≥M
M∏
m=1
|ˆΣmm|
1
D . (13)
The equality holds if and only if the term |ˆΣmm| has the
same value for allm.F u r t h e r ,Fischer’s inequality[9] gives:
M∏
m=1
|ˆΣmm|≥| ˆΣ|. (14)
The equality holds if the off-diagonal sub-matrices in ˆΣe-
qual to zero. Note that |ˆΣ|≡| Σ| is a constant given the
data distribution. Combining (13) and (14), we obtain the
constant lower bound for the objective in (12):
M∑
m=1
|ˆΣmm|
M
D ≥M|Σ|
1
D . (15)
The minimum is achieved if the following two criteria are
both satisﬁed:
(i) Independence . If we align the data by PCA, the e-
quality in Fischer’s inequality (14) is achieved. This implies
we make the dimensions independent to each other.
(ii) Balanced Subspaces’ Variance . The equality in
AM-GM (13) is achieved if |ˆΣmm| has the same value for
all subspaces. Suppose the data has been aligned by PCA.
Then |ˆΣ
mm| equals to the product of the eigenvalues of
Σmm. By re-ordering the principal components, we can
balance the product of eigenvalues for each subspace, thus
the values |ˆΣmm| for all subspaces. As a result, both equal-
ities in AM-GM (13) and Fischer’s (14) are satisﬁed, so the
objective function is minimized.
Based on the above analysis, we propose a simple
Eigenvalue Allocation method to optimize Eqn.(12). This
method is a greedy solution for the balanced partition prob-
lem. We ﬁrst align the data using PCA and sort the eigen-
values σ
2 in the descending order σ2
1 ≥... ≥σ2
D.N o t e
that we do not reduce dimensions. We prepare M empty
buckets, one for each of the M subspaces. We sequentially
pick out the largest eigenvalue and allocate it to the bucket
having the minimum product of the eigenvalues in it (un-
less the bucket is full, i.e., with D/M eigenvalues in it).
The principal directions corresponding to the eigenvalues
in each bucket form the subspace. In fact, the principal di-
rections are re-ordered to form the columns of R.
In real data sets, we ﬁnd this greedy algorithm is suf-
ﬁciently good for minimizing the objective function. To
show this fact, we compute the covariance matrix Σfrom
the SIFT/GIST datasets [10]. The following table shows the
theoretical minimum of the objective function (right hand

<!-- page 6 -->

side in (15)) and the objective function value (left hand side
in (15)) obtained by our Eigenvalue Allocation algorithm.
Here we use M =8 and k = 256. We can see the above
greedy algorithm achieves t he theoretical minimum:
theoretical minimum
 Eigenvalue Allocation
SIFT
 2.9286 × 103
 2.9287 × 103
GIST
 1.9870 × 10−3
 1.9870 × 10−3
Interestingly, existing encoding methods have adopted,
either heuristically or in objective functions different from
ours, the criteria of “independence” or “balance” mentioned
above. “Independence” was used in [20, 19, 3] in the for-
m of PCA projection. “Balance” was used in [11, 20, 3]:
the method in [11] rotates the data to “balance” variance
for each component but lost “independence”; the methods
in [20, 3] adaptively allocate the codeword bits to the princi-
pal components. Our derivation provides theoretical expla-
nations for the two criteria: they can be considered as a way
of minimizing the quantization distortion under a Gaussian
distribution assumption.
Summary of the parametric solution. Our parametric
solution ﬁrst computes the covariance matrix Σof the data
and uses Eigenvalue Allocation to generate the orthonormal
matrix R, which determines the space decomposition. The
data are then transformed by R. The original PQ algorithm
is then performed on these transformed data.
4. Experiments
We evaluate our method for ANN search using three
public datasets. The ﬁrst two datasets are from SIFT1M
and GIST1M [10]. The SIFT1 M consists of 1 million 128-d
SIFT features [12] and 10k queries. The GIST1M set con-
sists of 1 million 960-d GIST features [15] and 1k queries.
The third dataset MNIST 3 consists of 70k images of hand-
written digits, each as a 784-d vector concatenating all pix-
els. We randomly sample 1k as the queries and use the re-
maining as the data base. We further generate a synthet-
ic dataset subject to a 128-d independent Gaussian distri-
bution, where the eigenvalues of the covariance matrix are
given by σ
2
d = e−0.1d (d=1,...,128), a long-tail curve ﬁt to
the eigenvalues of the above real datasets. This synthetic set
has 1 million data points and 10k queries.
We consider K=100 Euclidean nearest neighbors as the
true neighbors. We ﬁnd that for lookup-based methods, K
is not inﬂuential for the comparisons among the methods.
We follow a common exhaustive search strategy ( e.g.,
[10, 6]). The data are ranked in the order of their approxi-
mate distances to the query. If both the query and data are to
be quantized, the method is termed as Symmetric Distance
Computation (SDC) [10]. SDC is equivalent to Hamming
3http://yann.lecun.com/exdb/mnist/
9_TZNKZOIHOZY9*)
4
8KIGRR
0 2500 50000
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
Figure 3: Comparison on Gaussian synthetic data using
Symmetric Distance Computation and 32-bit codes.
ranking for orthogonal hashing methods like ITQ [6]. If
only the data are quantized, the method is termed as Asym-
metric Distance Computation (ADC) [10]. ADC is more
accurate than SDC but has the same complexity. We have
tested both cases. The exhaustive search is fast using lookup
tables: e.g., for 64 bits indices it takes 20 ms per 1 million
distance computation. Experiments are on a PC with an In-
tel Core2 2.13GHz CPU and 8G RAM. We do not combine
with any non-exhaustive method like inverted ﬁles [17] as
this is not the focus of our paper. We study the following
methods:
•OPQ
P: our parametric solution.
•OPQNP: our non-parametric solution initialized by para-
metric solution.
•PQRO: randomly order dimensions as suggested in [10].
•PQRR: data is aligned using PCA and then randomly ro-
tated, as suggested in [11].
•TC (Transform Coding [3]): a scalar quantization
method that quantizes each principal component with an
adaptive number of bits.
•ITQ [6]: one of the state-of-the-art hashing methods, a
special vector quantization method.
Notice that in these settings we have assumed there is no
prior knowledge available. Later we will study the case with
prior knowledge.
Given the code-length B, all the PQ-based methods
(OPQNP,O P QP,P Q RO,P Q RR) assign 8 bits to each sub-
space (k = 256). The subspace number M isB/8.
We have published the Matlab code of our solutions 4.
Performance on the synthetic dataset
Fig. 3 shows the performance on the synthetic data sub-
ject to a 128-d independent Gaussian distribution, evaluated
through recall vs. N , i.e., the proportion of the true nearest
neighbors ranked in the ﬁrst N positions. We can see that
OPQNP and OPQP perform almost the same. OPQP achieves
4research.microsoft.com/en-us/um/people/kahe/

<!-- page 7 -->

G9/,:HOZY9*)
4
8KIGRR
0 1000 20000
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
16 32 64 1280
0.2
0.4
0.6
0.8


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
H9/,:9*)
HOZY
S'6
16 32 64 1280
0.2
0.4
0.6
0.8


OPQNP
OPQP
PQRO
PQRR
TC
I9/,:'*)
HOZY
S'6
Figure 4: Comparisons on SIFT1M. (a): recall at the N top ranked samples, using SDC and 64-bit codes. (b): mean Average
Precision vs. code-length, using SDC. (c): mean Average Precision vs. code-length, using ADC.
0 1000 20000
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
G-/,:HOZY9*)
4
8KIGRR
16 32 64 1280
0.1
0.2
0.3
0.4


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
H-/9:9*)
HOZY
S'6
I-/9:'*)
HOZY
S'6
16 32 64 1280
0.1
0.2
0.3
0.4


OPQNP
OPQP
PQRO
PQRR
TC
Figure 5: Comparisons on GIST1M.
theoretic minimum of 6.314× 10−3 in Eqn.(15). This im-
plies that, under a Gaussian distribution, our parametric so-
lution is optimal. On the contrary, PQ
RO and PQRR perform
substantially worse. This means that the subspace decom-
position is important for PQ even under a simple Gaussian
distribution. Besides, we ﬁnd PQ RO performs better than
PQRR. This is because in this distribution PQ RO automati-
cally satisﬁes the independent criterion (see 3.2.4).
Performance without prior knowledge
Next we evaluate the performance if the prior knowledge
is not available. In Fig. 4, 5, and 6 we compare the results of
SIFT1M, GIST1M, and MNIST. We show the recall in the
ﬁrst N positions with 64 bits using SDC (Fig. 4, 5, 6 (a)),
and the mAP (mean Average Pr ecision) vs. code-length us-
ing SDC and ADC, respectively (Fig. 4, 5, 6 (b)(c)).
We ﬁnd that both of our solutions substantially outper-
form the existing methods. The superiority of our methods
does not depends on the choice of SDC or ADC. Typical-
ly, in all cases even our simple parametric method OPQ
P
has shown prominent improvement over PQ RO and PQ RR.
This indicates that PQ-based methods strongly depend on
the space decomposition. We also notice the performance
of PQ
RR is disappointing. Although this method tries to
balance the variance using a random rotation, the indepen-
dence between subspaces is lost by such a random rotation.
Our non-parametric solution further improves the result-
s from the parametric one in the SIFT1M and MNIST sets.
This is because these two sets exhibit more than one cluster:
the SIFT1M set has two distinct clusters (this can be visual-
ized by the ﬁrst two principal components), and MNIST can
be expected to have 10 clusters due to the 10 digits. In these
cases the non-parametric solution is able to further reduce
the distortion. Such an improvement on MNIST is signiﬁ-
cant. In GIST1M our two methods are comparable, possibly
as this set is mostly subject to a Gaussian distribution.
We notice that TC performs clearly better than PQ
RO and
PQRR in the GIST1M set. This s calar quantization method
attempts to balance the bits assigned to the eigenvalues. So
it better satisﬁes the two criteria in Sec. 3.2.4. But TC is
inferior to our methods in all datasets. This is because our
method quantizes multi-dimensional subspaces instead of
each scalar dimension. And unlike TC that assigns a vari-
able number of bits to each eigenvalue, our method assigns
the eigenvalues to each subspace. Since bit numbers are
discrete but eigenvalues are continuous, it is easier for our
method to achieve balance.
Performance with prior knowledge
It has been noticed [10] that PQ works much better if uti-
lizing the prior knowledge that SIFT and GIST are concate-
nated histograms. Typically, the so-called “natural” order is
that each subspace consists of neighboring histograms. The
“structural” order (when M =8 ) is that each subspace con-
sists of the same bin of all histograms (each histogram has
8 bins). It is observed [10] that the natural order perform-

<!-- page 8 -->

0 100 200 300 400 5000
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
G34/9:HOZY9*)
4
8KIGRR
16 32 64 1280
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
ITQ
H34/9:9*)
HOZY
S'6
I34/9:'*)
HOZY
S'6
16 32 64 1280
0.2
0.4
0.6
0.8
1


OPQNP
OPQP
PQRO
PQRR
TC
Figure 6: Comparisons on MNIST.
0 1000 20000
0.2
0.4
0.6
0.8
1


OPQNP
OPQNP+pri
PQpri
G9/,:HOZY9*)
4
8KIGRR
H-/9:HOZY9*)
4
8KIGRR
0 1000 20000
0.2
0.4
0.6
0.8
1


OPQNP
OPQNP+pri
PQpri
Figure 7: Comparisons with prior knowledge using 64 bits
and SDC. (a): SIFT1M. (b): GIST1M.
s better for SIFT features, and the structural order is better
for GIST features. We denote PQ with priori knowledge as
PQpri (use the recommended orders). Note such priors may
limit the choices of M and are not always applicable for all
possible code-length.
In Fig. 7 we compare PQ pri with our prior-free non-
parametric method OPQ NP. We also evaluate our non-
parametric method using the prior orders as initialization,
denoted as OPQNP+pri. Our prior-free method outperforms
the prior-based PQ. In SIFT1M our prior-dependent method
improves further due to a better initialization. In GIST1M
our prior-dependent method is slightly inferior than our
prior-free method: this also implies that GIST1M largely
follows a Gaussian distribution.
5. Conclusion
We have proposed to optimize space decomposition in
product quantization. Since s pace decomposition has been
shown to have great impact on the search accuracy in our
experiments and in existing study [10], we believe our re-
sult has made product quantization a more practical and ef-
fective solution to general ANN problems.
References
[1] A. Andoni and P . Indyk. Near-optimal hashing algorithms for
approximate nearest neighbor in high dimensions. In FOCS,
pages 459–468. IEEE, 2006.
[2] O. Boiman, E. Shechtman, and M. Irani. In defense of
nearest-neighbor based image classiﬁcation. In CVPR, 2008.
[3] J. Brandt. Transform coding for fast approximate nearest
neighbor search in high dimensions. In CVPR, 2010.
[4] A. Cauchy. Cours d’analyse de l’ ´Ecole Royale Polytech-
nique. Imprimerie royale, 1821.
[5] T. Cover and J. Thomas. Elements of information theory ,
chapter 13, page 348. John Wiley & Sons, Inc., 1991.
[6] Y . Gong and S. Lazebnik. Iterative quantization: A pro-
crustean approach to learning binary codes. In CVPR, 2011.
[7] R. Gray. V ector quantization. ASSP Magazine, IEEE, 1984.
[8] K. He, F. Wen, and J. Sun. K-means Hashing: an Afﬁnity-
Preserving Quantization Method for Learning Binary Com-
pact Codes. In CVPR, 2013.
[9] R. A. Horn and C. R. Johnson. Matrix Analysis , chapter 7,
page 478. Cambridge University Press, 1990.
[10] H. Jegou, M. Douze, and C. Schmid. Product quantization
for nearest neighbor search. TPAMI, 33, 2011.
[11] H. Jegou, M. Douze, C. Schmid, and P . Perez. Aggregating
local descriptors into a compact image representation. In
CVPR, pages 3304–3311, 2010.
[12] D. G. Lowe. Distinctive image features from scale-invariant
keypoints. IJCV, 60:91–110, 2004.
[13] J. B. MacQueen. Some methods for classiﬁcation and analy-
sis of multivariate observations. In Proceedings of 5th Berke-
ley Symposium on Mathematic al Statistic s and Pr obability,
pages 281–297. University of California Press, 1967.
[14] M. Norouzi and D. Fleet. Cartesian k-means. In CVPR,
2013.
[15] A. Oliva and A. Torralba. Modeling the shape of the scene:
a holistic representation of the spatial envelope. IJCV, 2001.
[16] P . Sch¨onemann. A generalized solution of the orthogonal
procrustes problem. Psychometrika, 31(1):1–10, 1966.
[17] J. Sivic and A. Zisserman. Video google: a text retrieval
approach to object matching in videos. In ICCV, 2003.
[18] A. B. Torralba, R. Fergus, and Y . Weiss. Small codes and
large image databases for recognition. In CVPR, 2008.
[19] J. Wang, S. Kumar, and S.-F. Chang. Semi-supervised hash-
ing for scalable image retrieval. In CVPR, 2010.
[20] Y . Weiss, A. Torralba, and R. Fergus. Spectral hashing. In
NIPS, pages 1753–1760, 2008.
