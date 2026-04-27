# references/52_finding_structure_with_randomness.pdf

<!-- page 1 -->

arXiv:0909.4061v2  [math.NA]  14 Dec 2010
FINDING STRUCTURE WITH RANDOMNESS:
PROBABILISTIC ALGORITHMS FOR CONSTRUCTING
APPROXIMA TE MA TRIX DECOMPOSITIONS
N. HALKO †, P. G. MARTINSSON † , AND J. A. TROPP ‡
Abstract. Low-rank matrix approximations, such as the truncated sing ular value decompo-
sition and the rank-revealing QR decomposition, play a cent ral role in data analysis and scientiﬁc
computing. This work surveys and extends recent research wh ich demonstrates that randomization
oﬀers a powerful tool for performing low-rank matrix approx imation. These techniques exploit mod-
ern computational architectures more fully than classical methods and open the possibility of dealing
with truly massive data sets.
This paper presents a modular framework for constructing ra ndomized algorithms that compute
partial matrix decompositions. These methods use random sa mpling to identify a subspace that
captures most of the action of a matrix. The input matrix is th en compressed—either explicitly or
implicitly—to this subspace, and the reduced matrix is mani pulated deterministically to obtain the
desired low-rank factorization. In many cases, this approa ch beats its classical competitors in terms
of accuracy, speed, and robustness. These claims are suppor ted by extensive numerical experiments
and a detailed error analysis.
The speciﬁc beneﬁts of randomized techniques depend on the c omputational environment. Con-
sider the model problem of ﬁnding the k dominant components of the singular value decomposition
of an m × n matrix. (i) For a dense input matrix, randomized algorithms require O( mn log(k))
ﬂoating-point operations (ﬂops) in contrast with O( mnk) for classical algorithms. (ii) For a sparse
input matrix, the ﬂop count matches classical Krylov subspa ce methods, but the randomized ap-
proach is more robust and can easily be reorganized to exploi t multi-processor architectures. (iii) For
a matrix that is too large to ﬁt in fast memory, the randomized techniques require only a constant
number of passes over the data, as opposed to O( k) passes for classical algorithms. In fact, it is
sometimes possible to perform matrix approximation with a single pass over the data.
Key words. Dimension reduction, eigenvalue decomposition, interpol ative decomposition,
Johnson–Lindenstrauss lemma, matrix approximation, para llel algorithm, pass-eﬃcient algorithm,
principal component analysis, randomized algorithm, rand om matrix, rank-revealing QR factoriza-
tion, singular value decomposition, streaming algorithm.
AMS subject classiﬁcations. [MSC2010] Primary: 65F30. Secondary: 68W20, 60B20.
Part I: Introduction
1. Overview. On a well-known list of the “Top 10 Algorithms” that have in-
ﬂuenced the practice of science and engineering during the 20th ce ntury [40], we ﬁnd
an entry that is not really an algorithm: the idea of using matrix factorizations to
accomplish basic tasks in numerical linear algebra. In the accompany ing article [127],
Stewart explains that
The underlying principle of the decompositional approach to matrix
computation is that it is not the business of the matrix algorith-
micists to solve particular problems but to construct computationa l
platforms from which a variety of problems can be solved.
Stewart goes on to argue that this point of view has had many fruitf ul consequences,
including the development of robust software for performing thes e factorizations in a
†Department of Applied Mathematics, University of Colorado at Boulder, Boulder, CO 80309-
0526. Supported by NSF awards #0748488 and #0610097.
‡Computing & Mathematical Sciences, California Institute o f Technology, MC 305-16, Pasadena,
CA 91125-5000. Supported by ONR award #N000140810883.
1

<!-- page 2 -->

2 HALKO, MARTINSSON, AND TROPP
highly accurate and provably correct manner.
The decompositional approach to matrix computation remains fund amental, but
developments in computer hardware and the emergence of new app lications in the
information sciences have rendered the classical algorithms for th is task inadequate
in many situations:
• A salient feature of modern applications, especially in data mining, is th at
the matrices are stupendously big. Classical algorithms are not alwa ys well
adapted to solving the type of large-scale problems that now arise.
• In the information sciences, it is common that data are missing or inac curate.
Classical algorithms are designed to produce highly accurate matrix decompo-
sitions, but it seems proﬂigate to spend extra computational reso urces when
the imprecision of the data inherently limits the resolution of the outp ut.
• Data transfer now plays a major role in the computational cost of n umerical
algorithms. Techniques that require few passes over the data may be substan-
tially faster in practice, even if they require as many—or more—ﬂoat ing-point
operations.
• As the structure of computer processors continues to evolve, it becomes in-
creasingly important for numerical algorithms to adapt to a range o f novel
architectures, such as graphics processing units.
The purpose of this paper is make the case that randomized algorithms pro-
vide a powerful tool for constructing approximate matrix factor izations. These tech-
niques are simple and eﬀective, sometimes impressively so. Compared with stan-
dard deterministic algorithms, the randomized methods are often f aster and—perhaps
surprisingly—more robust. Furthermore, they can produce fact orizations that are ac-
curate to any speciﬁed tolerance above machine precision, which allo ws the user to
trade accuracy for speed if desired. We present numerical eviden ce that these algo-
rithms succeed for real computational problems.
In short, our goal is to demonstrate how randomized methods inte ract with classi-
cal techniques to yield eﬀective, modern algorithms supported by d etailed theoretical
guarantees. We have made a special eﬀort to help practitioners ide ntify situations
where randomized techniques may outperform established method s.
Throughout this article, we provide detailed citations to previous wo rk on ran-
domized techniques for computing low-rank approximations. The pr imary sources
that inform our presentation include [17, 46, 58, 91, 105, 112, 113 , 118, 137].
Remark 1.1. Our experience suggests that many practitioners of scientiﬁc com -
puting view randomized algorithms as a desperate and ﬁnal resort. Let us address
this concern immediately. Classical Monte Carlo methods are highly se nsitive to the
random number generator and typically produce output with low and uncertain ac-
curacy. In contrast, the algorithms discussed herein are relative ly insensitive to the
quality of randomness and produce highly accurate results. The pr obability of failure
is a user-speciﬁed parameter that can be rendered negligible (say, less than 10 − 15)
with a nominal impact on the computational resources required.
1.1. Approximation by low-rank matrices. The roster of standard matrix
decompositions includes the pivoted QR factorization, the eigenvalu e decomposition,
and the singular value decomposition (SVD), all of which expose the ( numerical) range
of a matrix. Truncated versions of these factorizations are ofte n used to express a

<!-- page 3 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 3
low-rank approximation of a given matrix:
A ≈ B C ,
m × n m × k k × n. (1.1)
The inner dimension k is sometimes called the numerical rank of the matrix. When
the numerical rank is much smaller than either dimension m or n, a factorization such
as (1.1) allows the matrix to be stored inexpensively and to be multiplied rapidly with
vectors or other matrices. The factorizations can also be used fo r data interpretation
or to solve computational problems, such as least squares.
Matrices with low numerical rank appear in a wide variety of scientiﬁc a pplica-
tions. We list only a few:
• A basic method in statistics and data mining is to compute the direction s of
maximal variance in vector-valued data by performing principal component
analysis (PCA) on the data matrix. PCA is nothing other than a low-rank
matrix approximation [71, §14.5].
• Another standard technique in data analysis is to perform low-dimen sional
embedding of data under the assumption that there are fewer deg rees of
freedom than the ambient dimension would suggest. In many cases, the
method reduces to computing a partial SVD of a matrix derived from the
data. See [71, §§14.8–14.9] or [30].
• The problem of estimating parameters from measured data via least -squares
ﬁtting often leads to very large systems of linear equations that ar e close to
linearly dependent. Eﬀective techniques for factoring the coeﬃcie nt matrix
lead to eﬃcient techniques for solving the least-squares problem, [1 13].
• Many fast numerical algorithms for solving PDEs and for rapidly evalu ating
potential ﬁelds such as the fast multipole method [66] and H-matrices [65],
rely on low-rank approximations of continuum operators.
• Models of multiscale physical phenomena often involve PDEs with rapid ly
oscillating coeﬃcients. Techniques for model reduction or coarse graining in
such environments are often based on the observation that the lin ear trans-
form that maps the input data to the requested output data can b e approxi-
mated by an operator of low rank [56].
1.2. Matrix approximation framework. The task of computing a low-rank
approximation to a given matrix can be split naturally into two computa tional stages.
The ﬁrst is to construct a low-dimensional subspace that capture s the action of the
matrix. The second is to restrict the matrix to the subspace and th en compute a
standard factorization (QR, SVD, etc.) of the reduced matrix. To be slightly more
formal, we subdivide the computation as follows.
Stage A: Compute an approximate basis for the range of the input matrix A. In
other words, we require a matrix Q for which
Q has orthonormal columns and A ≈ QQ∗ A. (1.2)
We would like the basis matrix Q to contain as few columns as possible, but it is
even more important to have an accurate approximation of the inpu t matrix.
Stage B: Given a matrix Q that satisﬁes (1.2), we use Q to help compute a
standard factorization (QR, SVD, etc.) of A.

<!-- page 4 -->

4 HALKO, MARTINSSON, AND TROPP
The task in Stage A can be executed very eﬃciently with random samp ling meth-
ods, and these methods are the primary subject of this work. In t he next subsection,
we oﬀer an overview of these ideas. The body of the paper provides details of the
algorithms ( §4) and a theoretical analysis of their performance ( §§8–11).
Stage B can be completed with well-established deterministic methods . Sec-
tion 3.3.3 contains an introduction to these techniques, and §5 shows how we apply
them to produce low-rank factorizations.
At this point in the development, it may not be clear why the output fr om Stage A
facilitates our job in Stage B. Let us illustrate by describing how to ob tain an ap-
proximate SVD of the input matrix A given a matrix Q that satisﬁes (1.2). More
precisely, we wish to compute matrices U and V with orthonormal columns and a
nonnegative, diagonal matrix Σ such that A ≈ U ΣV ∗ . This goal is achieved after
three simple steps:
1. Form B = Q∗ A, which yields the low-rank factorization A ≈ QB.
2. Compute an SVD of the small matrix: B = ˜U ΣV ∗ .
3. Set U = Q ˜U .
When Q has few columns, this procedure is eﬃcient because we can easily con -
struct the reduced matrix B and rapidly compute its SVD. In practice, we can often
avoid forming B explicitly by means of subtler techniques. In some cases, it is not
even necessary to revisit the input matrix A during Stage B. This observation allows
us to develop single-pass algorithms, which look at each entry of A only once.
Similar manipulations readily yield other standard factorizations, suc h as the
pivoted QR factorization, the eigenvalue decomposition, etc.
1.3. Randomized algorithms. This paper describes a class of randomized al-
gorithms for completing Stage A of the matrix approximation framew ork set forth
in §1.2. We begin with some details about the approximation problem these algo-
rithms target ( §1.3.1). Afterward, we motivate the random sampling technique with
a heuristic explanation ( §1.3.2) that leads to a prototype algorithm ( §1.3.3).
1.3.1. Problem formulations. The basic challenge in producing low-rank ma-
trix approximations is a primitive question that we call the ﬁxed-precision approxi-
mation problem . Suppose we are given a matrix A and a positive error tolerance ε.
We seek a matrix Q with k = k(ε) orthonormal columns such that
∥A − QQ∗ A∥ ≤ ε, (1.3)
where ∥·∥denotes the ℓ2 operator norm. The range of Q is a k-dimensional subspace
that captures most of the action of A, and we would like k to be as small as possible.
The singular value decomposition furnishes an optimal answer to the ﬁxed-precision
problem [97]. Let σj denote the jth largest singular value of A. For each j ≥ 0,
min
rank(X)≤ j
∥A − X∥ = σj+1. (1.4)
One way to construct a minimizer is to choose X = QQ∗ A, where the columns of Q
are k dominant left singular vectors of A. Consequently, the minimal rank k where
(1.3) holds equals the number of singular values of A that exceed the tolerance ε.
To simplify the development of algorithms, it is convenient to assume t hat the
desired rank k is speciﬁed in advance. We call the resulting problem the ﬁxed-rank
approximation problem . Given a matrix A, a target rank k, and an oversampling

<!-- page 5 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 5
parameter p, we seek to construct a matrix Q with k + p orthonormal columns such
that
∥A − QQ∗ A∥ ≈ min
rank(X)≤ k
∥A − X∥ . (1.5)
Although there exists a minimizer Q that solves the ﬁxed rank problem for p = 0, the
opportunity to use a small number of additional columns provides a ﬂ exibility that is
crucial for the eﬀectiveness of the computational methods we dis cuss.
We will demonstrate that algorithms for the ﬁxed-rank problem can be adapted
to solve the ﬁxed-precision problem. The connection is based on the observation that
we can build the basis matrix Q incrementally and, at any point in the computation,
we can inexpensively estimate the residual error ∥A − QQ∗ A∥. Refer to §4.4 for the
details of this reduction.
1.3.2. Intuition. To understand how randomness helps us solve the ﬁxed-rank
problem, it is helpful to consider some motivating examples.
First, suppose that we seek a basis for the range of a matrix A with exact rank
k. Draw a random vector ω, and form the product y = Aω. For now, the precise
distribution of the random vector is unimportant; just think of y as a random sample
from the range of A. Let us repeat this sampling process k times:
y(i) = Aω(i), i = 1, 2, . . . , k. (1.6)
Owing to the randomness, the set {ω(i) : i = 1 , 2, . . . , k } of random vectors is likely
to be in general linear position. In particular, the random vectors f orm a linearly
independent set and no linear combination falls in the null space of A. As a result,
the set {y(i) : i = 1 , 2, . . . , k } of sample vectors is also linearly independent, so it
spans the range of A. Therefore, to produce an orthonormal basis for the range of A,
we just need to orthonormalize the sample vectors.
Now, imagine that A = B + E where B is a rank- k matrix containing the
information we seek and E is a small perturbation. Our priority is to obtain a basis
that covers as much of the range of B as possible, rather than to minimize the number
of basis vectors. Therefore, we ﬁx a small number p, and we generate k + p samples
y(i) = Aω(i) = Bω (i) + Eω (i), i = 1, 2, . . . , k + p. (1.7)
The perturbation E shifts the direction of each sample vector outside the range of B,
which can prevent the span of {y(i) : i = 1 , 2, . . . , k } from covering the entire range
of B. In contrast, the enriched set {y(i) : i = 1 , 2, . . . , k + p} of samples has a much
better chance of spanning the required subspace.
Just how many extra samples do we need? Remarkably, for certain t ypes of
random sampling schemes, the failure probability decreases supere xponentially with
the oversampling parameter p; see (1.9). As a practical matter, setting p = 5 or p = 10
often gives superb results. This observation is one of the principal facts supporting
the randomized approach to numerical linear algebra.
1.3.3. A prototype algorithm. The intuitive approach of §1.3.2 can be ap-
plied to general matrices. Omitting computational details for now, w e formalize the
procedure in the ﬁgure labeled Proto-Algorithm.
This simple algorithm is by no means new. It is essentially the ﬁrst step o f a
subspace iteration with a random initial subspace [61, §7.3.2]. The novelty comes
from the additional observation that the initial subspace should ha ve a slightly higher

<!-- page 6 -->

6 HALKO, MARTINSSON, AND TROPP
Proto-Algorithm: Solving the Fixed-Rank Problem
Given an m × n matrix A, a target rank k, and an oversampling parameter
p, this procedure computes an m × (k + p) matrix Q whose columns are
orthonormal and whose range approximates the range of A.
1 Draw a random n × (k + p) test matrix Ω.
2 Form the matrix product Y = AΩ.
3 Construct a matrix Q whose columns form an orthonormal basis for
the range of Y .
dimension than the invariant subspace we are trying to approximate . With this
revision, it is often the case that no further iteration is required to obtain a high-
quality solution to (1.5). We believe this idea can be traced to [91, 105, 118].
In order to invoke the proto-algorithm with conﬁdence, we must ad dress several
practical and theoretical issues:
• What random matrix Ω should we use? How much oversampling do we need?
• The matrix Y is likely to be ill-conditioned. How do we orthonormalize its
columns to form the matrix Q?
• What are the computational costs?
• How can we solve the ﬁxed-precision problem (1.3) when the numerica l rank
of the matrix is not known in advance?
• How can we use the basis Q to compute other matrix factorizations?
• Does the randomized method work for problems of practical intere st? How
does its speed/accuracy/robustness compare with standard te chniques?
• What error bounds can we expect? With what probability?
The next few sections provide a summary of the answers to these q uestions. We
describe several problem regimes where the proto-algorithm can b e implemented ef-
ﬁciently, and we present a theorem that describes the performan ce of the most im-
portant instantiation. Finally, we elaborate on how these ideas can b e applied to
approximate the truncated SVD of a large data matrix. The rest of the paper con-
tains a more exhaustive treatment—including pseudocode, numeric al experiments,
and a detailed theory.
1.4. A comparison between randomized and traditional techn iques. To
select an appropriate computational method for ﬁnding a low-rank approximation to
a matrix, the practitioner must take into account the properties o f the matrix. Is
it dense or sparse? Does it ﬁt in fast memory or is it stored out of cor e? Does the
singular spectrum decay quickly or slowly? The behavior of a numerica l linear algebra
algorithm may depend on all these factors [13, 61, 132]. To facilitate a comparison be-
tween classical and randomized techniques, we summarize their rela tive performance
in each of three representative environments. Section 6 contains a more in-depth
treatment.
We focus on the task of computing an approximate SVD of an m × n matrix A
with numerical rank k. For randomized schemes, Stage A generally dominates the
cost of Stage B in our matrix approximation framework ( §1.2). Within Stage A, the
computational bottleneck is usually the matrix–matrix product AΩ in Step 2 of the
proto-algorithm ( §1.3.3). The power of randomized algorithms stems from the fact

<!-- page 7 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 7
that we can reorganize this matrix multiplication for maximum eﬃciency in a variety
of computational architectures.
1.4.1. A general dense matrix that ﬁts in fast memory . A standard deter-
ministic technique for computing an approximate SVD is to perform a r ank-revealing
QR factorization of the matrix, and then to manipulate the factors to obtain the ﬁnal
decomposition. The cost of this approach is typically O( kmn) ﬂoating-point opera-
tions, or ﬂops, although these methods require slightly longer running times in rare
cases [68].
In contrast, randomized schemes can produce an approximate SV D using only
O(mn log(k) + ( m + n)k2) ﬂops. The gain in asymptotic complexity is achieved by
using a random matrix Ω that has some internal structure, which allows us to evaluate
the product AΩ rapidly. For example, randomizing and subsampling the discrete
Fourier transform works well. Sections 4.6 and 11 contain more infor mation on this
approach.
1.4.2. A matrix for which matrix–vector products can be eval uated
rapidly . When the matrix A is sparse or structured, we may be able to apply it
rapidly to a vector. In this case, the classical prescription for com puting a partial SVD
is to invoke a Krylov subspace method, such as the Lanczos or Arno ldi algorithm.
It is diﬃcult to summarize the computational cost of these methods because their
performance depends heavily on properties of the input matrix and on the amount of
eﬀort spent to stabilize the algorithm. (Inherently, the Lanczos a nd Arnoldi methods
are numerically unstable.) For the same reasons, the error analysis of such schemes
is unsatisfactory in many important environments.
At the risk of being overly simplistic, we claim that the typical cost of a Krylov
method for approximating the k leading singular vectors of the input matrix is pro-
portional to k Tmult + (m + n)k2, where Tmult denotes the cost of a matrix–vector
multiplication with the input matrix and the constant of proportionalit y is small. We
can also apply randomized methods using a Gaussian test matrix Ω to complete the
factorization at the same cost, O( k Tmult + (m + n)k2) ﬂops.
With a given budget of ﬂoating-point operations, Krylov methods so metimes
deliver a more accurate approximation than randomized algorithms. Nevertheless, the
methods described in this survey have at least two powerful advan tages over Krylov
methods. First, the randomized schemes are inherently stable, an d they come with
very strong performance guarantees that do not depend on sub tle spectral properties
of the input matrix. Second, the matrix–vector multiplies required t o form AΩ can be
performed in parallel . This fact allows us to restructure the calculations to take full
advantage of the computational platform, which can lead to drama tic accelerations
in practice, especially for parallel and distributed machines.
A more detailed comparison or randomized schemes and Krylov subsp ace methods
is given in §6.2.
1.4.3. A general dense matrix stored in slow memory or stream ed.
When the input matrix is too large to ﬁt in core memory, the cost of tr ansferring the
matrix from slow memory typically dominates the cost of performing t he arithmetic.
The standard techniques for low-rank approximation described in §1.4.1 require O( k)
passes over the matrix, which can be prohibitively expensive.
In contrast, the proto-algorithm of §1.3.3 requires only one pass over the data to
produce the approximate basis Q for Stage A of the approximation framework. This
straightforward approach, unfortunately, is not accurate eno ugh for matrices whose

<!-- page 8 -->

8 HALKO, MARTINSSON, AND TROPP
singular spectrum decays slowly, but we can address this problem us ing very few (say,
2 to 4) additional passes over the data [112]. See §1.6 or §4.5 for more discussion.
Typically, Stage B uses one additional pass over the matrix to const ruct the ap-
proximate SVD. With slight modiﬁcations, however, the two-stage r andomized scheme
can be revised so that it only makes a single pass over the data. Refe r to §5.5 for
information.
1.5. Performance analysis. A principal goal of this paper is to provide a de-
tailed analysis of the performance of the proto-algorithm describe d in §1.3.3. This
investigation produces precise error bounds, expressed in terms of the singular values
of the input matrix. Furthermore, we determine how several choic es of the random
matrix Ω impact the behavior of the algorithm.
Let us oﬀer a taste of this theory. The following theorem describes the average-
case behavior of the proto-algorithm with a Gaussian test matrix, a ssuming we per-
form the computation in exact arithmetic. This result is a simpliﬁed ver sion of The-
orem 10.6.
Theorem 1.1. Suppose that A is a real m × n matrix. Select a target rank
k ≥ 2 and an oversampling parameter p ≥ 2, where k + p ≤ min{m, n }. Execute the
proto-algorithm with a standard Gaussian test matrix to obt ain an m × (k + p) matrix
Q with orthonormal columns. Then
E ∥A − QQ∗ A∥ ≤
[
1 + 4√ k + p
p − 1 ·
√
min{m, n }
]
σk+1, (1.8)
where E denotes expectation with respect to the random test matrix a nd σk+1 is the
(k + 1)th singular value of A.
We recall that the term σk+1 appearing in (1.8) is the smallest possible error (1.4)
achievable with any basis matrix Q. The theorem asserts that, on average, the al-
gorithm produces a basis whose error lies within a small polynomial fac tor of the
theoretical minimum. Moreover, the error bound (1.8) in the rando mized algorithm
is slightly sharper than comparable bounds for deterministic techniq ues based on
rank-revealing QR algorithms [68].
The reader might be worried about whether the expectation provid es a useful
account of the approximation error. Fear not: the actual outco me of the algorithm
is almost always very close to the typical outcome because of measure concentra tion
eﬀects. As we discuss in §10.3, the probability that the error satisﬁes
∥A − QQ∗ A∥ ≤
[
1 + 11
√
k + p ·
√
min{m, n }
]
σk+1 (1.9)
is at least 1 − 6 ·p− p under very mild assumptions on p. This fact justiﬁes the use of
an oversampling term as small as p = 5. This simpliﬁed estimate is very similar to
the major results in [91].
The theory developed in this paper provides much more detailed infor mation
about the performance of the proto-algorithm.
• When the singular values of A decay slightly, the error ∥A − QQ∗ A∥ does
not depend on the dimensions of the matrix ( §§10.2–10.3).
• We can reduce the size of the bracket in the error bound (1.8) by co mbining
the proto-algorithm with a power iteration ( §10.4). For an example, see §1.6
below.

<!-- page 9 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 9
Prototype for Randomized SVD
Given an m × n matrix A, a target number k of singular vectors, and an
exponent q (say q = 1 or q = 2 ), this procedure computes an approximate
rank-2k factorization U ΣV ∗ , where U and V are orthonormal, and Σ is
nonnegative and diagonal.
Stage A:
1 Generate an n × 2k Gaussian test matrix Ω.
2 Form Y = (AA∗ )qAΩ by multiplying alternately with A and A∗ .
3 Construct a matrix Q whose columns form an orthonormal basis for
the range of Y .
Stage B:
4 Form B = Q∗ A.
5 Compute an SVD of the small matrix: B = ˜U ΣV ∗ .
6 Set U = Q ˜U .
Note: The computation of Y in Step 2 is vulnerable to round-oﬀ errors.
When high accuracy is required, we must incorporate an orthonorm alization
step between each application of A and A∗ ; see Algorithm 4.4.
• For the structured random matrices we mentioned in §1.4.1, related error
bounds are in force ( §11).
• We can obtain inexpensive a posteriori error estimates to verify the quality
of the approximation ( §4.3).
1.6. Example: Randomized SVD. We conclude this introduction with a
short discussion of how these ideas allow us to perform an approxima te SVD of a
large data matrix, which is a compelling application of randomized matrix approxi-
mation [112].
The two-stage randomized method oﬀers a natural approach to S VD compu-
tations. Unfortunately, the simplest version of this scheme is inade quate in many
applications because the singular spectrum of the input matrix may d ecay slowly. To
address this diﬃculty, we incorporate q steps of a power iteration, where q = 1 or
q = 2 usually suﬃces in practice. The complete scheme appears in the bo x labeled
Prototype for Randomized SVD. For most applications, it is importan t to incorporate
additional reﬁnements, as we discuss in §§4–5.
The Randomized SVD procedure requires only 2( q + 1) passes over the matrix,
so it is eﬃcient even for matrices stored out-of-core. The ﬂop cou nt satisﬁes
TrandSVD = (2q + 2) k Tmult + O(k2(m + n)),
where Tmult is the ﬂop count of a matrix–vector multiply with A or A∗ . We have the
following theorem on the performance of this method in exact arithm etic, which is a
consequence of Corollary 10.10.
Theorem 1.2. Suppose that A is a real m × n matrix. Select an exponent q
and a target number k of singular vectors, where 2 ≤ k ≤ 0. 5 min{m, n }. Execute the

<!-- page 10 -->

10 HALKO, MARTINSSON, AND TROPP
Randomized SVD algorithm to obtain a rank- 2k factorization U ΣV ∗ . Then
E ∥A − U ΣV ∗ ∥ ≤
[
1 + 4
√
2 min{m, n }
k − 1
] 1/ (2q+1)
σk+1, (1.10)
where E denotes expectation with respect to the random test matrix a nd σk+1 is the
(k + 1)th singular value of A.
This result is new. Observe that the bracket in (1.10) is essentially th e same as
the bracket in the basic error bound (1.8). We ﬁnd that the power it eration drives
the leading constant to one exponentially fast as the power q increases. The rank- k
approximation of A can never achieve an error smaller than σk+1, so the randomized
procedure computes 2 k approximate singular vectors that capture as much of the
matrix as the ﬁrst k actual singular vectors.
In practice, we can truncate the approximate SVD, retaining only t he ﬁrst k
singular values and vectors. Equivalently, we replace the diagonal f actor Σ by the
matrix Σ(k) formed by zeroing out all but the largest k entries of Σ. For this truncated
SVD, we have the error bound
E

A − U Σ(k)V ∗ 
 ≤ σk+1 +
[
1 + 4
√
2 min{m, n }
k − 1
] 1/ (2q+1)
σk+1. (1.11)
In words, we pay no more than an additive term σk+1 when we perform the truncation
step. Our numerical experience suggests that the error bound ( 1.11) is pessimistic.
See Remark 5.1 and §9.4 for some discussion of truncation.
1.7. Outline of paper. The paper is organized into three parts: an introduction
(§§1–3), a description of the algorithms ( §§4–7), and a theoretical performance analysis
(§§8–11). The two latter parts commence with a short internal outline . Each part is
more or less self-contained, and after a brief review of our notatio n in §§3.1–3.2, the
reader can proceed to either the algorithms or the theory part.
2. Related work and historical context. Randomness has occasionally sur-
faced in the numerical linear algebra literature; in particular, it is quit e standard to
initialize iterative algorithms for constructing invariant subspaces w ith a randomly
chosen point. Nevertheless, we believe that sophisticated ideas fr om random matrix
theory have not been incorporated into classical matrix factoriza tion algorithms un-
til very recently. We can trace this development to earlier work in co mputer science
and—especially—to probabilistic methods in geometric analysis. This se ction presents
an overview of the relevant work. We begin with a survey of randomiz ed methods for
matrix approximation; then we attempt to trace some of the ideas b ackward to their
sources.
2.1. Randomized matrix approximation. Matrices of low numerical rank
contain little information relative to their apparent dimension owing to the linear de-
pendency in their columns (or rows). As a result, it is reasonable to e xpect that these
matrices can be approximated with far fewer degrees of freedom. A less obvious fact
is that randomized schemes can be used to produce these approxim ations eﬃciently.
Several types of approximation techniques build on this idea. These methods all
follow the same basic pattern:

<!-- page 11 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 11
1. Preprocess the matrix, usually to calculate sampling probabilities.
2. Take random samples from the matrix, where the term sample refers generi-
cally to a linear function of the matrix.
3. Postprocess the samples to compute a ﬁnal approximation, typ ically with
classical techniques from numerical linear algebra. This step may re quire
another look at the matrix.
We continue with a description of the most common approximation sch emes.
2.1.1. Sparsiﬁcation. The simplest approach to matrix approximation is the
method of sparsiﬁcation or the related technique of quantization. The goal of sparsiﬁ-
cation is to replace the matrix by a surrogate that contains far few er nonzero entries.
Quantization produces an approximation whose components are dr awn from a (small)
discrete set of values. These methods can be used to limit storage r equirements or
to accelerate computations by reducing the cost of matrix–vecto r and matrix–matrix
multiplies [94, Ch. 6]. The manuscript [33] describes applications in optim ization.
Sparsiﬁcation typically involves very simple elementwise calculations. E ach entry
in the approximation is drawn independently at random from a distribu tion deter-
mined from the corresponding entry of the input matrix. The expec ted value of the
random approximation equals the original matrix, but the distributio n is designed so
that a typical realization is much sparser.
The ﬁrst method of this form was devised by Achlioptas and McSherr y [2], who
built on earlier work on graph sparsiﬁcation due to Karger [75, 76]. Ar ora–Hazan–
Kale presented a diﬀerent sampling method in [7]. See [60, 123] for som e recent work
on sparsiﬁcation.
2.1.2. Column selection methods. A second approach to matrix approxima-
tion is based on the idea that a small set of columns describes most of the action of
a numerically low-rank matrix. Indeed, classical existential results [117] demonstrate
that every m × n matrix A contains a k-column submatrix C for which

A − CC †A

 ≤
√
1 + k(n − k) ·

A − A(k)

 , (2.1)
where k is a parameter, the dagger † denotes the pseudoinverse, and A(k) is a best
rank-k approximation of A. It is NP-hard to perform column selection by optimiz-
ing natural objective functions, such as the condition number of t he submatrix [27].
Nevertheless, there are eﬃcient deterministic algorithms, such as the rank-revealing
QR method of [68], that can nearly achieve the error bound (2.1).
There is a class of randomized algorithms that approach the ﬁxed-r ank approx-
imation problem (1.5) using this intuition. These methods ﬁrst comput e a sampling
probability for each column, either using the squared Euclidean norm s of the columns
or their leverage scores. (Leverage scores reﬂect the relative importance of the columns
to the action of the matrix; they can be calculated easily from the do minant k right
singular vectors of the matrix.) Columns are then selected randomly according to this
distribution. Afterward, a postprocessing step is invoked to prod uce a more reﬁned
approximation of the matrix.
We believe that the earliest method of this form appeared in a 1998 pa per of
Frieze–Kannan–Vempala [57, 58]. This work was reﬁned substantially in the pa-
pers [43, 44, 46]. The basic algorithm samples columns from a distribut ion related
to the squared ℓ2 norms of the columns. This sampling step produces a small column
submatrix whose range is aligned with the range of the input matrix. T he ﬁnal ap-
proximation is obtained from a truncated SVD of the submatrix. Give n a target rank

<!-- page 12 -->

12 HALKO, MARTINSSON, AND TROPP
k and a parameter ε > 0, this approach samples ℓ = ℓ(k, ε ) columns of the matrix to
produce a rank- k approximation B that satisﬁes
∥A − B∥F ≤

A − A(k)


F + ε ∥A∥F , (2.2)
where ∥·∥F denotes the Frobenius norm. We note that the algorithm of [46] req uires
only a constant number of passes over the data.
Rudelson and Vershynin later showed that the same type of column s ampling
method also yields spectral-norm error bounds [116]. The technique s in their paper
have been very inﬂuential; their work has found other applications in randomized
regression [52], sparse approximation [133], and compressive samplin g [19].
Deshpande et al. [37, 38] demonstrated that the error in the colum n sampling
approach can be improved by iteration and adaptive volume sampling. They showed
that it is possible to produce a rank- k matrix B that satisﬁes
∥A − B∥F ≤ (1 + ε)

A − A(k)


F (2.3)
using a k-pass algorithm. Around the same time, Har-Peled [70] independent ly de-
veloped a recursive algorithm that oﬀers the same approximation gu arantees. Very
recently, Desphande and Rademacher have improved the running t ime of volume-
based sampling methods [36].
Drineas et al. and Boutsidis et al. have also developed randomized algo rithms
for the column subset selection problem , which requests a column submatrix C that
achieves a bound of the form (2.1). Via the methods of Rudelson and Vershynin [116],
they showed that sampling columns according to their leverage scor es is likely to
produce the required submatrix [50, 51]. Subsequent work [17, 18] showed that post-
processing the sampled columns with a rank-revealing QR algorithm ca n reduce the
number of output columns required (2.1). The argument in [17] explic itly decou-
ples the linear algebraic part of the analysis from the random matrix t heory. The
theoretical analysis in the present work involves a very similar techn ique.
2.1.3. Approximation by dimension reduction. A third approach to matrix
approximation is based on the concept of dimension reduction . Since the rows of a
low-rank matrix are linearly dependent, they can be embedded into a low-dimensional
space without altering their geometric properties substantially. A r andom linear map
provides an eﬃcient, nonadaptive way to perform this embedding. ( Column sampling
can also be viewed as an adaptive form of dimension reduction.)
The proto-algorithm we set forth in §1.3.3 is simply a dual description of the
dimension reduction approach: collecting random samples from the c olumn space of
the matrix is equivalent to reducing the dimension of the rows. No pre computation
is required to obtain the sampling distribution, but the sample itself ta kes some work
to collect. Afterward, we orthogonalize the samples as preparatio n for constructing
various matrix approximations.
We believe that the idea of using dimension reduction for algorithmic ma trix
approximation ﬁrst appeared in a 1998 paper of Papadimitriou et al. [1 04, 105], who
described an application to latent semantic indexing (LSI). They sug gested projecting
the input matrix onto a random subspace and compressing the origin al matrix to (a
subspace of) the range of the projected matrix. They establishe d error bounds that
echo the result (2.2) of Frieze et al. [58]. Although the Euclidean colum n selection
method is a more computationally eﬃcient way to obtain this type of er ror bound,
dimension reduction has other advantages, e.g., in terms of accura cy.

<!-- page 13 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 13
Sarl´ os argued in [118] that the computational costs of dimension reduction can be
reduced substantially by means of the structured random maps pr oposed by Ailon–
Chazelle [3]. Sarl´ os used these ideas to develop eﬃcient randomized a lgorithms for
least-squares problems; he also studied approximate matrix multiplic ation and low-
rank matrix approximation. The recent paper [102] analyzes a very similar matrix
approximation algorithm using Rudelson and Vershynin’s methods [116 ].
The initial work of Sarl´ os on structured dimension reduction did not immediately
yield algorithms for low-rank matrix approximation that were superio r to classical
techniques. Woolfe et al. showed how to obtain an improvement in asy mptotic com-
putational cost, and they applied these techniques to problems in s cientiﬁc comput-
ing [137]. Related work includes [86, 88].
Martinsson–Rokhlin–Tygert have studied dimension reduction using a Gaussian
transform matrix, and they demonstrated that this approach pe rforms much better
than earlier analyses had suggested [91]. Their work highlights the imp ortance of over-
sampling, and their error bounds are very similar to the estimate (1.9 ) we presented
in the introduction. They also demonstrated that dimension reduct ion can be used
to compute an interpolative decomposition of the input matrix, which is essentially
equivalent to performing column subset selection.
Rokhlin–Szlam–Tygert have shown that combining dimension reductio n with a
power iteration is an eﬀective way to improve its performance [112]. T hese ideas
lead to very eﬃcient randomized methods for large-scale PCA [69]. An eﬃcient,
numerically stable version of the power iteration is discussed in §4.5, as well as [92].
Related ideas appear in a paper of Roweis [114].
Very recently, Clarkson and Woodruﬀ [29] have developed one-pas s algorithms for
performing low-rank matrix approximation, and they have establish ed lower bounds
which prove that many of their algorithms have optimal or near-opt imal resource
guarantees, modulo constants.
2.1.4. Approximation by submatrices. The matrix approximation literature
contains a subgenre that discusses methods for building an approx imation from a
submatrix and computed coeﬃcient matrices. For example, we can c onstruct an
approximation using a subcollection of columns (the interpolative dec omposition), a
subcollection of rows and a subcollection of columns (the CUR decomp osition), or a
square submatrix (the matrix skeleton). This type of decompositio n was developed
and studied in several papers, including [26, 64, 126]. For data analy sis applications,
see the recent paper [89].
A number of works develop randomized algorithms for this class of ma trix ap-
proximations. Drineas et al. have developed techniques for comput ing CUR decom-
positions, which express A ≈ CU R, where C and R denote small column and row
submatrices of A and where U is a small linkage matrix. These methods identify
columns (rows) that approximate the range (corange) of the mat rix; the linkage matrix
is then computed by solving a small least-squares problem. A random ized algorithm
for CUR approximation with controlled absolute error appears in [47]; a relative error
algorithm appears in [51]. We also mention a paper on computing a closely related
factorization called the compact matrix decomposition [129].
It is also possible to produce interpolative decompositions and matrix skeletons
using randomized methods, as discussed in [91, 112] and §5.2 of the present work.
2.1.5. Other numerical problems. The literature contains a variety of other
randomized algorithms for solving standard problems in and around n umerical linear
algebra. We list some of the basic references.

<!-- page 14 -->

14 HALKO, MARTINSSON, AND TROPP
T ensor skeletons. Randomized column selection methods can be used to produce
CUR-type decompositions of higher-order tensors [49].
Matrix multiplication. Column selection and dimension reduction techniques can
be used to accelerate the multiplication of rank-deﬁcient matrices [4 5, 118].
See also [10].
Overdetermined linear systems. The randomized Kaczmarz algorithm is a lin-
early convergent iterative method that can be used to solve overd etermined
linear systems [101, 128].
Overdetermined least squares. Fast dimension-reduction maps can sometimes ac-
celerate the solution of overdetermined least-squares problems [5 2, 118].
Nonnegative least squares. Fast dimension reduction maps can be used to reduce
the size of nonnegative least-squares problems [16].
Preconditioned least squares. Randomized matrix approximations can be used
to precondition conjugate gradient to solve least-squares proble ms [113].
Other regression problems. Randomized algorithms for ℓ1 regression are described
in [28]. Regression in ℓp for p ∈ [1, ∞ ) has also been considered [31].
F acility location. The Fermat–Weber facility location problem can be viewed as
matrix approximation with respect to a diﬀerent discrepancy measu re. Ran-
domized algorithms for this type of problem appear in [121].
2.1.6. Compressive sampling. Although randomized matrix approximation
and compressive sampling are based on some common intuitions, it is fa cile to con-
sider either one as a subspecies of the other. We oﬀer a short over view of the ﬁeld
of compressive sampling—especially the part connected with matrice s—so we can
highlight some of the diﬀerences.
The theory of compressive sampling starts with the observation th at many types
of vector-space data are compressible. That is, the data are approximated well using
a short linear combination of basis functions drawn from a ﬁxed collec tion [42]. For
example, natural images are well approximated in a wavelet basis; nu merically low-
rank matrices are well approximated as a sum of rank-one matrices . The idea behind
compressive sampling is that suitably chosen random samples from th is type of com-
pressible object carry a large amount of information. Furthermor e, it is possible to
reconstruct the compressible object from a small set of these ra ndom samples, often
by solving a convex optimization problem. The initial discovery works o f Cand` es–
Romberg–Tao [20] and Donoho [41] were written in 2004.
The earliest work in compressive sampling focused on vector-valued data; soon
after, researchers began to study compressive sampling for mat rices. In 2007, Recht–
Fazel–Parillo demonstrated that it is possible to reconstruct a rank -deﬁcient matrix
from Gaussian measurements [111]. More recently, Cand` es–Rech t [22] and Cand` es–
Tao [23] considered the problem of completing a low-rank matrix from a random
sample of its entries.
The usual goals of compressive sampling are (i) to design a method fo r collecting
informative, nonadaptive data about a compressible object and (ii) to reconstruct a
compressible object given some measured data. In both cases, th ere is an implicit
assumption that we have limited—if any—access to the underlying dat a.
In the problem of matrix approximation, we typically have a complete r epresenta-
tion of the matrix at our disposal. The point is to compute a simpler rep resentation as
eﬃciently as possible under some operational constraints. In part icular, we would like
to perform as little computation as we can, but we are usually allowed t o revisit the
input matrix. Because of the diﬀerent focus, randomized matrix ap proximation algo-

<!-- page 15 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 15
rithms require fewer random samples from the matrix and use fewer computational
resources than compressive sampling reconstruction algorithms.
2.2. Origins. This section attempts to identify some of the major threads of
research that ultimately led to the development of the randomized t echniques we
discuss in this paper.
2.2.1. Random embeddings. The ﬁeld of random embeddings is a major pre-
cursor to randomized matrix approximation. In a celebrated 1984 p aper [74], Johnson
and Lindenstrauss showed that the pairwise distances among a colle ction of N points
in a Euclidean space are approximately maintained when the points are mapped
randomly to a Euclidean space of dimension O(log N ). In other words, random em-
beddings preserve Euclidean geometry. Shortly afterward, Bour gain showed that ap-
propriate random low-dimensional embeddings preserve the geome try of point sets in
ﬁnite-dimensional ℓ1 spaces [15].
These observations suggest that we might be able to solve some com putational
problems of a geometric nature more eﬃciently by translating them in to a lower-
dimensional space and solving them there. This idea was cultivated by the theoretical
computer science community beginning in the late 1980s, with resear ch ﬂowering in
the late 1990s. In particular, nearest-neighbor search can bene ﬁt from dimension-
reduction techniques [73, 78, 80]. The papers [57, 104] were appar ently the ﬁrst to
apply this approach to linear algebra.
Around the same time, researchers became interested in simplifying the form
of dimension reduction maps and improving the computational cost o f applying the
map. Several researchers developed reﬁned results on the perf ormance of a Gaussian
matrix as a linear dimension reduction map [32, 73, 93]. Achlioptas demo nstrated that
discrete random matrices would serve nearly as well [1]. In 2006, Ailon and Chazelle
proposed the fast Johnson–Lindenstrauss transform [3], which combines the speed of
the FFT with the favorable embedding properties of a Gaussian matr ix. Subsequent
reﬁnements appear in [4, 87]. Sarl´ os then imported these techniqu es to study several
problems in numerical linear algebra, which has led to some of the fast est algorithms
currently available [88, 137].
2.2.2. Data streams. Muthukrishnan argues that a distinguishing feature of
modern data is the manner in which it is presented to us. The sheer volume of infor-
mation and the speed at which it must be processed tax our ability to transmit the
data elsewhere, to compute complicated functions on the data, or to store a substan-
tial part of the data [100, §3]. As a result, computer scientists have started to develop
algorithms that can address familiar computational problems under these novel con-
straints. The data stream phenomenon is one of the primary justiﬁ cations cited by [45]
for developing pass-eﬃcient methods for numerical linear algebra p roblems, and it is
also the focus of the recent treatment [29].
One of the methods for dealing with massive data sets is to maintain sketches,
which are small summaries that allow functions of interest to be calcu lated. In the
simplest case, a sketch is simply a random projection of the data, bu t it might be a
more sophisticated object [100, §5.1]. The idea of sketching can be traced to the work
of Alon et al. [5, 6].
2.2.3. Numerical linear algebra. Classically, the ﬁeld of numerical linear al-
gebra has focused on developing deterministic algorithms that prod uce highly ac-
curate matrix approximations with provable guarantees. Neverth eless, randomized
techniques have appeared in several environments.

<!-- page 16 -->

16 HALKO, MARTINSSON, AND TROPP
One of the original examples is the use of random models for arithmet ical errors,
which was pioneered by von Neumann and Goldstine. Their papers [135 , 136] stand
among the ﬁrst works to study the properties of random matrices . The earliest nu-
merical linear algebra algorithm that depends essentially on randomiz ed techniques
is probably Dixon’s method for estimating norms and condition number s [39].
Another situation where randomness commonly arises is the initializat ion of iter-
ative methods for computing invariant subspaces. For example, mo st numerical linear
algebra texts advocate random selection of the starting vector f or the power method
because it ensures that the vector has a nonzero component in th e direction of a dom-
inant eigenvector. Wo´ zniakowski and coauthors have analyzed the performance of the
power method and the Lanczos iteration given a random starting ve ctor [79, 85].
Among other interesting applications of randomness, we mention th e work by
Parker and Pierce, which applies a randomized FFT to eliminate pivoting in Gaus-
sian elimination [106], work by Demmel et al. who have studied randomiza tion in
connection with the stability of fast methods for linear algebra [35], a nd work by Le
and Parker utilizing randomized methods for stabilizing fast linear alge braic compu-
tations based on recursive algorithms, such as Strassen’s matrix m ultiplication [81].
2.2.4. Scientiﬁc computing. One of the ﬁrst algorithmic applications of ran-
domness is the method of Monte Carlo integration introduced by Von Neumann and
Ulam [95], and its extensions, such as the Metropolis algorithm for simu lations in sta-
tistical physics. (See [9] for an introduction.) The most basic techn ique is to estimate
an integral by sampling m points from the measure and computing an empirical mean
of the integrand evaluated at the sample locations:
∫
f (x) dµ (x) ≈ 1
m
m∑
i=1
f (Xi),
where Xi are independent and identically distributed according to the probab ility
measure µ . The law of large numbers (usually) ensures that this approach pro duces
the correct result in the limit as m → ∞ . Unfortunately, the approximation error
typically has a standard deviation of m− 1/ 2, and the method provides no certiﬁcate
of success.
The disappointing computational proﬁle of Monte Carlo integration s eems to
have inspired a distaste for randomized approaches within the scien tiﬁc computing
community. Fortunately, there are many other types of randomiz ed algorithms—such
as the ones in this paper—that do not suﬀer from the same shortco mings.
2.2.5. Geometric functional analysis. There is one more character that plays
a central role in our story: the probabilistic method in geometric ana lysis. Many of
the algorithms and proof techniques ultimately come from work in this beautiful but
recondite corner of mathematics.
Dvoretsky’s theorem [53] states (roughly) that every inﬁnite-dim ensional Banach
space contains an n-dimensional subspace whose geometry is essentially the same as
an n-dimensional Hilbert space, where n is an arbitrary natural number. In 1971,
V. D. Milman developed a striking proof of this result by showing that a random
n-dimensional subspace of an N -dimensional Banach space has this property with
exceedingly high probability, provided that N is large enough [96]. Milman’s article
debuted the concentration of measure phenomenon , which is a geometric interpreta-
tion of the classical idea that regular functions of independent ran dom variables rarely

<!-- page 17 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 17
deviate far from their mean. This work opened a new era in geometric analysis where
the probabilistic method became a basic instrument.
Another prominent example of measure concentration is Kashin’s co mputation
of the Gel’fand widths of the ℓ1 ball [77], subsequently reﬁned in [59]. This work
showed that a random (N − n)-dimensional projection of the N -dimensional ℓ1 ball has
an astonishingly small Euclidean diameter: approximately
√
(1 + log(N/n ))/n . In
contrast, a nonzero projection of the ℓ2 ball always has Euclidean diameter one. This
basic geometric fact undergirds recent developments in compress ive sampling [21].
We have already described a third class of examples: the randomized embeddings
of Johnson–Lindenstrauss [74] and of Bourgain [15].
Finally, we mention Maurey’s technique of empirical approximation. Th e original
work was unpublished; one of the earliest applications appears in [24, §1]. Although
Maurey’s idea has not received as much press as the examples above , it can lead
to simple and eﬃcient algorithms for sparse approximation. For some examples in
machine learning, consider [8, 84, 110, 119]
The importance of random constructions in the geometric analysis c ommunity
has led to the development of powerful techniques for studying ra ndom matrices.
Classical random matrix theory focuses on a detailed asymptotic an alysis of the spec-
tral properties of special classes of random matrices. In contra st, geometric analysts
know methods for determining the approximate behavior of rather complicated ﬁnite-
dimensional random matrices. See [34] for a fairly current survey a rticle. We also
mention the works of Rudelson [115] and Rudelson–Vershynin [116], w hich describe
powerful tools for studying random matrices drawn from certain d iscrete distribu-
tions. Their papers are rooted deeply in the ﬁeld of geometric funct ional analysis, but
they reach out toward computational applications.
3. Linear algebraic preliminaries. This section summarizes the background
we need for the detailed description of randomized algorithms in §§4–6 and the anal-
ysis in §§8–11. We introduce notation in §3.1, describe some standard matrix de-
compositions in §3.2, and brieﬂy review standard techniques for computing matrix
factorizations in §3.3.
3.1. Basic deﬁnitions. The standard Hermitian geometry for Cn is induced by
the inner product
⟨x, y⟩ = x ·y =
∑
j
xj yj.
The associated norm is
∥x∥2 = ⟨x, x⟩ =
∑
j
|xj|2.
We usually measure the magnitude of a matrix A with the operator norm
∥A∥ = max
x̸=0
∥Ax∥
∥x∥ ,
which is often referred to as the spectral norm. The Frobenius norm is given by
∥A∥F =
[∑
jk
|ajk|2
]1/ 2
.
The conjugate transpose, or adjoint, of a matrix A is denoted A∗ . The important
identities
∥A∥2 = ∥A∗ A∥ = ∥AA∗ ∥

<!-- page 18 -->

18 HALKO, MARTINSSON, AND TROPP
hold for each matrix A.
We say that a matrix U is orthonormal if its columns form an orthonormal set
with respect to the Hermitian inner product. An orthonormal matr ix U preserves
geometry in the sense that ∥U x∥ = ∥x∥ for every vector x. A unitary matrix is
a square orthonormal matrix, and an orthogonal matrix is a real unitary matrix.
Unitary matrices satisfy the relations U U∗ = U ∗ U = I. Both the operator norm and
the Frobenius norm are unitarily invariant , which means that
∥U AV ∗ ∥ = ∥A∥ and ∥U AV ∗ ∥F = ∥A∥F
for every matrix A and all orthonormal matrices U and V
We use the notation of [61] to denote submatrices. If A is a matrix with entries
aij, and if I = [i1, i 2, . . . , i p] and J = [j1, j 2, . . . , j q] are two index vectors, then the
associated p × q submatrix is expressed as
A(I,J ) =



ai1,j 1 · · · ai1,j q
.
.
. .
.
.
aip,j 1 · · · aip,j q


 .
For column- and row-submatrices, we use the standard abbreviat ions
A( : ,J ) = A([1, 2, ..., m ],J ), and A(I, : ) = A(I, [1, 2, ..., n ]).
3.2. Standard matrix factorizations. This section deﬁnes three basic matrix
decompositions. Methods for computing them are described in §3.3.
3.2.1. The pivoted QR factorization. Each m× n matrix A of rank k admits
a decomposition
A = QR,
where Q is an m × k orthonormal matrix, and R is a k × n weakly upper-triangular
matrix. That is, there exists a permutation J of the numbers {1, 2, . . . , n } such
that R( : ,J ) is upper triangular. Moreover, the diagonal entries of R( : ,J ) are weakly
decreasing. See [61, §5.4.1] for details.
3.2.2. The singular value decomposition (SVD). Each m × n matrix A of
rank k admits a factorization
A = U ΣV ∗ ,
where U is an m × k orthonormal matrix, V is an n × k orthonormal matrix, and Σ
is a k × k nonnegative, diagonal matrix
Σ =





σ1
σ2
. . .
σk




 .
The numbers σj are called the singular values of A. They are arranged in weakly
decreasing order:
σ1 ≥ σ2 ≥ · · · ≥ σk ≥ 0.

<!-- page 19 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 19
The columns of U and V are called left singular vectors and right singular vectors ,
respectively.
Singular values are connected with the approximability of matrices. F or each j,
the number σj+1 equals the spectral-norm discrepancy between A and an optimal
rank-j approximation [97]. That is,
σj+1 = min{∥A − B∥ : B has rank j}. (3.1)
In particular, σ1 = ∥A∥. See [61, §2.5.3 and §5.4.5] for additional details.
3.2.3. The interpolative decomposition (ID). Our ﬁnal factorization iden-
tiﬁes a collection of k columns from a rank- k matrix A that span the range of A. To
be precise, we can compute an index set J = [j1, . . . , j k] such that
A = A( : ,J ) X,
where X is a k × n matrix that satisﬁes X( : ,J ) = Ik. Furthermore, no entry of X
has magnitude larger than two. In other words, this decomposition expresses each
column of A using a linear combination of k ﬁxed columns with bounded coeﬃcients.
Stable and eﬃcient algorithms for computing the ID appear in the pap ers [26, 68].
It is also possible to compute a two-sided ID
A = W A (J ′,J ) X,
where J ′ is an index set identifying k of the rows of A, and W is an m × k matrix
that satisﬁes W(J ′, : ) = Ik and whose entries are all bounded by two.
Remark 3.1. There always exists an ID where the entries in the factor X have
magnitude bounded by one. Known proofs of this fact are constru ctive, e.g., [103,
Lem. 3.3], but they require us to ﬁnd a collection of k columns that has “maximum
volume.” It is NP-hard to identify a subset of columns with this type of extremal
property [27]. We ﬁnd it remarkable that ID computations are possib le as soon as the
bound on X is relaxed.
3.3. T echniques for computing standard factorizations. This section dis-
cusses some established deterministic techniques for computing th e factorizations pre-
sented in §3.2. The material on pivoted QR and SVD can be located in any major te xt
on numerical linear algebra, such as [61, 132]. References for the I D include [26, 68].
3.3.1. Computing the full decomposition. It is possible to compute the full
QR factorization or the full SVD of an m× n matrix to double-precision accuracy with
O(mn min{m, n }) ﬂops. Techniques for computing the SVD are iterative by necessit y,
but they converge so fast that we can treat them as ﬁnite for pra ctical purposes.
3.3.2. Computing partial decompositions. Suppose that an m × n matrix
has numerical rank k, where k is substantially smaller than m and n. In this case,
it is possible to produce a structured low-rank decomposition that a pproximates the
matrix well. Sections 4 and 5 describe a set of randomized techniques for obtaining
these partial decompositions. This section brieﬂy reviews the class ical techniques,
which also play a role in developing randomized methods.
To compute a partial QR decomposition, the classical device is the Bu singer–
Golub algorithm, which performs successive orthogonalization with p ivoting on the
columns of the matrix. The procedure halts when the Frobenius nor m of the remaining

<!-- page 20 -->

20 HALKO, MARTINSSON, AND TROPP
columns is less than a computational tolerance ε. Letting ℓ denote the number of steps
required, the process results in a partial factorization
A = QR + E, (3.2)
where Q is an m × ℓ orthonormal matrix, R is a ℓ× n weakly upper-triangular matrix,
and E is a residual that satisﬁes ∥E∥F ≤ ε. The computational cost is O( ℓmn),
and the number ℓ of steps taken is typically close to the minimal rank k for which
precision ε (in the Frobenius norm) is achievable. The Businger–Golub algorithm c an
in principle signiﬁcantly overpredict the rank, but in practice this pro blem is very
rare provided that orthonormality is maintained scrupulously.
Subsequent research has led to strong rank-revealing QR algorith ms that suc-
ceed for all matrices. For example, the Gu–Eisenstat algorithm [68] (setting their
parameter f = 2) produces an QR decomposition of the form (3.2), where
∥E∥ ≤
√
1 + 4k(n − k) ·σk+1.
Recall that σk+1 is the minimal error possible in a rank- k approximation [97]. The
cost of the Gu–Eisenstat algorithm is typically O(kmn), but it can be slightly higher
in rare cases. The algorithm can also be used to obtain an approximat e ID [26].
To compute an approximate SVD of a general m × n matrix, the most straightfor-
ward technique is to compute the full SVD and truncate it. This proc edure is stable
and accurate, but it requires O( mn min{m, n }) ﬂops. A more eﬃcient approach is to
compute a partial QR factorization and postprocess the factors to obtain a partial
SVD using the methods described below in §3.3.3. This scheme takes only O( kmn)
ﬂops. Krylov subspace methods can also compute partial SVDs at a comparable cost
of O( kmn), but they are less robust.
Note that all the techniques described in this section require exten sive random
access to the matrix, and they can be very slow when the matrix is st ored out-of-core.
3.3.3. Converting from one partial factorization to anothe r. Suppose
that we have obtained a partial decomposition of a matrix A by some means:
∥A − CB ∥ ≤ ε,
where B and C have rank k. Given this information, we can eﬃciently compute any
of the basic factorizations.
We construct a partial QR factorization using the following three st eps:
1. Compute a QR factorization of C so that C = Q1R1.
2. Form the product D = R1B, and compute a QR factorization: D = Q2R.
3. Form the product Q = Q1Q2.
The result is an orthonormal matrix Q and a weakly upper-triangular matrix R such
that ∥A − QR∥ ≤ ε.
An analogous technique yields a partial SVD:
1. Compute a QR factorization of C so that C = Q1R1.
2. Form the product D = R1B, and compute an SVD: D = U2ΣV ∗ .
3. Form the product U = Q1U2.
The result is a diagonal matrix Σ and orthonormal matrices U and V such that
∥A − U ΣV ∗ ∥ ≤ ε.

<!-- page 21 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 21
Converting B and C into a partial ID is a one-step process:
1. Compute J and X such that B = B( : ,J )X.
Then A ≈ A( : ,J )X, but the approximation error may deteriorate from the initial
estimate. For example, if we compute the ID using the Gu–Eisenstat algorithm [68]
with the parameter f = 2, then the error

A − A( : ,J )X

 ≤ (1 +
√
1 + 4k(n − k)) ·ε.
Compare this bound with Lemma 5.1 below.
3.3.4. Krylov-subspace methods. Suppose that the matrix A can be applied
rapidly to vectors, as happens when A is sparse or structured. Then Krylov subspace
techniques can very eﬀectively and accurately compute partial sp ectral decomposi-
tions. For concreteness, assume that A is Hermitian. The idea of these techniques is
to ﬁx a starting vector ω and to seek approximations to the eigenvectors within the
corresponding Krylov subspace
Vq(ω) = span {ω, Aω, A2ω, . . . , Aq− 1ω}.
Krylov methods also come in blocked versions, in which the starting vector ω is
replaced by a starting matrix Ω. A common recommendation is to draw a starting
vector ω (or starting matrix Ω) from a standardized Gaussian distribution, which
indicates a signiﬁcant overlap between Krylov methods and the meth ods in this paper.
The most basic versions of Krylov methods for computing spectral decompositions
are numerically unstable. High-quality implementations require that w e incorporate
restarting strategies, techniques for maintaining high-quality bas es for the Krylov
subspaces, etc. The diversity and complexity of such methods mak e it hard to state
a precise computational cost, but in the environment we consider in this paper, a
typical cost for a fully stable implementation would be
TKrylov ∼ k Tmult + k2(m + n), (3.3)
where Tmult is the cost of a matrix–vector multiplication.
Part II: Algorithms
This part of the paper, §§4–7, provides detailed descriptions of randomized algo-
rithms for constructing low-rank approximations to matrices. As d iscussed in §1.2,
we split the problem into two stages. In Stage A, we construct a sub space that cap-
tures the action of the input matrix. In Stage B, we use this subspa ce to obtain an
approximate factorization of the matrix.
Section 4 develops randomized methods for completing Stage A, and §5 describes
deterministic methods for Stage B. Section 6 compares the comput ational costs of the
resulting two-stage algorithm with the classical approaches outline d in §3. Finally, §7
illustrates the performance of the randomized schemes via numeric al examples.
4. Stage A: Randomized schemes for approximating the range. This
section outlines techniques for constructing a subspace that cap tures most of the
action of a matrix. We begin with a recapitulation of the proto-algorit hm that we
introduced in §1.3. We discuss how it can be implemented in practice ( §4.1) and then
consider the question of how many random samples to acquire ( §4.2). Afterward, we

<!-- page 22 -->

22 HALKO, MARTINSSON, AND TROPP
Algorithm 4.1: Randomized Range Finder
Given an m × n matrix A, and an integer ℓ, this scheme computes an m × ℓ
orthonormal matrix Q whose range approximates the range of A.
1 Draw an n × ℓ Gaussian random matrix Ω.
2 Form the m × ℓ matrix Y = AΩ.
3 Construct an m × ℓ matrix Q whose columns form an orthonormal
basis for the range of Y , e.g., using the QR factorization Y = QR.
present several ways in which the basic scheme can be improved. Se ctions 4.3 and 4.4
explain how to address the situation where the numerical rank of th e input matrix is
not known in advance. Section 4.5 shows how to modify the scheme to improve its
accuracy when the singular spectrum of the input matrix decays slo wly. Finally, §4.6
describes how the scheme can be accelerated by using a structure d random matrix.
4.1. The proto-algorithm revisited. The most natural way to implement
the proto-algorithm from §1.3 is to draw a random test matrix Ω from the standard
Gaussian distribution. That is, each entry of Ω is an independent Gaussian random
variable with mean zero and variance one. For reference, we formu late the resulting
scheme as Algorithm 4.1.
The number Tbasic of ﬂops required by Algorithm 4.1 satisﬁes
Tbasic ∼ ℓn Trand + ℓ Tmult + ℓ2m (4.1)
where Trand is the cost of generating a Gaussian random number and Tmult is the cost
of multiplying A by a vector. The three terms in (4.1) correspond directly with the
three steps of Algorithm 4.1.
Empirically, we have found that the performance of Algorithm 4.1 dep ends very
little on the quality of the random number generator used in Step 1.
The actual cost of Step 2 depends substantially on the matrix A and the com-
putational environment that we are working in. The estimate (4.1) s uggests that
Algorithm 4.1 is especially eﬃcient when the matrix–vector product x ↦→ Ax can be
evaluated rapidly. In particular, the scheme is appropriate for app roximating sparse
or structured matrices. Turn to §6 for more details.
The most important implementation issue arises when performing the basis cal-
culation in Step 3. Typically, the columns of the sample matrix Y are almost linearly
dependent, so it is imperative to use stable methods for performing the orthonor-
malization. We have found that the Gram–Schmidt procedure, augm ented with the
double orthogonalization described in [12], is both convenient and reliable. Methods
based on Householder reﬂectors or Givens rotations also work ver y well. Note that
very little is gained by pivoting because the columns of the random mat rix Y are
independent samples drawn from the same distribution.
4.2. The number of samples required. The goal of Algorithm 4.1 is to pro-
duce an orthonormal matrix Q with few columns that achieves
∥(I − QQ∗ )A∥ ≤ ε, (4.2)
where ε is a speciﬁed tolerance. The number of columns ℓ that the algorithm needs to
reach this threshold is usually slightly larger than the minimal rank k of the smallest

<!-- page 23 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 23
basis that veriﬁes (4.2). We refer to this discrepancy p = ℓ − k as the oversampling
parameter. The size of the oversampling parameter depends on several fact ors:
The matrix dimensions. Very large matrices may require more oversampling.
The singular spectrum. The more rapid the decay of the singular values, the less
oversampling is needed. In the extreme case that the matrix has ex act rank
k, it is not necessary to oversample.
The random test matrix. Gaussian matrices succeed with very little oversampling,
but are not always the most cost-eﬀective option. The structure d random ma-
trices discussed in §4.6 may require substantial oversampling, but they still
yield computational gains in certain settings.
The theoretical results in Part III provide detailed information abo ut how the
behavior of randomized schemes depends on these factors. For t he moment, we limit
ourselves to some general remarks on implementation issues.
For Gaussian test matrices, it is adequate to choose the oversamp ling parameter
to be a small constant, such as p = 5 or p = 10. There is rarely any advantage to
select p > k . This observation, ﬁrst presented in [91], demonstrates that a Ga ussian
test matrix results in a negligible amount of extra computation.
In practice, the target rank k is rarely known in advance. Randomized algorithms
are usually implemented in an adaptive fashion where the number of sa mples is in-
creased until the error satisﬁes the desired tolerance. In other words, the user never
chooses the oversampling parameter. Theoretical results that bound the amount of
oversampling are valuable primarily as aids for designing algorithms. We develop an
adaptive approach in §§4.3–4.4.
The computational bottleneck in Algorithm 4.1 is usually the formation of the
product AΩ. As a result, it often pays to draw a larger number ℓ of samples than
necessary because the user can minimize the cost of the matrix mult iplication with
tools such as blocking of operations, high-level linear algebra subro utines, parallel
processors, etc. This approach may lead to an ill-conditioned sample matrix Y , but
the orthogonalization in Step 3 of Algorithm 4.1 can easily identify the n umerical
rank of the sample matrix and ignore the excess samples. Furtherm ore, Stage B of
the matrix approximation process succeeds even when the basis ma trix Q has a larger
dimension than necessary.
4.3. A posteriori error estimation. Algorithm 4.1 is designed for solving the
ﬁxed-rank problem, where the target rank of the input matrix is sp eciﬁed in advance.
To handle the ﬁxed-precision problem, where the parameter is the c omputational
tolerance, we need a scheme for estimating how well a putative basis matrix Q captures
the action of the matrix A. To do so, we develop a probabilistic error estimator. These
methods are inspired by work of Dixon [39]; our treatment follows [88, 137].
The exact approximation error is ∥(I − QQ∗ )A∥. It is intuitively plausible that
we can obtain some information about this quantity by computing ∥(I − QQ∗ )Aω∥,
where ω is a standard Gaussian vector. This notion leads to the following meth od.
Draw a sequence {ω(i) : i = 1 , 2, . . . , r } of standard Gaussian vectors, where r is a
small integer that balances computational cost and reliability. Then
∥(I − QQ∗ )A∥ ≤ 10
√
2
π max
i=1,...,r

(I − QQ∗ )Aω(i)
 (4.3)
with probability at least 1 − 10− r. This statement follows by setting B = (I− QQ∗ )A
and α = 10 in the following lemma, whose proof appears in [137, §3.4].

<!-- page 24 -->

24 HALKO, MARTINSSON, AND TROPP
Lemma 4.1. Let B be a real m × n matrix. Fix a positive integer r and a
real number α > 1. Draw an independent family {ω(i) : i = 1 , 2, . . . , r } of standard
Gaussian vectors. Then
∥B∥ ≤ α
√
2
π max
i=1,...,r

Bω (i)

except with probability α − r.
The critical point is that the error estimate (4.3) is computationally in expensive
because it requires only a small number of matrix–vector products . Therefore, we
can make a lowball guess for the numerical rank of A and add more samples if the
error estimate is too large. The asymptotic cost of Algorithm 4.1 is pr eserved if we
double our guess for the rank at each step. For example, we can st art with 32 samples,
compute another 32, then another 64, etc.
Remark 4.1. The estimate (4.3) is actually somewhat crude. We can obtain a
better estimate at a similar computational cost by initializing a power it eration with
a random vector and repeating the process several times [88].
4.4. Error estimation (almost) for free. The error estimate described in §4.3
can be combined with any method for constructing an approximate b asis for the range
of a matrix. In this section, we explain how the error estimator can b e incorporated
into Algorithm 4.1 at almost no additional cost.
To be precise, let us suppose that A is an m × n matrix and ε is a computational
tolerance. We seek an integer ℓ and an m × ℓ orthonormal matrix Q(ℓ) such that

(
I − Q(ℓ)(Q(ℓ))∗ )
A

 ≤ ε. (4.4)
The size ℓ of the basis will typically be slightly larger than the size k of the smallest
basis that achieves this error.
The basic observation behind the adaptive scheme is that we can gen erate the
basis in Step 3 of Algorithm 4.1 incrementally. Starting with an empty ba sis matrix
Q(0), the following scheme generates an orthonormal matrix whose ran ge captures
the action of A:
for i = 1, 2, 3, . . .
Draw an n × 1 Gaussian random vector ω(i), and set y(i) = Aω(i).
Compute ˜q(i) =
(
I − Q(i− 1)(Q(i− 1))∗ )
y(i).
Normalize q(i) = ˜q(i)/

 ˜q(i)
, and form Q(i) = [Q(i− 1) q(i)].
end for
How do we know when we have reached a basis Q(ℓ) that veriﬁes (4.4)? The answer
becomes apparent once we observe that the vectors ˜q(i) are precisely the vectors that
appear in the error bound (4.3). The resulting rule is that we break t he loop once we
observe r consecutive vectors ˜q(i) whose norms are smaller than ε/ (10
√
2/π ).
A formal description of the resulting algorithm appears as Algorithm 4.2. A
potential complication of the method is that the vectors ˜q(i) become small as the
basis starts to capture most of the action of A. In ﬁnite-precision arithmetic, their
direction is extremely unreliable. To address this problem, we simply re project the
normalized vector q(i) onto range( Q(i− 1))⊥ in steps 7 and 8 of Algorithm 4.2.
The CPU time requirements of Algorithms 4.2 and 4.1 are essentially iden tical.
Although Algorithm 4.2 computes the last few samples purely to obtain the error

<!-- page 25 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 25
estimate, this apparent extra cost is oﬀset by the fact that Algor ithm 4.1 always
includes an oversampling factor. The failure probability stated for A lgorithm 4.2 is
pessimistic because it is derived from a simple union bound argument. I n practice,
the error estimator is reliable in a range of circumstances when we ta ke r = 10.
Algorithm 4.2: Adaptive Randomized Range Finder
Given an m × n matrix A, a tolerance ε, and an integer r (e.g. r = 10), the
following scheme computes an orthonormal matrix Q such that (4.2) holds
with probability at least 1 − min{m, n }10− r.
1 Draw standard Gaussian vectors ω(1), . . . , ω(r) of length n.
2 For i = 1, 2, . . . , r , compute y(i) = Aω(i).
3 j = 0.
4 Q(0) = [ ], the m × 0 empty matrix.
5 while max
{
y(j+1)
,

y(j+2)
, . . . ,

y(j+r)

}
> ε/ (10
√
2/π ),
6 j = j + 1.
7 Overwrite y(j) by
(
I − Q(j− 1)(Q(j− 1))∗ )
y(j).
8 q(j) = y(j)/

y(j)
.
9 Q(j) = [Q(j− 1) q(j)].
10 Draw a standard Gaussian vector ω(j+r) of length n.
11 y(j+r) =
(
I − Q(j)(Q(j))∗ )
Aω(j+r).
12 for i = (j + 1), (j + 2), . . . , (j + r − 1),
13 Overwrite y(i) by y(i) − q(j) ⟨
q(j), y(i)⟩
.
14 end for
15 end while
16 Q = Q(j).
Remark 4.2. The calculations in Algorithm 4.2 can be organized so that each
iteration processes a block of samples simultaneously. This revision c an lead to dra-
matic improvements in speed because it allows us to exploit higher-leve l linear alge-
bra subroutines (e.g., BLAS3) or parallel processors. Although blo cking can lead to
the generation of unnecessary samples, this outcome is generally h armless, as noted
in §4.2.
4.5. A modiﬁed scheme for matrices whose singular values dec ay slowly .
The techniques described in §4.1 and §4.4 work well for matrices whose singular val-
ues exhibit some decay, but they may produce a poor basis when the input matrix
has a ﬂat singular spectrum or when the input matrix is very large. In this section,
we describe techniques, originally proposed in [67, 112], for improving the accuracy
of randomized algorithms in these situations. Related earlier work inc ludes [114] and
the literature on classical orthogonal iteration methods [61, p. 33 2].
The intuition behind these techniques is that the singular vectors as sociated with
small singular values interfere with the calculation, so we reduce the ir weight relative
to the dominant singular vectors by taking powers of the matrix to b e analyzed.
More precisely, we wish to apply the randomized sampling scheme to th e matrix
B = ( AA∗ )qA, where q is a small integer. The matrix B has the same singular

<!-- page 26 -->

26 HALKO, MARTINSSON, AND TROPP
Algorithm 4.3: Randomized Power Iteration
Given an m × n matrix A and integers ℓ and q, this algorithm computes an
m × ℓ orthonormal matrix Q whose range approximates the range of A.
1 Draw an n × ℓ Gaussian random matrix Ω.
2 Form the m × ℓ matrix Y = (AA∗ )qAΩ via alternating application
of A and A∗ .
3 Construct an m × ℓ matrix Q whose columns form an orthonormal
basis for the range of Y , e.g., via the QR factorization Y = QR.
Note: This procedure is vulnerable to round-oﬀ errors; see Remark 4.3. T he
recommended implementation appears as Algorithm 4.4.
vectors as the input matrix A, but its singular values decay much more quickly:
σj(B) = σj(A)2q+1, j = 1, 2, 3, . . . . (4.5)
We modify Algorithm 4.1 by replacing the formula Y = AΩ in Step 2 by the formula
Y = BΩ =
(
AA∗ )q
AΩ, and we obtain Algorithm 4.3.
Algorithm 4.3 requires 2 q + 1 times as many matrix–vector multiplies as Algo-
rithm 4.1, but is far more accurate in situations where the singular va lues of A decay
slowly. A good heuristic is that when the original scheme produces a b asis whose
approximation error is within a factor C of the optimum, the power scheme produces
an approximation error within C1/ (2q+1) of the optimum. In other words, the power
iteration drives the approximation gap to one exponentially fast. Se e Theorem 9.2
and §10.4 for the details.
Algorithm 4.3 targets the ﬁxed-rank problem. To address the ﬁxed -precision prob-
lem, we can incorporate the error estimators described in §4.3 to obtain an adaptive
scheme analogous with Algorithm 4.2. In situations where it is critical t o achieve near-
optimal approximation errors, one can increase the oversampling b eyond our standard
recommendation ℓ = k + 5 all the way to ℓ = 2 k without changing the scaling of the
asymptotic computational cost. A supporting analysis appears in C orollary 10.10.
Remark 4.3. Unfortunately, when Algorithm 4.3 is executed in ﬂoating point
arithmetic, rounding errors will extinguish all information pertaining to singular
modes associated with singular values that are small compared with ∥A∥. (Roughly,
if machine precision is µ , then all information associated with singular values smaller
than µ 1/ (2q+1) ∥A∥ is lost.) This problem can easily be remedied by orthonormaliz-
ing the columns of the sample matrix between each application of A and A∗ . The
resulting scheme, summarized as Algorithm 4.4, is algebraically equivale nt to Algo-
rithm 4.3 when executed in exact arithmetic [92, 124]. We recommend A lgorithm 4.4
because its computational costs are similar to Algorithm 4.3, even th ough the former
is substantially more accurate in ﬂoating-point arithmetic.
4.6. An accelerated technique for general dense matrices. This section
describes a set of techniques that allow us to compute an approxima te rank- ℓ factor-
ization of a general dense m × n matrix in roughly O( mn log(ℓ)) ﬂops, in contrast to
the asymptotic cost O( mnℓ) required by earlier methods. We can tailor this scheme
for the real or complex case, but we focus on the conceptually simp ler complex case.
These algorithms were introduced in [137]; similar techniques were pro posed in [118].

<!-- page 27 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 27
Algorithm 4.4: Randomized Subspace Iteration
Given an m × n matrix A and integers ℓ and q, this algorithm computes an
m × ℓ orthonormal matrix Q whose range approximates the range of A.
1 Draw an n × ℓ standard Gaussian matrix Ω.
2 Form Y0 = AΩ and compute its QR factorization Y0 = Q0R0.
3 for j = 1, 2, . . . , q
4 Form ˜Yj = A∗ Qj− 1 and compute its QR factorization ˜Yj = ˜Qj ˜Rj .
5 Form Yj = A ˜Qj and compute its QR factorization Yj = QjRj .
6 end
7 Q = Qq.
The ﬁrst step toward this accelerated technique is to observe tha t the bottleneck
in Algorithm 4.1 is the computation of the matrix product AΩ. When the test matrix
Ω is standard Gaussian, the cost of this multiplication is O( mnℓ), the same as a rank-
revealing QR algorithm [68]. The key idea is to use a structured random matrix that
allows us to compute the product in O( mn log(ℓ)) ﬂops.
The subsampled random Fourier transform , or SRFT, is perhaps the simplest
example of a structured random matrix that meets our goals. An SR FT is an n × ℓ
matrix of the form
Ω =
√ n
ℓ DF R, (4.6)
where
• D is an n× n diagonal matrix whose entries are independent random variables
uniformly distributed on the complex unit circle,
• F is the n × n unitary discrete Fourier transform (DFT), whose entries take
the values fpq = n− 1/ 2 e− 2π i(p− 1)(q− 1)/n for p, q = 1, 2, . . . , n , and
• R is an n × ℓ matrix that samples ℓ coordinates from n uniformly at random,
i.e., its ℓ columns are drawn randomly without replacement from the columns
of the n × n identity matrix.
When Ω is deﬁned by (4.6), we can compute the sample matrix Y = AΩ us-
ing O( mn log(ℓ)) ﬂops via a subsampled FFT [137]. Then we form the basis Q by
orthonormalizing the columns of Y , as described in §4.1. This scheme appears as
Algorithm 4.5. The total number Tstruct of ﬂops required by this procedure is
Tstruct ∼ mn log(ℓ) + ℓ2n (4.7)
Note that if ℓ is substantially larger than the numerical rank k of the input matrix,
we can perform the orthogonalization with O( kℓn) ﬂops because the columns of the
sample matrix are almost linearly dependent.
The test matrix (4.6) is just one choice among many possibilities. Othe r sugges-
tions that appear in the literature include subsampled Hadamard tra nsforms, chains
of Givens rotations acting on randomly chosen coordinates, and ma ny more. See [86]
and its bibliography. Empirically, we have found that the transform s ummarized in
Remark 4.6 below performs very well in a variety of environments [113 ].

<!-- page 28 -->

28 HALKO, MARTINSSON, AND TROPP
Algorithm 4.5: F ast Randomized Range Finder
Given an m × n matrix A, and an integer ℓ, this scheme computes an m × ℓ
orthonormal matrix Q whose range approximates the range of A.
1 Draw an n × ℓ SRFT test matrix Ω, as deﬁned by (4.6).
2 Form the m × ℓ matrix Y = AΩ using a (subsampled) FFT.
3 Construct an m × ℓ matrix Q whose columns form an orthonormal
basis for the range of Y , e.g., using the QR factorization Y = QR.
At this point, it is not well understood how to quantify and compare t he behav-
ior of structured random transforms. One reason for this uncer tainty is that it has
been diﬃcult to analyze the amount of oversampling that various tra nsforms require.
Section 11 establishes that the random matrix (4.6) can be used to id entify a near-
optimal basis for a rank- k matrix using ℓ ∼ (k + log(n)) log(k) samples. In practice,
the transforms (4.6) and (4.8) typically require no more oversamplin g than a Gaussian
test matrix requires. (For a numerical example, see §7.4.) As a consequence, setting
ℓ = k + 10 or ℓ = k + 20 is typically more than adequate. Further research on these
questions would be valuable.
Remark 4.4. The structured random matrices discussed in this section do not
adapt readily to the ﬁxed-precision problem, where the computatio nal tolerance is
speciﬁed, because the samples from the range are usually compute d in bulk. Fortu-
nately, these schemes are suﬃciently inexpensive that we can prog ressively increase
the number of samples computed starting with ℓ = 32, say, and then proceeding to
ℓ = 64, 128, 256, . . . until we achieve the desired tolerance.
Remark 4.5. When using the SRFT (4.6) for matrix approximation, we have
a choice whether to use a subsampled FFT or a full FFT. The complete FFT is so
inexpensive that it often pays to construct an extended sample ma trix Ylarge = ADF
and then generate the actual samples by drawing columns at rando m from Ylarge and
rescaling as needed. The asymptotic cost increases to O( mn log(n)) ﬂops, but the full
FFT is actually faster for moderate problem sizes because the cons tant suppressed by
the big-O notation is so small. Adaptive rank determination is easy bec ause we just
examine extra samples as needed.
Remark 4.6. Among the structured random matrices that we have tried, one of
the strongest candidates involves sequences of random Givens ro tations [113]. This
matrix takes the form
Ω = D′′Θ′D′Θ D F R , (4.8)
where the prime symbol ′ indicates an independent realization of a random matrix.
The matrices R, F , and D are deﬁned after (4.6). The matrix Θ is a chain of random
Givens rotations:
Θ = Π G(1, 2; θ1) G(2, 3; θ2) · · ·G(n − 1, n ; θn− 1)
where Π is a random n × n permutation matrix; where θ1, . . . , θ n− 1 are independent
random variables uniformly distributed on the interval [0 , 2π ]; and where G(i, j ; θ)
denotes a rotation on Cn by the angle θ in the ( i, j ) coordinate plane [61, §5.1.8].

<!-- page 29 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 29
Algorithm 5.1: Direct SVD
Given matrices A and Q such that (5.1) holds, this procedure computes an
approximate factorization A ≈ U ΣV ∗ , where U and V are orthonormal,
and Σ is a nonnegative diagonal matrix.
1 Form the matrix B = Q∗ A.
2 Compute an SVD of the small matrix: B = ˜U ΣV ∗ .
3 Form the orthonormal matrix U = Q ˜U .
Remark 4.7. When the singular values of the input matrix A decay slowly,
Algorithm 4.5 may perform poorly in terms of accuracy. When random ized sampling
is used with a Gaussian random matrix, the recourse is to take a coup le of steps of
a power iteration; see Algorithm 4.4. However, it is not currently kno wn whether
such an iterative scheme can be accelerated to O( mn log(k)) complexity using “fast”
random transforms such as the SRFT.
5. Stage B: Construction of standard factorizations. The algorithms for
Stage A described in §4 produce an orthonormal matrix Q whose range captures the
action of an input matrix A:
∥A − QQ∗ A∥ ≤ ε, (5.1)
where ε is a computational tolerance. This section describes methods for a pproximat-
ing standard factorizations of A using the information in the basis Q.
To accomplish this task, we pursue the idea from §3.3.3 that any low-rank factor-
ization A ≈ CB can be manipulated to produce a standard decomposition. When
the bound (5.1) holds, the low-rank factors are simply C = Q and B = Q∗ A. The
simplest scheme ( §5.1) computes the factor B directly with a matrix–matrix product
to ensure a minimal error in the ﬁnal approximation. An alternative a pproach (§5.2)
constructs factors B and C without forming any matrix–matrix product. The ap-
proach of §5.2 is often faster than the approach of §5.1 but typically results in larger
errors. Both schemes can be streamlined for an Hermitian input mat rix ( §5.3) and a
positive semideﬁnite input matrix ( §5.4). Finally, we develop single-pass algorithms
that exploit other information generated in Stage A to avoid revisitin g the input
matrix ( §5.5).
Throughout this section, A denotes an m × n matrix, and Q is an m × k or-
thonormal matrix that veriﬁes (5.1). For purposes of exposition, we concentrate on
methods for constructing the partial SVD.
5.1. F actorizations based on forming Q∗ A directly . The relation (5.1) im-
plies that ∥A − QB∥ ≤ ε, where B = Q∗ A. Once we have computed B, we can
produce any standard factorization using the methods of §3.3.3. Algorithm 5.1 illus-
trates how to build an approximate SVD.
The factors produced by Algorithm 5.1 satisfy
∥A − U ΣV ∗ ∥ ≤ ε. (5.2)
In other words, the approximation error does not degrade.
The cost of Algorithm 5.1 is generally dominated by the cost of the pro duct Q∗ A
in Step 1, which takes O( kmn) ﬂops for a general dense matrix. Note that this scheme

<!-- page 30 -->

30 HALKO, MARTINSSON, AND TROPP
is particularly well suited to environments where we have a fast meth od for computing
the matrix–vector product x ↦→ A∗ x, for example when A is sparse or structured.
This approach retains a strong advantage over Krylov-subspace methods and rank-
revealing QR because Step 1 can be accelerated using BLAS3, paralle l processors, and
so forth. Steps 2 and 3 require O( k2n) and O( k2m) ﬂops respectively.
Remark 5.1. Algorithm 5.1 produces an approximate SVD with the same rank
as the basis matrix Q. When the size of the basis exceeds the desired rank k of the
SVD, it may be preferable to retain only the dominant k singular values and singular
vectors. Equivalently, we replace the diagonal matrix Σ of computed singular values
with the matrix Σ(k) formed by zeroing out all but the largest k entries of Σ. In
the worst case, this truncation step can increase the approximat ion error by σk+1;
see §9.4 for an analysis. Our numerical experience suggests that this er ror analysis is
pessimistic, and the term σk+1 often does not appear in practice.
5.2. Postprocessing via row extraction. Given a matrix Q such that (5.1)
holds, we can obtain a rank- k factorization
A ≈ XB , (5.3)
where B is a k × n matrix consisting of k rows extracted from A. The approxima-
tion (5.3) can be produced without computing any matrix–matrix pro ducts, which
makes this approach to postprocessing very fast. The drawback comes because the er-
ror ∥A − XB ∥ is usually larger than the initial error ∥A − QQ∗ A∥, especially when
the dimensions of A are large. See Remark 5.3 for more discussion.
To obtain the factorization (5.3), we simply construct the interpola tive decompo-
sition ( §3.2.3) of the matrix Q:
Q = XQ (J, : ) . (5.4)
The index set J marks k rows of Q that span the row space of Q, and X is an m × k
matrix whose entries are bounded in magnitude by two and contains t he k × k identity
as a submatrix: X(J, : ) = Ik. Combining (5.4) and (5.1), we reach
A ≈ QQ∗ A = XQ (J, : ) Q∗ A. (5.5)
Since X(J, : ) = Ik, equation (5.5) implies that A(J, : ) ≈ Q(J, : ) Q∗ A. Therefore, (5.3)
follows when we put B = A(J, : ) .
Provided with the factorization (5.3), we can obtain any standard f actorization
using the techniques of §3.3.3. Algorithm 5.2 illustrates an SVD calculation. This
procedure requires O( k2(m + n)) ﬂops. The following lemma guarantees the accuracy
of the computed factors.
Lemma 5.1. Let A be an m × n matrix and let Q be an m × k matrix that satisfy
(5.1). Suppose that U , Σ, and V are the matrices constructed by Algorithm 5.2. Then
∥A − U ΣV ∗ ∥ ≤
[
1 +
√
1 + 4k(n − k)
]
ε. (5.6)
Proof. The factors U , Σ, V constructed by the algorithm satisfy
U ΣV ∗ = U Σ ˜V ∗ W ∗ = ZW ∗ = XR ∗ W ∗ = XA (J, : ) .

<!-- page 31 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 31
Algorithm 5.2: SVD via Row Extraction
Given matrices A and Q such that (5.1) holds, this procedure computes an
approximate factorization A ≈ U ΣV ∗ , where U and V are orthonormal,
and Σ is a nonnegative diagonal matrix.
1 Compute an ID Q = XQ (J, : ) . (The ID is deﬁned in §3.2.3.)
2 Extract A(J, : ) , and compute a QR factorization A(J, : ) = R∗ W ∗ .
3 Form the product Z = XR ∗ .
4 Compute an SVD Z = U Σ ˜V ∗ .
5 Form the orthonormal matrix V = W ˜V .
Note: Algorithm 5.2 is faster than Algorithm 5.1 but less accurate.
Note: It is advantageous to replace the basis Q by the sample matrix Y
produced in Stage A, cf. Remark 5.2.
Deﬁne the approximation
ˆA = QQ∗ A. (5.7)
Since ˆA = XQ (J, : ) Q∗ A and since X(J, : ) = Ik, it must be that ˆA(J, : ) = Q(J, : ) Q∗ A.
Consequently,
ˆA = X ˆA(J, : ) .
We have the chain of relations
∥A − U ΣV ∗ ∥ =

A − XA (J, : )


=

(
A − X ˆA(J, : )
)
+
(
X ˆA(J, : ) − XA (J, : )
)

≤

A − ˆA

 +

X ˆA(J, : ) − XA (J, : )


≤

A − ˆA

 + ∥X∥

A(J, : ) − ˆA(J, : )

. (5.8)
Inequality (5.1) ensures that

A − ˆA

 ≤ ε. Since A(J, : ) − ˆA(J, : ) is a submatrix of
A − ˆA, we must also have

A(J, : ) − ˆA(J, : )

 ≤ ε. Thus, (5.8) reduces to
∥A − U ΣV ∗ ∥ ≤ (1 + ∥X∥) ε. (5.9)
The bound (5.6) follows from (5.9) after we observe that X contains a k × k identity
matrix and that the entries of the remaining ( n − k) × k submatrix are bounded in
magnitude by two.
Remark 5.2. To maintain a uniﬁed presentation, we have formulated all the
postprocessing techniques so they take an orthonormal matrix Q as input. Recall
that, in Stage A of our framework, we construct the matrix Q by orthonormalizing
the columns of the sample matrix Y . With ﬁnite-precision arithmetic, it is preferable
to adapt Algorithm 5.2 to start directly from the sample matrix Y . To be precise,
we modify Step 1 to compute X and J so that Y = XY (J, :). This revision is
recommended even when Q is available from the adaptive rank determination of
Algorithm 4.2.

<!-- page 32 -->

32 HALKO, MARTINSSON, AND TROPP
Algorithm 5.3: Direct Eigenvalue Decomposition
Given an Hermitian matrix A and a basis Q such that (5.1) holds, this
procedure computes an approximate eigenvalue decompositi on A ≈ U ΛU ∗ ,
where U is orthonormal, and Λ is a real diagonal matrix.
1 Form the small matrix B = Q∗ AQ.
2 Compute an eigenvalue decomposition B = V ΛV ∗ .
3 Form the orthonormal matrix U = QV .
Remark 5.3. As the inequality (5.6) suggests, the factorization produced by
Algorithm 5.2 is potentially less accurate than the basis that it uses as input. This
loss of accuracy is problematic when ε is not so small or when kn is large. In such
cases, we recommend Algorithm 5.1 over Algorithm 5.2; the former is m ore costly,
but it does not amplify the error, as shown in (5.2).
5.3. Postprocessing an Hermitian matrix. When A is Hermitian, the post-
processing becomes particularly elegant. In this case, the columns of Q form a
good basis for both the column space and the row space of A so that we have
A ≈ QQ∗ AQQ∗ . More precisely, when (5.1) is in force, we have
∥A − QQ∗ AQQ∗ ∥ = ∥A − QQ∗ A + QQ∗ A − QQ∗ AQQ∗ ∥
≤ ∥ A − QQ∗ A∥ +

QQ∗ (
A − AQQ∗ )
 ≤ 2ε. (5.10)
The last inequality relies on the facts that ∥QQ∗ ∥ = 1 and that
∥A − AQQ∗ ∥ =

(A − AQQ∗ )∗ 
 = ∥A − QQ∗ A∥ .
Since A ≈ Q
(
Q∗ AQ
)
Q∗ is a low-rank approximation of A, we can form any standard
factorization using the techniques from §3.3.3.
For Hermitian A, it is more common to compute an eigenvalue decomposition
than an SVD. We can accomplish this goal using Algorithm 5.3, which ada pts the
scheme from §5.1. This procedure delivers a factorization that satisﬁes the erro r
bound ∥A − U ΛU ∗ ∥ ≤ 2ε. The calculation requires O( kn2) ﬂops.
We can also pursue the row extraction approach from §5.2, which is faster but
less accurate. See Algorithm 5.4 for the details. The total cost is O( k2n) ﬂops.
5.4. Postprocessing a positive semideﬁnite matrix. When the input ma-
trix A is positive semideﬁnite, the Nystr¨ om methodcan be used to improve the quality
of standard factorizations at almost no additional cost; see [48] a nd its bibliography.
To describe the main idea, we ﬁrst recall that the direct method pre sented in §5.3
manipulates the approximate rank- k factorization
A ≈ Q
(
Q∗ AQ
)
Q∗ . (5.11)
In contrast, the Nystr¨ om scheme builds a more sophisticated ran k-k approximation,
namely
A ≈ (AQ)
(
Q∗ AQ
)− 1
(AQ)∗
=
[
(AQ)
(
Q∗ AQ
)− 1/ 2] [
(AQ)
(
Q∗ AQ
)− 1/ 2]∗
= F F ∗ , (5.12)

<!-- page 33 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 33
Algorithm 5.4: Eigenvalue Decomposition via Row Extractio n
Given an Hermitian matrix A and a basis Q such that (5.1) holds, this
procedure computes an approximate eigenvalue decompositi on A ≈ U ΛU ∗ ,
where U is orthonormal, and Λ is a real diagonal matrix.
1 Compute an ID Q = XQ (J, : ) .
2 Perform a QR factorization X = V R.
3 Form the product Z = RA(J,J )R∗ .
4 Compute an eigenvalue decomposition Z = W ΛW ∗ .
5 Form the orthonormal matrix U = V W .
Note: Algorithm 5.4 is faster than Algorithm 5.3 but less accurate.
Note: It is advantageous to replace the basis Q by the sample matrix Y
produced in Stage A, cf. Remark 5.2.
Algorithm 5.5: Eigenvalue Decomposition via Nystr ¨om Method
Given a positive semideﬁnite matrix A and a basis Q such that (5.1)
holds, this procedure computes an approximate eigenvalue d ecomposition
A ≈ U ΛU ∗ , where U is orthonormal, and Λ is nonnegative and diagonal.
1 Form the matrices B1 = AQ and B2 = Q∗ B1.
2 Perform a Cholesky factorization B2 = C ∗ C.
3 Form F = B1C − 1 using a triangular solve.
4 Compute an SVD F = U ΣV ∗ and set Λ = Σ2.
where F is an approximate Cholesky factor of A with dimension n × k. To compute
the factor F numerically, ﬁrst form the matrices B1 = AQ and B2 = Q∗ B1. Then
decompose the psd matrix B2 = C ∗ C into its Cholesky factors. Finally compute the
factor F = B1C − 1 by performing a triangular solve. The low-rank factorization (5.12)
can be converted to a standard decomposition using the technique s from §3.3.3.
The literature contains an explicit expression [48, Lem. 4] for the ap proximation
error in (5.12). This result implies that, in the spectral norm, the Ny str¨ om approx-
imation error never exceeds ∥A − QQ∗ A∥, and it is often substantially smaller. We
omit a detailed discussion.
For an example of the Nystr¨ om technique, consider Algorithm 5.5, w hich com-
putes an approximate eigenvalue decomposition of a positive semideﬁ nite matrix. This
method should be compared with the scheme for Hermitian matrices, Algorithm 5.3.
In both cases, the dominant cost occurs when we form AQ, so the two procedures
have roughly the same running time. On the other hand, Algorithm 5.5 is typically
much more accurate than Algorithm 5.3. In a sense, we are exploiting the fact that
A is positive semideﬁnite to take one step of subspace iteration (Algor ithm 4.4) for
free.
5.5. Single-pass algorithms. The techniques described in §§5.1–5.4 all require
us to revisit the input matrix. This may not be feasible in environments where the
matrix is too large to be stored. In this section, we develop a method that requires
just one pass over the matrix to construct not only an approximat e basis but also a
complete factorization. Similar techniques appear in [137] and [29].

<!-- page 34 -->

34 HALKO, MARTINSSON, AND TROPP
For motivation, we begin with the case where A is Hermitian. Let us recall the
proto-algorithm from §1.3.3: Draw a random test matrix Ω; form the sample matrix
Y = AΩ; then construct a basis Q for the range of Y . It turns out that the matrices
Ω, Y , and Q contain all the information we need to approximate A.
To see why, deﬁne the (currently unknown) matrix B via B = Q∗ AQ. Postmul-
tiplying the deﬁnition by Q∗ Ω, we obtain the identity BQ∗ Ω = Q∗ AQQ∗ Ω. The
relationships AQQ∗ ≈ A and AΩ = Y show that B must satisfy
BQ∗ Ω ≈ Q∗ Y . (5.13)
All three matrices Ω, Y , and Q are available, so we can solve (5.13) to obtain the
matrix B. Then the low-rank factorization A ≈ QBQ∗ can be converted to an eigen-
value decomposition via familiar techniques. The entire procedure re quires O( k2n)
ﬂops, and it is summarized as Algorithm 5.6.
Algorithm 5.6: Eigenvalue Decomposition in One Pass
Given an Hermitian matrix A, a random test matrix Ω, a sample matrix Y =
AΩ, and an orthonormal matrix Q that veriﬁes (5.1) and Y = QQ∗ Y , this
algorithm computes an approximate eigenvalue decompositi on A ≈ U ΛU ∗ .
1 Use a standard least-squares solver to ﬁnd an Hermitian matrix Bapprox
that approximately satisﬁes the equation Bapprox(Q∗ Ω) ≈ Q∗ Y .
2 Compute the eigenvalue decomposition Bapprox = V ΛV ∗ .
3 Form the product U = QV .
When A is not Hermitian, it is still possible to devise single-pass algorithms, but
we must modify the initial Stage A of the approximation framework to simultaneously
construct bases for the ranges of A and A∗ :
1. Generate random matrices Ω and ˜Ω.
2. Compute Y = AΩ and ˜Y = A∗ ˜Ω in a single pass over A.
3. Compute QR factorizations Y = QR and ˜Y = ˜Q ˜R.
This procedure results in matrices Q and ˜Q such that A ≈ QQ∗ A ˜Q ˜Q∗ . The reduced
matrix we must approximate is B = Q∗ A ˜Q. In analogy with (5.13), we ﬁnd that
Q∗ Y = Q∗ AΩ ≈ Q∗ A ˜Q ˜Q∗ Ω = B ˜Q∗ Ω. (5.14)
An analogous calculation shows that B should also satisfy
˜Q∗ ˜Y ≈ B∗ Q∗ ˜Ω. (5.15)
Now, the reduced matrix Bapprox can be determined by ﬁnding a minimum-residual
solution to the system of relations (5.14) and (5.15).
Remark 5.4. The single-pass approaches described in this section can degrade
the approximation error in the ﬁnal decomposition signiﬁcantly. To e xplain the issue,
we focus on the Hermitian case. It turns out that the coeﬃcient ma trix Q∗ Ω in
the linear system (5.13) is usually ill-conditioned. In a worst-case sce nario, the error
∥A − U ΛU ∗ ∥ in the factorization produced by Algorithm 5.6 could be larger than

<!-- page 35 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 35
the error resulting from the two-pass method of Section 5.3 by a fa ctor of 1 /τ min,
where τmin is the minimal singular value of the matrix Q∗ Ω.
The situation can be improved by oversampling. Suppose that we see k a rank- k
approximate eigenvalue decomposition. Pick a small oversampling par ameter p. Draw
an n × (k + p) random matrix Ω, and form the sample matrix Y = AΩ. Let Q denote
the n × k matrix formed by the k leading left singular vectors of Y . Now, the linear
system (5.13) has a coeﬃcient matrix Q∗ Ω of size k × (k + p), so it is overdetermined.
An approximate solution of this system yields a k × k matrix B.
6. Computational costs. So far, we have postponed a detailed discussion of
the computational cost of randomized matrix approximation algorit hms because it is
necessary to account for both the ﬁrst stage, where we comput e an approximate basis
for the range ( §4), and the second stage, where we postprocess the basis to com plete
the factorization ( §5). We are now prepared to compare the cost of the two-stage
scheme with the cost of traditional techniques.
Choosing an appropriate algorithm, whether classical or randomize d, requires us
to consider the properties of the input matrix. To draw a nuanced p icture, we discuss
three representative computational environments in §6.1–6.3. We close with some
comments on parallel implementations in §6.4.
For concreteness, we focus on the problem of computing an appro ximate SVD of
an m × n matrix A with numerical rank k. The costs for other factorizations are
similar.
6.1. General matrices that ﬁt in core memory . Suppose that A is a general
matrix presented as an array of numbers that ﬁts in core memory. In this case, the
appropriate method for Stage A is to use a structured random mat rix ( §4.6), which
allows us to ﬁnd a basis that captures the action of the matrix using O (mn log(k) +
k2m) ﬂops. For Stage B, we apply the row-extraction technique ( §5.2), which costs
an additional O( k2(m + n)) ﬂops. The total number of operations Trandom for this
approach satisﬁes
Trandom ∼ mn log(k) + k2(m + n).
As a rule of thumb, the approximation error of this procedure satis ﬁes
∥A − U ΣV ∗ ∥ ≲ n ·σk+1, (6.1)
where σk+1 is the ( k + 1)th singular value of A. The estimate (6.1), which follows
from Theorem 11.2 and Lemma 5.1, reﬂects the worst-case scenar io; actual errors are
usually smaller.
This algorithm should be compared with modern deterministic techniqu es, such
as rank-revealing QR followed by postprocessing ( §3.3.2) which typically require
TRRQR ∼ kmn
operations to achieve a comparable error.
In this setting, the randomized algorithm can be several times fast er than classical
techniques even for problems of moderate size, say m, n ∼ 103 and k ∼ 102. See §7.4
for numerical evidence.
Remark 6.1. In case row extraction is impractical, there is an alternative
O(mn log(k)) technique described in [137, §5.2]. When the error (6.1) is unacceptably
large, we can use the direct method ( §5.1) for Stage B, which brings the total cost to
O(kmn) ﬂops.

<!-- page 36 -->

36 HALKO, MARTINSSON, AND TROPP
6.2. Matrices for which matrix–vector products can be rapid ly evalu-
ated. In many problems in data mining and scientiﬁc computing, the cost Tmult of
performing the matrix–vector multiplication x ↦→ Ax is substantially smaller than
the nominal cost O( mn) for the dense case. It is not uncommon that O( m + n) ﬂops
suﬃce. Standard examples include (i) very sparse matrices; (ii) str uctured matrices,
such as T¨ oplitz operators, that can be applied using the FFT or oth er means; and
(iii) matrices that arise from physical problems, such as discretized integral operators,
that can be applied via, e.g., the fast multipole method [66].
Suppose that both A and A∗ admit fast multiplies. The appropriate randomized
approach for this scenario completes Stage A using Algorithm 4.1 with p constant (for
the ﬁxed-rank problem) or Algorithm 4.2 (for the ﬁxed-precision pr oblem) at a cost
of (k + p) Tmult + O(k2m) ﬂops. For Stage B, we invoke Algorithm 5.1, which requires
(k + p) Tmult + O(k2(m + n)) ﬂops. The total cost Tsparse satisﬁes
Tsparse = 2 (k + p) Tmult + O(k2(m + n)). (6.2)
As a rule of thumb, the approximation error of this procedure satis ﬁes
∥A − U ΣV ∗ ∥ ≲
√
kn ·σk+1. (6.3)
The estimate (6.3) follows from Corollary 10.9 and the discussion in §5.1. Actual
errors are usually smaller.
When the singular spectrum of A decays slowly, we can incorporate q iterations
of the power method (Algorithm 4.3) to obtain superior solutions to t he ﬁxed-rank
problem. The computational cost increases to, cf. (6.2),
Tsparse = (2q + 2) (k + p) Tmult + O(k2(m + n)), (6.4)
while the error (6.3) improves to
∥A − U ΣV ∗ ∥ ≲ (kn)1/ 2(2q+1) ·σk+1. (6.5)
The estimate (6.5) takes into account the discussion in §10.4. The power scheme can
also be adapted for the ﬁxed-precision problem ( §4.5).
In this setting, the classical prescription for obtaining a partial SV D is some vari-
ation of a Krylov-subspace method; see §3.3.4. These methods exhibit great diversity,
so it is hard to specify a “typical” computational cost. To a ﬁrst app roximation, it is
fair to say that in order to obtain an approximate SVD of rank k, the cost of a numer-
ically stable implementation of a Krylov method is no less than the cost ( 6.2) with p
set to zero. At this price, the Krylov method often obtains better accuracy than the
basic randomized method obtained by combining Algorithms 4.1 and 5.1, especially
for matrices whose singular values decay slowly. On the other hand, the randomized
schemes are inherently more robust and allow much more freedom in o rganizing the
computation to suit a particular application or a particular hardware architecture.
The latter point is in practice of crucial importance because it is usua lly much faster
to apply a matrix to k vectors simultaneously than it is to execute k matrix–vector
multiplications consecutively. In practice, blocking and parallelism can lead to enough
gain that a few steps of the power method (Algorithm 4.3) can be per formed more
quickly than k steps of a Krylov method.
Remark 6.2. Any comparison between randomized sampling schemes and Krylov
variants becomes complicated because of the fact that “basic” Kr ylov schemes such

<!-- page 37 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 37
as Lanczos [61, p. 473] or Arnoldi [61, p. 499] are inherently unsta ble. To obtain nu-
merical robustness, we must incorporate sophisticated modiﬁcat ions such as restarts,
reorthogonalization procedures, etc. Constructing a high-qualit y implementation is
suﬃciently hard that the authors of a popular book on “numerical r ecipes” qualify
their treatment of spectral computations as follows [109, p. 567]:
You have probably gathered by now that the solution of eigensyste ms
is a fairly complicated business. It is. It is one of the few subjects
covered in this book for which we do not recommend that you avoid
canned routines. On the contrary, the purpose of this chapter is
precisely to give you some appreciation of what is going on inside
such canned routines, so that you can make intelligent choices abou t
using them, and intelligent diagnoses when something goes wrong.
Randomized sampling does not eliminate the diﬃculties referred to in th is quotation;
however it reduces the task of computing a partial spectral decomposition of a very
large matrix to the task of computing a full decomposition of a small dense matrix.
(For example, in Algorithm 5.1, the input matrix A is large and B is small.) The latter
task is much better understood and is eminently suitable for using ca nned routines.
Random sampling schemes interact with the large matrix only through matrix–matrix
products, which can easily be implemented by a user in a manner appro priate to the
application and to the available hardware.
The comparison is further complicated by the fact that there is sign iﬁcant overlap
between the two sets of ideas. Algorithm 4.3 is conceptually similar to a “block
Lanczos method” [61, p. 485] with a random starting matrix. Indee d, we believe that
there are signiﬁcant opportunities for cross-fertilization in this ar ea. Hybrid schemes
that combine the best ideas from both ﬁelds may perform very well.
6.3. General matrices stored in slow memory or streamed. The tradi-
tional metric for numerical algorithms is the number of ﬂoating-poin t operations they
require. When the data does not ﬁt in fast memory, however, the c omputational time
is often dominated by the cost of memory access. In this setting, a more appropri-
ate measure of algorithmic performance is pass-eﬃciency, which counts how many
times the data needs to be cycled through fast memory. Flop count s become largely
irrelevant.
All the classical matrix factorization techniques that we discuss in §3.2—including
dense SVD, rank-revealing QR, Krylov methods, and so forth—req uire at least k
passes over the the matrix, which is prohibitively expensive for huge data matrices. A
desire to reduce the pass count of matrix approximation algorithms served as one of
the early motivations for developing randomized schemes [46, 58, 10 5]. Detailed recent
work appears in [29].
For many matrices, randomized techniques can produce an accura te approxima-
tion using just one pass over the data. For Hermitian matrices, we o btain a single-pass
algorithm by combining Algorithm 4.1, which constructs an approximat e basis, with
Algorithm 5.6, which produces an eigenvalue decomposition without an y additional
access to the matrix. Section 5.5 describes the analogous techniqu e for general ma-
trices.
For the huge matrices that arise in applications such as data mining, it is com-
mon that the singular spectrum decays slowly. Relevant applications include image
processing (see §§7.2–7.3 for numerical examples), statistical data analysis, and net -
work monitoring. To compute approximate factorizations in these e nvironments, it is

<!-- page 38 -->

38 HALKO, MARTINSSON, AND TROPP
crucial to enhance the accuracy of the randomized approach usin g the power scheme,
Algorithm 4.3, or some other device. This approach increases the pa ss count some-
what, but in our experience it is very rare that more than ﬁve passe s are required.
6.4. Gains from parallelization. As mentioned in §§6.2–6.3, randomized meth-
ods often outperform classical techniques not because they invo lve fewer ﬂoating-point
operations but rather because they allow us to reorganize the calc ulations to exploit
the matrix properties and the computer architecture more fully. I n addition, these
methods are well suited for parallel implementation. For example, in A lgorithm 4.1,
the computational bottleneck is the evaluation of the matrix produ ct AΩ, which is
embarrassingly parallelizable.
7. Numerical examples. By this time, the reader has surely formulated a
pointed question: Do these randomized matrix approximation algorit hms actually
work in practice? In this section, we attempt to address this conce rn by illustrating
how the algorithms perform on a diverse collection of test cases.
Section 7.1 starts with two examples from the physical sciences invo lving discrete
approximations to operators with exponentially decaying spectra. Sections 7.2 and
7.3 continue with two examples of matrices arising in “data mining.” Thes e are
large matrices whose singular spectra decay slowly; one is sparse an d ﬁts in RAM,
one is dense and is stored out-of-core. Finally, §7.4 investigates the performance of
randomized methods based on structured random matrices.
Sections 7.1–7.3 focus on the algorithms for Stage A that we presen ted in §4
because we wish to isolate the performance of the randomized step .
Computational examples illustrating truly large data matrices have b een reported
elsewhere, for instance in [69].
7.1. Two matrices with rapidly decaying singular values. We ﬁrst illus-
trate the behavior of the adaptive range approximation method, A lgorithm 4.2. We
apply it to two matrices associated with the numerical analysis of diﬀe rential and
integral operators. The matrices in question have rapidly decaying singular values
and our intent is to demonstrate that in this environment, the appr oximation error
of a bare-bones randomized method such as Algorithm 4.2 is very close to the mini-
mal error achievable by any method. We observe that the approxim ation error of a
randomized method is itself a random variable (it is a function of the ra ndom matrix
Ω) so what we need to demonstrate is not only that the error is small in a typical
realization, but also that it clusters tightly around the mean value.
We ﬁrst consider a 200 × 200 matrix A that results from discretizing the following
single-layer operator associated with the Laplace equation:
[Sσ ](x) = const ·
∫
Γ1
log |x − y| σ (y) dA(y), x ∈ Γ2, (7.1)
where Γ 1 and Γ 2 are the two contours in R2 illustrated in Figure 7.1(a). We ap-
proximate the integral with the trapezoidal rule, which converges superalgebraically
because the kernel is smooth. In the absence of ﬂoating-point er rors, we estimate that
the discretization error would be less than 10 − 20 for a smooth source σ . The leading
constant is selected so the matrix A has unit operator norm.
We implement Algorithm 4.2 in Matlab v6.5. Gaussian test matrices are ge nerated
using the randn command. For each number ℓ of samples, we compare the following
three quantities:

<!-- page 39 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 39
−3 −2 −1 0 1 2 3
−2.5
−2
−1.5
−1
−0.5
0
0.5
1
1.5
2
2.5
Γ1
Γ2
(a) (b)
Fig. 7.1 . Conﬁgurations for physical problems. (a) The contours Γ1 (red) and Γ2 (blue) for
the integral operator (7.1). (b) Geometry of the lattice problem associated with matrix B in §7.1.
1. The minimum rank- ℓ approximation error σℓ+1 is determined using svd.
2. The actual error eℓ =

(
I − Q(ℓ)(Q(ℓ))∗ )
A

 is computed with norm.
3. A random estimator fℓ for the actual error eℓ is obtained from (4.3), with the
parameter r set to 5.
Note that any values less than 10 − 15 should be considered numerical artifacts.
Figure 7.2 tracks a characteristic execution of Algorithm 4.2. We mak e three
observations: (i) The error eℓ incurred by the algorithm is remarkably close to the
theoretical minimum σℓ+1. (ii) The error estimate always produces an upper bound
for the actual error. Without the built-in 10 × safety margin, the estimate would
track the actual error almost exactly. (iii) The basis constructed by the algorithm
essentially reaches full double-precision accuracy.
How typical is the trial documented in Figure 7.2? To answer this ques tion, we
examine the empirical performance of the algorithm over 2000 indep endent trials.
Figure 7.3 charts the error estimate versus the actual error at f our points during the
course of execution: ℓ = 25 , 50, 75, 100. We oﬀer four observations: (i) The initial
run detailed in Figure 7.2 is entirely typical. (ii) Both the actual and est imated error
concentrate about their mean value. (iii) The actual error drifts s lowly away from the
optimal error as the number ℓ of samples increases. (iv) The error estimator is always
pessimistic by a factor of about ten, which means that the algorithm never produces
a basis with lower accuracy than requested. The only eﬀect of selec ting an unlucky
sample matrix Ω is that the algorithm proceeds for a few additional steps.
We next consider a matrix B which is deﬁned implicitly in the sense that we
cannot access its elements directly; we can only evaluate the map x ↦→ Bx for a given
vector x. To be precise, B represents a transfer matrix for a network of resistors like
the one shown in Figure 7.1(b). The vector x represents a set of electric potentials
speciﬁed on the red nodes in the ﬁgure. These potentials induce a un ique equilibrium
ﬁeld on the network in which the potential of each black and blue node is the average
of the potentials of its three or four neighbors. The vector Bx is then the restriction
of the potential to the blue exterior nodes. Given a vector x, the vector Bx can
be obtained by solving a large sparse linear system whose coeﬃcient m atrix is the
classical ﬁve-point stencil approximating the 2D Laplace operator .

<!-- page 40 -->

40 HALKO, MARTINSSON, AND TROPP
0 50 100 150
−18
−16
−14
−12
−10
−8
−6
−4
−2
0
2


ℓ
log10(fℓ )
log10(eℓ )
log10(σ ℓ +1)
Approximation errors
Order of magnitude
Fig. 7.2 . Approximating a Laplace integral operator. One execution of Algorithm 4.2 for the
200 × 200 input matrix A described in §7.1. The number ℓ of random samples varies along the
horizontal axis; the vertical axis measures the base-10 log arithm of error magnitudes. The dashed
vertical lines mark the points during execution at which Fig ure 7.3 provides additional statistics.
−5.5 −5 −4.5
−5
−4.5
−4
−3.5
−3
−9.5 −9 −8.5
−9
−8.5
−8
−7.5
−7
−13.5 −13 −12.5
−13
−12.5
−12
−11.5
−11
−16 −15.5 −15
−15.5
−15
−14.5
−14
−13.5
log10(eℓ)
log10(fℓ)
“y = x”
Minimal
error
ℓ = 25 ℓ = 50
ℓ = 75 ℓ = 100
Fig. 7.3 . Error statistics for approximating a Laplace integral ope rator. 2,000 trials of Al-
gorithm 4.2 applied to a 200 × 200 matrix approximating the integral operator (7.1). The panels
isolate the moments at which ℓ = 25, 50, 75, 100 random samples have been drawn. Each solid point
compares the estimated error fℓ versus the actual error eℓ in one trial; the open circle indicates the
trial detailed in Figure 7.2. The dashed line identiﬁes the m inimal error σ ℓ+1, and the solid line
marks the contour where the error estimator would equal the a ctual error.

<!-- page 41 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 41
We applied Algorithm 4.2 to the 1596 × 532 matrix B associated with a lattice
in which there were 532 nodes (red) on the “inner ring” and 1596 nod es on the (blue)
“outer ring.” Each application of B to a vector requires the solution of a sparse linear
system of size roughly 140 000 × 140 000. We implemented the scheme in Matlab using
the “backslash” operator for the linear solve. The results of a typ ical trial appear in
Figure 7.4. Qualitatively, the performance matches the results in Fig ure 7.3.
0 50 100 150
−20
−18
−16
−14
−12
−10
−8
−6
−4
−2
0


ℓ
log10(fℓ )
log10(eℓ )
log10(σ ℓ +1)
Approximation errors
Order of magnitude
Fig. 7.4 . Approximating the inverse of a discrete Laplacian. One execution of Algorithm 4.2
for the 1596 × 532 input matrix B described in §7.1. See Figure 7.2 for notations.
7.2. A large, sparse, noisy matrix arising in image processi ng. Our next
example involves a matrix that arises in image processing. A recent line of work uses
information about the local geometry of an image to develop promisin g new algorithms
for standard tasks, such as denoising, inpainting, and so forth. T hese methods are
based on approximating a graph Laplacian associated with the image. The dominant
eigenvectors of this matrix provide “coordinates” that help us smo oth out noisy image
patches [120, 131].
We begin with a 95 × 95 pixel grayscale image. The intensity of each pixel is
represented as an integer in the range 0 to 4095. We form for each pixel i a vector
x(i) ∈ R25 by gathering the 25 intensities of the pixels in a 5 × 5 neighborhood
centered at pixel i (with appropriate modiﬁcations near the edges). Next, we form
the 9025 × 9025 weight matrix ˜W that reﬂects the similarities between patches:
˜wij = exp
{
−

x(i) − x(j)
2
/σ 2}
,
where the parameter σ = 50 controls the level of sensitivity. We obtain a sparse
weight matrix W by zeroing out all entries in ˜W except the seven largest ones in
each row. The object is then to construct the low frequency eigen vectors of the graph

<!-- page 42 -->

42 HALKO, MARTINSSON, AND TROPP
Laplacian matrix
L = I − D− 1/ 2W D− 1/ 2,
where D is the diagonal matrix with entries dii = ∑
j wij. These are the eigenvectors
associated with the dominant eigenvalues of the auxiliary matrix A = D− 1/ 2W D− 1/ 2.
The matrix A is large, and its eigenvalues decay slowly, so we use the power
scheme summarized in Algorithm 4.3 to approximate it. Figure 7.5[left] illu strates
how the approximation error eℓ declines as the number ℓ of samples increases. When
we set the exponent q = 0, which corresponds with the basic Algorithm 4.1, the
approximation is rather poor. The graph illustrates that increasing the exponent q
slightly results in a tremendous improvement in the accuracy of the p ower scheme.
Next, we illustrate the results of using the two-stage approach to approximate the
eigenvalues of A. In Stage A, we construct a basis for A using Algorithm 4.3 with
ℓ = 100 samples for diﬀerent values of q. In Stage B, we apply the Hermitian variant of
Algorithm 5.1 described in §5.3 to compute an approximate eigenvalue decomposition.
Figure 7.5[right] shows the approximate eigenvalues and the actual eigenvalues of A.
Once again, we see that the minimal exponent q = 0 produces miserable results, but
the largest eigenvalues are quite accurate even for q = 1.
0 20 40 60 80 100
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
0 20 40 60 80 100
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1


ℓ j
Approximation error eℓ Estimated Eigenvalues λ j
Magnitude
“Exact” eigenvalues
λ j for q = 3
λ j for q = 2
λ j for q = 1
λ j for q = 0
Fig. 7.5 . Approximating a graph Laplacian. For varying exponent q, one trial of the power
scheme, Algorithm 4.3, applied to the 9025 × 9025 matrix A described in §7.2. [Left] Approximation
errors as a function of the number ℓ of random samples. [Right] Estimates for the 100 largest
eigenvalues given ℓ = 100 random samples compared with the 100 largest eigenvalues of A.
7.3. Eigenfaces. Our next example involves a large, dense matrix derived from
the FERET databank of face images [107, 108]. A simple method for pe rforming face
recognition is to identify the principal directions of the image data, w hich are called
eigenfaces. Each of the original photographs can be summarized by its compon ents

<!-- page 43 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 43
along these principal directions. To identify the subject in a new pict ure, we compute
its decomposition in this basis and use a classiﬁcation technique, such as nearest
neighbors, to select the closest image in the database [122].
We construct a data matrix A as follows: The FERET database contains 7254
images, and each 384 × 256 image contains 98 304 pixels. First, we build a 98 304 × 7254
matrix ˜A whose columns are the images. We form A by centering each column of
˜A and scaling it to unit norm, so that the images are roughly comparable . The
eigenfaces are the dominant left singular vectors of this matrix.
Our goal then is to compute an approximate SVD of the matrix A. Represented
as an array of double-precision real numbers, A would require 5. 4 GB of storage, which
does not ﬁt within the fast memory of many machines. It is possible to compress the
database down to at 57 MB or less (in JPEG format), but then the da ta would have
to be uncompressed with each sweep over the matrix. Furthermor e, the matrix A has
slowly decaying singular values, so we need to use the power scheme, Algorithm 4.3,
to capture the range of the matrix accurately.
To address these concerns, we implemented the power scheme to r un in a pass-
eﬃcient manner. An additional diﬃculty arises because the size of th e data makes
it expensive to calculate the actual error eℓ incurred by the approximation or to
determine the minimal error σℓ+1. To estimate the errors, we use the technique
described in Remark 4.1.
Figure 7.6 describes the behavior of the power scheme, which is similar to its
performance for the graph Laplacian in §7.2. When the exponent q = 0, the ap-
proximation of the data matrix is very poor, but it improves quickly as q increases.
Likewise, the estimate for the spectrum of A appears to converge rapidly; the largest
singular values are already quite accurate when q = 1. We see essentially no improve-
ment in the estimates after the ﬁrst 3–5 passes over the matrix.
7.4. Performance of structured random matrices. Our ﬁnal set of experi-
ments illustrates that the structured random matrices described in §4.6 lead to matrix
approximation algorithms that are both fast and accurate.
First, we compare the computational speeds of four methods for computing an
approximation to the ℓ dominant terms in the SVD of an n × n matrix A. For now,
we are interested in execution time only (not accuracy), so the cho ice of matrix is
irrelevant and we have selected A to be a Gaussian matrix. The four methods are
summarized in the following table; Remark 7.1 provides more details on t he imple-
mentation.
Method Stage A Stage B
direct Rank-revealing QR executed using column Algorithm 5.1
pivoting and Householder reﬂectors
gauss Algorithm 4.1 with a Gaussian random matrix Algorithm 5.1
srft Algorithm 4.1 with the modiﬁed SRFT (4.8) Algorithm 5.2
svd Full SVD with LAPACK routine dgesdd Truncate to ℓ terms
Table 7.1 lists the measured runtime of a single execution of each algor ithm for
various choices of the dimension n of the input matrix and the rank ℓ of the ap-
proximation. Of course, the cost of the full SVD does not depend o n the number ℓ
of components required. A more informative way to look at the runt ime data is to
compare the relative cost of the algorithms. The direct method is the best determin-

<!-- page 44 -->

44 HALKO, MARTINSSON, AND TROPP
0 20 40 60 80 100
10
0
10
1
10
2


0 20 40 60 80 100
10
0
10
1
10
2Approximation error eℓ Estimated Singular Values σ j
Magnitude
Minimal error (est)
q = 0
q = 1
q = 2
q = 3
ℓ j
Fig. 7.6 . Computing eigenfaces. For varying exponent q, one trial of the power scheme,
Algorithm 4.3, applied to the 98 304× 7254 matrix A described in §7.3. (Left) Approximation errors
as a function of the number ℓ of random samples. The red line indicates the minimal errors as
estimated by the singular values computed using ℓ = 100 and q = 3 . (Right) Estimates for the 100
largest eigenvalues given ℓ = 100 random samples.
istic approach for dense matrices, so we calculate the factor by wh ich the randomized
methods improve on this benchmark. Figure 7.7 displays the results. We make two
observations: (i) Using an SRFT often leads to a dramatic speed-up over classical
techniques, even for moderate problem sizes. (ii) Using a standard Gaussian test ma-
trix typically leads to a moderate speed-up over classical methods, primarily because
performing a matrix–matrix multiplication is faster than a QR factoriz ation.
Second, we investigate how the choice of random test matrix inﬂuen ces the error
in approximating an input matrix. For these experiments, we return to the 200 × 200
matrix A deﬁned in Section 7.1. Consider variations of Algorithm 4.1 obtained wh en
the random test matrix Ω is drawn from the following four distributions:
Gauss: The standard Gaussian distribution.
Ortho: The uniform distribution on n × ℓ orthonormal matrices.
SRFT: The SRFT distribution deﬁned in (4.6).
GSRFT: The modiﬁed SRFT distribution deﬁned in (4.8).
Intuitively, we expect that Ortho should provide the best performance.
For each distribution, we perform 100 000 trials of the following expe riment. Ap-
ply the corresponding version of Algorithm 4.1 to the matrix A, and calculate the
approximation error eℓ = ∥A − QℓQ∗
ℓ A∥. Figure 7.8 displays the empirical proba-
bility density function for the error eℓ obtained with each algorithm. We oﬀer three
observations: (i) The SRFT actually performs slightly better than a Gaussian ran-

<!-- page 45 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 45
Table 7.1
Computational times for a partial SVD. The time, in seconds, required to compute the ℓ
leading components in the SVD of an n × n matrix using each of the methods from §7.4. The last
row indicates the time needed to obtain a full SVD.
n = 1024 n = 2048 n = 4096
ℓ direct gauss srft direct gauss srft direct gauss srft
10 1.08e-1 5.63e-2 9.06e-2 4.22e-1 2.16e-1 3.56e-1 1.70e 0 8.94e-1 1.45e 0
20 1.97e-1 9.69e-2 1.03e-1 7.67e-1 3.69e-1 3.89e-1 3.07e 0 1.44e 0 1.53e 0
40 3.91e-1 1.84e-1 1.27e-1 1.50e 0 6.69e-1 4.33e-1 6.03e 0 2.64e 0 1.63e 0
80 7.84e-1 4.00e-1 2.19e-1 3.04e 0 1.43e 0 6.64e-1 1.20e 1 5.43e 0 2.08e 0
160 1.70e 0 9.92e-1 6.92e-1 6.36e 0 3.36e 0 1.61e 0 2.46e 1 1.16e 1 3.94e 0
320 3.89e 0 2.65e 0 2.98e 0 1.34e 1 7.45e 0 5.87e 0 5.00e 1 2.41e 1 1.21e 1
640 1.03e 1 8.75e 0 1.81e 1 3.14e 1 2.13e 1 2.99e 1 1.06e 2 5.80e 1 5.35e 1
1280 — — — 7.97e 1 6.69e 1 3.13e 2 2.40e 2 1.68e 2 4.03e 2
svd 1.19e 1 8.77e 1 6.90e 2
dom matrix for this example. (ii) The standard SRFT and the modiﬁed S RFT have
essentially identical errors. (iii) There is almost no diﬀerence betwee n the Gaussian
random matrix and the random orthonormal matrix in the ﬁrst thre e plots, while the
fourth plot shows that the random orthonormal matrix performs better. This behav-
ior occurs because, with high probability, a tall Gaussian matrix is well conditioned
and a (nearly) square Gaussian matrix is not.
Remark 7.1. The running times reported in Table 7.1 and in Figure 7.7 depend
strongly on both the computer hardware and the coding of the algo rithms. The ex-
periments reported here were performed on a standard oﬃce des ktop with a 3.2 GHz
Pentium IV processor and 2 GB of RAM. The algorithms were implement ed in For-
tran 90 and compiled with the Lahey compiler. The Lahey versions of B LAS and
LAPACK were used to accelerate all matrix–matrix multiplications, as well as the
SVD computations in Algorithms 5.1 and 5.2. We used the code for the m odiﬁed
SRFT (4.8) provided in the publicly available software package id dist [90].
Part III: Theory
This part of the paper, §§8–11, provides a detailed analysis of randomized sam-
pling schemes for constructing an approximate basis for the range of a matrix, the task
we refer to as Stage A in the framework of §1.2. More precisely, we assess the qual-
ity of the basis Q that the proto-algorithm of §1.3 produces by establishing rigorous
bounds for the approximation error
| | |A − QQ∗ A| | |, (7.2)
where | | |·| | |denotes either the spectral norm or the Frobenius norm. The diﬃc ulty in
developing these bounds is that the matrix Q is random, and its distribution is a
complicated nonlinear function of the input matrix A and the random test matrix Ω.
Naturally, any estimate for the approximation error must depend o n the properties
of the input matrix and the distribution of the test matrix.
To address these challenges, we split the argument into two pieces. The ﬁrst part
exploits techniques from linear algebra to deliver a generic error bou nd that depends
on the interaction between the test matrix Ω and the right singular vectors of the
input matrix A, as well as the tail singular values of A. In the second part of the
argument, we take into account the distribution of the random mat rix to estimate
the error for speciﬁc instantiations of the proto-algorithm. This b ipartite proof is

<!-- page 46 -->

46 HALKO, MARTINSSON, AND TROPP
10
1
10
2
10
30
1
2
3
4
5
6
7
10
1
10
2
10
30
1
2
3
4
5
6
7
10
1
10
2
10
30
1
2
3
4
5
6
7
ℓ ℓ ℓ
n = 1024 n = 2048 n = 4096
t(direct) /t (gauss)
t(direct) /t (srft)
t(direct) /t (svd)
Acceleration factor
Fig. 7.7 . Acceleration factor. The relative cost of computing an ℓ-term partial SVD of an n × n
Gaussian matrix using direct, a benchmark classical algorithm, versus each of the three c ompetitors
described in §7.4. The solid red curve shows the speedup using an SRFT test m atrix, and the dotted
blue curve shows the speedup with a Gaussian test matrix. The dashed green curve indicates that a
full SVD computation using classical methods is substantia lly slower. Table 7.1 reports the absolute
runtimes that yield the circled data points.
common in the literature on randomized linear algebra, but our argum ent is most
similar in spirit to [17].
Section 8 surveys the basic linear algebraic tools we need. Section 9 u ses these
methods to derive a generic error bound. Afterward, we specialize these results to the
case where the test matrix is Gaussian ( §10) and the case where the test matrix is a
subsampled random Fourier transform ( §11).
8. Theoretical preliminaries. We proceed with some additional background
from linear algebra. Section 8.1 sets out properties of positive-sem ideﬁnite matrices,
and §8.2 oﬀers some results for orthogonal projectors. Standard re ferences for this
material include [11, 72].
8.1. Positive semideﬁnite matrices. An Hermitian matrix M is positive
semideﬁnite (brieﬂy, psd) when u∗ M u ≥ 0 for all u ̸= 0. If the inequalities are
strict, M is positive deﬁnite (brieﬂy, pd). The psd matrices form a convex cone,
which induces a partial ordering on the linear space of Hermitian matr ices: M ≼ N
if and only if N − M is psd. This ordering allows us to write M ≽ 0 to indicate that
the matrix M is psd.

<!-- page 47 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 47
0 0.5 1 1.5
x 10
−4
0
2
4
6
8
10
x 10
4
0 0.5 1 1.5 2
x 10
−8
0
2
4
6
8
10
12
x 10
8
0 0.5 1 1.5
x 10
−12
0
2
4
6
8
10
12
x 10
12
2 4 6
x 10
−15
0
0.5
1
1.5
2
x 10
16
Empirical density
e25 e50
e75 e100
ℓ = 25
ℓ = 75
ℓ = 50
ℓ = 100
Gauss
Ortho
SRFT
GSRFT
Fig. 7.8 . Empirical probability density functions for the error in A lgorithm 4.1. As described
in §7.4, the algorithm is implemented with four distributions f or the random test matrix and used
to approximate the 200 × 200 input matrix obtained by discretizing the integral operato r (7.1). The
four panels capture the empirical error distribution for ea ch version of the algorithm at the moment
when ℓ = 25, 50, 75, 100 random samples have been drawn.
Alternatively, we can deﬁne a psd (resp., pd) matrix as an Hermitian m atrix with
nonnegative (resp., positive) eigenvalues. In particular, each psd matrix is diagonal-
izable, and the inverse of a pd matrix is also pd. The spectral norm of a psd matrix
M has the variational characterization
∥M ∥ = max
u̸=0
u∗ M u
u∗u , (8.1)
according to the Rayleigh–Ritz theorem [72, Thm. 4.2.2]. It follows tha t
M ≼ N =⇒ ∥ M ∥ ≤ ∥ N ∥ . (8.2)
A fundamental fact is that conjugation preserves the psd prope rty.
Proposition 8.1 ( Conjugation Rule ). Suppose that M ≽ 0. For every A, the
matrix A∗ M A ≽ 0. In particular,
M ≼ N =⇒ A∗ M A ≼ A∗ N A.
Our argument invokes the conjugation rule repeatedly. As a ﬁrst a pplication, we
establish a perturbation bound for the matrix inverse near the iden tity matrix.

<!-- page 48 -->

48 HALKO, MARTINSSON, AND TROPP
Proposition 8.2 ( Perturbation of Inverses ). Suppose that M ≽ 0. Then
I − (I + M )− 1 ≼ M
Proof. Deﬁne R = M 1/ 2, the psd square root of M promised by [72, Thm. 7.2.6].
We have the chain of relations
I − (I + R2)− 1 = (I + R2)− 1R2 = R(I + R2)− 1R ≼ R2.
The ﬁrst equality can be veriﬁed algebraically. The second holds beca use rational
functions of a diagonalizable matrix, such as R, commute. The last relation follows
from the conjugation rule because ( I + R2)− 1 ≼ I.
Next, we present a generalization of the fact that the spectral n orm of a psd
matrix is controlled by its trace.
Proposition 8.3. We have ∥M ∥ ≤ ∥ A∥ + ∥C∥ for each partitioned psd matrix
M =
[ A B
B∗ C
]
.
Proof. The variational characterization (8.1) of the spectral norm implie s that
∥M ∥ = sup
∥x∥2+∥y∥2=1
[x
y
]∗ [ A B
B∗ C
] [x
y
]
≤ sup
∥x∥2+∥y∥2=1
(
∥A∥ ∥x∥2 + 2 ∥B∥ ∥x∥ ∥y∥ + ∥C∥ ∥y∥2)
.
The block generalization of Hadamard’s psd criterion [72, Thm. 7.7.7] s tates that
∥B∥2 ≤ ∥ A∥ ∥C∥. Thus,
∥M ∥ ≤ sup
∥x∥2+∥y∥2=1
(
∥A∥1/ 2 ∥x∥ + ∥C∥1/ 2 ∥y∥
)2
= ∥A∥ + ∥C∥ .
This point completes the argument.
8.2. Orthogonal projectors. An orthogonal projector is an Hermitian matrix
P that satisﬁes the polynomial P 2 = P . This identity implies 0 ≼ P ≼ I. An
orthogonal projector is completely determined by its range. For a given matrix M ,
we write PM for the unique orthogonal projector with range( PM ) = range( M ).
When M has full column rank, we can express this projector explicitly:
PM = M (M ∗ M )− 1M ∗ . (8.3)
The orthogonal projector onto the complementary subspace, r ange(P )⊥ , is the matrix
I − P . Our argument hinges on several other facts about orthogonal projectors.
Proposition 8.4. Suppose U is unitary. Then U ∗ PM U = PU ∗ M .
Proof. Abbreviate P = U ∗ PM U . It is clear that P is an orthogonal projector
since it is Hermitian and P 2 = P . Evidently,
range(P ) = U ∗ range(M ) = range( U ∗ M ).

<!-- page 49 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 49
Since the range determines the orthogonal projector, we conclu de P = PU ∗ M .
Proposition 8.5. Suppose range(N ) ⊂ range(M ). Then, for each matrix A, it
holds that ∥PN A∥ ≤ ∥ PM A∥ and that ∥(I − PM )A∥ ≤ ∥ (I − PN )A∥.
Proof. The projector PN ≼ I, so the conjugation rule yields PM PN PM ≼ PM .
The hypothesis range( N ) ⊂ range(M ) implies that PM PN = PN , which results in
PM PN PM = PN PM = (PM PN )∗ = PN .
In summary, PN ≼ PM . The conjugation rule shows that A∗ PN A ≼ A∗ PM A. We
conclude from (8.2) that
∥PN A∥2 = ∥A∗ PN A∥ ≤ ∥ A∗ PM A∥ = ∥PM A∥2 .
The second statement follows from the ﬁrst by taking orthogonal complements.
Finally, we need a generalization of the scalar inequality |px|q ≤ | p| |x|q, which
holds when |p| ≤ 1 and q ≥ 1.
Proposition 8.6. Let P be an orthogonal projector, and let M be a matrix. For
each positive number q,
∥P M∥ ≤ ∥ P (M M∗ )qM ∥1/ (2q+1) . (8.4)
Proof. Suppose that R is an orthogonal projector, D is a nonnegative diagonal
matrix, and t ≥ 1. We claim that
∥RDR∥t ≤

RDtR

 . (8.5)
Granted this inequality, we quickly complete the proof. Using an SVD M = U ΣV ∗ ,
we compute
∥P M∥2(2q+1) = ∥P M M∗ P ∥2q+1 =

(U ∗ P U) ·Σ2 ·(U ∗ P U)

2q+1
≤

(U ∗ P U) ·Σ2(2q+1) ·(U ∗ P U)

 =

P (M M∗ )2(2q+1)P


=

P (M M∗ )qM ·M ∗ (M M∗ )qP

 = ∥P (M M∗ )qM ∥2 .
We have used the unitary invariance of the spectral norm in the sec ond and fourth
relations. The inequality (8.5) applies because U ∗ P U is an orthogonal projector.
Take a square root to ﬁnish the argument.
Now, we turn to the claim (8.5). This relation follows immediately from [11 ,
Thm. IX.2.10], but we oﬀer a direct argument based on more elementa ry considera-
tions. Let x be a unit vector at which
x∗ (RDR)x = ∥RDR∥ .
We must have Rx = x. Otherwise, ∥Rx∥ < 1 because R is an orthogonal projector,
which implies that the unit vector y = Rx/ ∥Rx∥ veriﬁes
y∗ (RDR)y = (Rx)∗ (RDR)(Rx)
∥Rx∥2 = x∗ (RDR)x
∥Rx∥2 > x∗ (RDR)x.

<!-- page 50 -->

50 HALKO, MARTINSSON, AND TROPP
Writing xj for the entries of x and dj for the diagonal entries of D, we ﬁnd that
∥RDR∥t = [x∗ (RDR)x]t = [x∗ Dx]t =
[∑
j
dj x2
j
]t
≤
[∑
j
dt
jx2
j
]
= x∗ Dtx = (Rx)∗ Dt(Rx) ≤

RDtR

 .
The inequality is Jensen’s, which applies because ∑x2
j = 1 and the function z ↦→ | z|t
is convex for t ≥ 1.
9. Error bounds via linear algebra. We are now prepared to develop a de-
terministic error analysis for the proto-algorithm described in §1.3. To begin, we
must introduce some notation. Afterward, we establish the key er ror bound, which
strengthens a result from the literature [17, Lem. 4.2]. Finally, we ex plain why the
power method can be used to improve the performance of the prot o-algorithm.
9.1. Setup. Let A be an m × n matrix that has a singular value decomposition
A = U ΣV ∗ , as described in Section 3.2.2. Roughly speaking, the proto-algorith m
tries to approximate the subspace spanned by the ﬁrst k left singular vectors, where
k is now a ﬁxed number. To perform the analysis, it is appropriate to pa rtition the
singular value decomposition as follows.
k n − k n
A = U
[
Σ1
Σ2
] [
V ∗
1
V ∗
2
]
k
n − k
(9.1)
The matrices Σ1 and Σ2 are square. We will see that the left unitary factor U does
not play a signiﬁcant role in the analysis.
Let Ω be an n× ℓ test matrix, where ℓ denotes the number of samples. We assume
only that ℓ ≥ k. Decompose the test matrix in the coordinate system determined b y
the right unitary factor of A:
Ω1 = V ∗
1 Ω and Ω2 = V ∗
2 Ω. (9.2)
The error bound for the proto-algorithm depends critically on the p roperties of the
matrices Ω1 and Ω2. With this notation, the sample matrix Y can be expressed as
ℓ
Y = AΩ = U
[ Σ1Ω1
Σ2Ω2
] k
n − k
It is a useful intuition that the block Σ1Ω1 in (9.1) reﬂects the gross behavior of A,
while the block Σ2Ω2 represents a perturbation.
9.2. A deterministic error bound for the proto-algorithm. The proto-
algorithm constructs an orthonormal basis Q for the range of the sample matrix Y ,
and our goal is to quantify how well this basis captures the action of the input A.
Since QQ∗ = PY , the challenge is to obtain bounds on the approximation error
| | |A − QQ∗ A| | |= | | |(I − PY )A| | |.
The following theorem shows that the behavior of the proto-algorit hm depends on
the interaction between the test matrix and the right singular vect ors of the input
matrix, as well as the singular spectrum of the input matrix.

<!-- page 51 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 51
Theorem 9.1 ( Deterministic error bound ). Let A be an m × n matrix with
singular value decomposition A = U ΣV ∗ , and ﬁx k ≥ 0. Choose a test matrix Ω,
and construct the sample matrix Y = AΩ. Partition Σ as speciﬁed in (9.1), and
deﬁne Ω1 and Ω2 via (9.2). Assuming that Ω1 has full row rank, the approximation
error satisﬁes
| | |(I − PY )A| | |2 ≤ | | |Σ2| | |2 +
⏐
⏐⏐
⏐⏐
⏐Σ2Ω2Ω†
1
⏐
⏐⏐
⏐⏐
⏐2
, (9.3)
where | | |·| | |denotes either the spectral norm or the Frobenius norm.
Theorem 9.1 sharpens the result [17, Lem. 2], which lacks the square s present
in (9.3). This reﬁnement yields slightly better error estimates than t he earlier bound,
and it has consequences for the probabilistic behavior of the error when the test
matrix Ω is random. The proof here is diﬀerent in spirit from the earlier analysis ; our
argument is inspired by the perturbation theory of orthogonal pr ojectors [125].
Proof. We establish the bound for the spectral-norm error. The bound f or the
Frobenius-norm error follows from an analogous argument that is s lightly easier.
Let us begin with some preliminary simpliﬁcations. First, we argue that the left
unitary factor U plays no essential role in the argument. In eﬀect, we execute the
proof for an auxiliary input matrix ˜A and an associated sample matrix ˜Y deﬁned by
˜A = U ∗ A =
[
Σ1V ∗
1
Σ2V ∗
2
]
and ˜Y = ˜AΩ =
[
Σ1Ω1
Σ2Ω2
]
. (9.4)
Owing to the unitary invariance of the spectral norm and to Propos ition 8.4, we have
the identity
∥(I − PY )A∥ =

U ∗ (I − PY )U ˜A

 =

(I − PU ∗ Y ) ˜A

 =

(I − P ˜Y ) ˜A

. (9.5)
In view of (9.5), it suﬃces to prove that

(I − P ˜Y ) ˜A

 ≤

Σ2

2
+

Σ2Ω2Ω†
1

2
. (9.6)
Second, we assume that the number k is chosen so the diagonal entries of Σ1 are
strictly positive. Suppose not. Then Σ2 is zero because of the ordering of the singular
values. As a consequence,
range( ˜A) = range
[
Σ1V ∗
1
0
]
= range
[
Σ1Ω1
0
]
= range( ˜Y ).
This calculation uses the decompositions presented in (9.4), as well a s the fact that
both V ∗
1 and Ω1 have full row rank. We conclude that

(I − P ˜Y ) ˜A

 = 0,
so the error bound (9.6) holds trivially. (In fact, both sides are zer o.)
The main argument is based on ideas from perturbation theory. To illu strate the
concept, we start with a matrix related to ˜Y :
ℓ
W =
[ Σ1Ω1
0
] k
n − k

<!-- page 52 -->

52 HALKO, MARTINSSON, AND TROPP
The matrix W has the same range as a related matrix formed by “ﬂattening out” t he
spectrum of the top block. Indeed, since Σ1Ω1 has full row rank,
k
range(W ) = range
[ I
0
] k
n − k
The matrix on the right-hand side has full column rank, so it is legal to apply the
formula (8.3) for an orthogonal projector, which immediately yields
PW =
[ I 0
0 0
]
and I − PW =
[0 0
0 I
]
. (9.7)
In words, the range of W aligns with the ﬁrst k coordinates, which span the same
subspace as the ﬁrst k left singular vectors of the auxiliary input matrix ˜A. Therefore,
range(W ) captures the action of ˜A, which is what we wanted from range( ˜Y ).
We treat the auxiliary sample matrix ˜Y as a perturbation of W , and we hope
that their ranges are close to each other. To make the comparison rigorous, let us
emulate the arguments outlined in the last paragraph. Referring to the display (9.4),
we ﬂatten out the top block of ˜Y to obtain the matrix
Z = ˜Y ·Ω†
1Σ− 1
1 =
[ I
F
]
where F = Σ2Ω2Ω†
1Σ− 1
1 . (9.8)
Let us return to the error bound (9.6). The construction (9.8) en sures that
range(Z) ⊂ range( ˜Y ), so Proposition 8.5 implies that the error satisﬁes

(I − P ˜Y ) ˜A

 ≤

(I − PZ) ˜A

.
Squaring this relation, we obtain

(I − P ˜Y ) ˜A

2
≤

(I − PZ) ˜A

2
=

 ˜A∗ (I − PZ ) ˜A

 = ∥Σ∗ (I − PZ)Σ∥ . (9.9)
The last identity follows from the deﬁnition ˜A = ΣV ∗ and the unitary invariance
of the spectral norm. Therefore, we can complete the proof of ( 9.6) by producing a
suitable bound for the right-hand side of (9.9).
To continue, we need a detailed representation of the projector I − PZ . The con-
struction (9.8) ensures that Z has full column rank, so we can apply the formula (8.3)
for an orthogonal projector to see that
PZ = Z(Z ∗ Z)− 1Z ∗ =
[ I
F
]
(I + F ∗ F )− 1
[ I
F
]∗
.
Expanding this expression, we determine that the complementary p rojector satisﬁes
I − PZ =
[I − (I + F ∗ F )− 1 − (I + F ∗ F )− 1F ∗
− F (I + F ∗ F )− 1 I − F (I + F ∗ F )− 1F ∗
]
. (9.10)
The partitioning here conforms with the partitioning of Σ. When we conjugate the
matrix by Σ, copies of Σ− 1
1 , presently hidden in the top-left block, will cancel to
happy eﬀect.

<!-- page 53 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 53
The latter point may not seem obvious, owing to the complicated form of (9.10).
In reality, the block matrix is less fearsome than it looks. Proposition 8.2, on the
perturbation of inverses, shows that the top-left block veriﬁes
I − (I + F ∗ F )− 1 ≼ F ∗ F .
The bottom-right block satisﬁes
I − F (I + F ∗ F )− 1F ∗ ≼ I
because the conjugation rule guarantees that F (I + F ∗ F )− 1F ∗ ≽ 0. We abbreviate
the oﬀ-diagonal blocks with the symbol B = − (I + F ∗ F )− 1F ∗ . In summary,
I − PZ ≼
[
F ∗ F B
B∗ I
]
.
This relation exposes the key structural properties of the proje ctor. Compare this
relation with the expression (9.7) for the “ideal” projector I − PW .
Moving toward the estimate required by (9.9), we conjugate the las t relation by
Σ to obtain
Σ∗ (I − PZ )Σ ≼
[
Σ∗
1F ∗ F Σ1 Σ∗
1BΣ2
Σ∗
2B∗ Σ1 Σ∗
2Σ2
]
.
The conjugation rule demonstrates that the matrix on the left-ha nd side is psd, so
the matrix on the right-hand side is too. Proposition 8.3 results in the norm bound
∥Σ∗ (I − PZ)Σ∥ ≤ ∥ Σ∗
1F ∗ F Σ1∥ + ∥Σ∗
2Σ2∥ = ∥F Σ1∥2 + ∥Σ2∥2 .
Recall that F = Σ2Ω2Ω†
1Σ− 1
1 , so the factor Σ1 cancels neatly. Therefore,
∥Σ∗ (I − PZ )Σ∥ ≤

Σ2Ω2Ω†
1

2
+ ∥Σ2∥2 .
Finally, introduce the latter inequality into (9.9) to complete the proo f.
9.3. Analysis of the power scheme. Theorem 9.1 suggests that the perfor-
mance of the proto-algorithm depends strongly on the relationship between the large
singular values of A listed in Σ1 and the small singular values listed in Σ2. When a
substantial proportion of the mass of A appears in the small singular values, the con-
structed basis Q may have low accuracy. Conversely, when the large singular values
dominate, it is much easier to identify a good low-rank basis.
To improve the performance of the proto-algorithm, we can run it w ith a closely
related input matrix whose singular values decay more rapidly [67, 112 ]. Fix a positive
integer q, and set
B = (AA∗ )qA = U Σ2q+1V ∗ .
We apply the proto-algorithm to B, which generates a sample matrix Z = BΩ and
constructs a basis Q for the range of Z. Section 4.5 elaborates on the implementation
details, and describes a reformulation that sometimes improves the accuracy when
the scheme is executed in ﬁnite-precision arithmetic. The following re sult describes
how well we can approximate the original matrix A within the range of Z.

<!-- page 54 -->

54 HALKO, MARTINSSON, AND TROPP
Theorem 9.2 ( Power scheme). Let A be an m × n matrix, and let Ω be an n × ℓ
matrix. Fix a nonnegative integer q, form B = ( A∗ A)qA, and compute the sample
matrix Z = BΩ. Then
∥(I − PZ)A∥ ≤ ∥ (I − PZ)B∥1/ (2q+1) .
Proof. We determine that
∥(I − PZ)A∥ ≤ ∥ (I − PZ )(AA∗ )qA∥1/ (2q+1) = ∥(I − PZ)B∥1/ (2q+1)
as a direct consequence of Proposition 8.6.
Let us illustrate how the power scheme interacts with the main error bound (9.3).
Let σk+1 denote the ( k + 1)th singular value of A. First, suppose we approximate A
in the range of the sample matrix Y = AΩ. Since ∥Σ2∥ = σk+1, Theorem 9.1 implies
that
∥(I − PY )A∥ ≤
(
1 +

Ω2Ω†
1

2)1/ 2
σk+1. (9.11)
Now, deﬁne B = ( AA∗ )qA, and suppose we approximate A within the range of the
sample matrix Z = BΩ. Together, Theorem 9.2 and Theorem 9.1 imply that
∥(I − PZ)A∥ ≤ ∥ (I − PZ )B∥1/ (2q+1) ≤
(
1 +

Ω2Ω†
1

2)1/ (4q+2)
σk+1
because σ 2q+1
k+1 is the ( k + 1)th singular value of B. In eﬀect, the power scheme
drives down the suboptimality of the bound (9.11) exponentially fast as the power
q increases. In principle, we can make the extra factor as close to on e as we like,
although this increases the cost of the algorithm.
9.4. Analysis of truncated SVD. Finally, let us study the truncated SVD
described in Remark 5.1. Suppose that we approximate the input mat rix A inside
the range of the sample matrix Z. In essence, the truncation step computes a best
rank-k approximation ˆA(k) of the compressed matrix PZA. The next result provides
a simple error bound for this method; this argument was proposed b y Ming Gu.
Theorem 9.3 ( Analysis of Truncated SVD ). Let A be an m × n matrix with
singular values σ1 ≥ σ2 ≥ σ3 ≥ . . . , and let Z be an m × ℓ matrix, where ℓ ≥ k.
Suppose that ˆA(k) is a best rank- k approximation of PZ A with respect to the spectral
norm. Then

A − ˆA(k)

 ≤ σk+1 + ∥(I − PZ )A∥ .
Proof. Apply the triangle inequality to split the error into two components.

A − ˆAk

 ≤

A − PZA

 +

PZ A − ˆA(k)

. (9.12)
We have already developed a detailed theory for estimating the ﬁrst term. To analyze
the second term, we introduce a best rank- k approximation A(k) of the matrix A.
Note that

PZA − ˆA(k)

 ≤

PZA − PZA(k)



<!-- page 55 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 55
because ˆA(k) is a best rank- k approximation to the matrix PZA, whereas PZA(k) is
an undistinguished rank- k matrix. It follows that

PZA − ˆA(k)

 ≤

PZ(A − A(k))

 ≤

A − A(k)

 = σk+1. (9.13)
The second inequality holds because the orthogonal projector is a contraction; the
last identity follows from Mirsky’s theorem [97]. Combine (9.12) and (9.1 3) to reach
the main result.
Remark 9.1. In the randomized setting, the truncation step appears to be less
damaging than the error bound of Theorem 9.3 suggests, but we cu rrently lack a
complete theoretical understanding of its behavior.
10. Gaussian test matrices. The error bound in Theorem 9.1 shows that
the performance of the proto-algorithm depends on the interact ion between the test
matrix Ω and the right singular vectors of the input matrix A. Algorithm 4.1 is a
particularly simple version of the proto-algorithm that draws the te st matrix according
to the standard Gaussian distribution. The literature contains a we alth of information
about these matrices, which allows us to perform a very precise err or analysis.
We focus on the real case in this section. Analogous results hold in th e complex
case, where the algorithm even exhibits superior performance.
10.1. T echnical background. A standard Gaussian matrix is a random ma-
trix whose entries are independent standard normal variables. Th e distribution of
a standard Gaussian matrix is rotationally invariant: If U and V are orthonormal
matrices, then U ∗ GV also has the standard Gaussian distribution.
Our analysis requires detailed information about the properties of G aussian ma-
trices. In particular, we must understand how the norm of a Gauss ian matrix and its
pseudoinverse vary. We summarize the relevant results and citatio ns here, reserving
the details for Appendix A.
Proposition 10.1 ( Expected norm of a scaled Gaussian matrix ). Fix matrices
S, T , and draw a standard Gaussian matrix G. Then
(
E ∥SGT ∥2
F
)1/ 2
= ∥S∥F ∥T ∥F and (10.1)
E ∥SGT ∥ ≤ ∥ S∥ ∥T ∥F + ∥S∥F ∥T ∥ . (10.2)
The identity (10.1) follows from a direct calculation. The second boun d (10.2)
relies on methods developed by Gordon [62, 63]. See Propositions A.1 a nd A.2.
Proposition 10.2 ( Expected norm of a pseudo-inverted Gaussian matrix ). Draw
a k × (k + p) standard Gaussian matrix G with k ≥ 2 and p ≥ 2. Then
(
E

G†
2
F
)1/ 2
=
√
k
p − 1 and (10.3)
E

G†
 ≤ e√
k + p
p . (10.4)

<!-- page 56 -->

56 HALKO, MARTINSSON, AND TROPP
The ﬁrst identity is a standard result from multivariate statistics [99 , p. 96]. The
second follows from work of Chen and Dongarra [25]. See Proposition A.4 and A.5.
To study the probability that Algorithm 4.1 produces a large error, w e rely on
tail bounds for functions of Gaussian matrices. The next proposit ion rephrases a
well-known result on concentration of measure [14, Thm. 4.5.7]. See a lso [83, §1.1]
and [82, §5.1].
Proposition 10.3 ( Concentration for functions of a Gaussian matrix ). Suppose
that h is a Lipschitz function on matrices:
|h(X) − h(Y )| ≤ L ∥X − Y ∥F for all X, Y .
Draw a standard Gaussian matrix G. Then
P {h(G) ≥ E h(G) + Lt} ≤ e− t2/ 2.
Finally, we state some large deviation bounds for the norm of a pseud o-inverted
Gaussian matrix.
Proposition 10.4 ( Norm bounds for a pseudo-inverted Gaussian matrix ). Let
G be a k × (k + p) Gaussian matrix where p ≥ 4. For all t ≥ 1,
P
{

G†

F ≥
√
12k
p ·t
}
≤ 4t− p and (10.5)
P
{
G†
 ≥ e√
k + p
p + 1 ·t
}
≤ t− (p+1). (10.6)
Compare these estimates with Proposition 10.2. It seems that (10.5 ) is new; we
were unable to ﬁnd a comparable analysis in the random matrix literatu re. Although
the form of (10.5) is not optimal, it allows us to produce more transpa rent results than
a fully detailed estimate. The bound (10.6) essentially appears in the w ork of Chen
and Dongarra [25]. See Propositions A.3 and Theorem A.6 for more info rmation.
10.2. Average-case analysis of Algorithm 4.1. We separate our analysis
into two pieces. First, we present information about expected valu es. In the next
subsection, we describe bounds on the probability of a large deviatio n.
We begin with the simplest result, which provides an estimate for the e xpected
approximation error in the Frobenius norm. All proofs are postpon ed to the end of
the section.
Theorem 10.5 ( Average Frobenius error ). Suppose that A is a real m × n
matrix with singular values σ1 ≥ σ2 ≥ σ3 ≥ . . . . Choose a target rank k ≥ 2 and
an oversampling parameter p ≥ 2, where k + p ≤ min{m, n }. Draw an n × (k + p)
standard Gaussian matrix Ω, and construct the sample matrix Y = AΩ. Then the
expected approximation error
E ∥(I − PY )A∥F ≤
(
1 + k
p − 1
)1/ 2 (∑
j>k
σ 2
j
)1/ 2
.

<!-- page 57 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 57
This theorem predicts several intriguing behaviors of Algorithm 4.1. The Eckart–
Young theorem [54] shows that ( ∑
j>k σ 2
j )1/ 2 is the minimal Frobenius-norm error
when approximating A with a rank- k matrix. This quantity is the appropriate bench-
mark for the performance of the algorithm. If the small singular va lues of A are very
ﬂat, the series may be as large as σk+1
√
min{m, n } − k. On the other hand, when
the singular values exhibit some decay, the error may be on the same order as σk+1.
The error bound always exceeds this baseline error, but it may be po lynomially
larger, depending on the ratio between the target rank k and the oversampling pa-
rameter p. For p small (say, less than ﬁve), the error is somewhat variable because
the small singular values of a nearly square Gaussian matrix are very unstable. As
the oversampling increases, the performance improves quickly. Wh en p ∼ k, the error
is already within a constant factor of the baseline.
The error bound for the spectral norm is somewhat more complicat ed, but it
reveals some interesting new features.
Theorem 10.6 ( Average spectral error ). Under the hypotheses of Theorem 10.5,
E ∥(I − PY )A∥ ≤
(
1 +
√
k
p − 1
)
σk+1 + e√ k + p
p
(∑
j>k
σ 2
j
)1/ 2
.
Mirsky [97] has shown that the quantity σk+1 is the minimum spectral-norm error
when approximating A with a rank- k matrix, so the ﬁrst term in Theorem 10.6 is
analogous with the error bound in Theorem 10.5. The second term re presents a new
phenomenon: we also pay for the Frobenius-norm error in approxim ating A. Note
that, as the amount p of oversampling increases, the polynomial factor in the second
term declines much more quickly than the factor in the ﬁrst term. Wh en p ∼ k, the
factor on the σk+1 term is constant, while the factor on the series has order k− 1/ 2
We also note that the bound in Theorem 10.6 implies
E ∥(I − PY )A∥ ≤
[
1 +
√
k
p − 1 + e√ k + p
p ·
√
min{m, n } − k
]
σk+1,
so the average spectral-norm error always lies within a small polynom ial factor of the
baseline σk+1.
Let us continue with the proofs of these results.
Proof. [Theorem 10.5] Let V be the right unitary factor of A. Partition V =
[V1 | V2] into blocks containing, respectively, k and n − k columns. Recall that
Ω1 = V ∗
1 Ω and Ω2 = V ∗
2 Ω.
The Gaussian distribution is rotationally invariant, so V ∗ Ω is also a standard Gaus-
sian matrix. Observe that Ω1 and Ω2 are nonoverlapping submatrices of V ∗ Ω, so
these two matrices are not only standard Gaussian but also stocha stically indepen-
dent. Furthermore, the rows of a (fat) Gaussian matrix are almos t surely in general
position, so the k × (k + p) matrix Ω1 has full row rank with probability one.
H¨ older’s inequality and Theorem 9.1 together imply that
E ∥(I − PY )A∥F ≤
(
E ∥(I − PY )A∥2
F
)1/ 2
≤
(
Σ2

2
F + E

Σ2Ω2Ω†
1

2
F
)1/ 2
.

<!-- page 58 -->

58 HALKO, MARTINSSON, AND TROPP
We compute this expectation by conditioning on the value of Ω1 and applying Propo-
sition 10.1 to the scaled Gaussian matrix Ω2. Thus,
E

Σ2Ω2Ω†
1

2
F = E
(
E
[
Σ2Ω2Ω†
1

2
F
⏐
⏐ Ω1
])
= E
(
∥Σ2∥2
F

Ω†
1

2
F
)
= ∥Σ2∥2
F ·E

Ω†
1

2
F = k
p − 1 · ∥Σ2∥2
F ,
where the last expectation follows from relation (10.3) of Propositio n 10.2. In sum-
mary,
E ∥(I − PY )A∥F ≤
(
1 + k
p − 1
)1/ 2
∥Σ2∥F .
Observe that ∥Σ2∥2
F = ∑
j>k σ 2
j to complete the proof.
Proof. [Theorem 10.6] The argument is similar to the proof of Theorem 10.5.
First, Theorem 9.1 implies that
E ∥(I − PY )A∥ ≤ E
(
∥Σ2∥2 +

Σ2Ω2Ω†
1

2)1/ 2
≤ ∥ Σ2∥ + E

Σ2Ω2Ω†
1

.
We condition on Ω1 and apply Proposition 10.1 to bound the expectation with respect
to Ω2. Thus,
E

Σ2Ω2Ω†
1

 ≤ E
(
∥Σ2∥

Ω†
1


F + ∥Σ2∥F

Ω†
1


)
≤ ∥ Σ2∥
(
E

Ω†
1

2
F
)1/ 2
+ ∥Σ2∥F ·E

Ω†
1

.
where the second relation requires H¨ older’s inequality. Applying both parts of Propo-
sition 10.2, we obtain
E

Σ2Ω2Ω†
1

 ≤
√
k
p − 1 ∥Σ2∥ + e√ k + p
p ∥Σ2∥F .
Note that ∥Σ2∥ = σk+1 to wrap up.
10.3. Probabilistic error bounds for Algorithm 4.1. We can develop tail
bounds for the approximation error, which demonstrate that the average performance
of the algorithm is representative of the actual performance. We begin with the
Frobenius norm because the result is somewhat simpler.
Theorem 10.7 ( Deviation bounds for the Frobenius error ). Frame the hypotheses
of Theorem 10.5. Assume further that p ≥ 4. For all u, t ≥ 1,
∥(I − PY )A∥F ≤
(
1 + t ·
√
12k/p
) (∑
j>k
σ 2
j
)1/ 2
+ ut · e√ k + p
p + 1 ·σk+1,
with failure probability at most 5t− p + 2e− u2/ 2.
To parse this theorem, observe that the ﬁrst term in the error bo und corresponds
with the expected approximation error in Theorem 10.5. The second term represents
a deviation above the mean.

<!-- page 59 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 59
An analogous result holds for the spectral norm.
Theorem 10.8 ( Deviation bounds for the spectral error ). Frame the hypotheses
of Theorem 10.5. Assume further that p ≥ 4. For all u, t ≥ 1,
∥(I − PY )A∥
≤
[(
1 + t ·
√
12k/p
)
σk+1 + t · e√ k + p
p + 1
(∑
j>k
σ 2
j
)1/ 2]
+ ut · e√ k + p
p + 1 σk+1,
with failure probability at most 5t− p + e− u2/ 2.
The bracket corresponds with the expected spectral-norm erro r while the remain-
ing term represents a deviation above the mean. Neither the numer ical constants nor
the precise form of the bound are optimal because of the slacknes s in Proposition 10.4.
Nevertheless, the theorem gives a fairly good picture of what is act ually happening.
We acknowledge that the current form of Theorem 10.8 is complicate d. To pro-
duce more transparent results, we make appropriate selections f or the parameters u, t
and bound the numerical constants.
Corollary 10.9 ( Simpliﬁed deviation bounds for the spectral error ). Frame
the hypotheses of Theorem 10.5, and assume further that p ≥ 4. Then
∥(I − PY )A∥ ≤
(
1 + 17
√
1 + k/p
)
σk+1 + 8√ k + p
p + 1
(∑
j>k
σ 2
j
)1/ 2
,
with failure probability at most 6e− p. Moreover,
∥(I − PY )A∥ ≤
(
1 + 8
√
(k + p) ·p log p
)
σk+1 + 3
√
k + p
(∑
j>k
σ 2
j
)1/ 2
,
with failure probability at most 6p− p.
Proof. The ﬁrst part of the result follows from the choices t = e and u = √ 2p,
and the second emerges when t = p and u = √ 2p log p. Another interesting parameter
selection is t = pc/p and u = √ 2c log p, which yields a failure probability 6 p− c.
Corollary 10.9 should be compared with [91, Obs. 4.4–4.5]. Although our result
contains sharper error estimates, the failure probabilities are usu ally worse. The error
bound (1.9) presented in §1.5 follows after further simpliﬁcation of the second bound
from Corollary 10.9.
We continue with a proof of Theorem 10.8. The same argument can be used to
obtain a bound for the Frobenius-norm error, but we omit a detailed account.
Proof. [Theorem 10.8] Since Ω1 and Ω2 are independent from each other, we can
study how the error depends on the matrix Ω2 by conditioning on the event that Ω1
is not too irregular. To that end, we deﬁne a (parameterized) even t on which the
spectral and Frobenius norms of the matrix Ω†
1 are both controlled. For t ≥ 1, let
Et =
{
Ω1 :

Ω†
1

 ≤ e√
k + p
p + 1 ·t and

Ω†
1


F ≤
√
12k
p ·t
}
.
Invoking both parts of Proposition 10.4, we ﬁnd that
P (Ec
t ) ≤ t− (p+1) + 4t− p ≤ 5t− p.

<!-- page 60 -->

60 HALKO, MARTINSSON, AND TROPP
Consider the function h(X) =

Σ2XΩ†
1

. We quickly compute its Lipschitz
constant L with the lower triangle inequality and some standard norm estimates:
|h(X) − h(Y )| ≤

Σ2(X − Y )Ω†
1


≤ ∥ Σ2∥ ∥X − Y ∥

Ω†
1

 ≤ ∥ Σ2∥

Ω†
1

 ∥X − Y ∥F .
Therefore, L ≤ ∥ Σ2∥

Ω†
1

. Relation (10.2) of Proposition 10.1 implies that
E[h(Ω2) |Ω1] ≤ ∥ Σ2∥

Ω†
1


F + ∥Σ2∥F

Ω†
1

.
Applying the concentration of measure inequality, Proposition 10.3, conditionally to
the random variable h(Ω2) =

Σ2Ω2Ω†
1

 results in
P
{
Σ2Ω2Ω†
1

 > ∥Σ2∥

Ω†
1


F + ∥Σ2∥F

Ω†
1

 + ∥Σ2∥

Ω†
1

 ·u
⏐
⏐ Et
}
≤ e− u2/ 2.
Under the event Et, we have explicit bounds on the norms of Ω†
1, so
P
{

Σ2Ω2Ω†
1

 > ∥Σ2∥
√
12k
p ·t + ∥Σ2∥F
e√ k + p
p + 1 ·t + ∥Σ2∥ e√ k + p
p + 1 ·ut
⏐
⏐
⏐
⏐ Et
}
≤ e− u2/ 2.
Use the fact P (Ec
t ) ≤ 5t− p to remove the conditioning. Therefore,
P
{

Σ2Ω2Ω†
1

 > ∥Σ2∥
√
12k
p ·t + ∥Σ2∥F
e√ k + p
p + 1 ·t + ∥Σ2∥ e√ k + p
p + 1 ·ut
}
≤ 5t− p + e− u2/ 2.
Insert the expressions for the norms of Σ2 into this result to complete the probability
bound. Finally, introduce this estimate into the error bound from Th eorem 9.1.
10.4. Analysis of the power scheme. Theorem 10.6 makes it clear that the
performance of the randomized approximation scheme, Algorithm 4 .1, depends heav-
ily on the singular spectrum of the input matrix. The power scheme ou tlined in
Algorithm 4.3 addresses this problem by enhancing the decay of spec trum. We can
combine our analysis of Algorithm 4.1 with Theorem 9.2 to obtain a detaile d report
on the behavior of the performance of the power scheme using a Ga ussian matrix.
Corollary 10.10 ( Average spectral error for the power scheme ). Frame the
hypotheses of Theorem 10.5. Deﬁne B = (AA∗ )qA for a nonnegative integer q, and
construct the sample matrix Z = BΩ. Then
E ∥(I − PZ)A∥ ≤
[(
1 +
√
k
p − 1
)
σ 2q+1
k+1 + e√ k + p
p
(∑
j>k
σ 2(2q+1)
j
)1/ 2
] 1/ (2q+1)
.
Proof. By H¨ older’s inequality and Theorem 9.2,
E ∥(I − PZ)A∥ ≤
(
E ∥(I − PZ )A∥2q+1
)1/ (2q+1)
≤ (E ∥(I − PZ)B∥)1/ (2q+1) .

<!-- page 61 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 61
Invoke Theorem 10.6 to bound the right-hand side, noting that σj(B) = σ 2q+1
j .
The true message of Corollary 10.10 emerges if we bound the series u sing its
largest term σ 4q+2
k+1 and draw the factor σk+1 out of the bracket:
E ∥(I − PZ)A∥ ≤
[
1 +
√
k
p − 1 + e√ k + p
p ·
√
min{m, n } − k
] 1/ (2q+1)
σk+1.
In words, as we increase the exponent q, the power scheme drives the extra factor in
the error to one exponentially fast. By the time q ∼ log (min{m, n }),
E ∥(I − PZ)A∥ ∼ σk+1,
which is the baseline for the spectral norm.
In most situations, the error bound given by Corollary 10.10 is subst antially better
than the estimates discussed in the last paragraph. For example, s uppose that the
tail singular values exhibit the decay proﬁle
σj ≲ j(1+ε)/ (4q+2) for j > k and ε > 0.
Then the series in Corollary 10.10 is comparable with its largest term, w hich allows
us to remove the dimensional factor min {m, n } from the error bound.
To obtain large deviation bounds for the performance of the power scheme, simply
combine Theorem 9.2 with Theorem 10.8. We omit a detailed statement.
Remark 10.1. We lack an analogous theory for the Frobenius norm because
Theorem 9.2 depends on Proposition 8.6, which is not true for the Fro benius norm.
It is possible to obtain some results by estimating the Frobenius norm in terms of the
spectral norm.
11. SRFT test matrices. Another way to implement the proto-algorithm from
§1.3 is to use a structured random matrix so that the matrix product in Step 2 can
be performed quickly. One type of structured random matrix that has been proposed
in the literature is the subsampled random Fourier transform , or SRFT, which we
discussed in §4.6. In this section, we present bounds on the performance of the proto-
algorithm when it is implemented with an SRFT test matrix. In contrast with the
results for Gaussian test matrices, the results in this section hold f or both real and
complex input matrices.
11.1. Construction and Properties. Recall from §4.6 that an SRFT is a tall
n × ℓ matrix of the form Ω =
√
n/ℓ ·DF R∗ where
• D is a random n × n diagonal matrix whose entries are independent and
uniformly distributed on the complex unit circle;
• F is the n × n unitary discrete Fourier transform; and
• R is a random ℓ × n matrix that restricts an n-dimensional vector to ℓ coor-
dinates, chosen uniformly at random.
Up to scaling, an SRFT is just a section of a unitary matrix, so it satisﬁ es the norm
identity ∥Ω∥ =
√
n/ℓ . The critical fact is that an appropriately designed SRFT
approximately preserves the geometry of an entire subspace of vectors .

<!-- page 62 -->

62 HALKO, MARTINSSON, AND TROPP
Theorem 11.1 ( The SRFT preserves geometry ). Fix an n × k orthonormal
matrix V , and draw an n × ℓ SRFT matrix Ω where the parameter ℓ satisﬁes
4
[√
k +
√
8 log(kn)
]2
log(k) ≤ ℓ ≤ n.
Then
0. 40 ≤ σk(V ∗ Ω) and σ1(V ∗ Ω) ≤ 1. 48
with failure probability at most O(k− 1).
In words, the kernel of an SRFT of dimension ℓ ∼ k log(k) is unlikely to intersect
a ﬁxed k-dimensional subspace. In contrast with the Gaussian case, the lo garithmic
factor log( k) in the lower bound on ℓ cannot generally be removed (Remark 11.2).
Theorem 11.1 follows from a straightforward variation of the argum ent in [134],
which establishes equivalent bounds for a real analog of the SRFT, c alled the subsam-
pled randomized Hadamard transform (SRHT). We omit further details.
Remark 11.1. For large problems, we can obtain better numerical constants [134 ,
Thm. 3.2]. Fix a small, positive number ι. If k ≫ log(n), then sampling
ℓ ≥ (1 + ι) ·k log(k)
coordinates is suﬃcient to ensure that σk(V ∗ Ω) ≥ ι with failure probability at most
O(k− cι). This sampling bound is essentially optimal because (1 − ι) ·k log(k) samples
are not adequate in the worst case; see Remark 11.2.
Remark 11.2. The logarithmic factor in Theorem 11.1 is necessary when the
orthonormal matrix V is particularly evil. Let us describe an inﬁnite family of worst-
case examples. Fix an integer k, and let n = k2. Form an n × k orthonormal matrix
V by regular decimation of the n × n identity matrix. More precisely, V is the
matrix whose jth row has a unit entry in column ( j − 1)/k when j ≡ 1 (mod k) and
is zero otherwise. To see why this type of matrix is nasty, it is helpful to consider
the auxiliary matrix W = V ∗ DF . Observe that, up to scaling and modulation of
columns, W consists of k copies of a k × k DFT concatenated horizontally.
Suppose that we apply the SRFT Ω = DF R∗ to the matrix V ∗ . We obtain a
matrix of the form X = V ∗ Ω = W R∗ , which consists of ℓ random columns sampled
from W . Theorem 11.1 certainly cannot hold unless σk(X) > 0. To ensure the
latter event occurs, we must pick at least one copy each of the k distinct columns
of W . This is the coupon collector’s problem [98, Sec. 3.6] in disguise. To obt ain
a complete set of k coupons (i.e., columns) with nonnegligible probability, we must
draw at least k log(k) columns. The fact that we are sampling without replacement
does not improve the analysis appreciably because the matrix has to o many columns.
11.2. Performance guarantees. We are now prepared to present detailed in-
formation on the performance of the proto-algorithm when the te st matrix Ω is an
SRFT.
Theorem 11.2 ( Error bounds for SRFT ). Fix an m × n matrix A with singular
values σ1 ≥ σ2 ≥ σ3 ≥ . . . . Draw an n × ℓ SRFT matrix Ω, where
4
[√
k +
√
8 log(kn)
]2
log(k) ≤ ℓ ≤ n.

<!-- page 63 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 63
Construct the sample matrix Y = AΩ. Then
∥(I − PY )A∥ ≤
√
1 + 7n/ℓ ·σk+1 and
∥(I − PY )A∥F ≤
√
1 + 7n/ℓ ·
(∑
j>k
σ 2
j
)1/ 2
with failure probability at most O(k− 1).
As we saw in §10.2, the quantity σk+1 is the minimal spectral-norm error possible
when approximating A with a rank- k matrix. Similarly, the series in the second bound
is the minimal Frobenius-norm error when approximating A with a rank- k matrix.
We see that both error bounds lie within a polynomial factor of the ba seline, and this
factor decreases with the number ℓ of samples we retain.
The likelihood of error with an SRFT test matrix is substantially worse t han in
the Gaussian case. The failure probability here is roughly k− 1, while in the Gaussian
case, the failure probability is roughly e − (ℓ− k). This qualitative diﬀerence is not
an artifact of the analysis; discrete sampling techniques inherently fail with higher
probability.
Matrix approximation schemes based on SRFTs often perform much better in
practice than the error analysis here would indicate. While it is not gen erally possible
to guarantee accuracy with a sampling parameter less than ℓ ∼ k log(k), we have found
empirically that the choice ℓ = k + 20 is adequate in almost all applications. Indeed,
SRFTs sometimes perform even better than Gaussian matrices (see, e.g., Figure 7.8).
We complete the section with the proof of Theorem 11.2.
Proof. [Theorem 11.2] Let V be the right unitary factor of matrix A, and partition
V = [V1 | V2] into blocks containing, respectively, k and n − k columns. Recall that
Ω1 = V ∗
1 Ω and Ω2 = V ∗
2 Ω.
where Ω is the conjugate transpose of an SRFT. Theorem 11.1 ensures tha t the
submatrix Ω1 has full row rank, with failure probability at most O( k− 1). Therefore,
Theorem 9.1 implies that
| | |(I − PY )A| | | ≤ | | |Σ2| | |
[
1 +

Ω†
1

2
· ∥Ω2∥2
]1/ 2
,
where | | |·| | |denotes either the spectral norm or the Frobenius norm. Our app lication
of Theorem 11.1 also ensures that the spectral norm of Ω†
1 is under control.

Ω†
1

2
≤ 1
0. 402 < 7.
We may bound the spectral norm of Ω2 deterministically.
∥Ω2∥ = ∥V ∗
2 Ω∥ ≤ ∥ V ∗
2 ∥ ∥Ω∥ =
√
n/ℓ
since V2 and
√
ℓ/n ·Ω are both orthonormal matrices. Combine these estimates to
complete the proof.
Acknowledgments. The authors have beneﬁted from valuable discussions with
many researchers, among them Inderjit Dhillon, Petros Drineas, M ing Gu, Edo Lib-
erty, Michael Mahoney, Vladimir Rokhlin, Yoel Shkolnisky, and Arthu r Szlam. In

<!-- page 64 -->

64 HALKO, MARTINSSON, AND TROPP
particular, we would like to thank Mark Tygert for his insightful rema rks on early
drafts of this paper. The example in Section 7.2 was provided by Fran ¸ cois Meyer of
the University of Colorado at Boulder. The example in Section 7.3 come s from the
FERET database of facial images collected under the FERET progra m, sponsored by
the DoD Counterdrug Technology Development Program Oﬃce. The work reported
was initiated during the program Mathematics of Knowledge and Search Engines held
at IPAM in the fall of 2007. Finally, we would like to thank the anonymou s referees,
whose thoughtful remarks have helped us to improve the manuscr ipt dramatically.
Appendix A. On Gaussian matrices. This appendix collects some of the
properties of Gaussian matrices that we use in our analysis. Most of the results follow
quickly from material that is already available in the literature. One fa ct, however,
requires a surprisingly diﬃcult new argument. We focus on the real c ase here; the
complex case is similar but actually yields better results.
A.1. Expectation of norms. We begin with the expected Frobenius norm of
a scaled Gaussian matrix, which follows from an easy calculation.
Proposition A.1. Fix real matrices S, T , and draw a standard Gaussian matrix
G. Then
(
E ∥SGT ∥2
F
)1/ 2
= ∥S∥F ∥T ∥F .
Proof. The distribution of a Gaussian matrix is invariant under orthogonal trans-
formations, and the Frobenius norm is also unitarily invariant. As a re sult, it repre-
sents no loss of generality to assume that S and T are diagonal. Therefore,
E ∥SGT ∥2
F = E
[∑
jk
|sjj gjktkk|2
]
=
∑
jk
|sjj |2|tkk|2 = ∥S∥2
F ∥T ∥2
F .
Since the right-hand side is unitarily invariant, we have also identiﬁed t he value of
the expectation for general matrices S and T .
The literature contains an excellent bound for the expected spect ral norm of a
scaled Gaussian matrix. The result is due to Gordon [62, 63], who esta blished the
bound using a sharp version of Slepian’s lemma. See [83, §3.3] and [34, §2.3] for
additional discussion.
Proposition A.2. Fix real matrices S, T , and draw a standard Gaussian matrix
G. Then
E ∥SGT ∥ ≤ ∥ S∥ ∥T ∥F + ∥S∥F ∥T ∥ .
A.2. Spectral norm of pseudoinverse. Now, we turn to the pseudoinverse
of a Gaussian matrix. Recently, Chen and Dongarra developed a goo d bound on
the probability that its spectral norm is large. The statement here follows from [25,
Lem. 4.1] after an application of Stirling’s approximation. See also [91, Lem. 2.14]
Proposition A.3. Let G be an m× n standard Gaussian matrix with n ≥ m ≥ 2.
For each t > 0,
P
{
G†
 > t
}
≤ 1
√
2π (n − m + 1)
[ e√ n
n − m + 1
]n− m+1
t− (n− m+1).

<!-- page 65 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 65
We can use Proposition A.3 to bound the expected spectral norm of a pseudo-
inverted Gaussian matrix.
Proposition A.4. Let G be a m × n standard Gaussian matrix with n − m ≥ 1
and m ≥ 2. Then
E

G†
 < e√
n
n − m
Proof. Let us make the abbreviations p = n − m and
C = 1√
2π (p + 1)
[ e√ n
p + 1
]p+1
.
We compute the expectation by way of a standard argument. The in tegral formula
for the mean of a nonnegative random variable implies that, for all E > 0,
E

G†
 =
∫ ∞
0
P
{
G†
 > t
}
dt ≤ E +
∫ ∞
E
P
{
G†
 > t
}
dt
≤ E + C
∫ ∞
E
t− (p+1) dt = E + 1
p CE − p,
where the second inequality follows from Proposition A.3. The right-h and side is
minimized when E = C1/ (p+1). Substitute and simplify.
A.3. F robenius norm of pseudoinverse. The squared Frobenius norm of a
pseudo-inverted Gaussian matrix is closely connected with the trac e of an inverted
Wishart matrix. This observation leads to an exact expression for t he expectation.
Proposition A.5. Let G be an m× n standard Gaussian matrix with n− m ≥ 2.
Then
E

G†
2
F = m
n − m − 1 .
Proof. Observe that

G†
2
F = trace
[
(G†)∗ G†]
= trace
[
(GG∗ )− 1]
.
The second identity holds almost surely because the Wishart matrix GG∗ is invertible
with probability one. The random matrix ( GG∗ )− 1 follows the inverted Wishart
distribution, so we can compute its expected trace explicitly using a f ormula from [99,
p. 97].
On the other hand, very little seems to be known about the tail beha vior of the
Frobenius norm of a pseudo-inverted Gaussian matrix. The following theorem, which
is new, provides an adequate bound on the probability of a large devia tion.
Theorem A.6. Let G be an m × n standard Gaussian matrix with n − m ≥ 4.
For each t ≥ 1,
P
{
G†
2
F > 12m
n − m ·t
}
≤ 4t− (n− m)/ 2.

<!-- page 66 -->

66 HALKO, MARTINSSON, AND TROPP
Neither the precise form of Theorem A.6 nor the constants are idea l; we have
focused instead on establishing a useful bound with minimal fuss. Th e rest of the
section is devoted to the rather lengthy proof. Unfortunately, m ost of the standard
methods for producing tail bounds fail for random variables that d o not exhibit normal
or exponential concentration. Our argument relies on special pro perties of Gaussian
matrices and a dose of brute force.
A.3.1. T echnical background. We begin with a piece of notation. For any
number q ≥ 1, we deﬁne the Lq norm of a random variable Z by
Eq(Z) = ( E |Z|q)
1/q
.
In particular, the Lq norm satisﬁes the triangle inequality.
We continue with a collection of technical results. First, we present a striking
fact about the structure of Gaussian matrices [55, §3.5].
Proposition A.7. For n ≥ m, an m × n standard Gaussian matrix is orthogo-
nally equivalent with a random bidiagonal matrix
L =







Xn
Ym− 1 Xn− 1
Ym− 2 Xn− 2
. . . . . .
Y1 Xn− (m− 1)







m× n
, (A.1)
where, for each j, the random variables X 2
j and Y 2
j follow the χ 2 distribution with j
degrees of freedom. Furthermore, these variates are mutual ly independent.
We also require the moments of a chi-square variate, which are expr essed in terms
of special functions.
Proposition A.8. Let Ξ be a χ 2 variate with k degrees of freedom. When
0 ≤ q < k/ 2,
E (Ξq) = 2qΓ(k/ 2 + q)
Γ(k/ 2) and E
(
Ξ− q)
= Γ(k/ 2 − q)
2qΓ(k/ 2) .
Proof. Recall that a χ 2 variate with k degrees of freedom has the probability
density function
f (t) = 1
2k/ 2Γ(k/ 2) tk/ 2− 1e− t/ 2, for t ≥ 0.
By the integral formula for expectation,
E(Ξq) =
∫ ∞
0
tqf (t) dt = 2qΓ(k/ 2 + q)
Γ(k/ 2) ,
where the second equality follows from Euler’s integral expression f or the gamma
function. The other calculation is similar.

<!-- page 67 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 67
To streamline the proof, we eliminate the gamma functions from Prop osition A.8.
The next result bounds the positive moments of a chi-square variat e.
Lemma A.9. Let Ξ be a χ 2 variate with k degrees of freedom. For q ≥ 1,
Eq(Ξ) ≤ k + q.
Proof. Write q = r + θ, where r = ⌊q⌋. Repeated application of the functional
equation zΓ(z) = Γ( z + 1) yields
Eq(Ξ) =

 2θΓ(k/ 2 + θ)
Γ(k/ 2) ·
r∏
j=1
(k + 2(q − j))


1/q
.
The gamma function is logarithmically convex, so
2θΓ(k/ 2 + θ)
Γ(k/ 2) ≤ 2θ ·Γ(k/ 2)1− θ ·Γ(k/ 2 + 1)θ
Γ(k/ 2) = kθ ≤


r∏
j=1
(k + 2(q − j))


θ/r
.
The second inequality holds because k is smaller than each term in the product, hence
is smaller than their geometric mean. As a consequence,
Eq(Ξ) ≤


r∏
j=1
(k + 2(q − j))


1/r
≤ 1
r
r∑
j=1
(k + 2(q − j)) ≤ k + q.
The second relation is the inequality between the geometric and arith metic mean.
Finally, we develop a bound for the negative moments of a chi-square variate.
Lemma A.10. Let Ξ be a χ 2 variate with k degrees of freedom, where k ≥ 5.
When 2 ≤ q ≤ (k − 1)/ 2,
Eq (
Ξ− 1)
< 3
k .
Proof. We establish the bound for q = ( k − 1)/ 2. For smaller values of q, the
result follows from H¨ older’s inequality. Proposition A.8 shows that
Eq (
Ξ− 1)
=
[ Γ(k/ 2 − q)
2qΓ(k/ 2)
]1/q
=
[ Γ(1/ 2)
2qΓ(k/ 2)
]1/q
.
Stirling’s approximation ensures that Γ( k/ 2) ≥
√
2π ·(k/ 2)(k− 1)/ 2 ·e− k/ 2. Since the
value Γ(1/ 2) = √ π ,
Eq (
Ξ− 1)
≤
[ √ π
2q√
2π ·(k/ 2)q ·e− q− 1/ 2
]1/q
= e
k
[ e
2
]1/ 2q
< 3
k ,
where we used the assumption q ≥ 2 to complete the numerical estimate.

<!-- page 68 -->

68 HALKO, MARTINSSON, AND TROPP
A.3.2. Proof of Theorem A.6. Let G be an m × n Gaussian matrix, where
we assume that n − m ≥ 4. Deﬁne the random variable
Z =

G†
2
F .
Our goal is to develop a tail bound for Z. The argument is inspired by work of
Szarek [130, §6] for square Gaussian matrices.
The ﬁrst step is to ﬁnd an explicit, tractable representation for th e random vari-
able. According to Proposition A.7, a Gaussian matrix G is orthogonally equivalent
with a bidiagonal matrix L of the form (A.1). Making an analogy with the inversion
formula for a triangular matrix, we realize that the pseudoinverse o f L is given by
L† =












X − 1
n
− Ym− 1
XnXn− 1
X − 1
n− 1
− Ym− 2
Xn− 1Xn− 2
X − 1
n− 2
. . . . . .
− Y1
Xn− (m− 2)Xn− (m− 1)
X − 1
n− (m− 1)












n× m
.
Because L† is orthogonally equivalent with G† and the Frobenius norm is unitarily
invariant, we have the relations
Z =

G†
2
F =

L†
2
F ≤
m− 1∑
j=0
1
X 2
n− j
(
1 + Y 2
m− j
X 2
n− j+1
)
,
where we have added an extra subdiagonal term (corresponding w ith j = 0) so that
we can avoid exceptional cases later. We abbreviate the summands as
Wj = 1
X 2
n− j
(
1 + Y 2
m− j
X 2
n− j+1
)
, j = 0, 1, 2, . . . , m − 1.
Next, we develop a large deviation bound for each summand by compu ting a mo-
ment and invoking Markov’s inequality. For the exponent q = (n− m)/ 2, Lemmas A.9
and A.10 yield
Eq(Wj) = Eq(X − 2
n− j) ·Eq
[
1 + Y 2
m− j
X 2
n− j+1
]
≤ Eq(X − 2
n− j)
[
1 + Eq(Y 2
m− j) ·Eq(X − 2
n− j+1)
]
≤ 3
n − j
[
1 + 3(m − j + q)
n − j + 1
]
= 3
n − j
[
1 + 3 − 3(n − m + 1 − q)
n − j + 1
]
Note that the ﬁrst two relations require the independence of the v ariates and the
triangle inequality for the Lq norm. The maximum value of the bracket evidently
occurs when j = 0, so
Eq(Wj ) < 12
n − j , j = 0, 1, 2, . . . , m − 1.

<!-- page 69 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 69
Markov’s inequality results in
P
{
Wj ≥ 12
n − j ·u
}
≤ u− q.
Select u = t ·(n − j)/ (n − m) to reach
P
{
Wj ≥ 12
n − m ·t
}
≤
[ n − m
n − j
]q
t− q.
To complete the argument, we combine these estimates by means of the union
bound and clean up the resulting mess. Since Z ≤ ∑m− 1
j=0 Wj,
P
{
Z ≥ 12m
n − m ·t
}
≤ t− q
m− 1∑
j=0
[ n − m
n − j
]q
.
To control the sum on the right-hand side, observe that
m− 1∑
j=0
[ n − m
n − j
]q
< (n − m)q
∫ m
0
(n − x)− q dx
< (n − m)q
q − 1 (n − m)− q+1 = 2(n − m)
n − m − 2 ≤ 4,
where the last inequality follows from the hypothesis n − m ≥ 4. Together, the
estimates in this paragraph produce the advertised bound.
Remark A.1. It would be very interesting to ﬁnd more conceptual and extensible
argument that yields accurate concentration results for inverse spectral functions of
a Gaussian matrix.
REFERENCES
[1] D. Achlioptas , Database-friendly random projections: Johnson–Lindenst rauss with binary
coins, J. Comput. System Sci., 66 (2003), pp. 671–687.
[2] D. Achlioptas and F. McSherry , Fast computation of low-rank matrix approximations , J.
Assoc. Comput. Mach., 54 (2007), pp. Art. 9, 19 pp. (electron ic).
[3] N. Ailon and B. Chazelle , Approximate nearest neighbors and the fast Johnson–
Lindenstrauss transform , in STOC ’06: Proc. 38th Ann. ACM Symp. Theory of Com-
puting, 2006, pp. 557–563.
[4] N. Ailon and E. Liberty , Fast dimension reduction using Rademacher series on dual BC H
codes, in STOC ’08: Proc. 40th Ann. ACM Symp. Theory of Computing, 2 008.
[5] N. Alon, P. Gibbons, Y. Matias, and M. Szegedy , Tracking join and self-join sizes in
limited storage, in Proc. 18th ACM Symp. Principles of Database Systems (POD S), 1999,
pp. 10–20.
[6] N. Alon, Y. Matias, and M. Szegedy , The space complexity of approximating frequency
moments, in STOC ’96: Proc. 28th Ann. ACM Symp. Theory of Algorithms, 1996, pp. 20–
29.
[7] S. Arora, E. Hazan, and S. Kale , A fast random sampling algorithm for sparsifying matri-
ces, in Approximation, randomization and combinatorial optim ization: Algorithms and
Techniques, Springer, Berlin, 2006, pp. 272–279.
[8] A. R. Barron , Universal approximation bounds for superpositions of a sig moidal function ,
IEEE Trans. Inform. Theory, 39 (1993), pp. 930–945.
[9] I. Beichl , The Metropolis algorithm , Comput. Sci. Eng., (2000), pp. 65–69.
[10] M.-A. Bellabas and P. J. Wolfe , On sparse representations of linear operators and the
approximation of matrix products , in Proc. 42nd Ann. Conf. Information Sciences and
Systems (CISS), 2008, pp. 258–263.

<!-- page 70 -->

70 HALKO, MARTINSSON, AND TROPP
[11] R. Bhatia , Matrix Analysis , no. 169 in GTM, Springer, Berlin, 1997.
[12] ˚
A. Bj ¨orck, Numerics of Gram–Schmidt orthogonalization , Linear Algebra Appl., 197–198
(1994), pp. 297–316.
[13]
, Numerical Methods for Least Squares Problems , SIAM, Philadelphia, PA, 1996.
[14] V. Bogdanov , Gaussian Measures, American Mathematical Society, Providence, RI, 1998.
[15] J. Bourgain , On Lipschitz embedding of ﬁnite metric spaces in Hilbert spa ce, Israel J. Math.,
52 (1985), pp. 46–52.
[16] C. Boutsidis and P. Drineas , Random projections for nonnegative least squares , Linear
Algebra Appl., 431 (2009), pp. 760–771.
[17] C. Boutsidis, P. Drineas, and M. W. Mahoney , An improved approximation algorithm
for the column subset selection problem , in Proc. 20th Ann. ACM–SIAM Symp. Discrete
Algorithms (SODA), 2009.
[18] C. Boutsidis, M. W. Mahoney, and P. Drineas , Unsupervised feature selection for principal
components analysis, in Proc. ACM SIGKDD Intl. Conf. Knowledge Discovery and Dat a
Mining (KDD), Aug. 2008.
[19] E. Cand `es and J. K. Romberg , Sparsity and incoherence in compressive sampling , Inverse
Problems, 23 (2007), pp. 969–985.
[20] E. Cand `es, J. K. Romberg, and T. Tao , Robust uncertainty principles: Exact signal recon-
struction from highly incomplete Fourier information , IEEE Trans. Inform. Theory, 52
(2006), pp. 489–509.
[21] E. J. Cand `es, Compressive sampling , in Proc. 2006 Intl. Cong. Mathematicians, Madrid,
2006.
[22] E. J. Cand `es and B. Recht , Exact matrix completion via convex optimization , Found.
Comput. Math., 9 (2009), pp. 717–772.
[23] E. J. Cand `es and T. Tao , The power of convex relaxation: Near-optimal matrix comple tion,
IEEE Trans. Inform. Theory, 56 (2010), pp. 2053–2080.
[24] B. Carl , Inequalities of Bernstein–Jackson-type and the degree of c ompactness in Banach
spaces, Ann. Inst. Fourier (Grenoble), 35 (1985), pp. 79–118.
[25] Z. Chen and J. J. Dongarra , Condition numbers of Gaussian random matrices , SIAM J.
Matrix Anal. Appl., 27 (2005), pp. 603–620.
[26] H. Cheng, Z. Gimbutas, P.-G. Martinsson, and V. Rokhlin , On the compression of low
rank matrices, SIAM J. Sci. Comput., 26 (2005), pp. 1389–1404 (electronic ).
[27] A. C ¸ ivril and M. Magdon-Ismail, On selecting a maximum volume sub-matrix of a matrix
and related problems , Theoret. Comput. Sci., 410 (2009), pp. 4801–4811.
[28] K. L. Clarkson , Subgradient and sampling algorithms for ℓ1 regression, in Proc. 16th Ann.
ACM–SIAM Symp. Discrete Algorithms (SODA), 2005, pp. 257–2 66.
[29] K. L. Clarkson and D. P. Woodruff , Numerical linear algebra in the streaming model , in
STOC ’09: Proc. 41st Ann. ACM Symp. Theory of Computing, 2009 .
[30] R. R. Coifman, S. Lafon, A. B. Lee, M. Maggioni, B. Nadler, F. W ar ner, and S. W.
Zucker, Geometric diﬀusions as a tool for harmonic analysis and stru cture deﬁnition of
data: Diﬀusion maps , Proc. Natl. Acad. Sci. USA, 102 (2005), pp. 7426–7431.
[31] A. Dasgupta, P. Drineas, B. Harb, R. Kumar, and M. W. Mahoney , Sampling algorithms
and coresets for ℓp regression, SIAM J. Comput., 38 (2009), pp. 2060–2078.
[32] S. Dasgupta and A. Gupta , An elementary proof of the Johnson–Lindenstrauss lemma ,
Computer Science Dept. Tech. Report 99-006, Univ. Californ ia at Berkeley, Mar. 1999.
[33] A. d’Aspremont , Subsampling algorithms for semideﬁnite programming . Available at arXiv:
0803.1990, Apr. 2009.
[34] K. R. Davidson and S. J. Szarek , Local operator theory, random matrices, and Banach
spaces, in Handbook of Banach Space Geometry, W. B. Johnson and J. Li ndenstrauss,
eds., Elsevier, 2002, pp. 317–366.
[35] J. Demmel, I. Dumitriu, and O. Holtz , Fast linear algebra is stable , Numer. Math., 108
(2007), pp. 59–91.
[36] A. Deshpande and L. Rademacher , Eﬃcient volume sampling for row/column subset se-
lection. Available at arXiv:1004.4057, Apr. 2010.
[37] A. Deshpande, L. Rademacher, S. Vempala, and G. W ang , Matrix approximation and
projective clustering via volume sampling , in Proc. 17th Ann. ACM–SIAM Symp. Discrete
Algorithms (SODA), 2006, pp. 1117–1126.
[38] A. Deshpande and S. Vempala , Adaptive sampling and fast low-rank matrix approximation ,
in Approximation, randomization and combinatorial optimi zation, vol. 4110 of LNCS,
Springer, Berlin, 2006, pp. 292–303.
[39] J. D. Dixon , Estimating extremal eigenvalues and condition numbers of m atrices, SIAM J.
Numer. Anal., 20 (1983), pp. 812–814.

<!-- page 71 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 71
[40] J. Dongarra and F. Sullivan , The top 10 algorithms , Comput. Sci. Eng., (2000), pp. 22–23.
[41] D. L. Donoho , Compressed sensing, IEEE Trans. Inform. Theory, 52 (2006), pp. 1289–1306.
[42] D. L. Donoho, M. Vetterli, R. A. DeVore, and I. Daubechies , Data compression and
harmonic analysis , IEEE Trans. Inform. Theory, 44 (1998), pp. 2433–2452.
[43] P. Drineas, A. Frieza, R. Kannan, S. Vempala, and V. Vinay , Clustering of large graphs
via the singular value decomposition , Machine Learning, 56 (2004), pp. 9–33.
[44] P. Drineas, A. Frieze, R. Kannan, S. Vempala, and V. Vinay , Clustering in large graphs
and matrices , in Proc. 10th Ann. ACM Symp. on Discrete Algorithms (SODA), 1999,
pp. 291–299.
[45] P. Drineas, R. Kannan, and M. W. Mahoney , Fast Monte Carlo algorithms for matrices.
I. Approximating matrix multiplication , SIAM J. Comput., 36 (2006), pp. 132–157.
[46] , Fast Monte Carlo algorithms for matrices. II. Computing a lo w-rank approximation
to a matrix , SIAM J. Comput., 36 (2006), pp. 158–183 (electronic).
[47] , Fast Monte Carlo algorithms for matrices. III. Computing a c ompressed approximate
matrix decomposition, SIAM J. Comput., 36 (2006), pp. 184–206.
[48] P. Drineas and M. W. Mahoney , On the Nystr¨ om method for approximating a Gram matrix
for improved kernel-based learning , J. Mach. Learn. Res., 6 (2005), pp. 2153–2175.
[49] , A randomized algorithm for a tensor-based generalization o f the singular value de-
composition, Linear Algebra Appl., 420 (2007), pp. 553–571.
[50] P. Drineas, M. W. Mahoney, and S. Muthukrishnan , Subspace sampling and relative-
error matrix approximation: Column-based methods , in APPROX and RANDOM 2006,
J. Diaz et al., ed., no. 4110 in LNCS, Springer, Berlin, 2006, pp. 321–326.
[51] , Relative-error CUR matrix decompositions , SIAM J. Matrix Anal. Appl., 30 (2008),
pp. 844–881.
[52] P. Drineas, M. W. Mahoney, S. Muthukrishnan, and T. Sarl ´os, Faster least squares
approximation, Num. Math., (2009). To appear. Available at arXiv:0710.1435.
[53] A. Dvoretsky, Some results on convex bodies and Banach spaces , in Proc. Intl. Symp. Linear
Spaces, Jerusalem, 1961, pp. 123–160.
[54] C. Eckart and G. Young , The approximation of one matrix by another of lower rank ,
Psychometrika, 1 (1936), pp. 211–218.
[55] A. Edelman , Eigenvalues and condition numbers of random matrices , Ph.D. thesis, Mathe-
matics Dept., Massachusetts Inst. Tech., Boston, MA, May 19 89.
[56] B. Engquist and O. Runborg , Wavelet-based numerical homogenization with application s,
in Multiscale and Multiresolution Methods: Theory and Appl ications, T. J. Barth et al.,
ed., vol. 20 of LNCSE, Springer, Berlin, 2001, pp. 97–148.
[57] A. Frieze, R. Kannan, and S. Vempala , Fast Monte Carlo algorithms for ﬁnding low-
rank approximations, in Proc. 39th Ann. IEEE Symp. Foundations of Computer Scien ce
(FOCS), 1998, pp. 370–378.
[58] , Fast Monte Carlo algorithms for ﬁnding low-rank approximat ions, J. Assoc. Comput.
Mach., 51 (2004), pp. 1025–1041. (electronic).
[59] A. Yu. Garnaev and E. D. Gluskin , The widths of a Euclidean ball , Dokl. Akad. Nauk.
SSSR, 277 (1984), pp. 1048–1052. In Russian.
[60] A. Gittens and J. A. Tropp , Error bounds for random matrix approximation schemes .
Available at arXiv:0911.4108.
[61] G. H. Golub and C. F. van Loan , Matrix Computations , Johns Hopkins Studies in the
Mathematical Sciences, Johns Hopkins Univ. Press, Baltimo re, MD, 3rd ed., 1996.
[62] Y. Gordon , Some inequalities for Gaussian processes and applications , Israel J. Math., 50
(1985), pp. 265–289.
[63] , Gaussian processes and almost spherical sections of convex bodies, Ann. Probab., 16
(1988), pp. 180–188.
[64] S. A. Goreinov, E. E. Tyrtyshnikov, and N. L. Zamarashkin , Theory of pseudo-skeleton
matrix approximations , Linear Algebra Appl., 261 (1997), pp. 1–21.
[65] L. Grasedyck and W. Hackbusch , Construction and arithmetics of H-matrices, Comput-
ing, 70 (2003), pp. 295–334.
[66] L. Greengard and V. Rokhlin , A new version of the fast multipole method for the Laplace
equation in three dimensions , Acta Numer., 17 (1997), pp. 229–269.
[67] M. Gu , 2007. Personal communication.
[68] M. Gu and S. C. Eisenstat , Eﬃcient algorithms for computing a strong rank-revealing Q R
factorization, SIAM J. Sci. Comput., 17 (1996), pp. 848–869.
[69] N. Halko, P.-G. Martinsson, Y. Shkolnisky, and M. Tygert , An algorithm for the prin-
cipal component analysis of large data sets , 2010.
[70] S. Har-Peled, Matrix approximation in linear time . Manuscript. Available at http://valis.

<!-- page 72 -->

72 HALKO, MARTINSSON, AND TROPP
cs.uiuc.edu/~sariel/research/papers/05/lrank/, 2006.
[71] T. Hastie, R. Tibshirani, and J. Friedman , The Elements of Statistical Learning: Data
Mining, Inference, and Prediction , Springer, Berlin, 2nd ed., 2008.
[72] R. A. Horn and C. R. Johnson , Matrix Analysis , Cambridge Univ. Press, Cambridge, 1985.
[73] P. Indyk and R. Motw ani , Approximate nearest neighbors: Toward removing the curse o f
dimensionality, in STOC ’98: Proc. 30th Ann. ACM Symp. Theory of Computing, 1 998,
pp. 604–613.
[74] W. B. Johnson and J. Lindenstrauss , Extensions of Lipschitz mappings into a Hilbert
space, Contemp. Math., 26 (1984), pp. 189–206.
[75] D. R. Karger , Random sampling in cut, ﬂow, and network design problems , Math. Oper.
Res., 24 (1999), pp. 383–413.
[76] , Minimum cuts in near-linear time , J. ACM, 47 (2000), pp. 46–76.
[77] B. S. Ka ˇsin, On the widths of certain ﬁnite-dimensional sets and classes of smooth functions ,
Izv. Akad. Nauk. SSSR Ser. Mat., 41 (1977), pp. 334–351, 478. In Russian.
[78] J. Kleinberg , Two algorithms for nearest neighbor search in high dimensio ns, in STOC ’97:
Proc. 29th ACM Symp. Theory of Computing, 1997, pp. 599–608.
[79] J. Kuczy ´nski and H. Wo ´zniakowski, Estimating the largest eigenvalue by the power and
Lanczos algorithms with a random start , SIAM J. Matrix Anal. Appl., 13 (1992),
pp. 1094–1122.
[80] E. Kushilevitz, R. Ostrovski, and Y. Rabani , Eﬃcient search for approximate nearest
neighbor in high-dimensional spaces , SIAM J. Comput., 30 (2000), pp. 457–474.
[81] D. Le and D. S. Parker , Using randomization to make recursive matrix algorithms pr actical,
J. Funct. Programming, 9 (1999), pp. 605–624.
[82] M. Ledoux , The Concentration of Measure Phenomenon , no. 89 in MSM, American Mathe-
matical Society, Providence, RI, 2001.
[83] M. Ledoux and M. Talagrand , Probability in Banach Spaces: Isoperimetry and Processes ,
Springer, Berlin, 1991.
[84] W. S. Lee, P. L. Bartlett, and R. C. Williamson , Eﬃcient agnostic learning of neural
networks with bounded fan-in , IEEE Trans. Inform. Theory, 42 (1996), pp. 2118–2132.
[85] Z. Leyk and H. Wo ´zniakowski, Estimating the largest eigenvector by Lanczos and polyno-
mial algorithms with a random start , Num. Linear Algebra Appl., 5 (1998), pp. 147–164.
[86] E. Liberty , Accelerated dense random projections , Ph.D. thesis, Computer Science Dept.,
Yale University, New Haven, CT, 2009.
[87] E. Liberty, N. Ailon, and A. Singer , Dense fast random projections and lean Walsh trans-
forms, in APPROX and RANDOM 2008, A. Goel et al., ed., no. 5171 in LNC S, Springer,
Berlin, 2008, pp. 512–522.
[88] E. Liberty, F. F. Woolfe, P.-G. Martinsson, V. Rokhlin, and M. Tygert, Randomized
algorithms for the low-rank approximation of matrices , Proc. Natl. Acad. Sci. USA, 104
(2007), pp. 20167–20172.
[89] M. W. Mahoney and P. Drineas , CUR matrix decompositions for improved data analysis ,
Proc. Natl. Acad. Sci. USA, 106 (2009), pp. 697–702.
[90] P.-G. Martinsson, V. Rokhlin, Y. Shkolnisky, and M. Tygert , ID: A software package
for low-rank approximation of matrices via interpolative d ecompositions, version 0.2 ,
2008.
[91] P.-G. Martinsson, V. Rokhlin, and M. Tygert , A randomized algorithm for the approxi-
mation of matrices , Computer Science Dept. Tech. Report 1361, Yale Univ., New H aven,
CT, 2006.
[92] P.-G. Martinsson, A. Szlam, and M. Tygert , Normalized power iterations for the compu-
tation of SVD . Manuscript., Nov. 2010.
[93] J. Matouˇsek, Lectures on Discrete Geometry , Springer, Berlin, 2002.
[94] F. McSherry , Spectral Methods in Data Analysis , Ph.D. thesis, Computer Science Dept.,
Univ. W ashington, Seattle, W A, 2004.
[95] N. Metropolis and S. Ulam , The Monte Carlo method , J. Amer. Statist. Assoc., 44 (1949),
pp. 335–341.
[96] V. D. Milman , A new proof of A. Dvoretsky’s theorem on cross-sections of co nvex bodies ,
Funkcional. Anal. i Priloˇ zen, 5 (1971), pp. 28–37.
[97] L. Mirsky, Symmetric gauge functions and unitarily invariant norms , Quart. J. Math. Oxford
Ser. (2), 11 (1960), pp. 50–59.
[98] R. Motw ani and P. Raghavan, Randomized Algorithms, Cambridge Univ. Press, Cambridge,
1995.
[99] R. J. Muirhead , Aspects of Multivariate Statistical Theory , Wiley, New York, NY, 1982.
[100] S. Muthukrishnan , Data Streams: Algorithms and Applications , Now Publ., Boston, MA,

<!-- page 73 -->

PROBABILISTIC ALGORITHMS FOR MATRIX APPROXIMATION 73
2005.
[101] D. Needell , Randomized Kaczmarz solver for noisy linear systems , BIT, 50 (2010), pp. 395–
403.
[102] N. H. Nguyen, T. T. Do, and T. D. Tran , A fast and eﬃcient algorithm for low-rank
approximation of a matrix , in STOC ’09: Proc. 41st Ann. ACM Symp. Theory of Com-
puting, 2009.
[103] C.-T. Pan , On the existence and computation of rank-revealing LU facto rizations, Linear
Algebra Appl., 316 (2000), pp. 199–222.
[104] C. H. Papadimitriou, P. Raghavan, H. Tamaki, and S. Vempala , Latent semantic index-
ing: A probabilistic analysis , in Proc. 17th ACM Symp. Principles of Database Systems
(PODS), 1998.
[105] , Latent semantic indexing: A probabilistic analysis , J. Comput. System Sci., 61 (2000),
pp. 217–235.
[106] D. S. Parker and B. Pierce , The randomizing FFT: An alternative to pivoting in Gaussian
elimination, Computer Science Dept. Tech. Report CSD 950037, Univ. Cali fornia at Los
Angeles, 1995.
[107] P. J. Phillips, H. Moon, S.A. Rizvi, and P.J. Rauss , The FERET evaluation methodology
for face recognition algorithms , IEEE Trans. Pattern Anal. Mach. Intelligence, 22 (2000),
pp. 1090–1104.
[108] P. J. Phillips, H. Wechsler, J. Huang, and P. Rauss , The FERET database and evaluation
procedure for face recognition algorithms, Image Vision Comput., 16 (1998), pp. 295–306.
[109] W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery , Numerical
Recipes: The Art of Scientiﬁc Computing , Cambridge University Press, Cambridge,
3rd ed., 2007.
[110] A. Rahimi and B. Recht , Random features for large-scale kernel machines , in Proc. 21st
Ann. Conf. Advances in Neural Information Processing Syste ms (NIPS), 2007.
[111] B. Recht, M. F azel, and P. Parillo , Guaranteed minimum-rank solutions ot matrix equa-
tions via nuclear-norm minimization , SIAM Rev., 52 (2010), pp. 471–501.
[112] V. Rokhlin, A. Szlam, and M. Tygert , A randomized algorithm for principal component
analysis, SIAM J. Matrix Anal. Appl., 31 (2009), pp. 1100–1124.
[113] V. Rokhlin and M. Tygert , A fast randomized algorithm for overdetermined linear leas t-
squares regression, Proc. Natl. Acad. Sci. USA, 105 (2008), pp. 13212–13217.
[114] S. Roweis, EM algorithms for PCA and SPCA , in Proc. 10th Ann. Conf. Advances in Neural
Information Processing Systems (NIPS), MIT Press, 1997, pp . 626–632.
[115] M. Rudelson , Random vectors in the isotropic position , J. Funct. Anal., 164 (1999), pp. 60–
72.
[116] M. Rudelson and R. Vershynin , Sampling from large matrices: An approach through ge-
ometric functional analysis , J. Assoc. Comput. Mach., 54 (2007), pp. Art. 21, 19 pp.
(electronic).
[117] A. F. Ruston , Auerbach’s theorem, Math. Proc. Cambridge Philos. Soc., 56 (1964), pp. 476–
480.
[118] T. Sarl ´os, Improved approximation algorithms for large matrices via r andom projections, in
Proc. 47th Ann. IEEE Symp. Foundations of Computer Science ( FOCS), 2006, pp. 143–
152.
[119] S. Shalev-Shw artz and N. Srebro , Low ℓ1-norm and guarantees on sparsiﬁability , in
ICML/COLT/UAI Sparse Optimization and Variable Selection W orkshop, July 2008.
[120] X. Shen and F. G. Meyer , Low-dimensional embedding of fMRI datasets , Neuroimage, 41
(2008), pp. 886–902.
[121] N. D. Shyamalkumar and K. V aradarajaran, Eﬃcient subspace approximation algorithms ,
in Proc. 18th Ann. ACM–SIAM Symp. Discrete Algorithms (SODA ), 2007, pp. 532–540.
[122] L. Sirovich and M. Kirby , Low-dimensional procedure for the characterization of hum an
faces., J. Optical Soc. Amer. A, 4 (1987), pp. 519–524.
[123] D. Spielman and N. Srivastasa , Graph sparsiﬁcation by eﬀective resistances , in STOC ’08:
Proc. 40th Ann. ACM Symp. Theory of Computing, 2008.
[124] G. Stew art, Accelerating the orthogonal iteration for the eigenvector s of a Hermitian matrix ,
Num. Math., 13 (1969), pp. 362–376. 10.1007/BF02165413.
[125] G. W. Stew art, Perturbation of pseudo-inverses, projections, and linear least squares prob-
lems, SIAM Rev., 19 (1977), pp. 634–662.
[126] , Four algorithms for the eﬃcient computation of truncated pi voted QR approximations
to a sparse matrix , Numer. Math., 83 (1999), pp. 313–323.
[127] , The decompositional approach to matrix computation , Comput. Sci. Eng., (2000),
pp. 50–59.

<!-- page 74 -->

74 HALKO, MARTINSSON, AND TROPP
[128] T. Strohmer and R. Vershynin , A randomized Kaczmarz algorithm with exponential con-
vergence, J. Fourier Anal. Appl., 15 (2009), pp. 262–278.
[129] J. Sun, Y. Xie, H. Zhang, and C. F aloutsos , Less is more: Compact matrix decomposition
for large sparse graphs , Stat. Anal. Data Min., 1 (2008), pp. 6–22.
[130] S. J. Szarek , Spaces with large distance from ℓn
∞ and random matrices , Amer. J. Math., 112
(1990), pp. 899–942.
[131] A. Szlam, M. Maggioni, and R. R. Coifman , Regularization on graphs with function-
adapted diﬀsion processes , J. Mach. Learn. Res., 9 (2008), pp. 1711–1739.
[132] L. N. Trefethen and D. Bau III , Numerical Linear Algebra, SIAM, Philadelphia, PA, 1997.
[133] J. A. Tropp , On the conditioning of random subdictionaries , Appl. Comput. Harmon. Anal.,
25 (2008), pp. 1–24.
[134]
, Improved analysis of the subsampled randomized Hadamard tr ansform, Adv. Adaptive
Data Anal., 3 (2011). To appear. Available at arXiv:1011.1595.
[135] J. von Neumann and H. H. Goldstine , Numerical inverting of matrices of high order , Bull.
Amer. Math. Soc., 53 (1947), pp. 1021–1099.
[136] , Numerical inverting of matrices of high order. II , Proc. Amer. Math. Soc., 2 (1952),
pp. 188–202.
[137] F. Woolfe, E. Liberty, V. Rokhlin, and M. Tygert , A fast randomized algorithm for the
approximation of matrices , Appl. Comp. Harmon. Anal., 25 (2008), pp. 335–366.
