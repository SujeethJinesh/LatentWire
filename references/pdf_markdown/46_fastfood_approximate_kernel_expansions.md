# references/46_fastfood_approximate_kernel_expansions.pdf

<!-- page 1 -->

Fastfood: Approximate Kernel Expansions in Loglinear Time
Quoc Viet Le
Google Research, 1600 Amphitheatre Pky, Mountain View 94043 CA, USA
Tamas Sarlos
Google Strategic Technologies, 1600 Amphitheatre Pky, Mountain View 94043 CA, USA
Alexander J. Smola
Carnegie Mellon University, 5000 Forbes Ave, Pittsburgh 15213 PA, USA
Google Strategic Technologies, 1600 Amphitheatre Pky, Mountain View 94043 CA, USA
October 1, 2018
Abstract
Despite their successes, what makes kernel methods diﬃcult to use in many large scale problems
is the fact that storing and computing the decision function is typically expensive, especially at
prediction time. In this paper, we overcome this diﬃculty by proposing Fastfood, an approxi-
mation that accelerates such computation signiﬁcantly. Key to Fastfood is the observation that
Hadamard matrices, when combined with diagonal Gaussian matrices, exhibit properties similar
to dense Gaussian random matrices. Yet unlike the latter, Hadamard and diagonal matrices are
inexpensive to multiply and store. These two matrices can be used in lieu of Gaussian matri-
ces in Random Kitchen Sinks proposed by Rahimi and Recht (2009) and thereby speeding up
the computation for a large range of kernel functions. Speciﬁcally, Fastfood requires O(n log d)
time and O(n) storage to compute n non-linear basis functions in d dimensions, a signiﬁcant
improvement from O(nd) computation and storage, without sacriﬁcing accuracy.
Our method applies to any translation invariant and any dot-product kernel, such as the
popular RBF kernels and polynomial kernels. We prove that the approximation is unbiased
and has low variance. Experiments show that we achieve similar accuracy to full kernel expan-
sions and Random Kitchen Sinks while being 100x faster and using 1000x less memory. These
improvements, especially in terms of memory usage, make kernel methods more practical for
applications that have large training sets and/or require real-time prediction.
1 Introduction
Kernel methods have proven to be a highly successful technique for solving many problems in
machine learning, ranging from classiﬁcation and regression to sequence annotation and feature
extraction (Boser et al., 1992; Cortes and Vapnik, 1995; Vapnik et al., 1997; Taskar et al., 2004;
Sch¨ olkopf et al., 1998). At their heart lies the idea that inner products in high-dimensional feature
spaces can be computed in implicit form via kernel function k:
k(x, x′) =
⟨
φ(x), φ(x′)
⟩
. (1)
Here φ : X → F is a feature map transporting elements of the observation space X into a possibly
inﬁnite-dimensional feature space F. This idea was ﬁrst used by Aizerman et al. (1964) to show
nonlinear separation. There exists a rich body of literature on Reproducing Kernel Hilbert Spaces
1
arXiv:1408.3060v1  [cs.LG]  13 Aug 2014

<!-- page 2 -->

(RKHS) (Aronszajn, 1944; Wahba, 1990; Micchelli, 1986) and one may show that estimators using
norms in feature space as penalty are equivalent to estimators using smoothness in an RKHS
(Girosi, 1998; Smola et al., 1998a). Furthermore, one may provide a Bayesian interpretation via
Gaussian Processes. See e.g. (Williams, 1998; Neal, 1994; MacKay, 2003) for details.
More concretely, to evaluate the decision function f(x) on an example x, one typically employs
the kernel trick as follows
f(x) = ⟨w, φ(x)⟩ =
⟨ N∑
i=1
αiφ(xi), φ(x)
⟩
=
N∑
i=1
αik(xi, x)
This has been viewed as a strength of kernel methods, especially in the days that datasets consisted
of ten thousands of examples. This is because the Representer Theorem of Kimeldorf and Wahba
(1970) states that such a function expansion in terms of ﬁnitely many coeﬃcients must exist under
fairly benign conditions even whenever the space is inﬁnite dimensional. Hence we can eﬀectively
perform optimization in inﬁnite dimensional spaces. This trick that was also exploited by Sch¨ olkopf
et al. (1998) for evaluating PCA. Frequently the coeﬃcient space is referred to as dual space. This
arises from the fact that the coeﬃcients are obtained by solving a dual optimization problem.
Unfortunately, on large amounts of data, this expansion becomes a signiﬁcant liability for
computational eﬃciency. For instance, Steinwart and Christmann (2008) show that the number of
nonzero αi (i.e., N, also known as the number of “support vectors”) in many estimation problems
can grow linearly in the size of the training set. As a consequence, as the dataset grows, the
expense of evaluating f also grows. This property makes kernel methods expensive in many large
scale problems: there the sample size m may well exceed billions of instances. The large scale
solvers of Fan et al. (2008) and Matsushima et al. (2012) work in primal space to sidestep these
problems, albeit at the cost of limiting themselves to linear kernels, a signiﬁcantly less powerful
function class.
2 Related Work
Numerous methods have been proposed to mitigate this issue. To compare computational cost of
these methods we make the following assumptions:
• We have m observations and access to an O(mβ) with β ≥ 1 algorithm for solving the
optimization problem at hand. In other words, the algorithm is linear or worse. This is a
reasonable assumption — almost all data analysis algorithm need to inspect the data at least
once to draw inference.
• Data has d dimensions. For simplicity we assume that it is dense with density rate ρ, i.e. on
average O(ρd) coordinates are nonzero.
• The number of nontrivial basis functions is O(γm). This is well motivated by Steinwart and
Christmann (2008) and it also follows from the fact that e.g. in regularized risk minimization
the subgradient of the loss function determines the value of the associated dual variable.
• We denote the number of (nonlinear) basis functions by n.
Reduced Set Expansions Burges (1996) focused on compressing function expansions after
the problem was solved by means of reduced-set expansions. That is, one ﬁrst solves the full
optimization problem at O(mβ+1ρd) cost and subsequently one minimizes the discrepancy between
2

<!-- page 3 -->

the full expansion and an expansion on a subset of basis functions. The exponent of mβ+1 arises
from the fact that we need to compute O(m) kernels O(mβ) times. Evaluation of the reduced
function set costs at least O(nρd) operations per instance and O(nρd) storage, since each kernel
function k(xi, ·) requires storage of xi.
Low Rank Expansions Subsequent work by Smola and Sch¨ olkopf (2000); Fine and Scheinberg
(2001) and Williams and Seeger (2001) aimed to reduce memory footprint and complexity by
ﬁnding subspaces to expand functions. The key diﬀerence is that these algorithms reduce the
function space before seeing labels. While this is suboptimal, experimental evidence shows that
for well designed kernels the basis functions extracted in this fashion are essentially as good as
reduced set expansions. This is to be expected. After all, the kernel encodes our prior belief in
which function space is most likely to capture the relevant dependencies between covariates and
labels. These projection-based algorithms generate an n-dimensional subspace:
• Compute the kernel matrix Knn on an n-dimensional subspace at O(n2ρd) cost.
• The matrix Knn is inverted at O(n3) cost.
• For all observations one computes an explicit feature map by projecting data in RKHS onto
the set of n basis vectors via φ(x) = Knn− 1
2 [k(x1, x), . . . , k(xn, x)]. That is, training proceeds
at O(nρmβ + n2m) cost.
• Prediction costs O(nρd) computation and O(nρd) memory, as in reduced set methods, albeit
with a diﬀerent set of basis functions.
Note that these methods temporarily require O(n2) storage during training, since we need to be
able to multiply with the inverse covariance matrix eﬃciently. This allows for solutions to problems
where m is in the order of millions and n is in the order of thousands: for n = 10 4 we need
approximately 1GB of memory to store and invert the covariance matrix. Preprocessing can be
parallelized eﬃciently. Obtaining a minimal set of observations to project on is even more diﬃcult
and only the recent work of Das and Kempe (2011) provides usable performance guarantees for it.
Multipole Methods Fast multipole expansions (Lee and Gray, 2009; Gray and Moore, 2003)
oﬀer one avenue for eﬃcient function expansions whenever the dimensionality of the underlying
space is relatively modest. However, for high dimensions they become computationally intractable
in terms of space partitioning, due to the curse of dimensionality. Moreover, they are typically
tuned for localized basis functions, speciﬁcally the Gaussian RBF kernel.
Random Subset Kernels A promising alternative to approximating an existing kernel function
is to design new ones that are immediately compatible with scalable data analysis. A recent instance
of such work is the algorithm of Davies and Ghahramani (2014) who map observations x into set
membership indicators si(x), where i denotes the random partitioning chosen at iterate i and s ∈ N
indicates the particular set.
While the paper suggests that the algorithm is scalable to large amounts of data, it suﬀers from
essentially the same problem as other feature generation methods insofar as it needs to evaluate set
membership for each of the partitions for all data, hence we have anO(knm) computational cost for
n partitions into k sets on m observations. Even this estimate is slightly optimistic since we assume
that computing the partitions is independent of the dimensionality of the data. In summary, while
the function class is potentially promising, its computational cost considerably exceeds that of the
other algorithms discussed below, hence we do not investigate it further.
3

<!-- page 4 -->

Random Kitchen Sinks A promising alternative was proposed by Rahimi and Recht (2009)
under the moniker of Random Kitchen Sinks . In contrast to previous work the authors attempt
to obtain an explicit function space expansion directly. This works for translation invariant kernel
functions by performing the following operations:
• Generate a (Gaussian) random matrix M of size n × d.
• For each observationx compute M x and apply a nonlinearity ψ to each coordinate separately,
i.e. φi(x) = ψ([M x]i).
The approach requires O(n × d) storage both at training and test time. Training costs O(mβnρd)
operations and prediction on a new observation costs O(nρd). This is potentially much cheaper
than reduced set kernel expansions. The experiments in (Rahimi and Recht, 2009) showed that
performance was very competitive with conventional RBF kernel approaches while providing dra-
matically simpliﬁed code.
Note that explicit spectral ﬁnite-rank expansions oﬀer potentially much faster rates of conver-
gence, since the spectrum decays as fast as the eigenvalues of the associated regularization operator
(Williamson et al., 2001). Nonetheless Random Kitchen Sinks are a very attractive alternative due
to their simple construction and the ﬂexility in synthesizing kernels with predeﬁned smoothness
properties.
Fastfood Our approach hews closely to random kitchen sinks. However, it succeeds at overcoming
their key obstacle — the need to store and to multiply by a random matrix. This way, fastfood,
accelerates Random Kitchen Sinks from O(nd) to O(n log d) time while only requiring O(n) rather
than O(nd) storage. The speedup is most signiﬁcant for large input dimensions, a common case in
many large-scale applications. For instance, a tiny 32x32x3 image in the CIFAR-10 (Krizhevsky,
2009) already has 3072 dimensions, and non-linear function classes have shown to work well for
MNIST (Sch¨ olkopf and Smola, 2002) and CIFAR-10. Our approach relies on the fact that Hadamard
matrices, when combined with Gaussian scaling matrices, behave very much like Gaussian random
matrices. That means these two matrices can be used in place of Gaussian matrices in Random
Kitchen Sinks and thereby speeding up the computation for a large range of kernel functions.
The computational gain is achieved because unlike Gaussian random matrices, Hadamard matrices
admit FFT-like multiplication and require no storage.
We prove that the Fastfood approximation is unbiased, has low variance, and concentrates
almost at the same rate as Random Kitchen Sinks. Moreover, we extend the range of applications
from radial basis functions k(∥x − x′∥) to any kernel that can be written as dot product k(⟨x, x′⟩).
Extensive experiments with a wide range of datasets show that Fastfood achieves similar accuracy to
full kernel expansions and Random Kitchen Sinks while being 100x faster with 1000x less memory.
These improvements, especially in terms of memory usage, make it possible to use kernel methods
even for embedded applications.
Our experiments also demonstrate that Fastfood, thanks to its speedup in training, achieves
state-of-the-art accuracy on the CIFAR-10 dataset (Krizhevsky, 2009) among permutation-invariant
methods. Table 1 summarizes the computational cost of the above algorithms.
Having an explicit function expansion is extremely beneﬁcial from an optimization point of
view. Recent advances in both online (Ratliﬀ et al., 2007) and batch (Teo et al., 2010; Boyd et al.,
2010) subgradient algorithms summarily rely on the ability to compute gradients in the feature
space F explicitly.
4

<!-- page 5 -->

Algorithm CPU Training RAM Training CPU Test RAM Test
Reduced set O(mβ+1ρd + mnρd) O(γmρd) O(nρd) O(nρd)
Low rank O(mβnρd + mn2) O(n2 + nρd) O(nρd) O(nρd)
Random Kitchen Sinks O(mβnρd) O(nd) O(nρd) O(nd)
Fastfood O(mβn log d) O(n) O(n log d) O(n)
Table 1: Computational cost for reduced rank expansions. Eﬃcient algorithms achieve β = 1 and
typical sparsity coeﬃcients are ρ = 0.01.
3 Kernels and Regularization
For concreteness and to allow for functional-analytic tools we need to introduce some machinery
from regularization theory and functional analysis. The derivation is kept brief but we aim to be
self-contained. A detailed overview can be found e.g. in the books of Sch¨ olkopf and Smola (2002)
and Wahba (1990).
3.1 Regularization Theory Basics
When solving a regularized risk minimization problem one needs to choose a penalty on the func-
tions employed. This can be achieved e.g. via a simple norm penalty on the coeﬃcients
f(x) = ⟨w, φ(x)⟩ with penalty Ω[ w] = 1
2 ∥w∥2
2 . (2)
Alternatively we could impose a smoothness requirement which emphasizes simple functions over
more complex ones via
Ω[w] = 1
2 ∥f ∥2
H such as Ω[w] = 1
2
[
∥f ∥2
L2 + ∥∇f ∥2
L2
]
One may show that the choice of feature map φ(x) and RKHS norm ∥·∥H are connected. This is
formalized in the reproducing property
f(x) = ⟨w, φ(x)⟩F = ⟨f, k(x, ·)⟩H . (3)
In other words, inner products in feature space F can be viewed as inner products in the RKHS.
An immediate consequence of the above is that k(x, x′) = ⟨k(x, ·), k(x′, ·)⟩H. It also means that
whenever norms can be written via regularization operator P , we may ﬁnd k as the Greens function
of the operator. That is, whenever ∥f ∥2
H = ∥P f∥2
L2 we have
f(x) = ⟨f, k(x, ·)⟩H =
⣨
f, P†P k(x, ·)
⟩
= ⟨f, δx⟩ . (4)
That is, P†P k(x, ·) as like a delta distribution on f ∈ H . This allows us to identify P†P from k
and vice versa (Smola et al., 1998a; Girosi, 1998; Girosi et al., 1995; Girosi and Anzellotti, 1993;
Wahba, 1990). Note, though, that this need not uniquely identify P , a property that we will be
taking advantage of when expressing a given kernel in terms of global and local basis functions.
For instance, any isometry U with U⊤U = 1 generates an equivalent P′ = U P. In other words,
there need not be a unique feature space representation that generates a given kernel (that said,
all such representations are equivalent).
5

<!-- page 6 -->

3.2 Mercer’s Theorem and Feature Spaces
A key tool is the theorem of Mercer (1909) which guarantees that kernels can be expressed as an
inner product in some Hilbert space.
Theorem 1 (Mercer) Any kernel k : X × X → R satisfying Mercer’s condition
∫
k(x, x′)f(x)f(x′)dxdx′ ≥ 0 for all f ∈ L2(X ) (5)
can be expanded into
k(x, x′) =
∑
j
λjφj(x)φj(x′) with λj ≥ 0 and ⟨φi, φj⟩ = δij. (6)
The key idea of Rahimi and Recht (2008) is to use sampling to approximate the sum in (6). Note
that for trace-class kernels, i.e. for kernels with ﬁnite ∑
j λj we can normalize the sum to mimic a
probability distribution, i.e. we have
k(x, x′) = ∥λ∥1 Eλ
[
φλ(x)φλ(x′)
]
where p(λ) =
{
∥λ∥−1
1 λ if λ ∈ {. . . λj . . .}
0 otherwise (7)
Consequently the following approximation converges for n → ∞ to the true kernel
λi ∼ p(λ) and k(x, x′) ≈ ∥λ∥1
n
n∑
i=1
φλi(x)φλi(x′) (8)
Note that the basic connection between random basis functions was well established, e.g., by Neal
(1994) in proving that the Gaussian Process is a limit of an inﬁnite number of basis functions. A
related strategy can be found in the so-called ‘empirical’ kernel map (Tsuda et al., 2002; Sch¨ olkopf
and Smola, 2002) where kernels are computed via
k(x, x′) = 1
n
n∑
i=1
κ(xi, x)κ(xi, x′) (9)
for xi often drawn from the same distribution as the training data. An explicit expression for this
map is given e.g. in (Smola et al., 1998b). The expansion (8) is possible whenever the following
conditions hold:
1. An inner product expansion of the form (6) is known for a given kernel k.
2. The basis functions φj are suﬃciently inexpensive to compute.
3. The norm ∥λ∥1 exists, i.e., k corresponds to a trace class operator Kreyszig (1989).
Although condition 2 is typically diﬃcult to achieve, there exist special classes of expansions that
are computationally attractive. Speciﬁcally, whenever the kernels are invariant under the action of
a symmetry group, we can use the eigenfunctions of its representation to diagonalize the kernel.
6

<!-- page 7 -->

3.3 Kernels via Symmetry Groups
Of particular interest in our case are kernels with some form of group invariance since in these
cases it is fairly straightforward to identify the basis functions φi(x). The reason is that whenever
k(x, x′) is invariant under a symmetry group transformation of its arguments, it means that we
can ﬁnd a matching eigensystem eﬃciently, simply by appealing to the functions that decompose
according to the irreducible representation of the group.
Theorem 2 Assume that a kernel k : X 2 → R is invariant under the action of a symmetry group
G, i.e. assume that k(x, x′) = k(g ◦ x, g ◦ x′) holds for all g ∈ G . In this case, the eigenfunctions φi
of k can be decomposed according to the irreducible representations of G on k(x, ·). The eigenvalues
within each such representation are identical.
For details see e.g. Berg et al. (1984). This means that knowledge of a group invariance dramatically
simpliﬁes the task of ﬁnding an eigensystem that satisﬁes the Mercer decomposition. Moreover, by
construction unitary representations are orthonormal.
Fourier Basis To make matters more concrete, consider translation invariant kernels
k(x, x′) = k(x − x′, 0). (10)
The matching symmetry group is translation group with the Fourier basis admitting a unitary
irreducible representation. Corresponding kernels can be expanded
k(x, x′) =
∫
z
dz exp (i ⟨z, x⟩) exp
(
−i
⟨
z, x′⟩)
λ(z) =
∫
z
dz exp
(
i
⟨
z, x − x′⟩)
λ(z). (11)
This expansion is particularly simple since the translation group is Abelian. By construction the
function λ(z) is obtained by applying the Fourier transform to k(x, 0) — in this case the above
expansion is simply the inverse Fourier transform. We have
λ(z) = (2π)−d
∫
dx exp (−i ⟨x, z⟩) k(x, 0). (12)
This is a well studied problem and for many kernels we may obtain explicit Fourier expansions.
For instance, for Gaussians it is a Gaussian with the inverse covariance structure. For the Laplace
kernel it yields the damped harmonic oscillator spectrum. That is, good choices of λ are
λ(z) = (2π)−d
2 σde− 1
2σ2∥z∥2
2 (Gaussian RBF Kernel) (13)
λ(z) =
l⨂
j=1
1Ud(z) (Matern Kernel) (14)
Here the ﬁrst follows from the fact that Fourier transforms of Gaussians are Gaussians and the
second equality follows from the fact that the Fourier spectrum of Bessel functions can be expressed
as multiple convolution of the unit sphere. For instance, this includes the Bernstein polynomials
as special case for one-dimensional problems. For a detailed discussion of spectral properties for a
broad range of kernels see e.g. (Smola, 1998).
7

<!-- page 8 -->

Spherical Harmonics Kernels that are rotation invariant can be written as an expansion of
spherical harmonics. (Smola et al., 2001, Theorem 5) shows that dot-product kernels of the form
k(x, x′) = κ(⟨x, x′⟩) can be expanded in terms of spherical harmonics. This provides necessary and
suﬃcient conditions for certain families of kernels. Since Smola et al. (2001) derive an incomplete
characterization involving an unspeciﬁed radial contribution we give a detailed derivation below.
Theorem 3 For a kernel k(x, x′) = κ(⟨x, x′⟩) with x, x′ ∈ Rd and with analytic κ, the eigenfunction
expansion can be written as
k(x, x′) =
∑
n
Ωd−1
N(d, n) λn ∥x∥nx′n∑
j
Yd
n,j
( x
∥x∥
)
Yd
n,j
( x′
∥x′∥
)
(15)
=
∑
n
λn ∥x∥nx′n Ln,d
( ⟨x, x′⟩
∥x∥ ∥x′∥
)
(16)
=
∑
n
N(d, n)
Ωd−1
λn ∥x∥nx′n
∫
Sd
Ln,d
(
∥x∥−1 ⟨x, z⟩
)
Ln,d
(x′−1⟨
x′, z
⟩)
dz (17)
Here Yd
n,j are orthogonal polynomials of degree n on the d-dimensional sphere. Moreover, N(d, n)
denotes the number of linearly independent homogeneous polynomials of degree n in d dimensions,
and Ωd−1 denotes the volume of the d−1 dimensional unit ball. Ln,d denotes the Legendre polynomial
of degree n in d dimensions. Finally, λn denotes the expansion coeﬃcients of κ in terms of Ln,d.
Proof Equality between the two expansions follows from the addition theorem of spherical har-
monics of order n in d dimensions. Hence, we only need to show that for ∥x∥ = ∥x′∥ = 1 the
expansion κ(⟨x, x′⟩) =∑
n λnLn,d(⟨x, x′⟩) holds.
First, observe, that such an expansion is always possible since the Legendre polynomials are
orthonormal with respect to the measure induced by the d − 1 dimensional unit sphere, i.e. with
respect to (1−t2)
d−3
2 . See e.g. (Hochstadt, 1961, Chapter 3) for details. Hence they form a complete
basis for one-dimensional expansions of κ(ξ) in terms of Ln,d(ξ). Since κ is analytic, we can extend
the homogeneous polynomials radially by expanding according to (16). This proves the correctness.
To show that this expansion provides necessary and suﬃcient conditions for positive semidef-
initeness, note that Yd
l,n are orthogonal polynomials. Hence, if we had λn < 0 we could use any
matching Yd
l,n to falsify the conditions of Mercer’s theorem.
Finally, the last equality follows from the fact that
∫
Sd−1
Yd
l,nYd
l′,n = δl,l′, i.e. the functions Yd
l,n
are orthogonal polynomials. Moreover, we use the series expansion of Ln,d that also established
equality between the ﬁrst and second line.
The integral representation of (17) may appear to be rather cumbersome. Quite counterintuitively,
it holds the key to a computationally eﬃcient expansion for kernels depending on ⟨x, x′⟩ only.
This is the case since we may sample from a spherically isotropic distribution of unit vectors z
and compute Legendre polynomials accordingly. As we will see, computing inner products with
spherically isotropic vectors can be accomplished very eﬃciently using a construction described in
Section 4.
Corollary 4 Denote by λn the coeﬃcients obtained by a Legendre polynomial series expansion of
κ(⟨x, x′⟩) and let N(d, n) = (d+n−1)!
n!(d−1)! be the number of linearly independent homogeneous polynomials
of degree n in d variables. Draw zi ∼ Sd−1 uniformly from the unit sphere and draw ni from a
8

<!-- page 9 -->

spectral distribution with p(n) ∝ λnN(d, n). Then
E
[
m−1
m∑
i=1
Lni,d(⟨x, zi⟩)Lni,d(
⟨
x′, zi
⟩
)
]
= κ(
⟨
x, x′⟩
) (18)
In other words, provided that we are able to compute the Legendre polynomials Ln,d eﬃciently,
and provided that it is possible to draw from the spectral distribution of λnN(d, n), we have an
eﬃcient means of computing dot-product kernels.
For kernels on the symmetric group that are invariant under group action, i.e. kernels satisfying
k(x, x′) = k(g ◦x, g◦x′) for permutations, expansions using Young Tableaux can be found in (Huang
et al., 2007). A very detailed discussion of kernels on symmetry groups is given in (Kondor, 2008,
Section 4). However, eﬃcient means of computing such kernels rapidly still remains an open
problem.
3.4 Explicit Templates
In some cases expanding into eigenfunctions of a symmetry group may be undesirable. For instance,
the Fourier basis is decidedly nonlocal and function expansions using it may exhibit undesirable
local deviations, eﬀectively empirical versions of the well-known Gibbs phenomenon. That is, local
changes in terms of observations can have far-reaching global eﬀects on observations quite distant
from the observed covariates.
This makes it desirable to expand estimates in terms of localized basis functions, such as Gaus-
sians, Epanechikov kernels, B-splines or Bessel functions. It turns out that the latter is just as
easily achievable as the more commonplace nonlocal basis function expansions. Likewise, in some
cases the eigenfunctions are expensive to compute and it would be desirable to replace them with
possibly less statistically eﬃcient alternatives that oﬀer cheap computation.
Consequently we generalize the above derivation to general nonlinear function classes dependent
on matrix multiplication or distance computation with respect to spherically symmetric sets of
instances. The key is that the feature map depends on x only via
φz(x) := κ(x⊤z, ∥x∥ , ∥z∥) for x, z ∈ Rd (19)
That is, the feature map depends on x and z only in terms of their norms and an inner product
between both terms. Here the dominant cost of evaluating φz(x) is the inner product x⊤z. All other
operations are O(1), provided that we computed ∥x∥ and ∥z∥ previously as a one-oﬀ operation.
Eq. (19) includes the squared distance as a special case:
∥x − z∥2 = ∥x∥2 + ∥z∥2 − 2x⊤z and (20)
κ(x⊤z, ∥x∥ , ∥z∥) := κ(∥x − z∥2) (21)
Here κ is suitably normalized, such as
∫
dzκ(z) = 1. In other words, we expand x in terms of how
close the observations are to a set of well-deﬁned anchored basis functions. It is clear that in this
case
k(x, x′) :=
∫
dµ(z)κz(x − z)φz(x′ − z) (22)
is a kernel function since it can be expressed as an inner product. Moreover, provided that the
basis functions φz(x) are well bounded, we can use sampling from the (normalized) measure µ(z)
9

<!-- page 10 -->

to obtain an approximate kernel expansion
k(x, x′) = 1
n
n∑
i=1
κ(x − zi)κ(x′ − zi). (23)
Note that there is no need to obtain an explicit closed-form expansion in (22). Instead, it suﬃces
to show that this expansion is well-enough approximated by draws from µ(z).
Gaussian RBF Expansion For concreteness consider the following:
φz(x) = exp
[
− a
2
[
∥x∥2 − 2x⊤z + ∥z∥2
]]
and µ(z) := exp
[
− b
2 ∥z∥2
]
(24)
Integrating out z yields
k(x, x′) ∝ exp
[
− a
2
b
2a + b
[
∥x∥2 +
x′2]
− a2
4a + 2b
x − x′2
]
. (25)
This is a locally weighted variant of the conventional Gaussian RBF kernel, e.g. as described by
Haussler (1999). While this loses its translation invariance, one can easily verify that for b → 0 it
converges to the conventional kernel. Note that the key operation in generating an explicit kernel
expansion is to evaluate ∥zi − x∥ for all i. We will explore settings where this can be achieved for n
locations zi that are approximately random at only O(n log d) cost, where d is the dimensionality
of the data. Any subsequent scaling operation is O(n), hence negligible in terms of aggregate cost.
Finally note that by dividing out the terms related only to ∥x∥ and ∥x′∥ respectively we obtain a
’proper’ Gaussian RBF kernel. That is, we use the following features:
˜φz(x) = exp
[
− a
2
[ 2a
2a + b ∥x∥2 − 2x⊤z + ∥z∥2
]]
(26)
and µ(z) = exp
[
− b
2 ∥z∥2
]
. (27)
Weighting functions µ(z) that are more spread-out than a Gaussian will yield basis function ex-
pansions that are more adapted to heavy-tailed distributions. It is easy to see that such expansions
can be obtained simply by specifying an algorithm to draw z rather than having to express the
kernel k in closed form at all.
Polynomial Expansions One of the main inconveniences in computational evaluation of Corol-
lary 4 is that we need to evaluate the associated Legendre polynomials Ln,d(ξ) directly. This is
costly since currently there are no known O(1) expansions for the associate Legendre polynomials,
although approximate O(1) variants for the regular Legendre polynomials exist (Bogaert et al.,
2012). This problem can be alleviated by considering the following form of polynomial kernels:
k(x, x′) =
∑
p
cp
|Sd−1|
∫
Sd−1
⟨x, v⟩p⟨
x′, v
⟩p dv (28)
In this case we only need the ability to draw from the uniform distribution over the unit sphere
to compute a kernel. The price to be paid for this is that the eﬀective basis function expansion is
rather more complex. To compute it we use the following tools from the theory of special functions.
10

<!-- page 11 -->

• For ﬁxed d ∈ N0 the associated kernel is a homogeneous polynomial of degree d in x and
x′ respectively and it only depends on ∥x∥ , ∥x′∥ and the cosine of the angle θ := ⟨x,x′⟩
∥x∥∥x′∥
between both vectors. This follows from the fact that convex combinations of homogeneous
polynomials remain homogeneous polynomials. Moreover, the dependence on lenghts and θ
follows from the fact that the expression is rotation invariant.
• The following integral has a closed-form solution for b ∈ N and for even a.
∫ 1
−1
xa(1 − x2)
b
2 dx = Γ
(a+1
2
)
Γ
(b+3
2
)
Γ
(a+b+3
2
) (29)
For odd a the integral vanishes, which follows immediately from the dependence on xa and
the symmetric domain of integration [ −1, 1].
• The integral over the unit-sphere Sd−1 ∈ Rd can be decomposed via
∫
Sd−1
f(x)dx =
∫ 1
−1
[∫
Sd−2
f
(
x1, x2
√
1 − x2
1
)
dx2
]
(1 − x1)
d−3
2 dx1 (30)
That is, we decompose x into its ﬁrst coordinate x1 and the remainder x2 that lies on Sd−2
with suitable rescaling by (1−x1)
d−3
2 . Note the exponent of d−3
2 that arises from the curvature
of the unit sphere. See e.g. (Hochstadt, 1961, Chapter 6) for details.
While (28) oﬀers a simple expansion for sampling, it is not immediately useful in terms of describing
the kernel as a function of ⟨x, x′⟩. For this we need to solve the integral in (28). Without loss of
generality we may assume that x = (x1, x2, 0, . . .0) and that x′ = (1, 0, . . .0) with ∥x∥ = ∥x′∥ = 1.
In this case a single summand of (28) becomes
∫
Sd−1
⟨x, v⟩p⟨
x′, v
⟩p dv =
∫
Sd−1
(v1x1 + v2x2)pvp
1dv (31)
=
p∑
i=0
(p
i
)
xp−i
1 xi
2
∫ 1
−1
v2p−i
1 (1 − v2
1)
i+d−3
2 dv1
∫
Sd−2
vi
2dv
=
p∑
i=0
(p
i
)
xp−i
1 xi
2
∫ 1
−1
v2p−i
1 (1 − v2
1)
i+d−3
2 dv1
∫ 1
−1
vi
2(1 − v2
2)
d−4
2 dv2 |Sd−3|
= |Sd−3|
p∑
i=0
(p
i
)
xp−i
1 xi
2
Γ
(
2p−i+1
2
)
Γ
(i+d−1
2
)
Γ
(
2p+d
2
) Γ
(i+1
2
)
Γ
(d−2
2
)
Γ
(i+d−1
2
) (32)
Using the fact that x1 = θ and x2 =
√
1 − θ2 we have the full expansion of (28) via
k(x, x′) =
∑
p
∥x∥px′p cp
|Sd−3|
|Sd−1|
p∑
i=0
θp−i[
1 − θ2]i
2
(p
i
)Γ
(
2p−i+1
2
)
Γ
(i+d−1
2
)
Γ
(
2p+d
2
) Γ
(i+1
2
)
Γ
(d−2
2
)
Γ
(i+d−1
2
)
The above form is quite diﬀerent from commonly used inner-product kernels, such as an inhomo-
geneous polynomial ( ⟨x, x′⟩ + d)p. That said, the computational savings are considerable and the
expansion bears suﬃcient resemblance to warrant its use due to signiﬁcantly faster evaluation.
11

<!-- page 12 -->

4 Sampling Basis Functions
4.1 Random Kitchen Sinks
We now discuss computationally eﬃcient strategies for approximating the function expansions intro-
duced in the previous section, beginning with Random Kitchen Sinks of Rahimi and Recht (2008), as
described in Section 3.2. Direct use for Gaussian RBF kernels yields the following algorithm to ap-
proximate kernel functions by explicit feature construction:
input Scale σ2, n, d
Draw Z ∈ Rn×d with iid entries Zij ∼ N (0, σ−2).
for all x do
Compute empirical feature map via φj(x) = c√n exp(i[Zx]j)
end for
As discussed previously, and as shown by Rahimi and Recht (2009), the associated feature map
converges in expectation to the Gaussian RBF kernel. Moreover, they also show that this conver-
gence occurs with high probability and at the rate of independent empirical averages. While this
allows one to use primal space methods, the approach remains limited by the fact that we need
to store Z and, more importantly, we need to compute Zx for each x. That is, each observation
costs O(n · d) operations. This seems wasteful, given that we are really only multiplying x with a
‘random’ matrix Z, hence it seems implausible to require a high degree of accuracy for Zx.
The above idea can be improved to extend matters beyond a Gaussian RBF kernel and to reduce
the memory footprint in computationally expensive settings. We summarize this in the following
two remarks:
Remark 5 (Reduced Memory Footprint) To avoid storing the Gaussian random matrix Z we
recompute Zij on the ﬂy. Assume that we have access to a random number generator which takes
samples from the uniform distribution ξ ∼ U[0, 1] as input and emits samples from a Gaussian,
e.g. by using the inverse cumulative distribution function z = F−1(ξ). Then we may replace the
random number generator by a hash function via ξij = N−1h(i, j) where N denotes the range of
the hash, and subsequently Zij = F−1(ξij).
Unfortunately this variant is computationally even more costly than Random Kitchen Sinks, its
only beneﬁt being the O(n) memory footprint relative to the O(nd) footprint for random kitchen
sinks. To make progress, a more eﬀective approximation of the Gaussian random matrix Z is
needed.
4.2 Fastfood
For simplicity we begin with the Gaussian RBF case and extend it to more general spectral dis-
tributions subsequently. Without loss of generality assume that d = 2l for some l ∈ N.1 For the
moment assume that d = n. The matrices that we consider instead of Z are parameterized by a
product of diagonal matrices and the Hadamard matrix:
V := 1
σ
√
d
SHG ΠHB. (33)
1If this is not the case, we can trivially pad the vectors with zeros until d = 2l holds.
12

<!-- page 13 -->

Here Π ∈ {0, 1}d×d is a permutation matrix and H is the Walsh-Hadamard matrix.2 S, G and B are
all diagonal random matrices. More speciﬁcally, B has random {±1} entries on its main diagonal,
G has random Gaussian entries, and S is a random scaling matrix. V is then used to compute the
feature map.
The coeﬃcients for S, G, B are computed once and stored. On the other hand, the Walsh-
Hadamard matrix is never computed explicitly. Instead we only multiply by it via the fast
Hadamard transform, a variant of the FFT which allows us to compute Hdx in O(d log d) time.
The Hadamard matrices are deﬁned as follows:
H2 :=
[ 1 1
1 −1
]
and H2d :=
[ Hd Hd
Hd −Hd
]
.
When n > d, we replicate (33) for n/d independent random matrices Vi and stack them via
VT = [ V1, V2, . . . Vn/d]T until we have enough dimensions. The feature map for Fastfood is then
deﬁned as
φj(x) = n− 1
2 exp(i[V x]j). (34)
Before proving that in expectation this transform yields a Gaussian random matrix, let us brieﬂy
verify the computational eﬃciency of the method.
Lemma 6 (Computational Eﬃciency) The features of (34) can be computed at O(n log d) cost
using O(n) permanent storage for n ≥ d.
Proof Storing the matrices S, G, B costs 3n entries and 3 n operations for a multiplication. The
permutation matrix Π costs n entries and n operations. The Hadamard matrix itself requires no
storage since it is only implicitly represented. Furthermore, the fast Hadamard transforms costs
O(n log d) operations to carry out since we have O(d log d) per block and n/d blocks. Computing
the Fourier basis for n numbers is an O(n) operation. Hence the total CPU budget is O(n log d)
and the storage is O(n).
Note that the construction of V is analogous to that of Dasgupta et al. (2011). We will use these
results in establishing a suﬃciently high degree of decorrelation between rows of V . Also note
that multiplying with a longer chain of Walsh-Hadamard matrices and permutations would yield
a distribution closer to independent Gaussians. However, as we shall see, two matrices provide a
suﬃcient amount of decorrelation.
4.3 Basic Properties
Now that we showed that the above operation is fast, let us give some initial indication why it is
also useful and how the remaining matrices S, G, B, Π are deﬁned.
Binary scaling matrix B: This is a diagonal matrix with Bii ∈ {±1} drawn iid from the uniform
distribution over {±1}. The initial HBd− 1
2 acts as an isometry that densiﬁes the input, as
pioneered by Ailon and Chazelle (2009).
Permutation Π: It ensures that the rows of the two Walsh-Hadamard matrices are incoherent
relative to each other. Π can be stored eﬃciently as a lookup table at O(d) cost and it can
be generated by sorting random numbers.
2We conjecture that H can be replaced by any matrix T ∈ Rd×d, such that T /
√
d is orthonormal, maxij |Tij| =
O(1), i.e. T is smooth, and T x can be computed in O(d log d) time. A natural candidate is the Discrete Cosine
Transform (DCT).
13

<!-- page 14 -->

Gaussian scaling matrix G: This is a diagonal matrix whose elements Gii ∼ N (0, 1) are drawn
iid from a Gaussian. The next Walsh-Hadamard matrices H will allow us to ’recycle’ n Gaus-
sians to make the resulting matrix closer to an iid Gaussian. The goal of the preconditioning
steps above is to guarantee that no single Gii can inﬂuence the output too much and hence
provide near-independence.
Scaling matrix S: Note that the length of all rows of HGΠHB are constant as equation (36)
shows below. In the Gaussian case S ensures that the length distribution of the row of V are
independent of each other. In the more general case, one may also adjust the capacity of the
function class via a suitably chosen scaling matrix S. That is, large values in Sii correspond
to high complexity basis functions whereas small Sii relate to simple functions with low total
variation. For the RBF kernel we choose
Sii = si ∥G∥
− 1
2
Frob where p(si) ∝ rd−1e−r2
2 . (35)
Thus si matches the radial part of a normal distribution and we rescale it using the Frobenius
norm of G.
We now analyze the distribution of entries in V .
The rows of HGΠHB have the same length. To compute their length we take
l2 :=
[
HGΠHB(HGΠHB)⊤
]
jj
= [HG2H]jj d =
∑
i
H2
ijG2
iid = ∥G∥2
Frob d (36)
In this we used the fact that H⊤H = d1 and moreover that |Hij| = 1. Consequently, rescaling
the entries by ∥G∥
− 1
2
Frob d− 1
2 yields rows of length 1.
Any given row of HGΠHB is iid Gaussian. Each entry of the matrix
[HGΠHB]ij = Bjj HT
i GΠHj
is zero-mean Gaussian as it consists of a sum of zero-mean independent Gaussian random
variables. Sign changes retain Gaussianity. Also note that Var [ HGΠHB]ij = d. B ensures
that diﬀerent entries in [HGΠHB]i· have 0 correlation. Hence they are iid Gaussian (checking
ﬁrst and second order moments suﬃces).
The rows of SHG ΠHB are Gaussian. Rescaling the length of a Gaussian vector using (35)
retains Gaussianity. Hence the rows of SHG ΠHB are Gaussian, albeit not independent.
Lemma 7 The expected feature map recovers the Gaussian RBF kernel, i.e.,
ES,G,B,Π
[
φ(x)
⊤
φ(x′)
]
= e−∥x−x′∥2
2σ2 .
Moreover, the same holds for V′ = 1
σ
√
d HGΠHB .
Proof We already proved that any given row in V is a random Gaussian vector with distribution
N (0, σ−2Id), hence we can directly appeal to the construction of Rahimi and Recht (2008). This
also holds for V′. The main diﬀerence being that the rows in V′ are considerably more correlated.
Note that by assembling several d × d blocks to obtain an n × d matrix this property is retained,
since each block is drawn independently.
14

<!-- page 15 -->

4.4 Changing the Spectrum
Changing the kernel from a Gaussian RBF to any other radial basis function kernel is straight-
forward. After all, HGΠHB provides a approximately spherically uniformly distributed random
vectors of the same length. Rescaling each direction of projection separately costs only O(n) space
and computation. Consequently we are free to choose diﬀerent coeﬃcients Sii rather than (35).
Instead, we may use
Sii ∼ c−1rd−1A−1
d−1λ(r).
Here c is a normalization constant and λ(r) is the radial part of the spectral density function of
the regularization operator associated with the kernel.
A key advantage over a conventional kernel approach is that we are not constrained by the
requirement that the spectral distributions be analytically computable. Even better, we only need
to be able to sample from the distribution rather than compute its Fourier integral in closed form.
For concreteness consider the Matern kernel. Its spectral properties are discussed, e.g. by
Sch¨ olkopf and Smola (2002). In a nutshell, given data in Rd denote by ν := d
2 a dimension
calibration and let t ∈ N be a ﬁxed parameter determining the degree of the Matern kernel (which
is usually determined experimentally). Moreover, denote by Jν(r) the Bessel function of the ﬁrst
kind of order ν. Then the kernel given by
k(x, x′) :=
x − x′−tν Jt
ν(
x − x′) for n ∈ N (37)
has as its associated Fourier transform
F k(ω) =
n⨂
i=1
χSd[ω].
Here χSd is the characteristic function on the unit ball in Rd and⨂denotes convolution. In words,
the Fourier transform of k is the n-fold convolution of χSd. Since convolutions of distributions
arise from adding independent random variables this yields a simple algorithm for computing the
Matern kernel:
for each Sii do
Draw t iid samples ξi uniformly from Sd.
Use Sii =
∑t
i=1 ξi
 as scale.
end for
While this may appear costly, it only needs to be carried out once at initialization time and it
allows us to sidestep computing the convolution entirely. After that we can store the coeﬃcients
Sii. Also note that this addresses a rather surprising problem with the Gaussian RBF kernel —
in high dimensional spaces draws from a Gaussian are strongly concentrated on the surface of a
sphere. That is, we only probe the data with a ﬁxed characteristic length. The Matern kernel, on
the other hand, spreads its capacity over a much larger range of frequencies.
4.5 Inner Product Kernels
We now put Theorem 3 and Corollary 4 to good use. Recall that the latter states that any dot-
product kernel can be obtained by taking expectations over draws from the degree of corresponding
Legendre polynomial and over a random direction of reference, as established by the integral rep-
resentation of (17).
15

<!-- page 16 -->

It is understood that the challenging part is to draw vectors uniformly from the unit sphere.
Note, though, that it is this very operation that Fastfood addresses by generating pseudo-Gaussian
vectors. Hence the modiﬁed algorithm works as follows:
Initialization
for j = 1 to n/d do
Generate matrix block Vj ← ∥G∥−1
Frob d− 1
2 HGΠHB implicitly as per (33).
Draw degrees ni from p(n) ∝ λnN(d, n).
end for
Computation
r ← ∥x∥ and t ← V x
for i = 1 to n do
ψi ← Lni,d(ti) = rniLni,d(ti/r)
end for
Note that the equality Lni,d(ti) = rniLni,d(ti/r) follows from the fact that Lni,d is a homogeneous
polynomial of degree ni. The second representation may sometimes be more eﬀective for reasons of
numerical stability. As can be seen, this relies on access to eﬃcient Legendre polynomial computa-
tion. Recent work of Bogaert et al. (2012) shows that (quite surprisingly) this is possible in O(1)
time for Ln(t) regardless of the degree of the polynomial. Extending these guarantees to associated
Legendre polynomials is unfortunately rather nontrivial. Hence, a direct expansion in terms of
⟨x, v⟩d, as discussed previously, or brute force computation may well be more eﬀective.
Remark 8 (Kernels on the Symmetric Group) We conclude our reasoning by providing an
extension of the above argument to the symmetric group. Clearly, by treating permutation matrices
Π ∈ Cn as d × d dimensional vectors, we can use them as inputs to a dot-product kernel. Subse-
quently, taking inner products with random reference vectors of unit length yields kernels which are
dependent on the matching between permutations only.
5 Analysis
The next step is to show that the feature map is well behaved also in terms of decorrelation between
rows of V . We focus on Gaussian RBF kernels in this context.
5.1 Low Variance
When using random kitchen sinks, the variance of the feature map is at least O(1/n) since we draw
n samples iid from the space of parameters. In the following we show that the variance of fastfood
is comparable, i.e. it is also O(1/n), albeit with a dependence on the magnitude of the magnitude
of the inputs of the feature map. This guarantee matches empirical evidence that both algorithms
perform equally well as the exact kernel expansion.
For convenience, since the kernel values are real numbers, let us simplify terms and rewrite the
inner product in terms of a sum of cosines. Trigonometric reformulation yields
1
n
∑
j
¯φj(x)φj(x′) = 1
n
∑
j
cos [V (x − x′)]j for V = d− 1
2 HGΠHB. (38)
We begin the analysis with a general variance bound for square matrices V ∈ Rd×d. The extension
to n/d iid drawn stacked matrices is deferred to a subsequent corollary.
16

<!-- page 17 -->

Theorem 9 Let v = x−x′
σ and let ψj(v) = cos[V v]j denote the estimate of the kernel value arising
from the jth pair of random features for each j ∈ {1 . . . d}. Then for each j we have
Var [ψj(v)] = 1
2
(
1 − e−∥v∥2)2
and Var


d∑
j=1
ψj(v)

 ≤ d
2
(
1 − e−∥v∥2)2
+ dC(∥v∥) (39)
where C(α) = 6α4
[
e−α2
+ α2
3
]
depends on the scale of the argument of the kernel.
Proof Since for any random variable Xj we can decompose Var (∑Xj) = ∑
j,t Cov(Xj, Xt) our
goal is to compute
Cov(ψ(v), ψ(v)) = E
[
ψ(v)ψ(v)⊤
]
− E [ψ(v)] E [ψ(v)]⊤ .
We decompose V v into a sequence of terms w = d− 1
2 HBv and u = Πw and z = HGu. Hence we
have ψj(v) = cos(zj). Note that ∥u∥ = ∥v∥ since by construction d− 1
2 H, B and Π are orthonormal
matrices.
Gaussian Integral Now condition on the value of u. Then it follows that Cov( zj, zt|u) =
ρjt(u) ∥v∥2 = ρ(u) ∥v∥2 where ρ ∈ [−1, 1] is the correlation of zj and zt. By symmetry all ρij
are identical.
Observe that the marginal distribution of each zj is N (0, ∥v∥2) since each element of H is ±1.
Thus the joint distribution of zj and zt is a Gaussian with mean 0 and covariance
Cov [[zj, zt]|u] =
[ 1 ρ
ρ 1
]
∥v∥2 = L · LT where L =
[ 1 0
ρ
√
1 − ρ2
]
∥v∥
is its Cholesky factor. Hence
Cov(ψj(v), ψt(v)|u) = Eg [cos([Lg]1) cos([Lg]2)] − Eg[cos(zj)]Eg[cos(zt)] (40)
where g ∈ R2 is drawn from N (0, 1). From the trigonometric identity
cos(α) cos(β) = 1
2 [cos(α − β) + cos(α + β)]
it follows that we can rewrite
Eg [cos([Lg]1) cos([Lg]2)] = 1
2Eh [cos(a−h) + cos(a+h)] = 1
2
[
e− 1
2a2
− + e− 1
2a2
+
]
where h ∼ N (0, 1) and a2
± =
L⊤[1, ±1]
2 = 2 ∥v∥2 (1 ± ρ). That is, after applying the addition
theorem we explicitly computed the now one-dimensional Gaussian integrals.
We compute the ﬁrst moment analogously. Since by construction zj and zj have zero mean and
variance ∥v∥2 we have that
Eg[cos(zj)]Eg[cos(zt)] = Eh[cos(∥v∥ h)]2 = e−∥v∥2
Combining both terms we obtain that the covariance can be written as
Cov[ψj(v), ψt(v)|u] = e−∥v∥2[
cosh[∥v∥2 ρ] − 1
]
(41)
17

<!-- page 18 -->

Taylor Expansion To prove the ﬁrst claim note that here j = t, since we are computing the
variance of a single feature. Correspondingly ρ(u) = 1. Plugging this into (41) and simplifying
terms yields the ﬁrst claim of (39).
To prove our second claim, observe that from the Taylor series of cosh with remainder in
Lagrange form, it follows that there exists η ∈ [− ∥v∥2 |ρ|, ∥v∥2 |ρ|] such
cosh(∥v∥2 ρ) − 1 = 1
2 ∥v∥4 ρ2 + 1
6 sinh(η) ∥v∥6 ρ3
≤ 1
2 ∥v∥4 ρ2 + 1
6 sinh(∥v∥2) ∥v∥6 ρ3
≤ ρ2 ∥v∥4 B(∥v∥),
where B(∥v∥) = 1
2 + sinh(∥v∥2)∥v∥2
6 . Plugging this into (41) yields
Cov[ψj(v), ψt(v)|u] ≤ ρ2 ∥v∥4 B(∥v∥).
Bounding Eu[ρ2] Note that the above is still conditioned on u. What remains is to bound
Eu[ρ2], which is small if E[∥u∥4
4] is small. The latter is ensured by HB, which acts as a randomized
preconditioner: Since G is diagonal and Gii ∼ N (0, 1) independently we have
Cov[z, z] = Cov[HGu, HGu] = H Cov[Gu, Gu]H⊤ = HE
[
diag(u2
1, . . . , u2
d)
]
H⊤.
Recall that Hij = Hji are elements of the Hadamard matrix. For ease of notation ﬁx j ̸= t and let
T = {i ∈ [1..d] : Hji = Hti} be the set of columns where the jth and the tth row of the Hadamard
matrix agree. Then
Cov(zj, zt|u) =
d∑
i=1
HjiHtiu2
i =
∑
i∈T
u2
i −
∑
i/∈T
u2
i = 2
∑
i∈T
u2
i −
d∑
i=1
u2
i = 2
∑
i∈T
u2
i − ∥v∥2 .
Now recall that u = Πw and that Π is a random permutation matrix. Therefore ui = wπ(i) for a
randomly chosen permutation π and thus the distribution of ρ ∥v∥2 and 2∑
i∈R w2
i − ∥v∥2 where R
is a randomly chosen subset of size d
2 in {1 . . . d} are the same. Let us ﬁx (condition on) w. Since
2ER
[∑
i∈R w2
i
]
= ∥v∥2 we have that
ER
[
ρ2 ∥v∥4
]
= 4ER


[∑
i∈R
w2
i
] 2
 − ∥v∥4 .
Now let δi = 1 if i ∈ R and 0 otherwise. Note that Eδ(δi) = 1
2 and if j ̸= k then Eδ(δiδk) ≤ 1
4 as δi
are (mildly) negatively correlated. From ∥w∥ = ∥v∥ it follows that
ER


[∑
i∈R
w2
i
] 2
 = Eδ


[ d∑
i=1
δiw2
i
] 2
 = Eδ

∑
i̸=k
δiδkw2
i w2
k

 + Eδ
∑
i
δiw4
i ≤ ∥v∥4
4 + ∥w∥4
4
2 .
From the two equations above it follows that
ER
[
ρ2 ∥v∥4
]
≤ 2∥w∥4
4. (42)
18

<!-- page 19 -->

Bounding the fourth moment of ∥w∥ Let bi = Bii be the independent ±1 random variables
of B. Using the fact wi = 1√
d
∑d
t=1 Hitbtvt and that bi are independent with similar calculations to
the above it follows that
Eb
[
w4
i
]
≤ 6
d2

v4
i +
∑
t̸=j
v2
t v2
j

 and hence Eb
[
∥w∥4
4
]
≤ 6
d ∥v∥4
2
which shows that 1√
d HB acts as preconditioner that densiﬁes the input. Putting it all together we
have
∑
j̸=t
Eu [Cov(ψj(v), ψt(v)|u)] ≤ d2e−∥v∥2
B(∥v∥)ER[ρ2 ∥v∥4] ≤ 12de−∥v∥2
B(∥v∥) ∥v∥4
= 6d ∥v∥4
(
e−∥v∥2
+ ∥v∥2/3
)
Combining the latter with the already proven ﬁrst claim establishes the second claim.
Corollary 10 Denote by V, V′ Gauss-like matrices of the form
V = σ−1d− 1
2 HGΠHB and V′ = σ−1d− 1
2 SHG ΠHB. (43)
Moreover, let C(α) = 6 α4
[
e−α2
+ α2
3
]
be a scaling function. Then for the feature maps obtained
by stacking n/d iid copies of either V or V′ we have
Var
[
φ′(x)⊤φ′(x′)
]
≤ 2
n
(
1 − e−∥v∥2)2
+ 1
n C(∥v∥) where v = σ−1(x − x′). (44)
Proof Since φ′(x)⊤φ′(x′) is the average of n/d independent estimates, each arising from 2 d fea-
tures. Hence we can appeal to Theorem 9 for a single block, i.e. when n = d. The near-identical
argument for V is omitted.
5.2 Concentration
The following theorem shows that for a given error probability δ, the approximation error of a d × d
block of Fastfood is at most logarithmically larger than the error of Random Kitchen Sinks. That
is, it is only logarithmically weaker. We believe that this bound is pessimistic and could be further
improved with considerable analytic eﬀort. That said, the O(m− 1
2 ) approximation guarantees to
the kernel matrix are likely rather conservative when it comes to generalization performance, as we
found in experiments. In other words, we found that the algorithm works much better in practice
than in theory, as conﬁrmed in Section 6. Nonetheless it is important to establish tail bounds, not
to the least since this way improved guarantees for random kitchen sinks also immediately beneﬁt
fastfood.
Theorem 11 For all x, x′ ∈ Rd let ˆk(x, x′) =∑d
j=1 cos(d− 1
2 [HGΠHB(x − x′)/σ]j)/d denote our
estimate of the RBF kernel k(x, x′) that arises from a d × d block of Fastfood. Then we have that
P
[⏐⏐⏐ˆk(x, x′) − k(x, x′)
⏐⏐⏐ ≥ 2σ−1d− 1
2
x − x′√
log(2/δ) log(2d/δ)
]
≤ 2δ for all δ > 0
19

<!-- page 20 -->

Theorem 11 demonstrates almost sub-Gaussian convergence Fastfood kernel for a ﬁxed pair of
points x, x′. A standard ϵ-net argument then shows uniform convergence over any compact set
of Rd with bounded diameter (Rahimi and Recht, 2008, Claim 1). Also, the small error of the
approximate kernel does not signiﬁcantly perturb the solution returned by wide range of learning
algorithms (Rahimi and Recht, 2008, Appendix B) or aﬀect their generalization error.
Our key tool is concentration of Lipschitz continuous functions under the Gaussian mea-
sure Ledoux (1996). We ensure that Fastfood construct has a small Lipschitz constant using
the following lemma, which is due to Ailon and Chazelle (2009).
Lemma 12 (Ailon and Chazelle, 2009) Let x ∈ Rd and t > 0. Let H, B ∈ Rd×d denote the
Hadamard and the binary random diagonal matrices respectively in our construction. Then for any
δ > 0 we have that
P
[
∥HBx ∥∞ ≥ ∥x∥2
√
2 log 2d/δ
]
≤ δ (45)
In other words, with high probability, the largest elements of d− 1
2 HBx are with high probability
no larger than what one could expect if all terms were of even size as per the ∥x∥2 norm.
To use concentration of the Gaussian measure we need Lipschitz continuity. We refer to a
function f : Rd → R as Lipschitz continuous with constant L if for all x, y ∈ Rd it holds that
|f(x) − f(y)| ≤ L ∥x − y∥2. Then the following holds (Ledoux, 1996, Inequality 2.9):
Theorem 13 Assume that f : Rd → R is Lipschitz continuous with constant L and let g ∼ N (0, 1)
be drawn from a d-dimensional Normal distribution. Then we have
P [|f(g) − Eg [f(g)]| ≥ t] ≤ 2e− t2
2L2 . (46)
Proof [Theorem 11] Since both k and ˆk are shift invariant we set v = σ(x − x′) and write
k(v) = k(x, x′) and ˆk(v) = ˆk(x, x′) to simplify the notation. Set u = Πd− 1
2 HBv , and z = HGu
and deﬁne
f(G, Π, B) = d−1
d∑
j=1
cos(zj).
Observe that Lemma 7 implies EG,Π,B [f(G, Π, B)] = k(v). Therefore it is suﬃcient to prove that
f(G, Π, B) concentrates around its mean. We will accomplish this by showing that f is Lipschitz
continuous as a function of G for most Π and B. For a ∈ Rd let
h(a) = d−1
d∑
j=1
cos(aj). (47)
Using the fact that cosine is Lipschitz continuous with constant 1 we observe that for any pair of
vectors a, b ∈ Rd it holds that
|h(a) − h(b)| ≤ d−1
d∑
j=1
| cos(aj) − cos(bj)| ≤ d−1 ∥a − b∥1 ≤ d− 1
2 ∥a − b∥2 . (48)
For any vector g ∈ Rd let diag(g) ∈ Rd×d denote the diagonal matrix whose diagonal is g. Observe
that for any pair of vectors g, g′ ∈ Rd we have that
Hdiag(g)u − Hdiag(g′)u

2 ≤ ∥H∥2
diag(g − g′)u

2
20

<!-- page 21 -->

Let G = Diag( g) in the Fastfood construct and recall the deﬁnition of function h as in (47).
Combining the above inequalities for any pair of vectors g, g′ ∈ Rd yields the following bound
|h(HDiag(g)u)) − h(HDiag(g′)u)| ≤ ∥ u∥∞
g − g′
2 . (49)
From u = Πd− 1
2 HBv and ∥Πw∥∞ = ∥w∥∞ combined with Lemma 12 it follows that
∥u∥∞ ≤ ∥v∥2
√
2
d log 2d
δ (50)
holds with probability at least 1 − δ, where the probability is over the choice of B.3 Now condition
on (50). From inequality (49) we have that the function
g → h(Hdiag(g)u) = f(diag(g), Π, B)
is Lipschitz continuous with Lipschitz constant
L = ∥v∥2
√
2
d log 2d
δ . (51)
Hence from Theorem 13 and from the independently chosen Gjj ∼ N (0, 1) it follows that
PG
[
|f(G, Π, B) − k(v)| ≥
√
2L log 2/δ
]
≤ δ. (52)
Combining inequalities (51) and (52) with the union bound concludes the proof.
6 Experiments
In the following we assess the performance of Random Kitchen Sinks and Fastfood. The results
show that Fastfood performs as well as Random Kitchen Sinks in terms of accuracy. Fastfood,
however, is orders of magnitude faster and exhibits a signiﬁcantly lower memory footprint. For
simplicity, we focus on penalized least squares regression since in this case we are able to compute
exact solutions and are independent of any other optimization algorithms. We also benchmark
Fastfood on CIFAR-10 (Krizhevsky, 2009) and observe that it achieves state-of-the-art accuracy.
This advocates for the use of non-linear expansions even when d is large.
6.1 Approximation quality
We begin by investigating how well our features can approximate the exact kernel computation as
n increases. For that purpose, we uniformly sample 4000 vectors from [0 , 1]10. We compare the
exact kernel values to Random Kitchen Sinks and Fastfood.
The results are shown in Figure 1. We used the absolute diﬀerence between the exact kernel and
the approximation to quantify the error (the relative diﬀerence also exhibits similar behavior and
is thus not shown due to space constraints). The results are presented as averages, averaging over
4000 samples. As can be seen, as n increases, both Random Kitchen Sinks and Fastfood converge
quickly to the exact kernel values. Their performance is indistinguishable, as expected from the
construction of the algorithm.
3Note that in contrast to Theorem 9, the permutation matrix Π does not play a role in the proof of Theorem 11.
21

<!-- page 22 -->

Figure 1: Kernel approximation errors of diﬀerent methods with respect to number of basis functions
n.
Note, though, that ﬁdelity in approximating k(x, x′) does not imply generalization performance
(unless the bounds are very tight). To assess this, we carried out experiments on all regression
datasets from the UCI repository (Frank and Asuncion, 2010) that are not too tiny, i.e., that
contained at least 4 , 000 instances.
We investigate estimation accuracy via Gaussian process regression (Rasmussen and Williams,
2006) using approximated kernel computation methods and we compare this to exact kernel com-
putation whenever the latter is feasible. For completeness, we compare the following methods:
Exact RBF uses the exact Gaussian RBF kernel, that is k(x, x′) = exp
(
− ∥x − x′∥2 /2σ2
)
. This
is possible, albeit not practically desirable due to its excessive cost, on all but the largest
datasets where the kernel matrix does not ﬁt into memory.
Nystrom uses the Nystrom approximation of the kernel matrix (Williams and Seeger, 2001).
These methods have received recent interest due to the improved approximation guarantees
of Jin et al. (2011) which indicate that approximation rates faster thanO(n− 1
2 ) are achievable.
Hence, theoretically, the Nystrom method could have a signiﬁcant accuracy advantage over
Random Kitchen Sinks and Fastfood when using the same number of basis functions, albeit
at exponentially higher cost of O(d) vs. O(log d) per function. We set n = 2, 048 to retain a
computationally feasible feature projection.
Random Kitchen Sinks uses the the Gaussian random projection matrices of Rahimi and Recht
(2008). As before, we use n = 2, 048 basis functions. Note that this is a rather diﬃcult setting
for Random Kitchen Sinks relative to the Nystrom decomposition, since the basis functions
22

<!-- page 23 -->

obtained in the latter are arguably better in terms of approximating the kernel. Hence, one
would naively expect slightly inferior performance from Random Kitchen Sinks relative to
direct Hilbert Space methods.
Fastfood (Hadamard features) uses the random matrix given by SHG ΠHB, again with n = 2, 048
dimensions. Based on the above reasoning one would expect that the performance of the
Hadamard features is even weaker than that of Random Kitchen Sinks since now the basis
functions are no longer even independently drawn from each other.
FFT Fastfood (Fourier features) uses a variant of the above construction. Instead of combining
two Hadamard matrices, a permutation and Gaussian scaling, we use a permutation in con-
junction with a Fourier Transform matrix F : the random matrix given by V = ΠF B. The
motivation is the Subsampled Random Fourier Transform, as described by Tropp (2010): by
picking a random subset of columns from a (unitary) Fourier matrix, we end up with vec-
tors that are almost spatially isotropic, albeit with slightly more dispersed lengths than in
Fastfood. We use this heuristic for comparison purposes.
Exact Poly uses the exact polynomial kernel, that is k(x, x′) = (⟨z, x⟩ + 1)d, with d = 10. Similar
to the case of Exact RBF, this method is only practical on small datasets.
Fastfood Poly uses the Fastfood trick via Spherical Harmonics to approximate the polynomial
kernels.
The results of the comparison are given in Table 3. As can be seen, and contrary to the intu-
ition above, there is virtually no diﬀerence between the exact kernel, the Nystrom approximation,
Random Kitchen Sinks and Fastfood. In other words, Fastfood performs just as well as the exact
method, while being substantially cheaper to compute. Somewhat surprisingly, the Fourier features
work very well. This indicates that the concentration of measure eﬀects impacting Gaussian RBF
kernels may actually be counterproductive at their extreme. This is corroborated by the good
performance observed with the Matern kernel.
Table 2: Speed and memory improvements of Fastfood relative to Random Kitchen Sinks
d n Fastfood RKS Speedup RAM
1, 024 16 , 384 0.00058s 0.0139s 24x 256x
4, 096 32 , 768 0.00137s 0.1222s 89x 1024x
8, 192 65 , 536 0.00269s 0.5351 199x 2048x
In Figure 2, we show regression performance as a function of the number of basis functions
n on the CPU dataset. As is evident, it is necessary to have a large n in order to learn highly
nonlinear functions. Interestingly, although the Fourier features do not seem to approximate the
Gaussian RBF kernel, they perform well compared to other variants and improve as n increases.
This suggests that learning the kernel by direct spectral adjustment might be a useful application
of our proposed method.
6.2 Speed of kernel computations
In the previous experiments, we observe that Fastfood is on par with exact kernel computation,
the Nystrom method, and Random Kitchen Sinks. The key point, however, is to establish whether
the algorithm oﬀers computational savings.
23

<!-- page 24 -->

Figure 2: Test RMSE on CPU dataset with respect to the number of basis functions. As number
of basis functions increases, the quality of regression generally improves.
For this purpose we compare Random Kitchen Sinks using Eigen4 and our method using Spiral5.
Both are highly optimized numerical linear algebra libraries in C++. We are interested in the time
it takes to go from raw features of a vector with dimension d to the label prediction of that vector.
On a small problem with d = 1, 024 and n = 16, 384, performing prediction with Random Kitchen
Sinks takes 0.07 seconds. Our method is around 24x faster, taking only 0.003 seconds to compute
the label for one input vector. The speed gain is even more signiﬁcant for larger problems, as is
evident in Table 2. This conﬁrms experimentally the O(n log d) vs. O(nd) runtime and the O(n) vs.
O(nd) storage of Fastfood relative to Random Kitchen Sinks. In other words, the computational
savings are substantial for large input dimensionality d.
6.3 Random features for CIF AR-10
To understand the importance of nonlinear feature expansions for a practical application, we bench-
marked Fastfood, Random Kitchen Sinks on the CIFAR-10 dataset Krizhevsky (2009) which has
50,000 training images and 10,000 test images. Each image has 32x32 pixels and 3 channels
(d = 3072). In our experiments, linear SVMs achieve 42.3% accuracy on the test set. Non-linear
expansions improve the classiﬁcation accuracy signiﬁcantly. In particular, Fastfood FFT (“Fourier
features”) achieve 63.1% while Fastfood (“Hadamard features”) and Random Kitchen Sinks achieve
62.4% with an expansion of n = 16, 384. These are also best known classiﬁcation accuracies using
permutation-invariant representations on this dataset. In terms of speed, Random Kitchen Sinks
is 5x slower (in total training time) and 20x slower (in predicting a label given an image) compared
to both Fastfood and and Fastfood FFT. This demonstrates that non-linear expansions are needed
4http://eigen.tuxfamily.org/index.php?title=Main_Page
5http://spiral.net
24

<!-- page 25 -->

even when the raw data is high-dimensional, and that Fastfood is more practical for such problems.
In particular, in many cases, linear function classes are used because they provide fast training
time, and especially test time, but not because they oﬀer better accuracy. The results on CIFAR-10
demonstrate that Fastfood can overcome this obstacle.
7 Summary
We demonstrated that it is possible to compute n nonlinear basis functions in O(n log d) time, a
signiﬁcant speedup over the best competitive algorithms. This means that kernel methods become
more practical for problems that have large datasets and/or require real-time prediction. In fact,
Fastfood can be used to run on cellphones because not only it is fast, but it also requires only a
small amount of storage.
Note that our analysis is not limited to translation invariant kernels but it also includes inner
product formulations. This means that for most practical kernels our tools oﬀer an easy means of
making kernel methods scalable beyond simple subspace decomposition strategies. Extending our
work to other symmetry groups is subject to future research. Also note that fast multiplications with
near-Gaussian matrices are a key building block of many randomized algorithms. It remains to be
seen whether one could use the proposed methods as a substitute and reap signiﬁcant computational
savings.
References
N. Ailon and B. Chazelle. The fast johnson-lindenstrauss transform and approximate nearest
neighbors. SIAM Journal on Computing , 39(1):302–322, 2009.
M. A. Aizerman, A. M. Braverman, and L. I. Rozono´ er. Theoretical foundations of the potential
function method in pattern recognition learning. Autom. Remote Control, 25:821–837, 1964.
N. Aronszajn. La th´ eorie g´ en´ erale des noyaux r´ eproduisants et ses applications.Proc. Cambridge
Philos. Soc., 39:133–153, 1944.
C. Berg, J. P. R. Christensen, and P. Ressel. Harmonic Analysis on Semigroups . Springer, New
York, 1984.
I. Bogaert, B. Michiels, and J. Fostier. O(1) computation of legendre polynomials and gauss–
legendre nodes and weights for parallel computing. SIAM Journal on Scientiﬁc Computing , 34
(3):C83–C101, 2012.
B. Boser, I. Guyon, and V. Vapnik. A training algorithm for optimal margin classiﬁers. In D. Haus-
sler, editor, Proc. Annual Conf. Computational Learning Theory, pages 144–152, Pittsburgh, PA,
July 1992. ACM Press.
S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical
learning via the alternating direction method of multipliers. Foundations and Trends in Machine
Learning, 3(1):1–123, 2010.
C. J. C. Burges. Simpliﬁed support vector decision rules. In L. Saitta, editor, Proc. Intl. Conf.
Machine Learning, pages 71–77, San Mateo, CA, 1996. Morgan Kaufmann Publishers.
C. Cortes and V. Vapnik. Support vector networks. Machine Learning, 20(3):273–297, 1995.
25

<!-- page 26 -->

A. Das and D. Kempe. Submodular meets spectral: Greedy algorithms for subset selection, sparse
approximation and dictionary selection. In L. Getoor and T. Scheﬀer, editors, Proceedings of the
28th International Conference on Machine Learning, ICML , pages 1057–1064. Omnipress, 2011.
A. Dasgupta, R. Kumar, and T. Sarl´ os. Fast locality-sensitive hashing. In Proceedings of the
17th ACM SIGKDD international conference on Knowledge discovery and data mining , pages
1073–1081. ACM, 2011.
A. Davies and Z. Ghahramani. The random forest kernel and other kernels for big data from
random partitions. arXiv preprint arXiv:1402.4293 , 2014.
R.-E. Fan, J.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A library for large
linear classiﬁcation. Journal of Machine Learning Research , 9:1871–1874, August 2008.
S. Fine and K. Scheinberg. Eﬃcient SVM training using low-rank kernel representations. Journal
of Machine Learning Research, 2:243–264, 2001.
A. Frank and A. Asuncion. UCI machine learning repository, 2010. URL http://archive.ics.
uci.edu/ml.
F. Girosi. An equivalence between sparse approximation and support vector machines. Neural
Computation, 10(6):1455–1480, 1998.
F. Girosi and G. Anzellotti. Rates of convergence for radial basis functions and neural networks. In
R. J. Mammone, editor, Artiﬁcial Neural Networks for Speech and Vision, pages 97–113, London,
1993. Chapman and Hall.
F. Girosi, M. Jones, and T. Poggio. Regularization theory and neural networks architectures.
Neural Computation, 7(2):219–269, 1995.
Alexander G. Gray and Andrew W. Moore. Rapid evaluation of multiple density models. In Proc.
Intl. Conference on Artiﬁcial Intelligence and Statistics , 2003.
David Haussler. Convolution kernels on discrete structures. Technical Report UCS-CRL-99-10, UC
Santa Cruz, 1999.
H. Hochstadt. Special functions of mathematical physics . Dover, 1961.
T. Huang, C. Guestrin, and L. Guibas. Eﬃcient inference for distributions on permutations. In
NIPS, 2007.
R. Jin, T. Yang, M. Mahdavi, Y.F. Li, and Z.H. Zhou. Improved bound for the nystrom’s method
and its application to kernel classiﬁcation, 2011. URL http://arxiv.org/abs/1111.2262.
G. S. Kimeldorf and G. Wahba. A correspondence between Bayesian estimation on stochastic
processes and smoothing by splines. Annals of Mathematical Statistics , 41:495–502, 1970.
R. Kondor. Group theoretical methods in machine learning. PhD thesis, Columbia University, 2008.
URL http://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf.
E. Kreyszig. Introductory Functional Analysis with Applications . Wiley, 1989.
A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University
of Toronto, 2009.
26

<!-- page 27 -->

M. Ledoux. Isoperimetry and gaussian analysis. In Lectures on probability theory and statistics ,
pages 165–294. Springer, 1996.
Dongryeol Lee and Alexander G. Gray. Fast high-dimensional kernel summations using the monte
carlo multipole method. In Neural Information Processing Systems. MIT Press, 2009.
David J. C. MacKay. Information Theory, Inference, and Learning Algorithms . Cambridge Uni-
versity Press, 2003.
S. Matsushima, S.V.N. Vishwanathan, and A.J. Smola. Linear support vector machines via dual
cached loops. In Q. Yang, D. Agarwal, and J. Pei, editors,The 18th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining, KDD , pages 177–185. ACM, 2012. URL
http://dl.acm.org/citation.cfm?id=2339530.
J. Mercer. Functions of positive and negative type and their connection with the theory of integral
equations. Philos. Trans. R. Soc. Lond. Ser. A Math. Phys. Eng. Sci. , A 209:415–446, 1909.
C. A. Micchelli. Interpolation of scattered data: distance matrices and conditionally positive deﬁnite
functions. Constructive Approximation, 2:11–22, 1986.
R. Neal. Priors for inﬁnite networks. Technical Report CRG-TR-94-1, Dept. of Computer Science,
University of Toronto, 1994.
A. Rahimi and B. Recht. Random features for large-scale kernel machines. In J.C. Platt, D. Koller,
Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems 20 . MIT
Press, Cambridge, MA, 2008.
Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization
with randomization in learning. In Neural Information Processing Systems, 2009.
C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning . MIT Press,
Cambridge, MA, 2006.
N. Ratliﬀ, J. Bagnell, and M. Zinkevich. (online) subgradient methods for structured prediction.
In Eleventh International Conference on Artiﬁcial Intelligence and Statistics (AIStats) , March
2007.
B. Sch¨ olkopf, A. J. Smola, and K.-R. M¨ uller. Nonlinear component analysis as a kernel eigenvalue
problem. Neural Comput., 10:1299–1319, 1998.
Bernhard Sch¨ olkopf and A. J. Smola.Learning with Kernels. MIT Press, Cambridge, MA, 2002.
A. J. Smola. Learning with Kernels . PhD thesis, Technische Universit¨ at Berlin, 1998. GMD
Research Series No. 25.
A. J. Smola and B. Sch¨ olkopf. Sparse greedy matrix approximation for machine learning. In
Proceedings of the International Conference on Machine Learning, pages 911–918, San Francisco,
2000. Morgan Kaufmann Publishers.
A. J. Smola, B. Sch¨ olkopf, and K.-R. M¨ uller. General cost functions for support vector regression.
In T. Downs, M. Frean, and M. Gallagher, editors, Proc. of the Ninth Australian Conf. on Neural
Networks, pages 79–83, Brisbane, Australia, 1998a. University of Queensland.
27

<!-- page 28 -->

A. J. Smola, B. Sch¨ olkopf, and K.-R. M¨ uller. The connection between regularization operators and
support vector kernels. Neural Networks, 11(5):637–649, 1998b.
A. J. Smola, Z. L. ´Ov´ ari, and R. C. Williamson. Regularization with dot-product kernels. In
T. K. Leen, T. G. Dietterich, and V. Tresp, editors, Advances in Neural Information Processing
Systems 13, pages 308–314. MIT Press, 2001.
Ingo Steinwart and Andreas Christmann. Support Vector Machines . Information Science and
Statistics. Springer, 2008.
B. Taskar, C. Guestrin, and D. Koller. Max-margin Markov networks. In S. Thrun, L. Saul,
and B. Sch¨ olkopf, editors,Advances in Neural Information Processing Systems 16 , pages 25–32,
Cambridge, MA, 2004. MIT Press.
Choon Hui Teo, S. V. N. Vishwanthan, A. J. Smola, and Quoc V. Le. Bundle methods for regularized
risk minimization. Journal of Machine Learning Research , 11:311–365, January 2010.
J. A. Tropp. Improved analysis of the subsampled randomized hadamard transform. CoRR,
abs/1011.1595, 2010. URL http://arxiv.org/abs/1011.1595.
K. Tsuda, T. Kin, and K. Asai. Marginalized kernels for biological sequences. Bioinformatics, 18
(Suppl. 2):S268–S275, 2002.
V. Vapnik, S. Golowich, and A. J. Smola. Support vector method for function approximation,
regression estimation, and signal processing. In M. C. Mozer, M. I. Jordan, and T. Petsche,
editors, Advances in Neural Information Processing Systems 9 , pages 281–287, Cambridge, MA,
1997. MIT Press.
G. Wahba. Spline Models for Observational Data , volume 59 of CBMS-NSF Regional Conference
Series in Applied Mathematics . SIAM, Philadelphia, 1990.
C. K. I. Williams. Prediction with Gaussian processes: From linear regression to linear prediction
and beyond. In M. I. Jordan, editor, Learning and Inference in Graphical Models, pages 599–621.
Kluwer Academic, 1998.
Christoper K. I. Williams and Matthias Seeger. Using the Nystrom method to speed up kernel
machines. In T. K. Leen, T. G. Dietterich, and V. Tresp, editors,Advances in Neural Information
Processing Systems 13, pages 682–688, Cambridge, MA, 2001. MIT Press.
R. C. Williamson, A. J. Smola, and B. Sch¨ olkopf. Generalization bounds for regularization networks
and support vector machines via entropy numbers of compact operators. IEEE Trans. Inform.
Theory, 47(6):2516–2532, 2001.
28

<!-- page 29 -->

Table 3: Test set RMSE of diﬀerent kernel computation methods. We can see Fastfood methods perform comparably with Exact RBF,
Nystrom, Random Kitchen Sinks (RKS) and Exact Polynomial (degree 10). m and d are the size of the training set the dimension of
the input. Note that the problem size made it impossible to compute the exact solution for datasets of size 40,000 and up.
Dataset m d Exact Nystrom RKS Fastfood Fastfood Exact Fastfood Exact Fastfood
RBF RBF RBF FFT RBF Matern Matern Poly Poly
Insurance 5, 822 85 0.231 0.232 0.266 0.266 0.264 0.234 0.235 0.256 0.271
Wine 4, 080 11 0.819 0.797 0.740 0.721 0.740 0.753 0.720 0.827 0.731
Quality
Parkinson 4, 700 21 0.059 0.058 0.054 0.052 0.054 0.053 0.052 0.061 0.055
CPU 6, 554 21 7.271 6.758 7.103 4.544 7.366 4.345 4.211 7.959 5.451
CT slices 42, 800 384 n.a. 60.683 49.491 58.425 43.858 n.a. 14.868 n.a. 53.793
(axial)
KEGG 51, 686 27 n.a. 17 .872 17 .837 17.826 17 .818 n.a. 17.846 n.a. 18.032
Network
Year 463, 715 90 n.a. 0.113 0.123 0.106 0.115 n.a. 0.116 n.a. 0.114
Prediction
Forest 522, 910 54 n.a. 0.837 0.840 0.838 0.840 n.a. 0.976 n.a. 0.894
29
