# references/180_constrained_gaussian_wasserstein_optimal_transport_with_commutative_covariance_matrices.pdf

<!-- page 1 -->

arXiv:2503.03744v1  [cs.IT]  5 Mar 2025
1
Constrained Gaussian Wasserstein Optimal
Transport with Commutative Covariance Matrices
Jun Chen, Jia Wang, Ruibin Li, Han Zhou, Wei Dong, Huan Liu, an d Y uanhao Y u
Abstract—Optimal transport has found widespread applica-
tions in signal processing and machine learning. Among its m any
equivalent formulations, optimal transport seeks to recon struct
a random variable/vector with a prescribed distribution at the
destination while minimizing the expected distortion rela tive
to a given random variable/vector at the source. However , in
practice, certain constraints may render the optimal trans port
plan infeasible. In this work, we consider three types of con -
straints: rate constraints, dimension constraints, and ch annel
constraints, motivated by perception-aware lossy compres sion,
generative principal component analysis, and deep joint so urce-
channel coding, respectively. Special attenion is given to the
setting termed Gaussian Wasserstein optimal transport, wh ere
both the source and reconstruction variables are multivari ate
Gaussian, and the end-to-end distortion is measured by the
mean squared error . We derive explicit results for the minim um
achievable mean squared error under the three aforemention ed
constraints when the covariance matrices of the source and
reconstruction variables commute.
Index Terms —Common randomness, dimension reduction,
generative model, hybrid coding, joint source-channel cod ing,
optimal transport, perception constraint, principal comp onent
analysis, reverse waterﬁlling, Wasserstein distance.
I. I NTRODUCTION
Since its inception, optimal transport has grown from a
purely mathematical theory [1] into a powerful tool with
widespread applications across numerous ﬁelds. Its inﬂuen ce
extends so broadly that it is difﬁcult to identify an area it
has not impacted. In particular, optimal transport has had
a profound effect on signal processing and machine learn-
ing, where it has shaped fundamental methodologies and
inspired innovative approaches. This signiﬁcant impact is well-
documented in various survey papers [2], [3], highlighting its
role in advancing these domains.
Optimal transport admits many equivalent formulations. In
this work, we adopt a formulation that frames the problem as
reconstructing a random variable/vector ˆS with a prescribed
distribution p ˆS at the destination while minimizing the ex-
pected distortion relative to a given S with distribution pS
at the source. Mathematically, this corresponds to solving the
following optimization problem:
inf
pS ˆS ∈ Π(pS ,p ˆS)
E[c(S, ˆS)], (1)
where Π(pS, p ˆS) denotes the set of joint distributions with
marginals being pS and p ˆS, and c(·, ·) is the transport cost
function, which in this formulation is more naturally inter -
preted as a distortion measure. In particular, when c(s, ˆs) =
∥s − ˆs∥2, the solution to (1) yields the squared Wassertein-2
distance between pS and p ˆS, denoted by W 2
2 (pS, p ˆS),
In practical applications, various constraints may render
the optimal transport plan associated with the joint distri bu-
tion pS ˆS that achieves the inﬁmum in (1) infeasible. These
constraints can arise from physical limitations, regulato ry
requirements, or structural restrictions imposed by the pr oblem
setting. Consequently, there is a need to study constrained opti-
mal transport, which aims to develop methods for determinin g
the best possible transport plan while ensuring compliance
with the given constraints.
One such scenario occurs when the source and destination
are connected by a rate-limited bit pipeline. In this case, c on-
tinuous transport plans are no longer realizable, necessit ating
some form of discretization. This challenge has led to the
investigation of distribution-preserving quantization [ 4], [5]
and output-constrained lossy source coding [6], [7]. Recen tly,
this line of research has gained renewed interest due to the
emergence of perception-aware lossy compression [8]–[22]
and cross-domain lossy compression [23], [24]. For related
developments in the quantum setting, see [25].
Another motivating scenario involves a dimensional bot-
tleneck between the source and destination. In this case,
it becomes essential to identify the minimum-dimensional
representation of the source variable S that enables faithful
reconstruction. This concept underpins compressed sensin g
[26], [27] and analog compression [28]. In the lossy setting ,
it leads to techniques such as principal component analysis
[29], among others. More generally, one may be interested in
generative tasks where the reconstruction variable ˆS does not
need to be identical to the source variable S. This requires
the development of dimension reduction methods speciﬁcall y
tailored for such purposes.
The emerging paradigm of generative communication [30],
which leverages deep generative models for joint source-
channel coding [31]–[35], also provides a compelling impet us
for studying constrained optimal transport. In this contex t, the
source must communicate with the destination through a chan -
nel. Notably, unlike conventional communication problems ,
here the source-channel separation architecture can be str ictly
suboptimal, even in the point-to-point scenario.
The present work focuses on the setting termed Gaussian
Wasserstein optimal transport, where both the source and
reconstruction variables are multivariate Gaussian, and t he
end-to-end distortion is measured by the mean squared error .
Speciﬁcally, we assume that S := ( S1, S 2, . . . , S L)T and
ˆS := ( ˆS1, ˆS2, . . . , ˆSL)T are L-dimensional random vectors
distributed according to N (µ, Σ) and N (ˆµ, ˆΣ), respectively,
and c(s, ˆs) = ∥s − ˆs∥2 for s, ˆs ∈ RL. Consequently, the
solution to (1) is given by the squared Wasserstein-2 distan ce

<!-- page 2 -->

2
between N (µ, Σ) and N (ˆµ, ˆΣ) [36]–[39] expressed as
W 2
2 (N (µ, Σ), N (ˆµ, ˆΣ))
= ∥µ − ˆµ ∥2 + tr
(
Σ + ˆΣ − 2(Σ
1
2 ˆΣΣ
1
2 )
1
2
)
. (2)
Special attention is given to the case where the covariance
matrices Σ and ˆΣ are positive deﬁnite and commute. This
allows us to write them as Σ = ΘΛΘ T and ˆΣ = Θ ˆΛΘT ,
where Θ is a unitary matrix, while Λ := diag( λ 1, λ 2, . . . , λ L)
and ˆΛ := diag( ˆλ 1, ˆλ 2, . . . , ˆλ L) are diagonal matrices with
positive diagonal entries. For this case, (2) simpliﬁes to
W 2
2 (N (µ, Σ), N (ˆµ, ˆΣ))
= ∥µ − ˆµ ∥2 +
L∑
ℓ=1
(√
λ ℓ −
√
ˆλ ℓ
)2
. (3)
The optimal transport plan that achieves (3) is given by
ˆS = Θdiag


√
ˆλ 1
λ 1
,
√
ˆλ 2
λ 2
, . . . ,
√
ˆλ L
λ L

 ΘT (S − µ ) + ˆµ,
(4)
which is an afﬁne transformation. For notational simplicit y,
we henceforth assume µ = ˆµ = 0 , Θ = I (i.e., Σ = Λ and
ˆΣ = ˆΛ), and
λ 1ˆλ 1 ≥ λ 2ˆλ 2 ≥ . . . ≥ λ L ˆλ L. (5)
It will be seen that the Gaussian Wasserstein optimal transp ort
problem serves as an ideal framework for examining the three
key constraints discussed earlier: rate constraints, dime nsion
constraints, and channel constraints. To ensure a coherent
treatment, we adopt the asymptotic setting rather than the o ne-
shot setting where a single reconstruction variable/vecto r ˆS is
generated for a single source variable/vector S. Speciﬁcally,
we consider the task of generating an i.i.d. reconstruction
sequence ˆSn with ˆS(t) ∼ N (0, ˆΛ), t = 1 , 2, . . . , n , in
response to an i.i.d. source sequence Sn with S(t) ∼ N (0, Λ),
t = 1, 2, . . . , n , while minimizing the average distortion
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2]. (6)
In the absence of constraints, there is no fundamental diffe r-
ence between the one-shot and asymptotic settings, as trans port
can be performed in a symbol-by-symbol manner without loss
of optimality.
Our main contributions are as follows:
1) For rate-constrained optimal transport, we distinguish
between the case with unlimited common randomness
and the case with no common randomness, deriving
reverse waterﬁlling-type formulas for the minimum
achievable distortion in both cases.
2) For dimension-constrained optimal transport, we extend
principal component analysis to generative tasks.
3) For channel-constrained optimal transport, we provide
a systematic comparison of separation-based, uncoded,
and hybrid schemes.
The remainder of this paper is organized as follows. Sec-
tions II, III, and IV explore rate-constrained, dimension-
constrained, and channel-constrained optimal transport, re-
spectively. Recurring themes, important connections, and key
differences across the three types of constraints are highl ighted
throughout these sections. Finally, we conclude the paper i n
Section V.
Throughout this paper, we adopt the standard notation
for information measures: I(·; ·) for mutual information and
h(·) for differential entropy. The set of nonnegative numbers
is denoted by R+. We deﬁne log+(a) := max {log(a), 0},
(a)+ := max{x, 0}, and a ∧ b := min{a, b }. A Gaussian dis-
tribution with mean µ and covariance matrix Σ is represented
as N (µ, Σ). For brevity, we use X n to denote the sequence
{X(t)}n
t=1. Summations of the form ∑k
i=j ai are deﬁned to be
zero whenever j > k . The expectation, trace, ﬂoor, and ceiling
operators are denoted by E[·], tr(·), ⌊·⌋, and ⌈·⌉, respectively.
For two matrices A and B, the notation A ⪯ B indicates that
B − A is positive semideﬁnite. Finally, we use log and ln to
denote logarithms with bases 2 and e, respectively.
II. R ATE-C ONSTRAINED OPTIMAL TRANSPORT
In this section, we examine the scenario where the source
and destination are connected by a rate-limited bit pipelin e,
necessitating the deployment of an encoder and a decoder.
Given the source sequence Sn, the encoder produces a length-
m bit string Bm ∈ { 0, 1}m and transmits it to the destination
via the bit pipeline. Upon receiving the bit string, the deco der
generates a reconstruction sequence ˆSn with the prescribed
distribution while minimizing the end-to-end distortion. This
scenario is ﬁrst studied in [7] from an information-theoret ic
perspective. By focusing on the Gaussian Wasserstein setti ng,
we are able to obtain explicit reverse waterﬁlling-type res ults
by leveraging convex optimization techniques. Notably, un like
conventional source coding problems, the minimum achievab le
distortion in this setting depends on the availability of co mmon
randomness. Accordingly, we structure our analysis to acco unt
for this dependency.
A. Unlimited Common Randomness
Here, the encoder and decoder are assumed to share a ran-
dom seed Q. Accordingly, their operations are governed by the
conditional distributions pBm|SnQ and p ˆSn|BmQ, respectively.
The overall system is characterized by the joint distributi on
pSnQBm ˆSn factorized as
pSnQBm ˆSn = pSn pQpBm|SnQp ˆSn|BmQ, (7)
where pSn = pn
S with pS = N (0, Λ).
Deﬁnition 1: With common randomness, a distortion level
D is said to be achievable under a rate constraint R if, for
all sufﬁciently large n, there exist a seed distribution pQ, an
encoding distribution pBm|SnQ, and a decoding distribution
p ˆSn|BmQ such that
m
n ≤ R, (8)
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≤ D, (9)

<!-- page 3 -->

3
and the reconstruction sequence ˆSn follows the i.i.d. distri-
bution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ). The inﬁmum of all
achievable distortion levels D under the rate constraint R with
common randomness is denoted by Dr(R).
The following result provides an explicit characterizatio n of
Dr(R). Its proof can be found in Appendix A.
Theorem 1: For R ≥ 0,
Dr(R) =
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√
(1 − 2− 2Rℓ (R))λ ℓˆλ ℓ
)
, (10)
where
Rℓ(R) := 1
2 log
(
1 +
√
1 + αλ ℓˆλ ℓ
2
)
, ℓ = 1, 2, . . . , L,
(11)
with α being the unique nonnegative number satisfying
1
2
L∑
ℓ=1
log
(
1 +
√
1 + αλ ℓˆλ ℓ
2
)
= R. (12)
Theorem 1 admits a natural operational interpretation. Wit h
the availability of common randomness, as n → ∞ , it takes
Rℓ := I(Sℓ, ˆSℓ) bits per symbol to simulate ˆSn
ℓ such that its
pairwise distributions with Sn
ℓ , i.e., pSℓ (t) ˆSℓ (t), t = 1, 2, . . . , n ,
are all equal to a prescribed bivariate Gaussian distributi on
pSℓ ˆSℓ
, where Sℓ ∼ N (0, λ ℓ) and ˆSℓ ∼ N (0, ˆλ ℓ) for ℓ =
1, 2, . . . , L . Assuming the correlation coefﬁcient of Sℓ and ˆSℓ
is ρℓ ≥ 0, we have
Rℓ = 1
2 log
( 1
1 − ρ2
ℓ
)
, (13)
which implies
ρℓ =
√
1 − 2− 2Rℓ , ℓ = 1, 2, . . . , L. (14)
Consequently,
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2]
= 1
nL
n∑
t=1
n∑
ℓ=1
E[(Sℓ(t) − ˆSℓ(t))2]
=
L∑
ℓ=1
E[(Sℓ − ˆSℓ)2]
=
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√
(1 − 2− 2Rℓ )λ ℓˆλ ℓ
)
. (15)
This leads to the following rate allocation problem:
min
(R1,R 2...,R L)∈ RL
+
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√(1 − 2− 2Rℓ )λ ℓˆλ ℓ
)
(16)
s.t.
L∑
ℓ=1
Rℓ ≤ R, (17)
with the minimizer given by (11).
B. No Common Randomness
Here, the encoder and decoder are assumed to operate
without a shared random seed. Speciﬁcally, their operation s are
governed by the conditional distributions pBm|Sn and p ˆSn|Bm,
respectively. The overall system is characterized by the jo int
distribution pSnBm ˆSn factorized as
pSnBm ˆSn = pSnpBm|Sn p ˆSn|Bm, (18)
where pSn = pn
S with pS = N (0, Λ).
Deﬁnition 2: Without common randomness, a distortion
level D is said to be achievable under a rate constraint R if,
for all sufﬁciently large n, there exist an encoding distribution
pBm|Sn and a decoding distribution p ˆSn|Bm such that
m
n ≤ R, (19)
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≤ D, (20)
and the reconstruction sequence ˆSn follows the i.i.d. distri-
bution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ). The inﬁmum of all
achievable distortion levels D under the rate constraint R
without common randomness is denoted by Dr(R).
The following result provides an explicit characterizatio n of
Dr(R). Its proof can be found in Appendix B.
Theorem 2: For R ≥ 0,
Dr(R) =
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2(1 − 2− 2Rℓ (R))
√
λ ℓˆλ ℓ
)
, (21)
where
Rℓ(R) := 1
2 log+
( √
λ ℓˆλ ℓ
β
)
, ℓ = 1, 2, . . . , L, (22)
with β being the unique number in (0,
√
λ 1ˆλ 1] satisfying
1
2
L∑
ℓ=1
log+
( √
λ ℓˆλ ℓ
β
)
= R. (23)
Theorem 2 also admits a natural operational interpretation .
First, the encoder maps Sn
ℓ to U n
ℓ using a rate- Rℓ vector
quantizer such that
1
n
n∑
t=1
E[(Sℓ(t) − Uℓ(t))2] = λ ℓ − γℓ ≈ 2− 2Rℓ λ ℓ, (24)
where γℓ := 1
n
∑n
t=1 E[U 2
ℓ (t)] for ℓ = 1 , 2, . . . , L . Next, it
converts U n
ℓ to ˆU n
ℓ (t) via the transport plan induced by a
coupling satisfying
1
n
n∑
t=1
E[(Uℓ(t) − ˆUℓ(t))2] ≈ (√ γ ℓ −
√
ˆγ ℓ)2, (25)
where ˆU n
ℓ is distributed according to the output of a rate- Rℓ
quantizer with the reconstruction ˆSn
ℓ serving as a ﬁctitious
source, and ˆγℓ := 1
n
∑n
t=1 E[ ˆU 2
ℓ (t)] for ℓ = 1 , 2, . . . , L .
The decoder then generates ˆSn
ℓ from ˆU n
ℓ by performing
dequantization (also known as posterior sampling), yieldi ng
1
n
n∑
t=1
E[( ˆSℓ(t) − ˆUℓ(t))2] = ˆλ ℓ − ˆγℓ ≈ 2− 2Rℓ ˆλ ℓ, (26)

<!-- page 4 -->

4
for ℓ = 1, 2, . . . , L . Consequently,
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2]
= 1
n
n∑
t=1
L∑
ℓ=1
E[(Sℓ(t) − Uℓ(t))2 + (Uℓ(t) − ˆUℓ(t))2
+ ( ˆSℓ(t) − ˆU (t))2]
=
L∑
ℓ=1
(λ ℓ + ˆλ ℓ − 2
√
γℓˆγℓ)
≈
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2(1 − 2− 2Rℓ )
√
λ ℓˆλ ℓ
)
. (27)
This leads to the following rate allocation problem:
min
(R1,R 2...,R L)∈ RL
+
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2(1 − 2− 2Rℓ )
√λ ℓˆλ ℓ
)
(28)
s.t.
L∑
ℓ=1
Rℓ ≤ R, (29)
with the minimizer given by (22).
0 1 2 3 4 5 6 7 8 9 10
0
2
4
6
8
10
12
Fig. 1. Plots of Dr(R) and Dr(R) for the case where (λ1, λ2, λ3) =
(2, 3, 1) and (ˆλ1, ˆλ2, ˆλ3) = (3 , 1, 1).
R = 0.1 R = 2.1 R = 4.1
R1(R) 0.058 0.929 1.641
R2(R) 0.031 0.726 1.407
R3(R) 0.011 0.445 1.051
R1(R) 0.1 0.999 1.665
R2(R) 0 0.749 1.415
R3(R) 0 0.353 1.019
TABLE I
COMPARISON OF (R1(R), R2(R), R3(R)) AND (R1(R), R2(R), R3(R))
FOR THE CASE WHERE (λ1, λ2, λ3) = (2 , 3, 1) AND
(ˆλ1, ˆλ2, ˆλ3) = (3 , 1, 1).
It is instructive to compare Dr(R) and Dr(R), as well as
their associated rate allocation schemes (see also Fig. 1 an d
Table I). Clearly, Dr(R) < Dr(R) for all R > 0. Both Dr(R)
and Dr(R) approach
Dmax :=
L∑
ℓ=1
(λ ℓ + ˆλ ℓ) (30)
as R → 0 and approach
Dmin :=
L∑
ℓ=1
(√
λ ℓ −
√
ˆλ ℓ
)2
(31)
as R → ∞ , where Dmax and Dmin are, respectively, the
distortion achieved by generating ˆSn independently of Sn
and the distortion achieved by unconstrained optimal trans -
port (cf. (3)). For each ℓ = 1 , 2, . . . , L , both Rℓ(R) and
Rℓ(R) are increasing functions of R. For a ﬁxed R, the
ordering of R1(R), R 2(R), . . . , R L(R) is determined by that
of λ 1ˆλ 1, λ 2ˆλ 2, . . . , λ L ˆλ L, with larger values of λ ℓˆλ ℓ cor-
responding to higher Rℓ(R). The same ordering applies to
R1(R), R2(R), . . . , RL(R). It will be seen that the value
of λ ℓˆλ ℓ as a measure of signiﬁcance is a recurring theme
across other constrained optimal transport problems. On th e
other hand, the two rate allocation schemes also have some
notable differences. In particular, R
1(R), R 2(R), . . . , R L(R)
are strictly positive whenever R > 0, whereas some of
R1(R), R2(R), . . . , RL(R) can be zero when R is sufﬁciently
small. Moreover, for a given total rate R > 0, the indi-
vidual rates R1(R), R2(R), . . . , RL(R) exhibit greater vari-
ation across all components compared to their counterparts
R
1(R), R 2(R), . . . , R L(R).
III. D IMENSION -C ONSTRAINED OPTIMAL TRANSPORT
In this section, we consider the scenario where the bottle-
neck between the source and destination takes the form of a di -
mensionality constraint. As a result, the encoder must iden tify
a low-dimensional representation of the source sequence Sn,
from which the decoder generates a reconstruction sequence
ˆSn, ensuring the prescribed distribution while minimizing th e
end-to-end distortion.
It is clear that a certain regularity condition needs to be
imposed on the encoder since otherwise the source sequence
could be losslessly represented using a single real number. For
this reason, we require the encoder to be a linear mapping φ.
On the other hand, the decoder is allowed to be a stochastic
function governed by the conditional distribution p ˆSn|φ (Sn).
The overall system is characterized by the joint distributi on
pSnφ (Sn) ˆSn factorized as
pSnφ (Sn) ˆSn = pSn pφ (Sn)|Sn p ˆSn|φ (Sn), (32)
where pSn = pn
S with pS = N (0, Λ).
Deﬁnition 3: A distortion level D is said to be achiev-
able under a normalized dimension constraint Γ if, for all
sufﬁciently large n, there exist a linear encoding function
φ : RnL → Rm and a decoding distribution p ˆSn|φ (Sn) such
that
m
n ≤ Γ, (33)
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≤ D, (34)

<!-- page 5 -->

5
and the reconstruction sequence ˆSn follows the i.i.d. distri-
bution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ). The inﬁmum of all
achievable distortion levels D under the normalized dimension
constraint Γ is denoted by Dd(Γ).
We ﬁrst prove the following one-shot result. Its proof can
be found in Appendix C.
Theorem 3: Given S ∼ N (0, Λ), for any linear encoding
function φ : RL → RK and decoding distribution p ˆS|φ (S)
such that the induced distribution p ˆS = N (0, ˆΛ), we have
E[∥S − ˆS∥2] ≥
K∧ L∑
ℓ=1
(√
λ ℓ −
√
ˆλ ℓ
)2
+
L∑
ℓ=(K∧ L)+1
(λ ℓ + ˆλ ℓ).
(35)
Moreover, this lower bound is achieved by selecting the
ﬁrst K ∧ L components of S, scaling them to obtain
ˆS1, ˆS2, . . . , ˆSK∧ L, and generating the remaining components
of ˆS from scratch.
The scheme achieving the lower bound in Theorem 3 can
be interpreted as a generative variant of principal compone nt
analysis, where the selection rule is determined by the orde ring
of λ 1ˆλ 1, λ 2ˆλ 2, . . . , λ Lˆλ L. This selection rule simpliﬁes to that
of conventional principal component analysis [29] when Λ =
ˆΛ.
Since Sn and ˆSn can be regarded as Gaussian random
vectors of dimension nL, with their covariance matrices
preserving a diagonal structure, we can directly infer the
following result from Theorem 3.
Theorem 4: For Γ ∈ [0, L ],
Dd(Γ) =
⌊Γ⌋∑
ℓ=1
(√
λ ℓ −
√
ˆλ ℓ
)2
+ (Γ − ⌊ Γ⌋)
(√
λ ⌈Γ⌉ −
√
ˆλ ⌈Γ⌉
)2
+ (⌈Γ⌉ − Γ)(λ ⌈Γ⌉ + ˆλ ⌈Γ⌉) +
L∑
ℓ=⌈Γ⌉+1
(λ ℓ + ˆλ ℓ).
(36)
Moreover, Dd(Γ) = Dd(L) for Γ > L.
0 0.5 1 1.5 2 2.5 3
0
2
4
6
8
10
12
Fig. 2. Plot of Dd(Γ) for the case where (λ1, λ2, λ3) = (2 , 3, 1) and
(ˆλ1, ˆλ2, ˆλ3) = (3 , 1, 1).
Dd(Γ) is a decreasing, convex, piecewise linear function of
Γ, approaching Dmax as Γ → 0 and Dmin as Γ → L (see Fig.
2). In comparison, Dr(R) and Dr(R) exhibit similar overall
behavior but are strictly convex.
IV. C HANNEL -C ONSTRAINED OPTIMAL TRANSPORT
In this section, we consider the scenario where the transpor t
must pass through an additive unit-variance white Gaussian
noise channel pY |X , denoted by A WGN(1), where Y = X +
N with N ∼ N (0, 1) independent of X. This scenario is ﬁrst
studied in [30, Section V] for the degenerate case Λ = ˆΛ.
We will again distinguish between the cases with and without
common randomness.
With common randomness, the encoder and decoder are
assumed to share a random seed Q. Accordingly, their oper-
ations are governed by the conditional distributions pX n|SnQ
and p ˆSn|Y nQ, respectively. The overall system is characterized
by the joint distribution pSnQX nY n ˆSn factorized as
pSnQX nY n ˆSn = pSn pQpX n|SnQpY n|X np ˆSn|Y nQ, (37)
where pSn = pn
S with pS = N (0, Λ) and pY n|X n = pn
Y |X
with pY |X = A WGN(1).
Deﬁnition 4: With common randomness, a distortion level
D is said to be achievable through A WGN(1)under an input
power constraint P if, for all sufﬁciently large n, there exist
a seed distribution pQ, an encoding distribution pX n|SnQ, and
a decoding distribution p ˆSn|Y nQ such that
1
n
n∑
t=1
E[X 2(t)] ≤ P, (38)
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≤ D, (39)
and the reconstruction sequence ˆSn follows the i.i.d. dis-
tribution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ). The inﬁmum of
all achievable distortion levels D through A WGN(1) under
the input power constraint P with commom randomness is
denoted by Dc(P ).
According to [30, Theorem 1], when unlimited common
randomness is available, the source-channel separation th eo-
rem holds for channel-constrained optimal transport, name ly,
there is no loss of optimality in ﬁrst converting the channel to
a bit pipeline using error correction codes and then perform ing
rate-constrained optimal transport. Combining this resul t with
the capacity formula of A WGN(1), we obtain
Dc(P ) = Dr
(1
2 log(P + 1)
)
. (40)
Without common randomess, the encoding and decod-
ing operations are governed by the conditional distributio ns
pX n|Sn and p ˆSn|Y n , respectively. The overall system is char-
acterized by the joint distribution pSnX nY n ˆSn factorized as
pSnX nY n ˆSn = pSn pX n|SnpY n|X np ˆSn|Y n , (41)
where pSn = pn
S with pS = N (0, Λ) and pY n|X n = pn
Y |X
with pY |X = A WGN(1).

<!-- page 6 -->

6
Deﬁnition 5: Without common randomness, a distortion
level D is said to be achievable through A WGN(1) under
an input power constraint P if, for all sufﬁciently large n,
there exist an encoding distribution pX n|Sn and a decoding
distribution p ˆSn|Y n such that
1
n
n∑
t=1
E[X 2(t)] ≤ P, (42)
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≤ D, (43)
and the reconstruction sequence ˆSn follows the i.i.d. distri-
bution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ). The inﬁmum of all
achievable distortion levels D through A WGN(1)under the
input power constraint P without commom randomness is
denoted by Dc(P ).
Let D(s)
c (P ) denote the minimum achievable distortion
under the separation-based scheme, i.e.,
D(s)
c (P ) := Dr
(1
2 log(P + 1)
)
. (44)
It turns out that D(s)
c (P ) is just an upper bound on Dc(P ). As
we will demonstrate, the source-channel separation archit ec-
ture is generally suboptimal for channel-constrained opti mal
transport when no common randomness is available.
To this end, consider the following uncoded scheme. The
encoder transmits X n :=
√
P
λ 1
Sn
1 , obtained by scaling the ﬁrst
component of each source symbol to meet the power contraint
while discarding the other components Sn
2 , S n
3 , . . . , S n
L; given
the channel output Y n, the decoder sets ˆSn
1 :=
√
ˆλ 1
P +1 Y n
and generates the remaining components ˆSn
2 , ˆSn
3 , . . . , ˆSn
L of
the reconstruction sequence from scratch. It can be veriﬁed
that the resulting distortion is given by
D(u)
c (P ) := − 2
√
P
P + 1 λ 1ˆλ 1 +
L∑
ℓ=1
(λ ℓ + ˆλ ℓ). (45)
The following result, which is a “noisy” variant of Theorem
4 for the special case Γ = 1 and a generalization of the one-
shot optimality result [30, Theorem 3] for the degenerate ca se
Λ = ˆΛ, indicates that this uncoded scheme is the best one
among all linear schemes. Its proof can be found in Appendix
D.
Theorem 5: Let X n := φ(Sn) be the channel input induced
by a linear mapping φ satisfying (42), where pSn = pn
S with
pS = N (0, Λ), and let Y n be the corresponding channel
output through A WGN(1). For any decoding distribution
p ˆSn|Y n such that the reconstruction sequence ˆSn follows the
i.i.d. distribution p ˆSn = pn
ˆS with p ˆS = N (0, ˆΛ), we have
1
n
n∑
t=1
E[∥S(t) − ˆS(t)∥2] ≥ D(u)
c (P ). (46)
When L = 1, we have
D(u)
c (P ) = Dc(P ) = − 2
√
P
P + 1 λ 1ˆλ 1 + λ 1 + ˆλ 1, (47)
which implies
Dc(P ) = − 2
√
P
P + 1 λ 1 ˆλ 1 + λ 1 + ˆλ 1. (48)
In contrast, when L = 1,
D(s)
c (P ) = − 2P
P + 1
√
λ 1 ˆλ 1 + λ 1 + ˆλ 1, (49)
which is strictly greater than Dc(P ) for P > 0.
When L ≥ 2, the separation-based scheme and the uncoded
scheme can be integrated into a hybrid scheme via superpo-
sition. Speciﬁcally, the encoder allocates a fraction 1 − δ of
the power to transmit Sn
1 using the uncoded scheme, referred
to as the analog part, and the remaining fraction δ to transmit
Sn
2 , S n
3 , . . . , S n
L using the separation-based scheme, referred to
as the digital part. The decoder ﬁrst decodes the digital par t
by treating the analog part as noise and uses it to generate
ˆSn
2 , ˆSn
3 , . . . , ˆSn
L. It then subtracts the digital part from the
channel output and scales the residual signal to produce ˆSn
1 .
The distortion associated with the analog part is
− 2
√
(1 − δ)P
(1 − δ)P + 1 λ 1ˆλ 1 + λ 1 + ˆλ 1 (50)
while the distortion associated with the digital part is
− 2
L∑
ℓ=2
(√
λ ℓˆλ ℓ − β (δ)
)
+
+
L∑
ℓ=2
(λ ℓ + ˆλ ℓ), (51)
with β (δ) being the unique number in (0,
√
λ 2ˆλ 2] satisfying
L∏
ℓ=2
max
{ √
λ ℓˆλ ℓ
β (δ) , 1
}
= P + 1
(1 − δ)P + 1 . (52)
By summing these two distortions and optimizing over the
power allocation parameter δ, we obtain the minimum achiev-
able distortion under the hybrid scheme:
D(h)
c (P ) := min
δ∈ [0, 1]
{
− 2
√
(1 − δ)P
(1 − δ)P + 1 λ 1ˆλ 1
− 2
L∑
ℓ=2
(√
λ ℓˆλ ℓ − β (δ)
)
+
}
+
L∑
ℓ=1
(λ ℓ + ˆλ ℓ). (53)
The following result shows that with an optimized δ, the hy-
brid scheme strictly outperforms the separation-based sch eme
when P > 0, but reduces to the uncoded scheme when P is
sufﬁciently small. Its proof can be found in Appendix E.
Theorem 6: For P > 0,
D(h)
c (P ) < D (s)
c (P ). (54)
Moreover, when L ≥ 2,
D(h)
c (P ) = D(u)
c (P ) (55)
if and only if P ∈ [0, P ∗ ], where
P ∗ :=
− 1 +
√
1 + λ 1 ˆλ 1
λ 2 ˆλ 2
2 . (56)
Fig. 3 illustrates Dc(P ), D(s)
c (P ), D(u)
c (P ), and D(h)
c (P )
for a representative example. Notably, Dc(P ), D(s)
c (P ), and

<!-- page 7 -->

7
10-3 10-2 10-1 100 101 102 103
0
2
4
6
8
10
12
Fig. 3. Plots of Dc(P ), D(s)
c (P ), D(u)
c (P ), and D(h)
c (P ) for the case
where (λ1, λ2, λ3) = (2 , 3, 1) and (ˆλ1, ˆλ2, ˆλ3) = (3 , 1, 1).
D(h)
c (P ) all converge to Dmax as P → 0 and to Dmin as
P → ∞ . While D(u)
c (P ) follows a similar trend in the low-
power regime, it saturates at (
√
λ 1−
√ˆλ 1)2+∑L
ℓ=2(λ ℓ+ˆλ ℓ) in
the high-power limit. This occurs because the uncoded schem e
transmits only the ﬁrst component of each source symbol,
preventing it from utilizing additional power to reduce the
distortion with respect to the remaining components. It can
also be seen that D(h)
c (P ) is strictly below D(s)
c (P ) for P > 0
and coincides with D(u)
c (P ) for sufﬁciently small P . On the
other hand, D(h)
c (P ) still falls short of matching D
c(P ),
the minimum distortion achievable with unlimited common
randomness, whenever P > 0. The exact characterization
of
Dc(P ) remains unknown, though it must lie somewhere
between D(h)
c (P ) and Dc(P ).
V. C ONCLUSION
We have studied the problem of Gaussian Wasserstein
optimal transport with commutative covariance matrices un -
der rate, dimension, and channel constraints. The extensio n
beyond commutative covariance matrices requires more ad-
vanced analytical techniques, which will be addressed in fu ture
work. Notably, the Gaussian distribution represents the wo rst-
case scenario under the squared error distortion measure.
Therefore, our ﬁndings can serve as a useful reference point for
understanding the limits and behavior of transport problem s in
more general settings.
Overall, constrained optimal transport provides a uniﬁed
framework with broad applicability across machine learnin g,
information theory, and signal processing, opening up seve ral
promising research avenues. In this regard, our work makes
an initial attempt to explore these possibilities, and we ho pe it
will inspire further studies that delve deeper into the nuan ces
of this theoretical framework and its real-world applicati ons.
APPENDIX A
PROOF OF THEOREM 1
In light of [7, Section III.A],
D
r(R) = inf
pS ˆS∈ Π(N (0, Λ), N (0, ˆΛ))
E[∥S − ˆS∥2] (57)
s.t. I(S; ˆS) ≤ R. (58)
For pS ˆS ∈ Π(N (0, Λ), N (0, ˆΛ)), we have
I(S; ˆS) = h(S) + h( ˆS) − h(S, ˆS)
≥ h(S) + h( ˆS) −
L∑
ℓ=1
h(Sℓ, ˆSℓ)
=
L∑
ℓ=1
h(Sℓ) +
L∑
ℓ=1
h( ˆSℓ) −
L∑
ℓ=1
h(Sℓ, ˆSℓ)
=
L∑
ℓ=1
Rℓ, (59)
and
E[∥S − ˆS∥2] =
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2ρℓ
√
λ ℓˆλ ℓ
)
, (60)
where Rℓ := I(Sℓ; ˆSℓ), and ρℓ denotes the correlation coefﬁ-
cient of Sℓ and ˆSℓ for ℓ = 1, 2, . . . , L . Note that
Rℓ ≥ 1
2 log
( 1
1 − ρ2
ℓ
)
, (61)
which implies
ρℓ ≤
√
1 − 2− 2Rℓ , ℓ = 1, 2, . . . , L. (62)
Therefore, Dr(R) is bounded below by the solution to the
following convex optimization problem:
min
(R1,R 2...,R L)∈ RL
+
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√(1 − 2− 2Rℓ )λ ℓˆλ ℓ
)
(63)
s.t.
L∑
ℓ=1
Rℓ ≤ R. (64)
Deﬁne the Lagrangian
G :=
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√
(1 − 2− 2Rℓ )λ ℓˆλ ℓ
)
+ ν
n∑
ℓ=1
Rℓ,
(65)
where ν ≥ 0. It can be veriﬁed that
dG
dRℓ
= − (2 ln 2)2− 2Rℓ
√
1 − 2− 2Rℓ
√
λ ℓˆλ ℓ + ν, ℓ = 1, 2, . . . , L. (66)
For ℓ = 1, 2, . . . , L , setting dG
dRℓ
= 0 gives
Rℓ = 1
2 log

 1 +
√
1 + (16 ln2 2)
v2 λ ℓˆλ ℓ
2


= 1
2 log
(
1 +
√
1 + αλ ℓˆλ ℓ
2
)
, (67)

<!-- page 8 -->

8
where α := (16 ln2 2)
v2 . We obtain the minimizer
(R1(R), R 2(R), . . . , R L(R)), as deﬁned in (11), by
choosing α to be the unique nonnegative number that
satisﬁes the constraint in (64) with equality. The proof is
complete since this lower bound is attained when Sℓ and
ˆSℓ are jointly Gaussian with the correlation coefﬁcient
ρℓ =
√
1 − 2− 2Rℓ (R) for ℓ = 1 , 2, . . . , L , and the pairs
(S1, ˆS1), (S2, ˆS2), . . . , (SL, ˆSL) are mutually independent.
APPENDIX B
PROOF OF THEOREM 2
In light of [21, Theorem 1], Dr(R) is given by the solution
to the following optimization problem:
inf
pU |S ,p ˆU | ˆS
E[∥S − U ∥2] + W 2
2 (pU , p ˆU ) + E[∥ ˆS − ˆU ∥2] (68)
s.t. max{I(S; U ); I( ˆS; ˆU )} ≤ R, (69)
E[S|U ] = U and E[ ˆS|ˆU ] = ˆU almost surely , (70)
where S ∼ N (0, Λ) and ˆS ∼ N (0, ˆΛ). Consider U :=
(U1, U 2, . . . , U L)T and ˆU := ( ˆU1, ˆU2, . . . , ˆUL)T that satisfy
(69) and (70). Let ξℓ := E[U 2
ℓ ], ˆξℓ := E[ ˆU 2
ℓ ], Rℓ := I(Sℓ; Uℓ),
and ˆRℓ := I( ˆSℓ; ˆUℓ) for ℓ = 1, 2, . . . , L . We have
E[∥S − U ∥2] =
L∑
ℓ=1
(λ ℓ − ξℓ), (71)
E[∥ ˆS − ˆU ∥2] =
L∑
ℓ=1
(ˆλ ℓ − ˆξℓ), (72)
W 2
2 (pU , p ˆU ) ≥
L∑
ℓ=1
(
√
ξℓ −
√
ˆξℓ)2. (73)
Moreover,
I(S; U ) ≥
L∑
ℓ=1
Rℓ, (74)
I( ˆS; ˆU ) ≥
L∑
ℓ=1
ˆRℓ. (75)
It can be veriﬁed that for ℓ = 1, 2, . . . , L ,
ξℓ ≤ (1 − 2− 2Rℓ )λ ℓ, (76)
ˆξℓ ≤ (1 − 2− 2 ˆRℓ )ˆλ ℓ. (77)
Therefore, Dr(R) is bounded below by the solution to the
following optimization problem:
min
(R1,R 2,...,R L), ( ˆR1, ˆR2,..., ˆRL)∈ RL
+
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√(1 − 2− 2Rℓ )
×
√
(1 − 2− 2 ˆRℓ )λ ℓˆλ ℓ
)
(78)
s.t. max
{ L∑
ℓ=1
Rℓ,
L∑
ℓ=1
ˆRℓ
}
≤ R. (79)
In view of the fact that
(1 − 2− 2Rℓ )(1 − 2− 2 ˆRℓ ) ≤ (1 − 2− (Rℓ + ˆRℓ ))2, (80)
there is no loss of generality in assuming Rℓ = ˆRℓ for ℓ =
1, 2, . . . , L . So the optimization problem reduces to
min
(R1,R 2,...,R ℓ )∈ RL
+
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2(1 − 2− 2Rℓ )
√λ ℓˆλ ℓ
)
(81)
s.t.
L∑
ℓ=1
Rℓ ≤ R. (82)
Deﬁne the Lagrangian
G :=
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2(1 − 2− 2Rℓ )
√
λ ℓˆλ ℓ
)
+ ν
n∑
ℓ=1
Rℓ,
(83)
where ν ≥ 0. Note that
dG
dRℓ
= − (4 ln 2)2− 2Rℓ
√
λ ℓˆλ ℓ + ν, ℓ = 1, 2, . . . , L. (84)
For ℓ = 1 , 2, . . . , L , setting dG
dRℓ
= 0 and taking into account
the constraint Rℓ ≥ 0 gives
Rℓ = 1
2 log+
(
(4 ln 2)
√
λ ℓˆλ ℓ
ν
)
= 1
2 log+
( √
λ ℓˆλ ℓ
β
)
, (85)
where β := ν
4 ln 2 . We obtain the minimizer
(R1(R), R2(R), . . . , RL(R)), as deﬁned in (22), by choosing
β to be the unique number in (0,
√
λ 1ˆλ 1] that satisﬁes the
constraint in (82) with equality. The proof is complete sinc e
this lower bound is attained when
1) U is jointly Gaussian with S such that the pairs
(S1, U 1), (S2, U 2), . . . , (SL, U L) are mutually indepen-
dent, and for ℓ = 1 , 2, . . . , L , the covariance matrix of
(Sℓ, U ℓ) is
(
λ ℓ (1 − 2− 2Rℓ (R))λ ℓ
(1 − 2− 2Rℓ (R))λ ℓ (1 − 2− 2Rℓ (R))λ ℓ
)
, (86)
2) ˆU is jointly Gaussian with ˆS such that the pairs
( ˆS1, ˆU1), ( ˆS2, ˆU2), . . . , ( ˆSL, ˆUL) are mutually indepen-
dent, and for ℓ = 1 , 2, . . . , L , the covariance matrix of
( ˆSℓ, ˆUℓ) is
(
ˆλ ℓ (1 − 2− 2Rℓ (R))ˆλ ℓ
(1 − 2− 2Rℓ (R))ˆλ ℓ (1 − 2− 2Rℓ (R))ˆλ ℓ
)
. (87)
It is instructive to compare Dr(R) with the greedy solution
proposed in [23], [24]. Let pU ′|S and p ˆU ′|ˆS be the minimizers
of
min
pU |S
E[∥S − U ∥2] (88)
s.t. I(S; U ) ≤ R, (89)
and
min
p ˆU |S
E[∥ ˆS − ˆU ∥2] (90)
s.t. I( ˆS; ˆU ) ≤ R, (91)

<!-- page 9 -->

9
respectively. According to the classical reverse waterﬁll ing
formula [40, Theorem 13.3.3], we have
E[∥S − U ′∥2] =
L∑
ℓ=1
2− 2R′
ℓ (R)λ ℓ, (92)
where
R′
ℓ(R) := 12 log+
(λ ℓ
̺
)
, ℓ = 1, 2, . . . , L, (93)
with ̺ being the unique number in (0, max{λ 1, λ 2, . . . , λ L}]
satisfying
1
2
L∑
ℓ=1
log+
(λ ℓ
̺
)
= R. (94)
Similary,
E[∥ ˆS − ˆU ′∥2] =
L∑
ℓ=1
2− 2 ˆR′
ℓ (R)ˆλ ℓ, (95)
where
ˆR′
ℓ(R) := 12 log+
( ˆλ ℓ
ˆ̺
)
, ℓ = 1, 2, . . . , L, (96)
with ˆ̺ being the unique number in (0, max{ˆλ 1, ˆλ 2, . . . , ˆλ L}]
satisfying
1
2
L∑
ℓ=1
log+
( ˆλ ℓ
ˆ̺
)
= R. (97)
Moreover, it can be veriﬁed that
W 2
2 (pU ′, p ˆU ′)
=
L∑
ℓ=1
(√
(1 − 2− 2R′
ℓ (R))λ ℓ −
√(1 − 2− 2 ˆR′
ℓ (R))ˆλ ℓ
)2
. (98)
Since U and U ′ automatically satisfy (70), summing (92), (95),
and (98) yields the following upper bound on
Dr(R):
L∑
ℓ=1
(
λ ℓ + ˆλ ℓ − 2
√
(1 − 2− 2R′
ℓ (R))(1 − 2− 2 ˆR′
ℓ (R))λ ℓˆλ ℓ
)
=:
D
′
r(R) (99)
This upper bound is not tight, except in certain special
cases (e.g., L = 1 or Λ = ˆΛ). Therefore, blindly applying
the reverse-waterﬁlling-based quantiztion and dequantiz ation
strategies is in general suboptimal for rate-constrained o ptimal
transport.
Fig. 4 compares
Dr(R) and D
′
r(R) for an illustrative
example. It can be seen that
D
′
r(R) is indeed suboptimal.
In particular,
D
′
r(R) = Dmax when R is sufﬁciently close
to zero. This occurs because the index sets corresponding
to the largest λ ℓ and the largest ˆλ ℓ are disjoint, leading
to the undesirable situation in the low-rate regime where
R′
ℓ(R) ˆR′
ℓ(R) = 0 for all ℓ. In contrast, for
Dr(R), the rates
allocated to Sℓ and ˆSℓ are both given by Rℓ(R), effectively
avoiding this issue.
0 1 2 3 4 5 6 7 8 9 10
0
2
4
6
8
10
12
Fig. 4. Plots of Dr(R) and D
′
r(R) for the case where (λ1, λ2, λ3) =
(2, 3, 1) and (ˆλ1, ˆλ2, ˆλ3) = (3 , 1, 1).
APPENDIX C
PROOF OF THEOREM 3
There is no loss of generality in assuming K ∈
{1, 2, . . . , L }. Let Z := E[S|φ(S)]. It can be veriﬁed that
E[∥S − ˆS∥2]
= E[∥S − Z∥2] + E[∥Z − ˆS∥2]
≥ E[∥S − Z∥2] + W 2
2 (N (0, ∆), N (0, ˆΛ))
= tr(Λ − ∆) + tr
(
∆ + ˆΛ − 2(ˆΛ
1
2 ∆ˆΛ
1
2 )
1
2
)
= tr
(
Λ + ˆΛ − 2(ˆΛ
1
2 ∆ˆΛ
1
2 )
1
2
)
, (100)
where ∆ denotes the covariance matrix of Z. Clearly, 0 ⪯
∆ ⪯ Λ and rank(∆) ≤ K. Therefore, E[∥S − ˆS∥2] is bounded
below by the solution to the following optimization problem :
min
∆
tr
(
Λ + ˆΛ − 2(ˆΛ
1
2 ∆ˆΛ
1
2 )
1
2
)
(101)
s.t. 0 ⪯ ∆ ⪯ Λ, rank(∆) ≤ K. (102)
Let Ξ := ( ˆΛ
1
2 ∆ˆΛ
1
2 )
1
2 . Since the square root is operator
monotone, it follows by the L¨ owner-Heinz theorem that Ξ ⪯
(ˆΛ
1
2 Λ ˆΛ
1
2 )
1
2 = Λ
1
2 ˆΛ
1
2 . Moreover, rank(Ξ) = rank(∆) . As a
consequence, we can establish a lower bound on E[∥S − ˆS∥2]
by relaxing the optimization problem in (101)–(102) to
min
Ξ
tr
(
Λ + ˆΛ − 2Ξ
)
(103)
s.t. 0 ⪯ Ξ ⪯ Λ
1
2 ˆΛ
1
2 , rank(Ξ) ≤ K. (104)
Note that (104) implies
tr(Ξ) ≤
K∑
ℓ=1
√
λ ℓˆλ ℓ (105)

<!-- page 10 -->

10
according to [41, Corollary 7.7.4]. This leads to the desire d
lower bound:
E[∥S − ˆS∥2]
≥ tr(Λ + ˆΛ) − 2
K∑
ℓ=1
√
λ ℓˆλ ℓ
=
K∑
ℓ=1
(√
λ ℓ −
√
ˆλ ℓ
)2
+
L∑
ℓ=K+1
(λ ℓ + ˆλ ℓ), (106)
which can be achieved by setting the ﬁrst K components of ˆS
to be the scaled versions of the corresponding components of
S and generating the remaining components of ˆS from scratch.
APPENDIX D
PROOF OF THEOREM 5
It is more convenient to represent Sn and ˆSn as two
nL-dimensional Gaussian random vectors with covariance
matrices
Λn :=





Λ 0 · · · 0
0 Λ · · · 0
.
.
.
.
.
. . . . .
.
.
0 0 · · · Λ




 , ˆΛn :=





ˆΛ 0 · · · 0
0 ˆΛ · · · 0
.
.
. .
.
. . . . .
.
.
0 0 · · · ˆΛ




 ,
respectively. Since φ is linear, we can write X n = AnSn,
where An is an n × (nL) matrix. Note that the input power
constraint (42) is equivalent to
tr(AnΛnAT
n ) ≤ nP. (107)
Let Z n := E[Sn|Y n]. The covariance matrix of Z n is given
by
∆n := Λ nAT
n (AnΛnAT
n + I)− 1AnΛn. (108)
It can be veriﬁed that
E[∥Sn − ˆSn∥2]
= E[∥Sn − Z n∥2] + E[∥Z n − ˆSn∥2]
≥ E[∥Sn − Z n∥2] + W 2
2 (N (0, ∆n), N (0, ˆΛn))
= tr(Λ n − ∆n) + tr
(
∆n + ˆΛn − 2(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
1
2
)
= tr
(
Λn + ˆΛn − 2(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
1
2
)
= − 2tr
(
(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
1
2
)
+ n
L∑
ℓ=1
(λ ℓ + ˆλ ℓ). (109)
Let σi(M ) denote the i-th largest eigenvalue of matrix M . We
have
tr
(
(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
1
2
)
=
nL∑
i=1
√
σi(ˆΛ
1
2
n ∆n ˆΛ
1
2
n ). (110)
Since rank(∆n) ≤ n, it follows that σi(ˆΛ
1
2
n ∆n ˆΛ
1
2
n ) = 0 for
i > n . As a consequence,
tr
(
(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
1
2
)
=
n∑
i=1
√
σi(ˆΛ
1
2
n ∆n ˆΛ
1
2
n ). (111)
For i = 1, 2, . . . , n ,
σi(ˆΛ
1
2
n ∆n ˆΛ
1
2
n )
= σi(ˆΛn∆n)
= σi(ˆΛnΛnAT
n (AnΛnAT
n + I)− 1AnΛn)
= σi(Λ
1
2
n ˆΛnΛnAT
n (AnΛnAT
n + I)− 1AnΛ
1
2
n )
≤ σ1(Λ
1
2
n ˆΛnΛ
1
2
n )σi(Λ
1
2
n AT
n (AnΛnAT
n + I)− 1AnΛ
1
2
n )
= λ 1ˆλ 1σi(Λ
1
2
n AT
n (AnΛnAT
n + I)− 1AnΛ
1
2
n )
= λ 1ˆλ 1σi(AnΛnAT
n (AnΛnAT
n + I)− 1)
= σi(AnΛnAT
n )
σi(AnΛnAT
n ) + 1 λ 1ˆλ 1, (112)
where the inequality follows by [42, Corollary 4.6.3]. More -
over, (107) can be written equivalently as
n∑
i=1
σi(AnΛnAT
n ) ≤ nP. (113)
Therefore, we have
E[∥Sn − ˆSn∥2] ≥ − 2ζ + n
L∑
ℓ=1
(λ ℓ + ˆλ ℓ), (114)
where
ζ := max
(σ 1,σ 2,...,σ n)∈ Rn
+
n∑
i=1
√
σi
σi + 1 λ 1ˆλ 1 (115)
s.t.
n∑
i=1
σi ≤ nP. (116)
Since
√
σ
σ +1 is concave in σ for σ ≥ 0, the maximum in (115)
is attained at σ1 = σ2 = . . . = σn = P , and consequently,
ζ = n
√
P
P + 1 λ 1ˆλ 1. (117)
Substituting (117) into (114) and dividing both sides by n
yields the desired lower bound.
APPENDIX E
PROOF OF THEOREM 6
It sufﬁces to consider the case L ≥ 2. In view of (44) and
Theorem 2,
D(s)
c (P ) = − 2
L∑
ℓ=1
(√
λ ℓˆλ ℓ − β
)
+
+
L∑
ℓ=1
(λ ℓ + ˆλ ℓ), (118)
with β being the unique number in (0,
√
λ 1ˆλ 1] satisfying
L∏
ℓ=1
max
{ √
λ ℓˆλ ℓ
β , 1
}
= P + 1. (119)
When P > 0, we must have β <
√
λ 1ˆλ 1. Let
δ∗ := 1 −
√
λ 1ˆλ 1 − β
βP . (120)

<!-- page 11 -->

11
It can be veriﬁed that δ∗ ∈ [0, 1) and
L∏
ℓ=2
max
{ √
λ ℓˆλ ℓ
β , 1
}
= P + 1
(1 − δ∗ )P + 1 . (121)
In light of (52), (53), and (121),
D(h)
c (P ) ≤ − 2
√
(1 − δ∗ )P
(1 − δ∗ )P + 1 λ 1ˆλ 1
− 2
L∑
ℓ=2
(√
λ ℓˆλ ℓ − β
)
+
+
L∑
ℓ=1
(λ ℓ + ˆλ ℓ). (122)
Therefore,
D(s)
c (P ) − D(h)
c (P )
≥ − 2
(√
λ 1ˆλ 1 − β
)
+ 2
√
(1 − δ∗ )P
(1 − δ∗ )P + 1 λ 1ˆλ 1
= − 2
(√
λ 1ˆλ 1 − β
)
+ 2
√
λ 1ˆλ 1 − β
√
λ 1 ˆλ 1
> − 2
(√
λ 1ˆλ 1 − β
)
+ 2
√
λ 1ˆλ 1 − 2β
√
λ 1 ˆλ 1 + β 2
= 0. (123)
This proves (54).
Note that
D(h)
c (P ) = min
δ∈ [0, 1]
{− 2f1(δ) − 2f2(δ)} +
L∑
ℓ=1
(λ ℓ + ˆλ ℓ),
(124)
where
f1(δ) :=
√
(1 − δ)P
(1 − δ)P + 1 λ 1ˆλ 1,
f2(δ) :=
L∑
ℓ=2
(√
λ ℓˆλ ℓ − β (δ)
)
+
. (125)
The proof of (55) boils down to determining the condition
under which the minimum in (124) is attained at δ = 0 .
Clearly,
df1(δ)
dδ = − 1
2
√
P
(1 − δ)((1 − δ)P + 1)3 λ 1ˆλ 1
≤ − 1
2
√
P
(P + 1)3 λ 1ˆλ 1. (126)
It can be veriﬁed that
f2(δ) = − κ(δ)

 (1 − δ)P + 1
P + 1
κ (δ)+1∏
ℓ=2
√
λ ℓˆλ ℓ


1
κ (δ )
+
κ (δ)+1∑
ℓ=2
√
λ ℓˆλ ℓ, (127)
where κ(δ) denotes the largest ℓ ∈ { 1, 2, . . . , L − 1} satisfying√
λ ℓ+1ˆλ ℓ+1 ≥ β (δ). Since κ(δ) is a piecewise constant
function of δ, we have
df2(δ)
dδ = P
(1 − δ)P + 1

 (1 − δ)P + 1
P + 1
κ (δ)+1∏
ℓ=2
√
λ ℓˆλ ℓ


1
κ (δ )
(128)
within each interval of δ where κ(δ) is ﬁxed. The expression
in (128) is maximized when κ(δ) = 1 , yielding
df2(δ)
dδ ≤ P
P + 1
√
λ 2ˆλ 2. (129)
Therefore,
df1(δ)
dδ + df2(δ)
dδ ≤ − 1
2
√
P
(P + 1)3 λ 1ˆλ 1 + P
P + 1
√
λ 2 ˆλ 2.
(130)
The solution to
− 1
2
√
P
(P + 1)3 λ 1ˆλ 1 + P
P + 1
√
λ 2ˆλ 2 = 0 (131)
is given by P = P ∗ deﬁned in (56). For P ∈ [0, P ∗],
df1(δ)
dδ + df2(δ)
dδ ≤ 0, (132)
which implies that the minimum in (124) is attained at δ = 0.
On the other hand, for P > P ∗ , we have
df1(δ)
dδ
⏐
⏐
⏐
⏐
δ=0
+ df2(δ)
dδ
⏐
⏐
⏐
⏐
δ=0
= − 1
2
√
P
(P + 1)3 λ 1ˆλ 1 + P
P + 1
√
λ 2ˆλ 2
> 0, (133)
and consequently, the minimum in (124) is not attained at
δ = 0. This completes the proof of (55).
REFERENCES
[1] C. Villani, Topics in Optimal Transport . Providence, RI, USA: American
Mathematical Society, 2003.
[2] S. Kolouri, S. R. Park, M. Thorpe, D. Slepcev and G. K. Rohd e, ”Optimal
Mass Transport: Signal processing and machine-learning ap plications,”
IEEE Signal Process. Mag. , vol. 34, no. 4, pp. 43–59, Jul. 2017.
[3] E. F. Montesuma, F. M. N. Mboula and A. Souloumiac, ”Recen t advances
in optimal transport for machine learning,” IEEE Trans. Pattern Anal.
Mach. Intell. , vol. 47, no. 2, pp. 1161–1180, Feb. 2025.
[4] M. Li, J. Klejsa, and W. B. Kleijn, “Distribution preserv ing quantization
with dithering and transformation,” IEEE Signal Process. Lett. , vol. 17,
no. 12, pp. 1014–1017, Dec. 2010.
[5] J. Klejsa, G. Zhang, M. Li, and W. B. Kleijn, “Multiple des cription
distribution preserving quantization,” IEEE Trans. Signal Process. , vol.
61, no. 24, pp. 6410–6422, Dec. 2013.
[6] N. Saldi, T. Linder, and S. Y¨ uksel, “Randomized quantiz ation and source
coding with constrained output distribution,” IEEE Trans. Inf. Theory ,
vol. 61, no. 1, pp. 91–106, Jan. 2015.
[7] N. Saldi, T. Linder, and S. Y¨ uksel, “Output constrained lossy source
coding with limited common randomness,” IEEE Trans. Inf. Theory ,
vol. 61, no. 9, pp. 4984–4998, Sep. 2015.
[8] Y . Blau and T. Michaeli, “Rethinking lossy compression: The rate-
distortion-perception tradeoff,” in Proc. ACM Int. Conf. Mach. Learn.
(ICML), 2019, pp. 675–685.

<!-- page 12 -->

12
[9] R. Matsumoto, “Introducing the perception-distortion tradeoff into the
rate-distortion theory of general information sources,” IEICE Comm.
Express, vol. 7, no. 11, pp. 427–431, 2018.
[10] R. Matsumoto, “Rate-distortion-perception tradeoff of variable-length
source coding for general information sources,” IEICE Comm. Express ,
vol. 8, no. 2, pp. 38–42, 2019.
[11] Z. Y an, F. Wen, R. Ying, C. Ma, and P . Liu, “On perceptual l ossy
compression: The cost of perceptual reconstruction and an o ptimal
training framework,” in Proc. ACM Int. Conf. Mach. Learn. (ICML), 2021,
pp. 11682–11692.
[12] L. Theis and A. B. Wagner, “A coding theorem for the rate- distortion-
perception function,” in Proc. Neural Compress. W orkshop Int. Conf.
Learn. Represent. (ICLR) , 2021, pp. 1–5.
[13] L. Theis and E. Agustsson, “On the advantages of stochas tic encoders,”
in Proc. Neural Compress. W orkshop Int. Conf. Learn. Represent. (ICLR),
2021, pp. 1–8.
[14] G. Zhang, J. Qian, J. Chen, and A. Khisti, ”Universal rat e-distortion-
perception representations for lossy compression,” in Proc. Adv. Neural
Inf. Process. Syst. (NeurIPS) , 2021, pp. 11517–11529.
[15] J. Chen, L. Y u, J. Wang, W. Shi, Y . Ge, and W. Tong, “On the r ate-
distortion-perception function,” IEEE J. Sel. Areas Inf. Theory , vol. 3, no.
4, pp. 664–673, Dec. 2022.
[16] S. Salehkalaibar, B. Phan, J. Chen, W. Y u, and A. Khisti, “On the choice
of perception loss function for learned video compression, ” in Proc. Adv.
Neural Inf. Process. Syst. (NeurIPS) , 2023, pp. 1–19.
[17] Y . Hamdi, A. B. Wagner and D. G¨ und¨ uz, “The rate-distortion-perception
trade-off: the role of private randomness,” in Proc. IEEE Int. Symp. Inf.
Theory (ISIT) , 2024, pp. 1083–1088.
[18] S. Salehkalaibar, J. Chen, A. Khisti, and W. Y u, “Rate-d istortion-
perception tradeoff based on the conditional-distributio n perception mea-
sure,” IEEE Trans. Inf. Theory , vol. 70, no. 12, pp. 8432–8454, Dec.
2024.
[19] G. Serra, P . A. Stavrou and M. Kountouris, “On the comput ation of
the Gaussian rate–distortion–perception function,” IEEE J. Sel. Areas Inf.
Theory, vol. 5, pp. 314–330, 2024.
[20] J. Qian, S. Salehkalaibar, J. Chen, A. Khisti, W. Y u, W. S hi, Y . Ge,
and W. Tong, “Rate-distortion-perception tradeoff for vec tor Gaussian
sources,” IEEE J. Sel. Areas Inf. Theory , vol. 6, pp. 1–17, 2025.
[21] L. Xie, L. Li, J. Chen, and Z. Zhang, “Output-constraine d lossy
source coding with application to rate-distortion-percep tion theory,” 2024,
arXiv:2403.14849. [Online] Available: https://arxiv.or g/abs/2403.14849
[22] L. Xie, L. Li, J. Chen, L. Y u, and Z. Zhang, “Gaussian rate -distortion-
perception coding and entropy-constrained scalar quantiz ation,” 2024,
arXiv:2409.02388. [Online] Available: https://arxiv.or g/abs/2409.02388
[23] H. Liu, G. Zhang, J. Chen, A. Khisti, “Lossy compression with distri-
bution shift as entropy constrained optimal transport,” in Proc. Int. Conf.
Learn. Represent. (ICLR) , 2022, pp. 1–34.
[24] H. Liu, G. Zhang, J. Chen and A. Khisti, “Cross-domain lo ssy com-
pression as entropy constrained optimal transport,” IEEE J. Sel. Areas
Inf. Theory , vol. 3, no. 3, pp. 513–527, Sep. 2022.
[25] H. M. Garmaroudi, S. Sandeep Pradhan and J. Chen, ”Rate- limited
quantum-to-classical optimal transport in ﬁnite and conti nuous-variable
quantum systems,” IEEE Trans. Inf. Theory , vol. 70, no. 11, pp. 7892–
7922, Nov. 2024.
[26] D. Donoho, “Compressed sensing,” IEEE Trans. Inf. Theory , vol. 52,
no. 4, pp. 1289–1306, Apr. 2006.
[27] E. Candes, J. Romberg, and T. Tao, “Robust uncertainty p rinciples: Exact
signal reconstruction from highly incomplete frequency in formation,”
IEEE Trans. Inf. Theory , vol. 52, no. 2, pp. 489–509, Feb. 2006.
[28] Y . Wu and S. V erd´ u, “R´ enyi information dimension: Fundamental limits
of almost lossless analog compression,” IEEE Trans. Inf. Theory , vol. 56,
no. 8, pp. 3721–3748, Aug. 2010.
[29] T. Jolliffe, Principal Component Analysis , 2nd ed. New Y ork, NY , USA:
Springer, 2002.
[30] X. Qu, R. Li, J. Chen, L. Y u, and X. Wang, “Channel-aware o ptimal
transport: A theoretical framework for generative communi cation,” 2024,
arXiv:2412.19025. [Online] Available: https://arxiv.or g/abs/2412.19025
[31] E. Bourtsoulatze, D. B. Kurka, and D. G¨ und¨ uz, “Deep jo int source-
channel coding for wireless image transmission,” IEEE Trans. on Cogn.
Commun. Netw., vol. 5, no. 3, pp. 567–579, Sep. 2019.
[32] D. B. Kurka and D. G¨ und¨ uz, “DeepJSCC-f: Deep joint sou rce-channel
coding of images with feedback,” IEEE J. Sel. Areas Commun. , vol. 1,
no. 1, pp. 178–193, May 2020,
[33] D. B. Kurka and D. G¨ und¨ uz, “Bandwidth-agile image tra nsmission
with deep joint source-channel coding,” IEEE Trans. Wireless Commun. ,
vol. 20, no. 12, pp. 8081–8095, Dec. 2021.
[34] T. -Y . Tung and D. G¨ und¨ uz, “DeepWiV e: Deep-learning-aided wireless
video transmission,” IEEE J. Sel. Areas Commun. , vol. 40, no. 9,
pp. 2570–2583, Sep. 2022.
[35] E. Erdemir, T. -Y . Tung, P . L. Dragotti and D. G¨ und¨ uz, “ Generative
joint source-channel coding for semantic image transmissi on,” IEEE J.
Sel. Areas Commun. , vol. 41, no. 8, pp. 2645–2657, Aug. 2023.
[36] D. C. Dowson and B. V . Landau, “ The Fr´ echet distance bet ween
multivariate normal distributions,” J. Multivariate Anal. , vol. 12, no. 3,
pp. 450—455, 1982.
[37] I. Olkin and F. Pukelsheim, “The distance between two ra ndom vectors
with given dispersion matrices,” Linear Algebra Appl. , vol. 48, pp. 257–
263, 1982.
[38] M. Knott and C. S. Smith, “On the optimal mapping of distr ibutions,”
J. Optim. Theory Appl. , vol. 43, no. 1, pp. 39–49, 1984.
[39] C. R. Givens and R. M. Shortt, “A class of Wasserstein met rics for
probability,” Michigan Math. J. , vol. 31, no. 2, pp. 231–240, 1984.
[40] T. M. Cover and J. A. Thomas, Elements of Information Theory . New
Y ork, NY , USA: Wiley, 1991.
[41] R. A. Horn and C. R. Johnson, Matrix Analysis . Cambridge, U.K.:
Cambridge Univ. Press, 1985.
[42] S. G. Wang, M. X. Wu, and Z. Z. Jia, Matrix Inequalities , 2nd ed.
Beijing, China: Chinese Science Press, 2006.
