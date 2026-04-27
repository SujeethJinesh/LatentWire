# references/126_nonlinear_shrinkage_estimation_of_large_dimensional_covariance_matrices.pdf

<!-- page 1 -->

arXiv:1207.5322v1  [math.ST]  23 Jul 2012
The Annals of Statistics
2012, Vol. 40, No. 2, 1024–1060
DOI:
10.1214/12-AOS989
c⃝ Institute of Mathematical Statistics , 2012
NONLINEAR SHRINKAGE ESTIMATION OF
LARGE-DIMENSIONAL COV ARIANCE MATRICES
By Olivier Ledoit and Michael Wolf 1
University of Zurich
Many statistical applications require an estimate of a cova riance
matrix and/or its inverse. When the matrix dimension is larg e com-
pared to the sample size, which happens frequently, the samp le co-
variance matrix is known to perform poorly and may suﬀer from
ill-conditioning. There already exists an extensive liter ature concern-
ing improved estimators in such situations. In the absence o f fur-
ther knowledge about the structure of the true covariance ma trix,
the most successful approach so far, arguably, has been shri nkage
estimation. Shrinking the sample covariance matrix to a mul tiple of
the identity, by taking a weighted average of the two, turns o ut to
be equivalent to linearly shrinking the sample eigenvalues to their
grand mean, while retaining the sample eigenvectors. Our pa per ex-
tends this approach by considering nonlinear transformati ons of the
sample eigenvalues. We show how to construct an estimator th at is
asymptotically equivalent to an oracle estimator suggeste d in pre-
vious work. As demonstrated in extensive Monte Carlo simula tions,
the resulting bona ﬁde estimator can result in sizeable improvements
over the sample covariance matrix and also over linear shrin kage.
1. Introduction. Many statistical applications require an estimate of
a covariance matrix and/or of its inverse when the matrix dim ension, p,
is large compared to the sample size, n. It is well known that in such situa-
tions, the usual estimator—the sample covariance matrix—p erforms poorly.
It tends to be far from the population covariance matrix and i ll-conditioned.
The goal then becomes to ﬁnd estimators that outperform the s ample co-
variance matrix, both in ﬁnite samples and asymptotically. For the purposes
of asymptotic analyses, to reﬂect the fact that p is large compared to n, one
Received October 2010; revised December 2011.
1Supported by the NCCR Finrisk project “New Methods in Theore tical and Empirical
Asset Pricing.”
AMS 2000 subject classiﬁcations. Primary 62H12; secondary 62G20, 15A52.
Key words and phrases. Large-dimensional asymptotics, nonlinear shrinkage, rot ation
equivariance.
This is an electronic reprint of the original article published by the
Institute of Mathematical Statistics in The Annals of Statistics ,
2012, Vol. 40, No. 2, 1024–1060 . This reprint diﬀers from the original in
pagination and typographic detail.
1

<!-- page 2 -->

2 O. LEDOIT AND M. WOLF
has to employ large-dimensional asymptotics where p is allowed to go to
inﬁnity together with n. In contrast, standard asymptotics would assume
that p remains ﬁxed while n tends to inﬁnity.
One way to come up with improved estimators is to incorporate additional
knowledge in the estimation process, such as sparseness, a g raph model or
a factor model; for example, see Bickel and Levina ( 2008), Rohde and Tsy-
bakov ( 2011), Cai and Zhou ( 2012), Ravikumar et al. ( 2008), Rajaratnam,
Massam and Carvalho ( 2008), Khare and Rajaratnam ( 2011), Fan, Fan and
Lv ( 2008) and the references therein.
However, not always is such additional knowledge available or trustwor-
thy. In this general case, it is reasonable to require that co variance matrix
estimators be rotation-equivariant. This means that rotat ing the data by
some orthogonal matrix rotates the estimator in exactly the same way. In
terms of the well-known decomposition of a matrix into eigen vectors and
eigenvalues, an estimator is rotation-equivariant if and o nly if it has the
same eigenvectors as the sample covariance matrix. Therefo re, it can only
diﬀerentiate itself by its eigenvalues.
Ledoit and Wolf ( 2004) demonstrate that the largest sample eigenvalues
are systematically biased upwards, and the smallest ones do wnwards. It is
advantageous to correct this bias by pulling down the larges t eigenvalues
and pushing up the smallest ones, toward the grand mean of all sample
eigenvalues. This is an application of the general shrinkag e principle, going
back to Stein ( 1956). Working under large-dimensional asymptotics, Ledoit
and Wolf ( 2004) derive the optimal linear shrinkage formula (when the loss
is deﬁned as the Frobenius norm of the diﬀerence between the es timator and
the true covariance matrix). The same shrinkage intensity i s applied to all
sample eigenvalues, regardless of their positions. For exa mple, if the linear
shrinkage intensity is 0.5, then every sample eigenvalue is moved half-way
toward the grand mean of all sample eigenvalues. Ledoit and W olf ( 2004)
both derive asymptotic optimality properties of the result ing estimator of
the covariance matrix and demonstrate that it has desirable ﬁnite-sample
properties via simulation studies.
A cursory glance at the Marˇ cenko and Pastur ( 1967) equation, which
governs the relationship between sample and population eig envalues under
large-dimensional asymptotics, shows that linear shrinka ge is the ﬁrst-order
approximation to a fundamentally nonlinear problem. How go od is this ap-
proximation? Ledoit and Wolf ( 2004) are very clear about this. Depending
on the situation at hand, the improvement over the sample cov ariance ma-
trix can either be gigantic or minuscule. When p/n is large, and/or the
population eigenvalues are close to one another, linear shr inkage captures
most of the potential improvement over the sample covarianc e matrix. In
the opposite case, that is, when p/n is small and/or the population eigen-
values are dispersed, linear shrinkage hardly improves at a ll over the sample
covariance matrix.

<!-- page 3 -->

NONLINEAR SHRINKAGE ESTIMATION 3
The intuition behind the present paper is that the ﬁrst-orde r approxima-
tion does not deliver a suﬃcient improvement when higher-or der eﬀects are
too pronounced. The cure is to upgrade to nonlinear shrinkage estimation
of the covariance matrix. We get away from the one-size-ﬁts- all approach by
applying an individualized shrinkage intensity to every sa mple eigenvalue.
This is more challenging mathematically than linear shrink age because many
more parameters need to be estimated, but it is worth the extr a eﬀort. Such
an estimator has the potential to asymptotically at least ma tch the linear
shrinkage estimator of Ledoit and Wolf ( 2004) and often do a lot better,
especially when linear shrinkage does not deliver a suﬃcien t improvement
over the sample covariance matrix. As will be shown later in t he paper,
this is indeed what we achieve here. By providing substantia l improvement
over the sample covariance matrix throughout the entire par ameter space,
instead of just part of it, the nonlinear shrinkage estimato r is as much of
a step forward relative to linear shrinkage as linear shrink age was relative to
the sample covariance matrix. In terms of ﬁnite-sample perf ormance, the lin-
ear shrinkage estimator rarely performs better than the non linear shrinkage
estimator. This happens only when the linear shrinkage esti mator is (nearly)
optimal already. However, as we show in simulations, the out performance
over the nonlinear shrinkage estimator is very small in such cases. Most of
the time, the linear shrinkage estimator is far from optimal , and nonlinear
shrinkage then oﬀers a considerable amount of ﬁnite-sample i mprovement.
A formula for nonlinear shrinkage intensities has recently been proposed
by Ledoit and P´ ech´ e (2011). It is motivated by a large-dimensional asymp-
totic approximation to the optimal ﬁnite-sample rotation- equivariant shrink-
age formula under the Frobenius norm. The advantage of the fo rmula of
Ledoit and P´ ech´ e (2011) is that it does not depend on the unobservable
population covariance matrix: it only depends on the distri bution of sam-
ple eigenvalues. The disadvantage is that the resulting cov ariance matrix
estimator is an oracle estimator in that it depends on the “limiting” distri-
bution of sample eigenvalues, not the observed one. These tw o objects are
very diﬀerent. Most critically, the limiting empirical cumu lative distribution
function (c.d.f.) of sample eigenvalues is continuously di ﬀerentiable, whereas
the observed one is, by construction, a step function.
The main contribution of the present paper is to obtain a bona ﬁde estima-
tor of the covariance matrix that is asymptotically as good a s the oracle esti-
mator. This is done by consistently estimating the oracle no nlinear shrinkage
intensities of Ledoit and P´ ech´ e (2011), in a uniform sense. As a by-product,
we also derive a new estimator of the limiting empirical c.d. f. of population
eigenvalues. A previous such estimator was proposed by El Ka roui ( 2008).
Extensive Monte Carlo simulations indicate that our covari ance matrix
estimator improves substantially over the sample covarian ce matrix, even
for matrix dimensions as low as p = 30. As expected, in some situations
the nonlinear shrinkage estimator performs as well as Ledoi t and Wolf’s

<!-- page 4 -->

4 O. LEDOIT AND M. WOLF
(2004) linear shrinkage estimator, while in others, where higher -order eﬀects
are more pronounced, it does substantially better. Since th e magnitude of
higher-order eﬀects depends on the population covariance ma trix, which is
unobservable, it is always safer a priori to use nonlinear shrinkage.
Many statistical applications require an estimate of the pr ecision matrix,
which is the inverse of the covariance matrix, instead of (or in addition to)
an estimate of the covariance matrix itself. Of course, one p ossibility is to
simply take the inverse of the nonlinear shrinkage estimate of the covariance
matrix itself. However, this would be ad hoc . The superior approach is to
estimate the inverse covariance matrix directly by nonline arly shrinking the
inverses of the sample eigenvalues. This gives quite diﬀeren t and markedly
better results. We provide a detailed, in-depth solution fo r this important
problem as well.
The remainder of the paper is organized as follows. Section 2 deﬁnes our
framework for large-dimensional asymptotics and reviews s ome fundamen-
tal results from the corresponding literature. Section 3 presents the oracle
shrinkage estimator that motivates our bona ﬁde nonlinear shrinkage esti-
mator. Sections 4 and 5 show that the bona ﬁde estimator is consistent for
the oracle estimator. Section 6 examines ﬁnite-sample behavior via Monte
Carlo simulations. Finally, Section 7 concludes. All mathematical proofs are
collected in the supplement [Ledoit and Wolf ( 2012)].
2. Large-dimensional asymptotics.
2.1. Basic framework. Let n denote the sample size and p ≡ p(n) the
number of variables, with p/n → c ∈ (0, 1) as n → ∞ . This framework is
known as large-dimensional asymptotics. The restriction t o the case c < 1
that we make here somewhat simpliﬁes certain mathematical r esults as well
as the implementation of our routines in software. The case c > 1, where the
sample covariance matrix is singular, could be handled by si milar methods,
but is left to future research.
The following set of assumptions will be maintained through out the paper.
(A1) The population covariance matrix Σ n is a nonrandom p-dimensional
positive deﬁnite matrix.
(A2) Let Xn be an n × p matrix of real independent and identically dis-
tributed (i.i.d.) random variables with zero mean and unit v ariance.
One only observes Yn ≡ XnΣ1/2
n , so neither Xn nor Σ n are observed
on their own.
(A3) Let (( τn,1, . . . , τn,p); (vn,1, . . . , vn,p)) denote a system of eigenvalues and
eigenvectors of Σ n. The empirical distribution function (e.d.f.) of the
population eigenvalues is deﬁned as ∀t ∈ R, Hn(t) ≡ p− 1 ∑p
i=1 1[τn,i,+∞ )(t),
where 1 denotes the indicator function of a set. We assume Hn(t) con-
verges to some limit H(t) at all points of continuity of H.

<!-- page 5 -->

NONLINEAR SHRINKAGE ESTIMATION 5
(A4) Supp( H), the support of H, is the union of a ﬁnite number of closed
intervals, bounded away from zero and inﬁnity. Furthermore , there
exists a compact interval in (0 , +∞ ) that contains Supp( Hn) for all n
large enough.
Let (( λn,1, . . . , λn,p); (un,1, . . . , un,p)) denote a system of eigenvalues and
eigenvectors of the sample covariance matrix Sn ≡ n− 1Y ′
nYn = n− 1Σ1/2
n X ′
nXn ×
Σ1/2
n . We can assume that the eigenvalues are sorted in increasing order
without loss of generality (w.l.o.g.). The ﬁrst subscript, n, will be omitted
when no confusion is possible. The e.d.f. of the sample eigen values is deﬁned
as ∀λ ∈ R, Fn(λ) ≡ p− 1 ∑p
i=1 1[λi,+∞ )(λ).
In the remainder of the paper, we shall use the notation Re( z) and Im( z)
for the real and imaginary parts, respectively, of a complex number z, so
that
∀z ∈ C z = Re(z) + i ·Im(z).
The Stieltjes transform of a nondecreasing function G is deﬁned by
∀z ∈ C+ mG(z) ≡
∫ +∞
−∞
1
λ − z dG(λ),(2.1)
where C+ is the half-plane of complex numbers with strictly positive imag-
inary part. The Stieltjes transform has a well-known invers ion formula,
G(b) − G(a) = lim
η→ 0+
1
π
∫ b
a
Im[mG(ξ + iη)] dξ,
which holds if G is continuous at a and b. Thus, the Stieltjes transform of
the e.d.f. of sample eigenvalues is
∀z ∈ C+ mFn(z) = 1
p
p∑
i=1
1
λi − z = 1
p Tr[(Sn − zI )− 1],
where I denotes a conformable identity matrix.
2.2. Marˇ cenko–Pastur equation and reformulations. Marˇ cenko and Pas-
tur ( 1967) and others have proven that Fn(λ) converges almost surely (a.s.)
to some nonrandom limit F (λ) at all points of continuity of F under certain
sets of assumptions. Furthermore, Marˇ cenko and Pastur discovered the equa-
tion that relates mF to H. The most convenient expression of the Marˇ cenko–
Pastur equation is the one found in Silverstein [( 1995), equation (1.4)],
∀z ∈ C+ mF (z) =
∫ +∞
−∞
1
τ [1 − c − czmF (z)] − z dH(τ ).(2.2)
This version of the Marˇ cenko–Pastur equation is the one tha t we start out
with. In addition, Silverstein and Choi (
1995) showed that
∀λ ∈ R − { 0} lim
z∈ C+→ λ
mF (z) ≡ ˘mF (λ)

<!-- page 6 -->

6 O. LEDOIT AND M. WOLF
exists, and that F has a continuous derivative F ′ = π− 1Im[ ˘mF ] on all of R
with F ′≡ 0 on ( −∞ , 0]. For purposes that will become apparent later, it is
useful to reformulate the Marˇ cenko–Pastur equation.
The limiting e.d.f. of the eigenvalues of n− 1Y ′
nYn = n− 1Σ1/2
n X ′
nXnΣ1/2
n
was deﬁned as F . In addition, deﬁne the limiting e.d.f. of the eigenvalues o f
n− 1YnY ′
n = n− 1XnΣnX ′
n as F . It then holds
∀x ∈ R F (x) = (1 − c)1[0,+∞ )(x) + cF (x),
∀x ∈ R F (x) = c − 1
c 1[0,+∞ )(x) + 1
c F (x),
∀z ∈ C+ mF (z) = c − 1
z + cmF (z),
∀z ∈ C+ mF (z) = 1 − c
cz + 1
c mF (z).
With this notation, equation (1.3) of Silverstein and Choi ( 1995) rewrites
the Marˇ cenko–Pastur equation in the following way: for each z ∈ C+, mF (z)
is the unique solution in C+ to the equation
mF (z) = −
[
z − c
∫ +∞
−∞
τ
1 + τ mF (z) dH(τ )
] − 1
.(2.3)
Now introduce uF
(z) ≡ − 1/mF (z). Notice that uF (z) ∈ C+ ⇐ ⇒ mF (z) ∈
C+. The mapping from uF (z) to mF (z) is one-to-one on C+.
With this change of variable, equation ( 2.3) is equivalent to saying that
for each z ∈ C+, uF (z) is the unique solution in C+ to the equation
uF (z) = z + cuF (z)
∫ +∞
−∞
τ
τ − uF (z) dH(τ ).(2.4)
Let the linear operator L transform any c.d.f. G into
LG(x) ≡
∫ x
−∞
τ dG(τ ).
Combining L with the Stieltjes transform, we get
mLG(z) =
∫ +∞
−∞
τ
τ − z dG(τ ) = 1 + zmG(z).
Thus, we can rewrite equation ( 2.4) more concisely as
uF (z) = z + cuF (z)mLH (uF (z)).(2.5)
As Silverstein and Choi [(
1995), equation (1.4)] explain, the function deﬁned
in equation ( 2.3) is invertible. Thus we can deﬁne the inverse function
zF (m) ≡ − 1
m + c
∫ +∞
−∞
τ
1 + τ m dH(τ ).(2.6)

<!-- page 7 -->

NONLINEAR SHRINKAGE ESTIMATION 7
We can do the same thing for equation ( 2.5) and deﬁne the inverse function
˜zF (u) ≡ u − cumLH (u).(2.7)
Equations (
2.2), ( 2.3), ( 2.5), ( 2.6) and ( 2.7) are all completely equivalent to
one another; solving any one of them means having solved them all. They
are all just reformulations of the Marˇ cenko–Pastur equati on.
As will be detailed in Section 3, the oracle nonlinear shrinkage estimator
of Σn involves the quantity ˘mF (λ), for various inputs λ. Section 2.3 describes
how this quantity can be found in the hypothetical case that F and H are
actually known. This will then allow us later to discuss cons istent estimation
of ˘mF (λ) in the realistic case when F and H are unknown.
2.3. Solving the Marˇ cenko–Pastur equation. Silverstein and Choi ( 1995)
explain how the support of F , denoted by Supp( F ), is determined. Let
B ≡ { u ∈ R : u ̸= 0, u ∈ Supp∁(H)}. Then plot the function ˜ zF (u) of ( 2.7) on
the set B. Find the extreme values on each interval. Delete these poin ts and
everything in between on the real line. Do this for all increa sing intervals.
What is left is just Supp( F ); see Figure 1 of Bai and Silverstein ( 1998) for
an illustration.
To simplify, we will assume from here on that Supp( F ) is a single com-
pact interval, bounded away from zero, with F ′ > 0 in the interior of this
interval. But if Supp( F ) is the union of a ﬁnite number of such intervals,
the arguments presented in this section as well as in the rema inder of the
paper apply separately to each interval. In particular, our consistency re-
sults presented in subsequent sections can be easily extend ed to this more
general case. On the other hand, the even more general case of Supp(F ) be-
ing the union of an inﬁnite number of such intervals or being a noncompact
interval is ruled out by assumption (A4). By our assumption t hen, Supp( F )
is given by the compact interval [ ˜zF (u1), ˜zF (u2)] for some u1 < u2. To keep
the notation shorter in what follows, let ˜z1 ≡ ˜zF (u1) and ˜z2 ≡ ˜zF (u2).
We know that for every λ in the interior of Supp( F ), there exists a unique
v ∈ C+, denoted by vλ, such that
vλ − cvλmLH (vλ) = λ.(2.8)
We further know that
F ′(λ) = 1
c F ′(λ) = 1
cπ Im[ ˘mF (λ)] = 1
cπ Im
[
− 1
vλ
]
.
The converse is also true. Since Supp( F ) = [˜zF (u1), ˜zF (u2)], for every x ∈
(u1, u2), there exists a unique y > 0, denoted by yx, such that
(x + iyx) − c(x + iyx)mLH (x + iyx) ∈ R.

<!-- page 8 -->

8 O. LEDOIT AND M. WOLF
In other words, yx is the unique value of y > 0 for which Im[( x + iy) − c(x +
iy)mLH (x + iy)] = 0. Also, if λx denotes the value of λ for which we have
(x + iyx) − c(x + iyx)mLH (x + iyx) = λ, then, by deﬁnition, zλx = x + iyx.
Once we ﬁnd a way to consistently estimate yx for any x ∈ [u1, u2], then
we have an estimate of the (asymptotic) solution to the Marˇ c enko–Pastur
equation. For example, Im[ − 1/(x + iyx)]/(cπ) is the value of the density F ′
evaluated at Re[( x + iyx) − c(x + iyx)mLH (x + iyx)] = ( x + iyx) − c(x +
iyx)mLH (x + iyx).
From the above arguments, it follows that
∀λ ∈ (˜z1, ˜z2) ˘ mF (λ) = − 1
vλ
and so ˘ mF (λ) = 1 − c
cλ − 1
c
1
vλ
.(2.9)
3. Oracle estimator.
3.1. Covariance matrix. In the absence of speciﬁc information about the
true covariance matrix Σ n, it appears reasonable to restrict attention to the
class of estimators that are equivariant with respect to rot ations of the
observed data. To be more speciﬁc, let W be an arbitrary p-dimensional or-
thogonal matrix. Let ˆΣn ≡ ˆΣn(Yn) be an estimator of Σ n. Then the estimator
is said to be rotation-equivariant if it satisﬁes ˆΣn(YnW ) = W ′ˆΣn(Yn)W . In
other words, the estimate based on the rotated data equals th e rotation of
the estimate based on the original data. The class of rotatio n-equivariant
estimators of the covariance matrix is constituted of all th e estimators that
have the same eigenvectors as the sample covariance matrix; for example,
see
Perlman [(2007), Section 5.4]. Every rotation-equivariant estimator is
thus of the form
UnDnU ′
n where Dn ≡ Diag(d1, . . . , dp) is diagonal ,
and where Un is the matrix whose ith column is the sample eigenvector
ui ≡ un,i. This is the class we consider.
The starting objective is to ﬁnd the matrix in this class that is closest
to Σ n. To measure distance, we choose the Frobenius norm deﬁned as
∥A∥ ≡
√
Tr(AA′)/r for any matrix A of dimension r × m.(3.1)
[Dividing by the dimension of the square matrix AA′ inside the root is not
standard, but we do this for asymptotic purposes so that the F robenius
norm remains constant equal to one for the identity matrix re gardless of the
dimension; see Ledoit and Wolf (
2004).] As a result, we end up with the
following minimization problem:
min
Dn
∥UnDnU ′
n − Σn∥.
Elementary matrix algebra shows that its solution is
D∗
n ≡ Diag(d∗
1, . . . , d∗
p) where d∗
i ≡ u′
iΣnui for i = 1, . . . , p.(3.2)

<!-- page 9 -->

NONLINEAR SHRINKAGE ESTIMATION 9
The interpretation of d∗
i is that it captures how the ith sample eigenvector ui
relates to the population covariance matrix Σ n as a whole. As a result, the
ﬁnite-sample optimal estimator is given by
S∗
n ≡ UnD∗
nU ′
n where D∗
n is deﬁned as in (
3.2).(3.3)
By generalizing the Marˇ cenko–Pastur equation ( 2.2), Ledoit and P´ ech´ e
(2011) show that d∗
i can be approximated by the quantity
dor
i ≡ λi
|1 − c − cλi ˘mF (λi)|2 for i = 1, . . . , p,(3.4)
from which they deduce their oracle estimator
Sor
n ≡ UnDor
n U ′
n where Dor
n ≡ Diag(dor
1 , . . . , dor
p ).(3.5)
The key diﬀerence between D∗
n and Dor
n is that the former depends on the
unobservable population covariance matrix, whereas the la tter depends on
the limiting distribution of sample eigenvalues, which mak es it amenable to
estimation, as explained below.
Note that Sor
n constitutes a nonlinear shrinkage estimator: since the val ue
of the denominator of dor
i varies with λi, the shrunken eigenvalues dor
i are ob-
tained by applying a nonlinear transformation to the sample eigenvalues λi;
see Figure
3 for an illustration. Ledoit and P´ ech´ e (2011) also illustrate in
some (limited) simulations that this oracle estimator can p rovide a magni-
tude of improvement over the linear shrinkage estimator of L edoit and Wolf
(2004).
3.2. Precision matrix. Often times an estimator of the inverse of the
covariance matrix, or the precision matrix, Σ − 1
n is required. A reasonable
strategy would be to ﬁrst estimate Σ n, and to then simply take the inverse
of the resulting estimator. However, such a strategy will ge nerally not be
optimal.
By arguments analogous to those leading up to ( 3.3), among the class of
rotation-equivariant estimators, the ﬁnite-sample optim al estimator of Σ − 1
n
with respect to the Frobenius norm is given by
P ∗
n ≡ UnA∗
nU ′
n where a∗
i ≡ u′
iΣ− 1
n ui for i = 1, . . . , p.(3.6)
In particular, note that P ∗
n ̸= (S∗
n)− 1 in general.
Studying the asymptotic behavior of the diagonal matrix A∗
n led Ledoit
and P´ ech´ e (
2011) to the following oracle estimator:
P or
n ≡ UnAor
n U ′
n
(3.7)
where aor
i ≡ λ− 1
i (1 − c − 2cλiRe[ ˘mF (λi)]) for i = 1, . . . , p.
In particular, note that P or
n ̸= (Sor
n )− 1 in general.

<!-- page 10 -->

10 O. LEDOIT AND M. WOLF
Remark 3.1. One can see that both oracle estimators Sor
n and P or
n in-
volve the unknown quantities ˘mF (λi), for i = 1, . . . , p. As a result, they are
not bona ﬁde estimators. However, being able to consistently estimate ˘mF (λ),
uniformly in λ, will allow us to construct bona ﬁde estimators ˆSn and ˆPn
that converge to their respective oracle counterparts almo st surely (in the
sense that the Frobenius norm of the diﬀerence converges to ze ro almost
surely).
Section 4 explains how to construct a uniformly consistent estimator
of ˘mF (λ) based on a consistent estimator of H, the limiting spectral distri-
bution of the population eigenvalues. Section 5 discusses how to construct
a consistent estimator of H from the data.
3.3. Further details on the results of Ledoit and P´ ech´ e ( 2011). Ledoit
and P´ ech´ e (2011) (hereafter LP) study functionals of the type
∀z ∈ C+ Θg
N (z) ≡ 1
N
N∑
i=1
1
λi − z
N∑
j=1
|u∗
i vj|2 × g(τj )
(3.8)
= 1
N Tr[(SN − zI )− 1g(ΣN )],
where g is any real-valued univariate function satisfying suitabl e regular-
ity conditions. Comparison with equation ( 2.1) reveals that this family of
functionals generalizes the Stieltjes transform, with the Stieltjes transform
corresponding to the special case g ≡ 1. What is of interest is what happens
for other, nonconstant functions g.
It turns out that it is possible to generalize the Marˇ cenko– Pastur re-
sult ( 2.2) to any function g with ﬁnitely many points of discontinuity. Un-
der assumptions that are usual in the Random Matrix Theory li terature,
LP prove in their Theorem 2 that there exists a nonrandom func tion Θ g
deﬁned over C+ such that Θ g
N (z) converges a.s. to Θ g(z) for all z ∈ C+.
Furthermore, Θ g is given by
∀z ∈ C+ Θg(z) ≡
∫ +∞
−∞
g(τ )
τ [1 − c − czmF (z)] − z dH(τ ).(3.9)
What is remarkable is that, as one moves from the constant fun ction g ≡ 1
to any other function g(τ ), the integration kernel g(τ )
τ [1− c− czmF (z)]− z remains
unchanged. Therefore equation ( 3.9) is a direct generalization of Marˇ cenko
and Pastur’s foundational result.
The power and usefulness of this generalization become appa rent once one
starts plugging speciﬁc, judiciously chosen functions g(τ ) into equation (3.9).
For the purpose of illustration, LP work out three examples o f functions g(τ ).
The ﬁrst example of LP is g(τ ) ≡ 1(−∞ ,τ ), where 1 denotes the indicator
function of a set. It enables them to characterize the asympt otic location of

<!-- page 11 -->

NONLINEAR SHRINKAGE ESTIMATION 11
sample eigenvectors relative to population eigenvectors. Since this result is
not directly relevant to the present paper, we will not elabo rate further, and
refer the interested reader to LP’s Section 1.2.
The second example of LP is g(τ ) ≡ τ . It enables them to characterize
the asymptotic behavior of the quantities dor
i introduced in equation (
3.4).
More formally, for any u ∈ (0, 1), deﬁne
∆∗
n(u) ≡ 1
p
⌊u·p⌋∑
i=1
d∗
i and ∆ or
n (u) ≡ 1
p
⌊u·p⌋∑
i=1
dor
i ,(3.10)
where ⌊·⌋ denotes the integer part. LP’s Theorem 4 proves that ∆ ∗
n(u) −
∆or
n (u) → 0 a.s.
The third example of LP is g(τ ) ≡ 1/τ . It enables them to characterize
the asymptotic behavior of the quantities aor
i introduced in equation (
3.7).
For any u ∈ (0, 1) deﬁne
Ψ∗
n(u) ≡ 1
p
⌊u·p⌋∑
i=1
a∗
i and Ψ or
n (u) ≡ 1
p
⌊u·p⌋∑
i=1
aor
i .(3.11)
LP’s Theorem 5 proves that Ψ ∗
n(u) − Ψor
n (u) → 0 a.s.
4. Estimation of ˘mF (λ). Fix x ∈ [u1 + η, u2 − η], where η > 0 is some
small number. From the previous discussion in Section
2, it follows that the
equation
Im[x + iy − c(x + iy)mLH (x + iy)] = 0
has a unique solution y ∈ (0, +∞ ), called yx. Since u1 < x < u 2, it follows
that yx > 0; for x = u1 or x = u2, we would have yx = 0 instead. The goal is
to consistently estimate yx, uniformly in x ∈ [u1 + η, u2 − η].
Deﬁne for any c.d.f. G and for any d > 0, the real function
gG,d(y, x) ≡ |Im[x + iy − d(x + iy)mLG(x + iy)]|.
With this notation, yx is the unique minimizer in (0 , +∞ ) of gH,c(y, x) then.
In particular, gH,c(yx, x) = 0.
In the remainder of the paper, the symbol ⇒ denotes weak convergence
(or convergence in distribution).
Proposition 4.1. (i) Let { ˆHn} be a sequence of probability measures
with ˆHn ⇒ H. Let {ˆcn} be a sequence of positive real numbers with ˆcn → c.
Let K ⊆ (0, ∞ ) be a compact interval satisfying {yx : x ∈ [u1 +η, u2 − η]} ⊆ K.
For a given x ∈ [u1 + η, u2 − η], let ˆyn,x ≡ miny∈ K g ˆHn,ˆcn
(y, x). It then holds
that ˆyn,x → yx uniformly in x ∈ [u1 + η, u2 − η].
(ii) In case of ˆHn ⇒ H a.s., it holds that ˆyn,x → yx a.s. uniformly in
x ∈ [u1 + η, u2 − η].

<!-- page 12 -->

12 O. LEDOIT AND M. WOLF
It should be pointed out that the assumption {yx : x ∈ [u1 + η, u2 − η]} ⊆ K
is not really restrictive, since one can choose K ≡ [ε, 1/ε], for ε arbitrarily
small.
We also need to solve the “inverse” estimation problem, name ly starting
with λ and recovering the corresponding vλ. Fix λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ], where
˜δ > 0 is some small number. From the previous discussion, it foll ows that
the equation
v − cvmLH (v) = λ
has a unique solution v ∈ C+, called vλ. The goal is to consistently esti-
mate vλ, uniformly in λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ].
Deﬁne for any c.d.f. G and for any d > 0, the real function
hG,d(v, λ) ≡ |v − dvmLG(v) − λ|.
With this notation, vλ is then the unique minimizer in C+ of hH,c(v, λ). In
particular, hH,c(vλ, λ) = 0.
Proposition 4.2. (i) Let { ˆHn} be a sequence of probability measures
with ˆHn ⇒ H. Let {ˆcn} be a sequence of positive real numbers with ˆcn → c.
Let K ⊆ C+ be a compact set satisfying {vλ : λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ]} ⊆ K. For
a given λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ], let ˆvn,λ ≡ minv∈ K h ˆHn,ˆcn
(v, λ). It then holds that
ˆvn,λ → vλ uniformly in λ ∈ [˜z1 + ˜δ, z2 − ˜δ].
(ii) In case of ˆHn ⇒ H a.s., it holds that ˆvn,λ → vλ a.s. uniformly in
λ ∈ [˜z1 + ˜δ, z2 − ˜δ].
Being able to ﬁnd consistent estimators of vλ, uniformly in λ, now allows
us to ﬁnd consistent estimators of ˘ mF (λ), uniformly in λ, based on ( 2.9).
Our estimator of ˘mF (λ) is given by
˘mF ˆHn,ˆcn
(λ) ≡ 1 − ˆcn
ˆcnλ − 1
ˆcn
1
ˆvn,λ
.(4.1)
This, in turn, provides us with a consistent estimator of Sor
n , the oracle
nonlinear shrinkage estimator of Σ n. Deﬁne
ˆSn ≡ Un ˆDnU ′
n
(4.2)
where ˆdi ≡ λi
|1 − ˆcn − ˆcnλi ˘mF ˆHn,ˆcn
(λi)|2 for i = 1, . . . , p.
It also provides us with a consistent estimator of P or
n , the oracle nonlinear
shrinkage estimator of Σ − 1
n . Deﬁne
ˆPn ≡ Un ˆAnU ′
n
(4.3)
where ˆai ≡ λ− 1
i (1 − ˆcn − 2ˆcnλiRe[ ˘mF ˆHn,ˆcn
(λi)]) for i = 1, . . . , p.

<!-- page 13 -->

NONLINEAR SHRINKAGE ESTIMATION 13
In particular, note that ˆPn ̸= ˆS− 1
n in general.
Proposition 4.3.
(i) Let { ˆHn} be a sequence of probability measures with ˆHn ⇒ H. Let {ˆcn}
be a sequence of positive real numbers with ˆcn → c. It then holds that:
(a) ˘mF ˆHn,ˆcn
(λ) → ˘mF (λ) uniformly in λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ];
(b) ∥ ˆSn − Sor
n ∥ → 0;
(c) ∥ ˆPn − P or
n ∥ → 0.
(ii) In case of ˆHn ⇒ H a.s., it holds that:
(a) ˘mF ˆHn,ˆcn
(λ) → ˘mF (λ) uniformly in λ ∈ [˜z1 + ˜δ, ˜z2 − ˜δ] a.s.;
(b) ∥ ˆSn − Sor
n ∥ → 0 a.s.;
(c) ∥ ˆPn − P or
n ∥ → 0 a.s.
5. Estimation of H. As described before, consistent estimation of the
oracle estimators of Ledoit and P´ ech´ e (2011) requires (uniformly) consis-
tent estimation of ˘mF (λ). Since Im[ ˘mF (λ)] = πF ′(λ), one possible approach
could be to take an oﬀ-the-shelf density estimator for F ′, based on the ob-
served sample eigenvalues λi. There exists a large literature on density es-
timation; for example, see Silverman ( 1986). The real part of ˘ mF (λi) could
be estimated in a similar manner.
However, the sample eigenvalues do not satisfy any of the reg ularity con-
ditions usually invoked for the underlying data. It really i s not clear at all
whether an oﬀ-the-shelf density estimator applied to the sam ple eigenvalues
would result in consistent estimation of F ′.
Even if this issue was somehow resolved, using such a generic procedure
would not exploit the speciﬁc features of the problem. Namel y: F is not just
any distribution; it is a distribution of sample eigenvalue s. It is the solution
to the Marˇ cenko–Pastur equation for some H. This is valuable informa-
tion that narrows down considerably the set of possible dist ributions F .
Therefore an estimation procedure speciﬁcally designed to incorporate this
a priori knowledge would be better suited to the problem at ha nd. This is
the approach we select.
In a nutshell: our estimator of F is the c.d.f. that is closest to Fn among
the c.d.f.’s that are a solution to the Marˇ cenko–Pastur equ ation for some ˜H
and for ˜c ≡ ˆcn ≡ p/n. The “underlying” distribution ˜H that produces the
thus obtained estimator of F is, in turn, our estimator of H. If we can show
that this estimator of H is consistent, then the results of the previous section
demonstrate that the implied estimator of ˘ mF (λ) is uniformly consistent.
Section 5.1 derives theoretical properties of this approach, while Sec -
tion 5.2 discusses various issues concerning the practical impleme ntation.

<!-- page 14 -->

14 O. LEDOIT AND M. WOLF
5.1. Consistency results. For a grid of real numbers Q ≡ { . . . , t− 1, t0,
t1, . . .} ⊆ R, with tk− 1 < tk, deﬁne the corresponding grid size γ as
γ ≡ sup
k
(tk − tk− 1).
A grid Q is said to cover a compact interval [ a, b] ⊆ R if there exists at least
one tk ∈ Q with tk ≤ a and at least another tk′ ∈ Q with b ≤ tk′. A sequence
of grids {Qn} is said to eventually cover a compact interval [ a, b] if for
every φ > 0 there exist N ≡ N (φ) such that Qn covers the compact interval
[a + φ, b − φ] for all n ≥ N .
For any probability measure ˜H on the real line and for any ˜c > 0, let F ˜H,˜c
denote the c.d.f. on the real line induced by the correspondi ng solution of
the Marˇ cenko–Pastur equation. More speciﬁcally, for each z ∈ C+, mF ˜H,˜c
(z)
is the unique solution for m ∈ C+ to the equation
m =
∫ +∞
−∞
1
τ [1 − ˜c − ˜czm] − z d ˜H(τ ).
In this notation, we then have F = FH,c.
It follows from Silverstein and Choi ( 1995) again that
∀λ ∈ R − { 0} lim
z∈ C+→ λ
mF ˜H,˜c
(z) ≡ ˘mF ˜H,˜c
(λ)
exists, and that F ˜H,˜c has a continuous derivative F ′
˜H,˜c = π− 1Im[ ˘mF ˜H,˜c
] on
(0, +∞ ). In the case ˜c < 1, F ˜H,˜c has a continuous derivative on all of R with
F ′
˜H,˜c ≡ 0 on ( −∞ , 0].
For a grid Q on the real line and for two c.d.f.’s G1 and G2, deﬁne
∥G1 − G2∥Q ≡ sup
t∈ Q
|G1(t) − G2(t)|.
The following theorem shows that both F and H can be estimated con-
sistently via an idealized algorithm.
Theorem 5.1. Let {Qn} be a sequence of grids on the real line eventu-
ally covering the support of F with corresponding grid sizes {γn} satisfying
γn → 0. Let {ˆcn} be a sequence of positive real numbers with ˆcn → c. Let ˆHn
be deﬁned as
ˆHn ≡ argmin
˜H
∥F ˜H,ˆcn
− Fn∥Qn,(5.1)
where ˜H is a probability measure.
Then we have (i) F ˆHn,ˆcn
⇒ F a.s.; and (ii) ˆHn ⇒ H a.s.
The algorithm used in the theorem is not practical for two rea sons. First,
it is not possible to optimize over all probability measures ˜H. But simi-
larly to El Karoui (
2008), we can show that it is suﬃcient to optimize over

<!-- page 15 -->

NONLINEAR SHRINKAGE ESTIMATION 15
all probability measures that are sums of atoms, the locatio n of which is
restricted to a ﬁxed-size grid, with the grid size vanishing asymptotically.
Corollary 5.1. Let {Qn} be a sequence of grids on the real line even-
tually covering the support of F with corresponding grid sizes {γn} satisfying
γn → 0. Let {ˆcn} be a sequence of positive real numbers with ˆcn → c. Let Pn
denote the set of all probability measures that are sums of at oms belonging to
the grid {Jn/Tn, (Jn + 1)/Tn, . . . , Kn/Tn} with Tn → ∞ , Jn being the largest
integer satisfying Jn/Tn ≤ λ1, and Kn being the smallest integer satisfying
Kn/Tn ≥ λp. Let ˆHn be deﬁned as
ˆHn ≡ argmin
˜H∈P n
∥F ˜H,ˆcn
− Fn∥Qn.(5.2)
Then we have (i) F ˆHn,ˆcn
⇒ F a.s.; and (ii) ˆHn ⇒ H a.s.
But even restricting the optimization over a manageable set of proba-
bility measures is not quite practical yet for a second reaso n. Namely, to
compute F ˜H,ˆcn
exactly for a given ˜H, one would have to (numerically)
solve the Marˇ cenko–Pastur equation for an inﬁnite number o f points. In
practice, we can only aﬀord to solve the equation for a ﬁnite nu mber of
points and then approximate F ˜H,ˆcn
by trapezoidal integration. Fortunately,
this approximation does not negatively aﬀect the consistenc y of our estima-
tors.
Let G be a c.d.f. with continuous density g and compact support [ a, b]. For
a grid Q ≡ { . . . , t− 1, t0, t1, . . .} covering the support of G, the approximation
to G via trapezoidal integration over the grid Q, denoted by ˆGQ, is obtained
as follows. For t ∈ [a, b], let Jlo ≡ max{k : tk ≤ a} and Jhi ≡ min{k : t < tk}.
Then
ˆGQ(t) ≡
Jhi− 1∑
k=Jlo
(tk+1 − tk)[g(tk) + g(tk+1)]
2 .(5.3)
Now turn to the special case G ≡ F ˜H,˜c and Q ≡ Qn. In this case, we denote
the approximation to F ˜H,˜c via trapezoidal integration over the grid Qn by
ˆF ˜H,˜c;Qn
.
Corollary 5.2. Assume the same assumptions as in Corollary 5.1.
Let ˆHn be deﬁned as
ˆHn ≡ argmin
˜H∈P n
∥ ˆF ˜H,ˆcn;Qn
− Fn∥Qn.(5.4)
Let ˘mF ˆHn,ˆcn
(λ), ˆSn, and ˆPn be deﬁned as in (
4.1), ( 4.2) and ( 4.3), respec-
tively. Then:

<!-- page 16 -->

16 O. LEDOIT AND M. WOLF
(i) F ˆHn,ˆcn
⇒ F a.s.
(ii) ˆHn ⇒ H a.s.
(iii) For any ˜δ > 0, ˘mF ˆHn,ˆcn
(λ) → ˘mF (λ) a.s. uniformly in λ ∈ [˜z1 +
˜δ, ˜z2 − ˜δ].
(iv) ∥ ˆSn − Sor
n ∥ → 0 a.s.
(v) ∥ ˆPn − P or
n ∥ → 0 a.s.
5.2. Implementation details.
Decomposition of the c.d.f. of population eigenvalues. As discussed be-
fore, it is not practical to search over the set of all possibl e c.d.f.’s ˜H.
Following El Karoui ( 2008), we project H onto a certain basis of c.d.f.’s
(Mk)k=1,...,K , where K goes to inﬁnity along with n and p. The projection
of H onto this basis is given by the nonnegative weights w1, . . . , wK , where
∀t ∈ R H(t) ≈ ˜H(t) ≡
K∑
k=1
wkMk(t) and
K∑
k=1
wk = 1.(5.5)
Thus, our estimator for F will be a solution to the Marˇ cenko–Pastur equa-
tion for ˜H given by equation (
5.5) for some ( wk)k=1,...,K , and for ˜c ≡ p/n. It
is just a matter of searching over all sets of nonnegative wei ghts summing
up to one.
Choice of basis. We base the c.d.f.’s ( Mk)k=1,...,K on a grid of p equally
spaced points on the interval [ λ1, λp].
xi ≡ λ1 + i − 1
p (λp − λ1) for i = 1, . . . , p.(5.6)
Thus x1 = λ1 and xp = λp. We then form the basis {M1, . . . , Mk} as the
union of three families of c.d.f.’s:
(1) the indicator functions 1[xi,+∞ ) (i = 1, . . . , p);
(2) the c.d.f.’s whose derivatives are linearly increasing on the interval [xi− 1, xi]
and zero everywhere else ( i = 2, . . . , p);
(3) the c.d.f.’s whose derivatives are linearly decreasing on the interval
[xi− 1, xi] and zero everywhere else ( i = 2, . . . , p).
This list yields a basis ( Mk)k=1,...,K of dimension K = 3p − 2. Notice that by
the theoretical results of Section
5.1, it would be suﬃcient to use the ﬁrst
family only. Including the second and third families in addi tion cannot make
the approximation to H any worse.
Trapezoidal integration. For a given ˜H ≡ ∑K
k=1 wkMk, it is computa-
tionally too expensive (in the context of an optimization pr ocedure) to solve
the Marˇ cenko–Pastur equation for mF (z) over all z ∈ C+. It is more eﬃ-

<!-- page 17 -->

NONLINEAR SHRINKAGE ESTIMATION 17
cient to solve the Marˇ cenko–Pastur equation only for ˘mF (xi) ( i = 1, . . . , p),
and to use the trapezoidal approximation formula to deduce f rom it F (xi)
(i = 1, . . . , p). The trapezoidal rule gives
∀i = 1, . . . , p F (xi) =
i− 1∑
j=1
xj+1 − xj− 1
2 F ′(xj) + xi − xi− 1
2 F ′(xi)
=
i− 1∑
j=1
(xj+1 − xj− 1)Im[ ˘mF (xj)]
2π(5.7)
+ (xi − xi− 1)Im[ ˘mF (xi)]
2π ,
with the convention x0 ≡ 0.
Objective function. The objective function measures the distance be-
tween Fn and the F that solves the Marˇ cenko–Pastur equation for ˜H ≡∑K
k=1 wkMk and for ˜c ≡ p/n. Traditionally, Fn is deﬁned as c` adl` ag, that is,
Fn(λ1) = 1/p and Fn(λp) = 1. However, there is a certain degree of arbitrari-
ness in this convention: why is Fn(λp) equal to one but Fn(λ1) not equal
to zero? By symmetry, there is no a priori justiﬁcation for sp ecifying that
the largest eigenvalue is closer to the supremum of the suppo rt of F than
the smallest to its inﬁmum. Therefore, a diﬀerent convention might be more
appropriate in this case, which leads us to the following deﬁ nition:
∀i = 1, . . . , p ˆFn(λi) ≡ i
p − 1
2p .(5.8)
This choice restores a certain element of symmetry to the tre atment of the
smallest vs. the largest eigenvalue. From equation (
5.8), we deduce ˆFn(xi),
for i = 2, . . . , p− 1, by linear interpolation. With a sup-norm error penalty,
this leads to the following objective function:
max
i=1,...,p
|F (xi) − ˆFn(xi)|,(5.9)
where F (xi) is given by equation (
5.7) for i = 1, . . . , p. Using equation ( 5.7),
we can rewrite this objective function as
max
i=1,...,p
⏐
⏐
⏐
⏐
⏐
i− 1∑
j=1
(xj+1 − xj− 1)Im[ ˘mF (xj)]
2π + (xi − xi− 1)Im[ ˘mF (xi)]
2π − ˆFn(xi)
⏐
⏐
⏐
⏐
⏐.
Optimization program. We now have all the ingredients needed to state
the optimization program that will extract the estimator of ˘mF (x1), . . . ,
˘mF (xp) from the observations λ1, . . . , λp. It is the following:
min
m1,...,mp
w1,...,wK
max
i=1,...,p
⏐
⏐
⏐
⏐
⏐
i− 1∑
j=1
(xj+1 − xj− 1)Im[mj]
2π + (xi − xi− 1)Im[mi]
2π − ˆFn(xi)
⏐
⏐
⏐
⏐
⏐

<!-- page 18 -->

18 O. LEDOIT AND M. WOLF
subject to
∀j = 1, . . . , p m j =
K∑
k=1
∫ +∞
−∞
wk
t[1 − (p/n) − (p/n)xjmj] − xj
dMk(t),
K∑
k=1
wk = 1,
(5.10)
∀j = 1, . . . , p m j ∈ C+,
∀k = 1, . . . , K w k ≥ 0.
The key is to introduce the variables mj ≡ ˘mF (xj), for j = 1, . . . , p. The con-
straint in equation ( 5.10) imposes that mj is the solution to the Marˇ cenko–
Pastur equation evaluated as z ∈ C+ → xj when ˜H = ∑K
k=1 wkMk.
Real optimization program. In practice, most optimizers only accept real
variables. Therefore it is necessary to decompose mj into its real and imag-
inary parts: aj ≡ Re[mj] and bj ≡ Im[mj]. Then we can optimize separately
over the two sets of real variables aj and bj for j = 1, . . . , p. The Marˇ cenko–
Pastur constraint in equation (
5.10) splits into two constraints: one for the
real part and the other for the imaginary part. The reformula ted optimiza-
tion program is
min
a1,...,ap
b1,...,bp
w1,...,wK
max
i=1,...,p
⏐
⏐
⏐
⏐
⏐
i− 1∑
j=1
(xj+1 − xj− 1)bj
2π + (xi − xi− 1)bi
2π − ˆFn(xi)
⏐
⏐
⏐
⏐
⏐(5.11)
subject to
∀j = 1, . . . , p
(5.12)
aj =
K∑
k=1
∫ +∞
−∞
Re
{ wk
t[1 − (p/n) − (p/n)xj(aj + ibj)] − xj
}
dMk(t),
∀j = 1, . . . , p
(5.13)
bj =
K∑
k=1
∫ +∞
−∞
Im
{ wk
t[1 − (p/n) − (p/n)xj(aj + ibj)] − xj
}
dMk(t),
K∑
k=1
wk = 1,(5.14)
∀j = 1, . . . , p b j ≥ 0,(5.15)
∀k = 1, . . . , K w k ≥ 0.(5.16)

<!-- page 19 -->

NONLINEAR SHRINKAGE ESTIMATION 19
Remark 5.1. Since the theory of Sections 4 and 5.1 partly assumes
that mj belongs to a compact set in C+ bounded away from the real line,
we might want to add to the real optimization program the cons traints that
− 1/ε ≤ aj ≤ 1/ε and that ε ≤ bj ≤ 1/ε, for some small ε > 0. Our simulations
indicate that for a small value of ε such as ε = 10− 6, this makes no diﬀerence
in practice.
Sequential linear programming. While the optimization program deﬁned
in equations ( 5.11)–(5.16) may appear daunting at ﬁrst sight because of its
non-convexity, it is, in fact, solved quickly and eﬃciently by oﬀ-the-shelf opti-
mization software implementing Sequential Linear Program ming (SLP). The
key is to linearize equations ( 5.12)–(5.13), the two constraints that embody
the Marˇ cenko–Pastur equation, around an approximate solution point. Once
they are linearized, the optimization program ( 5.11)–(5.16) becomes a stan-
dard Linear Programming (LP) problem, which can be solved ve ry quickly.
Then we linearize again equations ( 5.12)–(5.13) around the new point, and
this generates a new LP problem; hence the name: Sequential Linear Pro-
gramming. The software iterates until a satisfactory degre e of convergence
is achieved. All of this is handled automatically by the SLP o ptimizer. The
user only needs to specify the problem ( 5.11)–(5.16), as well as some starting
point, and then launch the SLP optimizer. For our SLP optimiz er, we se-
lected a standard oﬀ-the-shelf commercial software: SNOPT TM Version 7.2–
5; see Gill, Murray and Saunders ( 2002). While SNOPT TM was originally
designed for sequential quadratic programming, it also han dles SLP, since
linear programming can be viewed as a particular case of quad ratic pro-
gramming with no quadratic term.
Starting point. A neutral way to choose the starting point is to place
equal weights on all the c.d.f.’s in our basis: wk ≡ 1/K(k = 1, . . . , K). Then it
is necessary to solve the Marˇ cenko–Pastur equation numerically once before
launching the SLP optimizer, in order to compute the values o f ˘mF (xj)
(j = 1, . . . , p) that correspond to this initial choice of ˜H = ∑K
k=1 Mk/K. The
initial values for aj are taken to be Re[ ˘ mF (xj)], and Im[ ˘mF (xj)] for bj
(j = 1, . . . , p). If the choice of equal weights wk ≡ 1/K for the starting point
does not lead to convergence of the optimization program wit hin a pre-
speciﬁed limit on the maximum number of iterations, we choos e random
weights wk generated i.i.d. ∼ Uniform[0, 1] (rescaled to sum up to one),
repeating this process until convergence ﬁnally occurs. In the vast majority
of cases, the optimization program already converges on the ﬁrst try. For
example, over 1000 Monte Carlo simulations using the design of Section
6.1
with p = 100 and n = 300, the optimization program converged on the ﬁrst
try 994 times and on the second try the remaining 6 times.

<!-- page 20 -->

20 O. LEDOIT AND M. WOLF
Fig. 1. Mean and median CPU times (in seconds) for optimization prog ram as function
of matrix dimension. The design is the one of Section 6.1 with n = 3p. Every point is the
result of 1000 Monte Carlo simulations.
Optimization time. Figure 1 gives some information on how the opti-
mization time increases with the matrix dimension.
The main reason for the rate at which the optimization time in creases
with p is that the number of grid points in ( 5.6) increases linearly in p.
This linear rate is not a requirement for our asymptotic resu lts. Therefore,
if necessary, it is possible to pick a less-than-linear rate of increase in the
number of grid points to speed up the optimization for very la rge matrices.
Estimating the covariance matrix. Once the SLP optimizer has con-
verged, it generates optimal values ( a∗
1, . . . , a∗
p), (b∗
1, . . . , b∗
p) and ( w∗
1, . . . , w∗
K ).
The ﬁrst two sets of variables at the optimum are used to estim ate the oracle
shrinkage factors. From the reconstructed ˘m∗
F (xj) ≡ a∗
j + ib∗
j , we deduce by
linear interpolation ˘m∗
F (λj), for j = 1, . . . , p. Our estimator of the covariance
matrix ˆSn is built by keeping the same eigenvectors as the sample covar iance
matrix, and dividing each sample eigenvalue λj by the following correction
factor:
⏐
⏐
⏐
⏐1 − p
n − p
n λj ˘m∗
F (λj)
⏐
⏐
⏐
⏐
2
.
Corollary
5.2 assures us that the resulting bona ﬁde nonlinear shrinkage
estimator is asymptotically equivalent to the oracle estim ator Sor
n . Also,
we can see that, as the concentration ˆcn = p/n gets closer to zero, that
is, as we get closer to ﬁxed-dimension asymptotics, the magn itude of the
correction becomes smaller. This makes sense because under ﬁxed-dimension

<!-- page 21 -->

NONLINEAR SHRINKAGE ESTIMATION 21
asymptotics the sample covariance matrix is a consistent es timator of the
population covariance matrix.
Estimating the precision matrix. The output of the same optimization
process can also be used to estimate the oracle shrinkage fac tors for the pre-
cision matrix. Our estimator of the precision matrix Σ − 1
n is built by keeping
the same eigenvectors as the sample covariance matrix, and m ultiplying the
inverse λ− 1
j of each sample eigenvalue by the following correction facto r:
1 − p
n − 2 p
n λjRe[ ˘m∗
F (λj)].
Corollary
5.2 assures us that the resulting bona ﬁde nonlinear shrinkage
estimator is asymptotically equivalent to the oracle estim ator P or
n .
Estimating H. We point out that the optimal values ( w∗
1, . . . , w∗
K) gener-
ated from the SLP optimizer yield a consistent estimate of H in the following
fashion:
H ∗ ≡
K∑
k=1
w∗
kMk.
This estimator could be considered an alternative to the est imator in-
troduced by El Karoui ( 2008). The most salient diﬀerence between the two
optimization algorithms is that our objective function tri es to match Fn
on R, whereas his objective function tries to match (a function o f) mFn
on C+. The deeper we go into C+, the more “smoothed-out” is the Stielt-
jes transform, as it is an analytic function; therefore, the more information
is lost. However, the approach of El Karoui ( 2008) cannot get too close
to the real line because mFn starts looking like a sum of Dirac functions
(which are very ill-behaved) as one gets close to the real lin e, since Fn is
a step function. In a sense, the approach of El Karoui ( 2008) is to match
a smoothed-out version of a sum of ill-behaved Diracs. In thi s situation,
knowing “how much to smooth” is rather delicate, and even if i t is done
well, it still loses information. By contrast, we have no inf ormation loss be-
cause we operate directly on the real line, and we have no prob lems with
Diracs because we match Fn instead of its derivative. The price to pay is
that our optimization program is not convex, whereas the one of El Karoui
(2008) is. But extensive simulations reported in the next section show that
oﬀ-the-shelf nonconvex optimization software—as the comme rcial package
SNOPT—can handle this particular type of a nonconvex proble m in a fast,
robust and eﬃcient manner.
It would have been of additional interest to compare our esti mator of H
to the one of El Karoui ( 2008) in some simulations. But when we tried to
implement his estimator according to the implementation de tails provided,
we were not able to match the results presented in his paper. F urthermore,
we were not able to obtain his original software. As a result, we cannot make

<!-- page 22 -->

22 O. LEDOIT AND M. WOLF
any deﬁnite statements concerning the performance of our es timator of H
compared to the one of El Karoui ( 2008).
Remark 5.2 (Cross-validation estimator). The implementation of our
nonlinear shrinkage estimators is not trivial and also requ ires the use of
a third-party SLP optimizer. It is therefore of interest whe ther an alternative
version exists that is easier to implement and exhibits (nea rly) as good ﬁnite-
sample properties.
To this end an anonymous referee suggested to estimate the qu anti-
ties d∗
i of (
3.2) by a leave-one-out cross-validation method. In particula r, let
(λi[k], . . . , λp[k]); (u1[k], . . . , up[k]) denote a system of eigenvalues and eigen-
vectors of the sample covariance matrix computed from all th e observed
data, except for the kth observation. Then d∗
i of (
3.2) can be approximated
by
dcv
i ≡ 1
n
n∑
k=1
(ui[k]′yk)2,
where the p × 1 vector yk denotes the kth row of the matrix Yn ≡ XnΣ1/2
n .
The motivation here is that
(ui[k]′yk)2 = ui[k]′yky′
kui[k],
where yk is independent of ui[k] and E(yky′
k) = Σn (even though yky′
k is of
rank one only).
We are grateful for this suggestion, since the cross-valida tion quanti-
ties dcv
i can be computed without the use of any third-party optimizat ion
software, and the corresponding computer code is very short .
On the other hand, the cross-validation estimator has three disadvantages.
First, when p is large, it takes much longer to compute the cross-validati on
estimator. The reason is that the spectral decomposition of a p × p covariance
matrix has to be computed n times as opposed to only one time. Second,
the cross-validation method only applies to the estimation of the covariance
matrix Σ n itself. It is not clear how to adapt this method to the (direct )
estimation of the precision matrix Σ − 1
n or any other smooth function of Σ n.
Third, the performance of the cross-validation estimator c annot match the
performance of our method; see Section
6.8.
Remark 5.3. Another approach proposed recently is the one of Mestre
and Lagunas ( 2006). They use so-called “G-estimation,” that is, asymptotic
results that assume the sample size n and the matrix dimension p go to
inﬁnity together, to derive minimum variance beam formers i n the context
of the spatial ﬁltering of electronic signals. There are sev eral diﬀerences
between their paper and the present one. First, Mestre and La gunas ( 2006)

<!-- page 23 -->

NONLINEAR SHRINKAGE ESTIMATION 23
are interested in an optimal p × 1 weight vector wopt given by
wopt ≡ argmin
w
w′Σnw subject to w′sd = 1,
where sd is a p × 1 vector containing signal information. Consequently,
Mestre and Lagunas ( 2006) are “only” interested in a certain functional
of Σ n, while we are interested in the full covariance matrix Σ n and also in
the full precision matrix Σ − 1
n . Second, they use the real Stieltjes transform,
which is diﬀerent from the more conventional complex Stieltj es transform
used in random matrix theory and in the present paper. Third, their random
variables are complex whereas ours are real. The cumulative impact of these
diﬀerences is best exempliﬁed by the estimation of the precis ion matrix:
Mestre and Lagunas [(2006), page 76] recommend (1 − p/n)S− 1
n , which is
just a rescaling of the inverse of the sample covariance matr ix, whereas our
Section 3.2 points to a highly nonlinear transformation of the eigenval ues of
the sample covariance matrix.
6. Monte Carlo simulations. In this section, we present the results of var-
ious sets of Monte Carlo simulations designed to illustrate the ﬁnite-sample
properties of the nonlinear shrinkage estimator ˆSn. As detailed in Section 3,
the ﬁnite-sample optimal estimator in the class of rotation -equivariant es-
timators is given by S∗
n as deﬁned in ( 3.3). Thus, the improvement of the
shrinkage estimator ˆSn over the sample covariance matrix will be measured
by how closely this estimator approximates S∗
n relative to the sample covari-
ance matrix. More speciﬁcally, we report the Percentage Rel ative Improve-
ment in Average Loss (PRIAL), which is deﬁned as
PRIAL ≡ PRIAL(ˆΣn) ≡ 100 ×
{
1 − E[∥ˆΣn − S∗
n∥2]
E[∥Sn − S∗n∥2]
}
%,(6.1)
where ˆΣn is an arbitrary estimator of Σ n. By deﬁnition, the PRIAL of Sn
is 0%, while the PRIAL of S∗
n is 100%.
Most of the simulations will be designed around a population covariance
matrix Σ n that has 20% of its eigenvalues equal to 1, 40% equal to 3 and
40% equal to 10. This is a particularly interesting and diﬃcu lt example
introduced and analyzed in detail by Bai and Silverstein (
1998). For concen-
tration values such as c = 1/3 and below, it displays “spectral separation;”
that is, the support of the distribution of sample eigenvalu es is the union
of three disjoint intervals, each one corresponding to a Dir ac of population
eigenvalues. Detecting this pattern and handling it correc tly is a real chal-
lenge for any covariance matrix estimation method.
6.1. Convergence. The ﬁrst set of Monte Carlo simulations shows how
the nonlinear shrinkage estimator ˆSn behaves as the matrix dimension p and
the sample size n go to inﬁnity together. We assume that the concentration
ratio ˆcn = p/n remains constant and equal to 1 /3. For every value of p (and

<!-- page 24 -->

24 O. LEDOIT AND M. WOLF
Fig. 2. Comparison of the nonlinear vs. linear shrinkage estimator s. 20% of population
eigenvalues are equal to 1, 40% are equal to 3 and 40% are equal to 10. Every point is the
result of 1000 Monte Carlo simulations.
hence n), we run 1000 simulations with normally distributed variab les. The
PRIAL is plotted in Figure 2. For the sake of comparison, we also report
the PRIALs of the oracle Sor
n and the optimal linear shrinkage estimator Sn
developed by Ledoit and Wolf ( 2004).
One can see that the performance of the nonlinear shrinkage e stimator ˆSn
converges quickly toward that of the oracle and of S∗
n. Even for relatively
small matrices of dimension p = 30, it realizes 88% of the possible gains
over the sample covariance matrix. The optimal linear shrin kage estima-
tor Sn also performs well relative to the sample covariance matrix , but the
improvement is limited: in general, it does not converge to 1 00% under large-
dimensional asymptotics. This is because there are strong n onlinear eﬀects in
the optimal shrinkage of sample eigenvalues. These eﬀects ar e clearly visible
in Figure 3, which plots a typical simulation result for p = 100.
One can see that the nonlinear shrinkage estimator ˆSn shrinks the eigen-
values of the sample covariance matrix almost as if it “knew” the correct
shape of the distribution of population eigenvalues. In par ticular, the various
curves and gaps of the oracle nonlinear shrinkage formula ar e well picked up
and followed by this estimator. By contrast, the linear shri nkage estimator
can only use the best linear approximation to this highly non linear transfor-
mation. We also plot the 45-degrees line as a visual referenc e to show what
would happen if no shrinkage was applied to the sample eigenv alues, that
is, if we simply used Sn.
6.2. Concentration. The next set of Monte Carlo simulations shows how
the PRIAL of the shrinkage estimators varies as a function of the concen-
tration ratio ˆcn = p/n if we keep the product p × n constant and equal to

<!-- page 25 -->

NONLINEAR SHRINKAGE ESTIMATION 25
Fig. 3. Nonlinearity of the oracle shrinkage formula. 20% of population eigenvalues are
equal to 1, 40% are equal to 3 and 40% are equal to 10. p = 100 and n = 300.
9000. We keep the same population covariance matrix Σ n as in Section 6.1.
For every value of p/n, we run 1000 simulations with normally distributed
variables. The respective PRIALs of Sor
n , ˆSn and Sn are plotted in Figure 4.
One can see that the nonlinear shrinkage estimator performs well across
the board, closely in line with the oracle, and always achiev es at least 90%
of the possible improvement over the sample covariance matr ix. By contrast,
Fig. 4. Eﬀect of varying the concentration ratio ˆcn = p/n. 20% of population eigenvalues
are equal to 1, 40% are equal to 3 and 40% are equal to 10. Every point is the result of
1000 Monte Carlo simulations.

<!-- page 26 -->

26 O. LEDOIT AND M. WOLF
the linear shrinkage estimator achieves relatively little improvement over the
sample covariance matrix when the concentration is low. Thi s is because,
when the sample size is large relative to the matrix dimensio n, there is a lot
of precise information about the optimal nonlinear way to sh rink the sample
eigenvalues that is waiting to be extracted by a suitable non linear procedure.
By contrast, when the sample size is not so large, the informa tion about the
population covariance matrix is relatively fuzzy; therefo re a simple linear
approximation can achieve up to 93% of the potential gains.
6.3. Dispersion. The third set of Monte Carlo simulations shows how
the PRIAL of the shrinkage estimators varies as a function of the dispersion
of population eigenvalues. We take a population covariance matrix Σ n with
20% of its eigenvalues equal to 1, 40% equal to 1 + 2 d/9 and 40% equal
to 1 + d, where the dispersion parameter d varies from 0 to 20. Thus, for
d = 0, Σ n is the identity matrix and, for d = 9, Σ n is the same matrix as in
Section 6.1. The sample size is n = 300 and the matrix dimension is p = 100.
For every value of d, we run 1000 simulations with normally distributed
variables. The respective PRIALs of Sor
n , ˆSn and Sn are plotted in Figure 5.
One can see that the linear shrinkage estimator Sn beats the nonlinear
shrinkage estimator ˆSn for very low dispersion levels. For example, when
d = 0, that is, when the population covariance matrix is equal t o the iden-
tity matrix, Sn realizes 99 .9% of the possible improvement over the sample
covariance matrix, while ˆSn realizes “only” 99 .4% of the possible improve-
ment. This is because, in this case, linear shrinkage is opti mal or (when d
is strictly positive but still small) nearly optimal Hence t here is nothing too
little to be gained by resorting to a nonlinear shrinkage met hod. However, as
Fig. 5. Eﬀect of varying the dispersion of population eigenvalues. 20% of population
eigenvalues are equal to 1, 40% equal to 1 + 2 d/9 and 40% equal to 1 + d, where the
dispersion parameter d varies from 0 to 20. p = 100 and n = 300. Every point is the result
of 1000 Monte Carlo simulations.

<!-- page 27 -->

NONLINEAR SHRINKAGE ESTIMATION 27
Table 1
Eﬀect of nonnormality. 20% of population eigenvalues are equal to 1, 40% are equal to 3
and 40% are equal to 10. 1000 Monte Carlo simulations with p = 100 and n = 300
Average squared PRIALFrobenius loss
df = 3 df = ∞ df = 3 df = ∞
Sample covariance matrix 5.856 5.837 0% 0%
Linear shrinkage estimator 1.883 1.883 67.84% 67.74%
Nonlinear shrinkage estimator 0.128 0.133 97.81% 97.71%
Oracle 0.043 0.041 99.27% 99.30%
dispersion increases, linear shrinkage delivers less and l ess improvement over
the sample covariance matrix, while nonlinear shrinkage re tains a PRIAL
above 96%, and close to that of the oracle.
6.4. Fat tails. We also have some results on the eﬀect of non-normality
on the performance of the shrinkage estimators. We take the s ame popula-
tion covariance matrix as in Section 6.1, that is, Σ n has 20% of its eigen-
values equal to 1, 40% equal to 3 and 40% equal to 10. The sample size
is n = 300, and the matrix dimension is p = 100. We compare two types of
random variates: a Student t distribution with df = 3 degrees of freedom,
and a Student t distribution with df = ∞ degrees of freedom (which is the
Gaussian distribution). For each number of degrees of freed om df, we run
1000 simulations. The respective PRIALs of Sor
n , ˆSn and Sn are summarized
in Table 1.
One can see that departure from normality does not have any no ticeable
eﬀect on performance.
6.5. Precision matrix. The next set of Monte Carlo simulations focuses
on estimating the precision matrix Σ − 1
n . The deﬁnition of the PRIAL, in
this subsection only, is given by
PRIAL ≡ PRIAL( ˆΠn) ≡ 100 ×
{
1 − E[∥ˆΠn − P ∗
n ∥2]
E[∥S− 1n − P ∗n ∥2]
}
%,(6.2)
where ˆΠn is an arbitrary estimator of Σ − 1
n . By deﬁnition, the PRIAL of S− 1
n
is 0% while the PRIAL of P ∗
n is 100%.
We take the same population eigenvalues as in Section
6.1. The concentra-
tion ratio ˆcn = p/n is set to the value 1 /3. For various values of p between
30 and 200, we run 1000 simulations with normally distribute d variables.
The respective PRIALs of P or
n , ˆPn, ˆS− 1
n and S
− 1
n are plotted in Figure 6.
One can see that the nonlinear shrinkage method seems to be ju st as
eﬀective for the purpose of estimating the precision matrix a s it is for the

<!-- page 28 -->

28 O. LEDOIT AND M. WOLF
Fig. 6. Estimating the precision matrix. 20% of population eigenvalues are equal to 1,
40% are equal to 3 and 40% are equal to 10. Every point is the result of 1000 Monte Carlo
simulations.
purpose of estimating the covariance matrix itself. Moreov er, there is a clear
beneﬁt in directly estimating the precision matrix by means of ˆPn as opposed
to the indirect estimation by means of ˆS− 1
n (which on its own signiﬁcantly
outperforms
S
− 1
n ).
6.6. Shape. Next, we study how the nonlinear shrinkage estimator ˆSn
performs for a wide variety of shapes of population spectral densities. This
requires using a family of distributions with bounded suppo rt and which,
for various parameter values, can take on diﬀerent shapes. Th e best-suited
family for this purpose is the beta distribution. The c.d.f. of the beta distri-
bution with parameters ( α, β) is
∀x ∈ [0, 1] F(α,β)(x) = Γ(α + β)
Γ(α)Γ(β)
∫ x
0
tα− 1(1 − t)β− 1 dt.
While the support of the beta distribution is [0 , 1], we shift it to the interval
[1, 10] by applying a linear transformation. Thanks to the ﬂexib ility of the
beta family of densities, selecting diﬀerent parameters ( α, β) enables us to
generate eight diﬀerent shapes for the population spectral d ensity: rectan-
gular (1 , 1), linearly decreasing triangle (1 , 2), linearly increasing triangle
(2, 1), circular (1 .5, 1.5), U-shaped (0 .5, 0.5), bell-shaped (5 , 5), left-skewed
(5, 2) and right-skewed (2 , 5); see Figure 7 for a graphical illustration.
For every one of these eight beta densities, we take the popul ation eigen-
values to be equal to
1 + 9F − 1
(α,β)
( i
p − 1
2p
)
, i = 1, . . . , p.

<!-- page 29 -->

NONLINEAR SHRINKAGE ESTIMATION 29
Fig. 7. Shape of the beta density for various parameter values. The s upport of the beta
density has been shifted to the interval [1,10] by a linear transformation. To enhance clar-
ity, the densities corresponding to the parameters (2,1) and (5,2) have been omitted, since
they are symmetric to (1,2) and (2,5), respectively, about the mid-point of the support.
The concentration ratio ˆcn = p/n is equal to 1 /3. For various values of p
between 30 and 200, we run 1000 simulations with normally dis tributed
variables. The PRIAL of the nonlinear shrinkage estimator ˆSn is plotted in
Figure 8.
As in all the other simulations presented above, the PRIAL of the non-
linear shrinkage estimator always exceeds 88%, and more oft en than not
exceeds 95%. To preserve the clarity of the picture, we do not report the
Fig. 8. Performance of the nonlinear shrinkage with beta densities . The various curves
correspond to diﬀerent shapes of the population spectral den sity. The support of the popu-
lation spectral density is [1,10].

<!-- page 30 -->

30 O. LEDOIT AND M. WOLF
PRIALs of the oracle and of the linear shrinkage estimator; b ut as usual,
the nonlinear shrinkage estimator ranked between them.
6.7. Fixed-dimension asymptotics. Finally, we report a set of Monte Carlo
simulations that departs from the large-dimensional asymp totics assumption
under which the nonlinear shrinkage estimator ˆSn was derived. The goal is to
compare it against the sample covariance matrix Sn in the setting where Sn
is known to have certain optimality properties (at least in t he normal case):
traditional asymptotics, that is, when the number of variab les p remains
ﬁxed while the sample size n goes to inﬁnity. This gives as much advantage
to the sample covariance matrix as it can possibly have. We ﬁx the dimen-
sion p = 100 and let the sample size n vary from n = 125 to n = 10,000. In
practice, very few applied researchers are fortunate enoug h to have as many
as n = 10,000 i.i.d. observations, or a concentration ratio c = p/n as low as
0.01. The respective PRIALs of Sor
n , ˆSn and Sn are plotted in Figure 9.
One crucial diﬀerence with all the previous simulations is th at the tar-
get for the PRIAL is no longer S∗
n, but instead the population covariance
matrix Σ itself, because now Σ can be consistently estimated . Note that,
since the matrix dimension is ﬁxed, Σ n does not change with n; therefore,
we can drop the subscript n. Thus, in this subsection only, the deﬁnition of
the PRIAL is given by
PRIAL ≡ PRIAL(ˆΣn) ≡ 100 ×
{
1 − E[∥ˆΣn − Σ∥2]
E[∥Sn − Σ∥2]
}
%,
where ˆΣn is an arbitrary estimator of Σ. By deﬁnition, the PRIAL of Sn is
0% while the PRIAL of Σ is 100%.
Fig. 9. Fixed-dimension asymptotics. 20% of population eigenvalues are equal to 1, 40%
are equal to 3 and 40% are equal to 10. Variables are normally distributed. Every point is
the result of 1000 Monte Carlo simulations.

<!-- page 31 -->

NONLINEAR SHRINKAGE ESTIMATION 31
In this setting, Ledoit and Wolf ( 2004) acknowledge that the improve-
ment of the linear shrinkage estimator over the sample covar iance matrix
vanishes asymptotically, because the optimal linear shrin kage intensity van-
ishes. Therefore it should be no surprise that the PRIAL of Sn goes to
zero in Figure 9. Perhaps more surprising is the continued ability of the
oracle and the nonlinear shrinkage estimator to improve by a pproximately
60% over the sample covariance matrix, even for a sample size as large as
n = 10,000, and with no sign of abating as n goes to inﬁnity. This is an
encouraging result, as our simulation gave every possible a dvantage to the
sample covariance matrix by placing it in the asymptotic con ditions where
it possesses well-known optimality properties, and where t he earlier linear
shrinkage estimator of Ledoit and Wolf ( 2004) is most disadvantaged.
Intuitively, this is because the oracle shrinkage formula b ecomes more and
more nonlinear as n goes to inﬁnity for ﬁxed p. Bai and Silverstein ( 1998)
show that the sample covariance matrix exhibits “spectral s eparation” when
the concentration ratio p/n is suﬃciently small. It means that the sample
eigenvalues coalesce into clusters, each cluster correspo nding to a Dirac of
population eigenvalues. Within a given cluster, the smalle st sample eigen-
values need to be nudged upward, and the largest ones downwar d, to the
average of the cluster. In other words: full shrinkage withi n clusters, and
no shrinkage between clusters. This is illustrated in Figur e 10, which plots
a typical simulation result for n = 10,000.2
By detecting this intricate pattern automatically, that is , by discovering
where to shrink and where not to shrink, the nonlinear shrink age estima-
tor ˆSn showcases its ability to generate substantial improvement s over the
sample covariance matrix even for very low concentration ra tios.
6.8. Additional Monte Carlo simulations.
6.8.1. Comparisons with other estimators. So far, we have compared the
nonlinear shrinkage estimator ˆSn only to the linear shrinkage estimator Sn
and the oracle estimator Sor
n to keep the resulting ﬁgures concise and legible.
It is of additional interest to compare the nonlinear shrink age estimator
also to some other estimators from the literature. To this en d we consider
the following set of estimators:
• The estimator of Stein (
1975);
• The estimator of Haﬀ ( 1980);
• The estimator recently proposed by Won et al. ( 2009). This estimator is
based on a maximum likelihood approach, assuming normality , with an
explicit constraint on the condition number of the covarian ce matrix. The
2For enhanced ability to distinguish linear shrinkage from t he sample covariance matrix,
we plot the two uninterrupted lines, even though the sample e igenvalues lie in three disjoint
intervals (as can be seen from nonlinear shrinkage).

<!-- page 32 -->

32 O. LEDOIT AND M. WOLF
Fig. 10. Nonlinear shrinkage under ﬁxed-dimension aymptotics. 20% of population
eigenvalues are equal to 1, 40% are equal to 3 and 40% are equal to 10. p = 100 and
n = 10,000. The oracle is not shown because it is virtually identical to the nonlinear shrink-
age estimator.
resulting estimator turns out to be a nonlinear shrinkage es timator as
well: all “small” sample eigenvalues are brought up to a lowe r bound, all
“large” sample eigenvalues are brought down to an upper boun d, and all
“intermediate” sample eigenvalues are left unchanged.
Therefore, the corresponding transformation from sample e igenvalues
to shrunk eigenvalues is step-wise linear: ﬁrst ﬂat, then a 4 5-degree line,
and then ﬂat again. The upper and lower bounds are determined by the
desired constraint on the condition number κ. If such an explicit constraint
is not available from a priori information, a suitable const raint number ˆκ
can be computed in a data-dependent fashion by a K-fold cross-validation
method, which is the method we use. 3
In particular, the cross-validation method selects ˆκ by optimizing over
a ﬁnite grid {κ1, κ2, . . . , κL} that has to be supplied by the user. To
this end we choose L = 10 and the κl log-linearly spaced between 1 and
κ(Sn), for l = 1, . . . , L; here κ(Sn) denotes the condition number of the
sample covariance matrix. More precisely, for l = 1, . . . , L, κl ≡ exp(ωl),
where {ω1, ω2, . . . , ωL} is the equally-spaced grid with ω1 ≡ 0 and ωL ≡
log(κ(Sn)).
• The cross-validation version of the nonlinear shrinkage es timator ˆSn; see
Remark 5.2.
3We are grateful to Joong-Ho Won for supplying us with corresp onding Matlab code.

<!-- page 33 -->

NONLINEAR SHRINKAGE ESTIMATION 33
Fig. 11. Comparison of various estimators. 20% of population eigenvalues are equal to 1,
40% are equal to 3 and 40% are equal to 10. Every point is the result of 1000 Monte Carlo
simulations.
We repeat the simulation exercises of Sections 6.1–6.3, replacing the oracle
estimator and the linear shrinkage estimator with the above set of other
estimators. The respective PRIALs of the various estimator s are plotted in
Figures 11–13.
One can see that the nonlinear shrinkage estimator ˆSn outperforms all
other estimators, with the cross-validation version of ˆSn in second place,
followed by the estimators of Stein ( 1975), Won et al. ( 2009) and Haﬀ
(1980).
Fig. 12. Eﬀect of varying the concentration ratio ˆcn = p/n. 20% of population eigenval-
ues are equal to 1, 40% are equal to 3 and 40% are equal to 10. Every point is the result
of 1000 Monte Carlo simulations.

<!-- page 34 -->

34 O. LEDOIT AND M. WOLF
Fig. 13. Eﬀect of varying the dispersion of population eigenvalues. 20% of population
eigenvalues are equal to 1, 40% equal to 1 + 2 d/9 and 40% equal to 1 + d, where the
dispersion parameter d varies from 0 to 20. p = 100 and n = 300. Every point is the result
of 1000 Monte Carlo simulations.
6.8.2. Comparisons based on a diﬀerent loss function. So far, the PRIAL
has been based on the loss function
LF r(ˆΣn, Σn) ≡ ∥ ˆΣn − Σn∥2.
It is of additional interest to add some comparisons based on a diﬀerent loss
function. To this end we use the scale-invariant loss functi on proposed by
James and Stein ( 1961), namely
LJ S(ˆΣn, Σn) ≡ trace(ˆΣnΣ− 1
n ) − log det(ˆΣnΣ− 1
n ) − p.(6.3)
We repeat the simulation exercises of Sections 6.1–6.3, replacing LF r
with LJ S. The respective PRIALs of Sor
n , ˆSn, and Sn are plotted in Fig-
ures 14–16.
One can see that the results do not change much qualitatively . If anything,
the comparisons are now even more favorable to the nonlinear shrinkage
estimator, in particular when comparing Figure 5 to Figure 16.
7. Conclusion. Estimating a large-dimensional covariance matrix is a very
important and challenging problem. In the absence of additi onal information
concerning the structure of the true covariance matrix, a su ccessful approach
consists of appropriately shrinking the sample eigenvalue s, while retaining
the sample eigenvectors. In particular, such shrinkage est imators enjoy the
desirable property of being rotation-equivariant.
In this paper, we have extended the linear approach of Ledoit and Wolf
(2004) by applying a nonlinear transformation to the sample eigen values.

<!-- page 35 -->

NONLINEAR SHRINKAGE ESTIMATION 35
Fig. 14. Comparison of the nonlinear vs. linear shrinkage estimator s. 20% of population
eigenvalues are equal to 1, 40% are equal to 3 and 40% are equal to 10. The PRIALs
are based on the James–Stein ( 1961) loss function ( 6.3). Every point is the result of 1000
Monte Carlo simulations.
The speciﬁc transformation suggested is motivated by the or acle estima-
tor of Ledoit and P´ ech´ e (
2011), which in turn was derived by studying
the asymptotic behavior of the ﬁnite-sample optimal rotati on-equivariant
estimator (i.e., the estimator with the rotation-equivari ant property that
is closest to the true covariance matrix when distance is mea sured by the
Frobenius norm).
Fig. 15. Eﬀect of varying the concentration ratio ˆcn = p/n. 20% of population eigenval-
ues are equal to 1, 40% are equal to 3 and 40% are equal to 10. The PRIALs are based on
the James–Stein ( 1961) loss function ( 6.3). Every point is the result of 1000 Monte Carlo
simulations.

<!-- page 36 -->

36 O. LEDOIT AND M. WOLF
Fig. 16. Eﬀect of varying the dispersion of population eigenvalues. 20% of population
eigenvalues are equal to 1, 40% equal to 1 + 2 d/9 and 40% equal to 1 + d, where the
dispersion parameter d varies from 0 to 20. p = 100 and n = 300. The PRIALs are based
on the James and Stein ( 1961) loss function ( 6.3). Every point is the result of 1000 Monte
Carlo simulations.
The oracle estimator involves the Stieltjes transform of th e limiting spec-
tral distribution of the sample eigenvalues, evaluated at v arious points on
the real line. By ﬁnding a way to consistently estimate these quantities,
in a uniform sense, we have been able to construct a bona ﬁde nonlinear
shrinkage estimator that is asymptotically equivalent to t he oracle.
Extensive Monte Carlo studies have demonstrated the improv ed ﬁnite-
sample properties of our nonlinear shrinkage estimator com pared to the
sample covariance matrix and the linear shrinkage estimato r of Ledoit and
Wolf (2004), as well as its fast convergence to the performance of the or acle.
In particular, when the sample size is very large compared to the dimension,
or the population eigenvalues are very dispersed, the nonli near shrinkage
estimator still yields a signiﬁcant improvement over the sa mple covariance
matrix, while the linear shrinkage estimator no longer does .
Many statistical applications require an estimator of the i nverse of the
covariance matrix, which is called the precision matrix. We have modiﬁed
our nonlinear shrinkage approach to this alternative probl em, thereby con-
structing a direct estimator of the precision matrix. Monte Carlo studies
have conﬁrmed that this estimator yields a sizable improvem ent over the
indirect method of simply inverting the nonlinear shrinkag e estimator of the
covariance matrix itself.
The scope of this paper is limited to the case where the matrix dimension
is smaller than the sample size. The other case, where the mat rix dimension
exceeds the sample size, requires certain modiﬁcations in t he mathematical
treatment, and is left for future research.

<!-- page 37 -->

NONLINEAR SHRINKAGE ESTIMATION 37
Acknowledgments. We would like to thank two anonymous referees for
valuable comments, which have resulted in an improved expos ition of this
paper.
SUPPLEMENTARY MATERIAL
Mathematical proofs (DOI: 10.1214/12-AOS989SUPP; .pdf). This sup-
plement contains detailed proofs of all mathematical resul ts.
REFERENCES
Bai, Z. D. and Sil verstein, J. W. (1998). No eigenvalues outside the support of the
limiting spectral distribution of large-dimensional samp le covariance matrices. Ann.
Probab. 26 316–345. MR1617051
Bickel, P. J. and Levina, E. (2008). Regularized estimation of large covariance matric es.
Ann. Statist. 36 199–227. MR2387969
Cai, T. and Zhou, H. (2012). Minimax estimation of large covariance matrices un der ℓ1
norm. Statist. Sinica . To appear.
El Karoui, N. (2008). Spectrum estimation for large dimensional covaria nce matrices
using random matrix theory. Ann. Statist. 36 2757–2790. MR2485012
F an, J., F an, Y. and L v, J. (2008). High dimensional covariance matrix estimation usi ng
a factor model. J. Econometrics 147 186–197. MR2472991
Gill, P. E. , Murray, W. and Saunders, M. A. (2002). SNOPT: An SQP algorithm
for large-scale constrained optimization. SIAM J. Optim. 12 979–1006 (electronic).
MR1922505
Haff, L. R. (1980). Empirical Bayes estimation of the multivariate nor mal covariance
matrix. Ann. Statist. 8 586–597. MR0568722
James, W. and Stein, C. (1961). Estimation with quadratic loss. In Proc. 4th Berkeley
Sympos. Math. Statist. and Prob., Vol. I 361–379. Univ. California Press, Berkeley,
Calif. MR0133191
Khare, K. and Rajaratnam, B. (2011). Wishart distributions for decomposable covari-
ance graph models. Ann. Statist. 39 514–555. MR2797855
Ledoit, O. and P´ech´e, S. (2011). Eigenvectors of some large sample covariance matri x
ensembles. Probab. Theory Related Fields 151 233–264.
MR2834718
Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimension al
covariance matrices. J. Multivariate Anal. 88 365–411. MR2026339
Ledoit, O. and Wolf, M. (2012). Supplement to “Nonlinear shrinkage estimation of
large-dimensional covariance matrices.” DOI: 10.1214/12-AOS989SUPP.
Marˇcenko, V. A. and P astur, L. A. (1967). Distribution of eigenvalues for some sets
of random matrices. Sbornik: Mathematics 1 457–483.
Mestre, X. and Lagunas, M. A. (2006). Finite sample size eﬀect on minimum variance
beamformers: Optimum diagonal loading factor for large arr ays. IEEE Trans. Signal
Process. 54 69–82.
Perlman, M. D. (2007). STAT 542: Multivariate Statistical Analysis . Univ. Washington
(On-Line Class Notes), Seattle, Washington.
Rajaratnam, B. , Massam, H. and Carvalho, C. M. (2008). Flexible covariance esti-
mation in graphical Gaussian models. Ann. Statist. 36 2818–2849. MR2485014
Ravikumar, P., W awinwright, M., Raskutti, G. and Yu, B. (2008). High-dimensional
covariance estimation by minimizing ℓ1-penalized log-determinant divergence Technical
Report 797, Dept. Statistics, Univ. California, Berkeley.

<!-- page 38 -->

38 O. LEDOIT AND M. WOLF
Rohde, A. and Tsybakov, A. B. (2011). Estimation of high-dimensional low-rank ma-
trices. Ann. Statist. 39 887–930. MR2816342
Sil verman, B. W. (1986). Density Estimation for Statistics and Data Analysis . Chapman
& Hall, London. MR0848134
Sil verstein, J. W. (1995). Strong convergence of the empirical distribution o f eigenvalues
of large-dimensional random matrices. J. Multivariate Anal. 55 331–339. MR1370408
Sil verstein, J. W. and Choi, S.-I. (1995). Analysis of the limiting spectral distribution
of large-dimensional random matrices. J. Multivariate Anal. 54 295–309. MR1345541
Stein, C. (1956). Inadmissibility of the usual estimator for the mean of a multivariate
normal distribution. In Proceedings of the Third Berkeley Symposium on Mathematica l
Statistics and Probability, 1954–1955, Vol. I 197–206. Univ. California Press, Berkeley.
MR0084922
Stein, C. (1975). Estimation of a covariance matrix. Rietz lecture, 3 9th Annual Meeting
IMS. Atlanta, Georgia.
Won, J. H. , Lim, J. , Kim, S. J. and Rajaratnam, B. (2009). Maximum likelihood
covariance estimation with a condition number constraint. Technical Report 2009-10,
Dept. Statistics, Stanford Univ.
Department of Economics
University of Zurich
CH-8032 Zurich
Switzerland
E-mail:
olivier.ledoit@econ.uzh.ch
michael.wolf@econ.uzh.ch
