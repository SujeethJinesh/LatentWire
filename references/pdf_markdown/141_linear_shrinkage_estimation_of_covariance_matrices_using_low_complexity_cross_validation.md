# references/141_linear_shrinkage_estimation_of_covariance_matrices_using_low_complexity_cross_validation.pdf

<!-- page 1 -->

arXiv:1810.08360v1  [cs.IT]  19 Oct 2018
1
Linear Shrinkage Estimation of Covariance Matrices
Using Low-Complexity Cross-V alidation
Jun Tong, Rui Hu, Jiangtao Xi, Zhitao Xiao, Qinghua Guo and Y a nguang Y u
Abstract—Shrinkage can effectively improve the condition
number and accuracy of covariance matrix estimation, espec ially
for low-sample-support applications with the number of tra ining
samples smaller than the dimensionality. This paper invest igates
parameter choice for linear shrinkage estimators. We propo se
data-driven, leave-one-out cross-validation (LOOCV) met hods
for automatically choosing the shrinkage coefﬁcients, aim ing to
minimize the Frobenius norm of the estimation error . A quadratic
loss is used as the prediction error for LOOCV . The resulting
solutions can be found analytically or by solving optimizat ion
problems of small sizes and thus have low complexities. Our
proposed methods are compared with various existing techni ques.
We show that the LOOCV method achieves near-oracle per-
formance for shrinkage designs using sample covariance mat rix
(SCM) and several typical shrinkage targets. Furthermore, the
LOOCV method provides low-complexity solutions for estima tors
that use general shrinkage targets, multiple targets, and/ or
ordinary least squares (OLS)-based covariance matrix esti mation.
We also show applications of our proposed techniques to seve ral
different problems in array signal processing.
Index Terms —Covariance matrix, cross-validation, linear
shrinkage, ordinary least squares, sample covariance matr ix.
I. I NTRODUCTION
In statistical signal processing, one critical problem is t o es-
timate the covariance matrix, which has extensive applicat ions
in correlation analysis, portfolio optimization, and vari ous
signal processing tasks in radar and communication systems
[1]-[5]. One key challenge is that when the dimensionality i s
large but the sample support is relatively low, the estimate d
covariance matrix R, which may be obtained using a general
method such as sample covariance matrix (SCM) or ordinary
least squares (OLS), becomes ill-conditioned or even singu lar,
and suffers from signiﬁcant errors relative to the true cova ri-
ance matrix Σ. Consequently, signal processing tasks that rely
on covariance matrix estimation may perform poorly or fail
to apply. Regularization techniques have attracted tremen dous
attention recently for covariance matrix estimation. By im pos-
ing structural assumptions of the true covariance matrix Σ,
techniques such as banding [6], thresholding [7], and shrin kage
[8]-[18] have demonstrated great potential for improving t he
performance of covariance matrix estimation. See [19]-[21 ] for
recent surveys.
J. Tong, R. Hu, J. Xi, Q. Guo and Y . Y u are with the School of Elec trical,
Computer and Telecommunications Engineering, University of Wollongong,
Wollongong, NSW 2522, Australia. Email: rh546@uowmail.ed u.au, {jtong,
jiangtao, qguo, yanguang }@uow.edu.au.
Z. Xiao is with School of Electronic and Information Enginee ring, Tianjin
Polytechnic University, China.
This paper is concerned with the linear shrinkage estimatio n
of covariance matrices. Given an estimate R of the covariance
matrix, a linear shrinkage estimate is constructed as
ˆΣρ,τ = ρR + τ T0, (1)
where T0 is the shrinkage target and ρ and τ are nonnegative
shrinkage coefﬁcients. In general, the shrinkage target T0
is better-conditioned, more parsimonious or more structur ed,
with lower variance but higher bias compared to the original
estimate R [11]. The coefﬁcients ρ and τ are chosen to
provide a good tradeoff between bias and variance, such that
an estimate outperforming both R and T0 is achieved and a
better approximation to the true covariance matrix Σ can be
obtained. Compared to other regularized estimators such as
banding and thresholding, linear shrinkage estimators can be
easily designed to guarantee positive-deﬁniteness. Such s hrink-
age designs have been employed in various applications whic h
utilize covariance matrices and have demonstrated signiﬁc ant
performance improvements. The linear shrinkage approach
has also been generalized to nonlinear shrinkage estimatio n
of covariance matrices [22], [23], and is closely related to
several unitarily invariant covariance matrix estimators that
shrink the eigenvalues of the SCM, such as those imposing
condition number constraints on the estimate [24], [25]. Th ere
are also a body of studies on shrinkage estimation of precisi on
matrix (the inverse of covariance matrix) [26]-[30] and on
application-oriented design of shrinkage estimators. See [31]-
[36] for example applications in array signal processing.
Shrinkage has a Bayes interpretation [2], [9]. The true
covariance matrix Σ can be assumed to be within the neighbor-
hoods of the shrinkage target T0. There can be various differ-
ent approaches for constructing R and T0. For example, when
a generative model about the observation exists, one may ﬁrs t
estimate the model parameters and then construct R [20]. A
typical example of this is linear models seen in communicati on
systems. Furthermore, different types of shrinkage target s, not
necessarily limited to identity or diagonal targets, can be used
to better utilize prior knowledge. For example, knowledge-
aided space-time signal processing (KA-STAP) may set T0
using knowledge about the environment [3] or past covarianc e
matrix estimates [37]. Even multiple shrinkage targets can be
applied when distinct guesses about the true covariance mat rix
are available [17].
The choice of shrinkage coefﬁcients signiﬁcantly inﬂuence s
the performance of linear shrinkage estimators. V arious criteria
and methods have been studied. Under the mean squared error
(MSE) criterion, Ledoit and Wolf (LW) [2] derived closed-

<!-- page 2 -->

2
form solutions based on asymptotic estimates of the statist ics
needed for ﬁnding the optimal shrinkage coefﬁcients, where
R and T0 are assumed as the SCM and identity matrix,
respectively. Later the LW solution was extended for more
general shrinkage targets [3], [17]. Chen et al [4] assumed
Gaussian distribution and proposed an oracle approximatin g
shrinkage (OAS) estimator, which achieves near-optimal pa -
rameter choice for Gaussian data even with very low sample
supports. The shrinkage coefﬁcients determination can als o be
cast as a model selection problem and thus generic model
selection techniques such as cross-validation (CV) [38]-[ 40]
can be applied. In general, CV splits the training samples fo r
multiple times into disjoint subsets and then ﬁts and assess es
the models under different splits based on a properly chosen
prediction loss. This has been explored, e.g., in [10], [13] ,
where the Gaussian likelihood is used as the prediction loss .
All these data-driven techniques achieve near-optimal pa-
rameter choice when the underlying assumptions hold. How-
ever, there are also limitations to their applications: alm ost all
existing analytical solutions to shrinkage coefﬁcients [2 ]-[4],
[17] were derived under the assumption of SCM and certain
special forms of shrinkage targets. They need to be re-desig ned
when applied to other cases, which is generally nontrivial. The
asymptotic analysis-based methods [2], [3] may not perform
well when the sample support is very low. Although the
existing CV approaches [10], [13] have broader application s,
they assume Gaussian distribution and employ grid search
to determine the shrinkage coefﬁcients. The likelihood cos t
of [10], [13] must be computed for multiple data splits and
multiple candidates of shrinkage coefﬁcients, which can be
time-consuming.
In this paper, we further investigate data-driven techniqu es
that automatically tune the linear shrinkage coefﬁcients u s-
ing leave-one-out cross-validation (LOOCV). We choose a
simple quadratic loss as the prediction loss for LOOCV , and
derive analytical and computationally efﬁcient solutions . The
solutions do not need to specify the distribution of the data .
Furthermore, the LOOCV treatment is applicable to differen t
covariance matrix estimators including the SCM- and ordina ry
least squares (OLS)-based schemes. It can be used together
with general shrinkage targets and can also be easily extend ed
to incorporate multiple shrinkage targets. The numerical e x-
amples show that the proposed method can achieve oracle-
approximating performance for covariance matrix estimati on
and can improve the performance of several array signal
processing schemes.
The remainder of the paper is organized as follows. In
Section 2, we present computationally efﬁcient LOOCV meth-
ods for choosing the linear shrinkage coefﬁcients for both
SCM- and OLS-based covariance matrix estimators and also
compare the proposed LOOCV methods with several existing
methods which have attracted considerable attentions rece ntly.
In Section 3, we extend our results for multi-target shrinka ge.
Section 4 reports numerical examples, and ﬁnally Section 5
gives conclusions.
II. LOOCV C HOICE OF LINEAR SHRINKAGE
COEFFICIENTS
This paper deals with the estimation of covariance matrices
of zero-mean signals whose fourth-order moments exist. We
study the LOOCV choice of the shrinkage coefﬁcients for the
linear shrinkage covariance matrix estimator (1), i.e., ˆΣρ,τ =
ρR + τ T0. The following assumptions are made:
1) The true covariance matrix Σ, the estimated covariance
matrix R, and the shrinkage target T0 are all Hermitian
and positive-semideﬁnite (PSD).
2) T independent, identically distributed (i.i.d.) samples
{yt} of the signal are available.
3) The shrinkage coefﬁcients are nonnegative, i.e.,
ρ ≥ 0, τ ≥ 0. (2)
Assumption 3 follows the treatments in [2]-[4] and is sufﬁci ent
but not necessary to guarantee that the shrinkage estimate ˆΣρ,τ
is PSD when Assumption 1 holds 1. Two classes of shrinkage
targets will be considered in this paper. One is constructed
independent of the training samples {yt} for generating R,
similarly to the knowledge-aided targets as considered in [ 3].
The other is constructed from {yt}, but is highly structured
with signiﬁcantly fewer free parameters as compared to R.
Examples of the second class include those constructed usin g
only the diagonal entries of R [4], [20] and the Toeplitz
approximations of R [17].
A. Oracle Choice
Different criteria may be used for evaluating the covarianc e
matrix estimators. In this paper, we use the squared Frobeni us
norm of the estimation error as the performance measure.
Given Σ, R and T0, the oracle shrinkage coefﬁcients min-
imize
JO(ρ, τ) = ||ˆΣρ,τ − Σ||2
F = ||ρR + τ T0 − Σ||2
F , (3)
where || · ||F denotes the Frobenius norm. The cost function
in (3) can then be rewritten as a quadratic function of the
shrinkage coefﬁcients:
JO(ρ, τ) =
[ ρ
τ
]T
AO
[ ρ
τ
]
− 2
[ ρ
τ
]T
bO + tr(Σ2), (4)
AO =
[ tr(R2) tr( RT0)
tr(RT0) tr( T2
0)
]
, (5)
bO =
[
tr(RΣ)
tr(T0Σ)
]
, (6)
where tr(·) denotes the trace of a matrix. As AO is positive-
deﬁnite, we can ﬁnd the minimizer of JO(ρ, τ) by solving
the above bivariate convex optimization problem. We can als o
apply the Karush-Kuhn-Tucker (KKT) conditions to ﬁnd the
solution analytically. From (4), letting JO(ρ,τ )
∂ρ = JO(ρ,τ )
∂τ = 0
leads to
tr(R2)
tr(RΣ) ρ + tr(RT0)
tr(RΣ) τ = 1, (7)
1Imposing Assumption 3 may introduce performance loss. Alte rnatively,
one may remove the constraint ρ ≥ 0, τ ≥ 0 and impose a constraint that
ˆΣρ,τ is PSD, similar to a treatment in [5].

<!-- page 3 -->

3
tr(RT0)
tr(T0Σ) ρ + tr(T2
0)
tr(T0Σ) τ = 1. (8)
The oracle shrinkage coefﬁcients can be obtained by solving
(7) and (8): [
ρ⋆
O
τ ⋆
O
]
= A− 1
O bO. (9)
Note that (9) may produce negative shrinkage coefﬁcients,
which may not lead to a positive-deﬁnite estimate of the
covariance matrix. In this case, we clip the negative coefﬁc ient
to zero and then ﬁnd the other coefﬁcient using (7) or (8)
to guarantee the positive deﬁniteness, for τ = 0 or ρ = 0 ,
respectively. This treatment is similar to [2]-[5] and prov ides
a suboptimal yet simple solution. The oracle estimator requ ires
knowledge of Σ, which is unavailable in real applications, but
the result serves as an upper bound of the performance given
the linear shrinkage structure.
B. LOOCV Choice for General Cases
Let ˆΣ denote a positive-deﬁnite, Hermitian matrix. It can
be easily veriﬁed that the following cost
JS( ˆΣ) ≜ E[||ˆΣ − yy†||2
F ] (10)
is minimized when ˆΣ = Σ, where the expectation is taken
over y. In this paper, we apply LOOCV [38] to produce an
estimate of JS( ˆΣ) as the proxy for measuring the accuracy of
ˆΣ, based on which the shrinkage coefﬁcients can be selected.
With the LOOCV method, the length- T training data Y =
[y1, y2, · · ·, yT ] is repeatedly split into two sets with respect
to time. For the t-th split, where 1 ≤ t ≤ T , T − 1 samples
in Yt (with the t-th column yt omitted from Y) are used for
producing a covariance matrix estimate Rt and the remaining
sample yt is spared for parameter validation. In total, T splits
of the training data Y are used and all the training samples are
used for validation once. Assuming shrinkage estimation wi th
given shrinkage coefﬁcients (ρ, τ), we construct from each Yt
a shrinkage covariance matrix estimator as
ˆΣt,ρ,τ = ρRt + τ T0. (11)
We propose to use the following LOOCV cost function
JCV(ρ, τ) = 1
T
T∑
t=1
||ˆΣt,ρ,τ − yty†
t ||2
F (12)
= 1
T
T∑
t=1
||ρRt + τ T0 − yty†
t ||2
F (13)
to approximate the cost in (10) when ˆΣ is chosen as ˆΣt,ρ,τ .
For notational simplicity, deﬁne
St ≜ yty†
t . (14)
After some manipulations, the above cost function can be
written similarly to (4) as
JCV(ρ, τ) =
[ ρ
τ
]T
ACV
[ ρ
τ
]
− 2
[ ρ
τ
]T
bCV + 1
T
T∑
t=1
tr(S2
t ),
(15)
where
ACV =




1
T
T∑
t=1
tr(R2
t ) 1
T
T∑
t=1
tr(RtT0)
1
T
T∑
t=1
tr(RtT0) tr( T2
0)



 , (16)
bCV =




1
T
T∑
t=1
tr(RtSt)
1
T
T∑
t=1
tr(T0St)



 . (17)
The shrinkage coefﬁcients can then be found by solving the
above bivariate, constant-coefﬁcient quadratic program. Ana-
lytical solutions can be obtained under different conditio ns, as
shown below.
1) Unconstrained shrinkage: For unconstrained (ρ, τ), set-
ting the partial derivatives ∂J CV(ρ,τ )
∂ρ = ∂J CV(ρ,τ )
∂τ = 0 yields
T∑
t=1
tr(R2
t )
T∑
t=1
tr(RtSt)
ρ +
T∑
t=1
tr(RtT0)
T∑
t=1
tr(RtSt)
τ = 1, (18)
T∑
t=1
tr(RtT0)
T∑
t=1
tr(T0St)
ρ + T tr(T2
0)
T∑
t=1
tr(T0St)
τ = 1. (19)
Solving (18) and (19) produces the unconstrained solution
[ ρ⋆
CV
τ ⋆
CV
]
= A− 1
CVbCV. (20)
We choose (20) as the optimal shrinkage coefﬁcients if both
ρ⋆
CV and τ ⋆
CV are nonnegative. Otherwise, we consider the
optimal choices on the boundary of ρ ≥ 0, τ ≥ 0 speciﬁed by
(18) or (19) for τ = 0 or ρ = 0 as
ρ⋆
CV =
T∑
t=1
tr(RtSt)
T∑
t=1
tr(R2
t )
, τ ⋆
CV = 0, (21)
or
ρ⋆
CV = 0, τ ⋆
CV =
T∑
t=1
tr(T0St)
T tr(T2
0) . (22)
2) Constrained shrinkage: For the more parsimonious de-
sign using convex linear combination, the following constr aint
is imposed:
ρ = 1 − τ. (23)
By plugging (23) into the cost function (12) and taking the
minimizer, we can also easily ﬁnd the optimal shrinkage
coefﬁcients using
ρ⋆
CV =
T∑
t=1
(
tr(T2
0) − tr(RtT0) − tr(T0St) + tr(StRt)
)
T∑
t=1
(tr(R2
t ) − 2tr(RtT0) + tr(T2
0))
.
(24)

<!-- page 4 -->

4
In case a negative shrinkage coefﬁcient is produced, we set
it to zero and let the other be one according to (23). Note
that although the closed-form solution involves multiple m atrix
operations, the quantities involved need to be computed onl y
once. Furthermore, the computational complexity may be
greatly reduced given a speciﬁc method of covariance matrix
estimation. In the following two subsections, we will show
the simpliﬁed solutions for SCM- and OLS-based covariance
matrix estimation.
C. LOOCV Choice for SCM-Based Estimation
We consider in this subsection that R is the SCM estimate
of Σ. In this case,
R = 1
T
T∑
t=1
yty†
t = 1
T
T∑
t=1
Rt = 1
T
T∑
t=1
St, (25)
which is a sufﬁcient statistic for Gaussian-distributed da ta
when the mean vector is the zero vector. For the t-th split,
the SCM constructed from all the samples except the t-th is
Rt = 1
T − 1
∑
j̸=t
yjy†
j = T
T − 1 R − 1
T − 1 St. (26)
We can then verify the following expressions for quickly
computing the relevant quantities in (16) and (17):
1
T
T∑
t=1
tr(R2
t ) = T (T − 2)
(T − 1)2 tr(R2) − 1
T (T − 1)2
T∑
t=1
||yt||4
F ,
(27)
1
T
T∑
t=1
tr(RtSt) = T
T − 1 tr(R2) − 1
T (T − 1)
T∑
t=1
||yt||4
F ,
(28)
1
T
T∑
t=1
tr(RtT0) = tr( RT0), (29)
1
T
T∑
t=1
tr(StT0) = tr ( RT0) . (30)
Plugging these into (16) and (17) and after some manipula-
tions, we can rewrite the
LOOCV cost function (15) as
JCV(ρ, τ) = ρT (ρT − 2ρ − 2T + 2)
(T − 1)2 tr(R2)
+ 2τ (ρ − 1)tr(RT0) + τ 2tr(T2
0)
+ 1
T
( ρ
T − 1 + 1
) 2 T∑
t=1
∥yt∥4
F . (31)
The optimal shrinkage coefﬁcients can then be obtained ana-
lytically from the SCM R, the shrinkage target T0, and the
training samples {yt}, as discussed below.
1) Unconstrained shrinkage: It can be veriﬁed from (19)
and (30) that the optimal shrinkage coefﬁcients (ignoring t he
nonnegative constraint ρ ≥ 0, τ ≥ 0) satisfy
τ = (1 − ρ) tr(RT0)
tr(T2
0) . (32)
The closed-form solution to ρ is given by
ρ⋆
CV, SCM =
T tr(R2)
T − 1 − (tr(R T0))2
tr(T2
0) −
T∑
t=1
∥yt∥4
F
T (T − 1)
(T 2− 2T )tr(R2)
(T − 1)2 − (tr(R T0))2
tr(T2
0) +
T∑
t=1
∥yt∥4
F
T (T − 1)2
. (33)
In case ρ⋆
CV, SCM > 1 or ρ⋆
CV, SCM < 0, we apply (21) or (22),
respectively, to determine the solution, using the express ions
in (27)-(30).
Note that for the typical shrinkage target T0 = tr(R)
N I, (32)
results in τ ⋆
CV, SCM = 1 − ρ⋆
CV, SCM. This provides another
justiﬁcation for the convex linear combination design with an
identity target, which has been widely adopted in the litera ture,
e.g., [4]. This also shows that for such a special target
the unconstrained solution is equivalent to the constraine d
solution, which does not hold for more general shrinkage
targets.
2) Constrained shrinkage: For the widely considered con-
vex linear combination with constraint ρ + τ = 1, the optimal
ρ (ignoring the nonnegative constraint) is computed as
ρ⋆
CV, SCM =
T tr(R2)
T − 1 − 2tr(RT0) + tr(T2
0) −
T∑
t=1
∥yt∥4
F
T (T − 1)
(T 2− 2T )tr(R2)
(T − 1)2 − 2tr(RT0) + tr(T2
0) +
T∑
t=1
∥yt∥4
F
T (T − 1)2
.
(34)
Similarly, in case a negative shrinkage coefﬁcient is obtai ned,
we set it to zero and let the other be one.
The above results show that the optimal shrinkage coefﬁ-
cients for the covariance matrix estimate (1) can be compute d
directly from the samples and shrinkage target, without the
need of specifying any user parameters. The constrained
shrinkage design may lead to certain performance loss as
compared to the unconstrained one.
D. LOOCV Choice for OLS-Based Covariance Estimation
One advantage of the LOOCV method is that it can be
applied to different covariance matrix estimators. In this
subsection, we discuss the LOOCV method for OLS-based
covariance matrix estimation. Note that most existing anal yt-
ical solutions for choosing the shrinkage coefﬁcients assu me
SCM and speciﬁc shrinkage targets and need to be re-derived
for general cases. Also, in contrast to general application s of
LOOCV which require a grid search of the parameters and
thus a high computational complexity, we have shown that for
SCM, fast analytical solutions can be obtained for choosing
the shrinkage coefﬁcients. This will also be the case for the
OLS-based covariance matrix estimation.
Consider the case with observation y ∈ CN modeled as
y = Hx + z, (35)
where H ∈ CN × M is a deterministic channel matrix and
z ∈ CN a zero-mean, white noise with covariance matrix
σ2I, which is uncorrelated with the zero-mean input signal
x ∈ CM with covariance matrix I. If both training samples of
x and y are known, we may ﬁrst estimate the channel matrix
H and the covariance matrix of the noise z using the ordinary

<!-- page 5 -->

5
least squares (OLS) approach. Let the block of training data
be (X, Y), where the input signal X can be designed to have
certain properties such as being orthogonal. The OLS estima tes
of the channel matrix and noise variance are then obtained as
ˆH = YX† (
XX†) − 1
, (36)
ˆσ2 = 1
T N


Y − ˆHX



2
F
= 1
T N tr
(
(Y − ˆHX)(Y − ˆHX)†
)
= 1
T N tr
(
Y(I − X†(XX†)− 1X)Y†)
, (37)
where ˆ(·) denotes the estimate of a quantity. In this case, the
covariance matrix of y can be estimated as
R = ˆH ˆH† + ˆσ2I. (38)
Such OLS-based covariance matrix estimation may be useful
for designing signal estimation schemes in wireless commu-
nications. We can apply the linear shrinkage design (1) to
enhance its accuracy and apply the LOOCV method (12) to
choose the shrinkage coefﬁcients. Note that in this case, in
the t-th split, we generate the covariance matrix estimate Rt
by applying the OLS estimate to the leave-one-out samples
(Xt, Yt) which are the subset of (X, Y) with the pair (xt, yt)
omitted. The LOOCV cost is the same as (15). In this case,
the leave-one-out estimate of the covariance matrix for the t-th
data split is
Rt = ˆHt ˆH†
t + ˆσ2
t I, (39)
where ˆHt and ˆσ2
t denote the channel matrix and noise variance
estimated from (Xt, Yt), respectively. A direct computation of
(16) and (17) for evaluating the LOOCV cost performs OLS
estimation for T times, which incurs signiﬁcant complexity.
The complexity can be greatly reduced by observing that the
leave-one-out OLS estimate of the channel matrix is related
to the OLS channel matrix estimate ˆH in (36) by a rank-one
update:
ˆHt = YtX†
t
(
XtX†
t
) − 1
= ˆH − etf †
t , (40)
where
et ≜ yt − ˆHxt, (41)
ft ≜ 1
1 − Φt
(XX†)− 1xt. (42)
In the above,
Φt ≜ x†
t (XX†)− 1xt (43)
is the t-th diagonal entry of
Φ = X†(XX†)− 1X. (44)
Similarly, the leave-one-out estimate of the noise varianc e can
be updated as
ˆσ2
t = 1
N (T − 1) tr
(
(Yt − ˆHtXt)(Yt − ˆHtXt)†
)
= ˆσ2 − δt, (45)
where
δt = ∥et∥2
F
N (T − 1)(1 − Φt) −
ˆσ2
T − 1 . (46)
Note that both updates can be achieved with low complexity
when a few matrices are computed in advance and reused. In
this way, the covariance matrix estimate can be computed as
Rt = R − δtI − etφ†
t − ψte†
t , (47)
where φt and ψt are deﬁned as
φt = ˆHft, (48)
ψt = φt − ||ft||2
F et. (49)
This shows that the leave-one-out OLS covariance matrix
estimate can be obtained from R by corrections involving
a scaled identity matrix and two rank-one updates. Eqn.
(47) can be exploited to compute the closed-form LOOCV
solution quickly. From (47), the most involved computation
for ﬁnding the solution of the optimization problem (15) can
be implemented as
1
T
T∑
t=1
tr(R2
t ) = tr( R2) + N ∑T
i=1 δ2
t
T − 2 ∑T
i=1 δt
T tr(R)
+ 1
T
T∑
i=1
||et||2
F (||φt||2
F + ||ψt||2
F )
− 2
T
T∑
i=1
R(e†
t (R − δtI)(φt + ψt) − e†
t ψte†
t φt),
(50)
where R(·) denotes the real part of a scalar. When R is already
computed, the right-hand side of (50) can be evaluated using
inner products and matrix-vector products. The terms tr(T2
0)
and tr(T0St) are the same as those for SCM. For the other
two terms, we have
1
T
T∑
t=1
tr(RtSt) = 1
T tr
(
RYY†)
− 1
T
T∑
t=1
R(δt ∥yt∥2
F + y†
t etφ†
t yt + y†
t ψte†
t yt),
(51)
1
T
T∑
t=1
tr(RtT0) = tr ( RT0) −
∑T
t=1 δt
T tr(T0)
− 1
T
T∑
t=1
R(φ†
t T0et + e†
t T0ψt). (52)
Note that the computational complexities of (50)-(52) are l ow
because the major operations are matrix-vector products an d
inner products.
E. Comparisons with Alternative Choices of Linear Shrinkag e
Coefﬁcients
In the above, we have introduced LOOCV methods with
analytical solutions for choosing the coefﬁcients for line ar

<!-- page 6 -->

6
shrinkage covariance matrix estimators. We now discuss sev -
eral alternative techniques which have received considera ble
attentions recently and compare them with the LOOCV meth-
ods proposed in this paper.
In 2004, Ledoit and Wolf (LW) [2] studied estimators that
shrink SCM toward an identity target, i.e., T0 = I. Such
estimators do not alter the eigenvectors but shrink eigenva lues
of the SCM, which is well supported by the fact that sample
eigenvalues tend to be more spread than population eigen-
values. The optimal shrinkage coefﬁcients under the MMSE
criterion (3) can be written as
ρ⋆ = α2
δ2 , τ ⋆ = β2
δ2 µ, (53)
where the parameters µ ≜ tr(Σ)
N , δ2 ≜ E[∥R − µI∥2
F ],
β2 = E[ ∥Σ − R∥2
F ], and α2 = ∥Σ − µI∥2
F depend on the
true covariance matrix Σ and other unknown statistics. Ref.
[2] shows that δ2 = α2+β2 and proposes to approximate these
quantities by their asymptotic estimates under T → ∞ , N →
∞ , N/T → c < ∞ , as
ˆµ = tr(R)
N , ˆδ2 = ∥R − ˆµI∥2
F , (54)
ˆβ2 = min
(
ˆδ2, 1
T 2
T∑
t=1


yty†
t − R



2
F
)
, ˆα2 = ˆδ2 − ˆβ2,
(55)
which can all be computed from the training samples. By
substituting these into (53), estimators that signiﬁcantl y out-
perform SCM are obtained, which also approach the oracle
estimators when the training length is large enough.
The above LW estimator is extended by Stoica et al in
2008 [3] for complex-valued signals with general shrinkage
targets T0, with applications to knowledge-aided space-time
adaptive processing (KA-STAP) in radar applications. Seve ral
estimators with similar performance are derived there. For the
general linear combination (GLC) design of [3], it is shown
that the oracle shrinkage coefﬁcients for (1) satisfy
ρ⋆ = 1 − τ ⋆
ν , (56)
where
ν = tr(T0Σ)
∥T0∥2
F
, τ ⋆ = ν β2
E[∥R − νT0∥2
F ]
. (57)
The quantity β2 is estimated in the same way as (55), and a
computationally efﬁcient expression for ˆβ2 is given by
ˆβ2 = 1
T 2
T∑
t=1
∥yt∥4
F − 1
T ∥R∥2
F . (58)
Furthermore, ν and E[||R − νT0||2
F ] are estimated as ˆν =
tr(T0R)
||T0||2
F
and ||R− ˆνT0||2
F , respectively. This leads to the result
given by Eqns. (34) and (35) of [3], which can recover the
LW estimator [2] when the identity shrinkage target T0 = I
is assumed.
More recently, Chen et al [4] derived the oracle approx-
imating shrinkage (OAS) estimator, which assumes SCM,
real-valued Gaussian samples, and scaled identity target w ith
T0 = tr(R)
N I and ρ = 1 − τ . They ﬁrst derive the oracle
shrinkage coefﬁcients for SCM obtained from i.i.d. Gaussia n
samples, which is determined by N, T, tr(Σ) and tr(Σ2).
Then, they propose an iterative procedure to approach the
oracle estimator. In the iterations, tr(Σ2) and tr(Σ) are
estimated by tr( ˆΣj R) and tr( ˆΣj), respectively, where ˆΣj
is the covariance matrix estimate at the j-th iteration. It is
further proved that ˆΣj converges to the OAS estimator with
the following analytical expression for τ :
τ ⋆
OAS = min
(
1,
(
1 − 2
N
)
tr(R2) + (tr(R))2
(
T + 1 − 2
N
)
[tr(R2) − (tr(R))2
N ]
)
. (59)
This approach achieves superior performance for (scaled) i den-
tity target and Gaussian data and dominates the LW estimator
[2] when T is small. It was later generalized by Senneret et
al [20] to a shrinkage target chosen as the diagonal entries of
the SCM. Other related techniques include [14], which also
assumes SCM, Gaussian data, and identity/diagonal shrinka ge
targets.
All the above techniques provide analytical solutions and
achieve near-oracle performance when the underlying as-
sumptions (e.g., large dimensionality, large size of train ing
data, identity/diagonal shrinkage targets) hold. However , they
also have limitations. A common restriction is that all thes e
analytical solutions assume SCM and are not optimized for
other types of covariance matrix estimators such as model-
based estimators. In particular, the LW and GLC methods [2],
[3], which employ asymptotic approximations, may exhibit a
noticeable gap to the oracle choice when the sample support
is low, which may be relevant in some applications. The OAS
method [4] assumed identity target, but its extensions to mo re
general cases, e.g., with multiple/general shrinkage targ ets,
are not trivial. By contrast, the LOOCV method proposed in
this paper allows different designs and achieves near-orac le
performance in general.
Cross-validation has also been applied previously for choo s-
ing shrinkage coefﬁcients for covariance matrix estimatio n.
The key issues for applying this generic tool include ﬁnding
appropriate predictive metrics for scoring the different e stima-
tors and fast computation schemes. In [10], [13], the Gaussi an
likelihood was chosen as such a proxy. The computations
with likelihood are generally involved as multiple matrix i n-
verses/determinants are required, and a grid search is requ ired
for ﬁnding the optimal parameters. In this paper, we use the
distribution-free, Frobenius norm loss in (12) as the metri c,
which leads to analytical solutions and is computationally
more tractable.
III. M ULTI -TARGET SHRINKAGE
In Section 2, we have considered linear shrinkage designs
with a single target. Multiple shrinkage targets may be used to
further enhance performance, which may be obtained from a
priori knowledge, e.g., a past covariance matrix estimate f rom
older training samples or from neighboring frequencies. We
can easily extend our proposed LOOCV method to multiple
targets.

<!-- page 7 -->

7
A. Oracle choice of shrinkage coefﬁcients
Consider the multi-target shrinkage design
ˆΣρ, τ = ρR +
K∑
k=1
τkTk, (60)
where all the shrinkage coefﬁcients are nonnegative to guar -
antee PSD covariance matrix estimates, i.e.,
ρ ≥ 0; τk ≥ 0, ∀k. (61)
The oracle multi-target shrinkage minimizes the squared
Frobenius norm of the estimation error
JO, MT(ρ, τ ) =




ρR +
K∑
k=1
τkTk − Σ





2
F
, (62)
which can be rewritten as
JO, MT(ρ, τ ) =
[ ρ
τ
]T
AO, MT
[ ρ
τ
]
− 2
[ ρ
τ
]T
bO, MT+tr(Σ2),
(63)
where τ = [τ1, τ2, · · ·, τK ]T ,
AO, MT =





tr(R2) tr( RT1) · · · tr(RTK)
tr(T1R) tr( T2
1) · · ·tr(T1TK)
.
.
.
.
.
. . . . .
.
.
tr(TKR) tr( TKT1) · · · tr(T2
K)




 , (64)
bO, MT =





tr(RΣ)
tr(T1Σ)
.
.
.
tr(TK Σ)




 . (65)
The oracle shrinkage coefﬁcients can then be obtained by sol v-
ing the problem of minimizing the cost function JO, MT(ρ, τ )
of (63), which is a strictly convex quadratic program (SCQP)
with K + 1 variables.
B. LOOCV choice of shrinkage coefﬁcients
We now extend the LOOCV method in Section 2 to the
multi-target shrinkage here. Following the same treatment as
in Section II-B, in each split of the training data, Rt and St
are constructed to generate and validate the covariance mat rix
estimate, respectively. The multiple shrinkage coefﬁcien ts are
chosen to minimize the LOOCV cost
JCV, MT(ρ, τ ) = 1
T
T∑
t=1




ρRt +
K∑
k=1
τkTk − St





2
F
. (66)
The above cost function can be rewritten in a form similar to
(15) as
JCV, MT(ρ, τ ) =
[ ρ
τ
]T
ACV, MT
[ ρ
τ
]
− 2
[ ρ
τ
]T
bCV, MT
+ 1
T
T∑
t=1
tr(S2
t ) (67)
with
ACV, MT =
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

T∑
t=1
tr(R2
t )
T
T∑
t=1
tr(RtT1)
T · · ·
T∑
t=1
tr(RtTK )
T
T∑
t=1
tr(T1Rt)
T tr(T2
1) · · · tr(T1TK)
.
.
. .
.
. . . . .
.
.
T∑
t=1
tr(TK Rt)
T tr(TKT1) · · · tr(T2
K)
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

,
(68)
bCV, MT =
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

1
T
T∑
t=1
tr(RtSt)
1
T
T∑
t=1
tr(T1St)
.
.
.
1
T
T∑
t=1
tr(TK St)
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

. (69)
The constant entries of ACV, MT and bCV, MT can be computed
in the same way as for the single-target case. When K is small,
which is typically the case, the solution that minimizes the
LOOCV cost can be found quickly using standard optimization
tools. Alternatively, we may ﬁnd ﬁrst the global optimizer t hat
ignores the nonnegative constraint by
[ ρ⋆
CV, MT
τ ⋆
CV, MT
]
= A− 1
CV, MTbCV, MT, (70)
and check if the nonnegative condition is satisﬁed. If a
negative shrinkage coefﬁcient is produced, we then conside r
the boundaries of ρ ≥ 0, τk ≥ 0, k = 1 , 2, · · ·, K, which
are equivalent to removing a certain number of shrinkage
targets from the shrinkage design. The solution can be found
in exactly the same way as (70) but with fewer targets.
Similarly to the single-target case, we may also consider a
constrained case, where the shrinkage targets {Tk} have the
same trace as the estimated covariance matrix R, and
ρ +
K∑
k=1
τk = 1. (71)
Then the LOOCV cost function can be rewritten as
JCV, MT(τ ) = 1
T
T∑
t=1





K∑
k=1
τkAkt + Bt





2
F
, (72)
where
Akt ≜ Tk − Rt, 1 ≤ k ≤ K, 1 ≤ t ≤ T, (73)
Bt ≜ Rt − St, 1 ≤ t ≤ T. (74)
The optimal shrinkage coefﬁcients can be found similarly as
for the unconstrained case by minimizing
JCV, MT(τ ) = τ T A′
CV, MTτ − 2τ T b′
CV, MT + 1
T
T∑
t=1
tr(B2
t ),
(75)

<!-- page 8 -->

8
where the entries of A′
CV, MT and b′
CV, MT are deﬁned by
[A′
CV, MT]mn ≜ 1
T
T∑
t=1
tr(AmtAnt), 1 ≤ m, n ≤ K, (76)
[b′
CV, MT]k ≜ 1
T
T∑
t=1
tr(AktBt), 1 ≤ k ≤ K. (77)
These entries may also be evaluated quickly. For example,
with SCM,
[A′
CV, MT]mn = tr(TmTn) − tr((Tm + Tn)R) + η, (78)
[b′
CV, MT]k = T
T − 1 tr(R2)− 1
T (T − 1)
T∑
t=1
||yt||4
F − η, (79)
where
η = 1
T
T∑
t=1
tr(R2
t )
can be computed using (27). The solution to τ can be found
as
τ ⋆
CV, MT = A′− 1
CV, MTb′
CV, MT (80)
if the nonnegative condition is satisﬁed. Otherwise, ﬁnd th e
solution in a similar way as for the unconstrained case.
Note that for multi-target shrinkage, Lancewicki and Al-
adjem [17] recently introduced another method for ﬁnding
the shrinkage coefﬁcients. They assume SCM and shrinkage
targets which belong to a set that can be characterized by
Eqn. (21) of [17]. Then, they follow the Ledoit-Wolf (LW)
framework [2] to derive unbiased estimates of the unknown
coefﬁcients needed for minimizing the expectation of the co st
in (62), based on which {ρ, τ } can be optimized. By contrast,
our approach resorts to a LOOCV estimate of the cost in (10),
which does not rely on the aforementioned assumptions in
[17]. As will be shown later, the LOOCV method can achieve
similar performance as [17] for the shrinkage targets consi d-
ered there. However, it can be applied to general estimators
other than SCM and shrinkage targets which are not covered
by Eqn. (21) of [17], offering wider applicability.
IV. N UMERICAL EXAMPLES
In this section, we present numerical examples to demon-
strate the effectiveness of the proposed shrinkage design a nd
compare it with alternative methods. The quality of covaria nce
matrix estimation is measured by the MSE normalized by the
average of the squared Frobenius norm ||Σ||2
F , i.e.,
NMSEΣ ≜ E[||ˆΣρ,τ − Σ||2
F ]
E[||Σ||2
F ] . (81)
We show examples of covariance matrix estimation and its ap-
plications in array signal processing. We denote by N (µ, σ2)
a real-valued Gaussian distribution with mean µ and variance
σ2.
Example 1: Shrinkage toward an identity target: We ﬁrst
consider a real-valued example with an autoregressive (AR)
covariance matrix, whose (i, j)-th entry is given by
[Σ]i,j = r|i− j|, 1 ≤ i, j ≤ N, (82)
0 5 10 15 20 25 30
T
-5
0
5
10
15Normalized MSE (dB)
Figure 1. NMSE of single-target (ST) shrinkage estimates of an AR
covariance matrix with N = 100 , r = 0 .5, T0 = tr(R)
N I. “LW”, “GLC”
and “OAS” refer to the methods of [2], [3] and [4], respective ly, which are
also described in Section II-E; “CV” refers to our proposed L OOCV method;
“Oracle” refers to the coefﬁcient choice in Section II-A; an d “Con” and “Unc”
indicate that the constraint ρ + τ = 1 is imposed or not, respectively.
which has been widely considered for evaluating covariance
matrix estimation techniques [4]-[7]. Let Σ1/ 2 be the Cholesky
factor of Σ. The training samples are randomly generated
as yt = Σ1/ 2nt, where nt consists of i.i.d. entries drawn
from N (0, 1). The typical shrinkage target T0 = tr(R)
N I is
considered for single-target shrinkage. Our proposed LOOC V
method is compared with the widely used alternative methods
[2]-[4] for choosing the shrinkage coefﬁcients. The simula tion
results (averaged over 1000 repetitions for each training length
T ) in Fig. 1 conﬁrm that the LOOCV methods with and
without the constraint ρ + τ = 1 produce the same results
for the scaled identity target and they achieve performance
almost identical to the OAS estimator [4], which was derived
by assuming Gaussian data and identity target. The LW [2]
and GLC [3] methods, which are equivalent for the scaled
identity target here, do not perform well for very low sample
support, but are able to approximate the oracle choice very
well when more samples are available, which is consistent
with the observations from [4]. All of these shrinkage desig ns
signiﬁcantly outperform the SCM, conﬁrming the effectiven ess
of shrinkage for covariance matrix estimation. Recall that
these methods were derived using different strategies and
assumptions and have different analytical solutions.
Example 2: Shrinkage toward a nondiagonal target: We
then consider an example of the linear model given by
(35). For each training length, 1000 random realizations of
Σ = HH† +σ2I are generated and estimated through training,
where σ2 = 0.1. The entries of H are independently generated
from N (0, 1) and then ﬁxed for the whole training process.
Given H, T training samples are generated by y = Hx + z,
with the entries of x and z generated independently from
N (0, 1) and N (0, σ2), respectively. In order to demonstrate
the effectiveness of the LOOCV method for general shrinkage
targets, we assume a scenario where H is slowly time-varying
and the shrinkage target T0 can be constructed as a well-

<!-- page 9 -->

9
101 102
T
-10
-8
-6
-4
-2
0
2
Normalized MSE (dB)
Figure 2. NMSE of single-target (ST) shrinkage estimation o f covariance
matrix for the linear model (35) with N = 50, M = 50, σ2 = 0.1. The non-
diagonal shrinkage target is constructed from the estimate of a past covariance
matrix. The result indicated by “ T0” corresponds to estimating Σ as T0.
“Identity” indicates a scaled identity shrinkage target is used instead. We
can show that imposing the constraint (23) leads to negligib le change in
performance for the proposed LOOCV approach.
conditioned estimate of a past covariance matrix
Σpast = HpastHpast† + σ2I, (83)
where
Hpast = H + ∆, (84)
and the entries of ∆ are independently drawn from N (0, 0.2)
and are ﬁxed for each repetition. Speciﬁcally, we construct T0
as the shrinkage estimate of Σpast using SCM and the scaled
identity target. This construction is similar to the knowle dge-
aided target considered in [3] and the resulting T0 is not
diagonal. We assume that the numbers of samples used for
estimating Σ and T0 are both equal to T . The simulation
results are included in Fig. 2 for the normalized MSE. It can b e
seen that the LOOCV methods generally achieve near-oracle
performance and outperform the GLC method. Also, the non-
diagonal shrinkage target achieves better performance tha n the
scaled identity target.
Example 3: Shrinkage with multiple targets: A multi-target
example is illustrated in Fig. 3. An AR covariance matrix
is estimated by shrinking SCM with three targets which
can be represented by Eqn. (21) of [17]: T1 = tr(R)
N I,
T2 = Diag(R), and T3 is a symmetric, Toeplitz matrix which
was considered in [17]:
T3 = tr(R)
N I +
N − 1∑
i=1
tr(CiR)
2(N − i) Ci, (85)
where Ci is a symmetric, Toeplitz matrix with unit entries on
the i-th sub- and super-diagonals and zeros elsewhere. It is
seen that multi-target shrinkage can signiﬁcantly outperf orm
single-target shrinkage with T0 = tr(R)
N I when the number
of samples is large enough. For the oracle parameter choices ,
the unconstrained shrinkage design, which allows a larger s et
of shrinkage factors to be chosen, can noticeably outperfor m
the design constrained by (71). However, when the proposed
101 102
T
-18
-16
-14
-12
-10
-8
-6
-4
-2
0
2
Normalized MSE (dB)
Figure 3. MSE of covariance matrix estimation with multi-ta rget (MT)
shrinkage and LOOCV parameter choices. AR covariance matri x with N =
50, r = 0.9 is assumed. “LA” refers to the method proposed by Lancewicki
and Aladjem [17]. Note that the LOOCV methods and the LA metho d achieve
similar performance for this example.
LOOCV methods are used, the gap is signiﬁcantly reduced.
We can show that when the number of samples is small,
using a more parsimonious design with constrained shrinkag e
coefﬁcients or fewer shrinkage targets may achieve better
performance. It is seen that the multi-target shrinkage met hod
of [17] (indicated by “MT-LA” in Fig. 3) performs similarly
to the LOOCV method for this example. Note that the method
of [17] assumes SCM and shrinkage targets satisfying certai n
structures and does not apply directly to model-based covar i-
ance matrix estimation or more general shrinkage targets.
Example 4: Application to MMSE estimation of MIMO
channels. A potential application of the proposed technique is
the design of MMSE estimator of MIMO channels. Consider
a point-to-point MIMO system with Nt transmitting antennas
and Nr receiving antennas. Let B be the length of the pilot
sequence. The received signal matrix during the training st age
is modelled as
Y = HP + N, (86)
where Y ∈ CNr× B is the received signal matrix, H ∈
CNr× Nt the channel matrix, P ∈ CNt× B the pilot matrix,
and N ∈ CNr× B the noise which is uncorrelated with H.
V ectorizingY in (86) gives
y = ˜Ph + n, (87)
where y = vec(Y), ˜P = PT ⊗ I, h = vec(H), n = vec(N),
vec(·) denotes vectorization, and ⊗ denotes Kronecker prod-
uct. We assume a Rayleigh fading channel and denote by
Σh ∈ CNtNr× NtNr the covariance matrix of the channel
vector h. We also assume that the disturbance n is complex
Gaussian-distributed with a zero mean and identity covaria nce
matrix.
Given Σh, the MMSE estimate of h from y can be
computed as [41]
ˆhMMSE = Σh ˜P†( ˜PΣh ˜P† + I)− 1y. (88)
The covariance matrix Σh, which can be very large, must
be estimated in order to compute ˆhMMSE. In communication

<!-- page 10 -->

10
100 150 200 250 300 350 400 450 500 550 600
T
-9
-8
-7
-6
-5
-4
-3
-2
-1
0
NMSE of channel estimation (dB)
Figure 4. Performance of MMSE estimation of MIMO channels wi th the
channel covariance matrix estimated using different estim ators with Nt =
Nr = B = 10. Pilot-to-noise ratio is 5 dB. “LS” refers to the LS estimator
in (89); “MMSE” refers to the MMSE channel estimator (88) con structed
using estimated covariance matrices; “identity” and “past ” represent shrinkage
targets chosen as the scaled identity matrix and the estimat e of a past
covariance matrix, respectively; “MT-CV” uses both the ide ntity target and
the target set as a past estimate.
systems, h is not directly observable and thus the SCM
estimator can not be directly applied to generate Σh. One
may estimate Σh from least squares (LS) estimates of H, i.e.,
ˆHLS = YP†(PP†)− 1. (89)
When orthogonal training signal with P =
√
P I is applied,
where P determines the power for training signals, it can be
shown that
ˆHLS = 1√
P
Y = 1√
P
(HP + N) = H + 1√
P
N. (90)
Denote by ˆhLS the vectorization of ˆHLS. It can be shown that
the covariance matrix of ˆhLS is
ΣˆhLS
≜ E[ˆhLSˆh†
LS] = Σh + 1
P I. (91)
Therefore, if ΣˆhLS
is estimated as ˆΣˆhLS
, we can then use (91)
to estimate Σh as
ˆΣh = ˆΣˆhLS
− 1
P I,
which can be used in (88). The estimation of ΣˆhLS
can be
achieved using the different shrinkage estimators introdu ced
in this paper.
An example is shown in Fig. 4. The covariance matrix is
assumed to be
Σh = Σt ⊗ Σr, (92)
where Σt and Σr are, respectively, the transmitter side and
receiver side covariance matrix, with entries given by
[Σt]i,j =
{
r|i− j|
t , i ≥ j
(r∗
t )|i− j|, i < j , (93)
[Σr]i,j =
{
r|i− j|
r , i ≥ j
(r∗
r )|i− j|, i < j , (94)
rt = 0 .7e− j0. 9349π and rr = 0 .9e− j0. 9289π . While applying
shrinkage to estimate ΣˆhLS
, two shrinkage targets are tested:
the identity matrix and the shrinkage estimate (with a scale d
identity target) of a past covariance matrix. The second is
considered based on the assumption that Σh is slowly varying
in time and a well-conditioned estimate of a past covariance
matrix Σpast
h can be available. In our simulations, Σpast
h
is modeled by randomly perturbing rt and rr in (93) and
(94) by δt and δr whose real and imaginary parts are both
randomly and uniformly generated from
[
− 1
10
√
2 , 1
10
√
2
]
. The
normalized MSE of channel estimation is deﬁned as
NMSEh ≜ E[||ˆhMMSE − h||2
F ]
E[||h||2
F ] , (95)
where ˆhMMSE is the MMSE channel estimate obtained from
(88) with the true channel covariance matrix replaced by its
shrinkage estimate.
From the simulation results in Fig. 4, when the number of
samples T of channel estimates is small, the MMSE channel
estimator constructed using the SCM estimate of Σh is poorer
than the LS estimator which does not require any knowledge of
Σh. Therefore, an accurate estimate of the covariance matrix
is necessary to exploit the potential of the MMSE channel
estimator. Shrinkage with LOOCV choice of the shrinkage
coefﬁcients improves the performance of the MMSE channel
estimator by providing a better estimate of Σh. Two-target
shrinkage can further enhance performance. Note that the
multi-target method of [17] is not directly applicable to th e
shrinkage target used here. Similarly to [42], we do not expl oit
the Kronecker product structure in (92) and the exponential
modeling of (93) and (94) while estimating the covariance
matrix and similar trends can be observed when the channel
covariance matrix follows different models such as those in
[43], [44].
Example 5: Application to LMMSE signal estimation: An-
other example application is the design of linear minimum
mean squared error (LMMSE) estimator [45], [46] for estimat -
ing the transmitted signal x in MIMO communications. The
received signal is modeled by (35) and the LMMSE estimate
of x is obtained as
ˆx = H†Σ− 1
y y, (96)
where we have assumed that x has identity covariance ma-
trix and Σy is the covariance matrix of y. The OLS-based
covariance matrix estimation in Section II-D can be used to
estimate Σy in (96). In Fig. 5, we show an example where
the shrinkage target T0 is chosen as the diagonal matrix of
the OLS estimate (38) of the covariance matrix. This results
in a shrunk LMMSE signal estimator
ˆx = ˆH†(ρ( ˆH ˆH† + ˆσ2I) + τ T0)− 1y. (97)
Orthogonal training of length T constructed from the discrete
Fourier transform (DFT) matrix is assumed for the OLS
channel estimate and ﬁnding the shrinkage coefﬁcients usin g
our proposed LOOCV method is achieved at a low complexity.

<!-- page 11 -->

11
40 50 60 70 80 90 100 110 120 130 140
T
-8
-7
-6
-5
-4
-3
-2Normalized MSE of signal estimation (dB)
Standard OLS
OLS with shrinkage, CV
OLS with shrinkage, oracle
Nr = 60, Nt = 40
Nr = Nt = 40
Figure 5. Performance of the LMMSE signal estimator with cha nnel matrix
and received signal’s covariance matrices estimated using OLS and shrinkage.
The entries of H are independently generated from complex Gaussian distri-
bution with zero mean and variance 1/40, and the noise variance σ2 = 0.1.
The normalized MSE of signal estimation is deﬁned as
NMSEx ≜ E[||ˆx − x||2
F ]
E[||x||2
F ] . (98)
Fig. 5 presents the simulation results averaged over 1000
random realizations of H for each T . It can be seen that
the shrinkage estimate of the covariance matrix can lead to
noticeable improvement of the MSE performance of signal
estimation. The resulting performance can approach the ora cle
choice of (ρ, τ) that minimizes the MSE of estimating x [35].
Note that in contrast to the cross-validation methods in [35 ]
and [36] which choose shrinkage factors by a grid search
for optimizing the signal estimation performance, the meth od
proposed in this paper has an analytical solution and optimi zes
covariance matrix estimation. It also differs from [47] whi ch
targets the design of a signal estimator that shrinks the sam ple
LMMSE ﬁlter toward the matched ﬁlter.
Example 6: Application to MVDR beamforming: Finally, we
show an example application to minimum variance distortion -
less response (MVDR) beamforming [31], [33]. We assume
a N = 30 -element uniform linear array (ULA) with half-
wavelength spacing between neighboring antennas. As in [33 ],
we assume that the desired complex Gaussian signal has an
angle of arrival (AoA) of θ0 = 0 ◦ and there are 8 complex
Gaussian interferences in the directions {θm} = {8◦, − 15◦,
23◦, − 21◦, 46◦, − 44◦, − 85◦, 74◦}, all with an average
power 10 dB higher than the desired signal. The noise is
assumed to be additive white Gaussian noise (AWGN) with
an average power 10 dB lower than the desired signal. The
MVDR beamformer is given by
w = Σ− 1s
s†Σ− 1s , (99)
where s is the steering vector of the desired signal and Σ is
the covariance matrix of the received signal. We consider a
practical scenario where the desired signal’s steering vec tor
suffers from an AoA error uniformly distributed in [− 5◦ , 5◦]
and Σ is estimated from the training samples by shrinking
the SCM R toward the scaled identity matrix tr(R)
N I. We
5 10 15 20 25 30
T
-10
-8
-6
-4
-2
0
2
4
6
8
Average SINR (dB)
Figure 6. Average output SINR for a MVDR beamformer with AoA m ismatch
and the estimated covariance matrix. The results labeled by “SCM” is obtained
by replacing Σ−1 in (99) with the pseudo-inverse of the SCM. Note that the
LOOCV and OAS methods achieve almost the same performance, w hich is
slightly better than the GLC method when T is very small.
focus on the low-sample-support case and compare the result
with an approach that uses the pseudo-inverse of the SCM for
computing w. The output signal-to-interference-and-noise ra-
tio (SINR) averaged over 1000 repetitions are plotted in Fig. 6.
It is seen that though the proposed approach targets covaria nce
matrix estimation only and is not optimized for beamformer
designs, it still provides noticeable gains as compared to t he
pseudo-inverse approach in the low-sample-support regime .
V. C ONCLUSIONS
In this paper, we have introduced a leave-one-out cross-
validation (LOOCV) method for choosing the coefﬁcients for
linear shrinkage covariance matrix estimators. By employi ng
a quadratic loss as the LOOCV prediction error, analytical
expressions of the optimal shrinkage coefﬁcients are obtai ned,
which do not require a grid search of the parameters. As a
result, the coefﬁcients can be computed at low costs for the
SCM- and OLS-based estimation of the covariance matrix.
The LOOCV method is generic in the sense that it can be
applied to different covariance matrix estimation methods and
different shrinkage targets. Numerical examples show that
it can approximate the oracle parameter choices in general
and have wider applications than several existing analytic al
methods that have been widely applied.
Zero-mean signals have been assumed in this paper. When
nonzero-mean signals are considered, our proposed approac h
may be applied after subtracting an estimate of the mean
from the samples. However, the inaccuracy in the mean vector
estimate may introduce extra errors to the covariance matri x
estimation. Jointly estimating the mean and covariance mat rix
in a robust manner may be further explored. Other future work
includes theoretical study of the properties of the propose d
approach and low-complexity cross-validation schemes for
choosing shrinkage factors for speciﬁc signal processing a ppli-
cations such as beamforming, space-time adaptive processi ng,
correlation analysis, etc.

<!-- page 12 -->

12
ACKNOWLEDGMENTS
The authors wish to thank Prof. Antonio Napolitano and the
anonymous reviewers for their constructive comments which
have greatly improved the paper. This work was supported
in part by an International Links grant of University of
Wollongong (UOW), Australia, and in part by NSFC under
Grant 61601325.
REFERENCES
[1] L. L. Scharf, Statistical Signal Processing: Detection, Estimation, an d
Time Series Analysis , Addison–Wesley, Bosten, 1991.
[2] O. Ledoit and M. Wolf, “A well-conditioned estimator for large-
dimensional covariance matrices,” J. Multivar . Anal., vol. 88, pp. 365-
411, 2004.
[3] P . Stoica, J. Li, X. Zhu, and J. R. Guerci, “On using a prior i knowledge
in space-time adaptive processing,” IEEE Trans. Sig. Process. , vol. 56,
no. 6, pp. 2598-2602, 2008.
[4] Y . Chen, A. Wiesel, Y . C. Eldar, and A. O. Hero, “Shrinkage algorithms
for MMSE covariance estimation,” IEEE Trans. Sig. Process. , vol. 58,
no. 10, pp. 5016-5029, 2010.
[5] L. Du, J. Li, and P . Stoica, “Fully automatic computation of diagonal
loading levels for robust adaptive beamforming,” IEEE Trans. Aerosp.
Electron. Syst, vol. 46, no. 1, pp. 449-458, 2010.
[6] P . J. Bickel and E. Levina, “Regularized estimation of la rge covariance
matrices,” Ann. Statist. , vol. 36, no. 1, pp. 199-227, 2008.
[7] P . J. Bickel and E. Levina, “Covariance regularization b y thresholding,”
Ann. Statist. , vol. 36, no. 6, pp. 2577-2604, 2008.
[8] C. Stein, “Inadmissibility of the usual estimator for th e mean of a
multivariate normal distribution,” Proc. Third Berkeley Symp. Math.
Statist. Prob., 1, pp. 197-206, 1956.
[9] L. R. Haff, “Empirical Bayes estimation of the multivari ate normal
covariance matrix,” Ann. Statist. , vol. 8, no. 3, pp. 586597, 1980.
[10] J. P . Hoffbeck and D.A. Landgrebe, “Covariance matrix e stimation and
classiﬁcation with limited training data,” IEEE Trans. Pattern Anal.
Mach. Intell. , vol. 18, no. 7, pp.763-767, 1996.
[11] M. Daniels and R. Kass, “Shrinkage estimators for covar iance matrices,”
Biometrics, vol. 57, pp. 1173-1184, 2001.
[12] J. Sch¨ afer and K. Strimmer, “A shrinkage approach to la rge-scale
covariance matrix estimation and implications for functio nal genomics,”
Statist. Appl. Genetics Molecular Biol. , vol. 4, 2005.
[13] D. I. Warton, “Penalized normal likelihood and ridge re gularization of
correlation and covariance matrices,” JASA, vol. 103, no. 481, pp. 340-
49, 2008.
[14] T. J. Fisher and X. Sun, “Improved Stein-type shrinkage estimators
for the high-dimensional multivariate normal covariance m atrix,” Comp.
Statist. Data Analysis , 55, 1909-1918, 2011.
[15] X. Chen, Z. Jane Wang, and M. J. McKeown, “Shrinkage-to- tapering
estimation of large covariance matrices,” IEEE Trans. Sig. Process. , vol.
60, pp. 5640-5656, 2012.
[16] J. Theiler, “The incredible shrinking covariance esti mator,” Proc. SPIE.,
vol. 8391, pp. 83910P , 2012.
[17] T. Lancewicki and M. Aladjem, “Multi-target shrinkage estimation for
covariance matrices,” IEEE Trans. Sig. Process. , vol. 62, no. 24, pp.
6380-6390, 2014.
[18] Y . Ikeda, T. Kubokawa, and M. S. Srivastava, “Compariso n of linear
shrinkage estimators of a large covariance matrix in normal and non-
normal distributions,” Comput. Stat. Data Anal. , vol. 95, 95-108, 2016.
[19] T. Tong, C. Wang, and Y . Wang, “Estimation of variances a nd covari-
ances for high-dimensional data: a selective review,” Wiley Interdiscip.
Rev. Comput. Stat. , vol. 6, no. 4, pp. 255-264, 2014.
[20] M. Senneret, Y . Malevergne, P . Abry, G. Perrin, and L. Ja ffrs, “Covari-
ance versus precision matrix estimation for efﬁcient asset allocation,”
IEEE J. Sel. Topics Sig. Process. , vol. 10, no. 6, pp. 982-993, Sept.
2016.
[21] J. Fan, Y . Liao, and H. Liu, “An overview of the estimatio n of large
covariance and precision matrices,” Econom. J. 19, no. 1, C1-C32, 2016.
[22] O. Ledoit and M. Wolf, “Nonlinear shrinkage estimation of large-
dimensional covariance matrices,” Ann. Statist. , vol. 40, no. 2, 1024-
1060, 2012.
[23] C. Lam, “Nonparametric eigenvalue-regularized preci sion or covariance
matrix estimator,” Ann. Statist. , vol. 44, no. 3, pp. 928-953, 2016.
[24] A. Aubry, A. De Maio, L. Pallotta, and A. Farina, “Maximu m likelihood
estimation of a structured covariance matrix with a conditi on number
constraint,” IEEE Trans. Signal Process. , vol. 60, pp. 3004-3021, 2012.
[25] J.-H. Won, J. Lim, S.-J. Kim, and B. Rajaratnam, “Condit ion-number
regularized covariance estimation,” J. Roy. Statist. Soc. B , vol. 75, pp.
427-450, Jun. 2013.
[26] A. Kourtis, G. Dotsis, and R. N. Markellos, “Parameter u ncertainty in
portfolio selection: Shrinking the inverse covariance mat rix,” J. Bank.
Financ., vol. 36, no. 9, pp.2522-2531, 2012.
[27] M. Zhang, F. Rubio, and D. P . Palomar, “Improved calibra tion of high-
dimensional precision matrices,” IEEE Trans. Sig. Process. , vol. 61, no.
6, pp. 1509-1519, 2013.
[28] C. Wang, G. Pan, T. Tong, and L. Zhu, “Shrinkage estimati on of large
dimensional precision matrix using random matrix theory,” Statistica
Sinica, vol. 25, no. 3, pp. 993-1008, 2015.
[29] T. Ito and T. Kubokawa, “Linear ridge estimator of high- dimensional
precision matrix using random matrix theory,” Technical Report F-995,
CIRJE, Faculty of Economics, University of Tokyo, 2015.
[30] T. Bodnar, A. K. Gupta, and N. Parolya, “Direct shrinkag e estimation
of large dimensional precision matrix,” J. Multivar . Anal., vol. 146, pp.
223-236, 2016.
[31] X. Mestre and M. A. Lagunas, “Finite sample size effect o n minimum
variance beamformers: Optimum diagonal loading factor for large ar-
rays,” IEEE Trans. Sig. Process. , vol. 54, no. 1, pp. 69-82, 2006.
[32] C.-K. Wen, J.-C. Chen, and P . Ting, “A shrinkage linear m inimum
mean square error estimator,” IEEE Sig. Process. Lett., vol. 20, no. 12,
pp.1179-1182, 2013.
[33] J. Serra and M. N´ ajar, “Asymptotically optimal linear shrinkage of
sample LMMSE and MVDR ﬁlters,” IEEE Trans. Sig. Process. , vol.
62, no. 14, pp. 3552-3564, 2014.
[34] M. Zhang, F. Rubio, D. Palomar, and X. Mestre, “Finite-s ample linear
ﬁlter optimization in wireless communications and ﬁnancia l systems,”
IEEE Trans. Sig. Process., vol. 61, no. 20, pp. 5014-5025, 2013.
[35] J. Tong, P . J. Schreier, Q. Guo, S. Tong, J. Xi, and Y . Y u, “ Shrinkage of
covariance matrices for linear signal estimation using cro ss-validation,”
IEEE Trans. Sig. Process. , vol. 64, no. 11, pp. 2965-2975, 2016.
[36] J. Tong, Q. Guo, J. Xi, Y . Y u, and P . J. Schreier, “Choosin g the diagonal
loading factor for linear signal estimation using cross val idation,” in
Proc. IEEE ICASSP 2016 , pp. 3956-3959, 2016.
[37] J. R. Guerci and E. J. Baranoski, “Knowledge-aided adap tive radar at
DARPA: an overview,” IEEE Signal Process. Mag. , vol. 23, no. 1, pp.
41-50, Jan. 2006.
[38] S. Arlot and A. Celisse, “A survey of cross-validation p rocedures for
model selection,” Statist. Surv., vol. 4, pp. 40-79, 2010.
[39] G. H. Golub, M. Heath, and G. Wahba, “Generalized cross- validation
as a method for choosing a good ridge parameter,” Technometrics, vol.
21, no. 2, pp. 215-223, 1979.
[40] R. D. Nowak, “Optimal signal estimation using cross-va lidation,” IEEE
Sig. Process. Letters, vol. 4, no. 1, pp. 23-25, 1997.
[41] E. Bj¨ ornson and B. Ottersten, “A framework for trainin g-based esti-
mation in arbitrarily correlated Rician MIMO channels with Rician
disturbance,” IEEE Transactions on Signal Processing, vol. 58, no. 3,
pp. 18071820, March 2010.
[42] N. Shariati, E. Bj¨ ornson, M. Bengtsson and M. Debbah, “ Low-
complexity polynomial channel estimation in large-scale M IMO with
arbitrary statistics,” IEEE J. Sel. Topics Signal Process. , vol. 8, no. 5,
pp. 815-830, Oct. 2014.
[43] W. Weichselberger, M. Herdin, H. Ozcelik, and E. Bonek, “A stochastic
MIMO channel model with joint correlation of both link ends, ” IEEE
Trans. Wireless Commun., vol. 5, no. 1, pp. 90-100, 2006.
[44] J. Fang, X. Li, H. Li and F. Gao, “Low-rank covariance-as sisted
downlink training and channel estimation for FDD massive MI MO
systems,” IEEE Trans. Wireless Commun., vol. 16, no. 3, pp. 1935-1947,
March 2017.
[45] D. Tse and P . Viswanath, Fundamentals of Wireless Communications.
Cambridge, U.K.: Cambridge Univ. Press, 2005.
[46] N. Kim, Y . Lee and H. Park, “Performance analysis of MIMO system
with linear MMSE receiver,” IEEE Trans. Wireless Commun., vol. 7, no.
11, pp. 4474-4478, Nov. 2008.
[47] J. Tong, J. Xi, Q. Guo and Y . Y u, “Low-complexity cross-v alidation
design of a linear estimator,” Electronics Letters, vol. 53, no. 18, pp.
1252-1254, 2017.
