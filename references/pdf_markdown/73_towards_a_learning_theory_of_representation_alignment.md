# references/73_towards_a_learning_theory_of_representation_alignment.pdf

<!-- page 1 -->

arXiv:2502.14047v1  [cs.LG]  19 Feb 2025
Towards a Learning Theory of Representation
Alignment
Francesco Insulla * Shuo Huang † Lorenzo Rosasco ‡
Abstract
It has recently been argued that AI models’ representations are becoming aligned
as their scale and performance increase. Empirical analyse s have been designed to sup-
port this idea and conjecture the possible alignment of di ﬀerent representations toward
a shared statistical model of reality. In this paper, we prop ose a learning-theoretic per-
spective to representation alignment. First, we review and connect diﬀerent notions of
alignment based on metric, probabilistic, and spectral ideas. Then, we focus on stitching,
a particular approach to understanding the interplay betwe en diﬀerent representations in
the context of a task. Our main contribution here is relating properties of stitching to the
kernel alignment of the underlying representation. Our res ults can be seen as a ﬁrst step
toward casting representation alignment as a learning-the oretic problem.
1 Introduction
In recent years, as AI systems have grown in scale and perform ance, attention has moved
towards universal models that share architecture across mo dalities. Examples of such sys-
tems include CLIP ( Radford et al., 2021), VinVL (Zhang et al., 2021), FLA V A (Singh et al.,
2022), OpenAI’s GPT-4 ( OpenAI, 2023), and Google’s Gemini ( Google, 2023). These mod-
els are trained on diverse datasets containing both images a nd text and yield embeddings that
can be used for downstream tasks in either modality or for tas ks that require both modalities.
The emergence of this new class of multimodal models poses in teresting questions regarding
alignment and the trade-oﬀs between unimodal and multimodal modeling. While multimod al
models may provide access to greater scale through dataset size and computational eﬃciency,
*Institute of Computational and Mathematical Engineering S tanford University, Stanford, CA, USA.
Email:franinsu@stanford.edu
†Istituto Italiano di Tecnologia, Genoa, Italy. Email: shuo .huang@iit.it
‡MaLGa Center – DIBRIS – Universit` a di Genova, Genoa, Italy; also CBMM – Massachusets Institute of
Technology, USA; also Istituto Italiano di Tecnologia, Gen oa, Italy. Email: lrosasco@mit.edu
1

<!-- page 2 -->

how well do features learned from di ﬀerent modalities correspond to each other? How do we
mathematically quantify and evaluate this alignment and fe ature learning across modalities?
Regarding alignment, Huh et al. (2024) observed that as the scale and performance of
deep networks increase, the models’ representations tend t o align. They further conjectured
that the limiting representations accurately describe rea lity - known as Platonic represen-
tation hypothesis . Their analysis also suggests that alignment correlates wi th performance,
implying that improving the alignment of learned features a cross diﬀerent modalities could
enhance a model’s generalization ability. However, alignm ent across modalities has yet to
be evaluated in a more interpretable manner, and theoretica l guarantees of alignment under
realistic assumptions are still lacking.
Reality
(Ξ, ξ)
Objects/data
(X1, µ1)
Objects/data
(X2, µ2)
Representation
(Z1, λ1)
Representation
(Z2, λ2)
Output
(Y1, ν1)
Output
(Y2, ν2)
m1 m2
f1 f2
g1 g2
s1, 2
s2, 1
h1 h2
Figure 1: Diagram illustrating the process of multi-modal l earning. It contains spaces and
measures of reality, objects/data, representation, and outputs as well as the functions c onnect-
ing them. A detailed explanation of these symbols is in Secti on 2.
One way to quantify alignment is by kernel alignment, introd uced by Cristianini et al.
(2001), which evaluates the correlation of two kernel matrices K1, n, K2, n through Frobenius
norms
ˆA(K1, n, K2, n) = ⟨K1, n, K2, n⟩F
√
⟨K1, n, K1, n⟩F⟨K2, n, K2, n⟩F
.
Following this direction, methods like Centered Kernel Ali gnment (CKA) ( Kornblith et al.,
2019) and Singular V ector Canonical Correlation Analysis (SVCC A) ( Raghu et al. , 2017)
were developed to compare learned representations. Anothe r class of metrics is derived
2

<!-- page 3 -->

from independence testing, including the Hilbert-Schmidt Independence Criterion (HSIC)
(Gretton et al. , 2005a) and Mutual Information (MI) ( Hjelm et al. , 2019). However, further
research is needed to clarify the relationships among these methods and other frameworks
for assessing alignment.
To quantify the alignment of representation conditioned on a task, one approach is to
use the stitching method ( Lenc & V edaldi, 2015). Bansal et al. (2021) revisited this tech-
nique and used it to highlight that good models trained on di ﬀerent objectives (supervised
vs self-supervised) have similar representations. By eval uating how well one representation
integrates into another model, stitching provides a more in terpretable framework for assess-
ing alignment. To describe the setup, we use h1, 2 := g2 ◦s1, 2 ◦f1 to represent the function
after stitching from model 1 to model 2 (Figure 1 gives a detailed illustration of the whole
process). Here gq and fq are parts of model Hq : Xq → Y q with q = 1, 2, and s1, 2 is the
stitcher. We consider the generalization error after stitc hing between two models:
R(g2 ◦s1, 2 ◦f1) = E [ℓ(h1, 2(x), y)] .
We can use the risk of the stitched model in excess of the risk o f model 2
min
s1, 2
R(h1, 2) −min
h2∈H2
R(h2)
to quantify the impact of using di ﬀerent representations, ﬁxing g2.
In this paper, we aim to formalize and reﬁne some of these ques tions, and our contribu-
tions are summarized as follows:
(a) We compile di ﬀerent deﬁnitions of alignment from various communities, de monstrate
their connections, and give spectral interpretations.
• Starting from the empirical Kernel Alignment (KA), we refor mulate empirical KA
and population version of KA using feature /representation maps, operators in Repro-
ducing Kernel Hilbert Space (RKHS), and spectral interpret ation. In addition, we
discuss the statistical properties of KA.
• We integrate various notions of alignment, ranging from ker nel alignment in inde-
pendence testing and learning theory to measure and metric a lignment, and demon-
strate their relationships and correlations. This compreh ensive exploration provides a
deeper understanding for practical applications.
(b) We provide the generalization error bound of linear stit ching with the kernel alignment
of the underlying representation.
• A linear gq results in the stitching error being equivalent to the risk f rom the model
Hq. This occurs, for example, when Hq represents RKHSs or neural networks, then
gq is a linear combination of features in RKHSs or the output lin ear layers of neural
networks.
3

<!-- page 4 -->

• The excess stitching risk can be bounded by kernel alignment when gq are nonlin-
ear functions with the Lipschitz property. A typical scenar io is stitching across the
intermediate layers of neural networks.
• For models involving several compositions such as deep netw orks, if we stitch from
a layer further from the output to a layer closer to the output (stitching forward) and
gq is Lipschitz, the di ﬀerence in risk can be bounded by stitching.
Structure of the paper In the following of this paper, we introduce the problem sett ings
and some notations in Section 2. Di ﬀerent deﬁnitions for representation alignment from
diﬀerent communities and the relationship among them will be de rived in Section 3. Section
4 demonstrates that the stitching error could be bounded by th e kernel alignment metric. And
the conclusion is given in Section 5.
2 Preliminaries
Empirical results demonstrate that well-aligned features signiﬁcantly enhance task perfor-
mance. However, there is a pressing need for more rigorous ma thematical tools to formalize
and quantify these concepts in uni /multi-modal settings. In this section, we provide a math-
ematical formalization of uni / multi-modal learning, introducing key notations to facili tate a
deeper understanding of the underlying processes.
Setup Without loss of generality, we focus on the case of two modali ties, as illustrated in
Figure
1, which outlines the corresponding process. For q = 1, 2, let ( Xq, µq) and ( Zq, λq)
be probability spaces, and let Fq be the space of functions fq : Xq → Z q = Rdq. We regard
Xq as the space of objects (or data), Fq as the space of representation (or embedding) maps,
and Zq as the space of representations. We relate µq and λq by assuming λq = ( fq)#µq1.
We also assume that µ1 and µ2 are the marginals of a joint probability space ( X, µ) with
X = X1 × X2, µ q = (πq)#µ, where πq : X → X q is the projection map. Moreover, let
(Yq, νq) be the task-based output spaces and deﬁne Gq = {gq : Zq → Yq}with νq = (gq)#λq.
Each overall model is generated by Hq := {hq : Xq → Yq |hq = gq ◦fq}.
Reality Consider a space of abstract objects, called the reality space and denoted by Ξ,
which generates the observed data in various modalities thr ough maps mq : Ξ → X q. These
maps may be bijective, lossy, or stochastic. Reality can be m odeled as a probability space
(Ξ, ξ). Alternatively, one may deﬁne reality as the joint distribution over modalities by setting
mq = πq.
1( fq)#µq is the pushforward measure of µq deﬁned as ( fq)#µq(A) = µq( f −1
q (A)). In terms of random variables
Xq and Zq with measures µq and λq, respectively, this is equivalent to fq(Xq) and Zq being equal in law.
4

<!-- page 5 -->

Uni/Multi-modal We may want to consider the case of a single modality, where on ly one
data space exists, versus multiple modalities, where sever al such spaces are present. Two
modalities are deemed equal if π1 = π2.
Representation alignment A representation mapping is a function f : X → Rd that as-
signs a latent feature vector in Rd to each input in the data domain X. Alignment provides a
metric to evaluate how well the latent feature spaces obtain ed from di ﬀerent representation
mappings, whether from uni-modal or multi-modal data, are a ligned or similar. Commonly
used metrics for measuring alignment include those derived from kernel alignment, con-
trastive learning, mutual information, canonical correla tion analysis, and cross-modal mech-
anisms, among others. However, they are introduced in a very fragmented manner, without
an integrated or uniﬁed concept. A detailed introduction an d analysis of these methods will
be provided in Section 3.
3 Frameworks for Representation Alignment
In this section, we describe various deﬁnitions of represen tation alignment from di ﬀerent
communities and demonstrate the relationship among them. W e begin with a detailed presen-
tation of empirical and population Kernel Alignment and its statistical properties. We then
cover other notions of alignment coming from metrics, indep endence testing, and probability
measures, as well as their spectral interpretations. We dra w connections to kernel alignment
which emerges as a central object.
3.1 Kernel alignment (KA)
Based on the work of Cristianini et al. (2001), who introduced the deﬁnition of kernel
alignment using empirical kernel matrices, we propose di ﬀerent perspectives to understand
kernel alignment in both empirical and population settings and derive its statistical properties
accordingly.
A reproducing positive deﬁnite kernel K : X × X → R captures the notion of similarity
between objects by inducing an inner product in the associat ed reproducing kernel Hilbert
space (RKHS) H. Speciﬁcally, K(x, x′) = ⟨ f (x), f (x′)⟩ for any representation (feature) map
f ∈ H, and x, x′∈ X. For the multi-modal case, we deﬁne Kq(x, x′) : = ˜Kq(πq(x), πq(x′)) =
˜Kq(xq, x′
q), where ˜Kq is the reproducing kernel associated with Hq. In other words, Kq acts
on x = (x1, x2) by ﬁrst applying the projection πq(x) = xq. In the following, the subscript xq
denotes the q-th modality, and the superscript xi indicates the i-th sample.
3.1.1 Empirical KA
From
Cristianini et al. (2001), we adopt the following formulation for kernel alignment
for kernel matrix Kq, n ∈Rn×n with samples {xi}n
i=1 drawing according to the probability mea-
5

<!-- page 6 -->

sure µ
ˆA(K1, n, K2, n) = ⟨K1, n, K2, n⟩F
√
⟨K1, n, K1, n⟩F⟨K2, n, K2, n⟩F
,
where ⟨K1, n, K2, n⟩F = ∑n
i, j=1 K1, n(xi, x j)K2, n(xi, x j). One modiﬁcation is to ﬁrst demean the
kernel by applying a matrix H = In −1
n 1n1T
n on the left and right of each Kq, n with I ∈Rn×n
being the identity matrix and 1 n being the ones vectors. This results in Centered Kernel
Alignment (CKA).
Representation interpretation of KA Denote the empirical cross-covariance matrix be-
tween the representation maps f1 and f2 as ˆΣ1, 2 = En
[
f1(x) f2(x)T
]
= 1
n
∑n
i=1 f1(xi) f2(xi)T ∈
Rd1×d2. Then the empirical KA will become
ˆA(K1, n, K2, n) = ∥ˆΣ1, 2∥2
F
∥ˆΣ1, 1∥F∥ˆΣ2, 2∥F
. (1)
RKHS operator interpretation of KA Inspired by the equation 1, we construct a con-
sistent deﬁnition of Kernel Alignment using tools of RKHS, w here it su ﬃces to consider
output in one dimension 2. Consider RKHS Hq containing functions hq : Xq → R with kernel
Kq. Given evaluation (sampling) operators ˆS q : Hq → Rn deﬁned by ( ˆ
S qhq)i = hq(xi
q) =
⟨hq, Kq, xi
q ⟩. It is not hard to check that the adjoint operator ˆS ∗
q : Rn → H q can be written as
ˆ
S ∗
q(w1, . . . , wn) = ∑n
i=1 wiKq(xi
q, ·) and the empirical kernels can be written as Kq, n/ n = ˆS qˆS ∗
q
(
Smale & Zhou , 2004; De Vito et al., 2005). Then the empirical KA may be written as
ˆA(K1, n, K2, n) = ⟨ˆ
S 1ˆ
S ∗
1, ˆS 2ˆ
S ∗
2⟩F
∥ˆ
S 1ˆ
S ∗
1∥F∥ˆ
S 2ˆ
S ∗
2∥F
= ∥ˆ
S ∗
1ˆ
S 2∥2
F
∥ˆ
S ∗
1ˆ
S 1∥F∥ˆ
S ∗
2ˆS 2∥F
,
where ˆ
S ∗
1ˆ
S 2 = 1
n
∑
i K1, xi
1
⊗K2, xi
2
and it coincides with the literature about learning theory w ith
RKHS.
3.1.2 Population version of KA
For the population setting (inﬁnite data limit of the evalua tion operator) in L2, the re-
striction operator S q : Hq → L2(Xq, µ) is deﬁned by S qhq(x) = ⟨hq, Kq(x, ·)⟩Kq and its ad-
joint S ∗
q : L2(Xq, µ) → H q is given by S ∗
qg =
∫
X g(x)Kq(x, ·)dx. Then the integral operator
LKq = S qS ∗
q : L2(Xq, µ) → L2(Xq, µ) is given by LKqg(x) =
∫
X Kq(x, x′)g(x′)dµ(x′) and the op-
erator Σq = S ∗
qS q : Hq → Hq can be written as Σq =
∫
X Kq(x, ·) ⊗Kq(x, ·)dµ(x) (
De Vito et al.,
2We can generalize the deﬁnition to vector-valued functions by recasting hq : Xq → Rtq as hq : Xq ×[tq] → R
i.e. with kernels of the form Kq(xq, i, x′
q, i′) for integers 1 ≤i, i′≤tq.
6

<!-- page 7 -->

2005; Rosasco et al. , 2010). Similarly, the population KA between two kernels K1, K2 can be
deﬁned by
A(K1, K2) = Tr ( LK1 LK2
)
√
Tr
(
L2
K1
)
Tr
(
L2
K2
) ,
where the summation in ⟨K1, n, K2, n⟩F becomes the integration as
Tr ( LK1 LK2
) =
∫
dµ(x1, x2)dµ(x′
1, x′
2)K1(x1, x′
1)K2(x2, x′
2).
If Kq(x, x′) = ⟨ fq(x), fq(x′)⟩, then S ∗
qS q is a projection onto the span of coordinates of fq. The
population version of CKA is KA with S q replaced with HS q.
Spectral Interpretation of KA Furthermore, the understanding of kernel alignment (KA)
can be deepened via the spectral decomposition of the associ ated integral operator. The
mercer kernel K can be decomposed as K = ∑
i ηi φi ⊗φi, where ηi are the eigenvalues and φi
are the eigenfunctions of the integral operator LK (
Cucker & Smale, 2002; Sch¨ olkopf, 2002).
Deﬁning the features as fi = √ηiφi and expressing the target function as h = ∑
i wi fi, we
obtain
A(K, h ⊗h) =
∑
i η2
i w2
i
√∑
i η2
i
∑
i ηiw2
i
.
Similarly, given two kernels Kq = ∑
i ηq, iφq, i ⊗φq, i with fq, i = √
ηq, iφq, i, we have
A(K1, K2) =
∑
i, j⟨ f1, i, f2, j⟩
√∑
i η2
1, i
∑
i η2
2, i
=
∑
i, j η1, iη2, j⟨φ1, i, φ2, j⟩2
√∑
i η2
1, i
∑
i η2
2, i
.
Letting [C1, 2]i, j = ⟨φ1, i, φ2, j⟩ and deﬁning ˆηi = ηi/ ∥ηi∥, we can equivalently write
A(K1, K2) = Tr
[
C1, 2 diag( ˆη2) CT
1, 2 diag( ˆη1)
]
= ⟨ˆη1, (C1, 2 ⊙C1, 2) ˆη2⟩ = ⟨ˆη1 ˆηT
2 , C1, 2 ⊙C1, 2⟩
with ⊙as the Hadamard product. This formulation provides insight into kernel alignment
by relating it to the similarity between the eigenfunctions of the two integral operators.
In particular, if η1 and η2 are constant, then A(K1, K2) ∝ ∥C1, 2∥2; and if C1, 2 = I, then
A(K1, K2) = ⟨ˆη1, ˆη2⟩.
3.1.3 Statistical properties of KA.
Having introduced both the empirical and population versio ns of KA, we now explore
its statistical properties.
Cristianini et al. (2006) shows that empirical KA concentrates to its
expectation by McDiarmid’s inequality and gives an O(1/ √n) bound. For completeness, we
state the following lemma summarizing this statistical pro perty and the proof is provided in
Appendix 6.3.
7

<!-- page 8 -->

Lemma 1. Let K 1, K2 be two kernels for di ﬀerent representations and ˆK1, n, ˆK2, n ∈Rn×n be
kernel matrices generated by n samples, then with probabili ty at least 1 −δ, we have
ˆA(K1, n, K2, n) −A(K1, K2) ≤
√
(32/ n) log(2/δ).
3.2 Alignment from distance alignment
Distance alignment (DA) Given distances dq : Xq × Xq → R, then we can compare the
diﬀerence of two spaces by
D(d1, d2) =
∫
(d2
1(x, x′) −d2
2(x, x′))2dµ(x)dµ(x′).
Equivalence between KA and DA Suppose d2
q = 2(1−Kq) (
Igel et al. , 2007) and Kq(xq, xq) =
1, which emerge naturally from assuming Kq(x, x′) = ⟨ fq(x), fq(x′)⟩, ∥ fq(x)∥ = 1, and d2
q(xq, x′
q) =
∥ fq(xq) −fq(x′
q)∥2, (i.e., Kq represents a mapping onto a ball). Also assume ∥Kq∥ = C. Then,
D(d1, d2) = 8C(1 −A(K1, K2)), hence the two paradigms are equivalent.
3.3 Alignment from independence testing
Independence testing is a statistical method used to assess the degree of dependence be-
tween variables. It often involves examining the covariance and correlations between random
variables and can also be applied to quantify kernel-based i ndependence. In this section, we
outline several approaches from independence testing with in the alignment framework and
investigate their connections to the kernel alignment meth od discussed earlier.
Hilbert-Schmidt Independence Criterion (HSIC) The cross-covariance operator for two
functions (
Baker, 1973) is given byC1, 2[h1, h2] = Ex1, x2[(h1(x1)−Ex1 (h1(x1))(h2(x2)−Ex2 (h2(x2))]
for h1 ∈ H1, h2 ∈ H2. From Gretton et al. (2005a)
HSIC(µ,H1, H2) = ∥C1, 2∥2
HS ,
where µis the joint distribution of X1 and X2. We can also note that
HSIC(µ,H1, H2) = ∥E [Kx1 ⊗Kx2
] ∥2 = ∥Σ1, 2∥2
HS .
Hence HSIC is e ﬀectively and unnormalized version of CKA, or, more explicit ly,
CKA(K1, K2) = HSIC(H1, H2)√
HSIC(H1, H1)HSIC(H2, H2)
.
8

<!-- page 9 -->

Statistical property of HSIC Gretton et al. (2005b) shows that, excluding the O(n−1) di-
agonal bias, centered empirical HSIC concentrates to popul ation and Song et al. (2012) pro-
vides an unbiased estimator of HSIC and shows its concentrat ion, both by U-statistic argu-
ments.
Remark 1 (Other notions from independence testing) . There are other concepts of indepen-
dence testing for alignment such us Constrained Covariance (COCO) (Gretton et al. , 2005a),
Kernel Canonical Correlation (KCC), Kernel Mutual Informa tion (KMI) ( Bach & Jordan,
2002). They are also related to kernel alignment and more detaile d explanations can be
found in Appendix 6.2.
3.4 Alignment from measure alignment
There are several methods for comparing measures on the same space. One can then
quantify independence by comparing a joint measure with the product of its marginals. This
principle allows us to interpret HSIC as test for independen ce given two function classes.
MMD to HSIC Following Gretton et al. (2012), we start by introducing the so-called Max-
imum Mean Discrepancy (MMD). Let H be a class of functions h : X → R and let µq be
diﬀerent measures on X. Then, letting xq ∼µq,
MMD(µ1, µ2; H) = sup
h∈H
E [h(x1) −h(x2)] .
Let H be an RKHS and restrict to a ball of radius 1, then
MMD(µ1, µ2; H)2 = ∥E [Kx1 −Kx2
] ∥2
H = E [K(x1, x′
1) + K(x2, x′
2) −2K(x1, x2)] .
Now we construct a measure of independence by applying MMD on µversus µ1 ⊗µ2 where
H is replaced with H1 × H2 and get HSIC
MMD(µ, µ1 ⊗µ2; H1 ⊗ H2)2 = HSIC(µ,H1, H2) = ∥Σ1, 2∥2 =
∑
i
ρ2
i
where
{
ρ2
i
}
is the spectrum of Σ1, 2Σ2, 1.
We can also use tests of independence that don’t explicitly d epend on a function class,
such as mutual information, by letting µbe a Gaussian Process measure on two functions in
their respective RKHS with covariance deﬁned by their kerne ls.
KL Divergence to Mutual Information Given KL divergence
DKL(µ||ν) =
∫
dµ(x) log
( dµ
dν(x)
)
,
9

<!-- page 10 -->

we can deﬁne mutual information as
I(µ) = DKL(µ||µ1 ⊗µ2) =
∫
dµ(x1, x2) log
( µ(x1, x2)
µ1(x1)µ2(x2)
)
=
∫
dµ(x1, x2) log
( µ(x2|x1)
µ2(x2)
)
.
For multivariate Gaussian µ, with marginals µq = N(0, Σq),
MI(ν) = 1
2 log
( |Σ1||Σ2|
|Σ|
)
= 1
2 log
( |Σ2|
|Σ2 −Σ2, 1Σ−1
1 Σ1, 2|
)
.
For the simplest case of Σq = I, then this simpliﬁes to
MI(ν) = −1
2 log(|I −Σ1, 2Σ2, 1|) = −1
2
∑
i
log(1 −ρ2
i ).
Wasserstein distance For the Wasserstein distance
W2(µ, ν) = inf{E(x, y)∼γ
[
∥x −y∥2]
: γ1 = µ, γ2 = ν},
applying µand µ1 ⊗µ2 to measure independence, we have
W2(µ, µ1 ⊗µ2) = inf{E((x1, x2), (x′
1, x′
2))∼γ
[
∥x1 −x′
1∥2 + ∥x2 −x′
2∥2]
: γ1 = µ, γ2 = µ1 ⊗µ2}.
For mean zero Gaussians
W2(µ1, µ2) = Tr[Σ1 + Σ2 −2(Σ1/ 2
1 Σ2Σ1/ 2
1 )1/ 2]
and as a measure of independence with Σq = I
W2(µ, µ1 ⊗µ2) = 2Tr[I −(I −Σ1, 2Σ2, 1)1/ 2] = 2
∑
i
(
1 −
√
1 −ρ2
i
)
.
In summary, we’ve introduced several popular metrics for al ignment between two repre-
sentations and related them via spectral decompositions to a central notion of kernel align-
ment generalized for RKHS. Moreover, similar notions can be used to quantify alignment
between a model and a task to estimate generalization error, and more details are provided in
the Appendix
6.1.
4 Stitching: Task Aware Representation Alignment
Building on our understanding of kernel alignment—a fundam ental metric for evaluating
the alignment of representations detailed in the previous s ection—we now explore stitching,
a task-aware concept of alignment. Stitching involves comb ining layers or components from
various models to create a new model which can be used to under stand of how diﬀerent parts
contribute to overall performance or to compare the learned features for a task. In this section,
we mathematically formulate this process and provide some i ntuition by demonstrating that
the generalization error after stitching can be bounded by k ernel alignment using spectral
arguments.
10

<!-- page 11 -->

4.1 Stitching error between models
In the following, we focus on stitching between two modaliti es. Figure 1 provides a de-
tailed illustration of the functions, spaces, and composit ions in question. Denote the function
space for task learning as Hq := {hq : Xq → Yq|hq = gq ◦fq, gq ∈ Gq, fq ∈ Fq}with q = 1, 2.
Here Fq : Xq → Z q and Gq : Zq → Y q. Denote S1, 2 := {s1, 2 : Z1 → Z 2}as the stitching
map from Z1 to Z2 and S2, 1 := {s2, 1 : Z2 → Z 1}reversely. Deﬁne the risk concerning the
least squares loss as
Rq(hq) = E
[
∥hq(x) −y∥2]
=
∫
Xq×Yq
∥hq(x) −y∥2dρq(x, y), hq ∈ Hq.
Here, ρq(x, y) is the joint distribution of Xq and Yq and we use the notation ∥ · ∥to represent
∥ · ∥Yq associated with space Yq for simplicity, i.e. absolute value for Yq = R, l2 norm for
Yq = Rtq and L2 norm for Yq being the function space. For hq ∈ Hq, denote any minimizer
of R(hq) among Hq as h∗
q, that is,
Rq(Hq) := Rq(h∗
q) = min
h∈Hq
Rq(h), q = 1, 2.
Moreover, denote the function spaces generated after stitc hing from Z1 to Z2 as
H1, 2 = {h1, 2 = g2 ◦s1, 2 ◦f1 : s1, 2 ∈ S1, 2}
and conversely as H2, 1.
Lenc & V edaldi(2015) proposed to describe the similarity between two represent ations
by measuring how usable a representation f1 is when stitching with g2 through a function
s1, 2 : Z1 → Z 2 or oppositely through s2, 1 ∈ S2, 1. To quantify the similarity, we provide a
detailed deﬁnition of the stitching error.
Stitching error Deﬁne the stitching error as
Rstitch
1, 2 (s1, 2) := R2(g2 ◦s1, 2 ◦f1) = R2(h1, 2)
and the minimum as
Rstitch
1, 2 (S1, 2) := min
s1, 2∈S1, 2
R2(h1, 2) = R2(H1, 2).
To quantify the diﬀerence in the use of stitching, we deﬁne the excess stitching risk as
Rstitch
1, 2 (S1, 2) − R2(H2).
Note that Rstitch
1, 2 (S1, 2) − R2(H2) quantiﬁes a di ﬀerence in use of representation (ﬁx g2,
compare s1, 2 ◦f1 vs f2), while if Y1 = Y2 then Rstitch
1, 2 (S1, 2) − R1(H1) quantiﬁes di ﬀerence
between g2 ◦s1, 2 and g1 (ﬁx f1).
11

<!-- page 12 -->

The functions in S1, 2 are typically simple maps such as linear layers or convoluti ons of
size one, to avoid introducing any learning, as emphasized i n Bansal et al. (2021). The aim is
to measure the compatibility of two given representations w ithout ﬁtting a representation to
another. One perspective inspired by Lenc & V edaldi(2015) is that we should not penalize
certain symmetries, such as rotations, scaling, or transla tions, which do not alter the informa-
tion content of the representations. Furthermore, the amou nt of unwanted learning may be
quantiﬁed by stitching from a randomly initialized network .
4.2 Stitching error bounds with kernel alignment
In this section, we focus on a simpliﬁed setting where s1, 2 : Z1 → Z 2 is a linear
stitching, that is, s1, 2(z1) = S 1, 2z1 with S 1, 2 ∈Rd2×d1, zq ∈Rdq. Additionally, we assume
Y1 = Rt1, Y2 = Rt2. In this section, we quantify the stitching error and excess stitching
risk using kernel alignment and provide a lower bound for the stitching error when stitching
forward.
The following lemma shows that when Gq are linear, stitching error only measures the
diﬀerence in risk of H1 versus H2.
Lemma 2. Suppose dim (Y1) = dim(Y2) = d and R1 = R2. Let g q ∈ Gq be linear with
gq(zq) = Wqzq and W q ∈Rd×dq. Let s 1, 2 : Z1 → Z 2 be linear with s 1, 2(z1) = S 1, 2z1 and
S 1, 2 ∈Rd2×d1. Then Rstitch
1, 2 (S1, 2) = R1(H1).
Remark 2. The lemma applies when Hq represents a neural network with Gq as the output
linear layer, as well as when Hq is an RKHS with a Mercer kernel and Gq is the linear map
of representations 3.
The next theorem shows the case when Gq are nonlinear with the κ-Lipschitz property,
∥g(z)−g(z′)∥ ≤κ∥z−z′∥. One intermediate example is the stitching between the midd le layers
of neural networks.
Theorem 1. Suppose g 2 is κ2-Lipschitz. Again let s 1, 2 be linear , identiﬁed with matrix S 1, 2.
With the spectral interpretations of Σ1, 2 = E
[
f1 f T
2
]
= diag(η1)1/ 2C1, 2diag(η2)1/ 2 and ˜A2 =
∥I∥η2 − ∥C1, 2∥2
η2 as Paragraph
3.1.2, we have
Rstitch
1, 2 (S1, 2) ≤ R2(H2) + κ2
2 ˜A2 + 2κ2( ˜A2R2(H2))1/ 2. (2)
3More explicitly, if the RKHS kernel Kq is a sum of separable kernels, then by Mercer’s theorem we can
decompose it as Kq = ∑dq
ρ=1 ηq,ρφq,ρ ⊗φq,ρ where ηq,ρ ≥0 are the eigenvalues, and φq,ρ : RDq → Rdq are the
orthonormal eigenfunctions of the integral operator assoc iated with the kernel Kq. Then any hq ∈ Hq can be
decomposed as hq = gq ◦fq, where fq ∈ Fq is the feature map fq(Xq)ρ= √ηρφq,ρ(Xq) and gq ∈ Gq is linear
gq(zq) = wq ·zq.
12

<!-- page 13 -->

Proof. Breaking Rstitch
1, 2 (s1, 2) into two parts and using Cauchy-Schwarz we get
E
[
∥g2(S 1, 2 f1)(x) −y∥2]
=E
[
∥(g2(S 1, 2 f1)(x) −g2( f2)(x)) −(y −g2( f2)(x))∥2]
≤R2(h2) + E
[
∥g2(S 1, 2 f1)(x) −g2( f2)(x)∥2]
+ 2(E
[
∥g2(S 1, 2 f1)(x) −g2( f2)(x)∥2]
R2(h2))1/ 2.
As g2 is κ2-Lipschitz, we can bound with the error from linearly regres sing f2 on f1
E
[

g2(S 1, 2 f1)(x) −g2( f2)(x)



2
]
≤κ2
2E
[
∥S 1, 2 f1(x) −f2(x)∥2]
= κ2
2(∥S 1, 2∥2
η1 + ∥I∥2
η2 −2⟨S 1, 2, ΣT
1, 2⟩)
with ∥M∥2
η= ⟨M, Mdiag(η)⟩. Taking derivatives, we note that the minimizer of the RHS is
S 1, 2 = Σ T
1, 2diag(η1)−1. Plugging in, the RHS reduces to κ2
2 ˜A2. Thus
Rstitch
1, 2 (S1, 2) ≤ Rstitch
1, 2 (Σ1, 2)
≤ R2(H2) + κ2
2 ˜A2 + 2κ2( ˜A2R2(H2))1/ 2.
□
Remark 3. In arguing that kernel alignment bounds stitching error for Theorem
1, we made
several simplifying assumptions, which we now assess. Firs tly, we restricted the stitching
S1, 2 to linear maps, following the transformations commonly use d in practice ( Bansal et al. ,
2021; Csisz´ arik et al., 2021), and to preserve the signiﬁcance of the original represent ations.
If we relax this assumption, we observe that a similar resultholds, with ˜A2 = inf s1, 2∈S1, 2 E[∥s1, 2( f1(x))−
f2(x)∥2]. Interestingly, for s1, 2 to use only information about the covariance of f1, f2, similarly
to kernel alignment, s1, 2 must be linear. Furthermore, we note that for stitching clas ses that
include all linear maps, the linear result remains valid.
Remark 4. Note that the notion of alignment that appears here, namely ∥I∥2
η2 −˜A2 = ∥C1, 2∥2
η2 =
∥C1, 2diag(η2)∥2, is similar to, yet distinct from, kernel alignment, which i s given by ∥Σ1, 2∥2 =
∥diag(η1)1/ 2C1, 2diag(η2)1/ 2∥2. In particular, the spectrum η1 is irrelevant for the bound. How-
ever, this does not hold if regularization is added to S 1, 2 by analogy to linear regression.
Remark 5. If two representations are similar in the alignment sense, t hey are also similar in
the stitching sense; however, the converse does not necessa rily hold. By loose analogy to
topology, this suggests that kernel alignment is a stronger notion of similarity.
Excess stitching risk can also serve as an intermediate resu lt to bound the di ﬀerence in
risk. Let Y1 = Y2 and R1 = R2. To obtain a lower bound for Rstitch
1, 2 (S1, 2) in a practical setting,
we can assume that S1, 2 ◦ G2 ⊆ G1. For models involving several compositions, such as deep
networks, this condition can hold when stitching from a laye r further from the output to a
layer closer to the output (i.e., stitching forward), provi ded that the networks are similar and
the layer indices are aligned at the end.
13

<!-- page 14 -->

Lemma 3. Let Y1 = Y2 = Y and R1 = R2 = R. If S1, 2 ◦ G2 ⊆ G1 then Rstitch
1, 2 (S1, 2) ≥ R1(H1).
The following theorem derives directly from equation 2 and Lemma 3.
Theorem 2. Let Y1 = Y2 and R1 = R2 = R. Let S1, 2 ◦ G2 ⊆ G1 and let g q be κq-Lipschitz
for q = 1, 2. Then
R(H1) − R(H2) ≤ Rstitch
1, 2 (S1, 2) − R(H2) ≤κ2
2 ˜A2 + 2κ2( ˜A2R(H2))1/ 2.
Remark 6. If we consider deep models and keep the H1, H2 the same but iterate over layers
j stitching forward, then
R(H1) − R(H2) ≤min
j
{
(κ( j)
2 )2 ˜A( j)
2 + 2κ( j)
2 ( ˜A( j)
2 R(H2))1/ 2}
.
Alternatively, by making similar assumptions and swapping the index 1 ↔ 2, which
requires G1 = G2 up to a linear layer (due to the S1, 2 ◦ G2 ⊆ G1 condition), we get
|R(H1) − R(H2)| ≤max
i∈{1, 2}
{
κ2
i ˜Ai + 2κi( ˜AiR(Hq))1/ 2}
.
The above result can be stated informally as “alignment at si milar depth (measured back-
ward from the output) bounds di ﬀerences in risk”.
The results presented have several practical implications . First, we build on the experi-
ments from Huh et al. (2024), which provide evidence for the alignment of deep networks at
a large scale using measures similar to kernel alignment. By establishing a connection be-
tween kernel alignment and stitching, our work supports bui lding universal models that share
architectures across modalities as scale increases. Secon d, we can elucidate the experiments
from Bansal et al. (2021), which suggest that typical SGD minima have low stitching c osts
(stitching connectivity). This aligns with works that argu e feature learning under SGD can be
understood through the lens of adaptive kernels ( Radhakrishnan et al., 2022; Atanasov et al.,
2022).
5 Conclusion
In this paper, we review and unify several representation al ignment metrics, including
kernel alignment, distance alignment, and independence te sting, demonstrating their equiva-
lence and interrelationships. Additionally, we formalize the concept of stitching, a technique
used in uni /multi-modal settings to quantify alignment in relation to a given task. Further-
more, we establish bounds on stitching error across di ﬀerent modalities and derive stitching
error bounds based on misalignment, along with their genera lizations and implications.
14

<!-- page 15 -->

Acknowledgment
L.R. is thankful to the CBMM-Hebrew University workshop org anizers and Philipp Isola
for the talk that inspired this work. L. R. acknowledges the ﬁ nancial support of the European
Research Council (grant SLING 819789), the European Commis sion (Horizon Europe grant
ELIAS 101120237), the US Air Force Oﬃce of Scientiﬁc Research (FA8655-22-1-7034), the
Ministry of Education, University and Research (FARE grant ML4IP R205T7J2KP; grant
BAC FAIR PE00000013 funded by the EU - NGEU). This work repres ents only the view of
the authors. The European Commission and the other organiza tions are not responsible for
any use that may be made of the information it contains.
References
Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learn-
ers: The silent alignment e ﬀect. In International Conference on Learning Representations,
2022.
Francis R Bach and Michael I Jordan. Kernel independent comp onent analysis. Journal of
Machine Learning Research, 3(Jul):1–48, 2002.
Charles R Baker. Joint measures and cross-covariance opera tors. Transactions of the Ameri-
can Mathematical Society , 186:273–289, 1973.
Y amini Bansal, Preetum Nakkiran, and Boaz Barak. Revisitin g model stitching to compare
neural representations. Advances in Neural Information Processing Systems , 34:225–236,
2021.
Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. S pectral bias and task-model
alignment explain generalization in kernel regression and inﬁnitely wide neural networks.
Nature Communications, 12(1):2914, 2021.
Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Al gorithms for learning kernels
based on centered alignment. Journal of Machine Learning Research , 13:795–828, 2012.
Peter Craven and Grace Wahba. Smoothing noisy data with spli ne functions: estimating the
correct degree of smoothing by the method of generalized cro ss-validation. Numerische
mathematik, 31(4):377–403, 1978.
Nello Cristianini, John Shawe-Taylor, Andre Elissee ﬀ, and Jaz Kandola. On kernel-target
alignment. Advances in Neural Information Processing Systems , 14, 2001.
Nello Cristianini, Jaz Kandola, Andre Elissee ﬀ, and John Shawe-Taylor. On Kernel Target
Alignment, pp. 205–256. Springer Berlin Heidelberg, 2006.
15

<!-- page 16 -->

Adri´ an Csisz´ arik, P´ eter K˝ or¨ osi-Szab´ o, Akos Matszangosz, Gergely Papp, and D´ aniel V arga.
Similarity and matching of neural network representations . Advances in Neural Informa-
tion Processing Systems, 34:5656–5668, 2021.
Felipe Cucker and Steve Smale. On the mathematical foundati ons of learning. Bulletin of
the American mathematical society , 39(1):1–49, 2002.
Ernesto De Vito, Lorenzo Rosasco, Andrea Caponnetto, Umber to De Giovannini, Francesca
Odone, and Peter Bartlett. Learning from examples as an inve rse problem. Journal of
Machine Learning Research, 6(5), 2005.
Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul , Sepideh Kharaghani, Daniel M
Roy, and Surya Ganguli. Deep learning versus kernel learnin g: an empirical study of
loss landscape geometry and the time evolution of the neural tangent kernel. Advances in
Neural Information Processing Systems , 33:5850–5861, 2020.
Gene H Golub, Michael Heath, and Grace Wahba. Generalized cr oss-validation as a method
for choosing a good ridge parameter. Technometrics, 21(2):215–223, 1979.
Google. Gemini: a family of highly capable multimodal model s. arXiv preprint
arXiv:2312.11805, 2023.
Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Sch¨ olkopf. Measuring statisti-
cal dependence with hilbert-schmidt norms. In International Conference on Algorithmic
Learning Theory, pp. 63–77. Springer, 2005a.
Arthur Gretton, Ralf Herbrich, Alexander Smola, Olivier Bo usquet, Bernhard Sch¨ olkopf,
and Aapo Hyv¨ arinen. Kernel methods for measuring independ ence. Journal of Machine
Learning Research, 6(12), 2005b.
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhar d Sch¨ olkopf, and Alexander
Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(1):723–773,
2012.
R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Kar an Grewal, Phil Bachman,
Adam Trischler, and Y oshua Bengio. Learning deep representations by mutual information
estimation and maximization. In International Conference on Learning Representations ,
2019.
Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isol a. Position: The platonic
representation hypothesis. In F orty-ﬁrst International Conference on Machine Learning ,
2024.
16

<!-- page 17 -->

Christian Igel, Tobias Glasmachers, Britta Mersch, Nico Pf eifer, and Peter Meinicke.
Gradient-based optimization of kernel-target alignment f or sequence kernels applied to
bacterial gene start detection. IEEE/ACM Transactions on Computational Biology and
Bioinformatics, 4(2):216–226, 2007.
Arthur Jacot, Berﬁn Simsek, Francesco Spadaro, Cl´ ement Hongler, and Franck Gabriel. Ker-
nel alignment risk estimator: Risk prediction from trainin g data. Advances in Neural
Information Processing Systems, 33:15568–15578, 2020.
Dmitry Kopitkov and V adim Indelman. Neural spectrum alignm ent: Empirical study. In
Artiﬁcial Neural Networks and Machine Learning–ICANN 2020 : 29th International Con-
ference on Artiﬁcial Neural Networks, Bratislava, Slovaki a, September 15–18, 2020, Pro-
ceedings, Part II 29, pp. 168–179. Springer, 2020.
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geo ﬀrey Hinton. Similarity of
neural network representations revisited. In International Conference on Machine Learn-
ing, pp. 3519–3529. PMLR, 2019.
Karel Lenc and Andrea V edaldi. Understanding image represe ntations by measuring their
equivariance and equivalence. In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pp. 991–999, 2015.
OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
Jonas Paccolat, Leonardo Petrini, Mario Geiger, Kevin Tylo o, and Matthieu Wyart. Geomet-
ric compression of invariant manifolds in neural networks. Journal of Statistical Mechan-
ics: Theory and Experiment , 2021(4):044001, 2021.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, G abriel Goh, Sandhini Agar-
wal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Cla rk, et al. Learning trans-
ferable visual models from natural language supervision. I n International conference on
machine learning, pp. 8748–8763. PMLR, 2021.
Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parth e Pandit, and Mikhail Belkin.
Mechanism of feature learning in deep fully connected netwo rks and kernel machines
that recursively learn features. arXiv preprint arXiv:2212.13881, 2022.
Maithra Raghu, Justin Gilmer, Jason Y osinski, and Jascha So hl-Dickstein. Svcca: Singu-
lar vector canonical correlation analysis for deep learnin g dynamics and interpretability.
Advances in Neural Information Processing Systems , 30, 2017.
Lorenzo Rosasco, Ernesto De Vito, and Alessandro V erri. Spectral methods for regularization
in learning theory. DISI, Universita degli Studi di Genova, Italy, Technical Re port DISI-
TR-05-18, 2005.
17

<!-- page 18 -->

Lorenzo Rosasco, Mikhail Belkin, and Ernesto De Vito. On lea rning with integral operators.
Journal of Machine Learning Research , 11(2), 2010.
B Sch¨ olkopf.Learning with kernels: support vector machines, regulariz ation, optimization,
and beyond. The MIT Press, 2002.
Haozhe Shan and Blake Bordelon. A theory of neural tangent ke rnel alignment and its
inﬂuence on training. arXiv preprint arXiv:2105.14301, 2021.
Amanpreet Singh, Ronghang Hu, V edanuj Goswami, Guillaume Couairon, Wojciech Galuba,
Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment
model. In Proceedings of the IEEE /CVF Conference on Computer Vision and Pattern
Recognition, pp. 15638–15650, 2022.
Steve Smale and Ding-Xuan Zhou. Shannon sampling and functi on reconstruction from
point values. Bulletin of the American Mathematical Society , 41(3):279–305, 2004.
Le Song, Alex Smola, Arthur Gretton, Justin Bedo, and Karsten Borgwardt. Feature selection
via dependence maximization. Journal of Machine Learning Research , 13(5), 2012.
Alexander Wei, Wei Hu, and Jacob Steinhardt. More than a toy: Random matrix models
predict how real-world neural representations generalize . In International Conference on
Machine Learning, pp. 23549–23588. PMLR, 2022.
Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Y ang, Lei Zh ang, Lijuan Wang, Y ejin
Choi, and Jianfeng Gao. Vinvl: Revisiting visual represent ations in vision-language mod-
els. In Proceedings of the IEEE /CVF Conference on Computer Vision and Pattern Recog-
nition, pp. 5579–5588, 2021.
6 Appendix
6.1 Alignment to task
Here we mention ideas of alignment between a representation and task used to estimate
generalization error and characterize spectral contribut ions to sample complexity.
Kernel alignment risk estimator (KARE) In
Jacot et al. (2020) we have the following
deﬁnition for KARE which is an estimator for risk.
ρ(λ,yn, Kn) =
1
n⟨(Kn/ n + λI)−2, ynyT
n ⟩
( 1
nTr[(Kn/ n + λI)−1])2
This was also obtained in Golub et al. (1979), Wei et al. (2022), Craven & Wahba (1978).
18

<!-- page 19 -->

Spectral task-model alignment From Canatar et al. (2021), we have a deﬁnition for the
cumulative power distribution which quantiﬁes task-model alignment.
C(n) =
∑
i≤n ηiw2
i
∑
i ηiw2
i
Here K = ∑
i ηiφi ⊗φi, ⟨φi, φj⟩ = δi, j, and target hµ= ∑
i wi
√
ηiφi. C(n) can be interpreted as
fraction of variance of hµexplained by ﬁrst n features. The faster C(n) goes to 1, the higher
the alignment.
Source Condition From
Rosasco et al. (2005) we have bounds on generalization of kernel
ridge assuming some regularity of hµ, called source condition
hµ∈Ωr, R =
{
h ∈L2(X, ρ) : h = Lr
Kv, ∥v∥K ≤R
}
Assuming hµ= ∑
i wi
√ηiφi, then the statement can be rewritten as
∞∑
i=1
ηiw2
i
η2r
i
< ∞
Remark 7. KTA appears in several theoretical applications. Cristianini et al. (2001) bounds
generalization error of Parzen window classiﬁer 1. Cristianini et al. (2006); Cortes et al.
(2012) show that there exist predictors for which kernel target al ignment (KTA) A(K, yyT )
bounds risk.
h(x) = Ex′, y′
[K(x, x′)y′]
Ex′, x
[K(x, x′)2] ⇒ R(h) ≤2(1 −A(K, yyT ))
Furthermore, several authors including Atanasov et al. (2022); Paccolat et al. (2021); Kopitkov & Indelman
(2020); Fort et al. (2020); Shan & Bordelon (2021) use KTA to study feature learning and
Neural Tangent Kernel evolution.
6.2 Other notions for alignment from independence testing
Constrained Covariance (COCO) Then
Gretton et al. (2005a) proposed the concept of
constrained covariance as the largest singular value of the cross-covariance operator,
COCO(µ,H1, H2) = sup{cov[h1(x1), h2(x2)] : h1 ∈ H1, h2 ∈ H2}
1Cortes et al. (2012) notes error in proof since implicitly assumes max x Ex′
[
K2(x, x′)
]
/ Ex, x′
[
K2(x, x′)
]
= 1
making kernel constant. However proof can be saved with an ad ditional assumption.
19

<!-- page 20 -->

Kernel Canonical Correlation (KCC) From Bach & Jordan (2002)
KCC(µ,H1, H2, κ) = sup







cov[h1(x1), h2(x2)]
(var(h1(x1)) + κ∥h1∥2
H1
)1/ 2(var(h2(x2)) + κ∥h2∥2
H2
)1/ 2 : h1 ∈ H1, h2 ∈ H2







The next two are bounds on mutual information from correlati on and covariance respec-
tively
Kernel Mutual Information (KMI) From
Bach & Jordan (2002)
KMI(H1, H2) = −1
2 log(|I −(κ1, nκ2, n)K1, nK2, n|)
where kernels are centered and κq, n = mini
∑
j Kq(xq, i, xq, j) but empirically κ= 1/ n suﬃces.
6.3 Additional Proofs
In this section, we provide the detailed proofs of Lemmas pre sented in and Section 3 and
Section 4.
We begin with the proof of Lemma 1. For completeness, we ﬁrst restate the lemma below.
Lemma 4. Assume |Kq(xq, x′
q)| ≤Cq. Let ˆA1, 2(X) = ˆA1, 2((x1
1, x1
2), . . . , (xn
1, xn
2)) = 1
n2 ⟨K1, K2⟩F.
Let A 1, 2 = E ˆA1, 2, ˆA =
ˆA1, 2√
ˆA1, 1 ˆA2, 2
, and A = A1, 2√
A1, 1A2, 2
. Then with probability at least 1 −δ, and
ǫ=
√
(32/ n) log(2/δ), we have |ˆA −A| ≤C(X)ǫ, where C (X) is non-trivial function.
Proof. Let ( xi
1
′
, xi
2
′
) = (xi
1, xi
2) for all i = 1, . . . n except k. Then
Di j = K1(xi
1, x j
1)K2(xi
2, x j
2) −K1(xi
1
′
, x j
1
′
)K2(xi
2
′
, x j
2
′
) and note |Di j| ≤4C1C2. Then
|ˆA1, 2(X) − ˆA1, 2(X′)|= n−2







2
∑
j/nequali
|Di j|+ |Dii|







≤4C1C2
2n −1
n2 ≤8C
n
Applying McDiarmid, we get
P{|ˆA1, 2 −A1, 2| ≥ǫ} ≤2 exp
( −ǫ2n
32C2
)
which can also be read as, with probability at least 1 −δ, |ˆA1, 2 −A1, 2| ≤ǫ=
√
(32/ n) log(2/δ)
Finally, we show that |ˆAi, j −Ai, j| ≤ǫfor i, j ∈ {1, 2}gives |ˆA −A| ≤C(X)ǫ.
|ˆA −A|=
⏐
⏐
⏐ ˆA1, 2( ˆA1, 1 ˆA2, 2)−1/ 2 −A1, 2(A1, 1A2, 2)−1/ 2⏐
⏐
⏐
=|ˆA1, 2 −A1, 2|( ˆA1, 1 ˆA2, 2)−1/ 2 + A1, 2
⏐
⏐
⏐( ˆA1, 1 ˆA2, 2)−1/ 2 −(A1, 1A2, 2)−1/ 2⏐
⏐
⏐
=|ˆA1, 2 −A1, 2|( ˆA1, 1 ˆA2, 2)−1/ 2
+ A1, 2
( ⏐
⏐
⏐( ˆA1, 1 ˆA2, 2)−1/ 2 −(A1, 1 ˆA2, 2)−1/ 2⏐
⏐
⏐ +
⏐
⏐
⏐(A1, 1 ˆA2, 2)−1/ 2 −(A1, 1A2, 2)−1/ 2⏐
⏐
⏐
)
20

<!-- page 21 -->

Lastly, we can use
(x−1/ 2 −y−1/ 2) = y1/ 2 −x1/ 2
(xy)1/ 2 = y −x
(xy)1/ 2(y−1/ 2 + x−1/ 2)
□
Now we are in the position to prove Lemma 2. To complete this, we ﬁrst restate the
lemma below.
Lemma 5. Suppose dim (Y1) = dim(Y2) = d and R1 = R2. Let g q ∈ Gq be linear with
gq(zq) = Wqzq and W q ∈Rd×dq. Let s 1, 2 : Z1 → Z 2 be linear with s 1, 2(z1) = S 1, 2z1 and
S 1, 2 ∈Rd2×d1. Then Rstitch
1, 2 (S1, 2) = R1(H1).
Proof. For the linear case, there exists a vector Wq ∈Rd×dq, such that gq(zq) = Wqzq, zq ∈Rdq.
We can write the error of stitching as
Rstitch
1, 2 (s1, 2) = E
[
∥W2S 1, 2 f1 −y∥2]
= E
[
∥(W2S 1, 2 −W1) f1∥2]
+ E
[
∥W1 f1(x) −y∥2]
= ∥W2S 1, 2 −W1∥2
η1 + R1(h1),
where we used that for W1 to be optimal, we require ∂W1R1(h1) = E
[
(W1 f1 −y) f T
1
]
= 0.
Minimizing with respect to S 1, 2 yields
Rstitch
1, 2 (S1, 2) = ∥Π⊥
2 W1∥2
η1 + R1(H1),
where we use Π2 = I −(W T
2 diag(η1)W2)†W T
2 diag(η1) to denote the residual of the generalized
η1-projection onto (column) span of W2. We note that in general, as long as d ≤d2, we have
Rstitch
1, 2 (S1, 2) = R1(H1). □
21
