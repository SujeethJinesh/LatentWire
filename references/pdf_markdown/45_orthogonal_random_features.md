# references/45_orthogonal_random_features.pdf

<!-- page 1 -->

Orthogonal Random Features
Felix Xinnan Yu Ananda Theertha Suresh Krzysztof Choromanski
Daniel Holtmann-Rice Sanjiv Kumar
Google Research, New York
{felixyu, theertha, kchoro, dhr, sanjivk}@google.com
Abstract
We present an intriguing discovery related to Random Fourier Features: in Gaussian
kernel approximation, replacing the random Gaussian matrix by a properly scaled
random orthogonal matrix signiﬁcantly decreases kernel approximation error. We
call this technique Orthogonal Random Features (ORF), and provide theoretical
and empirical justiﬁcation for this behavior. Motivated by this discovery, we further
propose Structured Orthogonal Random Features (SORF), which uses a class of
structured discrete orthogonal matrices to speed up the computation. The method
reduces the time cost fromO(d2) toO(d logd), whered is the data dimensionality,
with almost no compromise in kernel approximation quality compared to ORF.
Experiments on several datasets verify the effectiveness of ORF and SORF over the
existing methods. We also provide discussions on using the same type of discrete
orthogonal structure for a broader range of applications.
1 Introduction
Kernel methods are widely used in nonlinear learning [9], but they are computationally expensive for
large datasets. Kernel approximation is a powerful technique to make kernel methods scalable, by
mapping input features into a new space where dot products approximate the kernel well [20]. With
accurate kernel approximation, efﬁcient linear classiﬁers can be trained in the transformed space
while retaining the expressive power of nonlinear methods [11, 22].
Formally, given a kernel K(·,·) : Rd× Rd→ R, kernel approximation methods seek to ﬁnd a
nonlinear transformationφ(·) : Rd→ Rd′
such that, for any x, y∈ Rd
K(x, y)≈ ˆK(x, y) =φ(x)Tφ(y).
Random Fourier Features [20] are used widely in approximating smooth, shift-invariant kernels. This
technique requires the kernel to exhibit two properties: 1) shift-invariance, i.e. K(x, y) = K(∆)
where ∆ = x− y; and 2) positive semi-deﬁniteness ofK(∆) on Rd. The second property guarantees
that the Fourier transform ofK(∆) is a nonnegative function [3]. Letp(w) be the Fourier transform
ofK(z). Then,
K(x− y) =
∫
Rd
p(w)ejwT (x−y)dw.
This means that one can treatp(w) as a density function and use Monte-Carlo sampling to derive the
following nonlinear map for a real-valued kernel:
φ(x) =
√
1/D
[
sin(wT
1 x),··· , sin(wT
Dx), cos(wT
1 x),··· , cos(wT
Dx)
]T
,
where wi is sampled i.i.d. from a probability distribution with density p(w). Let W =[
w1,··· , wD
]T
. The linear transformation Wx is central to the above computation since,
30th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain.
arXiv:1610.09072v1  [cs.LG]  28 Oct 2016

<!-- page 2 -->

1 2 3 4 5 6 7 8 9 100
0.5
1
1.5
2 x 10
−3
MSE
D / d


RFF (Random Gaussian)
ORF (Random Orthogonal)
(a) USPS
1 2 3 4 5 6 7 8 9 100
1
2
3
4 x 10
−4
MSE
D / d


RFF (Random Gaussian)
ORF (Random Orthogonal) (b) MNIST
1 2 3 4 5 6 7 8 9 100
2
4
6
8 x 10
−4
MSE
D / d


RFF (Random Gaussian)
ORF (Random Orthogonal) (c) CIFAR
Figure 1: Kernel approximation mean squared error (MSE) for the Gaussian kernel K(x, y) =
e−||x−y||2/2σ2
. D: number of rows in the linear transformation W. d: input dimension. ORF
imposes orthogonality on W (Section 3).
• The choice of matrix W determines how well the estimated kernel converges to the actual kernel;
• The computation of Wx has space and time costs of O(Dd). This is expensive for high-
dimensional data, especially since D is often required to be larger than d to achieve low ap-
proximation error.
In this work, we address both of the above issues. We ﬁrst show an intriguing discovery (Figure 1):
by enforcing orthogonality on the rows of W, the kernel approximation error can be signiﬁcantly
reduced. We call this method Orthogonal Random Features (ORF). Section 3 describes the method
and provides theoretical explanation for the improved performance.
Since both generating ad×d orthogonal matrix (O(d3) time andO(d2) space) and computing the
transformation (O(d2) time and space) are prohibitively expensive for high-dimensional data, we
further propose Structured Orthogonal Random Features (SORF) in Section 4. The idea is to replace
random orthogonal matrices by a class of special structured matrices consisting of products of binary
diagonal matrices and Walsh-Hadamard matrices. SORF has fast computation time, O(D logd),
and almost no extra memory cost (with efﬁcient in-place implementation). We show extensive
experiments in Section 5. We also provide theoretical discussions in Section 6 of applying the
structured matrices in a broader range of applications where random Gaussian matrix is used.
2 Related Works
Explicit nonlinear random feature maps have been constructed for many types of kernels, such as
intersection kernels [16], generalized RBF kernels [ 23], skewed multiplicative histogram kernels
[15], additive kernels [25], and polynomial kernels [12, 19]. In this paper, we focus on approximating
Gaussian kernels following the seminal Random Fourier Features (RFF) framework [20], which has
been extensively studied both theoretically and empirically [27, 21, 24].
Key to the RFF technique is Monte-Carlo sampling. It is well known that the convergence of Monte-
Carlo can be largely improved by carefully choosing a deterministic sequence instead of random
samples [18]. Following this line of reasoning, Yang et al. [26] proposed to use low-displacement
rank sequences in RFF. Yu et al. [29] studied optimizing the sequences in a data-dependent fashion to
achieve more compact maps. In contrast to the above works, this paper is motivated by an intriguing
new discovery that using orthogonal random samples provides much faster convergence. Compared
to [26], the proposed SORF method achieves both lower kernel approximation error and greatly
reduced computation and memory costs. Furthermore, unlike [29], the results in this paper are data
independent.
Structured matrices have been used for speeding up dimensionality reduction [1], binary embedding
[28], deep neural networks [6] and kernel approximation [14, 29, 8]. For the kernel approximation
works, in particular, the “structured randomness” leads to a minor loss of accuracy, but allows faster
computation since the structured matrices enable the use of FFT-like algorithms. Furthermore, these
matrices provide substantial model compression since they require subquadratic (usually only linear)
2

<!-- page 3 -->

Method Extra Memory Time Lower error than RFF?
Random Fourier Feature (RFF) [20] O(Dd) O(Dd) -
Compact Nonlinear Map (CNM) [29] O(Dd) O(Dd) Yes (data-dependent)
Quasi-Monte Carlo (QMC) [26] O(Dd) O(Dd) Yes
Structured (fastfood/circulant) [29, 14] O(D) O(D logd) No
Orthogonal Random Feature (ORF) O(Dd) O(Dd) Y es
Structured ORF (SORF) O(D) or O(1) O(D log d) Y es
Table 1: Comparison of different kernel approximation methods under the framework of Random
Fourier Features [20]. We assume D≥ d. The proposed SORF method have O(D) degrees of
freedom. The computations can be efﬁciently implemented as in-place operations with ﬁxed random
seeds. Therefore it can costO(1) in extra space.
space. In comparison with the above works, our proposed methods SORF and ORF are more effective
than RFF. In particular SORF demonstrates both lower approximation error and better efﬁciency than
RFF. Table 1 compares the space and time costs of different techniques.
3 Orthogonal Random Features
Our goal is to approximate a Gaussian kernel of the form
K(x, y) =e−||x−y||2/2σ2
.
In the paragraph below, we assume a square linear transformation matrix W∈ RD×d, D = d.
WhenD < d, we simply use the ﬁrstD dimensions of the result. When D > d, we use multiple
independently generated random features and concatenate the results. We comment on this setting at
the end of this section.
Recall that the linear transformation matrix of RFF can be written as
WRFF = 1
σ G, (1)
where G∈ Rd×d is a random Gaussian matrix, with every entry sampled independently from the
standard normal distribution. Denote the approximate kernel based on the aboveWRFF asKRFF(x, y).
For completeness, we ﬁrst show the expectation and variance ofKRFF(x, y).
Lemma 1. (Appendix A.2) KRFF(x, y) is an unbiased estimator of the Gaussian kernel, i.e.,
E(KRFF(x, y)) = e−||x−y||2/2σ2
. Let z = ||x− y||/σ. The variance of KRFF(x, y) is
Var(KRFF(x, y)) = 1
2D
(
1−e−z2
)2
.
The idea of Orthogonal Random Features (ORF) is to impose orthogonality on the matrix on the
linear transformation matrix G. Note that one cannot achieve unbiased kernel estimation by simply
replacing G by an orthogonal matrix, since the norms of the rows of G follow theχ-distribution,
while rows of an orthogonal matrix have the unit norm. The linear transformation matrix of ORF has
the following form
WORF = 1
σ SQ, (2)
where Q is a uniformly distributed random orthogonal matrix1. The set of rows of Q forms a bases in
Rd. S is a diagonal matrix, with diagonal entries sampled i.i.d. from theχ-distribution withd degrees
of freedom. S makes the norms of the rows of SQ and G identically distributed.
Denote the approximate kernel based on the above WORF asKORF(x, y). The following shows that
KORF(x, y) is an unbiased estimator of the kernel, and it has lower variance in comparison to RFF.
Theorem 1. KORF(x, y) is an unbiased estimator of the Gaussian kernel, i.e.,
E(KORF(x, y)) =e−||x−y||2/2σ2
.
1We ﬁrst generate the random Gaussian matrixG in (1). Q is the orthogonal matrix obtained from the QR
decomposition of G. Q is distributed uniformly on the Stiefel manifold (the space of all orthogonal matrices)
based on the Bartlett decomposition theorem [17].
3

<!-- page 4 -->

0 1 2 3 40
0.2
0.4
0.6
0.8
1
1.2
z
variance ratio


d =∞
(a) Variance ratio (whend is large)
0 1 2 3 40
0.2
0.4
0.6
0.8
1
1.2
z
variance ratio


d = 2
d = 4
d = 8
d = 16
d = 32
d =∞ (b) Variance ratio (simulation)
0 1 2 3 40
1
2
3
4
5
z
count


letter
forest
usps
cifar
mnist
gisette (c) Empirical distribution of z
Figure 2: (a) Var(KORF(x, y))/Var(KRFF(x, y)) whend is large andd =D. z =||x− y||/σ. (b)
Simulation of Var(KORF(x, y))/Var(KRFF(x, y)) whenD =d. Note that the empirical variance is
the Mean Squared Error (MSE). (c) Distribution ofz for several datasets, when we setσ as the mean
distance to 50th-nearest neighbor for samples from the dataset. The count is normalized such that the
area under curve for each dataset is 1. Observe that most points in all the datasets havez <2. As
shown in (a), for these values ofz, ORF has much smaller variance compared to the standard RFF.
LetD≤ d, and z =||x− y||/σ. There exists a function f such that for all z, the variance of
KORF(x, y) is bounded by
Var(KORF(x, y))≤ 1
2D
((
1−e−z2 )2
− D− 1
d e−z2
z4
)
+ f(z)
d2 .
Proof. We ﬁrst show the proof of the unbiasedness. Let z = x−y
σ , and z = ||z||, then
E(KORF (x, y)) = E
(
1
D
∑D
i=1 cos(wT
i z)
)
= 1
D
∑D
i=1 E
(
cos(wT
i z)
)
. Based on the deﬁnition
of ORF, w1, w2,..., wD areD random vectors given by wi = siui, with u1, u2,..., ud a uni-
formly chosen random orthonormal basis for Rd, and si’s are independentχ-distributed random
variables withd degrees of freedom. It is easy to show that for eachi, wi is distributed according to
N(0, Id), and hence by Bochner’s theorem,
E[cos(wT z)] =e−z2/2.
We now show a proof sketch of the variance. Suppose,ai = cos(wT
i z).
Var
(
1
D
D∑
i=1
ai
)
= E
[( ∑D
i=1ai
D
)2]
− E
[( ∑D
i=1ai
D
)]2
= 1
D2
∑
i
(
E[a2
i ]− E[ai]2)
+ 1
D2
∑
i
∑
j̸=i
(E[aiaj]− E[ai]E[aj])
=
(
1−e−z2 )2
2D + D(D− 1)
D2
(
E[a1a2]−e−z2 )
,
where the last equality follows from symmetry. The ﬁrst term in the resulting expression is exactly
the variance of RFF. In order to have lower variance,E[a1a2]−e−z2
must be negative. We use the
following lemma to quantify this term.
Lemma 2. (Appendix A.3) There is a functionf such that for anyz,
E[aiaj]≤e−z2
−e−z2z4
2d + f(z)
d2 .
Therefore, for a larged, andD≤d, the ratio of the variance of ORF and RFF is
Var(KORF(x, y))
Var(KRFF(x, y))≈ 1− (D− 1)e−z2
z4
d(1−e−z2
)2 . (3)
Figure 2(a) shows the ratio of the variance of ORF to that of RFF whenD =d andd is large. First
notice that this ratio is always smaller than 1, and hence ORF always provides improvement over
4

<!-- page 5 -->

0 2 4 6 8 10−0.5
−0.4
−0.3
−0.2
−0.1
0
0.1
0.2
0.3
z
bias


d=2d=4d=8d=16d=32
(a) Bias of ORF′
0 2 4 6 8 10
−1
−0.5
0
0.5
1
z
bias


d=2d=4d=8d=16d=32 (b) Bias of SORF
0 1 2 3 40
0.2
0.4
0.6
0.8
1
1.2
z
variance ratio


d = 2d = 4d = 8d = 16d = 32d =∞ (c) Variance ratio of ORF′
0 1 2 3 40
0.2
0.4
0.6
0.8
1
1.2
z
variance ratio


d = 16
d = 32d = 64
d =∞ (d) Variance ratio of SORF
Figure 3: Simulations of bias and variance of ORF ′and SORF. z = ||x− y||/σ. (a)
E(KORF′(x, y))−e−z2/2. (b) E(KSORF(x, y))−e−z2/2. (c) Var(KORF′(x, y))/Var(KRFF(x, y)).
(d) Var(KSORF(x, y))/Var(KRFF(x, y)). Each point on the curve is based on 20,000 choices of the
random matrices and two ﬁxed points with distancez. For both ORF and ORF′, even atd = 32, the
bias is close to 0 and the variance is close to that ofd =∞ (Figure 2(a)).
the conventional RFF. Interestingly, we gain signiﬁcantly for small values ofz. In fact, whenz→ 0
andd→∞ , the ratio is roughly z2 (noteex≈ 1 +x whenx→ 0), and ORF exhibits inﬁnitely
lower error relative to RFF. Figure 2(b) shows empirical simulations of this ratio. We can see that the
variance ratio is close to that ofd =∞ (3), even whend = 32, a fairly low-dimensional setting in
real-world cases.
Recall thatz =||x− y||/σ. This means that ORF preserves the kernel value especially well for data
points that are close, thereby retaining the local structure of the dataset. Furthermore, empiricallyσ
is typically not set too small in order to prevent overﬁtting—a common rule of thumb is to setσ to be
the average distance of 50th-nearest neighbors in a dataset. In Figure 2(c), we plot the distribution of
z for several datasets with this choice of σ. These distributions are all concentrated in the regime
where ORF yields substantial variance reduction.
The above analysis is under the assumption thatD≤d. Empirically, for RFF,D needs to be larger
thand in order to achieve low approximation error. In that case, we independently generate and apply
the transformation (2) multiple times. The next lemma bounds the variance for this case.
Corollary 1. LetD =m·d, for an integerm andz =||x− y||/σ. There exists a functionf such
that for allz, the variance ofKORF(x, y) is bounded by
Var(KORF(x, y))≤ 1
2D
((
1−e−z2 )2
− d− 1
d e−z2
z4
)
+ f(z)
dD .
4 Structured Orthogonal Random Features
In the previous section, we presented Orthogonal Random Features (ORF) and provided a theoretical
explanation for their effectiveness. Since generating orthogonal matrices in high dimensions can be
expensive, here we propose a fast version of ORF by imposing structure on the orthogonal matrices.
This method can provide drastic memory and time savings with minimal compromise on kernel
approximation quality. Note that the previous works on fast kernel approximation using structured
matrices do not use structured orthogonal matrices [14, 29, 8].
Let us ﬁrst introduce a simpliﬁed version of ORF: replaceS in (2) by a scalar
√
d. Let us call this
method ORF′. The transformation matrix thus has the following form:
WORF′ =
√
d
σ Q. (4)
Theorem 2. (Appendix B) LetKORF′(x, y) be the approximate kernel computed with linear transfor-
mation matrix (4). LetD≤d andz =||x− y||/σ. There exists a functionf such that the bias of
KORF′(x, y) satisﬁes
⏐⏐⏐E(KORF′(x, y))−e−z2/2
⏐⏐⏐≤e−z2/2z4
4d + f(z)
d2 ,
5

<!-- page 6 -->

1 2 3 4 5 6 7 8 9 100
0.01
0.02
0.03
0.04MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood
(a) LETTER (d = 16)
1 2 3 4 5 6 7 8 9 100
2
4
6
8 x 10
−3
MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood (b) FOREST (d = 64)
1 2 3 4 5 6 7 8 9 100
0.5
1
1.5
2
2.5
3 x 10
−3
MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood (c) USPS (d = 256)
1 2 3 4 5 6 7 8 9 100
0.2
0.4
0.6
0.8
1
1.2 x 10
−3
MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood
(d) CIFAR (d = 512)
1 2 3 4 5 6 7 8 9 100
1
2
3
4
5
6 x 10
−4
MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood (e) MNIST (d = 1024)
1 2 3 4 5 6 7 8 9 100
0.2
0.4
0.6
0.8
1
1.2 x 10
−4
MSE
D / d


RFF
ORF
SORF
QMC(digitalnet)
circulant
fastfood (f) GISETTE (d = 4096)
Figure 4: Kernel approximation mean squared error (MSE) for the Gaussian kernel K(x, y) =
e−||x−y||2/2σ2
. D: number of transformations. d: input feature dimension. For each dataset, σ
is chosen to be the mean distance of the 50th ℓ2 nearest neighbor for 1,000 sampled datapoints.
Empirically, this yields good classiﬁcation results. The curves for SORF and ORF overlap.
and the variance satisﬁes
Var(KORF′(x, y))≤ 1
2D
(
(1−e−z2
)2− D− 1
d e−z2
z4
)
+ f(z)
d2 .
The above implies that when d is large KORF′(x, y) is a good estimation of the kernel with low
variance. Figure 3(a) shows that even for relatively smalld, the estimation is almost unbiased. Figure
3(c) shows that whend≥ 32, the variance ratio is very close to that ofd =∞. We ﬁnd empirically
that ORF′also provides very similar MSE in comparison with ORF in real-world datasets.
We now introduce Structured Orthogonal Random Features (SORF). It replaces the random orthogonal
matrix Q of ORF′in (4) by a special type of structured matrix HD1HD2HD3:
WSORF =
√
d
σ HD1HD2HD3, (5)
where Di ∈ Rd×d,i = 1, 2, 3 are diagonal “sign-ﬂipping” matrices, with each diagonal entry
sampled from the Rademacher distribution. H is the normalized Walsh-Hadamard matrix.
Computing WSORFx has the time costO(d logd), since multiplication with D takesO(d) time and
multiplication with H takesO(d logd) time using fast Hadamard transformation. The computation
of SORF can also be carried out with almost no extra memory due to the fact that both sign ﬂipping
and the Walsh-Hadamard transformation can be efﬁciently implemented as in-place operations [10].
Figures 3(b)(d) show the bias and variance of SORF. Note that although the curves for smalld are
different from those of ORF, whend is large (d >32 in practice), the kernel estimation is almost
unbiased, and the variance ratio converges to that of ORF. In other words, it is clear that SORF can
provide almost identical kernel approximation quality as that of ORF. This is also conﬁrmed by the
experiments in Section 5. In Section 6, we provide theoretical discussions to show that the structure
of (5) can also be generally applied to many scenarios where random Gaussian matrices are used.
6

<!-- page 7 -->

Dataset Method D = 2d D = 4d D = 6d D = 8d D = 10d Exact
letter
d = 16
RFF 76.44± 1.04 81.61± 0.46 85.46± 0.56 86.58± 0.99 87.84± 0.59
90.10ORF 77.49± 0.95 82.49± 1.16 85.41± 0.60 87.17± 0.40 87.73± 0.63
SORF 76.18± 1.20 81.63± 0.77 84.43± 0.92 85.71± 0.52 86.78± 0.53
forest
d = 64
RFF 77.61± 0.23 78.92± 0.30 79.29± 0.24 79.57± 0.21 79.85± 0.10
80.43ORF 77.88± 0.24 78.71± 0.19 79.38± 0.19 79.63± 0.21 79.54± 0.15
SORF 77.64± 0.20 78.88± 0.14 79.31± 0.12 79.50± 0.14 79.56± 0.09
usps
d = 256
RFF 94.27± 0.38 94.98± 0.10 95.43± 0.22 95.66± 0.25 95.71± 0.18
95.57ORF 94.21± 0.51 95.26± 0.25 96.46± 0.18 95.52± 0.20 95.76± 0.17
SORF 94.45± 0.39 95.20± 0.43 95.51± 0.34 95.46± 0.34 95.67± 0.15
cifar
d = 512
RFF 73.19± 0.23 75.06± 0.33 75.85± 0.30 76.28± 0.30 76.54± 0.31
78.71ORF 73.59± 0.44 75.06± 0.28 76.00± 0.26 76.29± 0.26 76.69± 0.09
SORF 73.54± 0.26 75.11± 0.21 75.76± 0.21 76.48± 0.24 76.47± 0.28
mnist
d = 1024
RFF 94.83± 0.13 95.48± 0.10 95.85± 0.07 96.02± 0.06 95.98± 0.05
97.14ORF 94.95± 0.25 95.64± 0.06 95.85± 0.09 95.95± 0.08 96.06± 0.07
SORF 94.98± 0.18 95.48± 0.08 95.77± 0.09 95.98± 0.05 96.02± 0.07
gisette
d = 4096
RFF 97.68± 0.28 97.74± 0.11 97.66± 0.25 97.70± 0.16 97.74± 0.05
97.60ORF 97.56± 0.17 97.72± 0.15 97.80± 0.07 97.64± 0.09 97.68± 0.04
SORF 97.64± 0.17 97.62± 0.04 97.64± 0.11 97.68± 0.08 97.70± 0.14
Table 2: Classiﬁcation Accuracy based on SVM. ORF and SORF provide competitive classiﬁcation
accuracy for a givenD. Exact is based on kernel-SVM trained on the Gaussian kernel. Note that
in all the settings SORF is faster than RFF and ORF by a factor of O(d/ logd). For example, on
gisette withD = 2d, SORF provides 10 times speedup in comparison with RFF and ORF.
5 Experiments
Kernel Approximation. We ﬁrst show kernel approximation performance on six datasets. The input
feature dimensiond is set to be power of 2 by padding zeros or subsampling. Figure 4 compares the
mean squared error (MSE) of all methods. For ﬁxedD, the kernel approximation MSE exhibits the
following ordering:
SORF≃ ORF< QMC [26]< RFF [20]< Other fast kernel approximations [14, 29].
By imposing orthogonality on the linear transformation matrix, Orthogonal Random Features (ORF)
achieves signiﬁcantly lower approximation error than Random Fourier Features (RFF). The Structured
Orthogonal Random Features (SORF) have almost identical MSE to that of ORF. All other fast kernel
approximation methods, such as circulant [29] and FastFood [14] have higher MSE. We also include
DigitalNet, the best performing method among Quasi-Monte Carlo techniques [26]. Its MSE is lower
than that of RFF, but still higher than that of ORF and SORF. The order of time cost for a ﬁxedD is
SORF≃ Other fast kernel approximations [14, 29]≪ ORF = QMC [26] = RFF [20].
Remarkably, SORF has both better computational efﬁciency and higher kernel approximation quality
compared to other methods.
We also apply ORF and SORF on classiﬁcation tasks. Table 2 shows classiﬁcation accuracy for
different kernel approximation techniques with a (linear) SVM classiﬁer. SORF is competitive with
or better than RFF, and has greatly reduced time and space costs.
The Role of σ. Note that a very small σ will lead to overﬁtting, and a very large σ provides no
discriminative power for classiﬁcation. Throughout the experiments,σ for each dataset is chosen to
be the mean distance of the 50thℓ2 nearest neighbor, which empirically yields good classiﬁcation
results [29]. As shown in Section 3, the relative improvement over RFF is positively correlated with
σ. Figure 5(a)(b) verify this on the mnist dataset. Notice that the proposed methods (ORF and
SORF) consistently improve over RFF.
Simplifying SORF . The SORF transformation consists of three Hadamard-Diagonal blocks. A
natural question is whether using fewer computations and randomness can achieve similar empirical
performance. Figure 5(c) shows that reducing the number of blocks to two (HDHD) provides similar
performance, while reducing to one block (HD) leads to large error.
6 Analysis and General Applicability of the Hadamard-Diagonal Structure
We provide theoretical discussions of SORF in this section. We ﬁrst show that for larged, SORF is
an unbiased estimator of the Gaussian kernel.
7

<!-- page 8 -->

1 2 3 4 5 6 7 8 9 100
1
2
3
4
5
6 x 10
−4
MSE
D / d


RFF
ORF
SORF
(a) σ = 0.5× 50NN distance
1 2 3 4 5 6 7 8 9 100
1
2
x 10
−4
MSE
D / d


RFF
ORF
SORF (b) σ = 2× 50NN distance
1 2 3 4 5 6 7 8 9 100
0.2
0.4
0.6
0.8
1 x 10
−3
MSE
D / d


HDHDHD
HDHD
HD (c) Variants of SORF
Figure 5: (a) (b) MSE on mnist with differentσ. (c) Effect of using less randomness on mnist.
HDHDHD is the the proposed SORF method. HDHD reduces the number of Hadamard-Diagonal
blocks to two, and HD uses only one such block.
Theorem 3. (Appendix C) LetKSORF(x, y) be the approximate kernel computed with linear trans-
formation matrix
√
dHD1HD2HD3. Letz =||x− y||/σ. Then
⏐⏐⏐E(KSORF(x, y))−e−z2/2
⏐⏐⏐≤ 6z√
d
.
Even though SORF is nearly-unbiased, proving tight variance and concentration guarantees similar
to ORF remains an open question. The following discussion provides a sketch in that direction. We
ﬁrst show a lemma of RFF.
Lemma 3. Let W be a random Gaussian matrix as in RFF , for a givenz, the distribution of Wz is
N(0,||z||2Id).
Note that Wz in RFF can be written as Rg, where R is a scaled orthogonal matrix such that each
row has norm||z||2 and g is distributed according to N(0, Id). Hence the distribution of Rg is
N(0,||z||2Id), identical to Wz. The concentration results of RFF use the fact that the projections of
a Gaussian vector g onto orthogonal directions R are independent.
We show that
√
dHD1HD2HD3z has similar properties. In particular, we show that it can be
written as ˜R˜g, where rows of ˜R are “near-orthogonal” (with high probability) and have norm ||z||2,
and the vector ˜g is close to Gaussian ( ˜g has independent sub-Gaussian elements), and hence the
projections behave “near-independently”. Speciﬁcally, ˜g = vec(D1) (vector of diagonal entries of
D1), and ˜R is a function of D2, D3 and z.
Theorem 4. (Appendix D) For a given z, there exists a ˜R (function of D2, D3, z), such that√
dHD1HD2HD3z = ˜Rvec(D1). Each row of ˜R has norm||z||2 and for any t≥ 1/d, with
probability 1−de−c·t2/3d1/3
, the inner product between any two rows of ˜R is at mostt||z||2, wherec
is a constant.
The above result can also be applied to settings not limited to kernel approximation. In the appendix,
we show empirically that the same scheme can be successfully applied to angle estimation where the
nonlinear mapf is a non-smooth sign(·) function [4]. We note that the HD1HD2HD3 structure
has also been recently used in fast cross-polytope LSH [2, 13, 7].
7 Conclusions
We have demonstrated that imposing orthogonality on the transformation matrix can greatly reduce
the kernel approximation MSE of Random Fourier Features when approximating Gaussian kernels.
We further proposed a type of structured orthogonal matrices with substantially lower computation
and memory cost. We provided theoretical insights indicating that the Hadamard-Diagonal block
structure can be generally used to replace random Gaussian matrices in a broader range of applications.
Our method can also be generalized to other types of kernels such as general shift-invariant kernels
and polynomial kernels based on Schoenberg’s characterization as in [19].
8

<!-- page 9 -->

References
[1] N. Ailon and B. Chazelle. Approximate nearest neighbors and the fast Johnson-Lindenstrauss
transform. In STOC, 2006.
[2] A. Andoni, P. Indyk, T. Laarhoven, I. Razenshteyn, and L. Schmidt. Practical and optimal lsh
for angular distance. In NIPS, 2015.
[3] S. Bochner. Harmonic analysis and the theory of probability. Dover Publications, 1955.
[4] M. S. Charikar. Similarity estimation techniques from rounding algorithms. In STOC, 2002.
[5] S. Chatterjee. Lecture notes on Stein’s method and applications. 2007.
[6] Y . Cheng, F. X. Yu, R. S. Feris, S. Kumar, A. Choudhary, and S.-F. Chang. An exploration of
parameter redundancy in deep networks with circulant projections. In ICCV, 2015.
[7] K. Choromanski, F. Fagan, C. Gouy-Pailler, A. Morvan, T. Sarlos, and J. Atif. Triplespin-a
generic compact paradigm for fast machine learning computations. arXiv, 2016.
[8] K. Choromanski and V . Sindhwani. Recycling randomness with structure for sublinear time
kernel expansions. ICML, 2015.
[9] C. Cortes and V . Vapnik. Support-vector networks. Machine Learning, 20(3):273–297, 1995.
[10] B. J. Fino and V . R. Algazi. Uniﬁed matrix treatment of the fast walsh-hadamard transform.
IEEE Transactions on Computers, (11):1142–1146, 1976.
[11] T. Joachims. Training linear SVMs in linear time. In KDD, 2006.
[12] P. Kar and H. Karnick. Random feature maps for dot product kernels. In AISTATS, 2012.
[13] C. Kennedy and R. Ward. Fast cross-polytope locality-sensitive hashing. arXiv, 2016.
[14] Q. Le, T. Sarlós, and A. Smola. Fastfood – approximating kernel expansions in loglinear time.
In ICML, 2013.
[15] F. Li, C. Ionescu, and C. Sminchisescu. Random fourier approximations for skewed multiplica-
tive histogram kernels. Pattern Recognition, pages 262–271, 2010.
[16] S. Maji and A. C. Berg. Max-margin additive classiﬁers for detection. In ICCV, 2009.
[17] R. J. Muirhead. Aspects of multivariate statistical theory, volume 197. John Wiley & Sons,
2009.
[18] H. Niederreiter. Quasi-Monte Carlo Methods. Wiley Online Library, 2010.
[19] J. Pennington, F. X. Yu, and S. Kumar. Spherical random features for polynomial kernels. In
NIPS, 2015.
[20] A. Rahimi and B. Recht. Random features for large-scale kernel machines. In NIPS, 2007.
[21] A. Rudi, R. Camoriano, and L. Rosasco. Generalization properties of learning with random
features. arXiv:1602.04474, 2016.
[22] S. Shalev-Shwartz, Y . Singer, N. Srebro, and A. Cotter. Pegasos: Primal estimated sub-gradient
solver for SVM, volume = 127, year = 2011. Mathematical Programming, (1):3–30.
[23] V . Sreekanth, A. Vedaldi, A. Zisserman, and C. Jawahar. Generalized RBF feature maps for
efﬁcient detection. In BMVC, 2010.
[24] B. Sriperumbudur and Z. Szabó. Optimal rates for random fourier features. In NIPS, 2015.
[25] A. Vedaldi and A. Zisserman. Efﬁcient additive kernels via explicit feature maps. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 34(3):480–492, 2012.
[26] J. Yang, V . Sindhwani, H. Avron, and M. Mahoney. Quasi-monte carlo feature maps for
shift-invariant kernels. In ICML, 2014.
[27] T. Yang, Y .-F. Li, M. Mahdavi, R. Jin, and Z.-H. Zhou. Nyström method vs random fourier
features: A theoretical and empirical comparison. In NIPS, 2012.
[28] F. X. Yu, S. Kumar, Y . Gong, and S.-F. Chang. Circulant binary embedding. InICML, 2014.
[29] F. X. Yu, S. Kumar, H. Rowley, and S.-F. Chang. Compact nonlinear maps and circulant
extensions. arXiv:1503.03893, 2015.
[30] X. Zhang, F. X. Yu, R. Guo, S. Kumar, S. Wang, and S.-F. Chang. Fast orthogonal projection
based on kronecker product. In ICCV, 2015.
9

<!-- page 10 -->

Appendix A Variance Reduction via Orthogonal Random Features
A.1 Notation
Let z = x−y
σ , andz =||z||. For a vector y, lety(i) denote itsith coordinate. Letn!! be the double
factorial ofn, i.e., the product of every number from n to 1 that has the same parity as n.
A.2 Proof of Lemma 1
Let z = (x− y)/σ. Recall that in RFF, we compute the Kernel approximation as
D∑
i=1
1
D cos(wT
i z),
where each wi is a d dimensional vector distributed N(0,Id). Let w be a d dimensional vector
distributedN(0,Id). By Bochner’s theorem,
E[cos(wT z)] =e−z2/2,
and hence RFF yields an unbiased estimate.
We now compute the variance of RFF approximation. Observe that
cos2(wT z) = 1 + cos(2wT z)
2 = 1 + cos(wT (2z))
2 .
Hence by Bochner’s theorem
E[cos2(wT z)] = 1 +e−2z2
2 .
Therefore,
Var(cos(wT z)) = E[cos2(wT z)]− (E[cos(wT z)])2
= 1 +e−2z2
2 −e−z2
= (1−e−z2
)2
2 .
If we takeD such independent random variables w1, w2,... wD, since variance of the sum is sum
of variances,
Var
(
1
D
D∑
i=1
cos(wT
i z)
)
= (1−e−z2
)2
2D .
A.3 Proof of Lemma 2
The proof uses the following lemma.
Lemma 4. For a set of non-negative values α1,α 2,...α k and β1,β 2,...β k such that for all i,
βi≤αi,
⏐⏐⏐⏐⏐
1
∏k
i (1 +αi)
−
(
1−
k∑
i=1
αi
)⏐⏐⏐⏐⏐≤
( k∑
i=1
αi
)2
,
and
⏐⏐⏐⏐⏐
k∏
i
1 +βi
1 +αi
−
(
1 +
k∑
i=1
βi−
k∑
i=1
αi
)⏐⏐⏐⏐⏐≤
( k∑
i=1
(αi−βi)
)2
+
k∑
i=1
(αi−βi)βi.
10

<!-- page 11 -->

Proof. Sinceαis are non-negative,
1
∏k
i (1 +αi)
−
(
1−
k∑
i=1
αi
)
≤ 1
1 + ∑k
i αi
−
(
1−
k∑
i=1
αi
)
= 1− (1 + ∑k
i=1αi)(1− ∑k
i=1αi)
1 + ∑k
i αi
= (∑k
i=1αi)2
1 + ∑k
i αi
≤
(k∑
i=1
αi
)2
.
Furthermore, by convexity
1
∏k
i (1 +αi)
≥ 1
(1 + ∑k
i=1αi/k)k
≥e− ∑k
i=1αi≥ 1−
k∑
i=1
αi.
Combining the above two equations results in the ﬁrst part of the lemma. For the second part observe
that
k∏
i
1 +βi
1 +αi
= 1
∏k
i
(
1 + αi−βi
1+βi
).
Hence, by the ﬁrst part
⏐⏐⏐⏐⏐
k∏
i
1 +βi
1 +αi
−
(
1−
k∑
i=1
αi−βi
1 +βi
)⏐⏐⏐⏐⏐≤
( k∑
i=1
αi−βi
1 +βi
)2
≤
( k∑
i=1
(αi−βi)
)2
.
Furthermore, for everyi ⏐⏐⏐⏐
1
1 +βi
− 1
⏐⏐⏐⏐≤βi.
Combining the above two equations yields the second part of the lemma.
Proof of Lemma 2. Observe that
cos(wT
1 z) cos(wT
2 z) = cos(wT
1 z + wT
2 z) + cos(wT
1 z− wT
2 z)
2 .
Since the problem is rotation invariant, instead of projecting a vector z onto a randomly chosen two
orthogonal vectors u1 and u2, we can choose a vector y that is uniformly distributed on a sphere of
radiusz and project it on to the ﬁrst two dimensions. Thus,
E[cos(wT
1 z + wT
2 z)] = E[cos((s1y(1) +s2y(2))z)].
Similarly,
E[cos(wT
1 z− wT
2 z)] = E[cos((s1y(1)−s2y(2))z)].
Thekth term in the Taylor’s series expansion of sum of above two terms is
(−1)k
(2k)! ((s1y(1) +s2y(2))z)2k + (−1)k
(2k)! ((s1y(1)−s2y(2))z)2k
= (−z2)k
(2k)!
k∑
i=0
(2k
2i
)
s2i
1 y2i(1)s2k−2i
2 y2k−2i(2).
A way to compute a uniformly distributed random variable on a sphere with radiusz is to generate
d independent random variables x = (x(1),x (2),...,x (d)) each distributed N(0, 1) and setting
11

<!-- page 12 -->

y(i) =zx(i)/||x||. Hence,
E
[k∑
i=0
(2k
2i
)
s2i
1 y2i(1)s2k−2i
2 y2k−2i(2)
]
(a)
= E
[2k∑
i=0
(2k
2i
)s2i
1 x2i(1)s2k−2i
2 x2k−2i(2)
||x||2k
]
(b)
=
k∑
i=0
(2k
2i
)
E[s2i
1 ]E[s2k−2i
2 ]E
[x2i(1)x2k−2i(2)
||x||2k
]
(c)
=
k∑
i=0
(2k
2i
)
E[s2i
1 ]E[s2k−2i
2 ] E[x2i(1)]E[x2k−2i(2)]
E[||x||2k]
(d)
=
k∑
i=0
(2k
2i
)(d + 2i− 2)!!(d + 2k− 2i− 2)!!· (2i− 1)!!(2k− 2i− 1)!!
(d + 2k− 2)!!(d− 2)!!
(e)
= (2k)!
2kk!
k∑
i=0
(k
i
)(d + 2i− 2)!!(d + 2k− 2i− 2)!!
(d + 2k− 2)!!(d− 2)!! .
(a) follows from linearity of expectation and the observation above.(b) follows from the independence
ofs1,s2, and x. (d) follows from substituting the moments of chi and Gaussian distributions. (e)
follows from numerical simpliﬁcation. We now describe the reasoning behind(c). Let z = x||y||
||x|| ,
where y and x are independentN(0,Id) random variables. By the properties of the Gaussian random
variables z is also aN(0,Id) random variable. Thus,
E[z2i(1)]E[z2k−2i(2)] = E
[x2i(1)x2k−2i(2)
||x||2k
]
E[||y||2k].
Rearranging terms, we get
E
[x2i(1)x2k−2i(2)
||x||2k
]
= E[z2i(1)]E[z2k−2i(2)]
E[||y||2k = E[x2i(1)]E[x2k−2i(2)]
E[||x||2k] ,
and hence (c). Substituting the above equation in the cosine expansion, we get that the expectation is
E[cos(s1y(1) +s2y(2)]] =
∞∑
k=0
(−z2)k
k!
k∑
i=0
(k
i
) 1
2k
(d + 2i− 2)!!(d + 2k− 2i− 2)!!
(d + 2k− 2)!!(d− 2)!! .
Observe that
(d + 2i− 2)!!(d + 2k− 2i− 2)!!
(d + 2k− 2)!!(d− 2)!! =
∏k−i−1
j=0 (1 + 2j/d)
∏k−i−1
j=0 (1 + 2(j +i)/d)
,
Hence by Lemma 4,
⏐⏐⏐⏐⏐⏐
∏k−i−1
j=0 (1 + 2j/d)
∏k−i−1
j=0 (1 + 2(j +i)/d)
−

1 +
k−i−1∑
j=0
2j
d −
k−i−1∑
j=0
2(j +i)
d


⏐⏐⏐⏐⏐⏐
≤


k−i−1∑
j=0
2i
d


2
+
k−i−1∑
j=0
2i
d
(2j
d
)
.
Simplifying we get,
⏐⏐⏐⏐⏐
∏k−i−1
j=0 (1 + 2j/d)
∏k−i−1
j=0 (1 + 2(j +i)/d)
−
(
1 + 2i2− 2ik
d
)⏐⏐⏐⏐⏐≤ 4i2(k−i)2
d2 + 2i(k−i)(k−i− 1)
d3 .
12

<!-- page 13 -->

Hence summing overi,
⏐⏐⏐⏐⏐
k∑
i=0
(k
i
) 1
2k
∏k−i−1
j=0 (1 + 2j/d)
∏k−i−1
j=0 (1 + 2(j +i)/d)
−
(
1 +k−k2
2d
)⏐⏐⏐⏐⏐≤ k4
4d2 + k2(k− 1)
2d3 .
Substituting,
E[cos((s1y(1) +s2y(2))z]] + E[cos((s1y(1)−s2y(2))z]]
2 =
∞∑
k=0
(−z2)k
k!
(
1 +k−k2
2d +ck,d
)
,
where|ck,d|≤ k4
4d2 + k2(k−1)
2d3 . Thus,
E[cos((s1y(1) +s2y(2))z]] + E[cos((s1y(1)−s2y(2))z]]
2
=
∞∑
k=0
(−z2)k
k!
(
1 +k−k2
2d +ck,d
)
≤
∞∑
k=0
(−z2)k
k!
(
1 +k−k2
2d
)
+
∞∑
k=0
(z2)k
k!
(k4
4d2 + k2(k− 1)
2d3
)
≤e−z2
−e−z2z4
2d + ez2
(z8 + 6z6 + 7z4 +z2)
4d2 + ez2
z4(z6 + 2z4)
2d3 .
Appendix B Proof of Theorem 2
The proof of the theorem is similar to that of Lemma 2 and we outline some key steps. We ﬁrst bound
the bias in Lemma 5 and then the variance in Lemma 6.
Lemma 5. If w =
√
dy, wherey is distributed uniformly on a unit sphere, then
⏐⏐⏐⏐E[cos wT z]−
(
e−z2/2−e−z2/2z4
4d
)⏐⏐⏐⏐≤ ez2/2z4(z4 + 8z2 + 8)
16d2 .
Proof. Without loss of generality, we can assume z is along the ﬁrst coordinate and hencewT z =√
dzy(1). A way to compute a uniformly distributed random variable on a sphere with radiusz is to
generated independent random variables x = (x(1),x (2),...,x (d)) each distributedN(0, 1) and
settingy(i) =zx(i)/||x||. hence,
E[cos wT z] = E
[
cos
(
z
√
dx(1)
||x||
)]
.
Thekth term in the Taylor’s series expansion of cosine in the above equation is
(−1)k
(2k)!
(√
dx(1)z
||x||
)2k
Similar to the proof of Lemma 2, it can be shown that the expectation of this term is
E

(−1)k
(2k)!
(
z
√
dx(1)
||x||
)2k
 = (−z2)k
2kk!
dk
(d, 2k− 2)!!.
Applying Lemma 4 and simplifying,
E[cos(dy(1))] =
∞∑
k=0
(−z2)k
2kk!
(
1− k(k− 1)
d +c′
k,d
)
,
13

<!-- page 14 -->

where|c′
k,d|≤
(
k(k−1)
d
)2
. Hence,
⏐⏐⏐⏐⏐E[cos(dy(1))]−
∞∑
k=0
(−z2)k
2kk!
(
1− k(k− 1)
d
)⏐⏐⏐⏐⏐≤
∞∑
k=0
(z2)k
2kk!
(k(k− 1)
d
)2
,
and thus ⏐⏐⏐⏐E[cos(dy(1))]−e−z2/2 +e−z2/2z4
4d
⏐⏐⏐⏐≤ ez2/2z4(z4 + 8z2 + 8)
16d2 .
Lemma 6. LetD≤d. If W =
√
dQ, where Q is a uniformly chosen random rotation, then
Var
(
1
D
D∑
i=1
cos(wT
i z)
)
≤ 1
2D
(
(1−e−z2
)2− D− 1
d e−z2
z4
)
+O(e3z2
)
d2 .
Proof. Letai = cos(wT
i z). Expanding the variance we have,
Var
(
1
D
D∑
i=1
ai
)
= 1
D2
∑
i
(
E[a2
i ]− (E[ai])2)
+ 1
D2
∑
i
∑
j̸=i
(E[aiaj]− E[ai]E[aj])
= 1
D
(
E[a2
1]− (E[a1]2)
)
+ D− 1
D (E[a1a2]− E[a1]E[a2]).
For the ﬁrst term, rewritingcos2(wT z) = 1+cos(2wT z)
2 , similar to the proof of Lemma 5 it can be
shown that
(E[a2
1]− (E[a1]2)≤ (1−e−z2
)2
2 +O(e3z2
)
d .
Second term can be bounded similar to Lemma 2 and here we just sketch an outline. Similar to
the proof of Lemma 2, the variance boils down to computing the expectation ofcos(wT
1 z + wT
2 z).
Using Lemma 4 and summing Taylor’s series we get
⏐⏐⏐⏐E[cos(wT
1 z + wT
2 z)]−e−z2
+e−z2z4
d
⏐⏐⏐⏐≤ ez2
z4(z4 + 4z2 + 2)
d2 .
Substituting the above bound and the expectation from Lemma 5, we get
E[a1a2]− E[a1]E[a2]≤−ez2z4
2d +O(e3z2
)
d2 ,
and hence the lemma.
Appendix C Proof of Theorem 3
The proof follows from the following two technical lemmas.
Lemma 7. Letz′ be distributed according to N(0,||x||2
2) andy′ = ∑d
i=1x(i)di, where dis are
independent Rademacher random variables. For any functiong such that|g′|≤ 1 and|g|≤ 1,
|E[g(z′)]− E[g(y′)]|≤ 3
2
d∑
i=1
x3(i)
||x||2
2
.
Proof. Letz =z′/||x||2,y =y′/||x||2, andh(x) =g(||x||2x), for allx. Hence h(z) =g(z′) and
h(y) =g(y′). By a lemma due to Stein [5],
|E[g(z′)]− E[g(y′)]| =|E[h(z)]− E[h(y)]|
≤ supf{|E[f′(y)−yf (y)]| :|f|∞≤|| x||2,|f′|∞≤
√
2/π||x||2,|f′′|∞≤ 2||x||2}.
14

<!-- page 15 -->

We now bound the term on the right hand side by classic Stein-type arguments.
E[yf (y)] =
d∑
i=1
x(i)di
||x||2
E[f(y)].
Letyi =y− x(i)di
||x||2
. Observe that
E[dif(y)] = E[di(f(y)−f(yi))]
= E[di(f(y)−f(yi))−di(y−yi)f′(yi)] + E[di(y−yi)f′(yi)],
where the ﬁrst equality follows from the fact thatyi anddi are independent anddi has zero mean. By
Taylor series approximation, the ﬁrst term is bounded by
|E[dif(y)−f(yi)−di(y−yi)f′(yi)]|≤ 1
2(y−yi)2|f′′|∞ = 1
2
x2(i)
||x||2
2
|f′′|∞.
Similarly,
E[di(y−yi)f′(yi)] = x(i)
||x||2
f′(yi).
Combining the above four equations, we get
⏐⏐⏐⏐⏐E
[
yf (y)−
d∑
i=1
x2(i)
||x||2
2
f′(yi)
]⏐⏐⏐⏐⏐≤
d∑
i=1
|x3(i)|
||x||3
2
|f′′|∞.
Similarly, note that
⏐⏐⏐⏐⏐E
[
f′(y)−
d∑
i=1
x2(i)
||x||2
2
f′(yi)
]⏐⏐⏐⏐⏐≤
d∑
i=1
|f′′|∞
x2(i)
||x||2
2
E[|y−yi|] =
d∑
i=1
|f′′|∞
|x3(i)|
||x||3
2
.
Combining the above two equations, we get
||E[yf (y)−f′(y))]|≤ 3|f′′|∞
2
d∑
i=1
|x3(i)|
||x||3
2
.
Substituting the bound on the second moment off yields the result.
Let G be a random matrix with i.i.d. N(0, 1) entries as before. Using the above lemma we show that√
dHD1HD2 behaves like G while computing the bias.
Lemma 8. For a given x, let z = Gx and y =
√
dHD1HD2x. For any function g such that
|g′|≤ 1 and|g|≤ 1,
⏐⏐⏐⏐⏐
1
d
d∑
i=1
E [g(z(i))]− 1
d
d∑
i=1
E [g(y(i))]
⏐⏐⏐⏐⏐≤ 6∥x∥2√
d
.
Proof. By triangle inequality,
⏐⏐⏐⏐⏐
1
d
d∑
i=1
E [g(z(i))]− 1
d
d∑
i=1
E [g(y(i))]
⏐⏐⏐⏐⏐≤ 1
d
d∑
i=1
|E[g(z(i))]− E[g(y(i))]|.
Let u = HD2x. Then for everyi,y(i) = ∑
jH(i,j )D2(j)u(j). Hence by Lemma 7, we can relate
expectation undery to the expectation under Gaussian distribution:
|E[g(z(i))]− E[g(y(i))]|
(a)
=|E[E[g(z(i))]− E[g(y(i))|u]]|
≤ 3
2
d∑
i=1
E
[|u3(i)|
||u||2
2
]
= 3
2
d∑
i=1
E
[|u3(i)|
||x||2
2
]
,
15

<!-- page 16 -->

where the last equality follows from the fact that HD2 does not change rotation and (a) follows from
the law of total expectation. By Cauchy-Schwartz inequality, for eachi
E[|u(i)|3]≤
√
E[u6(i)],
It can be shown that
E[u6(i)]≤ 15||x||6
2
d3 ,
Summing over all the indices yields the lemma.
Theorem 3 follows from the Bochner’s theorem and the fact that cos(·) satisﬁes requirements for
the above lemma. We note that Theorem 3 holds for the matrix
√
dHD1HD2 itself and the third
component HD3 is not necessary to bound the bias.
Appendix D Proof of Theorem 4
To prove Theorem 4, we use the Hanson-Wright Inequality.
Lemma 9 (Hanson-Wright Inequality) . Let X = (X1,...,X n)∈ Rn be a random vector with
independent subgaussian components Xi which satisfy: E[Xi] = 0 and∥Xi∥sg ≤ K for some
constantK >0. Let A∈ Rn×n. Then for anyt> 0 the following holds:
P[|XT AX− E[XT AX]|>t ]≤ 2e
−c min( t2
K4 ∥A∥2
F
, t
K2 ∥A∥2
)
,
for some universal positive constantc> 0.
Proof of Theorem 4. For a vector u, let diag(u) denote the diagonal matrix whose entries correspond
to the entries of u. For a diagonal matrix D, let vec(D) denote the vector corresponding to the
diagonal entries of D. Let v = HD3z and u = HD2v = Hdiag(v)vec(D2). Observe that
√
dHD1HD2HD3z =
√
dHdiag(HD2HD3z)vec(D1).
Hence ˜R =
√
dHdiag(HD2HD3z). Note that all the entries of
√
dH have magnitude 1 and
HD2HD3 do not change norm of the vector. Hence, each row of ˜R has norm||z||2. To prove the
orthogonality of rows of ˜R, we need to show that for anyi andj̸=i,
√
d
d∑
k=1
H(i,k )H(j,k )u2(k)
is small. We ﬁrst show that the expectation of the above quantity is0 and then use the Hanson-Wright
inequality to prove concentration. Let A be a diagonal matrix withkth entry being
√
dH(i,k )H(j,k ).
The above equation can be rewriten as
d∑
k=1
H(i,k )H(j,k )u2(k) = vec(D2)T diag(v)HT AHdiag(v)vec(D2).
Observe that the (l,l ) entry of the HT AH is
d∑
k=1
HT (l,k )A(k,k )H(k,l ) =
d∑
k=1
H(k,l )A(k,k )H(k,l )
= 1
d
d∑
k=1
A(k,k )
=
d∑
k=1
H(i,k )H(j,k ) = 0,
16

<!-- page 17 -->

10 20 30 40 50 60 70 80 90 1000.5
0.6
0.7
0.8
0.9
1
Recall
Number of retrieved points


LSH
CBE
KBE
ours(HDHDHD)
(a) D =d/2
10 20 30 40 50 60 70 80 90 100
0.65
0.7
0.75
0.8
0.85
0.9
0.95Recall
Number of retrieved points


LSH
CBE
KBE
ours(HDHDHD) (b) D =d
1 2 3 4 50
1
2
x 10
−4
MSE
D / d


LSH
CBE
KBE
ours(HDHDHD) (c) MSE
Figure 6: Recall and angular MSE on a 16384-dimensional dataset of natural images [30].
where the last equality follows from observing that the rows of H are orthogonal to each other.
Together with the fact that elements of D2 are independent of each other, we get
E[uT Au] = E[vec(D2)T diag(v)HT AHdiag(v)vec(D2)] = 0,
To prove the concentration result, observe that the entries of vec(D2) are independent and sub-
Gaussian, and hence we can use the Hanson-Wright inequality. To this end, we bound the Frobenius
and the spectral norm of the underlying matrix. For the Frobenius norm, observe that
||diag(v)HT AHdiag(v)||F
(a)
≤ (||v||∞)4||HT AH||F
(b)
= (||v||∞)4||A||F
(c)
= d (||v||∞)4,
where (a) follows by observing that each diag(v) changes the Frobenius norm by at most||v||2
∞, (b)
follows from the fact that H does not change the Frobenius norm, and (c) follows by substituting A.
To bound the spectral norm, observe that
||diag(v)HT AHdiag(v)||2
(a)
≤ (||v||∞)2||HT AH||2
(b)
= (||v||∞)2||A||2
(c)
= (||v||∞)2,
where (a) follows by observing that each diag(v) changes the spectral norm by at most||v||∞, (b)
follows from the fact that rotation does not change the spectral norm, and(c) follows by substitutingA.
Since v = HD3z, by McDiarmid’s inequality, it can be shown that with probability≥ 1−2de−dϵ2/2,
||v||∞≤ϵ||z||2. Hence, by the Hanson-Wright inequality, we get
Pr
(
√
d
d∑
k=1
H(i,k )H(j,k )u2(k)>t||z||2
)
≤ 2de−dϵ2/2 + 2e−c min(t2/(dϵ4),t/ϵ2),
wherec is a constant. Choosingϵ = (t/d)1/3 results in the theorem.
Appendix E Discrete Hadamard-Diagonal Structure in Binary Embedding
Motivated by the recent advances in using structured matrices in binary embedding, we show
empirically that the same type of structured discrete orthogonal matrices (three blocks of Hadamard-
Diagonal matrices) can also be applied to approximate angular distances for high-dimensional data.
Let W∈ RD×d be a random matrix with i.i.d. normally distributed entries. The classic Locality
17

<!-- page 18 -->

Sensitive Hashing (LSH) result shows that the sign nonlinear mapφ :φ(x) = 1√
D sign(Wx) can be
used to approximate the angle, i.e., for any x, y∈ Rd
φ(x)Tφ(y)≈θ(x, y)/π.
We compare random projection based Locality Sensitive Hashing (LSH) [ 4], Circulant Binary
Embedding (CBE) [ 28] and Kronecker Binary Embedding (KBE) [ 30]. We closely follow the
experimental settings of [30]. We choose to compare with [30] because it proposed to use another
type of structured random orthogonal matrix (Kronecker product of orthogonal matrices). As shown
in Figure 6, our result (HDHDHD) provides higher recall and lower angular MSE in comparison with
other methods.
18
