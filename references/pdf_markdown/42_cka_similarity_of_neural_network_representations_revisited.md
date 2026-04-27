# references/42_cka_similarity_of_neural_network_representations_revisited.pdf

<!-- page 1 -->

Similarity of Neural Network Representations Revisited
Simon Kornblith 1 Mohammad Norouzi 1 Honglak Lee 1 Geoffrey Hinton 1
Abstract
Recent work has sought to understand the behav-
ior of neural networks by comparing representa-
tions between layers and between different trained
models. We examine methods for comparing neu-
ral network representations based on canonical
correlation analysis (CCA). We show that CCA
belongs to a family of statistics for measuring mul-
tivariate similarity, but that neither CCA nor any
other statistic that is invariant to invertible linear
transformation can measure meaningful similari-
ties between representations of higher dimension
than the number of data points. We introduce
a similarity index that measures the relationship
between representational similarity matrices and
does not suffer from this limitation. This simi-
larity index is equivalent to centered kernel align-
ment (CKA) and is also closely connected to CCA.
Unlike CCA, CKA can reliably identify corre-
spondences between representations in networks
trained from different initializations.
1. Introduction
Across a wide range of machine learning tasks, deep neural
networks enable learning powerful feature representations
automatically from data. Despite impressive empirical ad-
vances of deep neural networks in solving various tasks,
the problem of understanding and characterizing the neu-
ral network representations learned from data remains rel-
atively under-explored. Previous work ( e.g. Advani &
Saxe (2017); Amari et al. (2018); Saxe et al. (2014)) has
made progress in understanding the theoretical dynamics
of the neural network training process. These studies are
insightful, but fundamentally limited, because they ignore
the complex interaction between the training dynamics and
structured data. A window into the network’s representation
can provide more information about the interaction between
machine learning algorithms and data than the value of the
loss function alone.
1Google Brain. Correspondence to: Simon Kornblith<skorn-
blith@google.com>.
Proceedings of the 36 th International Conference on Machine
Learning, Long Beach, California, PMLR 97, 2019. Copyright
2019 by the author(s).
This paper investigates the problem of measuring similari-
ties between deep neural network representations. An effec-
tive method for measuring representational similarity could
help answer many interesting questions, including: (1) Do
deep neural networks with the same architecture trained
from different random initializations learn similar repre-
sentations? (2) Can we establish correspondences between
layers of different network architectures? (3) How simi-
lar are the representations learned using the same network
architecture from different datasets?
We build upon previous studies investigating similarity be-
tween the representations of neural networks (Laakso &
Cottrell, 2000; Li et al., 2015; Raghu et al., 2017; Morcos
et al., 2018; Wang et al., 2018). We are also inspired by the
extensive neuroscience literature that uses representational
similarity analysis (Kriegeskorte et al., 2008a; Edelman,
1998) to compare representations across brain areas (Haxby
et al., 2001; Freiwald & Tsao, 2010), individuals (Connolly
et al., 2012), species (Kriegeskorte et al., 2008b), and be-
haviors (Elsayed et al., 2016), as well as between brains
and neural networks (Yamins et al., 2014; Khaligh-Razavi
& Kriegeskorte, 2014; Sussillo et al., 2015).
Our key contributions are summarized as follows:
• We discuss the invariance properties of similarity in-
dexes and their implications for measuring similarity of
neural network representations.
• We motivate and introduce centered kernel alignment
(CKA) as a similarity index and analyze the relationship
between CKA, linear regression, canonical correlation
analysis (CCA), and related methods (Raghu et al., 2017;
Morcos et al., 2018).
• We show that CKA is able to determine the correspon-
dence between the hidden layers of neural networks
trained from different random initializations and with
different widths, scenarios where previously proposed
similarity indexes fail.
• We verify that wider networks learn more similar repre-
sentations, and show that the similarity of early layers
saturates at fewer channels than later layers. We demon-
strate that early layers, but not later layers, learn similar
representations on different datasets.
arXiv:1905.00414v4  [cs.LG]  19 Jul 2019

<!-- page 2 -->

Similarity of Neural Network Representations Revisited
Problem Statement
LetX∈ Rn×p1 denote a matrix of activations of p1 neu-
rons for n examples, and Y ∈ Rn×p2 denote a matrix of
activations of p2 neurons for the same n examples. We
assume that these matrices have been preprocessed to center
the columns. Without loss of generality we assume that
p1≤p2. We are concerned with the design and analysis of
a scalar similarity indexs(X,Y ) that can be used to com-
pare representations within and across neural networks, in
order to help visualize and understand the effect of different
factors of variation in deep learning.
2. What Should Similarity Be Invariant To?
This section discusses the invariance properties of similarity
indexes and their implications for measuring similarity of
neural network representations. We argue that both intuitive
notions of similarity and the dynamics of neural network
training call for a similarity index that is invariant to orthog-
onal transformation and isotropic scaling, but not invertible
linear transformation.
2.1. Invariance to Invertible Linear Transformation
A similarity index is invariant to invertible linear transfor-
mation ifs(X,Y ) = s(XA,YB ) for any full rankA and
B. If activationsX are followed by a fully-connected layer
f(X) = σ(XW +β), then transforming the activations
by a full rank matrixA asX′ =XA and transforming the
weights by the inverseA−1 asW′ =A−1W preserves the
output of f(X). This transformation does not appear to
change how the network operates, so intuitively, one might
prefer a similarity index that is invariant to invertible linear
transformation, as argued by Raghu et al. (2017).
However, a limitation of invariance to invertible linear trans-
formation is that any invariant similarity index gives the
same result for any representation of width greater than or
equal to the dataset size, i.e.p2≥n. We provide a simple
proof in Appendix A.
Theorem 1. LetX andY ben×p matrices. Suppose s
is invariant to invertible linear transformation in the ﬁrst
argument, i.e.s(X,Z ) = s(XA,Z ) for arbitrary Z and
anyA with rank(A) =p. If rank(X) = rank(Y ) =n, then
s(X,Z ) =s(Y,Z ).
There is thus a practical problem with invariance to invert-
ible linear transformation: Some neural networks, especially
convolutional networks, have more neurons in some layers
than there are examples the training dataset (Springenberg
et al., 2015; Lee et al., 2018; Zagoruyko & Komodakis,
2016). It is somewhat unnatural that a similarity index
could require more examples than were used for training.
A deeper issue is that neural network training is not invari-
Net A PC 2
Net A PC 1
Net B PC 2
Net B PC 1
Examples Colored By Net A Principal Components
Figure 1. First principal components of representations of net-
works trained from different random initializations are similar.
Each example from the CIFAR-10 test set is shown as a dot col-
ored according to the value of the ﬁrst two principal components of
an intermediate layer of one network (left) and plotted on the ﬁrst
two principal components of the same layer of an architecturally
identical network trained from a different initialization (right).
ant to arbitrary invertible linear transformation of inputs
or activations. Even in the linear case, gradient descent
converges ﬁrst along the eigenvectors corresponding to the
largest eigenvalues of the input covariance matrix (LeCun
et al., 1991), and in cases of overparameterization or early
stopping, the solution reached depends on the scale of the
input. Similar results hold for gradient descent training
of neural networks in the inﬁnite width limit (Jacot et al.,
2018). The sensitivity of neural networks training to linear
transformation is further demonstrated by the popularity of
batch normalization (Ioffe & Szegedy, 2015).
Invariance to invertible linear transformation implies that the
scale of directions in activation space is irrelevant. Empiri-
cally, however, scale information is both consistent across
networks and useful across tasks. Neural networks trained
from different random initializations develop representa-
tions with similar large principal components, as shown in
Figure 1. Consequently, Euclidean distances between ex-
amples, which depend primarily upon large principal com-
ponents, are similar across networks. These distances are
meaningful, as demonstrated by the success of perceptual
loss and style transfer (Gatys et al., 2016; Johnson et al.,
2016; Dumoulin et al., 2017). A similarity index that is
invariant to invertible linear transformation ignores this as-
pect of the representation, and assigns the same score to
networks that match only in large principal components or
networks that match only in small principal components.
2.2. Invariance to Orthogonal Transformation
Rather than requiring invariance to any invertible linear
transformation, one could require a weaker condition; in-
variance to orthogonal transformation, i.e. s(X,Y ) =
s(XU,YV ) for full-rank orthonormal matrices U andV
such thatU TU =I andV TV =I.

<!-- page 3 -->

Similarity of Neural Network Representations Revisited
Indexes invariant to orthogonal transformations do not share
the limitations of indexes invariant to invertible linear trans-
formation. When p2 > n, indexes invariant to orthogonal
transformation remain well-deﬁned. Moreover, orthogo-
nal transformations preserve scalar products and Euclidean
distances between examples.
Invariance to orthogonal transformation seems desirable for
neural networks trained by gradient descent. Invariance to
orthogonal transformation implies invariance to permuta-
tion, which is needed to accommodate symmetries of neural
networks (Chen et al., 1993; Orhan & Pitkow, 2018). In
the linear case, orthogonal transformation of the input does
not affect the dynamics of gradient descent training (LeCun
et al., 1991), and for neural networks initialized with rota-
tionally symmetric weight distributions, e.g. i.i.d. Gaussian
weight initialization, training with ﬁxed orthogonal trans-
formations of activations yields the same distribution of
training trajectories as untransformed activations, whereas
an arbitrary linear transformation would not.
Given a similarity index s(·,·) that is invariant to orthog-
onal transformation, one can construct a similarity index
s′(·,·) that is invariant to any invertible linear transforma-
tion by ﬁrst orthonormalizing the columns of X and Y ,
and then applying s(·,·). Given thin QR decompositions
X =QARA andY =QBRB one can construct a similarity
indexs′(X,Y ) =s(QX,QY ), wheres′(·,·) is invariant to
invertible linear transformation because orthonormal bases
with the same span are related to each other by orthonormal
transformation (see Appendix B).
2.3. Invariance to Isotropic Scaling
We expect similarity indexes to be invariant to isotropic scal-
ing, i.e.s(X,Y ) = s(αX,βY ) for anyα,β ∈ R+. That
said, a similarity index that is invariant to both orthogonal
transformation and non-isotropic scaling, i.e. rescaling of
individual features, is invariant to any invertible linear trans-
formation. This follows from the existence of the singular
value decomposition of the transformation matrix. Gener-
ally, we are interested in similarity indexes that are invariant
to isotropic but not necessarily non-isotropic scaling.
3. Comparing Similarity Structures
Our key insight is that instead of comparing multivariate
features of an example in the two representations (e.g. via re-
gression), one can ﬁrst measure the similarity between every
pair of examples in each representation separately, and then
compare the similarity structures. In neuroscience, such
matrices representing the similarities between examples
are called representational similarity matrices (Kriegesko-
rte et al., 2008a). We show below that, if we use an inner
product to measure similarity, the similarity between repre-
sentational similarity matrices reduces to another intuitive
notion of pairwise feature similarity.
Dot Product-Based Similarity. A simple formula relates
dot products between examples to dot products between
features:
⟨vec(XX T), vec(YY T)⟩ = tr(XX TYY T) =||Y TX||2
F.
(1)
The elements ofXX T andYY T are dot products between
the representations of the ith and jth examples, and indi-
cate the similarity between these examples according to the
respective networks. The left-hand side of (1) thus mea-
sures the similarity between the inter-example similarity
structures. The right-hand side yields the same result by
measuring the similarity between features fromX andY ,
by summing the squared dot products between every pair.
Hilbert-Schmidt Independence Criterion. Equation 1
implies that, for centeredX andY :
1
(n− 1)2 tr(XX TYY T) =||cov(XT,Y T)||2
F. (2)
The Hilbert-Schmidt Independence Criterion (Gretton et al.,
2005) generalizes Equations 1 and 2 to inner products
from reproducing kernel Hilbert spaces, where the squared
Frobenius norm of the cross-covariance matrix becomes
the squared Hilbert-Schmidt norm of the cross-covariance
operator. LetKij =k(xi, xj) andLij =l(yi, yj) wherek
andl are two kernels. The empirical estimator of HSIC is:
HSIC(K,L ) = 1
(n− 1)2 tr(KHLH ), (3)
whereH is the centering matrix Hn = In− 1
n 11T. For
linear kernelsk(x, y) =l(x, y) = xTy, HSIC yields (2).
Gretton et al. (2005) originally proposed HSIC as a test
statistic for determining whether two sets of variables are
independent. They prove that the empirical estimator con-
verges to the population value at a rate of 1/√n, and Song
et al. (2007) provide an unbiased estimator. When k and
l are universal kernels, HSIC = 0 implies independence,
but HSIC is not an estimator of mutual information. HSIC
is equivalent to maximum mean discrepancy between the
joint distribution and the product of the marginal distribu-
tions, and HSIC with a speciﬁc kernel family is equivalent
to distance covariance (Sejdinovic et al., 2013).
Centered Kernel Alignment. HSIC is not invariant to
isotropic scaling, but it can be made invariant through nor-
malization. This normalized index is known as centered ker-
nel alignment (Cortes et al., 2012; Cristianini et al., 2002):
CKA(K,L ) = HSIC(K,L )√
HSIC(K,K )HSIC(L,L )
. (4)

<!-- page 4 -->

Similarity of Neural Network Representations Revisited
Invariant to
Invertible Linear Orthogonal Isotropic
Similarity Index Formula Transform Transform Scaling
Linear Reg. (R2
LR) ||QT
YX||2
F/||X||2
F Y only  
CCA (R2
CCA) ||QT
YQX||2
F/p1   
CCA (¯ρCCA) ||QT
YQX||∗/p1   
SVCCA (R2
SVCCA) ||(UYTY )TUXTX||2
F/min(||TX||2
F,||TY||2
F) If same subspace kept  
SVCCA (¯ρSVCCA) ||(UYTY )TUXTX||∗/min(||TX||2
F,||TY||2
F) If same subspace kept  
PWCCA ∑p1
i=1αiρi/||α||1,αi =∑
j|⟨hi, xj⟩|   
Linear HSIC ||Y TX||2
F/(n− 1)2   
Linear CKA ||Y TX||2
F/(||XTX||F||Y TY||F)   
RBF CKA tr(KHLH )/
√
tr(KHKH )tr(LHLH )   ∗
Table 1. Summary of similarity methods investigated. QX andQY are orthonormal bases for the columns of X andY . UX andUY
are the left-singular vectors ofX andY sorted in descending order according to the corresponding singular vectors.||·|| ∗ denotes the
nuclear norm.TX andTY are truncated identity matrices that select left-singular vectors such that the cumulative variance explained
reaches some threshold. For RBF CKA,K andL are kernel matrices constructed by evaluating the RBF kernel between the examples as
in Section 3, andH is the centering matrixHn =In− 1
n 11T. See Appendix C for more detail about each technique.
∗Invariance of RBF CKA to isotropic scaling depends on the procedure used to select the RBF kernel bandwidth parameter. In our
experiments, we selected the bandwidth as a fraction of the median distance, which ensures that the similarity index is invariant to
isotropic scaling.
For a linear kernel, CKA is equivalent to the RV coefﬁcient
(Robert & Escouﬁer, 1976) and to Tucker’s congruence co-
efﬁcient (Tucker, 1951; Lorenzo-Seva & Ten Berge, 2006).
Kernel Selection. Below, we report results of CKA with
a linear kernel and the RBF kernelk(xi, xj) = exp(−||xi−
xj||2
2/(2σ2)). For the RBF kernel, there are several possible
strategies for selecting the bandwidthσ, which controls the
extent to which similarity of small distances is emphasized
over large distances. We setσ as a fraction of the median
distance between examples. In practice, we ﬁnd that RBF
and linear kernels give similar results across most exper-
iments, so we use linear CKA unless otherwise speciﬁed.
Our framework extends to any valid kernel, including ker-
nels equivalent to neural networks (Lee et al., 2018; Jacot
et al., 2018; Garriga-Alonso et al., 2019; Novak et al., 2019).
4. Related Similarity Indexes
In this section, we brieﬂy review linear regression, canon-
ical correlation, and other related methods in the context
of measuring similarity between neural network representa-
tions. We letQX andQY represent any orthonormal bases
for the columns of X and Y , i.e. QX = X(XTX)−1/2,
QY =Y (Y TY )−1/2 or orthogonal transformations thereof.
Table 1 summarizes the formulae and invariance properties
of the indexes used in experiments. For a comprehensive
general review of linear indexes for measuring multivariate
similarity, see Ramsay et al. (1984).
Linear Regression. A simple way to relate neural net-
work representations is via linear regression. One can ﬁt
every feature inX as a linear combination of features from
Y . A suitable summary statistic is the total fraction of vari-
ance explained by the ﬁt:
R2
LR = 1− minB||X−YB||2
F
||X||2
F
=||QT
YX||2
F
||X||2
F
. (5)
We are unaware of any application of linear regression to
measuring similarity of neural network representations, al-
though Romero et al. (2015) used a least squares loss be-
tween activations of two networks to encourage thin and
deep “student” networks to learn functions similar to wide
and shallow “teacher” networks.
Canonical Correlation Analysis (CCA). Canonical cor-
relation ﬁnds bases for two matrices such that, when the
original matrices are projected onto these bases, the cor-
relation is maximized. For 1≤ i≤ p1, the ith canonical
correlation coefﬁcientρi is given by:
ρi = max
wi
X,wi
Y
corr(Xwi
X,Y wi
Y )
subject to ∀j<i Xwi
X⊥Xwj
X
∀j<i Y wi
Y ⊥Y wj
Y.
(6)
The vectors wi
X∈ Rp1 and wi
Y ∈ Rp2 that maximize ρi
are the canonical weights, which transform the original data
into canonical variablesXwi
X andY wi
Y . The constraints
in (6) enforce orthogonality of the canonical variables.
For the purpose of this work, we consider two summary

<!-- page 5 -->

Similarity of Neural Network Representations Revisited
statistics of the goodness of ﬁt of CCA:
R2
CCA =
∑p1
i=1ρ2
i
p1
=||QT
YQX||2
F
p1
(7)
¯ρCCA =
∑p1
i=1ρi
p1
=||QT
YQX||∗
p1
, (8)
where||·|| ∗ denotes the nuclear norm. The mean squared
CCA correlationR2
CCA is also known as Yanai’s GCD mea-
sure (Ramsay et al., 1984), and several statistical pack-
ages report the sum of the squared canonical correlations
p1R2
CCA =∑p1
i=1ρ2
i under the name Pillai’s trace (SAS In-
stitute, 2015; StataCorp, 2015). The mean CCA correlation
¯ρCCA was previously used to measure similarity between
neural network representations in Raghu et al. (2017).
SVCCA. CCA is sensitive to perturbation when the con-
dition number ofX orY is large (Golub & Zha, 1995). To
improve robustness, singular vector CCA (SVCCA) per-
forms CCA on truncated singular value decompositions of
X andY (Raghu et al., 2017; Mroueh et al., 2015; Kuss
& Graepel, 2003). As formulated in Raghu et al. (2017),
SVCCA keeps enough principal components of the input
matrices to explain a ﬁxed proportion of the variance, and
drops remaining components. Thus, it is invariant to invert-
ible linear transformation only if the retained subspace does
not change.
Projection-Weighted CCA. Morcos et al. (2018) pro-
pose a different strategy to reduce the sensitivity of CCA to
perturbation, which they term “projection-weighted canoni-
cal correlation” (PWCCA):
ρPW =
∑c
i=1αiρi∑
i=1αi
αi =
∑
j
|⟨hi, xj⟩|, (9)
where xj is the jth column of X, and hi = Xwi
X is the
vector of canonical variables formed by projectingX to the
ith canonical coordinate frame. As we show in Appendix
C.3, PWCCA is closely related to linear regression, since:
R2
LR =
∑c
i=1α′
iρ2
i∑
i=1α′
i
α′
i =
∑
j
⟨hi, xj⟩2. (10)
Neuron Alignment Procedures. Other work has studied
alignment between individual neurons, rather than align-
ment between subspaces. Li et al. (2015) examined correla-
tion between the neurons in different neural networks, and
attempt to ﬁnd a bipartite match or semi-match that maxi-
mizes the sum of the correlations between the neurons, and
then to measure the average correlations. Wang et al. (2018)
proposed to search for subsets of neurons ˜X ⊂ X and
˜Y ⊂Y such that, to within some tolerance, every neuron
in ˜X can be represented by a linear combination of neu-
rons from ˜Y and vice versa. They found that the maximum
matching subsets are very small for intermediate layers.
Mutual Information. Among non-linear measures, one
candidate is mutual information, which is invariant not only
to invertible linear transformation, but to any invertible trans-
formation. Li et al. (2015) previously used mutual infor-
mation to measure neuronal alignment. In the context of
comparing representations, we believe mutual information
is not useful. Given any pair of representations produced by
deterministic functions of the same input, mutual informa-
tion between either and the input must be at least as large as
mutual information between the representations. Moreover,
in fully invertible neural networks (Dinh et al., 2017; Jacob-
sen et al., 2018), the mutual information between any two
layers is equal to the entropy of the input.
5. Linear CKA versus CCA and Regression
Linear CKA is closely related to CCA and linear regression.
IfX andY are centered, thenQX andQY are also centered,
so:
R2
CCA = CKA(QXQT
X,QYQT
Y )
√p2
p1
. (11)
When performing the linear regression ﬁt ofX with design
matrixY ,R2
LR =||QT
YX||2
F/||X||2
F , so:
R2
LR = CKA(XX T,QYQT
Y )
√p1||XTX||F
||X||2
F
. (12)
When might we prefer linear CKA over CCA? One way
to show the difference is to rewrite X andY in terms of
their singular value decompositionsX =UXΣXV T
X,Y =
UY ΣYV T
Y . Let the ith eigenvector of XX T (left-singular
vector ofX) be indexed as ui
X. ThenR2
CCA is:
R2
CCA =||U T
YUX||2
F/p1 =
p1∑
i=1
p2∑
j=1
⟨ui
X, uj
Y⟩2/p1. (13)
Let theith eigenvalue ofXX T (squared singular value of
X) be indexed asλi
X. Linear CKA can be written as:
CKA(XX T,YY T) = ||Y TX||2
F
||XTX||F||Y TY||F
=
∑p1
i=1
∑p2
j=1λi
Xλj
Y⟨ui
X, uj
Y⟩2
√∑p1
i=1(λi
X)2
√∑p2
j=1(λj
Y )2
.
(14)
Linear CKA thus resembles CCA weighted by the eigen-
values of the corresponding eigenvectors, i.e. the amount
of variance inX orY that each explains. SVCCA (Raghu
et al., 2017) and projection-weighted CCA (Morcos et al.,
2018) were also motivated by the idea that eigenvectors
that correspond to small eigenvalues are less important, but

<!-- page 6 -->

Similarity of Neural Network Representations Revisited
2
4
6
8Layer
CCA (R 2
CCA)
0.2
0.3
0.4
0.5
Linear Regression
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Similarity
2
4
6
8Layer
SVCCA (¯ρ)
0.3
0.4
0.5
0.6
0.7
0.8
SVCCA (R 2
CCA)
0.2
0.3
0.4
0.5
0.6
0.7
Similarity
2 4 6 8
Layer
2
4
6
8Layer
CKA (Linear)
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8
Layer
CKA (RBF 0.4)
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Similarity
Figure 2. CKA reveals consistent relationships between layers of
CNNs trained with different random initializations, whereas CCA,
linear regression, and SVCCA do not. For linear regression, which
is asymmetric, we plotR2 for the ﬁt of the layer on the x-axis with
the layer on the y-axis. Results are averaged over 10 networks on
the CIFAR-10 training set. See Table 2 for a numerical summary.
linear CKA incorporates this weighting symmetrically and
can be computed without a matrix decomposition.
Comparison of (13) and (14) immediately suggests the pos-
sibility of alternative weightings of scalar products between
eigenvectors. Indeed, as we show in Appendix D.1, the sim-
ilarity index induced by “canonical ridge” regularized CCA
(Vinod, 1976), when appropriately normalized, interpolates
betweenR2
CCA, linear regression, and linear CKA.
6. Results
6.1. A Sanity Check for Similarity Indexes
We propose a simple sanity check for similarity indexes:
Given a pair of architecturally identical networks trained
from different random initializations, for each layer in the
ﬁrst network, the most similar layer in the second network
should be the architecturally corresponding layer. We train
10 networks and, for each layer of each network, we com-
pute the accuracy with which we can ﬁnd the corresponding
layer in each of the other networks by maximum similarity.
We then average the resulting accuracies. We compare CKA
with CCA, SVCCA, PWCCA, and linear regression.
Index Accuracy
CCA (¯ρ) 1.4
CCA (R2
CCA) 10.6
SVCCA (¯ρ) 9.9
SVCCA (R2
CCA) 15.1
PWCCA 11.1
Linear Reg. 45.4
Linear HSIC 22.2
CKA (Linear) 99.3
CKA (RBF 0.2) 80.6
CKA (RBF 0.4) 99.1
CKA (RBF 0.8) 99.3
Table 2. Accuracy of identifying corresponding layers based on
maximum similarity for 10 architecturally identical 10-layer CNNs
trained from different initializations, with logits layers excluded.
For SVCCA, we used a truncation threshold of 0.99 as recom-
mended in Raghu et al. (2017). For asymmetric indexes (PWCCA
and linear regression) we symmetrized the similarity asS +ST.
CKA RBF kernel parameters reﬂect the fraction of the median
Euclidean distance used asσ. Results not signiﬁcantly different
from the best result are bold-faced (p< 0.05, jackknife z-test).
We ﬁrst investigate a simple VGG-like convolutional net-
work based on All-CNN-C (Springenberg et al., 2015) (see
Appendix E) on CIFAR-10. Figure 2 and Table 2 show that
CKA passes our sanity check, but other methods perform
substantially worse. For SVCCA, we experimented with
a range of truncation thresholds, but no threshold revealed
the layer structure (Appendix F.2); our results are consistent
with those in Appendix E of Raghu et al. (2017).
We also investigate Transformer networks, where all layers
are of equal width. In Appendix F.1, we show similarity
between the 12 sublayers of the encoders of Transformer
models (Vaswani et al., 2017) trained from different random
initializations. All similarity indexes achieve non-trivial
accuracy and thus pass the sanity check, although RBF CKA
and R2
CCA performed slightly better than other methods.
However, we found that there are differences in feature
scale between representations of feed-forward network and
self-attention sublayers that CCA does not capture because
it is invariant to non-isotropic scaling.
6.2. Using CKA to Understand Network Architectures
CKA can reveal pathology in neural networks representa-
tions. In Figure 3, we show CKA between layers of individ-
ual CNNs with different depths, where layers are repeated
2, 4, or 8 times. Doubling depth improved accuracy, but
greater multipliers hurt accuracy. At 8x depth, CKA indi-
cates that representations of more than half of the network
are very similar to the last layer. We validated that these
later layers do not reﬁne the representation by training anℓ2-
regularized logistic regression classiﬁer on each layer of the
network. Classiﬁcation accuracy in shallower architectures
progressively improves with depth, but for the 8x deeper

<!-- page 7 -->

Similarity of Neural Network Representations Revisited
1 2 3 4 5 6 7 8 9
1
2
3
4
5
6
7
8
9Layer
1x Depth (94.1%)
5 10 15
5
10
15
2x Depth (95.0%)
5 10 15 20 25 30
5
10
15
20
25
30
4x Depth (93.2%)
10 20 30 40 50 60
10
20
30
40
50
60
8x Depth (91.9%)
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Similarity
1 2 3 4 5 6 7 8 9
Layer
0.6
0.8
1.0Accuracy
5 10 15
Layer
0.6
0.8
1.0
5 10 15 20 25 30
Layer
0.6
0.8
1.0
10 20 30 40 50 60
Layer
0.6
0.8
1.0
Figure 3. CKA reveals when depth becomes pathological. Top: Linear CKA between layers of individual networks of different depths on
the CIFAR-10 test set. Titles show accuracy of each network. Later layers of the 8x depth network are similar to the last layer. Bottom:
Accuracy of a logistic regression classiﬁer trained on layers of the same networks is consistent with CKA.
15 30 45 60
Layer
15
30
45
60Layer
All Layers
8 16 24
Layer
8
16
24
Even Layers
8 16 24
Layer
8
16
24
Odd Layers
0.20.30.40.50.60.70.80.91.0
Similarity
Figure 4. Linear CKA between layers of a ResNet-62 model on
the CIFAR-10 test set. The grid pattern in the left panel arises
from the architecture. Right panels show similarity separately for
even layer (post-residual) and odd layer (block interior) activations.
Layers in the same block group (i.e. at the same feature map scale)
are more similar than layers in different block groups.
network, accuracy plateaus less than halfway through the
network. When applied to ResNets (He et al., 2016), CKA
reveals no pathology (Figure 4). We instead observe a grid
pattern that originates from the architecture: Post-residual
activations are similar to other post-residual activations, but
activations within blocks are not.
CKA is equally effective at revealing relationships between
layers of different architectures. Figure 5 shows the relation-
ship between different layers of networks with and without
residual connections. CKA indicates that, as networks are
made deeper, the new layers are effectively inserted in be-
tween the old layers. Other similarity indexes fail to reveal
meaningful relationships between different architectures, as
we show in Appendix F.5.
In Figure 6, we show CKA between networks with differ-
ent layer widths. Like Morcos et al. (2018), we ﬁnd that
increasing layer width leads to more similar representations
between networks. As width increases, CKA approaches 1;
CKA of earlier layers saturates faster than later layers. Net-
works are generally more similar to other networks of the
same width than they are to the widest network we trained.
2 4 6 8 10121416
Plain-18 Layer
2
4
6
8Plain-10 Layer
2 4 6 8 10 12
ResNet-14 Layer
2
4
6
8Plain-10 Layer
5 10 15 20 25 30
ResNet-32 Layer
2
4
6
8
10
12
14
16Plain-18 Layer
5 10 15 20 25 30
ResNet-32 Layer
2
4
6
8
10
12ResNet-14 Layer
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
CKA (Linear)
Figure 5. Linear CKA between layers of networks with different
architectures on the CIFAR-10 test set.
4 16 64 256 1024 4096
Width
0.5
0.6
0.7
0.8
0.9
1.0CKA (Linear)
Similarity with Width 4096
4 16 64 256 1024 4096
Width
Similarity with Same Width
conv1
conv2
conv3
conv4
conv5
conv6
conv7
conv8
Figure 6. Layers become more similar to each other and to wide
networks as width increases, but similarity of earlier layers satu-
rates ﬁrst. Left: Similarity of networks with the widest network
we trained. Middle: Similarity of networks with other networks
of the same width trained from random initialization. All CKA
values are computed between 10 networks on the CIFAR-10 test
set; shaded regions reﬂect jackknife standard error.

<!-- page 8 -->

Similarity of Neural Network Representations Revisited
1 2 3 4 5 6 7 8 9
Layer
0.0
0.2
0.4
0.6
0.8
1.0CKA (Linear)
Similarity on CIFAR-10
1 2 3 4 5 6 7 8 9
Layer
Similarity on CIFAR-100
CIFAR-10 Net vs. CIFAR-10 Net
CIFAR-100 Net vs. CIFAR-100 Net
CIFAR-10 Net vs. CIFAR-100 Net
Untrained vs. CIFAR-10 Net
Untrained vs. CIFAR-100 Net
Figure 7. CKA shows that models trained on different datasets
(CIFAR-10 and CIFAR-100) develop similar representations, and
these representations differ from untrained models. The left panel
shows similarity between the same layer of different models on the
CIFAR-10 test set, while the right panel shows similarity computed
on CIFAR-100 test set. CKA is averaged over 10 models of each
type (45 pairs).
6.3. Similar Representations Across Datasets
CKA can also be used to compare networks trained on dif-
ferent datasets. In Figure 7, we show that models trained on
CIFAR-10 and CIFAR-100 develop similar representations
in their early layers. These representations require training;
similarity with untrained networks is much lower. We fur-
ther explore similarity between layers of untrained networks
in Appendix F.3.
6.4. Analysis of the Shared Subspace
Equation 14 suggests a way to further elucidating what CKA
is measuring, based on the action of one representational
similarity matrix (RSM)YY T applied to the eigenvectors
ui
X of the other RSMXX T. By deﬁnition,XX Tui
X points
in the same direction as ui
X, and its norm||XX Tui
X||2 is
the corresponding eigenvalue. The degree of scaling and
rotation by YY T thus indicates how similar the action of
YY T is toXX T, for each eigenvector of XX T. For visu-
alization purposes, this approach is somewhat less useful
than the CKA summary statistic, since it does not collapse
the similarity to a single number, but it provides a more
complete picture of what CKA measures. Figure 8 shows
that, for large eigenvectors, XX T andYY T have similar
actions, but the rank of the subspace where this holds is
substantially lower than the dimensionality of the activa-
tions. In the penultimate (global average pooling) layer, the
dimensionality of the shared subspace is approximately 10,
which is the number of classes in the CIFAR-10 dataset.
7. Conclusion and Future Work
Measuring similarity between the representations learned
by neural networks is an ill-deﬁned problem, since it is not
entirely clear what aspects of the representation a similarity
Figure 8. The shared subspace of two networks trained on CIFAR-
10 from different random initializations is spanned primarily by the
eigenvectors corresponding to the largest eigenvalues. Each row
represents a different network layer. Note that the average pooling
layer has only 64 units. Left: Scaling of the eigenvectors ui
X of
the RSMXX T from network A by RSMs of networks A and B.
Orange lines show||XX Tui
X||2, i.e. the eigenvalues. Purple dots
show||YY Tui
X||2, the scaling of the eigenvectors of the RSM
of networkA by the RSM of network B. Right: Cosine of the
rotation by the RSM of network B, (ui
X)TYY Tui
X/||YY Tui
X||2.
index should focus on. Previous work has suggested that
there is little similarity between intermediate layers of neu-
ral networks trained from different random initializations
(Raghu et al., 2017; Wang et al., 2018). We propose CKA as
a method for comparing representations of neural networks,
and show that it consistently identiﬁes correspondences be-
tween layers, not only in the same network trained from
different initializations, but across entirely different archi-
tectures, whereas other methods do not. We also provide a
uniﬁed framework for understanding the space of similarity
indexes, as well as an empirical framework for evaluation.
We show that CKA captures intuitive notions of similarity,
i.e. that neural networks trained from different initializa-
tions should be similar to each other. However, it remains
an open question whether there exist kernels beyond the
linear and RBF kernels that would be better for analyzing
neural network representations. Moreover, there are other
potential choices of weighting in Equation 14 that may be
more appropriate in certain settings. We leave these ques-
tions as future work. Nevertheless, CKA seems to be much
better than previous methods at ﬁnding correspondences be-
tween the learned representations in hidden layers of neural
networks.

<!-- page 9 -->

Similarity of Neural Network Representations Revisited
Acknowledgements
We thank Gamaleldin Elsayed, Jaehoon Lee, Paul-Henri
Mignot, Maithra Raghu, Samuel L. Smith, Alex Williams,
and Michael Wu for comments on the manuscript, Rishabh
Agarwal for ideas, and Aliza Elkin for support.
References
Advani, M. S. and Saxe, A. M. High-dimensional dynamics
of generalization error in neural networks. arXiv preprint
arXiv:1710.03667, 2017.
Amari, S.-i., Ozeki, T., Karakida, R., Yoshida, Y ., and
Okada, M. Dynamics of learning in MLP: Natural gradi-
ent and singularity revisited. Neural Computation, 30(1):
1–33, 2018.
Bj¨orck, ˚A. and Golub, G. H. Numerical methods for com-
puting angles between linear subspaces. Mathematics of
Computation, 27(123):579–594, 1973.
Bojar, O., Federmann, C., Fishel, M., Graham, Y ., Haddow,
B., Huck, M., Koehn, P., and Monz, C. Findings of the
2018 Conference on Machine Translation (WMT18). In
EMNLP 2018 Third Conference on Machine Translation
(WMT18), 2018.
Chen, A. M., Lu, H.-m., and Hecht-Nielsen, R. On the
geometry of feedforward neural network error surfaces.
Neural Computation, 5(6):910–927, 1993.
Connolly, A. C., Guntupalli, J. S., Gors, J., Hanke, M.,
Halchenko, Y . O., Wu, Y .-C., Abdi, H., and Haxby, J. V .
The representation of biological classes in the human
brain. Journal of Neuroscience, 32(8):2608–2618, 2012.
Cortes, C., Mohri, M., and Rostamizadeh, A. Algorithms
for learning kernels based on centered alignment.Journal
of Machine Learning Research, 13(Mar):795–828, 2012.
Cristianini, N., Shawe-Taylor, J., Elisseeff, A., and Kandola,
J. S. On kernel-target alignment. In Advances in Neural
Information Processing Systems, 2002.
Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density esti-
mation using real NVP. In International Conference on
Learning Representations, 2017.
Dumoulin, V ., Shlens, J., and Kudlur, M. A learned repre-
sentation for artistic style. In International Conference
on Learning Representations, 2017.
Edelman, S. Representation is representation of similarities.
Behavioral and Brain Sciences, 21(4):449–467, 1998.
Elsayed, G. F., Lara, A. H., Kaufman, M. T., Churchland,
M. M., and Cunningham, J. P. Reorganization between
preparatory and movement population responses in motor
cortex. Nature Communications, 7:13239, 2016.
Freiwald, W. A. and Tsao, D. Y . Functional compartmental-
ization and viewpoint generalization within the macaque
face-processing system. Science, 330(6005):845–851,
2010.
Garriga-Alonso, A., Rasmussen, C. E., and Aitchison, L.
Deep convolutional networks as shallow Gaussian pro-
cesses. In International Conference on Learning Repre-
sentations, 2019.
Gatys, L. A., Ecker, A. S., and Bethge, M. Image style
transfer using convolutional neural networks. In IEEE
Conference on Computer Vision and Pattern Recognition,
2016.
Golub, G. H. and Zha, H. The canonical correlations of
matrix pairs and their numerical computation. In Linear
Algebra for Signal Processing, pp. 27–49. Springer, 1995.
Gretton, A., Bousquet, O., Smola, A., and Sch ¨olkopf, B.
Measuring statistical dependence with Hilbert-Schmidt
norms. In International Conference on Algorithmic
Learning Theory, 2005.
Haxby, J. V ., Gobbini, M. I., Furey, M. L., Ishai, A.,
Schouten, J. L., and Pietrini, P. Distributed and over-
lapping representations of faces and objects in ventral
temporal cortex. Science, 293(5539):2425–2430, 2001.
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
learning for image recognition. In IEEE Conference on
Computer Vision and Pattern Recognition, 2016.
Ioffe, S. and Szegedy, C. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In International Conference on Machine Learning, 2015.
Jacobsen, J.-H., Smeulders, A. W., and Oyallon, E. i-
RevNet: Deep invertible networks. In International Con-
ference on Learning Representations, 2018.
Jacot, A., Gabriel, F., and Hongler, C. Neural tangent ker-
nel: Convergence and generalization in neural networks.
In Advances in Neural Information Processing Systems,
2018.
Johnson, J., Alahi, A., and Fei-Fei, L. Perceptual losses for
real-time style transfer and super-resolution. In European
Conference on Computer Vision, 2016.
Khaligh-Razavi, S.-M. and Kriegeskorte, N. Deep super-
vised, but not unsupervised, models may explain it corti-
cal representation. PLoS Computational Biology, 10(11):
e1003915, 2014.
Kriegeskorte, N., Mur, M., and Bandettini, P. A. Repre-
sentational similarity analysis-connecting the branches of
systems neuroscience. Frontiers in Systems Neuroscience,
2:4, 2008a.

<!-- page 10 -->

Similarity of Neural Network Representations Revisited
Kriegeskorte, N., Mur, M., Ruff, D. A., Kiani, R., Bodurka,
J., Esteky, H., Tanaka, K., and Bandettini, P. A. Matching
categorical object representations in inferior temporal
cortex of man and monkey. Neuron, 60(6):1126–1141,
2008b.
Kuss, M. and Graepel, T. The geometry of kernel canon-
ical correlation analysis. Technical report, Max Planck
Institute for Biological Cybernetics, 2003.
Laakso, A. and Cottrell, G. Content and cluster analysis:
assessing representational similarity in neural systems.
Philosophical Psychology, 13(1):47–76, 2000.
LeCun, Y ., Kanter, I., and Solla, S. A. Second order proper-
ties of error surfaces: Learning time and generalization.
In Advances in Neural Information Processing Systems,
1991.
Lee, J., Sohl-dickstein, J., Pennington, J., Novak, R.,
Schoenholz, S., and Bahri, Y . Deep neural networks
as gaussian processes. In International Conference on
Learning Representations, 2018.
Li, Y ., Yosinski, J., Clune, J., Lipson, H., and Hopcroft, J.
Convergent learning: Do different neural networks learn
the same representations? In NIPS 2015 Workshop on
Feature Extraction: Modern Questions and Challenges,
2015.
Lorenzo-Seva, U. and Ten Berge, J. M. Tucker’s congru-
ence coefﬁcient as a meaningful index of factor similarity.
Methodology, 2(2):57–64, 2006.
Morcos, A., Raghu, M., and Bengio, S. Insights on repre-
sentational similarity in neural networks with canonical
correlation. In Advances in Neural Information Process-
ing Systems, 2018.
Mroueh, Y ., Marcheret, E., and Goel, V . Asymmetri-
cally weighted CCA and hierarchical kernel sentence
embedding for multimodal retrieval. arXiv preprint
arXiv:1511.06267, 2015.
Novak, R., Xiao, L., Bahri, Y ., Lee, J., Yang, G., Abolaﬁa,
D. A., Pennington, J., and Sohl-dickstein, J. Bayesian
deep convolutional networks with many channels are
Gaussian processes. In International Conference on
Learning Representations, 2019.
Orhan, E. and Pitkow, X. Skip connections eliminate singu-
larities. In International Conference on Learning Repre-
sentations, 2018.
Press, W. H. Canonical correlation clariﬁed by
singular value decomposition, 2011. URL
http://numerical.recipes/whp/notes/
CanonCorrBySVD.pdf.
Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J.
SVCCA: Singular vector canonical correlation analysis
for deep learning dynamics and interpretability. In Ad-
vances in Neural Information Processing Systems, 2017.
Ramsay, J., ten Berge, J., and Styan, G. Matrix correlation.
Psychometrika, 49(3):403–423, 1984.
Robert, P. and Escouﬁer, Y . A unifying tool for linear multi-
variate statistical methods: the RV-coefﬁcient. Applied
Statistics, 25(3):257–265, 1976.
Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta,
C., and Bengio, Y . FitNets: Hints for thin deep nets. In
International Conference on Learning Representations,
2015.
SAS Institute. Introduction to Regression Proce-
dures. 2015. URL https://support.sas.
com/documentation/onlinedoc/stat/141/
introreg.pdf.
Saxe, A. M., McClelland, J. L., and Ganguli, S. Exact
solutions to the nonlinear dynamics of learning in deep
linear neural networks. In International Conference on
Learning Representations, 2014.
Sejdinovic, D., Sriperumbudur, B., Gretton, A., and Fuku-
mizu, K. Equivalence of distance-based and RKHS-based
statistics in hypothesis testing. The Annals of Statistics,
pp. 2263–2291, 2013.
Smith, S. L., Turban, D. H., Hamblin, S., and Hammerla,
N. Y . Ofﬂine bilingual word vectors, orthogonal trans-
formations and the inverted softmax. In International
Conference on Learning Representations, 2017.
Song, L., Smola, A., Gretton, A., Borgwardt, K. M., and
Bedo, J. Supervised feature selection via dependence esti-
mation. In International Conference on Machine learning,
2007.
Springenberg, J. T., Dosovitskiy, A., Brox, T., and Ried-
miller, M. Striving for simplicity: The all convolutional
net. In International Conference on Learning Represen-
tations Workshop, 2015.
StataCorp. Stata Multivariate Statistics Reference Man-
ual. 2015. URL https://www.stata.com/
manuals14/mv.pdf.
Sussillo, D., Churchland, M. M., Kaufman, M. T., and
Shenoy, K. V . A neural network that ﬁnds a naturalistic
solution for the production of muscle activity. Nature
Neuroscience, 18(7):1025, 2015.
Tucker, L. R. A method for synthesis of factor analysis
studies. Technical report, Educational Testing Service,
Princeton, NJ, 1951.

<!-- page 11 -->

Similarity of Neural Network Representations Revisited
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Atten-
tion is all you need. In Advances in Neural Information
Processing Systems, pp. 5998–6008, 2017.
Vaswani, A., Bengio, S., Brevdo, E., Chollet, F., Gomez,
A. N., Gouws, S., Jones, L., Kaiser, Ł., Kalchbrenner,
N., Parmar, N., et al. Tensor2tensor for neural machine
translation. arXiv preprint arXiv:1803.07416, 2018.
Vinod, H. D. Canonical ridge and econometrics of joint pro-
duction. Journal of Econometrics, 4(2):147–166, 1976.
Wang, L., Hu, L., Gu, J., Wu, Y ., Hu, Z., He, K., and
Hopcroft, J. E. Towards understanding learning repre-
sentations: To what extent do different neural networks
learn the same representation. In Advances in Neural
Information Processing Systems, 2018.
Yamins, D. L., Hong, H., Cadieu, C. F., Solomon, E. A.,
Seibert, D., and DiCarlo, J. J. Performance-optimized
hierarchical models predict neural responses in higher
visual cortex. Proceedings of the National Academy of
Sciences, 111(23):8619–8624, 2014.
Zagoruyko, S. and Komodakis, N. Wide residual networks.
In British Machine Vision Conference, 2016.

<!-- page 12 -->

Similarity of Neural Network Representations Revisited
A. Proof of Theorem 1
Theorem. LetX andY ben×p matrices. Supposes is invariant to invertible linear transformation in the ﬁrst argument,
i.e.s(X,Z ) = s(XA,Z ) for arbitrary Z and any A with rank(A) = p. If rank(X) = rank(Y ) = n, then s(X,Z ) =
s(Y,Z ).
Proof. Let
X′ =
[X
KX
]
Y′ =
[Y
KY
]
,
whereKX is a basis for the null space of the rows of X andKY is a basis for the null space of the rows of Y . Then let
A =X′−1Y′. [X
KX
]
A =
[Y
KY
]
=⇒ XA =Y.
BecauseX′ andY′ have rankp by construction,A also has rankp. Thus,s(X,Z ) =s(XA,Z ) =s(Y,Z ).
B. Orthogonalization and Invariance to Invertible Linear Transformation
Here we show that any similarity index that is invariant to orthogonal transformation can be made invariant to invertible
linear transformation by orthogonalizing the columns of the input.
Proposition 1. LetX be ann×p matrix of full column rank and letA be an invertiblep×p matrix. LetX =QXRX and
XA =QXARXA, whereQT
XQX =QT
XAQXA =I andRX andRXA are invertible. Ifs(·,·) is invariant to orthogonal
transformation, thens(QX,Y ) =s(QXA,Y ).
Proof. LetB =RXAR−1
XA. ThenQXB =QXA, and B is an orthogonal transformation:
BTB =BTQT
XQXB =QT
XAQXA =I.
Thuss(QX,Y ) =s(QXB,Y ) =s(QXA,Y ).
C. CCA and Linear Regression
C.1. Linear Regression
Consider the linear regression ﬁt of the columns of ann×m matrixC with ann×p matrixA:
ˆB = arg min
B
||C−AB||2
F = (ATA)−1ATC.
LetA =QR, the thin QR decomposition of A. Then the ﬁtted values are given by:
ˆC =A ˆB
=A(ATA)−1ATC
=QR(RTQTQR)−1RTQTC
=QRR−1(RT)−1RTQTC
=QQTC.
The residualsE =C− ˆC are orthogonal to the ﬁtted values, i.e.
ET ˆC = (C−QQTC)TQQTC
=CTQQTC−CTQQTC = 0.

<!-- page 13 -->

Similarity of Neural Network Representations Revisited
Thus:
||E||2
F = tr(ETE)
= tr(ETC−ET ˆC)
= tr((C− ˆC)TC)
= tr(CTC)− tr(CTQQTC)
=||C||2
F−||QTC||2
F. (15)
Assuming thatC was centered by subtracting its column means prior to the linear regression ﬁt, the total fraction of variance
explained by the ﬁt is:
R2 = 1−||E||2
F
||C||2
F
= 1−||C||2
F−||QTC||2
F
||C||2
F
=||QTC||2
F
||C||2
F
. (16)
Although we have assumed that Q is obtained from QR decomposition, any orthonormal basis with the same span will
sufﬁce, because orthogonal transformations do not change the Frobenius norm.
C.2. CCA
LetX be ann×p1 matrix andY be ann×p2 matrix, and letp = min(p1,p 2). Given the thin QR decompositions ofX
andY ,X =QXRX,Y =QYRY such thatQT
XQX =I,QT
YQY =I, the canonical correlationsρi are the singular values
ofA =QT
XQY (Bj¨orck & Golub, 1973; Press, 2011) and thus the square roots of the eigenvalues of ATA. The squared
canonical correlationsρ2
i are the eigenvalues ofATA =QT
YQXQT
XQY . Their sum is∑p
i=1ρ2
i = tr(ATA) =||QT
YQX||2
F.
Now consider the linear regression ﬁt of the columns ofQX withY . Assume thatQX has zero mean. SubstitutingQY for
Q andQX forC in Equation 16, and noting that||QX||2
F =p1:
R2 =||QT
YQX||2
F
p1
=
∑p
i=1ρ2
i
p1
. (17)
C.3. Projection-Weighted CCA
Let X be an n×p1 matrix and Y be an n×p2 matrix, with p1 ≤ p2. Morcos et al. (2018) proposed to compute
projection-weighted canonical correlation as:
¯ρPW =
∑c
i=1αiρi∑
i=1αi
αi =
∑
j
|⟨hi, xj⟩|,
where the xj are the columns ofX, and the hi are the canonical variables formed by projectingX to the canonical coordinate
frame. Below, we show that if we modify ¯ρPW by squaring the dot products andρi, we recover linear regression.:
R2
MPW =
∑c
i=1α′
iρ2
i∑
i=1α′
i
=R2
LR α′
i =
∑
j
⟨hi, xj⟩2.
Our derivation begins by forming the SVDQT
XQY =UΣV T. Σ is a diagonal matrix of the canonical correlationsρi, and
the matrix of canonical variablesH =QXU. ThenR2
MPW is:
R2
MPW =||XTHΣ||2
F
||XTH||2
F
(18)
= tr((XTHΣ)T(XTHΣ))
tr((XTH)T(XTH))
= tr(ΣHTXX THΣ)
tr(HTXX TH)
= tr(XTHΣ2HTX)
tr(XTHH TX)
= tr(RT
XQT
XHΣ2HTQXRX)
tr(RT
XQT
XQXUU TQT
XQXRX).

<!-- page 14 -->

Similarity of Neural Network Representations Revisited
Because we assume p1 ≤ p2, U is a square orthogonal matrix and UU T = I. Further noting that QT
XH = U and
UΣ =QT
XQYV :
R2
MPW = tr(RT
XUΣ2U TRX)
tr(RT
XQT
XQXRX)
= tr(RT
XQT
XQYV ΣU TRX)
tr(XTX)
= tr(XTQYQT
YQXRX)
tr(XTX)
= tr(XTQYQT
YX)
tr(XTX)
=||QT
YX||2
F
||X||2
F
.
SubstitutingQY forQ andX forC in Equation 16:
R2
LR =||QT
YX||2
F
||X||2
F
=R2
MPW.
D. Notes on Other Methods
D.1. Canonical Ridge
Beyond CCA, we could also consider the “canonical ridge” regularized CCA objective (Vinod, 1976):
σi = max
wi
X,wi
Y
(Xwi
X)T(Y wi
Y )√
||Xwi
X||2 +κX||wi
X||2
2
√
||Y wi
Y||2 +κY||wi
Y||2
subject to ∀j<i (wi
X)T(XTX +κI)wj
X = 0
∀j<i (wi
Y )T(Y TY +κI)wj
Y = 0.
(19)
Given the singular value decompositionsX =UXΣXV T
X andY =UY ΣYV T
Y , one can form “partially orthogonalized” bases
˜QX =UXΣX(Σ2
X+κXI)−1/2 and ˜QY =UY ΣY (Σ2
Y +κYI)−1/2. Given the singular value decomposition of their product
˜U ˜Σ ˜V T = ˜QT
X ˜QY , the canonical weights are given byWX =VX(Σ2
X +κXI)−1/2 ˜U andWY =VY (Σ2
Y +κYI)−1/2 ˜V ,
as previously shown by Mroueh et al. (2015). As in the unregularized case (Equation 13), there is a convenient expression
for the sum of the squared singular values∑˜σ2
i in terms of the eigenvalues and eigenvectors ofXX T andYY T. Let theith
left-singular vector ofX (eigenvector ofXX T) be indexed as ui
X and let theith eigenvalue ofXX T (squared singular value
ofX) be indexed asλi
X, and similarly let the left-singular vectors of YY T be indexed as ui
Y and the eigenvalues asλi
Y .
Then:
p1∑
i=1
˜σ2
i =|| ˜QT
Y ˜QX||2
F (20)
=||(Σ2
Y +κYI)−1/2ΣYU T
YUXΣX(Σ2
X +κXI)−1/2||2
F (21)
=
p1∑
i=1
p2∑
j=1
λi
Xλj
Y
(λi
X +κX)(λj
Y +κY )
⟨ui
X, uj
Y⟩2. (22)
Unlike in the unregularized case, the singular values σi do not measure the correlation between the canonical variables.
Instead, they become arbitrarily small as κX or κY increase. Thus, we need to normalize the statistic to remove the
dependency on the regularization parameters.

<!-- page 15 -->

Similarity of Neural Network Representations Revisited
Applying von Neumann’s trace inequality yields a bound:
p1∑
i=1
˜σ2
i = tr( ˜QY ˜QT
Y ˜QX ˜QT
X) (23)
= tr((UY Σ2
Y (Σ2
Y +κYI)−1U T
Y )(UXΣ2
X(Σ2
X +κXI)−1U T
X)) (24)
≤
p1∑
i=1
λi
Xλi
Y
(λi
X +κX)(λi
Y +κY ). (25)
Applying the Cauchy-Schwarz inequality to (25) yields the alternative bounds:
p1∑
i=1
˜σ2
i≤
√
p1∑
i=1
( λi
X
λi
X +κX
)2
√
p1∑
i=1
( λi
Y
λi
Y +κY
)2
(26)
≤
√
p1∑
i=1
( λi
X
λi
X +κX
)2
√
p2∑
i=1
( λi
Y
λi
Y +κY
)2
. (27)
A normalized form of (22) could be produced by dividing by any of (25), (26), or (27).
IfκX =κY = 0, then (25) and (26) are equal top1. In this case, (22) is simply the sum of the squared canonical correlations,
so normalizing by either of these bounds recoversR2
CCA.
IfκY = 0, then asκX→∞ , normalizing by the bound from (25) recoversR2:
lim
κX→∞
∑p1
i=1
∑p2
j=1
λi
Xλj
Y
(λi
X +κX )(λj
Y +0)⟨ui
X, uj
Y⟩2
∑p1
i=1
λi
Xλi
Y
(λi
X +κX )(λi
Y +0)
(28)
= lim
κX→∞
∑p1
i=1
∑p2
j=1
λi
X(
λi
X
κX
+1
)⟨ui
X, uj
Y⟩2
∑p1
i=1
λi
X(
λi
X
κX
+1
)
(29)
=
∑p1
i=1
∑p2
j=1λi
X⟨ui
X, uj
Y⟩2
∑p1
i=1λi
X
(30)
=||U T
YUXΣX||2
F
||X||2
F
=||QT
YX||2
F
||X||2
F
=R2
LR. (31)
The bound from (27) differs from the bounds in (25) and (26) because it is multiplicatively separable in X and Y .
Normalizing by this bound leads to CKA( ˜QX ˜QT
X, ˜QY ˜QT
Y ):
∑p1
i=1
∑p2
j=1
λi
Xλj
Y
(λi
X +κX )(λj
Y +κY )⟨ui
X, uj
Y⟩2
√
∑p1
i=1
(
λi
X
λi
X +κX
)2
√
∑p2
i=1
(
λi
Y
λi
Y +κY
)2
(32)
= || ˜QT
Y ˜QX||2
F
|| ˜QT
X ˜QX||F|| ˜QT
Y ˜QY||F
= CKA( ˜QX ˜QT
X, ˜QY ˜QT
Y ). (33)

<!-- page 16 -->

Similarity of Neural Network Representations Revisited
Moreover, settingκX =κY =κ and taking the limit asκ→∞ , the normalization from (27) leads to CKA(XX T,YY T):
lim
κ→∞
∑p1
i=1
∑p2
j=1
λi
Xλj
Y
(λi
X +κ)(λj
Y +κ)⟨ui
X, uj
Y⟩2
√
∑p1
i=1
(
λi
X
λi
X +κ
)2
√
∑p2
i=1
(
λi
Y
λi
Y +κ
)2
(34)
= lim
κ→∞
∑p1
i=1
∑p2
j=1
λi
Xλj
Y(
λi
X
κ +1
)(
λj
Y
κ +1
)⟨ui
X, uj
Y⟩2
√
∑p1
i=1
(
λi
X
λi
X
κ +1
)2
√
∑p2
i=1
(
λi
Y
λi
Y
κ +1
)2
(35)
=
∑p1
i=1
∑p2
j=1λi
Xλj
Y⟨ui
X, uj
Y⟩2
√∑p1
i=1
(
λi
X
)2
√∑p2
i=1
(
λi
Y
)2
(36)
= CKA(XX T,YY T).
Overall, the hyperparameters of the canonical ridge objective make it less useful for exploratory analysis. These hyperpa-
rameters could be selected by cross-validation, but this is computationally expensive, and the resulting estimator would be
biased by sample size. Moreover, our goal is not to map representations of networks to a common space, but to measure the
similarity between networks. Appropriately chosen regularization will improve out-of-sample performance of the mapping,
but it makes the meaning of “similarity” more ambiguous.
D.2. The Orthogonal Procrustes Problem
The orthogonal Procrustes problem consists of ﬁnding an orthogonal rotation in feature space that produces the smallest
error:
ˆQ = arg min
Q
||Y−XQ||2
F subject to QTQ =I. (37)
The objective can be written as:
||Y−XQ||2
F = tr((Y−XQ)T(Y−XQ))
= tr(Y TY )− tr(Y TXQ)− tr(QTXTY ) + tr(QTXTXQ)
=||Y||2
F +||X||2
F− 2tr(Y TXQ). (38)
Thus, an equivalent objective is:
ˆQ = arg max
Q
tr(Y TXQ) subject to QTQ =I. (39)
The solution is ˆQ =UV T whereUΣV T =XTY , the singular value decomposition. At the maximum of (39):
tr(Y TX ˆQ) = tr(V ΣU TUV T) = tr(Σ) =||XTY||∗ =||Y TX||∗, (40)
which is similar to what we call “dot product-based similarity” (Equation 1), but with the squared Frobenius norm ofY TX
(the sum of the squared singular values) replaced by the nuclear norm (the sum of the singular values). The Frobenius norm
ofY TX can be obtained as the solution to a similar optimization problem:
||Y TX||F = max
W
tr(Y TXW ) subject to tr(W TW ) = 1. (41)
In the context of neural networks, Smith et al. (2017) previously proposed using the solution to the orthogonal Procrustes
problem to align word embeddings from different languages, and demonstrated that it outperformed CCA.

<!-- page 17 -->

Similarity of Neural Network Representations Revisited
E. Architecture Details
All non-ResNet architectures are based on All-CNN-C (Springenberg et al., 2015), but none are architecturally identical. The
Plain-10 model is very similar, but we place the ﬁnal linear layer after the average pooling layer and use batch normalization
because these are common choices in modern architectures. We use these models because they train in minutes on modern
hardware.
Tiny-10
3× 3 conv. 16-BN-ReLu×2
3× 3 conv. 32 stride 2-BN-ReLu
3× 3 conv. 32-BN-ReLu×2
3× 3 conv. 64 stride 2-BN-ReLu
3× 3 conv. 64 valid padding-BN-ReLu
1× 1 conv. 64-BN-ReLu
Global average pooling
Logits
Table E.1. The Tiny-10 architecture, used in Figures 2, 8, F.3, and F.5. The average Tiny-10 model achieved 89.4% accuracy.
Plain-(8n + 2)
3× 3 conv. 96-BN-ReLu×(3n− 1)
3× 3 conv. 96 stride 2-BN-ReLu
3× 3 conv. 192-BN-ReLu×(3n− 1)
3× 3 conv. 192 stride 2-BN-ReLu
3× 3 conv. 192 BN-ReLu×(n− 1)
3×3 conv. 192 valid padding-BN-ReLu
1× 1 conv. 192-BN-ReLu×n
Global average pooling
Logits
Table E.2. The Plain-(8n + 2) architecture, used in Figures 3, 5, 7, F.4, F.5, F.6, and F.7. Mean accuracies: Plain-10, 93.9%; Plain-18:
94.8%; Plain-34: 93.7%; Plain-66: 91.3%
Width-n
3× 3 conv.n-BN-ReLu×2
3× 3 conv.n stride 2-BN-ReLu
3× 3 conv.n-BN-ReLu×2
3× 3 conv.n stride 2-BN-ReLu
3× 3 conv.n valid padding-BN-ReLu
1× 1 conv.n-BN-ReLu
Global average pooling
Logits
Table E.3. The architectures used for width experiments in Figure 6.

<!-- page 18 -->

Similarity of Neural Network Representations Revisited
F . Additional Experiments
F .1. Sanity Check for Transformer Encoders
CCA (R 2
CCA)
SVCCA (R 2
CCA)
Linear Regression
CKA (Linear)
CKA (RBF 0.4)
3
6
9
12Sublayer
Layer Norm
 Scale
 Attention/FFN
 Residual
3
6
9
12Sublayer
3
6
9
12Sublayer
3
6
9
12Sublayer
3 6 9 12
Sublayer
3
6
9
12Sublayer
3 6 9 12
Sublayer
3 6 9 12
Sublayer
3 6 9 12
Sublayer
Figure F.1. All similarity indices broadly reﬂect the structure of
Transformer encoders. Similarity indexes are computed between
the 12 sublayers of Transformer encoders, for each of the 4 possible
places in each sublayer that representations may be taken (see Fig-
ure F.2), averaged across 10 models trained from different random
initializations.
Layer Normalization
Channel-wise Scale
Self-Attention or
Feed-Forward Network
+Residual
From Previous Sublayer
To Next Sublayer
Figure F.2. Architecture of a single sublayer of the Transformer
encoder used for our experiments. The full encoder includes 12 sub-
layers, alternating between self-attention and feed-forward network
sublayers.
Index Layer Norm Scale Attn/FFN Residual
CCA (¯ρ) 85.3 85.3 94.9 90.9
CCA (R2
CCA) 87.8 87.8 95.3 95.2
SVCCA (¯ρ) 78.2 83.0 89.5 75.9
SVCCA (R2
CCA) 85.4 86.9 90.8 84.7
PWCCA 88.5 88.9 96.1 87.0
Linear Reg. 78.1 83.7 76.0 36.9
CKA (Linear) 78.6 95.6 86.0 73.6
CKA (RBF 0.2) 76.5 73.1 70.5 76.2
CKA (RBF 0.4) 92.3 96.5 89.1 98.1
CKA (RBF 0.8) 80.8 95.8 93.6 90.0
Table F.1. Accuracy of identifying corresponding sublayers based
maximum similarity, for 10 architecturally identical 12-sublayer
Transformer encoders at the 4 locations in each sublayer after which
the representation may be taken (see Figure F.2). Results not sig-
niﬁcantly different from the best result are bold-faced (p <0.05,
jackknife z-test).
When applied to Transformer encoders, all similarity indexes we investigated passed the sanity check described in Section 6.1.
We trained Transformer models using the tensor2tensor library (Vaswani et al., 2018) on the English to German
translation task from the WMT18 dataset (Bojar et al., 2018) (Europarl v7, Common Crawl, and News Commentary v13
corpora) and computed representations of each of the 75,804 tokens from the 3,000 sentencenewstest2013 development
set, ignoring end of sentence tokens. In Figure F.1, we show similarity between the 12 sublayers of the encoders of 10
Transformer models (45 pairs) trained from different random initializations. Each Transformer sublayer contains four
operations, shown in Figure F.2; results vary based which operation the representation is taken after. Table F.1 shows the
accuracy with which we identify corresponding layers between network pairs by maximal similarity.
The Transformer architecture alternates between self-attention and feed-forward network (FFN) sublayers. The checkerboard
pattern in representational similarity after the self-attention/feed-forward network operation in Figure F.1 indicates that
representations of attention sublayers are more similar to other attention sublayers than to FFN sublayers, and similarly,
representations of FFN sublayers are more similar to other FFN than to feed-forward network layers. CKA reveals a
checkerboard pattern for activations after the channel-wise scale operation (before the attention/FFN operation) that other
methods do not. Because CCA is invariant to non-isotropic scaling, CCA similarities before and after channel-wise scaling
are identical. Thus, CCA cannot capture this structure, even though the structure is consistent across networks.

<!-- page 19 -->

Similarity of Neural Network Representations Revisited
F .2. SVCCA at Alternative Thresholds
2
4
6
8Layer
R 2
SVCCA Threshold 0.5
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
R 2
SVCCA Threshold 0.6
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
R 2
SVCCA Threshold 0.7
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
R 2
SVCCA Threshold 0.8
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8
Layer
2
4
6
8Layer
R 2
SVCCA Threshold 0.9
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8
Layer
R 2
SVCCA Threshold 0.99
0.2
0.3
0.4
0.5
0.6
0.7
2 4 6 8
Layer
SVCCA ¯ρ Threshold 0.9
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8
Layer
SVCCA ¯ρ Threshold 0.99
0.3
0.4
0.5
0.6
0.7
0.8
Figure F.3. Same as Figure 2 row 2, but for more SVCCA thresholds than the 0.99 threshold suggested by Raghu et al. (2017). No
threshold reveals the structure of the network.
F .3. CKA at Initialization
5 10 15
Layer
5
10
15Layer
Same Net
5 10 15
Layer
5
10
15
Different Nets
5 10 15
Layer
5
10
15
Vs. Trained
0.2
0.4
0.6
0.8
1.0
CKA (Linear)
Figure F.4. Similarity of the Plain-18 network at initialization.Left:
Similarity between layers of the same network. Middle: Similarity
between untrained networks with different initializations. Right:
Similarity between untrained and trained networks.
5
10
15
20
25
30
Layer
5
10
15
20
25
30Layer
Plain-34
10
20
30
40
50
60
Layer
10
20
30
40
50
60
Plain-66
10
20
30
40
50
60
Layer
10
20
30
40
50
60
ResNet-62
0.2
0.4
0.6
0.8
1.0
CKA (Linear)
Figure F.5. Similarity between layers at initialization for deeper
architectures.
F .4. Additional CKA Results
2 4 6 8
Layer
2
4
6
8Layer
BN vs. BN
2 4 6 8
Layer
No BN vs. No BN
2 4 6 8
Layer
BN vs. No BN
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
CKA (Linear)
Figure F.6. Networks with and without batch normalization trained
from different random initializations learn similar representations
according to CKA. The largest difference between networks is at
the last convolutional layer. Optimal hyperparameters were sepa-
rately selected for the batch normalized network (93.9% average
accuracy) and the network without batch normalization (91.5%
average accuracy).
2 4 6 8
Layer
2
4
6
8Layer
All Examples
2 4 6 8
Layer
Within Class
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
CKA (Linear)
Figure F.7. Within-class CKA is similar to CKA based on all exam-
ples. To measure within-class CKA, we computed CKA separately
for examples belonging to each CIFAR-10 class based on represen-
tations from Plain-10 networks, and averaged the resulting CKA
values across classes.

<!-- page 20 -->

Similarity of Neural Network Representations Revisited
F .5. Similarity Between Different Architectures with Other Indexes
2
4
6
8Tiny-10 Layer
CCA (¯ρ)
0.4
0.5
0.6
0.7
CCA (R 2
CCA)
0.2
0.3
0.4
0.5
0.6
SVCCA (¯ρ)
0.4
0.5
0.6
0.7
0.8
SVCCA (R 2
CCA)
0.2
0.3
0.4
0.5
0.6
2 4 6 8 10 12
ResNet-14 Layer
2
4
6
8Tiny-10 Layer
PWCCA
0.5
0.6
0.7
0.8
2 4 6 8 10 12
ResNet-14 Layer
Linear Regression
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8 10 12
ResNet-14 Layer
CKA (Linear)
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
2 4 6 8 10 12
ResNet-14 Layer
CKA (RBF)
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Figure F.8. Similarity between layers of different architectures (Tiny-10 and ResNet-14) for all methods investigated. Only CKA reveals
meaningful correspondence. SVCCA results resemble Figure 7 of Raghu et al. (2017). In order to achieve better performance for
CCA-based techniques, which are sensitive to the number of examples used to compute similarity, all plots show similarity on the
CIFAR-10 training set.
