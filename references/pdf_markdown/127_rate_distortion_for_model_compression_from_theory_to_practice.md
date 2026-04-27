# references/127_rate_distortion_for_model_compression_from_theory_to_practice.pdf

<!-- page 1 -->

Rate Distortion For Model Compression:
From Theory To Practice
Weihao Gao 1 Yu-Han Liu 2 Chong Wang 3 Sewoong Oh 4
Abstract
The enormous size of modern deep neural net-
works makes it challenging to deploy those mod-
els in memory and communication limited scenar-
ios. Thus, compressing a trained model without a
signiﬁcant loss in performance has become an in-
creasingly important task. Tremendous advances
has been made recently, where the main techni-
cal building blocks are pruning, quantization, and
low-rank factorization. In this paper, we propose
principled approaches to improve upon the com-
mon heuristics used in those building blocks, by
studying the fundamental limit for model com-
pression via the rate distortion theory. We prove
a lower bound for the rate distortion function for
model compression and prove its achievability
for linear models. Although this achievable com-
pression scheme is intractable in practice, this
analysis motivates a novel objective function for
model compression, which can be used to improve
classes of model compressor such as pruning or
quantization. Theoretically, we prove that the
proposed scheme is optimal for compressing one-
hidden-layer ReLU neural networks. Empirically,
we show that the proposed scheme improves upon
the baseline in the compression-accuracy tradeoff.
1. Introduction
Deep neural networks have been successful, for example,
in the application of computer vision (
Krizhevsky et al. ,
2012), machine translation ( Wu et al., 2016) and game play-
ing (Silver et al. , 2017). With increasing data and compu-
1Department of Electrical and Computer Engineering, Uni-
versity of Illinois at Urbana-Champaign. Work done as an in-
tern in Google. 2Google, Inc. 3Bytedance, Inc. 4Department of
Computer Science, University of Washington. Correspondence
to: Weihao Gao
<wgao9@illinois.edu>, Y u-Han Liu <yuhan-
liu@google.com>, Chong Wang <mr.chongwang@gmail.com>,
Sewoong Oh <sewoong@cs.washington.edu>.
Proceedings of the 36 th International Conference on Machine
Learning, Long Beach, California, PMLR 97, 2019. Copyright
2019 by the author(s).
tational power, the number of weights in practical neural
network model also grows rapidly. For example, in the ap-
plication of image recognition, the LeNet-5 model ( LeCun
et al. , 1998) only has 400K weights. After two decades,
AlexNet ( Krizhevsky et al. , 2012) has more than 60M
weights, and VGG-16 net ( Simonyan & Zisserman, 2014)
has more than 130M weights. Coates et al. (2013) even
tried a neural network with 11B weights. The huge size of
neural networks brings many challenges, including large
storage, difﬁculty in training, and large energy consumption.
In particular, deploying such extreme models to embedded
mobile systems is not feasible.
Several approaches have been proposed to reduce the size of
large neural networks while preserving the performance as
much as possible. Most of those approaches fall into one of
the two broad categories. The ﬁrst category designs novel
network structures with small number of parameters, such as
SqueezeNet (Iandola et al., 2016) and MobileNet ( Howard
et al. , 2017). The other category directly compresses a
given large neural network using pruning, quantization, and
matrix factorization, including ( LeCun et al. , 1990; Hassibi
& Stork , 1993; Han et al. , 2015b;a; Cheng et al. , 2015).
There are also advanced methods to train the neural network
using Bayesian methods to help pruning or quantization at
a later stage, such as ( Ullrich et al. , 2017; Louizos et al. ,
2017; Federici et al. , 2017).
As more and more model compression algorithms are pro-
posed and compression ratio becomes larger and larger, it
motivates us to think about the fundamental question —
How well can we do for model compression? The goal of
model compression is to trade off the number of bits used to
describe the model parameters, and the distortion between
the compressed model and original model. We wonder at
least how many bits is needed to achieve certain distortion?
Despite many successful model compression algorithms,
these theoretical questions still remain unclear.
In this paper, we ﬁll in this gap by bringing tools from rate
distortion theory to identify the fundamental limit on how
much a model can be compressed. Speciﬁcally, we focus on
compression of a pretrained model, rather than designing
new structures or retraining models. Our approach builds
upon rate-distortion theory introduced by Shannon (1959)

<!-- page 2 -->

Rate Distortion For Model Compression: From Theory To Practice
and further developed by Berger (1971). The approach
also connects to modeling neural networks as random vari-
ables in Mandt et al. (2017), which has many practical
usages (Cao et al. , 2018).
Our contribution for model compression is twofold: theo-
retical and practical. We ﬁrst apply theoretical tools from
rate distortion theory to provide a lower bound on the fun-
damental trade-off between rate (number of bits to describe
the model) and distortion between compressed and origi-
nal models, and prove the tightness of the lower bound for
a linear model. This analysis seamlessly incorporate the
structure of the neural network architecture into model com-
pression via backpropagation. Motivated by the theory, we
design an improved objective for compression algorithms
and show that the improved objective gives optimal prun-
ing and quantization algorithm for one-hidden-layer ReLU
neural network, and has better performance in real neural
networks as well.
The rest of the paper is organized as follows.
• In Section 2, we brieﬂy review some previous work on
model compression.
• In Section 3, we introduce the background of the rate
distortion theory for data compression, and formally
state the rate distortion theory for model compression.
• In Section 4, we give a lower bound of the rate distor-
tion function, which quantiﬁes the fundamental limit
for model compression. We then prove that the lower
bound is achievable for linear model.
• In Section 5, motivated by the achievable compressor
for linear model, we proposed an improved objective
for model compression, which takes consideration of
the sturcture of the neural network. We then prove that
the improved objective gives optimal compressor for
one-hidden-layer ReLU neural network.
• In Section 6, we demonstrate the empirical perfor-
mance of the proposed objective on fully-connected
neural networks on MNIST dataset and convolutional
networks on CIFAR dataset.
2. Related work on model compression
The study of model compression of neural networks ap-
peared as long as neural network was invented. Here we
mainly discuss the literature on directly compressing large
models, which are more relevant to our work. They usually
contain three types of methods — pruning, quantization and
matrix factorization.
Pruning methods set unimportant weights to zero to reduce
the number of parameters. Early works of model pruning
includes biased weight decay ( Hanson & Pratt , 1989), op-
timal brain damage ( LeCun et al., 1990) and optimal brain
surgeon (Hassibi & Stork , 1993). Early methods utilize the
Hessian matrix of the loss function to prune the weights,
however, Hessian matrix is inefﬁcient to compute for mod-
ern large neural networks with millions of parameters. More
recently, Han et al. (2015b) proposed an iterative pruning
and retraining algorithm that works for large neural net-
works.
Quantization, or weight sharing methods group the weights
into clusters and use one value to represent the weights in
the same group. This category includes ﬁxed-point quan-
tization by
V anhoucke et al. (2011), vector quantization
by Gong et al. (2014), HashedNets by Chen et al. (2015),
Hessian-weighted quantizaiton by Choi et al. (2016) and
Diameter-regularized Hessian-weighted quantization by Bu
et al. (2019).
Matrix factorization assumes the weight matrix in each layer
could be factored as a low rank matrix plus a sparse matrix.
Hence, storing low rank and sparse matrices is cheaper than
storing the whole matrix. This category includes Denton
et al. (2014) and Cheng et al. (2015).
There are some recent advanced method beyond pruning,
quantization and matrix factorization. Han et al. (2015a)
assembles pruning, quantization and Huffman coding to
achieve better compression rate. Bayesian methods ( Ullrich
et al., 2017; Louizos et al., 2017; Federici et al., 2017) are
also used to retrain the model such that the model has more
space to be compressed. He et al. (2018) uses reinforcement
learning to design a compression algorithm.
Despite these aforementioned works for model compression,
no one has studied the fundamental limit of model compres-
sion, as far as we know. More speciﬁcally, in this paper,
we focus on the study of theory of model compression for
pretrained neural network models and then derive practical
compression algorithms given the proposed theory.
3. Rate distortion theory for model
compression
In this section, we brieﬂy introduce the rate distortion the-
ory for data compression. Then we extend the theory to
compression of model parameters.
3.1. Review of rate distortion theory for data
compression
Rate distortion theory, ﬁrstly introduced by Shannon (1959)
and further developed by Berger (1971), is an important con-
cept in information theory which gives theoretical descrip-
tion of lossy data compression. It addressed the minimum
average number of R bits, to transmit a random variable

<!-- page 3 -->

Rate Distortion For Model Compression: From Theory To Practice
Figure 1. An illustration of encoder and decoder.
such that the receiver can reconstruct the random variable
with distortion D.
Precisely, let X n = {X1, X 2 . . . X n} ∈ X n be i.i.d. ran-
dom variables from distribution PX . An encoder fn :
X n → { 1, 2, . . . , 2nR} maps the message X n into code-
word, and a decoder gn : {1, 2, . . . , 2nR} → X n recon-
struct the message by an estimate ˆX n from the codeword.
See Figure 1 for an illustration.
A distortion function d : X × X → R+ quantiﬁes the
difference of the original and reconstructed message. Dis-
tortion between sequence
X n and ˆX n is deﬁned as the
average distortion of Xi’s and ˆXi’s. Commonly used
distortion function includes Hamming distortion function
d(x, ˆx) = I[x ̸= ˆx] for X = {0, 1} and square distortion
function d(x, ˆx) = ( x − ˆx)2 for X = R.
Now we are ready to deﬁne the rate-distortion function for
data compression.
Deﬁnition 1
A rate-distortion pair (R, D ) is achievable
if there exists a series of (probabilistic) encoder-
decoder
(fn, g n) such that the alphabet of code-
word has size 2nR and the expected distortion
limn→∞ E[d(X n, g n(fn(X n)))] ≤ D.
Deﬁnition 2 Rate-distortion function R(D) equals to the
inﬁmum of rate R such that rate-distortion pair (R, D ) is
achievable.
The main theorem of rate-distortion theory ( Cover &
Thomas (2012, Theorem 10.2.1)) states as follows,
Theorem 1 Rate distortion theorem for data compression.
R(D) = min
P ˆX|X :E[d(X, ˆX)]≤ D
I(X; ˆX) . (1)
The rate distortion quantiﬁes the fundamental limit of data
compression, i.e., at least how many bits are needed to
compress the data, given the quality of the reconstructed
data. Here is an example for rate-distortion function.
Example 1
If X ∼ N (0, σ 2), the rate distortion function
is given by
R(D) =
{
1
2 log2(σ 2/D ) if D ≤ σ 2
0 if D > σ 2 .
If the required distortion D is larger than the variance of the
Gaussian variable σ 2, we simply transmit ˆX = 0; otherwise,
we will transmit ˆX such that ˆX ∼ N (0, σ 2 − D), X − ˆX ∼
N (0, D ) where ˆX and X − ˆX are independent.
3.2. Rate distortion theory for model compression
Now we extend the rate distortion theory for data compres-
sion to model compression. To apply the rate distortion
theory to model compression, we view the weights in the
model as a multi-dimensional random variable
W ∈ Rm
following distribution PW . The randomness comes from
multiple sources including different distributions of training
data, randomness of training data and randomness of train-
ing algorithm. The compressor can also be random hence
we describe the compressor by a conditional probability
P ˆW |W . Now we deﬁne the distortion and rate in model
compression, analogously to the data compression scenario.
Distortion. Assume we have a neural network fw that maps
input x ∈ Rdx to fw(x) in output space S. For regressors,
fw(x) is deﬁned as the output of the neural network on Rdy .
Analogous to the square distortion in data compression, We
deﬁne the distortion to be the expected ℓ2 distance between
fw and f ˆw, i.e.
d(w, ˆw) ≡ EX
[
∥fw(X) − f ˆw(X)∥2
2
]
. (2)
For classﬁers, fw(x) is deﬁned as the output probability
distribution over C classes on the simplex ∆C− 1. We deﬁne
the distortion to be the expected distance between fw and
f ˆw, i.e.
d(w, ˆw) ≡ EX [ D(f ˆw(X)||fw(X)) ] . (3)
Here D could be any statistical distance, including KL diver-
gence, Hellinger distance, total variation distance, etc. Such
a deﬁnition of distortion captures the difference between
the original model and the compressed model, averaged
over data
X, and measures the quality of a compression
algorithm.
Rate.
In data compression, the rate is deﬁned as the de-
scription length of the bits necessary to communicate the
compressed data ˆX. The compressor outputs ˆX from a ﬁnite
code book X . The description consists the code word which
are the indices of ˆx in the code book, and the description of
the code book.
In rate distortion theory, we ignore the code book length.
Since we are transmitting a sequence of data X n, the code
word has to be transmitted for each Xi but the code book is
only transmitted once. In asymptotic setting, the description
length of code book can be ignored, and the rate is deﬁned
as the description length of the code word.
In model compression, we also deﬁne the rate as the code
word length, by assuming that an underlying distribution

<!-- page 4 -->

Rate Distortion For Model Compression: From Theory To Practice
PW of the parameters exists and inﬁnitely many models
whose parameters are i.i.d. from PW will be compressed.
In practice, we only compress the parameters once so there
is no distribution of the parameters. Nevertheless, the rate
distortion theory can also provide important intuitions for
one-time compression, explained in Section 5.
Now we can deﬁne the rate distortion function for model
compression. Analogously to Theorem 1, the rate distortion
function for model compression is deﬁned as follows,
Deﬁnition 3
Rate distortion function for model compres-
sion.
R(D) = min
P ˆW |W :EW, ˆW [d(W, ˆW )]≤ D
I(W ; ˆW ). (4)
In the following sections we establish a lower bound of the
rate-distortion function.
4. Lower bound and achievability for rate
distortion function
In this section, we study the lower bound for rate distortion
function in Deﬁnition 3. We provide a lower bound for the
rate distortion function, and prove that this lower bound is
achivable for linear regression models.
4.1. Lower bound for linear model
Assume that we are going to compress a linear regression
model fw(x) = wT x. We assume that the mean of data
x ∈ Rm is zero and the covariance matrix is diagonal, i.e.,
EX [X 2
i ] = λ x,i > 0 and EX [XiXj] = 0 for i ̸= j. Further-
more, assume that the parameters W ∈ Rm are drawn from
a Gaussian distribution N (0, ΣW ). The following theorem
gives the lower bound of the rate distortion function for the
linear regression model.
Theorem 2
The rate-distortion function of the linear re-
gression model fw(x) = wT x is lower bounded by
R(D) ≥ R(D) = 1
2 log det(ΣW ) −
m∑
i=1
1
2 log(Di),
where
Di =
{
µ/λ x,i ifµ < λ x,iEW [W 2
i ] ,
EW [W 2
i ] if µ ≥ λ x,iEW [W 2
i ] ,
where µ is chosen that ∑m
i=1 λ x,iDi = D.
This lower bound gives rise to a “weighted water-ﬁlling”
approach, which differs from the classical “water-ﬁlling”
for rate distortion of colored Gaussian source in
Cover &
Thomas (2012, Figure 13.7). The details and graphical
explanation of the “weighted water-ﬁlling” can be found in
Appendix A.
4.2. Achievability
We show that, the lower bound give in Theorem 2 is achiev-
able. Precisely, we have the following theorem.
Theorem 3
There exists a class of probabilistic compres-
sors P (D)
ˆW ∗|W such that EPW ◦P (D)
ˆW ∗ |W
[
d(W, ˆW ∗ )
]
= D and
I(W ; ˆW ∗ ) = R(D).
The optimal compressor is Algorithm 1 in Appendix A.
Intuitively, the optimal compressor does the following
• Find the optimal water levels Di for “weighted wa-
ter ﬁlling”, such that the expected distortion D =
EW, ˆW [d(W, ˆW )] = EW, ˆW [ ˆW T ΣX (W − ˆW )] is min-
imized given certain rate.
• Add a noise Zi which is independent of ˆWi = Wi + Zi
and has a variance proportional to the water level. That
is possible since W is Gaussian.
We can check that the compressor makes all the inequalities
become equality, hence achieve the lower bound. The full
proof of the lower bound and achievability can be found in
Appendix A.
5. Improved objective for model compression
In the previous sections, we study the rate-distortion theory
for model compression. In rate-distortion theory, we assume
that there exists a prior distribution PW on the weights W ,
and prove the tightness of the lower bound in the asymptotic
scenario. However, in practice, we only compress one par-
ticular pre-trained model, so there are no prior distribution
of W . Nonetheless, we can still learn something impor-
tant from the achivability of the lower bound, by extracting
two “golden rules” from the optimal algorithm for linear
regression.
5.1. Two golden rules
Recall that for linear regression model, to achieve the
smallest rate given certain distortion (or, equivalently,
achieve the smallest distortion given certain rate), the
optimal compressor need to do the following: (1) ﬁnd
appropriate “water levels” such that the expected distor-
tion
EW, ˆW [d(W, ˆW )] = EW, ˆW ,X[(W T X − ˆW T X)2] =
EW, ˆW [(W − ˆW )T ΣX (W − ˆW )] is minimized. (2) make
sure that ˆWi is independent with Wi − ˆWi, in other words,
EW, ˆW [ ˆW T ΣX (W − ˆW )] = 0 . Hence, we extract the fol-
lowing two “golden rules”:

<!-- page 5 -->

Rate Distortion For Model Compression: From Theory To Practice
1. EW, ˆW [ ˆW T ΣX (W − ˆW )] = 0
2. EW, ˆW [(W − ˆW )T ΣX (W − ˆW )] should be minimized,
given certain rate.
For practical model compression, we adopt these two
“golden rules”, by making the following amendments. First,
we discard the expectation over W and ˆW since there is
only one model to be compressed. Second, the distor-
tion can be written as
d(w, ˆw) = ( w − ˆw)T ΣX (w − ˆw)
only for linear models. For non-linear models, the dis-
tortion function is complicated, but can be approximated
by a simpler formula. For non-linear regression mod-
els, we take ﬁrst order Taylor expansion of the function
f ˆw(x) ≈ fw(x) + ( ˆw − w)T ∇ wfw(x), and have
d(w, ˆw)
= EX
[
∥fw(X) − f ˆw(X)∥2
2
]
≈ EX
[
(w − ˆw)T ∇ wfw(X)(∇ wfw(X))T (w − ˆw)
]
= ( w − ˆw)T Iw(w − ˆw)
where the “weight importance matrix” deﬁned as
Iw = EX
[
∇ wfw(X)(∇ wfw(X))T ]
, (5)
quantiﬁes the relative importance of each weight to the
output. For linear regression models, weight importance
matrix Iw equals to ΣX .
For classiﬁcation models, we will ﬁrst approximate the
KL divergence. Using the Taylor expansion
x log(x/a ) ≈
(x − a) + ( x − a)2/ (2a) for x/a ≈ 1, the KL
divergence DKL (P ||Q) for can be approximated by
DKL (P ||Q) ≈ ∑
i(Pi − Qi) + ( Pi − Qi)2/ (2Pi) =∑
i(Pi − Qi)2/ (2Pi), or in vector form DKL (P ||Q) ≈
1
2 (P − Q)T diag[P − 1](P − Q). Therefore,
d(w, ˆw)
= EX [DKL (f ˆw(X)||fw(X))]
≈ 1
2 EX
[
(fw(X) − f ˆw(X))T diag[f − 1
w (X)]
(fw(X) − f ˆw(X))
]
≈ 1
2 EX
[
(w − ˆw)T (∇ wfw(X))diag[f − 1
w (X)]
(∇ wfw(X))T (w − ˆw)
]
.
So the weight importance matrix is given by
Iw = EX
[
(∇ wfw(X))diag[f − 1
w (X)](∇ wfw(X))T ]
. (6)
This weight importance matrix is also valid for many
other statistical distances, including reverse KL divergence,
Hellinger distance and Jenson-Shannon distance.
Now we deﬁne the two “golden rules” for practical model
compression algorithms,
1. ˆwT Iw(w − ˆw) = 0 ,
2. (w − ˆw)T Iw(w − ˆw) is minimized given certain con-
straints.
In the following subsections we will show the optimality of
the “golden rules” for a one-hidden-layer neural network,
and apply the “golden rules” to derive new objective func-
tion for pruning and quantization.
5.2. Optimality for one-hidden-layer ReLU network
We show that if a compressor of a one-hidden-layer ReLU
network satisﬁes the two “golden rules”, it will be the op-
timal compressor, with respect to mean-square-error. Pre-
cisely, consider the one-hidden layer ReLU neural network
fw(x) = ReLU (wT x), where the distribution of input
x ∈ Rm is N (0, ΣX ). Furthermore, we assume that the
covariance matrix ΣX = diag[ λ x,1, . . . , λ x,m] is diagonal
and λ x,i > 0 for all i. We have the following theorem.
Theorem 4 If compressed weight ˆw∗ satisﬁes ˆw∗ Iw( ˆw∗ −
w) = 0 and
ˆw∗ = arg min
ˆw∈ ˆW
(w − ˆw)T Iw(w − ˆw),
where ˆW is some class of compressors, then
ˆw∗ = arg min
ˆw∈ ˆW
EX
[
(fw(X) − f ˆw(X))2]
.
The proof uses the techniques of Hermite polynomials and
Fourier analysis on Gaussian spaces, inspired by Ge et al.
(2017). The full proof can be found in Appendix B. Gener-
alizing this result to other activation functions and deeper
neural networks are possible future directions.
Here ˆW denotes a class of compressors, with some con-
straints. For example, ˆW could be the class of pruning
algorithms where no more than 50% weights are pruned, or
ˆW could be the class of quantization algorithm where each
weight is quantized to 4 bits. Theoretically, it is not guaran-
teed that the two “golden rules” can be satisﬁed simultane-
ously for every ˆW, but in the following subsection we show
that they can be satisﬁed simultaneously for two of the most
commonly used class of compressors — pruning and quanti-
zation. Hence, minimizing the objective
(w− ˆw)T Iw(w− ˆw)
will be optimal for pruning and quantization.
5.3. Improved objective for pruning and quantization
Pruning and quantization are two most basic and useful
building blocks of modern model compression algorithms,
For example, DeepCompress ( Han et al., 2015a) iteratively
prune, retrain and quantize the neural network and achieve
state-of-the-art performances on large neural networks.

<!-- page 6 -->

Rate Distortion For Model Compression: From Theory To Practice
In pruning algorithms, we choose a subset S ∈ [m] and
set ˆwi = 0 for all i ∈ S and ˆwi = wi for i ̸∈ S. The
compression ratio is evaluated by the proportion of unpruned
weights r = ( m − | S|)/m . Since either ˆwi or wi − ˆwi is
zero, so the ﬁrst “golden rule” is automatically satisﬁed, so
we have the following corollary.
Corollary 1 F or any ﬁxedr, let
ˆw∗
r = arg min
S: d−|S|
d =r
(w − ˆw)T Iw(w − ˆw),
Then
ˆw∗
r = arg min
S: d−|S|
d =r
EX
[
(fw(X) − f ˆw(X))2]
.
In quantization algorithms, we cluster the weights into k
centroids {c1, . . . , c k}. The algorithm optimize the cen-
troids as long as the assignments of each weight Ai ∈ [k].
The ﬁnal compressed weight is given by ˆwi = cAi . Usually
k-means algorithm are utilized to minimize the centroids
and assignments alternatively. The compression ratio of
quantization algorithm is given by
r = mb
m ∑k
j=1
mj
m ⌈log2
m
mj
⌉ + kb
,
where m is the number of weights and b is the number of
bits to represent one weight before quantization (usually
32). By using Huffman coding, the average number of bits
for each weight is given by ∑k
j=1(mj/m )⌈log2(m/m j)⌉
,
where mj is the number of weights assigned to the j-th
cluster. The deﬁnition of compression ratio of pruning and
quantization is consistent since both of them equals to the
number of bits representing compressed model parameters
divided by the number of bits representing original model
parameters.
If we can ﬁnd the optimal quantization algorithm with re-
spect to (w − ˆw)T Iw(w − ˆw), then each centroids cj should
be optimal, i.e.
0 = ∂
∂c j
(w − ˆw)T Iw(w − ˆw)
= − 2

 ∑
i:Ai=j
eT
i

 Iw(w − ˆw)
where ei is the i-th standard basis. Therefore, we have
ˆwIw( ˆw − w) =


k∑
j=1
cj(
∑
i:Ai=j
ei)


T
Iw(w − ˆw)
=
k∑
j=1
cj

 (
∑
i:Ai=j
eT
i )Iw(w − ˆw)

 = 0.
Hence the ﬁrst “golden rule” is satisﬁed if the second
“golden rule” is satisﬁed. So we have
Corollary 2 F or any ﬁxed number of centroids k, let
ˆw∗
k = arg min
{c1,...,ck},A∈ [k]m
(w − ˆw)T Iw(w − ˆw),
then
ˆw∗
k = arg min
{c1,...,ck},A∈ [k]m
EX
[
(fw(X) − f ˆw(X))2]
.
As corollaries of Theorem 4, we proposed to use (w −
ˆw)T Iw(w− ˆw) as the objective for pruning and quantization
algorithms, which can achieve the minimum MSE for one-
hidden-layer ReLU neural network.
6. Experiments
In the previous section, we proved that a pruning or quantiza-
tion algorithm that minimizes the objective
(w− ˆw)T Iw(w−
ˆw) also minimizes the MSE loss for one-hidden-layer ReLU
neural network. In this section, we show that this objective
can also improve pruning and quantization algorithm for
larger neural networks on real data. 1
We test the objectives on the following neural network and
datasets. 2
1. 3-layer fully connected neural network on MNIST.
2.
Convolutional neural network with 5 convolutional
layers and 3 fully connected layers on CIFAR 10 and
CIFAR 100.
In Section 6.1, we use the weight importance matrix for
classiﬁcation in Eq. (6), which is derived by approximat-
ing the distortion of KL-divergence. This weight impor-
tance matrix does not depend on the training labels, so
the induced pruning/quantization algorithms is called “un-
supervised compression”. Furthermore, if the training la-
bels are available, we treat the loss function
Lw(X, Y ) :
X × Y → R+ as the function to be compressed, and de-
rive several pruning/quantization objectives. The induced
pruning/quantization methods are called “supervised com-
pression” and are studied in Section 6.2.
6.1. Unsupervised Compression Experiments
Recall that for classiﬁcation problems, the weight impor-
tance matrix is deﬁned as
Iw = EX
[
∇ wfw(X)diag[f − 1
w (X)](∇ wfw(X))T ]
.
1We leave combinations of pruning, model retraining and quan-
tization like Han et al. (2015a) as future work.
2We load the pretrained models from https://github.
com/aaron-xichen/pytorch-playground .

<!-- page 7 -->

Rate Distortion For Model Compression: From Theory To Practice
For computational simplicity, we drop the off-
diagonal terms of
Iw, and simplify the objective to∑m
i=1 EX [
(∇ wi fw(X))2
fw(X) ](wi − ˆwi)2. To minimize the pro-
posed objective, a pruning algorithm just prune the weights
with smaller EX [
(∇ wi fw(X))2
fw(X) ]w2
i greedily. A quantization
algorithm uses the weighted k-means algorithm ( Choi et al. ,
2016) to ﬁnd the optimal centroids and assignments. We
compare the proposed objective with the baseline objective∑m
i=1(wi − ˆwi)2
, which were used as building blocks
in DeepCompress ( Han et al. , 2015a). We compare the
objectives in Table 6.1.
Name Minimizing objective
Baseline ∑m
i=1(wi − ˆwi)2
Proposed ∑m
i=1 EX [
(∇ wi fw(X))2
fw(X) ](wi − ˆwi)2
Table 1. Comparison of unsupervised compression objectives.
For pruning experiment, we choose the same compression
rate for every convolutional layer and fully-connected layer,
and plot the test accuracy and test cross-entropy loss against
compression rate. For quantization experiment, we choose
the same number of clusters for every convolutional and
fully-connected layer. Also we plot the test accuracy and
test cross-entropy loss against compression rate. To reduce
the variance of estimating the weight importance matrix Iw,
we use the temperature scaling method introduced by Guo
et al. (2017) to improve model calibration.
We show that results of pruning experiment in Figure 2,
and the results of quantization experiment in Figure 3. We
can see that the proposed objective gives better validation
cross-entropy loss than the baseline, for every different com-
pression ratios. The proposed objective also gives better
validation accuracy in most scenarios. Occasionally the
proposed objective can not improve the accuracy (top left of
Figure
2). We conjecture that the reason is the ill-calibration
of the original model. We relegate the results for CIFAR100
in Appendix C.
6.2. Supervised Compression Experiments
In the previous experiment, we only use the training data
to compute the weight importance matrix. But if we can
use the training label as well, we can further improve the
performance of pruning and quantization algorithms. If the
training label is available, we can view the cross-entropy
loss function
L(fw(x), y ) = Lw(x, y ) as a function from
X × Y → R+, and deﬁne the distortion function as
d(w, ˆw) = EX,Y
[
(Lw(X, Y ) − L ˆw(X, Y ))2]
.
Taking ﬁrst order approximation of the loss function gives
the supervised weight importance matrix,
Iw = E
[
∇ wLw(X, Y )(∇ wLw(X, Y ))T ]
.
0
0.2
0.4
0.6
0.8
1.0
0% 5% 10% 15% 20% 25%
uncompressed
baseline
proposed
Accuracy
0
0.2
0.4
0.6
0.8
1.0
0% 20% 40% 60% 80% 100%
uncompressed
baseline
proposed
0.0
0.5
1.0
1.5
2.0
2.5
0% 5% 10% 15% 20% 25%
uncompressed
baseline
proposed
Cross Entropy 0.0
2.5
5.0
7.5
10.0
12.5
0% 20% 40% 60% 80% 100%
uncompressed
baseline
proposed
Compression RatioCompression Ratio
Figure 2. Result for unsupervised pruning experiment. Left: fully-
connected NN on MNIST (Top: test accuracy, Bottom: test cross
entropy). Right: ConvNN on CIFAR10 (Top: test accuracy, Bot-
tom: test cross entropy).
0.94
0.96
0.98
1.0
2% 4% 6% 8% 10%
uncompressed
baseline
proposed
Accuracy
0
0.2
0.4
0.6
0.8
1.0
4% 6% 8% 10% 12%
uncompressed
baseline
proposed
0.0
0.1
0.2
0.3
0.4
0.5
2% 4% 6% 8% 10%
uncompressed
baseline
proposed
Cross Entropy 0.0
1.0
2.0
3.0
4.0
5.0
6.0
4% 6% 8% 10% 12%
uncompressed
baseline
proposed
Compression RatioCompression Ratio
Figure 3. Result for unsupervised quantization experiment. Left:
fully-connected NN on MNIST (Top: test accuracy, Bottom: test
cross entropy). Right: ConvNN on CIFAR10 (Top: test accuracy,
Bottom: test cross entropy).
We write E instead of EX,Y for simplicity. Similarly, we
drop the off-diagonal terms for ease of computation, and
simplify the objective to ∑m
i=1 E[(∇ wi Lw(X, Y ))2](wi −
ˆwi)2
, which is called gradient-based objective. Note that
for well-trained model, the expected value of gradient
E[∇ wLw(X, Y )] is closed to zero, but the second moment
of the gradient E[∇ wLw(X, Y )(∇ wLw(X, Y ))T ] could be
large. We compare this objective with the baseline objective∑m
i=1(wi − ˆwi)2
. We also compare with the hessian-based
objective ∑m
i=1 E[∇ 2
wi Lw(X, Y )](wi − ˆwi)2
, which is used
in (LeCun et al. , 1990) and ( Hassibi & Stork , 1993) for net-
work pruning and ( Choi et al., 2016) for network quantiza-
tion. To estimate the diagonal entries of the Hessian matrix
of the loss function with respect to the model parameters, we
implemented Curvature Propagation ( Martens et al., 2012)
treating each layer and activation as a node. The running
time is proportional to the running time of the usual gradient
back-propagation by a factor that does not depend on the

<!-- page 8 -->

Rate Distortion For Model Compression: From Theory To Practice
size of the model. Manually optimizing the local Hessian
calculation at each node reduces memory usage and allows
us to use larger batch size and larger number of samples for
more accurate estimates.
Furthermore, if we take second order approximation of the
loss function, and drop the off-diagonal terms of the squared
gradient matrix and squared hessian tensor, we have the
following approximation
d(w, ˆw) = E
[
(Lw(X, Y ) − L ˆw(X, Y ))2]
≈ E
[
(∇ wLw(X, Y )T (w − ˆw)
+ 1
2 (w − ˆw)T ∇ 2
wLw(X, Y )(w − ˆw))2]
≈
m∑
i=1
E[(∇ wi Lw(X, Y ))2](wi − ˆwi)2
+ 1
4
m∑
i=1
E[(∇ 2
wi Lw(X, Y ))2](wi − ˆwi)4,
which is called gradient+hessian based objective. For prun-
ing algorithm, we can prune the weights with smaller
E[(∇ wi Lw(X, Y ))2]w2
i + 1
4 E[(∇ 2
wi Lw(X, Y ))2]w4
i
greed-
ily. For quantization algorithm, we use an alternatice mini-
mization algorithm in Appendix C to ﬁnd the minimum. We
conclude the different supervised objectives in Table 6.2.
Name Minimizing objective
Baseline ∑m
i=1(wi − ˆwi)2
Gradient ∑m
i=1 E[(∇ wi Lw(X, Y ))2](wi − ˆwi)2
Hessian ∑m
i=1 E[∇ 2
wi Lw(X, Y )](wi − ˆwi)2
Gradient ∑m
i=1 E[(∇ wi Lw(X, Y ))2](wi − ˆwi)2
+ Hessian
+ 1
4
∑m
i=1 E[(∇ 2
wi Lw(X, Y ))2](wi − ˆwi)4
Table 2. Comparison of supervised compression objectives.
We show that results of pruning experiment in Figure 4, and
quantization experiment in Figure 5. Generally, the gradient
objective and hessian objective both give better performance
than baseline objective , while gradient objective is slightly
than hessian objective at some points. Gradient + hessian
objective gives the best overall performance. We relegate
the results for CIFAR100 in Appendix C.
Remark. Here we deﬁne the supervised distortion function
as d(w, ˆw) = EX,Y
[
(Lw(X, Y ) − L ˆw(X, Y ))2]
, analo-
gously to the distortion of regression. However, since
the goal of classiﬁcation is to minimize the loss function,
the following deﬁnition of distortion function
˜d(w, ˆw) =
EX,Y [L ˆw(X, Y ) − L w(X, Y )] is also valid and has been
adopted in LeCun et al. (1990) and Choi et al. (2016). The
main difference is — d(w, ˆw) focus on the quality of com-
pression algorithm , i.e., how similar is the compressed
model compared to uncompressed model, whereas ˜d(w, ˆw)
focus on the quality of compressed model, i.e. how good
is the compressed model. So d(w, ˆw) is a better criteria for
the compression algorithm. Additionally, by taking second
order approximation of d(w, ˆw), we have gradient+hessian
objective, which shows better empirical performance than
hessian objective, derived by taking second order approxi-
mation of ˜d(w, ˆw).
0
0.2
0.4
0.6
0.8
1.0
0% 5% 10% 15% 20% 25%
uncompressed
baseline
gradient
hessian
gradient+hessian
Accuracy
0
0.2
0.4
0.6
0.8
1.0
0% 20% 40% 60% 80% 100%
uncompressed
baseline
gradient
hessian
gradient+hessian
0.0
0.5
1.0
1.5
2.0
2.5
0% 5% 10% 15% 20% 25%
uncompressed
baseline
gradient
hessian
gradient+hessian
Cross Entropy 0.0
2.5
5.0
7.5
10.0
12.5
0% 20% 40% 60% 80% 100%
uncompressed
baseline
gradient
hessian
gradient+hessian
Compression RatioCompression Ratio
Figure 4. Result for supervised pruning experiment.Left: fully-
connected NN on MNIST (Top: test accuracy, Bottom: test cross
entropy). Right: ConvNN on CIFAR10 (Top: test accuracy, Bot-
tom: test cross entropy).
0.94
0.96
0.98
1.0
2% 4% 6% 8% 10%
uncompressed
baseline
gradient
hessian
gradient+hessian
Accuracy
0
0.2
0.4
0.6
0.8
1.0
4% 6% 8% 10% 12%
uncompressed
baseline
gradient
hessian
gradient+hessian
0.0
0.1
0.2
0.3
0.4
0.5
2% 4% 6% 8% 10%
uncompressed
baseline
gradient
hessian
gradient+hessian
Cross Entropy 0.0
1.0
2.0
3.0
4.0
5.0
6.0
4% 6% 8% 10% 12%
uncompressed
baseline
gradient
hessian
gradient+hessian
Compression RatioCompression Ratio
Figure 5. Result for supervised quantization experiment. Left:
fully-connected NN on MNIST (Top: test accuracy, Bottom: test
cross entropy). Right: ConvNN on CIFAR10 (Top: test accuracy,
Bottom: test cross entropy).
7. Conclusion
In this paper, we investigate the fundamental limit of neural
network model compression algorithms. We prove a lower
bound for the rate distortion function for model compres-
sion, and prove its achievability for linear model. Motivated
by the rate distortion function, we propose the weight im-
portance matrtix, and show that for one-hidden-layer ReLU
network, pruning and quantization that minimizes the pro-
posed objective is optimal. We also show the superiority of
proposed objective in real neural networks.

<!-- page 9 -->

Rate Distortion For Model Compression: From Theory To Practice
Acknowledgement
The authors thank Denny Zhou for initial comments and
helpful discussions. This work is partially supported by
Google and NSF award 1815535.
References
Berger, T. Rate distortion theory: A mathematical basis for
data compression. 1971.
Bu, Y ., Gao, W., Zou, S., and V eeravalli, V . V .
Information-theoretic understanding of population risk
improvement with model compression. arXiv preprint
arXiv:1901.09421, 2019.
Cao, W., Wang, X., Ming, Z., and Gao, J. A review on
neural networks with random weights. Neurocomputing,
275:278–287, 2018.
Chen, W., Wilson, J., Tyree, S., Weinberger, K., and Chen,
Y . Compressing neural networks with the hashing trick.
In International Conference on Machine Learning , pp.
2285–2294, 2015.
Cheng, Y ., Y u, F. X., Feris, R. S., Kumar, S., Choudhary, A.,
and Chang, S.-F. An exploration of parameter redundancy
in deep networks with circulant projections. In Proceed-
ings of the IEEE International Conference on Computer
Vision, pp. 2857–2865, 2015.
Choi, Y ., El-Khamy, M., and Lee, J. Towards the limit of
network quantization. arXiv preprint arXiv:1612.01543,
2016.
Coates, A., Huval, B., Wang, T., Wu, D., Catanzaro, B.,
and Andrew, N. Deep learning with cots hpc systems.
In International Conference on Machine Learning , pp.
1337–1345, 2013.
Cover, T. M. and Thomas, J. A. Elements of information
theory. John Wiley & Sons, 2012.
Denton, E. L., Zaremba, W., Bruna, J., LeCun, Y ., and Fer-
gus, R. Exploiting linear structure within convolutional
networks for efﬁcient evaluation. In Advances in neural
information processing systems, pp. 1269–1277, 2014.
Federici, M., Ullrich, K., and Welling, M. Improved
bayesian compression. arXiv preprint arXiv:1711.06494,
2017.
Ge, R., Lee, J. D., and Ma, T. Learning one-hidden-layer
neural networks with landscape design. arXiv preprint
arXiv:1711.00501, 2017.
Gong, Y ., Liu, L., Y ang, M., and Bourdev, L. Compressing
deep convolutional networks using vector quantization.
arXiv preprint arXiv:1412.6115, 2014.
Guo, C., Pleiss, G., Sun, Y ., and Weinberger, K. Q. On
calibration of modern neural networks. arXiv preprint
arXiv:1706.04599, 2017.
Han, S., Mao, H., and Dally, W. J. Deep compres-
sion: Compressing deep neural networks with pruning,
trained quantization and huffman coding. arXiv preprint
arXiv:1510.00149, 2015a.
Han, S., Pool, J., Tran, J., and Dally, W. Learning both
weights and connections for efﬁcient neural network. In
Advances in neural information processing systems , pp.
1135–1143, 2015b.
Hanson, S. J. and Pratt, L. Y . Comparing biases for minimal
network construction with back-propagation. In Advances
in neural information processing systems , pp. 177–185,
1989.
Hassibi, B. and Stork, D. G. Second order derivatives for
network pruning: Optimal brain surgeon. In Advances
in neural information processing systems , pp. 164–171,
1993.
He, Y ., Lin, J., Liu, Z., Wang, H., Li, L.-J., and Han, S.
Amc: Automl for model compression and acceleration
on mobile devices. 2018.
Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang,
W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets:
Efﬁcient convolutional neural networks for mobile vision
applications. arXiv preprint arXiv:1704.04861, 2017.
Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K.,
Dally, W. J., and Keutzer, K. Squeezenet: Alexnet-level
accuracy with 50x fewer parameters and¡ 0.5 mb model
size. arXiv preprint arXiv:1602.07360, 2016.
Jiao, J., Gao, W., and Han, Y . The nearest neighbor infor-
mation estimator is adaptively near minimax rate-optimal.
arXiv preprint arXiv:1711.08824, 2017.
Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet
classiﬁcation with deep convolutional neural networks.
In Advances in neural information processing systems ,
pp. 1097–1105, 2012.
LeCun, Y ., Denker, J. S., and Solla, S. A. Optimal brain
damage. In Advances in neural information processing
systems, pp. 598–605, 1990.
LeCun, Y ., Bottou, L., Bengio, Y ., and Haffner, P . Gradient-
based learning applied to document recognition. Proceed-
ings of the IEEE , 86(11):2278–2324, 1998.
Louizos, C., Ullrich, K., and Welling, M. Bayesian compres-
sion for deep learning. In Advances in Neural Information
Processing Systems, pp. 3288–3298, 2017.

<!-- page 10 -->

Rate Distortion For Model Compression: From Theory To Practice
Mandt, S., Hoffman, M. D., and Blei, D. M. Stochastic
gradient descent as approximate bayesian inference. The
Journal of Machine Learning Research, 18(1):4873–4907,
2017.
Martens, J., Sutskever, I., and Swersky, K. Estimating the
hessian by back-propagating curvature. arXiv preprint
arXiv:1206.6464, 2012.
McDonald, R. and Schultheiss, P . Information rates of gaus-
sian signals under criteria constraining the error spectrum.
Proceedings of the IEEE , 52(4):415–416, 1964.
Shannon, C. E. Coding theorems for a discrete source with
a ﬁdelity criterion. IRE Nat. Conv. Rec , 4(142-163):1,
1959.
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou,
I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M.,
Bolton, A., et al. Mastering the game of go without
human knowledge. Nature, 550(7676):354, 2017.
Simonyan, K. and Zisserman, A. V ery deep convolu-
tional networks for large-scale image recognition. arXiv
preprint arXiv:1409.1556, 2014.
Ullrich, K., Meeds, E., and Welling, M. Soft weight-
sharing for neural network compression. arXiv preprint
arXiv:1702.04008, 2017.
V anhoucke, V ., Senior, A., and Mao, M. Z. Improving the
speed of neural networks on cpus. In Proc. Deep Learn-
ing and Unsupervised Feature Learning NIPS Workshop ,
volume 1, pp. 4. Citeseer, 2011.
Wu, Y ., Schuster, M., Chen, Z., Le, Q. V ., Norouzi, M.,
Macherey, W., Krikun, M., Cao, Y ., Gao, Q., Macherey,
K., et al. Google’s neural machine translation system:
Bridging the gap between human and machine translation.
arXiv preprint arXiv:1609.08144, 2016.
