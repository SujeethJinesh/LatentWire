# references/70_rate_distortion_for_model_compression_from_theory_to_practice.pdf

<!-- page 1 -->

Rate Distortion For Model Compression: From Theory To Practice
Weihao Gao∗, Yu-Han Liu †, Chong Wang †, Sewoong Oh ‡
January 25, 2019
Abstract
The enormous size of modern deep neural networks makes it challenging to deploy those models in
memory and communication limited scenarios. Thus, compressing a trained model without a signiﬁcant
loss in performance has become an increasingly important task. Tremendous advances has been made
recently, where the main technical building blocks are parameter pruning, parameter sharing (quantization),
and low-rank factorization. In this paper, we propose principled approaches to improve upon the common
heuristics used in those building blocks, namely pruning and quantization.
We ﬁrst study the fundamental limit for model compression via the rate distortion theory. We bring
the rate distortion function from data compression to model compression to quantify this fundamental
limit. We prove a lower bound for the rate distortion function and prove its achievability for linear models.
Although this achievable compression scheme is intractable in practice, this analysis motivates a novel
model compression framework. This framework provides a new objective function in model compression,
which can be applied together with other classes of model compressor such as pruning or quantization.
Theoretically, we prove that the proposed scheme is optimal for compressing one-hidden-layer ReLU
neural networks. Empirically, we show that the proposed scheme improves upon the baseline in the
compression-accuracy tradeoﬀ.
1 Introduction
Deep neural networks have been successful, for example, in the application of computer vision (Krizhevsky
et al., 2012), machine translation (Wu et al., 2016) and game playing (Silver et al., 2017). With increasing
data and computational power, the number of weights in practical neural network model also grows rapidly.
For example, in the application of image recognition, the LeNet-5 model (LeCun et al., 1998) only has 400K
weights. After two decades, AlexNet (Krizhevsky et al., 2012) has more than 60M weights, and VGG-16
net (Simonyan and Zisserman, 2014) has more than 130M weights. Coates et al. (2013) even tried a
neural network with 11B weights. The huge size of neural networks brings many challenges, including large
storage, diﬃculty in training, and large energy consumption. In particular, deploying such extreme models to
embedded mobile systems is not feasible.
Several approaches have been proposed to reduce the size of large neural networks while preserving the
performance as much as possible. Most of those approaches fall into one of the two broad categories. The
ﬁrst category designs novel network structures with small number of parameters, such as SqueezeNet Iandola
et al. (2016) and MobileNet Howard et al. (2017). The other category directly compresses a given large neural
network using pruning, quantization, and matrix factorization, including LeCun et al. (1990); Hassibi and
Stork (1993); Han et al. (2015b,a); Cheng et al. (2015). There are also advanced methods to train the neural
network using Bayesian methods to help pruning or quantization at a later stage, such as Ullrich et al. (2017);
Louizos et al. (2017); Federici et al. (2017).
∗Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign. Email:
wgao9@illinois.edu, work done as an intern in Google Inc.
†Google Inc. Email: {yuhanliu, chongw}@google.com
‡Allen School of Computer Science and Engineering. Email: sewoong@cs.washington.edu
1
arXiv:1810.06401v2  [cs.IT]  23 Jan 2019

<!-- page 2 -->

As more and more model compression algorithms are proposed and compression ratio becomes larger and
larger, it motivates us to think about the fundamental question — How well can we do for model compression?
The goal of model compression is to trade oﬀ the number of bits used to describe the model parameters,
and the distortion between the compressed model and original model. We wonder at least how many bits is
needed to achieve certain distortion? Despite many successful model compression algorithms, these theoretical
questions still remain unclear.
In this paper, we ﬁll in this gap by bringing tools from rate distortion theory to identify the fundamental
limit on how much a model can be compressed. Speciﬁcally, we focus on compression of a pretrained model,
rather than designing new structures or retraining models. Our approach builds upon rate-distortion theory
introduced by Shannon (1959) and further developed by Berger (1971). The approach also connects to
modeling neural networks as random variables in Mandt et al. (2017), which has many practical usages (Cao
et al., 2018).
Our contribution for model compression is twofold: theoretical and practical. We ﬁrst apply theoretical
tools from rate distortion theory to provide a lower bound on the fundamental trade-oﬀ between rate (number
of bits to describe the model) and distortion between compressed and original models, and prove the tightness
of the lower bound for a linear model. This analysis seamlessly incorporate the structure of the neural
network architecture into model compression via backpropagation. Motivated by the theory, we design an
improved objective for compression algorithms and show that the improved objective gives optimal pruning
and quantization algorithm for one-hidden-layer ReLU neural network, and has better performance in real
neural networks as well.
The rest of the paper is organized as follows.
• In Section 2, we brieﬂy review some previous work on model compression.
• In Section 3, we introduce the background of the rate distortion theory for data compression, and
formally state the rate distortion theory for model compression.
• In Section 4, we give a lower bound of the rate distortion function, which quantiﬁes the fundamental
limit for model compression. We then prove that the lower bound is achievable for linear model.
• In Section 5, motivated by the achievable compressor for linear model, we proposed an improved
objective for model compression, which takes consideration of the sturcture of the neural network. We
then prove that the improved objective gives optimal compressor for one-hidden-layer ReLU neural
network.
• In Section 6, we demonstrate the empirical performance of the proposed objective on fully-connected
neural networks on MNIST dataset and convolutional networks on CIFAR dataset.
2 Related work on model compression
The study of model compression of neural networks appeared as long as neural network was invented. Here
we mainly discuss the literature on directly compressing large models, which are more relevant to our work.
They usually contain three types of methods — pruning, quantization and matrix factorization.
Pruning methods set unimportant weights to zero to reduce the number of parameters. Early works of
model pruning includes biased weight decay (Hanson and Pratt, 1989), optimal brain damage (LeCun et al.,
1990) and optimal brain surgeon (Hassibi and Stork, 1993). Early methods utilize the Hessian matrix of the
loss function to prune the weights, however, Hessian matrix is ineﬃcient to compute for modern large neural
networks with millions of parameters. More recently, Han et al. (2015b) proposed an iterative pruning and
retraining algorithm that works for large neural networks.
Quantization, or weight sharing methods group the weights into clusters and use one value to represent
the weights in the same group. This category includes ﬁxed-point quantization by Vanhoucke et al. (2011),
vector quantization by Gong et al. (2014), HashedNets by Chen et al. (2015), Hessian-weighted quantizaiton
by Choi et al. (2016).
2

<!-- page 3 -->

Figure 1: An illustration of encoder and decoder.
Matrix factorization assumes the weight matrix in each layer could be factored as a low rank matrix plus
a sparse matrix. Hence, storing low rank and sparse matrices is cheaper than storing the whole matrix. This
category includes Denton et al. (2014) and Cheng et al. (2015).
There are some recent advanced method beyond pruning, quantization and matrix factorization. Han et al.
(2015a) assembles pruning, quantization and Huﬀman coding to achieve better compression rate. Bayesian
methods Ullrich et al. (2017); Louizos et al. (2017); Federici et al. (2017) are also used to retrain the model
such that the model has more space to be compressed. He et al. (2018) uses reinforcement learning to design
a compression algorithm.
Despite these aforementioned works for model compression, no one has studied the fundamental limit of
model compression, as far as we know. More speciﬁcally, in this paper, we focus on the study of theory of
model compression for pretrained neural network models and then derive practical compression algorithms
given the proposed theory.
3 Rate distortion theory for model compression
In this section, we brieﬂy introduce the rate distortion theory for data compression. Then we extend the
theory to compression of model parameters.
3.1 Review of rate distortion theory for data compression
Rate distortion theory, ﬁrstly introduced by Shannon (1959) and further developed by Berger (1971), is an
important concept in information theory which gives theoretical description of lossy data compression. It
addressed the minimum average number of R bits, to transmit a random variable such that the receiver can
reconstruct the random variable with distortion D.
Precisely, letXn ={X1,X 2...X n}∈X n be i.i.d. random variables from distribution PX. An encoder
fn :Xn→{ 1, 2,..., 2nR} maps the message Xn into codeword, and a decoder gn :{1, 2,..., 2nR}→X n
reconstruct the message by an estimate ˆXn from the codeword. See Figure 1 for an illustration.
A distortion function d :X×X→ R+ quantiﬁes the diﬀerence of the original and reconstructed message.
Distortion between sequence Xn and ˆXn is deﬁned as the average distortion of Xi’s and ˆXi’s. Commonly
used distortion function includes Hamming distortion function d(x, ˆx) = 1 [x̸= ˆx] forX ={0, 1} and square
distortion function d(x, ˆx) = (x− ˆx)2 forX = R.
Now we are ready to deﬁne the rate-distortion function for data compression.
Deﬁnition 1 A rate-distortion pair (R,D ) is achievable if there exists a series of (probabilistic) encoder-
decoder (fn,gn) such that the alphabet of codeword has size 2nR and the expected distortion
limn→∞ E[d(Xn,gn(fn(Xn)))]≤D.
Deﬁnition 2 Rate-distortion function R(D) equals to the inﬁmum of rate R such that rate-distortion pair
(R,D ) is achievable.
The main theorem of rate-distortion theory (Cover and Thomas (2012, Theorem 10.2.1)) states as follows,
3

<!-- page 4 -->

Theorem 1 Rate distortion theorem for data compression.
R(D) = min
P ˆX|X :E[d(X, ˆX)]≤D
I(X; ˆX). (1)
The rate distortion quantiﬁes the fundamental limit of data compression, i.e., at least how many bits are
needed to compress the data, given the quality of the reconstructed data. Here is an example for rate-distortion
function.
Example 1 If X∼N (0,σ 2), the rate distortion function is given by
R(D) =
{
1
2 log2(σ2/D) if D≤σ2
0 if D>σ 2 .
If the required distortion D is larger than the variance of the Gaussian variable σ2, we simply transmit ˆX = 0;
otherwise, we will transmit ˆX such that ˆX∼N (0,σ 2−D), X− ˆX∼N (0,D ) where ˆX and X− ˆX are
independent.
3.2 Rate distortion theory for model compression
Now we extend the rate distortion theory for data compression to model compression. To apply the rate
distortion theory to model compression, we view the weights in the model as a multi-dimensional random
variableW∈ Rm following distribution PW . The randomness comes from multiple sources including diﬀerent
distributions of training data, randomness of training data and randomness of training algorithm. The
compressor can also be random hence we describe the compressor by a conditional probability P ˆW|W . Now
we deﬁne the distortion and rate in model compression, analogously to the data compression scenario.
Distortion. Assume we have a neural network fw that maps input x∈ Rdx to fw(x) in output space
S. For regressors, fw(x) is deﬁned as the output of the neural network on Rdy. Analogous to the square
distortion in data compression, We deﬁne the distortion to be the expected ℓ2 distance between fw and f ˆw,
i.e.
d(w, ˆw)≡ EX
[
∥fw(X)−f ˆw(X)∥2
2
]
. (2)
For classﬁers,fw(x) is deﬁned as the output probability distribution over C classes on the simplex ∆C−1.
We deﬁne the distortion to be the expected distance between fw and f ˆw, i.e.
d(w, ˆw)≡ EX [D(f ˆw(X)||fw(X)) ] . (3)
Here D could be any statistical distance, including KL divergence, Hellinger distance, total variation
distance, etc. Such a deﬁnition of distortion captures the diﬀerence between the original model and the
compressed model, averaged over data X, and measures the quality of a compression algorithm.
Rate. In data compression, the rate is deﬁned as the description length of the bits necessary to
communicate the compressed data ˆX. The compressor outputs ˆX from a ﬁnite code bookX . The description
consists the code word which are the indices of ˆx in the code book, and the description of the code book.
In rate distortion theory, we ignore the code book length. Since we are transmitting a sequence of data Xn,
the code word has to be transmitted for each Xi but the code book is only transmitted once. In asymptotic
setting, the description length of code book can be ignored, and the rate is deﬁned as the description length
of the code word.
In model compression, we also deﬁne the rate as the code word length, by assuming that an underlying
distribution PW of the parameters exists and inﬁnitely many models whose parameters are i.i.d. from PW
will be compressed. In practice, we only compress the parameters once so there is no distribution of the
4

<!-- page 5 -->

parameters. Nevertheless, the rate distortion theory can also provide important intuitions for one-time
compression, explained in Section 5.
Now we can deﬁne the rate distortion function for model compression. Analogously to Theorem 1, the
rate distortion function for model compression is deﬁned as follows,
Deﬁnition 3 Rate distortion function for model compression.
R(D) = min
P ˆW|W :EW, ˆW[d(W, ˆW )]≤D
I(W ; ˆW ). (4)
In the following sections we establish a lower bound of the rate-distortion function.
4 Lower bound and achievability for rate distortion function
In this section, we study the lower bound for rate distortion function in Deﬁnition 3. We provide a lower
bound for the rate distortion function, and prove that this lower bound is achivable for linear regression
models.
4.1 Lower bound for linear model
Assume that we are going to compress a linear regression model fw(x) =wTx. We assume that the mean of
data x∈ Rm is zero and the covariance matrix is diagonal, i.e., EX[X 2
i ] =λx,i > 0 and EX[XiXj] = 0 for
i̸=j. Furthermore, assume that the parameters W∈ Rm are drawn from a Gaussian distribution N (0, ΣW ).
The following theorem gives the lower bound of the rate distortion function for the linear regression model.
Theorem 2 The rate-distortion function of the linear regression model fw(x) =wTx is lower bounded by
R(D)≥R(D) = 1
2 log det(ΣW )−
m∑
i=1
1
2 log(Di),
where
Di =
{
µ/λx,i ifµ<λ x,iEW [W 2
i ],
EW [W 2
i ] if µ≥λx,iEW [W 2
i ],
whereµ is chosen that ∑m
i=1λx,iDi =D.
This lower bound gives rise to a “weighted water-ﬁlling” approach, which diﬀers from the classical
“water-ﬁlling” for rate distortion of colored Gaussian source in Cover and Thomas (2012, Figure 13.7). The
details and graphical explanation of the “weighted water-ﬁlling” can be found in Appendix A.
4.2 Achievability
We show that, the lower bound give in Theorem 2 is achievable. Precisely, we have the following theorem.
Theorem 3 There exists a class of probabilistic compressors P (D)
ˆW∗|W such that EPW◦P (D)
ˆW∗|W
[
d(W, ˆW∗)
]
=D
and I(W ; ˆW∗) =R(D).
The optimal compressor is Algorithm 1 in Appendix A. Intuitively, the optimal compressor does the following
• Find the optimal water levels Di for “weighted water ﬁlling”, such that the expected distortion
D = EW, ˆW [d(W, ˆW )] = EW, ˆW [ ˆWT ΣX(W− ˆW )] is minimized given certain rate.
5

<!-- page 6 -->

• Add a noise Zi which is independent of ˆWi =Wi +Zi and has a variance proportional to the water
level. That is possible since W is Gaussian.
We can check that the compressor makes all the inequalities become equality, hence achieve the lower bound.
The full proof of the lower bound and achievability can be found in Appendix A.
5 Improved objective for model compression
In the previous sections, we study the rate-distortion theory for model compression. In rate-distortion theory,
we assume that there exists a prior distribution PW on the weights W , and prove the tightness of the lower
bound in the asymptotic scenario. However, in practice, we only compress one particular pre-trained model, so
there are no prior distribution of W . Nonetheless, we can still learn something important from the achivability
of the lower bound, by extracting two “golden rules” from the optimal algorithm for linear regression.
5.1 Two golden rules
Recall that for linear regression model, to achieve the smallest rate given certain distortion (or, equivalently,
achieve the smallest distortion given certain rate), the optimal compressor need to do the following: (1) ﬁnd
appropriate “water levels” such that the expected distortion EW, ˆW [d(W, ˆW )] = EW, ˆW,X [(WTX− ˆWTX)2] =
EW, ˆW [(W− ˆW )T ΣX(W− ˆW )] is minimized. (2) make sure that ˆWi is independent with Wi− ˆWi, in other
words, EW, ˆW [ ˆWT ΣX(W− ˆW )] = 0. Hence, we extract the following two “golden rules”:
1. EW, ˆW [ ˆWT ΣX(W− ˆW )] = 0
2. EW, ˆW [(W− ˆW )T ΣX(W− ˆW )] should be minimized, given certain rate.
For practical model compression, we adopt these two “golden rules”, by making the following amendments.
First, we discard the expectation over W and ˆW since there is only one model to be compressed. Second, the
distortion can be written as d(w, ˆw) = (w− ˆw)T ΣX(w− ˆw) only for linear models. For non-linear models, the
distortion function is complicated, but can be approximated by a simpler formula. For non-linear regression
models, we take ﬁrst order Taylor expansion of the function f ˆw(x)≈fw(x) + ( ˆw−w)T∇wfw(x), and have
d(w, ˆw) = EX
[
∥fw(X)−f ˆw(X)∥2
2
]
≈ EX
[
(w− ˆw)T∇wfw(X)(∇wfw(X))T (w− ˆw)
]
= ( w− ˆw)TIw(w− ˆw)
where the “weight importance matrix” deﬁned as
Iw = EX
[
∇wfw(X)(∇wfw(X))T]
, (5)
quantiﬁes the relative importance of each weight to the output. For linear regression models, weight importance
matrix Iw equals to ΣX.
For classiﬁcation models, we will ﬁrst approximate the KL divergence. Using the Taylor expansion
x log(x/a)≈ (x−a) + (x−a)2/(2a) for x/a≈ 1, the KL divergence DKL(P||Q) for can be approximated
byDKL(P||Q)≈∑
i(Pi−Qi) + (Pi−Qi)2/(2Pi) =∑
i(Pi−Qi)2/(2Pi), or in vector form DKL(P||Q)≈
1
2(P−Q)T diag[P−1](P−Q). Therefore,
d(w, ˆw) = EX [DKL(f ˆw(X)||fw(X))]
≈ 1
2 EX
[
(fw(X)−f ˆw(X))T diag[f−1
w (X)](fw(X)−f ˆw(X))
]
≈ 1
2 EX
[
(w− ˆw)T (∇wfw(X))diag[f−1
w (X)](∇wfw(X))T (w− ˆw)
]
.
6

<!-- page 7 -->

So the weight importance matrix is given by
Iw = EX
[
(∇wfw(X))diag[f−1
w (X)](∇wfw(X))T]
. (6)
This weight importance matrix is also valid for many other statistical distances, including reverse KL
divergence, Hellinger distance and Jenson-Shannon distance.
Now we deﬁne the two “golden rules” for practical model compression algorithms,
1. ˆwTIw(w− ˆw) = 0,
2. (w− ˆw)TIw(w− ˆw) is minimized given certain constraints.
In the following subsection we will show the optimality of the “golden rules” for a one-hidden-layer neural
network.
5.2 Optimality for one-hidden-layer ReLU network
We show that if a compressor of a one-hidden-layer ReLU network satisﬁes the two “golden rules”, it will be
the optimal compressor, with respect to mean-square-error. Precisely, consider the one-hidden layer ReLU
neural network fw(x) =ReLU(wTx), where the distribution of input x∈ Rm isN (0, ΣX). Furthermore, we
assume that the covariance matrix ΣX = diag[λx,1,...,λ x,m] is diagonal and λx,i > 0 for all i. We have the
following theorem.
Theorem 4 If compressed weight ˆw∗ satisﬁes ˆw∗Iw( ˆw∗−w) = 0 and
ˆw∗ = arg min
ˆw∈ ˆW
(w− ˆw)TIw(w− ˆw),
where ˆW is some class of compressors, then
ˆw∗ = arg min
ˆw∈ ˆW
EX
[
(fw(X)−f ˆw(X))2]
.
The proof uses the techniques of Hermite polynomials and Fourier analysis on Gaussian spaces, inspired
by Ge et al. (2017). The full proof can be found in Appendix B.
Here ˆW denotes a class of compressors, with some constraints. For example, ˆW could be the class of
pruning algorithms where no more than 50% weights are pruned, or ˆW could be the class of quantization
algorithm where each weight is quantized to 4 bits. Theoretically, it is not guaranteed that the two “golden
rules” can be satisﬁed simultaneously for every ˆW, but in the following subsection we show that they can be
satisﬁed simultaneously for two of the most commonly used class of compressors — pruning and quantization.
Hence, minimizing the objective (w− ˆw)TIw(w− ˆw) will be optimal for pruning and quantization.
5.3 Improved objective for pruning and quantization
Pruning and quantization are two most basic and useful building blocks of modern model compression
algorithms, For example, DeepCompress Han et al. (2015a) iteratively prune, retrain and quantize the neural
network and achieve state-of-the-art performances on large neural networks.
In pruning algorithms, we choose a subset S∈ [m] and set ˆwi = 0 for all i∈S and ˆwi =wi for i̸∈S.
The compression ratio is evaluated by the proportion of unpruned weights r = (m−|S|)/m. Since either ˆwi
or wi− ˆwi is zero, so the ﬁrst “golden rule” is automatically satisﬁed, so we have the following corollary.
Corollary 1 For any ﬁxed r, let
ˆw∗
r = arg min
S: d−|S|
d =r
(w− ˆw)TIw(w− ˆw),
7

<!-- page 8 -->

Then
ˆw∗
r = arg min
S: d−|S|
d =r
EX
[
(fw(X)−f ˆw(X))2]
.
In quantization algorithms, we cluster the weights into k centroids{c1,...,c k}. The algorithm optimize
the centroids as long as the assignments of each weight Ai∈ [k]. The ﬁnal compressed weight is given by
ˆwi =cAi. Usually k-means algorithm are utilized to minimize the centroids and assignments alternatively.
The compression ratio of quantization algorithm is given by
r = mb
m∑k
j=1
mj
m⌈log2
m
mj
⌉ +kb
,
where m is the number of weights and b is the number of bits to represent one weight before quanti-
zation (usually 32). By using Huﬀman coding, the average number of bits for each weight is given by∑k
j=1(mj/m)⌈log2(m/mj)⌉, where mj is the number of weights assigned to the j-th cluster.
If we can ﬁnd the optimal quantization algorithm with respect to ( w− ˆw)TIw(w− ˆw), then each centroids
cj should be optimal, i.e.
0 = ∂
∂cj
(w− ˆw)TIw(w− ˆw) =−2

 ∑
i:Ai=j
eT
i

Iw(w− ˆw)
where ei is the i-th standard basis. Therefore, we have
ˆwIw( ˆw−w) =
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
Iw(w− ˆw) =
k∑
j=1
cj

(
∑
i:Ai=j
eT
i )Iw(w− ˆw)

 = 0.
Hence the ﬁrst “golden rule” is satisﬁed if the second “golden rule” is satisﬁed. So we have
Corollary 2 For any ﬁxed number of centroids k, let
ˆw∗
k = arg min
{c1,...,ck},A∈[k]m
(w− ˆw)TIw(w− ˆw),
then
ˆw∗
k = arg min
{c1,...,ck},A∈[k]m
EX
[
(fw(X)−f ˆw(X))2]
.
As corollaries of Theorem 4, we proposed to use ( w− ˆw)TIw(w− ˆw) as the objective for pruning and
quantization algorithms, which can achieve the minimum MSE for one-hidden-layer ReLU neural network.
6 Experiments
In the previous section, we proved that a pruning or quantization algorithm that minimizes the objective
(w− ˆw)TIw(w− ˆw) also minimizes the MSE loss for one-hidden-layer ReLU neural network. In this section,
we show that this objective can also improve pruning and quantization algorithm for larger neural networks
on real data. 1
We test the objectives on the following neural network and datasets.
1. 3-layer fully connected neural network on MNIST.
1We leave combinations of pruning, model retraining and quantization like Han et al. (2015a) as future work.
8

<!-- page 9 -->

2. Convolutional neural network with 5 convolutional layers and 3 fully connected layers on CIFAR 10
and CIFAR 100.
We load the pretrained models from https://github.com/aaron-xichen/pytorch-playground.
In Section 6.1, we use the weight importance matrix for classiﬁcation in Eq. (6), which is derived by
approximating the distortion of KL-divergence. This weight importance matrix does not depend on the training
labels, so the induced pruning/quantization algorithms is called “unsupervised compression”. Furthermore, if
the training labels are available, we treat the loss function Lw(X,Y ) :X×Y→ R+ as the function to be
compressed, and derive several pruning/quantization objectives. The induced pruning/quantization methods
are called “supervised compression” and are studied in Section 6.2.
6.1 Unsupervised Compression Experiments
Recall that for classiﬁcation problems, the weight importance matrix is deﬁned as
Iw = EX
[
∇wfw(X)diag[f−1
w (X)](∇wfw(X))T]
.
For computational simplicity, we drop the oﬀ-diagonal terms of Iw, and simplify the objective to∑m
i=1 EX[
(∇wifw(X))2
fw(X) ](wi− ˆwi)2. To minimize the proposed objective, a pruning algorithm just prune
the weights with smaller EX[
(∇wifw(X))2
fw(X) ]w2
i greedily. A quantization algorithm uses the weighted k-means
algorithm Choi et al. (2016) to ﬁnd the optimal centroids and assignments. We compare the proposed objective
with the baseline objective ∑m
i=1(wi− ˆwi)2, which were used as building blocks in DeepCompress Han et al.
(2015a). We compare the objectives in Table 6.1.
Name Minimizing objective
Baseline ∑m
i=1(wi− ˆwi)2
Proposed ∑m
i=1 EX[
(∇wifw(X))2
fw(X) ](wi− ˆwi)2
Table 1: Comparison of unsupervised compression objectives.
For pruning experiment, we choose the same compression rate for every convolutional layer and fully-
connected layer, and plot the test accuracy and test cross-entropy loss against compression rate. For
quantization experiment, we choose the same number of clusters for every convolutional and fully-connected
layer. Also we plot the test accuracy and test cross-entropy loss against compression rate. To reduce the
variance of estimating the weight importance matrix Iw, we use the temperature scaling method introduced
by Guo et al. (2017) to improve model calibration.
We show that results of pruning experiment in Figure 2, and the results of quantization experiment in
Figure 3. We can see that the proposed objective gives better validation cross-entropy loss than the baseline,
for every diﬀerent compression ratios. The proposed objective also gives better validation accuracy in most
scenarios. We relegate the results for CIFAR100 in Appendix C.
6.2 Supervised Compression Experiments
In the previous experiment, we only use the training data to compute the weight importance matrix. But if
we can use the training label as well, we can further improve the performance of pruning and quantization
algorithms. If the training label is available, we can view the cross-entropy loss functionL(fw(x),y ) =Lw(x,y )
as a function from X×Y→ R+, and deﬁne the distortion function as
d(w, ˆw) = EX,Y
[
(Lw(X,Y )−L ˆw(X,Y ))2]
.
Taking ﬁrst order approximation of the loss function gives the supervised weight importance matrix,
Iw = E
[
∇wLw(X,Y )(∇wLw(X,Y ))T]
.
9

<!-- page 10 -->

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
proposedCross Entropy
0.0
2.5
5.0
7.5
10.0
12.5
0% 20% 40% 60% 80% 100%
uncompressed
baseline
proposed
Compression Ratio Compression Ratio
Figure 2: Result for unsupervised pruning experiment. Left: fully-connected neural network on MNIST (Top:
test accuracy, Bottom: test cross entropy loss). Right: convolutional neural network on CIFAR10 (Top: test
accuracy, Bottom: test cross entropy loss).
We write E instead of EX,Y for simplicity. Similarly, we drop the oﬀ-diagonal terms for ease of computation,
and simplify the objective to ∑m
i=1 E[(∇wiLw(X,Y ))2](wi− ˆwi)2, which is called gradient-based objective.
Note that for well-trained model, the expected value of gradient E[∇wLw(X,Y )] is closed to zero, but
the second moment of the gradient E[∇wLw(X,Y )(∇wLw(X,Y ))T ] could be large. We compare this
objective with the baseline objective ∑m
i=1(wi− ˆwi)2. We also compare with the hessian-based objective∑m
i=1 E[∇2
wiLw(X,Y )](wi− ˆwi)2, which is used in LeCun et al. (1990) and Hassibi and Stork (1993) for
network pruning and Choi et al. (2016) for network quantization. To estimate the diagonal entries of
the Hessian matrix of the loss function with respect to the model parameters, we implemented Curvature
Propagation Martens et al. (2012) treating each layer and activation as a node. The running time is
proportional to the running time of the usual gradient back-propagation by a factor that does not depend on
the size of the model. Manually optimizing the local Hessian calculation at each node reduces memory usage
and allows us to use larger batch size and larger number of samples for more accurate estimates.
Furthermore, if we take second order approximation of the loss function, and drop the oﬀ-diagonal terms
of the squared gradient matrix and squared hessian tensor, we have the following approximation
d(w, ˆw) = E
[
(Lw(X,Y )−L ˆw(X,Y ))2]
≈ E
[
(∇wLw(X,Y )T (w− ˆw) + 1
2(w− ˆw)T∇2
wLw(X,Y )(w− ˆw))2]
≈
m∑
i=1
E[(∇wiLw(X,Y ))2](wi− ˆwi)2 + 1
4
m∑
i=1
E[(∇2
wiLw(X,Y ))2](wi− ˆwi)4,
which is called gradient+hessian based objective. For pruning algorithm, we can prune the weights with
smaller E[(∇wiLw(X,Y ))2]w2
i + 1
4 E[(∇2
wiLw(X,Y ))2]w4
i greedily. For quantization algorithm, we use an
alternatice minimization algorithm in Appendix C to ﬁnd the minimum. We conclude the diﬀerent supervised
objectives in Table 6.2.
We show that results of pruning experiment in Figure 4, and the results of quantization experiment in
10

<!-- page 11 -->

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
proposedCross Entropy
0.0
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
Compression Ratio Compression Ratio
Figure 3: Result for unsupervised quantization experiment. Left: fully-connected neural network on MNIST
(Top: test accuracy, Bottom: test cross entropy loss). Right: convolutional neural network on CIFAR10 (Top:
test accuracy, Bottom: test cross entropy loss).
Name Minimizing objective
Baseline ∑m
i=1(wi− ˆwi)2
Gradient ∑m
i=1 E[(∇wiLw(X,Y ))2](wi− ˆwi)2
Hessian ∑m
i=1 E[∇2
wiLw(X,Y )](wi− ˆwi)2
Gradient ∑m
i=1 E[(∇wiLw(X,Y ))2](wi− ˆwi)2
+ Hessian + 1
4
∑m
i=1 E[(∇2
wiLw(X,Y ))2](wi− ˆwi)4
Table 2: Comparison of supervised compression objectives.
Figure 5. Generally, the gradient objective and hessian objective both give better performance than baseline
objective , while gradient objective is slightly than hessian objective at some points. Gradient + hessian
objective gives the best overall performance. We relegate the results for CIFAR100 in Appendix C.
Remark. Here we deﬁne the supervised distortion function asd(w, ˆw) = EX,Y
[
(Lw(X,Y )−L ˆw(X,Y ))2]
,
analogously to the distortion of regression. However, since the goal of classiﬁcation is to minimize the loss
function, the following deﬁnition of distortion function ˜d(w, ˆw) = EX,Y [L ˆw(X,Y )−Lw(X,Y )] is also valid
and has been adopted in LeCun et al. (1990) and Choi et al. (2016). The main diﬀerence is — d(w, ˆw) focus
on the quality of compression algorithm, i.e., how similar is the compressed model compared to uncompressed
model, whereas ˜d(w, ˆw) focus on the quality of compressed model, i.e. how good is the compressed model. So
d(w, ˆw) is a better criteria for the compression algorithm. Additionally, by taking second order approximation
of d(w, ˆw), we have gradient+hessian objective, which shows better empirical performance than hessian
objective, derived by taking second order approximation of ˜d(w, ˆw).
11

<!-- page 12 -->

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
gradient+hessianCross Entropy
0.0
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
Compression Ratio Compression Ratio
Figure 4: Result for supervised pruning experiment. Left: fully-connected neural network on MNIST (Top:
test accuracy, Bottom: test cross entropy loss). Right: convolutional neural network on CIFAR10 (Top: test
accuracy, Bottom: test cross entropy loss).
7 Conclusion
In this paper, we investigate the fundamental limit of neural network model compression algorithms. We
prove a lower bound for the rate distortion function for model compression, and prove its achievability for
linear model. Motivated by the rate distortion function, we propose the weight importance matrtix, and show
that for one-hidden-layer ReLU network, pruning and quantization that minimizes the proposed objective is
optimal. We also show the superiority of proposed objective in real neural networks.
Acknowledgement
The authors thank Denny Zhou for initial comments and helpful discussions.
References
Berger, T. (1971). Rate distortion theory: A mathematical basis for data compression.
Cao, W., Wang, X., Ming, Z., and Gao, J. (2018). A review on neural networks with random weights.
Neurocomputing, 275:278–287.
Chen, W., Wilson, J., Tyree, S., Weinberger, K., and Chen, Y. (2015). Compressing neural networks with
the hashing trick. In International Conference on Machine Learning , pages 2285–2294.
Cheng, Y., Yu, F. X., Feris, R. S., Kumar, S., Choudhary, A., and Chang, S.-F. (2015). An exploration of
parameter redundancy in deep networks with circulant projections. In Proceedings of the IEEE International
Conference on Computer Vision , pages 2857–2865.
12

<!-- page 13 -->

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
gradient+hessianCross Entropy
0.0
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
Compression Ratio Compression Ratio
Figure 5: Result for supervised quantization experiment. Left: fully-connected neural network on MNIST
(Top: test accuracy, Bottom: test cross entropy loss). Right: convolutional neural network on CIFAR10 (Top:
test accuracy, Bottom: test cross entropy loss).
Choi, Y., El-Khamy, M., and Lee, J. (2016). Towards the limit of network quantization. arXiv preprint
arXiv:1612.01543.
Coates, A., Huval, B., Wang, T., Wu, D., Catanzaro, B., and Andrew, N. (2013). Deep learning with cots
hpc systems. In International Conference on Machine Learning , pages 1337–1345.
Cover, T. M. and Thomas, J. A. (2012). Elements of information theory . John Wiley & Sons.
Denton, E. L., Zaremba, W., Bruna, J., LeCun, Y., and Fergus, R. (2014). Exploiting linear structure within
convolutional networks for eﬃcient evaluation. In Advances in neural information processing systems, pages
1269–1277.
Federici, M., Ullrich, K., and Welling, M. (2017). Improved bayesian compression. arXiv preprint
arXiv:1711.06494.
Ge, R., Lee, J. D., and Ma, T. (2017). Learning one-hidden-layer neural networks with landscape design.
arXiv preprint arXiv:1711.00501 .
Gong, Y., Liu, L., Yang, M., and Bourdev, L. (2014). Compressing deep convolutional networks using vector
quantization. arXiv preprint arXiv:1412.6115 .
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. arXiv
preprint arXiv:1706.04599.
Han, S., Mao, H., and Dally, W. J. (2015a). Deep compression: Compressing deep neural networks with
pruning, trained quantization and huﬀman coding. arXiv preprint arXiv:1510.00149 .
Han, S., Pool, J., Tran, J., and Dally, W. (2015b). Learning both weights and connections for eﬃcient neural
network. In Advances in neural information processing systems , pages 1135–1143.
13

<!-- page 14 -->

Hanson, S. J. and Pratt, L. Y. (1989). Comparing biases for minimal network construction with back-
propagation. In Advances in neural information processing systems , pages 177–185.
Hassibi, B. and Stork, D. G. (1993). Second order derivatives for network pruning: Optimal brain surgeon.
In Advances in neural information processing systems , pages 164–171.
He, Y., Lin, J., Liu, Z., Wang, H., Li, L.-J., and Han, S. (2018). Amc: Automl for model compression and
acceleration on mobile devices.
Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H.
(2017). Mobilenets: Eﬃcient convolutional neural networks for mobile vision applications. arXiv preprint
arXiv:1704.04861.
Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., and Keutzer, K. (2016). Squeezenet:
Alexnet-level accuracy with 50x fewer parameters and¡ 0.5 mb model size. arXiv preprint arXiv:1602.07360.
Jiao, J., Gao, W., and Han, Y. (2017). The nearest neighbor information estimator is adaptively near minimax
rate-optimal. arXiv preprint arXiv:1711.08824 .
Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). Imagenet classiﬁcation with deep convolutional
neural networks. In Advances in neural information processing systems , pages 1097–1105.
LeCun, Y., Bottou, L., Bengio, Y., and Haﬀner, P. (1998). Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11):2278–2324.
LeCun, Y., Denker, J. S., and Solla, S. A. (1990). Optimal brain damage. In Advances in neural information
processing systems, pages 598–605.
Louizos, C., Ullrich, K., and Welling, M. (2017). Bayesian compression for deep learning. In Advances in
Neural Information Processing Systems, pages 3288–3298.
Mandt, S., Hoﬀman, M. D., and Blei, D. M. (2017). Stochastic gradient descent as approximate bayesian
inference. The Journal of Machine Learning Research , 18(1):4873–4907.
Martens, J., Sutskever, I., and Swersky, K. (2012). Estimating the hessian by back-propagating curvature.
arXiv preprint arXiv:1206.6464 .
McDonald, R. and Schultheiss, P. (1964). Information rates of gaussian signals under criteria constraining
the error spectrum. Proceedings of the IEEE, 52(4):415–416.
Shannon, C. E. (1959). Coding theorems for a discrete source with a ﬁdelity criterion. IRE Nat. Conv. Rec ,
4(142-163):1.
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai,
M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. Nature, 550(7676):354.
Simonyan, K. and Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
arXiv preprint arXiv:1409.1556 .
Ullrich, K., Meeds, E., and Welling, M. (2017). Soft weight-sharing for neural network compression. arXiv
preprint arXiv:1702.04008.
Vanhoucke, V., Senior, A., and Mao, M. Z. (2011). Improving the speed of neural networks on cpus. In Proc.
Deep Learning and Unsupervised Feature Learning NIPS Workshop , volume 1, page 4. Citeseer.
Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q.,
Macherey, K., et al. (2016). Google’s neural machine translation system: Bridging the gap between human
and machine translation. arXiv preprint arXiv:1609.08144 .
14

<!-- page 15 -->

A Lower bound for rate distortion function
In this section, we ﬁnish the proof of the lower bound and achievability in Section 4. Our approach is based
on the water-ﬁlling approach McDonald and Schultheiss (1964).
A.1 General lower bound
First, we establish establishes a lower bound of the rate distortion function, which works for general models..
Lemma 1 The rate-distortion function R(D)≥ R(D) = h(W )−C, where C is the optimal value of the
following optimization problem.
max
P ˆW|W
m∑
i=1
min
{
h(Wi), 1
2 log(2πeEW, ˆW [(Wi− ˆWi)2])
}
s.t. E W, ˆW
[
d(W, ˆW )
]
≤D.
where h(W ) =−
∫
w∈WPW (w) logPW (w)dw is the diﬀerential entropy of W and h(Wi) is the diﬀerential
entropy of the i-th entry of W .
A.1.1 Proof of Lemma 1
Recall that the rate distortion function for model compression is deﬁned asR(D) = minP ˆW|W :EW, ˆW [d(W, ˆW )]≤DI(W ; ˆW ).
Now we lower bound the mutual information I(W, ˆW ) by
I(W ; ˆW ) = h(W )−h(W| ˆW ),
= h(W )−
m∑
i=1
h(Wi|W1,...,W i−1, ˆWi,..., ˆWm)
≥ h(W )−
m∑
i=1
h(Wi| ˆWi).
Here the last inequality comes from the fact that conditioning does not increase entropy. Notice that the ﬁrst
term h(W ) does not depend on the compressor. For the last term, we upper bound each term h(Wi| ˆWi)
in two ways. On one hand, h(Wi| ˆWi) is upper bounded by h(Wi) because conditioning does not increase
entropy. On the other hand, h(Wi| ˆWi) =h(Wi− ˆWi| ˆWi)≤h(Wi− ˆWi), and by Cover and Thomas (2012,
Theorem 8.6.5), diﬀerential entropy is maximized by Gaussian distribution, for given second moment. We
then have:
h(Wi| ˆWi) ≤ min
{
h(Wi),h (Wi− ˆWi)
}
≤ min
{
h(Wi), 1
2 log
(
2πeEW, ˆW [(Wi− ˆWi)2]
)}
= min
{
h(Wi), 1
2 log(2πeEW, ˆW [(Wi− ˆWi)2])
}
.
Therefore, the lower bound of the mutual information is given by,
I(W ; ˆW )≥h(W )−
m∑
i=1
min
{
h(Wi), 1
2 log(2πeEW, ˆW [(Wi− ˆWi)2])
}
.
15

<!-- page 16 -->

A.2 Lower bound for linear model
For complex models, the general lower bound in Lemma 1 is diﬃcult to evaluate, due to the large dimension
of parameters. It was shown by Jiao et al. (2017) that the sample complexity to estimate diﬀerential entropy
is exponential to the dimension. It’s even harder to design an algorithm to achieve the lower bound. But for
linear model, the lower bound can be simpliﬁed. For fw(x) =wTx, the distortion function d(w, ˆw) can be
written as
d(w, ˆw) = EX
[
(fw(X)−f ˆw(X))2]
= EX
[
(wTX− ˆwTX)2]
= EX
[
(w− ˆw)TXXT (w− ˆw)
]
= (w− ˆw)T EX[XXT ](w− ˆw).
Since we assumed that E[X] = 0, E[X 2
i ] =λx,i > 0 and E[XiXj] = 0, so the constraint in Lemma 1 is
given by
D ≥ EW, ˆW
[
(W− ˆW )T EX[XXT ](W− ˆW )
]
=
m∑
i=1
λx,i EW, ˆW
[
(Wi− ˆWi)2
]
  
Di
.
Then the optimization problem in Lemma 1 can be written as follows
max
p( ˆw|w)
m∑
i=1
min{h(Wi), 1
2 log(2πeDi)}
s.t.
m∑
i=1
λx,iDi≤D.
Here Wi is a Gaussian random variable, so h(Wi) = 1
2 log(2πeE[W 2
i ]). The Lagrangian function of the
problem is given by
L(D1,...,D m,µ )
=
m∑
i=1
(
min{1
2 log E[W 2
i ], 1
2 logDi} + 1
2 log(2πe)−µλx,iDi
)
.
By setting the derivative w.r.t. Di to 0, we have
0 = ∂L
∂Di
= 1
2Di
−µλx,i.
for all Di such that Di < E[W 2
i ]. So the optimal Di should satisfy that Diλx,i is constant, for all Di such
that Di < E[W 2
i ]. Also the optimal Di is at most E[W 2
i ]. Also, since h(W ) = m
2 log(2πe) + 1
2 log det(ΣW )
the lower bound is given by
R(D)≥ 1
2 log det(ΣW )−
m∑
i=1
1
2 log(Di),
where
Di =
{
µ/λx,i ifµ<λ x,iEW [W 2
i ],
EW [W 2
i ] if µ≥λx,iEW [W 2
i ],
where µ is chosen that ∑m
i=1λx,iDi =D.
16

<!-- page 17 -->

Figure 6: Illustration of “weighted water-ﬁlling” process.
This lower bound gives rise to a “weighted water-ﬁlling”, which diﬀers from the classical “water-ﬁlling”
for rate-distortion of colored Gaussian source in Cover and Thomas (2012, Figure 13.7), since the water
level’sDi are proportional to 1/λx,i, which is related to the input of the model rather than the parameters
to be compressed. To illustrate the “weighted water-ﬁlling” process, we choose a simple example where
ΣW = ΣX = diag[3, 2, 1]. In Figure 6, the widths of each rectangle are proportional to λx,i, and the heights
are proportional to ΣW = [3, 2, 1]. The water level in each rectangle is Di and the volume of water is µ. As
D starts to increase from 0, each rectangle is ﬁlled with same volume of water ( µ is the same), but the water
levelDi’s increase with speed 1/λx,i respectively (Figure 6.(a)). This gives segment (a) of the rate distortion
curve in Figure 6.(d). If D is large enough such that the third rectangle is full, then D3 is ﬁxed to be
E[W 2
3 ] = 1, whereas D1 and D2 continuously increase (Figure 6.(b)). This gives segment (b) in Figure 6.(d).
Keep increasing D until the second rectangle is also full, then D2 is ﬁxed to be E[W 2
2 ] = 2 and D1 continuous
increasing (Figure 6 (c)). This gives segment (c) in Figure 6.(d). The entire rate-distortion function is shown
in Figure 6(d), where the ﬁrst red dot corresponds to the moment that the third rectangle is exactly full, and
the second red dot corresponds to moment that the second rectangle is exactly full.
A.3 Achievability
We prove that this lower bound is achievable. To achieve the lower bound, we construct the compression
algorithm in Algorithm 1,
Intuitively, the optimal compressor does the following: (1) Find the optimal water levels Di for “weighted
water ﬁlling”. (2) For the entries where the corresponding rectangles are full, simply discard the entries;
(3) for the entries where the corresponding rectangles are not full, add a noise which is independent of ˆWi
and has a variance proportional to the water level. That is possible since W is Gaussian. (4) Combine the
conditional probabilities.
To see that this compressor is optimal, we will check that the compressor makes all the inequalities become
equality. Here is all the inequalities used in the proof.
• h(Wi|W1,...,W i−1, ˆWi,..., ˆWm)≤ h(Wi| ˆWi) for all i = 1...m. It becomes equality by P ˆW|W =∏m
i=1P ˆWi|W .
• Either
– h(Wi| ˆWi)≤h(Wi). It becomes equality for those ˆWi = 0.
– h(Wi− ˆWi| ˆWi)≤h(Wi− ˆWi)≤ 1
2 log(2πeEW, ˆW [(Wi− ˆW )2]). It becomes equality for those ˆWi’s
such that Wi− ˆWi is independent of ˆWi and Wi− ˆWi is Gaussian.
17

<!-- page 18 -->

Algorithm 1 Optimal compression algorithm for linear regression
Input: distortion D, covariance matrix of parameters Σ W , covariance matrix of data Σ X =
diag[λx,1,...,λ x,m].
Choose Di’s such that
Di =
{
µ/λx,i ifµ<λ x,iEW [W 2
i ],
EW [W 2
i ] if µ≥λx,iEW [W 2
i ],
where∑m
i=1λx,iDi =D.
for i = 1 to m do
if Di =µ/λx,i then
Choose ˆWi = 0
else
Choose a conditional distribution P ˆWi|Wi
such that Wi = ˆW +Zi where Zi ∼ N(0,Di), ˆWi ∼
N (0, EW [W 2
i ]−Di) and ˆWi is independent of Zi.
end if
end for
Combine the conditional probability distributions by P ˆW|W =∏m
i=1P ˆWi|Wi
.
• The “water levels” Di. It becomes equality by choosing the Di’s according to Lagrangian conditions.
Therefore, Algorithm 1 gives a compressor P (D)
ˆW|W such that EPW◦P (D)
ˆW|W
[d(W, ˆW )] =D and I(W ; ˆW ) =R(D),
hence the lower bound is tight.
B Proof of Theorem 4
In this section, we provide the proof of Theorem 4. For simplicity let σ(t) = tI{t≥ 0} denotes the ReLU
activation function. First we deal with the objective of the compression algorithm,
(w− ˆw)TIw(w− ˆw) = ( w− ˆw)T EX
[
∇wfw(x)∇wfw(x)T]
(w− ˆw)
= ( w− ˆw)T EX
[
∇wσ(wTx)∇wσ(wTx)T]
(w− ˆw)
= ( w− ˆw)T EX
[
xT (σ′(wTx))2x
]
(w− ˆw)
= EX
[
I{wTx≥ 0}((w− ˆw)Tx)2]
Notice that x is jointly Gaussian random variable with zero mean and non-degenerate variance, so the
distribution of x is equivalent to the distribution of −x. Therefore,
EX[I{wTx≥ 0}((w− ˆw)Tx)2] =
∫
x:wTx≥0
((w− ˆwT )x)2dx
= 1
2
(∫
x:wTx≥0
((w− ˆwT )x)2dx +
∫
x:wTx≤0
((w− ˆwT )x)2dx
)
= 1
2
∫ d
x∈R
((w− ˆwT )x)2dx = 1
2(w− ˆw)T ΣX(w− ˆw)
So minimizing the gradient-squared based loss is equivalent to minimizing ( w− ˆw)T ΣX(w− ˆw). Similarly,
the condition ˆwIw(w− ˆw) = 0 is equivalent to ˆwΣX(w− ˆw) = 0. Now we deal with the MSE loss function
E[(fw(x)−f ˆw(x))2]. We utilize the Hermite polynomials and Fourier analysis on Gaussian space. We use the
following key lemma,
18

<!-- page 19 -->

Lemma 2 (Ge et al. (2017, Claim 4.3)) Let f, g be two functions from R to R such that f 2,g 2 ∈
L2(R,e−x2/2). The for any unit vectors u,v , we have that
Ex∈N (0,Id×d)[f(uTx)g(vTx)] =
∞∑
p=0
ˆfpˆgp(uTv)p
where ˆfp = Ex∈N (0,1)[f(x)hp(x)] is the p-th order coeﬃcient of f, where hp is the p-th order probabilists’
Hermite polynomial.
Please see Section 4.1 in Ge et al. (2017) for more backgrounds of the Hermite polynomials and Fourier
analysis on Gaussian space. For ReLU function, the coeﬃcients are given by ˆσ0 = 1√
2π , ˆσ1 = 1
2. For p≥ 2
and even, ˆσp = ((p−3)!!)2
√2πp! . For p≥ 2 and odd, ˆσp = 0. Since X∼N (0, ΣX), we can write x = Σ1/2
X z, where
z∼N (0,Id). So for any compressed weight ˆw, we have
EX
[
(fw(x)−f ˆw(x))2]
= EX
[
(σ(wTx)−σ( ˆwTx))2]
= Ez∈N (0,Id)[(σ(wT Σ1/2
X z)−σ( ˆwT Σ1/2
X z))2]
= Ez∈N (0,Id)[σ(wT Σ1/2
X z)2]− 2Ez∈N (0,Id)[σ(wT Σ1/2
X z)σ( ˆwT Σ1/2
X z)] + Ez∈N (0,Id)[σ( ˆwT Σ1/2
X z)2]
=
∞∑
p=0
ˆσ2
p(wT ΣXw)p− 2
∞∑
p=0
ˆσ2
p(wT ΣX ˆw)p +
∞∑
p=0
ˆσp
2( ˆwT ΣX ˆw)p
=
∞∑
p=0
ˆσ2
p

(wT ΣXw)p− 2(wT ΣX ˆw)p + ( ˆwT ΣX ˆw)p
  
Dp(w, ˆw)


Now we can see that D0(w, ˆw) = 0. D1(w, ˆw) =wT ΣXw− 2wT ΣX ˆw + ˆwT ΣXw = (w− ˆw)T ΣX(w− ˆw),
is just the objective. The following lemma gives the minimizer of Dp(w, ˆw) for higher order p.
Lemma 3 If ˆw∗ satisﬁes ˆw∗ΣX( ˆw−w) = 0 and
ˆw∗ = arg min
ˆs∈W
D1(w, ˆw)
for some constrained set W. Then for any p≥ 2 and even, we have
ˆw∗ = arg min
ˆw∈W
Dp(w, ˆw)
Since the coeﬃcients ˆσp is zero for p≥ 3 and odd, so if a compressed weight ˆw satisﬁed ˆwΣX( ˆw−w) = 0
and minimizes D1( ˆw,w ) = ( ˆw−w)T ΣX( ˆw−w), then it is the minimizer for all Dp(w, ˆw) for even p, therefore
a minimizer of the MSE loss.
B.1 Proof of Lemma 3
For simplicity of notation, deﬁneA =wT ΣXw,B = ˆwT ΣX( ˆw−w) andC =D1(w, ˆw) = ( ˆw−w)T ΣX( ˆw−w).
For all compressors, we have C≤A. Therefore, wT ΣX ˆw =A +B−C and ˆwT ΣX ˆw =A + 2B−C. So
Dp(w, ˆw) = Ap− 2(A +B−C)p + (A + 2B−C)p
First notice that
∂Dp(w, ˆw)
∂B = 2p((A + 2B−C)p−1− (A +B−C)p−1).
19

<!-- page 20 -->

For even p ≥ 2, xp−1 is monotonically increasing, so ( A + 2B−C)p−1 > (A +B−C)p−1 if B > 0
and vice versa. Therefore, for ﬁxed A and C, Dp(w, ˆw) is monotonically increasing for positive B and
decreasing for negative B. Therefore, Dp(w, ˆw) is minimized when B = 0, and the minimal value is
Dp(w, ˆw) =Ap− 2(A−C)p + (A−C)p =Ap− (A−C)p, which is monotonically increasing with respect to
C. So if ˆw∗ satisﬁes B = 0 and is a minimzer of C =D1(w, ˆw), it is also a minimizer for Dp(w, ˆw) for all
p≥ 2 and even.
C Details of the experiments
In this appendix, we give some details of the experiment and additional experiments which are omitted in the
main text.
C.1 Additional experiment results
We present the experiment results for CIFAR100 here, due to page limit of the main text.
In Figure 7 and Figure 8, we show the result for unsupervised pruning and quantization, introduced in
Section 6.1. We can see that, similar to the experiments of MNIST and CIFAR10, the proposed objectives
gives better accuracy and smaller loss than the baseline.
In Figure 9 and Figure 10, we show the result for supervised pruning and quantization, introduced in
Section 6.2. Due to the slow running speed for estimating the Hessian ∇2
wiLw(x,y ), we only compare two
objectives — baseline and gradient. It is shown that the gradient objective gives better accuracy and smaller
loss.
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
2.5
5.0
7.5
10.0
12.5
0% 20% 40% 60% 80% 100%
uncompressed
baseline
proposed
Cross Entropy
Compression Ratio Compression Ratio Compression Ratio
Figure 7: Result for unsupervised pruning experiment for CIFAR 100 experiment. Left: top-1 accuracy.
Middle: top-5 accuracy. Right: cross entropy loss.
0
0.2
0.4
0.6
0.8
1.0
4% 6% 8% 10% 12% 14%
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
4% 6% 8% 10% 12% 14%
uncompressed
baseline
proposed
0.0
3.0
6.0
9.0
12.0
15.0
4% 6% 8% 10% 12%
uncompressed
baseline
proposed
Cross Entropy
Compression Ratio Compression Ratio Compression Ratio
Figure 8: Result for unsupervised quantization experiment for CIFAR 100 experiment. Left: top-1 accuracy.
Middle: top-5 accuracy. Right: cross entropy loss.
20

<!-- page 21 -->

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
0.0
2.5
5.0
7.5
10.0
12.5
0% 20% 40% 60% 80% 100%
uncompressed
baseline
gradient
Cross Entropy
Compression Ratio Compression Ratio Compression Ratio
Figure 9: Result for supervised pruning experiment for CIFAR 100 experiment. Left: top-1 accuracy. Middle:
top-5 accuracy. Right: cross entropy loss.
0
0.2
0.4
0.6
0.8
1.0
4% 6% 8% 10% 12% 14%
uncompressed
baseline
gradient
Accuracy
0
0.2
0.4
0.6
0.8
1.0
4% 6% 8% 10% 12% 14%
uncompressed
baseline
gradient
0.0
3.0
6.0
9.0
12.0
15.0
4% 6% 8% 10% 12%
uncompressed
baseline
gradient
Cross Entropy
Compression Ratio Compression Ratio Compression Ratio
Figure 10: Result for supervised quantization experiment for CIFAR 100 experiment. Left: top-1 accuracy.
Middle: top-5 accuracy. Right: cross entropy loss.
C.2 Algorithm for ﬁnding optimal quantization
We present a variation of k-means algorithm which are used to ﬁnd the optimal quantization for the following
objective,
min
c1,...,ck,A∈[k]m
m∑
i=1
(
Ii(wi−cAi)2 +Hi(wi−cAi)4)
where Ii is positive weight importance for quadratic term and Hi is positive weight importance for quartic
term. Basic idea of the algorithm is — the assignment step ﬁnds the optimal assignment given ﬁxed centroids,
and the update step ﬁnds the optimal centroids given ﬁxed assignments. This is used for gradient+hessian
objective in Section 6.2.
Here we show that the cubic equation in Algorithm 2 has only one real root. It was know that if the
determinant ∆0 =b2− 3ac of a cubic equation ax3 +bx2 +cx +d = 0 is negative, then the cubic equation
is strictly increasing or decreasing, hence only have one real root. Now we show that the determinant is
negative in this case (we drop the subsripts of the summation for simplicity).
∆0 = (
∑
12Hiwi)2− 3(
∑
4Hi)(
∑
12Hiw2
i + 2Ii)
= 144
(
(
∑
Hiwi)2− (
∑
Hi)(
∑
Hiw2
i )
)
− 24(
∑
Hi)(
∑
Ii)
The ﬁrst term is non-positive because of Cauchy-Schwarz inequality. The second term is negative since Hi’s
and Ii’s are all positive. Hence the determinant is negative.
21

<!-- page 22 -->

Algorithm 2 Quartic weighted k-means
input Weights{w1,...,w m}, weight importances{I1,...,I m}, quartic weight importances{H1,...,H m},
number of clusters k, iterations T
Initialize the centroid of k clusters{c(0)
1 ,...,c (0)
k }
for t = 1 to T do
Assignment step:
for i = 1 to m do
Assign wi to the nearest cluster centroid, i.e. A(t)
i = arg minj∈[k](wi−c(t−1)
j )2.
end for
Update step:
for j = 1 to k do
Find the only real root x∗ of the cubic equation
(
∑
i:A(t)
i =j
4Hi)x3− (
∑
i:A(t)
i =j
12Hiwi)x2 + (
∑
i:A(t)
i =j
(12Hiw2
i + 2Ii))x− (
∑
i:A(t)
i =j
(4Hiw3
i + 2Iiwi)) = 0
Update the cluster centroids c(t)
j be the real root x∗.
end for
end for
output Centroids{c(T )
1 ,...,c (T )
k } and assignments A(T )∈ [k]m.
C.3 Eﬀects of hyperparameters
Here we brieﬂy talk about the hyperparameters used in estimating the gradients E[∇wiLw(X,Y )] and hessians
E[∇2
wiLw(X,Y )].
C.3.1 Temperature scaling method
The temperature scaling method proposed by Guo et al. (2017), aims to improve the conﬁdence calibration
of a classiﬁcation model. Denote zw(x)∈ RC is the output of the neural network, and classical softmax gives
f (c)
w (x) = exp{z(c)
w (x)}∑
c∈C exp{z(c)
w (x)}. The temperature sclaed softmax gives
f (c)
w (x) = exp{z(c)
w (x)/T}
∑
c∈C exp{z(c)
w (x)/T}
by choosing diﬀerent T , the prediction of the model does not change, but the cross entropy loss may change.
Hence, we can ﬁnetune T to get a better model calibration. In our experiment, we found that in MNIST
experiment, the model is poorly calibrated. Hence, the variance of estimating gradient and hessian is very
large. To solve this, we adopt a temperature T >1 such that the loss from correctly-predicted data can also
be backpropagated.
In Figure 11, we show the eﬀect of T for supervised pruning for MNIST. We can see that as T increases
from 1, the performance become better at ﬁrst, then become worse. In our experiment, we choose T ∈
{1.0, 2.0,..., 9.0} which gives best accuracy.
C.3.2 Regularizer of hessian
In the experiments, we estimate the hessians E[∇2
wiLw(X,Y )] using the curvature propagation algo-
rithm Martens et al. (2012). However, due to the sparsity introduced by ReLU, there are many zero
entries of the estimated hessians, which hurts the performance of the algorithm. Hence, we add a constant
µ> 0 to the estimated hessians. In Figure 12, we show that eﬀect of µ for supervised pruning for CIFAR10.
22

<!-- page 23 -->

0.4
0.6
0.8
1.0
1 3 5 7 9
ratio=0.05
ratio=0.075
ratio=0.1
Accuracy
0.0
0.5
1.0
1.5
2.0
1 3 5 7 9
ratio=0.05
ratio=0.075
ratio=0.1
Cross Entropy
Temperature Temperature
Figure 11: Eﬀect of the temperature T . Left: accuracy of supervised pruning for MNIST. Right: cross entropy
of supervised pruning for MNIST. Diﬀerent lines denote diﬀerent compression ratio ∈{ 0.05, 0.075, 0.1}
We can see that as µ increases from 0, the performance increase ﬁrst then decrease. We use simple binary
search to ﬁnd the best µ.
0.7
0.8
0.9
0 0.05 0.1 0.15 0.2 0.25
ratio=0.4
ratio=0.5
ratio=0.6
Accuracy
0.5
1.0
1.5
0 0.05 0.1 0.15 0.2 0.25
ratio=0.4
ratio=0.5
ratio=0.6
Cross Entropy
µ µ
Figure 12: Eﬀect of the regularizer µ. Left: accuracy of supervised pruning for CIFAR10. Right: cross
entropy of supervised pruning for CIFAR10. Diﬀerent lines denote diﬀerent compression ratio ∈{ 0.4, 0.5, 0.6}
23
