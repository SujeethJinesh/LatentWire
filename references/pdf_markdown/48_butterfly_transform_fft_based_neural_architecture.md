# references/48_butterfly_transform_fft_based_neural_architecture.pdf

<!-- page 1 -->

Butterﬂy Transform: An Efﬁcient FFT Based Neural Architecture Design
Keivan Alizadeh vahid, Anish Prabhu, Ali Farhadi, Mohammad Rastegari
University of Washington
keivan@cs.washington.edu
Abstract
In this paper, we show that extending the butterﬂy opera-
tions from the FFT algorithm to a general Butterﬂy Trans-
form (BFT) can be beneﬁcial in building an efﬁcient block
structure for CNN designs. Pointwise convolutions, which
we refer to as channel fusions, are the main computational
bottleneck in the state-of-the-art efﬁcient CNNs (e.g. Mo-
bileNets [15, 38, 14]).We introduce a set of criterion for
channel fusion, and prove that BFT yields an asymptoti-
cally optimal FLOP count with respect to these criteria.
By replacing pointwise convolutions with BFT, we reduce
the computational complexity of these layers from O(n2)
to O(n log n) with respect to the number of channels. Our
experimental evaluations show that our method results in
signiﬁcant accuracy gains across a wide range of network
architectures, especially at low FLOP ranges. For example,
BFT results in up to a 6.75% absolute Top-1 improvement
for MobileNetV1[15], 4.4% for ShufﬂeNet V2[28] and 5.4%
for MobileNetV3[14] on ImageNet under a similar number
of FLOPS. Notably, ShufﬂeNet-V2+BFT outperforms state-
of-the-art architecture search methods MNasNet[43], FBNet
[46] and MobilenetV3[14] in the low FLOP regime.
1. Introduction
Devising Convolutional Neural Networks (CNN) that can
run efﬁciently on resource-constrained edge devices has be-
come an important research area. There is a continued push
to put increasingly more capabilities on-device for personal
privacy, latency, and scale-ability of solutions. On these
constrained devices, there is often extremely high demand
for a limited amount of resources, including computation
and memory, as well as power constraints to increase battery
life. Along with this trend, there has also been greater ubiq-
uity of custom chip-sets, Field Programmable Gate Arrays
(FPGAs), and low-end processors that can be used to run
CNNs, rather than traditional GPUs.
A common design choice is to reduce the FLOPs and
parameters of a network by factorizing convolutional lay-
ers [15, 38, 28, 50] into a depth-wise separable convolution
Figure 1: Replacing pointwise convolutions with BFT in state-of-the-art
architectures results in signiﬁcant accuracy gains in resource constrained
settings.
that consists of two components: (1) spatial fusion, where
each spatial channel is convolved independently by a depth-
wise convolution, and (2) channel fusion, where all the spa-
tial channels are linearly combined by 1 × 1 convolutions,
known as pointwise convolutions. Inspecting the computa-
tional proﬁle of these networks at inference time reveals that
the computational burden of the spatial fusion is relatively
negligible compared to that of the channel fusion[ 15]. In
this paper we focus on designing an efﬁcient replacement
for these pointwise convolutions.
We propose a set of principles to design a replacement
for pointwise convolutions motivated by both efﬁciency and
accuracy. The proposed principles are as follows: (1) full
connectivity from every input to all outputs: to allow outputs
to use all available information, (2) large information bot-
tleneck: to increase representational power throughout the
network, (3) low operation count: to reduce the computa-
tional cost, (4) operation symmetry: to allow operations to be
stacked into dense matrix multiplications. In Section 3, we
formally deﬁne these principles, and mathematically prove a
lower-bound of O(n log n) operations to satisfy these princi-
ples. We propose a novel, lightweight convolutional building
block based on the Butterﬂy Transform (BFT). We prove that
BFT yields an asymptotically optimal FLOP count under
arXiv:1906.02256v2  [cs.CV]  16 Apr 2020

<!-- page 2 -->

these principles.
We show that BFT can be used as a drop-in replacement
for pointwise convolutions in several state-of-the-art efﬁ-
cient CNNs. This signiﬁcantly reduces the computational
bottleneck for these networks. For example, replacing point-
wise convolutions with BFT decreases the computational
bottleneck of MobileNetV1 from 95% to 60%, as shown in
Figure 3. We empirically demonstrate that using BFT leads
to signiﬁcant increases in accuracy in constrained settings, in-
cluding up to a 6.75% absolute Top-1 gain for MobileNetV1,
4.4% for ShufﬂeNet V2 and 5.4% for MobileNetV3 on the
ImageNet[7] dataset. There have been several efforts on us-
ing butterﬂy operations in neural networks [ 20, 6, 33] but, to
the best of our knowledge, our method outperforms all other
structured matrix methods (Table 2b) for replacing pointwise
convolutions as well as state-of-the-art Neural Architecture
Search (Table 2a) by a large margin at low FLOP ranges.
2. Related Work
Deep neural networks suffer from intensive computations.
Several approaches have been proposed to address efﬁcient
training and inference in deep neural networks.
Efﬁcient CNN Architecture Designs: Recent successes
in visual recognition tasks, including object classiﬁcation,
detection, and segmentation, can be attributed to exploration
of different CNN designs [23, 39, 13, 21, 42, 17]. To make
these network designs more efﬁcient, some methods have
factorized convolutions into different steps, enforcing dis-
tinct focuses on spatial and channel fusion [15, 38]. Further,
other approaches extended the factorization schema with
sparse structure either in channel fusion [28, 50] or spatial
fusion [30]. [16] forced more connections between the lay-
ers of the network but reduced the computation by designing
smaller layers. Our method follows the same direction of
designing a sparse structure on channel fusion that enables
lower computation with a minimal loss in accuracy.
Structured Matrices: There have been many methods
which attempt to reduce the computation in CNNs, [44, 24, 8,
19] by exploiting the fact that CNNs are often extremely over-
parameterized. These models learn a CNN or fully connected
layer by enforcing a linear transformation structure during
the training process which has less parameters and compu-
tation than the original linear transform. Different kinds of
structured matrices have been studied for compressing deep
neural networks, including circulant matrices[ 9], toeplitz-
like matrices[40], low rank matrices[37], and fourier-related
matrices[32]. These structured matrices have been used for
approximating kernels or replacing fully connected layers.
UGConv [51] has considered replacing one of the point-
wise convolutions in the ShufﬂeNet structure with unitary
group convolutions, while our Butterﬂy Transform is able
to replace all of the pointwise convolutions. The butterﬂy
structure has been studied for a long time in linear algebra
[34, 26] and neural network models [ 31]. Recently, it has
received more attention from researchers who have used it in
RNNs [20], kernel approximation[33, 29, 4] and fully con-
nected layers[6]. We have generalized butterﬂy structures
to replace pointwise convolutions, and have signiﬁcantly
outperformed all known structured matrix methods for this
task, as shown in Table 2b.
Network pruning: This line of work focuses on reducing
the substantial redundant parameters in CNNs by pruning
out either neurons or weights [11, 12, 45, 2]. Our method is
different from these type methods in the way that we enforce
a predeﬁned sparse channel structure to begin with and we do
not change the structure of the network during the training.
Quantization: Another approach to improve the efﬁciency
of the deep networks is low-bit representation of network
weights and neurons using quantization [ 41, 35, 47, 5, 52,
18, 1]. These approaches use fewer bits (instead of 32-bit
high-precision ﬂoating points) to represent weights and neu-
rons for the standard training procedure of a network. In the
case of extremely low bitwidth (1-bit) [ 35] had to modify
the training procedure to ﬁnd the discrete binary values for
the weights and the neurons in the network. Our method is
orthogonal to this line of work and these method are com-
plementary to our network.
Neural architecture search: Recently, neural search
methods, including reinforcement learning and genetic al-
gorithms, have been proposed to automatically construct
network architectures [53, 48, 36, 54, 43, 27]. Recent search-
based methods [43, 3, 46, 14] use Inverted Residual Blocks
[38] as a basic search block for automatic network design.
The main computational bottleneck in most of the search
based method is in the channel fusion and our butterﬂy struc-
ture does not exist in any of the predeﬁned blocks of these
methods. Our efﬁcient channel fusion can be augmented
with these models to further improve the efﬁciency of these
networks. Our experiments shows that our proposed butter-
ﬂy structure outperforms recent architecture search based
models on small network design.
3. Model
In this section, we outline the details of the proposed
model. As discussed above, the main computational bot-
tleneck in current efﬁcient neural architecture design is in
the channel fusion step, which is implemented with a point-
wise convolution layer. The input to this layer is a tensor
X of size nin × h × w, where n is the number of channels

<!-- page 3 -->

1−BFLayer 2−BFLayer log
  𝑛−BFLayer
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
BFT(n
2,2)
BFT(n
2,2)
y1
y2
.
.
.
yn
2
− 1
yn
2
yn
2
+1
yn
2
+2
.
.
.
yn− 1
yn
BFT(n,2)
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
𝑤
ℎ
𝑛
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
x1
x2
.
.
.
x n
2
− 1
x n
2
x n
2
+1
x n
2
+2
.
.
.
xn− 1
xn
Figure 2: BFT Architecture: This ﬁgure illustrates the graph structure of the proposed Butterﬂy Transform. The left ﬁgure shows the recursive procedure
of the BFT that is applied to an input tensor and the right ﬁgure shows the expanded version of the recursive procedure as log n Butterﬂy Layers in the
network.
and w, h are the width and height respectively. The size of
the weight tensor W is nout × nin × 1 × 1 and the output
tensor Y is nout × h × w. For the sake of simplicity, we
assume n = nin = nout. The complexity of a pointwise
convolution layer is O(n2wh), and this is mainly inﬂuenced
by the number of channels n. We propose to use Butterﬂy
Transform as a layer, which has O((n log n)wh) complexity.
This design is inspired by the Fast Fourier Transform (FFT)
algorithm, which has been widely used in computational
engines for a variety of applications and there exist many
optimized hardware/software designs for the key operations
of this algorithm, which are applicable to our method. In the
following subsections we explain the problem formulation
and the structure of our butterﬂy transform.
3.1. Pointwise Convolution as Matrix-Vector Prod-
ucts
A pointwise convolution can be deﬁned as a functionP
as follows:
Y = P(X; W) (1)
This can be written as a matrix product by reshaping the input
tensor X to a 2-D matrix ˆX with size n×(hw) (each column
vector in the ˆX corresponds to a spatial vector X[:, i, j]) and
reshaping the weight tensor to a 2-D matrix ˆW with size
n × n,
ˆY = ˆW ˆX (2)
where ˆY is the matrix representation of the output tensor Y.
This can be seen as a linear transformation of the vectors
in the columns of ˆX using ˆW as a transformation matrix.
The linear transformation is a matrix-vector product and
its complexity is O(n2). By enforcing structure on this
transformation matrix, one can reduce the complexity of the
transformation. However, to be effective as a channel fusion
transform, it is critical that this transformation respects the
desirable characteristics detailed below.
Fusion network design principles: 1) full connectivity
from every input to all outputs: This condition allows every
single output to have access to all available information in the
inputs. 2) large information bottleneck: The bottleneck size
is deﬁned as the minimum number of nodes in the network
that if removed, the information ﬂow from input channels to
output channels would be completely cut off (i.e. there would
be no path from any input channel to any output channel).
The representational power of the network is bound by the
bottleneck size. To ensure that information is not lost while
passed through the channel fusion, we set the minimum
bottleneck size to n. 3) low operation count : The fewer
operations, or equivalently edges in the graph, that there are,
the less computation the fusion will take. Therefore we want
to reduce the number of edges. 4) operation symmetry: By
enforcing that there is an equal out-degree in each layer, the
operations can be stacked into dense matrix multiplications,
which is in practice much faster for inference than sparse
computation.
Claim: A multi-layer network with these properties has
at least O(n log n) edges.
Proof: Suppose there exist ni nodes in ith layer. Remov-
ing all the nodes in one layer will disconnect inputs from
outputs. Since the maximum possible bottleneck size is n,
therefore ni ≥ n. Now suppose that out degree of each
node at layer i is di. Number of nodes in layer i, which are
reachable from an input channel is ∏i−1
j=0 dj. Because of
the every-to-all connectivity, all of the n nodes in the output
layer are reachable. Therefore ∏m−1
j=0 dj ≥ n. This implies
that ∑m−1
j=0 log2(dj) ≥ log2(n). The total number of edges
will be:

<!-- page 4 -->

∑m−1
j=0 njdj ≥ n ∑m−1
j=0 dj ≥ n ∑m−1
j=0 log2(dj) ≥
n log2 n■
In the following section we present a network structure
that satisﬁes all the design principles for fusion network.
3.2. Butterﬂy Transform (BFT)
As mentioned above we can reduce the complexity of
a matrix-vector product by enforcing structure on the ma-
trix. There are several ways to enforce structure on the
matrix. Here we ﬁrst explain how the channel fusion is done
through BFT and then show a family of the structured matrix
equivalent to this fusion leads to a O(n log n) complexity of
operations and parameters while maintaining accuracy.
Channel Fusion through BFT: We want to fuse informa-
tion among all channels. We do it in sequential layers. In the
ﬁrst layer we partition channels tok parts with size n
k each,
x1, .., xk. We also partition output channels of this ﬁrst layer
to k parts with n
k size each, y1, .., yk. We connect elements
of xi to yj with n
k parallel edges Dij. After combining
information this way, each yi contains the information from
all channels, then we recursively fuse information of each
yi in the next layers.
Butterﬂy Matrix: In terms of matrices B(n,k) is a butter-
ﬂy matrix of order n and base k where B(n,k) ∈ I Rn×n is
equivalent to fusion process described earlier.
B(n,k) =


M
( n
k ,k)
1 D11 . . . M
( n
k ,k)
1 D1k
... ... ...
M
( n
k ,k)
k Dk1 . . . M
( n
k ,k)
k Dkk

 (3)
Where M
( n
k ,k)
i is a butterﬂy matrices of order n
k and
base k and Dij is an arbitrary diagonal n
k × n
k matrix. The
matrix-vector product between a butterﬂy matrix B(n,k) and
a vector x ∈ I Rn is :
B(n,k)x =


M
( n
k ,k)
1 D11 . . . M
( n
k ,k)
1 D1k
... ... ...
M
( n
k ,k)
k Dk1 . . . M
( n
k ,k)
k Dkk




x1
...
xk


(4)
where xi ∈ I R
n
k is a subsection of x that is achieved by
breaking x into k equal sized vector. Therefore, the product
can be simpliﬁed by factoring outM as follow:
B(n,k)x =


M
( n
k ,k)
1
∑k
j=1 D1jxj
...
M
( n
k ,k)
i
∑k
j=1 Dijxj
...
M
( n
k ,k)
k
∑k
j=1 Dkj xj


=


M
( n
k ,k)
1 y1
...
M
( n
k ,k)
i yi
...
M
( n
k ,k)
k yk


(5)
where yi = ∑k
j=1 Dijxj. Note that M
( n
k ,k)
i yi is a smaller
product between a butterﬂy matrix of order n
k and a vector
of size n
k therefore, we can use divide-and-conquer to recur-
sively calculate the product B(n,k)x. If we consider T (n, k)
as the computational complexity of the product between a
(n, k) butterﬂy matrix and an n-D vector. From equation 5,
the product can be calculated by k products of butterﬂy ma-
trices of order n
k which its complexity is kT (n/k, k). The
complexity of calculating yi for all i ∈ {1, . . . , k} is O(kn)
therefore:
T (n, k) = kT (n/k, k) + O(kn) (6)
T (n, k) = O(k(n logk n)) (7)
With a smaller choice of k(2 ≤ k ≤ n) we can achieve
a lower complexity. Algorithm 1 illustrates the recursive
procedure of a butterﬂy transform when k = 2.
Algorithm 1: Recursive Butterﬂy Transform
1 Function ButterflyTransform(W, X, n): ;
/* algorithm as a recursive
function */
2
Data: W Weights containing 2n log(n) numbers
Data: X An input containing n numbers
3 if n == 1 then
4 return [X] ;
5 Make D11, D12, D21, D22 using ﬁrst2n numbers
of W ;
6 Split rest 2n(log(n) − 1) numbers to two
sequences W1, W2 with length n(log(n) − 1) ;
7 Split X to X1, X2;
8 y1 ← −D11X1 + D12X2;
9 y2 ← −D21X1 + D22X2;
10 M y1 ← −
ButterflyTransform(W1, y1, n − 1);
11 M y2 ← −
ButterflyTransform(W2, y2, n − 1);
12 return Concat(M y1, M y2);
3.3. Butterﬂy Neural Network
The procedure explained in Algorithm 1 can be represented
by a butterﬂy graph similar to the FFT’s graph. The butterﬂy
network structure has been used for function representation [ 25]
and fast factorization for approximating linear transformation [6].
We adopt this graph as an architecture design for the layers of a
neural network. Figure 2 illustrates the architecture of a butterﬂy
network of base k = 2 applied on an input tensor of sizen× h× w.
The left ﬁgure shows how the recursive structure of the BFT as
a network. The right ﬁgure shows the constructed multi-layer
network which has log n Butterﬂy Layers (BFLayer). Note that

<!-- page 5 -->

Figure 3: Distribution of FLOPs: This ﬁgure shows that replacing the pointwise convolution with BFT reduces the size of the computational bottleneck.
the complexity of each Butterﬂy Layer is O(n) (2n operations),
therefore, the total complexity of the BFT architecture will be
O(n log n).
Each Butterﬂy layer can be augmented by batch norm and non-
linearity functions (e.g. ReLU, Sigmoid). In Section 4.2 we study
the effect of using different choices of these functions. We found
that both batch norm and nonlinear functions (ReLU and Sigmoid)
are not effective within BFLayers. Batch norm is not effective
mainly because its complexity is the same as the BFLayerO(n),
therefore, it doubles the computation of the entire transform. We
use batch norm only at the end of the transform. The non-linear
activation ReLU and Sigmoid zero out almost half of the values
in each BFLayer, thus multiplication of these values throughout the
forward propagation destroys all the information. The BFLayers
can be internally connected with residual connections in different
ways. In our experiments, we found that the best residual connec-
tions are the one that connect the input of the ﬁrst BFLayer to the
output of the last BFLayer. The base of the BFT affects the shape
and the number of FLOPs. We have empirically found that base
k = 4 achieves the highest accuracy while having the same number
FLOPs as the base k = 2 as shown in Figure 5c.
Butterﬂy network satisﬁes all the fusion network design prin-
ciples. There exist exactly one path between every input channel
to all the output channels, the degree of each node in the graph is
exactly k, the bottleneck size is n, and the number of edges are
O(n log n).
We use the BFT architecture as a replacement of the point-
wise convolution layer ( 1× 1 convs) in different CNN ar-
chitectures including MobileNetV1[ 15], ShufﬂeNetV2[ 28] and
MobileNetV3[14]. Our experimental results shows that under the
same number of FLOPs, the efﬁciency gain by BFT is more ef-
fective in terms of accuracy compared to the original model with
smaller channel rate. We show consistent accuracy improvement
across several architecture settings.
Fusing channels using BFT, instead of pointwise convolution
reduces the size of the computational bottleneck by a large-margin.
Figure 3 illustrate the percentage of the number of operations by
each block type throughout a forward pass in the network. Note that
when BFT is applied, the percentage of the depth-wise convolutions
increases by 8×.
4. Experiments
In this section, we demonstrate the performance of the pro-
posed BFT on large-scale image classiﬁcation tasks. To show-
case the strength of our method in designing very small networks,
we compare performance of Butterﬂy Transform with pointwise
convolutions in three state-of-the-art efﬁcient architectures: (1)
MobileNetV1, (2) ShufﬂeNetV2, and (3) MobileNetV3. We com-
pare our results with other type of structured matrices that have
O(n log n) computation (e.g. low-rank transform and circulant
transform). We also show that our method outperforms state-of-the
art architecture search methods at low FLOP ranges.
4.1. Image Classiﬁcation
4.1.1 Implementation and Dataset Details:
Following standard practice, we evaluate the performance of But-
terﬂy Transforms on the ImageNet dataset, at different levels of
complexity, ranging from 14 MFLOPS to 150 MFLOPs. ImageNet
classiﬁcation dataset contains 1.2M training samples and 50K vali-
dation samples, uniformly distributed across 1000 classes.
For each architecture, we substitute pointwise convolutions with
Butterﬂy Transforms. To keep the FLOP count similar between
BFT and pointwise convolutions, we adjust the channel numbers
in the base architectures (MobileNetV1, ShufﬂeNetV2, and Mo-
bileNetV3). For all architectures, we optimize our network by
minimizing cross-entropy loss using SGD. Speciﬁc learning rate
regimes are used for each architecture which can be found in the
Appendix. Since BFT is sensitive to weight decay, we found that
using little or no weight decay provides much better accuracy. We
experimentally found (Figure 5c) that butterﬂy base k = 4 per-
forms the best. We also used a custom weight initialization for the
internal weights of the Butterﬂy Transform which we outline below.
More information and intuition on these hyper-parameters can be
found in our ablation studies (Section 4.2).
Weight initialization: Proper weight initialization is critical
for convergence of neural networks, and if done improperly can
lead to instability in training, and poor performance. This is espe-
cially true for Butterﬂy Transforms due to the amplifying effect

<!-- page 6 -->

(a)
Flops ShufﬂeNetV2 ShufﬂeNetV2 +BFT Gain
14 M 50.86 (14 M)* 55.26 (14 M) 4.40
21 M 55.21 (21 M)* 57.83 (21 M) 2.62
40 M 59.70(41 M)*
60.30 (41 M) 61.33 (41 M) 1.63
1.03
(b)
Flops MobileNetV3 MobileNetV3+BFT Gain
10-15 M 49.8 (13 M) 55.21 (15 M) 5.41
(c)
Flops MobileNet MobileNet+BFT Gain
14 M 41.50 (14 M) 46.58 (14 M) 5.08
20 M 45.50 (21 M) 52.26 (23 M) 6.76
40 M 47.70 (34 M)
50.60 (41 M) 54.30 (35 M) 6.60
3.70
50 M 56.30 (49 M) 57.56 (51 M)
58.35 (52 M)
1.26
2.05
110 M 61.70 (110 M) 63.03 (112 M) 1.33
150 M 63.30 (150 M) 64.32 (150 M) 1.02
Table 1: These tables compare the accuracy of ShufﬂeNetV2, MobileNetV1 and MobileNetV3 when using standard pointwise convolution vs using BFTs
of the multiplications within the layer, which can create extremely
large or small values. A common technique for initializing point-
wise convolutions is to initialize weights uniformly from the range
(−x, x) where x =
√
6
nin+nout
, which is referred to as Xavier
initialization [10]. We cannot simply apply this initialization to
butterﬂy layers, since we are changing the internal structure.
We denote each entry B(n,k)
u,v as the multiplication of all the
edges in path from node u to v. We propose initializing the weights
of the butterﬂy layers from a range (−y, y), such that the multipli-
cation of all edges along paths, or equivalently values in B(n,k),
are initialized close to the range (−x, x). To do this, we solve for a
y which makes the expectation of the absolute value of elements of
B(n,k) equal to the expectation of the absolute value of the weights
with standard Xavier initialization, which is x/2. Let e1, .., elog(n)
be edges on the path p from input node u to output node v. We
have the following:
E[|B(n,k)
u,v |] = E[|
log(n)∏
i=1
ei|] = x
2 (8)
We initialize each ei in range (−y, y) where
( y
2 )log(n) = x
2 =⇒ y = x
1
log(n)∗ 2
log(n)−1
log(n) . (9)
4.1.2 MobileNetV1 + BFT
Figure 4:
MobileNetV1+BFT
Block
To add BFT to MobileNeV1, for all Mo-
bileNetV1 blocks, which consist of a
depthwise layer followed by a pointwise
layer, we replace the pointwise convo-
lution with our Butterﬂy Transform, as
shown in Figure 4. We would like to em-
phasize that this means we replace all
pointwise convolution in MobileNetV1,
with BFT. In Table 1, we show that we
outperform a spectrum of MobileNetV1s
from about 14M to 150M FLOPs with a
spectrum of MobileNetV1s+BFT within
the same FLOP range. Our experi-
ments with MobileNetV1+BFT include all combinations of width-
multiplier 1.00 and 2.00, as well as input resolutions 128, 160, 192,
and 224. We also add a width-multiplier 1.00 with input resolution
96 to cover the low FLOP range (14M). A full table of results can
be found in the Appendix.
In Table 1c we showcase that using BFT outperforms traditional
MobileNets across the entire spectrum, but is especially effective in
the low FLOP range. For example using BFT results in an increase
of 6.75% in top-1 accuracy at 23 MFLOPs. Note that MobileNetV1
+ BFT at 23 MFLOPs has much higher accuracy than MobileNetV1
at 41 MFLOPs, which means it can get higher accuracy with al-
most half of the FLOPs. This was achieved without changing the
architecture at all, other than simply replacing pointwise convo-
lutions, which means there are likely further gains by designing
architectures with BFT in mind.
4.1.3 ShufﬂeNetV2 + BFT
We modify the ShufﬂeNet block to add BFT to ShufﬂeNetv2. In
Table 1a we show results for ShufﬂeNetV2+BFT, versus the original
ShufﬂeNetV2. We have interpolated the number of output channels
to build ShufﬂeNetV2-1.25+BFT, to be comparable in FLOPs with
a ShufﬂeNetV2-0.5. We have compared these two methods for
different input resolutions (128, 160, 224) which results in FLOPs
ranging from 14M to 41M. ShufﬂeNetV2-1.25+BFT achieves about
1.6% better accuracy than our implementation of ShufﬂeNetV2-0.5
which uses pointwise convolutions. It achieves 1% better accuracy
than the reported numbers for ShufﬂeNetV2 [28] at 41 MFLOPs.
4.1.4 MobileNetV3 + BFT
We follow a procedure which is very similar to that of Mo-
bileNetV1+BFT, and simply replace all pointwise convolutions
with Butterﬂy Transforms. We trained a MobileNetV3+BFT Small
with a network-width of 0.5 and an input resolution 224, which
achieves 55.21% Top-1 accuracy. This model outperforms Mo-
bileNetV3 Small network-width of 0.35 and input resolution 224
at a similar FLOP range by about 5.4% Top-1, as shown in 1b.
Due to resource constraints, we only trained one variant of Mo-
bileNetV3+BFT.
4.1.5 Comparison with Neural Architecture Search
Including BFT in ShufﬂeNetV2 allows us to achieve higher
accuracy than state-of-the-art architecture search methods,
MNasNet[43], FBNet [46], and MobileNetV3 [14] on an extremely
low resource setting (∼ 14M FLOPs). These architecture search

<!-- page 7 -->

(a) BFT vs. Architecture Search
Model Accuracy
ShufﬂeNetV2+ BFT (14 M) 55.26
MobileNetV3Small-224-0.5+BFT (15 M) 55.21
FBNet-96-0.35-1 (12.9 M) 50.2
FBNet-96-0.35-2 (13.7 M) 51.9
MNasNet (12.7 M) 49.3
MobileNetV3Small-224-0.35 (13 M) 49.8
MobileNetV3Small-128-1.0 (12 M) 51.7
(b) BFT vs. Other Structured Matrix Approaches
Model Accuracy
MobilenetV1+BFT (35 M) 54.3
MobilenetV1 (42 M) 50.6
MobilenetV1+Circulant* (42 M) 35.68
MobilenetV1+low-rank* (37 M) 43.78
MobilenetV1+BPBP (35 M) 49.65
MobilenetV1+Toeplitz* (37 M) 40.09
MobilenetV1+FastFood* (37 M) 39.22
Table 2: These tables compare BFT with other efﬁcient network design approaches. In Table (a), we show that ShufﬂeNetV2 + BFT outperforms
state-of-the-art neural architecture search methods (MNasNet [43], FBNet[46], MobilenetV3[14]). In Table (b), we show that BFT achieves signiﬁcantly
higher accuracy than other structured matrix approaches which can be used for channel fusion. The * denotes that this is our implementation.
methods search a space of predeﬁned building blocks, where the
most efﬁcient block for channel fusion is the pointwise convolu-
tion. In Table 2a, we show that by simply replacing pointwise
convolutions in ShufﬂeNetv2, we are able to outperform state-of-
the-art architecture search methods in terms of Top-1 accuracy on
ImageNet. We hope that this leads to future work where BFT is in-
cluded as one of the building blocks in architecture searches, since
it provides an extremely low FLOP method for channel fusion.
4.1.6 Comparison with Structured Matrices
To further illustrate the beneﬁts of Butterﬂy Transforms, we com-
pare them with other structured matrix methods which can be used
to reduce the computational complexity of pointwise convolutions.
In Table 2b we show that BFT signiﬁcantly outperforms all these
other methods at a similar FLOP range. For comparability, we have
extended all the other methods to be used as replacements for point-
wise convolutions, if necessary. We then replaced all pointwise
convolutions in MobileNetV1 for each of the methods and report
Top-1 validation accuracy on ImageNet. Here we summarize these
other methods:
Circulant block: In this block, the matrix that represents the
pointwise convolution is a circulant matrix. In a circulant matrix
rows are cyclically shifted versions of one another [9]. The product
of this circulant matrix by a column can be efﬁciently computed in
O(n log(n)) using the Fast Fourier Transform (FFT).
Low-rank matrix: In this block, the matrix that represents
the pointwise convolution is the product of two log(n) rank ma-
trices (W = U V T ). Therefore the pointwise convolution can be
performed by two consequent small matrix product and the total
complexity isO(n log n).
Toeplitz Like: Toeplitz like matrices have been introduced in
[40]. They have been proven to work well on kernel approximation.
We have used displacement rank r = 1 in our experiments.
Fastfood: This block has been introduce in [22] and used in
Deep Fried ConvNets[49]. In Deep Fried Nets they replace fully
connected layers with FastFood. By unifying batch, height and
width dimension, we can use a fully connected layer as a pointwise
convolution.
BPBP: This method uses the butterﬂy network structure for
fast factorization for approximating linear transformation, such as
Discrete Fourier Transform (DFT) and the Hadamard transform[6].
We extend BPBP to work with pointwise convolutions by using
the trick explained in the Fastfood section above, and performed
experiments on ImageNet.
4.2. Ablation Study
Now, we study different elements of our BFT model. As men-
tioned earlier, residual connections and non-linear activations can
be augmented within our BFLayers. Here we show the perfor-
mance of these elements in isolation on CIFAR-10 dataset using
MobileNetv1 as the base network. The only exception is the But-
terﬂy Base experiment which was performed on ImageNet.
Model Accuracy
No residual 79.2
Every-other-Layer 81.12
First-to-Last 81.75
Table 3: Residual connections
Residual connections:
The graphs that are obtained
by replacing BFTransform
with pointwise convolutions
are very deep. Residual
connections generally help
when training deep networks.
We experimented with three
different ways of adding residual connections (1) First-to-Last,
which connects the input of the ﬁrst BFLayer to the output
of last BFLayer, (2) Every-other-Layer, which connects every
other BFLayer and (3) No-residual, where there is no residual
connection. We found the First-to-last is the most effective type of
residual connection as shown in Table 3.
With/Without Non-Linearity: As studied by [ 38] adding a
non-linearity function like ReLU or Sigmoid to a narrow layer
(with few channels) reduces the accuracy because it cuts off half
of the values of an internal layer to zero. In BFT, the effect of an
input channel i on an output channel o, is determined by the mul-
tiplication of all the edges on the path between i and o. Dropping
any value along the path to zero will destroy all the information
transferred between the two nodes. Dropping half of the values of
each internal layer destroys almost all the information in the entire
layer. Because of this, we don’t use any activation in the internal
Butterﬂy Layers. Figure 5b compares the the learning curves of
BFT models with and without non-linear activation functions.
With/Without Weight-Decay: We found that BFT is very sen-
sitive to the weight decay. This is because in BFT there is only one
path from an input channel i to an output channel o. The effect of
i on o is determined by the multiplication of all the intermediate
edges along the path between i and o. Pushing all weight values
toowards zero, will signiﬁcantly reduce the effect of the i on o.

<!-- page 8 -->

(a) Effect of weight-decay
 (b) Effect of activations
 (c) Effect of butterﬂy base
Figure 5: Design choices for BFT: a) In BFT we should not enforce weight decay, because it signiﬁcantly reduces the effect of input channels on output
channels. b) Similarly, we should not apply the common non-linear activation functions. These functions zero out almost half of the values in the intermediate
BFLayers, which leads to a catastrophic drop in the information ﬂow from input channels to the output channels. c) Butterﬂy base determines the structure of
BFT. Under 40M FLOP budget base k = 4 works the best.
Therefore, weight decay is very destructive in BFT. Figure 5a illus-
trates the learning curves with and without using weight decay on
BFT.
Butterﬂy base: The parameter k in B(n,k) determines the struc-
ture of the Butterﬂy Transform and has a signiﬁcant impact on the
accuracy of the model. The internal structure of the BFT will
contain logk(n) layers. Because of this, very small values of k lead
to deeper internal structures, which can be more difﬁcult to train.
Larger values of k are shallower, but have more computation, since
each node in layers inside the BFT has an out-degreee of k. With
large values of k, this extra computation comes at the cost of more
FLOPs.
We tested the values of k = 2 , 4, 8, n on MobileNetV1+BFT
with an input resolution of 160x160 which results in ∼ 40M
FLOPs. When k = n, this is equivalent to a standard pointwise
convolution. For a fair comparison, we made sure to hold FLOPs
consistent across all our experiments by varying the number of
channels, and tested all models with the same hyper-parameters on
ImageNet. Our results in Figure 5c show that k = 4 signiﬁcantly
outperforms all other values of k. Our intuition is that this setting
allows the block to be trained easily, due to its shallowness, and
that more computation than this is better spent elsewhere, such as in
this case increasing the number of channels. It is a likely possibility
that there is a more optimal value for k, which varies throughout
the model, rather than being ﬁxed. We have also only performed
this ablation study on a relatively low FLOP range ( 40M), so it
might be the case that larger architectures perform better with a
different value of k. There is lots of room for future exploration in
this design choice.
5. Drawbacks
A weakness of our model is that there is an increase in working
memory when using BFT since we must add substantially more
channels to maintain the same number of FLOPs as the original
network. For example, a MobileNetV1-2.0+BFT has the same
number of FLOPS as a MobileNetV1-0.5, which means it will use
about four times as much working memory. Please note that the
intermediate BFLayers can be computed in-place so they do not
increase the amount of working memory needed. Due to using
wider channels, GPU training time is also increased. In our im-
plementation, at the forward pass, we calculate B(n,k) from the
current weights of the BFLayers, which is a bottleneck in training.
Introducing a GPU implementation of butterﬂy operations would
greatly reduce training time.
6. Conclusion and Future Work
In this paper, we demonstrated how a family of efﬁcient trans-
formations referred to as the Butterﬂy Transforms can replace
pointwise convolutions in various neural architectures to reduce
the computation while maintaining accuracy. We explored many
design decisions for this block including residual connections, non-
linearities, weight decay, the power of theBFT , and also introduce
a new weight initialization, which allows us to signiﬁcantly outper-
form all other structured matrix approaches for efﬁcient channel
fusion that we are aware of. We also provided a set of principles
for fusion network design, and BFT exhibits all these properties.
As a drop-in replacement for pointwise convolutions in efﬁ-
cient Convolutional Neural Networks, we have shown that our
method signiﬁcantly increases accuracy of models, especially at
the low FLOP range, and can enable new capabilities on resource
constrained edge devices. It is worth noting that these neural archi-
tectures have not at all been optimized for BFT , and we hope that
this work will lead to more research towards networks designed
speciﬁcally with the Butterﬂy Transform in mind, whether through
manual design or architecture search. BFT can also be extended
to other domains, such as language and speech, as well as new
types of architectures, such as Recurrent Neural Networks and
Transformers.
We look forward to future inference implementations of But-
terﬂy structures which will hopefully validate our hypothesis that
this block can be implemented extremely efﬁciently, especially on
embedded devices and FPGAs. Finally, one of the major challenges
we faced was the large amount of time and GPU memory necessary
to train BFT , and we believe there is a lot of room for optimizing
training of this block as future work.

<!-- page 9 -->

Acknowledgement
Thanks Aditya Kusupati, Carlo Del Mundo, Golnoosh Samei,
Hessam Bagherinezhad, James Gabriel and Tim Dettmers for their
help and valuable comments. This work is in part supported by NSF
IIS 1652052, IIS 17303166, DARPA N66001-19-2-4031, 67102239
and gifts from Allen Institute for Artiﬁcial Intelligence.
References
[1] Renzo Andri, Lukas Cavigelli, Davide Rossi, and Luca Benini.
Yodann: An architecture for ultralow power binary-weight
cnn acceleration. IEEE Transactions on Computer-Aided
Design of Integrated Circuits and Systems, 2018.
[2] Hessam Bagherinezhad, Mohammad Rastegari, and Ali
Farhadi. LCNN: lookup-based convolutional neural network.
CoRR, abs/1611.06473, 2016.
[3] Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct
neural architecture search on target task and hardware. In
ICLR, 2019.
[4] Krzysztof Choromanski, Mark Rowland, Wenyu Chen, and
Adrian Weller. Unifying orthogonal Monte Carlo methods.
In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
Proceedings of the 36th International Conference on Machine
Learning, volume 97 of Proceedings of Machine Learning
Research, pages 1203–1212, Long Beach, California, USA,
09–15 Jun 2019. PMLR.
[5] Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-
Yaniv, and Yoshua Bengio. Binarized neural networks: Train-
ing neural networks with weights and activations constrained
to+ 1 or- 1. arXiv preprint arXiv:1602.02830, 2016.
[6] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and
Christopher Ré. Learning fast algorithms for linear
transforms using butterﬂy factorizations. arXiv preprint
arXiv:1903.05895, 2019.
[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li
Fei-Fei. Imagenet: A large-scale hierarchical image database.
In 2009 IEEE conference on computer vision and pattern
recognition, pages 248–255. Ieee, 2009.
[8] Emily L Denton, Wojciech Zaremba, Joan Bruna, Yann Le-
Cun, and Rob Fergus. Exploiting linear structure within
convolutional networks for efﬁcient evaluation. InAdvances
in neural information processing systems, pages 1269–1277,
2014.
[9] Caiwen Ding, Siyu Liao, Yanzhi Wang, Zhe Li, Ning Liu,
Youwei Zhuo, Chao Wang, Xuehai Qian, Yu Bai, Geng Yuan,
et al. C ir cnn: accelerating and compressing deep neural net-
works using block-circulant weight matrices. In Proceedings
of the 50th Annual IEEE/ACM International Symposium on
Microarchitecture, pages 395–408. ACM, 2017.
[10] Xavier Glorot and Yoshua Bengio. Understanding the dif-
ﬁculty of training deep feedforward neural networks. In
Yee Whye Teh and Mike Titterington, editors, Proceedings
of the Thirteenth International Conference on Artiﬁcial Intel-
ligence and Statistics, volume 9 of Proceedings of Machine
Learning Research, pages 249–256, Chia Laguna Resort, Sar-
dinia, Italy, 13–15 May 2010. PMLR.
[11] Song Han, Huizi Mao, and William J Dally. Deep com-
pression: Compressing deep neural networks with pruning,
trained quantization and huffman coding. arXiv preprint
arXiv:1510.00149, 2015.
[12] Song Han, Jeff Pool, John Tran, and William Dally. Learning
both weights and connections for efﬁcient neural network. In
NIPS, 2015.
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In CVPR, 2016.
[14] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh
Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu,
Ruoming Pang, Vijay Vasudevan, Quoc V . Le, and Hartwig
Adam. Searching for mobilenetv3. CoRR, abs/1905.02244,
2019.
[15] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry
Kalenichenko, Weijun Wang, Tobias Weyand, Marco An-
dreetto, and Hartwig Adam. Mobilenets: Efﬁcient convolu-
tional neural networks for mobile vision applications. arXiv
preprint arXiv:1704.04861, 2017.
[16] Gao Huang, Shichen Liu, Laurens van der Maaten, and Kil-
ian Q Weinberger. Condensenet: An efﬁcient densenet using
learned group convolutions. In CVPR, 2018.
[17] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kil-
ian Q Weinberger. Densely connected convolutional networks.
In CVPR, 2017.
[18] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-
Yaniv, and Yoshua Bengio. Quantized neural networks: Train-
ing neural networks with low precision weights and activa-
tions. arXiv preprint arXiv:1609.07061, 2016.
[19] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman.
Speeding up convolutional neural networks with low rank
expansions. arXiv preprint arXiv:1405.3866, 2014.
[20] Li Jing, Yichen Shen, Tena Dubcek, John Peurifoy, Scott A.
Skirlo, Max Tegmark, and Marin Soljacic. Tunable efﬁcient
unitary neural networks (EUNN) and their application to
RNN. CoRR, abs/1612.05231, 2016.
[21] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Im-
agenet classiﬁcation with deep convolutional neural networks.
In NIPS, 2012.
[22] Quoc Le, Tamas Sarlos, and Alex Smola. Fastfood - approxi-
mating kernel expansions in loglinear time. In 30th Interna-
tional Conference on Machine Learning (ICML), 2013.
[23] Yann LeCun, Bernhard E Boser, John S Denker, Donnie
Henderson, Richard E Howard, Wayne E Hubbard, and
Lawrence D Jackel. Handwritten digit recognition with a
back-propagation network. In Advances in neural informa-
tion processing systems, pages 396–404, 1990.
[24] Chong Li and CJ Richard Shi. Constrained optimization
based low-rank approximation of deep neural networks. In
ECCV, 2018.
[25] Yingzhou Li, Xiuyuan Cheng, and Jianfeng Lu. Butterﬂy-
net: Optimal function representation based on convolutional
neural networks. arXiv preprint arXiv:1805.07451, 2018.
[26] Yingzhou Li, Haizhao Yang, Eileen R. Martin, Kenneth L.
Ho, and Lexing Ying. Butterﬂy factorization. Multiscale
Modeling & Simulation, 13:714–732, 2015.

<!-- page 10 -->

[27] Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens,
Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang,
and Kevin Murphy. Progressive neural architecture search. In
Proceedings of the European Conference on Computer Vision
(ECCV), pages 19–34, 2018.
[28] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun.
Shufﬂenet v2: Practical guidelines for efﬁcient cnn architec-
ture design. In ECCV, 2018.
[29] Michaël Mathieu and Yann LeCun. Fast approximation of
rotations and hessians matrices. CoRR, abs/1404.7195, 2014.
[30] Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and
Hannaneh Hajishirzi. Espnetv2: A light-weight, power efﬁ-
cient, and general purpose convolutional neural network. In
CVPR, 2019.
[31] Tatsuya Member and Kazuyoshi Member. Bidirectional learn-
ing for neural network having butterﬂy structure. Systems and
Computers in Japan, 26:64 – 73, 04 1995.
[32] Marcin Moczulski, Misha Denil, Jeremy Appleyard, and
Nando de Freitas. Acdc: A structured efﬁcient linear layer.
CoRR, abs/1511.05946, 2015.
[33] Marina Munkhoeva, Yermek Kapushev, Evgeny Burnaev, and
Ivan V . Oseledets. Quadrature-based features for kernel ap-
proximation. CoRR, abs/1802.03832, 2018.
[34] D. Stott Parker. Random butterﬂy transformations with ap-
plications in computational linear algebra. Technical report,
1995.
[35] Mohammad Rastegari, Vicente Ordonez, Joseph Redmon,
and Ali Farhadi. Xnor-net: Imagenet classiﬁcation using
binary convolutional neural networks. In ECCV, 2016.
[36] Esteban Real, Sherry Moore, Andrew Selle, Saurabh Sax-
ena, Yutaka Leon Suematsu, Jie Tan, Quoc V Le, and Alexey
Kurakin. Large-scale evolution of image classiﬁers. In Pro-
ceedings of the 34th International Conference on Machine
Learning-Volume 70, pages 2902–2911. JMLR. org, 2017.
[37] Tara N. Sainath, Brian Kingsbury, Vikas Sindhwani, Ebru
Arisoy, and Bhuvana Ramabhadran. Low-rank matrix
factorization for deep neural network training with high-
dimensional output targets. 2013 IEEE International Con-
ference on Acoustics, Speech and Signal Processing, pages
6655–6659, 2013.
[38] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zh-
moginov, and Liang-Chieh Chen. Mobilenetv2: Inverted
residuals and linear bottlenecks. In CVPR, 2018.
[39] Karen Simonyan and Andrew Zisserman. Very deep convolu-
tional networks for large-scale image recognition. In ICLR,
2014.
[40] Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured
transforms for small-footprint deep learning. In C. Cortes,
N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett,
editors, Advances in Neural Information Processing Systems
28, pages 3088–3096. Curran Associates, Inc., 2015.
[41] Daniel Soudry, Itay Hubara, and Ron Meir. Expectation
backpropagation: Parameter-free training of multilayer neural
networks with continuous or discrete weights. In NIPS, 2014.
[42] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet,
Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent
Vanhoucke, and Andrew Rabinovich. Going deeper with
convolutions. In CVPR, 2015.
[43] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan,
and Quoc V Le. Mnasnet: Platform-aware neural architecture
search for mobile. arXiv preprint arXiv:1807.11626, 2018.
[44] Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai
Li. Learning structured sparsity in deep neural networks. In
NIPS, 2016.
[45] Mitchell Wortsman, Ali Farhadi, and Mohammad Rastegari.
Discovering neural wirings. CoRR, abs/1906.00586, 2019.
[46] Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang,
Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing
Jia, and Kurt Keutzer. Fbnet: Hardware-aware efﬁcient con-
vnet design via differentiable neural architecture search.arXiv
preprint arXiv:1812.03443, 2018.
[47] Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and
Jian Cheng. Quantized convolutional neural networks for
mobile devices. In CVPR, 2016.
[48] Lingxi Xie and Alan Yuille. Genetic cnn. In Proceedings
of the IEEE International Conference on Computer Vision ,
pages 1379–1388, 2017.
[49] Zichao Yang, Marcin Moczulski, Misha Denil, Nando de Fre-
itas, Alexander J. Smola, Le Song, and Ziyu Wang. Deep fried
convnets. 2015 IEEE International Conference on Computer
Vision (ICCV), pages 1476–1483, 2014.
[50] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun.
Shufﬂenet: An extremely efﬁcient convolutional neural net-
work for mobile devices. In CVPR, 2018.
[51] Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa,
and Zhiru Zhang. Building efﬁcient deep neural networks
with unitary group convolutions. CoRR, abs/1811.07755,
2018.
[52] Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen,
and Yuheng Zou. Dorefa-net: Training low bitwidth convo-
lutional neural networks with low bitwidth gradients. arXiv
preprint arXiv:1606.06160, 2016.
[53] Barret Zoph and Quoc V Le. Neural architecture search with
reinforcement learning. arXiv preprint arXiv:1611.01578 ,
2016.
[54] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V
Le. Learning transferable architectures for scalable image
recognition. In Proceedings of the IEEE conference on com-
puter vision and pattern recognition, pages 8697–8710, 2018.

<!-- page 11 -->

A. Experimental details
Here we explain our experimental setup. For all architectures,
we optimize our network by minimizing cross-entropy loss using
SGD.
A.1. MobileNetV1+BFT
We have used weight decay of 10−5. We train for 170 epochs.
We have used a constant learning rate 0.5 and decay it by 1
10 at
epochs 140, 160. For details on width multiplier of MobileNet and
input resolution on each experiment look at Table 4.
A.2. ShufﬂeNetV2+BFT
We have used weight decay of 10−5. We train for 300 epochs.
We start with a learning rate of 0.5 linearly decaying it to 0. All of
the pointwise convolutions are replaced by BFT as shown in Figure
6, except the ﬁrst pointwise convolution with input channel size of
24. For comparing under the similar number of FLOPs we have
slightly changed ShufﬂeNet’s layer width to create ShufﬂeNetV2-
1.25. This is the structure which is used for shufﬂeNetV2-1.25:
Layer output size Kernel Stride Repeat Width
Image 224×224 3
Conv1
Max pool
112×112
56×56
3×3
3×3
2
2 1 24
Stage 2 28×28
28×28
2
1
1
3 128
Stage 3 14×14
14×14
2
1
1
7 256
Stage 4 7×7
7×7
2
1
1
3 1024
Conv 5 7×7 BFT 1 1 1024
Global Pool 1×1 7×7
FC 1000
FLOPS 41
For details on input resolution on each experiment look at Table
5.
A.3. MobileNetV3+BFT
We have used weight decay of 10−5. We train for 200 epochs.
We start with a warm-up for the ﬁrst 5 epochs, starting from a
learning rate 0.1 and linearly increasing it to 0.5. Then we decay
learning rate from 0.5 to 0.0 using a cosine scheme in the remaining
195 epochs. For details on width multiplier and input resolution on
each experiment look at Table 6.
.
3x3 DWConv
Concat

3x3 DWConv
(stride = 2)
Concat
3x3 DWConv
(stride = 2)
Channel Split
Channel Shufﬂe Channel Shufﬂe
1x1 GConv
Channel Shufﬂe
3x3 DWConv
Add
1x1 GConv
1x1 GConv
Channel Shufﬂe
3x3 DWConv
(stride = 2)
Concat
1x1 GConv
3x3 AVG Pool
(stride = 2)
BN ReLU
BN
BN
ReLU ReLU
BN
BN
BN ReLU
BN ReLU
BN
BN ReLU
BN
(a) (b)
BN ReLU BN ReLU
BN
BN ReLU

BFT
BFT
BFT
BFT
BFT
Figure 6:
ShufﬂeNetV2+BFT Block

<!-- page 12 -->

MobileNet MobileNet+BFT gainwidth resolution ﬂops Accuracy width resolution ﬂops Accuracy
0.25 128 14 M 41.50 1.0 96 14 M 46.58 5.08
0.25 160 21 M 45.50 1.0 128 23 M 52.26 6.76
0.25 192
224
34 M
41 M
47.70
50.60 1.0 160 35 M 54.30 6.60
3.70
0.50 128 49 M 56.30 1.0
2.0
192
128
51 M
52 M
57.56
58.35
1.26
2.05
0.50 192 110 M 61.70 2.0 192 112 M 63.03 1.33
0.50 224 150 M 63.30 2.0 224 150 M 64.32 1.02
Table 4: Comparision between Mobilenet and Mobilenet+BFT. For comparision under similar number of FLOPs we have used wider channels in
MobileNet+BFT.
ShufﬂeNetV2 ShufﬂeNetV2 +BFT gainwidth resolution ﬂops Accuracy width resolution ﬂops Accuracy
0.50 128 14 M 50.86* 1.25 128 14 M 55.26 4.4
0.50 160 21 M 55.21* 1.25 160 21 M 57.83 2.62
0.50 224 41 M 59.70*
60.30 1.25 224 41 M 61.33 1.63
1.03
Table 5: Comparision between ShufﬂeNetV2 and ShufﬂeNetV2+BFT. For comparision under similar number of FLOPs we have used wider channels in
ShufﬂeNetV2+BFT.
MobileNetV3 MobileNetV3+BFT gainwidth resolution ﬂops Accuracy width resolution ﬂops Accuracy
Small-0.35 224 13 M 49.8 Small-0.5 224 15 M 55.21 5.41
Table 6: Comparision between MobileNetV3 and MobileNetV3+BFT. For comparision under similar number of FLOPs we have used wider channels in
MobileNetV3+BFT.
