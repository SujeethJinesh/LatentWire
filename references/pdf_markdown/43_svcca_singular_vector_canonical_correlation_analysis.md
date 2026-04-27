# references/43_svcca_singular_vector_canonical_correlation_analysis.pdf

<!-- page 1 -->

SVCCA: Singular Vector Canonical Correlation
Analysis for Deep Learning Dynamics and
Interpretability
Maithra Raghu,1,2 Justin Gilmer,1 Jason Yosinski,3 & Jascha Sohl-Dickstein1
1Google Brain 2Cornell University 3Uber AI Labs
maithrar
@ gmail
• com, gilmer
@ google
• com, yosinski
@ uber
• com, jaschasd
@ google
• com
Abstract
We propose a new technique, Singular Vector Canonical Correlation Analysis
(SVCCA), a tool for quickly comparing two representations in a way that is both
invariant to afﬁne transform (allowing comparison between different layers and
networks) and fast to compute (allowing more comparisons to be calculated than
with previous methods). We deploy this tool to measure the intrinsic dimension-
ality of layers, showing in some cases needless over-parameterization; to probe
learning dynamics throughout training, ﬁnding that networks converge to ﬁnal
representations from the bottom up; to show where class-speciﬁc information in
networks is formed; and to suggest new training regimes that simultaneously save
computation and overﬁt less.
1 Introduction
As the empirical success of deep neural networks ([7, 9, 18]) become an indisputable fact, the goal
of better understanding these models escalates in importance. Central to this aim is a core issue
of deciphering learned representations. Facets of this key question have been explored empirically,
particularly for image models, in [1, 2, 10, 12, 13, 14, 15, 19, 20]. Most of these approaches are
motivated by interpretability of learned representations. More recently, [11] studied the similarities
of representations learned by multiple networks by ﬁnding permutations of neurons with maximal
correlation.
In this work we introduce a new approach to the study of network representations, based on an
analysis of each neuron’s activation vector – the scalar outputs it emits on input datapoints. With
this interpretation of neurons as vectors (and layers as subspaces, spanned by neurons), we intro-
duce SVCCA, Singular Vector Canonical Correlation Analysis, an amalgamation of Singular Value
Decomposition and Canonical Correlation Analysis (CCA) [5], as a powerful method for analyzing
deep representations. Although CCA has not previously been used to compare deep representations,
it has been used for related tasks such as computing the similarity between modeled and measured
brain activity [16], and training multi-lingual word embeddings in language models [3].
The main contributions resulting from the introduction of SVCCA are the following:
1. We ask: is the dimensionality of a layer’s learned representation the same as the number
of neurons in the layer? Answer: No. We show that trained networks perform equally well
with a number of directions just a fraction of the number of neurons with no additional
training, provided they are carefully chosen with SVCCA (Section 2.1). We explore the
consequences for model compression (Section 4.4).
2. We ask: what do deep representation learning dynamics look like? Answer: Networks
broadly converge bottom up. Using SVCCA, we compare layers across time and ﬁnd they
31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
arXiv:1706.05806v2  [stat.ML]  8 Nov 2017

<!-- page 2 -->

index over dataset index over dataset index over dataset
Neurons with highest activations
(net1, net2)
Top SVD Directions
(net1, net2)
Top SVCCA directions
(net1, net2)
200
200
200
200
4
1
network
(a) (b) (c) (d)
Figure 1: To demonstrate SVCCA, we consider a toy regression task (regression target as in Figure 3). (a)
We train two networks with four fully connected hidden layers starting from different random initializations,
and examine the representation learned by the penultimate (shaded) layer in each network. (b) The neurons
with the highest activations in net 1 (maroon) and in net 2 (green). The x-axis indexes over the dataset: in
our formulation, the representation of a neuron is simply its value over a dataset (Section 2). (c) The SVD
directions — i.e. the directions of maximal variance — for each network. (d) The top SVCCA directions. We
see that each pair of maroon/green lines (starting from the top) are almost visually identical (up to a sign). Thus,
although looking at just neurons (b) seems to indicate that the networks learn very different representations,
looking at the SVCCA subspace (d) shows that the information in the representations are (up to a sign) nearly
identical.
solidify from the bottom up. This suggests a simple, computationally more efﬁcient method
of training networks, Freeze Training, where lower layers are sequentially frozen after a
certain number of timesteps (Sections 4.1, 4.2).
3. We develop a method based on the discrete Fourier transform which greatly speeds up the
application of SVCCA to convolutional neural networks (Section 3).
4. We also explore an interpretability question, of when an architecture becomes sensitive to
different classes. We ﬁnd that SVCCA captures the semantics of different classes, with
similar classes having similar sensitivities, and vice versa. (Section 4.3).
Experimental Details Most of our experiments are performed on CIFAR-10 (augmented with
random translations). The main architectures we use are a convolutional network and a residual
network1. To produce a few ﬁgures, we also use a toy regression task: training a four hidden layer
fully connected network with 1D input and 4D output, to regress on four different simple functions.
2 Measuring Representations in Neural Networks
Our goal in this paper is to analyze and interpret the representations learned by neural networks. The
critical question from which our investigation departs is: how should we deﬁne the representation
of a neuron? Consider that a neuron at a particular layer in a network computes a real-valued
function over the network’s input domain. In other words, if we had a lookup table of all possible
input→ output mappings for a neuron, it would be a complete portrayal of that neuron’s functional
form.
However, such inﬁnite tables are not only practically infeasible, but are also problematic to process
into a set of conclusions. Our primary interest is not in the neuron’s response to random data, but
rather in how it represents features of a speciﬁc dataset (e.g. natural images). Therefore, in this
study we take a neuron’s representation to be its set of responses over a ﬁnite set of inputs — those
drawn from some training or validation set.
More concretely, for a given datasetX ={x1,··· xm} and a neuroni on layerl,zzzl
i, we deﬁnezzzl
i to
be the vector of outputs onX, i.e.
zzzl
i = (zzzl
i(x1),··· ,zzzl
i(xm))
1Convnet layers: conv-conv-bn-pool-conv-conv-conv-bn-pool-fc-bn-fc-bn-out. Resnet layers:
conv-(x10 c/bn/r block)-(x10 c/bn/r block)-(x10 c/bn/r block)-bn-fc-out .
2

<!-- page 3 -->

Note that this is a different vector from the often-considered vector of the “representation at a layer
of a single input.” Herezzzl
i is a single neuron’s response over the entire dataset, not an entire layer’s
response for a single input. In this view, a neuron’s representation can be thought of as a single
vector in a high-dimensional space. Broadening our view from a single neuron to the collection of
neurons in a layer, the layer can be thought of as the set of neuron vectors contained within that
layer. This set of vectors will span some subspace. To summarize:
Considered over a datasetX withm examples, a neuron is a vector in Rm.
A layer is the subspace of Rm spanned by its neurons’ vectors.
Within this formalism, we introduce Singular Vector Canonical Correlation Analysis (SVCCA) as
a method for analysing representations. SVCCA proceeds as follows:
• Input: SVCCA takes as input two (not necessarily different) sets of neurons (typically
layers of a network)l1 ={zzzl1
1,...,zzzl1
m1} andl2 ={zzzl2
1,...,zzzl2
m2}
• Step 1 First SVCCA performs a singular value decomposition of each subspace to get sub-
subspacesl′
1⊂l1,l′
2⊂l2 which comprise of the most important directions of the original
subspacesl1,l 2. In general we take enough directions to explain 99% of variance in the
subspace. This is especially important in neural network representations, where as we will
show many low variance directions (neurons) are primarily noise.
• Step 2 Second, compute the Canonical Correlation similarity ([5]) of l′
1,l′
2: linearly trans-
form l′
1,l′
2 to be as aligned as possible and compute correlation coefﬁcients. In particu-
lar, given the output of step 1, l′
1 ={zzz′l1
1,...,zzz′l1
m′
1
},l′
2 ={zzz′l2
1,...,zzz′l2
m′
2
}, CCA linearly
transforms these subspaces ˜l1 = WXl′
1, ˜l2 = WYl′
2 such as to maximize the correlations
corrs ={ρ1,...ρ min(m′
1,m′
2)} between the transformed subspaces.
• Output: With these steps, SVCCA outputs pairs of aligned directions, (˜zzzl1
i , ˜zzzl2
i ) and how
well they correlate, ρi. Step 1 also produces intermediate output in the form of the top
singular values and directions.
For a more detailed description of each step, see the Appendix. SVCCA can be used to analyse
any two sets of neurons. In our experiments, we utilize this ﬂexibility to compare representations
across different random initializations, architectures, timesteps during training, and speciﬁc classes
and layers.
Figure 1 shows a simple, intuitive demonstration of SVCCA. We train a small network on a toy
regression task and show each step of SVCCA, along with the resulting very similar representations.
SVCCA is able to ﬁnd hidden similarities in the representations.
2.1 Distributed Representations
An important property of SVCCA is that it is truly a subspace method: both SVD and CCA work
with span(zzz1,...,z zzm) instead of being axis aligned to thezzzi directions. SVD ﬁnds singular vectors
zzz′
i = ∑m
j=1sijzzzj, and the subsequent CCA ﬁnds a linear transform W , giving orthogonal canon-
ically correlated directions {˜zzz1,..., ˜zzzm} = {∑m
j=1w1jzzz′
j,..., ∑m
j=1wmjzzz′
j}. In other words,
SVCCA has no preference for representations that are neuron (axes) aligned.
If representations are distributed across many dimensions, then this is a desirable property of a
representation analysis method. Previous studies have reported that representations may be more
complex than either fully distributed or axis-aligned [17, 21, 11] but this question remains open.
We use SVCCA as a tool to probe the nature of representations via two experiments:
(a) We ﬁnd that the subspace directions found by SVCCA are disproportionately important to
the representation learned by a layer, relative to neuron-aligned directions.
(b) We show that at least some of these directions are distributed across many neurons.
Experiments for (a), (b) are shown in Figure 2 as (a), (b) respectively. For both experiments, we ﬁrst
acquire two different representations,l1,l 2, for a layerl by training two different random initializa-
tions of a convolutional network on CIFAR-10. We then apply SVCCA tol1 andl2 to get directions
3

<!-- page 4 -->

0 100 200 300 400 500
Number of directions
0.2
0.4
0.6
0.8Accuracy
CIFAR10: Accuracy with SVCCA directions
 and random neurons
p2 (4096 neurons) SVCCA
p2 max acts neurons
p2 random neurons
fc1 (512 neurons) SVCCA
fc1 random neurons
fc2 (256 neurons) SVCCA
fc2 max acts neurons
0 10 20 30 40 50
Number of directions
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
Accuracy
CIFAR10 acc vs neurons used for SVCCA dirns
SVCCA fc1 (512 neurons)
SVCCA p2 (4096 neurons)
50 neurons for fc1
150 neurons for p2
300 neurons for p2
100 neurons for fc1
(a) (b)
Figure 2: Demonstration of (a) disproportionate importance of SVCCA directions, and (b) distributed nature
of some of these directions. For both panes, we ﬁrst ﬁnd the topk SVCCA directions by training two conv nets
on CIFAR-10 and comparing corresponding layers.(a) We project the output of the top three layers, pool1, fc1,
fc2, onto this top-k subspace. We see accuracy rises rapidly with increasing k, with evenk ≪ num neurons
giving reasonable performance, withno retraining. Baselines of randomk neuron subspaces and max activation
neurons require larger k to perform as well. (b): after projecting onto top k subspace (like left), dotted lines
then project again onto m neurons, chosen to correspond highly to the top k-SVCCA subspace. Many more
neurons are needed thank for better performance, suggesting distributedness of SVCCA directions.
{˜zzzl1
1,..., ˜zzzl1
m} and{˜zzzl2
1,..., ˜zzzl2
m}, ordered according to importance by SVCCA, with each ˜zzzli
j being a
linear combination of the original neurons, i.e. ˜zzzli
j = ∑m
r=1α(li)
jr zzzli
r .
For different values of k < m, we can then restrict layer li’s output to lie in the subspace of
span(˜zzzli
1,..., ˜zzzli
k ), the most usefulk-dimensional subspace as found by SVCCA, done by projecting
each neuron into thisk dimensional space.
We ﬁnd — somewhat surprisingly — that very few SVCCA directions are required for the network
to perform the task well. As shown in Figure 2(a), for a network trained on CIFAR-10, the ﬁrst
25 dimensions provide nearly the same accuracy as using all 512 dimensions of a fully connected
layer with 512 neurons. The accuracy curve rises rapidly with the ﬁrst few SVCCA directions, and
plateaus quickly afterwards, for k≪ m. This suggests that the useful information contained in m
neurons is well summarized by the subspace formed by the top k SVCCA directions. Two base-
lines for comparison are picking random and maximum activation neuron aligned subspaces and
projecting outputs onto these. Both of these baselines require far more directions (in this case: neu-
rons) before matching the accuracy achieved by the SVCCA directions. These results also suggest
approaches to model compression, which are explored in more detail in Section 4.4.
Figure 2(b) next demonstrates that these useful SVCCA directions are at least somewhat distributed
over neurons rather than axis-aligned. First, the top k SVCCA directions are picked and the rep-
resentation is projected onto this subspace. Next, the representation is further projected onto m
neurons, where them are chosen as those most important to the SVCCA directions . The resulting
accuracy is plotted for different choices ofk (given by x-axis) and different choices ofm (different
lines). That, for example, keeping even 100 fc1 neurons (dashed green line) cannot maintain the
accuracy of the ﬁrst 20 SVCCA directions (solid green line at x-axis 20) suggests that those 20
SVCCA directions are distributed across 5 or more neurons each, on average. Figure 3 shows a
further demonstration of the effect on the output of projecting onto top SVCCA directions, here for
the toy regression case.
Why the two step SV + CCA method is needed. Both SVD and CCA have important properties
for analysing network representations and SVCCA consequently beneﬁts greatly from being a two
step method. CCA is invariant to afﬁne transformations, enabling comparisons without natural
alignment (e.g. different architectures, Section 4.4). See Appendix B for proofs and a demonstrative
ﬁgure. While CCA is a powerful method, it also suffers from certain shortcomings, particularly in
determining how many directions were important to the original space X, which is the strength of
4

<!-- page 5 -->

0 50000 100000 150000 200000
4
3
2
1
0
1
2
3
4
Original output
 using 200 directions
0 50000 100000 150000 200000
Projection on top
 02 SVCCA directions
0 50000 100000 150000 200000
Projection on top
 06 SVCCA directions
0 50000 100000 150000 200000
Projection on top
 15 SVCCA directions
0 50000 100000 150000 200000
Projection on top
 30 SVCCA directions
Figure 3: The effect on the output of a latent representation being projected onto top SVCCA directions in
the toy regression task. Representations of the penultimate layer are projected onto 2, 6, 15, 30 top SVCCA
directions (from second pane). By 30, the output looks very similar to the full 200 neuron output (left).
SVD. See Appendix for an example where naive CCA performs badly. Both the SVD and CCA
steps are critical to the analysis of learning dynamics in Section 4.1.
3 Scaling SVCCA for Convolutional Layers
Applying SVCCA to convolutional layers can be done in two natural ways:
(1) Same layer comparisons: IfX,Y are the same layer (at different timesteps or across ran-
dom initializations) receiving the same input we can concatenate along the pixel (heighth,
width w) coordinates to form a vector: a conv layer h×w×c maps to c vectors, each
of dimension hwd, where d is the number of datapoints. This is a natural choice because
neurons at different pixel coordinates see different image data patches to each other. When
X,Y are two versions of the same layer, thesec different views correspond perfectly.
(2) Different layer comparisons: WhenX,Y are not the same layer, the image patches seen by
different neurons have no natural correspondence. But we can ﬂatten anh×w×c conv into
hwc neurons, each of dimension d. This approach is valid for convs in different networks
or at different depths.
3.1 Scaling SVCCA with Discrete Fourier Transforms
Applying SVCCA to convolutions introduces a computational challenge: the number of neurons
(h×w×c) in convolutional layers, especially early ones, is very large, making SVCCA prohibitively
expensive due to the large matrices involved. Luckily the problem of approximate dimensionality
reduction of large matrices is well studied, and efﬁcient algorithms exist, e.g. [4].
For convolutional layers however, we can avoid dimensionality reduction and perform exact
SVCCA, even for large networks. This is achieved by preprocessing each channel with a Discrete
Fourier Transform (which preserves CCA due to invariances, see Appendix), causing all (covari-
ance) matrices to be block-diagonal. This allows all matrix operations to be performed block by
block, and only over the diagonal blocks, vastly reducing computation. We show:
Theorem 1. Suppose we have a translation invariant (image) dataset X and convolutional layers
l1, l2. Letting DFT (li) denote the discrete fourier transform applied to each channel of li, the
covariancecov(DFT (l1),DFT (l2)) is block diagonal, with blocks of sizec×c.
We make only two assumptions: 1) all layers below l1, l2 are either conv or pooling layers with
circular boundary conditions (translation equivariance) 2) The dataset X has all translations of the
imagesXi. This is necessary in the proof for certain symmetries in neuron activations, but these
symmetries typically exist in natural images even without translation invariance, as shown in Fig-
ure App.2 in the Appendix. Below are key statements, with proofs in Appendix.
Deﬁnition 1. Say a single channel image dataset X of images is translation invariant if for any
(wlogn×n) imageXi∈X, with pixel values{zzz11,...zzznn},X (a,b)
i ={zzzσa(1)σb(1),...zzzσa(n)σb(n)}
is also inX, for all 0≤a,b≤n− 1, whereσa(i) =a +i mod n (and similarly forb).
For a multiple channel imageXi, an (a,b ) translation is an(a,b ) height/width shift on every channel
separately.X is then translation invariant as above.
5

<!-- page 6 -->

To prove Theorem 1, we ﬁrst show another theorem:
Theorem 2. Given a translation invariant dataset X, and a convolutional layer l with channels
{c1,...c k} applied toX
(a) the DFT of ci,FcF T has diagonal covariance matrix (with itself).
(b) the DFT of ci,cj,FciFT ,FcjFT have diagonal covariance with each other.
Finally, both of these theorems rely on properties of circulant matrices and their DFTs:
Lemma 1. The covariance matrix of ci applied to translation invariant X is circulant and block
circulant.
Lemma 2. The DFT of a circulant matrix is diagonal.
4 Applications of SVCCA
4.1 Learning Dynamics with SVCCA
We can use SVCCA as a window into learning dynamics by comparing the representation at a
layer at different points during training to its ﬁnal representation. Furthermore, as the SVCCA
computations are relatively cheap to compute compared to methods that require training an auxiliary
network for each comparison [1, 10, 11], we can compare all layers during training at all timesteps
to all layers at the ﬁnal time step, producing a rich view into the learning process.
The outputs of SVCCA are the aligned directions (˜xi, ˜yi), how well they align, ρi, as well as in-
termediate output from the ﬁrst step, of singular values and directions, λ(i)
X,x′(i), λ(j)
Y ,y′(j). We
condense these outputs into a single value, the SVCCA similarity ¯ρ, that encapsulates how well the
representations of two layers are aligned with each other,
¯ρ = 1
min (m1,m 2)
∑
i
ρi, (1)
where min (m1,m 2) is the size of the smaller of the two layers being compared. The SVCCA
similarity ¯ρ is the average correlation across aligned directions, and is a direct multidimensional
analogue of Pearson correlation.
The SVCCA similarity for all pairs of layers, and all time steps, is shown in Figure 4 for a convnet
and a resnet architecture trained on CIFAR10.
4.2 Freeze Training
Observing in Figure 4 that networks broadly converge from the bottom up, we propose a training
method where we successively freeze lower layers during training, only updating higher and higher
layers, saving all computation needed for deriving gradients and updating in lower layers.
We apply this method to convolutional and residual networks trained on CIFAR-10, Figure 5, using
a linear freezing regime: in the convolutional network, each layer is frozen at a fraction (layer num-
ber/total layers) of total training time, while for resnets, each residual block is frozen at a fraction
(block number/total blocks). The vertical grey dotted lines show which steps have another set of lay-
ers frozen. Aside from saving computation, Freeze Training appears to actively help generalization
accuracy, like early stopping but with different layers requiring different stopping points.
4.3 Interpreting Representations: when are classes learned?
We also can use SVCCA to compare how correlated representations in each layer are with the logits
of each class in order to measure how knowledge about the target evolves throughout the network.
In Figure 6 we apply the DFT CCA technique on the Imagenet Resnet [6]. We take ﬁve different
classes and for different layers in the network, compute the DFT CCA similarity between the logit
of that class and the network layer. The results successfully reﬂect semantic aspects of the classes:
the ﬁretruck class sensitivity line is clearly distinct from the two pairs of dog breeds, and network
develops greater sensitivity to ﬁretruck earlier on. The two pairs of dog breeds, purposefully chosen
so that each pair is similar to the other in appearance, have cca similarity lines that are very close to
each other through the network, indicating these classes are similar to each other.
6

<!-- page 7 -->

layer (during training)
layer (end of training)
Convnet, CIFAR-10Resnet, CIFAR-10
layer (during training)
layer (end of training) layer (end of training) layer (end of training)
Weighted SVCCA scale
0% trained 35% trained 75% trained 100% trained
Figure 4: Learning dynamics plots for conv (top) and res (bottom) nets trained on CIFAR-10. Each pane is
a matrix of size layers × layers, with each entry showing the SVCCA similarity ¯ρ between the two layers.
Note that learning broadly happens ‘bottom up’ – layers closer to the input seem to solidify into their ﬁnal
representations with the exception of the very top layers. Per layer plots are included in the Appendix. Other
patterns are also visible – batch norm layers maintain nearly perfect similarity to the layer preceding them due
to scaling invariance (with a slight reduction since batch norm changes the SVD directions which capture 99%
of the variance). In the resnet plot, we see a stripe like pattern due to skip connections inducing high similarities
to previous layers.
0 20000 40000 60000 80000 100000 120000 140000 160000
Train step
0.70
0.75
0.80
0.85
0.90
Accuracy
CIFAR10 Conv Freeze Training
test acc base
test acc freeze
0 20000 40000 60000 80000 100000 120000 140000 160000
Train step
0.70
0.75
0.80
0.85
0.90 CIFAR10 Resnet Freeze Training
test acc base
 test acc freeze
Figure 5: Freeze Training reduces training cost and improves generalization. We apply Freeze Training to a
convolutional network on CIFAR-10 and a residual network on CIFAR-10. As shown by the grey dotted lines
(which indicate the timestep at which another layer is frozen), both networks have a ‘linear’ freezing regime:
for the convolutional network, we freeze individual layers at evenly spaced timesteps throughout training. For
the residual network, we freeze entire residual blocks at each freeze step. The curves were averaged over ten
runs.
4.4 Other Applications: Cross Model Comparison and compression
SVCCA similarity can also be used to compare the similarity of representations across different
random initializations, and even different architectures. We compare convolutional networks on
CIFAR-10 across random initializations (Appendix) and also a convolutional network to a residual
network in Figure 7, using the DFT method described in 3.
In Figure 3, we saw that projecting onto the subspace of the top few SVCCA directions resulted in
comparable accuracy. This observations motivates an approach to model compression. In particular,
letting the output vector of layer l bexxx(l) ∈ Rn×1, and the weights W (l), we replace the usual
W (l)xxx(l) with (W (l)PT
x )(Pxxxx(l)) wherePx is ak×n projection matrix, projectingxxx onto the top
SVCCA directions. This bottleneck reduces both parameter count and inference computational cost
7

<!-- page 8 -->

0 10 20 30 40 50 60 70 80
Layer Number
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
CCA Similarity with Class
CCA Similarity (using DFT) of Layers in
 Imagenet Resnet to Different Classes
s_terrier
w_terrier
husky
eskimo_dog
fire truck
Figure 6: We plot the CCA similarity using the Discrete Fourier Transform between the logits of ﬁve classes
and layers in the Imagenet Resnet. The classes are ﬁretruck and two pairs of dog breeds (terriers and husky
like dogs: husky and eskimo dog) that are chosen to be similar to each other. These semantic properties are
captured in CCA similarity, where we see that the line corresponding to ﬁretruck is clearly distinct from the
two pairs of dog breeds, and the two lines in each pair are both very close to each other, reﬂecting the fact that
each pair consists of visually similar looking images. Firetruck also appears to be easier for the network to
learn, with greater sensitivity displayed much sooner.
in bncv bncv bncv bncv bncv bncv bncv bn
Resnet layers
p2
bn2
c5
c4
c3
p1
bn1
c2
c1
in
Convnet layers
DFT CCA similarity between
 Resnet and Convnet on CIFAR10
0.0
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
Figure 7: We plot the CCA similarity using the Discrete Fourier Transform between convolutional layers of a
Resnet and Convnet trained on CIFAR-10. We ﬁnd that the lower layrs of both models are noticeably similar to
each other, and get progressively less similar as we compare higher layers. Note that the highest layers of the
resnet are least similar to the lower layers of the convnet.
for the layer by a factor∼ k
n. In Figure App.5 in the Appendix, we show that we can consecutively
compress top layers with SVCCA by a signiﬁcant amount (in one case reducing each layer to 0.35
original size) and hardly affect performance.
5 Conclusion
In this paper we present SVCCA, a general method which allows for comparison of the learned dis-
tributed representations between different neural network layers and architectures. Using SVCCA
we obtain novel insights into the learning dynamics and learned representations of common neural
network architectures. These insights motivated a new Freeze Training technique which can reduce
the number of ﬂops required to train networks and potentially even increase generalization perfor-
mance. We observe that CCA similarity can be a helpful tool for interpretability, with sensitivity
to different classes reﬂecting their semantic properties. This technique also motivates a new algo-
rithm for model compression. Finally, the “lower layers learn ﬁrst” behavior was also observed for
recurrent neural networks as shown in Figure App.6 in the Appendix.
8

<!-- page 9 -->

References
[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classiﬁer
probes. arXiv preprint arXiv:1610.01644, 2016.
[2] David Eigen, Jason Rolfe, Rob Fergus, and Yann LeCun. Understanding deep architectures
using a recursive convolutional network. arXiv preprint arXiv:1312.1847, 2013.
[3] Manaal Faruqui and Chris Dyer. Improving vector space word representations using multilin-
gual correlation. Association for Computational Linguistics, 2014.
[4] Nathan Halko, Martinsson Per-Gunnar, and Joel A. Tropp. Finding structure with random-
ness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM
Rev., 53:217–288, 2011.
[5] D. R. Hardoon, S. Szedmak, and J. Shawe-Taylor. Canonical correlation analysis: An overview
with application to learning methods. Neural Computation, 16:2639–2664, 2004.
[6] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. CoRR, abs/1512.03385, 2015.
[7] Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep
Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neu-
ral networks for acoustic modeling in speech recognition: The shared views of four research
groups. IEEE Signal Processing Magazine, 29(6):82–97, 2012.
[8] Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 1985.
[9] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classiﬁcation with deep
convolutional neural networks. In Advances in neural information processing systems , pages
1097–1105, 2012.
[10] Karel Lenc and Andrea Vedaldi. Understanding image representations by measuring their
equivariance and equivalence. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 991–999, 2015.
[11] Y . Li, J. Yosinski, J. Clune, H. Lipson, and J. Hopcroft. Convergent Learning: Do different
neural networks learn the same representations? In International Conference on Learning
Representations (ICLR), May 2016.
[12] Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning:
Do different neural networks learn the same representations? In Feature Extraction: Modern
Questions and Challenges, pages 196–212, 2015.
[13] Aravindh Mahendran and Andrea Vedaldi. Understanding deep image representations by in-
verting them. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recog-
nition, pages 5188–5196, 2015.
[14] Gr ´egoire Montavon, Mikio L Braun, and Klaus-Robert M ¨uller. Kernel analysis of deep net-
works. Journal of Machine Learning Research, 12(Sep):2563–2581, 2011.
[15] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional
networks: Visualising image classiﬁcation models and saliency maps. arXiv preprint
arXiv:1312.6034, 2013.
[16] David Sussillo, Mark M Churchland, Matthew T Kaufman, and Krishna V Shenoy. A neural
network that ﬁnds a naturalistic solution for the production of muscle activity. Nature neuro-
science, 18(7):1025–1033, 2015.
[17] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian
Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint
arXiv:1312.6199, 2013.
[18] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V . Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural ma-
chine translation system: Bridging the gap between human and machine translation. arXiv
preprint arXiv:1609.08144, 2016.
[19] Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding
neural networks through deep visualization. In Deep Learning Workshop, International Con-
ference on Machine Learning (ICML), 2015.
9

<!-- page 10 -->

[20] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
European conference on computer vision, pages 818–833. Springer, 2014.
[21] Bolei Zhou, Aditya Khosla, `Agata Lapedriza, Aude Oliva, and Antonio Torralba. Object de-
tectors emerge in deep scene cnns. In International Conference on Learning Representations
(ICLR), volume abs/1412.6856, 2014.
10

<!-- page 11 -->

Appendix
A Mathematical details of CCA and SVCCA
Canonical Correlation ofX,Y Finding maximal correlations betweenX,Y can be expressed as
ﬁndinga,b to maximise:
aT ΣXYb√
aT ΣXXa
√
bT ΣYY b
where ΣXX, ΣXY, ΣYX, ΣYY are the covariance and cross-covariance terms. By performing the
change of basis ˜x˜x˜x1 = Σ1/2
xx a and ˜y˜y˜y1 = Σ1/2
YY b and using Cauchy-Schwarz we recover an eigenvalue
problem:
˜x˜x˜x1 = argmax
[
xT Σ−1/2
XX ΣXY Σ−1
YY ΣYX Σ−1/2
XX x
||x||
]
(*)
SVCCA Given two subspaces X ={xxx1,...,xxxm1},Y ={yyy1,...,yyym2}, SVCCA ﬁrst performs a
singular value decomposition onX,Y . This results in singular vectors{x′x′x′1,...,x′x′x′m1} with associ-
ated singular values{λ1,...,λ m1} (forX, and similarly for Y ). Of these m1 singular vectors, we
keep the topm′
1 wherem′
1 is the smallest value that ∑m′
1
i=1|λi|(≥ 0.99 ∑m1
i=1|λi|). That is, 99% of
the variation ofX is explainable by the top m′
1 vectors. This helps remove directions/neurons that
are constant zero, or noise with small magnitude.
Then, we apply Canonical Correlation Analysis (CCA) to the sets{x′x′x′1,...,x′x′x′m′
1},{y′y′y′
1,...,y′y′y′
m′
2
} of
top singular vectors.
CCA is a well established statistical method for understanding the similarity of two different sets
of random variables – given our two sets of vectors{x′x′x′1,...,x′x′x′m′
1},{y′y′y′
1,...,y′y′y′
m′
2
}, we wish to ﬁnd
linear transformations, WX,WY that maximally correlate the subspaces. This can be reduced to
an eigenvalue problem. Solving this results in linearly transformed subspaces ˜X, ˜Y with directions
˜xxxi, ˜yyyi that are maximally correlated with each other, and orthogonal to ˜xxxj, ˜yyyj,j < i. We let ρi =
corr(˜xxxi, ˜yyyi). In summary, we have:
SVCCA Summary
1. Input: X,Y
2. Perform: SVD(X), SVD(Y). Output: X′ =UX,Y ′ =VY
3. Perform CCA( X′, Y′). Output: ˜X = WXX′, ˜Y = WYY′ and corrs =
{ρ1,...ρ min(m1,m2)}
B Additional Proofs and Figures from Section 2.1
Proof of Orthonormal and Scaling Invariance of CCA:
We can see this using equation (*) as follows: suppose U,V are orthonormal transforms applied to
the sets X,Y . Then it follows that Σa
XX becomes UΣa
XXUT , for a ={1,−1, 1/2,−1/2}, and
similarly forY andV . Also note ΣXY becomesUΣXYVT . Equation (*) then becomes
˜x1 = argmax
[
xTUΣ−1/2
XX ΣXY Σ−1
YY ΣYX Σ−1/2
XX UTx
||x||
]
So if ˜u is a solution to equation (*), thenU ˜u is a solution to the equation above, which results in the
same correlation coefﬁcients.
B.0.1 The importance of SVD: how many directions matter?
While CCA is excellent at identifying useful learned directions that correlate, independent of certain
common transforms, it doesn’t capture the full picture entirely. Consider the following setting:
11

<!-- page 12 -->

5
0
5
10
15
20
80
60
40
20
0
20
40
0 2000 4000 6000 8000 10000
40
20
0
20
40
60
80
CIFAR10 Signal and Distorted Version
5
0
5
10
15
20
0 2000 4000 6000 8000 10000
5
0
5
10
15
20
Output after Canonical Correlation
Figure App.1: This ﬁgure shows the ability of CCA to deal with orthogonal and scaling transforms.
In the ﬁrst pane, the maroon plot shows one of the highest activation neurons in the penultimate
layer of a network trained on CIFAR-10, with the x-axis being (ordered) image ids and the y-axis
being activation on that image. The green plots show two resulting distorted directions after this and
two of the other top activation neurons are permuted, rotated and scaled. Pane two shows the result
of applying CCA to the distorted directions and the original signal, which succeeds in recovering
the original signal.
suppose we have subspaces A,B,C , with A being 50 dimensions,B being 200 dimensions, 50 of
which are perfectly aligned with A and the other 150 being noise, and C being 200 dimensions, 50
of which are aligned withA (andB) and the other 150 being useful, but different directions.
Then looking at the canonical correlation coefﬁcients of(A,B ) and (A,C ) will give thesame result,
both being 1 for 50 values and 0 for everything else. But these are two very different cases – the
subspaceB is indeed well represented by the50 directions that are aligned withA. But the subspace
C has 150 more useful directions.
This distinction becomes particularly important when aggregating canonical correlation coefﬁcients
as a measure of similarity, as used in analysing network learning dynamics. However, by ﬁrst ap-
plying SVD to determine the number of directions needed to explain 99% of the observed variance,
we can distinguish between pathological cases like the one above.
C Proof of Theorem 1
Here we provide the proofs for Lemma 1, Lemma 2, Theorem 2 and ﬁnally Theorem 1.
A preliminary note before we begin:
When we consider a (wlog)n byn channelc of a convolutional layer, we assume it has shape


zzz0,0 zzz1,2 ... z zz0,n−1
zzz1,0 zzz2,2 ... z zz1,n−1
... ... ... ...
zzzn−1,0 zzzn−1,1 ... z zzn−1,n−1


12

<!-- page 13 -->

(a)
 (b)
 (c)
 (d)
Figure App.2: This ﬁgure visualizes the covariance matrix of one of the channels of a resnet
trained on Imagenet. Black correspond to large values and white to small values.(a) we compute the
covariance without a translation invariant dataset and without ﬁrst preprocessing the images by DFT.
We see that the covariance matrix is dense. (b) We compute the covariance after applying DFT, but
without augmenting the dataset with translations. Even without enforcing translation invariance, we
see that the covariance in the DFT basis is approximately diagonal. (c) Same as (a), but the dataset
is augmented to be fully translation invariant. The covariance in the pixel basis is still dense. (d)
Same as (c), but with dataset augmented to be translation invariant. The covariance matrix is exactly
diagonal for a translation invariant dataset in a DFT basis.
When computing the covariance matrix however, we vectorizec by stacking the columns under each
other, and call the resultvec(c):
vec(c) =


zzz0,0
zzz1,0
...
zzzn−1,0
zzz0,1
...
zzzn−1,n−1


:=


zzz0
zzz1
...
zzzn−1
zzzn
...
zzzn2−1


One useful identity when switching between these two notations (see e.g. [8]) is
vec(AcB) = (BT⊗A)vec(c)
whereA,B are matrices and⊗ is the Kronecker product. A useful observation arising from this is:
Lemma 3. The CCA vectors of DFT (ci),DFT (cj) are the same (up to a rotation by F ) as the
CCA ofci,cj.
Proof: From Section B we know that unitary transforms only rotate theCCA directions. But while
DFT pre and postmultiplies byF,F T – unitary matrices, we cannot directly apply this as the result
is for unitary transforms on vec(ci). But, using the identity above, we see that vec(DFT (ci)) =
vec(FciFT ) = (F⊗F )vec(ci), which is unitary asF is unitary. Applying the same identity tocj,
we can thus conclude that the DFT preserves CCA (up to rotations).
As Theorem 1 preprocesses the neurons with DFT, it is important to note that by the Lemma above,
we do not change the CCA vectors (except by a rotation).
C.1 Proof of Lemma 1
Proof. Translation invariance is preserved We show inductively that any translation invariant input
to a convolutional channel results in a translation invariant output: Suppose the input to channel c,
(n byn) is translation invariant. It is sufﬁcient to show that for inputsXi,Xj and 0≤a,b,≤n− 1,
c(Xi) + (a,b ) mod n = c(Xj). But an (a,b ) shift in neuron coordinates in c corresponds to a
(height stride·a, width stride·b) shift in the input. And as X is translation invariant, there is some
Xj =Xi + (height stride·a, width stride·b).
cov(c) is circulant:
13

<!-- page 14 -->

LetX be (by proof above) a translation invariant input to a channelc in some convolution or pooling
layer. The empirical covariance,cov(c) is then2 byn2 matrix computed by (assumingc is centered)
1
|X|
∑
Xi∈X
vec(c(Xi))·vec(c(Xi))T
So,cov(c)ij = 1
|X|zzzT
izzzj = 1
|X|
∑
Xl∈XzzzT
i (Xl)zzzj(Xl), i.e. the inner products of the neurons i and
j.
The indexesi andj refer to the neurons in their vectorized order invec(c). But in the matrix ordering
of neurons inc,i andj correspond to some (a1,b 1) and (a2,b 2). If we applied a translation (a,b ),
to both, we would get new neuron coordinates (a1 +a,b 1 +b), (a2 +a,b 2 +b) (all coordinates
mod n) which would correspond toi +an +b mod n2 andj +an +b mod n2, by our stacking
of columns and reindexing.
Let τa,b be the translation in inputs corresponding to an (a,b ) translation in c, i.e. τa,b =
(height stride·a, width stride·b). Then clearlyzzz(a1,b1)(Xi) =zzz(a1+a,b1+b)(τ(a,b)(Xi), and similarly
forzzz(a2,b2)
It follows that 1
|X|zzzT
(a1,b1)zzz(a2,b2) = 1
|X|zzzT
(a1+b,b1+b)zzz(a2+a,b2+b), or, withvec(c) indexing
1
|X|zzzT
izzzj = 1
|X|zzzT
(i+an+b mod n2)zzz(j+an+b mod n2)
This gives us the circulant structure ofcov(c).
cov(c) is block circulant: Letzzz(i) be the ith column ofc, andzzz(j) the jth. Invec(c), these correspond
tozzz(i−1)n,...zzzin−1 andzzz(j−1)n,...zzzjn−1, and then byn submatrix at those row and column in-
dexes ofcov(vec(c)) corresponds to the covariance of columni,j . But then we see that the covari-
ance of columnsi+k,j +k, corresponding to the covariance of neuronszzz(i−1)n+k·n,...zzzin−1+k·n,
andzzz(j−1)n+k·n,...zzzjn−1+k·n, which corresponds to the 2-d shift (1, 0), applied to every neuron.
So by an identical argument to above, we see that for all 0≤k≤n− 1
cov(zzz(i),zzz(j)) =cov(zzz(i+k),zzz(j+k))
In particular,cov(vec(c)) is block circulant.
An examplecov(vec(c)) withc being 3 by 3 look like below:
[A0 A1 A2
A2 A0 A1
A1 A2 A0
]
where eachAi is itself a circulant matrix.
C.2 Proof of Lemma 2
Proof. This is a standard result, following from expressing a circulant matrix A in terms of its
diagonal form , i.e. A =V ΣVT with the columns ofV being its eigenvectors. Noting thatV =F ,
the DFT matrix, and that vectors of powers of ωk = exp( 2πik
n ), ωj = exp( 2πik
n ) are orthogonal
gives the result.
C.3 Proof of Theorem 2
Proof. Starting with (a), we need to show thatcov(vec(DFT (ci)),vec (DFT (ci)) is diagonal. But
by the identity above, this becomes:
cov(vec(DFT (ci)),vec (DFT (ci)) = (F⊗F )vec(ci)vec(ci)T (F⊗F )∗
14

<!-- page 15 -->

By Lemma 1, we see that
cov(vec(ci)) =vec(ci)vec(ci)T =


A0 A1 ... A n−1
An−1 A0 ... A n−2
... ... ... ...
A1 A2 ... A 0


with eachAi circulant.
And socov(vec(DFT (ci)),vec (DFT (ci)) becomes


f00F f 01F ... f 0,n−1F
f10F f 11F ... f 1,n−1F
... ... ... ...
fn−1,0F f n−1,1F ... f n−1,n−1F




A0 A1 ... A n−1
An−1 A0 ... A n−2
... ... ... ...
A1 A2 ... A 0




f∗
00F∗ f∗
10F∗ ... f ∗
n−1,0F∗
f∗
01F∗ f∗
11F∗ ... f ∗
n−1,1F∗
... ... ... ...
f∗
0,n−1F∗ f∗
1,n−1F∗ ... f ∗
n−1,n−1F∗


From this, we see that thesjth entry has the form
n−1∑
l=0
(n−1∑
k=0
fskFAl−k
)
f∗
ljF∗ =
∑
k,l
fskf∗
ljFAl−kF∗
Letting [FArF∗] denote the coefﬁcient of the termFArF∗, we see that (addition being mod n)
[FArF∗] =
n−1∑
k=0
fskf∗
(k+r)j =
∑
k
e
2πisk
n ·e
−2πij(k+r)
n =e
−2πijr
n
n−1∑
k=0
e
2πik(s−j)
n =e
−2πijr
n ·δsj
with the last step following by the fact that the sum of powers of non trivial roots of unity are 0.
In particular, we see that only the diagonal entries (of the n byn matrix of matrices) are non zero.
The diagonal elements are linear combinations of terms of formFArF∗, and by Lemma 2 these are
diagonal. So the covariance of the DFT is diagonal as desired.
Part (b) follows almost identically to part (a), but by ﬁrst noting that exactly by the proof of Lemma
1,cov(ci,cj) is also a circulant and block circulant matrix.
C.4 Proof of Theorem 1
Proof. This Theorem now follows easily from the previous. Suppose we have a layer l, with chan-
nelsc1,...,c k. And let vec(DFT (ci)) have directions ˜zzz(i)
0 ,··· ˜zzz(i)
n2−1. By the previous theorem, we
know that the covariance of all of these neurons only has non-zero termscov(˜zzz(i)
k , ˜zzz(j)
k .
So arranging the full covariance matrix to have row and column indexes being
˜zzz(1)
0 , ˜zzz(1)
0 ,... ˜zzz(k)
0 , ˜zzz(1)
1 ... ˜zzz(k)
n2 the nonzero terms all live in the n2 k by k blocks down the
diagonal of the matrix, proving the theorem.
C.5 Computational Gains
As the covariance matrix is block diagonal, our more efﬁcient algorithm for computation is as fol-
lows: take the DFT of every channel (n logn due to FFT) and then compute covariances according
to blocks: partition thekn directions into then2k byk matrices that are non-zero, and compute the
covariance, inverses and square roots along these.
A rough computational budget for the covariance is therefore kn logn +n2k2.5, while the naive
computation would be of order (kn2)2.5, a polynomial difference. Furthermore, the DFT method
also makes for easy parallelization as each of then2 blocks does not interact with any of the others.
15

<!-- page 16 -->

0 20000 40000 60000 80000 100000 120000 140000 160000
Train step
0.0
0.2
0.4
0.6
0.8
1.0
1.2
SVCCA of layer with final step
Layer dynamics with SVCCA
in
c1
c2
bn1
p1
c3
c4
c5
bn2
p2
fc1
bn3
fc2
bn4
logits
0 20000 40000 60000 80000 100000 120000 140000 160000
Train step
0.0
0.2
0.4
0.6
0.8
1.0
1.2
SVCCA of layer with final step
Layer dynamics with SVCCA
in
res
bn_cv
bn_cv
res
bn_cv
bn_cv
res
bn_cv
bn_cv
out
Figure App.3: Learning dynamics per layer plots for conv (left pane) and res (right pane) nets trained on
CIFAR-10. Each line plots the SVCCA similarity of each layer with its ﬁnal representation, as a function of
training step, for both the conv (left pane) and res (right pane) nets. Note the bottom up convergence of different
layers
D Per Layer Learning Dynamics Plots from Section 4.1
E Additional Figure from Section 4.4
Figure App.4 compares the converged representations of two different initializations of the same
convolutional network on CIFAR-10.
in c1 c2 bn1 p1 c3 c4 c5 bn2 p2 fc1 bn3 fc2 bn4 logits
out
Initialization 2
out
logits
bn4
fc2
bn3
fc1
p2
bn2
c5
c4
c3
p1
bn1
c2
c1
in
Initialization 1
SVCCA similarity of CIFAR10 conv nets over
different random initializations
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Figure App.4: Comparing the converged representations of two different initializations of the same
convolutional architecture. The results support ﬁndings in [12], where initial and ﬁnal layers are
found to be similar, with middle layers differing in representation similarity.
F Experiment from Section 4.4
G Learning Dynamics for an LSTM
16

<!-- page 17 -->

p2 fc1 bn3 fc2 bn4
Number of top layers consecutively compressed
0.70
0.75
0.80
0.85
0.90
0.95
1.00
1.05
1.10
Accuracy
CIFAR10: Accuracy after compression by projecting
 layers onto top SVCCA directions
baseline
45% (SVCCA two nets)
63% (SVCCA two nets)
22% (SVCCA against logits)
35% (SVCCA against logits)
Figure App.5: Using SVCCA to perform model compression on the fully connected layers in a CIFAR-
10 convnet. The two gray lines indicate the original train (top) and test (bottom) accuracy. The two sets of
representations for SVCCA are obtained through 1) two different initialization and training of convnets on
CIFAR-10 2) the layer activations and the activations of the logits. The latter provides better results, with the
ﬁnal ﬁve layers: pool1, fc1, bn3, fc2 and bn4 all being compressed to0.35 of their original size.
Figure App.6: Learning dynamics of the different layers of a stacked LSTM trained on the Penn Tree
Bank language modeling task. We observe a similar pattern to that of convolutional architectures
trained on image data: lower layer converge faster than upper layers.
17
