# references/173_prac_principal_random_subspace_for_llm_activation_compression_and_memory_efficient_training.pdf

<!-- page 1 -->

PRAC: Principal-Random Subspace for LLM Activation
Compression and Memory-Efficient Training
Yanyi Li1∗ yyl0605@foxmail.com
Yimu Zhang1∗ zym24@stu.pku.edu.cn
Cong Fang1,2 fangcong@pku.edu.cn
1 School of Intelligence Science and Technology, Peking University, China
2 Institute for Artificial Intelligence, Peking University, China
∗ Equal contribution.
Abstract
Activations have become the primary memory bottleneck in large-batch LLM training.
However, existing compression methods fail to exploit the spectral structure of activa-
tions, resulting in slow convergence or limited compression. To address this, we bridge the
relationship between the algorithm’s fast convergence and the requirements for subspace
projection, and show that an effective compression should yield an unbiased estimate of
the original activation with low variance. We proposePrincipal-Random Subspace for
LLMActivationCompression (PRAC), which novelly decomposes activations into two
components: a principal subspace captured via SVD to retain dominant information, and
a random subspace sampled from the orthogonal complement to approximate the tail.
By introducing a precise scaling factor, we prove that PRAC yields an unbiased gradient
estimator with minimum variance under certain conditions. Extensive experiments on pre-
training and fine-tuning tasks demonstrate that PRAC achieves up to 36% total memory
reduction with negligible performance degradation and minimal computational cost.
1. Introduction
Memory footprint has emerged as a critical bottleneck in the training of large language mod-
els (LLMs). During training, the memory overhead primarily consists of three components:
model parameters (weights), optimizer states, and activations memory. The activations rep-
resent the intermediate outputs of each layer computed during the forward pass, which are
typically retained to compute gradients in the subsequent backward propagation. In prac-
tice, a moderately large batch size generally promotes more stable training dynamics and
improves GPU parallelism utilization, thereby accelerating convergence efficiently. How-
ever, a larger batch size substantially increases the activation memory footprint, turning it
into the primary bottleneck and limiting both training efficiency and scalability. As illus-
trated in Figure 1 (Right), training a LLaMA-1B model (batch size 1 of 128 and sequence
length of 256) requires 94.5GB of memory, with model parameters 2.48GB, optimizer states
and weight gradients 7.95GB for popular used ADAM algorithm (Adam et al., 2014), and
activation 84.17GB, where activations account for a staggering 89% of the total usage.
A straightforward approach to reducing activation memory is to modify the implemen-
tation of the training algorithm while preserving its underlying logic. For instance, gra-
dient checkpointing (Chen et al., 2016) selectively re-computes certain activations during
1. In this paper, “batch size” denotes the micro-batch size, representing the maximum number of samples
processed during a single forward and backward pass on an individual GPU.
©2026 Yanyi Li, Yimu Zhang, and Cong Fang.
License: CC-BY 4.0, seehttps://creativecommons.org/licenses/by/4.0/.
arXiv:2602.23111v1  [cs.LG]  26 Feb 2026

<!-- page 2 -->

Li, Zhang, F ang
Forward Pass Backward Pass
Activation Compression
PAC
RAC
PRAC
Baseline
High Variance for RAC
Low Variance
for PRAC
Bias
 PAC
RAC
PRAC
Baseline
High Variance for RAC
Minimum
Variance
Bias
Baseline GaLore RSO PRAC
0
20
40
60
80
100
120
140
160Total Memory (GB) =34.12GB
 ( 36%)
Activations
Non-Activations
Eval Loss
2.60
2.65
2.70
2.75
2.80
Eval Loss
Figure 1: The proposed PRAC projects activations onto both the principal and random
subspaces and yields the minimum variance unbiased estimator, thus achieving up
to 36% total memory reduction with negligible performance degradation. (Left)
The flowchart of PRAC; (Middle) A conceptual comparison of subspace strategies;
(Right) Performance comparison on LLaMA-1B.
the back-propagation phase, rather than storing them throughout the forward pass. This
technique introduces additional computational overhead in the backward pass, thereby in-
creasing overall training time.
The other line of research focuses on reducing activation memory through novel algo-
rithmic design. The core idea of such approaches is to iteratively optimize within a special
chosen subspace of the model parameters.
One representative example is zeroth-order optimization, which has attracted consider-
able attention in recent years, leading to multiple proposed variants (Malladi et al., 2023;
Gautam et al., 2024; Chen et al., 2024b; Zhang et al., 2024a; Zhao et al., 2024b; Shu et al.,
2025; Petrov et al., 2025). From an algorithmic perspective, the algorithm per-step sam-
ples a random directionξ, typically drawn from a standard normal or uniform spherical
distribution, and constructs a gradient estimator of the form 1
η[f(w+ηξ)−f(w)]ξ, where
wdenotes the model parameters andηis a small positive constant. Intuitively, asη→0,
this estimator approximates [ξ ⊤∇f(W)]ξ, which corresponds to performing an update only
along the normalized directionξ/∥ξ∥, scaled by the directional derivative of the objective
function. In implementation, such methods only require access to the final output of the
forward pass, allowing intermediate activations to be released immediately. This substan-
tially reduces the peak memory footprint during training. However, since these approaches
typically update only a one-dimensional subspace per iteration, they converge much more
slowly than first-order methods in practice and are generally not suitable for challenging
tasks such as large-scale pre-training.
To improve efficiency, some advanced methods incorporate architectural information to
enable multi-dimensional subspace updates. For example, RSO (Random Subspace Opti-
mization) introduced by Chen et al. (2025) in the outerloop uniformly samples subspaces
and in the inner loop optimizes weights in the subspaces by inserting a low-dimensional
trainable module. Activation memory is reduced because only the projected activations
need to be stored—a principle inspired by LoRA (Hu et al., 2022) and ReLoRA (Lialin
et al., 2023). Despite these memory savings, this method still results in non-negligible
performance degradation during pre-training, limiting its practical applicability.
2

<!-- page 3 -->

Principal-Random Subspace for LLM Activation Compression
We emphasize that a central limitation of existing activation-efficient methods is their
inability to leverage the spectral structure of activations, which ultimately compromises
training effectiveness. In this paper, we directly focus on how to compress activations during
training with minimal impact on convergence rate. Following Zhao et al. (2024a); He et al.
(2024); Malladi et al. (2023); Gautam et al. (2024), we still adopt a linear compression
scheme: given a projection matrixP, the activation matrixXis compressed asXPand
reconstructed asXP P ⊤.
Our starting point is a standard stochastic optimization formulation, through which
we bridge the relationship between the algorithm’s fast convergence and the requirements
for subspace projection. We demonstrate that a well-chosen projection should yield an
unbiased estimate of the original activation with low variance that ensures provable and
efficient training convergence.
To design an efficient estimator, we leverage the spectral structure of activations. We
formalize widely observed “low-rank” phenomenon (Hu et al., 2022; Zhao et al., 2024a; Chen
et al., 2024a; Zhang et al., 2025) as theActivation Degenerate Condition, which assumes
that the singular values of activations consist of a few dominant ones, followed by a long
tail of smaller values whose cumulative energy (squared sum) is bounded.
The primary contribution of this paper is the introduction of an activation compression
technique, termed PRAC (Principal-Random Subspace for LLMActivationCompression).
We show this method is optimal in the sense that it achievesminimumvariance among
all unbiased estimators under the Activation Degenerate Condition. PRAC consists of two
simple components: a principal subspace projection, which obtains a subspace projection
matrixQ 1 via SVD, and a random subspace projection, whereQ 2 is sampled uniformly
from the orthogonal complement ofQ 1. Additionally, akeyscaling factorkis introduced
to ensure the unbiasedness of the estimator.
˜X= (XP)P ⊤ = (XQ1)Q⊤
1 +k(XQ 2)Q⊤
2 .
In implemtation for memory-efficient LLM training, PRAC employs a dynamic subspace
update schedule that refreshes projection matrices at fixed intervals, thereby rendering the
cost of SVD and QR decompositions negligible. The method further maximizes memory
efficiency through subspace sharing and a layer-wise policy (see details in Section 5.1). We
analyze the memory footprint and computational overhead of PRAC, demonstrating its
efficiency in both aspects.
We conduct extensive experiments on both pre-training and fine-tuning tasks. Our
experiments demonstrate that PRAC matches or surpasses baselines across various tasks,
achieving up to 36% memory saving while maintaining competitive performance.
Our key contributions are summarized as follows:
•We propose PRAC, a novel activation compression method that effectively integrates
principal and random subspace projections. PRAC is thefirstmethod to leverage the
structural information of activations for memory-efficient pre-training. Theoretically,
we prove that PRAC yields an unbiased estimator withminimumvariance under the
Activation Degeneration Condition.
3

<!-- page 4 -->

Li, Zhang, F ang
•We demonstrate the practical efficacy of PRAC through extensive experiments on both
pre-training and fine-tuning tasks. PRAC consistently achieves substantial memory
reduction with negligible performance loss and very small computational overhead.
2. Related Works
Activation-Efficient Methods.To alleviate activation memory overhead, a technique
modifying the checkpointing mechanism during backpropagation was proposed in Chen et al.
(2016). This approach has not only achieved widespread adoption but has also catalyzed
subsequent research into modifications of the backpropagation algorithm itself (Yang et al.,
2024; Luo et al., 2025).
Furthermore, alternative approaches employ zeroth-order optimization to bypass back-
propagation entirely (Malladi et al., 2023; Gautam et al., 2024; Chen et al., 2024b; Miles
et al., 2024; Chen et al., 2025; Shamshoum et al., 2025; Petrov et al., 2025), while others
retain first-order methods but address the optimization problem within a low-dimensional
subspace (Chen et al., 2025).
Compression-based approaches (Yang et al., 2024; Shamshoum et al., 2025; Miles et al.,
2024) offer an alternative strategy by approximating intermediate variables. However, these
methods are generally designed for specific sub-components of the model architecture. Fur-
thermore, they fail to explicitly address how compression artifacts influence the convergence
rate, limiting their potential in general-purpose compression for memory-efficient training.
Optimization States-Efficient Methods.Considerable research aim to improve the
memory efficiency for optimizer states. GaLore (Gradient Low-Rank Projection) and its
variants (Zhao et al., 2024a; Liang et al., 2024; He et al., 2024) perform low-rank compression
on gradients. To address the compression of momentum terms in the Adam optimizer,
Adafactor (Shazeer and Stern, 2018) utilizes factorization techniques to approximate the
storage of the second moment. Adam-mini (Zhang et al., 2024b) designs a block-wise
learning rate strategy based on the heterogeneity of the neural network’s Hessian matrix,
reducing the memory footprint of the second moment. Fira (Chen et al., 2024a) and Apollo
(Zhu et al., 2025) utilize optimizer states to update adaptive learning rates. However, these
algorithms do not modify the backpropagation process and thus cannot reduce the memory
usage of activations, which is the dominant factor. Our method is orthogonal to this kind
of method and can be combined with them for additional memory savings.
3. Starting Point of Activation Compression
3.1 Problem Formulation
The training of Large Language Models (LLMs) can be formulated as the following stochas-
tic optimization problem:
min
W
f(W) :=E ζ[F(W;ζ)],(1)
whereWdenotes the model parameters andζrepresents the random data batch. We
simply study the Stochastic Gradient Descent (SGD) (Bottou, 2010) method, noting that
4

<!-- page 5 -->

Principal-Random Subspace for LLM Activation Compression
the analysis extends naturally to other optimizers. The update of SGD goes as:
Wt+1 =W t −η t ∇F(W t;ζ t)| {z }
update term
.(2)
To mitigate memory bottlenecks, various methods compress training artifacts (parameters,
gradients, or activations). Consequently, the exact gradient is replaced by an approximate
update term, denoted as ˜G(Wt;ζ t), leading to the modified update rule:
Wt+1 =W t −η t ˜G(Wt;ζ t).(3)
A fundamental question arises:What properties should the compression satisfy
to guarantee fast convergence?
To address this, we first examine the impact of systematic bias in the gradient estimator.
Proposition 1(Non-convergence under Constant Bias).Consider two-dimensional strongly
convex problem:f(w (1), w(2)) =E ξ′(w(1) −3ξ ′)2 + (w(2))2 withξ ′ following a Bernoulli dis-
tribution. The parameters are initialized atw (1) = 0, w(2) = 1. At each step, only the first
coordinate is updated by a stochastic gradient: ˜G= [2(w (1) −3ξ ′),0] ⊤.Then for any{η t}
satisfies the Robbins-Monro condition: P∞
t=0 ηt =∞, P∞
t=0 η2
t <∞, one hasw (2)
t = 1for
allt. Consequently,(w (1)
t , w(2)
t )will never converge to an approximate stationary point.
In the simple example above, the gradient estimator is biased by Θ(1) becausew (2) is
never updated. It is interesting to observe that when|w (1)|<2, the magnitude of the
stochastic gradient forw (1) (i.e., 2|w (1) −3ξ ′|) is always larger than that forw (2) (i.e.,
2|w(2)|). Consequently, even ifw (1) converges to its minimizer 0, the maximum selection
rule that consistently picks the coordinate with the largest gradient magnitude will never
updatew (2). It implies that principal methods, such as Galore (Zhao et al., 2024a), may
fail to converge in general stochastic optimization settings. Proposition 1 also establishes
that a constant biased estimator cannot ensure convergence. We then consider unbiased
estimator with non-negligible variance.
Proposition 2(Convergence Rate with Bounded Gradient Variance, Ghadimi and Lan
(2013)).Let the objective functionfis L-smooth and bounded below, and denote∆ =
f(W 1)−inff. Assume that at each iterationtwe have access to an unbiased stochastic
gradient ˜G(Wt, ζt)satisfyingE[ ˜G(Wt, ζt)|W t] =∇f(W t),E[∥ ˜G(Wt, ζt)− ∇f(W t)∥2 |W t]≤
σ2.Consider the update rule in(3)with constant step sizesη t = min

1
L ,
√
2∆/L
σ
√
N

and then
randomly selects an outputW R from{W 1,· · ·, W T }according to the certain probability mass
function. Then the following bound holds:
E[∥∇f(W R)∥2]≤ 2L∆
T + 2
r
2L∆
T σ.
Proposition 2 is a standard result in non-convex optimization (Nesterov, 2013; Ghadimi
and Lan, 2013). It demonstrates that unbiased compression ensures convergence. Moreover,
the convergence rate is hindered by the varianceσ 2. Therefore, an effective compression
should yield a gradient estimator that is bothunbiasedandlow-variance.
Since activations dominate memory usage in LLM training, we translate the gradient-
level requirements established above into concrete criteria for activation compression.
5

<!-- page 6 -->

Li, Zhang, F ang
3.2 Criteria for Activation Compression
The theoretical constraints established in Section 3.1 for general gradients directly inform
the design of activation compression. For a linear layer, such as the Query layer in the
attention block, the quality of the activation reconstruction dictates the quality of the
gradient estimate.
Lemma 3.Consider a linear layer with forward passY=XW. Let ˜Xbe a compressed
estimator of the activationX. If ˜Xis unbiased (E[ ˜X] =X) with bounded varianceE[∥ ˜X−
X∥2
F ]≤σ 2, then the resulting gradient estimator ˜G= ˜X ⊤(∇Y L)satisfies:
1.Unbiasedness:The gradient estimator is unbiased with respect to the true gradient:
E[ ˜G] =E[ ˜X ⊤](∇Y L) =∇ W L.
2.Bounded Variance:The gradient variance is bounded by the activation variance
and the upstream gradient norm:
E[∥ ˜G− ∇ W L∥2
F ]≤σ 2∥∇Y L∥2
2.
Lemma 3 shows that constructing an unbiased and low-variance approximation of acti-
vations suffices to satisfy the convergence guarantees in Propositions 1 and 2.
Following Zhao et al. (2024a); He et al. (2024); Malladi et al. (2023); Gautam et al.
(2024), we adopt a subspace projection approach. LetP∈R n×r be a projection matrix,
and the activationXis compressed to a lower-dimensional representationXPand recon-
structed as ˜X= (XP)P ⊤. The central challenge, therefore, lies in designing the projection
matrixPsuch that the reconstruction ˜Xminimizes variance while maintaining unbiased-
ness, specifically tailored to the spectral structure of the activations.
4. Proposed Method: PRAC
In this section, we introduce PRAC, a hybrid framework motivated by the spectral structure
of activations. By integrating the principal and random subspaces to address their respective
limitations, we balance the bias-variance trade-off. Finally, we prove that PRAC yields the
minimum variance unbiased estimator under certain conditions.
4.1 Spectral Analysis of Activations
/uni00000013/uni00000015/uni00000013/uni00000013/uni00000017/uni00000013/uni00000013/uni00000019/uni00000013/uni00000013/uni0000001b/uni00000013/uni00000013
/uni00000036/uni0000004c/uni00000051/uni0000004a/uni00000058/uni0000004f/uni00000044/uni00000055/uni00000003/uni00000039 /uni00000044/uni0000004f/uni00000058/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000015/uni00000013/uni00000013
/uni00000017/uni00000013/uni00000013
/uni00000019/uni00000013/uni00000013
/uni0000001b/uni00000013/uni00000013
/uni00000014/uni00000013/uni00000013/uni00000013
/uni00000014/uni00000015/uni00000013/uni00000013
/uni00000014/uni00000017/uni00000013/uni00000013
/uni00000014/uni00000019/uni00000013/uni00000013/uni00000036/uni0000004c/uni00000051/uni0000004a/uni00000058/uni0000004f/uni00000044/uni00000055/uni00000003/uni00000039 /uni00000044/uni0000004f/uni00000058/uni00000048
/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni00000032/uni00000058/uni00000057/uni00000053/uni00000058/uni00000057/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni00000038/uni00000053/uni00000003/uni00000033/uni00000055/uni00000052/uni0000004d/uni00000048/uni00000046/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni00000013/uni00000015/uni00000013/uni00000013/uni00000017/uni00000013/uni00000013/uni00000019/uni00000013/uni00000013/uni0000001b/uni00000013/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000036/uni0000004c/uni00000051/uni0000004a/uni00000058/uni0000004f/uni00000044/uni00000055/uni00000003/uni00000039 /uni00000044/uni0000004f/uni00000058/uni00000048/uni00000056
/uni00000015/uni00000013
/uni00000017/uni00000013
/uni00000019/uni00000013
/uni0000001b/uni00000013
/uni00000014/uni00000013/uni00000013/uni00000026/uni00000058/uni00000050/uni00000058/uni0000004f/uni00000044/uni00000057/uni0000004c/uni00000059/uni00000048/uni00000003/uni00000028/uni00000051/uni00000048/uni00000055/uni0000004a/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni00000032/uni00000058/uni00000057/uni00000053/uni00000058/uni00000057/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni00000038/uni00000053/uni00000003/uni00000033/uni00000055/uni00000052/uni0000004d/uni00000048/uni00000046/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057
/uni0000001b/uni00000013/uni00000008/uni00000003/uni00000028/uni00000051/uni00000048/uni00000055/uni0000004a/uni0000005c
Figure 2: Singular value spectrum (Left) and
cumulative energy ratio (Right).
To design an effective projection matrixP
in Section 3.2, we analyze the spectral struc-
ture of activations using the 10th layer of
LLaMA-130M (at 10% training progress) as
an example.
As shown in Figure 2, the spectrum ex-
hibits two distinct characteristics: a few
dominant singular values capture the ma-
jority of the spectral energy; and the re-
maining singular values form a long tail that
decays slowly.
6

<!-- page 7 -->

Principal-Random Subspace for LLM Activation Compression
Note the “low-rank” structure is commonly observed in weight matrices (Hu et al., 2022;
Lialin et al., 2023). We study activation matrix and emphasize that the long-tail directions
cannot be ignored—truncating them significantly slows down convergence (Chen et al.,
2024a; Zhang et al., 2025). We introduce the termdegenerate conditionto characterize this
observed phenomenon.
Assumption 4(Activation Degenerate Condition).For activation matrixX, letσ i be
thei-th largest singular value of the activations. We sayXsatifies the(s, q)-degenerate
condition (X∼(s, q)−Din short) if it admits: Pn
s+1 σ2
i ≤q.
This assumption posits that only the topssingular values of the activation matrix are
large. Since the remaining singular values are typically small and roughly equal in practice,
we bound the rest squared sum (same as the square of the Frobenius norm for the rest
matrix) by a constantq.
Based on Assumption 4, we can conduct a theoretical analysis of the optimal design
of the projection matrixP(see Section 4.3). Before that, let us first introduce two key
components of PRAC, which are also subspace projection methods.
4.2 Key Components of PRAC
Intuitively, one might consider two natural approaches to activation compression: projecting
onto the principal subspace of the activations, or onto a random subspace. However, both
have inherent drawbacks that our method addresses.
Principal Subspace for Activation Compression (PAC).PAC retains the significant
information by projecting activations onto their principal subspaces. Let the Singular Value
Decomposition (SVD) of the activation matrix beX=UΣV ⊤ =Pn
i=1 siuiv⊤
i . We define
the projection matrixQ 1 =
h
v(1)
1 ,· · ·, v (1)
r1
i
∈R n×r1 using the top-r1 right singular vectors.
ThenXQ 1 captures the rank-r1 principal information. For the rankr 1 < n, PAC introduces
a systematic reconstruction bias ∆ =∥X−XQ 1Q⊤
1 ∥2
F =Pn
i=r1+1 σ2
i >0. As discussed in
Section 3.2, this bias can lead to non-convergence in the optimization problem (1).
Random Subspace for Activation Compression (RAC).To avoid bias, RAC projects
activations onto a random subspace sampled uniformly from the Stiefel manifold St n,r2,
defined as St n,r =

P∈R n×r|P ⊤P=I r

. The construction proceeds as follows: First, we
generate a random matrixS∈R n×r2 with entriesS ij ∼ N(0,1). We then perform QR
decomposition onSto obtain an orthogonal basisQ 2 ∈R n×r2, such thatQ ⊤
2 Q2 =I.
While RAC can provide an unbiased estimator of the activation (via setting a scal-
ing factor), the inherent randomness introduces high variance:E∥ ˜Xrac −X∥ 2
F ≤( n
r2
−
1)
 Ps
i=1 σ2
i +q

due to the top singular values. (see Theorem 14 and 16 in Appendix C.2).
This high variance destabilizes the training process and slows convergence.
4.3 Optimal Design via Hybrid Projection
We now establish the lower bound variance to estimate the activations. Specifically, for any
activationX, we consider the random subspace projection method where the projection
matrixP∈R n×r is drawn from a distributionP X, which must satisfy the unbiasedness
condition (i.e.E P∼P X[XP P ⊤] =X) and depend onX, then the lower bound variance can
7

<!-- page 8 -->

Li, Zhang, F ang
be written as
sup
X∼(s,q)−D

 inf
P∼P X ,
EPX [XP P ⊤]=X
EPX ∥XP P ⊤ −X∥ 2
F

 .(4)
For this min-max optimization problem, if the small singular values ofXexhibit non-
uniformity,Pcan be tailored to exploit this structure. Intuitively, a distribution where
Xpossesses approximately uniform tail singular values represents a relatively worst-case
scenario. The following lemma gives the lower bound explicity.
Lemma 5(Lower Bound).Under Assumption 4, whens < r < n, we have(4)≥( n−s
r−s −1)q.
We now propose our method and demonstrate that its variance achieves the lower bound
in Lemma 5, thereby demonstraining its optimality in attaining the minimum variance.
Construction of the Projection Matrix.We first extract the principal basis vectors
v(1)
i (i∈[1, r 1]) via SVD, and let the projection matrixQ 1 =
h
v(1)
1 ,· · ·, v (1)
r1
i
. Then we
sample a random matrix whose columns lie in the orthogonal complement of the column
space ofQ 1:
So = (I−Q 1Q⊤
1 )S,whereS ij
i.i.d∼ N(0,1).(5)
Letv (2)
j (j∈[1, r 2]) be the column vectors obtained from QR(S o). By construction, the
principal and random subspaces are orthogonal (v (1)
i ⊥v (2)
j ). The unified projection matrix
P∈R n×(r1+r2) is defined as:
P=

v(1)
1 ,· · ·, v (1)
r1| {z }
Principal
,
√
kv(2)
1 ,· · ·,
√
kv(2)
r2| {z }
Random

 ,(6)
wherekis a scaling factor used to ensure unbiased reconstruction. It’s critical, since
we only sample a subspace of dimensionr 2 to represent the entire tail of dimensionn−r 1,
the energy of the random component must be amplified to ensure the estimator remains
unbiased in expectation.
Unbiased Estimation.By selecting an appropriatek, PRAC ensures the unbiasedness.
Theorem 6(Unbiased Reconstruction of PRAC).LetPbe the projection matrix defined
in(6). If the scaling factor is set tok= n−r1
r2
, the reconstruction ˜X=XP P ⊤ satisfies
E[ ˜X] =X.
Variance Reduction Analysis.A key advantage of PRAC is its ability to achieve the
minimum variance.
Theorem 7(Variance Bound of PRAC).ForXadmiting(s, q)-D condition withs < r < n,
by choosingr 1 =sandr 2 =r−r 1, we have:
E

∥XP P ⊤ −X∥ 2
F

≤( n−r 1
r2
−1)q.
8

<!-- page 9 -->

Principal-Random Subspace for LLM Activation Compression
Table 1: Comparison of Estimation Unbiasedness and Variance across Projection Methods.
Algorithm Form Scaling Unbiasedness Variance
PAC ˜Xpac = (XQ 1)Q⊤
1 -×-
RAC ˜Xrac = (XQ 2)Q⊤
2 n/r2 ✓High
PRAC ˜Xprac = (XP)P ⊤ (n−r 1)/r2 ✓Minimum
Theorem 6 confirms that settingk= n−r1
r2
yields an unbiased reconstruction of activa-
tions and, by extension, gradients (Lemma 3). Theorem 7 demonstrates its optimality in
the sense that it achieves the minimum variance among all unbiased estimators under the
degenerate condition. Ablation study in Section 6.3 empirically validates this theoretical
finding. A comparison of the three projection methods is presented in Table 1.
5. PRAC for Memory-Efficient Training
In this section, we detail the practical implementation of PRAC for memory-efficient LLM
training. We introduce dynamic and sharing strategy to minimize computational overhead.
Furthermore, we present a layer-wise adaptation scheme tailored to the distinct sensitivity
of different model components. Finally, we provide a complexity analysis of PRAC, focusing
on activation memory reduction and the amortized computational cost.
5.1 PRAC for LLM Training
Dynamic Subspace Update.Recall that the PRAC reconstruction can be written as:
XP P ⊤ =X
" r1X
i=1
v(1)
i (v(1)
i )⊤ +k
r2X
i=1
v(2)
i (v(2)
i )⊤
#
=XQ 1Q⊤
1 +kXQ 2Q⊤
2 , (7)
whereQ 1 = [v (1)
1 ,· · ·, v (1)
r1 ], Q2 = [v (2)
1 ,· · ·, v (2)
r2 ] are the principal and random projection
matrices respectively.
Computing the exact principal subspace (via SVD) and generating orthogonal random
projections (via QR) at every step incurs significant computational cost. To mitigate this,
we employ a lazy update strategy. The principal and random componentsQ 1, Q2 are
updated only at fixed intervalsT 1 andT 2, respectively. The rationale is that the activation
statistics (and thus the related subspace) change slowly over training steps, especially after
the initial warm-up phase. This lazy update mechanism drastically reduces the amortized
computational overhead. The full procedure is outlined in Algorithm 1.
Subspace Sharing.In Transformer architectures, many layers share the same input acti-
vationXsuch as the Query, Key and Value projections. To cut memory usage, we enforce
subspace sharing: a single set of projection matrices (Q 1, Q2) and compressed activations
(XQ1, XQ2) is computed and shared across these layers. Beyond memory savings, this
strategy ensures that gradient updates for parallel heads reside in a consistent subspace,
reducing gradient noise variance and stabilizing optimization.
9

<!-- page 10 -->

Li, Zhang, F ang
Algorithm 1Dynamic Activation Compression via PRAC
Require:Input activationX, ranks (r 1, r2), update intervals (T 1, T2), current stept
Ensure:Projection matricesQ (t)
1 ,Q (t)
2 , Compressed ActivationsX (t)
1 ,X (t)
2
1:k←(n−r 1)/r2
2:// Update Principal Subspace:
3:iftmodT 1 = 0then
4:Q (t)
1 ←top-r 1 right singular vectors ofX
5:else
6:Q (t)
1 ←Q (t−1)
1
7:end if
8:// Update Random Subspace
9:iftmodT 2 = 0then
10:SampleS∼ N(0, I n×r2)
11:Q (t)
2 ←orthonormal basis of (I−Q (t)
1 Q(t)⊤
1 )S
12:else
13:Q (t)
2 ←Q (t−1)
2
14:end if
15:// Compression
16:X (t)
1 ←XQ (t)
1 ,X (t)
2 ←k·XQ (t)
2
17:Return:Q (t)
1 , Q(t)
2 , X(t)
1 , X(t)
2
Layer-Wise Design.Building on the periodic update strategy, we further tailor the
compression to the specific roles of different layer types, as not all layers contribute equally
to training dynamics. To this end, we adopt a layer-wise configuration tailored to the
specific characteristics of linear and non-linear modules:
•Linear Layers (MLP, Attention Projections):Due to their large parameter
scale, the updates of their weight matrices impose a higher demand on gradient quality.
Therefore, more activation information needs to be retained to ensure optimization
performance. During compression, we allocate a relatively higher rank to these layers
(e.g.,r 1 =r 2 =⌊0.3n⌋).
•Non-linear Layers (LayerNorm, RMSNorm, GeLU, SiLU):These layers typ-
ically involve element-wise operations or vector-wise scaling parameters, resulting in
lower information density. Although our theoretical analysis does not directly cover
all cases, we find that our method remains effective for most layers. It is worth noting
that certain intermediate variables in these layers, such as the mean and variance in
LayerNorm, are not compressed due to their negligible memory footprint (e.g.,bsvs.
bsn). Likewise, operations like Flash Attention are excluded from PRAC compression
because of their intricate coupled structure and engineering optimizations. In practice,
we process all the selected non-linear layers using a lower rank (r 1 =r 2 =⌊0.2n⌋).
5.2 Memory and Computational Efficiency
10

<!-- page 11 -->

Principal-Random Subspace for LLM Activation Compression
Table 2: Comparison with memory-efficient algorithms on pretraining tasks. Validation
perplexity (PPL, lower is better) and peak memory consumption (MEM, in GB,
lower is better) are reported. For methods marked out-of-memory (OOM), per-
plexity is measured using half the batch size specified above.B, S,∆ denote the
micro-batch size, sequence length and total memory reduction respectively.
LLaMA-130M LLaMA-350M LLaMA-1B GPT-2-124M GPT-2-355M
Method PPL MEM ∆ PPL MEM ∆ PPL MEM ∆ PPL MEM ∆ PPL MEM ∆
Baseline 24.41 21.26 - 18.77 46.10 - 15.33 OOM 19.83 62.27 - 16.26 55.94 -
GaLore 25.12 21.08 -1% 19.65 45.37 -2% 15.64 OOM 21.01 62.15 -0% 17.88 55.64 -1%
RSO 25.41 19.57 -8% 19.57 40.36 -13% 15.72 79.32 -16% 22.31 54.79 -12% 18.29 46.90 -16%
PRAC24.6615.65-27%18.9232.21-30%15.4160.48-36%19.9549.68-20%16.3544.21-21%
B / S 128/256 128/256 128/256 64/1024 32/1024
Table 3: Activation memory footprint com-
parison for the GELU and linear
layer.
Baseline PRAC
GeLU Inputbsn bs(r 1 +r 2)
Wdown Inputbsn bs(r 1 +r 2)
Total 2bsn2bs(r 1 +r 2)
Theoretical Memory Footprint.We
analyze memory usage in the GeLU layer
Au = GeLU(A ′
u) and linear layerA ′
d =
AuW ⊤
down. Standard backpropagation re-
quires storing full tensorsA u, A′
u ∈R b·s×n,
costing 2bsn. In contrast, PRAC projects
these activations onto principal (r 1) and
random (r 2) subspaces, caching only the
low-dimensional projections inR b·s×(r1+r2).
In particular, sinceP u, P ′
u are independent of the batch size, their memory overhead is
negligible. The memory usage is summarized in Table 3.
In the experiments,r 1 +r 2 is typically set to less than⌊0.6n⌋, which results in over
40% reduction in the activations. The complete analysis for the full GPT architecture is
provided in Appendix B.
Computational Overhead.The primary computational cost of PRAC arises from SVD
and QR decompositions. However, by setting the update intervalsT 1 andT 2 to sufficiently
large values (e.g.,T 1 =T 2 = 500 steps), the amortized cost becomes negligible. Further-
more, the additional matrix multiplications introduced by the projection operationXP
incur minimal overhead compared to the memory throughput gains. The experimental
results of training efficiency are shown in Section 6.1.
6. Experiments
In this section, we extensively evaluate PRAC on both pre-training and fine-tuning tasks
across various model architectures. PRAC consistently achieves up to 36% memory reduc-
tion while maintaining competitive performance.
11

<!-- page 12 -->

Li, Zhang, F ang
6.1 Memory-Efficient Pre-training
Setup.We evaluate PRAC by pre-training LLaMA (Touvron et al., 2023) models (130M,
350M, 1B parameters) on the C4 dataset (Raffel et al., 2020) and GPT-2 (Brown et al., 2020)
models (124M, 355M) on OpenWebText (Gokaslan et al., 2019), following the standard
configurations in the nanoGPT codebase 2. For PRAC, we configure the subspace ranks
asr 1 =r 2 =⌊0.3n⌋for linear layers andr 1 =r 2 =⌊0.2n⌋for non-linear layers (e.g.,
RMSNorm, SiLU), wherenis the hidden dimension. We compare against two memory-
efficient baselines: GaLore (Zhao et al., 2024a) and RSO (Chen et al., 2025), using their
reported hyperparameters. Detailed configurations are provided in Appendix D.1. To
ensure statistical robustness, all experiments are conducted over multiple runs with different
random seeds, and we report the averaged results.
7500 10000 12500 15000 17500 20000
Iteration
3.20
3.25
3.30
3.35
3.40
3.45Validation Loss
Baseline
PRAC
GaLore
RSO
30000 35000 40000 45000 50000 55000 60000
Iteration
2.95
3.00
3.05
3.10Validation Loss
Baseline
PRAC
GaLore
RSO
(a) LLaMA 130M (b) LLaMA 350M
Figure 3: Loss curves of pre-training LLaMA-
130M and LLaMA-350M model.
Results.Table 2 summarizes the quanti-
tative results. PRAC achieves the optimal
trade-off between training performance and
memory usage, achieving a 36% total mem-
ory reduction for the LLaMA-1B model.
Figure 3 illustrates the validation loss tra-
jectories. PRAC maintains competitive
convergence throughout the entire training
process. In contrast, GaLore (Principal-
only) suffers from stagnation in the later
stages due to bias, while RSO (Random-only) exhibits slower initial convergence due to
high variance. In the GPT-2 experiments, we restrict the batch size to avoid OOM failures
in the comparison methods; under these conditions, our method achieves 20×and 21×total
memory compression for GPT-124M and 355M respectively.
Table 4: Training Time Comparison on
4×A800 GPU for LLaMA-1B.
Method Batch SizeMemory (GB)Training Time (h)
Baseline 64 50.35 156
PRAC 64 36.90 179
96 (↑50%) 48.70 117 (↓25%)
Training Efficiency.To verify the prac-
tical speedup of PRAC, we measure train-
ing time and peak memory on LLaMA-1B
using 4×A800 GPUs. As shown in Table
4, PRAC reduces peak memory by approx-
imately 27% at a fixed batch size of 64 (to
prevent OOM for the baseline). Crucially,
this memory headroom allows for a larger
batch size (increasing from 64 to 96), which
reduces the total training time by 25%.
Scaling Laws.We validate the scalability of PRAC by training a LLaMA series ranging
from 35M to 1B parameters. Following Chinchilla’s law (Hoffmann et al., 2022), we set
the token budget to 20×the parameter count. Figure 4(a) shows that PRAC tracks the
AdamW baseline loss curves almost identically across all scales while using around 30% less
memory. The linear fit in Figure 4(b) confirms that PRAC adheres to the power-law scaling
relationship, suggesting robust performance for even larger models.
2.https://github.com/karpathy/nanoGPT
12

<!-- page 13 -->

Principal-Random Subspace for LLM Activation Compression
1016
1017
1018
1019
1020
Flops (log)
3 × 100
4 × 100
Validation Loss (log)
Baseline-1e-3-35M
Baseline-1e-3-60M
Baseline-1e-3-130M
Baseline-1e-3-350M
Baseline-4e-4-1B
PRAC-1e-3-35M
PRAC-1e-3-60M
PRAC-1e-3-130M
PRAC-1e-3-350M
PRAC-4e-4-1B
107
108
109
Parameters (log)
2.5×100
3.0×100
3.5×100
4.0×100
4.5×100
Final Validation loss (log)
Baseline fit
Baseline
PRAC fit
PRAC
(a) Scaling laws in terms of compute (b) Scaling laws in terms of parameters
Figure 4: Scaling behavior of PRAC across model sizes (35M to 1B parameters)
Table 5: Comparison with memory-efficient algorithms on fine-tuning RoBERTa models.
Average scores across the GLUE benchmark and the peak memory usage of acti-
vations in MRPC task are provided.
Method Memory (GB) COLA STSB MRPC RTE SST2 MNLI QNLI QQP AVG
Full Fine-Tuning 0.42 62.24 90.92 91.30 79.42 94.57 87.18 92.33 92.28 86.28
RSO (rank=4) 0.29 62.47 90.62 92.25 78.7094.84 86.6792.29 90.94 86.10
PRAC (rank=4) 0.26 (↓38%) 63.56 90.83 93.28 78.7194.72 86.6092.35 90.95 86.38
6.2 Memory-Efficient Fine-tuning
We evaluate PRAC on the GLUE benchmark (Wang et al., 2018) by fine-tuning RoBERTa
(Liu et al., 2019). For a fair comparison with RSO (rankr= 4), we setr 1 =r 2 = 2 for
PRAC. As shown in Table 5, PRAC not only outperforms the RSO baseline but, in certain
tasks, surpasses full fine-tuning. This suggests that the random subspace component may
act as a beneficial regularizer during fine-tuning by introducing stochasticity.
6.3 Ablation Study
4000 5000 6000 7000 8000 9000 10000
Iteration
3.50
3.55
3.60
3.65Validation Loss
PRAC (r=0.3)
PAC (r=0.3)
RAC (r=0.6)
12000 14000 16000 18000 20000
Iteration
3.18
3.20
3.22
3.24
3.26
3.28
3.30Validation Loss
PRAC (r=0.3)
PAC (r=0.3)
RAC (r=0.6)
(a) LLaMA 60M (b) LLaMA 130M
Figure 5: Loss curve of using PRAC, PAC,
RAC in LLaMA series pre-training.
RAC reports the result ofr= 0.6
due to the divergence ofr= 0.3.
We assess the efficacy of the hybrid projec-
tion by comparing PAC, RAC, and PRAC
with equivalent total rank (usingr 1 +r 2
for PRAC). As shown in Figure 5, PAC
converges slowly in the later stages, where
gradient noise dominates the true gradient
(Proposition 1). In contrast, RAC strug-
gles in the early stages due to its inabil-
ity to capture principal subspace properties.
PRAC overcomes these limitations, achiev-
ing faster convergence throughout the entire
training process and validating the advan-
tage of combining dual subspaces.
Appendix D.3 details the ablation analysis for the scaling factor. Results indicate that
the theoretical settingk= n−r1
r2
is effective across both linear and non-linear layers.
13

<!-- page 14 -->

Li, Zhang, F ang
7. Conclusion
In this work, we propose PRAC, a theoretically optimal activation compression strategy
that provides unbiased estimates and leverages the low-rank structure of LLM activations
to minimize estimation variance. Unlike prior subspace methods, PRAC ensures stable
convergence while reducing total memory by up to 36% across both pre-training and fine-
tuning tasks. With negligible computational overhead, PRAC offers a robust and scalable
solution for training large language models on memory-constrained hardware.
Impact Statement
This paper presents a method to substantially improve the memory and computational
efficiency of LLM training. By compressing activations via principal and random subspaces,
our approach reduces the hardware requirements for pre-training and fine-tuning, thereby
minimizing the energy consumption and carbon emissions associated with large-scale deep
learning workloads.
References
Kingma DP Ba J Adam et al. A method for stochastic optimization.arXiv preprint
arXiv:1412.6980, 1412(6), 2014.
L´ eon Bottou. Large-scale machine learning with stochastic gradient descent. In Yves
Lechevallier and Gilbert Saporta, editors,Proceedings of COMPSTAT’2010, pages 177–
186, Heidelberg, 2010. Physica-Verlag HD. ISBN 978-3-7908-2604-3.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini
Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya
Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric
Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner,
Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models
are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin,
editors,Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.
Curran Associates, Inc., 2020.
Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with
sublinear memory cost.CoRR, abs/1604.06174, 2016.
Xi Chen, Kaituo Feng, Changsheng Li, Xunhao Lai, Xiangyu Yue, Ye Yuan, and Guoren
Wang. Fira: Can we achieve full-rank training of llms under low-rank constraint?arXiv
preprint arXiv:2410.01623, 2024a.
Yiming Chen, Yuan Zhang, Liyuan Cao, Kun Yuan, and Zaiwen Wen. Enhancing
zeroth-order fine-tuning for language models with low-rank structures.arXiv preprint
arXiv:2410.07698, 2024b.
14

<!-- page 15 -->

Principal-Random Subspace for LLM Activation Compression
Yiming Chen, Yuan Zhang, Yin Liu, Kun Yuan, and Zaiwen Wen. A memory efficient
randomized subspace optimization method for training large language models.arXiv
preprint arXiv:2502.07222, 2025.
Tanmay Gautam, Youngsuk Park, Hao Zhou, Parameswaran Raman, and Wooseok Ha.
Variance-reduced zeroth-order methods for fine-tuning language models.arXiv preprint
arXiv:2404.08080, 2024.
Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex
stochastic programming.SIAM journal on optimization, 23(4):2341–2368, 2013.
Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus,
Mar 2019.
Yutong He, Pengrui Li, Yipeng Hu, Chuyan Chen, and Kun Yuan. Subspace optimization
for large language models with convergence guarantees.arXiv preprint arXiv:2410.11289,
2024.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai,
Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan
Clark, et al. Training compute-optimal large language models.arXiv preprint
arXiv:2203.15556, 2022.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models.
ICLR, 1(2):3, 2022.
Vladislav Lialin, Namrata Shivagunde, Sherin Muckatira, and Anna Rumshisky. Relora:
High-rank training through low-rank updates.arXiv preprint arXiv:2307.05695, 2023.
Kaizhao Liang, Bo Liu, Lizhang Chen, and Qiang Liu. Memory-efficient llm training with
online subspace descent. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,
J. Tomczak, and C. Zhang, editors,Advances in Neural Information Processing Systems,
volume 37, pages 64412–64432. Curran Associates, Inc., 2024. doi: 10.52202/079017-2054.
Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin,
Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training.arXiv preprint
arXiv:2502.16982, 2025.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy,
Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized
bert pretraining approach.arXiv preprint arXiv:1907.11692, 2019.
Qijun Luo, Mengqi Li, Lei Zhao, and Xiao Li. Streambp: Memory-efficient exact backprop-
agation for long sequence training of llms.arXiv preprint arXiv:2506.03077, 2025.
Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D Lee, Danqi Chen,
and Sanjeev Arora. Fine-tuning language models with just forward passes. In A. Oh,
T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors,Advances in
15

<!-- page 16 -->

Li, Zhang, F ang
Neural Information Processing Systems, volume 36, pages 53038–53075. Curran Asso-
ciates, Inc., 2023.
Roy Miles, Pradyumna Reddy, Ismail Elezi, and Jiankang Deng. Velora: Memory efficient
training using rank-1 sub-token projections. In A. Globerson, L. Mackey, D. Belgrave,
A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors,Advances in Neural Information
Processing Systems, volume 37, pages 42292–42310. Curran Associates, Inc., 2024. doi:
10.52202/079017-1338.
Yurii Nesterov.Introductory lectures on convex optimization: A basic course, volume 87.
Springer Science & Business Media, 2013.
Egor Petrov, Grigoriy Evseev, Aleksey Antonov, Andrey Veprikov, Nikolay Bushkov,
Stanislav Moiseev, and Aleksandr Beznosikov. Leveraging coordinate momentum in
signsgd and muon: Memory-optimized zero-order.arXiv preprint arXiv:2506.04430, 2025.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning
with a unified text-to-text transformer.Journal of machine learning research, 21(140):
1–67, 2020.
Yara Shamshoum, Nitzan Hodos, Yuval Sieradzki, and Assaf Schuster. CompAct: Com-
pressed activations for memory-efficient LLM training. In Luis Chiruzzo, Alan Rit-
ter, and Lu Wang, editors,Proceedings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 1511–1524, Albuquerque, New Mexico,
April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi:
10.18653/v1/2025.naacl-long.71.
Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear mem-
ory cost. In Jennifer Dy and Andreas Krause, editors,Proceedings of the 35th Interna-
tional Conference on Machine Learning, volume 80 ofProceedings of Machine Learning
Research, pages 4596–4604. PMLR, 10–15 Jul 2018.
Yao Shu, Qixin Zhang, Kun He, and Zhongxiang Dai. Refining adaptive zeroth-order
optimization at ease.arXiv preprint arXiv:2502.01014, 2025.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux,
Timoth´ ee Lacroix, Baptiste Rozi` ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
Llama: Open and efficient foundation language models.arXiv preprint arXiv:2302.13971,
2023.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman.
Glue: A multi-task benchmark and analysis platform for natural language understand-
ing. In Tal Linzen, Grzegorz Chrupa la, and Afra Alishahi, editors,Proceedings of the
2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for
NLP, pages 353–355, Brussels, Belgium, November 2018. Association for Computational
Linguistics. doi: 10.18653/v1/W18-5446.
16

<!-- page 17 -->

Principal-Random Subspace for LLM Activation Compression
Yuchen Yang, Yingdong Shi, Cheems Wang, Xiantong Zhen, Yuxuan Shi, and Jun Xu.
Reducing fine-tuning memory overhead by approximate and memory-sharing backprop-
agation.arXiv preprint arXiv:2406.16282, 2024.
Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-
Yu Chen, Jason D Lee, Wotao Yin, Mingyi Hong, et al. Revisiting zeroth-order optimiza-
tion for memory-efficient llm fine-tuning: A benchmark.arXiv preprint arXiv:2402.11592,
2024a.
Yimu Zhang, Yuanshi Liu, and Cong Fang. Adapm: a partial momentum algorithm for llm
training.arXiv preprint arXiv:2510.09103, 2025.
Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P Kingma,
Yinyu Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain
more.arXiv preprint arXiv:2406.16793, 2024b.
Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuan-
dong Tian. Galore: Memory-efficient llm training by gradient low-rank projection.arXiv
preprint arXiv:2403.03507, 2024a.
Yanjun Zhao, Sizhe Dang, Haishan Ye, Guang Dai, Yi Qian, and Ivor W Tsang. Second-
order fine-tuning without pain for llms: A hessian informed zeroth-order optimizer.arXiv
preprint arXiv:2402.15173, 2024b.
Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park, Vikas Chandra, Bo Long,
David Z. Pan, Zhangyang Wang, and Jinwon Lee. Apollo: Sgd-like memory, adamw-
level performance. In M. Zaharia, G. Joshi, and Y. Lin, editors,Proceedings of Machine
Learning and Systems, volume 7. MLSys, 2025.
17

<!-- page 18 -->

Li, Zhang, F ang
Appendix A. Implementation of PRAC
We present the PRAC method for compressing activations in Algorithm 1, and introduce
its application during forward and backward propagation in Algorithm 2.
Algorithm 2Activation Storage and Recovery for Backpropagation
Require:InputX, parametersW, forward functionf, ranks (r 1,r 2), step intervals (T1,T 2), current
training stept
1:Forward Propagation:
2:Compute output:Y←f(X;W)
3:Store compression components:{Q (t)
1 , Q(t)
2 ,X (t)
1 ,X (t)
2 } ← C(X, r 1, r2, T1, T2, t) (Algorithm 1)
4:Return:Y
5:
6:Backward Propagation:
7:Receive gradient w.r.t output:∇ Y L ← ∂L
∂Y
8:Reconstruct input approximation from cache: ˜X← X (t)
1 Q(t)
1 +X (t)
2 Q(t)
2
9:Compute parameter and input gradients:∇ X L,∇ W L ←backward(∇ Y L, ˜X, W)
10:Return:∇ W L,∇ X L
Appendix B. Memory Complexity Analysis
We take GPT structure as an example to show theoretical activation memory of PRAC. We
assume a uniform rank ratior 1, r2 ≥0 for the principal and random subspaces, respectively.
Usually,r 1 +r 2 ≤0.6.
A comparison of activation compression between PRAC and baseline methods is pre-
sented in Table 6. The proposed PRAC compresses both linear layers (including the pro-
jection layers in Attention and the MLP) and non-linear layers (including LayerNorm and
the GeLU activation). The mean and variance results in LayerNorm are not compressed,
as they contribute negligibly to the total memory footprint (i.e.,bs≪bsn). Flash Atten-
tion operations are not compressed by PRAC due to their intricate coupled structure and
engineering optimizations. Nevertheless, the compressions above are sufficient for PRAC to
achieve an overall memory reduction of nearly 40% in large-batch-size training tasks.
Appendix C. Theoretical Results for PRAC
C.1 Notations and Useful Lemma
Definition 8(Orthogonal Group).For a positive integerm, letM m(R)denote the set of
allm×mreal matrices. The orthogonal group, denotedO(m), is defined as:
O(m) :=
n
A∈M m(R)|A ⊤A=I m
o
.(8)
Lemma 9.Let ˜S∈R m×k be a random matrix whose entries are independent and identically
distributed (i.i.d.) standard normal variables, i.e., ˜Sij
i.i.d.∼ N(0,1). Consider the (thin) QR
decomposition of ˜S=V R, whereV∈R m×k satisfiesV ⊤V=I k andR∈R k×k is an upper-
triangular matrix with positive diagonal entries. Then for any orthogonal matrixO∈ O(m),
18

<!-- page 19 -->

Principal-Random Subspace for LLM Activation Compression
Table 6: Comparison of activation stored in the GPT series architecture, with decreased
activation values highlighted in red for PRAC.
Operation
Activation Saved
Baseline PRAC (0≤r 1 +r 2 ≤0.6)
X= LayerNorm( ˜X)
˜X∈R b×s×n
(µ( ˜X), σ( ˜X)2)∈R b×s
˜XP x ∈R b×s×(nr1), ˜XQ x ∈R b×s×(nr2)
(µ(X), σ(X) 2)∈R b×s
q=XW q
k=XW k
v=XW v
X∈R b×s×n XP q ∈R b×s×(nr1)
XQ q ∈R b×s×(nr2)
reshapeq, k, v
to (b, h, s, n/h) None None
Ah = flash attn(q, k, v) q, k, v∈R b×h×s×n/h
two buffers∈R b×s×h
q, k, v∈R b×h×s×n/h
two buffers∈R b×s×h
reshapeA h to (b, s, n) None None
A′′
o =A hW T
o Ah ∈R b×s×n AhPh ∈R b×s×(nr1)
AhQh ∈R b×s×(nr2)
residual:A ′
o =A ′′
o + ˜X None None
Ao = LayerNorm(A′
o) A′
o ∈R b×s×n
(µ(A′
o), σ(A′
o)2)∈R b×s
A′
oP ′
o ∈R b×s×(nr1), A′
oQ′
o ∈R b×s×(nr2)
(µ(A′
o), σ(A′
o)2)∈R b×s
A′
u =A oW T
up Ao ∈R b×s×n AoPo ∈R b×s×(nr1)
AoQo ∈R b×s×(nr2)
Au = GeLU(A′
u) A′
u ∈R b×s×m A′
uP ′
u ∈R b×s×(mr1)
A′
uQ′
u ∈R b×s×(mr2)
A′
d =A uW T
down Au ∈R b×s×m AuPu ∈R b×s×(mr1)
AuQu ∈R b×s×(mr2)
residual:A d =A ′
d +A ′
o None None
OVandVare identically distributed, i.e.
OV d=V∀O∈ O(m).(9)
ProofLetO∈ O(m) be an arbitrary fixed orthogonal matrix and set ˜S′ =O ˜S. Consider
the QR decomposition of ˜Sand ˜S′, we have
˜S=V R, ˜S′ =V ′R′,(10)
whereV, V ′ ∈R m×k satisfieV ⊤V=I k,(V ′)⊤V ′ =I k andR, R ′ are upper-triangular ma-
trices with positive diagonal entries. Since the standard normal distribution is rotationally
invariant, ˜S′ has the same distribution as ˜S, i.e., ˜S′ d= ˜S, therefore we haveV d=V ′. What’s
more,
˜S′ =O ˜S=OV R.(11)
19

<!-- page 20 -->

Li, Zhang, F ang
Comparing (10) and (11), from the uniqueness of the thin QR decomposition we have
V ′ =OV, R ′ =R, then
V ′ =OV d=V∀O∈ O(m).(12)
Lemma 10.Suppose the matrixM∈R m×m satisfiesM O=OM∀O∈ O(m), thenM
must be a scalar matrix (i.e.,M=cI m).
ProofWe construct the reflection matrix as follows:
Ji = diag(1,· · ·,1,−1|{z}
i
,1,· · ·,1)∈R m×m,(13)
where only thei-th diagonal entry is -1 and the others are 1. Clearly,J i ∈ O(m). From
the commutativity conditionJ k, compare the entries in thei-th row andj-th column (with
i̸=j), we have
(JiM) ij =−M ij,(M J k)ij =M ij.(14)
Equating the two expressions gives:
−Mij =M ij ⇒M ij = 0 (i̸=j).
Therefore,Mis a diagonal matrix. Denote it as:
M= diag(λ 1, λ2,· · ·, λ m).(15)
Then consider a permutation matrixP ij that swaps thei-th andj-th rows (and columns). It
is also orthogonal, withP −1
ij =P ⊤
ij =P ij. From the commutativity conditionP ijM=M P ij,
we have:
M=P ijM Pij.(16)
ConjugatingMbyP ij swaps thei-th andj-th diagonal entries ofM. Since the matrices
MandP ijM Pij are equal, their diagonal entries must coincide, givingλ i =λ j for any pair
i, j. Hence,Mmust be a scalar matrix (i.e.,M=cI m).
Lemma 11.Suppose the random matrixV∈R m×k satisfiesV ⊤V=I k and∀O∈
O(m), OV d=V. Then we have
E[V V ⊤] = k
m Im.(17)
ProofFor any orthogonal matrixO∈ O(m),OVhas the same distribution asV. Then,
E[V V ⊤] =E[(OV)(OV) ⊤] =OE[V V ⊤]O⊤,
due toO ⊤ =O −1,
E[V V ⊤]O=OE[V V ⊤]∀O∈ O(m).
20

<!-- page 21 -->

Principal-Random Subspace for LLM Activation Compression
From Lemma 10, the expectationE[V V ⊤] of the random matrix must be a scaler matrix,
i.e.,
E[V V ⊤] =cI m.(18)
The constantcis determined by computing the trace:
tr(E[V V ⊤]) =E[tr(V V ⊤)] =E[tr(V ⊤V)] =k.(19)
Meanwhile, tr(cIm) =cm, thuscm=k⇒c= k
m.
Lemma 12.SupposeQ 1 ∈R n×r1 be a fixed orthogonal matrix, random matrixS o be gener-
ated by (5), and let the random matrixQ 2 ∈R n×r2 be the first component of QR(S o), then
we have:
E[Q2Q⊤
2 ] = r2
n−r 1
(In −Q 1Q⊤
1 ).(20)
ProofLetU∈R n×(n−r1) be an orthonormal basis for the orthogonal complement ofP:
U ⊤U=I n−r1, Q⊤
1 U=O r1×(n−r1), Q1Q⊤
1 +U U ⊤ =I n,(21)
then
So =S−Q 1Q⊤
1 S=U U ⊤S.(22)
We define ˜S=U ⊤S∈R (n−r1)×r2, thenS o =U ˜S. SinceUhas orthonormal columns and
the entries ofSare i.i.d Gaussian, the entries of ˜Sare also i.i.d Gaussian. Perform QR
decomposition on ˜S: ˜S=V R, then
So =U ˜S= (U V)R,(23)
according to the uniqueness of the thin QR decomposition, the projection matrix in the
algorithm satisfiesQ 2 =U V∈R n×r2. Then,
E[Q2Q⊤
2 ] =E[U V V ⊤U ⊤] =UE[V V ⊤]U ⊤
(a)
=U( r2
n−r 1
In−r1)U ⊤
= r2
n−r 1
(In −Q 1Q⊤
1 )
(24)
where the equality (a) holds according to Lemma 11 andV∈R (n−r1)×r2.
C.2 Omitted Proofs in Section 4
Theorem 13(Unbiased Reconstruction of PRAC).LetP∈R n×r1 be the projection matrix
defined in(6). If the scaling factor is set tok= n−r1
r2
, the reconstruction ˜X=XP P ⊤
satisfiesE[ ˜X] =X.
21

<!-- page 22 -->

Li, Zhang, F ang
ProofThe expectations for the reconstruction ˜Xare as follows:
E[ ˜X] =E[XQ 1QT
1 +kXQ 2Q⊤
2 ] =XQ 1Q⊤
1 +kXE[Q 2Q⊤
2 ]
(a)
=XQ 1Q⊤
1 +k r2
n−r 1
X(I n −Q 2Q⊤
2 )
=XQ 1Q⊤
1 + n−r 1
r2
r2
n−r 1
X(I n −Q 2Q⊤
2 )
=X,
(25)
where in (a) we use Lemma 12 to replace theE[Q 2Q⊤
2 ] term.
Theorem 14(Unbiased Reconstruction of RAC).LetQ 2 ∈R n×r2 sample uniformly from
the Stiefel manifold. Set the scaling factork= n
r2
. Then the reconstruction ˜Xrac =
(kXQ 2)Q⊤
2 satisfiesE[ ˜Xrac] =X.
ProofThis result follows as a special case of Theorem 13 by settingr 1 = 0 and then
k=n/r 2.
Theorem 15(Variance Bound of PRAC).Under the conditions of Theorem 13, the recon-
struction error variance is bounded by:
E

∥ ˜X−X∥ 2
F

≤( n−r 1
r2
−1)


X−XQ 1Q⊤
1


2
F .
ProofWe evaluate the expected squared Frobenius norm of the reconstruction error,
E[∥X− ˜X∥2
F ] =E[∥X(I n −Q 1Q⊤
1 −kQ 2Q⊤
2 )∥2
F ]. (26)
Recall from the properties ofQ 1 thatI n −Q 1Q⊤
1 =U U ⊤. Substituting this relation, we
obtain
E[∥X− ˜X∥2
F ] =E[∥X(U U ⊤ −kQ 2Q⊤
2 )∥2
F ] =E[tr(X(U U ⊤ −kQ 2Q⊤
2 )2X ⊤)], (27)
where the last equality follows from the identity∥A∥ 2
F = tr(AA⊤). Then, we compute the
expectation of the squared matrix term:
E
h
(U U⊤ −kQ 2Q⊤
2 )2
i
=U U ⊤ −2kE[Q 2Q⊤
2 ] +k 2E[Q2Q⊤
2 ] (28)
From Lemma 11, we haveE[Q 2Q⊤
2 ] = 1
k U U⊤. Substituting this yields
E
h
(U U⊤ −kQ 2Q⊤
2 )2
i
=U U ⊤(1−2k 1
k +k 2 1
k ) =U U ⊤(k−1),(29)
Substituting (29) into (27), we have:
E[∥X− ˜X∥2
F ] = tr(XU U ⊤X ⊤)(k−1) = ( n−r 1
r2
−1)∥X−XQ 1Q⊤
1 ∥2
F (30)
22

<!-- page 23 -->

Principal-Random Subspace for LLM Activation Compression
Theorem 16(Variance Bound of RAC).Under the conditions of Theorem 14, then the
reconstruction error variance is bounded by:
E

∥ ˜Xrac −X∥ 2
F

≤( n
r2
−1)∥X∥ 2
F .
ProofThis result follows as a special case of Theorem 15 by settingr 1 = 0 and then
k=n/r 2.
Theorem 17(Lower Bound).Under Assumption 4, ifs < r < n, then
(4)≥
 n−s
r−s −1

q.
ProofWithout loss of generality, assume the activation matrix isX= diag(σ 1, . . . , σn)∈
Rn×n withσ 1 ≥σ 2 ≥ · · · ≥σ n >0 and, by Assumption 4,Pn
i=s+1 σ2
i ≤q. The unbiasedness
conditionE[XP P ⊤] =XimpliesE[P P ⊤] =I. SetA=P P ⊤ ∈R n×n; then the expected
error can be written as
ϵ=E∥X(A−I)∥ 2
F =
nX
i=1
σ2
i
h
E(Aii −1) 2 +
X
j̸=i
EA2
ij
i
≡
nX
i=1
σ2
i αi,(31)
whereα i ≥0 is defined by the bracket.
1.Vanishing of the firstscoefficients.Consider a family ofXwhereσ i =tfor
i≤swhileσ s+1, . . . , σn are fixed and satisfy the sum-of-squares constraint. If for
some distribution ofPwe have Ps
i=1 αi >0, then sendingt→ ∞would makeϵ
arbitrarily large, so sup X inf P ϵ=∞and the bound holds trivially. Hence we need
only consider strategies with Ps
i=1 αi = 0, i.e.α i = 0 for alli≤s. This forcesPto
be block-diagonal almost surely:
P=
Is 0
0 ˜P

, ˜P∈R (n−s)×(r−s).(32)
Consequently the error reduces to
ϵ=
nX
i=s+1
αiσ2
i .(33)
2.Lower bound on the sum of the remainingα i.Let ˜A= ˜P ˜P ⊤ ∈R (n−s)×(n−s).
FromE[A] =Iwe obtainE[tr( ˜A)] =E[tr(A)]−s=n−s. Because rank( ˜A)≤r−s,
Cauchy–Schwarz gives
tr( ˜A2)≥ [tr( ˜A)]2
r−s .(34)
Taking expectations and using Jensen’s inequality,
E[tr( ˜A2)]≥ (E[tr( ˜A)])2
r−s = (n−s) 2
r−s .(35)
23

<!-- page 24 -->

Li, Zhang, F ang
Now compute the sum ofα i fori > s:
nX
i=s+1
αi =
nX
i=s+1
h
E(Aii −1) 2 +
X
j̸=i
EA2
ij
i
=
nX
i=s+1
h nX
j=1
EA2
ij −1
i
(sinceEA ii = 1)
=
nX
i=s+1
nX
j=s+1
EA2
ij −(n−s) (by the block-diagonal form)
=E[tr( ˜A2)]−(n−s)
≥ (n−s) 2
r−s −(n−s)≡C.
(36)
3.Completion of the lower bound.For any fixedσ s+1, . . . , σn, usingα i ≥0 we have
nX
i=s+1
αiσ2
i ≥
 nX
i=s+1
αi

min
s+1≤i≤n
σ2
i ≥C·min
i
σ2
i .(37)
Taking the infimum over admissibleα i (i.e. over distributions ofP) and then the
supremum overσ i satisfyingPn
i=s+1 σ2
i ≤qyields
sup
σi
inf
αi
nX
i=s+1
αiσ2
i ≥C·sup
σi
min
i
σ2
i .(38)
Under the sum-of-squares constraint, sup min i σ2
i is attained when allσ 2
i are equal,
i.e.σ 2
i =q/(n−s); hence sup min i σ2
i =q/(n−s). Substituting this together with
the value ofCfrom (36) gives
sup
σi
inf
αi
nX
i=s+1
αiσ2
i ≥
(n−s) 2
r−s −(n−s)
 q
n−s =
 n−s
r−s −1

q.(39)
By (33) this is exactly a lower bound for the quantity (4), completing the proof.
Appendix D. Experiment Details
D.1 Pre-training Setting
We detail the architectural configurations and pre-training hyperparameters for both the
LLaMA and GPT models. To ensure numerical stability and computational efficiency, all
LLaMA models are trained using bfloat16 precision and GPT models are trained using
Automatic Mixed Precision. The key hyperparameters for LLaMA and GPT models across
different scales are summarized in Table 7.
24

<!-- page 25 -->

Principal-Random Subspace for LLM Activation Compression
LLaMA Settings.The LLaMA models are trained with a maximum sequence length
of 256 and a global batch size of 512. We employ a learning rate schedule comprising a
linear warmup over the first 10% of training steps, followed by a cosine annealing phase
that decays the learning rate to 10% of its peak value. Across all LLaMA model scales, we
adopt the optimal learning rates reported in the original papers for comparison methods.
For PRAC, the learning rate is selected from the grid{0.0006,0.0008,0.001,0.002,0.004},
which is closely aligned with the settings used by the baselines.
GPT Settings.For the GPT models, we adopt a maximum sequence length of 1024 while
maintaining a global batch size of 512. We set the warmup steps to 2K, which corresponds
to 4% of the total steps 3. Notably, given that the official code and hyperparameters for
GaLore (Zhao et al., 2024a) and RSO (Chen et al., 2025) on GPT models are not publicly
available, we train these models using learning rates aligned with the baseline. For these
methods, we configure the ranks as 256 and 512 for the 124M and 355M models, respectively.
Table 7: Hyperparameter settings for pre-traing LLaMA and GPT-2 model.
Params Hidden Intermediate Heads Layers Steps
LLaMA
35M 384 1024 8 6 5K
60M 512 1376 8 8 10K
130M 768 2048 12 12 20K
350M 1024 2736 16 24 60K
1B 2048 5461 24 32 150K
GPT-2 124M 768 3072 12 12 50K
355M 1024 4096 16 24 50K
D.2 Fine-tuning Setting
We fine-tune the pre-trained RoBERTa-Base model on the GLUE benchmark using the
Hugging Face library4. All tasks are trained for 30 epochs with a batch size of 16. Refer to
Table 8 for the specific hyperparameters employed for PRAC.
Table 8: Hyperparameter settings for fine-tuning RoBERTa-Base model on the GLUE
benchmark using the PRAC method.
COLA STSB MRPC RTE SST2 MNLI QNLI QQP
Batch Size 16 16 16 16 16 16 16 16
Epochs 30 30 30 30 30 30 30 30
Learing Rate 3E-05 3E-05 3E-05 3E-05 1E-05 1E-05 1E-05 1E-05
Rank (r1 +r 2) 4
PRAC Interval 200
Max Seq Length 512
3.https://github.com/zyushun/Adam-mini/tree/main
4.https://huggingface.co/transformers/model_doc/roberta.html
25

<!-- page 26 -->

Li, Zhang, F ang
D.3 Additional Result
Compatibility with Advanced Optimizers.PRAC is orthogonal to optimizer choice.
We integrate PRAC with memory-efficient optimizers Muon (Liu et al., 2025) and Adam-
mini (Zhang et al., 2024b). As shown in Figure 6 and Table 9, PRAC successfully reduces
memory by an additional approximately 30% on top of these optimizers with negligible
perplexity degradation, highlighting its versatility.
0 5000 10000 15000 20000
Iteration
3.2
3.4
3.6
3.8
4.0
4.2Validation Loss
Muon
PRAC
0 10000 20000 30000 40000 50000 60000
Iteration
3.0
3.5
4.0
4.5
5.0Validation Loss
Muon
PRAC
0 5000 10000 15000 20000
Iteration
3.5
4.0
4.5Validation Loss
Adam_mini
PRAC
0 10000 20000 30000 40000 50000 60000
Iteration
3
4
5Validation Loss
Adam_mini
PRAC
(a) Muon (LLaMA 130M) (b) Muon (LLaMA 350M) (c) Adam-mini (LLaMA 130M) (d) Adam-mini (LLaMA 350M)
Figure 6: Loss curves of pre-training LLaMA model based on the Muon and Adam-mini
optimizers w./w.o. PRAC.
Table 9: Combining with advanced optimizers on pretraining LLaMA. Report validation
perplexity (PPL, lower is better) and the algorithm’s peak memory usage (lower
is better).
Method PRAC LLaMA-130M LLaMA-350M
Ppl Mem Ppl Mem
Muon ×23.90 21.04 17.98 45.26
✓24.16 15.37 (↓27%) 18.04 30.82 (↓32%)
Adam-
mini
×24.22 21.01 18.60 45.42
✓24.48 15.36 (↓27%) 18.92 31.09 (↓31%)
Selection of the Scaling Factork.We evaluate different values ofkon the linear and
normalization layers of LLaMA-130M. Defining the theoretical baseline ask 0 = n−r1
r2
, we
testk∈ {k 0,0.5k 0,0.2k 0,1.2k 0}. The results in Figure 7 indicate thatk=k 0 yields the
best convergence, whereas other settings result in performance degradation. This empirical
evidence aligns with our theoretical findings (See Section 4.3).
17000 18000 19000 20000
Iteration
3.196
3.198
3.200
3.202Validation Loss
k0
0.5k0
0.2k0
1.2k0
14000 16000 18000 20000
Iteration
3.20
3.21
3.22
3.23
3.24
3.25
3.26Validation Loss
k0
0.5k0
0.2k0
1.2k0
(a) Linear Layer (b) Norm Layer
Figure 7: Loss curve of using differentkin LLaMA-130M pre-training, wherek 0 = n−r1
r2
,
Settingk=k 0 performs better than larger or smaller settings, whether in linear
or nonlinear layers.
26
