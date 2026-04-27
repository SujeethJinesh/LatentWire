# references/153_permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.pdf

<!-- page 1 -->

PermLLM: Learnable Channel Permutation for
N:M Sparse Large Language Models
Lancheng Zou1, Shuo Yin1, Zehua Pei1, Tsung-Yi Ho1, Farzan Farnia1, and Bei Yu1
1The Chinese University of Hong Kong
Abstract
Channel permutation is a powerful technique for enhancing the accuracy of N:M
sparse models by reordering the channels of weight matrices to prioritize the re-
tention of important weights. However, traditional channel permutation methods
rely on handcrafted quality metrics, which often fail to accurately capture the true
impact of pruning on model performance. To address this limitation, we propose
PermLLM, a novel post-training pruning framework that introduces learnable chan-
nel permutation (LCP) for N:M sparsity. LCP leverages Sinkhorn normalization to
transform discrete permutation matrices into differentiable soft permutation matri-
ces, enabling end-to-end optimization. Additionally, PermLLM incorporates an
efficient block-wise channel permutation strategy, which significantly reduces the
number of learnable parameters and computational complexity. PermLLM seam-
lessly integrates with existing one-shot pruning methods to adaptively optimize
channel permutations, effectively mitigating pruning-induced errors. Extensive
experiments on the LLaMA series, Qwen, and OPT models demonstrate that
PermLLM achieves superior performance in optimizing N:M sparse models. The
code is available athttps://github.com/lanchengzou/PermLLM.
1 Introduction
The rapid advancements in large language models (LLMs) [ 6, 61, 52, 1] have led to a notable
enhancement in their capabilities across a broad range of domains. However, the growing scale of
LLMs presents substantial challenges for efficient deployment. To address these challenges, model
compression techniques, such as quantization [ 57, 16, 10, 31, 65] and pruning [15, 50, 62], offer
promising solutions to reduce memory usage and computational overhead.
In this paper, we focus on network pruning [28, 22, 21], particularly semi-structured pruning [46, 42].
The core idea of network pruning is to eliminate redundancies within the model by preserving only
the essential weights while setting the less important ones to zero. Semi-structured pruning takes this
a step further by enforcing N:M sparsity, where N out of every M consecutive elements are set to
zero. The N:M sparsity pattern is natively supported by Sparse Tensor Core in NVIDIA GPUs [45]
to achieve speed-up, which makes semi-structured pruning a practical approach for efficient model
inference.
Recent studies on LLM pruning primarily focus on designing a better pruning metric to obtain
higher-quality masks to improve the accuracy of the sparse models [15, 50, 62]. RIA [62] introduces
a novel pruning metric that avoids channel corruption while accounting for the effect of activations.
Additionally, it proposes a two-stage channel permutation strategy to maximize the sum of retained
weight importance, which serves as the quality metric to evaluate channel permutation solution.
However, it is important to note that a discrepancy may exist between the handcrafted quality metric
and the actual impact on output loss, as illustrated in Figure 1. Moreover, it fails to fully capture the
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
arXiv:2510.10136v1  [cs.LG]  11 Oct 2025

<!-- page 2 -->

Min. Loss Channel PermutationMax. Score Channel Permutation
Loss=14.375 Loss=4.75
S=96 S=93
114-217-10-6-2116
0-4
31
21
31
1-4
10
-42
3-4
3
-3
-2
0
-1
3
-1
-3
-1
0
-4
-4
-3
-3
1
2
-21
2-2
-1-3
-32
3-2
-1-2
3-3
1-2
-32
-4-1
03
-1-2
02
-42
-20
-10
3-213-12-1-2
-4
3
2
3
-4
-42
3-4
3-2
-4
-4
-3
-3
3
-3
-4
-4
-1
3
-3
3
-3
1
-2
2
-3
3
3
-1
-3
2
-2
-2
-3
-2
2
3
-2
168-184-8-6-2510
0 1 2 3 4 5 6 7 0 1 3 6 2 4 5 7
3
3
-4
3
3
-3
3
-3
-4
-4
-3
-3
-1
-3
-2
-3
3132
0 2 3 5
-4
-4
-4
2
-1
-3
3
3
-3
-4
3
-2
-4 2
-2
-1
-2-1-1-2
1 6 4 7
118-234-8-6-1915
x
W
y
x
W
y
x
W
y
Original
Pruned Weight Unpruned Weight
Figure 1: Effects of different channel permutation strategies on the outputs. Channel order is in
purple. We use magnitude pruning [21] for 2:4 sparsity in this example. Score S denotes the sum of
retained weight importance, which is used as the quality metric for channel permutation [46, 62]. Loss
is the mean square error between the original output y and the output of the pruned one. The output
loss of direct 2:4 sparsity (i.e., without channel permutation) is 12.375. The results demonstrate that
channel permutation which maximizes the score may lead to performance degradation.
complex inter-layer interactions, thereby missing opportunities to compensate for pruning errors and
improve the overall performance of the sparse model.
To overcome the limitations of prior channel permutation methods, we are the first to present learnable
channel permutation (LCP) for N:M sparsity. Unlike previous approaches that rely on handcrafted
quality metrics as optimization proxies, the proposed post-training framework, PermLLM, directly
minimizes the output errors between the dense model and the sparse model.
However, achieving feasible and practical permutation learning for pruning presents two major
challenges: (1) the discrete nature and strict combinatorial constraints of permutation matrices
render them non-differentiable, hindering effective optimization; (2) the vast solution space of
permutations, particularly in LLMs with high-dimensional weight matrices, results in prohibitively
high computational complexity.
To address these challenges, we first relax hard permutation matrices into soft permutation matrices
using Sinkhorn normalization [48], enabling gradient-based optimization. Then, we introduce an
efficient block-wise channel permutation strategy, which significantly reduces the number of learnable
parameters and computational overhead. PermLLM is fully compatible with existing efficient one-
shot pruning methods, such as Wanda [ 50] and RIA [ 62], enabling pruning-aware permutation
learning that adaptively minimizes pruning-induced errors. Moreover, a customized CUDA kernel is
developed to accelerate the channel permutation operation, achieving a significant speedup compared
to the Pytorch implementation. Extensive experiments underscore the effectiveness of PermLLM,
demonstrating its ability to enhance the performance of existing one-shot pruning methods across
various LLMs, particularly for updated models such as LLaMA-3.1 and Qwen-2.5.
2 Preliminaries
2.1 Large Language Models Pruning
The effectiveness of network pruning [ 28, 22, 20, 21, 23] has garnered significant attention from
researchers, prompting extensive exploration for LLM pruning.
Based on the granularity of pruning, prior works can be categorized into three types: structured
pruning [36, 3, 49, 55, 38, 44], semi-structured pruning [62, 14] and unstructured pruning [15, 50,
4, 11]. Unstructured pruning is the most flexible approach, as it is not constrained by specific
patterns. This flexibility often leads to improved accuracy; however, it comes at the expense
of limited efficiency gains. In contrast, structured pruning removes weights at a coarse-grained
level, such as channels [ 36], layers [38, 7], or blocks [ 49, 44], thereby enabling more substantial
2

<!-- page 3 -->

improvements in computational efficiency. However, the structural removal often leads to significant
accuracy degradation, necessitating retraining or fine-tuning to mitigate pruning-induced errors.
Semi-structured pruning serves as an intermediate approach, introducing hardware-friendly patterns,
such as N:M sparsity [42] which retains only N zero values within each group of M values. This
method achieves a compromise between the acceleration benefits of structured pruning and the
flexibility of fine-grained sparsity.
In general, there are three pipelines to obtain a sparse model [8]: pruning before training (PBT) [29,
54], pruning during training (PDP) [13, 32] and post-training pruning (PTP) [28, 22]. PBT and PDP
typically demand substantial training efforts, which makes PTP the widely adopted pipeline for LLM
pruning due to its lower computational cost. The objective of PTP can be formulated as follows:
arg min
M
∥WX−(M⊙W)·X∥ 2
2,s.t.∥M∥ 0 ≤k,(1)
where W∈R Cout×Cin represents the pre-trained weight with Cout output channels and Cin input
channels. The goal of PTP is to determine a mask M that minimizes the reconstruction error under
the given input X from calibration dataset and specific sparsity constraints (e.g., sparsity ratio and
pruning granularity).
2.2 N:M Sparsity
NVIDIA Ampere architecture [ 45] leverages Sparse Tensor Core to accelerate model inference
with N:M sparsity [ 42]. For instance, compressing the model with 2:4 sparsity can theoretically
achieve a 2× increase in compute throughput for sparse matrix multiplication compared to its
dense counterpart. Thus, this approach has garnered significant attention for its ability to improve
computational efficiency while maintaining model accuracy.
RIA [62] introduces a one-shot pruning method based on a handcrafted importance metric for semi-
structured pruning. While one-shot pruning is highly efficient, it relies on handcrafted importance
metrics as proxies for true discrepancy, resulting in a significant gap with the actual pruning-induced
discrepancy. To address this issue, researchers have introduced various methodologies for learnable
N:M masks [ 64, 34, 24, 26, 14]. Sparse-Refined Straight-Through Estimator (SR-STE) [ 64] is
proposed by extending original Straight-Through Estimator (STE) [5] to train N:M sparse models
from scratch.
2.3 Channel Permutation
Channel permutation [25, 46, 37] has proven to be an effective technique for improving the accuracy
of pruning with specific sparsity patterns (e.g., N:M sparsity) by reordering the input channels of
the weight matrix. More recently, researchers have explored reordering to enhance quantization
performance [16, 59, 30], highlighting channel permutation as a promising approach that merits
further investigation.
For a linear layer with Cin input channels, there are Cin! possible permutation candidates. Due
to the nature of N:M sparsity, channel permutation can be formulated as the following problem:
distributing Cin distinguishable balls into Cin/M indistinguishable boxes, where each box contains
exactly M balls. In this case, the solution space is reduced to Cin!
(M!) G·G!, where G=C in/M denotes
the number of pruning groups. When Cin = 16 and M= 4 , the reduced solution space still
contains approximately 2.6 million candidates. The solution space grows rapidly with increasing Cin,
leading to significant computational challenges for large values of Cin. Exhaustive search algorithm
combined with a greedy incremental refinement strategy is applied for channel permutation [ 46].
However, this approach is primarily suitable for models with a small number of channels and becomes
computationally expensive when applied to LLMs with large hidden dimensions. To address the
computational overhead, RIA [62] adopts a heuristic channel allocation method that iteratively assign
important channels to different blocks efficiently. Subsequently, a refinement process is applied,
formulated as a linear sum assignment problem, to maximize the sum of retained weight importance
scores.
Nevertheless, the handcrafted weight importance metric, used as a quality proxy in previous channel
permutation methods [46, 62], fails to accurately capture the relationship between pruning error
and channel permutation, resulting in suboptimal solutions. As illustrated in Figure 1, channel
3

<!-- page 4 -->

permutation based on maximum importance score does not necessarily reduce pruning error and may
even lead to an increase in error.
To address the aforementioned challenges and limitations, this study pushes the boundaries of post-
training semi-structured pruning for LLMs by learnable channel permutation (LCP). This approach
enables an end-to-end learning of channel reordering, eliminating the need for the handcrafted
quality metrics. The proposed LCP serves as an effective plugin for existing one-shot pruning
methods [50, 62] by identifying appropriate channel reordering to mitigate mask quality limitations
and reduce pruning errors.
3 Learnable Channel Permutation
The objective of channel permutation is to determine a permutation matrix P∈R Cin×Cin for the
weight matrix W∈R Cout×Cin, such that the reordered weight matrix, cW=WP , can achieve
improved accuracy after applying N:M sparsity.
However, there are two major challenges to learn the permutation matrix P: (1) P is a binary matrix
containing only 0s and 1s, which makes it inherently discrete and thus non-differentiable. The discrete
nature of P poses a significant challenge for gradient-based learning methods. Moreover, P must
satisfy the properties of a permutation matrix—each row and column must contain exactly one “1"
(with all other entries being “0"). This introduces strict combinatorial constraints that significantly
increase the complexity of the learning process. (2) The number of possible permutation candidates
increases factorially with Cin. In LLMs, Cin typically exceeds one thousand, leading to an extremely
vast solution space and posing a significant challenge for the design of efficient algorithms.
3.1 Relaxation to Soft Permutation Matrix
Some existing mask learning methods assign a learnable score [63] or probability [14] to each mask
candidate to identify the best option. Although permutation learning can also be formulated as a
combinatorial problem, the vast solution space of permutations renders these previously proposed
methods impractical. Consequently, directly learning the permutation matrix tends to be more
feasible.
To address the challenges associated with the discrete nature and properties of permutation matrix, a
common approach is to relax the hard constraints and represent the permutation using a soft permuta-
tion matrix. The soft permutation matrix, denoted as bP, serves as a continuous and differentiable
approximation of the discrete permutation matrixP, thereby enabling gradient-based learning method.
Adoubly stochastic matrixcan be used as bP [2], where all entries are non-negative and each row
and column sums to 1. This contrasts with P, in which each row and column contains exactly
one “1". By leveragingSinkhorn normalization[ 48, 2, 39, 12, 35], a nonnegative square matrix
can be converted into a doubly stochastic matrix through an iterative process of row and column
normalization.
Thus, any square matrixXcan be transformed into a doubly stochastic matrix as follows:
S0(X) =exp(X),(2)
Si(X) =T c
 
Tr(Si−1(X))

,(3)
S(X) = lim
l→∞
Sl(X),(4)
where a non-negative square matrix is first obtained by Equation (2). Then iterative row and column
normalization is performed by Equations (3) and (4). Tr(X) =X⊘(X1 N 1⊤
N ) is the row-wise
normalization operation and Tc(X) =X⊘(1 N 1⊤
N X) is used for column normalization. ⊘ represents
element-wise division and 1N denotes a column vector of one. Thus, the soft permutation matrix bP
can be obtained by
bP=S L(WP /τ),(5)
where WP is a learnable matrix with the same shape as bP. Since the limit in Equation (4) cannot be
computed exactly in practice, a truncated version withl→L is typically used for implementation [39,
12]. The temperature coefficient τ controls the hardness of the soft permutation matrix: as τ
approaches zero, the entries ofbPconverge to either 0 or 1.
4

<!-- page 5 -->

AsbP is not a strict permutation matrix, directly using it for channel permutation modifies both the
channel order and the weight values. To avoid its impact on mask selection, bP is hardened into a
strict permutation matrix P during the forward pass. This hardening process can be formulated as
a linear sum assignment problem and solved by using Hungarian algorithm [27]. Specifically, this
process identifies the hard permutation matrix P that is closest to the soft permutation matrixbP. It
achieves this by solving the following optimization problem:
P= arg max
P∈P
Tr(P⊤bP),(6)
where P represents the set of all valid permutation matrices and Tr(·) denotes the trace operator. The
objective is to maximize the alignment between P and bP by selecting the entries of bP that yield
the highest overall score. Unfortunately, the hardening process is not differentiable. To address this
limitation, STE [5] is employed to approximate the gradient in the backward pass, i.e., ∂P/∂bP= 1 .
By propagating gradients through this approximation, the STE preserves gradient flow across the
computational graph, thereby ensuring end-to-end trainability of the permutation learning framework.
3.2 Block-wise Learnable Channel Permutation
Block-wise LCP
<latexit sha1_base64="9uOmMm80DPYIK5s7B6aEsjBJJ8Q=">AAACz3icjVHLTsJAFD3UF+ILdemmEUxckcICXRLduIREHgkYMx0GaOgr7VRDCMatP+BW/8r4B/oX3hlLohKj07Q9c+49Z+bea4euE0vLes0YS8srq2vZ9dzG5tb2Tn53rxUHScRFkwduEHVsFgvX8UVTOtIVnTASzLNd0bbH5yrevhFR7AT+pZyE4spjQ98ZOJxJonrFnsfkyB5M27Pidb5glSy9zEVQTkEB6aoH+Rf00EcAjgQeBHxIwi4YYnq6KMNCSNwVpsRFhBwdF5ghR9qEsgRlMGLH9B3SrpuyPu2VZ6zVnE5x6Y1IaeKINAHlRYTVaaaOJ9pZsb95T7WnutuE/nbq5RErMSL2L9088786VYvEAKe6BodqCjWjquOpS6K7om5ufqlKkkNInMJ9ikeEuVbO+2xqTaxrV71lOv6mMxWr9jzNTfCubkkDLv8c5yJoVUrlaqnaqBRqZ+moszjAIY5pnieo4QJ1NMk7xCOe8Gw0jFvjzrj/TDUyqWYf35bx8AF4/JPh</latexit>
W
<latexit sha1_base64="9uOmMm80DPYIK5s7B6aEsjBJJ8Q=">AAACz3icjVHLTsJAFD3UF+ILdemmEUxckcICXRLduIREHgkYMx0GaOgr7VRDCMatP+BW/8r4B/oX3hlLohKj07Q9c+49Z+bea4euE0vLes0YS8srq2vZ9dzG5tb2Tn53rxUHScRFkwduEHVsFgvX8UVTOtIVnTASzLNd0bbH5yrevhFR7AT+pZyE4spjQ98ZOJxJonrFnsfkyB5M27Pidb5glSy9zEVQTkEB6aoH+Rf00EcAjgQeBHxIwi4YYnq6KMNCSNwVpsRFhBwdF5ghR9qEsgRlMGLH9B3SrpuyPu2VZ6zVnE5x6Y1IaeKINAHlRYTVaaaOJ9pZsb95T7WnutuE/nbq5RErMSL2L9088786VYvEAKe6BodqCjWjquOpS6K7om5ufqlKkkNInMJ9ikeEuVbO+2xqTaxrV71lOv6mMxWr9jzNTfCubkkDLv8c5yJoVUrlaqnaqBRqZ+moszjAIY5pnieo4QJ1NMk7xCOe8Gw0jFvjzrj/TDUyqWYf35bx8AF4/JPh</latexit>
W
<latexit sha1_base64="XaqQzsYlEym5J5YX9U1eDmwNEWA=">AAACz3icjVHLTsJAFD3UF+ILdemmEUxckcICXRLduIREHgkYMx0GaOgr7VRDCMatP+BW/8r4B/oX3hlLohKj07Q9c+49Z+bea4euE0vLes0YS8srq2vZ9dzG5tb2Tn53rxUHScRFkwduEHVsFgvX8UVTOtIVnTASzLNd0bbH5yrevhFR7AT+pZyE4spjQ98ZOJxJonrFnsfkyB5M67Pidb5glSy9zEVQTkEB6aoH+Rf00EcAjgQeBHxIwi4YYnq6KMNCSNwVpsRFhBwdF5ghR9qEsgRlMGLH9B3SrpuyPu2VZ6zVnE5x6Y1IaeKINAHlRYTVaaaOJ9pZsb95T7WnutuE/nbq5RErMSL2L9088786VYvEAKe6BodqCjWjquOpS6K7om5ufqlKkkNInMJ9ikeEuVbO+2xqTaxrV71lOv6mMxWr9jzNTfCubkkDLv8c5yJoVUrlaqnaqBRqZ+moszjAIY5pnieo4QJ1NMk7xCOe8Gw0jFvjzrj/TDUyqWYf35bx8AFoTpPa</latexit>
P
Frozen Learnable
Full Matrix LCP
<latexit sha1_base64="Cej6bF6JE5eK33fQTkUYzRe5Tcw=">AAAC0XicjVHLTsJAFD3UF+ILdemmEUxckZYFuiS6cYlRHgkgacsADX1lOjUhhMS49Qfc6k8Z/0D/wjtjSVRidJq2Z86958zce+3Ic2NhGK8ZbWl5ZXUtu57b2Nza3snv7jXiMOEOqzuhF/KWbcXMcwNWF67wWCvizPJtjzXt8bmMN28Zj90wuBaTiHV9axi4A9exBFE3xY5viZE9mDZnvVqxly8YJUMtfRGYKSggXbUw/4IO+gjhIIEPhgCCsAcLMT1tmDAQEdfFlDhOyFVxhhlypE0oi1GGReyYvkPatVM2oL30jJXaoVM8ejkpdRyRJqQ8Tliepqt4opwl+5v3VHnKu03ob6dePrECI2L/0s0z/6uTtQgMcKpqcKmmSDGyOid1SVRX5M31L1UJcoiIk7hPcU7YUcp5n3WliVXtsreWir+pTMnKvZPmJniXt6QBmz/HuQga5ZJZKVUuy4XqWTrqLA5wiGOa5wmquEANdfLmeMQTnrUrbaLdafefqVom1ezj29IePgBt+JSk</latexit>
W P
Sinkhorn
Norm.
LSA
<latexit sha1_base64="xT3prabmjygJxnXeCYN8l9/gmbg=">AAAC23icjVHLSsNAFD2N73dUcOMm2AquStpFdSm6cVnBqtAWmaTTdmheJBOlxK7ciVt/wK3+j/gH+hfeGVPwgeiEJGfOvefM3HudyBOJtO2XgjExOTU9Mzs3v7C4tLxirq6dJmEau7zhhl4Ynzss4Z4IeEMK6fHzKObMdzx+5gwOVfzskseJCIMTOYx422e9QHSFyyRRF+ZGqXUlOrzPZNbymew73aw+GpUuzKJdtvWyfoJKDorIVz00n9FCByFcpPDBEUAS9sCQ0NNEBTYi4trIiIsJCR3nGGGetCllccpgxA7o26NdM2cD2ivPRKtdOsWjNyalhW3ShJQXE1anWTqeamfF/uadaU91tyH9ndzLJ1aiT+xfunHmf3WqFoku9nQNgmqKNKOqc3OXVHdF3dz6VJUkh4g4hTsUjwm7Wjnus6U1ia5d9Zbp+KvOVKzau3luijd1Sxpw5fs4f4LTarlSK9eOq8X9g3zUs9jEFnZonrvYxxHqaJD3NR7wiCejbdwYt8bdR6pRyDXr+LKM+3fCDpip</latexit>
bP
1
1
1
1
1
1
1
1
ABCDEFGH
AB CDE FGH
<latexit sha1_base64="pp2ATsYqBD9QbTbA6zd6dzzhn2I=">AAAC23icjVHLSsNAFD3Gd31FBTdugq3gqqRdVJeiG5cK1hbaIpN02g7Ni2SilNiVO3HrD7jV/xH/QP/CO2MEH4hOSHLm3HvOzL3XiTyRSNt+njAmp6ZnZufmCwuLS8sr5uraWRKmscvrbuiFcdNhCfdEwOtSSI83o5gz3/F4wxkeqnjjgseJCINTOYp4x2f9QPSEyyRR5+ZGqX0punzAZNb2mRw4vawxHpfOzaJdtvWyfoJKDorI13FoPqGNLkK4SOGDI4Ak7IEhoaeFCmxExHWQERcTEjrOMUaBtCllccpgxA7p26ddK2cD2ivPRKtdOsWjNyalhW3ShJQXE1anWTqeamfF/uadaU91txH9ndzLJ1ZiQOxfuo/M/+pULRI97OkaBNUUaUZV5+Yuqe6Kurn1qSpJDhFxCncpHhN2tfKjz5bWJLp21Vum4y86U7Fq7+a5KV7VLWnAle/j/AnOquVKrVw7qRb3D/JRz2ETW9ihee5iH0c4Rp28r3CPBzwaHePauDFu31ONiVyzji/LuHsD0sOYsA==</latexit>
cW
ABCDEFGH
<latexit sha1_base64="BYYo34W65SSyfzLcthCdKVPyDzI=">AAAC1XicjVHLTsJAFD3UF+Kr6tJNI5i4Ii0LdEl04xIToSSApC0DNPSVdkpCCDvj1h9wq79k/AP9C++MJVGJ0Wnanjn3njNz77Ujz024rr/mlJXVtfWN/GZha3tnd0/dP2gmYRo7rOGEXhi3bCthnhuwBne5x1pRzCzf9phpjy9F3JywOHHD4IZPI9b1rWHgDlzH4kT1VLXU8S0+sgczc96r3xqlnlrUy7pc2jIwMlBEtuqh+oIO+gjhIIUPhgCcsAcLCT1tGNAREdfFjLiYkCvjDHMUSJtSFqMMi9gxfYe0a2dsQHvhmUi1Q6d49Mak1HBCmpDyYsLiNE3GU+ks2N+8Z9JT3G1Kfzvz8onlGBH7l26R+V+dqIVjgHNZg0s1RZIR1TmZSyq7Im6ufamKk0NEnMB9iseEHalc9FmTmkTWLnpryfibzBSs2DtZbop3cUsasPFznMugWSkb1XL1ulKsXWSjzuMIxzileZ6hhivU0SDvCR7xhGfFVObKnXL/markMs0hvi3l4QOqsZV4</latexit>
W1
P
<latexit sha1_base64="LGpijO6NSf0bSnEduhdpYfP/n8Q=">AAAC1XicjVHLTsJAFD3UF+Kr6tJNI5i4Ii0LdEl04xIToSSApC0DNPSVdkpCCDvj1h9wq79k/AP9C++MJVGJ0Wnanjn3njNz77Ujz024rr/mlJXVtfWN/GZha3tnd0/dP2gmYRo7rOGEXhi3bCthnhuwBne5x1pRzCzf9phpjy9F3JywOHHD4IZPI9b1rWHgDlzH4kT1VLXU8S0+sgczc96r31ZKPbWol3W5tGVgZKCIbNVD9QUd9BHCQQofDAE4YQ8WEnraMKAjIq6LGXExIVfGGeYokDalLEYZFrFj+g5p187YgPbCM5Fqh07x6I1JqeGENCHlxYTFaZqMp9JZsL95z6SnuNuU/nbm5RPLMSL2L90i8786UQvHAOeyBpdqiiQjqnMyl1R2Rdxc+1IVJ4eIOIH7FI8JO1K56LMmNYmsXfTWkvE3mSlYsXey3BTv4pY0YOPnOJdBs1I2quXqdaVYu8hGnccRjnFK8zxDDVeoo0HeEzziCc+KqcyVO+X+M1XJZZpDfFvKwwetEpV5</latexit>
W2
P
<latexit sha1_base64="GWKEH2ylz3Qh5mRfhFCV7wD0RFs=">AAAC1XicjVHLTsJAFD3UF+Kr6tJNI5i4Ii0m6JLoxiUmQk0ASVsGaOgr7ZSEEHbGrT/gVn/J+Af6F94ZS6ISo9O0PXPuPWfm3mtHnptwXX/NKUvLK6tr+fXCxubW9o66u9dMwjR2WMMJvTC+sa2EeW7AGtzlHruJYmb5tsdMe3Qh4uaYxYkbBtd8ErGObw0Ct+86Fieqq6qltm/xod2fmrNu/fak1FWLelmXS1sERgaKyFY9VF/QRg8hHKTwwRCAE/ZgIaGnBQM6IuI6mBIXE3JlnGGGAmlTymKUYRE7ou+Adq2MDWgvPBOpdugUj96YlBqOSBNSXkxYnKbJeCqdBfub91R6irtN6G9nXj6xHENi/9LNM/+rE7Vw9HEma3Cppkgyojonc0llV8TNtS9VcXKIiBO4R/GYsCOV8z5rUpPI2kVvLRl/k5mCFXsny03xLm5JAzZ+jnMRNCtlo1quXlWKtfNs1Hkc4BDHNM9T1HCJOhrkPcYjnvCsmMpMuVPuP1OVXKbZx7elPHwAr3OVeg==</latexit>
W3
P
<latexit sha1_base64="23oTSVR5FiRz5H9MxNEWCyNGI2k=">AAAC1XicjVHLTsJAFD3UF+Kr6tJNI5i4Ii0x6JLoxiUmQk0ASVsGaOgr7ZSEEHbGrT/gVn/J+Af6F94ZS6ISo9O0PXPuPWfm3mtHnptwXX/NKUvLK6tr+fXCxubW9o66u9dMwjR2WMMJvTC+sa2EeW7AGtzlHruJYmb5tsdMe3Qh4uaYxYkbBtd8ErGObw0Ct+86Fieqq6qltm/xod2fmrNu/fak1FWLelmXS1sERgaKyFY9VF/QRg8hHKTwwRCAE/ZgIaGnBQM6IuI6mBIXE3JlnGGGAmlTymKUYRE7ou+Adq2MDWgvPBOpdugUj96YlBqOSBNSXkxYnKbJeCqdBfub91R6irtN6G9nXj6xHENi/9LNM/+rE7Vw9HEma3Cppkgyojonc0llV8TNtS9VcXKIiBO4R/GYsCOV8z5rUpPI2kVvLRl/k5mCFXsny03xLm5JAzZ+jnMRNCtlo1quXlWKtfNs1Hkc4BDHNM9T1HCJOhrkPcYjnvCsmMpMuVPuP1OVXKbZx7elPHwAsdSVew==</latexit>
W4
P
Block-wise
Sinkhorn
Norm.
<latexit sha1_base64="f3Yh4B2OcSbF+C1cgtLecL6Go4I=">AAAC3XicjVHLSsNAFD3GV31X3Qhugq3gqiRdqEvRjcsKtgptCZN02g7mRTJRpNSdO3HrD7jV3xH/QP/CO2MKahGdkOTMufecmXuvG/silZb1OmFMTk3PzBbm5hcWl5ZXiqtrjTTKEo/XvciPknOXpdwXIa9LIX1+HiecBa7Pz9yLIxU/u+RJKqLwVF7HvB2wXii6wmOSKKe4UW5diQ7vMzloBUz23e6gNhw6h2WnWLIqll7mOLBzUEK+alHxBS10EMFDhgAcISRhHwwpPU3YsBAT18aAuISQ0HGOIeZJm1EWpwxG7AV9e7Rr5mxIe+WZarVHp/j0JqQ0sU2aiPISwuo0U8cz7azY37wH2lPd7Zr+bu4VECvRJ/Yv3SjzvzpVi0QX+7oGQTXFmlHVeblLpruibm5+qUqSQ0ycwh2KJ4Q9rRz12dSaVNeuest0/E1nKlbtvTw3w7u6JQ3Y/jnOcdCoVuzdyu5JtXRwmI+6gE1sYYfmuYcDHKOGOnnf4BFPeDYc49a4M+4/U42JXLOOb8t4+ACeiJle</latexit>
bPB
Block-wise
LSA
AB CD EF GH
1
1
1
1
1
1
1
1
<latexit sha1_base64="RwKWt6WWH8IcRUTOPSXkryvBmRs=">AAAC0XicjVHLTsJAFD3UF+ILdemmEUxckZYFuiS4cYlRwASQtGWAhr4ynZoQQmLc+gNu9aeMf6B/4Z2xJCoxOk3bM+fec2buvXbkubEwjNeMtrS8srqWXc9tbG5t7+R395pxmHCHNZzQC/m1bcXMcwPWEK7w2HXEmeXbHmvZ4zMZb90yHrthcCUmEev61jBwB65jCaJuih3fEiN7MK3PerViL18wSoZa+iIwU1BAuuph/gUd9BHCQQIfDAEEYQ8WYnraMGEgIq6LKXGckKviDDPkSJtQFqMMi9gxfYe0a6dsQHvpGSu1Q6d49HJS6jgiTUh5nLA8TVfxRDlL9jfvqfKUd5vQ3069fGIFRsT+pZtn/lcnaxEY4FTV4FJNkWJkdU7qkqiuyJvrX6oS5BARJ3Gf4pywo5TzPutKE6vaZW8tFX9TmZKVeyfNTfAub0kDNn+OcxE0yyWzUqpclAvVWjrqLA5wiGOa5wmqOEcdDfLmeMQTnrVLbaLdafefqVom1ezj29IePgA77pSP</latexit>
PB
<latexit sha1_base64="FyjzbXvK+lInR9PuELwCOTmTNvU=">AAAC3XicjVHLSsNAFD3G9zvqRnATbAVXJe1CXZa6calgH9CWMkmn7WBeJBNFQt25E7f+gFv9HfEP9C+8M0bwgeiEJGfOvefM3HudyBOJtO3nCWNyanpmdm5+YXFpeWXVXFtvJGEau7zuhl4YtxyWcE8EvC6F9HgrijnzHY83nbNDFW+e8zgRYXAqLyPe9dkwEAPhMklUz9wsdi5En4+YzDo+kyNnkDXH416t2DMLdsnWy/oJyjkoIF/HofmEDvoI4SKFD44AkrAHhoSeNsqwERHXRUZcTEjoOMcYC6RNKYtTBiP2jL5D2rVzNqC98ky02qVTPHpjUlrYIU1IeTFhdZql46l2Vuxv3pn2VHe7pL+Te/nESoyI/Uv3kflfnapFYoADXYOgmiLNqOrc3CXVXVE3tz5VJckhIk7hPsVjwq5WfvTZ0ppE1656y3T8RWcqVu3dPDfFq7olDbj8fZw/QaNSKu+V9k4qhWotH/UctrCNXZrnPqo4wjHq5H2Fezzg0egZ18aNcfueakzkmg18WcbdG69LmWU=</latexit>
cW B
(a) (b)
Figure 2: Illustration of learnable channel permuta-
tion with different granularity: (a) full matrix LCP;
(b) block-wise LCP.
According to Equation (5), if channel is allowed
to be permuted flexibly, the learnable param-
eter matrix will be WP ∈R Cin×Cin, which
usually has the similar or same shape with the
weight matrix W. If each weight were to have
its own learnable permutation, the learning bur-
den would become prohibitively large.
To address this, we apply a block-wise learnable
channel permutation that only allows channel
permutation operates within the block to reduce
the training cost. It is inspired by the widely
adopted block-wise operations in model com-
pression [16, 15, 66, 30].
Originally, the number of parameters in WP is
C 2
in for full matrix learnable channel permuta-
tion. Let the block size be B. For permutation
for a single block, the number of parameters in
Wi
P is B2. With NB representing the total num-
ber of blocks, the overall number of parameters
is given by NB ×B 2 = Cin
B ×B 2 =C in ×B .
By block-wise learnable channel permutation,
we reduce the number of parameters to B
Cin
of the original, achieving significant parameter savings
whenB≪C in.
Another advantage of block-wise learnable channel permutation is its enhanced computational
efficiency when hardening the soft permutation matrix. This process is solved using the Hungarian
algorithm [27], which has a time complexity of O(N 3). For a full matrix permutation, the time
complexity becomes O(C 3
in). In contrast, by adopting block-wise manner, the time complexity
for a single block is O(B 3). Given that there are NB blocks in total, the overall complexity is
O(NB ·B 3) =O(C in ·B 2). This demonstrates that it significantly reduces the computational
cost of hardening process by utilizing block-wise learnable channel permutation, particularly when
B≪C in.
To perform block-wise learnable channel permutation for W, each learnable matrix Wi
P is trans-
formed into a hard permutation matrix Pi for the i-th block. Unlike the reordered weight matrix
cW=WP obtained through full matrix permutation, the reordered weight matrix under block-wise
permutation is given by cWB =WP B, where PB =diag(P 1,P 2, . . . ,PNB ) represents a block
diagonal matrix andN B is the number of blocks.
As illustrated in Figure 2, an example of block-wise learnable channel permutation with NB = 4 is
shown. In this case, the channels of W are partitioned into four blocks, with each block consisting of
5

<!-- page 6 -->

consecutive Cin/4 channels. Pi only affects the channel permutation within thei-th block. Moreover,
compared to full matrix permutation, only the diagonal blocks are learnable, while all other entries
are fixed to zero, which significantly reduces the training overhead.
Given the advantages of the block-wise manner, it is adopted as the default setting for the proposed
learnable channel permutation in the following sections. The full matrix approach can be considered
a special case when the number of blocks is set to one.
4 PermLLM: Pruning with Learnable Channel Permutation
In this section, we will introduce the proposed novel N:M semi-structured pruning framework
that combines the existing one-shot pruning methods [50, 62] with the proposed learnable channel
permutation (LCP), which can further improve the performance of N:M sparse LLMs.
One-shot pruning eliminates weights by applying a predefined, handcrafted weight importance
metric. For example, the weight importance metric proposed by Wanda [ 50] is defined as Sij =
|Wij| · ||X j||2, where W∈R Cout×Cin and X is the input from calibration. Subsequently, the
pruning mask M∈R Cout×Cin is determined to maximize the sum of the retained importance
metrics, which can be formulated as
arg max
M
CoutX
i=0
Cin/MX
k=0
X
(M⊙S) i,kM:(k+1)M ,s.t.∥M i,kM:(k+1)M ∥0 =M−N,(7)
where Mi,kM:(k+1)M is constructed by setting the entries corresponding to the largestM−N values
in Si,kM:(k+1)M to 1, while all other entries are set to 0. This approach achieves N:M sparsity
by ensuring that N out of every M consecutive elements are set to 0, while preserving the most
important weights based on their importance metrics.
With channel permutation, the order of the channels is rearranged, and consequently, the channels in
the importance matrix S are permuted accordingly. The permuted importance matrix is represented
asbS=SP B, where PB is the permutation matrix. As a result, the mask M varies depending on the
specific permutation solution for the permuted weightcW=WP B:
arg max
M
CoutX
i=0
Cin/MX
k=0
X
(M⊙ bS)i,kM:(k+1)M ,s.t.∥M i,kM:(k+1)M ∥0 =M−N.(8)
However, the non-differentiability of the argmax operation hinders gradient backpropagation, render-
ing it unsuitable for gradient-based learning frameworks. To address this, STE [5] is employed to
approximate the gradients during the backward pass. Specifically, while the forward pass uses the
non-differentiable argmax operation to obtain a discrete hard mask M, the backward pass introduces
a soft maskcMto enable gradient computation. The soft mask is defined as:
cMi,kM:(k+1)M =Softmax(bSi,kM:(k+1)M ),(9)
where the softmax function provides a continuous and differentiable approximation. This approach
allows the forward pass to retain the discrete selection behavior of argmax, while the backward pass
leverages the smooth and differentiable properties ofsoftmaxto compute gradients effectively.
Existing channel permutation methods [46, 62] are primarily designed to find the optimal permutation
matrixP ∗ and the corresponding maskM ∗ that maximize the sum of retained importance score, as
defined in Equation (8). However, the handcrafted quality metric used to evaluate channel permutation
solutions often fails to accurately reflect the true effectiveness of the permutation, potentially leading
to suboptimal outcomes or even worse performance as illustrated in Figure 1.
To address the aforementioned issue, PermLLM aims to directly minimize the output discrepancy
between the dense model and the sparse N:M model by incorporating learnable channel permutations.
Specifically, we utilize a cosine similarity loss to encourage alignment between the outputs of the
two models, which is defined as:
Lcosine(y, ey) = 1− y· ey
||y|| · ||ey|| ,(10)
6

<!-- page 7 -->

whereyand eyrepresent the outputs of the original dense model and the sparse N:M model.
During the proposed post-training pruning process, only Wi
P for each permutation matrix PB is
learnable, while all weight matrices remain fixed, as illustrated in Figure 2. Additionally, each mask
M is directly obtained from Equation (8), with its values dynamically updated based on changes in
PB. By leveraging the proposed relaxation and gradient approximation techniques, the optimization
of each PB is effectively guided toward solutions that maximize the preservation of the dense model’s
performance while adhering to the N:M sparsity constraint.
After training, weightWwill be permuted and pruned by
cW′ =M ∗ ⊙(WP ∗
B),(11)
where P∗
B denotes the learned channel permutation matrix and M∗ is the corresponding pruning
mask.
Notably, the channels of the input activations must also be permuted to align with the channel order
of the weight matrix. It can be accomplished by permuting the output channels of the preceding layer.
Let P∗
l,B denote the permutation matrix for the current layer, and letcW′
l−1 represent the permuted
and pruned weight matrix of the preceding layer. The row of cW′
l−1 should be reordered for input
activation permutation of its succeeding layer, which can be expressed as:
cW′′
l−1 =P ∗
l,BcW′
l−1.(12)
Since it is a row-wise operation, it preserves the N:M sparsity ofcW′
l−1. To further reduce the runtime
overhead introduced by channel permutations, we developed a customized CUDA kernel specifically
for the channel permutation operation. Experimental results evaluated on LLaMA-2 7B demonstrate
that this kernel achieves an average speedup of 84× compared to the Pytorch implementation, thereby
making pruning with channel permutations significantly more practical.
5 Experiments
5.1 Setups
We compare with three baselines in N:M sparsity, especially 2:4 sparsity: SparseGPT [15], Wanda [50]
and RIA [62]. Wanda/RIA-CP enables channel permutation for N:M sparsity introduced in RIA.
PermLLMW anda/RIA indicates that Wanda or RIA is employed as the pruning metric in our
PermLLM framework.
The proposed method is evaluated on various open source representative models: LLaMA 7B-
13B [51], LLaMA-2 7B-13B [52], LLaMA-3.1 8B [19], Qwen-2.5 7B [58], and OPT 6.7B [61]. We
randomly select 128 samples from the C4 dataset [47], each comprising 1024 tokens, to serve as the
calibration data for all evaluated models. We utilize five zero-shot evaluation tasks: HellaSwag [60],
ARC-(Easy and Challenge) [9], OpenBookQA [41] and RTE [53] from lm-evaluation-harness [18]
and one language modeling dataset: Wikitext2 [40] to evaluate the performance of the sparse models.
We implement PermLLM with Pytorch [ 43] and HuggingFace Transformers library [ 56]. The
experiments of PermLLM are conducted on A100 GPUs. We employ N:M semi-structured pruning
for linear layers, skipping the initial embedding layer and the final classification head. These linear
layers constitute approximately 99% of the total parameters in LLMs.
For the proposed PermLLM framework, we utilize AdamW [33] as the optimizer, with the learning
rate set from {1e-3, 5e-3} for all models. The iteration of Sinkhorn normalization is 5. The
temperature τ is linearly decayed from 1 to 0.1 to control the hardness of the soft permutation matrix
in Equation (5). The block size for block-wise learnable channel permutation is set to 64, as it offers a
balanced trade-off between performance and efficiency. Specifically, a block size of 64 is considered
a more practical choice, as increasing the block size to 128 results in a twofold increase in runtime.
This is because a larger block size not only raises computational complexity but also requires more
iterations to achieve convergence due to the significantly expanded solution space. The pruning
duration is about 2.5 hours for the 7B model with 4 GPUs and 5.5 hours for the 13B model with
8 GPUs, which is considered acceptable given the extremely large-scale nature of pruning-aware
permutation problem. More efficient implementation scheme of PermLLM is discussed in Section A.
7

<!-- page 8 -->

Table 1: 2:4 semi-structured pruning results on Wikitext2 with perplexity as the evaluation metric.
Method OPT 6.7B LLaMA 7B LLaMA 13B LLaMA-2 7B LLaMA-2 13B LLaMA-3.1 8B Qwen-2.5 7B
Dense 10.86 5.68 5.09 5.47 4.89 6.24 7.74
SparseGPT 14.33 11.19 9.17 11.12 9.03 16.62 14.34
Wanda 16.29 11.59 9.60 12.16 9.05 23.42 24.44
Wanda+CP 15.28 11.07 8.69 11.00 8.51 21.09 18.76
PermLLMW anda 14.27 9.41 8.06 9.39 8.20 14.03 13.58
RIA 15.93 11.14 8.96 11.30 8.51 22.62 22.67
RIA+CP 15.13 10.99 8.15 10.26 8.08 19.80 17.58
PermLLMRIA 14.23 9.95 7.81 9.60 7.97 15.79 15.93
Table 2: Zero-shot performance of 2:4 sparse models.
Model Method Weight Update HellaSwag ARC_E ARC_C OBQA RTE Average
OPT 6.7B
Dense - 50.46 65.49 30.12 26.80 55.23 45.62
SparseGPT!43.4060.8226.6224.4052.71 41.59
Wanda%41.56 57.62 24.83 23.00 53.43 40.09
Wanda+CP%42.87 59.51 26.02 22.00 52.71 40.62
PermLLMW anda % 44.27 59.43 27.22 24.00 54.15 41.81
LLaMA 7B
Dense - 56.95 75.38 41.89 34.80 65.34 54.87
SparseGPT!43.55 61.78 27.90 22.80 58.12 42.83
Wanda%42.33 61.57 28.07 23.60 51.26 41.37
Wanda+CP%44.21 63.51 29.86 24.00 58.12 43.94
PermLLMW anda % 47.03 63.30 30.55 25.00 62.45 45.67
LLaMA-2 7B
Dense - 57.13 76.30 43.26 31.60 62.45 54.15
SparseGPT!44.11 64.1431.3124.20 58.84 44.52
Wanda%41.59 61.74 30.20 24.00 53.07 42.12
Wanda+CP%43.40 64.69 30.03 26.00 53.07 43.44
PermLLMW anda % 46.60 65.49 31.14 26.20 63.54 46.59
LLaMA-3.1 8B
Dense - 60.06 81.48 51.28 33.40 70.04 59.25
SparseGPT!44.2563.7630.5524.20 53.7943.31
Wanda%38.45 58.00 26.37 19.40 52.35 38.91
Wanda+CP%39.32 62.25 28.92 20.40 52.71 40.72
PermLLMW anda % 45.33 62.58 30.97 24.00 53.79 43.33
Qwen-2.5 7B
Dense - 58.79 79.56 46.08 33.00 76.90 58.87
SparseGPT!46.2071.1337.46 26.00 75.45 51.25
Wanda%40.60 67.17 33.45 25.40 72.92 47.91
Wanda+CP%42.92 70.50 36.09 25.20 72.20 49.38
PermLLMW anda % 47.30 70.58 38.13 27.60 77.26 52.17
5.2 N:M Semi-structured Pruning for LLMs
Language Modeling.In Table 1, we evaluate the language modeling performance of the 2:4 sparse
models on Wikitext2. Perplexity is used as the evaluation metric, with lower values indicating better
language modeling performance. SparseGPT updates the remaining unpruned weights during pruning
to compensate the pruning error. Other pruning methods, including our proposed PermLLM, do not
modify weight values.
Empirical results demonstrate that channel permutations effectively mitigate performance degradation
in pruned models. However, existing channel permutation algorithms rely on handcrafted heuristic
metrics to generate permutations, often yielding suboptimal solutions. In contrast, PermLLM
employs end-to-end learnable optimization to derive superior permutations by directly minimizing
the performance gap between the dense and pruned models. Compared to SparseGPT, both Wanda
and RIA initially demonstrate superior performance on LLaMA and LLaMA-2. The proposed
PermLLM framework further unlocks their potential. On the other hand, for other models, Wanda and
RIA underperform relative to SparseGPT, even with channel permutations. Specifically, significant
performance degradations are observed in LLaMA-3.1 and Qwen-2.5 even using Wanda+CP and
RIA+CP. However, with the incorporation of learnable channel permutations,PermLLM surpasses
8

<!-- page 9 -->

Table 3: Runtime for the different layers and channel permutations in LLaMA-2 7B using 2048
tokens.
Method Q/K/V/O_proj Up/Gate_proj Down_proj CP
Dense 1.513ms 2.607ms 2.614ms -
2:4 sparsity + CP 0.927ms 1.526ms 1.535ms 0.039ms
Speedup 1.632×1.708×1.703×-
Table 4: Evaluation onPermLLMwanda for LLaMA-2 7B with different iteration number of Sinkhorn
normalization.
Model # of Iter. HellaSwag ARC_E ARC_C OBQA RTE Average Wikitext2
Qwen-2.5 7B 0 45.2864.6529.86 21.2053.7942.96 14.12
545.3362.5830.97 24.00 53.79 43.33 14.03
LLaMA-3.1 8B 0 45.9371.0037.88 25.40 65.70 49.18 14.43
547.3070.5838.13 27.60 77.26 52.17 13.58
SparseGPT due to its accurate and model-wise optimization, demonstrating the effectiveness and
superiority of the proposed framework.
Zero-shot Performance.In Table 2, we report the zero-shot performance of the 2:4 sparse models on
five evaluation tasks. The average accuracy across all tasks is presented in the last column. PermLLM
significantly enhances the effectiveness of channel permutations for pruning, outperforming existing
methods on the majority of tasks and achieving the highest average accuracy. This highlights the
significant potential of channel permutation as an effective tool for semi-structured pruning. We
also evaluate PermLLM for 4:8 sparsity on LLaMA-2 7B in Table 8, which shows PermLLM is not
limited to 2:4 sparsity.
Inference Speedup. Inference runtime evaluation is crucial to validate the practicability of the
proposed framework. In Table 3, we report the runtime speedup of 2:4 sparse LLaMA-2 model using
a batch of 2048 tokens following SparseGPT and RIA. The customized CUDA kernel of channel
permutation reduces the total runtime from 3.288ms to 0.039ms, providing 84× speedup compared to
Pytorch implementation. Thus, the overhead of channel permutations is minimal with the customized
CUDA kernel. The overall acceleration across all linear layers, even with channel permutations, is
approximately 1.67×.
5.3 Ablation Study
We conduct an ablation study on the relaxation of the soft permutation matrix to evaluate its impact
on our framework. A larger iteration number in Sinkhorn normalization allows the soft permutation
matrix to converge more closely to a doubly stochastic matrix (DSM). By default, we set the iteration
number of Sinkhorn normalization to 5, which generally yields satisfactory performance. In Table 4,
we evaluate the pruning performance of PermLLMWanda under different Sinkhorn normalization
iterations. When the iteration number is set to 0, the soft permutation matrix deviates the most from
a DSM. The results demonstrate the benefits of using a DSM as the soft permutation matrix for
permutation learning. It helps to enhance the learning process by providing a more structured and
meaningful representation.
In Table 5, we evaluate PermLLMwanda on the LLaMA-2 7B model using different calibration
datasets: Pile [ 17], Wikitext2 [40], and C4 [ 47]. Each dataset consists of 128 randomly selected
samples. The results demonstrate that the learned permutation performs consistently well across
different datasets, which indicates robustness of PermLLM.
Additionally, we conduct experiments to analyze the trade-off between performance and training cost
under varying block sizes. As shown in Table 6, larger block sizes provide a greater optimization
space. However, this increased space comes at the cost of longer exploration and convergence times.
We select a block size of 64 as the default, as it strikes a good balance between pruning performance
and training efficiency.
9

<!-- page 10 -->

Table 5: Evaluation on PermLLMwanda for LlaMA-2 7B with different calibration dataset.
Dataset HellaSwag ARC_E ARC_C OBQA RTE Average Wikitext2
Pile 45.83 64.31 32.08 26.60 54.87 44.74 8.96
Wikitext2 45.42 66.41 32.34 25.80 53.07 44.61 8.31
C4 46.60 65.49 31.14 26.20 63.54 46.59 9.39
Table 6: Evaluation on PermLLMwanda for LlaMA-2 7B with different block size.
Block size HellaSwag ARC_E ARC_C OBQA RTE Average Wikitext2 Time
32 46.13 64.39 29.69 24.60 53.07 43.58 9.50 2h
64 46.60 65.49 31.14 26.20 63.54 46.59 9.39 2.5h
128 46.47 66.08 32.08 27.40 64.43 47.09 9.07 6h
6 Conclusion
This paper introduces PermLLM, a novel pruning framework leveraging learnable channel permuta-
tions (LCP) to optimize N:M sparsity in large language models. By minimizing pruning errors through
end-to-end optimization, PermLLM significantly enhances the performance of N:M semi-structured
pruning. Experimental results validate its superiority over existing methods.
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report.arXiv preprint, 2023.
[2] Ryan Prescott Adams and Richard S Zemel. Ranking via sinkhorn propagation.arXiv preprint,
2011.
[3] Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and
James Hensman. Slicegpt: Compress large language models by deleting rows and columns. In
International Conference on Learning Representations (ICLR), 2024.
[4] Guangji Bai, Yijiang Li, Chen Ling, Kibaek Kim, and Liang Zhao. Sparsellm: Towards
global pruning of pre-trained language models. InAnnual Conference on Neural Information
Processing Systems (NeurIPS), 2024.
[5] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients
through stochastic neurons for conditional computation.arXiv preprint, 2013.
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners.Annual Conference on Neural Information Processing Systems (NeurIPS),
33:1877–1901, 2020.
[7] Xiaodong Chen, Yuxuan Hu, and Jing Zhang. Compressing large language models by stream-
lining the unimportant layer.arXiv preprint, 2024.
[8] Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. A survey on deep neural network
pruning: Taxonomy, comparison, analysis, and recommendations.IEEE Transactions on
Pattern Analysis and Machine Intelligence (TPAMI), 2024.
[9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge.arXiv preprint, 2018.
[10] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8 () 8-bit matrix
multiplication for transformers at scale. InAnnual Conference on Neural Information Processing
Systems (NeurIPS), pages 30318–30332, 2022.
10

<!-- page 11 -->

[11] Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen
Chu. Pruner-zero: Evolving symbolic pruning metric from scratch for large language models.
InInternational Conference on Machine Learning (ICML), pages 11346–11374, 2024.
[12] Patrick Emami and Sanjay Ranka. Learning permutations with sinkhorn policy gradient.arXiv
preprint arXiv:1805.07010, 2018.
[13] Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. Rigging the
lottery: Making all tickets winners. InInternational Conference on Machine Learning (ICML),
pages 2943–2952. PMLR, 2020.
[14] Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo
Molchanov, and Xinchao Wang. Maskllm: Learnable semi-structured sparsity for large language
models. InAnnual Conference on Neural Information Processing Systems (NeurIPS), 2024.
[15] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned
in one-shot. InInternational Conference on Machine Learning (ICML), pages 10323–10337,
2023.
[16] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training
quantization for generative pre-trained transformers.arXiv preprint, 2022.
[17] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason
Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse
text for language modeling.arXiv preprint, 2020.
[18] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas
Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron,
Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework
for few-shot language model evaluation, 12 2023.
[19] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama
3 herd of models.arXiv preprint, 2024.
[20] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural
networks with pruning, trained quantization and huffman coding.arXiv preprint, 2015.
[21] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for
efficient neural network.Annual Conference on Neural Information Processing Systems (NIPS),
28, 2015.
[22] Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network
pruning. InIEEE International Conference on Neural Networks (ICNN), pages 293–299. IEEE,
1993.
[23] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural
networks. InIEEE International Conference on Computer Vision (ICCV), pages 1389–1397,
2017.
[24] Itay Hubara, Brian Chmiel, Moshe Island, Ron Banner, Joseph Naor, and Daniel Soudry.
Accelerated sparse neural training: A provable and efficient method to find n: m transposable
masks.Annual Conference on Neural Information Processing Systems (NeurIPS), 34:21099–
21111, 2021.
[25] Yu Ji, Ling Liang, Lei Deng, Youyang Zhang, Youhui Zhang, and Yuan Xie. Tetris: Tile-
matching the tremendous irregular sparsity.Annual Conference on Neural Information Process-
ing Systems (NeurIPS), 31, 2018.
[26] Sheng-Chun Kao, Amir Yazdanbakhsh, Suvinay Subramanian, Shivani Agrawal, Utku Evci,
and Tushar Krishna. Training recipe for n: M structured sparsity with decaying pruning mask.
arXiv preprint, 2022.
11

<!-- page 12 -->

[27] Harold W Kuhn. The hungarian method for the assignment problem.Naval research logistics
quarterly, 2(1-2):83–97, 1955.
[28] Yann LeCun, John Denker, and Sara Solla. Optimal brain damage.Annual Conference on
Neural Information Processing Systems (NIPS), 2, 1989.
[29] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip HS Torr. Snip: Single-shot network
pruning based on connection sensitivity.arXiv preprint, 2018.
[30] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song,
Zhenan Sun, and Ying Wei. Duquant: Distributing outliers via dual transformation makes
stronger quantized llms. InAnnual Conference on Neural Information Processing Systems
(NeurIPS), 2024.
[31] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan
Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization
for on-device llm compression and acceleration.Machine Learning and Systems (MLSys),
6:87–100, 2024.
[32] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Zahra Atashgahi, Lu Yin, Huanyu Kou, Li Shen,
Mykola Pechenizkiy, Zhangyang Wang, and Decebal Constantin Mocanu. Sparse training via
boosting pruning plasticity with neuroregeneration.Annual Conference on Neural Information
Processing Systems (NeurIPS), 34:9908–9922, 2021.
[33] I Loshchilov. Decoupled weight decay regularization.arXiv preprint, 2017.
[34] Yucheng Lu, Shivani Agrawal, Suvinay Subramanian, Oleg Rybakov, Christopher De Sa,
and Amir Yazdanbakhsh. Step: learning n: M structured sparsity masks from scratch with
precondition. InInternational Conference on Machine Learning (ICML), pages 22812–22824.
PMLR, 2023.
[35] Jiancheng Lyu, Shuai Zhang, Yingyong Qi, and Jack Xin. Autoshufflenet: Learning permutation
matrices via an exact lipschitz continuous penalty in deep convolutional neural networks. In
ACM International Conference on Knowledge Discovery and Data Mining (KDD), pages
608–616, 2020.
[36] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large
language models.Annual Conference on Neural Information Processing Systems (NeurIPS),
36:21702–21720, 2023.
[37] Mohit Mahajan, Wen-Mei Hwu, and Rakesh Nagi. Determining optimal channel partition for 2:
4 fine grained structured sparsity.Optimization Letters, 18(9):2079–2090, 2024.
[38] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han,
and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you
expect.arXiv preprint, 2024.
[39] Gonzalo Mena, David Belanger, Scott Linderman, and Jasper Snoek. Learning latent permuta-
tions with gumbel-sinkhorn networks. InInternational Conference on Learning Representations
(ICLR), 2018.
[40] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture
models.arXiv preprint, 2016.
[41] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
electricity? a new dataset for open book question answering.arXiv preprint, 2018.
[42] Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko Stosic, Dusan Stosic, Ganesh Venkatesh,
Chong Yu, and Paulius Micikevicius. Accelerating sparse deep neural networks.arXiv preprint,
2021.
[43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
style, high-performance deep learning library.Annual Conference on Neural Information
Processing Systems (NeurIPS), 32, 2019.
12

<!-- page 13 -->

[44] Zehua Pei, Hui-Ling Zhen, Xianzhi Yu, Sinno Jialin Pan, Mingxuan Yuan, and Bei Yu. Fusegpt:
Learnable layers fusion of generative pre-trained transformers.arXiv preprint, 2024.
[45] Jeff Pool, Abhishek Sawarkar, and Jay Rodge. Accelerating inference with sparsity us-
ing the nvidia ampere architecture and nvidia tensorrt.NVIDIA Developer Technical
Blog, https://developer . nvidia. com/blog/accelerating-inference-with-sparsityusing-ampere-
and-tensorrt, 2021.
[46] Jeff Pool and Chong Yu. Channel permutations for n: m sparsity.Annual Conference on Neural
Information Processing Systems (NeurIPS), 34:13316–13327, 2021.
[47] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified
text-to-text transformer.Journal of Machine Learning Research (JMLR), 21(140):1–67, 2020.
[48] Richard Sinkhorn. A relationship between arbitrary positive matrices and doubly stochastic
matrices.The annals of mathematical statistics, 35(2):876–879, 1964.
[49] Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, et al. Sleb: Streamlining
llms through redundancy verification and elimination of transformer blocks. InInternational
Conference on Machine Learning (ICML), 2024.
[50] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning
approach for large language models. InInternational Conference on Learning Representations
(ICLR), 2024.
[51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timo-
thée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open
and efficient foundation language models.arXiv preprint, 2023.
[52] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
foundation and fine-tuned chat models.arXiv preprint, 2023.
[53] Alex Wang. Glue: A multi-task benchmark and analysis platform for natural language under-
standing.arXiv preprint, 2018.
[54] Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by
preserving gradient flow.arXiv preprint, 2020.
[55] Zixiao Wang, Jingwei Zhang, Wenqian Zhao, Farzan Farnia, and Bei Yu. Moreaupruner: Robust
pruning of large language models against weight perturbations.arXiv preprint, 2024.
[56] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,
Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain
Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-
art natural language processing. InProceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020.
Association for Computational Linguistics.
[57] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han.
Smoothquant: Accurate and efficient post-training quantization for large language models.
InInternational Conference on Machine Learning (ICML), pages 38087–38099. PMLR, 2023.
[58] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report.arXiv preprint, 2024.
[59] Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu
Sun, Qiang Wu, Jiaxiang Wu, and Bingzhe Wu. Rptq: Reorder-based post-training quantization
for large language models.arXiv preprint, 2023.
[60] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a
machine really finish your sentence?arXiv preprint, 2019.
13

<!-- page 14 -->

[61] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen,
Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained
transformer language models.arXiv preprint, 2022.
[62] Yingtao Zhang, Haoli Bai, Haokun Lin, Jialin Zhao, Lu Hou, and Carlo Vittorio Cannistraci.
Plug-and-play: An efficient post-training pruning method for large language models. In
International Conference on Learning Representations (ICLR), 2024.
[63] Yuxin Zhang, Mingbao Lin, Zhihang Lin, Yiting Luo, Ke Li, Fei Chao, Yongjian Wu, and
Rongrong Ji. Learning best combination for efficient n: M sparsity.Annual Conference on
Neural Information Processing Systems (NeurIPS), 35:941–953, 2022.
[64] Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, and
Hongsheng Li. Learning n: M fine-grained structured sparse neural networks from scratch. In
International Conference on Learning Representations (ICLR), 2021.
[65] Lancheng Zou, Shuo Yin, Mingjun Li, Mingzi Wang, Chen Bai, Wenqian Zhao, and Bei
Yu. Oiso: Outlier-isolated data format for low-bit large language model quantization.IEEE
Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2025.
[66] Lancheng Zou, Wenqian Zhao, Shuo Yin, Chen Bai, Qi Sun, and Bei Yu. BiE: Bi-exponent
block floating-point for large language models quantization. InInternational Conference on
Machine Learning (ICML), volume 235, pages 62978–62992. PMLR, 2024.
14

<!-- page 15 -->

A Implementations
Hyperparameters Configurations.The learning rate is set from {1e-3, 5e-3}. Specifically, we use
1e-3 for PermLLMW anda and 5e-3 for PermLLMRIA. The iteration of Sinkhorn normalization is 5.
The temperature τ is linearly decayed from 1 to 0.1 to control the hardness of the soft permutation
matrix in Equation (5). The block size for block-wise learnable channel permutation is set to 64, as it
offers a balanced trade-off between performance and efficiency. We use 50 iterations for permutation
learning.
More Efficient Implementation.PermLLM serves as an effective plugin to improve performance
of existing zero-shot pruning methods. As observed in other studies, different layers have varying
impacts on the output. To further enhance the efficiency of PermLLM, learnable channel permutation
modules can be inserted into only a subset of layers, while the traditional channel permutation method
is applied to the remaining layers. For instance, we apply learnable channel permutations only to the
last six decoder layers of the LLaMA-2-7B model. In this case, only a single GPU is required for
permutation learning, reducing the runtime to 0.4 hours, which is similar to the runtime of traditional
channel permutation method.
The experimental results are shown in Table 7. Although partial PermLLM does not match the perfor-
mance of full PermLLM due to its limited optimization space, it still provides notable improvements
over traditional channel permutation methods. This approach also represents a balanced trade-off
between performance and efficiency, making it particularly suitable for scenarios with relatively
limited computational resources.
Table 7: Experimental results on LLaMA-2-7B with partial PermLLM. We highlight the top-2 results.
Method HellaSwag ARC_E ARC_C OBQA RTE Average Wikitext2
RIA+CP 42.8664.6930.29 24.4054.8743.42 10.26
PermLLMRIA (partial) 44.4664.1031.74 24.8053.7943.78 10.10
PermLLMRIA (full) 45.15 64.77 32.25 24.80 54.51 44.30 9.60
B PermLLM for 4:8 Sparsity
In Table 8, we present the detailed results about the 4:8 sparsity on LLaMA-2 model with different
pruning methods. The experimental results demonstrate that PermLLM is not limited to 2:4 sparsity
and can still outperform traditional method for 4:8 sparsity.
Table 8: Evaluation on 4:8 sparse LLaMA-2-7B with different pruning methods.
Method Weight Update HellaSwag ARC_E ARC_C OBQA RTE Average Wikitext2
Dense - 57.13 76.30 43.26 31.60 62.45 54.15 5.47
SparseGPT !48.77 67.68 34.81 26.20 53.79 46.25 8.56
Wanda %46.87 66.92 34.04 26.40 54.87 45.82 8.63
Wanda+CP %48.6170.6235.15 28.6055.2347.64 8.26
PermLLMW anda % 49.02 70.20 36.35 29.40 54.87 47.97 7.96
C Visualization of Mask
Figure 3 illustrates the masks of layer.30.down_proj in the pruned LLaMA-2-7B by different methods.
For methods involving channel permutations (e.g., RIA+CP and PermLLMRIA), the channels are
permuted back to their original order for better comparison. We extract a 128×128 portion of the mask
(i.e., mask[:128, :128]) for better visualization. It’s observed that the retained weights differ between
the previous channel permutation method and our proposed learnable channel permutation. This
is because we utilize different strategies: previous one aims at maximizing the sum of the retained
importance metrics and PermLLM is to minimize the output discrepancy between dense model and
the pruned model.
15

<!-- page 16 -->

(a) Wanda
 (b) Wanda+CP
 (c) PermLLMW anda
(d) RIA
 (e) RIA+CP
 (f) PermLLMRIA
Figure 3: Visualization of mask obtained by different pruning methods. The blue part means the
pruned weights and the white part is the retained weights
D Limitations
This paper introduces a innovative learnable channel permutation method to enhance semi-structured
pruning for the first time. Although the method is tailored for semi-structured pruning, channel
permutation or channel reordering has also been shown to be beneficial in other areas, such as
quantization [30, 59]. This suggests that the broader applicability of the proposed approach to tasks
beyond pruning, such as optimizing quantization performance, remains an open area for future
exploration. Moreover, while the proposed block-wise channel permutation scheme significantly
reduces training overhead compared to the full matrix scheme, the training of PermLLM still requires
more computational resources compared to traditional channel permutation methods. Enhancing the
training efficiency for pruning-aware permutation learning remains an important direction for future
research.
E Broader Impacts
The proposed learnable channel permutation for N:M sparsity is not expected to have any negative
societal impacts. Instead, it has the potential to advance the field of machine learning, particularly in
the area of model compression.
16
