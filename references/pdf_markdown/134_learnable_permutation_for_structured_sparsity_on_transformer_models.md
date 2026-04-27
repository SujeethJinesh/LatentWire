# references/134_learnable_permutation_for_structured_sparsity_on_transformer_models.pdf

<!-- page 1 -->

Learnable Permutation for Structured Sparsity on Transformer Models
Zekai Li*, Ji Liu*, Guanchen Li, Yixing Xu, Ziqiong Liu, Xuanwu Yin, Dong Li, Emad Barsoum
Advanced Micro Devices, Inc.
{Zekai.Li, Ji.Liu, Guanchen.Li, Yixing.Xu, Ziqiong.Liu, Xuanwu.Yin, d.li, Emad.Barsoum}@amd.com
Abstract
Structured sparsity has emerged as a popular model prun-
ing technique, widely adopted in various architectures, in-
cluding CNNs, Transformer models, and especially large lan-
guage models (LLMs) in recent years. A promising direction
to further improve post-pruning performance is weight per-
mutation, which reorders model weights into patterns more
amenable to pruning. However, the exponential growth of the
permutation search space with the scale of Transformer ar-
chitectures forces most methods to rely on greedy or heuristic
algorithms, limiting the effectiveness of reordering.
In this work, we propose a novelend-to-end learnableper-
mutation framework. Our method introduces a learnable per-
mutation cost matrix to quantify the cost of swapping any
two input channels of a given weight matrix, a differentiable
bipartite matching solver to obtain the optimal binary permu-
tation matrix given a cost matrix, and a sparsity optimization
loss function to directly optimize the permutation operator.
We extensively validate our approach on vision and language
Transformers, demonstrating that our method achieves state-
of-the-art permutation results for structured sparsity.
Introduction
Transformer architectures (Vaswani et al. 2017) have
achieved remarkable success across diverse AI applications,
including vision models such as ViT (Yuan et al. 2021),
DETR (Carion et al. 2020), DiT (Peebles and Xie 2023),
and large language models (LLMs) such as GPT (Floridi and
Chiriatti 2020), LLaMA (Touvron et al. 2023), Qwen (Bai
et al. 2023), and DeepSeek (Guo et al. 2025). Their
strong representational capacity and generalizability have
made Transformers the preferred architecture for founda-
tion models. However, deploying these large-scale models
on resource-constrained hardware remains challenging, as
inference cost grows rapidly with increasing model size. To
this end, structured pruning under N:M sparsity constraints,
which requires that only N out of every group of M consec-
utive weights remain nonzero, has emerged as an efficient
solution (Bengio, L ´eonard, and Courville 2013; Han et al.
2015; Sun et al. 2023a; Fang et al. 2024). Its regular struc-
ture enables significant parameter reduction while maintain-
ing hardware compatibility, as demonstrated by recent GPUs
*These authors contributed equally.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
that accelerate structured sparse patterns such as 2:4 spar-
sity (Zhou et al. 2021).
Despite their effectiveness, structured pruning methods
such as Wanda (Sun et al. 2023b), SparseGPT (Frantar and
Alistarh 2023), and PrunerZero (Dong et al. 2024) still de-
grade accuracy due to a fundamental mismatch between
rigid sparsity patterns and inherent weight distributions.
Standard N:M pruning preserves only the top-N weights
within fixed-size groups, irrespective of actual weight im-
portance. Since Transformer channels are initially ordered
arbitrarily, important weights can easily be pruned uninten-
tionally. To mitigate this, channel-wise weight permutation
methods reorder weight matrices before pruning to better
align weight saliency with sparsity patterns, significantly re-
ducing accuracy loss (Pool and Yu 2021).
However, current permutation approaches mainly rely on
heuristic or greedy algorithms (Zhang et al. 2023), which op-
timize local importance scores rather than directly improv-
ing end-to-end task performance. Furthermore, these heuris-
tics neglect global coordination and are computationally ex-
pensive, typically employing costly linear sum assignment
or searching algorithms (Pool and Yu 2021). Such complex-
ity becomes impractical for large Transformer models with
numerous layers and channels, limiting the efficiency and
quality of resulting permutations.
To address these shortcomings, we propose a fully learn-
able permutation framework to jointly optimize channel per-
mutation and structured pruning in an end-to-end manner.
However, there are two significant challenges.First, the
permutation operation is inherently discrete and non-
differentiable, complicating its integration with gradient-
based training.Second, existing importance heuristics are
insufficient for guiding permutation decisions that af-
fect overall model performance.This calls for an end-to-
end optimization framework that directly links permutation
learning with task-level objectives. In response, our frame-
work introduces three key innovations:
• Alearnable permutation cost matrixthat explicitly
quantifies the cost of swapping any two input channels
of a given weight matrix.
• To address the non-differentiability of discrete permuta-
tion, we design adifferentiable approximation of bi-
partite matchingguided by the learnable cost matrix,
arXiv:2601.22980v1  [cs.LG]  30 Jan 2026

<!-- page 2 -->

enabling efficient and accurate binary permutation ma-
trix learning with minimal computational overhead.
• Anend-to-end sparsity optimization lossfunction is
proposed to jointly guide the optimization of the per-
mutation operator, achieving a fine balance between
task-specific performance and alignment with the dense
teacher model through knowledge distillation.
Through end-to-end learning, the proposed framework
derives a dedicated permutation matrix for each weight ten-
sor, which is then multiplied with the original weights to
produce reordered weights that align more naturally with
the target sparsity pattern. We apply this approach to both
vision and language models, including ViT, LLM, and VLM
backbones, and conduct extensive experiments to validate
its effectiveness. Experimental results demonstrate that our
framework achieves state-of-the-art structured sparsity with
significantly reduced accuracy degradation, outperforming
traditional greedy baselines on a variety of benchmarks.
Related Work
Model Pruning.Model pruning compresses a pre-trained
model by reducing its parameter count, memory usage,
and computational footprint (Li et al. 2025). Contemporary
pruning approaches can be broadly categorized into three
types.Unstructured pruningeliminates individual weight
elements, offering fine-grained sparsity control. However,
the resulting irregular sparsity patterns pose challenges for
hardware acceleration, often requiring extremely high spar-
sity levels to achieve meaningful speedup (Han, Mao, and
Dally 2016; Han et al. 2015; Liao et al. 2025).Structural
pruningremoves entire filters, channels, or layers, producing
regular sparsity patterns that are hardware friendly. While
it simplifies deployment, this coarse-grained pruning often
leads to considerable accuracy degradation and typically de-
mands retraining to recover performance (Ma, Fang, and
Wang 2023; Xia et al. 2023; He, Zhang, and Sun 2017).
(Semi-)Structured pruning, or N:M sparsity, enforces a fixed
number of nonzero weights per block, balancing accuracy
preservation with hardware efficiency. It retains much of
the flexibility of unstructured pruning while producing reg-
ular memory layouts suitable for modern accelerators (Pool,
Sawarkar, and Rodge 2021; Pool and Yu 2021; Frantar and
Alistarh 2023).
N:M Sparsity.The N:M sparsity constraint enforces at
most N nonzero values within each block of M weights,
achieving a favorable trade-off between compression and
inference efficiency on sparsity-aware hardware (Pool,
Sawarkar, and Rodge 2021; Pool and Yu 2021; Fang et al.
2024; Hu, Zhu, and Chen 2024). Earlier methods applied
static pruning masks after training, while recent techniques
integrate mask learning into the optimization loop using
continuous relaxations and gradient-based updates (Zhou
et al. 2021; Lu et al. 2023; Fang et al. 2024; Liu et al. 2025).
Approaches such as sparse-refined straight-through estima-
tors further promote the retention of important weights
while maintaining strict adherence to the N:M sparsity con-
straint (Bengio, L ´eonard, and Courville 2013; Han et al.
2015). Beyond mask selection, post pruning weight update
0 1 2 3 4 5 6 7
0 1234 56 7
0
0
1
0
2
0
3
0
4
0
5
0
6
0
7
0
0
0
0
1
0
0
2
0
0
3
0
4
0
56 7
salient non-salient
Permuted
Original
Prune
Prune
High total saliency
Permute
Low total saliency
Figure 1: The channel permutation process enhances the
friendliness of the 2:4 sparsification, making the overall
saliency of the pruning metric more preserved.
methods recover accuracy under the N:M constraint by solv-
ing local reconstruction subproblems via second order up-
dates or constrained quadratic optimization (Frantar and Al-
istarh 2023; Bo ˇza 2024). Our work is orthogonal to these
pruning and weight update techniques. We focus on learn-
ing channel permutations that align saliency with the N:M
mask, which can be integrated seamlessly.
Matrix Permutation for Pruning Optimization.Ma-
trix permutation aims to rearrange weights such that salient
and non-salient values are distributed more uniformly across
pruning groups. This improves alignment with structured
sparsity patterns like N:M, enhancing pruning compatibil-
ity and preserving model performance. Channel permuta-
tion was first introduced in (Pool and Yu 2021), which iden-
tifies an optimal reordering via exhaustive greedy search.
However, such methods become impractical when applied to
large language models due to the computational cost of pro-
cessing high-dimensional weight matrices. The Plug-and-
Play method (Zhang et al. 2024) formulates permutation as a
combinatorial optimization problem and solves it efficiently
using the Hungarian algorithm. However, as a rule-based
method, it does not support end-to-end optimization and in-
curs high cost in large-scale models due to the linear sum
assignment operation over large weight tensors.
In this work, we tackle the large search space of channel
permutations by introducing a learnable permutation mech-
anism for GPT-scale Transformers. Our method enables ef-
fective channel reordering that improves model accuracy un-
der N:M sparsity constraints.
Methods
We begin by introducing preliminaries on channel permu-
tation in the context of optimizing structured N:M sparsity
for Transformers. We then describe our proposed end-to-end
learnable permutation framework, which consists of a learn-
able cost prediction module, a differentiable bipartite match-
ing solver, and optimization objectives. The unified frame-
work facilitates end-to-end optimization of permutation op-
erators for diverse Transformer-based architectures, includ-
ing vision, language, and vision-language models.
Preliminaries
Optimize N:M Sparsity via Channel Permutation.
Channel permutation enhances the compatibility of weight

<!-- page 3 -->

𝑊22
T 𝑊20
T 𝑊21
T
𝑊02
T 𝑊00
T 𝑊01
T
𝑊12
T 𝑊10
T 𝑊11
T
𝐴20 𝐴21 𝐴22
𝐴00 𝐴01 𝐴02
𝐴10 𝐴11 𝐴12
𝐴30 𝐴31 𝐴32
𝐴40 𝐴41 𝐴42
input dim
output dim
input dim
batch dim
𝑂22 𝑂20 𝑂21
𝑂02 𝑂00 𝑂01
𝑂12 𝑂10 𝑂11
𝑂32 𝑂30 𝑂31
𝑂42 𝑂40 𝑂41
𝑊10
T 𝑊11
T 𝑊12
T
𝑊20
T 𝑊21
T 𝑊22
T
𝑊00
T 𝑊01
T 𝑊02
T
𝐴22 𝐴20 𝐴21
𝐴02 𝐴00 𝐴01
𝐴12 𝐴10 𝐴11
𝐴32 𝐴30 𝐴31
𝐴42 𝐴40 𝐴41
input dim
output dim
input dim
batch dim
𝑂20 𝑂21 𝑂22
𝑂00 𝑂01 𝑂02
𝑂10 𝑂11 𝑂12
𝑂30 𝑂31 𝑂32
𝑂40 𝑂41 𝑂42
Layer 𝑖 − 1 Layer 𝑖
Weights Activations Input channel perm. Output channel perm.
Figure 2: Channel permutation for Linear layers. To guar-
antee output consistency, after the input channel of thei-th
layer weight is permuted, the input activation of that layer
should also be permuted accordingly, which can be realized
by permuting the output channel of the previous (i−1)-th
layer weight accordingly.
tensors with structured sparsity patterns such as N:M spar-
sity. As shown in Figure 1, the weight layout often exhibits
uneven saliency, with important weights clustered within
certain groups. This skewed distribution lowers the chance
of retaining key weights under fixed 2:4 sparsity, resulting
in suboptimal pruning with reduced preserved saliency.
By permuting channels before pruning, saliency becomes
more evenly distributed across groups. This increases the
likelihood that each M-element group contains a mix of im-
portant and unimportant weights, allowing structured prun-
ing to retain more informative elements. Channel permuta-
tion thus improves N:M pruning by aligning weight layout
with sparsity constraints.
Channel Permutation for Linear Layers.Applying a
channel permutation to a linear layer’s input dimension re-
quires aligning the permuted weights with its input activa-
tion to preserve output consistency. LetW ⊤
i ∈R din×dout
be the weight matrix of thei-th linear layer, and letP∈
{0,1} din×din be a permutation matrix. Applying input chan-
nel permutation to the weight yields cW⊤
i =PW ⊤
i . To
maintain correct computation, the corresponding activation
Ai ∈R dbatch×din must also be transformed as bAi =A iP⊤,
so that the output remains unchanged (P⊤ =P −1):
bAi+1 = bAicW⊤
i =A iP⊤PW⊤
i =A iW⊤
i =A i+1,(1)
To avoid runtime activation permutation, we propagate
the permutation backward to the output channels of the pre-
ceding(i−1)-th layer ( cWi−1 =W i−1P⊤), as shown in
Figure 2, so that:
bAi =A i−1cWi−1 =A i−1Wi−1P⊤ =A iP⊤.(2)
Channel Permutation for Transformer Layers.Apply-
ing channel permutations in Transformer architectures is
challenging due to the structural coupling within multi-head
self-attention (MHA) and feed-forward networks (FFN).
Unlike sequential linear layers, Transformer blocks use par-
allel projections that share inputs and have interdependent
𝐖𝐪
𝐖𝐤
𝐖𝐯
Multi-Head
Attention
𝐖𝐨
𝐖𝐮𝐩
𝐖𝐠𝐚𝐭𝐞
Act & Product
𝐖𝐝𝐨𝐰𝐧
Inputs
Pre-
Layer
Next
Layer
Attn FFN
In
Out
In
Out
Figure 3: An overview of channel permutation for Trans-
former layers. The alignment of the input channel permuta-
tion of the current layer’s weights to the output dimension of
the previous layer’s weights reflects a structural coupling.
weights, requiring coordinated, structure-aware permuta-
tions, as shown in Figure 3.
Based on the rule that the input channel permutation of
the current layer’s weights will affect the output channel per-
mutation of the previous layer’s weights, the input channel
permutation ofW o in the Transformer model will be ex-
ecuted as a binding toW q,W k, andW v. Similarly, the
input channel permutation ofW down will be executed in a
binding toW up andW gate. A detailed proof can be found
in supplementary materials.
Learnable Channel Permutator
To support semi-structured N:M sparsity constraint in Trans-
former models, we propose a learnable channel permutation
framework that enables end-to-end optimization of permu-
tation operators. As shown in Figure 4, the framework con-
sists of a permutation cost predictor, a differentiable bipar-
tite matching solver, and an optimization training objective.
The cost predictor produces layer-wise cost matrices, which
are used to generate permutation matrices. These reorder the
weights before pruning by an N:M mask generator (actually
structured sparsity pruning method, we use Wanda in this pa-
per). During training, pruned weights are inversely permuted
for loss computation, preserving gradient flow through the
pipeline. Unlike heuristic approaches based on local im-
portance or static ranking, our method learns permutation
jointly across gradient-based end-to-end optimization.
Permutation Cost Predictor.The core of our method is a
learnable permutation cost predictor, which produces a cost
matrix to guide the reordering of channels or features. Given
a weight matrixW∈R dout×din from a weight matrix, the
goal is to construct a permutation matrixP∈ {0,1} din×din
that rearranges the input channels such that the pruned
model aligns more effectively with structured N:M sparsity
constraints.
To enable differentiable learning ofP, we introduce a
real-valued cost matrixC∈R din×din, where each element
Ci,j reflects the cost of assigning the original input channeli
to positionj. Intuitively,Cencodes the pairwise preference
for spatial relocation, integrating both structured sparsity
alignment and semantic preservation objectives. We param-
eterize the cost matrix using learnable parameters. For each
input channel, we implement ad in ×d in learnable param-
eter as the cost predictor. The predictor outputs a normal-

<!-- page 4 -->

Block iWeight
Wn
0 1 0 1
0 0 1 1
0 0 1 1
1 0 1 0
1 0 0 0 0 0 0 00 0 0 0 1 0 0 00 0 1 0 0 0 0 0
0 0 0 0 0 0 1 0
0 1 0 0 0 0 0 0
0 0 0 0 0 1 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 1
N:M Mask Generator
(e.g., Wanda)
Permutation matrix
Permutation
Cost Predictor
Differentiable
Biparatite Matching
Solver
Permuted weight
Permuted N:M Mask
=
Pruned permuted weight
Cost matrix
W r×c
Wcc×c
WPc×c
Original weight
Original
inputIn
Original
outputOn Output O'n
Weight
Wn-1
Block NBlock 1
Weight
Wn
Input x
Output
f(x)
Weight
Wn
CE(f(x), y) + Distil(On, O'n)
E2E Optimization
GT label
y
0 0
0 0
0 0
0 0
Original
inputIn (For training)
1 0 0 0 0 0 0 00 0 0 0 1 0 0 00 0 1 0 0 0 0 0
0 0 0 0 0 0 1 0
0 1 0 0 0 0 0 0
0 0 0 0 0 1 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 1
Inverse permutation
Tunable
Frozen
Figure 4: Overview of our learnable permutation framework. A permutation cost predictor generates cost matrices for each
linear layer, which are converted into permutation matrices via a differentiable bipartite matching solver. The original weights
are permuted accordingly and then sparsified using an N:M mask generator. During training, the pruned weights are inversely
permuted and used for loss computation. The entire process supports end-to-end optimization while maintaining gradient flow
through the binary permutation matrix generated by the differentiable solver.
ized cost matrix, which quantifies the cost of swapping two
input channels. We minimize the cumulative cost of each
cost matrix with our proposed bipartite matching solver.
Our experimental results (Table 1) demonstrate that the pro-
posed permutation cost predictor achieves strong pruning
performance despite its simple design, which is intention-
ally lightweight to minimize additional training overhead.
Furthermore, directly learning full permutation matrices
for large-dimensional weight tensors in Transformer mod-
els is computationally expensive and often unnecessary. To
improve scalability and reduce overhead, we adopt a group-
wise permutation strategy, where the input channels of each
layer are partitioned into non-overlapping groups sizeG,
and a separate permutation is learned within each group.
Differentiable Bipartite Matching Solver.To enable
gradient-based optimization of permutation matrices, we in-
troduce a differentiable bipartite matching solver. Given a
learned cost matrixC∈R N×N , whereC i,j indicates the
cost of mapping theith input channel to thejth output, our
goal is to find a permutation matrixP∈ Pminimizing:
min
P∈P
⟨C,P⟩,(3)
wherePis the set of allN×Npermutation matrices.
SincePis discrete and non-differentiable, we relax the
optimization over the Birkhoff polytopeB N —the convex
hull ofP—which comprises all doubly stochastic matrices:
BN =

P∈R N×N P≥0,P1=1,P ⊤1=1

.(4)
We adopt an entropy-regularized formulation to approx-
imate soft permutations within the Birkhoff polytopeB N ,
solved via Sinkhorn iterations (Mena et al. 2018):
min
P∈BN
⟨C,P⟩+ε
X
i,j
Pij(logP ij −1),(5)
whereεis the temperature parameter controlling the en-
tropy strength. The solution has a closed-form structure
P= Diag(u)KDiag(v), withK= exp(−C/ε), and the
scaling vectorsuandvare iteratively updated (in the log
domain) to ensure numerical stability and convergence.
This soft matching mechanism, known as Sinkhorn-
Pop (Knight 2008; Mena et al. 2018), provides a differen-
tiable and numerically stable approximation of the optimal
permutation. It circumvents the need for non-differentiable
alternatives such as the Hungarian algorithm combined with
straight-through estimation (STE). During training, the tem-
peratureεis gradually annealed to sharpen the relaxed per-
mutation matrix towards discreteness. At inference time, the
final discrete permutation is recovered by solving the origi-
nal assignment problem using the Hungarian algorithm.
End-to-End Optimization Objectives.To jointly opti-
mize channel permutation for structured pruning, we opti-
mize the framework using a composite loss that combines
task-level supervision with intermediate feature alignment.
Specifically, the total objective includes two components:
atask-level cross-entropy lossand alayer-wise distillation
loss, encouraging both strong downstream performance and
internal structural consistency.
The task-level loss directly optimizes the pruned model’s
predictions. Letf perm+prune(x)denote the output of the model
after applying the learned permutation and N:M structured
pruning. The cross-entropy loss is given by:
Ltask = CE(fperm+prune(x), y),(6)

<!-- page 5 -->

whereCE(·)is the cross-entropy loss andyis the ground-
truth label. This objective ensures the learned permutation
contributes to preserving task-level accuracy.
To preserve semantic consistency at the feature level, we
incorporate a layer-wise distillation loss that aligns interme-
diate representations between the original and pruned mod-
els. Leth orig
l andh perm+prune
l denote the output features of the
l-th layer in the original and permuted-pruned models, re-
spectively. The distillation loss is defined as:
Ldistill =
LX
l=1
∥horig
l −h perm+prune
l ∥2
2,(7)
whereLis the number of pruned layers. This loss pro-
motes the retention of task-relevant features despite struc-
tural modifications. And the final training objective com-
bines both losses as:
Ltotal =L task +αL distill,(8)
whereαbalances the two losses. This joint objective pro-
vides both global supervision and local guidance for effec-
tive permutation learning under structured sparsity.
Training and Inference Details
Training.We begin by collecting input feature statistics
for each layer of the pretrained Transformer, which are later
used by the structured pruning method (Wanda) adopted in
our framework. After this preprocessing stage, all Trans-
former weights are frozen, and only the parameters of the
permutation cost predictor remain trainable.
To preserve structural consistency in attention layers, we
impose a synchronized permutation constraint across cou-
pled projection matrices, such asW q,W k, andW v. These
matrices share the same input representation, and apply con-
sistent permutations. By enforcing a shared permutation
across structurally dependent components, our method en-
sures correctness and preserves the potential for real-world
acceleration under structured sparsity.
Inference.At inference time, the learned permutation cost
predictor is frozen. For each group of channels, we can ob-
tain the optimal permutation matrix via our bipartite match-
ing solver. These permutations are applied to reorder the
weights of each layer, followed by the application of N:M
sparsity masks generated by the structured pruning method
(Wanda). The resulting pruned model, with permuted and
sparsified weights, is then used for standard inference.
Experiments
Experimental Setup
Models.We evaluate our framework for structured spar-
sity on several Transformer backbones and task do-
mains. For vision domain, we selectViT-Base/16and
ViT-Large/14(Dosovitskiy et al. 2020) for experiments.
For language domain, we employLLaMA-3.2-1B(Dubey
et al. 2024) andLLaMA-2-7B(Touvron et al. 2023) to rep-
resent the common small and large language models. For
multimodal domain, we choseQwen2.5-VL-3B(Bai et al.
2025) as the object of study.
Model Sparsity Method Top-1 (%) Top-5 (%)
ViT-Base/16
0% Dense 79.1 94.1
2:4
CP 66.2 86.4Wanda 65.8 86.4RIA 66.6 86.6Ours(Wanda)67.9 87.9
4:8
CP 66.8 88.2Wanda 66.4 86.6RIA 71.4 89.9Ours(Wanda)71.8 90.2
ViT-Large/14
0% Dense 85.2 97.9
2:4
CP 79.5 94.7Wanda 79.2 95.2RIA 79.6 95.1Ours(Wanda)80.7 95.8
4:8
CP 82.0 96.4Wanda 82.0 96.4RIA 82.3 96.4Ours(Wanda)82.7 96.5
Table 1: Performance of different approaches on ImageNet
after pruning ViT-Base/16 and ViT-Large/14 to the 2:4 and
4:8 structured pattern (50 % non-zero weights).
Datasets.For the vision Transformers, we use the canon-
icalImageNet-1Kdataset (Deng et al. 2009), which con-
sists of 1.28M training images and 50K validation images.
All models are trained with the official train/validation split
with standard input resolution of224×224. For the lan-
guage models, permutations are learned on the training set of
C4dataset (Raffel et al. 2020) and Alpaca-en dataset (Taori
et al. 2023), which comprises approximately 806MB of
cleaned English text. For the vision-language models, per-
mutations are learned on the training set of Alpaca-en
dataset (Taori et al. 2023) and LLaV A-Instruct dataset (Liu
et al. 2023) dataset, which is a set of GPT-generated multi-
modal instruction-following data.
Baselines.Our method is compared with a variety of clas-
sic and state-of-the-art pruning baselines, including Magni-
tude (Han et al. 2015), Wanda (Sun et al. 2023a), SparseGPT
(Frantar and Alistarh 2023), PrunerZero (Dong et al. 2024),
CP (Pool and Yu 2021), and RIA (Zhang et al. 2023).
Evaluations.For vision evaluations, we use the stan-
dardImageNet-1Kvalidation set with top-1/5 accu-
racy as the main metric. For language models, we mea-
sure perplexity on theWikiText2test set (Merity et al.
2016). To evaluate compression effects across tasks, we
report zero-shot accuracy onARC(Clark et al. 2018),
BoolQ(Christopher et al. 2019),HellaSwag(Zellers
et al. 2019),OpenBookQA(Mihaylov et al. 2018), and
WinoGrande(Sakaguchi et al. 2021), along with 5-shot
accuracy onMMLU(Hendrycks et al. 2020). For multimodal
tasks, we report zero-shot accuracy onMMMU(Yue et al.
2024),MMStar(Chen et al. 2024), andTextVQA(Singh
et al. 2019).
Implementation Details.We use Wanda as our mask gen-
erator by default. Vision models are trained for 20 epochs
on 2 AMD MI250 GPUs with AdamW (weight decay0.01,
base learning rate0.1) under a cosine decay schedule and
no warm-up, which takes around 4 hours. Language models
and vision-language models are trained for 20 and 10 update

<!-- page 6 -->

Model Method Wikitext2Arc-Easy Arc-Challenge BoolQ HellaSwag OpenBookQA WinoGrande MMLUAverage
LLaMA-3.2-1B(2:4)
Dense 9.06 65.36 31.40 63.82 47.73 26.60 60.69 31.19 46.68
Magnitude 4808.42 27.95 19.45 38.50 26.13 11.80 51.78 23.80 28.49
SparseGPT 32.20 45.2920.65 62.1131.9915.20 54.54 24.68 36.35
Wanda 75.76 37.16 18.17 62.05 28.57 12.00 50.20 24.45 33.23
PrunerZero 141.40 36.70 19.28 57.43 27.72 13.40 50.12 25.72 32.91
CP 68.17 38.35 18.14 62.08 28.56 12.40 53.31 23.82 33.81
RIA 72.56 38.12 20.33 61.34 27.12 12.80 52.76 24.31 33.83
Ours(Wanda) 45.32 42.26 21.08 62.11 29.12 15.60 54.85 26.27 35.90
LLaMA-2-7B(2:4)
Dense 5.12 76.30 43.43 77.68 57.14 31.40 69.06 45.84 57.26
Magnitude 52.00 61.87 30.20 59.79 45.42 21.80 60.93 26.87 43.84
SparseGPT 10.30 64.1032.51 71.25 43.35 25.00 67.25 28.56 47.43
Wanda 11.38 62.75 30.38 67.65 41.18 23.60 62.59 27.82 45.14
PrunerZero 12.91 61.20 27.47 66.15 39.43 24.40 61.01 27.41 43.87
CP 10.68 63.32 30.96 66.92 41.32 23.80 63.56 26.51 45.20
RIA 10.52 63.67 31.82 67.13 42.03 23.00 64.13 27.56 45.62
Ours(Wanda) 10.17 64.23 32.00 68.17 43.31 23.60 63.77 28.13 46.17
Table 2: Comparisons of different pruning methods on the LLaMA3.2-1B and LLaMA2-7B language models. Performance
across more sparsity patterns can be found in the supplementary materials.
Sparsity Method MMMU MMStar TextVQA Average
0% Dense 53.1 55.8 79.3 62.7
2:4
Magnitude 34.1 48.7 76.5 53.1
Wanda 37.2 51.2 77.2 55.2
RIA 37.3 51.4 77.1 55.3
Ours(Wanda) 38.1 51.9 77.8 55.9
Table 3: Performance of Qwen2.5-VL-3B under different
pruning methods.
epochs, respectively, with AdamW (lr= 10 −4). Training
takes approximately 10 hours for a 1B model and 40 hours
for a 7B model. The default permutation group numberG
is 4. All runs use native AMP with FP16 precision, gradient
accumulation set to one, and a small Smooth-L1 distillation
term weighted10 −5. The default N:M sparsity is 2:4.
Comparison with State-of-the-Art Methods
Results on Vision Transformers.On the vision side,
we apply structured pruning toViT-Base/16and
ViT-Large/14under 2:4 and 4:8 sparsity, retaining
50% of weights. Our method consistently achieves the
highest top-1 and top-5 accuracy across all settings. For
ViT-Base/16, it reaches 67.9% / 87.9% (top-1 / top-5)
at 2:4 sparsity, outperforming the strong permutation base-
line (RIA) by 1.3 points. At 4:8 sparsity, accuracy improves
to 71.8% / 90.2%, ahead of RIA by 0.4 / 0.3 points. Similar
gains are observed onViT-Large/14, achieving 80.7% /
95.8% at 2:4 and 82.7% / 96.5% at 4:8 sparsity, both surpass-
ing prior methods. These results demonstrate the accuracy-
preserving strength of our approach on Vision Transformers.
Results on Language Transformers.On the lan-
guage side, we evaluate pruning performance on both
LLaMA-3.2-1BandLLaMA-2-7Bmodels across a range
of context modeling and commonsense benchmarks. Our
method—using Wanda pruning followed by a learned chan-
nel permutation—consistently improves downstream perfor-
mance over baseline methods without requiring weight up-
dates. OnLLaMA-3.2-1B, our approach improves the av-
erage accuracy to 35.90%, outperforming Wanda (33.23%),
CP (33.81%), and RIA (33.83%) by 2.1 to 2.7 points. Simi-
Epochs Wikitext2 Arc-Easy Arc-Chall. MMLU Average
0 11.38 62.75 30.38 27.82 40.32
1 10.56 62.88 30.89 27.61 40.46
2 10.31 63.13 31.06 27.96 40.72
5 10.27 64.10 31.91 28.05 41.36
10 10.21 64.2331.88 28.12 41.42
20 10.17 64.23 32.00 28.13 41.46
Table 4: Performance over training epochs on LLaMA-2-7B.
lar trends are observed onLLaMA-2-7B, where our method
reaches 46.17% average accuracy, compared to 45.14%
(Wanda), 45.20% (CP), and 45.62% (RIA). Notably, our
method improves performance on challenging tasks such
asARC-Challenge(e.g., 3.79 points over Wanda) and
maintains strong results onBoolQandWinoGrande.
While methods like SparseGPT achieve higher accuracy due
to weight updates, our approach operates in the same con-
strained setting as Wanda, offering a fair and efficient com-
parison. Moreover, onWikiText2, our method yields bet-
ter perplexity than baselines, demonstrating its ability to pre-
serve language modeling quality.
Results on Multimodal Transformers.As shown in Ta-
ble 3, our method achieves the highest performance on
theMMMUbenchmark using theQwen2.5-VL-3Bmodel.
While baseline methods such as Magnitude, Wanda, and
RIA yield scores of 34.1, 37.2, and 37.3 respectively, our
approach outperforms with a score of 38.1. For Transformer-
based models of various types, the learnable permutation
consistently lifts performance beyond magnitude-only or
rule-based permutation strategies, and does so while pre-
serving the full latency advantage of2:4sparsity.
Training Convergence and Efficiency
To assess convergence, we report performance from epoch
1 to 20 in Table 4. The initial performance is limited (av-
erage 33.23;Wikitext2perplexity 75.76), but improves
markedly after one epoch, reaching 40.47 and 10.56 re-
spectively, reflecting an 86% reduction in perplexity and
indicating fast optimization of the permutator. Subsequent
gains taper off: 40.71 at epoch 2, 41.36 at epoch 5, 41.41 at

<!-- page 7 -->

Number of GroupsTrainable ParametersWikitext2Arc-Easy Arc-Chall. BoolQ HellaSwag OpenBookQA WinoGrande MMLUAverage
1 1.0× 10.11 64.90 32.00 68.01 43.56 23.80 63.93 28.07 46.33
2 0.68× 10.12 64.35 31.83 68.20 43.82 23.60 64.01 28.25 46.30
4 0.41× 10.17 64.23 32.00 68.17 43.31 23.60 63.77 28.13 46.17
8 0.23× 10.25 64.14 31.83 68.20 43.15 23.60 63.85 28.15 46.13
16 0.12× 10.63 63.59 31.14 67.92 42.89 23.00 63.69 27.67 45.71
Table 5: Performance of our approach on LLaMA-2-7B with different permutation group numbers.
Model Pruning MethodsWikitext2(Ours/Baseline)Arc-Easy Arc-Chall. BoolQ HellaSwag OpenBookQA WinoGrande MMLUAverage(Ours/Baseline)
LLaMA-2-7B
Magnitude(Ours) 45.82 /52.00 62.88 30.89 61.6845.4722.20 62.12 27.12 44.63 /43.84
Wanda(Ours) 10.17/11.38 64.23 32.00 68.17 43.31 23.60 63.77 28.13 46.17/45.14
RIA(Ours) 10.02/10.52 63.89 32.5168.8442.59 24.00 64.33 29.02 46.45/45.62
AdmmPruner(Ours) 9.56/9.68 69.02 32.68 63.39 45.12 26.00 65.98 29.62 47.40/47.05
Table 6: Performance of different pruning methods when integrated with our approach on LLaMA-2-7B.
Loss Wikitext2Arc-EasyArc-Chall.MMLUAverage
CE Loss 46.15 42.01 20.53 26.12 29.55
Distillation Loss 52.58 40.51 20.12 25.97 28.87
CE+Distillation loss45.32 42.27 21.08 26.27 29.87
Table 7: Ablation of optimization loss on LLaMA3.2-1B.
epoch 10, and 41.44 at epoch 20. Task-level improvements
show a similar pattern.Arc-Easyimproves from 37.16 to
64.22,Arc-Challengefrom 18.17 to 31.96, andMMLU
from 24.45 to 28.13, with most gains observed after the first
epoch, which takes approximately 3 to 4 hours.
We also evaluate other heuristic permutation methods un-
der the same settings, such as the LSA algorithm in RIA and
the search-based approach in CP. These methods typically
require 5 to 10 times longer to converge compared to our
approach. These results suggest that the method converges
efficiently, with nearly all achievable performance obtained
within five epochs, only 0.08 below the final result at epoch
20, while maintaining low per-epoch computational cost.
Ablation Study
Impact of Permutation Group Number.We investigate
how the number of permutation groupsG, defined as the
granularity of partitioning the weight matrix before learning
permutations, affects performance and efficiency. A smaller
Genables more global reordering, potentially leading to bet-
ter permutation masks. A largerGreduces the number of
parameters but constrains the search space. As shown in Ta-
ble 5, we evaluateG∈ {1,2,4,8,16}on LLAMA-2-7B.
The average score decreases only marginally from 46.33 at
G= 1to 46.30, 46.17, 46.13, and 45.71 as the number
of groups increases.Wikitext2perplexity rises slightly
from 10.11 to 10.63. Meanwhile, the number of trainable pa-
rameters is significantly reduced. Considering the reduced
number of trainable parameters, along with the preserva-
tion ofWikitext2perplexity and task accuracy, we adopt
G= 4as the default setting in this paper. Notably,G= 8is
also a viable option when a bigger accuracy drop is accept-
able and computational resources are limited.
Performance over Different Pruning Methods.To as-
sess the generality of our approach, we integrate the learn-
able permutator into four representative pruning baselines:
Magnitude, Wanda, RIA, and AdmmPruner. As shown in
Table 6, our method enhances performance across all set-
tings. Specifically, the average score improves from 43.84
to 44.63 for Magnitude, from 45.14 to 46.17 for Wanda,
and from 45.62 to 46.45 for RIA. The perplexity on
Wikitext2also decreases in every case—for example,
from 52.00 to 45.82 for Magnitude, from 11.38 to 10.17
for Wanda, and from 10.52 to 10.02 for RIA. OnMMLU,
the scores rise from 26.87, 27.82, and 27.56 to 27.12, 28.13,
and 29.02, respectively. Furthermore, the permutator is com-
patible with post-pruning weight updates. When applied to
AdmmPruner, it achieves the highest overall average score
(47.40), the bestMMLUaccuracy (29.62), and the lowest
Wikitext2perplexity (9.56). These results demonstrate
that the permutator is broadly applicable to both pruning
baselines and post-pruning optimization techniques.
Effect of Our Optimization Loss.To evaluate the con-
tributions of our optimization components end-to-end cross
entropy (CE) loss and layer wise distillation we conduct an
ablation study. As shown in Table 7, CE loss alone achieves
an average score of 29.55 and perplexity of 46.15, while dis-
tillation alone yields 28.87. Their combination leads to the
best results: an average score of 29.87 with top performance
on Arc Easy (42.27), Arc Challenge (21.08), and MMLU
(26.27), and a low perplexity of 45.32. These findings con-
firm the complementary benefits of both loss terms.
Conclusion
In this work, we present a novel end-to-end learnable per-
mutation framework to enhance structured sparsity in large-
scale transformer-based models. By introducing a differen-
tiable permutation cost predictor and a bipartite matching
solver, our approach learns optimal weight reorderings that
align better with N:M sparsity constraints. Extensive exper-
iments on vision and language backbones demonstrate that
our method consistently outperforms state-of-the-art base-
lines, offering a powerful and generalizable strategy for
model compression with minimal performance loss.

<!-- page 8 -->

References
Bai, J.; Bai, S.; Chu, Y .; Cui, Z.; Dang, K.; Deng, X.; Fan,
Y .; Ge, W.; Han, Y .; Huang, F.; et al. 2023. Qwen technical
report.arXiv preprint arXiv:2309.16609.
Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang,
K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl
technical report.arXiv preprint arXiv:2502.13923.
Bengio, Y .; L´eonard, N.; and Courville, A. 2013. Estimat-
ing or propagating gradients through stochastic neurons for
conditional computation.arXiv preprint arXiv:1308.3432.
Boˇza, V . 2024. Fast and optimal weight update for pruned
large language models.arXiv preprint arXiv:2401.02938.
Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov,
A.; and Zagoruyko, S. 2020. End-to-end object detection
with transformers. InEuropean conference on computer vi-
sion, 213–229. Springer.
Chen, L.; Li, J.; Dong, X.; Zhang, P.; Zang, Y .; Chen, Z.;
Duan, H.; Wang, J.; Qiao, Y .; Lin, D.; et al. 2024. Are we
on the right way for evaluating large vision-language mod-
els?Advances in Neural Information Processing Systems,
37: 27056–27087.
Christopher, C.; Kenton, L.; Ming-Wei, C.; Tom, K.;
Michael, C.; and Kristina, T. 2019. BoolQ: Exploring
the Surprising Difficulty of Natural Yes/No Questions. In
NAACL.
Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.;
Schoenick, C.; and Tafjord, O. 2018. Think you have Solved
Question Answering? Try ARC, the AI2 Reasoning Chal-
lenge.arXiv:1803.05457v1.
Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-
Fei, L. 2009. Imagenet: A large-scale hierarchical image
database. In2009 IEEE conference on computer vision and
pattern recognition, 248–255. Ieee.
Dong, P.; Li, L.; Tang, Z.; Liu, X.; Pan, X.; Wang, Q.; and
Chu, X. 2024. Pruner-Zero: Evolving Symbolic Pruning
Metric From Scratch for Large Language Models. InInter-
national Conference on Machine Learning, 11346–11374.
PMLR.
Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn,
D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.;
Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16
words: Transformers for image recognition at scale.arXiv
preprint arXiv:2010.11929.
Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.;
Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.;
et al. 2024. The llama 3 herd of models.arXiv e-prints,
arXiv–2407.
Fang, G.; Yin, H.; Muralidharan, S.; Heinrich, G.; Pool, J.;
Kautz, J.; Molchanov, P.; and Wang, X. 2024. Maskllm:
Learnable semi-structured sparsity for large language mod-
els.arXiv preprint arXiv:2409.17481.
Floridi, L.; and Chiriatti, M. 2020. GPT-3: Its nature, scope,
limits, and consequences.Minds and Machines, 30: 681–
694.
Frantar, E.; and Alistarh, D. 2023. Sparsegpt: Massive lan-
guage models can be accurately pruned in one-shot. InInter-
national Conference on Machine Learning, 10323–10337.
PMLR.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning.arXiv preprint arXiv:2501.12948.
Han, S.; Mao, H.; and Dally, W. J. 2016. Deep Compression:
Compressing Deep Neural Networks with Pruning, Trained
Quantization and Huffman Coding.International Confer-
ence on Learning Representations (ICLR).
Han, S.; Pool, J.; Tran, J.; and Dally, W. 2015. Learning
both weights and connections for efficient neural network.
Advances in neural information processing systems, 28.
He, Y .; Zhang, X.; and Sun, J. 2017. Channel Pruning for
Accelerating Very Deep Neural Networks. InInternational
Conference on Computer Vision (ICCV).
Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika,
M.; Song, D.; and Steinhardt, J. 2020. Measuring mas-
sive multitask language understanding.arXiv preprint
arXiv:2009.03300.
Hu, Y .; Zhu, J.; and Chen, J. 2024. S-ste: Continu-
ous pruning function for efficient 2: 4 sparse pre-training.
Advances in Neural Information Processing Systems, 37:
33756–33778.
Knight, P. A. 2008. The Sinkhorn–Knopp algorithm: con-
vergence and applications.SIAM Journal on Matrix Analy-
sis and Applications, 30(1): 261–275.
Li, J.; Xu, Y .; Huang, H.; Yin, X.; Li, D.; Ngai, E. C.; and
Barsoum, E. 2025. Gumiho: A Hybrid Architecture to Prior-
itize Early Tokens in Speculative Decoding.arXiv preprint
arXiv:2503.10135.
Liao, H.; Xu, Y .; He, S.; Li, G.; Yin, X.; Li, D.; Barsoum, E.;
Zhao, J.; and Liu, K. 2025. SparK: Query-Aware Unstruc-
tured Sparsity with Recoverable KV Cache Channel Prun-
ing.arXiv preprint arXiv:2508.15212.
Liu, H.; Li, C.; Wu, Q.; and Lee, Y . J. 2023. Visual in-
struction tuning.Advances in neural information processing
systems, 36: 34892–34916.
Liu, H.; Saha, R.; Jia, Z.; Park, Y .; Huang, J.; Sabach,
S.; Wang, Y .-X.; and Karypis, G. 2025. PROXSPARSE:
REGULARIZED LEARNING OF SEMI-STRUCTURED
SPARSITY MASKS FOR PRETRAINED LLMS. InF orty-
second International Conference on Machine Learning.
Lu, Y .; Agrawal, S.; Subramanian, S.; Rybakov, O.; De Sa,
C.; and Yazdanbakhsh, A. 2023. Step: Learning n: M
structured sparsity masks from scratch with precondition.
InInternational Conference on Machine Learning, 22812–
22824. PMLR.
Ma, X.; Fang, G.; and Wang, X. 2023. LLM-Pruner: On the
Structural Pruning of Large Language Models. InAdvances
in Neural Information Processing Systems.
Mena, G.; Belanger, D.; Linderman, S.; and Snoek, J. 2018.
Learning Latent Permutations with Gumbel-Sinkhorn Net-
works. InInternational Conference on Learning Represen-
tations.

<!-- page 9 -->

Merity, S.; Xiong, C.; Bradbury, J.; and Socher, R. 2016.
Pointer Sentinel Mixture Models. arXiv:1609.07843.
Mihaylov, T.; Clark, P.; Khot, T.; and Sabharwal, A. 2018.
Can a Suit of Armor Conduct Electricity? A New Dataset
for Open Book Question Answering. InEMNLP.
Peebles, W.; and Xie, S. 2023. Scalable diffusion models
with transformers. InProceedings of the IEEE/CVF inter-
national conference on computer vision, 4195–4205.
Pool, J.; Sawarkar, A.; and Rodge, J. 2021. Accelerating
inference with sparsity using the nvidia ampere architec-
ture and nvidia tensorrt.NVIDIA Developer Technical Blog,
https://developer . nvidia. com/blog/accelerating-inference-
with-sparsityusing-ampere-and-tensorrt.
Pool, J.; and Yu, C. 2021. Channel permutations for n: m
sparsity.Advances in neural information processing sys-
tems, 34: 13316–13327.
Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.;
Matena, M.; Zhou, Y .; Li, W.; and Liu, P. J. 2020. Explor-
ing the limits of transfer learning with a unified text-to-text
transformer.Journal of machine learning research, 21(140):
1–67.
Sakaguchi, K.; Bras, R. L.; Bhagavatula, C.; and Choi, Y .
2021. Winogrande: An adversarial winograd schema chal-
lenge at scale.Communications of the ACM, 64(9): 99–106.
Singh, A.; Natarajan, V .; Shah, M.; Jiang, Y .; Chen, X.; Ba-
tra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa
models that can read. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, 8317–
8326.
Sun, M.; Liu, Z.; Bair, A.; and Kolter, J. Z. 2023a. A Simple
and Effective Pruning Approach for Large Language Mod-
els.arXiv preprint arXiv:2306.11695.
Sun, M.; Liu, Z.; Bair, A.; and Kolter, J. Z. 2023b. A simple
and effective pruning approach for large language models.
arXiv preprint arXiv:2306.11695.
Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y .; Li, X.;
Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023. Stanford
Alpaca: An Instruction-following LLaMA model. https:
//github.com/tatsu-lab/stanford alpaca.
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux,
M.-A.; Lacroix, T.; Rozi `ere, B.; Goyal, N.; Hambro, E.;
Azhar, F.; et al. 2023. Llama: Open and efficient founda-
tion language models.arXiv preprint arXiv:2302.13971.
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. At-
tention Is All You Need. arXiv:1706.03762.
Xia, M.; Gao, T.; Zeng, Z.; and Chen, D. 2023. Sheared
llama: Accelerating language model pre-training via struc-
tured pruning.arXiv preprint arXiv:2310.06694.
Yuan, L.; Chen, Y .; Wang, T.; Yu, W.; Shi, Y .; Jiang, Z.-H.;
Tay, F. E.; Feng, J.; and Yan, S. 2021. Tokens-to-token vit:
Training vision transformers from scratch on imagenet. In
Proceedings of the IEEE/CVF international conference on
computer vision, 558–567.
Yue, X.; Ni, Y .; Zhang, K.; Zheng, T.; Liu, R.; Zhang, G.;
Stevens, S.; Jiang, D.; Ren, W.; Sun, Y .; et al. 2024. Mmmu:
A massive multi-discipline multimodal understanding and
reasoning benchmark for expert agi. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 9556–9567.
Zellers, R.; Holtzman, A.; Bisk, Y .; Farhadi, A.; and Choi,
Y . 2019. Hellaswag: Can a machine really finish your sen-
tence?arXiv preprint arXiv:1905.07830.
Zhang, Y .; Bai, H.; Lin, H.; Zhao, J.; Hou, L.; and Cannis-
traci, C. V . 2023. Plug-and-Play: An Efficient Post-training
Pruning Method for Large Language Models. InThe Twelfth
International Conference on Learning Representations.
Zhang, Y .; Bai, H.; Lin, H.; Zhao, J.; Hou, L.; and Cannis-
traci, C. V . 2024. Plug-and-Play: An Efficient Post-training
Pruning Method for Large Language Models. In12th In-
ternational Conference on Learning Representations (ICLR
2024).
Zhou, A.; Ma, Y .; Zhu, J.; Liu, J.; Zhang, Z.; Yuan, K.; Sun,
W.; and Li, H. 2021. Learning n: m fine-grained struc-
tured sparse neural networks from scratch.arXiv preprint
arXiv:2102.04010.
