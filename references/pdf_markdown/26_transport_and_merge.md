# references/26_transport_and_merge.pdf

<!-- page 1 -->

Transport and Merge:
Cross-Architecture Merging for Large Language Models
Chenhang Cui1,∗, Binyun Yang2,∗, Fei Shen1,†, Yuxin Chen1, Jingnan Zheng1,
Xiang Wang3, An Zhang3, Tat-Seng Chua1
1National University of Singapore (NUS), Singapore
2University of Electronic Science and Technology of China (UESTC), China
3University of Science and Technology of China (USTC), China
Abstract
Large language models (LLMs) achieve strong
capabilities by scaling model capacity and
training data, yet many real-world deployments
rely on smaller models trained or adapted
from low-resource data. This gap motivates
the need for mechanisms to transfer knowl-
edge from large, high-resource models to
smaller, resource-constrained targets. While
model merging provides an effective transfer
mechanism, most existing approaches assume
architecture-compatible models and therefore
cannot directly transfer knowledge from large
high-resource LLMs to heterogeneous low-
resource targets. In this work, we propose a
cross-architecture merging framework based on
optimal transport (OT) that aligns activations
to infer cross-neuron correspondences between
heterogeneous models. The resulting transport
matrices are then used to guide direct weight-
space fusion, enabling effective high-resource
to low-resource transfer using only a small
set of inputs. Extensive experiments across
low-resource languages and specialized domains
demonstrate consistent improvements over target
base models. The code is available at https:
//github.com/chenhangcuisg-code/
Cross-Architecture-Merging-for-\
Large-Language-Models/.
1. Introduction
Large language models (LLMs) have achieved remark-
able success in general language understanding and genera-
tion (Achiam et al., 2023; Grattafiori et al., 2024; Bai et al.,
∗Equal contribution. †Corresponding authors..
Preprint.
2023), enabling a wide range of downstream applications,
e.g., in medicine (Xie et al., 2024), finance (Konstantinidis
et al., 2024), and education (Wen et al., 2024). This success
is largely driven by scaling model capacity and pretraining
on massive, high-quality, and diverse corpora.
In practice, many real-world deployments rely on models
with limited parameter budgets trained or adapted from
low-resource data. Typical examples include low-resource
languages such as Malaysian languages (Hew et al., 2025)
and Cantonese dialects (Liu, 2022). Due to constraints on
both model capacity and data availability, models trained
from low-resource languages are unable to acquire specific
knowledge comparable to large and data-rich models. A
natural direction to address this challenge is high-resource
to low-resource transfer (Adimulam et al., 2022; Myakala
& Naayini, 2023; Cao et al., 2025), which leverages repre-
sentations learned by large, well-trained models to improve
performance in low-resource domains. Among existing ap-
proaches, model merging (Wortsman et al., 2022a; Jin et al.,
2022; Imfeld et al., 2023a) has emerged as an alternative
to gradient-based adaptation. It directly aggregates parame-
ters from multiple expert models, enabling capability fusion
without access to massive training data or repeated optimiza-
tion. With the rise of large language models (LLMs) (Yang
et al., 2024; Tao et al., 2024; Perin et al., 2024), this idea
naturally extends to LLM merging.
Despite its appeal, many current techniques (Tao et al., 2024;
Perin et al., 2024; Tian et al., 2025) assume that LLMs be-
ing merged share the same architecture, enabling direct
weight-space operations. This assumption restricts their
applicability in low-resource transfer scenarios, where the
source model is often a large high-resource LLM, while the
target model is a smaller architecture designed for deploy-
ment efficiency or domain-specific constraints (Liu et al.,
2024; 2023a; Kolomeitsev, 2025). Direct parameter merging
between heterogeneous architectures becomes challenging.
Although some recent methods attempt cross-architecture
merging or knowledge fusion for LLMs (Wan et al., 2024;
1
arXiv:2602.05495v2  [cs.CL]  22 Feb 2026

<!-- page 2 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
2025), they typically rely on distillation-based training (Hin-
ton et al., 2015; Hsieh et al., 2023; Cai et al., 2025) to
first distill knowledge into the parameter space of a target
model, followed by merging among architecture-compatible
models.
These limitations naturally raise the following question:
Can we enable high-resource to low-resource knowl-
edge transfer across heterogeneous model architectures
through direct model merging?Motivated by this ques-
tion, we propose a cross-model merging framework based
on optimal transport (Peyr ´e & Cuturi, 2019; Rout et al.,
2021) to enable parameter transfer between heterogeneous
architectures. Our approach is motivated by the observation
that, despite differences in architecture and model capacity,
language models often exhibit correlated internal activations
when processing the same inputs (Huh et al., 2024; Shah &
Khosla, 2025). Building on this insight, we establish a struc-
tured correspondence between heterogeneous models by
aligning their internal activations using an optimal transport
formulation. Given a small set of inputs, we extract interme-
diate activations from both the source and target models and
compute cross-model similarity signals at the feature level.
These signals are used to infer cross-neuron relationships
between the two models, yielding a global transfer plan that
captures transferable structure across layers and neurons.
The resulting transport relationships are then converted into
weight-space fusion operators, enabling direct parameter
fusion across architectures.
In summary, our contribution is a cross-architecture model
merging framework that enables high-resource to low-
resource knowledge transfer. By aligning heterogeneous
language models in activation space through an optimal
transport formulation, our method infers structured neuron-
level correspondences across models with different architec-
tures and capacities using only a small set of inputs. Cru-
cially, we show that these activation-space correspondences
admit a principled weight-space interpretation, allowing
them to be converted into direct fusion for parameter trans-
fer across heterogeneous models. Extensive experiments on
low-resource languages and domains demonstrate consis-
tent improvements, establishing our approach as a practical
alternative to distillation-based transfer. As shown in Fig-
ure 1, our method consistently improves performance across
different settings under multiple strategies.
2. Related Work
Model Merging.Model merging seeks to combine multiple
expert models into a single model, typically without access
to the original training data and without costly training (Il-
harco et al., 2022; Wortsman et al., 2022b; Ainsworth et al.,
2022). Existing approaches primarily involve two stages:
pre-merging and during-merging, covering both foundation
Figure 1.Cross-domain comparison using normalized scores
(higher is better).Top:Domain-specific performance trajectories
for Cantonese, Malaysian, and Thai under different transfer strate-
gies.Bottom-left:Relative improvements within each domain.
Bottom-right:Average normalized performance across all do-
mains. See Section 5.3 for detailed analysis.
models (LLMs and MLLMs) (Yang et al., 2024) and other
deep learning settings (Li et al., 2023). Pre-merging meth-
ods focus on preparing models for subsequent fusion by
modifying their parameters or representations in advance.
Typical approaches include adaptation schemes designed
to reduce parameter interference (Jacot et al., 2018; Ortiz-
Jimenez et al., 2023) and alignment techniques that map dif-
ferent checkpoints into compatible parameter spaces prior to
merging (Singh & Jaggi, 2020; Imfeld et al., 2023b). During-
merging methods perform the actual fusion, ranging from
simple parameter averaging (Wortsman et al., 2022a) to
more structured strategies such as subspace-based merging
(Yadav et al., 2023), as well as optimization-based merging
(Wei et al., 2025). Although some methods attempt cross-
architecture merging or knowledge fusion in LLMs (Wan
et al., 2024; 2025), they generally rely on distillation-based
training to first transfer knowledge into the parameter space
of a target model.
Optimal Transport and Applications.Optimal transport
(OT) provides a principled framework for comparing proba-
bility measures by finding a minimum-cost plan that trans-
ports mass from a source distribution to a target distribu-
tion (Peyr´e & Cuturi, 2019). Building on solid theory, OT
has become a versatile tool across a broad range of appli-
cations. In machine learning, OT is commonly used for
distribution matching in generative modeling (Rout et al.,
2021), domain adaptation under distribution shift (Liu et al.,
2023b; Redko et al., 2019), imitation learning (Luo et al.,
2023), and clustering (Del Barrio et al., 2019; Lin et al.,
2023). More recently, OT has been explored as a mecha-
nism for model merging, where a transport plan provides a
soft correspondence between neurons and enables weight
merging across models (Imfeld et al., 2023a; Singh & Jaggi,
2020). However, while recent studies have used optimal
transport (OT) to reveal cross-model feature similarity even
2

<!-- page 3 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
between heterogeneous LLMs (Shah & Khosla, 2025), how
to effectively leverage OT for model merging with different
architectures remains underexplored.
3. Preliminary
3.1. Optimal Transport
Optimal transport (OT) provides a principled way to com-
pute a soft correspondence between two sets of objects under
a pairwise cost. Given a cost matrix C∈R n×m and two
discrete distributionsa∈∆ n,b∈∆ m, OT solves
min
Q∈Rn×m
+
⟨C, Q⟩s.t.Q1=a, Q ⊤1=b.(1)
We adopt the entropically regularized variant,
min
Q∈Rn×m
+
⟨C, Q⟩−εH(Q)s.t.Q1=a, Q ⊤1=b,(2)
where H(Q) =− P
i,j Qij(logQ ij −1) and ε >0 . The
solution admits the Sinkhorn scaling form (Cuturi, 2013)
Q= diag(u)Kdiag(v) with K= exp(−C/ε) , and can
be computed by alternating updates onuandv.
3.2. Neurons and Features in Transformers
In this paper, we adopt a structural view of neurons in Trans-
former models. Specifically, we use the term “neuron” to
refer to a unit in the weight space of a linear sublayer, corre-
sponding to a single row or column of the associated weight
matrix. The values propagated through the network are
referred to as activated features.
Specifically, consider a linear transformation
y=W x,(3)
where W∈R dout×din. We distinguish two neuron spaces
induced by W : (i) input-side neurons correspond to the
columnsof W (indexed by din), and (ii) output-side neurons
correspond to therowsof W (indexed by dout). Accord-
ingly, x∈R din and y∈R dout represent features in the
input and output spaces, respectively.
In a Transformer block, the primary linear sublayers include
the attention projections
Q=W Qh, K=W Kh, V=W V h,(4)
and the MLP projections
z=W 1h, h ′ =σ(z), o=W 2h′,(5)
where σ(·) denotes an elementwise nonlinearity. In all these
cases, rows and columns of the projection matrices define
neurons in weight space, while each dimension of the acti-
vations corresponds to a feature induced by these neurons.
4. Method
Problem Definition.Our goal is to merge models across
heterogeneous architectures. We consider a target model
MA withLlayers and a source modelM B withMlayers.
Given a set of T samples D={x 1, . . . , xT }, we record
activations from both models across transformer modules:
Xℓ ∈R T×n ℓ , ℓ= 1, . . . , L,
Ym ∈R T×n ′
m, m= 1, . . . , M,
(6)
where each row corresponds to one input sample and each
column corresponds to one feature channel. Details of the
data used for activation extraction and transport estimation
are provided in Appendix B.1.
4.1. Finding Feature and Layer Relationships using
Optimal Transport
We use optimal transport (OT) (Peyr´e & Cuturi, 2019) to
infer transport relationship matrices from activations, which
reveal how activated feature channels in a source model
can be recombined to match those in a target model. Al-
though the ultimate goal is to merge parameters (i.e., neuron
weights), directly establishing neuron-to-neuron correspon-
dences is challenging across heterogeneous architectures.
Instead, we first infer correspondences in the activation
space and then lift these activation-level relationships to the
neuron level for parameter transport.
Concretely, for each target layer ℓ and source layer m, we
quantify cross-model similarity at the level of activation
channels by constructing a correlation-based cost matrix
C ℓm[i, j] =d corr(Xℓ[:, i], Y m[:, j])≜1−ρ(X ℓ[:, i], Y m[:, j]),
(7)
whereρ(·,·)denotes the Pearson correlation coefficient.
Based on this cost matrix, we compute a feature relationship
matrix Qℓm ∈R nℓ×n′
m by solving an entropically regular-
ized optimal transport problem:
Qℓm = arg min
Q∈T(n ℓ,n′m)
⟨C ℓm, Q⟩ −εH(Q),(8)
where H(Q) =− P
i,j Qij(logQ ij −1) and ε >0 . The
transportation polytope enforces balanced marginals,
T(n ℓ, n′
m) =
n
Q∈R nℓ×n′
m
+ :Q1= 1
nℓ
1, Q ⊤1= 1
n′m
1
o
.
(9)
The resulting transport plan Qℓm specifies how each target
activation channel can be represented as a weighted combi-
nation of source channels. Although inferred purely from
activation statistics, Qℓm will later be used as a neuron-level
mixing operator for transporting source parameters into the
target neuron coordinates.
3

<!-- page 4 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Figure 2.Illustration of cross-architecture merging pipeline.Given a small dataset D, we extract intermediate activations from a
high-resource source model and a low-resource target model with heterogeneous architectures. We then use optimal transport to infer
layer- and feature-level correspondences, and leverage the resulting transport plans for direct parameter fusion. Finally, the fused model
can be optionally refined via residual-frozen adaptation, where the transferred residuals are kept fixed and only base weights are updated.
To summarize the compatibility between target layer ℓ and
source layer m, we aggregate the feature-level correspon-
dence into a scalar transport cost inspired by (Shah &
Khosla, 2025)
Clayer[ℓ, m] =⟨C ℓm, Qℓm⟩,(10)
which yields a layer-to-layer cost matrixC layer ∈R L×M .
Using these layer-wise costs, we further infer a global cor-
respondence matrix P∈R L×M by solving another entropi-
cally regularized optimal transport problem:
P= arg min
P∈T(L,M)
⟨Clayer, P⟩ −ηH(P),(11)
whereη >0and
T(L, M) =
n
P∈R L×M
+ :P1= 1
L 1, P ⊤1= 1
M 1
o
.
(12)
Each entry Pℓm quantifies the degree of correspondence
between target layer ℓ and source layer m. Together, the
feature-level transport plans {Qℓm} and the global matrix
P specify how source neurons are used to construct target
neurons. All optimal transport problems are solved using
Sinkhorn iterations (Cuturi, 2013); implementation details
about the algorithm are provided in Appendix A.
4.2. From Activated Features to Neurons: Weight
Fusion via Feature and Layer Transport
Our transport matrices are estimated from activated features,
while the ultimate objective is to merge neurons and their
associated parameters. The key observation is that each
feature dimension corresponds to a well-defined neuron axis
in the underlying weight matrix, specifically, a column in
the input space or a row in the output space. This correspon-
dence allows us to lift feature-level relations into neuron-
level mixing operators, which are then used to transport
source weights into the target neuron basis.
Feature- and Layer-wise Transport.In attention modules,
input-side (pre-projection) and output-side (post-projection)
features lie in different representation spaces and may in-
duce different cross-layer alignment signals. Accordingly,
we estimate two feature-level transport plans,Qℓm
in from pre-
activation features and Qℓm
out from post-activation features.
At the layer level, we compute two compatibility costs by
aggregating the corresponding feature-level transport objec-
tives, and solve the layer-level OTseparatelyfor the pre-
and post-sides, yielding Ppre and Ppost, respectively. We
then combine them into an effective layer correspondence
Peff[ℓ, m] =
q
Ppre[ℓ, m]·P post[ℓ, m],(13)
Selective transport via top- k neuron replacement.To
improve robustness across heterogeneous architectures and
avoid unnecessary interference, we restrict fusion to a small
subset of neurons that are highly activated on the same
dataset D used for alignment. Concretely, for each layer
ℓ and each projection module, we run the target model on
D and record neuron activations via forward hooks. For
each neuron, we compute an activation strength score as the
mean absolute activation over samples sj = 1
T
PT
t=1
ht,j
,
where ht,j denotes the activation of neuron j at sample t.
We then select the top-k neurons with the largest scores in
4

<!-- page 5 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
each layer and projection module, and record their indices.
Details about selection can be found in Appendix A.5
Fusion is applied only to these selected neuron indices,
while all remaining parameters in the target model are left
unchanged. For attention projections, this corresponds to
replacing the rows or columns of the weight matrices as-
sociated with the selected neurons. This design yields a
partial weight fusion scheme that injects source knowledge
only into neurons that are both highly active on the target
data and well-aligned across models. Under this design, the
fused weights at target layerℓcan be written as:
W ℓ,fused
A =W ℓ
A
+α·M ℓ ⊙
 MX
m=1
Peff[ℓ, m]Q ℓm
out W m
B
 
Qℓm
in
⊤
−W ℓ
A
!
,
(14)
where Mℓ is a binary neuron-level mask that is nonzero only
on the selected top- k neuron indices, ⊙ denotes masking
at the neuron level, and α∈[0,1] controls the strength of
replacement. Bias terms are handled analogously.
4.3. Residual-Frozen Adaptation after Fusion
After fusion, only a small subset of neurons in the target
model are modified by transported source parameters, while
the remaining parameters remain identical to the original
target model. During this stage we freeze the transported
residual parameters and train only the original base parame-
ters of the target model. This design ensures that the trans-
ferred knowledge is not overwritten by subsequent optimiza-
tion, while still allowing the target model to adapt to the
downstream task distribution. Empirically, this residual-
frozen adaptation consistently outperforms training the tar-
get model alone under the same conditions, indicating that
fusion provides a strong and stable initialization.
Residual Parameterization.We parameterize fusion
through a residual form that separates transferred neuron-
level knowledge from the original model parameters. Specif-
ically, weight of target layerℓis expressed as
W ℓ,fused
A =W ℓ,base
A +α·M ℓ ⊙∆W ℓ
A,(15)
where W ℓ,base
A denotes the original weights of the target
model MA, Mℓ is a neuron-level mask that is nonzero only
on the selected top-k neurons, ⊙ denotes neuron-level mask-
ing, and α∈[0,1] controls the strength of replacement.
The transported residual ∆W ℓ
A aggregates source operators
across layers via activation-aligned transport,
∆W ℓ
A =
MX
m=1
Peff[ℓ, m]Q ℓm
out W m
B (Qℓm
in )⊤,(16)
where Peff[ℓ, m] is the effective layer correspondence and
Qℓm
in , Qℓm
out are the feature-level transport maps inferred
from activation statistics. Only the neurons selected by
Mℓ participate in fusion; all others remain unchanged.
Residual-Freezing and Adaptation.During post-fusion
adaptation, we freeze the transported residual∆W ℓ
A entirely
and optimize only the base parameters W ℓ,base
A . Gradients
are therefore applied uniformly to the base weights, without
any neuron-level freezing:
∂L
∂∆W ℓ
A
= 0, ∂L
∂W ℓ,base
A
̸= 0.(17)
This separation preserves the transferred neuron-level repre-
sentations while allowing the target model to adapt globally.
Weight Folding.After adaptation, we fold the residual back
into the base weights to obtain a parameter representation
for inference:
W ℓ,final
A =W ℓ,base
A +α·M ℓ ⊙∆W ℓ
A.(18)
This folding operation yields a model that is architecturally
identical to the target model, with transferred neuron-level
knowledge permanently absorbed into its parameters.
We summarize the overall transport-and-merge procedure
with masked neuron replacement in Alg. 1.
4.4. Weight Transport as Representation-Space
Transfer
Transporting source weights into the target model admits an
exact interpretation in representation space: it is equivalent
to mapping target features into the source feature coordi-
nates, applying the source linear map there, and mapping
the result back to the target space.
Setup.Fix a target sublayer ℓ in MA and a source sublayer
minM B with linear operators (bias omitted)
W ℓ
A ∈R dℓ
A,out×dℓ
A,in , W m
B ∈R dm
B,out ×dm
B,in .
Let Qℓm
in and Qℓm
out be transport relationship matrices esti-
mated from pre- and post-activation features. Define the
induced coordinate maps
Φℓm
in ≜(Q ℓm
in )⊤,Φ ℓm
out ≜Q ℓm
out.(19)
We then define the transported source operator acting on
target features as
fW ℓm
B→A ≜Φ ℓm
out W m
B Φℓm
in .(20)
Theorem 4.1(Representation-Space Interpretation of
Weight Transport).For any target representation hA ∈
Rdℓ
A,in and any(ℓ, m),
fW ℓm
B→AhA = Φℓm
out

W m
B
 
Φℓm
in hA

,(21)
5

<!-- page 6 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Algorithm 1Transport and Merge for Cross-Architecture
Model Merging
Require: Target model MA with L layers, source model
MB with M layers, samples D={x t}T
t=1, fusion
coefficientα
Ensure:Fused target modelM fused
A
1:Activation extraction:
2: Run MA and MB on D to obtain activations {Xℓ}L
ℓ=1
and{Y m}M
m=1
3:Feature-level transport:
4:foreach target layerℓand source layermdo
5:Construct cost matrixC ℓm (Eq. 7)
6:Solve entropic OT to obtainQ ℓm
in , Qℓm
out (Eq. 8)
7:end for
8:Layer-level transport:
9: Compute pre/post layer costs from the corresponding
Qℓm
in andQ ℓm
out (Eq. 10)
10:Solve OT to obtainP pre andP post (Eq. 11)
11:ComputeP eff (Eq. 13)
12:Weight fusion:
13:foreach target layerℓdo
14:Fuse weights using Eq. 14
15:end for
16:Optional post-fusion adaptation:
17: Freeze transferred residuals, fine-tune base parameters,
and fold residuals for inference (Eqs. 15–18)
18:Output:M fused
A
Our fused target operator aggregates transported source
operators across layers:
W ℓ,fused
A = (1−α)W ℓ
A +α
MX
m=1
Peff[ℓ, m]fW ℓm
B→A.(22)
Combining Eq. 22 with Theorem 4.1 shows that fusion
realizes a mixture of source-space computations driven by
target features transferred into the corresponding source
coordinate systems.
Corollary 4.2(Uniqueness Under Invertible Coordinate
Maps).If Φℓm
in and Φℓm
out are invertible, then fW ℓm
B→A is
the unique operator on the target space that reproduces
the source-layer computation under the target-to-source
coordinate transfer in Eq. 21.
Proofs are provided in Appendix A.4.
5. Experiments
5.1. High- to Low-Resource Language Transfer
Setup.We refer to the original unfused target model as
theBase Model. We consider two variants of our method:
Fused w/o Adaptation, which performs transport-guided
Table 1.Low-resource language transfer results on MalayMMLU.
Accuracy (%; higher is better).
Category base model Fused w/o adaptation Fused w/ adaptation
Humanities 42.30 46.1948.81
Language 38.37 40.5241.68
Others 44.50 46.6348.69
STEM 42.61 45.1946.58
Social Science 40.92 43.7746.91
Table 2.Low-resource language transfer results on Indonesian
benchmarks. Accuracy (%; higher is better).
Benchmark Base ModelFused w/o AdaptationFused w/ Adaptation
ARC (Indonesian) 23.424.023.7Belebele (Indonesian) 34.8 36.436.9TruthfulQA-MC (Indonesian) 35.9 36.536.6XCOPA (Indonesian) 59.6 59.860.0
Average 38.43 39.18 39.30
parameter fusion without any post-training, and Fused w/
Adaptation, which applies an adaptation stage. To ensure
experimental consistency, we fix the source architecture to
the LLaMA-3 8B family (Grattafiori et al., 2024) in this
section. Comparisons with other model families are de-
ferred to later experiments in Section 5.4. We adopt four
low-resource languages.Malaysian:Target is Malaysian-
LLaMA-3-1B-Instruct (Mesolitica, 2025).Indonesian:
Target is LLaMA-3-1B-Indonesian-QLoRA (Wahyurejeki,
2025).Thai:Target is Typhoon2-1B-Instruct (Pipatanakul
et al., 2024).Cantonese:To better match linguistic prox-
imity, we use a Chinese instruction-tuned source LLaMA3-
Chinese-8B-Instruct (FlagAlpha, 2025), and fuse into the
target model LLaMA-3-1B-Instruct. Details of the data used
for adaptation and evaluation benchmarks are provided in
Appendix B.1 and Appendix B.2, respectively.
Malaysian.Table 1 shows results of our method on
MalayMMLU (Poh et al., 2024). We can observe that our
method consistently achieves the best performance. Rela-
tive to the Base Model, Fused w/ Adaptation gains +6.51
(Humanities), +3.31 (Language), +4.19 (Others), +3.97
(STEM), and +5.99 (Social Science), indicating successful
transfer from the high-resource language.
Indonesian.Table 2 reports results on Indonesian bench-
marks (Clark et al., 2018; Bandarkar et al., 2024; Lin et al.,
2022; Ponti et al., 2020). Fusion alone already improves
most tasks, suggesting effective language capability trans-
fer without extra training. With post-fusion adaptation, the
model attains the best scores on three of four benchmarks,
while Fused w/o Adaptation remains best on ARC.
Cantonese.Table 3 shows accuracy of CMMLU bench-
mark (Jiang et al., 2025). Fused w/ Adaptation improves
all categories and raises the overall average from 25.26 to
27.44 (+2.18). The consistent gains suggest our fusion can
transfer relevant knowledge despite different architectures.
Thai.Table 4 summarizes Thai benchmark (Hendrycks
6

<!-- page 7 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Table 3.Low-resource language transfer results on CMMLU (Can-
tonese) (Jiang et al., 2025). Accuracy (%; higher is better).
Category Base ModelFused w/o AdaptationFused w/ Adaptation
Humanities 25.4427.7227.34
Social Science 25.07 27.1027.36
STEM 25.2226.63 26.63
Others 25.8429.2128.99
Average 25.26 27.41 27.44
Table 4.Low-resource language transfer results on Thai bench-
marks. Score (higher is better).
Benchmark Base ModelFused w/o AdaptationFused w/ Adaptation
MMLU (Thai) 0.15 0.130.17
MGSM (Thai) 0.50 0.640.72
XCOPA (Thai) 0.58 0.580.60
Average 0.41 0.45 0.50
et al., 2020; Ponti et al., 2020; Shi et al., 2022) results.
Fusion substantially boosts the MGSM task even without
adaptation (0.50→0.64 ), and adaptation further improves
all reported benchmarks, achieving the best overall scores.
This indicates fusion provides effective transfer, while adap-
tation stabilizes and calibrates task performance.
5.2. General-to-Expert Domain Transfer
Setup.We also evaluate general-domain to expert-domain
transfer. ForFinance, we fuse LLaMA-3-1B-TEL-A-
finance (TEL-LLM, 2025) with Qwen2.5-7B (Bai et al.,
2023). ForMedical, we fuse medical version LLaMA
(Grattafiori et al., 2024) with LLaMA-3.2-8B.
Finance.Table 5 reports results on three finance-related
MMLU subsets (Hendrycks et al., 2020). Direct fusion with-
out adaptation largely preserves the Base Model’s perfor-
mance, yielding either marginal gains or near-parity across
tasks, which indicates that cross-domain fusion does not in-
troduce negative interference. With post-fusion adaptation,
the fused model consistently achieves the best performance
on all three benchmarks. These results suggest that fusion
serves as an effective initialization that transfers general
knowledge into the target domain.
Medical.Table 6 shows consistent improvements from fu-
sion on medical benchmarks (Hendrycks et al., 2020). Even
without adaptation, fusion yields positive gains on anatomy
and professional medicine. Moreover, Fused w/ Adaptation
achieves the best results across all three tasks, improving
over the Base Model by +1.5, +2.0, and +1.2, indicating
stable consolidation of transferred medical knowledge.
5.3. Effectiveness of Cross-Architecture Merging
Evidence of cross-architecture Representation Similarity.
We analyze representation similarity across heterogeneous
Table 5.General-domain to finance-domain transfer results on
financial benchmarks. Score (higher is better).
Benchmark Base ModelFused w/o AdaptationFused w/ Adaptation
MMLU (Business Ethics) 0.36 0.370.38MMLU (Microeconomics) 0.340.390.37MMLU (Professional Accounting) 0.26 0.270.29
Average 0.32 0.34 0.35
Table 6.General-domain to medical-domain transfer results on
medical benchmarks. Accuracy (%; higher is better).
Benchmark Base ModelFused w/o AdaptationFused w/ Adaptation
MMLU (Anatomy) 49.6 50.051.1MMLU (Medical Genetics) 50.0 49.052.0MMLU (Professional Medicine) 39.3 39.740.5
Average 46.30 46.23 47.87
models using the feature-level optimal transport plans Qℓm
learned during fusion (Eq. (8)). Each entry Qℓm
ij measures
the amount of activation mass transported from a source
feature channel to a target feature channel, under balanced
marginal constraints. To quantify how concentrated these
correspondences are, we compute the percentage of trans-
port mass captured by the top-k entries of Qℓm. Specifically,
for each Qℓm, we sort all entries in descending order and
accumulate the transport mass of the top-k entries, reporting
this quantity as a percentage of the total transport mass. We
then average this quantity across all layer pairs and modules.
Figure 3 plots the averaged transport mass. The horizontal
axis corresponds to the top-k neuron correspondences, while
the vertical axis shows the average percentage of source-to-
target transport mass. Across all domains, a small number
of neuron pairs explains a large fraction of the transport
mass, indicating sparse yet aligned internal representations
across heterogeneous LLMs.
Performance Comparison with Merging Source Models
of Different Sizes.We further analyze how the capacity of
the source model affects cross-architecture fusion perfor-
mance. Using MalayMMLU as a representative benchmark,
we merge Malaysian-LLaMA-3-1B-Instruct with general-
purpose LLaMA models of different scales (1B, 8B, and
32B). Table 7 shows that increasing the source model size
leads to consistent performance improvements. While merg-
ing with an 8B source already yields substantial gains across
all categories, scaling the source to 32B provides additional
improvements. This indicates that larger source models of-
fer richer transferable representations that can be selectively
injected through our neuron-level fusion mechanism.
Compared with SFT Training and Distillation.As illus-
trated in Figure 1, our method consistently outperforms both
SFT and distillation (Hinton et al., 2015) under normalized
evaluation. Notably, the adapted fusion achieves the highest
relative performance in every domain, whereas SFT and dis-
tillation show domain-dependent variability. These results
suggest that effective cross-domain transfer requires not
7

<!-- page 8 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Figure 3.Average transport mass explainedby the top- k neuron
correspondences, computed from the optimal transport plans and
averaged over layers and modules.
Table 7.Effect of source model size on MalayMMLU. Results
report accuracy (%) of Fused w/ Adaptation with a fixed 1B target
model. Higher is better.
Source SizeHumanitiesLanguage Others STEM Social Science
1B 47.65 41.76 47.97 46.05 46.00
8B 48.81 41.68 48.69 46.58 46.91
32B49.17 42.27 49.17 46.75 47.28
only shared representations, but also post-fusion adaptation
to properly align domain-specific features. Details can be
found in Appendix C.2.
5.4. Robustness of Cross-Architecture Merging
Sensitivity to Source Model Backbone.Figure 4 studies
the sensitivity of our approach to the choice of high-resource
source backbone. We compare our method (Fused w/ Adap-
tation) using two different source models, LLaMA3-8B and
Qwen2.5-7B, while keeping the same low-resource target
model. Across both Malay and Indonesian benchmarks,
we observe consistent improvements over the Base Model.
This suggests that our method transfers task-relevant fea-
tures rather than overfitting to a specific source architecture,
and remains robust across heterogeneous source models.
Sensitivity to Fusion Coefficient α.Figure 5 analyzes the
sensitivity of performance to the fusion coefficientα, which
controls the contribution of the source model during param-
eter merging. Across all categories, performance exhibits a
consistent trend as α varies, indicating that our method is not
overly sensitive to precise hyperparameter tuning. Moderate
values of α (around 0.05–0.15) yield the strongest overall
performance. However, when α is too small, the influence
of the source model is limited, resulting in conservative
improvements. Conversely, overly large α values lead to
performance degradation in several categories, suggesting
that excessive source injection may introduce mismatched
or noisy information under architectural differences.
Figure 4.Sensitivity to the choice of source backbone.On
the Malay and Indonesian benchmarks, our method consistently
outperforms the Base Model when using either LLaMA3-8B or
Qwen2.5-7B as the source model.
Figure 5.Sensitivity analysis of the fusion coefficient α for
our method (Fused w/ Adaptation) on MalayMMLU. We report
category-wise accuracy (%; higher is better); dashed lines denote
the corresponding base-model performance.
Impact on General Abilities.Figure 6 evaluates how cross-
architecture fusion affects general capabilities. Details of
the general benchmarks used in this evaluation are provided
in Appendix B.2. Overall, fusion does not degrade gen-
eral abilities, and preserves performance close to the Base
Model. For Malay and Cantonese, fusion with adaptation
slightly improves the average general score, indicating that
transferred knowledge can be integrated without harming
generalization. For Indonesian, Thai, and Medical settings,
fused models achieve performance comparable to the Base
Model, with differences within a narrow margin. These
results suggest that our transport-based merging selectively
Malay Indonesian Cantonese Thai Medical Finance
Domain
0.50
0.52
0.54
0.56
0.58
0.60
0.62
0.64
0.66Average General Score
0.563
0.620 0.616
0.582
0.589
0.636
0.542
0.611 0.615
0.580
0.588
0.639
0.562
0.611
0.620
0.580
0.588
0.637
Base Model Fused w/o adaptation Fused w adaptation
Figure 6.Impact of cross-architecture fusion on general abili-
ties across domains.Bars show the average general score (higher
is better) for the Base Model and fused variants, with and without
post-fusion adaptation.
8

<!-- page 9 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
injects transferable knowledge while largely maintaining
the target model’s original general capabilities.
6. Conclusion
We proposed an OT-based framework for cross-architecture
model merging that aligns heterogeneous models in activa-
tion space and converts the resulting correspondences into
direct weight-space fusion. Across low-resource languages
and expert domains, our approach consistently improves tar-
get models using only a small calibration set, and residual-
frozen adaptation further strengthens performance while
preserving general capabilities. Overall, transport-guided
cross-architecture merging provides a practical and princi-
pled alternative to distillation when architectures differ.
Impact Statement
This work improves knowledge transfer from high-resource
models to smaller low-resource or domain-specific models
with heterogeneous architectures. We use activation-based
optimal transport to estimate cross-architecture correspon-
dences from a small input set for direct parameter fusion,
optionally followed by lightweight adaptation. Positive
impacts include reducing data/compute requirements for
low-resource languages and specialized domains. Negative
impacts include propagating biases, factual errors, or unsafe
behaviors from the source model, and overstating robust-
ness if evaluation is insufficient. We therefore emphasize
rigorous evaluation, transparent reporting, and appropriate
safeguards when applying the method in practice.
References
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
Adimulam, T., Chinta, S., and Pattanayak, S. K. Transfer
learning in natural language processing: Overcoming low-
resource challenges.International Journal of Enhanced
Research In Science Technology & Engineering, 11:65–
79, 2022.
Ainsworth, S. K., Hayase, J., and Srinivasa, S. Git re-basin:
Merging models modulo permutation symmetries.arXiv
preprint arXiv:2209.04836, 2022.
Bai, J., Bai, S., Chu, Y ., Cui, Z., Dang, K., Deng, X., Fan,
Y ., Ge, W., Han, Y ., Huang, F., et al. Qwen technical
report.arXiv preprint arXiv:2309.16609, 2023.
Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla,
S. N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer,
L., and Khabsa, M. The belebele benchmark: a parallel
reading comprehension dataset in 122 language variants.
InProceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long
Papers), pp. 749–775, 2024.
Bisk, Y ., Zellers, R., Gao, J., Choi, Y ., et al. Piqa: Reasoning
about physical commonsense in natural language. InPro-
ceedings of the AAAI conference on artificial intelligence,
volume 34, pp. 7432–7439, 2020.
Cai, Y ., Zhang, J., He, H., He, X., Tong, A., Gan, Z., Wang,
C., Xue, Z., Liu, Y ., and Bai, X. Llava-kd: A framework
of distilling multimodal large language models. InPro-
ceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 239–249, 2025.
9

<!-- page 10 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Cao, X., Xu, M., Yu, X., Yao, J., Ye, W., Huang, S., Zhang,
M., Tsang, I., Ong, Y .-S., Kwok, J. T., et al. Analytical
survey of learning with low-resource data: From analysis
to investigation.ACM Computing Surveys, 58(6):1–47,
2025.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
arXiv preprint arXiv:1803.05457, 2018.
Cuturi, M. Sinkhorn distances: Lightspeed computation
of optimal transport.Advances in neural information
processing systems, 26, 2013.
Del Barrio, E., Cuesta-Albertos, J. A., Matr ´an, C., and
Mayo-´Iscar, A. Robust clustering tools based on optimal
transportation.Statistics and Computing, 29(1):139–160,
2019.
FlagAlpha. Llama3-chinese-8b-instruct. Hug-
ging Face Model Card, 2025. URL
https://huggingface.co/FlagAlpha/
LLaMA3-Chinese-8B-Instruct.
Flowers, J. G. Finance-Instruct-500k.
https://huggingface.co/datasets/
Josephgflowers/Finance-Instruct-500k,
2025. Accessed: 2026-01-23.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
M., Song, D., and Steinhardt, J. Measuring mas-
sive multitask language understanding.arXiv preprint
arXiv:2009.03300, 2020.
Hew, Z. K., Low, J. X., Yang, S. J., and Chan, C. S.
Myculture: Exploring malaysia’s diverse culture un-
der low-resource language constraints.arXiv preprint
arXiv:2508.05429, 2025.
Hinton, G., Vinyals, O., and Dean, J. Distilling
the knowledge in a neural network.arXiv preprint
arXiv:1503.02531, 2015.
Hsieh, C.-Y ., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y .,
Ratner, A., Krishna, R., Lee, C.-Y ., and Pfister, T. Distill-
ing step-by-step! outperforming larger language models
with less training data and smaller model sizes. InFind-
ings of the Association for Computational Linguistics:
ACL 2023, pp. 8003–8017, 2023.
Huh, M., Cheung, B., Wang, T., and Isola, P. Position: the
platonic representation hypothesis. InProceedings of the
41st International Conference on Machine Learning, pp.
20617–20642, 2024.
Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S.,
Schmidt, L., Hajishirzi, H., and Farhadi, A. Editing mod-
els with task arithmetic.arXiv preprint arXiv:2212.04089,
2022.
Imfeld, M., Graldi, J., Giordano, M., Hofmann, T., Anagnos-
tidis, S., and Singh, S. P. Transformer fusion with optimal
transport.arXiv preprint arXiv:2310.05719, 2023a.
Imfeld, M., Graldi, J., Giordano, M., Hofmann, T., Anagnos-
tidis, S., and Singh, S. P. Transformer fusion with optimal
transport.arXiv preprint arXiv:2310.05719, 2023b.
Jacot, A., Gabriel, F., and Hongler, C. Neural tangent ker-
nel: Convergence and generalization in neural networks.
Advances in neural information processing systems, 31,
2018.
Jiang, J., Chen, P., Chen, L., Wang, S., Bao, Q., Kong, L.,
Li, Y ., and Wu, C. How well do llms handle cantonese?
benchmarking cantonese capabilities of large language
models. InFindings of the Association for Computational
Linguistics: NAACL 2025, pp. 4464–4505, 2025.
Jin, X., Ren, X., Preotiuc-Pietro, D., and Cheng, P. Data-
less knowledge fusion by merging weights of language
models.arXiv preprint arXiv:2212.09849, 2022.
Kolomeitsev, K. Llm modules: Knowledge transfer from
a large to a small model using enhanced cross-attention.
arXiv preprint arXiv:2502.08213, 2025.
Konstantinidis, T., Iacovides, G., Xu, M., Constantinides,
T. G., and Mandic, D. Finllama: Financial sentiment
classification for algorithmic trading applications.arXiv
preprint arXiv:2403.12285, 2024.
Li, W., Peng, Y ., Zhang, M., Ding, L., Hu, H., and Shen,
L. Deep model fusion: A survey.arXiv preprint
arXiv:2309.15698, 2023.
Lin, R., Du, S., Wang, S., and Guo, W. Multi-view cluster-
ing via optimal transport algorithm.Knowledge-Based
Systems, 279:110954, 2023.
Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring
how models mimic human falsehoods. InProceedings of
the 60th annual meeting of the association for computa-
tional linguistics (volume 1: long papers), pp. 3214–3252,
2022.
Liu, E. K.-Y . Low-resource neural machine translation: A
case study of cantonese. InProceedings of the Ninth
Workshop on NLP for Similar Languages, Varieties and
Dialects, pp. 28–40, 2022.
10

<!-- page 11 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Liu, J., Liu, J., Wang, Q., Wang, J., Cai, X., Zhao, D., Wang,
R., and Yan, R. Retrieval-based knowledge transfer: An
effective approach for extreme large language model com-
pression. InFindings of the association for computational
linguistics: EMNLP 2023, pp. 8643–8657, 2023a.
Liu, Y ., Zhou, Z., and Sun, B. Cot: Unsupervised domain
adaptation with clustering and optimal transport. InPro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pp. 19998–20007, 2023b.
Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y ., Fedorov,
I., Xiong, Y ., Chang, E., Shi, Y ., Krishnamoorthi, R.,
et al. Mobilellm: Optimizing sub-billion parameter lan-
guage models for on-device use cases.arXiv preprint
arXiv:2402.14905, 2024.
Luo, Y ., Jiang, Z., Cohen, S., Grefenstette, E., and Deisen-
roth, M. P. Optimal transport for offline imitation learning.
arXiv preprint arXiv:2303.13971, 2023.
Lynn, S. Cantonese Dialogue. https:
//huggingface.co/datasets/stvlynn/
Cantonese-Dialogue, 2024. Accessed: 2026-01-
23.
Mesolitica. Malaysian SFT. https://huggingface.
co/datasets/mesolitica/Malaysian-SFT,
2024. Accessed: 2026-01-23.
Mesolitica. Malaysian-llama-3.2-1b-instruct.
Hugging Face Model Card, 2025. URL
https://huggingface.co/mesolitica/
Malaysian-LLaMA-3.2-1B-Instruct.
Myakala, P. K. and Naayini, P. Bridging the gap: Leveraging
transfer learning for low-resource nlp tasks.International
Journal of Computer Techniques, 10(5), 2023.
Ortiz-Jimenez, G., Favero, A., and Frossard, P. Task arith-
metic in the tangent space: Improved editing of pre-
trained models.Advances in Neural Information Pro-
cessing Systems, 36:66727–66754, 2023.
Penedo, G., Kydl´ıˇcek, H., Lozhkov, A., Mitchell, M., Raffel,
C. A., V on Werra, L., Wolf, T., et al. The fineweb datasets:
Decanting the web for the finest text data at scale.Ad-
vances in Neural Information Processing Systems, 37:
30811–30849, 2024.
Perin, G., Chen, X., Liu, S., Kailkhura, B., Wang, Z., and
Gallagher, B. Rankmean: Module-level importance score
for merging fine-tuned llm models. InFindings of the
Association for Computational Linguistics: ACL 2024,
pp. 1776–1782, 2024.
Peyr´e, G. and Cuturi, M.Computational optimal transport:
With applications to data science. Now Foundations and
Trends, 2019.
Pipatanakul, K., Manakul, P., Nitarach, N., Sirichote-
dumrong, W., Nonesung, S., Jaknamon, T., Pengpun,
P., Taveekitworachai, P., Na-Thalang, A., Sripaisarn-
mongkol, S., Jirayoot, K., and Tharnpipitchai, K. Ty-
phoon 2: A family of open text and multimodal thai large
language models, 2024. URL https://arxiv.org/
abs/2412.13702.
Poh, S. C., Yang, S. J., Tan, J. M. L., Chieng, L. L. T. Y ., Tan,
J. X., Yu, Z., Mun, F. C., and Chan, C. S. Malaymmlu:
A multitask benchmark for the low-resource malay lan-
guage. InFindings of the Association for Computational
Linguistics: EMNLP 2024, pp. 650–669, 2024.
Ponti, E. M., Glava ˇs, G., Majewska, O., Liu, Q., Vuli ´c,
I., and Korhonen, A. Xcopa: A multilingual dataset
for causal commonsense reasoning.arXiv preprint
arXiv:2005.00333, 2020.
Redko, I., Courty, N., Flamary, R., and Tuia, D. Optimal
transport for multi-source domain adaptation under target
shift. InThe 22nd International Conference on artificial
intelligence and statistics, pp. 849–858. PMLR, 2019.
Rout, L., Korotin, A., and Burnaev, E. Generative mod-
eling with optimal transport maps.arXiv preprint
arXiv:2110.02999, 2021.
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y .
Winogrande: An adversarial winograd schema challenge
at scale.Communications of the ACM, 64(9):99–106,
2021.
Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y .
Socialiqa: Commonsense reasoning about social interac-
tions.arXiv preprint arXiv:1904.09728, 2019.
Shah, S. and Khosla, M. Representational alignment across
model layers and brain regions with hierarchical optimal
transport.arXiv preprint arXiv:2510.01706, 2025.
Shekswess. Medical LLaMA3 Instruct Dataset. https:
//huggingface.co/datasets/Shekswess/
medical_LLaMA3_instruct_dataset, 2024.
Accessed: 2026-01-23.
Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S.,
V osoughi, S., Chung, H. W., Tay, Y ., Ruder, S., Zhou, D.,
et al. Language models are multilingual chain-of-thought
reasoners.arXiv preprint arXiv:2210.03057, 2022.
Singh, S. P. and Jaggi, M. Model fusion via optimal trans-
port.Advances in Neural Information Processing Systems,
33:22045–22055, 2020.
Talmor, A., Herzig, J., Lourie, N., and Berant, J. Com-
monsenseqa: A question answering challenge targeting
commonsense knowledge. InProceedings of the 2019
11

<!-- page 12 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Conference of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pp.
4149–4158, 2019.
Tao, M., Zhang, C., Huang, Q., Ma, T., Huang, S., Zhao,
D., and Feng, Y . Unlocking the potential of model
merging for low-resource languages.arXiv preprint
arXiv:2407.03994, 2024.
TEL-LLM. Llama-3.2-1b-tel-a-finance. Hug-
ging Face Model Card, 2025. URL https:
//huggingface.co/TEL-LLM/LLaMA-3.
2-1B-TEL-A-finance.
Tian, Y ., Han, Y ., Chen, X., Wang, W., and Chawla, N. V .
Beyond answers: Transferring reasoning capabilities to
smaller llms using multi-teacher knowledge distillation.
InProceedings of the Eighteenth ACM International Con-
ference on Web Search and Data Mining, pp. 251–260,
2025.
Wahyurejeki. Llama-3.2-1b-indonesian-qlora.
Hugging Face Model Card, 2025. URL
https://huggingface.co/wahyurejeki/
LLaMA-3.2-1B-Indonesian-QLora.
Wan, F., Huang, X., Cai, D., Quan, X., Bi, W., and Shi,
S. Knowledge fusion of large language models.arXiv
preprint arXiv:2401.10491, 2024.
Wan, F., Zhong, L., Yang, Z., Chen, R., and Quan, X.
Fusechat: Knowledge fusion of chat models. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pp. 21629–21653, 2025.
Wei, Y ., Tang, A., Shen, L., Hu, Z., Yuan, C., and Cao, X.
Modeling multi-task model merging as adaptive projec-
tive gradient descent.arXiv preprint arXiv:2501.01230,
2025.
Wen, Q., Liang, J., Sierra, C., Luckin, R., Tong, R., Liu, Z.,
Cui, P., and Tang, J. Ai for education (ai4edu): Advancing
personalized education with llm and adaptive learning. In
Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, pp. 6743–6744,
2024.
Wortsman, M., Ilharco, G., Gadre, S. Y ., Roelofs, R.,
Gontijo-Lopes, R., Morcos, A. S., Namkoong, H.,
Farhadi, A., Carmon, Y ., Kornblith, S., et al. Model
soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time. In
International conference on machine learning, pp. 23965–
23998. PMLR, 2022a.
Wortsman, M., Ilharco, G., Gadre, S. Y ., Roelofs, R.,
Gontijo-Lopes, R., Morcos, A. S., Namkoong, H.,
Farhadi, A., Carmon, Y ., Kornblith, S., et al. Model
soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time. In
International conference on machine learning, pp. 23965–
23998. PMLR, 2022b.
Xie, Q., Chen, Q., Chen, A., Peng, C., Hu, Y ., Lin, F.,
Peng, X., Huang, J., Zhang, J., Keloth, V ., et al. Me-
llama: Foundation large language models for medical
applications.Research square, pp. rs–3, 2024.
Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal,
M. Resolving interference when merging models.arXiv
preprint arXiv:2306.01708, 1, 2023.
Yang, E., Shen, L., Guo, G., Wang, X., Cao, X., Zhang, J.,
and Tao, D. Model merging in llms, mllms, and beyond:
Methods, theories, applications, and opportunities.ACM
Computing Surveys, 2024.
12

<!-- page 13 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
A. Optimal Transport Formulation Details
This appendix provides the full formulations underlying the
transport plans used for model merging. We present both
the neuron-level transport within each layer pair and the
global layer-level mixing transport across the full depth.
A.1. Neuron-Level Transport Polytope and Objective
To manage computation and filter out noise, we apply top-k
selection strategies at both the neuron and transport matrix
levels.
Transport Matrix Sparsification.For a target layer ℓ
with nℓ feature channels and a source layer m with n′
m fea-
ture channels, we define a cost matrix C ℓm
inner ∈R nℓ×n′
m,
where each entry measures the dissimilarity between activa-
tion channels (Eq. 7).
We compute a soft relationship matrix Qℓm ∈R nℓ×n′
m
by solving an entropically regularized optimal transport
problem:
Qℓm = arg min
Q∈T(n ℓ,n′m)
⟨C ℓm
inner, Q⟩ −εH(Q),(23)
where ε >0 , ⟨A, B⟩= P
i,j AijBij, and the entropy term
is
H(Q) =−
nℓX
i=1
n′
mX
j=1
Qij
 
logQ ij −1

.(24)
The transportation polytope enforces balanced marginals:
T(n ℓ, n′
m) =
n
Q∈R nℓ×n′
m
+ :Q1=a, Q ⊤1=b
o
,
(25)
where we use uniform marginals
a= 1
nℓ
1∈R nℓ , b= 1
n′m
1∈R n′
m .(26)
Uniform marginals ensure that each target channel dis-
tributes equal total mass and each source channel receives
equal total mass, preventing degenerate, one-sided match-
ings.
A.2. Layer-Level Transport for Global Mixing
Given neuron-level solutions {Qℓm}, we form the layer-
level cost matrixC layer ∈R L×M by
Clayer[ℓ, m] =⟨C ℓm
inner, Qℓm⟩.(27)
We then compute the global layer mixing matrix P∈
RL×M via
P= arg min
P∈T(L,M)
⟨Clayer, P⟩ −ηH(P),(28)
whereη >0and
T(L, M) =
n
P∈R L×M
+ :P1= ¯a, P ⊤1= ¯b
o
,(29)
with uniform marginals
¯a= 1
L 1∈R L, ¯b= 1
M 1∈R M .(30)
These constraints enforce global balance: each target layer
allocates its mass across source layers, and each source
layer receives comparable total mass, improving utilization
during merging.
A.3. Sinkhorn Solver and Stabilization
We solve the entropically regularized OT problems in
Eqs. (23) and (28) using Sinkhorn iterations. Below we
summarize the scaling-form derivation and the iterative up-
dates.
A.3.1. SCALINGFORM
Consider the generic entropic OT problem
min
Q∈Rn×m
+
⟨C, Q⟩ −εH(Q)s.t.Q1=a, Q ⊤1=b.
(31)
Define the Gibbs kernel
K= exp(−C/ε).(32)
The optimal solution has the form
Q⋆ = diag(u)Kdiag(v),(33)
where u∈R n
+ and v∈R m
+ are scaling vectors chosen to
match the marginals.
A.3.2. SINKHORNITERATIONS
Given K, Sinkhorn iterations alternately normalize rows
and columns to satisfy the marginal constraints:
u←a./(Kv), v←b./(K ⊤u),(34)
where ./ denotes element-wise division. After convergence,
Q= diag(u)Kdiag(v)satisfiesQ1≈aandQ ⊤1≈b.
For the neuron-level problem, we useC=C ℓm
inner, a= 1
nℓ
1,
b= 1
n′m
1. For the layer-level problem, we use C=C layer,
¯a= 1
L 1, ¯b= 1
M 1.
A.3.3. PRACTICALSTABILIZATION
In practice, the kernel K= exp(−C/ε) may suffer from
numerical underflow when costs are large or ε is small. We
therefore optionally employ standard stabilization strate-
gies: (i) log-domain Sinkhorn updates, and/or (ii) periodic
rescaling of (u, v). We stop iterations when the maximum
marginal violation max{∥Q1−a∥ ∞,∥Q ⊤1−b∥ ∞} falls
below a tolerance.
13

<!-- page 14 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
A.3.4. HYPERPARAMETERSETTINGS
We solve the entropically regularized OT problems using
the Sinkhorn algorithm with the following specific hyperpa-
rameters:
Inner OT (Feature Alignment).For computing Qℓm
(Eq. (23)):
• Regularization (ε): We use ε= 0.1 for standard text
datasets (e.g., Malay, Cantonese) and a tighter ε=
0.03for the GSM8K math reasoning task.
• Iterations: We use a memory-efficient streaming
Sinkhorn solver with fixed200iterations.
• Tolerance: Convergence tolerance is set to10 −6.
Outer OT (Layer Mixing).For computingP(Eq. (28)):
• Regularization (η): We set the default η= 0.1 . We
employ an adaptive scaling mechanism where η is in-
creased if the maximum value of the cost matrix Clayer
exceeds 1000, ensuring numerical stability.
• Iterations: We allow up to1000iterations.
• Tolerance: We use a strict convergence tolerance of
10−9 with a numerical stability epsilon of10 −12.
A.4. Proofs for Representation-Space Interpretation
Proof of Theorem 4.1.By definition,
fW ℓm
B→AhA = (Φℓm
outW m
B Φℓm
in )hA = Φℓm
out
 
W m
B (Φℓm
in hA)

,
which is exactly Eq. (21).
Proof of Corollary 4.2. Assume Φℓm
in and Φℓm
out are invert-
ible. Let U:R dℓ
A,in →R dℓ
A,out be any linear map such
that
U hA = Φℓm
out

W m
B
 
Φℓm
in hA

for allh A.
Then
U= Φ ℓm
outW m
B Φℓm
in =fW ℓm
B→A,
establishing uniqueness.
A.5. Neuron Selection for Replacement.
For the top-k neuron replacement strategy, we set the default
number of neurons to k= 128 . The selection rule matches
the main text: we run the target model on the calibration
set D (used for transport estimation) and record neuron
activations via forward hooks. For each neuron j in layer
ℓ, we compute its activation strength as the mean absolute
activation over theTsamples,
sj = 1
T
TX
t=1
ht,j
,
where ht,j denotes the activation of neuron j on sample t.
We then select the indices of the k neurons with the highest
sj.
A.6. Computational Complexity
We analyze the computational cost of the proposed optimal
transport (OT)–based alignment in terms of feature- and
layer-level transport.
Feature-level OT.For each target layer ℓ and source layer
m, we solve an entropically regularized OT problem be-
tween nℓ target features and n′
m source features using
Sinkhorn iterations. Each Sinkhorn update involves ma-
trix–vector multiplications with the kernel K ℓm ∈R nℓ×n′
m,
incurring O(nℓn′
m) time per iteration. With Iin Sinkhorn
iterations, the total cost of one inner OT problem is
O(Iin nℓn′
m).
Across all layer pairs, the total feature-level OT cost is
O

Iin
LX
ℓ=1
MX
m=1
nℓn′
m
!
.
Layer-level OT.The layer-level transport problem operates
on a cost matrix Clayer ∈R L×M . Each Sinkhorn iteration
costs O(LM) , and with Iout iterations, the total cost is
O(Iout LM), which is negligible compared to feature-level
OT whenn ℓ, n′
m ≫L, M.
Overall complexity and practicality.In practice, Iin and
Iout denote the numbers of Sinkhorn iterations for the inner
and outer OT problems (not to be confused with T , the
number of samples used for activation extraction in Eq. (6)).
We fix Iin and Iout to small constants (e.g., 200 and 1000),
and feature dimensions are moderate for selected layers and
projection modules. Moreover, OT estimation is performed
onceusing a small dataset and does not introduce overhead
during inference. As a result, the OT-based alignment adds
a one-time preprocessing cost and remains practical for
cross-architecture fusion.
B. Experimental Setup
In this section, we detail the datasets used for training our
models and the comprehensive benchmarks employed to
evaluate their performance across diverse domains and lan-
guages.
14

<!-- page 15 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
B.1. Stimulus and Training Datasets
We use stimulus datasets to extract representative activations
for correspondence estimation, and training datasets for
task-specific adaptation when required. Both are tailored to
each domain and language. For each dataset, we randomly
sample 2000 examples.
B.1.1. DOMAIN-SPECIFIC
• Medical LLaMA3:We employ a specialized medical
dataset (medical LLaMA3) (Shekswess, 2024) contain-
ing high-quality biomedical literature, clinical case re-
ports, and medical guidelines to instill domain-specific
medical knowledge.
• Finance:To cover the financial domain, we utilize
a dedicated finance dataset (finance) (Flowers, 2025)
comprising financial news, reports, and economic anal-
ysis texts, enabling the model to grasp complex eco-
nomic concepts and terminology.
B.1.2. MULTILINGUALLANGUAGES
• Thai:We utilize fineweb thai (Penedo et al., 2024), a
subset of the FineWeb corpus focused on high-quality
Thai web text, to improve language modeling perfor-
mance in Thai.
• Indonesian:We incorporate indonesian conversation,
a dataset designed to capture natural dialogue and con-
versational nuances in the Indonesian language.
• Malay:For the Malay language, we use malaysian sft
(Mesolitica, 2024), a supervised adaptation dataset tai-
lored to improve instruction following in Malaysian
contexts.
• Cantonese:We include a dedicated cantonese dataset
(Lynn, 2024) to address the linguistic specificities and
colloquialisms of this regional language.
B.2. Evaluation Benchmarks
We evaluate our model on a diverse set of benchmarks,
categorized by language and domain to provide a holistic
view of performance.
B.2.1. MALAYBENCHMARKS
• MMLU (Malay):To assess general world knowledge
in Malay, we utilize the Malay version of the Massive
Multitask Language Understanding (MMLU) bench-
mark (Poh et al., 2024). This task evaluates the model’s
accuracy in a zero-shot setting across a wide range
of subjects—including STEM, humanities, and social
sciences—adapted to the Malay language to test cross-
lingual knowledge transfer.
B.2.2. INDONESIANBENCHMARKS
• ARC (Indonesian):The AI2 Reasoning Challenge
(ARC) (Clark et al., 2018) consists of grade-school
science questions designed to test complex reasoning
and scientific knowledge. We utilize the Indonesian
version to evaluate the model’s reasoning capabilities
in a low-resource language context.
• Belebele (Indonesian):Belebele (Bandarkar et al.,
2024) is a multilingual machine reading comprehen-
sion (MRC) dataset. Based on the FLORES-200
passages, it evaluates the model’s ability to answer
multiple-choice questions given a specific context. We
report results on the Indonesian (ind Latn) subset.
• TruthfulQA-MC (Indonesian):TruthfulQA (Lin
et al., 2022) evaluates the model’s truthfulness and its
tendency to generate imitative falsehoods. We use the
Indonesian translated version and report the average
accuracy across the single-true (MC1) and multi-true
(MC2) multiple-choice tasks.
• XCOPA (Indonesian):The Cross-lingual Choice of
Plausible Alternatives (XCOPA) (Ponti et al., 2020)
assesses causal commonsense reasoning. The task re-
quires the model to identify the correct cause or effect
given a premise. We evaluate performance on the In-
donesian subset (xcopa id).
B.2.3. CANTONESEBENCHMARKS
• CMMLU (Cantonese):We evaluate the model’s
proficiency in the Cantonese (Yue) dialect using the
CMMLU dataset from the Yue-Benchmark suite (Jiang
et al., 2025). This benchmark consists of multiple-
choice questions covering diverse disciplines such as
history, physics, and culture. It is specifically designed
to test the model’s ability to understand and reason
with Cantonese-specific vocabulary, grammar, and col-
loquialisms.
B.2.4. THAIBENCHMARKS
• MMLU (Thai):We assess general world knowledge
in Thai from the MMLU benchmark (Hendrycks et al.,
2020) (adapted as mmlu prox lite th other).
This subset comprises diverse multiple-choice ques-
tions across various uncategorized domains, evaluating
the model’s breadth of knowledge beyond specific aca-
demic disciplines in the Thai language.
• XCOPA (Thai):To evaluate causal reasoning in Thai,
we utilize the Thai subset (xcopa th) of the XCOPA
benchmark (Ponti et al., 2020). Similar to the Indone-
sian task, the model must select the most plausible al-
ternative between cause and effect based on a premise.
15

<!-- page 16 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
• MGSM (Thai):The Multilingual Grade School Math
(MGSM) benchmark (Shi et al., 2022) evaluates arith-
metic reasoning capabilities. We employ the Chain-of-
Thought (CoT) prompting setting on the Thai subset
(mgsm cot) to assess the model’s ability to perform
multi-step mathematical reasoning in Thai.
B.2.5. FINANCEBENCHMARKS
To assess the model’s capabilities in the finance and business
domains, we utilize three relevant subsets from the MMLU
benchmark (Hendrycks et al., 2020):
• MMLU (Business Ethics):This task
(global mmlu full en business ethics)
evaluates the model’s ability to apply ethical principles
in business scenarios, covering topics such as corpo-
rate governance, moral reasoning, and professional
standards.
• MMLU (Microeconomics):This subset
(global mmlu full en high school
microeconomics) covers fundamental eco-
nomic concepts including supply and demand, market
structures, and consumer behavior, reflecting a high
school level understanding of microeconomic theory.
• MMLU (Professional Accounting):Designed
to test advanced accounting knowledge, this task
(global mmlu full en professional
accounting) includes questions on financial
accounting, reporting standards, and auditing,
corresponding to professional certification levels.
B.2.6. MEDICALBENCHMARKS
We evaluate domain-specific medical knowledge using three
subsets from the Massive Multitask Language Understand-
ing (MMLU) benchmark (Hendrycks et al., 2020):
• MMLU (Anatomy):This task tests the model’s knowl-
edge of human anatomy through multiple-choice ques-
tions covering various body systems and structures,
derived from academic and professional sources.
• MMLU (Medical Genetics):This subset evaluates the
model’s understanding of genetic principles, hereditary
diseases, and clinical genetics, requiring specialized
medical knowledge to answer correctly.
• MMLU (Professional Medicine):This task assesses
the model’s ability to apply medical knowledge in pro-
fessional contexts. It includes questions typical of med-
ical board examinations, covering diagnosis, treatment,
and clinical decision-making.
B.3. General Capabilities
To assess the model’s fundamental reasoning and common-
sense abilities independent of specific domains or languages,
we employ the following benchmarks:
• ARC-Easy (Clark et al., 2018):The AI2 Reasoning
Challenge (Easy Set) consists of grade-school science
questions. It is designed to test the model’s ability
to answer questions that require basic commonsense
reasoning and world knowledge.
• CommonsenseQA (Talmor et al., 2019):This dataset
evaluates commonsense reasoning through multiple-
choice questions that require prior knowledge to dis-
tinguish between plausible answers, challenging the
model’s understanding of semantic relationships.
• PIQA (Bisk et al., 2020):The Physical Interaction QA
(PIQA) benchmark focuses on physical commonsense
reasoning. It presents the model with everyday situa-
tions and asks it to predict the most plausible physical
outcome or interaction.
• Social IQA (Sap et al., 2019):To evaluate social in-
telligence, we use Social IQA, which tests the model’s
ability to reason about social interactions, including
understanding motivations, emotional reactions, and
likely next steps in social contexts.
• WinoGrande (Sakaguchi et al., 2021):This bench-
mark is a large-scale dataset for commonsense rea-
soning, formulated as fill-in-the-blank problems. It
is designed to be more robust against annotation arti-
facts than the original Winograd Schema Challenge,
requiring deep understanding to resolve ambiguous
pronouns.
C. Details about Experiments
C.1. Visualization of Feature-Level Transport Maps
We provide qualitative visualizations of feature-level opti-
mal transport maps to further illustrate cross-architecture
representation similarity across heterogeneous architectures.
Figure 7 and Figure 8 visualize representative transport
plansQ ℓm computed between target and source layers.
Each entry of Qℓm encodes the amount of transport mass be-
tween a target neuron (horizontal axis) and a source neuron
(vertical axis), where brighter values indicate stronger cross-
architecture feature similarity. Across both language set-
tings, the transport mass is highly non-uniform and concen-
trates on sparse, localized regions rather than being evenly
distributed. Only a small subset of neuron pairs exhibits
strong transport values, while the majority of entries remain
close to zero.
16

<!-- page 17 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Figure 7.Feature-level optimal transport map between a LLaMA-
3-1B model and a Chinese LLaMA-8B model at layer 5 (K projec-
tion). Brighter values indicate stronger neuron-level alignment.
Figure 8.Feature-level optimal transport map between a
Malaysian-version LLaMA-3-1B model and a LLaMA-8B model
at layer 0 ( K projection). Similar to Cantonese, transport mass
concentrates on sparse, high-similarity neuron pairs.
These sparse yet structured patterns indicate that heteroge-
neous language models share well-aligned internal represen-
tations at the neuron level, rather than exhibiting random or
diffuse correspondence. Such visual evidence complements
our quantitative analysis and supports the design choice of
top-k neuron replacement, as the most informative cross-
architecture alignments are concentrated in a small number
of highly similar neuron pairs.
C.2. Details about Comparison with SFT training and
Distillation.
Evaluation Protocol and Normalization.To enable a
fair comparison across domains with heterogeneous score
ranges, we normalize results on a per-domain basis. Specif-
ically, let xd,m denote the raw performance of method m
on domain d. We apply two complementary normalization
schemes.
Z-score normalization.For each domaind, we compute
zd,m = xd,m −µ d
σd
,
where µd and σd are the mean and standard deviation of
scores across all methods for domain d. This normalization
highlights relative performance within each domain and
is used to visualize method-wise trajectories and average
trends. Z-scores are shown in the top row (domain-specific
trajectories) and the bottom-right panel (average across do-
mains).
Min–max scaling.We additionally apply min–max normal-
ization per domain,
ˆxd,m = xd,m −min d
maxd −min d
,
where mind and maxd denote the minimum and maximum
scores across methods for domain d. This scaling preserves
the relative shape of improvements within each domain and
is used in the bottom-left panel to compare improvement
patterns across domains.
Figure Interpretation.The top row of Figure 1 shows
z-score–normalized performance trajectories for representa-
tive domains (Cantonese, Malay, and Thai), illustrating how
different training strategies compare within each domain.
The bottom-left panel visualizes min–max–scaled scores
across all domains, highlighting the relative improvement
patterns of each method. The bottom-right panel reports
the average z-score across domains, summarizing overall
cross-domain performance.
Together, these normalized views demonstrate that our
adapted merging method consistently achieves the strongest
relative performance across domains, while SFT and distil-
lation exhibit more domain-dependent variability.
D. Additional Discussion
Q1: How sensitive is the method to the choice of the cal-
ibration set D?Our framework estimates transport plans
{Qℓm} and P from activation statistics, and therefore relies
on D to elicit representative behaviors of the target task or
domain. If D is severely mismatched, the resulting corre-
spondences may become noisy and less informative. To
reduce this sensitivity, our design explicitly incorporates
several stabilizing choices: (i) using a moderately sized but
task-relevant calibration set, (ii) employing entropic regu-
larization in OT to smooth noisy similarity estimates, and
(iii) restricting fusion to a small set of top- k highly acti-
vated neurons to limit interference. In practice, we find
that lightweight data from the target language or domain
is usually sufficient. When no such data is available, ap-
proaches that rely on explicit supervision (e.g., distillation
or adapters) may be more appropriate for that setting.
17

<!-- page 18 -->

Transport and Merge: Cross-Architecture Merging for Large Language Models
Q2: Under what conditions can cross-architecture fu-
sion fail despite sharp OT correspondences?Even when
OT yields seemingly confident correspondences, effective
fusion is not guaranteed if the target model lacks sufficient
capacity to absorb the injected knowledge, or if the trans-
ferred features are misaligned with the target task. These
cases reflect fundamental capacity or task-mismatch con-
straints rather than failures of the transport mechanism itself.
Our method adopts a conservative strategy—top- k neuron
replacement combined with optional residual-frozen adapta-
tion—to mitigate such risks by limiting the scope of transfer
and allowing controlled recalibration of the remaining pa-
rameters.
Q3: Can fusion propagate undesirable behaviors or bi-
ases from the source model?As with any direct parameter
transfer method, our approach may propagate both bene-
ficial capabilities and undesirable behaviors present in the
source model. This is an inherent consideration of weight-
level reuse rather than a property unique to our framework.
In practical deployments, we recommend applying standard
post-hoc safety and alignment pipelines (e.g., safety evalu-
ation, filtering, or targeted alignment) to the fused model.
Incorporating safety-aware objectives directly into the trans-
port formulation is an interesting direction for future work.
18
