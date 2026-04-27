# references/122_entropy_meets_importance_unified_head_importance_entropy_score_for_stable_efficient_transformer_pruning.pdf

<!-- page 1 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for
Stable and Efficient Transformer Pruning
Minsik Choi * 1 Hyegang Son * 1 Changhoon Kim 2 Young Geun Kim1
Abstract
Transformer-based models have achieved remark-
able performance in NLP tasks. However, their
structural characteristics—multiple layers and at-
tention heads—introduce efficiency challenges in
inference and deployment. To address these chal-
lenges, various pruning methods have recently
been proposed. Notably, gradient-based methods
using Head Importance Scores (HIS) have gained
traction for interpretability, efficiency, and ability
to identify redundant heads. However, HIS alone
has limitations as it captures only the gradient-
driven contribution, overlooking the diversity of
attention patterns. To overcome these limitations,
we introduce a novel pruning criterion,HIES
(Head Importance-Entropy Score), which in-
tegrates head importance scores with attention
entropy, providing complementary evidence on
per-head contribution. Empirically, HIES-based
pruning yields up to 15.2% improvement in model
quality and 2.04× improvement in stability over
HIS-only methods, enabling substantial model
compression without sacrificing either accuracy
or stability. Code will be released upon publica-
tion.
1. Introduction
Recent advances in Large Language Models (LLMs) have
led to remarkable performance. In pursuit of better mod-
eling of long-range dependencies, LLMs have scaled up
both context lengths and attention head counts, guided by
empirical scaling laws that correlate model capacity with
performance (Kaplan et al., 2020; Chen et al., 2025b). This
scaling, however, incurs substantial computational and mem-
ory costs during inference, resulting in prohibitive latency
*Equal contribution 1 Department of Computer Science and
Engineering, Korea University, Seoul, South Korea 2 School of
Software, Soongsil University, Seoul, South Korea . Correspon-
dence to: Changhoon Kim <changhoon.kim@ssu.ac.kr>, Young
Geun Kim<younggeun kim@korea.ac.kr>.
Preprint. February 3, 2026.
and energy consumption (Yang et al., 2020; Kim and Wu,
2020; Hoefler et al., 2021; Zhou et al., 2024). These con-
straints become critical barriers when LLMs are deployed to
resource-constrained environments such as consumer-grade
mobile devices or edge devices, for applications including
real-time translation, intelligent voice assistants, and per-
sonalized recommendation systems.
To improve deployability of LLMs for resource-constrained
environments, various pruning methods have been pro-
posed (Ma et al., 2023; Yang et al., 2024). Typically, these
methods selectively reduce computations by removing less
important weights, channels, or attention heads. Among
them,head pruninghas gained considerable attention due
to its structural simplicity, interpretability, and ability to
directly target redundancy within the attention mechanism.
Existing head pruning methods typically identifies less im-
portant heads based on Head Importance Score (HIS), which
quantifies the gradient-based contribution of each head to
the loss function. By leveraging gradient-based sensitivity
to the loss, HIS prioritizes heads that have the most direct
impact on accuracy of model inference.
However, HIS-based methods often exhibit limited stability
in their performance. For clarity, prior works (Bair et al.,
2023; Blanchet et al., 2024) motivate treating stability as
a practical surrogate for robustness—namely, a model’s
resilience to input perturbations and pruning-induced distri-
butional shifts. Such stability is crucial in real-world deploy-
ments where distribution shifts are common and aggressive
compression is often required. In our observations, this
instability appears to stem from two key factors. First, exist-
ing HIS-based methods solely rely on the loss gradient with
respect to each head’s output, which fails to capture token-
level attention allocation or its alignment with the task’s
empirical distribution. Consequently, a concentrated head
and a diffuse head can yield similar important scores, con-
cealing their functionally distinct roles on the task-specific
data manifold. Second, a uniform, layer-agnostic criterion
precludes layer-specific adaptation despite evidence that dif-
ferent layers require distinct attention behaviors (Artzy and
Schwartz, 2024). Lacking such layer-specific characteristics
often results in imbalanced pruning—preserving redundant
heads in some layers while removing functionally critical
1
arXiv:2510.13832v2  [cs.CL]  2 Feb 2026

<!-- page 2 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
ones in others. This imbalance not only degrades accuracy
but also undermines stability, leading to unpredictable per-
formance fluctuations across inputs or compression levels,
particularly under aggressive pruning ratios.
This work aims to address the aforementioned limitations
by proposing anEntropy-Aware Pruning Criterion, termed
HIES (Head Importance-Entropy Score), which jointly
considers a head’s gradient-based contribution to the loss
and the distributional structure of its attention—specifically,
the extent to which its attention is concentrated or dispersed
across input tokens. We compute the HIS to quantify a
head’s loss relevance and Attention Entropy (AE) to mea-
sure how evenly a head distributes attention over input to-
kens. Their principled combination in HIES enables layer-
adaptive pruning decisions and preserves functionally im-
portant heads. This allows for more balanced pruning across
layers, improving both accuracy and stability under aggres-
sive compression. Empirically, HIES yields up to a 15.2%
improvement in model quality and 2.04× improvement in
stability over HIS-only methods. By preserving both ac-
curacy and stability even under aggressive pruning ratios,
HIES represents a more practical and robust solution com-
pared to existing pruning methods. It is expected to offer
more stable performance in resource-constrained environ-
ments.
2. Background
Attention head pruning.To compress large language
models efficiently, structured pruning methods (Han et al.,
2015; Wang et al., 2019; Xia et al., 2022; Ma et al., 2023;
Ashkboos et al., 2024), which remove specific architectural
components from Transformer models, have been widely
adopted. Among these, attention head pruning has gained
traction. This is largely because it directly reduces attention
FLOPs and KV-cache memory while preserving the layer
topology, thereby simplifying checkpoint compatibility and
serving integration. Consequently, large-scale studies adopt
head-level pruning as a practical axis in LLM compres-
sion pipelines (Jaradat et al., 2024; Muralidharan et al.,
2024)1. Attention head pruning removes selected heads
from a trained Transformer’s multi-head attention with min-
imal impact on end-task performance (Vaswani et al., 2017).
A widely adopted criterion is the HIS of Michel et al. (2019),
which introduces mask variables mh ∈ {0,1} multiplying
the output of head h and defines importance as the expected
first-order loss increase under masking:
HISh =E x∼D

∂L(x)
∂mh
 =E x∼D
Ah(x)⊤ ∂L(x)
∂Ah(x)
 ,(1)
1For more detailed discussions on related work, please refer to
Appendix A.
where D denotes an input sample drawn from the data dis-
tribution D, L(x) is the loss for sample x, and Ah(x) is
the output of head h. The second equality follows from the
chain rule and the observation that gating scales the head’s
activation. Heads are then ranked by HISh and pruned in
ascending order of importance.
Attention Entropy and Stability.Zhai et al. (2023) quan-
tify the concentration of each attention head’s focus over
input tokens via the entropy of its attention weight distribu-
tion AEh = (H
 
p(h)
=− Pn
i=1 p(h)
i logp (h)
i , where p(h)
i
is the normalized attention probability assigned by head h
to the i-th input token subject to Pn
i=1 p(h)
i = 1. Higher
entropy indicates a diffuse focus over the sequence, whereas
lower entropy corresponds to highly concentrated attention
patterns. Their empirical findings reveal a strong correlation
between persistently low entropy (i.e., entropy collapse) and
instability during training, including oscillations in the loss
landscape and even divergence across various model scales
and tasks.
3. Motivation
Pruning Transformer models is most commonly driven by
gradient-based criteria, such as HIS and variants used in re-
cent pruning frameworks (e.g., LLM-Pruner) (Michel et al.,
2019; Ma et al., 2023). While gradient-based methods are
often effective at moderate sparsity, they exhibit sharp ac-
curacy degradation once the pruning ratio exceeds a certain
threshold, as shown in Figure 1 (a). Suchsharp-dropshave
been widely observed across various attention variants and
tasks (Ma et al., 2023; Li et al., 2023a; Ghattas et al., 2025),
underscoring the generality of this phenomenon.
We focus on this “sharp-drop” regime and contrast two
groups of heads. The first group consists of low-HIS heads
that are pruned during the sharp-drop of accuracy. The
second group consists of high-HIS heads that remain un-
pruned. The attention score heatmap in Figure 1 (b) reveals
that some pruned heads (red table) assign high attention
scores—i.e., the weights computed by the softmax over
token-token similarity that indicate how strongly a token
attends to another—to sentiment-discriminative tokens (to-
kens relevant for classification). Nonetheless, these heads
are pruned due to their low HIS values and end up caus-
ing the sharp accuracy drop observed in Figure 1 (a). In
contrast, some unpruned heads (gray table) often allocate
strong attention to non-informative tokens. These heads,
however, have high HIS values and thus remain unpruned,
though they contribute little to overall model quality. These
analysis results demonstrate that the gradient-based HIS is
insufficient to capture the token-level attention score dis-
tributions (the detailed mathematical analysis is provided
in Section 4.2.1), thereby resulting in suboptimal pruning
decisions for heads focusing on decisive tokens.
2

<!-- page 3 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 1.Analysis of accuracy degradation and head behaviors under HIS-based pruning. In our diagnostic study, we analyze the
phenomena of pruning by HIS on BERT, focusing on detailed attention head behaviors during inference. (a) Accuracy curves of HIS-based
pruning on CoLA and SST-2. The bold color highlights the sharp-drop regime of HIS-based pruning. (b) Head-level analysis on SST-2
with a validation example misclassified by the HIS-pruned model. The attention score heatmap shows heads on each column, where
pruned low-HIS heads are indicated with colored layer–head labels and unpruned high-HIS heads with gray table. It then shows token-wise
distributions, where tokens deemed important for classification (e.g., sentiment-discriminative tokens in SST-2) are marked withO, and
non-critical tokens withX. The left plot shows the distribution of heads by normalized HIS, and the right plot shows the distribution by
normalized AE, where pruned low-HIS heads are highlighted with red boxes and unpruned high-HIS heads with gray boxes . See
Appendix C for experimental setup details.
Thesharp-dropobserved in pruning can be interpreted as a
collapse of structural diversity in attention behaviors, caused
by the elimination of heads that concentrate on decisive to-
kens. To capture and prevent such collapse of structural
diversity in attention, we employ attention entropy, a mea-
sure widely used to prevent policy collapse in reinforcement
learning (Bharadhwaj et al., 2020; Xiao et al., 2021; Wang
et al., 2025). As shown in the bottom-right plots of Fig-
ure 1 (b), incorporating AE helps retain low-entropy heads
by recognizing their concentrated focus on decisive tokens,
thereby preventing them from being pruned and mitigating
thesharp-dropin accuracy.
Building on our analysis, we posit that attention entropy
captures structural signals that reflect unstable behavior
during deployment, leading to the following hypothesis:
Attention entropy serves as an indicator of inference-time
stability, mitigating accuracy sharp-drops
In particular, low-entropy heads may correlate with in-
creased sensitivity to input perturbations, leading to unstable
predictions under distribution shifts. This perspective moti-
vates our investigation of entropy as a proxy for robustness
and consistency during inference.
4. Proposed Method
Figure 2 provides an overview of our proposedHead Im-
portance–Entropy Score (HIES). Figure 2 (a) outlines the
Transformer architecture, where HIS computes the impor-
tance score of each of the h attention heads within a Trans-
former block. Figure 2 (b) displays the heatmap of the
first-order, loss-based HIS. Combining the normalized HIS
and AE yields the HIES heatmap in Figure 2 (c), which inte-
grates complementary signals provide a more stable assess-
ment of attention heads across layers. This design captures
the key intuition of HIES: HIS can be reinforced by AE,
providing a more robust basis for pruning head selections.
Section 4.1 formalizes HIES, and Section 4.2 develops a
risk-decomposition analysis that clarifies the complemen-
tary roles of HIS and AE and motivates HIES as a robust
criterion for head selection in pruning.
4.1. Head Importance Entropy Score
We define theHead Importance–Entropy Score (HIES)
as a weighted combination:
HIESh =α dHISh + (1−α)(1− cAEh), α∈[0,1),(1)
whereα 2 is a tunable hyperparameter.
Min-Max NormalizationDirectly comparing raw HIS and
AE is inherently problematic, as the two metrics reside on
different scales and encode distinct types of signals. To en-
able meaningful integration and ranking, we applymin–max
normalizationto both metrics, rescaling their values to
the interval [0,1] : dHISh = HISh−min(HIS)
max(HIS)−min(HIS) , cAEh =
AEh−min(AE)
max(AE)−min(AE) . This distribution-agnostic normalization
improves cross-criterion interpretability; lower normalized
2To determine the optimal combination of HIS and AE for
each task, we adopt a task-specific tuning procedure, where the
trade-off hyperparameter α is systematically explored under each
compression setting. Sensitivity to α is analyzed in Appendix D.8.
3

<!-- page 4 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 2.Design overview of Head Importance-Entropy Score (HIES). Darker cells correspond to values closer to 1, while lighter cells
correspond to values closer to 0. Note the heatmap utilizes each metric across attention heads in BERT on the CoLA dataset.
scores denote higher pruning priority. Prior studies show
min–max scaling outperforms z-score standardization in sta-
bility and reproducibility across diverse tasks (De Amorim
et al., 2023; Lima and Souza, 2023).
4.2. Theoretical Analysis
We analyze pruning through a risk decomposition that
combines a loss-increase term controlled by HIS, with a
generalization-gap term upper-bounded in terms of AE via
its token-wise deficit. We further show that the gradients
of HIS and AE are orthogonal in expectation, indicating
complementary axes: magnitude of contribution (HIS) and
dispersion of attention (AE). This perspective motivates the
composite importance measure HIES. By retaining heads
with high HIES, we simultaneously minimize our theoreti-
cal bound and enhance pruning stability. Conceptually, this
analysis formalizes importance-based selection into a prin-
cipled framework and offers a rigorous rationale for HIES’s
safety and effectiveness.
4.2.1. LOSS-INCREASECONTROL VIAHEAD
IMPORTANCE(HIS)
Setup.Let n be the sequence length, |H| the number of
heads, dv the value dimension per head, and d=|H|d v
the model width (i.e., hidden size). For head h, let Ah ∈
Rn×dv denote the head output, i.e., the value-projected at-
tention representation Ah = softmax

QhK⊤
h√dk

Vh, where
Qh =XW Q
h ∈R n×dk, Kh =XW K
h ∈R n×dk, and
Vh =XW V
h ∈R n×dv are the query, key, and value
projections of the input X∈R n×dmodel, with parame-
ter matrices W Q
h ∈R dmodel×dk, W K
h ∈R dmodel×dk, and
W V
h ∈R dmodel×dv.
We then define y= Concat(A 1, . . . , A|H|)∈R n×d as
the pre-projection representation, which is subsequently
projected through W O ∈R d×d. Head removal is modeled
by mask variables mh ∈ {0,1} : δAh =−(1−m h)Ah
and δy= Concat(δA 1, . . . , δA|H|). Formal definitions and
implementation notes are deferred to Appendix B.1.2.
Head Importance Score (HIS).We define
HISh :=E x∼D

∂L(x)
∂mh

=E x∼D

∇Ah(x)L(x), A h(x)

F

=E D
h
| ⟨∇Ah L, A h⟩F |
i
.
(2)
This quantity is a first-order activation–gradient correlation
whose absolute value prevents cross-sample cancellation,
yielding the additive upper boundP
h HISh on the loss (cf.
Appendix B.1.6).
Lemma 4.1(Loss-increase upper bound under head mask-
ing).Let βy :=∥∇ 2
yL∥2. For any mask variables
{mh}|H|
h=1,
∆L :=E D

L(y+δy)− L(y)

≤
|H|X
h=1
(1−m h) HISh + βy
2
|H|X
h=1
(1−m h)∥A h∥2
F .
(3)
Moreover, underbinary (sigmoid) cross-entropywe have
∥∇2
zL∥2 ≤ 1
4; with the linear projection z=yW O, this
yieldsβ y ≤ 1
4 ∥W O∥2
2, hence
∆L ≤
|H|X
h=1
(1−mh) HISh + 1
8 ∥W O∥2
2
|H|X
h=1
(1−mh)∥A h∥2
F .
(4)
Remark.Formulticlass softmaxcross-entropy, ∥∇2
zL∥2 ≤
1
2 (cf. Appendix B.1.4); thus the quadratic coefficient be-
comes 1
4 instead of 1
8.
4

<!-- page 5 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Implication for pruning.Eq. (4) shows that, for a fixed
pruning fraction ρ= 1
|H|
P
h(1−m h), selecting heads with
the smallest HISh minimizes the dominant first-order term,
while the quadratic term is controlled by ∥W O∥2 (or block-
wise norms) and token-averaged activations. Under standard
normalization, the quadratic contribution is typically domi-
nated by the first-order term (cf. Appendix B.1.5), justifying
the use of HISh as a practical surrogate importance for head
pruning.
However, since Ah = AttnhVh, HISh in Eq. (2) depends
solely on Ah and ∇Ah L, without explicitly incorporating
the distribution Attnh. Consequently, two heads with very
different attention patterns (e.g., sharply focusing on tokens
versus spreading over tokens) can yield similar HISh when-
ever the resulting Ah is comparable. This is supported by
our empirical analysis in Section 3.
4.2.2. GENERALIZATIONGAP ANDATTENTION
ENTROPY(AE)
Setup.Let S={(x i, yi)}N
i=1 ∼ D. We write ES and ED
for empirical and population expectations. We assume the
per-example loss ℓ is Lℓ-Lipschitz in its first argument, a
standard assumption that yields stability-based generaliza-
tion bounds (Bousquet and Elisseeff, 2002; Shalev-Shwartz
and Ben-David, 2014; Hardt et al., 2016).
Notation (attention entropy deficit).For head h and
query token t∈[n(x)] , let α(h)
t (x)∈∆ n(x)−1 denote
the attention over keys and H(p) :=− P
j pj logp j. We
define the token-averaged, length-normalized deficit (AD)
ADh(x) := 1
n(x) logn(x)
n(x)X
t=1

logn(x)−H
 
α(h)
t (x)

= 1−AE h(x)∈[0,1].
(5)
Main bound (loss–entropy link).Let M:=
maxh maxj ∥Vh(j,:)∥ 2 and CAE :=
√
8M
p
|H|ρlogn
for a representative effective length n. For the pruned
modelf S,m, the expected generalization gap (G) satisfies
G:=E D

ℓ(fS,m(x), y)

−E S

ℓ(fS,m(x), y)

≤2L ℓ CAE
vuut
|H|X
h=1
(1−m h)E S

ADh(x)

+ B
N .
(6)
Interpretation.For a fixed pruning ratio ρ, pruning heads
with smaller deficit (i.e., higher entropy) minimizes the
bound’s increase; pruning low-entropy (high-deficit) heads
worsens it. Proof details, operator-norm assumptions, and
variable-length handling are deferred to Appendix B.2.
4.2.3. RISKUPPERBOUND ANDHIES MINIMIZATION
Composite Risk Bound.Given the HIES defined above,
the overall risk upper bound is
R(m) :=
|H|X
h=1
(1−m h)HIES h (7)
Pruning Objective (fixed budget).Let k:= (1−ρ)|H|
be the number of heads to retain. We solve the cardinality-
constrained selection problem
min
m∈{0,1}|H|
|H|X
h=1
(1−m h)HIES h s.t.
|H|X
h=1
mh =k.(8)
Lemma 4.2(Optimality).Selecting the k heads with the
largest HIES values (equivalently, pruning the|H|−k heads
with the smallest HIES values) yields the globally optimal
maskm ∗ that minimizes(7)subject to(8).
4.2.4. ORTHOGONALITY ANDCOMPLEMENTARITY
Lemma 4.3(Orthogonality).Let
uh := sign
 
α(h)⊤gh

gh, vh :=1+ logα (h),
˜uh :=P u h,˜v h :=P v h,
where P:=I− 1
n 11⊤ projects onto {w:1 ⊤w= 0} .
Assume Cov(˜uh,˜vh) = 0 (the cross-covariance matrix is
zero). If, in addition, either ES[˜uh] = 0 ormore generally
⟨ES[˜uh],E S[˜vh]⟩= 0 3, then
ES
h
e∇α(h)HISh, e∇α(h)AEh
i
= 0,(9)
i.e., the two gradient directions are orthogonal in expecta-
tion4.
Complementarity.Because the gradients point along sta-
tistically orthogonal directions, HIS captures the magnitude
of loss sensitivity whereas AE captures the dispersion of
attention. Thus, they serve complementary roles: HIS em-
phasizes the magnitude of contribution, while AE character-
izes distributional concentration. Combined, they balance
pruning minimally influential heads and preserving heads
important for generalization, underpinning HIES’s effective-
ness.
5. Experimental Results
5.1. Experimental Setup
Model.We use publicly available BERTbase checkpoints
that have been fine-tuned and released by prior work (Devlin
et al., 2019), and LLaMA-27B checkpoint from Hugging
3We provide an empirical sanity check supporting the as-
sumption, which is consistently validated on both BERTbase and
LLaMA-27B in Appendix D.1.
4Detailed derivations and preliminaries are deferred to Ap-
pendix B.4.
5

<!-- page 6 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Table 1.Experimental results with BERT base on natural language understanding task. We report percentage improvements in blue.
Pruning Method SST-2 CoLA MRPC QQP STS-B QNLI MNLI RTE Average
Ratio Accuracy Matthews corr F1 Score Accuracy Pearson corr Accuracy Accuracy Accuracy
0% BERT base 92.43 76.69 91.35 91.27 94.02 91.54 84.57 72.56 86.80
10%
Random 92.09 75.25 89.90 91.00 93.88 89.87 82.62 70.76 85.670.00%
AD 92.55 75.48 90.56 91.10 93.65 90.87 83.50 70.40 86.01+0.40%
HIS 91.7477.2190.65 91.23 94.0490.6384.04 71.84 86.42+0.88%
L2 90.37 74.00 83.78 69.23 83.12 60.81 67.75 49.82 72.36-15.54%
LLM-Pruner (Channel) 91.97 76.05 79.19 83.53 93.06 87.39 80.46 67.51 82.40-3.82%
LLM-Pruner (Block) 91.06 76.69 89.83 84.96 93.77 87.42 82.03 64.62 83.80-2.19%
SliceGPT (w/o tune) 51.38 49.86 81.46 63.18 58.92 53.91 36.84 49.46 55.63-35.07%
SliceGPT (w/ tune) 86.47 61.87 82.30 88.28 62.47 83.45 77.34 54.51 74.59-12.94%
HIES (ours) 92.66 75.48 91.04 91.93 94.00 91.03 84.04 71.84 86.50 +0.97%
30%
Random 90.29 69.02 85.27 84.00 92.90 79.80 75.15 60.29 79.590.00%
AD 86.58 50.00 84.53 84.57 81.94 67.82 77.00 56.68 73.64-7.47%
HIS 89.56 73.1789.3789.95 93.82 89.04 82.25 68.59 84.47+6.13%
L2 86.58 67.52 81.58 64.83 76.52 51.09 56.00 50.90 66.88-15.97%
LLM-Pruner (Channel) 88.53 70.36 86.89 81.23 92.50 67.38 66.67 64.26 77.23-2.97%
LLM-Pruner (Block) 88.99 73.93 84.76 80.09 93.40 82.68 78.33 66.79 81.12+1.92%
SliceGPT (w/o tune) 50.80 53.29 78.05 63.18 54.64 53.18 34.90 51.99 55.00-30.90%
SliceGPT (w/ tune) 83.49 60.14 81.80 85.80 60.89 67.36 75.50 54.87 71.23-10.49%
HIES (ours) 91.86 74.97 88.81 90.37 93.89 89.13 82.50 70.04 85.20 +7.04%
50%
Random 78.74 61.02 72.53 66.25 91.40 67.53 67.32 53.79 69.820.00%
AD 82.91 50.00 54.50 76.18 75.00 68.94 68.00 55.96 66.44-4.84%
HIS 87.27 59.48 86.52 85.9192.6182.6878.67 62.82 79.50+13.84%
L2 82.80 60.98 85.30 64.83 69.76 50.54 44.42 47.29 63.24-9.39%
LLM-Pruner (Channel) 86.47 61.64 83.92 81.47 89.74 60.66 67.42 62.82 74.27+6.37%
LLM-Pruner (Block) 87.84 70.0983.84 78.8092.6972.60 73.60 61.73 77.65+11.20%
SliceGPT (w/o tune) 50.92 52.79 81.22 63.19 47.70 50.98 34.93 50.90 54.08-22.54%
SliceGPT (w/ tune) 83.49 57.45 81.37 82.16 55.05 65.79 71.70 51.62 68.58-1.78%
HIES (ours) 90.71 68.52 86.80 85.73 92.65 82.68 79.00 65.34 81.43 +16.63%
Face (Touvron et al., 2023). To examine the generalizability
to attention variants and tasks, we further employ ViTLarge
(Dosovitskiy, 2020) and LLaV A-1.57B (Liu et al., 2023).
Datasets.We evaluate on various widely-adopted bench-
marks: GLUE (Wang et al., 2018), HellaSwag (Zellers
et al., 2019), Winogrande (Sakaguchi et al., 2021), the
AI2 Reasoning Challenge—ARC-e/ARC-c (Clark et al.,
2018), OBQA (Mihaylov et al., 2018), ImageNet1k (Deng
et al., 2009), CIFAR-100 (Krizhevsky et al., 2009), Food-
101 (Bossard et al., 2014), Fashion MNIST (Xiao et al.,
2017), MMLU (Hendrycks et al., 2020), GSM8K (Cobbe
et al., 2021), VizWiz-VQA (Gurari et al., 2018), and MM-
Vet (Yu et al., 2023).
Baselines.
•Random: Prune attention heads uniformly at random.
• L2-Norm: Prune attention heads with smaller weight
magnitudes under the ℓ2 norm. This criterion lever-
ages parameter norms as a direct measure of structural
salience.
• HIS(Michel et al., 2019): Prune attention heads with the
smallest head-importance first.
• Attention Deficit (AD; 1−Attention Entropy)(Zhai
et al., 2023): Prune attention heads with smaller attention
entropy first, i.e., heads exhibiting more concentrated
attention patterns.
• LLM-Pruner (Channel-wise)(Ma et al., 2023): Prune
attention-layer channels based on first-order Taylor ex-
pansion of the loss. Importance scores are computed per
attention channel, and pruning proceeds while preserving
the most critical channels.
• LLM-Pruner (Block-wise)(Ma et al., 2023): Extend the
channel-wise pruning strategy to whole attention blocks,
guided by global importance ranking. Following the best-
performing configuration reported in prior work, we re-
tain the first three layers and the final layer, pruning the
others. This variant also restricts pruning to attention
layers.
• SliceGPT: Project activations onto principal components
estimated from calibration data, removing directions cor-
responding to less important subspaces (Ashkboos et al.,
2024). This preserves semantic subspaces while reducing
redundancy.
5.2. Main Results
We evaluate HIES using two key metrics: model quality and
stability5. As reported in Table 1, HIES improves model
quality by 8.21% on average. Table 2 further demonstrates
a 3.3% average stability gain over HIS. Notably, at a prun-
5Experimental details are provided in Appendix C. We also
conduct sensitivity analysis across multiple random seeds and ob-
serve consistent improvements for bothBERTbase and LLaMA-27B.
Details are provided in Appendix D.2. Note that we use a cali-
bration size of 32 for computing HIES; additional analyses are
reported in Appendix D.3.
6

<!-- page 7 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Table 2.Stability results with BERTbase on GLUE tasks. We report stability (%) against the unpruned model. Percentage improvements
(in blue) are relative to HIS within each pruning ratio.
Pruning Ratio Method SST-2 CoLA MRPC QQP STS-B QNLI MNLI RTE Average
10%
HIS 97.7196.55 94.3697.30 99.4795.79 95.66 94.22 96.380.00%
L2 95.18 76.41 26.72 68.63 21.87 60.33 72.44 39.71 57.66-40.17%
AD 98.5196.36 93.6397.8089.0797.25 96.15 94.95 95.47-0.94%
LLM-Pruner 97.02 85.81 75.25 85.40 39.53 90.39 84.98 86.28 80.58-16.39%
HIES (ours) 98.17 96.36 94.12 97.80 96.67 97.25 95.74 93.50 96.20 -0.19%
30%
HIS 94.38 90.03 90.0394.77 78.93 92.95 91.5687.36 90.000.00%
L2 90.94 83.13 26.72 63.90 20.73 50.39 59.23 42.24 54.66-39.27%
AD 88.53 81.21 78.68 87.90 24.53 63.04 79.07 69.68 71.58-20.47%
LLM-Pruner 92.66 63.09 84.80 82.37 62.13 68.70 68.29 70.04 74.01-17.77%
HIES (ours) 97.13 93.38 89.46 94.97 86.67 93.23 91.56 88.81 91.90 +2.11%
50%
HIS 90.02 40.84 83.33 88.8069.13 85.54 81.4778.70 77.230.00%
L2 86.47 82.26 26.72 63.90 20.40 49.81 45.58 34.30 51.18-33.73%
AD 52.18 81.21 51.72 72.33 21.87 73.49 68.49 62.82 60.51-21.65%
LLM-Pruner 89.68 41.32 78.19 83.27 44.13 61.58 67.52 75.81 67.69-12.35%
HIES (ours) 95.07 83.13 84.56 88.27 75.20 85.54 78.29 81.23 83.91 +8.65%
Figure 3.Head-level pruning patterns on the CoLA dataset across
pruning ratios from 30% to 50% pruning ratios. Pruned heads are
shaded in gray.
ing ratio of 50%, HIES achieves gains of up to 15.2% in
model quality and 2.04× in stability compared to the best-
performing baseline. These results corroborate our theoret-
ical analysis, demonstrating that HIES preserves end-task
performance while markedly enhancing robustness, both
critical for reliable and efficient deployment.
5.3. Extended Experimental Results
5.3.1. HEADREMOVALPATTERNS(HEATMAP)
In our main results, HIES exhibits more stable performance
gains than HIS at aggressive pruning ratios (e.g., ≥,30%).
At lower ratios (e.g., ≤,10%), HIS and HIES perform com-
parably, while HIS achieves marginally higher accuracy in a
few cases. We posit distinct prioritization. HIS, a gradient-
based metric tends to retain redundant, low-risk heads at
low sparsity, while HIES adds attention entropy to capture
stability, thereby preserving specialized low-entropy heads
that enhance robustness.
Pruning heatmap analysis provides empirical support for this
distinction. As shown in Figure 3, HIS (left panels) tends to
remove heads primarily from the lower layers, producing an
approximately bottom-up pattern. This behavior is guided
by a one-step gradient saliency, which estimates the impor-
tance of each attention head based on a single backward
pass through the model— this assigns higher importance to
heads whose activations have a larger immediate effect on
the loss. Meanwhile, HIES (right panels) yields a more dis-
persed selection spanning lower, middle, and upper layers —
other tasks also exhibit similar patterns (details are provided
in Appendix D.4). We attribute this to the entropy-aware
term, which leverages structural properties of the attention
distribution (concentration vs. dispersion) in addition to gra-
dient sensitivity, thereby promoting diversity across layers
in pruning decisions. Consequently, HIES exhibits more
stable performance across pruning ratios, yielding flatter
accuracy–sparsity curves than HIS.
5.3.2. SCALABILITY TOLARGERTRANSFORMER
MODELS
On LLaMA-27B, we evaluate pruning on HellaSwag, Wino-
grande, ARC-Easy (ARC-e), ARC-Challenge (ARC-c), and
Open Book Question Answering (OBQA), comparing HIES
with HIS across pruning ratios of 10–60%. HIES improves
accuracy by up to +10.54% and stability by up to +6.21%
relative to HIS, averaged over tasks. This advantage persists
uniformly across pruning ratios, indicating that the same
pruning mechanism scales effectively to larger models with
higher head counts. Notably, ARC-c is the most difficult
benchmark in this suite, as reflected by its lowest base ac-
curacy. Even under this challenging setting, HIES achieves
consistent and often larger accuracy gains relative to HIS,
underscoring its robustness not only on easier tasks but also
on the hardest ones.
5.3.3. GENERALIZABILITY TOATTENTIONVARIANTS
ANDTASKS
Table 3 evaluates the generalizability of HIES across at-
tention variants and task domains, spanning vision classi-
fication with ViTLarge and visual-language reasoning with
7

<!-- page 8 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 4.Comparison of HIES (Ours) and HIS on HellaSwag, Winogrande, ARC-e, ARC-c and OBQA across pruning ratios from 10% to
60% (x-axis). The top row reports task accuracy, while the bottom row reports stability. The rightmost panels summarize the mean over
tasks.
Table 3.Experimental results with ViTLarge on image classification benchmarks and LLaV A-1.5Large on multi-modal tasks. Relative
improvements of HIES (Ours) over HIS are shown in blue.
Vision Transformer Model (ViTLarge) Vision-Langauge Model (LLaV A-1.57B)
Image Classification Visual Question AnsweringComplex Multimodal
ImageNet1k CIFAR-100 Avg. VizWiz-VQA MM-Vet Avg.
HIS HIES (Ours) HIS HIES (Ours) HIS HIES (Ours) HIS HIES (Ours) HIS HIES (Ours) HIS HIES (Ours)
10% 86.40% 84.40% 91.40% 92.40% 88.90% 88.40%-0.56% 10% 48.00% 47.33% 41.20% 39.40% 44.60% 43.37%-2.84%
20% 44.80% 80.20% 65.60% 85.20% 55.20% 82.70%49.82% 30% 44.00% 47.00% 29.80% 34.00% 36.90% 40.50%8.89%
30% 6.20% 55.40% 19.40% 40.00% 12.80% 47.70%256.66% 50% 32.67% 39.67% 24.60% 25.20% 28.64% 32.43%11.71%
LLaV A-1.57B.
First, across different attention-based variants, including
ViT and LLaV A, HIES consistently shifts the region of
sharp accuracy drop to more aggressive sparsity levels com-
pared to HIS. In particular, at high pruning ratios where HIS
accuracy collapses, HIES achieves 49.82% higher accuracy
on ViTLarge at 20% sparsity, compared to HIS (with 55.20%
of accuracy). This indicates that HIES effectively captures
structural importance regardless of the specific model con-
figuration, enabling stable and reliable pruning.
Second, the advantages of HIES are further extended on
complex multi-modal tasks. HIES shows improvement of
11.71% at 50% sparsity compared to HIS on VizWiz-VQA
and MM-Vet. HIS frequently suffers from sharp accuracy
drops on these tasks, as it relies solely on gradient-based
head importance and fails to account for the structural diver-
sity of attention heads. As a result, pruning based on HIS
alone may remove heads that are essential for preserving
cross-modal alignment and semantic grounding. In con-
trast, HIES leverages AE to capture these structural signals,
resulting in more stable performance across challenging
vision-language benchmarks.
Third, we evaluate HIES on large-scale reasoning bench-
marks, including GSM8K and MMLU, to assess its robust-
ness beyond classification-style tasks. The results show
consistent gains under reasoning-intensive settings and are
reported in Appendix D.6. We also report downstream eval-
uations on CIFAR-100, Food-101, and Fashion-MNIST,
demonstrating that the benefits of HIES extend to diverse
vision benchmarks. Detailed results are provided in Ap-
pendix D.7.
Overall, these results confirm that combining HIS with AE
produces pruning signals that generalize across architec-
tures and modalities, preserving accuracy and stability. This
underscores the potential of HIES as a broadly applicable
criterion for efficient and reliable pruning of Transformer
models.
6. Conclusion
In this paper, we present HIES, a novel pruning criterion
that jointly leverages gradient-based head importance and at-
tention entropy to better characterize per-head contributions.
By combining complementary structural and behavioral sig-
nals, HIES outperforms HIS and other baselines, delivering
both higher accuracy and greater inference-time stability.
Beyond empirical gains, our analysis highlights the critical
role of entropy. We believe HIES offers a principled direc-
tion for stable and efficient pruning of Transformer-based
models, with potential to extend toward broader structured
sparsity and large-scale model deployment.
8

<!-- page 9 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Impact Statements
This paper presents work whose goal is to advance the field
of machine learning. There are many potential societal
consequences of our work, none of which we feel must be
specifically highlighted here.
References
Argenis Arriojas, Jacob Adamczyk, Stas Tiomkin, and
Rahul V Kulkarni. Entropy regularized reinforcement
learning using large deviation theory.Physical Review
Research, 5(2):023085, 2023.
Amit Ben Artzy and Roy Schwartz. Attend first, consolidate
later: On the importance of attention in different llm
layers. InProceedings of the 7th BlackboxNLP Workshop:
Analyzing and Interpreting Neural Networks for NLP,
pages 177–184, 2024.
Saleh Ashkboos, Maximilian L Croci, Marcelo Gen-
nari do Nascimento, Torsten Hoefler, and James Hensman.
Slicegpt: Compress large language models by deleting
rows and columns.arXiv preprint arXiv:2401.15024,
2024.
Yonatan Ashlag, Uri Koren, Mirco Mutti, Esther Derman,
Pierre-Luc Bacon, and Shie Mannor. State entropy reg-
ularization for robust reinforcement learning.arXiv
preprint arXiv:2506.07085, 2025.
Haoli Bai, Wei Zhang, Lu Hou, Lifeng Shang, Jin Jin, Xin
Jiang, Qun Liu, Michael Lyu, and Irwin King. Binarybert:
Pushing the limit of bert quantization. InProceedings
of the 59th Annual Meeting of the Association for Com-
putational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1:
Long Papers), pages 4334–4348, 2021.
Anna Bair, Hongxu Yin, Maying Shen, Pavlo Molchanov,
and Jose Alvarez. Adaptive sharpness-aware pruning for
robust sparse networks.arXiv preprint arXiv:2306.14306,
2023.
Homanga Bharadhwaj, Kevin Xie, and Florian Shkurti.
Model-predictive control via cross-entropy and gradient-
based optimization. InLearning for Dynamics and Con-
trol, pages 277–286. PMLR, 2020.
Jose Blanchet, Peng Cui, Jiajin Li, and Jiashuo Liu. Stability
evaluation through distributional perturbation analysis. In
Forty-first International Conference on Machine Learn-
ing, 2024.
Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.
Food-101–mining discriminative components with ran-
dom forests. InEuropean conference on computer vision,
pages 446–461. Springer, 2014.
Olivier Bousquet and Andr´e Elisseeff. Stability and gener-
alization.Journal of machine learning research, 2(Mar):
499–526, 2002.
Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng
Gao, Kaipeng Zhang, and Ping Luo. Efficientqat: Effi-
cient quantization-aware training for large language mod-
els. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 10081–10100, 2025a.
Yingfa Chen, Yutong Wu, Chenyang Song, Zhen Leng Thai,
Xingyu Shen, Xu Han, Zhiyuan Liu, and Maosong Sun.
Cost-optimal grouped-query attention for long-context
modeling.arXiv preprint arXiv:2503.09579, 2025b.
Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christo-
pher D Manning. What does bert look at? an analysis of
bert’s attention.arXiv preprint arXiv:1906.04341, 2019.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind
Tafjord. Think you have solved question answering?
try arc, the ai2 reasoning challenge.arXiv preprint
arXiv:1803.05457, 2018.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark
Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.
Training verifiers to solve math word problems.arXiv
preprint arXiv:2110.14168, 2021.
Lucas BV De Amorim, George DC Cavalcanti, and
Rafael MO Cruz. The choice of scaling technique matters
for classification performance.Applied Soft Computing,
133:109924, 2023.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and
Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In2009 IEEE conference on computer vision
and pattern recognition, pages 248–255. Ieee, 2009.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. InProceedings of
the 2019 conference of the North American chapter of
the association for computational linguistics: human lan-
guage technologies, volume 1 (long and short papers),
pages 4171–4186, 2019.
Alexey Dosovitskiy. An image is worth 16x16 words: Trans-
formers for image recognition at scale.arXiv preprint
arXiv:2010.11929, 2020.
Angela Fan, Edouard Grave, and Armand Joulin. Reducing
transformer depth on demand with structured dropout.
arXiv preprint arXiv:1909.11556, 2019.
9

<!-- page 10 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Tamer Ghattas, Michael Hassid, and Roy Schwartz.
On pruning state-space llms.arXiv preprint
arXiv:2502.18886, 2025.
Mitchell Gordon, Kevin Duh, and Nicholas Andrews. Com-
pressing bert: Studying the effects of weight pruning on
transfer learning. InProceedings of the 5th Workshop on
Representation Learning for NLP, pages 143–155, 2020.
Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi
Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.
Vizwiz grand challenge: Answering visual questions from
blind people. InProceedings of the IEEE conference on
computer vision and pattern recognition, pages 3608–
3617, 2018.
Shuai Han, Wenbo Zhou, Shuai L¨u, Sheng Zhu, and Xiaoyu
Gong. Entropy regularization methods for parameter
space exploration.Information Sciences, 622:476–489,
2023.
Song Han, Jeff Pool, John Tran, and William Dally. Learn-
ing both weights and connections for efficient neural net-
work.Advances in neural information processing systems,
28, 2015.
Moritz Hardt, Ben Recht, and Yoram Singer. Train faster,
generalize better: Stability of stochastic gradient descent.
InInternational conference on machine learning, pages
1225–1234. PMLR, 2016.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Mea-
suring massive multitask language understanding.arXiv
preprint arXiv:2009.03300, 2020.
Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden,
and Alexandra Peste. Sparsity in deep learning: Pruning
and growth for efficient inference and training in neural
networks.Journal of Machine Learning Research, 22
(241):1–124, 2021.
Jonghyun Hong and Sungyoon Lee. Variance sensitivity
induces attention entropy collapse and instability in trans-
formers. InProceedings of the 2025 Conference on Em-
pirical Methods in Natural Language Processing, pages
8371–8389, 2025.
Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen,
and Qun Liu. Dynabert: Dynamic bert with adaptive
width and depth.Advances in Neural Information Pro-
cessing Systems, 33:9782–9793, 2020.
Ghadeer Jaradat, Mohammed Tolba, Ghada Alsuhli, Hani
Saleh, Mahmoud Al-Qutayri, Thanos Stouraitis, and
Baker Mohammad. Hybrid dynamic pruning: A path-
way to efficient transformer inference.arXiv preprint
arXiv:2407.12893, 2024.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B
Brown, Benjamin Chess, Rewon Child, Scott Gray,
Alec Radford, Jeffrey Wu, and Dario Amodei. Scal-
ing laws for neural language models.arXiv preprint
arXiv:2001.08361, 2020.
Young Geun Kim and Carole-Jean Wu. Autoscale: En-
ergy efficiency optimization for stochastic edge infer-
ence using reinforcement learning. In2020 53rd Annual
IEEE/ACM international symposium on microarchitec-
ture (MICRO), pages 1082–1096. IEEE, 2020.
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
layers of features from tiny images. 2009.
Eldar Kurtic, Daniel Campos, Tuan Nguyen, Elias Frantar,
Mark Kurtz, Benjamin Fineran, Michael Goin, and Dan
Alistarh. The optimal bert surgeon: Scalable and accurate
second-order pruning for large language models.arXiv
preprint arXiv:2203.07259, 2022.
Franc ¸ois Lagunas, Ella Charlaix, Victor Sanh, and Alexan-
der M Rush. Block pruning for faster transformers.arXiv
preprint arXiv:2109.04838, 2021.
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvinine-
jad, Abdelrahman Mohamed, Omer Levy, Veselin Stoy-
anov, and Luke Zettlemoyer. Bart: Denoising sequence-
to-sequence pre-training for natural language generation,
translation, and comprehension. InProceedings of the
58th annual meeting of the association for computational
linguistics, pages 7871–7880, 2020.
Bingbing Li, Zigeng Wang, Shaoyi Huang, Mikhail A Bra-
gin, Ji Li, and Caiwen Ding. Towards lossless head
pruning through automatic peer distillation for language
models. InIJCAI, pages 5113–5121, 2023a.
Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang, Pengcheng
He, Weizhu Chen, and Tuo Zhao. Losparse: Structured
compression of large language models based on low-rank
and sparse approximation. InInternational Conference on
Machine Learning, pages 20336–20350. PMLR, 2023b.
Felipe Tomazelli Lima and Vinicius MA Souza. A large
comparison of normalization methods on time series.Big
Data Research, 34:100407, 2023.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke
Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly
optimized bert pretraining approach.arXiv preprint
arXiv:1907.11692, 2019.
10

<!-- page 11 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Zejian Liu, Fanrong Li, Gang Li, and Jian Cheng. Ebert:
Efficient bert inference with dynamic structured prun-
ing. InFindings of the Association for Computational
Linguistics: ACL-IJCNLP 2021, pages 4814–4823, 2021.
Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner:
On the structural pruning of large language models.Ad-
vances in neural information processing systems, 36:
21702–21720, 2023.
Paul Michel, Omer Levy, and Graham Neubig. Are six-
teen heads really better than one?Advances in neural
information processing systems, 32, 2019.
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sab-
harwal. Can a suit of armor conduct electricity? a new
dataset for open book question answering.arXiv preprint
arXiv:1809.02789, 2018.
Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov.
Variational dropout sparsifies deep neural networks. InIn-
ternational conference on machine learning, pages 2498–
2507. PMLR, 2017.
Saurav Muralidharan, Sharath Turuvekere Sreenivas, Ravi-
raj Joshi, Marcin Chochowski, Mostofa Patwary, Moham-
mad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo
Molchanov. Compact language models via pruning and
knowledge distillation.Advances in Neural Information
Processing Systems, 37:41076–41102, 2024.
Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang,
Yaliang Li, and Jun Huang. Meta-kd: A meta knowl-
edge distillation framework for language model compres-
sion across domains. InProceedings of the 59th Annual
Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural
Language Processing (Volume 1: Long Papers), pages
3026–3036, 2021.
Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A
primer in bertology: What we know about how bert works.
Transactions of the association for computational linguis-
tics, 8:842–866, 2020.
Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea
Goldsmith, and Mert Pilanci. Compressing large lan-
guage models using low rank and low precision decom-
position.Advances in Neural Information Processing
Systems, 37:88981–89018, 2024.
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula,
and Yejin Choi. Winogrande: An adversarial winograd
schema challenge at scale.Communications of the ACM,
64(9):99–106, 2021.
Shai Shalev-Shwartz and Shai Ben-David.Understanding
machine learning: From theory to algorithms. Cambridge
university press, 2014.
Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient
knowledge distillation for bert model compression.arXiv
preprint arXiv:1908.09355, 2019.
Siqi Sun, Zhe Gan, Yuwei Fang, Yu Cheng, Shuohang Wang,
and Jingjing Liu. Contrastive distillation on intermedi-
ate representations for language model compression. In
Proceedings of the 2020 Conference on Empirical Meth-
ods in Natural Language Processing (EMNLP), pages
498–508, 2020.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,
Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288, 2023.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
Illia Polosukhin. Attention is all you need.Advances in
neural information processing systems, 30, 2017.
Elena V oita, David Talbot, Fedor Moiseev, Rico Sennrich,
and Ivan Titov. Analyzing multi-head self-attention: Spe-
cialized heads do the heavy lifting, the rest can be pruned.
arXiv preprint arXiv:1905.09418, 2019.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill,
Omer Levy, and Samuel Bowman. Glue: A multi-task
benchmark and analysis platform for natural language
understanding. InProceedings of the 2018 EMNLP work-
shop BlackboxNLP: Analyzing and interpreting neural
networks for NLP, pages 353–355, 2018.
Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokking
of implicit reasoning in transformers: A mechanistic jour-
ney to the edge of generalization.Advances in Neural
Information Processing Systems, 37:95238–95265, 2024.
Chaoqi Wang, Roger Grosse, Sanja Fidler, and Guodong
Zhang. Eigendamage: Structured pruning in the
kronecker-factored eigenbasis. InInternational confer-
ence on machine learning, pages 6566–6575. PMLR,
2019.
Jiawei Wang, Jiacai Liu, Yuqian Fu, Yingru Li, Xintao
Wang, Yuan Lin, Yu Yue, Lin Zhang, Yang Wang, and
Ke Wang. Harnessing uncertainty: Entropy-modulated
policy gradients for long-horizon llm agents.arXiv
preprint arXiv:2509.09265, 2025.
Jeffrey TH Wong, Cheng Zhang, Xinye Cao, Pedro
Gimenes, George A Constantinides, Wayne Luk, and
Yiren Zhao. A3: an analytical low-rank approx-
imation framework for attention.arXiv preprint
arXiv:2505.12942, 2025.
11

<!-- page 12 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Struc-
tured pruning learns compact and accurate models.arXiv
preprint arXiv:2204.00408, 2022.
Changnan Xiao, Haosen Shi, Jiajun Fan, and Shihong
Deng. An entropy regularization free mechanism for
policy-based reinforcement learning.arXiv preprint
arXiv:2106.00707, 2021.
Han Xiao, Kashif Rasul, and Roland V ollgraf. Fashion-
mnist: a novel image dataset for benchmarking machine
learning algorithms.arXiv preprint arXiv:1708.07747,
2017.
Dongkuan Xu, Ian EH Yen, Jinxi Zhao, and Zhibin Xiao.
Rethinking network pruning–under the pre-train and fine-
tune paradigm.arXiv preprint arXiv:2104.08682, 2021.
Dingqing Yang, Amin Ghasemazar, Xiaowei Ren, Maximil-
ian Golub, Guy Lemieux, and Mieszko Lis. Procrustes: a
dataflow and accelerator for sparse deep neural network
training. In2020 53rd Annual IEEE/ACM International
Symposium on Microarchitecture (MICRO), pages 711–
724. IEEE, 2020.
Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large
language model pruning via layer collapse.arXiv preprint
arXiv:2402.11187, 2024.
Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xi-
aoxia Wu, Conglong Li, and Yuxiong He. Zeroquant:
Efficient and affordable post-training quantization for
large-scale transformers.Advances in neural information
processing systems, 35:27168–27183, 2022.
Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang,
Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang.
Mm-vet: Evaluating large multimodal models for inte-
grated capabilities.arXiv preprint arXiv:2308.02490,
2023.
Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe
Wasserblat. Q8bert: Quantized 8bit bert. In2019 Fifth
Workshop on Energy Efficient Machine Learning and Cog-
nitive Computing-NeurIPS Edition (EMC2-NIPS), pages
36–39. IEEE, 2019.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi,
and Yejin Choi. Hellaswag: Can a machine really finish
your sentence?arXiv preprint arXiv:1905.07830, 2019.
Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan
Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu,
and Joshua M Susskind. Stabilizing transformer training
by preventing attention entropy collapse. InInternational
Conference on Machine Learning, pages 40770–40803.
PMLR, 2023.
Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming
Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan,
Xiuhong Li, et al. A survey on efficient inference for
large language models.arXiv preprint arXiv:2404.14294,
2024.
12

<!-- page 13 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Appendix
Table of Contents
A. Related Work p. 14
B. Proofs and Details p. 15
B.1. Loss-increase Control via Head Importance (Section 4.2.1) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 15
B.1.1. Loss Bound with Operator-Norm Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 15
B.1.2. Norms and Inner Products . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 15
B.1.3. Estimating∥W O∥2 and Blockwise Norms via Power Iteration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 16
B.1.4. Cross-Entropy Curvature and Propagation toy. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 16
B.1.5. Why the Quadratic Term Is Typically Negligible . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
B.1.6. Remarks on HIS with Absolute Values . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
B.2. Generalization Gap and Attention Entropy (Section 4.2.2) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
B.2.1. Notation and Token Averaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
B.2.2. Neighboring Datasets and Why They Appear . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18
B.2.3. From Attention Perturbation to Output Perturbation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18
B.2.4. Entropy–Total Variation (TV) Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18
B.2.5. Stability and Generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 19
B.2.6. Constants and Practical Remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 19
B.3. Risk Upper Bound and HIES Minimization (Section 4.2.3) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 19
B.4. Orthogonality and Complementarity (Section 4.2.4) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 19
C. Experimental Setup p. 21
C.1. Experimental Setup for Motivation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
C.2. Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
C.3. Computing Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
C.4. Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
C.4.1. Natural Language Understanding Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
C.4.2. Image Classification Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 22
C.4.3. Reasoning Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 22
C.4.4. Multi-modal Vision-Language Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 22
D. Additional Experimental Results p. 25
D.1. Orthogonality Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 25
D.2. Seed Sensitivity and Statistical Significance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 26
D.3. Effect of Calibration Dataset Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 28
D.4. Heatmap of Importance Scores and Pruning Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 30
D.5. 3D Analysis of Attention Head Importance Scores . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 31
D.5.1. Difference in Pruning Patterns . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 32
D.5.2. Performance Inversion Across Sparsity Regimes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 32
D.6. Experimental Results on Large-scale Reasoning Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 34
D.7. Experimental Results on Downstream Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 35
D.8. Sensitivity Analysis - Ablation onα. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 36
D.9. Combining Attention Entropy with Other Importance Signals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 37
E. Efficiency Analysis p. 38
E.1. Computational Efficiency and FLOPs Reduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 38
E.2. Runtime Overhead and Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 38
13

<!-- page 14 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
A. Related Work
Model Compression.Language models (Devlin et al., 2019; Liu et al., 2019; Lewis et al., 2020) have rapidly advanced in
scale and capability, intensifying the demand to reduce their parameter sizes and inference latency (Molchanov et al., 2017;
Gordon et al., 2020). To compress large language models efficiently, various approaches have been explored, including
pruning (Liu et al., 2021; Kurtic et al., 2022; Xu et al., 2021), knowledge distillation (Sun et al., 2019; 2020; Pan et al.,
2021), quantization (Bai et al., 2021; Yao et al., 2022; Zafrir et al., 2019; Chen et al., 2025a), and other techniques, like
low-rank approximation methods (Saha et al., 2024; Li et al., 2023b; Wong et al., 2025) or weight space decomposition
methods such as (Ashkboos et al., 2024).
Structured Pruning.Among these, structured pruning—which removes entire architectural components rather than
individual weights— can be performed at various granularities, such as whole layers (Fan et al., 2019), multi-head attention
modules (Michel et al., 2019; V oita et al., 2019), or feed-forward networks (Lagunas et al., 2021; Hou et al., 2020). Recently,
attention head pruning has gained particular traction in the context of large language models. It directly reduces attention
FLOPs and KV-cache size by lowering the number of active heads while preserving the layer topology, thereby simplifying
checkpoint compatibility and serving integration. Consequently, several large-scale studies adopt head-level pruning as a
practical axis in LLM compression pipelines (Ma et al., 2023; Jaradat et al., 2024; Muralidharan et al., 2024). Ma et al.
(2023) propose a unified framework that integrates structured head pruning into the training pipeline of large language
models, achieving substantial sparsity without accuracy degradation.
Entropy in Reinforcement Learning and Transformers.Entropy has long been employed in reinforcement learning
(RL) as a means of encouraging exploration and preserving policy diversity. By introducing an entropy term into the policy
objective, RL methods prevent premature convergence to deterministic strategies and mitigate policy collapse (Bharadhwaj
et al., 2020; Xiao et al., 2021; Wang et al., 2025). Extensions of this idea include parameter-space entropy regularization
to explicitly control diversity (Han et al., 2023), large-deviation interpretations of entropy-regularized RL (Arriojas et al.,
2023), and state-distribution entropy regularization for improved robustness and generalization (Ashlag et al., 2025).
In Transformer architectures, attention entropy (AE) has been introduced to quantify the concentration of each head’s focus
across input tokens. Zhai et al. (2023) report that persistently low AE—an “entropy collapse” state—correlates strongly
with training instabilities such as oscillations in the loss surface and divergence across model scales. Their theoretical
analysis ties AE to the spectral norm of attention logits, and they propose σReparam to prevent collapse by enforcing a
lower bound on entropy. More recent work identifies variance sensitivity in the softmax transformation as another source of
entropy collapse (Hong and Lee, 2025). Together, these findings highlight attention entropy as a critical factor for stability,
motivating its integration as a complementary signal in pruning frameworks.
14

<!-- page 15 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
B. Proofs and Details
Overview.This appendix collects the formal analyses that underpin Section 4.2 and fixes notation and technical conventions
used throughout. We first develop loss-increase control for head pruning via gradient-based HIS, derive operator-norm
curvature bounds for the cross-entropy objective, and justify the first-order approximation by quantifying when quadratic
terms are negligible (Appendix B.1).
We then turn to generalization: starting from token-averaged notation and a neighboring-dataset construction, we link
perturbations of attention distributions to output deviations and establish an entropy–total variation inequality that couples
attention entropy with stability, culminating in a stability–generalization connection and practical constraints (Appendix B.2).
Building on these components, we present a risk upper bound whose surrogate minimization yields the proposed HIES
objective and clarifies its role as a principled pruning criterion (Appendix B.3). Finally, we prove an orthogonality result
between the centered HIS and entropy directions, showing their complementarity and explaining why combining the two
signals improves robustness across pruning regimes (Appendix B.4).
Collectively, these results provide (i) tight loss-control guarantees under operator-norm curvature, (ii) an entropy-based
route from stability to generalization, and (iii) a unified risk-motivated justification for HIES.
B.1. Loss-increase Control via Head Importance (Section 4.2.1)
B.1.1. LOSSBOUND WITHOPERATOR-NORMCONTROL
We writeβ y :=∥∇ 2
yL∥2 for the operator-norm curvature at the representationy.
Consider the second-order Taylor expansion iny:
L(y+δy)≤ L(y) +⟨∇ yL, δy⟩ F + 1
2 δy⊤∇2
yLδy.
Taking expectations and using the operator-norm bound yields
∆L:=E D

L(y+δy)− L(y)

≤E

⟨∇yL, δy⟩ F

+ βy
2 E

∥δy∥2
F

.
Since δy= Concat(δA 1, . . . , δA|H|), ∥δy∥2
F = P
h ∥δAh∥2
F and δAh =−(1−m h)Ah, the quadratic term equals
βy
2
P
h(1−m h)∥Ah∥2
F . For the first-order term, using the absolute value in the HIS definition and head-wise triangle
inequality,
E

⟨∇yL, δy⟩ F

=−
X
h
(1−m h)E

⟨∇Ah L, A h⟩F

≤
X
h
(1−m h) HISh,
hence
∆L ≤
|H|X
h=1
(1−m h) HISh + βy
2
|H|X
h=1
(1−m h)∥A h∥2
F .(10)
Plug-ins (default: binary).
Binary (Sigmoid) CE.β y ≤ 1
4 ∥W O∥2
2,
⇒∆L ≤
X
h
(1−m h) HISh + 1
8 ∥W O∥2
2
X
h
(1−m h)∥A h∥2
F .
Multiclass softmax CE.β y ≤ 1
2 ∥W O∥2
2,
⇒∆L ≤
X
h
(1−m h) HISh + 1
4 ∥W O∥2
2
X
h
(1−m h)∥A h∥2
F .
B.1.2. NORMS ANDINNERPRODUCTS
We use token-averaged Frobenius norms and inner products: ∥Ah∥2
F = 1
n
Pn
i=1 ∥Ah(i)∥2
2 and ⟨U, V⟩ F =
1
n
Pn
i=1⟨U(i), V(i)⟩(with batching: replace 1
n by 1
Bn ).
15

<!-- page 16 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
B.1.3. ESTIMATING∥W O∥2 ANDBLOCKWISENORMS VIAPOWERITERATION
The spectral norm of a matrixM∈R m×n is defined as
∥M∥ 2 := max
∥x∥2=1
∥M x∥2,
which measures the maximum ℓ2-amplification factor over all unit vectors. By the singular value decomposition (SVD),
M=UΣV ⊤, whereΣ = diag(σ 1, σ2, . . .)withσ 1 ≥σ 2 ≥ · · · ≥0, we have
∥M∥ 2 =σ max(M),
i.e., the spectral norm equals the largest singular value. This follows since U and V are orthogonal and preserve the ℓ2-norm,
so the maximization reduces to aligning x with the right singular vector corresponding to σmax. Exact computation via
full SVD costs O(min{m2n, mn2}), which is prohibitive for large M. Instead, we approximate σmax(M) using thepower
iterationmethod: starting from a random unit vectorv 0, iterate
ut ← M vt
∥M vt∥2
,
vt+1 ← M ⊤ut
∥M ⊤ut∥2
.
After T iterations, ∥M vT ∥2 converges to σmax(M), and vT approximates the corresponding right singular vector. We apply
this procedure toW O and, for a tighter quadratic term in our bound, to its head-wise column blocksW O
(:,Ih), forming
X
h
(1−m h)∥W O
(:,Ih)∥2
2 ∥Ah∥2
F .
Here, Ih ⊂ {1, . . . , d} denotes the set of column indices in W O corresponding to the dv output dimensions of head h. Thus,
W O
(:,Ih) ∈R d×dv is the column block of W O mapping the dv-dimensional output of head h to the d-dimensional model
space.
B.1.4. CROSS-ENTROPYCURVATURE ANDPROPAGATION TOy
Logit-space Hessian (binary vs. multiclass). Binary (sigmoid) CE.For a single logitzwithp=σ(z),
d2L
dz2 =p(1−p)≤ 1
4 ,
hence∥∇ 2
zL∥2 ≤ 1
4.
Multiclass softmax CE.For logitsz∈R C andp= softmax(z),
∇2
zL= diag(p)−pp ⊤,∥∇ 2
zL∥2 ≤ 1
2 .
Proof sketch.For any unit vector v, v⊤(diag(p)−pp ⊤)v= P
i piv2
i −(P
i pivi)2 = Varp(v). By Popoviciu’s inequality,
Varp(v)≤ (maxi vi−mini vi)2
4 ≤ 1
2 for∥v∥ 2 = 1. Tightness holds atC= 2,p= ( 1
2 , 1
2).
Mapping throughW O.With the immediate linear projectionz=yW O,
∇2
yL= (W O)⊤ ∇2
zLW O, β y :=∥∇ 2
yL∥2 ≤
( 1
4 ∥W O∥2
2,binary CE,
1
2 ∥W O∥2
2,multiclass softmax CE.
(A blockwise refinement replaces∥W O∥2 by∥W O
(:,Ih)∥2 head-wise.)
16

<!-- page 17 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
B.1.5. WHY THEQUADRATICTERMISTYPICALLYNEGLIGIBLE
We first note
HISh =E

|⟨∇Ah L, Ah⟩F |

=E

|cosϕ h| ∥∇Ah L∥F ∥Ah∥F

,
where the expectation is over x∼ D with token-averaging as in Appendix B.1.2. Assume there exists g >0 such that, for all
heads under consideration,
HISh ≥gE

∥Ah∥F

,
e.g., define
g:= min
h∈{1,...,—H—}
E

|cosϕ h| ∥∇Ah L∥F

,
wherecosϕ h :=
⟨∇Ah L, Ah⟩F
∥∇Ah L∥F ∥Ah∥F
denotes the cosine alignment between the head’s gradient and activation.
Then
quadratic
first-order ≤
βy
2
P
h(1−m h)E[∥A h∥2
F ]P
h(1−m h) HISh
≤ βy
2 · maxh E[∥Ah∥F ]
g ,
where the second inequality usesP
h(1−m h)E[∥Ah∥2
F ]≤
 
maxh E[∥Ah∥F ]
P
h(1−m h)E[∥Ah∥F ] and the per-head
lower boundHIS h ≥gE[∥A h∥F ].
Recalling βy ≤c∥W O∥2
2 with c= 1
4 for binary CE and c= 1
2 for multiclass softmax CE (cf. Appendix B.1.4), we obtain
quadratic
first-order ≤ c
2 ∥W O∥2
2 · maxh E[∥Ah∥F ]
g .
A blockwise refinement further tightens this by replacing ∥W O∥2
2 with maxh ∥W O
(:,Ih)∥2
2. Since (i) LayerNorm controls
token-wise activation scales (thus maxh E[∥Ah∥F ]), and (ii) g is bounded away from zero under non-degenerate alignment,
the ratio is typically small. Hence the first-order term dominates in practice, while the second-order term remains explicitly
controlled by the plug-in bounds in Appendix B.1.1.
B.1.6. REMARKS ONHISWITHABSOLUTEVALUES
The absolute value in (2) is part of the definition to prevent cancellation across samples; consequently, the triangle inequality
turns the first-order term into an additive upper boundP
h(1−m h) HISh (cf. Appendix B.1.1). If ⟨∇Ah L, Ah⟩F <0 on
some samples, masking that head could locally decrease the loss; the metric remains conservative by construction.
B.2. Generalization Gap and Attention Entropy (Section 4.2.2)
B.2.1. NOTATION ANDTOKENAVERAGING
For head h and query token t∈ {1, . . . , n(x)} , let α(h)
t (x)∈∆ n(x)−1 denote the attention distribution over keys, and
H(p) :=− P
j pj logp j the entropy. Define thetoken-averaged, length-normalized entropyanddeficitby
AEh(x) := 1
n(x) logn(x)
n(x)X
t=1
H
 
α(h)
t (x)

∈[0,1],
ADh(x) := 1
n(x) logn(x)
n(x)X
t=1

logn(x)−H
 
α(h)
t (x)

= 1−AE h(x)∈[0,1].
For neighboring datasets(S,S ′), write the symmetric aggregation
ADh(x) := 1
2
 
ADh(x) + AD′
h(x)

.
All token averages exclude padding positions and use the effective context length for causal masking(cf. Appendix B.2.1).
Here(S,S ′)areneighboringdatasets that differ in one example.
17

<!-- page 18 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
B.2.2. NEIGHBORINGDATASETS ANDWHYTHEYAPPEAR
We call two datasets S= (z 1, . . . , zN) and S′ = (z1, . . . , zi−1, z′
i, zi+1, . . . , zN) neighboringif they differ in exactly one
example.
Why neighboring datasets?
• Symmetrization.Introduce an i.i.d. ghost sample S′ ∼ DN to rewrite the expected generalization gap as an average of
sample-wise differences, e.g., ES,S′
 1
N
PN
i=1
 
ℓ(fS;z i)−ℓ(f S′;z ′
i)

, which is amenable to concentration and stability
arguments.
• Replace-one stability.Measure sensitivity to a single replacement by comparing fS with fS(i←z′), where S(i←z′)
replacesz i byz ′
i; underγ-uniform stability and bounded lossB, this yieldsE S[G(S)]≤2γ+ B
N .
• Symmetric inequalities.Our entropy–total variation (TV) control is symmetric in two distributions (α, α′); we
thus aggregate via ADh(x) := 1
2
 
ADh(x) + AD′
h(x)

, which streamlines notation and tightens constants in the
perturbation bound.
B.2.3. FROMATTENTIONPERTURBATION TOOUTPUTPERTURBATION
For tokent, the head output isa h(t) =
 
α(h)
t
⊤
Vh ∈R dv, hence for neighboring datasets,
∆h(t) :=a h(t)−a ′
h(t) =
 
α(h)
t −α ′(h)
t
⊤
Vh.
With∥V h∥∞→2 := maxj ∥Vh(j,:)∥ 2 and∥V h∥∞→2 ≤M,
∥∆h(t)∥2 ≤ ∥α (h)
t −α ′(h)
t ∥1 · ∥Vh∥∞→2 ≤M∥α (h)
t −α ′(h)
t ∥1.(11)
Averaging over tokens and applying the maskm h,
∥∆(x)∥2 := 1
n(x)
n(x)X
t=1
|H|X
h=1
(1−m h)∥∆ h(t)∥2.
B.2.4. ENTROPY–TOTALVARIATION(TV) CONTROL
Lemma B.1(Entropy–TV inequality).For p,q∈∆ n−1 and u uniform, ∥p−q∥ 2
1 ≤4

H(u)−H(p) +H(u)−H(q)

.
Proof. Triangle inequality and (a+b) 2 ≤2(a 2 +b 2) give ∥p−q∥ 2
1 ≤2(∥p−u∥ 2
1 +∥q−u∥ 2
1). Pinsker w.r.t. u yields
∥p−u∥ 2
1 ≤2(logn−H(p))and likewise forq.
Applying Lemma B.1 to (11) token-wise and averaging,
1
n(x)
n(x)X
t=1
∥α(h)
t −α ′(h)
t ∥1 ≤
vuut 1
n(x)
n(x)X
t=1
∥α(h)
t −α ′(h)
t ∥2
1 ≤
p
8 logn(x)
q
ADh(x).
Therefore,
∥∆(x)∥2 ≤M
p
8 logn(x)
|H|X
h=1
(1−m h)
q
ADh(x).
By Cauchy–Schwarz andP
h(1−m h) =|H|ρ,
∥∆(x)∥2 ≤
√
8M
p
|H|ρlogn(x)| {z }
=:C AE(x)
·
vuut
|H|X
h=1
(1−m h) ADh(x).(12)
18

<!-- page 19 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
B.2.5. STABILITY ANDGENERALIZATION
Letγ:=L ℓ ES[∥∆(x)∥2]. By on-average replace-one stability (Bousquet and Elisseeff, 2002),
ES

G(S)

≤2γ,G(S)≤2γ+ B
N .
Using (12) and Jensen for √·,
γ≤L ℓ ES

CAE(x)

·
vuut
|H|X
h=1
(1−m h)E S

ADh(x)

.
Taking a representative n (e.g., average/max effective length) yields the main-text constant CAE =
√
8M
p
|H|ρlogn and
Eq. (6).
B.2.6. CONSTANTS ANDPRACTICALREMARKS
• Operator norm. ∥Vh∥∞→2 := max j ∥Vh(j,:)∥ 2; take M:= max h ∥Vh∥∞→2 (controlled by LayerNorm/weight
norms).
• Sequence length.For padding/causal masking, replace n(x) by the effective context length; averages exclude padded
positions.
•Deficit aggregation.On-average: ADh = 1
2(ADh + AD′
h); Uniform: ADh = max{ADh,AD ′
h}.
• Do not pool entropies.Since using H( 1
n
P
t αt) can underestimate deficit (Jensen) and weaken control, token-wise
entropies are required.
B.3. Risk Upper Bound and HIES Minimization (Section 4.2.3)
Proof. Let supp(m) :={h:m h = 1} denote the set of retained heads. Suppose an admissible maskm′ with |supp(m ′)|=
k is not optimal. Then there exist i∈supp(m ′) and j /∈supp(m′) such that HIESj >HIES i. Consider the mask ˜mthat
swapsiandj(retainj, prunei); the constraint in (8) is preserved. The objective in (7) changes by
∆R=

HIESi

−

HIESj

<0,
since j was contributing to the sum (pruned) and i was not (retained). Hence ˜mhas a strictly smaller objective, contradicting
the minimality of m′. Therefore retaining the k heads with the largest HIES is optimal; equivalently, pruning the|H| −k
smallest HIES is optimal.
B.4. Orthogonality and Complementarity (Section 4.2.4)
Preliminaries.For headh, letα (h) ∈∆ n−1 be the attention probability vector,V h ∈R n×dv the value matrix, and
Ah =α (h)Vh ∈R 1×dv .
Define
gh :=V h
 
∇Ah L
⊤
∈R n,HIS h =
α(h)⊤gh
,AE h =−
nX
j=1
α(h)
j logα (h)
j .
Gradients w.r.t. attention (interior points).Forα (h)
j >0,
∇α(h)HISh = sign
 
α(h)⊤gh

gh,∇ α(h)AEh =−
 
1+ logα (h)
,
where log is applied elementwise. (At α(h)⊤gh = 0, any subgradient in {s gh :s∈[−1,1]} is valid; this does not affect
the result in expectation.)
19

<!-- page 20 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Simplex projection.Sinceα (h) ∈∆ n−1, we project onto the tangent space withP:=I− 1
n 11⊤ and define
e∇HISh :=P∇HIS h, e∇AEh :=P∇AE h.
Proof.By definition,u h := sign(α(h)⊤gh)g h, v h :=1+ logα (h), and˜uh :=P u h,˜vh :=P v h. Then
e∇α(h)HISh =P∇ α(h)HISh = ˜uh, e∇α(h)AEh =P∇ α(h)AEh =−˜vh.
Hence
ES

⟨e∇HISh, e∇AEh⟩

=E S

⟨˜uh,−˜vh⟩

=−tr

ES

˜uh˜v⊤
h

.
Decomposing the second moment,
ES

˜uh˜v⊤
h

= Cov(˜uh,˜vh) +E S[˜uh]E S[˜vh]⊤.
UnderCov(˜uh,˜vh) = 0and⟨E S[˜uh],E S[˜vh]⟩= 0(orthe strongerE S[˜uh] = 0), we obtain
ES

⟨e∇HISh, e∇AEh⟩

= 0.
Technical remarks.(i) At α(h)⊤gh = 0, use any subgradient of | · | for ∇α(h)HISh. (ii) Since α(h) = softmax(·), we
have α(h)
j >0 , so logα (h) (elementwise) is well-defined. (iii) “Cov(x, y) = 0” denotes thecross-covariance matrixbeing
zero, not merely componentwise uncorrelatedness. (iv) If one omits the projection P , the same argument applies with uh, vh
replacing˜uh,˜vh under the analogous conditionsCov(u h, vh) = 0and⟨E S[uh],E S[vh]⟩= 0.
20

<!-- page 21 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
C. Experimental Setup
C.1. Experimental Setup for Motivation Study
We analyze accuracy degradation and head behaviors under HIS-based pruning. In our diagnostic study, we analyze the
phenomena of pruning by HIS on BERT, focusing on detailed attention head behaviors during inference. Following prior
work analyzing BERT’s attention geometry and mechanisms (Clark et al., 2019; Rogers et al., 2020; Wang et al., 2024), we
further explore attention head pruning dynamics.
C.2. Model
Table 4.Summary of model parameters and architectures.
Model Parameters # Layers # Attention Heads Architecture / Key Details
BERTbase 110M 12 12 Transformer encoder; pre-trained on masked lan-
guage modeling and next sentence prediction
tasks.
LLaMA-27B 7B 32 32 Transformer decoder-only; trained on large-scale
text corpora for general-purpose language model-
ing.
ViTLarge 307M 24 16 Vision Transformer; patch-based image tokeniza-
tion (16×16), pre-trained on ImageNet for image
classification tasks.
LLaV A-1.57B 7B 32 32 Multi-modal LLaMA variant integrating a visual
encoder; capable of joint image-text understand-
ing and generation.
C.3. Computing Resources
Our experimental setup leverages two RTX 4090 GPUs with 24GB memory for NLU tasks using BERT and for image
classification tasks using ViT. Experiments involving LLMs such as LLaMA and multi-modal VLMs such as LLaV A were
conducted on H100 GPU with 80GB memory. For the MM-Vet benchmark, we evaluated model responses using the OpenAI
API to handle open-ended answer scoring.
C.4. Dataset Statistics
C.4.1. NATURALLANGUAGEUNDERSTANDINGTASK
We present the dataset statistics of GLUE (Wang et al., 2018) in Table 5.
Table 5.Summary of the NLU benchmark.
NLU Benchmark
Dataset # Train # Valid # Test # Label Task Evaluation Metric
Single-Sentence Classification (GLUE)
CoLA 8,551 521 522 2 Acceptability Matthews corr
SST-2 66,349 1,000 872 2 Sentiment Accuracy
Pairwise Text Classification (GLUE)
MNLI 392,702 9,832 9,815 3 NLI Accuracy
RTE 2,490 138 139 2 NLI Accuracy
QQP 362,846 1,000 40,431 2 Paraphrase Accuracy
MRPC 3,668 204 204 2 Paraphrase F1 score
QNLI 103,743 1,000 5,463 2 QA/NLI Accuracy
Pairwise Text Classification (GLUE)
STS-B 5,749 750 750 1 Similarity Pearson corr
21

<!-- page 22 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
C.4.2. IMAGECLASSIFICATIONTASK
Table 6 lists dataset statistics for the image classification task in the Computer Vision (CV) domain.
Table 6.Summary of the CV benchmark.
CV Benchmark
Dataset # Train # Valid # Test # Label Task Evaluation Metric
ImageNet1k 1,281,167 50,000 100,000 1,000 Classification Accuracy
CIFAR-100 45,000 5,000 10,000 100 Classification Accuracy
Fashion MNIST 54,000 6,000 10,000 10 Classification Accuracy
Oxford Flowers 1,020 1,020 6,150 102 Classification Accuracy
C.4.3. REASONINGTASK
To evaluate the effectiveness of our pruning method on reasoning tasks, we use two benchmark datasets:GSM8K(Cobbe
et al., 2021) andMMLU(Hendrycks et al., 2020).
GSM8K:is a dataset of 8.5K high quality, linguistically diverse grade school math word problems. GSM8K supports the
task of question answering on basic mathematical problems requiring multi-step reasoning.
MMLU:is a massive multitask benchmark consisting of multiple-choice questions spanning diverse domains, including
the humanities, social sciences, and hard sciences. The benchmark covers 57 tasks such as elementary mathematics, U.S.
history, computer science, and law, and requires models to possess broad world knowledge and strong problem-solving
abilities to achieve high performance.
By using both GSM8K and MMLU, we evaluate our pruning methods from complementary perspectives: GSM8K assesses
mathematical reasoning under multi-step computation, while MMLU measures the preservation of broad world knowledge
and multi-domain problem-solving ability.
C.4.4. MULTI-MODALVISION-LANGUAGETASK
To evaluate the effectiveness of our pruning method on multi-modal vision-language models (VLMs), we use two benchmark
datasets:VizWiz-VQA(Gurari et al., 2018) andMM-Vet(Yu et al., 2023). The evaluation was conducted using the
LLaV A1.57B model.
VizWiz-VQA:is designed for Visual Question Answering (VQA) in the context of assisting people who are blind. Each
visual question originates from a real-world setting where blind users captured images and recorded spoken questions, and
is accompanied by ten crowdsourced answers. The dataset poses two evaluation tasks: predicting the correct answer given
an image and question, and detecting whether a question cannot be answered.
MM-Vet:is a benchmark intended to evaluate large multimodal models on complex tasks that require the integration of
multiple vision-language capabilities. It defines six core VL skills and sixteen combinations of these skills, and employs
an LLM-based evaluator to provide a unified scoring metric across diverse question types and answer formats. MM-Vet
enables a systematic assessment of models’ generalization, reasoning, and open-ended answer generation abilities.
By using both VizWiz-VQA and MM-Vet, we comprehensively evaluate our pruning method across real-world visual
questions, complex multimodal reasoning, and diverse answer styles, providing a thorough assessment of its impact on the
overall quality of the pruned model. Note that our evaluation is conducted on a subset of the datasets. To illustrate the nature
of the datasets used in our evaluation, we provide example entries from both VizWiz-VQA and MM-Vet.
22

<!-- page 23 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 5.Examples from VizWiz-VQA showing visual questions asked by blind users and the corresponding answers from crowd workers.
The examples include both questions that can be answered from the image and questions that cannot.
23

<!-- page 24 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 6.Eight example queries from the MM-Vet benchmark, each requiring different integrations of core vision–language capabilities
to solve complicated multimodal tasks.
24

<!-- page 25 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D. Additional Experimental Results
D.1. Orthogonality Analysis
We provide an empirical sanity check supporting the Lemma 4.3. Experiments use TinyBERT on SST-2
(Vishnou/TinyBERT SST2). For each head we compute layerwise-normalized HIS and AE scores, stack them into
vectors u and v, and form the centered versions ˜uand ˜vby subtracting each vector’s mean (a finite-sample proxy for
projection onto the zero-sum subspace). The following sample statistics were obtained:
dCov(˜u,˜v) = 0.030853, u= 3.73×10 −9, v= 1.61×10 −8.
Consequently,
bE

⟨˜u,−˜v⟩

=−0.030853 =−
 dCov(˜u,˜v) +u v

(up to numerical precision).
The covariance magnitude is small on this batch, indicating weak coupling between the two directions and lending empirical
support to the “(near) uncorrelatedness” assumption.
We further extend the orthogonality analysis across models, with results reported in Table 7 and Table 8.
Table 7.Orthogonality analysis using BERT base on the GLUE benchmark.
Task E[˜u] E[ ˜v] Cov( ˜u,˜v) E[〈 ˜u,˜v〉] Correlation
COLA 6.94E-03 4.08E-01 -1.40E-05 2.82E-03 -1.54E-02
SST2 6.94E-03 4.42E-01 -3.60E-05 3.03E-03 -3.41E-02
MRPC 6.94E-03 5.27E-01 2.41E-04 3.90E-03 2.32E-01
STSB 6.94E-03 4.30E-01 -1.96E-04 2.79E-03 -1.65E-01
QQP 6.94E-03 5.05E-01 1.56E-04 3.67E-03 1.95E-01
MNLI 6.94E-03 4.89E-01 1.73E-04 3.57E-03 2.27E-01
QNLI 6.94E-03 4.62E-01 1.53E-04 3.36E-03 2.23E-01
RTE 6.94E-03 5.05E-01 2.15E-04 3.72E-03 2.88E-01
Table 8.Orthogonality analysis using LLaMA-2 7B on 5 tasks.
Task E[˜u] E[ ˜v] Cov( ˜u,˜v) E[〈 ˜u,˜v〉] Correlation
HellaSwag 9.77E-04 7.56E-01 -1.47E-04 5.92E-04 -1.44E-01
Winogrande 9.77E-04 7.66E-01 -1.85E-04 5.63E-04 -1.66E-01
ARC-e 9.77E-04 7.76E-01 -1.42E-04 6.16E-04 -1.44E-01
ARC-c 9.77E-04 7.71E-01 -1.27E-04 6.26E-04 -1.33E-01
OBQA 9.77E-04 8.01E-01 -1.29E-04 6.53E-04 -1.61E-01
25

<!-- page 26 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.2. Seed Sensitivity and Statistical Significance
We evaluate the seed sensitivity of HIS and HIES, with results reported in Table 9 and Table 10 .
Table 9.Accuracy and standard deviation in accuracy across 5 random seeds for BERT base on GLUE benchamrk.
Pruning Ratio (%) HIS HIES (ours)
Accuracy (%) Standard Deviation Accuracy (%) Standard Deviation
CoLA
10 74.18 1.60E-02 75.85 1.21E-02
20 74.18 5.44E-04 73.04 1.09E-02
30 72.70 6.91E-03 75.17 2.82E-03
40 70.28 5.52E-03 70.20 3.68E-02
50 60.37 2.18E-02 71.17 5.29E-03
MNLI
10 84.13 0.00E+00 84.06 1.64E-01
20 82.93 9.13E-02 82.87 0.00E+00
30 82.40 0.00E+00 82.29 3.58E-01
40 81.46 2.39E-01 81.39 1.98E-01
50 79.47 4.34E-01 78.87 3.27E-01
MRPC
10 90.29 4.58E-03 90.97 8.04E-04
20 89.69 4.50E-03 89.17 3.55E-03
30 88.79 2.36E-03 89.52 3.69E-03
40 88.02 7.41E-03 87.12 9.06E-03
50 86.85 1.65E-02 87.19 1.66E-02
QNLI
10 90.55 8.69E-02 90.51 1.34E-01
20 90.23 3.48E-01 90.05 2.73E-01
30 88.33 7.47E-01 88.15 7.43E-01
40 84.75 1.45E+00 85.02 8.90E-01
50 80.81 3.51E+00 82.83 1.74E+00
QQP
10 90.68 2.33E-02 91.26 8.00E-03
20 90.56 1.52E-01 90.56 7.31E-02
30 89.84 3.67E-01 89.72 3.45E-01
40 88.07 1.84E-01 87.96 2.54E-01
50 85.33 9.93E-01 85.90 6.97E-01
RTE
10 72.42 2.89E-01 72.13 2.70E-01
20 71.91 5.78E-01 70.76 9.69E-01
30 71.26 1.18E+00 71.34 1.47E+00
40 67.94 2.10E+00 66.79 1.92E+00
50 65.99 1.36E+00 65.42 1.56E+00
SST-2
10 91.86 1.62E-01 92.64 4.59E-02
20 91.77 2.22E-01 92.11 1.69E-01
30 91.19 3.03E-01 92.20 1.92E-01
40 88.67 1.62E+00 90.96 7.16E-01
50 86.31 1.35E+00 90.55 1.24E+00
STS-B
10 87.85 1.88E-03 87.95 3.37E-03
20 87.33 3.52E-03 87.35 2.06E-04
30 86.82 2.15E-03 86.76 3.25E-03
40 85.89 4.49E-03 85.87 4.41E-03
50 83.38 8.95E-03 82.27 1.37E-02
Avg.
10 85.25 7.30E-02 85.67 7.98E-02
20 84.82 1.75E-01 84.49 1.87E-01
30 83.92 3.26E-01 84.39 3.90E-01
40 81.89 7.00E-01 81.91 5.04E-01
50 78.56 9.62E-01 80.53 6.99E-01
26

<!-- page 27 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Table 10.Accuracy and standard deviation in accuracy across 5 random seeds for LLaMA-27B on HellaSwag, Winogrande, and ARC-c.
Pruning Ratio (%) HIS HIES (ours)
Accuracy (%) Standard Deviation Accuracy (%) Standard Deviation
HellaSwag
10 53.63 5.36E-01 53.980.66%↑5.40E-01
20 51.70 5.17E-01 52.361.28%↑5.24E-01
30 49.00 4.90E-01 49.841.71%↑4.98E-01
40 45.27 4.53E-01 46.272.20%↑4.63E-01
50 37.01 3.70E-01 36.132.37%↓3.61E-01
Winogrande
10 66.38 6.64E-01 67.010.95%↑6.70E-01
20 64.07 6.41E-01 66.273.42%↑6.63E-01
30 63.03 6.30E-01 64.672.60%↑6.47E-01
40 60.16 6.02E-01 61.061.50%↑6.11E-01
50 56.16 5.62E-01 55.830.59%↓5.58E-01
ARC-c
10 33.15 3.71E-03 35.055.73%↑7.03E-03
20 33.15 5.57E-03 36.279.41%↑0.00E+00
30 32.34 1.86E-03 35.469.64%↑5.67E-03
40 29.69 1.86E-03 34.2415.30%↑1.15E-02
50 25.29 7.43E-03 28.9514.48%↑4.22E-02
Avg.
10 51.05 5.70E-03 52.011.88%↑5.06E-03
20 49.64 5.12E-03 51.634.01%↑2.72E-03
30 48.12 5.30E-03 49.993.88%↑6.11E-03
40 45.04 5.49E-03 47.194.76%↑8.82E-03
50 39.49 9.20E-03 40.302.07%↑2.17E-02
27

<!-- page 28 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.3. Effect of Calibration Dataset Size
We analyze the effect of calibration dataset size on the performance and stability of HIES. Tables 11 and 12 report accuracy
and standard deviation across calibration sizes ranging from 1 to 1024 samples for BERTbase and LLaMA-27B, respectively.
Across both model families, HIES exhibits stable performance once a modest number of calibration samples is used. In
particular, increasing the calibration set size beyond small values yields diminishing returns in accuracy, while the variance
across random seeds remains low. These results indicate that HIES does not require large calibration datasets to obtain
reliable head-level statistics, supporting its practical applicability in low-cost calibration settings.
Table 11.Accuracy and standard deviation in accuracy for different calibration dataset sizes (1, 16, 32, 64, 128, 512, 1024) for BERTbase
on GLUE Benchmarks. Note that we use a default size of 32 in the main results.
Pruning Ratio (%) HIS HIES (ours)
Accuracy (%) Standard Deviation Accuracy (%) Standard Deviation
CoLA
10 74.46 1.33E-02 75.71 1.34E-02
30 74.09 8.40E-03 75.10 3.04E-03
50 62.01 3.37E-02 71.04 1.13E-02
MNLI
10 84.13 0.00E+00 84.13 0.00E+00
30 82.40 0.00E+00 82.35 3.48E-01
50 78.93 5.72E-01 78.65 1.98E-01
MRPC
10 90.52 2.82E-03 91.05 2.08E-03
30 89.12 5.97E-03 89.61 2.94E-03
50 86.45 1.55E-02 86.92 1.44E-02
QNLI
10 90.51 9.57E-02 90.54 1.25E-01
30 88.25 5.97E-01 88.01 5.53E-01
50 80.50 2.96E+00 81.97 1.39E+00
QQP
10 90.74 7.25E-02 91.24 4.16E-02
30 90.12 2.31E-01 90.01 2.37E-01
50 84.75 6.91E-01 85.60 4.29E-01
RTE
10 72.36 4.09E-01 72.25 3.25E-01
30 71.17 1.13E+00 70.81 1.13E+00
50 65.45 1.14E+00 65.50 2.13E+00
SST-2
10 91.79 1.30E-01 92.56 1.68E-01
30 90.78 4.53E-01 92.12 2.06E-01
50 85.76 2.30E+00 90.33 1.16E+00
STS-B
10 87.98 7.08E-04 87.96 2.88E-03
30 86.87 2.03E-03 86.75 2.64E-03
50 83.53 6.39E-03 82.80 1.37E-02
Avg.
10 85.31 9.06E-02 85.68 8.48E-02
30 84.10 3.04E-01 84.34 3.10E-01
50 78.42 9.63E-01 80.35 6.68E-01
28

<!-- page 29 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Table 12.Accuracy and standard deviation in accuracy for different calibration dataset sizes (1, 16, 32, 64, 128, 512, 1024) forLLaMA-27B
on HellaSwag, Winogrande, and ARC-c. Note that we use a default size of 32 in the main results.
Pruning Ratio (%) HIS HIES (ours)
Accuracy (%) Standard Deviation Accuracy (%) Standard Deviation
HellaSwag
10 53.54 1.57E-03 54.091.02%↑1.99E-03
20 51.44 3.51E-03 52.371.82%↑1.93E-03
30 49.09 4.54E-03 50.222.30%↑1.66E-03
40 44.66 2.80E-03 45.592.09%↑4.20E-03
50 34.91 3.60E-03 34.431.36%↓8.42E-03
60 28.46 4.47E-03 28.911.57%↑2.64E-03
Winogrande
10 65.87 5.51E-03 67.422.35%↑3.14E-03
20 64.31 9.97E-03 66.423.27%↑4.20E-03
30 63.21 5.59E-03 65.423.49%↑5.50E-03
40 60.38 9.65E-03 61.471.81%↑5.67E-03
50 56.17 7.89E-03 56.660.86%↑5.77E-03
60 51.23 8.90E-03 51.330.18%↑1.06E-02
ARC-c
10 34.51 1.66E-02 34.410.29%↓8.51E-03
20 33.83 6.95E-03 35.685.46%↑6.61E-03
30 33.02 7.73E-03 34.755.24%↑1.19E-02
40 31.25 1.50E-02 33.477.10%↑4.42E-03
50 26.10 1.58E-02 26.862.92%↑1.24E-02
60 22.17 2.37E-02 23.646.65%↑6.78E-03
Avg.
10 51.31 7.90E-03 51.971.29%↑4.54E-03
20 49.86 6.81E-03 51.493.27%↑4.25E-03
30 48.44 5.95E-03 50.133.49%↑6.35E-03
40 45.43 9.17E-03 46.853.12%↑4.76E-03
50 39.06 9.11E-03 39.320.66%↑8.85E-03
60 33.95 1.23E-02 34.631.98%↑6.66E-03
29

<!-- page 30 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.4. Heatmap of Importance Scores and Pruning Results
Figure 7.Heatmaps of head-importance scores across four GLUE tasks (CoLA, MRPC, QNLI, QQP). Left: HIS; Right: HIES (ours).
Rows = layers (L0–L11); columns = heads (H0–H11).
We analyze the pruning patterns and performance dynamics of HIS- and HIES-based methods across varying sparsity levels.
This section highlights the fundamental distinctions in head selection strategies and the underlying mechanisms responsible
for the observed performance inversion.
30

<!-- page 31 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.5. 3D Analysis of Attention Head Importance Scores
Figure 8.3D Analysis of Attention Head Importance Scores
31

<!-- page 32 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.5.1. DIFFERENCE INPRUNINGPATTERNS
As shown in Figure 9, Pruning heatmaps reveal systematic differences between the methods. HIS-based pruning tends to
remove heads primarily from the lower layers, producing an approximately bottom-up pattern consistent with its one-step
gradient saliency. In contrast, HIES yields a more dispersed selection spanning lower, middle, and upper layers. We
attribute this to the entropy-aware term, which leverages structural properties of the attention distribution (concentration vs.
dispersion) in addition to gradient sensitivity, thereby promoting diversity across layers in pruning decisions.
D.5.2. PERFORMANCEINVERSIONACROSSSPARSITYREGIMES
We identify two distinct pruning regimes:
Redundancy Regime (≤ 10% pruning).In the early pruning phase, the model contains a substantial number of redundant
heads. Here, gradient-based HIS is sufficient to identify and remove low-sensitivity heads, as they reflect the immediate
(one-step) loss change. Consequently, HIS performs slightly better than HIES in both accuracy and stability under light
pruning.
Specialization Regime (≥ 30% pruning).As pruning becomes more aggressive, redundant heads are mostly exhausted,
and specialized heads begin to be targeted. In this regime, HIS alone struggles to distinguish critical heads from less
important ones, as gradient magnitudes no longer capture long-term utility. In contrast, HIES leverages attention entropy
to preferentially preserve highly concentrated (low-entropy) heads—which are typically more specialized—and prune
high-entropy, less task-specific heads. This leads to superior accuracy and stability under higher pruning ratios.
Summary
•Pruning≤10%:Redundancy regime⇒HIS outperforms HIES.
•Pruning≥30%:Specialization regime⇒HIES outperforms HIS.
These findings demonstrate that HIS and HIES prioritize head preservation differently—HIS reflects short-horizon gradient
sensitivity, whereas HIES incorporates extended inference-time stability by preserving low-entropy specialized heads.
32

<!-- page 33 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
Figure 9.CoLA: heatmaps of head importance and pruning across sparsity levels. For each pruning ratio (10%, 30%, 50%, 70%), we show
HIS (left) and HIES (right). Rows = layers (L0–L11); columns = heads (H0–H11). Dark/grey cells mark heads pruned at the target ratio.
33

<!-- page 34 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.6. Experimental Results on Large-scale Reasoning Tasks
Figure 10 reports reasoning performance on GSM8K and MMLU with LLaMA-27B, comparing HIES against HIS-based
pruning. Across both benchmarks, HIES consistently outperforms HIS across pruning regimes, achieving an average
accuracy improvement of 14.67% and demonstrating improved robustness on reasoning tasks.
Figure 10.Results on GSM8K (math word-problem reasoning) and MMLU (10-task knowledge reasoning) with LLaMA-2 7B.
34

<!-- page 35 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.7. Experimental Results on Downstream Tasks
Figure 11 reports downstream evaluations of HIES versus the HIS baseline on CIFAR-100, Food-101, and Fashion-MNIST.
Across all three benchmarks, HIES consistently sustains higher accuracy under aggressive pruning, whereas HIS exhibits
rapid degradation once the pruning ratio exceeds 20%. On CIFAR-100, HIS collapses beyond moderate sparsity, while HIES
exhibits slower degradation and retains substantially higher accuracy relative to HIS even at 40–50%. Food-101 reveals a
similar trend, with HIES delivering substantial and consistent gains over HIS across all pruning levels. On Fashion-MNIST,
HIS undergoes steep drops after 20% pruning, in contrast to the stable performance of HIES up to 50%. These results
demonstrate that HIES reliably mitigates sharp-drop phenomena and delivers robust, stable improvements over HIS across
heterogeneous downstream tasks.
Figure 11.Evaluation of HIES on the image classification benchmarks. HIES consistently outperforms baseline, demonstrating robust and
stable performance across downstream tasks.
35

<!-- page 36 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
D.8. Sensitivity Analysis - Ablation onα
Figure 12.HIES sensitivity to the mixing coefficient α on GLUE. For each task, we sweep α and report three choices— αbest, αmedian,
αworst—selected by weighted AUC (wAUC) across pruning ratios. Curves plot performance versus pruning ratio for these three settings.
36

<!-- page 37 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
We sweep the mixing coefficientα∈[0,1)that interpolates the gradient-based HIS and AE signals in HIES,
HIESh(α) =α dHISh + (1−α) cAEh.
As expected, larger α upweights HIS and preserves heads with strong task relevance, whereas smaller α upweights AE
and retains low-entropy, focused heads. We choose a single α⋆ on a held-out validation split and fix it for all reported
experiments; the resulting accuracy–sparsity profiles are shown in Figure 12.
D.9. Combining Attention Entropy with Other Importance Signals
To examine the applicability of AE beyond gradient-based scores, we combine it with a different importance signal based
on the L2 norm of attention outputs. This experiment assesses the generality of AE and verifies that it can serve as a
complementary stabilization term under different importance signals.
Figure 13 reports the impact of combining AE with the L2-norm–based importance score on LLaMA-27B across five
benchmarks: HellaSwag, Winogrande, ARC-e, ARC-c, and OBQA. Across all benchmarks, incorporating AE consistently
mitigates performance degradation under aggressive pruning, indicating that the stabilizing effect of AE is not restricted to a
particular importance formulation.
Figure 13.Accuracy and stability improvements when combining AE with gradient L2-norm–based importance scores, compared to using
the L2-norm alone. We conduct experiments on LLaMA-27B for HellaSwag, Winogrande, PIQA, ARC-e, ARC-c, and OBQA.
37

<!-- page 38 -->

Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning
E. Efficiency Analysis
E.1. Computational Efficiency and FLOPs Reduction
As the pruning ratio increases, the total FLOPs decrease approximately linearly: removing 10% of attention heads yields
an ≈4% reduction in FLOPs, while pruning 50% of heads achieves an ≈20% reduction. Under HIS-only pruning, model
accuracy on TinyBERT with SST-2 sharply degrades beyond a 42% pruning ratio, at which point only an≈16% FLOPs
reduction can be attained without critical performance loss. In contrast, HIES-based pruning maintains at least 80%
validation accuracy up to a 60% pruning ratio, corresponding to an ≈23% FLOPs reduction relative to the original model.
Extending the feasible pruning regime from 42% to 60% therefore delivers an additional ≈7 percentage-point reduction
in FLOPs, demonstrating that HIES enables substantially greater computational efficiency without compromising task
performance.
E.2. Runtime Overhead and Implementation Details
The above efficiency gains are achieved with only a small computational overhead. HIES is computed using lightweight for-
ward hooks attached to the attention modules. Specifically, HIS reuses gradients produced during standard backpropagation,
while AE is obtained directly from attention probabilities in the forward pass. Both signals are collected via non-intrusive
forward hooks without modifying model internals or introducing additional forward or backward passes. As a result, the
runtime overhead is minimal—typically on the order of a few seconds— with negligible additional memory cost beyond a
small buffer for storing per-head statistics. Despite this low overhead, HIES consistently outperforms baselines that rely on
full fine-tuning, achieving superior efficiency in both computation and accuracy.
38
