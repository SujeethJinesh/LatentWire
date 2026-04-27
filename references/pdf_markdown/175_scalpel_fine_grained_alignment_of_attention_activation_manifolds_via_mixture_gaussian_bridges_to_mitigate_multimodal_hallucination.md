# references/175_scalpel_fine_grained_alignment_of_attention_activation_manifolds_via_mixture_gaussian_bridges_to_mitigate_multimodal_hallucination.pdf

<!-- page 1 -->

Scalpel: Fine-Grained Alignment of Attention Activation Manifolds via Mixture
Gaussian Bridges to Mitigate Multimodal Hallucination
Ziqiang Shi⋆*, Rujie Liu⋆, Shanshan Yu†, Satoshi Munakata†, Koichi Shirahata†
⋆ Fujitsu Research & Development Center Co.,LTD., Beijing, China
† Fujitsu Limited, Tokyo, Japan
Abstract
Rapid progress in large vision-language models (LVLMs)
has achieved unprecedented performance in vision-
language tasks. However, due to the strong prior of large
language models (LLMs) and misaligned attention across
modalities, LVLMs often generate outputs inconsistent with
visual content - termed hallucination. To address this, we
proposeScalpel, a method that reduces hallucination by re-
fining attention activation distributions toward more credi-
ble regions. Scalpel predicts trusted attention directions for
each head in Transformer layers during inference and ad-
justs activations accordingly. It employs a Gaussian mix-
ture model to capture multi-peak distributions of attention
in trust and hallucination manifolds, and uses entropic op-
timal transport (equivalent to Schr¨odinger bridge problem)
to map Gaussian components precisely. During mitiga-
tion, Scalpel dynamically adjusts intervention strength and
direction based on component membership and mapping
relationships between hallucination and trust activations.
Extensive experiments across multiple datasets and bench-
marks demonstrate that Scalpel effectively mitigates hallu-
cinations, outperforming previous methods and achieving
state-of-the-art performance. Moreover, Scalpel is model-
and data-agnostic, requiring no additional computation,
only a single decoding step.
1. Introduction
Large visual language models (LVLMs) have become es-
sential tools for handling diverse visual tasks and per-
forming complex visual question-answering due to their
strong capabilities in content understanding and genera-
tion [19, 33, 36, 40]. Despite these advancements, LVLMs
often suffer from the “hallucination” problem, where gen-
erated text appears plausible but misaligns with image con-
tent or even fabricates elements not present in the in-
put [10, 14, 14, 20, 27, 29, 37]. This issue significantly un-
*Corresponding author: shiziqiang@fujitsu.com
dermines their reliability in critical domains such as health-
care, autonomous driving, and security monitoring.
The hallucinations in current LVLMs are generally at-
tributed to two main factors: the strong language priors in-
herited from pre-trained large language models (LLMs) [14,
20], and the model’s tendency to over-attend to irrelevant vi-
sual tokens or system tokens unrelated to the given instruc-
tion [27, 29, 34, 37]. To mitigate this, researchers have ex-
plored reinforcement learning (RL) strategies that fine-tune
LVLMs using high-quality [5, 20, 21, 27], hallucination-
free feedback—either human-labeled or AI-generated—to
improve alignment between vision and language while re-
ducing reliance on LLM priors. However, these meth-
ods demand substantial computational resources for anno-
tation and training, making them impractical in resource-
constrained environments. Consequently, recent studies
have shifted focus toward inference-time optimization tech-
niques that do not require retraining, such as contrastive de-
coding [8, 10, 14, 35], which reduces hallucinations by ad-
justing logit scores. Although training-free, these methods
still fall short in effectiveness and often introduce noticeable
latency due to multi-step decoding.
While both approaches yield some success, they rarely
investigate the internal mechanisms of the LVLM itself
to uncover the root causes of hallucinations. In con-
trast, this paper delves into the role of attention activation
across different heads and layers within the LVLM’s Trans-
former [31] architecture during the generation of halluci-
nated content. We discover that individual attention heads
contribute differently to hallucinations. By precisely identi-
fying those responsible and applying targeted interventions,
we can significantly enhance output quality and multimodal
reasoning performance.
Our key contributions are summarized as follows:
• We propose Scalpel, a novel, training-free, plug-and-play
method that enables customized correction of attention
activations at the token level without compromising the
LLM’s knowledge capacity, effectively reducing halluci-
nations in LVLMs.
• To achieve precise attention correction per token, Scalpel
arXiv:2602.09541v1  [cs.CV]  10 Feb 2026

<!-- page 2 -->

Figure 1. Schematic diagram illustrating the principle of the Scalpel method. First, the trusted and hallucinated attention activations of all
heads are obtained by inputting both correct and hallucinated data into the LVLM. These activations serve two purposes. The first is to
identify the heads with the highest hallucination discrimination capability using a classifier. The second is to analyze the distribution of
credible and hallucinated attention activations across tokens, thereby determining the influence of each individual token on each head.
employs Gaussian mixture models (GMMs) to represent
trusted and hallucinated attention activation manifolds,
then applies Schr ¨odinger bridge problem solving theory
to compute optimal transport between these GMMs as
shown in Figure 1. This allows for component-specific
correction based on the current token’s activation.
• Experimental results on POPE [17] and MME [9] bench-
marks demonstrate that Scalpel significantly improves
performance on LLaV A-1.5 [19] and Qwen2.5-VL [2],
proving its model-agnostic and task-agnostic applicabil-
ity across multiple databases.
2. Related Work
2.1. Hallucination in LVLMs
Although LVLMs, built upon open-source language models
such as LLaMA [28] and Vicuna [7], have achieved effec-
tive multimodal fusion of text and images—greatly enhanc-
ing cross-modal understanding and generation—they still
face a critical challenge: hallucination. Hallucination refers
to factual inconsistencies between the generated text and the
input image, often manifesting as fabricated objects or in-
accurate descriptions of attributes and relationships. This
issue primarily stems from two key factors: (1) an over-
reliance on prior knowledge (e.g., biases in training data
or inherent language priors within LLMs) [14, 20], and (2)
limitations in visual encoder localization, misalignment of
multimodal information [27, 34], or suboptimal attention
modeling during decoding [29, 37], all of which hinder ac-
curate associations between inputs and outputs.
2.2. Hallucination Mitigation in LVLMs
LVLM hallucination mitigation falls into two categories:
training-based and training-free post-processing methods.
Training-based approaches include: MIA-DPO [20] cre-
ates multi-image data via grid collages from single inputs;
LLaV A-RLHF [27] reduces reward hacking through fact-
grounded reward models; DAMA [21] dynamically opti-
mizes training by aligning data difficulty with response
strategies; PerturboLLA V A [5] weakens language prior
reliance via adversarial text perturbations; EACO [34]
achieves alignment with only 5,000 images (high ef-
ficiency); AMP [38] enhances fine-grained recognition
through multi-level preference optimization.
Training-free post-processing improves generation accu-
racy without retraining. These methods fall into two cate-
gories: contrastive decoding [8, 10, 14, 35] and attention
reallocation [29, 35, 37, 39]. Contrastive decoding reduces
bias via cross-image comparison (VCD [14]), uses penal-
ties/rollback for refinement (OPERA [10]), and strengthens
vision-language alignment (M3ID [8]). Attention realloca-
tion improves responsiveness with blind-token calibration
(A VISC [35]), provides plug-and-play enhancement (Clear-
Sight [37]), optimizes resource use through recycling (At-
tnReal [29]), and broadcasts attention matrices for focus en-
hancement (EAH [39]), achieving minimal computational
overhead.
Our approach fundamentally differs by leveraging
inference-time intervention [6, 16]. The core procedure
comprises: (1) identifying factual-response directions in ac-

<!-- page 3 -->

tivation space; (2) steering activation vectors toward these
directions during reasoning. Compared to existing methods,
this technique offers: negligible computational overhead;
strong interpretability through explicit activation manipula-
tion; and significant hallucination reduction while preserv-
ing generation quality.
3. Methodology
3.1. Background and Notation
The input to an LVLM is multimodal, consisting of both
text and visual components. For simplicity, we do not
distinguish between system prompts and user instructions
in text input. Letx t ={x 1, x2, . . . , xM }denote the to-
ken sequence encoded by the text encoder, representing
a textual feature sequence of lengthM. Similarly, let
xv ={x M+1 , xM+2 , . . . , xM+N }represent the visual to-
ken sequence encoded by the vision encoder, with length
N. These sequences are concatenated into a unified input
sequencex={x 1, x2, . . . , xM+N }before being fed into
the LVLM.
The LVLM processes this input token sequencex
throughLTransformer layers. The output at layerl+
1, denoted ash (l+1), is computed using multi-head self-
attention [31]:
h(l+1) =h (l) +
NhX
n=1
Al
n(h(l))P l
n (1)
whereN h is the total number of heads in a Transformer
layer,dis the dimension of the operation space of each
head;A l
n(·)is the attention operator of then-th head in
thel-th layer;P l
n ∈R d×dNh maps the activated output of
attention head back to the operation space of thel-th Trans-
former layer. Each layer performs self-attention to capture
interactions between text and visual features. At the final
layer (layerL), the hidden statesh (L) are passed through
an affine layer to produce logits of size equal to the vocab-
ulary, followed by Softmax normalization to generate the
probability distribution for the next tokeny t:
p(yt|y<t) =Softmax(Affine(h (L)
t ))
wherey <t represents the previously generated tokens.
3.2. Modeling Trusted and Hallucinated Manifolds
of Attention Activations
Our method, Scalpel, is based on two key findings from
studies on LLMs and LVLMs. First, these models often
know the correct answer but fail to express it clearly [12,
24, 32]. Second, their activation space has key directions
that guide truthful responses [4, 26]. Earlier methods like
ITI [16] and ICT [6] reduce hallucinations by steering at-
tention heads toward trusted activation spaces. Yet, they
use fixed correction directions for all activations in a head,
which may not be ideal. Scalpel improves this by customiz-
ing intervention directions for different activations during
inference, even within the same attention head. To do this
efficiently, we separately discretize and tokenize trusted and
hallucinated manifolds, then create a one-to-one mapping
between them. GMMs [1] are well-suited for this task. We
use GMMs to model attention activations from both trusted
and hallucinated data, approximating their manifolds.
Trusted activation manifolds are built from valid (image,
Q,A) triplets (e.g., Q: What is in the image? A: Zebra.),
extracting attention activations across all layers/heads, as
shown in the left part of Figure 1. Hallucinated manifolds
are created by perturbing images while keeping QA pairs
correct, or by altering specific image regions (e.g., zebra
bounding boxes). For each attention head/layer, we extract
three manifolds: (1) trusted, (2) image-perturbed halluci-
nated, (3) object-perturbed hallucinated. Activation dis-
tributions differ slightly between image and object levels,
leading to different intervention directions. At the object
level, as shown in Figure 2, most background remains un-
changed, so only specific components need large-scale in-
tervention. However, intervention principles are identical,
so we omit level distinctions in the following. Our goal is to
map hallucinated to trusted manifolds, enabling corrective
adjustments to activations. Letρ H denote the hallucinated
manifold’s distribution with samplez 0, andρ T the trusted
withz 1. The optimal mapping is given by the entropic op-
timal transport (EOT) problem [22]:
min
π∈Π(ρH ,ρT )
Z
∥z0 −z 1∥2dπ−ϵh(π)(2)
whereπ(z 0,z 1)is the transport plan (coupling),Π(ρ H , ρT )
the joint distributions with these marginals, andhthe dif-
ferential entropy:
h(ρ)≜−
Z
ρ(z) logρ(z)dz.(3)
Directly solving the EOT problem (Eq. 2) from halluci-
natory to trusted activations is difficult. Leonard reformu-
lates it as an equivalent Schr ¨odinger bridge problem (SBP)
for tractability [15]:
min
u∈U
Et∼ρt
Z 1
0
1
2ϵ ∥ut(zt)∥2 dt

,(4)
dzt =u t(zt)dt+ √ϵ dwt,(5)
z0 ∼ρ H ,z 1 ∼ρ T ,(6)
whereUis the set of adapted finite-energy controls (i.e.,
drift in diffusion/SDEs). The goal is minimal-energy con-
trol ensuring initial distributionρ H and terminalρ T . We
solve this SBP to derive the optimal path from hallucination
to trusted manifolds.

<!-- page 4 -->

3.3. Optimal Transport Mapping Between Trusted
and Hallucinated GMMs
Both trusted/hallucinated GMMs contain multiple compo-
nents. When an activation belongs to a hallucinated compo-
nent, mapping requires not only identifying its trusted coun-
terpart but determining the optimal transformation path. To
address this, we propose an alignment algorithm superior to
random matching. Intuitively, minimal corrections preserve
data manifold integrity, thus we seek minimum-cost flow
mapping between components, as depicted in the right part
of Figure 1.
This process is performed independently per attention
head. Let hallucinated GMM (for a head) be:
z0 ∼ρ H ≈
N0X
i=1
wi
0N(µ i
0,Σ i
0),(7)
and trusted GMM:
z1 ∼ρ T ≈
N1X
j=1
wj
1N(µ j
1,Σ j
1),(8)
whereµ i
0,Σ i
0 andµ j
1,Σ j
1 are mean/covariance parameters
∀i, j.z 0,z 1 represent hallucinated/trusted activations.
Eq. (4) becomes optimal mapping between GMMs:
min
u∈U
E
Z 1
0
∥ut(z)∥2dt

, dz=u t(z)dt+ √ϵdw(9)
withz 0,z 1 satisfying Eq. (7) and (8).
This formulation physically avoids excessive pertur-
bation of LVLM attention activations, preserving visual-
language alignment. Instead, it achieves hallucinated-to-
trusted state shifts via minimal integral intervention.
The Schr ¨odinger bridge solves optimal transport be-
tween distributions. Bunne et al. [3] derived a closed-
form Gaussian bridge solution for single Gaussians (i.e.,
N0 =N 1 = 1in Eq. 9 constrained by Eq. 7,8). This
yields the most probable diffusion path under Brownian mo-
tion, with Gaussian marginals at all times. Rapakoulias et
al. [23] extended this to GMMs by combining Gaussian
bridges via linear programming, ensuring theoretical feasi-
bility and performance bounds.
Thus, we establish optimal hallucination-to-trust GMM
mapping (Proposition 1), maximally preserving LVLM
vision-language alignment.
Proposition 3.1.(Optimal GMM bridge policy for
hallucination-to-trust transition) For the GMM bridge
problem in Eq. (9), letu t|ij be the optimal control policy
for the Gaussian bridge between hallucinated component
i(N(µ i
0,Σ i
0)) and trusted componentj(N(µ j
1,Σ j
1)), with
induced flowρ t|ij and optimal costJ ij.
Consider this linear program:
min
λ
X
i,j
λijJij
under optimal transport constraints withλ ij ≥0,∀i, j:
N1X
j=1
λij =w i
0, i= 1, . . . , N 0,
N0X
i=1
λij =w j
1, j= 1, . . . , N 1.
The solutionλ ∗
ij gives the optimal lower bound for
Eq. (9). The optimal mixture policy:
ut(z) =
X
i,j
ut|ij(z) ρt|ij(z)λ∗
ijP
r,ℓ ρt|rℓ(z)λ∗
rℓ
,(10)
is feasible for (9) with induced flow:
ρt(z) =
X
i,j
ρt|ij(z)λ∗
ij.
Proof sketch. For systemdz t =u(z t)dt+ √ϵdwt, the
Fokker–Planck–Kolmogorov (FPK) [13] equation uniquely
determinesρ t’s evolution and enables control policy deriva-
tion. The parabolic PDE guarantees existence/uniqueness
ofρ t, forming the basis for state-feedback control mini-
mizing quadratic costs. Additive Gaussian noise and linear
dynamics ensure tractability via Riccati equations, yielding
closed-form optimal policies.
Per Proposition 3.1, the solution uses a mixture strat-
egy weighting conditional policies byλ ∗
ijρt|ij(z), normal-
ized by P
i,j ρt|ij(z)λ∗
ij. Sinceρ t|ij(z)is Gaussian cen-
tered at the(i, j)-bridge mean att, this prioritizes strategies
with means closer toz. Applied independently per atten-
tion head, the Schr¨odinger bridge framework maps halluci-
nated to trusted Gaussian components with minimal cost.
Hereλ ∗
ij represents transport weight from hallucinatedi
to trustedj. For eachi, selectj ∗ = arg max j λ∗
ij as the
trusted counterpart for mitigation (used next section). Fig-
ure 2 shows optimal transfer examples from hallucination
to trusted manifold. These cases, from different attention
heads, demonstrate both image-level and object-level inter-
ventions. We observe that tailored interventions can be ap-
plied for distinct tokens, enabling greater customization.
3.4. Hallucination Mitigation Based on Optimal
Mapping Between Manifolds
To identify critical attention heads, we train logistic re-
gression probes over allLlayers andN h heads of the
vision-language transformer. Given activation tensorsA ∈

<!-- page 5 -->

(a) This comparison shows the distribution of trusted and hallucinated attention activations at the image level. The activations are extracted from the 73rd
attention head in layer 1 of LLaV A-1.5. As seen in the middle image, most of the eight hallucination components require noticeable interventions, varying
in both direction and scale.
(b) This comparison shows distributions of trusted and hallucinated object-level attention activations, illustrating GMM transitions. Activations originate
from LLaV A-1.5’s 12th layer, 62nd head. Of 8 hallucinated components, 6-7 require minimal intervention while the rightmost needs substantial correction.
This occurs because hallucinations typically affect single bounding boxes, leaving other image areas intact - as shown in Figure 1 (zebra example).
Figure 2. Comparison of the distributions of trusted and hallucinated attention activations at the image and object levels, including the
component composition of GMMs and the transition flow from the hallucinated GMMs to the trusted GMMs. For simplicity, we use
t-SNE [30] to map the original attention activations into a two-dimensional space before performing all subsequent operations.
RB×L×Nh×d (batch sizeB, dimensiond), we train a logis-
tic model on each head’s activations to distinguish halluci-
nated (1) vs factual (0) inputs using cross-entropy loss with
L2-regularized weights. The accuracy matrixM ∈R L×Nh
reveals top-kheads for intervention analysis (Figure 1),
where dark orange highlights indicate strong hallucination
discrimination capacity.
The mitigation strategy dynamically adjusts key at-
tention heads during answer generation to influence out-
puts. At each step, we extract the current token’s high-
dimensional attention vector, reflecting the model’s inter-
nal state critical for next-word prediction in autoregres-
sive models. Using pre-trained hallucinated and trusted
GMMs (with multiple components), and their optimal map-
ping (Section 3.3), we analyze activation distributions to
determine most probable component membership. This
clustering interprets activation roles by comparing positions
against learned patterns. Letz current =z (l,h)
t ∈R d denote
the output activation vector at steptfrom layerl, headh.
Compute posterior probabilities for hallucation GMM com-
ponents:
P(r|z) = wr
0N(z|µ r
0,Σ r
0)PN0
i=1 wi
0N(z|µ i
0,Σ i
0)
.(11)
Let
r∗ = arg max
r
P(r|z current)(12)
denote the hallucinated Gaussian component to which the
current activationz current is most likely assigned, and let
c= max
r
P(r|z current)(13)
represent the corresponding assignment probability.
The intervention uses precomputed Schr ¨odinger bridge
mappings (Section 3.3). When identifying a hallucinated
component via Eq. (12), we retrieve its trusted counterpart
throughλ ∗
ij in Eq. (10) and compute the transfer vector as
vr∗ =µ j∗
1 −µ r∗
0 , wherej ∗ = arg maxj λr∗j.
Intervention strength combines base coefficientαbase and
Eq. (13) confidence:
αdynamic =α base ·c(14)
Strong matches (c≈1) get stronger interventions, while am-
biguous cases receive milder adjustments to maintain sta-
bility. Apply the scaled vector:
h(l+1) =h (l)+
NhX
n=1
h
Al
n(h(l)) +1 top-k(l, n)αdynamicvr∗
i
P l
n
(15)
Only top-kheads (Figure 1) get modified via indicator func-
tion1 top-k, ensuring minimal perturbation while guiding ac-
tivations toward trusted distributions. This correction ap-
plies at each generation step to reduce hallucinations while
preserving generation coherence.

<!-- page 6 -->

Table 1. Performance comparison of Scalpel and other baseline methods on the POPE benchmark across three datasets — MSCOCO,
A-OKVQA, and GQA — using the LLaV A-1.5-7B model.Boldvalues indicate the best performance on each dataset and metric.
Acc.↑/F1↑ MS COCO A-OKVQA GQA
Random Popular Adversarial Random Popular Adversarial Random Popular Adversarial
Vanilla [19] 83.29/81.33 81.88/80.06 78.96/77.57 83.45/82.56 79.90/79.59 74.04/75.15 83.73/82.95 78.17/78.37 75.08/76.06
VCD [14] 87.73/87.16 85.38/85.06 80.88/81.3 86.15/86.34 81.85/82.82 74.97/77.73 86.65/86.99 80.73/82.24 76.09/78.78
OPERA [10] 89.20/88.81 86.64/86.62 81.24/81.38 88.02/84.59 83.22/84.67 73.82/77.91 88.13/88.91 79.27/82.11 75.00/78.71
ICT [6] 89.1/88.48 86.76/86.40 83.83/83.84 89.3/89.40 83.4/84.45 75.56/78.68 89.3/89.49 80.86/82.64 77.4/80.11
Scalpel (ours) 90.67/90.74 87.87/88.36 85.97/86.00 89.87/89.93 85.00/85.18 78.40/79.71 89.93/89.87 84.57/85.31 81.00/82.09
4. Experiments
4.1. Benchmarks and Experimental Setup
Polling-based Object Probing Evaluation (POPE).
POPE [17] evaluates object hallucinations in LVLMs via bi-
nary queries (e.g., “Is there a chair?”). Unlike caption-based
methods, it directly probes object recognition and hallu-
cination. The balanced dataset (27K pairs) contains 50%
real/50% absent objects from COCO [18], A-OKVQA [25],
and GQA [11]. Three sampling strategies: random, popular
(frequent objects), adversarial (challenging cases). Evalua-
tion uses Accuracy and F1.
Multimodal Model Evaluation (MME).MME [9]
comprehensively assesses LVLMs across 14 subtasks: 10
perception, 4 cognition. Perception includes object exis-
tence/count (object hallucinations), position/color (attribute
hallucinations). Cognition covers commonsense reason-
ing (CSR), numerical, translation, and code reasoning.
Accuracy-based metrics used.
Figure 3. On the MME benchmark, Scalpel outperforms prior
SOTA methods - ICT, Vanilla LLaV A-1.5 and Qwen2.5-VL. The
radar chart highlights improvements across key categories: exis-
tence, location, counting, color perception, common sense reason-
ing, and overall performance.
Implementation details.We evaluated Scalpel on two
established LVLMs: LLaV A-1.5-7B [19] and Qwen2.5-VL-
7B [2], comparing with VCD [19], OPERA, and ICT 1 hal-
lucination mitigation methods. Hyperparameters included:
top-khead selection (Eq. (15)), intervention strengthα base
(Eq. (14)), and GMM component countN 0 (Eq. (7)). To
simplify tuning while preserving discriminative power, we
enforced equal component counts for trusted/hallucinated
GMMs. For head selection, we adopted ICT’s frame-
work [6] with 1,500 trusted vs. hallucination samples
(image/object-level), training logistic regression classifiers
on attention head activations to identify top-kcritical heads.
This prioritized heads showing significant activation pat-
terns during hallucinatory phases.
4.2. Results and Discussion
Tables 1, 2, and 3 compare Scalpel with leading meth-
ods—VCD, OPERA, and ICT—across nine POPE datasets
under LLaV A-1.5 and Qwen2.5-VL frameworks. Scalpel
excels on MS COCO, A-OKVQA, and GQA, outperform-
ing prior methods in Random, Popular, and Adversarial sub-
sets. Compared to Vanilla, it improves accuracy by 7.61%
and F1 by 8.88%, with notable gains in adversarial cases
(e.g., +10.87% F1 on MS COCO-Adversarial), demonstrat-
ing strong robustness and generalization. Against ICT—the
previous best—it achieves average relative improvements
of 2.43% (Acc.) and 1.81% (F1), with no performance
drop. Largest gains appear in Popular and Adversarial sub-
sets (e.g., +4.59% Acc. on GQA-Popular), highlighting its
ability to reduce bias and handle hard data.
Table 2 presents detailed analysis of Scalpel’s modu-
lar intervention vs. ICT under LLaV A. Image-Level (w/o
obj) gains 1.63% avg F1 (+4.93% peak on A-OKVQA-
Random), with strong adversarial performance (+2.47%
Acc. on COCO-Adversarial). Object-Level (w/o img) im-
proves 1.85% avg F1 (+5.85% peak) while avoiding ICT’s
84.06→84.47 Acc. drop on GQA-Popular. Combined,
Scalpel achieves max gains (+2.43% Acc./1.81% F1) with
super-additive effects: joint modules deliver +8.22% F1 on
COCO-Adversarial over ICT’s single module. Crucially, all
18 comparative indicators (9 subsets×Acc/F1) show strict
dominance, validating Scalpel’s hierarchical design through
both superior individual modules and synergistic collabora-
tion.
Table 3 confirms Scalpel’s advantages in Qwen2.5-VL
with +5.19% average Acc. (peak +9.94% on COCO-
Random) and +7.16% F1 improvement (peak +9.20%
COCO-Popular), achieving 85.40 F1 on GQA-Adversarial
1All ICT results are from our re-implementation (based onhttps:
//github.com/THU-BPM/ICT) for fair comparison.

<!-- page 7 -->

Table 2. Performance comparison of Scalpel and the previous SOTA method ICT using the LLaV A-1.5-7Bmodel on the POPE benchmark
across three datasets: MS COCO, A-OKVQA, and GQA. Results are shown for three configurations: image-level intervention (w/o obj),
object-level intervention (w/o img), and their combination. The outputs marked with a blue background represent the results of our Scalpel.
Acc.↑/F1↑ COCO A-OKVQA GQA
Random Popular Adversarial Random Popular Adversarial Random Popular Adversarial
Vanilla [19] 83.29/81.33 81.88/80.06 78.96/77.57 83.45/82.56 79.90/79.59 74.04/75.15 83.73/82.95 78.17/78.37 75.08/76.06
ICT w/o obj 87.53/86.20 86.56/85.31 85.1/83.97 85.33/85.16 88.7/88.16 78.7/79.81 89.06/88.53 84.73/84.69 81.43/81.97
Scalpel w/o obj 89.07/88.18 87.07/86.24 85.77/85.29 89.53/89.36 85.50/85.29 79.30/79.90 89.67/89.40 84.63/84.70 81.47/82.08
ICT w/o img 89.4/88.85 86.83/86.49 83.7/83.82 83.23/84.34 89.13/89.26 75.6/78.72 89.2/89.40 84.06/83.44 74.3/74.91
Scalpel w/o img 89.90/90.08 87.6/86.97 85.53/84.66 89.47/89.27 85.27/85.22 78.47/80.18 89.87/89.81 84.47/85.04 80.93/81.89
ICT [6] 89.1/88.48 86.76/86.40 83.83/83.84 89.3/89.40 83.4/84.45 75.56/78.68 89.3/89.49 80.86/82.64 77.4/80.11
Scalpel (ours) 90.67/90.74 87.87/88.36 85.97/86.00 89.87/89.93 85.00/85.18 78.40/79.71 89.93/89.87 84.57/85.31 81.00/82.09
Table 3. Performance comparison of Scalpel and the previous SOTA method ICT using the Qwen2.5-VL-7Bmodel on the POPE benchmark
across three datasets: MS COCO, A-OKVQA, and GQA. Results are shown for three configurations: image-level intervention (w/o obj),
object-level intervention (w/o img), and their combination. The results highlighted with a blue background were produced by our Scalpel.
Acc.↑/F1↑ COCO A-OKVQA GQA
Random Popular Adversarial Random Popular Adversarial Random Popular Adversarial
Vanilla [2] 85.4/82.97 85.13/82.71 84.83/82.42 87.76/86.41 86.43/85.15 81.5/80.78 87.1/85.63 84/82.74 81.6/80.65
ICT w/o obj 86.4/84.31 86.3/84.27 85.7/83.53 89.93/89.15 87.46/86.76 82.5/82.54 88.06/87.01 83.9/83.23 81.63/81.17
Scalpel w/o obj 86.27/84.84 88.97/87.87 86.57/84.87 91.20/90.54 88.83/88.37 81.70/80.94 88.03/89.67 84.43/84.16 85.47/85.96
ICT w/o img 85.4/82.97 85.4/83.07 85.2/82.90 88.76/87.64 86.63/85.50 81.7/81.16 87.16/85.66 83.93/82.67 81.33/80.40
Scalpel w/o img 86.37/84.26 85.80/83.60 85.37/82.98 88.73/87.66 87.17/85.99 82.57/82.08 87.73/86.02 85.27/85.89 83.00/82.29
ICT [6] 87.53/85.84 86.76/84.94 86.16/84.32 88.96/87.90 87.43/86.39 83.6/83.02 88.96/87.89 86.43/85.47 84.1/83.53
Scalpel (ours) 91.17/90.41 90.80/90.33 88.83/88.49 91.00/90.55 90.50/90.41 84.97/85.82 89.20/89.84 87.20/86.68 85.90/85.40
Figure 4. Ablation study of Scalpel under different GMM com-
ponent settings, evaluated on the POPE benchmark (MS COCO
dataset) with LLaV A-1.5-7B.
(vs. 80.65). Modular analysis shows Image-Level (w/o
obj) +3.07% avg F1 (A-OKVQA-Random 90.54 vs. 89.15)
and Object-Level (w/o img) +1.92% avg F1 (GQA-Popular
85.89 vs. 82.67). Dual-module synergy delivers 1.67× F1
gain over ICT (Scalpel 2.97% vs. ICT 1.78%), maintaining
1.77% absolute advantage on GQA-Adversarial. Adversar-
ial F1 gains exceed ICT by 1.65× (GQA: 5.89% vs. 3.57%).
Figure 3 shows multi-dimensional improvements:
LLaV A- based Scalpel scores 758.09 (+8.52% vs. Vanilla),
excelling in counting (+14.46%), positioning (+14.93%),
and common sense (+11.84%). Removing modules
reduces performance (w/o obj: 728.09/+4.23%; w/o
img: 736.66/+5.45%), highlighting image-object synergy.
Qwen-based Scalpel scores 851.42 (+6.14%), achieving
SOTA in positioning (+14.29%), color (+5.56%), and
common sense (+15.80%). Module analysis reveals:
image removal maintains positional/color performance
but degrades reasoning (+12.86% vs. +15.80%), while
object removal causes 9.09% counting drop but improves
color (+8.33%) and reasoning (+14.03%) via reduced
interference. Scalpel surpasses ICT (822.14/+2.49%), with
absolute dominance in complex reasoning (141.42 vs.
137.85).
4.3. Ablation study
Scalpel’s efficacy was validated through three aspects:
GMM components (intervention granularity), image-level,
and object-level interventions. Tables 2, 3, and Figure 4
present ablation studies. Specifically, Tables 2 and 3 com-
pare Scalpel variants (LLaV A-1.5/Qwen2.5-VL) across 9
POPE subsets when removing modules. Quantitative results
show: adding image-level intervention improves LLaV A
by +7.46% Acc./+8.43% F1 (Qwen: +2.32%/+3.69%);
object-level adds +7.37% Acc./+8.14% F1 (LLaV A) and
+1.09%/+1.52% (Qwen). Combined modules show peak
gains: +7.61% Acc./+8.78% F1 (LLaV A) and +4.70%
Acc./+6.48% F1 (Qwen). These cross-model improvements
validate dual levels’ critical role in hallucination mitigation.

<!-- page 8 -->

Figure 5. A comparative analysis of Scalpel, ICT, and the original Qwen2.5-VL-7B across selected test cases is presented.Qdenotes
the question,GTrepresents the ground truth, and each method’s answer follows its name inbold. Extended hallucinated responses are
highlighted in red for emphasis. Notably, the original questions included the instruction “Please answer yes or no” immediately after the
question mark, which has been omitted to optimize space usage.
Figure 4 analyzes GMM component count’s impact on
hallucination reduction and accuracy. Tests on COCO’s
POPE subsets (Random, Popular, Adversarial) evaluated
1–32 components. Higher counts improve performance,
with 32 components yielding best results across all subsets.
Image-level intervention peaks at 16 components for Ran-
dom/Popular, fewer for Adversarial. Object-level requires
higher counts even in adversarial cases. Complementary
patterns justify recommending full Scalpel with 32 GMM
components, balancing granularity needs across interven-
tion levels and data complexities.
Figure 5 presents qualitative comparisons. Scalpel sur-
passes ICT in object perception, spatial analysis, scenario
suitability, and action trace recognition. First, compared to
ICT/Vanilla Qwen2.5-VL, Scalpel enhances image under-
standing accuracy by removing hallucinated elements (e.g.,
non-visable floors) and detecting missed objects (e.g., fris-
bees/chairs), significantly reducing hallucinations. Second,
Scalpel perfectly matches GT in object positioning (e.g.,
occlusion under kites/umbrellas) and spatial arrangements
(elephant positioning), while ICT errors arise from over-
reliance on localized features/color cues. Third, Scalpel de-
tects subtle visual traces (e.g., cake cuts) rather than sur-
face appearances. It also eliminates ICT’s redundant as-
sumptions/logical inconsistencies (e.g., “requires verifica-
tio”), producing streamlined reasoning paths. These advan-
tages reflect superior vision-language alignment and noise
filtering during complex reasoning.
5. Conclusion
To correct hallucinated attention activations in LVLM
Transformers without added computational cost, we pro-
pose Scalpel. This method models hallucinated and
trusted attention activations via Gaussian Mixture Models
(GMMs), yielding hallucinated and trusted GMMs respec-
tively. The Schr”odinger Bridge (equivalent to entropic op-
timal transport) then constructs the minimal transport-cost
correction scheme by treating these GMMs as marginal dis-
tributions. This preserves LVLMs’ data-driven learning ca-
pabilities while effectively suppressing hallucinations. Ex-
periments across datasets show Scalpel achieves SOTA per-
formance, surpassing non-customized methods. A promis-
ing future direction is investigating refined correction for-
mulations via the infinite-component limit of GMMs.
References
[1] Robust text-independent speaker identification using gaus-
sian mixture speaker models.IEEE transactions on speech

<!-- page 9 -->

and audio processing, 3(1):72–83, 1995. 3
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report.arXiv preprint
arXiv:2502.13923, 2025. 2, 6, 7
[3] Charlotte Bunne, Ya-Ping Hsieh, Marco Cuturi, and Andreas
Krause. The schr ¨odinger bridge between gaussian measures
has a closed form. InInternational Conference on Artificial
Intelligence and Statistics, pages 5802–5833. PMLR, 2023.
4
[4] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt.
Discovering latent knowledge in language models without
supervision.arXiv preprint arXiv:2212.03827, 2022. 3
[5] Cong Chen, Mingyu Liu, Chenchen Jing, Yizhou Zhou,
Fengyun Rao, Hao Chen, Bo Zhang, and Chunhua Shen.
Perturbollava: Reducing multimodal hallucinations with per-
turbative visual training.arXiv preprint arXiv:2503.06486,
2025. 1, 2
[6] Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Lin-
feng Zhang, Lijie Wen, and Xuming Hu. Ict: Image-object
cross-level trusted intervention for mitigating object hallu-
cination in large vision-language models.Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognitio, 2025. 2, 3, 6, 7
[7] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang,
Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An
open-source chatbot impressing gpt-4 with 90%* chatgpt
quality.See https://vicuna. lmsys. org (accessed 14 April
2023), 2(3):6, 2023. 2
[8] Alessandro Favero, Luca Zancato, Matthew Trager, Sid-
dharth Choudhary, Pramuditha Perera, Alessandro Achille,
Ashwin Swaminathan, and Stefano Soatto. Multi-modal hal-
lucination control by visual information grounding. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 14303–14312, 2024. 1, 2
[9] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin,
Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li,
Xing Sun, et al. Mme: A comprehensive evaluation bench-
mark for multimodal large language models.arXiv preprint
arXiv:2306.13394, 2023. 2, 6
[10] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Con-
ghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and
Nenghai Yu. Opera: Alleviating hallucination in multi-
modal large language models via over-trust penalty and
retrospection-allocation. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 13418–13427, 2024. 1, 2, 6
[11] Drew A Hudson and Christopher D Manning. Gqa: A new
dataset for real-world visual reasoning and compositional
question answering. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
6700–6709, 2019. 6
[12] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac
Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al.
Language models (mostly) know what they know.arXiv
preprint arXiv:2207.05221, 2022. 3
[13] A Kolmogorov. ¨Uber die analytischen methoden in der
wahrscheinlichkeitstheorie.Math Annal, 104:415–458,
1931. 4
[14] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian
Lu, Chunyan Miao, and Lidong Bing. Mitigating object hal-
lucinations in large vision-language models through visual
contrastive decoding. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
13872–13882, 2024. 1, 2, 6
[15] Christian L ´eonard. A survey of the schr¨odinger problem and
some of its connections with optimal transport.Discrete and
Continuous Dynamical Systems-Series A, 34(4):1533–1574,
2014. 3
[16] Kenneth Li, Oam Patel, Fernanda Vi ´egas, Hanspeter Pfister,
and Martin Wattenberg. Inference-time intervention: Elicit-
ing truthful answers from a language model.Advances in
Neural Information Processing Systems, 36:41451–41530,
2023. 2, 3
[17] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin
Zhao, and Ji-Rong Wen. Evaluating object hallucina-
tion in large vision-language models.arXiv preprint
arXiv:2305.10355, 2023. 2, 6
[18] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
Computer vision–ECCV 2014: 13th European conference,
zurich, Switzerland, September 6-12, 2014, proceedings,
part v 13, pages 740–755. Springer, 2014. 6
[19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023. 1, 2, 6, 7
[20] Ziyu Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang
Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua
Lin, and Jiaqi Wang. Mia-dpo: Multi-image augmented di-
rect preference optimization for large vision-language mod-
els.arXiv preprint arXiv:2410.17637, 2024. 1, 2
[21] Jinda Lu, Junkang Wu, Jinghan Li, Xiaojun Jia, Shuo Wang,
YiFan Zhang, Junfeng Fang, Xiang Wang, and Xiangnan
He. Dama: Data-and model-aware alignment of multi-modal
llms.arXiv preprint arXiv:2502.01943, 2025. 1, 2
[22] Gabriel Peyr ´e, Marco Cuturi, et al. Computational optimal
transport: With applications to data science.Foundations
and Trends® in Machine Learning, 11(5-6):355–607, 2019.
3
[23] George Rapakoulias, Ali Reza Pedram, and Panagiotis Tsio-
tras. Go with the flow: Fast diffusion for gaussian mixture
models.arXiv preprint arXiv:2412.09059, 2024. 4
[24] William Saunders, Catherine Yeh, Jeff Wu, Steven Bills,
Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing
models for assisting human evaluators.arXiv preprint
arXiv:2206.05802, 2022. 3
[25] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark,
Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A
benchmark for visual question answering using world knowl-
edge. InEuropean conference on computer vision, pages
146–162. Springer, 2022. 6

<!-- page 10 -->

[26] Nishant Subramani, Nivedita Suresh, and Matthew E Peters.
Extracting latent steering vectors from pretrained language
models.arXiv preprint arXiv:2205.05124, 2022. 3
[27] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu,
Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui,
Yu-Xiong Wang, Yiming Yang, et al. Aligning large multi-
modal models with factually augmented rlhf.arXiv preprint
arXiv:2309.14525, 2023. 1, 2
[28] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste
Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
Llama: Open and efficient foundation language models.
arXiv preprint arXiv:2302.13971, 2023. 2
[29] Chongjun Tu, Peng Ye, Dongzhan Zhou, Lei Bai, Gang Yu,
Tao Chen, and Wanli Ouyang. Attention reallocation: To-
wards zero-cost and controllable hallucination mitigation of
mllms.arXiv preprint arXiv:2503.08342, 2025. 1, 2
[30] Laurens Van der Maaten and Geoffrey Hinton. Visualizing
data using t-sne.Journal of machine learning research, 9
(11), 2008. 5
[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need.Advances in neural
information processing systems, 30, 2017. 1, 3
[32] Chenguang Wang, Xiao Liu, and Dawn Song. Lan-
guage models are open knowledge graphs.arXiv preprint
arXiv:2010.11967, 2020. 3
[33] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, et al. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution.arXiv preprint
arXiv:2409.12191, 2024. 1
[34] Yongxin Wang, Meng Cao, Haokun Lin, Mingfei Han, Liang
Ma, Jin Jiang, Yuhao Cheng, and Xiaodan Liang. Eaco: En-
hancing alignment in multimodal llms via critical observa-
tion.arXiv preprint arXiv:2412.04903, 2024. 1, 2
[35] Sangmin Woo, Donguk Kim, Jaehyuk Jang, Yubin Choi, and
Changick Kim. Don’t miss the forest for the trees: Atten-
tional vision calibration for large vision language models.
arXiv preprint arXiv:2405.17820, 2024. 1, 2
[36] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke
Shen, Yunhai Tong, and Mengdi Wang. Mmada: Mul-
timodal large diffusion language models.arXiv preprint
arXiv:2505.15809, 2025. 1
[37] Hao Yin, Guangzong Si, and Zilei Wang. Clearsight: Vi-
sual signal enhancement for object hallucination mitigation
in multimodal large language models. InProceedings of the
Computer Vision and Pattern Recognition Conference, pages
14625–14634, 2025. 1, 2
[38] Mengxi Zhang, Wenhao Wu, Yu Lu, Yuxin Song, Kang
Rong, Huanjin Yao, Jianbo Zhao, Fanglong Liu, Haocheng
Feng, Jingdong Wang, et al. Automated multi-level prefer-
ence for mllms.Advances in Neural Information Processing
Systems, 37:26171–26194, 2024. 2
[39] Xiaofeng Zhang, Yihao Quan, Chaochen Gu, Chen Shen,
Xiaosong Yuan, Shaotian Yan, Hao Cheng, Kaijie Wu, and
Jieping Ye. Seeing clearly by layer two: Enhancing atten-
tion heads to alleviate hallucination in lvlms.arXiv preprint
arXiv:2411.09968, 2024. 2
[40] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mo-
hamed Elhoseiny. Minigpt-4: Enhancing vision-language
understanding with advanced large language models.arXiv
preprint arXiv:2304.10592, 2023. 1
