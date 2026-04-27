# references/132_retrieval_heads_are_dynamic.pdf

<!-- page 1 -->

Retrieval Heads are Dynamic
Yuping Lin1*, Zitao Li2†, Yue Xing1, Pengfei He1, Yingqian Cui1,
Yaliang Li3,Bolin Ding 3,Jingren Zhou 3,Jiliang Tang 1,
1Michigan State University, 2Zoom Communications, 3Tongyi Lab, Alibaba Group,
{linyupin, xingyue1, hepengf1, cuiyingq, tangjili}@msu.edu,
zitao.li@zoom.us,
{yaliang.li, bolin.ding, jingren.zhou}@alibaba-inc.com
Abstract
Recent studies have identified “retrieval heads”
in Large Language Models (LLMs) responsi-
ble for extracting information from input con-
texts. However, prior works largely rely on
static statistics aggregated across datasets, iden-
tifying heads that perform retrieval on average.
This perspective overlooks the fine-grained tem-
poral dynamics of autoregressive generation.
In this paper, we investigate retrieval heads
from a dynamic perspective. Through extensive
analysis, we establish three core claims: (1)
Dynamism: Retrieval heads vary dynamically
across timesteps; (2) Irreplaceability: Dynamic
retrieval heads are specific at each timestep and
cannot be effectively replaced by static retrieval
heads; and (3) Correlation: The model’s hidden
state encodes a predictive signal for future re-
trieval head patterns, indicating an internal plan-
ning mechanism. We validate these findings on
the Needle-in-a-Haystack task and a multi-hop
QA task, and quantify the differences on the
utility of dynamic and static retrieval heads in
a Dynamic Retrieval-Augmented Generation
framework. Our study provides new insights
into the internal mechanisms of LLMs.
1 Introduction
Recently, there is a growing interest in Large
Language Models (LLMs) (Radford et al., 2019;
Brown et al., 2020; Vaswani et al., 2017; Chowd-
hery et al., 2023; Hoffmann et al., 2022; Touvron
et al., 2023) to understand how they process con-
text, particularly focusing on their ability to extract
key information from the input. Prior work has
shown that although LLMs demonstrate strong in-
context learning abilities (Garg et al., 2022; Xie
et al., 2021), their ability to utilize long contexts
often needs improvement (Liu et al., 2024; Press
et al., 2023). A line of mechanistic interpretability
*Work done during internship at Tongyi Lab, Alibaba
Group.
†Work done during at Tongyi Lab, Alibaba Group.
Thebestthing
to do in San
Francisco
is to eat a
sandwich
and sit in DoloresPark on a
sunny
day
.
Generation Timestep
L5-H10
L8-H1
L16-H1
L16-H23
L20-H1
L20-H14
L20-H23
L24-H27
L27-H5
L27-H7
Attention Head Index
Figure 1:Dynamism of Retrieval Heads.The retrieval
scores of individual attention heads fluctuate across the
generation process.Dark colorindicates heads having
a retrieval score of 1, as defined by Equation (1). The
x-axis denotes the generation step, labeled by the token
generated at that step. The y-axis shows the 10 most
variated retrieval heads, selected based on their retrieval
score variance over the entire generation process. L x-
Hy denotes the y-th head (starting from 0) on the layer
x(starting from 0).
works suggests that attention heads exhibit func-
tional specialization (V oita et al., 2019; Michel
et al., 2019; Elhage et al., 2021): For example,
a pioneer work (Wu et al., 2024) analyzes the
model from an attention-head perspective, iden-
tifying a specific set of heads termed “retrieval
heads” that are responsible for the copy-paste be-
havior of LLMs from the given inputs. Recent
studies (Zhang et al., 2025; Fu et al., 2024) provide
further evidence of the existence of retrieval heads,
even in tasks requiring complex reasoning.
While these works provide valuable insights on
the mechanism of LLMs, they have been mainly
constrained to afixedsubset of attention heads.
For example, Wu et al. (2024) aggregated attention
patterns across datasets to find the heads which fre-
quently perform copy-paste operations, identifying
a fixed set of heads for each model that perform
retrieval on average. However, treating retrieval
heads as a fixed set assumes that this average behav-
ior is a heuristic approximation for the model’s real-
time operation. Given the autoregressive nature of
LLMs, it is natural to question whether the retrieval
heads should instead be adynamicset conditioned
1
arXiv:2602.11162v1  [cs.CL]  7 Jan 2026

<!-- page 2 -->

on the given context. Relying on static definitions
risks oversimplifying the model’s real-time behav-
ior, as heads that are statistically dominant may
not be active at every critical timestep, while “less
significant” heads might play irreplaceable roles in
specific contexts. As illustrated in Figure 1, the set
of heads acting as retrieval heads actually fluctuates
significantly across token generation steps. This
observation challenges the completeness of static
definitions, raising the following fundamental ques-
tions:1) How does the set of retrieval heads evolve
during generation? 2) Are these dynamic heads
functionally interchangeable with static ones? 3)
Is this dynamism predictable given a model and a
context?
To answer these questions, we present the first
systematic study of retrieval heads from adynamic
perspective. Our key contributions are as follows:
1. We demonstrate that the retrieval heads are
highlydynamicthat statistical methods fail to
capture. (Claim 1 in Section 3.2)
2. Given the dynamic nature of the retrieval head
pattern, we show that the specific retrieval heads
at a given generation step arenot replaceable,
and ablating them causes severe performance
degradation. (Claim 2 in Section 3.3)
3. We reveal that the model’s final hidden state ex-
hibits a strongcorrelationwith future retrieval
head patterns, revealing a planning mechanism
of LLMs. (Claim 3 in Section 3.4)
4. While the above claims are based on a Needle-
in-a-Haystack task in Section 3, we validate
them in a question-answering task where rea-
soning efforts are needed. (Section 4)
5. We use the dynamic and static retrieval heads
in a Dynamic RAG scenario to compare their
practical utility, demonstrating that dynamically
selecting heads based on the current generative
state significantly improves retrieval accuracy
and downstream performance compared to static
retrieval heads. (Section 5)
We hope these findings will serve as a founda-
tion for future research in model interpretability
and the development of more precise, state-aware
intervention techniques.
2 Related Works
Mechanistic InterpretabilityUnderstanding the
internal mechanisms of Transformer-based mod-
els (Vaswani et al., 2017) has been a focal point
of recent research. Mechanistic interpretability
has emerged as a principled approach to under-
standing neural networks beyond input-output be-
havior (Olah et al., 2020; Elhage et al., 2021).
Early work by Olsson et al. (2022) identified In-
duction Heads, a specialized circuit responsible for
in-context learning by copying previous tokens that
follow similar patterns. Subsequent work further
investigated induction-like circuits (Elhage et al.,
2022; Wang et al., 2022). This laid the theoreti-
cal groundwork for understanding how attention
heads perform “copy-paste” operations. In the con-
text of long-sequence modeling, Xiao et al. (2023)
discovered Attention Sinks, revealing that models
dedicate massive attention to initial tokens (e.g.,
BOS) to maintain numerical stability. Related anal-
yses have also examined positional and numerical
artifacts in attention mechanisms (Press et al., 2021;
Su et al., 2024a). Furthermore, studies on the “Lost
in the Middle” phenomenon (Liu et al., 2024) have
highlighted the non-uniform capability of models
to access information across long contexts. These
studies primarily focus on static circuit structures
or attention biases, and do not fully explain how
the model dynamically modulates its attention al-
location step-by-step to perform precise retrieval
during the autoregressive generation.
Retrieval HeadsInformation retrieval within
LLMs has been studied both implicitly through
attention mechanisms and explicitly through
retrieval-augmented generation (RAG) frame-
works (Lewis et al., 2020; Borgeaud et al., 2022).
Building on the concept of functional specializa-
tion, recent studies have isolated specific attention
heads responsible for information retrieval. Wu
et al. (2024) pioneered this direction by identify-
ing Retrieval Heads via the Needle-in-a-Haystack
(NIAH) test, characterizing them as a sparse, intrin-
sic subset of heads that perform copy-paste oper-
ations from long contexts. Addressing the limita-
tions of synthetic benchmarks, Zhang et al. (2025)
proposed QRHead, which refines head detection us-
ing query-aware attention scores on realistic tasks
to improve downstream retrieval and re-ranking
performance. Similarly, Fu et al. (2024) intro-
duced HeadKV , a method that leverages retrieval
and reasoning importance scores to perform head-
level KV cache compression, significantly out-
performing layer-level methods like SnapKV (Li
et al., 2024), H2O (Zhang et al., 2023), and Pyra-
midKV (Cai et al., 2024). A common limitation
across these works is their reliance on astatic per-
2

<!-- page 3 -->

spective. However, this approach overlooks the
temporal dynamismof the generation process.
3 Analysis of Dynamic Retrieval Heads
3.1 Setup
This section focuses on the traditionalcopy-paste
retrieval headas in Wu et al. (2024). This type of
retrieval head considers the exact copy-paste of the
input tokens to the next generated token.
To better trace and analyze the exact copy-paste
behavior, following Wu et al. (2024), we consider
the Needle-in-a-Haystack (NIAH) task (Kamradt,
2023), which evaluates a model’s ability to pre-
cisely retrieve the specific piece of information (the
“needle”) embedded at a random location within a
long, distracting document (the “haystack”).
Definition of Retrieval HeadFollowing Wu
et al. (2024), to define the copy-paste retrieval head,
an attention head is considered to be performing a
retrieval operation if and only if two conditions are
satisfied: (1) at the current inference step, the gen-
erated token is identical to the token receiving the
highest attention weight from that head, and (2) the
token with the highest attention weight lies within
the “needle” context, i.e., it is aneedle token. When
these conditions are satisfied, its retrieval score is
set to be 1.1
Formally, let i∗ = arg max i(ah,t
i ) be the index
of the token that receives the maximum attention
from head h at timestep t, where ah,t is the vector
of attention scores from the final token of the input
xt (as query) to all tokens in xt (as keys) for head
h, i.e., ah,t = AttnScoreh[t,:] . The retrieval score
of headhon inputx t is then defined as:
Scopy−paste(xt, h) =1

i∗ ∈I needle ∧x t
i∗ = ˆy

(1)
where Ineedle is the set of indices for tokens within
the needle, and ˆyis the token predicted to be gen-
erated at the current timestep.
Overview of ClaimsGiven the above task de-
scription and definition of retrieval heads, we
present our central claims:
• Claim 1: Dynamism.The patterns of retrieval
heads are dynamic throughout the autoregressive
generation process.
1The original work (Wu et al., 2024) normalizes this score
by the length of the needle. We omit this normalization, as-
signing a binary score of 1 and 0, because our analysis is
conducted at the token level, in contrast to the sample-level
analysis of the original work.
• Claim 2: Irreplaceability.The retrieval func-
tionality of the specific retrieval heads at a given
timestep cannot be replaced by other heads. If
these heads are disabled, the model will suffer
from performance degradation.
• Claim 3: Correlation.A strong correlation ex-
ists between the model’s hidden state and the
patterns of retrieval heads in the future.
3.2 Retrieval Heads are Dynamic
Different from existing literature (Wu et al., 2024;
Zhang et al., 2025; Fu et al., 2024) where a large
corpus of samples are collected to identify a set of
statistically significant retrieval heads (i.e.,static
retrieval heads), we argue the existence of unique
patterns of retrieval heads that emerge at individ-
ual timesteps during the generation process (i.e.,
dynamic retrieval heads). Specifically, the re-
trieval score of attention heads fluctuates across
timesteps. Therefore, we hypothesize that the dy-
namic retrieval heads at a particular timestep do
not always align with the static retrieval heads.
To verify our hypothesis, Figure 1 in Section 1
plots the retrieval scores calculated as per Equa-
tion (1) for some attention heads over the course
of an autoregressive generation process for a given
sample. The plot clearly demonstrates that the re-
trieval scores for individual heads fluctuate signif-
icantly across timesteps, confirming the dynamic
nature of retrieval heads.
Furthermore, to rigorously quantify this dy-
namism, we conducted a statistical analysis. The
results for all models are summarized in Table 1.
Model Jaccard w/ Static Adj. Jaccard Entropy
llama3.1-8b 0.3512 0.2793 3.8154
llama3.2-3b 0.3126 0.3188 3.0083
qwen3-8b 0.4611 0.3668 4.1038
llama2-13b 0.2077 0.4979 4.8973
phi4-mini 0.1845 0.5056 3.5532
Table 1:Quantitative Statistics of Retrieval Head
Dynamism. Jaccard w/ Static: Similarity between
dynamic retrieval heads and the top-20 static retrieval
heads; lower values indicate less static heads are in the
set of dynamic heads.Adj. Jaccard: Similarity of dy-
namic retrieval heads between consecutive steps; lower
values indicate rapid switching.Entropy: Measure of
distribution spread; higher values indicate broader head
involvement into dynamic retrieval.
There are several observations from the table:
First,the low Jaccard similarity (“Jaccard w/
Static”, ranging from 0.1845 to 0.4611) indicate
3

<!-- page 4 -->

5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
Accuracy (%)
Figure 2:Impact of Head Ablation on Retrieval Performance.Comparison of NIAH test scores after masking
three different sets of attention heads: dynamic retrieval heads, top-ranked static retrieval heads, and randomly
selected heads on llama3.1-8b. The x-axis shows different haystack lengths. The y-axis shows the different locations
(“depth”) where the needle is inserted. The evaluation metric is Accuracy (exact string match). The average number
of masked heads is kept consistent across all conditions. Masking dynamic heads (identified at each timestep via
Eq. (1)) results in the most significant performance degradation, indicating their critical role in retrieval.
that only a small proportion of static heads are iden-
tified as retrieval heads in each generation step.Sec-
ond,the lowAdjacent Jaccardsimilarity scores
(ranging from 0.2793 to 0.5056) reveal a rapid
turnover rate, confirming that the model frequently
switches its active retrieval heads from one token
to the next.Finally,while Jaccard metrics measure
the involvement of the static heads, theentropy
indicates that many dynamic retrieval heads are
not static heads. As a baseline, a uniform distri-
bution over just 20 heads would yield an entropy
of ln 20≈2.99 . The entropy values in the table
exceed 3.0 (up to 4.89), indicating that at least 20
heads are dynamic retrieval heads. Together with
the small Jaccard similarity (at most 6-10 static
heads are active in each step), this implies that at
least 10-14 other heads are dynamic retrieval heads.
(See Appendix A.3 for detailed formulations).
3.3 Dynamic Retrieval Heads are
Irreplaceable
We further claim that the dynamic retrieval heads
at a specific timestepcan not be replaced by the
static retrieval heads.
3.3.1 What will Happen without Dynamic
Retrieval Heads?
To verify Claim 2, we conducted a head ablation
study. The experiment procedure is as follows:
1. For each token generation step, we first execute
a standard forward pass without any interven-
tions. Discard the generated token.
2. We locate the retrieval heads at this step, as
defined by Equation (1), and label them as the
set of dynamic retrieval heads.
3. We mask all the dynamic retrieval heads of this
timestep, and execute a second forward pass to
re-generate the token for this timestep.
For comparison, we considered two baseline ap-
proaches. We masked heads drawn from either (a)
the top-ranked static retrieval heads or (b) a ran-
domly selected set of heads. For fair comparison,
we mask the same number of static retrieval head-
s/random heads as the average number of masked
dynamic retrieval heads. (Details in Appendix A.1)
This experimental design allows us to directly
test our hypothesis: if dynamic retrieval heads in-
deed carry the primary functionality of retrieval at
a given timestep, then masking them should cause a
significantly greater performance degradation than
masking any other set of heads.
Figure 2 presents the NIAH test results onmeta-
llama/Llama-3.1-8B-Instruct(llama3.1-8b). The
results clearly show that masking the dynamic re-
trieval heads leads to the most severe degradation
in retrieval performance, as the colors are almost
red. This far exceeds the impact of masking an
equal number of static or random heads, in which
only some of the pieces in the heatmap are red.
For ROUGE-L results and other models, see Ap-
pendix B.1. This verifies our hypothesis that the
dynamic retrieval heads are not replaceable.
3.3.2 To What Extent do Static Retrieval
Heads Help?
To further investigate the irreplaceability of dy-
namic retrieval heads, we also designed a progres-
sive ablation study to analyze the model’s compen-
satory mechanisms.
4

<!-- page 5 -->

Our primary objective is to quantify the extent to
which the model compensates for the loss of these
optimal heads by activating additional strong static
retrieval heads. To measure this, we first identify
the set ofcompensated headsat each timestep, i.e.,
heads that become retrieval heads only after k dy-
namic heads are ablated. We then count how many
of these compensated heads are in the top-20 static
retrieval heads. Detailed experiment descriptions
can be found in Appendix A.2.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6
7|compensated  static_top20|
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy (%)
Figure 3:Irreplaceability of Dynamic Retrieval
Heads.The plots show the degradation in NIAH perfor-
mance as an increasing number (k) of dynamic retrieval
heads are masked on llama3.1-8b. Even though the
model compensates by activating top-20 static retrieval
heads (blue line, left y-axis), the overall retrieval perfor-
mance, measured by Accuracy (red line, right y-axis),
continues to decline sharply. This demonstrates that
static retrieval heads cannot effectively substitute for
context-specific dynamic heads.
Figure 3 illustrates the result on llama3.1-8b,
plotting the retrieval performance (red line, right y-
axis) against the number of compensated heads that
overlap with the top-20 static retrieval heads (blue
line, left y-axis). The result reveals two critical ob-
servations. First, as the number of masked dynamic
heads (k) increases, the model attempts to compen-
sate by activating new heads as retrieval heads. A
significant proportion of these compensated heads
are indeed static retrieval heads, specifically, the
overlap with the top-20 static set rises sharply to
range between 4.5 and 7 for k≥5 . Second, how-
ever, this compensation is insufficient. As shown
by the red line, the overall retrieval performance
degrades significantly. For instance, the Accuracy
drops sharply from 1.0 to 0.0 as k reaches 20, sug-
gesting that the function of dynamic retrieval heads
is irreplaceable, and cannot be fully compensated
for by static retrieval heads. For ROUGE-L metric
and other models, see Appendix B.2.
3.4 Retrieval Scores are Correlated with
Hidden States
Our final claim is that astrong correlationex-
ists between the model’s hidden state and its fu-
ture retrieval activations, indicating that the model
may proactively plan ahead its functional behavior,
specifically retrieval in this case.
To validate this, we employed Canonical Cor-
relation Analysis (CCA) with atemporal offset,
denoted as k. Specifically, we measured the linear
correlation between thefinal hidden state(the out-
put embedding of the last input token at the last
layer) at timestepnand theretrieval scoresof all
attention heads at a future timestep n+k . Detailed
experiment settings can be found in Appendix A.4.
0 1 2 3 4 5 6 7 8 9 10
T emporal Distance k
0.4
0.5
0.6
0.7
0.8
0.9Canonical Correlation
0.965
0.931 0.915
0.835 0.816 0.805 0.792 0.776 0.753 0.742 0.743
Correlation Decay with Temporal Distance
Component 1
T op-10 mean
T op-50 mean
Figure 4:Predictive Correlation Between Hidden
States and Future Retrieval Scores.Canonical Cor-
relation Analysis (CCA) coefficients between the final
hidden state at timestep n and the retrieval scores at
a future timestep n+k . The plot shows the decay of
the leading (Top-1) canonical correlation, as well as
the average of the Top-10 and Top-50 correlations, as
the temporal offset k increases. The high correlation at
k >0 demonstrates the model’s anticipatory encoding
of future retrieval intent.
As shown in Figure 4, the canonical correlation
decreases as the temporal offset k increases. At
k= 0 , the first canonical correlation is an ex-
ceptionally high 0.966, confirming a strong syn-
chronous relationship between the hidden states
and the retrieval scores. Besides, this correlation
remains extremely high for future steps, at 0.931
for k= 1 and 0.915 for k= 2 , indicating that the
model’s state is pre-configured for retrieval opera-
tions several steps before they are executed.
While CCA demonstrates a strong linear rela-
tionship in this model, we additionally trained an
MLP probe to further capture the non-linear rela-
tionships. We focused this analysis on the k= 0
case, i.e., the retrieval scores at the same timestep
as the hidden states. The probe’s task was to learn
the mapping from the hidden state to the specific
5

<!-- page 6 -->

retrieval score pattern of all heads. Detailed experi-
ment settings can be found in Appendix A.5.
As shown in Table 2, the probes achieve strong
performance across all models, with F1-scores
ranging from 0.80 to 0.86 and AU-PRC values
all exceeding 0.88, confirming that the dynamic
retrieval head patterns are predictable.
LLM F1 Precision Recall AUPRC
llama3.1-8b 0.8349 0.8344 0.8353 0.9173
llama3.2-3b 0.8456 0.8564 0.8351 0.9289
qwen3-8b 0.8566 0.8780 0.8362 0.9339
llama2-13b 0.8336 0.8455 0.8220 0.9183
phi4-mini 0.8038 0.8219 0.7865 0.8862
Table 2:Performance of MLP Probes in Decoding
Retrieval Scores.The probes were trained to predict
retrieval head scores from the final hidden state for var-
ious LLMs. The reported metrics (Precision, Recall,
F1-Score) are calculated at the optimal decision thresh-
old, which was determined by maximizing the F1-score
on the validation set’s Precision-Recall curve.
4 Generalizing Dynamic Retrieval Heads
in a Question Answering Task
In Section 3, we systematically investigated three
core claims of retrieval heads within the controlled
experimental setting of the NIAH task. However,
NIAH represents a simplified “copy-paste” sce-
nario where the retrieved token is directly gen-
erated. To consider reasoning tasks, we need a
broader definition of “retrieval” that based on atten-
tion allocation rather than token copying.
4.1 A Reasoning-Oriented Definition of
Retrieval Score
In complex reasoning tasks such as Question An-
swering, the model’s behavior goes beyond sim-
ple “copy-paste” operations. The model must in-
tegrate multiple supporting facts to derive an an-
swer. Therefore, we propose a more generalized
retrieval score, and correspondingly name the re-
trieval heads asreasoning retrieval headswhose
retrieval score exceeds a pre-defined threshold.
Based on the works of Fu et al. (2024); Zhang
et al. (2025) with slight adaptation, we define the
reasoning retrieval score Sreasoning(xt, h) for an
attention head h at timestep t as the proportion
of attention it allocates to all supporting facts (the
“needles”) relative to the total attention it distributes
across the entire effective context (excluding the
interference from attention sinks (Xiao et al., 2023)
and local attention).2 Formally,
Sreasoning(xt, h) =
P
i∈Ineedle ah,t
i
P
j∈I\{I sink∪Ilocal} ah,t
j
(2)
where Ineedle is the set of indices for all support-
ing fact tokens, while Isink and Ilocal represent the
indices of the attention sink and local attention win-
dow, respectively. Intuitively, this score relaxes the
strong copy-paste condition and measures a head’s
focus on the correct information at a given step,
better aligning with how facts are used in the LLM
reasoning process.
4.2 Experiments
We use the HotpotQA dataset (Yang et al., 2018)
as our testbed. HotpotQA is a question-answering
dataset that requires multi-hop reasoning, where
the model must find and integrate multiple discrete
supporting facts from the context to formulate a cor-
rect answer. To examine whether the three claims
in Section 3 are still valid for the reasoning re-
trieval heads on the HotpotQA task, we adopt the
analytical framework from Section 3.
DynamismWe first validate Claim 1. Figure 5
visualizes the retrieval scores of the top-10 variated
retrieval heads during a generation. Consistent with
the NIAH task, the retrieval head pattern is highly
dynamic: no single head dominates throughout the
entire process. Instead, different heads operate as
retrieval heads at distinct generation stages, con-
firming that the dynamic nature of retrieval heads
is a general phenomenon that persists in complex
reasoning tasks.
Thebestthing
to do inSan
Francisco
is toeat a
sandwich
andsit inDoloresParkon a
sunny
day
.
Generation Timestep
L2-H22
L5-H8
L9-H27
L9-H31
L10-H13
L10-H14
L10-H18
L11-H6
L19-H3
L26-H3
Attention Head Index
0.0
0.2
0.4
0.6
0.8
1.0
Figure 5:Dynamic Pattern of Retrieval Heads in a
Multi-Hop Reasoning Task.The heatmap illustrates
the retrieval scores (defined in Eq. 2) for ten active
retrieval heads over the course of the generation process.
2Following common practice, we exclude two types of
attention patterns that are not directly related to long-range re-
trieval: (1)Attention Sinks(Xiao et al., 2023), where certain
tokens (e.g., the initial BOS token) often receive high attention
regardless of content, and (2)Local Attention, where heads
focus on a small, fixed window of recent tokens.
6

<!-- page 7 -->

IrreplaceabilityNext, we validate Claim 2 (Irre-
placeability) through the head ablation experiment
described in Section 3.3. The results in both Fig-
ure 6 and Figure 7 show that masking the dynamic
retrieval heads at each step (identified using Equa-
tion (2)) leads to a far more severe performance
degradation on the HotpotQA task than masking
an equivalent number of top static retrieval heads
or random heads.
250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
F1 Score (%)
Figure 6:Impact of Head Ablation on Multi-Hop
Reasoning Performance.F1-score comparison of Hot-
potQA test scores after ablating three different sets of
attention heads on llama3.1-8b. The opacity of each cell
corresponds to the number of valid samples it contains,
with blank as no valid samples.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6
7|compensated  static_top20|
10
20
30
40
50
60
70
F1 Score (%)
Figure 7:Irreplaceability of Dynamic Heads in a
Multi-Hop Reasoning Context.The plots show the
degradation in HotpotQA performance as an increasing
number (k) of dynamic retrieval heads are ablated on
llama3.1-8b.
CorrelationFinally, we validate Claim 3. Using
the same Temporal Offset CCA, Figure 8 shows
that the strong linear correlation between hidden
states and retrieval scores persists in HotpotQA. In
terms of the MLP experiment, unlike the NIAH
task where retrieval is binary, in HotpotQA, the
retrieval score (defined in Eq. 2) is a continuous
value representing the intensity of attention on sup-
porting facts. Consequently, we trained an MLP
regressor rather than a classifier to predict the pre-
cise retrieval score vector for all heads from the
final hidden state at the synchronous step (k= 0 ).
As shown in Table 3, the probes achieve high R2
scores (up to 0.81) across all models. This indicates
that the information of retrieval scores is effectively
encoded in the hidden state, verifying our claim of
the strong correlation.
0 1 2 3 4 5 6 7 8 9 10
T emporal Distance k
0.3
0.4
0.5
0.6
0.7
0.8Canonical Correlation
0.856
0.835 0.819 0.804 0.795 0.781 0.765 0.751 0.738 0.724 0.712
Correlation Decay with Temporal Distance
Component 1
T op-10 mean
T op-50 mean
Figure 8:Temporal Decay of Correlation on a Multi-
hop Reasoning Task.
Model MSE(↓)MAE(↓)R 2 (↑)
llama3.1-8b 0.0023 0.0177 0.8120
llama3.2-3b 0.0036 0.0247 0.8015
qwen3-8b 0.0050 0.0255 0.7200
llama2-13b 0.0009 0.0121 0.7669
phi4-mini 0.0014 0.0109 0.7333
Table 3:Performance of MLP Probes in Predicting
Reasoning Retrieval Scores on HotpotQA.
5 Case Study: Applying Dynamic
Retrieval Heads to Dynamic RAG
In the previous sections, we verified that the LLM’s
retrieval behavior is better characterized by dy-
namic retrieval heads rather than a fixed set of static
retrieval heads. However, while Sections 3 and 4
focus on comparing the properties of dynamic and
static retrieval heads themselves, an important re-
maining question is how much practical advantage
dynamic retrieval heads provide in real-world ap-
plications. Therefore, in this section, we designed
an case study integrating them into an existing Dy-
namic Retrieval-Augmented Generation (Dynamic
RAG) framework for question answering.
5.1 Task Description
Unlike traditional RAG that retrieves from external
knowledge bases, retrieval heads are specialized
for sourcing information already present within the
model’s input context. We therefore focus on an
in-context retrievaltask, evaluating the model’s
ability to accurately attend to and utilize key infor-
mation provided in its input.
5.2 Method
We adapt DRAGIN (Su et al., 2024b), a prominent
Dynamic RAG framework, for our in-context re-
trieval task. We make the following changes to
integrate the retrieval heads into this framework.
At each generation step, the model’s access to the
7

<!-- page 8 -->

Model Dynamic Static Dynamic Random Fixed Random w/o RAG
EM / F1 EM / F1 EM / F1 EM / F1 EM / F1
llama3.1-8b0.456 / 0.55860.398 / 0.5098 0.272 / 0.3670 0.272 / 0.3763 0.252 / 0.3257
llama3.2-3b 0.384 / 0.49930.428 / 0.53860.224 / 0.3143 0.226 / 0.3051 0.184 / 0.2439
qwen3-8b0.286 / 0.35800.278 / 0.3429 0.210 / 0.2804 0.210 / 0.2804 0.220 / 0.2961
llama2-13b0.284 / 0.38380.278 / 0.3789 0.276 / 0.3762 0.272 / 0.3751 0.192 / 0.2750
phi4-mini0.202 / 0.26900.186 / 0.2505 0.082 / 0.1090 0.086 / 0.1111 0.172 / 0.2331
Table 4: Performance comparison of different retrieval strategies on the HotpotQA dataset (Exact Match (EM) /
F1-Score (F1)). For each model, the better-performing strategy between Dynamic and Static is highlighted inbold.
context is controlled via attention masks. When
no retrieval is needed, the entire context is masked,
while the question and the currently generated text
remain visible. When retrieval is needed, we iden-
tify the active retrieval heads for that step, deter-
mine their top-k most attended-to positions by aver-
aging their attention scores, cluster those positions
and expand a fixed-size window around each clus-
ter. In the subsequent regeneration step, only the
tokens corresponding to these clusters are made
visible to the model via the attention mask. 3 We
use DRAGIN’s RIND algorithm to determine when
to retrieve, and the detailed pipeline can be found
in Algorithm 1 in Appendix C.
5.3 Experiment Setup
DatasetsFollowing Section 4, we employ the
HotpotQA dataset (Yang et al., 2018) for better
practical utility evaluation compared to NIAH.
ModelsWe selected five popular open-source
models of varying sizes and architectures to
ensure the generalizability of our findings:
meta-llama/Llama-3.1-8B-Instruct(llama3.1-8b),
meta-llama/Llama-3.2-3B-Instruct(llama3.2-3b),
Qwen/Qwen3-8B(qwen3-8b),meta-llama/Llama-
2-13b-chat-hf(llama2-13b), andmicrosoft/Phi-4-
mini-instruct(phi4-mini) (Dubey et al., 2024; Yang
et al., 2025; Touvron et al., 2023; Abouelenin et al.,
2025).
BaselinesWe compare five configurations of our
adapted DRAGIN framework to evaluate the effi-
cacy of different head selection strategies:
• Dynamic: Use dynamic retrieval heads identified
by the MLP probe at each step. MLP probe from
Section 4.2 is used to predict the top-5 dynamic
retrieval heads.
• Static: Use 5 pre-identified static retrieval heads.
3We choose to use attention masking over directly rewrit-
ing the context to minimize potential disruptions to the autore-
gressive process, thereby isolating the impact of the retrieved
information.
• Dynamic Random: Use a new set of 5 randomly
selected heads at each retrieval step.
• Fixed Random: Use a fixed set of 5 randomly
selected heads for the entire generation.
• w/o RAG: Perform no retrieval with no context
provided, relying solely on the model’s paramet-
ric knowledge.
5.4 Results
The results are summarized in Table 4. The ob-
servations indicate a superiority of dynamic re-
trieval heads over the other baselines: For the ma-
jority of the tested models (llama3.1-8b, qwen3-8b,
llama2-13b, and phi4-mini), the Dynamic strategy
using dynamic retrieval heads achieves higher or
comparable performance in both EM and F1-score
compared to the Static strategy. The advantage
is particularly pronounced for llama3.1-8b, where
the dynamic strategy’s F1-score (0.5586) is nearly
10% higher than that of the static one (0.5098).
This aligns with our findings in Sections 3 and 4,
suggesting that “expert” heads selected dynami-
cally at each timestep can more precisely locate
the information required for the current reasoning
step than a fixed set of “generalist” heads. An
exception is the llama3.2-3b, with the Static strat-
egy (F1=0.5386) outperforming the Dynamic one
(F1=0.4993). We hypothesize this may be related
to the model size: compared to other models, this
model has the least number of layers with the least
attention heads, indicating a possibility that each
head needs to perform multiple tasks.
6 Conclusion
This paper presents the first systematic study
of retrieval heads from a dynamic perspective.
Through extensive analysis on NIAH and Hot-
potQA tasks, we establish that retrieval head ac-
tivation is dynamic, irreplaceable, and correlated
with the model’s internal state. Furthermore, we
demonstrate the practical utility of these insights
8

<!-- page 9 -->

by integrating dynamic retrieval heads into a Dy-
namic RAG framework, achieving significant per-
formance gains compared to the static retrieval
heads. Our work offers a more granular under-
standing of LLM internal mechanisms and paves
the way for future research in interpretable and
efficient model steering.
Limitations
First, our Dynamic RAG experiment utilizes atten-
tion masking to simulate retrieval for validation
purposes, rather than physically selecting and con-
catenating context as in standard production RAG
pipelines; bridging this gap for practical deploy-
ment remains a direction for future work. Second,
our method in the case study in Section 5 relies
on a learned MLP probe to predict head activa-
tion. While the probe achieves high accuracy, it is
not an oracle; any prediction errors imply that the
identified heads may not perfectly match the true
optimal dynamic retrieval heads, potentially intro-
ducing noise into the retrieval process. Finally, our
analysis primarily focuses on retrieval-intensive
QA tasks (NIAH and HotpotQA); whether these
findings generalize to other long-context domains,
such as summarization or long-context QA, war-
rants further investigation.
References
Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkin-
son, Hany Awadalla, Nguyen Bach, Jianmin Bao,
Alon Benhaim, Martin Cai, Vishrav Chaudhary, Con-
gcong Chen, and 1 others. 2025. Phi-4-mini tech-
nical report: Compact yet powerful multimodal lan-
guage models via mixture-of-loras.arXiv preprint
arXiv:2503.01743.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu,
Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong,
Yue Dong, Junjie Hu, and 1 others. 2024. Pyra-
midkv: Dynamic kv cache compression based on
pyramidal information funneling.arXiv preprint
arXiv:2406.02069.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebas-
tian Gehrmann, and 1 others. 2023. Palm: Scaling
language modeling with pathways.Journal of Ma-
chine Learning Research, 24(240):1–113.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Nelson Elhage, Tristan Hume, Catherine Olsson,
Nicholas Schiefer, Tom Henighan, Shauna Kravec,
Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain,
Carol Chen, and 1 others. 2022. Toy models of su-
perposition.arXiv preprint arXiv:2209.10652.
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom
Henighan, Nicholas Joseph, Ben Mann, Amanda
Askell, Yuntao Bai, Anna Chen, Tom Conerly, and
1 others. 2021. A mathematical framework for
transformer circuits.Transformer Circuits Thread,
1(1):12.
Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue
Dong, and Wen Xiao. 2024. Not all heads matter:
A head-level kv cache compression method with
integrated retrieval and reasoning.arXiv preprint
arXiv:2410.19258.
Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gre-
gory Valiant. 2022. What can transformers learn
in-context? a case study of simple function classes.
Advances in neural information processing systems,
35:30583–30598.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Men-
sch, Elena Buchatskaya, Trevor Cai, Eliza Ruther-
ford, Diego de Las Casas, Lisa Anne Hendricks,
Johannes Welbl, Aidan Clark, and 1 others. 2022.
Training compute-optimal large language models.
arXiv preprint arXiv:2203.15556.
Greg Kamradt. 2023. LLMTest_NeedleInAHaystack:
A test to measure llm performance over long con-
texts. https://github.com/gkamradt/LLMTest_
NeedleInAHaystack.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. Snapkv:
Llm knows what you are looking for before gener-
ation.Advances in Neural Information Processing
Systems, 37:22947–22970.
9

<!-- page 10 -->

Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Paul Michel, Omer Levy, and Graham Neubig. 2019.
Are sixteen heads really better than one?Advances
in neural information processing systems, 32.
Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel
Goh, Michael Petrov, and Shan Carter. 2020. Zoom
in: An introduction to circuits.Distill, 5(3):e00024–
001.
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas
Joseph, Nova DasSarma, Tom Henighan, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, and 1 oth-
ers. 2022. In-context learning and induction heads.
arXiv preprint arXiv:2209.11895.
Ofir Press, Noah A Smith, and Mike Lewis. 2021.
Train short, test long: Attention with linear biases
enables input length extrapolation.arXiv preprint
arXiv:2108.12409.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, and 1 others. 2019.
Language models are unsupervised multitask learn-
ers.OpenAI blog, 1(8):9.
Tal Ridnik, Emanuel Ben-Baruch, Nadav Zamir, Asaf
Noy, Itamar Friedman, Matan Protter, and Lihi
Zelnik-Manor. 2021. Asymmetric loss for multi-
label classification. InProceedings of the IEEE/CVF
international conference on computer vision, pages
82–91.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. 2024a. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomputing, 568:127063.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024b. Dragin: dynamic retrieval
augmented generation based on the information
needs of large language models.arXiv preprint
arXiv:2403.10081.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing
systems, 30.
Elena V oita, David Talbot, Fedor Moiseev, Rico Sen-
nrich, and Ivan Titov. 2019. Analyzing multi-
head self-attention: Specialized heads do the heavy
lifting, the rest can be pruned.arXiv preprint
arXiv:1905.09418.
Kevin Wang, Alexandre Variengien, Arthur Conmy,
Buck Shlegeris, and Jacob Steinhardt. 2022. In-
terpretability in the wild: a circuit for indirect ob-
ject identification in gpt-2 small.arXiv preprint
arXiv:2211.00593.
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2024. Retrieval head mechanisti-
cally explains long-context factuality.arXiv preprint
arXiv:2404.15574.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2023. Efficient streaming
language models with attention sinks.arXiv preprint
arXiv:2309.17453.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and
Tengyu Ma. 2021. An explanation of in-context learn-
ing as implicit bayesian inference.arXiv preprint
arXiv:2111.02080.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600.
Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen,
and Xi Ye. 2025. Query-focused retrieval heads im-
prove long-context reasoning and re-ranking.arXiv
preprint arXiv:2506.09944.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, and 1 oth-
ers. 2023. H2o: Heavy-hitter oracle for efficient
generative inference of large language models.Ad-
vances in Neural Information Processing Systems,
36:34661–34710.
A Experimental Setup for
Needle-in-a-Haystack Masking
For reproducibility, all experiments in this paper
use greedy sampling as the LLM decoding strat-
egy. Each experiment can be perform on a single
NVIDIA H200 GPU.
10

<!-- page 11 -->

A.1 Needle-in-a-Haystack Test Setting
The original Needle-in-a-Haystack (NIAH)
test (Kamradt, 2023) was proposed in 2023, while
most of the models we study in this work were
released after this. This raises the possibility that
these models were exposed to the NIAH data
during their training phase. To prevent potential
data leakage, we usedynamically, randomly
generated UUID stringsas the needle string as a
substitute, to exclude the possibility that the model
has seen the Needle data during its training phase.
Its specific format is as follows:
• Needle: The magic word is [UUID].
• Question: What is the magic word?
For the experiment conducted in Section 3.3, the
detailed experiment setting is as follows:
For evaluation criteria, we use two types: ac-
curacy and ROUGE-L (Lin, 2004). Accuracy is
to check whether the model’s response completely
contains the UUID string. If it does, the score
is 1.0; otherwise, it is 0.0, even if there is only a
one-character difference.
We extend the context length to the specified
length by repeatedly concatenating and truncating
the haystack text. Then, we randomly select a
depth, backtrack to the nearest end of a sentence
to insert the needle, and apply a dialogue template
to construct the input. The dialogue template is as
follows:
NIAH Test Prompt
System
You are a helpful AI bot that answers ques-
tions for a user. Keep your response short
and direct.
User
Context:
{A Haystack with a Needle inserted in}
Question:
{Question}
Instruction:
Don’t give information outside the docu-
ment or repeat your findings.
To obtain the data for Figure 2, we conducted
5 independent runs for each grid cell and reported
the average metric value.
To collect the data for Figure 3, we conducted
20 independent runs at intervals of k= 5 , with the
haystack length fixed at 5000 tokens. The reported
results are the averages of these trials.
A.2 Detailed Setting for the Experiment in
Section 3.3.2
In the ablation study involving the masking of k
dynamic heads, our goal is to quantify the model’s
attempt to compensate using static retrieval heads.
To do this, we track the compensated heads and
measure their overlap with the Top-20 static re-
trieval heads.
A key methodological challenge is how to ag-
gregate this metric over a full generation sequence.
We observed that under heavy ablation (i.e., large
k), the model’s retrieval mechanism often col-
lapses in later generation timesteps, resulting in
zero retrieval heads. Consequently, a simple aver-
age across all timesteps would include these zero-
values, artificially deflating the metric and failing
to reflect the model’s actual capacity to utilize
compensated heads. To address this and robustly
capture the model’s peak compensatory effort, we
record the maximum number of compensated heads
observed at any single timestep within each sam-
ple’s generation.
Formally, for each sample s, let Hs,t be the
set of dynamic retrieval heads at timestep t be-
fore masking, let H ′
s,t be the set of dynamic re-
trieval heads at timestep t after masking. Let
Es,t =H ′
s,t −H s,t be the set of compensated heads
at timestep t. We compute the maximum intersec-
tion with the top-20 static set, Hstatic, within the
sample s: ms = max t |Es,t ∩H static|. The final
metric is the average ofm s over all samples.
A.3 Details for Entropy Metric in Dynamism
Analysis
We calculate the entropy of the retrieval score dis-
tribution to measure how broadly retrieval respon-
sibility is shared. Let ph be the probability that
head h is activated as a retrieval head across all
timesteps. The entropy is defined as:
S=−
X
h
ph lnp h (3)
To provide a baseline for interpretation, consider
a scenario where retrieval is exclusively and uni-
formly performed by the top-20 static heads. In this
case, ph = 1
20 or these 20 heads and 0 for others.
The resulting entropy would be:
11

<!-- page 12 -->

Sbaseline =−
20X
i=1
1
20 ln
 1
20

= ln(20)≈2.9957
(4)
Our observed entropy values are consistently
higher than this baseline (e.g., 3.8154 for llama3.1-
8b). Combining with the low Jaccard w/ Static
values, this indicates that the effective number
of heads participating in retrieval is significantly
larger than 20.
A.4 Experimental Settings for Canonical
Correlation Analysis
To analyze the correlation between the model’s fi-
nal hidden states and retrieval scores, we employed
Canonical Correlation Analysis (CCA). The de-
tailed experimental procedure is as follows:
Data PreprocessingWe first standardized the re-
trieval scores (min 0, max 1) to ensure consistent
scaling across attention heads. To improve compu-
tational efficiency and focus on the principal signal
subspaces, we applied Principal Component Anal-
ysis (PCA) to both the hidden states and retrieval
scores prior to CCA. We retained principal compo-
nents explaining 95% of the variance for the hidden
states and 99% for the retrieval scores.
CCA ConfigurationWe set the number of
canonical components to 50.
Temporal OffsetWe analyzed the correlation
with a temporal offset k ranging from 0 to 10. For
each offset k, we paired the hidden state at timestep
n with the retrieval scores at timestep n+k . Any
samples where n+k exceeded the sequence length
were excluded from the analysis.
A.5 Experimental Settings for MLP Probe
Training
To investigate the fine-grained correlation of re-
trieval head patterns, we trained a Multi-Layer Per-
ceptron (MLP) probe for each LLM. The detailed
configuration is as follows:
Model ArchitectureThe probe is a feed-forward
neural network consisting of three hidden layers
with dimensions [8192,4096,4096]. We applied
a dropout rate of 0.1 after each hidden layer to
prevent overfitting. The input dimension matches
the hidden size of the respective LLM, and the
output dimension corresponds to the total number
of attention heads.
Training ConfigurationThe models were
trained for 100 epochs with a batch size of 128.
We used the Adam optimizer with a learning rate
of 3×10 −4 and a scheduler that reduces the learn-
ing rate upon a plateau in validation loss (patience
set to 3 epochs). To handle the sparse retrieval
score distribution of attention heads, we employed
the Asymmetric Loss (Ridnik et al., 2021) as the
objective function. Gradients were clipped at a
maximum norm of 1.0 to ensure stability.
Data SplitThe dataset collected from the NIAH
runs was split into training (70%), validation (20%),
and testing (10%) sets. All reported metrics (Pre-
cision, Recall, F1, AUPRC) are evaluated on the
held-out test set.
B Additional Experiment Results
B.1 All Heads Ablation Results
See Figure 9, Figure 10, Figure 11, Figure 12, Fig-
ure 13, Figure 14.
B.2 Different Numbers of Heads Ablation
Results
See Figure 15, Figure 16, Figure 17, Figure 18,
Figure 19, Figure 20.
C Algorithms for the Dynamic RAG
Method in Section 5
See Algorithm 1, Algorithm 2.
12

<!-- page 13 -->

5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
ROUGE-L Score (%)
Figure 9: Head Ablation on NIAH test on llama3.1-8b. Using ROUGE-L as the metric.
Algorithm 1Dynamic RAG with In-Context Retrieval
Require:ContextC, QuestionQ, ModelM
1:Generated TextG ←‘’
2:Visible MaskV ←MaskAll(C)▷Initially mask context
3:whilenot finisheddo
4:InputI ←Concat(C,Q,G)
5:DraftD, AttentionsA ← M.GenerateDraft(I,mask=V)
6:is_hallucination, pos←RIND(D)
7:ifis_hallucinationthen
8:G ←Retract(G,to sentence of pos)
9:V ←Retrieve(C,Q,G)▷See Alg. 2
10:InputI ←Concat(C,Q,G)
11:Rewritten← M.Generate(I,mask=V)
12:G ←Append(G,Rewritten)
13:else
14:G ←Append(G,D)
15:V ←MaskAll(C)▷Re-mask for next draft
16:end if
17:end while
18:returnG
Algorithm 2In-Context Retrieval via Attention Heads
1:functionRETRIEVE(C,Q,G)
2:UnMaskAll(C)
3:Active HeadsH dyn ←IdentifyHeads(C,Q,G)
4:MaskAll(C)
5:Avg Scoress←AverageAttention(H dyn)
6:Top-k IndicesK ←TopKIndices(s, k)
7:Index ClustersC idx ←ClusterIndices(K)
8:Expanded WindowsW ← ∅
9:foreach clusterc∈ C idx do
10:Representative Indexi rep ←GetRepresentative(c)
11:W ← W ∪ExpandWindow(i rep,size)
12:end for
13:Final WindowsW f inal ←MergeOverlapping(W)
14:Visible MaskV ←CreateMaskFromWindows(W f inal)
15:returnV
16:end function
13

<!-- page 14 -->

5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
Accuracy (%)
(a) Accuracy
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
ROUGE-L Score (%)
(b) ROUGE-L
Figure 10: Head Ablation on NIAH test on llama3.2-3b.
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
Accuracy (%)
(a) Accuracy
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
0%
10%
20%
30%
40%
50%
60%
70%
80%
90%
100% Depth
Dynamic
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Static
5k 10k 15k 20k 25k 30k 35k 40k 45k 50k
Context Length
Random
0
20
40
60
80
100
ROUGE-L Score (%)
(b) ROUGE-L
Figure 11: Head Ablation on NIAH test on qwen3-8b.
14

<!-- page 15 -->

250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
Exact Match (%)
Figure 12: Head Ablation on HotpotQA test on llama3.1-8b. Using EM as the metric.
250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
Exact Match (%)
(a) EM
250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
F1 Score (%)
(b) F1
Figure 13: Head Ablation on HotpotQA test on llama3.2-3b.
15

<!-- page 16 -->

250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
Exact Match (%)
(a) EM
250 750 1250 1750 2250 2750 3250 3750
Context Length
5%
15%
25%
35%
45%
55%
65%
75%
85%
95% Depth
Dynamic
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Static
250 750 1250 1750 2250 2750 3250 3750
Context Length
0
2
4
6
8
Random
0
20
40
60
80
100
F1 Score (%)
(b) F1
Figure 14: Head Ablation on HotpotQA test on qwen3-8b.
16

<!-- page 17 -->

0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6
7|compensated  static_top20|
10
20
30
40
50
60
70
ROUGE-L Score (%)
Figure 15: Different Numbers of Head Ablation on
NIAH test on llama3.1-8b. Using ROUGE-L as the
metric.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4|compensated  static_top20|
0.0
0.2
0.4
0.6
0.8
Accuracy (%)
(a) Accuracy
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4|compensated  static_top20|
0
10
20
30
40
50
60
ROUGE-L Score (%)
(b) ROUGE-L
Figure 16: Different Numbers of Head Ablation on
NIAH test on llama3.2-3b.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6|compensated  static_top20|
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy (%)
(a) Accuracy
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6|compensated  static_top20|
20
30
40
50
60
70
ROUGE-L Score (%)
(b) ROUGE-L
Figure 17: Different Numbers of Head Ablation on
NIAH test on qwen3-8b.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3
4
5
6
7|compensated  static_top20|
10
20
30
40
50
Exact Match (EM) (%)
Figure 18: Different Numbers of Head Ablation on
NIAH test on llama3.1-8b. Using EM as the metric.
17

<!-- page 18 -->

0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3|compensated  static_top20|
20
30
40
50
60
Exact Match (EM) (%)
(a) EM
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0
1
2
3|compensated  static_top20|
30
40
50
60
70
F1 Score (%)
(b) F1
Figure 19: Different Numbers of Head Ablation on
HotpotQA test on llama3.2-3b.
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5|compensated  static_top20|
35
40
45
50
55
60
Exact Match (EM) (%)
(a) EM
0 10 20 30 40 50 60 70
k (number of masked dynamic retrieval heads)
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5|compensated  static_top20|
50
55
60
65
70
75
F1 Score (%)
(b) F1
Figure 20: Different Numbers of Head Ablation on
HotpotQA test on qwen3-8b.
18
