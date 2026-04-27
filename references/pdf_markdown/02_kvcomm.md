# references/02_kvcomm.pdf

<!-- page 1 -->

Published as a conference paper at ICLR 2026
KVCOMM: ENABLINGEFFICIENTLLM COMMUNICA-
TION THROUGHSELECTIVEKV SHARING
Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., Dejan Kosti´c
KTH Royal Institute of Technology
{xiangyus,mchiesa,maguire,dmk}@kth.se
ABSTRACT
Large Language Models (LLMs) are increasingly deployed in multi-agent sys-
tems, where effective inter-model communication is crucial. Existing communi-
cation protocols either rely on natural language, incurring high inference costs
and information loss, or on hidden states, which suffer from information concen-
tration bias and inefficiency. To address these limitations, we propose KVComm,
a novel communication framework that enables efficient communication between
LLMs through selective sharing of KV pairs. KVComm leverages the rich infor-
mation encoded in the KV pairs while avoiding the pitfalls of hidden states. We
introduce a KV layer-wise selection strategy based on attention importance scores
with a Gaussian prior to identify the most informative KV pairs for communica-
tion. Extensive experiments across diverse tasks and model pairs demonstrate that
KVComm achieves comparable performance to the upper-bound method, which
directly merges inputs to one model without any communication, while transmit-
ting as few as 30% of layers’ KV pairs. Our study highlights the potential of KV
pairs as an effective medium for inter-LLM communication, paving the way for
scalable and efficient multi-agent systems1.
1 INTRODUCTION
Large Language Models (LLMs) have catalyzed a paradigm shift from isolated model capabilities
towards collaborative multi-agent systems (Guo et al., 2024; Tran et al., 2025). CAMEL (Li et al.,
2023), AutoGen (Wu et al., 2024), and ChatDev (Qian et al., 2023) have demonstrated the potential
of LLMs to collaborate effectively in multi-agent systems, achieving impressive results in various
tasks. These systems leverage the strengths of individual LLMs and enable them to work together
to solve complex problems that are beyond the capabilities of a single model (Yang et al., 2024a).
While multi-agent systems have shown great promise, they also introduce new challenges, particu-
larly in the area of inter-agent communication. Effective communication between LLMs is crucial
for the success of multi-agent systems. Explicit communication through natural language has been
explored in several works, enabling the models to share information (Du et al., 2023), coordinate
their actions (Sun et al., 2025), and make collective decisions (Yang et al., 2024b).
However, natural language communication leads to high inference costs due to the need for multi-
ple decoding steps, and may not fully capture the rich information that needs to be shared between
Large Language Models, as information is lost in the sampling process (Pham et al., 2023; Ramesh
& Li, 2025) that occurs as each new token is produced. To address this limitation, recent works have
explored alternative communication protocols that leverage the internal representations of LLMs.
CIPHER (Pham et al., 2023) proposed to use the embedding space as the medium of communica-
tion between LLMs. Namely, they pass the weighted average of the token embeddings from one
LLM to another, facilitating more efficient information exchange. Rather than using the embedding
space, AC (Ramesh & Li, 2025) transmits the intermediate activations, specifically the last token’s
hidden state. They replace the last token’s hidden state of thereceiver’s model(M r) with that of
thesender’s model(M s), allowing a more direct transfer of information. While these methods have
shown promising results, they still face challenges in terms of communication efficiency and effec-
tiveness. CIPHER (Pham et al., 2023) still requires multiple decoding steps, which can be costly,
1Our code and data can be found at https://github.com/Zephyroam/KVComm
1
arXiv:2510.03346v3  [cs.LG]  22 Feb 2026

<!-- page 2 -->

Published as a conference paper at ICLR 2026
Layer
Selected
KV
Output
Model Input/Output
Prefill activation of
KV pairs of
Prefill activation of Attention within        /
Attention across models
KV pairs with score
Selected KV pairs
Decoding activation
CalibrationInference
Extract
attention Compute
score
Figure 1: KVComm framework for efficient LLM communication through selective KV sharing.
and AC (Ramesh & Li, 2025) may lead to information loss as only limited activation information is
transmitted.
We start with the question:What is the most effective way to communicate between LLMs?We
argue that an ideal communication protocol should satisfy the following criteria:①Effectiveness: It
should enableM r to effectively utilize the information fromMs.②Efficiency: It should minimize
the computation needed byM s and the amount of data transmitted between models.③Generality:
It should be applicable to a wide range of tasks and model architectures, ensuring its versatility in
different scenarios. We choose to use activation information as the medium of communication, as
no decoding steps are needed forM s, andM r can directly utilize the rich information encoded in
the activations. We study different types of activation information (i.e., hidden states and KV pairs),
and in Section 2.2, we show that hidden states suffer from information concentration bias, where
the last token’s hidden state contains most of the information needed for the model’s output. This
makes it challenging to design an effective communication protocol using the last token’s hidden
state. Furthermore, we find that using all tokens’ hidden states from a single layer of the sender
Ms does not guarantee effective communication. A dilemma arises: if the hidden states are taken
from the early layers ofM s, the computation benefit is limited since the computation cost is similar
to concatenating the two inputs; if the hidden states are prepended to the later layers ofM r, the
performance drops significantly.
Based on these observations, we proposeKVComm, a novel communication protocol that enables
efficient communication between LLMs through selective sharing of KV pairs. KV pairs are the
most representative activation information in each layer, and sharing them does not interact with
the hidden states ofM r directly, whileM r can decide how to utilize the information through the
attention mechanism. To further improve the efficiency of communication, we propose a selection
strategy to choose which (potentially non-contiguous) layers’ KV pairs to share. We formulate
hypotheses that (H1)KV pairs from intermediate layers encode transferable semantic knowledge,
and (H2)KV pairs from layers exhibiting stronger attention distributions are more effective for
communication. These hypotheses are validated by our experiments in Sections 4.3 and 4.5. Based
on these hypotheses, we define attention importance scores for each layer based on the average
attention weights assigned to the context tokens. We also apply a Gaussian distribution centered
at a certain layer as a prior on the attention importance scores. The intuition is that the Gaussian
distribution encourages selecting layers around a certain depth, which aligns with hypothesisH1.
The general framework is illustrated in Figure 1.
We evaluate KVComm on a diverse set of tasks with nine model pairs (see Section 4.1), showing
that it consistently outperforms existing communication protocols while significantly reducing the
data transmitted between models. In summary, our work makes three key contributions:
• We evaluate different types of activation information for communication between LLMs, and
identify the limitations of using hidden states as the medium of communication. We show
that the last token’s hidden state suffers from information concentration bias, and point out a
dilemma that arises when using all tokens’ hidden states.
• We propose KVComm, a novel communication protocol that enables efficient communication
between LLMs through selective sharing of KV pairs. We design a selection strategy based on
attention importance scores and a Gaussian prior to choose which layers’ KV pairs to share.
2

<!-- page 3 -->

Published as a conference paper at ICLR 2026
This is the first approach that makes it possible to choose non-contiguous layers of KV . More-
over, we show the feasibility of using a single context/question pair for guiding the selection
for a given pair of models, prior to deployment.
• We conduct extensive experiments on a diverse set of tasks and model pairs, demonstrating
that KVComm enables effective and efficient communication between LLMs, achieving com-
parable performance to the Skyline method, which is the upper-bound and directly merges the
inputs without any communication, while reducing the computation costs by 2.5x to 6x. In
particular, KVComm enables up to a 3x reduction in communication relative to approaches
that transmit the entire set of KV pairs. Moreover, we demonstrate the performance benefits of
non-contiguous selection of KV layers. Finally, we demonstrate the increase in performance
that KVComm brings even over Skyline on two datasets, further illustrating the need to com-
municate in a non-strictly textual manner.
2 PROBLEM ANDMOTIVATION
2.1 PROBLEMFORMULATION
We formally define the problem of solving a contextual task through the communication of two
LLMs:M s andM r.M s takes as input a contextC, and generates the required informationI C to
be communicated.M r takes as input the queryQand the informationI C fromM s, and produces
the final output. In this work, we limit the choices of the two LLMs to (1) two instances of the same
LLM, and (2) two models that are fine-tuned versions of the same base LLM. The objective is to
design a communication protocol that jointly optimizes the communication, computation efficiency,
and the information fidelity betweenM s andM r.
2.2 WHYHIDDENSTATESFALLSHORT
When Decoder-Only LLMs infer, the input information flows through the model in the form of
activation values, which refer to the intermediate results output by each decoder layer during the
forward pass. We refer to the intermediate activation values that are passed between adjacent layers
as hidden states. We also consider the KV pairs used in the attention mechanism within each layer
as another type of activation information. In this section, we investigate the effectiveness of using
hidden states as the medium of communication by studying two questions:How important are
hidden states of tokens at different positions in the sequence?(Section 2.2.1)Are hidden states of
all tokens effective for communication?(Section 2.2.2)
2.2.1 TOKENIMPORTANCE ATDIFFERENTPOSITIONS
We begin with a simple experiment examining how token positions affect performance. Using
Llama-3.1-8B on MMLU Social Science, we remove or retain the hidden state ofonlyspecific
tokens at a given layer and measure the performance change. As shown in Figure 2, different to-
kens vary in importance across layers, with the last token becoming most critical in later layers.
This aligns with the intuition that the last token is often the most relevant to the current prediction.
Thus, the last token’s hidden state carries the most influential information for both model output and
inter-LLM communication. Results on additional datasets and models are provided in Section C.
To ensure efficient communication with hidden states built on this observation, two conditions must
hold: (1)M s must send at least the last token’s hidden state, and (2) the communication protocol
should preserveM r’s last token state as much as possible. The protocol in Ramesh & Li (2025)
either replacesM r’s last token state with that ofMs or averages the two, but both cause information
loss inM r’s last token state, harming its performance.
2.2.2 UTILIZINGALLTOKENS
Another straightforward approach to ensure the last token’s hidden state is preserved is to prepend
all tokens’ hidden states fromM s toM r. The experiments on HotpotQA with Llama-3.1-8B,
presented in Figure 3, demonstrate that prepending all tokens’ hidden states fromM s toM r is
effective if the hidden states are taken from the early layers ofMs and prepended to the early layers
3

<!-- page 4 -->

Published as a conference paper at ICLR 2026
0 5 10 15 20 25
0.2
0.3
0.4
0.5
0.6
Retain the token
Last
None
Random
First
0 5 10 15 20 25
0.2
0.3
0.4
0.5
0.6
Remove the token
Layer
Accuracy
Figure 2: Compared to other token positions, the last token’s
hidden state is the most critical, especially in later layers.
0 10 20 30
0
5
10
15
20
25
30
F1 Score of Prepending Hidden States
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
F1 Score
Receiver r layer index (j)
Sender s layer index (k)
Figure 3: Prepending helps only
when using early-layer hidden
states.
ofM r. Section D shows experimental results on other datasets. We find that this method is caught
in a dilemma: (1) if the hidden states are taken from the early layers ofMs, the computation benefit
is limited since it is similar to concatenating the two inputs; (2) if the hidden states are prepended to
the later layers ofM r, the performance drops significantly.
These findings suggest that while utilizing all tokens’ hidden states can preserve the last token’s
information, it does not guarantee effective communication between LLMs.
3 EFFICIENTLLM COMMUNICATION THROUGHSELECTIVEKV SHARING
We propose a simple yet effective communication protocol that enables efficient communication
between LLMs by selectively sharing KV pairs. This approach addresses the limitations observed in
previous methods by ensuring that the most critical information is preserved. Our design satisfies the
three criteria outlined below: it enhances effectiveness by allowingM r to utilize essential context
(①), improves efficiency by reducing unnecessary computation and transmission overhead (②), and
ensures generality by being applicable across diverse tasks and architectures (③).
3.1 COMMUNICATIONFRAMEWORK
For a given contextCand queryQ,M s processes the contextCand runs one forward pass (prefill
stage) to generate the KV pairs{(k l
s,v l
s)}at each layerl, wherel= 1,2, . . . , LandLis the total
number of layers inM s. We apply a selection strategy to choose a subset of KV pairs{(k li
s ,v li
s )},
wherei= 1,2, . . . , MandMis the number of selected layers. The selected KV pairs are then
transmitted toM r.
Mr processes the queryQand incorporates the received KV pairs during its forward passes (prefill
and decoding stages). Specifically, at each layerlofM r, iflcorresponds to a selected layerl i2,
the KV pairs fromM s are integrated into the attention mechanism. We simply concatenate the KV
pairs fromM s with those ofM r:k l
r ←[k li
s ;k l
r], andv l
r ←[v li
s ;v l
r]. This integration allowsM r
to attend to both its own context and the information provided byM s. After processing the query
Qwith the integrated KV pairs,M r generates the final output.
3.2 KV SELECTIONSTRATEGIES
The communication protocol critically depends on the selection strategy for choosing which KV
pairs to transmit fromM s toM r. Not all layers or attention heads contribute equally to encoding
task-relevant knowledge. A fundamental question when designing selection strategies is:Which
parts of the KV pairs encode the most relevant knowledge for communication?
Formally, given the set of candidate KV pairs{(k l
s,v l
s)}L
l=1, our goal is to select a subsetS ⊆
{1, . . . , L}such that the receiver’s output retains maximal information from the sender, given a
2The layer indices are 1-to-1 matched betweenM s andM r since we only consider the case where the two
models are the same or fine-tuned versions of the same base LLM.
4

<!-- page 5 -->

Published as a conference paper at ICLR 2026
constraint on the number of selected layers|S|=M, which is determined by the desired communi-
cation efficiency. This can be formulated as the following optimization problem:
max
S⊆{1,...,L},|S|=M
f(M r(Q,{(k l
s,v l
s)}l∈S )),
wheref(·)is a performance metric (e.g., accuracy, F1 score), andM r(Q,{(k l
s,v l
s)}l∈S )denotes
the output of the receiver model given the queryQand the selected KV pairs. Since direct compu-
tation of this objective is intractable, we instead propose two hypothesesH1andH2that serve as
priors for designing practical heuristics.
The first hypothesisH1is thatKV pairs from intermediate layers contain the most readily trans-
ferable semantic knowledge. Prior analyses (Jawahar et al., 2019; Geva et al., 2020) suggest a hi-
erarchy: early layers capture surface patterns, middle layers encode semantic abstractions, and late
layers specialize in task predictions. Thus, intermediate KV pairs should carry the richest generaliz-
able information, making them most effective for communication. Experiment results in Section 4.3
support this hypothesis.
Another hypothesisH2is thatKV pairs from layers exhibiting stronger attention distributions are
more effective for communication. We quantify this notion ofattention distributionusing the at-
tention importance scoreS l
a, defined in Equation (1) below. We deem layerl i to exhibit stronger
attention distribution thanl j, ifS li
a > S lj
a . Intuitively, if a head consistently allocates high attention
mass to the given tokens, its KV cache encodes salient contextual relations that are critical for the
model’s reasoning. Attention concentration thus serves as a proxy for the communication value of
a KV subset, suggesting that such heads should be prioritized for selection. This hypothesis is also
validated by our experiments in Section 4.5.
Our selection strategy is based on these two hypotheses. We first define attention importance scores
for each layer, which are calculated as the average attention weights that have been assigned to the
context tokens by all heads in that layer during the prefill stage. We then take a Gaussian distribution
centered at a certain layer as a prior to select layers with high attention importance scores. The
intuition is that the Gaussian prior encourages selecting layers around a certain depth, which aligns
with hypothesisH1that intermediate layers are more likely to contain transferable knowledge.
Mathematically, the attention importance score for each layerlis computed as:
ˆSl
a = 1
H|Q|
HX
h=1
|Q|X
q=1
|C|X
c=1
al
h,q,c,(1)
whereHis the number of attention heads,|Q|is the number of tokens in the query,|C|is the number
of context tokens, anda l
h,q,c is the attention weight assigned by headhat layerlfrom tokenqto
context tokenc. ˆSl
a is then normalized to the range[0,1]across all layers to obtain the final attention
importance scoreS l
a =
ˆSl
a−minl′ ˆSl′
a
maxl′ ˆSl′
a −minl′ ˆSl′
a
.
We define a Gaussian prior centered at layerµwith standard deviationσasP l = exp

− (l−µ)2
2σ2

.
The final selection score for each layerlis computed as a weighted combination of the attention
importance score and the Gaussian prior:
Sl =αS l
a + (1−α)P l,
whereα∈[0,1]is a hyperparameter that balances the two components. We then select the topM
layers with the highest selection scoresS l to form the subsetSfor communication.
For each model pair and dataset, the topMlayers are selected based on the selection scores com-
puted from a calibration set. The selected layers are then fixed and used for all samples in the test set.
We found that a calibration set as small as a single sample is sufficient to obtain a robust selection
that generalizes well to the entire test set, as shown in the experiments in Section H.
3.3 COMPLEXITYANALYSIS
We analyze the computational complexity of our KVComm framework compared to baseline meth-
ods. Compared to the NLD (Du et al., 2023) method, our method does not require multiple decoding
5

<!-- page 6 -->

Published as a conference paper at ICLR 2026
steps forM s, which significantly reduces the computation cost. When the number of tokens gen-
erated during debate (Du et al., 2023) is large, the computation margin of our method over NLD is
on the order ofO(L(T s +T r +|Q|) 2d),whereT s andT r are the number of tokens generated by
Ms andM r in the debate, respectively, and|Q|anddare the number of tokens in the query and the
hidden dimension of the model, respectively. Compared to the Skyline (Section 4.1) method, our
method also reduces the computation cost, especially whenMis small. The computation margin of
our method over Skyline is on the order ofO(|C|d(L(2|Q|+T r)−M(|Q|+T r))), where|C|is
the number of tokens in the context.
4 EXPERIMENTS
4.1 EXPERIMENTALSETUP
DatasetsWe evaluate KVComm on a diverse set of contextual reasoning tasks. Following Ramesh
& Li (2025), we synthetically generate two datasets, Countries, which asks questions about coun-
tries based on landmark information, and Tipsheets, which requires investment decisions from fi-
nancial tips. Examples of these two datasets are shown in Table 3 in Section B.1. Moreover, we
select six benchmarks, including HotpotQA (Yang et al., 2018), QASPER (Dasigi et al., 2021),
MuSiQuest (Trivedi et al., 2022), two subsets of LongBench (Bai et al., 2024)(MultiFieldQA-en
and 2WikiMQA), and TMATH (Qi et al., 2025). The last dataset is a mathematical problem-solving
dataset that contains hints as context. We use ROUGE-L Recall as the evaluation metric for the last
dataset, and F1 score for all other datasets. Statistics are summarized in Table 4 in Section B.1.
ModelsWe conduct experiments on nine different model pairs, shown in Table 5 in Section B.3.
The model pairs include two instances of the same LLM and two models that are fine-tuned versions
of the same base LLM. These models cover different families, including LLaMA (Dubey et al.,
2024), Qwen (Qwen et al., 2024), and Falcon (Almazrouei et al., 2023).
Compared MethodsWe compare KVComm with several representative approaches:Baseline
(no communication betweenM r andM s),Skyline(concatenating contextCand queryQas an
upper bound),Natural Language Debate (NLD)(Du et al., 2023),CIPHER(Pham et al., 2023),
andAC(Ramesh & Li, 2025). Detailed descriptions for these methods are provided in Section B.4.
Implementation details are provided in Section B.2.
4.2 COMMUNICATIONRESULTS
Table 1 reports results on three model pairs fine-tuned from the same base LLM. The results on
other model pairs are provided in Table 8 in Section F, which show similar trends. We observe that
KVComm consistently outperforms all baseline communication methods across datasets and model
pairs. AC can outperform the Baseline method on some datasets, but they are still significantly
worse than KVComm and Skyline, as hidden states ofM r are corrupted during communication.
NLD and CIPHER can achieve performance close to that of KVComm or Skyline on Countries
and Tipsheets datasets, which is because these datasets require only a very small and highly salient
amount of information to be transferred. For all other datasets, the sender has access to the entire
context but not the question, and natural-language communication cannot reliably extract and trans-
mit the task-relevant subset of information. As a result, NLD and CIPHER perform substantially
below KVComm on complex, long-context reasoning tasks. We conduct further experiments in
Section I to eliminate the influence of hyperparameters.
KVComm can achieve comparable performance to Skyline when selecting 70% of layers’ KV pairs
for communication, demonstrating the effectiveness of our selection strategy. Even when selecting
only 30% of layers’ KV pairs, KVComm can still outperform most baseline communication methods
on many datasets, showing its potential for efficient communication with minimal overhead.
Note that KVComm can outperform Skyline on some datasets. We attribute this to two factors: (1)
Ms may complementM r with stronger capabilities in certain aspects, and (2) selective KV sharing
provides a regularization effect, which helpsM r to focus on the most relevant information and
6

<!-- page 7 -->

Published as a conference paper at ICLR 2026
Table 1: Communication results of different methods. Best results arebolded, second best
underlined (excluding Baseline and Skyline). We report the results withM r for Baseline and Sky-
line for fairness.KVComm (0.3/0.5/0.7)denotes selecting 30%/50%/70% of layers’ KV pairs for
communication, i.e.,M=⌈0.3L⌉,M=⌈0.5L⌉,M=⌈0.7L⌉.
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Ms: huihui-ai/Llama-3.2-3B-Instruct-abliterated;M r: suayptalha/DeepSeek-R1-Distill-Llama-3B
Baseline0.05 0.32 0.23 0.05 0.02 0.11 0.27 0.34
Skyline0.57 0.91 0.73 0.25 0.51 0.47 0.40 0.36
NLD0.430.72 0.43 0.10 0.18 0.09 0.30 0.33
CIPHER0.42 0.69 0.50 0.10 0.18 0.13 0.32 0.32
AC (mean)0.03 0.45 0.25 0.05 0.02 0.13 0.230.35
AC (replace)0.00 0.49 0.05 0.01 0.01 0.12 0.030.34
AC (sum)0.02 0.46 0.23 0.05 0.01 0.13 0.240.34
KVComm (0.3)0.46 0.45 0.46 0.09 0.28 0.15 0.280.35
KVComm (0.5) 0.57 0.810.57 0.27 0.32 0.510.36 0.35
KVComm (0.7) 0.57 0.81 0.65 0.29 0.360.47 0.37 0.35
Ms: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored;M r: bespokelabs/Bespoke-Stratos-7B
Baseline0.01 0.36 0.13 0.05 0.03 0.08 0.09 0.35
Skyline0.51 0.97 0.53 0.10 0.25 0.40 0.09 0.35
NLD0.21 0.80 0.16 0.02 0.04 0.11 0.020.35
CIPHER0.04 0.60 0.03 0.01 0.03 0.07 0.030.34
AC (mean)0.00 0.00 0.03 0.00 0.00 0.08 0.01 0.01
AC (replace)0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
AC (sum)0.00 0.00 0.02 0.00 0.00 0.07 0.04 0.03
KVComm (0.3)0.04 0.26 0.02 0.01 0.01 0.09 0.08 0.31
KVComm (0.5)0.190.88 0.28 0.07 0.12 0.26 0.10 0.33
KVComm (0.7) 0.41 0.89 0.41 0.21 0.25 0.29 0.150.34
Ms: ehristoforu/falcon3-ultraset;M r: huihui-ai/Falcon3-7B-Instruct-abliterated
Baseline0.08 0.36 0.21 0.06 0.04 0.09 0.23 0.31
Skyline0.56 0.95 0.76 0.32 0.56 0.51 0.45 0.37
NLD 0.460.80 0.52 0.19 0.25 0.11 0.24 0.15
CIPHER0.30 0.19 0.27 0.02 0.07 0.06 0.25 0.17
AC (mean)0.01 0.46 0.25 0.06 0.04 0.09 0.23 0.31
AC (replace)0.00 0.49 0.12 0.00 0.01 0.13 0.17 0.31
AC (sum)0.01 0.46 0.25 0.06 0.03 0.10 0.24 0.31
KVComm (0.3) 0.460.690.59 0.19 0.40 0.35 0.29 0.32
KVComm (0.5)0.40 0.92 0.630.25 0.440.45 0.340.35
KVComm (0.7)0.190.960.550.260.42 0.510.31 0.36
avoid wasting its capacity on less important signals. This also explains why using fewer layers can
sometimes yield better performance than using more.
Also note that the performance gain of KVComm is not substantial on TMATH. We attribute this
to that pretraining gives LLMs solid capabilities in mathematical reasoning, which may not dra-
matically benefit from additional context or hints. Moreover, AC performs relatively well on this
dataset, which we consider is because the hints contain information about questions, so even if the
last token’s hidden states are corrupted, it can still generate some useful information.
4.3 BENEFIT OFSELECTIVEKV OVERONECONTIGUOUSCHUNK
DroidSpeak (Liu et al., 2024b) chooses to use one contiguous chunk of context for communication
between LLMs. Despite different problem settings, we evaluate KVComm by replacing the selec-
tion strategy with two hyperparameters, which are two layer indices layer from and layerto, then all
layers between layerfrom and layerto are selected for communication. This is equivalent to using one
contiguous chunk of context for communication. We vary them to select different chunks of layers.
Figure 4 shows that using a single contiguous chunk for communication yields good performance
only in a small region of the hyperparameter space, making it tricky to find the right hyperparam-
eters. In contrast, the scatter and curve plots in Figure 5 demonstrate that KVComm consistently
achieves the best or even outperforms the best contiguous chunk setting for the same number of
layers. Line plots in Figure 6 show that contiguous chunks are most effective when taken from in-
7

<!-- page 8 -->

Published as a conference paper at ICLR 2026
termediate layers, consistent with hypothesisH1in Section 3.2. All results are on HotpotQA with
the Llama-3.1-8B pair, with more in Section O.
4.4 ABLATIONSTUDY ONSELECTIONSTRATEGY
Table 2 compares KVComm with random selection. We find that KVComm consistently outper-
forms random selection across different datasets and selection ratios. When the ratio is high (i.e.,
0.7), the performance gap between our selection strategy and random selection becomes smaller, as
more layers are selected and the impact of the selection strategy is reduced. However, when the ratio
is low (i.e., 0.3), our selection strategy significantly outperforms random selection, demonstrating
its effectiveness in selecting the most informative layers for communication. Comparison results on
other model pairs are provided in Table 9 in Section G, which show similar trends.
Table 2: Comparison with random selection. Best results for each selection ratio arebolded.
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Ms: huihui-ai/Llama-3.2-3B-Instruct-abliterated;M r: suayptalha/DeepSeek-R1-Distill-Llama-3B
Random (0.3)0.05 0.32 0.18 0.07 0.01 0.06 0.17 0.33
KVComm (0.3) 0.46 0.45 0.46 0.09 0.28 0.15 0.28 0.35
Random (0.5)0.26 0.44 0.37 0.08 0.10 0.09 0.21 0.34
KVComm (0.5) 0.57 0.81 0.57 0.27 0.32 0.51 0.36 0.35
Random (0.7) 0.57 0.820.62 0.20 0.34 0.30 0.280.35
KVComm (0.7) 0.570.810.65 0.29 0.36 0.47 0.37 0.35
0 10 20 30
layerto
0
10
20
30layerfrom
F1 Score with One Contig. Chunk of Context
0.2
0.4
0.6
Figure 4: Effective commu-
nication with limited hyperpa-
rameters.
0 10 20 30
#Layers
0.0
0.2
0.4
0.6
0.8F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
Figure 5: KVComm achieves
nearly the best or even outper-
forms contig. chunks.
0 5 10 15 20
layerfrom
0.2
0.4
0.6F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
Figure 6: Chunks in intermedi-
ate layers achieve the most ef-
fective communication.
4.5 ATTENTIONDISTRIBUTIONANALYSIS
We validate hypothesisH2in Section 3.2 by selecting layers with different attention importance
scores for communication. We select9layers with different levels of attention importance scores,
and test the communication performance with Llama-3.2-3B model. The results are shown in Fig-
ure 7. We find that selecting layers with higher scores can achieve better performance, while select-
ing layers with lower scores can diminish the performance. This validates hypothesisH2that layers
with higher attention importance scores are more effective for communication.
4.6 SYSTEMEFFICIENCY
Mathematically, we have shown in Section 3.3 that KVComm can reduce the computation cost
compared to Skyline. We validate this through experiments on the Llama-3.2-3B model pair with
Tipsheets and MultiFieldQA-en datasets. We report the relative FLOPs of KVComm and Skyline
over AC in Figure 8. NLD and CIPHER are not included since they require multiple decoding steps
forM s, which makes the computation cost significantly higher than AC. We find that KVComm
has a significant computation advantage over Skyline, especially when selecting fewer layers for
communication. This demonstrates the efficiency of our KVComm framework in enabling effective
communication with reduced computational overhead by 2.5x to 6x.
8

<!-- page 9 -->

Published as a conference paper at ICLR 2026
In addition to FLOPs, we also report the memory consumption among methods. KVComm similarly
shows a substantial memory advantage over Skyline, as the reduced number of communicated layers
not only lowers computation but also alleviates memory pressure. On Tipsheets, KVComm uses
23% to 73% less memory than Skyline. This further highlights the efficiency of our framework in
achieving lightweight inter-model communication.
Attention Level (high  low)
0.0
0.2
0.4F1 Score
F1 Score Over Attention Level with Llama-3.2-3B
HotpotQA
Tipsheets
Countries
Figure 7: Better communication per-
formance with higher attention level.
AC
Skyline
KVComm (0.3)KVComm (0.5)KVComm (0.7)
0.0
0.5
1.0
1.5
2.0
2.5 GFLOPs
Tipsheets
AC
Skyline
KVComm (0.3)KVComm (0.5)KVComm (0.7)
0
50
100
150
200
MultiFieldQA-en
0.000
0.005
0.010
0.015
0.0
0.1
0.2
0.3
0.4
 Memory (GB)
Relative FLOPs and Memory Cost Over AC with Llama-3.2-3B (per Query)
Figure 8: KVComm requires less computation and mem-
ory compared to Skyline.
5 RELATEDWORK
LLM Inference AccelerationLots of work has focused on accelerating LLM inference.
Computation-level methods such as FlashAttention (Dao et al., 2022) and Memory-Efficient At-
tention (Rabe & Staats, 2021) reduce memory and speed up attention; system-level methods such
as vLLM (Kwon et al., 2023) and DeepSpeed-Inference (Aminabadi et al., 2022) improve overall
throughput and latency; and model-level methods such as quantization (Lin et al., 2024) and prun-
ing (Ma et al., 2023) reduce model size and complexity. These works mainly focus on working
with only one model processing a single long input with the aim of minimizing computation cost.
These approaches are orthogonal to ours and can be combined with KVComm to further improve
efficiency.
Closest to our work are methods that reuse computation across decoding steps or requests. Gao
et al. (2024) introduces a hierarchical KV caching system for all requests; Gim et al. (2024) reuses
prompt KV caches across queries by decomposing inputs; Liu et al. (2024c) compresses KV caches
into compact bitstreams; and Yao et al. (2025) combines multiple chunks’ KV caches by selectively
recomputing a few tokens. In contrast, our work targets communication across different LLMs,
which is more challenging due to parameter differences. Moreover, while prior methods reuse KV
caches uniformly across layers, we enable selective sharing of KV caches from different layers,
further improving efficiency. We do not compare with these works since they are orthogonal to ours.
DroidSpeak (Liu et al., 2024b) aims to accelerate inference for queries with shared prefixes. It
reuses the partial KV cache of these prefixes among different queries. Specifically, it empirically
selects a single contiguous chunk of layers and recomputes the rest with large calibration overhead,
whereas our strategy flexibly selects non-contiguous layers with low overhead, without needing to
recompute the remaining layers. Despite different problem settings, we compare their contiguous-
chunk strategy with ours in Section 4.3, showing the advantages of our approach.
Ye et al. (2025) adjusts KV cache for shared content by referencing a pool of cached examples-
termed anchors that store observed cache deviations under varying prefixes. Our work goes beyond
this related work by: 1) enabling a different type of communication, where the receiver does not
have access to the context, 2) making it possible to efficiently and selectively choose layers of KV
pairs that will be transmitted, and 3) being able to work effectively across different models that are
fine-tuned from one model.
Inter-LLM CommunicationCommunication between multiple LLMs has been explored in sev-
eral recent works. Most works focus on using natural language as the medium of communication.
For example, Du et al. (2023) proposed a natural language debate framework where LLMs itera-
tively critique each other’s answers in natural language to improve the final answer. Liang et al.
(2023) followed a similar idea but introduced a judge model to manage the debate process.
9

<!-- page 10 -->

Published as a conference paper at ICLR 2026
CIPHER (Pham et al., 2023) proposed using embedding space as the medium of communication.
They pass the weighted average of the token embeddings from one LLM to another. Moreover,
AC (Ramesh & Li, 2025) proposed to use the last token’s hidden state as the medium of commu-
nication. They replace the last token’s hidden state of the receiver model with that of the sender
model. Instead, we propose to use the KV pairs as the medium, which can preserve more informa-
tion than just using the last token’s hidden state. We also propose a more effective selection strategy
for choosing which KV pairs to share, which can further improve efficiency.
KV Cache OptimizationSeveral works have explored optimizing KV caches for a single LLM
by (1) compressing the KV caches to reduce memory usage (Ge et al., 2023; Liu et al., 2024a) or (2)
managing the KV caches (offloading) to improve the inference speed (Lee et al., 2024; Xiong et al.,
2024). As our work focuses on layer-wise selection of KV caches for communication between two
LLMs, these methods are orthogonal and can be combined with our method.
6 DISCUSSION
In this section, we discuss the limitations, clarify the scope of current design choices, and outline
promising directions for future research. Additional discussions can be found in Section K.
Heterogeneous Model ArchitecturesOur current KVComm framework assumes that both LLMs
share the same base architecture, i.e., identical models or fine-tuned versions of the same base LLM.
This is because KV pair structures differ substantially across model families, making direct KV ex-
change undefined. This architecture dependency is a practical limitation but not a fundamental one.
Future work could explore learning latent projections, adapters, or other transformation functions to
enable KV exchange across heterogeneous architectures.
Multiple Sender/Receiver ExtensionsWhile we focus on a single sender-receiver pair in this
work, KVComm can be naturally extended to multiple senders and/or receivers. KVComm can
integrate information from multiple senders by concatenating KV caches, and multiple receivers
can independently select layers based on their own attention patterns. As shown in Section J, we
mathematically extend our framework to multiple senders, and perform a preliminary experiment
with two senders and one receiver, showing that multiple senders can improve performance due to
diversified information sources. However, a systematic study of scaling behaviors in larger multi-
agent networks remains future work.
Context-adaptive Online CalibrationKVComm currently adopts a fixed layer-selection strat-
egy after calibration for simplicity and computational efficiency, while context-adaptive selection is
a promising extension. KVComm can naturally support online and dynamic selection. A demon-
stration and analysis of this mechanism is provided in Section L.
Layer Selection PriorsGiven our goal of keeping the method simple, efficient, and broadly repro-
ducible, we opt for the Gaussian prior. Other alternatives, such as entropy-weighted or data-driven,
are promising but introduce significantly higher complexity, e.g., larger calibration sets, training a
selector, or risking overfitting to a particular task distribution. Exploring more sophisticated priors
is an interesting direction for future work.
7 CONCLUSION
In this work, we identified the potential of using KV pairs as an effective medium for communication
between two LLMs. We proposed a novel KVComm framework that enables efficient communica-
tion by selectively sharing KV pairs between LLM models. We designed a selection strategy based
on attention importance scores and a Gaussian prior to select the most relevant layers. Extensive
experiments on diverse datasets and model pairs demonstrated that KVComm can achieve compa-
rable or even superior performance to the Skyline upper bound and other methods, while reducing
communication costs by up to 3x. We highlight the generalization ability of our selection strategy,
which can be effectively calibrated with only a single sample. Our work opens up new possibilities
for efficient inter-LLM communication and paves the way for future research in this direction.
10

<!-- page 11 -->

Published as a conference paper at ICLR 2026
ACKNOWLEDGMENTS
This work was supported by the Knut and Alice Wallenberg Foundation through a Wallenberg
Scholar Grant to Prof. Dejan Kosti ´c. This work has been partially supported by Vinnova (the Swe-
den’s Innovation Agency), the Swedish Research Council (agreement No. 2021-04212), and KTH
Digital Futures. We thank the anonymous reviewers for their insightful comments and constructive
suggestions. We also thank Nicolae Filat for providing an early version of the code for CIPHER
comparison and for helpful suggestions on the illustrations.
Computations were enabled by the Berzelius resource provided by the Knut and Alice Wallenberg
Foundation at the National Supercomputer Centre. We also acknowledge the EuroHPC Joint Un-
dertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by
CINECA (Italy) and the LEONARDO consortium through an EuroHPC Development Access call.
REPRODUCIBILITYSTATEMENT
We provide detailed descriptions of the datasets, model pairs, and implementation details in Sec-
tion B. The code and synthetic datasets, Countries and Tipsheets, are uploaded to GitHub to facilitate
reproducibility upon the publication of this work.
REFERENCES
Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Co-
jocaru, M ´erouane Debbah, ´Etienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic,
et al. The falcon series of open language models.arXiv preprint arXiv:2311.16867, 2023.
Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton
Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. Deepspeed-inference:
enabling efficient inference of transformer models at unprecedented scale. InSC22: International
Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–15. IEEE,
2022.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du,
Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual,
multitask benchmark for long context understanding. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/
2024.acl-long.172. URLhttps://aclanthology.org/2024.acl-long.172.
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-
efficient exact attention with io-awareness.Advances in neural information processing systems,
35:16344–16359, 2022.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset
of information-seeking questions and answers anchored in research papers.arXiv preprint
arXiv:2105.03011, 2021.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving fac-
tuality and reasoning in language models through multiagent debate. InForty-first International
Conference on Machine Learning, 2023.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
arXiv e-prints, pp. arXiv–2407, 2024.
Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang,
Zhou Yu, and Pengfei Zuo.{Cost-Efficient}large language model serving for multi-turn conver-
sations with{CachedAttention}. In2024 USENIX Annual Technical Conference (USENIX ATC
24), pp. 111–126, 2024.
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells
you what to discard: Adaptive kv cache compression for llms.arXiv preprint arXiv:2310.01801,
2023.
11

<!-- page 12 -->

Published as a conference paper at ICLR 2026
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are
key-value memories.arXiv preprint arXiv:2012.14913, 2020.
In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt
cache: Modular attention reuse for low-latency inference.Proceedings of Machine Learning and
Systems, 6:325–338, 2024.
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A human-
annotated dialogue dataset for abstractive summarization. InProceedings of the 2nd Workshop
on New Frontiers in Summarization, pp. 70–79, Hong Kong, China, November 2019. Association
for Computational Linguistics. doi: 10.18653/v1/D19-5409. URLhttps://www.aclweb.
org/anthology/D19-5409.
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V . Chawla, Olaf Wiest,
and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and
challenges, 2024. URLhttps://arxiv.org/abs/2402.01680.
Ganesh Jawahar, Beno ˆıt Sagot, and Djam ´e Seddah. What does bert learn about the structure of
language? InACL 2019-57th Annual Meeting of the Association for Computational Linguistics,
2019.
Shuowei Jin, Xueshen Liu, Qingzhao Zhang, and Z Morley Mao. Compute or load kv cache? why
not both?arXiv preprint arXiv:2410.03065, 2024.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. InProceedings of the 29th symposium on operating systems princi-
ples, pp. 611–626, 2023.
Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim.{InfiniGen}: Efficient generative
inference of large language models with dynamic{KV}cache management. In18th USENIX
Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 155–172, 2024.
Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Com-
municative agents for” mind” exploration of large language model society.Advances in Neural
Information Processing Systems, 36:51991–52008, 2023.
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming
Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-
agent debate.arXiv preprint arXiv:2305.19118, 2023.
Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan
Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization
for on-device llm compression and acceleration.Proceedings of machine learning and systems,
6:87–100, 2024.
Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. Minicache:
Kv cache compression in depth dimension for large language models.Advances in Neural Infor-
mation Processing Systems, 37:139997–140031, 2024a.
Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua
Cheng, Junchen Jiang, Shan Lu, et al. Droidspeak: Kv cache sharing for cross-llm communication
and multi-llm serving.arXiv preprint arXiv:2411.02820, 2024b.
Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du,
Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al. Cachegen: Kv cache compression and
streaming for fast large language model serving. InProceedings of the ACM SIGCOMM 2024
Conference, pp. 38–56, 2024c.
Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large
language models.Advances in neural information processing systems, 36:21702–21720, 2023.
12

<!-- page 13 -->

Published as a conference paper at ICLR 2026
Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A Plum-
mer, Zhaoran Wang, and Hongxia Yang. Let models speak ciphers: Multiagent debate through
embeddings.arXiv preprint arXiv:2310.06272, 2023.
Changyong Qi, Yu’ang Wei, Haoxin Xu, Longwei Zheng, Peiji Chen, and Xiaoqing Gu. Tmath
a dataset for evaluating large language models in generating educational hints for math word
problems. InProceedings of the 31st International Conference on Computational Linguistics, pp.
5082–5093, 2025.
Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu,
and Maosong Sun. Communicative agents for software development.arXiv preprint
arXiv:2307.07924, 6(3):1, 2023.
A Yang Qwen, Baosong Yang, B Zhang, B Hui, B Zheng, B Yu, Chengpeng Li, D Liu, F Huang,
H Wei, et al. Qwen2. 5 technical report.arXiv preprint, 2024.
Markus N Rabe and Charles Staats. Self-attention does not need o (n2) memory.arXiv preprint
arXiv:2112.05682, 2021.
Vignav Ramesh and Kenneth Li. Communicating activations between language model agents.arXiv
preprint arXiv:2501.14082, 2025.
Lijun Sun, Yijun Yang, Qiqi Duan, Yuhui Shi, Chao Lyu, Yu-Cheng Chang, Chin-Teng Lin, and
Yang Shen. Multi-agent coordination across diverse applications: A survey, 2025.
Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and
Hoang D. Nguyen. Multi-agent collaboration mechanisms: A survey of llms, 2025.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554, 2022.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, R ´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick
von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gug-
ger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art
natural language processing. InProceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. As-
sociation for Computational Linguistics. URLhttps://www.aclweb.org/anthology/
2020.emnlp-demos.6.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-
agent conversations. InFirst Conference on Language Modeling, 2024.
Yi Xiong, Hao Wu, Changxu Shao, Ziqing Wang, Rui Zhang, Yuhong Guo, Junping Zhao,
Ke Zhang, and Zhenxuan Pan. Layerkv: Optimizing large language model serving with layer-
wise kv cache management.arXiv preprint arXiv:2410.00428, 2024.
Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen
Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt
and beyond.ACM Trans. Knowl. Discov. Data, 18(6), 2024a. ISSN 1556-4681. doi: 10.1145/
3649506. URLhttps://doi.org/10.1145/3649506.
Joshua C Yang, Damian Dalisan, Marcin Korecki, Carina I Hausladen, and Dirk Helbing. Llm
voting: Human choices and ai collective decision-making. InProceedings of the AAAI/ACM
Conference on AI, Ethics, and Society, volume 7, pp. 1696–1708, 2024b.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InConference on Empirical Methods in Natural Language Processing (EMNLP),
2018.
13

<!-- page 14 -->

Published as a conference paper at ICLR 2026
Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan
Lu, and Junchen Jiang. Cacheblend: Fast large language model serving for rag with cached
knowledge fusion. InProceedings of the Twentieth European Conference on Computer Systems,
pp. 94–109, 2025.
Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin,
Zhijian Liu, Jianyi Zhang, Danyang Zhuo, et al. Kvcomm: Online cross-context kv-cache com-
munication for efficient llm-based multi-agent systems.arXiv preprint arXiv:2510.12872, 2025.
14

<!-- page 15 -->

Published as a conference paper at ICLR 2026
A THEUSE OFLARGELANGUAGEMODELS(LLMS)
Large language models, including ChatGPT, were employed to provide assistance in improving
the clarity, coherence, and fluency of the manuscript. These tools were used solely for language
refinement, and all scientific content and interpretations remain the responsibility of the authors.
B EXPERIMENTALSETUP
In this appendix, we provide more details about the experimental setup, including dataset details,
implementation details, fine-tuned model pairs, and descriptions of compared methods.
B.1 DATASET
We provide sample prompts and expected answers for the Countries and Tipsheets datasets in Ta-
ble 3, which are inspired by Ramesh & Li (2025). We also provide the statistics of all datasets
used in our experiments in Table 4. HotpotQA, QASPER, MuSiQuest, and TMATH datasets are
randomly sampled from their original datasets to reduce the evaluation cost. Extended results on the
full datasets are provided in Section E.
Table 3: Sample prompts and expected answers for Countries and Tipsheets datasets inspired by
Ramesh & Li (2025).
Dataset Role Content
Countries
CUma is at the Mahaffie House.
QWhich country is Uma located in?
AnswerUnited States
Tipsheets
CAtlas LLC is under pressure amid softer trends; EPS -17%; won a sizable
customer contract but faces a lawsuit. Sable LLC shows clear momentum
and improving execution; authorized a buyback but reported a cyber in-
cident. Trace LLC looks balanced with a mixed near-term setup.
QYou must invest in exactly one company from Atlas LLC, Sable LLC, Trace
LLC. Which do you choose?
AnswerSable LLC
Table 4: Statistics of the datasets in our experiments.
Dataset Size
Countries 200
Tipsheets 500
HotpotQA (Yang et al., 2018) 500
QASPER (Dasigi et al., 2021) 500
MuSiQuest (Trivedi et al., 2022) 500
MultiFieldQA-en (Bai et al., 2024) 150
2WikiMQA (Bai et al., 2024) 200
TMATH (Qi et al., 2025) 300
B.2 IMPLEMENTATIONDETAILS
We implement our KVComm framework based on the Hugging Face Transformers library (Wolf
et al., 2020), and models are loaded in bfloat16 precision. We set the hyperparameters of our selec-
tion strategy asµ=L/2, andσ= 10, whereLis the total number of layers in the model. For NLD
and CIPHER methods, we set the number of debate rounds to 2, and the maximum generation length
to 256 in the debate process. For KVComm,αis set to1for Llama family models, and0.8for Qwen
and Falcon family models. These values are obtained by validating on a left-out set. All experiments
are conducted on a cluster of nodes, each equipped with an Intel®Xeon®Platinum 8358 Processor
15

<!-- page 16 -->

Published as a conference paper at ICLR 2026
@2.60 GHzand 4 NVIDIA A100 GPUs with64 GBmemory. We obtain the FLOPs with PyTorch
Profiler3.
B.3 MODELPAIRS
We conduct experiments on nine different model pairs, shown in Table 5. The first four pairs consist
of the same LLMs, while the last five pairs consist of models that are fine-tuned on the same base
LLM.
Table 5: Model pairs in the evaluation.M s is the sender model, andM r is the receiver model.
IndexM s Mr Note
1 meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct Same model2 meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-3B-Instruct Same model3 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct Same model4 tiiuae/Falcon3-7B-Instruct tiiuae/Falcon3-7B-Instruct Same model5 yuvraj17/EvolCodeLlama-3.1-8B-Instruct Team-ACE/ToolACE-2-Llama-3.1-8B Fine-tuned on 16 huihui-ai/Llama-3.2-3B-Instruct-abliterated suayptalha/DeepSeek-R1-Distill-Llama-3B Fine-tuned on 27 Orion-zhen/Qwen2.5-7B-Instruct-Uncensored bespokelabs/Bespoke-Stratos-7B Fine-tuned on 38 ehristoforu/falcon3-ultraset huihui-ai/Falcon3-7B-Instruct-abliterated Fine-tuned on 49 arcee-ai/Llama-3.1-SuperNova-Lite deepseek-ai/DeepSeek-R1-Distill-Llama-8B Fine-tuned on 1
B.4 COMPAREDMETHODDESCRIPTIONS
We compare our proposed KVComm framework with the following methods:
•Baseline:M r processes the queryQwithoutany communication fromM s.
•Skyline:M r directly processes the concatenation of the contextCand queryQ. This
serves as an upper bound for performance.
•Natural Language Debate (NLD)(Du et al., 2023): Each model generates an initial an-
swer, and then they iteratively critique each other’s answers in natural language for a fixed
number of rounds. Finally, one model produces the final answer based on the entire debate
history. Compared to the original debate style setting, we use an information-transfer style,
which explicitly promptsM s that it has to summarize the contextCin its initial answer.
We set the number of debate rounds to 2.
•CIPHER(Pham et al., 2023): Similar to NLD, but instead of communicating in natural
language, the models communicate by passing the weighted average of the token embed-
dings from one LLM to another. We use the same prompt as NLD, and set the number of
debate rounds to 2.
•AC(Ramesh & Li, 2025): Communicate with the last token’s hidden state. Replace the last
token’s hidden state ofMr with that ofM s. We also test with mean and sum operations.
C TOKENIMPORTANCE ATDIFFERENTPOSITIONS
We add more details and experiments related to Section 2.2.1 in this appendix.
C.1 DETAILEDEXPERIMENTPROCEDURE
We provide a detailed description of the experiment procedure in Section 2.2.1. Considering a model
MwithLlayers, given an inputXwithNtokens, we run a partial forward pass until layerlto
obtain the hidden states{h l
i}N
i=1. Then, given a specific token positionk, if we perform theRetain
operation, we create a modified set of hidden states{ ˜hl
i}N
i=1 as follows:
˜hl
i =
hl
i,ifi=k
0,otherwise
3https://docs.pytorch.org/docs/stable/profiler.html
16

<!-- page 17 -->

Published as a conference paper at ICLR 2026
If we perform theRemoveoperation, we create the modified set of hidden states{ ˜hl
i}N
i=1 as follows:
˜hl
i =
0,ifi=k
hl
i,otherwise
We then continue the forward pass from layerl+ 1to layerLusing the modified hidden states
{˜hl
i}N
i=1 as input, and obtain the final output of the model. We evaluate the model’s performance on
the task with different token positionskand layerl.
C.2 MOREEXPERIMENTS ONTOKENIMPORTANCE
We conduct the same experiment as in Section 2.2.1 on other datasets and models to investigate the
effect of tokens at different positions in the sequence on the model’s output. We report the results on
MMLU Social Science, MMLU STEM, and MMLU Humanities using Llama-3.1-8B and Llama-
3.2-3B models in Figure 9. We can see that the last token’s hidden state plays the most critical role
in the latter layers, which is consistent with the observation in Section 2.2.1.
0 10 20 30
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Retain the token
Last
None
Random
First
0 10 20 30
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Remove the token
Layer
Accuracy
(a) Llama-3.2-3B on MMLU Social Science
0 10 20 30
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
Retain the token
Last
None
Random
First
0 10 20 30
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
Remove the token
Layer
Accuracy (b) Llama-3.1-8B on MMLU STEM
0 5 10 15 20 25
0.20
0.25
0.30
0.35
0.40
0.45
0.50
Retain the token
Last
None
Random
First
0 5 10 15 20 25
0.25
0.30
0.35
0.40
0.45
0.50
Remove the token
Layer
Accuracy
(c) Llama-3.2-3B on MMLU STEM
0 10 20 30
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
Retain the token
Last
None
Random
First
0 10 20 30
0.3
0.4
0.5
0.6
Remove the token
Layer
Accuracy (d) Llama-3.1-8B on MMLU Humanities
0 5 10 15 20 25
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
Retain the token
Last
None
Random
First
0 5 10 15 20 25
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
Remove the token
Layer
Accuracy
(e) Llama-3.2-3B on MMLU Humanities
Figure 9: Effect of removing or retaining a token’s hidden state across different positions on MMLU
Social Science, MMLU STEM, and MMLU Humanities accuracy using Llama-3.1-8B and Llama-
3.2-3B models.
D UTILIZINGALLTOKENS
We add more details and experiments related to Section 2.2.2 in this appendix.
D.1 DETAILEDEXPERIMENTPROCEDURE
We provide a detailed description of the experiment procedure in Section 2.2.2. Considering two
modelsM s andM r, each withLlayers, givenCandQas input, we run a partial forward pass
17

<!-- page 18 -->

Published as a conference paper at ICLR 2026
ofM s until layerkto obtain the hidden stateH s
k ∈R |C|×d for all tokens inC, where|C|is the
number of tokens inC, anddis the hidden dimension. Another partial forward pass ofM r is run
until layerjto obtain the hidden stateH r
j ∈R |Q|×d for all tokens inQ, where|Q|is the number
of tokens inQ. We then modify the hidden states ofM r at layerjby prepending the hidden states
fromM s at layerkas follows:
˜Hr
j =

Hs
k
Hr
j

We continue the forward pass from layerj+ 1to layerLusing the modified hidden states ˜Hr
j as
input, and obtain the final output of the model. We evaluate the model’s performance on the task
with different layerskandj.
D.2 MOREEXPERIMENTS ONUTILIZINGALLTOKENS
We conduct the same experiment as in Section 2.2.2 on Countries, Tipsheets, and HotpotQA datasets
using Llama-3.1-8B, Llama-3.2-3B, and Qwen2.5-7B models. The results are shown in Figure 10.
We can see the results are consistent with the observation in Section 2.2.2.
0 10 20
0
5
10
15
20
25
Llama-3.1-8B
0 10 20
Llama-3.2-3B
0 10 20
Qwen2.5-7B
0.0
0.1
0.2
0.3
0.4
0.5
0.6
F1 Score
Receiver r layer index (j)
Sender s layer index (k)
(a) Countries
0 10 20
0
5
10
15
20
25
Llama-3.1-8B
0 10 20
Llama-3.2-3B
0 10 20
Qwen2.5-7B
0.0
0.2
0.4
0.6
0.8
F1 Score
Receiver r layer index (j)
Sender s layer index (k)
(b) Tipsheets
0 10 20
0
5
10
15
20
25
Llama-3.1-8B
0 10 20
Llama-3.2-3B
0 10 20
Qwen2.5-7B
0.0
0.2
0.4
0.6
F1 Score
Receiver r layer index (j)
Sender s layer index (k)
(c) HotpotQA
Figure 10: Performance heatmap of prepending the hidden states from certain layers ofM s to
certain layers ofM r on Countries, Tipsheets, and HotpotQA.
18

<!-- page 19 -->

Published as a conference paper at ICLR 2026
E EXPERIMENT RESULTS WITH EXTENDED DATASETS
To further validate the effectiveness of our KVComm framework, we conduct experiments on the full
datasets of HotpotQA, QASPER, MuSiQuest, mainly HotpotQA-E, QASPER-E, and MuSiQuest-E.
Moreover, we include a new human-created summarization dataset, SAMSum, which represents a
different task type. The statistics of these datasets are shown in Table 7. We report the results on
these extended datasets in Table 6. The results show similar trends as in Section 4.2, demonstrating
the robustness of our KVComm framework across different datasets and tasks.
Table 6: Communication results on extended communication tasks. The best results in each block
are inbold, and the second best results are underlined .
Method HotpotQA-E QASPER-E MuSiQuest-E SAMSum
Ms: huihui-ai/Llama-3.2-3B-Instruct-abliterated;
Mr: suayptalha/DeepSeek-R1-Distill-Llama-3B
Baseline0.22 0.03 0.06 0.26
Skyline0.77 0.52 0.25 0.33
NLD0.45 0.10 0.180.28
CIPHER0.51 0.10 0.200.28
AC (mean)0.24 0.03 0.06 0.26
AC (replace)0.06 0.00 0.01 0.26
AC (sum)0.23 0.03 0.06 0.26
KVComm (0.3)0.44 0.25 0.11 0.25
KVComm (0.5) 0.61 0.36 0.25 0.28
KVComm (0.7)0.71 0.38 0.30 0.29
Ms: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored;
Mr: bespokelabs/Bespoke-Stratos-7B
Baseline0.15 0.04 0.06 0.25
Skyline0.58 0.27 0.10 0.35
NLD 0.24 0.02 0.07 0.28
CIPHER0.04 0.01 0.020.37
AC (mean)0.03 0.00 0.00 0.01
AC (replace)0.00 0.00 0.00 0.00
AC (sum)0.03 0.00 0.00 0.04
KVComm (0.3)0.02 0.00 0.01 0.18
KVComm (0.5)0.210.14 0.08 0.30
KVComm (0.7)0.40 0.34 0.210.35
Ms: Orion-ehristoforu/falcon3-ultraset;
Mr: huihui-ai/Falcon3-7B-Instruct-abliterated
Baseline0.21 0.06 0.06 0.27
Skyline0.78 0.60 0.33 0.36
NLD 0.52 0.10 0.28 0.28
CIPHER0.28 0.03 0.09 0.17
AC (mean)0.24 0.06 0.07 0.26
AC (replace)0.12 0.02 0.01 0.26
AC (sum)0.23 0.05 0.06 0.26
KVComm (0.3)0.590.15 0.40 0.28
KVComm (0.5)0.590.22 0.460.31
KVComm (0.7)0.59 0.260.36 0.32
Table 7: Statistics of extended datasets.
Dataset Size
HotpotQA-E (Yang et al., 2018) 7,405
QASPER-E (Dasigi et al., 2021) 1,726
MuSiQuest-E (Trivedi et al., 2022) 2,417
SAMSum (Gliwa et al., 2019) 819
F MORECOMMUNICATIONRESULTS
We provide more communication results on different model pairs in Table 8, which show similar
trends as in Section 4.2.
19

<!-- page 20 -->

Published as a conference paper at ICLR 2026
Table 8: More communication results of different methods. Best results arebolded, second best
underlined (excluding Baseline and Skyline). We reportM r for Baseline and Skyline for fairness.
KVComm (0.3/0.5/0.7)denotes selecting 30%/50%/70% of layers’ KV pairs for communication,
i.e.,M=⌈0.3L⌉,M=⌈0.5L⌉,M=⌈0.7L⌉.
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Ms: meta-llama/Llama-3.1-8B-Instruct;M r: meta-llama/Llama-3.1-8B-Instruct
Baseline0.00 0.05 0.19 0.02 0.01 0.07 0.06 0.35
Skyline0.62 0.92 0.74 0.35 0.54 0.56 0.52 0.36
NLD0.58 0.87 0.52 0.13 0.25 0.17 0.10 0.36
CIPHER0.57 0.84 0.57 0.13 0.25 0.15 0.10 0.36
AC (mean)0.00 0.12 0.19 0.02 0.01 0.08 0.03 0.35
AC (replace)0.00 0.36 0.15 0.02 0.01 0.07 0.05 0.35
AC (sum)0.00 0.09 0.20 0.02 0.01 0.09 0.04 0.35
KVComm (0.3)0.51 0.93 0.330.07 0.11 0.21 0.290.37
KVComm (0.5) 0.620.95 0.60 0.290.34 0.50 0.37 0.37
KVComm (0.7) 0.62 0.96 0.69 0.29 0.39 0.53 0.38 0.38
Ms: meta-llama/Llama-3.2-3B-Instruct;M r: meta-llama/Llama-3.2-3B-Instruct
Baseline0.02 0.01 0.16 0.00 0.02 0.10 0.09 0.35
Skyline0.56 0.87 0.72 0.23 0.45 0.45 0.37 0.38
NLD0.51 0.71 0.49 0.09 0.18 0.11 0.07 0.34
CIPHER0.45 0.73 0.46 0.08 0.17 0.09 0.07 0.33
AC (mean)0.00 0.07 0.18 0.01 0.02 0.09 0.06 0.35
AC (replace)0.01 0.37 0.13 0.01 0.02 0.06 0.03 0.34
AC (sum)0.00 0.34 0.20 0.02 0.02 0.10 0.07 0.34
KVComm (0.3)0.51 0.48 0.47 0.10 0.20 0.17 0.28 0.35
KVComm (0.5)0.55 0.79 0.58 0.24 0.27 0.47 0.350.36
KVComm (0.7) 0.57 0.80 0.65 0.27 0.29 0.480.31 0.37
Ms: Qwen/Qwen2.5-7B-Instruct;M r: Qwen/Qwen2.5-7B-Instruct
Baseline0.00 0.32 0.19 0.05 0.03 0.06 0.17 0.32
Skyline0.54 0.97 0.68 0.30 0.48 0.49 0.45 0.33
NLD0.18 0.86 0.37 0.09 0.11 0.11 0.19 0.30
CIPHER0.18 0.87 0.34 0.07 0.10 0.11 0.16 0.31
AC (mean)0.00 0.37 0.15 0.01 0.02 0.10 0.200.33
AC (replace)0.00 0.35 0.02 0.00 0.00 0.10 0.090.32
AC (sum)0.00 0.41 0.14 0.02 0.02 0.08 0.170.32
KVComm (0.3)0.04 0.31 0.06 0.02 0.01 0.19 0.190.32
KVComm (0.5) 0.570.92 0.49 0.18 0.20 0.40 0.25 0.32
KVComm (0.7)0.56 0.98 0.72 0.29 0.48 0.45 0.35 0.33
Ms: tiiuae/Falcon3-7B-Instruct;M r: tiiuae/Falcon3-7B-Instruct
Baseline0.06 0.33 0.19 0.04 0.04 0.09 0.21 0.31
Skyline0.57 0.95 0.70 0.24 0.50 0.49 0.48 0.35
NLD0.38 0.71 0.44 0.07 0.19 0.13 0.24 0.20
CIPHER 0.470.63 0.41 0.03 0.19 0.09 0.21 0.21
AC (mean)0.03 0.51 0.22 0.04 0.04 0.09 0.220.32
AC (replace)0.00 0.57 0.09 0.00 0.02 0.12 0.140.31
AC (sum)0.04 0.51 0.22 0.04 0.03 0.09 0.220.32
KVComm (0.3)0.06 0.67 0.41 0.12 0.22 0.41 0.230.32
KVComm (0.5)0.160.94 0.52 0.22 0.33 0.47 0.33 0.32
KVComm (0.7)0.230.96 0.54 0.220.32 0.470.29 0.32
Ms: yuvraj17/EvolCodeLlama-3.1-8B-Instruct;M r: Team-ACE/ToolACE-2-Llama-3.1-8B
Baseline0.00 0.07 0.04 0.00 0.01 0.08 0.01 0.34
Skyline0.24 0.95 0.37 0.17 0.15 0.51 0.25 0.39
NLD0.29 0.82 0.17 0.04 0.05 0.13 0.02 0.34
CIPHER0.21 0.86 0.19 0.03 0.06 0.15 0.03 0.33
AC (mean)0.00 0.31 0.03 0.00 0.01 0.11 0.01 0.34
AC (replace)0.00 0.30 0.05 0.00 0.01 0.10 0.02 0.33
AC (sum)0.00 0.27 0.04 0.00 0.01 0.09 0.01 0.34
KVComm (0.3)0.12 0.95 0.12 0.05 0.04 0.26 0.190.36
KVComm (0.5) 0.55 0.980.38 0.15 0.14 0.43 0.28 0.38
KVComm (0.7)0.53 0.97 0.51 0.22 0.25 0.49 0.33 0.38
Ms: arcee-ai/Llama-3.1-SuperNova-Lite;M r: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
Baseline0.07 0.30 0.11 0.01 0.03 0.09 0.16 0.23
Skyline0.55 0.80 0.52 0.17 0.40 0.41 0.16 0.29
Continued on next page
20

<!-- page 21 -->

Published as a conference paper at ICLR 2026
Table 8 – continued from previous page
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
NLD0.30 0.39 0.20 0.02 0.06 0.080.190.22
CIPHER0.47 0.71 0.27 0.03 0.11 0.14 0.14 0.18
AC (mean)0.00 0.31 0.08 0.02 0.02 0.09 0.14 0.25
AC (replace)0.00 0.39 0.04 0.01 0.00 0.160.16 0.28
AC (sum)0.02 0.34 0.07 0.02 0.02 0.080.16 0.24
KVComm (0.3)0.09 0.52 0.10 0.01 0.03 0.09 0.080.28
KVComm (0.5)0.410.760.33 0.05 0.21 0.23 0.090.29
KVComm (0.7) 0.53 0.76 0.47 0.12 0.28 0.310.140.29
G ABLATIONSTUDY ONSELECTIONSTRATEGY
We conduct more ablation studies on the selection strategy by comparing with random selection and selection
based on only attention importance scores. The results are shown in Table 9, which show similar trends as in
Section 4.4.
Table 9: More comparison results with random selection. Best results for each selection ratio are
bolded.
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Ms: meta-llama/Llama-3.1-8B-Instruct;M r: meta-llama/Llama-3.1-8B-Instruct
Random (0.3)0.02 0.35 0.240.070.04 0.07 0.12 0.35
KVComm (0.3) 0.51 0.93 0.33 0.07 0.11 0.21 0.29 0.37
Random (0.5)0.49 0.76 0.58 0.15 0.29 0.29 0.27 0.36
KVComm (0.5) 0.62 0.95 0.60 0.29 0.34 0.50 0.37 0.37
Random (0.7) 0.630.880.76 0.32 0.490.52 0.34 0.37
KVComm (0.7)0.620.960.69 0.29 0.390.53 0.38 0.38
Ms: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored;M r: bespokelabs/Bespoke-Stratos-7B
Random (0.3)0.00 0.09 0.00 0.00 0.00 0.06 0.010.31
KVComm (0.3) 0.04 0.26 0.02 0.01 0.01 0.09 0.08 0.31
Random (0.5)0.12 0.32 0.06 0.00 0.03 0.15 0.040.33
KVComm (0.5) 0.19 0.88 0.28 0.07 0.12 0.26 0.10 0.33
Random (0.7)0.16 0.76 0.14 0.03 0.02 0.20 0.040.34
KVComm (0.7) 0.41 0.89 0.41 0.21 0.25 0.29 0.15 0.34
Ms: ehristoforu/falcon3-ultraset;M r: huihui-ai/Falcon3-7B-Instruct-abliterated
Random (0.3)0.35 0.36 0.23 0.06 0.07 0.14 0.240.31
KVComm (0.3) 0.46 0.69 0.59 0.19 0.40 0.35 0.29 0.32
Random (0.5)0.23 0.42 0.27 0.09 0.08 0.15 0.28 0.31
KVComm (0.5) 0.40 0.92 0.63 0.25 0.44 0.45 0.34 0.35
Random (0.7)0.18 0.94 0.51 0.23 0.35 0.47 0.30 0.34
KVComm (0.7) 0.19 0.96 0.55 0.26 0.42 0.51 0.31 0.36
Ms: meta-llama/Llama-3.2-3B-Instruct;M r: meta-llama/Llama-3.2-3B-Instruct
Random (0.3)0.02 0.29 0.11 0.06 0.02 0.07 0.16 0.34
KVComm (0.3) 0.51 0.48 0.47 0.10 0.20 0.17 0.28 0.35
Random (0.5)0.28 0.44 0.30 0.06 0.06 0.06 0.19 0.35
KVComm (0.5) 0.55 0.79 0.58 0.24 0.27 0.47 0.35 0.36
Random (0.7)0.540.810.62 0.210.300.30 0.26 0.36
KVComm (0.7) 0.570.800.65 0.270.290.48 0.31 0.37
Ms: Qwen/Qwen2.5-7B-Instruct;M r: Qwen/Qwen2.5-7B-Instruct
Random (0.3)0.00 0.34 0.05 0.00 0.00 0.08 0.100.30
KVComm (0.3) 0.04 0.31 0.06 0.02 0.01 0.19 0.19 0.32
Random (0.5)0.00 0.32 0.10 0.02 0.02 0.10 0.160.32
KVComm (0.5) 0.57 0.92 0.49 0.18 0.20 0.40 0.25 0.32
Random (0.7)0.41 0.71 0.28 0.04 0.04 0.21 0.17 0.32
KVComm (0.7) 0.56 0.98 0.72 0.29 0.48 0.45 0.35 0.33
Ms: tiiuae/Falcon3-7B-Instruct;M r: tiiuae/Falcon3-7B-Instruct
Random (0.3)0.01 0.35 0.18 0.04 0.03 0.12 0.21 0.30
KVComm (0.3) 0.06 0.67 0.41 0.12 0.22 0.41 0.23 0.32
Random (0.5)0.04 0.41 0.24 0.03 0.05 0.16 0.24 0.31
KVComm (0.5) 0.16 0.94 0.52 0.22 0.33 0.47 0.33 0.32
Continued on next page
21

<!-- page 22 -->

Published as a conference paper at ICLR 2026
Table 9 – continued from previous page
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Random (0.7)0.19 0.95 0.51 0.20 0.29 0.42 0.260.32
KVComm (0.7) 0.23 0.96 0.54 0.22 0.32 0.47 0.29 0.32
Ms: yuvraj17/EvolCodeLlama-3.1-8B-Instruct;M r: Team-ACE/ToolACE-2-Llama-3.1-8B
Random (0.3)0.00 0.34 0.06 0.00 0.01 0.13 0.03 0.34
KVComm (0.3) 0.12 0.95 0.12 0.05 0.04 0.26 0.19 0.36
Random (0.5)0.03 0.79 0.29 0.06 0.09 0.32 0.16 0.35
KVComm (0.5) 0.55 0.98 0.38 0.15 0.14 0.43 0.28 0.38
Random (0.7)0.37 0.850.590.210.270.470.330.36
KVComm (0.7) 0.53 0.970.510.220.250.49 0.33 0.38
H CALIBRATIONSETSIZE
0 25 50 75 100 125
Calibration size
0.6
0.7
0.8F1 Score
Countries
HotpotQA
Tipsheets
Figure 11: Effect of calibration set size. Calibration set size does not significantly affect the test
performance.
We investigate how many samples are needed in the calibration set so that the selection strategy can generalize
well to the test set. If a smaller calibration set can achieve good performance on the test set, it would be more
practical since it would require less cost to obtain the selected layers. We conduct the experiment on Countries,
Tipsheets, and HotpotQA datasets using the Llama-3.2-3B model. As the results in Figure 11 show, we can
see that using only one sample in the calibration set can already achieve the same performance as using more
samples (up to 128 samples). This suggests that our selection strategy can generalize well to the test set even
with a very small calibration set. In all other experiments in the paper, we use one sample in the calibration set.
I IMPACT OFTRANSMITTEDTOKENLENGTH ONNLD
Transmitted token length is an important factor affecting the performance of natural language-based commu-
nication methods like NLD, which refers to the maximum number of tokens generated by the sender model to
communicate with the receiver model. To investigate the impact of transmitted token length on NLD, we con-
duct experiments on HotpotQA, MultiFieldQA-en, and 2WikiMQA datasets with different transmitted token
lengths ranging from 64 to 1024 tokens. The results are shown in Figure 12. We can see that as the transmitted
token length increases from 64 to 128, the performance of NLD improves. However, as the transmitted token
length continues to increase beyond 128 tokens, the performance gains become marginal. This suggests that
there is a moderate transmitted token length (e.g., 128 tokens) is sufficient without incurring excessive com-
munication overhead. In our main experiments, we set the transmitted token length to 256 tokens for NLD to
ensure a fair comparison with other methods.
J MULTI-SOURCEKVCOMM
J.1 EXTENDINGKVCOMM TOMULTIPLESOURCES
KVComm can be naturally extended to multiple sources by integrating the KV pairs from different sender
models. Mathematically, if we haveN s sender modelsM s1 ,M s2 , . . . ,MsNs and one receiver modelM r,
each senderM si processes the contextC i and generates its own KV pairs{(k l
si ,v l
si )}at each layerl. The
receiverM r can then receive the KV pairs from all senders and use them to compute the attention scores. The
22

<!-- page 23 -->

Published as a conference paper at ICLR 2026
200 400 600 800 1000
Transmitted T oken Length
0.10
0.15
0.20
0.25Mean NLD Result
NLD Result with Different Trans. T oken Lengths
Model Pair Index 6
Model Pair Index 7
Model Pair Index 8
Figure 12: Effect of transmitted token length on NLD. A moderate length is sufficient for NLD.
attention scores can be computed as follows:
ˆSl
a = 1
HT
HX
h=1
|Q|X
q=1
NsX
i=1
|Ci|X
c=1
al
h,q,i,c,
where|C i|is the number of tokens in the contextC i, anda l
h,q,i,c is the attention weight assigned by headhat
layerlfrom tokenqto the context tokencof senderM si. The attention scores ˆSl
a are then integrated with the
Gaussian prior to compute the selection scores.
Given the selection scores, a subset of KV pairs{(k
lj
si ,v
lj
si )}can be selected from each sender modelM si at
each layerl j. The selected KV pairs are concatenated to form the final KV pairs for the receiver modelM r:
kl
r ←[k
lj
s1 ;k
lj
s2 ;. . .;k
lj
sNs ;k l
r],
vl
r ←[v
lj
s1 ;v
lj
s2 ;. . .;v
lj
sNs ;v l
r].
wherelcorresponds to a selected layerl j.
J.2 EXPERIMENT WITHTWOSENDERS ANDONERECEIVER
We experiment with the scenario of two senders and one receiver to demonstrate the feasibility of extending
KVComm to multiple sources. As shown in Table 10, we find that two senders can outperform one sender, for
17 out of 27 cases. We argue this is because of the diversification of information sources and agent thought.
Owing to the usage of KV pairs, we can naturally integrate multiple sources, while NLD and CIPHER cannot,
suffering performance degradation.
K ADDITIONALDISCUSSION
We have additional discussions on the details and choices of our method.
Positional Embedding CoherenceKVComm is designed to preserve positional coherence across all
layers. For the receiver model, in each layer, we shift all its positions by|C|, where|C|is the length of the
context. For selected layers, we concatenate the KV pairs of the sender at positions[0,|C|), and the KV pairs
of the receiver follow at positions[|C|,|C|+|Q|). For non-selected layers, positions[0,|C|)are left empty
(unattended), but the KV of the receiver still starts at position|C|. This approach ensures that all layers share a
consistent positional frame, so the attention mechanism sees the same offsets at every depth, avoiding positional
drift across layers. We perform an ablation study to validate this design in Section M.
Communication CostUnder the scenario where agents are connected with high-bandwidth links, the com-
munication cost is relatively low compared to recomputation cost (Jin et al., 2024; Liu et al., 2024c). KVComm
is more preferred when the information exchange volume is large (e.g., long contexts) and the communication
bandwidth is sufficient. In scenarios with limited bandwidth, further compression of KV pairs or more aggres-
sive layer selection may be necessary, which we leave for future work.
L CONTEXT-ADAPTIVEONLINECALIBRATION
A simple yet effective dynamic selection mechanism is to recompute the selected layers everyTqueries, where
Tis a hyperparameter that can be dynamically determined by server workload. We make two fully mixed
23

<!-- page 24 -->

Published as a conference paper at ICLR 2026
Table 10: Communication results for one sender and two senders scenarios. Weboldthe better
result comparing one sender and two senders for each method.KVComm (0.3/0.5/0.7)denotes
selecting 30%/50%/70% of layers’ KV pairs for communication, i.e.,M=⌈0.3L⌉,M=⌈0.5L⌉,
M=⌈0.7L⌉.
Method Sender HotpotQA MuSiQuest 2WikiMQA
Ms1: arcee-ai/Llama-3.1-SuperNova-Lite;
Ms2: yuvraj17/EvolCodeLlama-3.1-8B-Instruct;
Mr: Team-ACE/ToolACE-2-Llama-3.1-8B
Baseline NA 0.04 0.01 0.01
Skyline0.37 0.15 0.25
NLD Ms2 0.17 0.050.02
Ms1 andM s2 0.14 0.040.03
CIPHER Ms2 0.19 0.06 0.03
Ms1 andM s2 0.16 0.050.03
KVComm (0.3) Ms2 0.12 0.04 0.19
Ms1 andM s2 0.16 0.06 0.23
KVComm (0.5) Ms2 0.38 0.140.28
Ms1 andM s2 0.39 0.200.21
KVComm (0.7) Ms2 0.51 0.25 0.33
Ms1 andM s2 0.53 0.29 0.34
Ms1: cooperleong00/Qwen2.5-7B-Instruct-Jailbroken;
Ms2: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored;
Mr: bespokelabs/Bespoke-Stratos-7B
Baseline NA 0.13 0.03 0.09
Skyline0.53 0.25 0.09
NLD Ms2 0.16 0.040.02
Ms1 andM s2 0.18 0.06 0.02
CIPHER Ms2 0.03 0.03 0.03
Ms1 andM s2 0.02 0.01 0.01
KVComm (0.3) Ms2 0.02 0.01 0.08
Ms1 andM s2 0.020.00 0.05
KVComm (0.5) Ms2 0.28 0.120.10
Ms1 andM s2 0.32 0.180.08
KVComm (0.7) Ms2 0.410.250.15
Ms1 andM s2 0.500.240.24
Ms1: RedaAlami/Falcon3-7B-Instruct-Distill-DS-v1;
Ms2: ehristoforu/falcon3-ultraset;
Mr: huihui-ai/Falcon3-7B-Instruct-abliterated
Baseline NA 0.21 0.04 0.23
Skyline0.76 0.56 0.45
NLD Ms2 0.52 0.25 0.24
Ms1 andM s2 0.22 0.13 0.20
CIPHER Ms2 0.27 0.07 0.25
Ms1 andM s2 0.15 0.04 0.14
KVComm (0.3) Ms2 0.59 0.40 0.29
Ms1 andM s2 0.51 0.30 0.27
KVComm (0.5) Ms2 0.63 0.440.34
Ms1 andM s2 0.49 0.430.35
KVComm (0.7) Ms2 0.55 0.420.31
Ms1 andM s2 0.60 0.46 0.31
datasets, i.e., mixing all the samples from two datasets: Countries and Tipsheets; Countries and MultiFieldQA-
en. We then perform online calibration and evaluate with different calibration intervalsT. As shown in Fig-
ure 13, we find the performance drops whenTincreases, which is consistent with intuition.
Beyond periodic recomputation, more sophisticated adaptive mechanisms are also feasible. For example, the
receiver model could leverage lightweight signals, such as token-level entropy and attention sparsity patterns,
to trigger on-demand re-selection of informative layers. This is an exciting direction for future work, and
KVComm provides a clean foundation for such extensions.
Additionally, to illustrate how different the selected layers are for different datasets, we calculate the Kendall’s
Tau similarity of layer rankings for each pair of datasets across all models. As shown in Figure 14, some
tasks share quite a similar layer ranking for a given model pair, e.g., model pair index 6 shares a similar layer
ranking for HotpotQA and MuSiQuest datasets. This phenomenon could guide the design of dynamic selection
mechanisms in future work.
24

<!-- page 25 -->

Published as a conference paper at ICLR 2026
100 101 102 103 104
Calibration Interval T
0.2
0.4
0.6F1 Score
Countries_MultiFieldQA_en
Model Pair Index 6
Model Pair Index 7
Model Pair Index 8
100 101 102 103 104
Calibration Interval T
Countries_TipSheets
KVComm (0.3) Results on Mixed tasks with Online Calibration
Figure 13: Online calibration performance drops when the calibration interval increases.
TipsheetsCountriesHotpotQA
QASPER
MuSiQuest
MultiFieldQA-en
2WikiMQA
TMATH
Tipsheets
Countries
HotpotQA
QASPER
MuSiQuest
MultiFieldQA-en
2WikiMQA
TMATH
Model Pair Index 6
TipsheetsCountriesHotpotQA
QASPER
MuSiQuest
MultiFieldQA-en
2WikiMQA
TMATH
Model Pair Index 7
TipsheetsCountriesHotpotQA
QASPER
MuSiQuest
MultiFieldQA-en
2WikiMQA
TMATH
Model Pair Index 8
0.7
0.8
0.9
1.0
Kendall's T au Similarity
Layer Ranking Similarity Across Dataset
Figure 14: Kendall’s Tau similarity of layer rankings between different datasets.
M POSITIONALEMBEDDINGCOHERENCE
Table 11: Comparison of KVComm and KVComm-S. KVComm-S denotes shifting back the token
positions of non-selected layers to 0. We bold the best results between KVComm and KVComm-S
under the same settings.
MethodCountries Tipsheets HotpotQA QASPER MuSiQuest MultiField
-QA-en
2WikiM
-QA TMATH
Ms: huihui-ai/Llama-3.2-3B-Instruct-abliterated;M r: suayptalha/DeepSeek-R1-Distill-Llama-3B
KVComm-S (0.3)0.260.650.400.100.140.190.210.36
KVComm (0.3) 0.460.450.460.090.280.150.280.35
KVComm-S (0.5)0.49 0.740.57 0.28 0.320.45 0.300.35
KVComm (0.5) 0.57 0.81 0.570.270.32 0.51 0.36 0.35
KVComm-S (0.7)0.52 0.760.65 0.30 0.390.46 0.320.35
KVComm (0.7) 0.57 0.81 0.650.29 0.360.47 0.37 0.35
Ms: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored;M r: bespokelabs/Bespoke-Stratos-7B
KVComm-S (0.3)0.00 0.200.02 0.02 0.01 0.090.050.34
KVComm (0.3) 0.04 0.26 0.020.01 0.010.09 0.080.31
KVComm-S (0.5)0.040.90 0.33 0.13 0.18 0.35 0.16 0.35
KVComm (0.5) 0.190.88 0.28 0.07 0.12 0.26 0.10 0.33
KVComm-S (0.7)0.360.94 0.420.19 0.240.35 0.16 0.34
KVComm (0.7) 0.410.89 0.410.21 0.250.29 0.150.34
Ms: ehristoforu/falcon3-ultraset;M r: huihui-ai/Falcon3-7B-Instruct-abliterated
KVComm-S (0.3) 0.47 0.710.54 0.10 0.36 0.19 0.260.32
KVComm (0.3)0.46 0.690.59 0.19 0.40 0.35 0.29 0.32
KVComm-S (0.5)0.360.97 0.67 0.27 0.460.340.360.34
KVComm (0.5) 0.400.92 0.63 0.25 0.440.450.340.35
KVComm-S (0.7) 0.210.950.59 0.26 0.520.460.37 0.36
KVComm (0.7)0.190.960.550.260.420.510.310.36
We quantify the importance of positional embedding coherence between the sender and receiver models. We
perform an ablation experiment where, for non-selected layers, instead of shifting the receiver’s tokens to
25

<!-- page 26 -->

Published as a conference paper at ICLR 2026
position|C|, we place them back to position 0, creating a positional inconsistency with selected layers. As
shown in Table 11, positional inconsistency does not have a detrimental effect on performance, but overall, our
approach has merit.
N COMPLEXITYANALYSISDETAILS
We compare the computational complexity of our KVComm framework with the Skyline method and the NLD
method. Recall thatLis the total number of layers in the model,Mis the number of selected layers for
communication. We usedto denote the hidden dimension of the model, and|Q|and|C|to denote the number
of tokens in the query and context, respectively. SupposeM r would generateT r tokens in total, and the
number of generated tokens is the same across different methods. For NLD,M s andM r would each generate
Ts andT r tokens for the initial answer, respectively.
Ignoring the embedding, output layers, and other minor components, the computational complexity of prefilling
a sequence of lengthNwith a single decoder layer isO(N d 2 +N 2d), while the complexity of decoding a
single token isO(d 2 +(N+i)d), whereiis the index of the generated token. Therefore, the total computational
complexity ofM s to process the contextCisO(L(|C|d 2 +|C| 2d)).
The total computational complexity of KVComm consists of three parts: (1) the complexity ofM s to process
the contextC, which isO(L(|C|d 2 +|C| 2d)), (2) the complexity ofM r to process the queryQwith the
selectedMKV pairs fromM s, which isO(L|Q|d 2 +M(|C|+|Q|)|Q|d+ (L−M)|Q| 2d), and (3) the
complexity ofM r to generateT r tokens with the selectedMKV pairs fromM s, which isO(T r(Ld2 +
M(|C|+|Q|+T r)d+ (L−M)(|Q|+T r)d)). Therefore, the total computational complexity of KVComm
is:
T(KVComm) =O

L(|C|+|Q|+T r)d 2

+O
 
L
 
|C| 2 +|Q| 2 +T 2
r +T r|Q|

+CM(|Q|+T r)

d

The computational complexity of Skyline method consists of two parts: (1) the complexity of prefilling the
concatenation of the contextCand queryQ, which isO(L(|C|+|Q|)d 2 +L(|C|+|Q|) 2d), and (2) the
complexity of decodingT r tokens, which isO(T L(d 2+(|C|+|Q|+T r)d)). Therefore, the total computational
complexity of the Skyline method is:
T(Skyline) =O

L
 
|C|+|Q|+T r

d2

+O

L

(|C|+|Q|) 2 +T r
 
|C|+|Q|+T r

d

The margin of KVComm over Skyline is:
T(Skyline)− T(KVComm) =O

|C|d
 
L(2|Q|+T r)−M(|Q|+T r)

For NLD, the total computational complexity consists of three parts: (1) the complexity ofM s to process the
contextCand generateT s tokens, which isO(L(|C|d 2 +|C| 2d)+T sL(d2 +(|C|+T s)d)), (2) the complexity
ofM r to process the queryQand generateT r tokens, which isO(L(|Q|d 2 +|Q| 2d) +T rL(d2 + (|Q|+
Tr)d)), and (3) the complexity ofM r to process the entire debate history and generateT r tokens, which is
O(L((Ts +T r +|Q|)d 2 + (Ts +T r +|Q|) 2d) +T L(d 2 + (Ts +T r +|Q|+T r)d)). Therefore, the total
computational complexity of NLD is:
T(NLD) =O

L

|C|+ 2|Q|+ 2T s + 2Tr +T r

d2
!
+O

L

|C| 2 +T 2
s +|Q| 2 +T 2
r +
 
Ts +T r +|Q|
2
+T r
 
Ts +T r +T r +|Q|

+T s|C|+T r|Q|

d
!
26

<!-- page 27 -->

Published as a conference paper at ICLR 2026
The margin of KVComm over NLD is:
T(NLD)− T(KVComm) =O

L
 
2Ts + 2Tr +|Q|

d2

+O

L

T 2
s +T 2
r +
 
Ts +T r +|Q|
2
+T s|C|+T r|Q|+T r(Ts +T r)

−CM
 
|Q|+T r

!
d
!
O USINGONECHUNK OFLAYERS
We conduct the same experiment as in Section 4.3 on the HotpotQA dataset using other model pairs in Table 5.
The results are shown in Figure 15. We can see that the results are consistent with the observation in Section 4.3.
27

<!-- page 28 -->

Published as a conference paper at ICLR 2026
0 10 20
layerto
0
5
10
15
20
25layerfrom
F1 Score with One Contig. Chunk of Context
0.2
0.4
0.6
0 10 20
#Layers
0.2
0.4
0.6F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
0 5 10 15
layerfrom
0.2
0.3
0.4
0.5
0.6F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
(a) Llama-3.2-3B-Instruct as bothM s andM r
0 10 20
layerto
0
5
10
15
20
25layerfrom
F1 Score with One Contig. Chunk of Context
0.2
0.4
0.6
0 10 20
#Layers
0.0
0.2
0.4
0.6F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
0 5 10 15
layerfrom
0.0
0.1
0.2
0.3
0.4
0.5F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
(b) Qwen2.5-7B-Instruct as bothM s andM r
0 10 20
layerto
0
5
10
15
20
25layerfrom
F1 Score with One Contig. Chunk of Context
0.0
0.1
0.2
0.3
0.4
0.5
0 10 20
#Layers
0.0
0.2
0.4
0.6F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
0 5 10 15
layerfrom
0.0
0.1
0.2
0.3F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
(c) Qwen2.5-7B-Instruct-Uncensored asM s and Bespoke-Stratos-7B asM r
0 10 20 30
layerto
0
10
20
30layerfrom
F1 Score with One Contig. Chunk of Context
0.2
0.4
0.6
0 10 20 30
#Layers
0.0
0.2
0.4
0.6F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
0 5 10 15 20
layerfrom
0.05
0.10
0.15
0.20
0.25F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
(d) EvolCodeLlama-3.1-8B-Instruct asM s and ToolACE-2-Llama-3.1-8B asM r
0 10 20
layerto
0
5
10
15
20
25layerfrom
F1 Score with One Contig. Chunk of Context
0.2
0.3
0.4
0.5
0.6
0.7
0 10 20
#Layers
0.2
0.4
0.6F1 Score
F1 Score w.r.t. #Layers
Contig.
=1.0
=0.9
=0.8
=0.7
=0.6
=0.5
0 5 10 15
layerfrom
0.2
0.3
0.4
0.5
0.6F1 Score
Impact of Chunk Position under Fixed #Layers
#layers=10
#layers=16
(e) Llama-3.2-3B-Instruct-abliterated asM s and DeepSeek-R1-Distill-Llama-3B asM r
Figure 15: Experiment results of using one chunk of layers for communication on HotpotQA dataset
using different model pairs.
28
