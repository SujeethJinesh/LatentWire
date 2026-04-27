# references/04_communicating_activations_between_language_model_agents.pdf

<!-- page 1 -->

Communicating Activations Between Language Model Agents
Vignav Ramesh 1 Kenneth Li 1
Abstract
Communication between multiple language
model (LM) agents has been shown to scale up
the reasoning ability of LMs. While natural lan-
guage has been the dominant medium for inter-
LM communication, it is not obvious this should
be the standard: not only does natural language
communication incur high inference costs that
scale quickly with the number of both agents and
messages, but also the decoding process abstracts
away too much rich information that could be oth-
erwise accessed from the internal activations. In
this work, we propose a simple technique whereby
LMs communicate via activations; concretely, we
pause an LM B’s computation at an intermediate
layer, combine its current activation with another
LM A’s intermediate activation via some func-
tion f, then pass f’s output into the next layer
of B and continue the forward pass till decod-
ing is complete. This approach scales up LMs
on new tasks with zero additional parameters and
data, and saves a substantial amount of compute
over natural language communication. We test
our method with various functional forms f on
two experimental setups—multi-player coordina-
tion games and reasoning benchmarks—and find
that it achieves up to 27.0% improvement over
natural language communication across datasets
with <1/4 the compute, illustrating the superior-
ity and robustness of activations as an alternative
“language” for communication between LMs.
1. Introduction
Language is for the purpose of communication. As large
language models (LLMs) have been increasingly used to
power autonomous, goal-driven agents capable of reason-
ing, tool usage, and adaptive decision-making (Yao et al.,
1Kempner Institute for AI, Harvard University, Cam-
bridge, MA, USA. Correspondence to: Vignav Ramesh <vig-
navramesh@college.harvard.edu>.
Proceedings of the 42 nd International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).
2023; Xi et al., 2023; Wang et al., 2024; Ahn et al., 2022;
Schick et al., 2023; Shen et al., 2023; Park et al., 2023;
Nakano et al., 2022), communication between multiple co-
operating agents has emerged as an intuitive approach to
amplify the reasoning capabilities of LLMs (Wu et al., 2023).
Explicit communication in natural language between multi-
ple LLMs has been shown to encourage divergent thinking
(Liang et al., 2023), improve factuality and reasoning (Du
et al., 2023), enable integration of cross-domain knowledge
(Sukhbaatar et al., 2024), and allow for modular compo-
sition of abilities in a complementary manner (Wu et al.,
2023; Prasad et al., 2023).
A critical problem with natural language communication,
however, is that it incurs extremely high inference costs
that scale quickly with the number of agents as well as
length and number of messages (Du et al., 2023; Yang et al.,
2023; Wu et al., 2023). Restricting LLM communication
to natural language also raises the question: as LLMs are
increasingly capable of handling larger, more complex tasks
(sometimes with “super-human” ability) (Wei et al., 2022;
Burns et al., 2023), might they communicate more effec-
tively in representations of higher dimension than natural
language? While using natural language as a communica-
tive medium is appealing due to its interpretability, we claim
that it may not be optimal for inter-LLM communication.
Natural language generation uses only one token to repre-
sent the model’s belief over the entire vocabulary, which
risks losing information embedded within the model output
logits (Pham et al., 2024); furthermore, a model’s belief
over the entire vocabulary is itself not always better (for
communicative purposes) than the model’s (often richer)
representation of the input in earlier layers. Indeed, Her-
nandez et al. (2024) find that by around the halfway point
of an LM’s computation, it has developed “enriched entity
representations” of the input, where entities in the prompt
are populated with additional facts about that entity encoded
in the model’s weights; but by the later layers these embed-
dings are transformed into a representation of the next word
which leverages only parts of the previous, richer represen-
tations, when that full embedding would be quite useful for
communication.
Motivated by these concerns, this work outlines a simple
technique whereby LLM agents communicate via activa-
tions, thus enabling more efficient (i.e., higher-entropy) com-
1
arXiv:2501.14082v2  [cs.CL]  7 May 2025

<!-- page 2 -->

Communicating Activations Between Language Model Agents
munication at a fraction of the number of forward passes
required at inference time. Concretely, we (1) pause a Trans-
former LM B’s computation at intermediate layerj in the
residual stream; (2) combine its post-layer j activation with
another LM A’s post-layerk activation via some function f;
and then (3) pass f’s output into the next layerj + 1 of B
and continue its forward pass till decoding is complete. This
approach scales up LLMs on new tasks by leveraging exist-
ing, frozen LLMs along with zero task-specific parameters
and data, applying to diverse domains and settings. Further-
more, in requiring only a partial forward pass through A
and one forward pass through B, this method saves a sub-
stantial amount of compute over traditional natural language
communication, which we quantify in Section 3.2.
We validate our method by testing this approach with various
functional forms f on two experimental setups: two multi-
player coordination games, where B is asked to complete a
task requiring information provided in a prompt to A; and
seven reasoning benchmarks spanning multiple domains:
Biographies (Du et al., 2023), GSM8k (Cobbe et al., 2021),
MMLU High School Psychology, MMLU Formal Logic,
MMLU College Biology, MMLU Professional Law, and
MMLU Public Relations (Hendrycks et al., 2021). Our
activation communication protocol exhibits up to 27.0%
improvement over natural language communication across
these datasets, using <1/4 the compute. Critically, unlike
prior work which test inter-LLM communication only on
large-scale (>70B) models (Du et al., 2023; Liang et al.,
2023), we find that our approach generalizes across a wide
array of LLM suites and sizes, enabling even smaller LLMs
to unlock the benefits of communication.
In summary, our contributions are two-fold:
• We propose a novel inter-model communication proto-
col for LLM agents that is purely activation-based.
• We perform comprehensive experiments to validate the
improved performance of activation communication
over traditional natural language communication. We
also formally quantify our approach’s compute savings
over natural language communication, illustrating the
superiority and robustness of activations as an alterna-
tive “language” for communication between LMs.
2. Related Work
Multi-agent communication The field of multi-agent
communication has a long-standing history. Notably, prior
works on emergent communication have showed that agents
can autonomously evolve communication protocols when
deployed in multi-agent environments that enable cooper-
ative and competitive game-play (Sukhbaatar et al., 2016;
Foerster et al., 2016; Lazaridou et al., 2017). However, re-
cent experiments have demonstrated that learning meaning-
ful languages from scratch, even with centralized training,
remains difficult (Lowe et al., 2020; Chaabouni et al., 2019;
Jaques et al., 2019).
With the emergence of large pre-trained language models,
allowing communication between LLMs in natural language
has hence become a promising approach to enable coordina-
tion among multiple LLM agents (Li et al., 2023). Recent
works have demonstrated that such conversations enable
integration of cross-domain knowledge (Sukhbaatar et al.,
2024), modular composition of abilities in a complementary
manner (Wu et al., 2023), and improved task performance
via splitting into subtasks (Prasad et al., 2023). Most notable
is multiagent debate introduced by Du et al. (2023), where
LLMs provide initial responses and then make refinements
by iteratively considering inputs from peers. While such
methods have been shown to improve performance on vari-
ous tasks over vanilla and majority-vote (Wang et al., 2023)
style prompting, these experiments have only focused on
large models (GPT-3.5/4, LLaMA2-70B and up), leaving the
efficacy of debate on smaller, open-source models underex-
plored; our study addresses this gap by reimplementing Du
et al. (2023) in experiments with smaller-scale ( 1 − 70B)
models. More crucially, debate and similar natural language
communication methods are extremely computationally ex-
pensive, which this work addresses (Yang et al., 2023; Wu
et al., 2023).
Notably, Pham et al. (2024) propose CIPHER, which uses
input (tokenizer) embeddings (as opposed to activations) to
enable multi-agent communication; specifically, CIPHER
passes the average tokenizer embedding (weighted by the
LLM’s next-token probabilities) between models. While
(Pham et al., 2024) show this approach outperforms natural
language debate, it (i) still faces substantial information loss
relative to the model activations and (ii) does not save com-
pute, as the number of these “average embeddings” passed
between models is the same as the number of tokens passed
between models in natural language communication.
A related class of methods involves spending extra test-time
compute reasoning in latent space (Geiping et al., 2025;
Hao et al., 2024). Such latent reasoning approaches in-
volving doing ”chain-of-thought in activation space,” e.g.
by grafting LM activations into other layers/later forward
passes through the same model (e.g., a form of “recurrent
AC” within a single model); our approach can be viewed
as doing exactly the same thing, but instead ”outsourcing”
the CoT to another model (and thus reaping benefits from
greater diversity of thoughts/reasoning paths from distinct
models).
Activation engineering Activation engineering involves
editing an LLM’s intermediate layer representations during
a forward pass to create desired changes to output text (Li
2

<!-- page 3 -->

Communicating Activations Between Language Model Agents
et al., 2024; Turner et al., 2023). Past work has explored
extracting latent steering vectors from a frozen LLM to con-
trol quality and content of completions (Subramani et al.,
2022), as well as using “direction” vectors (computed as
the difference in activations between two prompts) that en-
able inference-time control over high-level properties of
generations (Li et al., 2024; Turner et al., 2023). This work
involves activation editing that is similar to such prior works
at a high level, though for the purpose of communication
between LLM agents.
Model composition and grafting Composing expert
models has been a recurring strategy to improve large mod-
els, with different methods imposing different restrictions
on the types of base LLMs that can be combined. Mixture
of Experts (Shazeer et al., 2017) requires that all experts
are trained simultaneously using the same data; Branch-
Train-Mix (Sukhbaatar et al., 2024) trains a single base LM
multiple times on different datasets, then learns a router on
outputs. Crucially, these methods do not work when nei-
ther model can do the task at hand well (i.e., they solve the
problem of choosing which of several outputs is best, not
that of generating a high-quality output by recombining the
disparate abilities of the various base LMs).
Model grafting, in contrast, seeks to merge different mod-
els immediately prior to or at inference-time. Past works
have explored this at the parameter level (e.g., task vector
averaging as in Ilharco et al. (2023), which requires that
the base models be well aligned), probability distribution
/ token level as in Shen et al. (2024) (which imposes few
restrictions on the relationship between the base models,
but by virtue of being token-based can result in cascading
errors during decoding), and activation level (e.g., CALM
(Bansal et al., 2024) which learns an attention layer on top
of two models’ intermediate layer activations and thus en-
ables broader integration of model abilities than token-level
methods, but requires re-tuning of the attention mechanism
for every model pair). In this work, we seek to unify CALM
and other activation-level grafting techniques under a single
framework, parameterized by the function f used to com-
bine activations; crucially, we explore simple forms of f
(e.g., sum, mean) that—unlike Bansal et al. (2024)—require
zero additional task-specific parameters and data, and are
far more compute-efficient.
3. Communicating Activations Between
Language Models
We propose a simple yet effective technique whereby lan-
guage models communicate via activations. We detail our
approach in Section 3.1; provide analytical models of the
compute saved over natural language communication in Sec-
tion 3.2; and discuss the intuition behind this approach in
Section 3.3.
3.1. Method
Consider two language models, A and B, and some setting
in which B must perform a task where it would benefit from
knowledge given to A as a prompt/encoded in A’s weights
(example settings in Section 4.1/Section 4.2 respectively).
We propose incorporating information from A’s post-layer
k activation hA,k into B’s post-layerj activation hB,j (and
vice versa, though for simplicity we henceforth only discuss
the first direction) (Figure 1, left).
More formally, suppose A and B (which have model di-
mensions dA and dB respectively) are given prompts xA
and xB respectively, where xA is of length tA tokens and
xB is of length tB tokens. We first run a partial forward
pass of B until layer j (henceforth denoted B≤j(xB)) to
get hB,j ∈ RtB ×dB. Then we (1) run a partial forward pass
of A until layer k to get A≤k(x1) := hA,k ∈ RtA×dA; (2)
replace the activation of the last token(hB,j )tB ∈ RdB ← −
f ((hA,k)tA , (hB,j )tB ) for some function f : RdA+dB →
RdB; then (3) continue B’s forward pass till decoding is
complete, resulting in an output y = B>k(hB,j ).
Let a = ( hA,k)tA, b = ( hB,j )tB. For sake of simplicity
assume dA = dB.1 We consider three non-learned functions
f:
f (a, b) = a + b (sum)
f (a, b) = 1
2 (a + b) ( mean)
f (a, b) = a (replace)
For cases where, due to differences in A and B’s training,
A and B’s activation spaces are quite different, we propose
learning a task-agnostic (depends only on the models A and
B) linear layer W ∈ RdB × RdA that projects a onto B’s
activation space. Note that this introduces zero additional
task-specific parameters and data, as we propose learning
this “mapping matrix” W only once for each model pair
(A, B) using general text, e.g. sequences from A and/or B’s
pretraining data mixes. We can then perform sum, mean, or
replace with W a, b instead of a, b. We propose training
W to minimize MSE loss over a dataset of N sentences
LMSE

{y(i)}N
i=1, {z(i)}N
i=1

= 1
N
NX
i=1



z(i) − W y(i)



2
2
1When dA ̸= dB, the sum, mean, and replace functions
are defined as follows. Let d = min( dA, dB) and ◦ the
concatenation operator. Then f (a, b) = b1:max(dB −d,0) ◦ 
bmax(dB −d,0)+1:dB + amax(dA−d,0)+1:dA

(sum), f (a, b) =
b1:max(dB −d,0) ◦ 1
2
 
bmax(dB −d,0)+1:dB + amax(dA−d,0)+1:dA

(mean), and f (a, b) = b1:max(dB −d,0) ◦ amax(dA−d,0)+1:dA
(replace).
3

<!-- page 4 -->

Communicating Activations Between Language Model Agents
Figure 1. Overview of activation communication. (Left) Our method involves (1) pausing a Transformer LM B ’s computation at layer
j in the residual stream; (2) combining its post-layer j activation with another LM A ’s post-layerk activation via some function f ;
then (3) passing f ’s output into the next layer j + 1 of B and continuing the forward pass till decoding is complete. (Right) Any
function f can be used to combine A and B’s activations; we explore letting f be the sum, mean, and replacement functions, as well
as a task-agnostic learned linear layer (details in Section 3.1).
where each (y(i), z(i)) pair denotes the final-token layer-26
activations of A and B at layers k and j respectively given
the same sentence as input.
3.2. Compute Analysis
To understand the significance of activation communication,
we must formally quantify the compute this procedure saves
over natural language communication. For simplicity sup-
pose the following (similar calculations can be made for the
cases where A and B have differing model architectures
and/or are given different prompts):
• A and B both have L layers (each with H attention
heads, key size K, and feedforward size F ), dimension
D, and vocab size V
• A and B are both given a prompt of P tokens
• A can send B a single M-token message
• B must produce an output ofT tokens, given its prompt
and A’s message
Traditional methods require M forward passes of A given a
P -length input, plusT forward passes ofB given a (P +M )-
length input. Following Hoffmann et al. (2022), this requires
M
 
4P V D + L(8P DKH + 4P 2KH + 3HP 2
+ 4P DF )

+ T
 
4(P + M )V D + L(8(P + M )DKH
+ 4(P + M )2KH + 3H(P + M )2 + 4(P + M )DF )

(1)
FLOPs. In contrast, at inference time, our method requires
only 1 partial (up till the kth layer) forward pass of A given
a P -length input, T forward passes of B given a P -length
input, and the activation replacement procedure. This re-
quires
2P V D + k(8P DKH + 4P 2KH + 3HP 2
+ 4P DF ) + T
 
4P V D + L(8P DKH + 4P 2KH
+ 3HP 2 + 4P DF )

+ F (D)
(2)
FLOPs, where F (D) = O(D) for non-learned f and
O(D2) when f is the mapping matrix.
In all practical cases, (2) is substantially lower than (1).
3.3. Why should this work?
Recall that Pham et al. (2024) propose CIPHER—
communicating the average tokenizer embedding (weighted
by the LLM’s next-token probabilities) between models.
We build upon the intuition behind CIPHER, which goes
as follows: the token sampling process during decoding
risks substantial information loss from the model’s output
logits, and communicating a model’s weighted-average tok-
enizer embedding essentially entails communicating both
that model’s final answer and its belief in that answer (over
the entire vocabulary).
Communicating activations, then, can be thought of as com-
municating a strict superset of {next-token prediction, belief
over entire vocabulary}, as activations of late-enough lay-
ers essentially encode the model’s entire knowledge about
the provided context as well as its predicted completion
and confidence in that completion (see Figures 1 and 7 in
Hewitt & Manning (2019) and Hernandez et al. (2024),
4

<!-- page 5 -->

Communicating Activations Between Language Model Agents
respectively, which show that linear probes tasked with pre-
dicting certain output characteristics from a Transformer’s
intermediate layer embeddings of its input work poorly for
early layers, extremely well after around the halfway point
of computation, but then probe accuracy drops closer to
the final layers).2 Indeed, these curves of probe accuracy
by layer indicate that the final layers and LM head “throw
away” information not useful for next-token prediction that
very well could be useful for communicative purposes; this
is precisely why our proposed activation communication
technique is not an iterative approach (there is no notion of
“rounds” like in debate and CIPHER, which require an addi-
tional token budget to extract more and more information
out of the LM), as one activation grafting step from A to B
inherently communicates to B all of A’s knowledge/beliefs
about the prompt it was given. Moreover, the extra informa-
tion over the model’s next-token prediction and confidence
that is encoded in its activations is what makes activation
communication more performant than its natural language
counterpart, as we will see in Section 4.
4. Experiments
We test our method on two distinct experimental setups:
multi-player coordination games (Section 4.1) and reasoning
benchmarks (Section 4.2). Qualitative results are available
in Appendix A.
4.1. Multi-player coordination games
Drawing from existing literature on multi-agent communi-
cation, we design two Lewis signaling games (Lewis, 2008;
Lazaridou et al., 2016) to test the efficacy of activation com-
munication (example prompts and answers in Table 1):
1. Countries, where A is given as input a string of the
format “ [PERSON] is at the [LANDMARK]” and B is
asked “Which country is [PERSON] located in?”
2. Tip Sheets (inspired by Lewis et al. (2017)), where A
is given a simulated “tip sheet” andB is asked to make
an informed investment decision in accordance with
the information in the tip sheet.
We synthetically generate 100 (Countries) and 70 (Tip
Sheets) different prompts and answers of the same format
as the samples in Table 1, and report the proportion out of
those samples that B responds with an exact string match to
the ground truth answer. As baselines, we consider a “silent”
2Note one important critique of multiagent debate: that in cases
where multiple agents are uncertain about the answer, there is no
reason why referencing other agents’ answers would generate more
factual reasoning. Both CIPHER and activation communication
solve this problem, as some notion of model confidence is being
communicated along with its next-token prediction.
(✗) setup, where the agents are not allowed to communicate;
a “single-agent skyline,” where a single LLM is given the
concatenation of A and B’s prompts; and traditional natural
language communication, where A is asked to output a
message that is then given toB along with xB. All decoding
is done greedily.
Table 2 presents the results for both coordination games
using 2 different instances of the same model as the agents
(A = B). Across the 3B and 8B model sizes, activation
communication (AC) with f = replace almost completely
recovers the gap between the zero-communication (✗) and
the single-agent skyline (SKYLINE ), outperforming natural
language communication (NL) using far less compute. We
hypothesize that replace is more effective than mean and
sum as the former is guaranteed to output a vector within
B’s activation space, while the latter two likely do not (e.g.,
the norm of the vector outputted by sum will be around
double that of a typical activation). Furthermore, most of
the information B needs is likely contained in its represen-
tations of previous tokens in the sequence, hence losing its
final-token representation does not hurt.
4.2. Reasoning Benchmarks
Next, we test our methods on a variety of reasoning bench-
marks, spanning several real-world tasks and domains.
Baselines We benchmark activation communication
against the following two baselines:
• Single Model: A single LLM responds to the prompt
in natural language.
• Natural Language Debate (NLD) (Du et al., 2023):
Each LLM provides an initial response to the given
prompt. Then, for each of r − 1 subsequent rounds,
each LLM is prompted to refine its previous response
given the other agents’ responses as input. Note that
NLD is the most direct baseline for our approach, as
it is a state-of-the-art natural language communication
protocol. We fix r = 2 in our experiments.
Note that we do not compare to Pham et al. (2024), as they
communicate the input (tokenizer) embeddings rather than
activations/output embeddings between models, and hence
require a shared tokenizer and embedding table between
agents which is extremely restrictive and prevents applica-
bility to our experimental setup.
To determine the values of k and j for activation commu-
nication (AC), we compute the accuracy on Countries and
Tip Sheets for every pair (k, j) ∈ { 1, . . . ,30}2. Based on
these results (shown in Figure 2) as well as Table 2, we fix
k = j = 26 and f = replace for the following experi-
ments.
5

<!-- page 6 -->

Communicating Activations Between Language Model Agents
Table 1: Multi-player coordination games. Sample (prompt, answer) pairs for each game.
Game Sample Prompts & Ground-Truth Answer
Countries
xA: “Alice is at the Acropolis of Athens .”
xB: “Which country is Alice located in ?”
B’s Expected Answer: “Greece”
Tip Sheets
xA: “Acme Inc. has taken a nosedive , as its quarterly earnings have dipped 8 %.
Meanwhile Doe LLC and Kiteflyer Labs have both reached record -high stock
prices of 89 , but Kiteflyer is involved in an IP lawsuit with its competitors .′′
xB: “You must invest in one company out of {Acme Inc., Doe LLC, Kiteflyer Labs}.
Which do you invest in ?”
B’s Expected Answer:“Doe LLC”
Table 2: Accuracies (%) on both coordination games using two identical LLaMA family models. Communication at layer
k = j = 26. 95% confidence intervals (1000 bootstrap iterations) reported in parentheses.
Model Method Accuracy (Countries) Accuracy (Tip Sheets)
LLaMA-3.2-3B
✗ 0.0 (0.0, 0.0) 38.6 (38.6, 39.4)
SKYLINE 84.0 (83.5, 84.1) 100.0 (100.0, 100.0)
NL 69.0 (68.7, 69.3) 74.3 (74.0, 74.6)
AC (sum) 34.0 (33.9, 34.4) 50.0 (49.6, 50.3)
AC (mean) 36.0 (35.5, 36.1) 80.0 (79.8, 80.4)
AC (replace) 78.0 (77.7, 78.2) 90.0 (89.9, 90.3)
LLaMA-3.1-8B
✗ 2.0 (1.9, 2.1) 54.3 (54.2, 54.5)
SKYLINE 86.0 (85.7, 86.1) 100.0 (100.0, 100.0)
NL 77.0 (76.6, 77.1) 85.7 (85.3, 85.8)
AC (sum) 71.0 (70.9, 71.4) 85.7 (85.5, 86.0)
AC (mean) 70.0 (69.7, 70.3) 92.9 (92.7, 93.1)
AC (replace) 83.0 (82.7, 83.1) 95.7 (95.6, 95.9)
Across all experiment configurations, we fix the decoding
strategy to nucleus sampling with p = 0.9.
Models We conduct most of our experiments usingLLaMA-
3.2-3B and LLaMA-3.1-8B as the two agents. Additionally,
to test our approach’s robustness and generalizability, we
conduct experiments with models belonging to various other
suites within the LLaMA family and of several different sizes.
Note that for these experiments, we restrict the setting to
communication between different models (rather than multi-
ple instances of the same model in Section 4.1), since the
same model would have identical activations for the same
prompts, meaning no information would be communicated
in the grafting process. We argue that the multiple-model
setting is realistic (perhaps more so than the setting of mul-
tiple instances of the same model), as recent advances in
LLM development have led to the release of models with
specialized abilities (Singhal et al., 2023) and of different
sizes (Dubey et al., 2024) that merit complementary usage.
Our work thus answers the question: How can we get the
best performance by leveraging multiple models of distinct
capabilities and sizes, relative to the added inference-time
compute over a single forward pass through any single
model?
Datasets We evaluate our technique on seven reasoning
datasets that span various real-world tasks and domains:
(i) Biographies (Du et al., 2023), which asks the LLM to
generate a factual biography of a famous computer scientist;
(ii) GSM8k (Cobbe et al., 2021), a variety of grade school
math problems created by human problem writers; and (iii)
5 datasets randomly drawn from MMLU (Hendrycks et al.,
2021): High School Psychology (from the Social Sciences
category), Formal Logic (from the Humanities category),
College Biology (from the STEM category), Professional
Law (from the Humanities Category), and Public Rela-
tions (from the Social Sciences category). We evaluate on a
randomly-sampled size-100 subset of each dataset.
In experiments involving the mapping matrix W , we in-
stantiate W ∈ R4096×3072 using Xavier initialization and
6

<!-- page 7 -->

Communicating Activations Between Language Model Agents
Figure 2. 2D contour plots of accuracy over different values of k and j (the layers at which we access/edit activations for A/B
respectively). k = j = 26 is roughly optimal ( ) for both (a) Countries and (b) Tip Sheets.
train for 10 epochs on a dataset of 3072 sentences3 ran-
domly drawn from the Colossal Clean Crawled Corpus (C4)
(Dodge et al., 2021). We use batch size 32 and the Adam
optimizer with learning rate 0.001.
Metrics We measure the accuracy of the final response
for the single models and AC. For NLD, we measure the
accuracy of the majority-held final-round answer across
agents when the answer is automatically verifiable (numeric
in GSM8k, multiple choice for the MMLU datasets) or the
average final-round answer across agents otherwise (Biogra-
phies).
For GSM8k and the MMLU datasets, we report the pro-
portion of samples in the dataset for which the generated
answer exactly matches the ground-truth answer. For Bi-
ographies, following Du et al. (2023), we prompt an LLM
judge ( LLaMA-3.1-8B) to check whether each manually-
decomposed fact in a ground-truth biography is supported
(1), partially supported (0.5), or unsupported (0) in the gen-
erated biography, taking the mean of these scores over all
facts as the per-biography accuracy and the mean over all
dataset samples as the total accuracy.
Comprehensive evaluation with the LLaMA family Ta-
ble 3 presents results on each of the seven reasoning bench-
marks across various baselines and activation communica-
tion. Notably, while NLD consistently outperforms LLaMA-
3.2-3B, it does not always display a performance improve-
ment over LLaMA-3.1-8B; but remarkably, ACconsistently
3We use 3072 sentences as linear regression with d-
dimensional input has a sample complexity of O(d) (Vapnik,
1999).
outperforms both single-model baselines. In fact, AC offers
an up to 27.0% improvement over NLD across six of the
seven reasoning datasets. When applying W to A’s acti-
vation before performing the replacement function, we see
even further gains of 2.6 − 50.0% over vanilla AC for four
of the seven datasets. We hypothesize that the benefits from
the learned linear layer are less consistent across datasets be-
cause the subset of C4 data used to train W likely contains
text more semantically similar to some datasets than others,
hence some datasets provide W with out-of-distribution
inputs which reduces performance compared to vanilla AC.
While we fix A as the smaller model and B as the larger
model in Table 3 (so as to ensure decoding happens with
the presumably more capable model), this need not be the
case; swapping A and B yields results of 81.5 ± 0.0 and
61.0±4.8 on Biographies and GSM8k respectively (without
the linear layer). While these accuracies are lower than
their non-swapped counterparts, notably they still are higher
than both single-model baselines (and higher than NLD for
Biographies); plus this is much more compute-efficient as
the smaller model is now the one requiring the full instead
of partial forward pass.
Note that we find AC outperforms NLD on 48 of the 57
datasets in the full MMLU benchmark; complete MMLU
results, as well as a suite of additional experiments, are
shown in Appendix B.
Performance-compute tradeoff and generalization to dif-
ferent model scales Thus far, we have been considering
the absolute performance of AC with respect to NLD, for
which our method attains state-of-the-art results; however
the superiority of activations as a language for inter-LLM
7

<!-- page 8 -->

Communicating Activations Between Language Model Agents
Table 3: Accuracies (%) on all seven reasoning benchmarks. NLD and all AC variants involve communication between
LLaMA-3.2-3B (A) and LLaMA-3.1-8B (B); the performance of these models individually are presented in the first two rows
of the table. NLD typically improves performance over at least one of the single model baselines; AC— both with and
without the task-agnostic linear layer—consistently beats both baselines and NLD as well.
Method Biog. GSM8k HS Psych. Logic Col. Bio. Prof. Law Pub. Rel.
3.2-3B 79.4±0.0 58.0±4.9 30.0±1.0 16.0±0.8 11.0±0.7 0.0±0.0 26.0±0.1
3.1-8B 83.9±0.0 60.0±4.9 65.0±0.1 42.0±0.1 50.0±0.2 20.0±0.8 53.0±0.2
NLD 80.2±0.1 75.0±4.3 83.0±0.8 37.0±0.1 71.0±0.1 30.0±0.1 63.0±0.7
AC 84.6±0.0 64.0±4.8 85.0±0.8 47.0±0.1 78.0±0.9 30.0±0.1 74.0±0.1
AC (W ) 86.8±0.0 66.0±4.8 70.0±0.1 35.0±0.1 79.0±0.9 45.0±0.1 63.0±0.1
communication is further illustrated by AC’s largerratio of
performance improvement to added inference-time compute
over individual LMs. Figure 3 displays the results of single
models, AC, and NLD across model scales and suites within
the LLaMA family on the Biographies dataset. Incoming
arrows to AC and NLD nodes denote the base models be-
tween which communication occurred. Not only does AC
consistently outperform both single-model baselines unlike
NLD, but also notice that the slope of each black line is
far greater than the slope of each gray line, indicating that
AC consistently achieves greater increases in accuracy per
additional unit of inference-time compute (normalized by
the compute of a single forward pass through LLaMA-3.2-1B
on the given prompt) compared to NLD.
Communication across model families Table 4 displays
results for AC between models from theQwen-2.5, Gemma-2,
and LLaMA-3 families. We see that AC beats NLD across
the board, and beats both individual models for 4/5 of
the 6 model pairs on Biographies/GSM8k respectively—
demonstrating the efficacy of AC irrespective of model ar-
chitecture, size, tokenizer, and training data. Moreover,
these results are obtained without training W , meaning we
do not need a separate projection layer between activation
spaces to attain SOTA results, even for extremely distinct
models! (We hypothesize this is because we are only re-
placing B’s last-token activation, henceB can learn from A
without an extreme alteration to its activation distribution.
An alternative explanation is to see this result as proof of the
platonic representation hypothesis (Huh et al., 2024), which
historical deep learning works have oft alluded to, includ-
ing in the context of cross-model representation stitching
(Moschella et al., 2023; Kornblith et al., 2019).)
5. Conclusion
We present a simple approach to enable effective and compu-
tationally efficient communication between language mod-
els by injecting information from the activations of one
model into the activations of another during the forward
pass. Salient features of this approach include: (i) Scales up
Figure 3. Accuracy (%) vs. compute (# FLOPs normalized by
single LLaMA-3.2-1B forward pass) for various configurations
of AC and NLD on the Biographies dataset. AC ( ) yields the
greatest performance gains per additional unit of inference-time
compute over each baseline ( ).
LLMs on new tasks by leveraging existing, frozen LLMs
along with zero additional task-specific parameters and data,
(ii) Applies to diverse domains and settings, and (iii) Saves
a substantial amount of compute.
There are some limitations to this method. First, when
not using the learned model-specific mapping discussed
in Section 3.1, our method requires both models to have
aligned embedding spaces, such that the activation of one
model roughly retains its meaning in the other’s activation
space (note that unlike past works such as Pham et al. (2024)
we do not require shared tokenizers or aligned vocabularies,
only aligned embeddings). While less restrictive than past
works (Pham et al., 2024), this assumption is somewhat
limiting, but can be relaxed when we let f be the learned
model-specific mapping; and in practice we find that even
amongst different models in the LLaMA family, no such
mapping is required for state-of-the-art results.
Second, this method requires access to embeddings and will
8

<!-- page 9 -->

Communicating Activations Between Language Model Agents
Table 4: Individual model, AC, and NLD accuracies across three model families. Each cell displays two values:
Biographies score / GSM8k score.
Model Pair (A, B) A B NLD AC
LLaMA-3.2-3B, LLaMA-3.1-8B 79.4±0.0 / 58.0±4.9 83.9±0.0 / 60.0±4.9 80.2±0.1 / 75.0±4.3 84.6±0.0 / 64.0±4.8
Qwen-2.5-1.5B, Qwen-2.5-3B 59.4±0.9 / 20.0±0.9 85.5±1.1 / 35.0±1.1 63.2±1.1 / 65.0±1.1 89.6±1.0 / 70.0±1.0
Gemma-2-2B, Gemma-2-9B 83.0±1.1 / 45.0±1.1 94.6±0.9 / 80.0±0.9 70.3±1.0 / 70.0±1.0 88.1±0.7 / 90.0±0.7
Qwen-2.5-1.5B, LLaMA-3.2-3B 59.4±0.9 / 20.0±0.9 79.4±0.0 / 58.0±4.9 75.4±1.0 / 75.0±1.0 79.5±1.0 / 75.0±1.0
LLaMA-3.2-3B, Gemma-2-2B 79.4±0.0 / 58.0±4.9 83.0±1.1 / 45.0±1.1 62.5±1.1 / 55.0±1.1 84.0±0.1 / 60.0±1.1
Qwen-2.5-1.5B, Gemma-2-2B 59.4±0.9 / 20.0±0.9 83.0±1.1 / 45.0±1.1 49.3±1.1 / 50.0±1.1 73.0±1.1 / 55.0±1.1
not work with black-box API access; however exploring
API-only approaches is highly limiting, and recent releases
of powerful open-source models (Dubey et al., 2024) merit
the development of embedding-based techniques.
Third, while a concern might be the limited interpretabil-
ity of communicating activations as opposed to natural
language, we note the following. First, there is a funda-
mental tradeoff between interpretability and information
preservation (as activations, by virtue of being much higher-
dimensional than the space of natural language, allow pro-
portionally higher-entropy communication) (Pham et al.,
2024), which merits discussion beyond the scope of this
work. But second, we actually posit that our method sug-
gests a new avenue towards interpreting LM activations:
“translating” activations based on the beliefs they induce
as messages in listening agents, similar to the method put
forward in Andreas et al. (2018). We recognize this as a
promising avenue for future research.
Additional directions of future work include using AC to
allow large LMs to leverage small, tunable LMs as “knowl-
edge bases” during decoding (Lee et al., 2024), as in col-
laborative decoding (Shen et al., 2024) setups; and testing
our approach on more complex coordination games (e.g.,
Lewis-style negotiation games (Lewis et al., 2017), Diplo-
macy).
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
Acknowledgements
The authors are grateful to Jacob Andreas, Yoon Kim, and
Sham Kakade for their valuable discussions and feedback.
References
Ahn, M., Brohan, A., Brown, N., Chebotar, Y ., Cortes, O.,
David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman,
K., Herzog, A., Ho, D., Hsu, J., Ibarz, J., Ichter, B.,
Irpan, A., Jang, E., Ruano, R. J., Jeffrey, K., Jesmonth,
S., Joshi, N. J., Julian, R., Kalashnikov, D., Kuang, Y .,
Lee, K.-H., Levine, S., Lu, Y ., Luu, L., Parada, C., Pastor,
P., Quiambao, J., Rao, K., Rettinghouse, J., Reyes, D.,
Sermanet, P., Sievers, N., Tan, C., Toshev, A., Vanhoucke,
V ., Xia, F., Xiao, T., Xu, P., Xu, S., Yan, M., and Zeng, A.
Do as i can, not as i say: Grounding language in robotic
affordances, 2022.
Andreas, J., Dragan, A., and Klein, D. Translating neuralese,
2018.
Bansal, R., Samanta, B., Dalmia, S., Gupta, N., Vashishth,
S., Ganapathy, S., Bapna, A., Jain, P., and Talukdar, P.
Llm augmented llms: Expanding capabilities through
composition, 2024.
Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao,
L., Aschenbrenner, L., Chen, Y ., Ecoffet, A., Joglekar,
M., Leike, J., Sutskever, I., and Wu, J. Weak-to-strong
generalization: Eliciting strong capabilities with weak
supervision, 2023.
Chaabouni, R., Kharitonov, E., Lazaric, A., Dupoux, E.,
and Baroni, M. Word-order biases in deep-agent emer-
gent communication. In Korhonen, A., Traum, D., and
M`arquez, L. (eds.), Proceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics, pp.
5166–5175, Florence, Italy, July 2019. Association for
Computational Linguistics. doi: 10.18653/v1/P19-1509.
URL https://aclanthology.org/P19-1509.
Cobbe, K., Kosaraju, V ., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., Hesse, C., and Schulman, J. Training verifiers to solve
math word problems, 2021.
Dodge, J., Sap, M., Marasovi ´c, A., Agnew, W., Ilharco,
G., Groeneveld, D., Mitchell, M., and Gardner, M. Doc-
umenting large webtext corpora: A case study on the
9

<!-- page 10 -->

Communicating Activations Between Language Model Agents
colossal clean crawled corpus, 2021. URL https:
//arxiv.org/abs/2104.08758.
Du, Y ., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch,
I. Improving factuality and reasoning in language models
through multiagent debate, 2023.
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A.,
Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A.,
Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravanku-
mar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A.,
Rodriguez, A., Gregerson, A., Spataru, A., Roziere, B.,
Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak,
C., Bi, C., Marra, C., McConnell, C., Keller, C., Touret,
C., Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Al-
lonsius, D., Song, D., Pintz, D., Livshits, D., Esiobu, D.,
Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino,
D., Hupkes, D., Lakomkin, E., AlBadawy, E., Lobanova,
E., Dinan, E., Smith, E. M., Radenovic, F., Zhang, F., Syn-
naeve, G., Lee, G., Anderson, G. L., Nail, G., Mialon, G.,
Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H.,
Touvron, H., Zarov, I., Ibarra, I. A., Kloumann, I., Misra,
I., Evtimov, I., Copet, J., Lee, J., Geffert, J., Vranes,
J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J.,
Billock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J.,
Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J.,
Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V .,
Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K.,
El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K.,
Rantala-Yeary, L., van der Maaten, L., Chen, L., Tan, L.,
Jenkins, L., Martin, L., Madaan, L., Malo, L., Blecher, L.,
Landzaat, L., de Oliveira, L., Muzzi, M., Pasupuleti, M.,
Singh, M., Paluri, M., Kardas, M., Oldham, M., Rita, M.,
Pavlova, M., Kambadur, M., Lewis, M., Si, M., Singh,
M. K., Hassan, M., Goyal, N., Torabi, N., Bashlykov, N.,
Bogoychev, N., Chatterji, N., Duchenne, O.,C ¸elebi, O.,
Alrassy, P., Zhang, P., Li, P., Vasic, P., Weng, P., Bhargava,
P., Dubal, P., Krishnan, P., Koura, P. S., Xu, P., He, Q.,
Dong, Q., Srinivasan, R., Ganapathy, R., Calderer, R.,
Cabral, R. S., Stojnic, R., Raileanu, R., Girdhar, R., Patel,
R., Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R.,
Silva, R., Hou, R., Wang, R., Hosseini, S., Chennabas-
appa, S., Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie,
S., Narang, S., Raparthy, S., Shen, S., Wan, S., Bhosale,
S., Zhang, S., Vandenhende, S., Batra, S., Whitman, S.,
Sootla, S., Collot, S., Gururangan, S., Borodinsky, S., Her-
man, T., Fowler, T., Sheasha, T., Georgiou, T., Scialom,
T., Speckbacher, T., Mihaylov, T., Xiao, T., Karn, U.,
Goswami, V ., Gupta, V ., Ramanathan, V ., Kerkez, V .,
Gonguet, V ., Do, V ., V ogeti, V ., Petrovic, V ., Chu, W.,
Xiong, W., Fu, W., Meers, W., Martinet, X., Wang, X.,
Tan, X. E., Xie, X., Jia, X., Wang, X., Goldschlag, Y .,
Gaur, Y ., Babaei, Y ., Wen, Y ., Song, Y ., Zhang, Y ., Li, Y .,
Mao, Y ., Coudert, Z. D., Yan, Z., Chen, Z., Papakipos, Z.,
Singh, A., Grattafiori, A., Jain, A., Kelsey, A., Shajnfeld,
A., Gangidi, A., Victoria, A., Goldstand, A., Menon, A.,
Sharma, A., Boesenberg, A., Vaughan, A., Baevski, A.,
Feinstein, A., Kallet, A., Sangani, A., Yunus, A., Lupu,
A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton,
A., Ryan, A., Ramchandani, A., Franco, A., Saraf, A.,
Chowdhury, A., Gabriel, A., Bharambe, A., Eisenman, A.,
Yazdan, A., James, B., Maurer, B., Leonhardi, B., Huang,
B., Loyd, B., Paola, B. D., Paranjape, B., Liu, B., Wu, B.,
Ni, B., Hancock, B., Wasti, B., Spence, B., Stojkovic, B.,
Gamido, B., Montalvo, B., Parker, C., Burton, C., Mejia,
C., Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H.,
Cai, C., Tindal, C., Feichtenhofer, C., Civin, D., Beaty,
D., Kreymer, D., Li, D., Wyatt, D., Adkins, D., Xu, D.,
Testuggine, D., David, D., Parikh, D., Liskovich, D., Foss,
D., Wang, D., Le, D., Holland, D., Dowling, E., Jamil,
E., Montgomery, E., Presani, E., Hahn, E., Wood, E.,
Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun,
F., Kreuk, F., Tian, F., Ozgenel, F., Caggioni, F., Guzm´an,
F., Kanayet, F., Seide, F., Florez, G. M., Schwarz, G.,
Badeer, G., Swee, G., Halpern, G., Thattai, G., Herman,
G., Sizov, G., Guangyi, Zhang, Lakshminarayanan, G.,
Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb,
H., Rudolph, H., Suk, H., Aspegren, H., Goldman, H.,
Damlaj, I., Molybog, I., Tufanov, I., Veliche, I.-E., Gat,
I., Weissman, J., Geboski, J., Kohli, J., Asher, J., Gaya,
J.-B., Marcus, J., Tang, J., Chan, J., Zhen, J., Reizenstein,
J., Teboul, J., Zhong, J., Jin, J., Yang, J., Cummings, J.,
Carvill, J., Shepard, J., McPhie, J., Torres, J., Ginsburg,
J., Wang, J., Wu, K., U, K. H., Saxena, K., Prasad, K.,
Khandelwal, K., Zand, K., Matosich, K., Veeraragha-
van, K., Michelena, K., Li, K., Huang, K., Chawla, K.,
Lakhotia, K., Huang, K., Chen, L., Garg, L., A, L., Silva,
L., Bell, L., Zhang, L., Guo, L., Yu, L., Moshkovich,
L., Wehrstedt, L., Khabsa, M., Avalani, M., Bhatt, M.,
Tsimpoukelli, M., Mankus, M., Hasson, M., Lennie, M.,
Reso, M., Groshev, M., Naumov, M., Lathi, M., Ke-
neally, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel,
M., Vyatskov, M., Samvelyan, M., Clark, M., Macey,
M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari,
M., Bansal, M., Santhanam, N., Parks, N., White, N.,
Bawa, N., Singhal, N., Egebo, N., Usunier, N., Laptev,
N. P., Dong, N., Zhang, N., Cheng, N., Chernoguz, O.,
Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P.,
Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P.,
Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P.,
Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy,
R., Nayani, R., Mitra, R., Li, R., Hogan, R., Battey, R.,
Wang, R., Maheswari, R., Howes, R., Rinott, R., Bondu,
S. J., Datta, S., Chugh, S., Hunt, S., Dhillon, S., Sidorov,
S., Pan, S., Verma, S., Yamamoto, S., Ramaswamy, S.,
Lindsay, S., Lindsay, S., Feng, S., Lin, S., Zha, S. C.,
Shankar, S., Zhang, S., Zhang, S., Wang, S., Agarwal,
S., Sajuyigbe, S., Chintala, S., Max, S., Chen, S., Kehoe,
S., Satterfield, S., Govindaprasad, S., Gupta, S., Cho,
10

<!-- page 11 -->

Communicating Activations Between Language Model Agents
S., Virk, S., Subramanian, S., Choudhury, S., Goldman,
S., Remez, T., Glaser, T., Best, T., Kohler, T., Robinson,
T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked,
T., V ontimitta, V ., Ajayi, V ., Montanez, V ., Mohan, V .,
Kumar, V . S., Mangla, V ., Albiero, V ., Ionescu, V ., Poe-
naru, V ., Mihailescu, V . T., Ivanov, V ., Li, W., Wang, W.,
Jiang, W., Bouaziz, W., Constable, W., Tang, X., Wang,
X., Wu, X., Wang, X., Xia, X., Wu, X., Gao, X., Chen,
Y ., Hu, Y ., Jia, Y ., Qi, Y ., Li, Y ., Zhang, Y ., Zhang, Y .,
Adi, Y ., Nam, Y ., Yu, Wang, Hao, Y ., Qian, Y ., He, Y .,
Rait, Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z.,
and Zhao, Z. The llama 3 herd of models, 2024. URL
https://arxiv.org/abs/2407.21783.
Foerster, J. N., Assael, Y . M., de Freitas, N., and White-
son, S. Learning to communicate with deep multi-agent
reinforcement learning, 2016.
Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh,
S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., and
Goldstein, T. Scaling up test-time compute with latent
reasoning: A recurrent depth approach, 2025. URL
https://arxiv.org/abs/2502.05171.
Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J.,
and Tian, Y . Training large language models to reason
in a continuous latent space, 2024. URL https://
arxiv.org/abs/2412.06769.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
M., Song, D., and Steinhardt, J. Measuring massive
multitask language understanding, 2021. URL https:
//arxiv.org/abs/2009.03300.
Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Watten-
berg, M., Andreas, J., Belinkov, Y ., and Bau, D. Linear-
ity of relation decoding in transformer language models,
2024.
Hewitt, J. and Manning, C. D. A structural probe for finding
syntax in word representations. In Burstein, J., Doran, C.,
and Solorio, T. (eds.), Proceedings of the 2019 Confer-
ence of the North American Chapter of the Association for
Computational Linguistics: Human Language Technolo-
gies, Volume 1 (Long and Short Papers), pp. 4129–4138,
Minneapolis, Minnesota, June 2019. Association for
Computational Linguistics. doi: 10.18653/v1/N19-1419.
URL https://aclanthology.org/N19-1419.
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., de Las Casas, D., Hendricks,
L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E.,
Millican, K., van den Driessche, G., Damoc, B., Guy,
A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W.,
Vinyals, O., and Sifre, L. Training compute-optimal large
language models, 2022.
Huh, M., Cheung, B., Wang, T., and Isola, P. The pla-
tonic representation hypothesis, 2024. URL https:
//arxiv.org/abs/2405.07987.
Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan,
S., Schmidt, L., Hajishirzi, H., and Farhadi, A. Editing
models with task arithmetic, 2023.
Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega,
P. A., Strouse, D., Leibo, J. Z., and de Freitas, N. Social
influence as intrinsic motivation for multi-agent deep
reinforcement learning, 2019.
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Simi-
larity of neural network representations revisited, 2019.
URL https://arxiv.org/abs/1905.00414.
Lazaridou, A., Peysakhovich, A., and Baroni, M. Multi-
agent cooperation and the emergence of (natural) lan-
guage. arXiv preprint arXiv:1612.07182, 2016.
Lazaridou, A., Peysakhovich, A., and Baroni, M. Multi-
agent cooperation and the emergence of (natural) lan-
guage, 2017.
Lee, J., Yang, F., Tran, T., Hu, Q., Barut, E., Chang, K.-
W., and Su, C. Can small language models help large
language models reason better?: Lm-guided chain-of-
thought, 2024. URL https://arxiv.org/abs/
2404.03414.
Lewis, D. Convention: A philosophical study. John Wiley
& Sons, 2008.
Lewis, M., Yarats, D., Dauphin, Y . N., Parikh, D., and Batra,
D. Deal or no deal? end-to-end learning for negotiation
dialogues, 2017.
Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and
Ghanem, B. Camel: Communicative agents for ”mind”
exploration of large language model society, 2023. URL
https://arxiv.org/abs/2303.17760.
Li, K., Patel, O., Vi´egas, F., Pfister, H., and Wattenberg, M.
Inference-time intervention: Eliciting truthful answers
from a language model. Advances in Neural Information
Processing Systems, 36, 2024.
Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y ., Wang,
R., Yang, Y ., Tu, Z., and Shi, S. Encouraging divergent
thinking in large language models through multi-agent
debate, 2023.
Lowe, R., Wu, Y ., Tamar, A., Harb, J., Abbeel, P., and Mor-
datch, I. Multi-agent actor-critic for mixed cooperative-
competitive environments, 2020.
11

<!-- page 12 -->

Communicating Activations Between Language Model Agents
Moschella, L., Maiorca, V ., Fumero, M., Norelli, A., Lo-
catello, F., and Rodol`a, E. Relative representations en-
able zero-shot latent space communication, 2023. URL
https://arxiv.org/abs/2209.15430.
Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L.,
Kim, C., Hesse, C., Jain, S., Kosaraju, V ., Saunders, W.,
Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button,
K., Knight, M., Chess, B., and Schulman, J. Webgpt:
Browser-assisted question-answering with human feed-
back, 2022.
Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang,
P., and Bernstein, M. S. Generative agents: Interactive
simulacra of human behavior, 2023.
Pham, C., Liu, B., Yang, Y ., Chen, Z., Liu, T., Yuan, J.,
Plummer, B. A., Wang, Z., and Yang, H. Let models
speak ciphers: Multiagent debate through embeddings,
2024.
Prasad, A., Koller, A., Hartmann, M., Clark, P., Sabharwal,
A., Bansal, M., and Khot, T. Adapt: As-needed decom-
position and planning with language models, 2023.
Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli,
M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Tool-
former: Language models can teach themselves to use
tools, 2023.
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
Q., Hinton, G., and Dean, J. Outrageously large neural
networks: The sparsely-gated mixture-of-experts layer,
2017.
Shen, S. Z., Lang, H., Wang, B., Kim, Y ., and Sontag,
D. Learning to decode collaboratively with multiple
language models, 2024.
Shen, Y ., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y .
Hugginggpt: Solving ai tasks with chatgpt and its friends
in hugging face, 2023.
Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E.,
Hou, L., Clark, K., Pfohl, S., Cole-Lewis, H., Neal, D.,
Schaekermann, M., Wang, A., Amin, M., Lachgar, S.,
Mansfield, P., Prakash, S., Green, B., Dominowska, E.,
y Arcas, B. A., Tomasev, N., Liu, Y ., Wong, R., Sem-
turs, C., Mahdavi, S. S., Barral, J., Webster, D., Cor-
rado, G. S., Matias, Y ., Azizi, S., Karthikesalingam, A.,
and Natarajan, V . Towards expert-level medical ques-
tion answering with large language models, 2023. URL
https://arxiv.org/abs/2305.09617.
Subramani, N., Suresh, N., and Peters, M. E. Extracting
latent steering vectors from pretrained language models,
2022.
Sukhbaatar, S., Szlam, A., and Fergus, R. Learning multia-
gent communication with backpropagation, 2016.
Sukhbaatar, S., Golovneva, O., Sharma, V ., Xu, H., Lin,
X. V ., Rozi`ere, B., Kahn, J., Li, D., tau Yih, W., Weston,
J., and Li, X. Branch-train-mix: Mixing expert llms into
a mixture-of-experts llm, 2024.
Turner, A. M., Thiergart, L., Udell, D., Leech, G., Mini,
U., and MacDiarmid, M. Activation addition: Steering
language models without optimization, 2023.
Vapnik, V . N. An overview of statistical learning theory.
IEEE transactions on neural networks, 10(5):988–999,
1999.
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J.,
Chen, Z., Tang, J., Chen, X., Lin, Y ., Zhao, W. X., Wei,
Z., and Wen, J.-R. A survey on large language model
based autonomous agents, 2024.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang,
S., Chowdhery, A., and Zhou, D. Self-consistency im-
proves chain of thought reasoning in language mod-
els, 2023. URL https://arxiv.org/abs/2203.
11171.
Wei, J., Tay, Y ., Bommasani, R., Raffel, C., Zoph, B.,
Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Met-
zler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang,
P., Dean, J., and Fedus, W. Emergent abilities of large
language models, 2022.
Wu, Q., Bansal, G., Zhang, J., Wu, Y ., Li, B., Zhu, E., Jiang,
L., Zhang, X., Zhang, S., Liu, J., Awadallah, A. H., White,
R. W., Burger, D., and Wang, C. Autogen: Enabling next-
gen llm applications via multi-agent conversation, 2023.
Xi, Z., Chen, W., Guo, X., He, W., Ding, Y ., Hong, B.,
Zhang, M., Wang, J., Jin, S., Zhou, E., Zheng, R., Fan,
X., Wang, X., Xiong, L., Zhou, Y ., Wang, W., Jiang, C.,
Zou, Y ., Liu, X., Yin, Z., Dou, S., Weng, R., Cheng, W.,
Zhang, Q., Qin, W., Zheng, Y ., Qiu, X., Huang, X., and
Gui, T. The rise and potential of large language model
based agents: A survey, 2023.
Yang, H., Yue, S., and He, Y . Auto-gpt for online decision
making: Benchmarks and additional opinions, 2023.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K., and Cao, Y . React: Synergizing reasoning and acting
in language models, 2023.
12

<!-- page 13 -->

Communicating Activations Between Language Model Agents
A. Qualitative Results
Figure 4. Example of AC on Biographies dataset.
13

<!-- page 14 -->

Communicating Activations Between Language Model Agents
Figure 5. Example of AC on GSM8k dataset.
14

<!-- page 15 -->

Communicating Activations Between Language Model Agents
Figure 6. Example of AC on MMLU High School Psychology dataset.
15

<!-- page 16 -->

Communicating Activations Between Language Model Agents
Figure 7. Example of AC on MMLU Formal Logic dataset.
16

<!-- page 17 -->

Communicating Activations Between Language Model Agents
Figure 8. Example of AC on MMLU College Biology dataset.
17

<!-- page 18 -->

Communicating Activations Between Language Model Agents
Figure 9. Example of AC on MMLU Professional Law dataset.
18

<!-- page 19 -->

Communicating Activations Between Language Model Agents
Figure 10. Example of AC on MMLU Public Relations dataset.
19

<!-- page 20 -->

Communicating Activations Between Language Model Agents
Table 5: Reasoning benchmark performance when varying tokens modified during AC. All methods involve communi-
cation between LLaMA-3.2-3B (A) and LLaMA-3.1-8B (B). The functional form f is varied between last-token replacement,
last-token summation, and summation for all tokens.
Method Biog. GSM8k HS Psych. Logic Col. Bio. Prof. Law Pub. Rel.
AC (replace) 84.6±0.0 64.0±4.8 85.0±0.8 47.0±0.1 78.0±0.9 30.0±0.1 74.0±0.1
AC (sum) 79.7±0.0 66.0±4.7 65.0±4.8 42.0±4.9 50.0±5.0 25.0±4.3 37.0±4.8
AC (all tokens) 76.0±0.0 62.0±4.9 35.0±4.8 42.0±4.9 61.0±4.9 15.0±3.6 26.0±4.4
Table 6: Reasoning benchmark performance when sampling from A with CoT. All methods involve communication
between LLaMA-3.2-3B (A) and LLaMA-3.1-8B (B).
Method Biog. GSM8k HS Psych. Logic Col. Bio. Prof. Law Pub. Rel.
AC 84.6±0.0 64.0±4.8 85.0±0.8 47.0±0.1 78.0±0.9 30.0±0.1 74.0±0.1
AC (W ) 86.8±0.0 66.0±4.8 70.0±0.1 35.0±0.1 79.0±0.9 45.0±0.1 63.0±0.1
AC (CoT) 82.1±0.0 66.0±4.0 80.0±4.0 26.0±4.4 67.0±4.7 40.0±4.9 63.0±4.8
B. Additional Experiments
B.1. Modifying Activations of All Tokens
Recall that AC grafts the last-token layer-k activation of A into B’s last-token layer-j activation. But is modifying just the
last token activation enough to communicate information from A to B?
Note that after applying masked attention in each of the previous Transformer layers, the last token activation of A attends
to all tokens before it, hence incorporating information from the entire sequence. Indeed, this must be the case for activation
communication to recover the gap between the zero-communication and skyline setups on both coordination games, which
(for Tip Sheets in particular) require information starting at the first few tokens of A’s prompt to be communicated.
To verify this empirically, we experiment with summing the activations of all tokens in the sequence rather than just the
last (we cannot replace all tokens as this would just replace B’s layer-j activation with A’s layerk-activation). Results are
shown in Table 5.
Indeed, applying f to all tokens decreases performance relative to applying f to just the last token. Note that the fact
performance generally decreases from f = replace to f = sum, and further with all tokens, is expected. The high
performance of AC with f = replace means that the edited last-token activation b retains some meaning in B’s activation
space; it is less likely for this to be the case when f = sum (at the very least b has norm roughly 2× that of B’s original
last-token activation), and when doing this for all tokens we’d expect performance to decrease even further as now all
activation vectors, not just the last, are out-of-distribution with respect to B’s activation space.
B.2. Incorporating Chain-of-Thought Prompting
How does AC perform in relation to NLD in cases where A might incur a long response (possibly with chain-of-thought for
intermediate answer computation)? I.e., does AC lose out on the benefits of CoT?
First, note that we still reap the benefits of CoT when we sample a completion from B after AC (where B gets all the
information encoding A’s “beliefs” about the prompt via AC, hence CoT on A’s side is not needed). To verify this, we
experiment with prompting A with CoT, generating a full response, and then passing the layer-k last-token activation of the
CoT response to B as part of AC. Results are shown in Table 6.
Indeed, we empirically find our above intuition (in orange) to hold, as there is no significant improvement over vanilla AC
when generating from A using CoT.
B.3. Learning W In-Distribution
Recall our reasoning about the AC (W ) results from Section 4.2: “We hypothesize that the benefits from the learned
linear layer are less consistent across datasets because the subset of C4 data used to train W likely contains text more
20

<!-- page 21 -->

Communicating Activations Between Language Model Agents
Table 7: GSM8k performance when learning W in-distribution. All AC variants involve communication between
LLaMA-3.2-3B (A) and LLaMA-3.1-8B (B).
AC AC ( W ) AC ( Win dist)
64.0±4.8 66.0±4.8 78.0±4.1
Table 8: Reasoning benchmark performance of communication between identical models. Both NLD and AC involve
communication between 2 instances of LLaMA-3.1-8B. 512-token completions are sampled with temperature 0.7 and debate
is run for 2 rounds.
Method Biog. GSM8k HS Psych. Logic Col. Bio. Prof. Law Pub. Rel.
LLaMA-3.1-8B 83.9±0.0 60.0±4.9 65.0±0.1 42.0±0.1 50.0±0.2 20.0±0.8 53.0±0.2
NLD 80.8±0.0 70.0±3.7 85.0±3.6 35.0±4.8 78.0±4.1 40.0±4.9 53.0±5.1
AC 83.7±0.0 60.0±4.9 85.0±3.6 40.0±4.9 74.0±4.4 40.0±4.9 79.0±4.1
semantically similar to some datasets than others, hence some datasets provide W with out-of-distribution inputs which
reduces performance compared to vanilla AC.”
Indeed, we verify this hypothesis by training W on the GSM8k train set (to produce Win dist) and then evaluating with this
task-specific linear layer on the GSM8k test set. Results are shown in Table 7.
Indeed, learning W in-distribution significantly boosts performance, confirming our hypothesis. Unfortunately we cannot
run this experiment for the other datasets, as there is no in-distribution training data available for MMLU (we use all public
data for testing).
Hence, this suggests that AC ( W ) should unilaterally improve over vanilla AC if we choose a training set with good
coverage across many tasks and distributions, such that there are sentences semantically similar to prompts across the span
of downstream task datasets.
B.4. Activation Space Similarity ∝ AC Performance Gain
We conduct the following experiment: for each of the six pairs of models A, B in the above experiment (see Table 4), we
compute the increase in Biographies performance with AC relative to the average individual performance of A and B. We
also compute the matrix analogue of the squared cosine similarity between the models’ activation spaces,
∥Y ⊤X∥2
F
∥X∥2
F ∥Y ∥2
F
,
where X is the matrix of A’s activations on 3072 sentences from C4 (the same dataset used to train W ), Y is the
corresponding matrix for B, and ∥·∥F denotes the Frobenius norm. This yields the plot in Figure 11.
There is a clear positive correlation between the similarity of the activation distributions and the AC performance gain, as
expected; the more aligned A and B’s activation spaces are, the more semantically meaningful and useful the embedding we
graft from A to B becomes.
B.5. Communicating Activations Between Identical Models
Note that AC as described in Section 3.1 only supports communication between distinct models. We can extend AC to work
for communication between identical models as follows: let A and B be instances of the same model. We can sample a
completion from A with temperature and graft the last-token layer-k activation of the completion into B at layer j as part of
the AC procedure. This still saves a substantial amount of compute over NLD between 2 model instances, showing our
technique can apply to this setting. Table 8 shows the results of this experiment.
Indeed, while communication between multiple model instances doesn’t always show improvement over the single model
itself (a well-known result from (Du et al., 2023)), AC matches/outperforms NLD on five of the seven datasets.
The intuition behind debate between multiple identical model instances is that sampling multiple completions (with
temperature) from the same model yields diverse reasoning paths that can be recombined into a stronger final answer. The
21

<!-- page 22 -->

Communicating Activations Between Language Model Agents
Figure 11. AC performance gain over average A/B individual performance on Biographies, as a function of matrix “cosine similarity”
between A and B’s activation spaces.
Table 9: Reasoning benchmark performance of AC and NLD with varying number of rounds. All methods involve
communication between LLaMA-3.2-3B (A) and LLaMA-3.1-8B (B).
Method Biog. GSM8k HS Psych. Logic Col. Bio. Prof. Law Pub. Rel.
NLD (1 round) 83.6±0.0 72.0±4.5 65.0±4.8 40.0±4.9 68.0±4.6 30.0±4.6 63.0±4.8
NLD (2 rounds) 80.2±0.1 75.0±4.3 83.0±0.8 37.0±0.1 71.0±0.1 30.0±0.1 63.0±0.7
NLD (3 rounds) 80.1±4.6 79.0±4.1 70.0±4.6 45.0±5.0 63.0±4.8 40.0±4.9 74.0±4.4
NLD (4 rounds) 78.0±0.0 79.0±4.1 * * * * *
AC 84.6±0.0 64.0±4.8 85.0±0.8 47.0±0.1 78.0±0.9 30.0±0.1 74.0±0.1
∗Runs required too much compute
above experiment shows that the same intuition holds for AC—we are sampling multiple times from the same model, but
passing responses between agents via AC rather than as NL messages.
B.6. Additional Rounds of Natural Language Debate
In Section 4.2 we fix NLD to 2 agents and 2 rounds, however we find in additional experiments that AC outperforms NLD
even with additional rounds, highlighting the superiority and robustness of activations as an alternative “language” for
inter-LM communication. Results are shown in Table 9; we see that for 5 of the 7 reasoning benchmarks, AC beats NLD
even with 3-4 rounds while using substantially less compute.
B.7. Full MMLU Benchmark Results
Table 10 below displays complete results of both AC and NLD on the full MMLU benchmark. Notably, AC
matches/outperforms NLD on 48/57 datasets, with substantially less compute used , indicating its superiority and
robustness as an alternative “language” for inter-LLM communication.
22

<!-- page 23 -->

Communicating Activations Between Language Model Agents
Table 10: Comparison of NLD vs. AC on the full MMLU benchmark (Hendrycks et al., 2021).
Dataset NLD AC
Conceptual Physics 60.0 ± 4.9 68.0 ± 4.6
High School Chemistry 50.0 ± 5.0 37.0 ± 4.8
Security Studies 60.0 ± 4.9 60 .0 ± 4.9
Jurisprudence 84.0 ± 3.6 84 .0 ± 3.6
Logical Fallacies 63.0 ± 4.8 72.0 ± 4.5
College Computer Science 44.0 ± 5.0 44 .0 ± 5.0
International Law 55.0 ± 5.0 59.0 ± 4.9
Miscellaneous 90.0 ± 3.0 95.0 ± 2.2
Marketing 70.0 ± 4.6 85.0 ± 3.6
Elementary Mathematics 75.0 ± 4.3 58.0 ± 4.9
Machine Learning 42.0 ± 4.9 42 .0 ± 4.9
High School Macroeconomics 44.0 ± 5.0 75.0 ± 4.3
High School US History 45.0 ± 5.0 71.0 ± 4.6
Human Aging 56.0 ± 5.0 72.0 ± 4.5
Astronomy 79.0 ± 4.1 80.0 ± 4.0
Computer Security 56.0 ± 5.0 75.0 ± 4.3
High School Statistics 55.0 ± 5.0 42.0 ± 4.9
Professional Medicine 79.0 ± 4.1 65.0 ± 4.8
Electrical Engineering 58.0 ± 4.9 60.0 ± 4.9
High School Computer Science 63.0 ± 4.8 70.0 ± 4.6
College Physics 50.0 ± 5.0 28.0 ± 4.5
Management 74.0 ± 4.1 75.0 ± 4.3
Moral Scenarios 40.0 ± 4.9 40 .0 ± 4.9
World Religions 58.0 ± 4.9 72.0 ± 4.5
Virology 47.0 ± 5.0 50.0 ± 5.0
Philosophy 67.0 ± 4.7 70.0 ± 4.6
Abstract Algebra 50.0 ± 5.0 28.0 ± 4.5
High School Government and Politics 80.0 ± 4.0 61.0 ± 4.9
High School Biology 60.0 ± 4.9 65.0 ± 4.8
College Mathematics 64.0 ± 4.8 66.0 ± 2.4
Global Facts 33.0 ± 5.0 37.0 ± 4.8
High School World History 71.0 ± 4.0 74.0 ± 4.4
High School European History 68.0 ± 4.0 71.0 ± 4.6
College Medicine 65.0 ± 4.8 53.0 ± 5.0
High School Geography 67.0 ± 4.7 79.0 ± 4.1
Anatomy 74.0 ± 4.4 74 .0 ± 4.4
Human Sexuality 75.0 ± 4.3 75 .0 ± 4.3
Medical Genetics 79.0 ± 4.1 82.0 ± 3.8
Professional Accounting 40.0 ± 4.9 48.0 ± 4.5
US Foreign Policy 89.0 ± 3.1 90.0 ± 3.1
Business Ethics 43.0 ± 5.0 44.0 ± 5.0
College Chemistry 41.0 ± 5.0 47.0 ± 5.0
High School Physics 40.0 ± 5.0 47.0 ± 5.0
Professional Psychology 54.0 ± 4.8 55.0 ± 5.0
Sociology 68.0 ± 4.1 68.0 ± 4.6
High School Microeconomics 95.0 ± 2.2 95 .0 ± 2.2
High School Mathematics 55.0 ± 5.0 55 .0 ± 5.0
Prehistory 75.0 ± 4.3 60.0 ± 4.9
Nutrition 64.0 ± 4.5 70.0 ± 4.6
Clinical Knowledge 65.0 ± 4.3 65 .0 ± 4.8
Moral Disputes 58.0 ± 4.8 60.0 ± 4.9
Econometrics 40.0 ± 5.0 40 .0 ± 4.9
High School Psychology 83.0 ± 0.8 85.0 ± 0.8
Formal Logic 37.0 ± 0.1 47.0 ± 0.1
College Biology 71.0 ± 0.1 78.0 ± 0.9
Professional Law 30.0 ± 0.1 30 .0 ± 0.1
Public Relations 63.0 ± 0.7 74.0 ± 0.1
Average 60.7 ± 2.0 62.7 ± 2.2
23
