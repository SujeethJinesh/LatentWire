# references/20_droidspeak.pdf

<!-- page 1 -->

DroidSpeak: KV Cache Sharing for Cross-LLM Communication
and Multi-LLM Serving
Yuhan Liu1 Yuyang Huang1 Jiayi Yao1 Shaoting Feng1 Zhuohan Gu1 Kuntai Du1 Hanchen Li1 Yihua Cheng1
Junchen Jiang1 Shan Lu2 Madan Musuvathi2 Esha Choukse2
1University of Chicago 2Microsoft
Abstract
Compound AI systems, such as agentic systems, are an emerg-
ing trend in large-scale enterprise settings, with multiple
LLMs specialized for different users, tasks, and/or roles work-
ing together. In these scenarios, different models often process
inputs that share the same context prefix. Although much work
was done in the past to enable the reuse of prefix KV caches
across inputs for a single model, how to enable one model to
reuse the prefix KV caches of a different model remains an
open question.
We introduce DroidSpeak, the first distributed LLM infer-
ence system that enables KV cache reuse across distributed
nodes running inference of different LLMs, so long as the
LLMs have the same architecture. We present the first study
that aims at understanding the impact of sharing KV caches
across different LLMs, and if/when such sharing affects qual-
ity. Inspired by the findings, we present DroidSpeak, which
selectively recomputes a few layers of the KV cache produced
by another LLM and reuses the remaining layers, with neg-
ligible quality loss. Moreover, carefully pipelining the layer-
wise re-computation and the loading of reused KV cache
further improves the inference performance. Experiments on
diverse datasets and model pairs demonstrate that DroidSpeak
achieves up to 4× throughput improvement and about 3.1×
faster prefill (time to first token), with negligible loss of qual-
ity in F1 scores, Rouge-L or code similarity score, compared
to the baseline which does not allow any sharing across mod-
els.
1 Introduction
Nowadays, LLM inference has become one of the most
resource-consuming workloads in industry, demanding ever
larger clusters of GPU machines [4, 72, 85–87, 106]. To re-
duce the computation demand, a common optimization is for
GPU machines that run the same LLM to share KV caches of
reused input prefixes over the network [20, 42, 53, 61]. How-
ever, how to make that optimization work across different
LLMs is yet to be studied.
Indeed, an emerging trend is hosting multiple different
LLMs in one GPU cluster. The reason is that with LLMs used
in more complex or personalized tasks, multiple LLMs, of-
ten fine-tuned from the same foundational model, are needed
to serve different users, tasks, or roles. These LLMs work
together to perform one complex task or to offer different
customized services [9, 12, 29, 65, 95]. Since standard prefix-
caching techniques work only when the KV cache is reused by
Baseline
agent
Expert
 agent
A) Workflows with multiple LLM
agents
Model BModel A
B) Multiple models working on
same content, storing their own
KV cache
KV cache KV cache
User
C) Personalized agents accessing
the same content
Coding
agent
Validation
agent
User Debugging
agent
E Cache transfer
Q
Q+A
Figure 1: Various scenarios in which same context is shared
by multiple LLMs. DroidSpeak brings down the computation
latency by up to 3.1×, increases throughput by 4×.
the same LLM, the use of multiple LLMs poses a direct chal-
lenge — how to reuse KV cache efficiently across different
LLMs in distributed settings.
To motivate this challenge, we highlight three common use
cases of multiple fine-tuned models working together in a sys-
tem. The first one is multi-agent systems, which use different
fine-tuned models to serve as the agents and to accomplish
collaborative tasks [14, 40]. The second use case is serving
multi-LoRA or multiple fine-tuned models , such as models
that are continuously updated over time with newer data [46],
or concurrently serving LoRA adapters [18, 79, 92]. The third
use case is personalized assistant systems , where a model
is fine-tuned for each user (or user type) according to their
personal preferences in coding or writing.
In all of these scenarios, prefix sharing is common. For ex-
ample, in multi-agent systems (Figure 1A), when the coding
agent talks to the validation agent, the conversation history of
the coding agent will be prepended to the input of the valida-
tion agent, to ensure the coherence of the conversation. In a
multi-LLM or multi-LoRA inference system (Figure 1B), an
updated model in a chatbot application could refer to the same
piece of conversation history produced by an older version
of the model. In a personalized assistant system (Figure 1C),
where each assistant is fine-tuned for a different user’s pref-
erence, the same news (i.e., the query prefix) can be used to
answer different users’ queries.
Based on the observations above, we pose the question:can
we share the intermediate states (i.e., KV cache) produced
1
arXiv:2411.02820v4  [cs.MA]  14 Jul 2025

<!-- page 2 -->

by one LLM on a given context to accelerate the prefill for
another LLM? In this paper, we seek to answer this ques-
tion under the assumption that the two LLMs have different
weights but the same architecture.
Intuitively, the potential performance benefit of such in-
termediate state sharing is huge — previous work on single-
LLM KV cache sharing has shown up to 8 × latency and
throughput improvement by speeding up the expensive pre-
fill phase of LLM inference, particularly for long context
workloads [31, 32, 49, 61]. However, just like a video decoder
could not decode a video encoded with another codec, naively
sharing the KV cache across different LLMs would cause
generation quality to drop greatly.
Our hypothesis is that there should be a way to re-compute
a small portion and reuse a large portion, although not all, of
the KV cache between two models that arefine-tuned from the
same base model — since the models are only fine-tuned, they
should share similar understanding of the same input context
and hence help accelerate each other’s inference. Of course,
the challenge is to validate this hypothesis, and to figure out
which part to re-compute or re-use without incurring much
delay overhead or degradation of quality.
Through a thorough empirical study (§3.2), we measured
eight representative model pairs and found that only a small
subset, often around 10%, of layers are sensitive to the KV
cache difference between two models in a pair. The identities
of these layers vary with different model pairs, but are largely
consistent across different inputs for the same model pair. We
refer to these layers as critical layers in this paper. Therefore,
for each pair of LLMs, we propose to selectively recompute
these critical layers in the KV cache.
We build our insights into DroidSpeak, the first distributed
multiple-LLM inference system that enables efficient sharing
of KV caches across different LLMs. First, DroidSpeak iden-
tifies the critical layer groups through offline profiling on a
held-out “training” set, ensuring sufficient re-computation for
accuracy purposes while reusing as many layers’ KV cache
as possible. Second, DroidSpeak implements smart KV cache
loading, which pipelines the loading of KV cache with the
re-computation of critical layers, to hide the loading delay
from remote nodes as much as possible.
We should note that the high-level idea of selectively re-
computing KV cache is not exactly new. DroidSpeak differs
with prior work [31, 32, 61, 97] inwhy and how to re-compute
KV cache: DroidSpeak is designed for KV cache reuse across
different LLMs, while prior work assumes a single LLM; due
to the different purposes, none of the prior work chooses to
re-compute or re-use a group of layers like in DroidSpeak.
For example, CacheBlend [97] updates a reused KV cache,
but it still feeds the KV cache to thesame LLM that generated
the KV cache. Moreover, it updates the KV cache of certain
tokens, rather than certain layers like in DroidSpeak.
We evaluate DroidSpeak on six datasets across three dif-
ferent tasks, including question answering, text summariza-
E
Transformer
Layer 1
Transformer
Layer 2
E
Wk
Wv
Wq Q
K
V
((Q x K)/√d) x V
...
Figure 2: Illustration of the use of embedding (E), query (Q),
key (K), and value (V) tensors in self-attention in transformer-
based LLMs.
tion, and coding, across eight different model pairs. We com-
pare DroidSpeak with various baselines, including direct KV
reuse [31], CacheBlend [97], and smaller models. Across
these setups, we can reduce the latency by up to 3.1 ×, im-
prove throughput by up to 4× with negligible drop in quality
(measured in F1 score, Rouge-L, or code similarity score).
While the concept of “translating” KV caches between
models involves a machine-learning challenge, our primary
contribution lies in making it practical in a distributed system
setting, which requires the transfer and computation of such
KV-cache translation to be done efficiently.
We have integrated our implementation with LMCache [25]
and vLLM [51], each being the state-of-the-art KV cache
management library and LLM inference engine, and it has
been tested in enterprise settings. The link to the code is
anonymized here for double-blind review requirements.
2 Background & Motivation
In this section, we give a brief introduction to the background
of the emerging workload of context sharing between different
fine-tuned model versions and the motivation for DroidSpeak.
2.1 Basic Transformer Concepts
The recent wave of generative AI is fueled by the advent
of high-performing models that are transformer-based and
decoder-only [27, 30, 37].
Query, Key, Value, and Embedding: In transformers, Q
(Query), K (Key), and V (Value) are the core components of
the attention mechanism [13, 58, 67, 84, 99]. An LLM model
comprises many layers. Each layer generates its E/Q/K/V
tensors given an input (Figure 2). We denote the K and V
tensors altogether as KV cache, and the embedding E tensors
as E cache. Within each layer, embeddings E are the starting
point for subsequent transformer computations (including
attention).
The quality of E/Q/K/V tensors directly affects the model’s
ability to understand and process the input context effectively.
Prefill and Decode phases: LLMs process input and generate
output in two distinct phases: the prefill phase and the decode
phase, as shown in Figure 3. In the Prefill Phase, the LLM
2

<!-- page 3 -->

LLM iteration 1 LLM
iteration 2
LLM iteration
3
KV
cache
KV
cacheIs tomato a
fruit?
It is
Prefill Phase Decode Phase
ai
Figure 3: Prefill and decode phases.
A B C D
Model pairs
0
25
50F1 score
Base Fine-tuned Figure 4: Fine-tuned model gives
higher accuracy than baseline.
1/8 1/4 1/2 1 2 8 16
Queries per Second (QPS)
0
10End-to-End Time (s)
context length = 5k
context length = 20k
Figure 5: Shorter input leads to
smaller end-to-end time.
processes the entire input context to produce the embeddings
and the KV caches across all layers. In the Decode Phase,
the model uses the KV cache generated in the prefill phase to
autoregressively produce tokens one by one as the output.
Fine-tuned LLMs: Despite being versatile, foundational
LLMs’ capabilities on specific tasks can improve through
fine-tuning on specialized domain data. For example, one
can turn a foundational LLM into a customer-support
agent by finetuning on troubleshooting requests [73],
or into a legal assistant by finetuning on case law and
statutes [101]. Fine-tuning can greatly improve the accuracy
of an LLM on a target domain, as shown in Figure 4,
where the fine-tuned models (Llama-3-70B-Instruct [5],
Mistrallite [100], Llama-3-8B-Instruct [5], and
MAmmoTH2 [102] ) greatly outperform the foundational
models they originate from, on the HotpotQA dataset [96].
Recent works in parameter efficient fine-tuning, like
LoRA [43] have made fine-tuned models even more accessi-
ble by updating part of the model weights, reducing computa-
tional and memory resources during fine-tuning.
2.2 Context Sharing Across LLMs
In compound AI systems, prefix contexts1 are often shared
across different LLMs—either to enhance the coherence of
the chat experience or to reference the same set of background
documents. In this paper, for convenience, we use the follow-
ing terminology:
• Sender model produces the KV cache of a context;
• Receiver model reuses the KV cache (with limited recom-
putation) of the reused context.
We describe several concrete use cases of context sharing
between different LLMs below.
Agentic Workflows: Agentic workflows represent a
paradigm shift in automation and collaboration in the LLM
space [3, 7, 33, 44, 54, 55, 90]. These workflows integrate mul-
tiple specialized LLM agents, each fine-tuned for specific
tasks, to collaborate and solve complex, multi-step problems,
or let them play different roles in agent debating. For exam-
ple, prior works propose using fine-tuned models as differ-
ent agents to improve the generation quality and generaliza-
1Since a shared context is often the prefix of different inputs, we use
prefix (caching/sharing) and context (caching/sharing) interchangeably.
tion of agents [14, 22, 23, 81]. Compared with using different
prompts on the same model for different agents, using fine-
tuned models as different agents can better improve output
quality [14, 81].
Need for context sharing: In agentic workflows, different
agents often share a common context, often the conversation
history of other agents, to ensure the coherence and consis-
tency between agents [36, 40, 89]. As a concrete example, in
coding agentic workflows with a coding and a testing agent,
the testing agent (receiver model) has to read both the input
instructions and generated code from the coding agent (sender
model) to write appropriate unit tests to meet user’s needs
[40, 76, 89].
Personalized Models: Personalized models tailored to in-
dividual users or tasks are increasingly prevalent in LLM
systems, particularly in applications like chatbots, virtual as-
sistants, and recommendation engines [11,17,83]. In these ap-
plications, different assistants are typically LLMs fine-tuned
for different users’ preferences [41, 55]. For example, the per-
sonal assistant for a software engineer can be fine-tuned to
generate high-quality and concise code snippets, while the
personal assistant for a financial analyst identifies marketing
angles in the documents without technical detail.
Need for context sharing: These models often share over-
lapping contexts, such as common conversation histories or
shared knowledge bases, to ensure continuity and relevance.
For instance, two assistants answering similar queries about
current events will process the same top news.
Multiple-LLM or multi-LoRA serving: In chatbot applica-
tions, LLMs often require continuous updates to incorporate
new information to provide up-to-date support and higher-
quality answers for users [8, 26]. As an example, ChatGPT
APIs release new API versions based on the same founda-
tional model about every two months, which fine-tunes on
the emerging new data [64]. Furthermore, multiple LoRA
adapters often need to be concurrently served [18, 79] to ac-
complish different tasks or serve different users.
Need for context sharing: In this case, the updated model
(receiver model) often needs to re-process the same sets of
popular contexts processed by the older model (sender model)
before. Multiple LoRA adapters can share their KV cache
when processing the same context.
We motivate DroidSpeak with these emerging trends in
3

<!-- page 4 -->

the workloads today that fuel the need for efficient context
sharing across fine-tuned LLMs.
2.3 Distributed LLM Inference Systems
As the demand for LLM inference continues to grow, it has
become common to serve LLMs in a cluster of GPU nodes.
Many companies have developed their own distributed infer-
ence systems, such as vLLM Production Stack [85], NVIDIA
Dynamo [4], and ByteDance AIBrix [86]. Among all these
frameworks, KV cache sharing across nodes is one of the
most important features for reducing prefill computation and
increasing overall throughput. Specifically, when there are
multiple requests to the same model querying a common pre-
fix context, these systems can transfer KV cache generated
by another user request, either from another GPU node or
through a centralized storage backend [20, 31, 50].
However, current distributed inference systems have not
optimized for multiple-LLM inference yet–they do not ex-
plore the opportunity to share KV cache across GPU nodes
when the requests are querying different models.
2.4 Prefill interference
Given the distinct characteristics of the prefill and decode
phases described in Section 2.1, lengthy prefill phases can
significantly reduce an inference system’s goodput—defined
as the number of queries processed per second within a la-
tency SLO [106]. This reduction occurs because TTFT (time
to first token) grows super-linearly with input length, and
the decoding phase cannot start until the prefill phase com-
pletes. Consequently, long inputs often turn prefill delays
into the end-to-end bottleneck. For example, as shown in Fig-
ure 5—where Llama-3-8B runs on a single A100 GPU with
synthetic input lengths of 5K and 20K tokens—setting a 3-
second latency SLO reveals that increasing the input size by
only 4× can reduce goodput by as much as 32×.
Receiver Model Sender Model
evolcode-3.1-8b toolace-3.1-8b
glue_sst2 conllpp
mistral-blitz mistral-24b
phi3.5-mini-instr-adapter-v2 phi-3.5-mini-instr-task15
llama-3-8b-sft-lora-ultrachat fingpt-llama-3-8b
llama-3-70b-instruct llama-3-70b
mistrallite mistral-7b
llama-3.1-70b-instruct llama-3.1-70b
Table 1: The model pairs used in our paper. For each pair of
models, we use the datasets that meet the requirements listed
in §3.1 (i.e., the receiver model has better quality than the
sender model).
3 Reusing KV cache across LLMs
A simple way to eliminate the overhead of repetitively
re-computing the KV cache that another LLM has gener-
ated before is to reuse the KV cache produced by another
model. As a concrete example, on an A100 GPU and us-
ing Llama-3.1-8B-Instruct, reusing the KV cache for a
40K-token input can reduce prefill latency from 4s to 0.08
seconds.
This naturally leads to the key question: What effect does
directly reusing another LLM’s KV cache have on generation
quality?
In this section, we present the first empirical study of how
reusing another model’s KV cache impacts output quality,
and we further examine whether these effects vary across
individual layers of the cache.
3.1 Building the benchmarks and datasets
Before getting into the KV cache sharing and patterns, we
describe the benchmark set we build for DroidSpeak. The
study needs pairs of models that share the context provided by
the datasets. The following assumptions are also made when
building the benchmark.
• The pair of models should share the same foundational
model. Specifically, the pair can either consist of the foun-
dational model and a fine-tuned model based on it, or, two
fine-tuned models based on the same foundational model.
• The dataset selected should be related to the task for which
one of the models has been fine-tuned. This is important
since in any context-sharing scenario, the receiver model is
performing the specialized task.
• The receiver model fine-tuned on the task in the correspond-
ing dataset should yield better quality than the sender model
in the pair.
Using these assumptions, we formulate the benchmark
as shown in Table 1. We use 8 pairs of models across 6
datasets (including HotpotQA [96], multifieldQA_en [48],
2wikimQA [38], multi_news [28], lcc [35], and repobench-
p [60]). The quality metric used is taken directly from the
dataset.
We focus on the use case where the sender model generates
the intermediate states for the context and the receiver model
reuses its intermediate states. This is a challenging use case
because the sender model has worse accuracy than the receiver
model, so achieving high quality requires properly refreshing
the KV cache.
3.2 Empirical insights of KV cache
Naive reusing is suboptimal: The first observation is about
naively reusing the sender model’s KV cache on the receiver
model. Specifically, we observe that:
Insight 1 Reusing the whole KV cache between models leads
to a huge loss in accuracy.
4

<!-- page 5 -->

lcc
2wikimQAhotpotQA
0
50
Quality score (%)
evolcode-3.1-8b reuses
 toolace-3.1-8b's KV
hotpotQAmulti_newsrepobench-p
Mistrallite reuses
 Mistral-7B's KV
Receiver Model Sender Model Direct KV reuses
hotpotQA2wikimQAmultifieldQA
glue_sst2 reuses
 conllpp's KV
hotpotQA2wikimQAmultifieldQA
mistral-blitz reuses
 mistral-24b's KV
hotpotQA2wikimQAmultifieldQA
0
50
Quality score (%)
llama-8b-ultrachat
 reuses fingpt-llama-8b's KV
hotpotQArepobench-p
lcc
phi-3.5-adapter-v2 reuses
 phi-3.5-task15's KV
hotpotQA2wikimQAmultifieldQA
Llama-3.1-70B-Instr reuses
 Llama-3.1-70B's KV
hotpotQA2wikimQAmultifieldQA
Llama-3-70B-Instr reuses
 Llama-3-70B's KV
Figure 6: Directly reusing the full KV cache greatly degrades generation quality.
0 10 20 30
20
40
60
F1 score (%)
evolcode
 reuses toolace-2-8b
0 10 20 30
50
60
70
mistral-blitzz
 reuses mistral-small-24b
0 10 20 30
30
40
glue_sst2
 reuses conllpp
0 10 20 30
60
70
mistrallite
 reuses mistral-7b
0 10 20 30
Layer
30
40
50
F1 score (%)
phi-3.5-mini-instr-adapter-v2
 reuses phi-3-5-mini-instr-task15
0 20 40 60 80
0
20
40
60
llama-3-70b-instruct-awq
 reuses llama-3-70b-awq
0 10 20 30
30
40
50
llama-3-8b-sft-lora-ultrachat
 reuses fingpt-llama-3-8b
0 20 40 60 80
50
60
llama-3.1-70b-instruct
 reuses llama-3.1-70b
Figure 7: Different layers have different sensitivities to deviation in KV cache. Plotted by reusing only one layer’s KV cache from
the sender model on the receiver model. The red dashed line is the original accuracy of the receiver model. The bars colored red
are those that have an F1 score drop of over 10% compared to the original receiver model.
0 5 10 15 20 25 30
Layer
0.0
0.5
1.0
Diff in F1 score
Figure 8: Variation in F1 score per input within a single dataset (HotpotQA) for model pair Llama-3-8B-sft-lora-ultrachat
reusing fingpt-llama-3-8B. We plot the 25 and 75 percentiles. Except layer 23, the 25 and 75 percentiles overlap, indicating a
low variance of error sensitivity across all layers except 23.
A naïve way to reuse the intermediate state between models
is to reuse the KV cache as is. In this case, the receiver model
receives the KV cache for the whole input prompt from the
sender model. It then uses this to generate the output tokens
in the decode phase, thereby skipping the prefill phase.
We show the impact of this on quality in Figure 6. For each
pair of models and dataset, we show the F1 score (higher is
better) of a) the receiver model, b) the receiver model while
reusing the KV cache generated by the sender model, and c)
the sender model alone.
Although the quality of the receiver model with the sender
model’s KV cache is still better than the sender model alone,
we lose a lot of accuracy. HotpotQA tends to lose more than
50% of the accuracy points across most of the model pairs,
while the other datasets show varying amounts of changes
across model pairs.
Layer-wise sensitivity to KV cache reuse: Our second
observation is about whether KV cache reusing leads to the
same impact across all layers.
Insight 2 Only a small subset of layers are sensitive to KV
cache reuse in terms of accuracy.
Figure 7 shows the quality drop by reusing part of KV
cache from the sender model. Specifically, each bar represents
the quality achieved by the receiver model reusing the KV
5

<!-- page 6 -->

cache for that corresponding layer from the sender model,
with everything else being recomputed.
For most of the model pairs, we find only a small subset of
layers are sensitive to the deviation in KV cache (i.e., F1 score
drops significantly), and we refer to these layers as critical
layers, which are colored red. On average across all pairs of
models, we identify 11% of layers to be critical.
Similarity of sensitivity across different inputs: Our third
observation is about whether different inputs show similar
patterns in layer-wise sensitivity.
Insight 3 The variation in KV cache patterns across inputs
is only notable for critical layers.
Figure 8 shows the violin plot of the normalized
change in F1 score per input in hotpotQA dataset,
when llama-3-8b-sft-lora-ultrachat reusing
fingpt-llama-3-8b’s KV cache of each layer only.
Layer 23, which is also marked as the most critical for this
model pair in Figure 7 ( i.e., the largest F1 score change),
shows a wider variation across different data points from the
dataset, with a lot of them observing F1 score change > 50%.
However, for all the non-critical layers, the variance in the F1
score change is insignificant, meaning that such non-critical
layers do not change across various inputs.
This phenomenon is also observed across other model pairs.
Intuitively, this can be because critical layers are essential for
the reasoning capabilities [21] or the ability to accomplish
specific downstream tasks [19]. These reasoning capabilities
must remain accurate to interpret any input to the LLMs.
4 DroidSpeak Design
Building on the insights in the previous section, we designed
DroidSpeak to enhance the context sharing between two
LLMs. The central questions that DroidSpeak targets are the
following: how do we determine the layers to re-compute to
reduce latency, while keeping the quality loss minimal?
4.1 Challenges with Selective KV Cache Reuse
Insight 2 suggests selectively reusing the KV cache while
recomputing it for critical layers might preserve quality. How-
ever, selecting all critical layers scattered across different parts
of the LLM is suboptimal for both efficiency and quality.
The efficiency challenge: Recomputing critical layers that
are non-contiguously placed is inefficient.
During the prefill phase, the output of a layer where the
KV cache is reused only contains information about the first
generated token. In contrast, recomputing the KV cache needs
to start from the E cache of this layer for the whole context.
While it is possible to obtain the E cache for layer l by per-
forming a full prefill from the context starting from the first
layer till layer l, this approach completely defeats the purpose
of KV cache reuse.
Recomputed KV cache
Decoding
Decoding
E cache
Reused KV cacheRecomputed KV cacheReused KV cache
SkippedlayersRecomputedlayersSkippedlayers
Figure 9: DroidSpeak chooses the critical layer groups (lay-
ers 4-5) to re-compute, and reuse KV cache for other layers.
To address this, we use the E cache from the sender model
to start the recomputing at the layer when transitioning from
KV cache reuse to recompute. We refer to this layer as a
transition layer. As depicted in Figure 9, for any transition
layer (between reuse and recompute), the sender model must
store and transmit the E cache to the receiver model.
The E cache is typically large, reaching up to twice the size
of the KV cache for the Mistral-7B or Llama-3-8B model
families, and up to four times larger for Llama-3.1-70B since
the KV cache size is optimized by group-query attention [6].
Thus, the overhead of storing the E cache in GPU memory and
the delays caused by loading it from remote GPU nodes can
be substantial, far exceeding the cost of storing and loading
KV cache alone.
The accuracy challenge: Furthermore, reusing the sender
model’s E cache at the transition layer also hurts the accuracy
of the final output. This is because the E cache loaded from
the sender model (starting point of the recomputation) already
differs from the receiver model. Such difference eventually
will introduce deviation from the point of recomputation and
propagate over all later layers. It is crucial to minimize the
error caused by such deviation.
If we pick all the critical layers, which are often not appear-
ing in contiguous groups (Figure 7), there will be multiple
transition layers from reuse to recompute, introducing multi-
ple deviations in E cache.
Figure 10 illustrates this. If we choose to recompute only
critical layers ( i.e., layers 16–18, 20, and 25–27), we need
to load E cache at layers 16, 20, and 25. However, whenever
we load E cache, the error from E cache will be propagated
to subsequent critical layers ( e.g., loading E cache at layer
16 populates errors to 16–18) and eventually to the output.
Thus, even if all the critical layers are recomputed, this will
lead to a substantial output error. In contrast, recomputing a
contiguous group of layers from 16 to 27 avoids this prob-
lem by recomputing the KV cache of non-critical layers that
are located between critical layers. As shown in Figure 10,
6

<!-- page 7 -->

Reusing E
Figure 10: More transition points lead to higher output error.
Pareto
frontie
r
Figure 11: An example result obtained by offline profil-
ing. Each point represents the F1 score achieved when re-
computing a specific group of contiguous layers. The Pareto
frontier shows the maximum F1 score attainable for each
number of re-computed layers.
re-computing a contiguous group of layers has much lower
output error than only re-computing the critical layers.
4.2 Profiling for re-computation configuration
A key question still remains: how to determine the critical
layer groups to be re-computed?
Based on Insight 3, the critical layers that are sensitive to
KV cache differences vary little with the inputs. This mo-
tivates our design to choose the critical layer group based
on some example inputs, and then apply to other new inputs.
Specifically, for each model pair, we use a “training” dataset
to determine the critical layer group. We refer to the critical
layer group as the re-computation configuration. The goal
is to understand the relationship between the critical layer
groups to perform re-computation and its impact on genera-
tion quality.
Figure 11 shows an example profiling result for the
glue_sst2 and conllpp model pair. Each point in the scatter
plot represents the F1 score achieved for a specific number of
re-computed layers. From these points, we obtain the Pareto
frontier, which captures the highest F1 score achieved at a
specific group of layers. For instance, a configuration with 11
re-computed layers is a good example on the Pareto frontier,
since the F1 score drop is within 5% of the original accuracy
of the receiver model, while the number of re-computed layers
is minimized.
Profiling Overhead: The profiling overhead has a complex-
Sender model
Profiler Training datasetOffline stage
Sender model GPUnode0
Online stage
Critical layer groups
KV cacheReceiver model GPU node1
Receiver model
KV cache
Network
Figure 12: Overall system design of DroidSpeak, including
the offline stage (top) that profiles the critical layers of each
pair of sender and receiver models (§4.2); and the online
stage (bottom) that uses the profile to dynamically pick the
critical layers and KV-cache loading strategy for each job
(§4.3).
ity of O(l2) where l is the number of layers in the LLMs.
For example, for Llama-3-8B with 32 layers, it takes three
hours on an A100 GPU. This one-time cost is negligible since
these models are deployed at a very large scale [72, 106]. Fur-
thermore, this cost can be substantially reduced by grouping
layers when profiling. Profiling at the granularity of 2-layer
groups, for instance, reduces the profiling time by about 3x. In
the evaluation we show soon in §5, we use the 2-layer group
profiling granularity.
4.3 DroidSpeak runtime design
As shown in Figure 12, DroidSpeak consists of two stages.
The offline stage (as detailed in §4.2) uses a training dataset
to profile the relationship between the value change of each
layer on the generation quality. This step allows DroidSpeak
to dynamically choose the right re-computation configuration
based on available resources. Next, we detail the online stage
that uses the offline profile to dynamically decide the critical
layers of the KV cache and executes the partial recomputation
on these critical layers to preserve high generation quality.
Specifically, in the online stage, DroidSpeak dynamically
decides which point on the Pareto frontier should be used
based on the latency SLO (details in Appendix§B). We as-
sume the E cache and KV cache at each transition point are
precomputed and stored, but they can also be generated by the
sender model in real-time and sent directly from the sender
model. Then, the receiver model selectively recomputes crit-
ical layer groups while reusing others, achieving a balance
between computational efficiency and quality.
Smart KV Cache Loading: Since GPUs hosting differ-
ent LLMs may reside in separate nodes, DroidSpeak needs
to fetch KV cache from remote nodes frequently. Without
careful design, this transfer can significantly increase the end-
to-end latency, particularly when fewer layers are recomputed
(i.e., more KV cache layers need to be transferred).
7

<!-- page 8 -->

Time
Re-compute
Load reused
(a) Send all layers of KV cache
(b) Only sends the KV cacheof reused layers
Time(c) Pipelining compute and loading
Time
L1-L10
L4-L10
L1-L10
L4-L10
L4-L10L1-L3
L4-L10
L4-L10
Re-compute
Load reused
Re-compute
Load reusedL1-L3
L1-L3
L1-L3
L1-L3
Figure 13: Illustration of different KV cache transferring
strategies. Each color represents the work (KV cache recom-
putation and loading reused KV caches) for the same query
for one model.
Consider an example with two receiver models A and B.
Each model has ten layers. Model A recomputes layers 4–10
and reuses KV cache for layers 1–3, while model B recom-
putes layers 1–3 and reuses KV cache for layers 4–10.
Figure 13 illustrates the timeline of three different loading
and recomputation strategies. We assume that a request arrives
at A at time=0, followed by a request to B after 2 time units,
and that recomputing or transferring one layer (KV or E cache)
takes 1 unit of time. The most naive approach is to load all
layers of the KV caches before recomputation begins in each
job. For example, in modelA (orange), this means first loading
the E cache for layer 4 and the entire KV cache, followed by
7 units of re-computation time, similarly for B, resulting in a
total TTFT of 47 (Figure 13(a)).
A slight improvement is to load only the reusable layers’
KV cache for A and B, right before recomputation. This re-
duces the total TTFT to 30, as shown in Figure 13(b).
However, both approaches miss the opportunity to overlap
recomputation and the loading of reused KV caches. Recom-
putation can start immediately after receiving the E cache
of the transition layer. The optimal solution, shown in Fig-
ure 13(c), pipelines loading and recomputation. For model
A, the E cache for layer 4 is transmitted first, enabling re-
computation of layers 4–10 to start while the KV cache for
layers 1–3 transfers in parallel. Similarly, for B, the loading
of E/KV cache can begin before A finishes recomputation.
This pipelined strategy reduces the total TTFT to 17—approx-
imately a 2× improvement over the baseline in (b).
4.4 Implementation
We implement DroidSpeak with about 3K lines of code in
Python, based on PyTorch v2.0, CUDA 12.0, and LMCache
0.1.4 [24, 25]. DroidSpeak operates the LLM inference serv-
ing engines through these interfaces:
• store(Cache, context, LLM) : We split the KV or E
cache into layers, and store it in a key-value store in GPU
memory.
• fetch(context, LLM, layer_id) -> Cache : Depend-
ing on what was stored previous store() call, this loads
layer’s KV or E cache of the correspondingLLM.
• partial_prefill(recompute_config, context)->
text: it takes in the recomputation configuration and the
context, including which layers to recompute during prefill,
and then generates the output text.
We implement these three interfaces in vLLM [51] and LM-
Cache [25]. For store_kv, after an LLM generates the KV
cache for a piece of context, we calculate the hash of the con-
text text, and put it into the key-value store if the context does
not exist in the current store. Before we run the inference
for any LLM, we obtain the re-computation configuration
from the offline profiling (§4.2), which includes the layer
numbers for recompute and KV cache reuse. During the on-
line inference stage, we call the partial_prefill function,
which calls fetch_kv for the layers for KV cache reusing,
and fetch_e at the transition layers. Both fetch_kv and
fetch_e are implemented with torch.distributed [74] to
fetch KV or E cache from a remote GPU node. All trans-
mission will be placed on a CUDA Stream different from
PyTorch’s default computation stream [75], enabling us to
overlap transmission of KV cache with recomputation and
hiding the transmission delay.
5 Evaluation
The key takeaways from the evaluation are:
• Across three datasets and eight model pairs, DroidSpeak
can reduce the prefill latency by 1.7–3.1× without compro-
mising accuracy, with an average prefill speedup of 2.1×.
• In the online serving system, DroidSpeak achieves up to
4× improvement in throughput.
• DroidSpeak’s profiling of recomputing layers is robust
across different datasets and model types.
5.1 Experiment Setup
Models: We evaluate DroidSpeak on eight pairs of models
(Table 1) of different sizes, specifically the fine-tuned versions
of Mistral-7B, Mistral-24B, Llama-3.1-8B, Llama-3-8B, Phi-
3.5-mini-instruct, Llama-3-70B and Llama-3.1-70B, selected
with the criteria in §3.1. These models are fine-tuned on the
base foundation model for chat-enhancing tasks, coding tasks,
and long context reasoning et. al. For Llama-3.1-70B and
Llama-3-70B models, we use 4-bit quantized models with
AWQ [57] to fit on one A100 GPU.
Note that the receiver models are not all directly fine-tuned
from the sender model; they also include two fine-tuned vari-
ants derived from the same foundation model.
Hardware setting: We run the experiments on two A100
virtual machines connected with InfiniBand link on Microsoft
Azure, namely Standard_ND96amsr_A100_v4, which con-
tains 8 × 80GB A100 GPUs on each virtual machine.
8

<!-- page 9 -->

Ours
0.0 0.2 0.4
30
40F1 Score (%)
Better
phi-adapter_v2 vs phi-task-15
 HotpotQA
0.0 0.5 1.0
40
60F1 Score (%)
Better
mistrallite
vs mistral-7b
HotpotQA
0 2 4
25
50F1 Score (%)
Better
llama-3.1-70b-instruct
vs llama-3.1-70b
HotpotQA
0 2 4
25
50F1 Score (%)
Better
llama-3-70b-instruct
vs llama-3-70b
HotpotQA
0.00 0.25 0.50
40
50Edit Similarity (%)
Better
repobench-p
0.00 0.25 0.50
0
20Rouge-L (%)
Better
multi_news
0 2 4
25
50F1 Score (%)
Better
2wikimqa
0 2 4
20
40F1 Score (%)
Better
2wikimqa
0.0 0.2 0.4
Latency (s)
25
50Edit Similarity (%)
Better
lcc
0 1
Latency (s)
10
20Edit Similarity (%)
Better
repobench-p
0 2 4
Latency (s)
25
50F1 Score (%)
Better
multifieldQA
0 2 4
Latency (s)
25
50F1 Score (%)
Better
multifieldQA
Figure 14: Prefill delay and F1 score trade-off. DroidSpeak greatly reduces prefill latency while maintaining generation quality.
Datasets: We evaluate DroidSpeak on six datasets, with
their context length statistics summarized in Table 2. For
each model pair, we report results on three of these datasets,
following the criteria that the receiver model must achieve
higher generation quality than the sender model on the chosen
datasets. The tasks drawn from the LongBench evaluation
suite [10] evaluate LLM capabilities in multi-hop reasoning,
summarization, and code understanding or completion.
Train/test split: As discussed in §4.2, DroidSpeak profiles
the critical layer groups that has quality drop within 5% of the
original quality with a “training” dataset offline. Specifically,
we use 50 contexts from HotpotQA dataset as the “training”
dataset to obtain the critical layer groups with the profiling
mechanism mentioned in §4.2. We apply the critical layer
groups on all the other testing datasets in the benchmark.
Quality metrics: We measure generation quality using the
standard metric of each dataset, following prior work [10, 61,
97]. Specifically, we use F1 score for QA tasks (hotpotQA,
2wikimQA, multifieldQA_en ), which measures the proba-
bility that the generated answer matches the ground-truth an-
swer for the question-answering task; Rouge-L score for sum-
marization tasks (multi_news), which measures the longest
common subsequence between the generated summarization
and the ground-truth answer; and finally the code similarity
score for code completion tasks (lcc, repobench-p ), which
measures the edit distance between the generated completed
code and the ground-truth code.
System metrics: We use the system metrics listed in §2.1 to
evaluate DroidSpeak compared with the baselines, including
TTFT, TBT, E2E. In §5.2 we also measure prefill latency,
which includes the prefill computation time on GPU and the
loading delay to fetch KV and E cache through InfiniBand
9

<!-- page 10 -->

0
1
2
3
4
TTFT (s)
gsm8k
reuses glue_stsb
Ours
Full Prefill
0.0
2.5
5.0
7.5
10.0
Llama-3.1-70B-Instruct
reuses Llama-3.1-70B
2
4
Mistral-blitz
reuses Mistral-24B
0
1
2
3
Phi-adapter_v2
reuses Phi-task-15
0.0
0.5
1.0
TBT (s)
0
1
2
0
1
2
0.1
0.2
10 20 30
QPS
0
1
2
3
E2E (s)
0 5 10
0
10
20
30
40
0 10 20
0.0
2.5
5.0
7.5
10.0
0 10 20
0
2
4
Figure 15: The impact of arrival rate on Time-to-first-token (TTFT), Time-between-tokens (TBT), and end-to-end latency (E2E),
when the DroidSpeak’s quality is same as full prefill. Ran the models with eight replicas, with round-robin routing.
bandwidth link across two GPU nodes.
Baselines: We compare with the following baselines:
• Full prefill: the receiver model prefills the text of the context
with vLLM [51], representing the baseline of the highest
computation overhead but the best quality we can get.
• Full KV cache reuse [31]: the receiver model directly reuses
the KV cache from the sender model, and the receiver
model runs decoding with the transferred KV cache.
• CacheBlend [97]: we extend CacheBlend’s algorithm to
determine the important tokens to re-compute for cross-
LLM KV cache sharing, based on the difference between
the re-computed KV cache and sender model’s original KV
cache for the first layer.
• Smaller models: In §5.6, we also compare
Llama-3.1-70B-Instruct’s accuracy and latency trade-
off with DroidSpeak with Llama-3.1-8B-Instruct,
which is fine-tuned with the same instruct-tuning dataset.
5.2 Lower Latency with Preserved Accuracy
We first demonstrate DroidSpeak’s reduction in prefill de-
lay and accuracy trade-off in Figure 14. Across 8 pairs of
models on three datasets, DroidSpeak achieves 1.7–3.1× re-
duction in prefill delay over the full prefill method, without
compromising generation quality. On the other hand, when
compared with reusing all of sender model’s KV cache, Droid-
Speak successfully preserves the improved quality of the re-
ceiver model despite a slightly higher delay. Compared to
CacheBlend, DroidSpeak can achieve much better latency
and quality trade-off. Specifically, DroidSpeak has 5–33%
(average 16%) higher quality than CacheBlend at a similar
prefill latency.
Understanding DroidSpeak’s improvement: DroidSpeak
outperforms the baselines for various reasons. Compared to
the full prefill baseline, DroidSpeak achieves significantly
lower prefill delay as only a small fraction of layers are pre-
filled. In contrast to full KV reuse, DroidSpeak has a longer
prefill latency because it does not perform prefill at all. How-
ever, it greatly reduces accuracy because it misses the op-
portunity to leverage layer-wise sensitivity in the KV cache
difference. DroidSpeak is better than CacheBlend in terms
of quality because DroidSpeak re-computes the critical layer
groups that are most sensitive to KV cache deviation between
models, while CacheBlend fails to spot the critical layers
since it selects the tokens to re-compute based on the first
layer.
5.3 Throughput and Latency Improvement
To see the impact of DroidSpeak on improving the throughput
of an online LLM inference system, we emulated an online
inference scenario by pairing the datasets with request arrival
times following a Poisson distribution under different incom-
ing rates to evaluate the performance of DroidSpeak in more
practical workloads.
For the experiment, we deployed a Kubernetes cluster run-
ning vLLM Production Stack [85], using DroidSpeak ’s cus-
tomized Docker image. The cluster consisted of two nodes,
each equipped with 8 A100 GPUs. We configured eight repli-
cas for each model across these nodes. Model placement fol-
lowed a simple strategy: for each model pair, we deployed four
replicas of both the sender and receiver models on each node.
Request routing was handled by vLLM Production Stack’s
round-robin algorithm, which splits incoming requests evenly
across all model replicas.
As demonstrated in Figure 15, we compare the TTFT, TBT,
and E2E impact under various request rates on the HotpotQA
dataset. Due to the limit in space, we only show four pairs of
models to illustrate. For DroidSpeak, we chose the configura-
tion within 1% accuracy drop for these pairs of models.
TTFT: Since the full-recompute baseline has much higher
10

<!-- page 11 -->

2.5 5.0 7.5 10.0
0.0
0.2
0.4
TTFT (s)
Ours
Full Prefill
2.5 5.0 7.5 10.0
0
2
4
6
E2E (s)
Figure 16: Impact of the arrival rate on the Time-to-first-
token (TTFT), and end-to-end latency (E2E) of code agentic
workflow.
0 20
0
20
40
F1 score (%)Test dataset:
hotpotqa
0 20
# of Reuse Layers
Test dataset:
multifieldqa_en
Profiled w/ hotpotqa w/ multifieldqa_en w/ 2wikimqa
0 20
Test dataset:
2wikimqa
Figure 17: Using the recompute layers profiled on training
datasets works well on testing datasets.
prefill latency than DroidSpeak, the queuing delay affects
(knee in the curve) its TTFT at a much lower QPS than what
DroidSpeak can support.
TBT and E2E: Although we are only reducing the TTFT
directly in DroidSpeak, the second-degree effect through less
interference and better scheduling brings down the TBT and
E2E latency too, as shown in Figure 15.
Throughput: Assuming an SLO that avoids the effects of
high queuing delays (knee of full prefill) on TTFT, TBT, and
E2E latency, DroidSpeak can support 2-4× higher throughput.
5.4 Robustness across datasets
As discussed in §4.3, DroidSpeak profiles the KV cache reuse
pattern using a single profiling run on a "training dataset"
during the offline stage and then generalizes the profile results
to other datasets during the online stage.
Figure 17 illustrates whether the profile obtained on one
dataset offline generalizes well to other datasets. In each sub-
figure, we plot the Pareto frontier of the F1 score versus the
number of reused layers, obtained through profiling on the
original testing dataset vs two other datasets in our benchmark
using glue_sst2 and conllpp model pair.
The figure demonstrates that the Pareto frontier obtained us-
ing the profile from the training dataset on the testing dataset
closely resembles the frontier obtained using the profile di-
rectly from the testing dataset. Across all the pertinent con-
figurations, the maximum difference in the score is 4 points,
with the average being 2 points. This result further validates
the sufficiency and robustness of our profiling strategy.
5.5 Case study of other tasks and other models
Math task: To demonstrate that the mechanisms of Droid-
Speak apply to other types of datasets, we apply DroidSpeak
on a model pair where the receiver model is fine-tuned on
math reasoning, and test on a math reasoning task.
In Figure 18(a), we run GSM8K [102] dataset on MAm-
moTH2 [102]. Note that the Pareto frontier obtained follows a
very similar pattern compared to the LongBench models and
dataset, demonstrating the wide applicability of DroidSpeak.
Case study on coding agentic workflow: Next, we study
how DroidSpeak performs under a real agentic workflow by
orchestrating a coding agent system using MetaGPT [40], a
state-of-the-art multi-agent framework. The system consists
of two agents, a coder using evolcode model and a tester us-
ing tool-8b model. The coder is responsible for implementing
Python functions according to the input prompt, and the tester
is responsible for testing the coder’s code and providing com-
ments to the coder agent for the next round’s modification.
We send the problems from the HumanEval dataset at
various rates. In Figure 16, we plot the TTFT and E2E im-
pact under different QPS, similar to the setup in §5.3, where
DroidSpeak is plotted with the re-computation configuration
that maintains the generation quality (pass@1 score of 52.5).
DroidSpeak significantly improves the TTFT by 2.7 × and
brings down the E2E delay to finish the problem as well.
Mixture-of-Experts Model: We also evaluate DroidSpeak
on a Mixture-of-Experts (MoE) model. As shown in Fig-
ure 19, where the sender model is Mixtral-8x7B and the re-
ceiver is Mixtral-8x7B-Instruct, DroidSpeak is able to achieve
significant reductions in prefill latency as well. While MoE
models may activate different experts for different inputs,
this selection occurs only at the linear layers after the atten-
tion modules—where the KV cache is used in. As a result,
DroidSpeak ’s KV cache sharing remains effective for MoE
architectures.
0.00 0.02 0.04
Prefill Delay (s)
 (a)
20
30
40F1 score (%)
Ours
Full Prefill
Full Reuse
0 2 4
Prefill Delay (s)
 (b)
0
20
40
60F1 score (%)
Ours on 70B
8B
70B
Figure 18: (a) Prefill delay and accuracy trade-off for MAm-
moTH2 (fine-tuned on math reasoning tasks). (b) DroidSpeak
applied on Llama-3.1-70B-Instruct has higher accuracy than
Llama-3-8B-Instruct.
5.6 Comparison against a smaller model
Since DroidSpeak trades off minimal accuracy impact for
latency, we compare using DroidSpeak on a larger model
with a smaller model of the same architecture to show our
superior performance in the quality and delay trade-off.
In Figure 18(b), we compare DroidSpeak on
Llama-3.1-70B-Instruct and Llama-3.1-8B-Instruct,
which is a smaller version of Llama-3.1-70B-Instruct
11

<!-- page 12 -->

and fine-tuned on the same dataset to enhance the base LLM’s
ability to follow instructions. As shown, Llama-3.1-8B
achieves approximately a 4 × reduction in prefill delay but
suffers a reduction in F1 score of about half compared to the
original F1 score of Llama-3.1-70B-Instruct.
One significant drawback of using a smaller model to
achieve speedup is the overhead of switching between small
and large models. For example, when additional resources
become available, switching back to the larger model to im-
prove serving quality incurs the overhead of loading the larger
model back onto the GPU. In contrast, DroidSpeak can eas-
ily adapt to the available compute resources by adjusting
the number of layers to be recomputed. This enables more
possibilities for efficient scaling up or down on demand.
0.0 0.5
Prefill Latency (s)
0
20F1 score (%)
Better
2wikimqa
0.0 0.5 1.0
Prefill Latency (s)
20
40
60Code Sim
Better
repobench-p
Ours Full Prefill Full Reuse
Figure 19: Applying DroidSpeak on Mixtral-8x7B-Instruct
sharing Mixtral-8x7B’s KV cache, which are MoE models
based on Mistral model architecture.
22
24
26
Bandwidth (Gbps)
100
200
300TTFT (s)
Ours
Send reused layers
Send all layers
Re-compute
Figure 20: Pipelining re-compute and loading the reused lay-
ers greatly reduces the total TTFT compared to the baselines
of sequentially send and re-computes the KV cache.
5.7 Impact of different network bandwidth
Figure 20 compares DroidSpeak and the baselines where the
re-computation and loading of KV cache are not pipelined.
We can see that DroidSpeak outperforms baselines under
almost all bandwidth situations. Arguably, the absolute reduc-
tion in TTFT becomes smaller under high bandwidth, because
the transmission delay becomes a smaller amount of the over-
all delay when the bandwidth is very high.
6 Limitation
KV cache sharing across different foundation mod-
els: DroidSpeak as-is does not support KV cache sharing
across LLMs originating from different foundation models,
whose resulting KV caches are of different sizes. We leave
this scenario to future work.
Re-computation adaptation with network bandwidth: In
§4.3, we only consider adjusting the re-computation ratio
based on system load. Future work may extend the adapta-
tion algorithm to consider changes in network bandwidth, for
example, expanding the range of critical layer groups when
network bandwidth is limited.
Data drift in the re-computation configuration: Droid-
Speak profiles the re-computation configuration for different
models offline. However, this approach may result in quality
degradation if the real test data drifts significantly from the
“training” data used for profiling. Periodic re-profiling can be
performed to update the re-computation configuration in line
with new data. We leave this enhancement to future work.
7 Related Work
Fine-tuning: Fine-tuning LLMs for specific tasks has
gained importance, but it remains resource-intensive. Meth-
ods like parameter-efficient fine-tuning, including LoRA and
LISA [69, 79, 88, 98] reduce the memory and computation
needed for fine-tuning.
Multi-agent systems: Multi-agent systems show promise
in areas such as coding [15, 39, 40, 45, 47, 76, 82], gam-
ing [1,16,34,59,66,103,104], and social simulations [70,71].
Fine-tuned LLMs as agents improve outcomes in question
answering [14], tool learning [78], and personalization [55].
DroidSpeak focuses on reducing communication delays in
such systems.
Faster LLM serving: One line of work speeds up LoRA
model serving by hosting many LoRA models in memory
at the same time. DroidSpeak is faster than them due to the
elimination of prefill computation. Other works improve LLM
serving including better scheduling [2, 51, 56, 63, 72, 80, 106],
memory management for LoRA models [18, 79], and KV
cache offloading [31, 42, 49, 52]. All of these works are or-
thogonal and complementary to DroidSpeak.
Another closely related line of work also trades speed for
quality but uses more compact model architectures [62,77,91].
However, to smoothly adapt the amount of computation, they
need to host multiple models of different sizes in GPU at the
same time, which degrades the serving capacity in the system.
DroidSpeak does not suffer from it as it simply changes the
number of recomputed layers.
KV cache optimization: Lots of prior work has focused on
optimizing KV caches for a single model. Some work focuses
on compressing or offloading KV cache for reduced memory
and transmission costs [52, 68, 93, 94, 105]. Another line of
research reduces the prefill delay when blending non-prefix
KV caches from multiple contexts for the same model [32,97].
Since DroidSpeak focuses on sharing KV cache across mod-
els, these works are orthogonal and can be used in conjunction.
8 Conclusion
In this work, we identified the core challenge of reducing
repetitive computation in systems where multiple models
12

<!-- page 13 -->

work on a shared context. We presented DroidSpeak, a frame-
work for KV cache sharing in compound AI systems. We
identified only a subset of layers that require recomputation
to maintain quality and show the robustness of our solution
for a diverse range of model pairs, model types, and datasets.
References
[1] Saaket Agashe, Yue Fan, Anthony Reyna, and Xin Eric
Wang. Llm-coordination: Evaluating and analyzing
multi-agent coordination abilities in large language
models, 2024.
[2] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree
Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey
Tumanov, and Ramachandran Ramjee. Taming
throughput-latency tradeoff in llm inference with
sarathi-serve, 2024.
[3] Monica (Butterfly Effect AI). Manus ai agent. https:
//manus.im, 2025. Accessed: 2025-04-23.
[4] ai-dynamo. Dynamo: A datacenter scale distributed
inference serving framework. https://github.com/a
i-dynamo/dynamo, 2025. GitHub repository, release
v0.1.1, accessed 21 April 2025.
[5] AI@Meta. Llama 3 model card. 2024.
[6] Joshua Ainslie, James Lee-Thorp, Michiel de Jong,
Yury Zemlyanskiy, Federico Lebrón, and Sumit Sang-
hai. GQA: Training generalized multi-query trans-
former models from multi-head checkpoints. arXiv
preprint arXiv:2305.13245, 2023.
[7] Anysphere Inc. Cursor: The ai code editor. https:
//www.cursor.com/, 2025. Accessed: 2025-04-23.
[8] Samuel Arcadinho, David Aparicio, and Mariana
Almeida. Automated test generation to evaluate tool-
augmented llms as conversational ai agents, 2024.
[9] Muhammad Arslan, Saba Munawar, and Christophe
Cruz. Sustainable digitalization of business with
multi-agent rag and llm. Procedia Computer Science,
246:4722–4731, 2024.
[10] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu,
Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and
Juanzi Li. Longbench: A bilingual, multitask bench-
mark for long context understanding, 2024.
[11] Mohammad Shafiquzzaman Bhuiyan. The role of
ai-enhanced personalization in customer experiences.
Journal of Computer Science and Technology Studies,
6(1):162–169, 2024.
[12] Alexander Borzunov, Max Ryabinin, Artem Chu-
machenko, Dmitry Baranchuk, Tim Dettmers, Younes
Belkada, Pavel Samygin, and Colin Raffel. Distributed
inference and fine-tuning of large language models
over the internet, 2023.
[13] Gianni Brauwers and Flavius Frasincar. A general
survey on attention mechanisms in deep learning.IEEE
Transactions on Knowledge and Data Engineering ,
35(4):3279–3298, 2021.
[14] Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier,
Karthik Narasimhan, and Shunyu Yao. Fireact: Toward
language agent fine-tuning, 2023.
[15] Dong Chen, Shaoxin Lin, Muhan Zeng, Daoguang Zan,
Jian-Gang Wang, Anton Cheshkov, Jun Sun, Hao Yu,
Guoliang Dong, Artem Aliev, Jie Wang, Xiao Cheng,
Guangtai Liang, Yuchi Ma, Pan Bian, Tao Xie, and
Qianxiang Wang. Coder: Issue resolving with multi-
agent and task graphs, 2024.
[16] Jiaqi Chen, Yuxian Jiang, Jiachen Lu, and Li Zhang.
S-agents: Self-organizing agents in open-ended envi-
ronments, 2024.
[17] Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu,
Qi Liu, Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiao-
long Chen, Xingmei Wang, et al. When large language
models meet personalization: Perspectives of chal-
lenges and opportunities. World Wide Web, 27(4):42,
2024.
[18] Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo,
Luis Ceze, and Arvind Krishnamurthy. Punica: Multi-
tenant lora serving, 2023.
[19] Nuo Chen, Ning Wu, Shining Liang, Ming Gong, Lin-
jun Shou, Dongmei Zhang, and Jia Li. Is bigger and
deeper always better? probing llama across scales and
layers, 2024.
[20] Shiyang Chen, Rain Jiang, Dezhi Yu, Jinlai Xu,
Mengyuan Chao, Fanlong Meng, Chenyu Jiang, Wei
Xu, and Hang Liu. Kvdirect: Distributed disaggregated
llm inference, 2024.
[21] Xinshi Chen, Yufei Zhang, Christoph Reisinger, and
Le Song. Understanding deep architecture with reason-
ing layer. Advances in Neural Information Processing
Systems, 33:1240–1252, 2020.
[22] Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei
Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and Feng
Zhao. Agent-flan: Designing data and methods of ef-
fective agent tuning for large language models, 2024.
13

<!-- page 14 -->

[23] Zhixun Chen, Ming Li, Yuxuan Huang, Yali Du, Meng
Fang, and Tianyi Zhou. Atlas: Agent tuning via learn-
ing critical steps, 2025.
[24] Yihua Cheng, Kuntai Du, Jiayi Yao, and Junchen Jiang.
Do large language models need a content delivery net-
work? arXiv preprint arXiv:2409.13761, 2024.
[25] LMCache Contributors. Lmcache: A kv cache sharing
layer for fast distributed llm serving. https://github.c
om/LMCache/LMCache, 2024. Commit #<commit-
hash>.
[26] Sumit Kumar Dam, Choong Seon Hong, Yu Qiao, and
Chaoning Zhang. A complete survey on llm-based ai
chatbots, 2024.
[27] Chenpeng Du, Yiwei Guo, Hankun Wang, Yifan Yang,
Zhikang Niu, Shuai Wang, Hui Zhang, Xie Chen, and
Kai Yu. Vall-t: Decoder-only generative transducer for
robust and decoding-controllable text-to-speech, 2024.
[28] Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li,
and Dragomir R. Radev. Multi-news: a large-scale
multi-document summarization dataset and abstractive
hierarchical model, 2019.
[29] Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen,
Suzan Verberne, and Zhaochun Ren. A multi-agent
conversational recommender system, 2024.
[30] Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and
Marta R. Costa-jussà. A primer on the inner workings
of transformer-based language models, 2024.
[31] Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang,
Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu,
and Pengfei Zuo. Cost-efficient large language model
serving for multi-turn conversations with cachedatten-
tion, 2024.
[32] In Gim, Guojun Chen, Seung seob Lee, Nikhil Sarda,
Anurag Khandelwal, and Lin Zhong. Prompt cache:
Modular attention reuse for low-latency inference,
2024.
[33] GitHub. Github copilot. https://github.com/features/
copilot, 2025. Accessed: 2025-04-23.
[34] Ran Gong, Qiuyuan Huang, Xiaojian Ma, Hoi V o, Zane
Durante, Yusuke Noda, Zilong Zheng, Song-Chun Zhu,
Demetri Terzopoulos, Li Fei-Fei, and Jianfeng Gao.
Mindagent: Emergent gaming interaction, 2023.
[35] Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Ju-
lian McAuley. Longcoder: A long-range pre-trained
language model for code completion, 2023.
[36] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi
Chang, Shichao Pei, Nitesh V . Chawla, Olaf Wiest, and
Xiangliang Zhang. Large language model based multi-
agents: A survey of progress and challenges, 2024.
[37] Desta Haileselassie Hagos, Rick Battle, and Danda B.
Rawat. Recent advances in generative ai and large
language models: Current status, challenges, and per-
spectives, 2024.
[38] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. Constructing a multi-hop qa
dataset for comprehensive evaluation of reasoning
steps, 2020.
[39] Samuel Holt, Max Ruiz Luyten, and Mihaela van der
Schaar. L2mac: Large language model automatic com-
puter for extensive code generation, 2024.
[40] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu
Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili
Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou,
Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen
Schmidhuber. MetaGPT: Meta programming for a
multi-agent collaborative framework. In The Twelfth
International Conference on Learning Representations,
2024.
[41] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng
Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang,
Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming
Ding, and Jie Tang. Cogagent: A visual language
model for gui agents, 2024.
[42] Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu,
Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yun-
gang Bao, Ninghui Sun, and Yizhou Shan. Memserve:
Context caching for disaggregated llm serving with
elastic memory pool, 2024.
[43] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. Lora: Low-rank adaptation of large
language models, 2021.
[44] Shengran Hu, Cong Lu, and Jeff Clune. Automated
design of agentic systems, 2024.
[45] Dong Huang, Jie M. Zhang, Michael Luck, Qingwen
Bu, Yuhao Qing, and Heming Cui. Agentcoder: Multi-
agent-based code generation with iterative testing and
optimisation, 2024.
[46] Yuyang Huang, Yuhan Liu, Haryadi S. Gunawi, Beibin
Li, and Changho Hwang. Alchemist: Towards the
design of efficient online continual learning system,
2025.
14

<!-- page 15 -->

[47] Md Ashraful Islam, Mohammed Eunus Ali, and
Md Rizwan Parvez. Mapcoder: Multi-agent code gen-
eration for competitive problem solving.arXiv preprint
arXiv:2405.11403, 2024.
[48] Ziyan Jiang, Xueguang Ma, and Wenhu Chen. Longrag:
Enhancing retrieval-augmented generation with long-
context llms, 2024.
[49] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin
Liu, Xuanzhe Liu, and Xin Jin. Ragcache: Efficient
knowledge caching for retrieval-augmented generation,
2024.
[50] Shuowei Jin, Xueshen Liu, Qingzhao Zhang, and
Z. Morley Mao. Compute or load kv cache? why not
both?, 2025.
[51] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-
zalez, Hao Zhang, and Ion Stoica. Efficient memory
management for large language model serving with
pagedattention, 2023.
[52] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jae-
woong Sim. InfiniGen: Efficient generative inference
of large language models with dynamic KV cache man-
agement. In 18th USENIX Symposium on Operating
Systems Design and Implementation (OSDI 24), pages
155–172, Santa Clara, CA, July 2024. USENIX Asso-
ciation.
[53] Baolin Li, Yankai Jiang, Vijay Gadepally, and Devesh
Tiwari. Llm inference serving: Survey of recent ad-
vances and opportunities, 2024.
[54] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and
Deheng Ye. More agents is all you need, 2024.
[55] Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li,
Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing
Xu, Xiang Wang, Yi Sun, Rui Kong, Yile Wang, Han-
fei Geng, Jian Luan, Xuefeng Jin, Zilong Ye, Guanjing
Xiong, Fan Zhang, Xiang Li, Mengwei Xu, Zhijun Li,
Peng Li, Yang Liu, Ya-Qin Zhang, and Yunxin Liu.
Personal llm agents: Insights and survey about the ca-
pability, efficiency and security, 2024.
[56] Chaofan Lin, Zhenhua Han, Chengruidong Zhang,
Yuqing Yang, Fan Yang, Chen Chen, and Lili Qiu. Par-
rot: Efficient serving of LLM-based applications with
semantic variable. In 18th USENIX Symposium on Op-
erating Systems Design and Implementation (OSDI 24),
pages 929–945, Santa Clara, CA, July 2024. USENIX
Association.
[57] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang,
Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao,
Xingyu Dang, Chuang Gan, and Song Han. Awq:
Activation-aware weight quantization for llm compres-
sion and acceleration, 2024.
[58] Tianyang Lin, Yuxin Wang, Xiangyang Liu, and
Xipeng Qiu. A survey of transformers. AI open, 3:111–
132, 2022.
[59] Jijia Liu, Chao Yu, Jiaxuan Gao, Yuqing Xie, Qingmin
Liao, Yi Wu, and Yu Wang. Llm-powered hierarchical
language agent for real-time human-ai coordination,
2024.
[60] Tianyang Liu, Canwen Xu, and Julian McAuley. Re-
pobench: Benchmarking repository-level code auto-
completion systems, 2023.
[61] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray,
Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao,
Shan Lu, Ganesh Ananthanarayanan, Michael Maire,
Henry Hoffmann, Ari Holtzman, and Junchen Jiang.
Cachegen: Kv cache compression and streaming for
fast large language model serving, 2024.
[62] Zhenhua Liu, Zhiwei Hao, Kai Han, Yehui Tang, and
Yunhe Wang. Ghostnetv3: Exploring the training strate-
gies for compact models, 2024.
[63] Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi,
Dahua Lin, Bin Cui, and Zhihao Jia. Spotserve: Serv-
ing generative large language models on preemptible
instances, 2023.
[64] Microsoft Docs. Azure openai service api version
lifecycle. https://github.com/MicrosoftDocs/azure-a
i-docs/blob/main/articles/ai-services/openai/api-versi
on-deprecation.md, 2025. Accessed: 2025-04-23.
[65] Paul Mineiro. Online joint fine-tuning of multi-agent
flows, 2024.
[66] Manuel Mosquera, Juan Sebastian Pinzon, Manuel
Rios, Yesid Fonseca, Luis Felipe Giraldo, Nicanor Qui-
jano, and Ruben Manrique. Can llm-augmented au-
tonomous agents cooperate?, an evaluation of their
cooperative capabilities through melting pot, 2024.
[67] Zhaoyang Niu, Guoqiang Zhong, and Hui Yu. A review
on the attention mechanism of deep learning. Neuro-
computing, 452:48–62, 2021.
[68] Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi,
and Roy Schwartz. Transformers are multi-state rnns,
2024.
15

<!-- page 16 -->

[69] Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng
Zhang, Chi Han, and Tong Zhang. Lisa: Layerwise im-
portance sampling for memory-efficient large language
model fine-tuning, 2024.
[70] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai,
Meredith Ringel Morris, Percy Liang, and Michael S.
Bernstein. Generative agents: Interactive simulacra
of human behavior. In Proceedings of the 36th An-
nual ACM Symposium on User Interface Software and
Technology, UIST ’23, New York, NY , USA, 2023. As-
sociation for Computing Machinery.
[71] Joon Sung Park, Lindsay Popowski, Carrie Jun Cai,
Meredith Ringel Morris, Percy Liang, and Michael S.
Bernstein. Social simulacra: Creating populated proto-
types for social computing systems. 2022.
[72] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka
Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bian-
chini. Splitwise: Efficient generative llm inference
using phase splitting, 2024.
[73] Predibase. Predibase finetuning for customer service.
https://predibase.com/customer-service-automation,
Dec 2023. Accessed: 2024-12-09.
[74] PyTorch Contributors. Distributed Communication
Package - torch.distributed, 2024. Accessed: 2024-12-
10.
[75] PyTorch Team. torch.cuda.stream — pytorch 2.2 doc-
umentation. https://pytorch.org/docs/stable/generated/
torch.cuda.Stream.html, 2024. Accessed: 2025-04-25.
[76] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan
Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng
Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and
Maosong Sun. Chatdev: Communicative agents for
software development, 2024.
[77] Anthony Sarah, Sharath Nittur Sridhar, Maciej Szankin,
and Sairam Sundaresan. Llama-nas: Efficient neural
architecture search for large language models, 2024.
[78] Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming
Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei
Huang. Small llms are weak tool learners: A multi-llm
agent, 2024.
[79] Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper,
Nicholas Lee, Shuo Yang, Christopher Chou, Banghua
Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gon-
zalez, and Ion Stoica. S-lora: Serving thousands of
concurrent lora adapters, 2024.
[80] Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu,
Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, and
Ion Stoica. Fairness in serving large language models,
2024.
[81] Yifan Song, Weimin Xiong, Xiutian Zhao, Dawei Zhu,
Wenhao Wu, Ke Wang, Cheng Li, Wei Peng, and Sujian
Li. Agentbank: Towards generalized llm agents via
fine-tuning on 50000+ interaction trajectories. arXiv
preprint arXiv:2410.07706, 2024.
[82] Microsoft AutoGen Team. Autogen 0.2 documentation
- agentchat auto feedback from code execution. https:
//microsoft.github.io/autogen/0.2/docs/notebooks/age
ntchat_auto_feedback_from_code_execution, 2024.
Accessed: 2024-10-14.
[83] Athanasios Valavanidis. Artificial intelligence (ai) ap-
plications. Department of Chemistry, National and
Kapodistrian University of Athens, University Campus
Zografou, 15784, 2023.
[84] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need,
2023.
[85] vLLM Production Stack Team. vLLM Production
Stack: reference system for k8s-native cluster-wide
deployment with community-driven performance op-
timization. https://github.com/vllm- project/p
roduction-stack, 2025. GitHub repository, release
vllm-stack-0.1.1, accessed 21 April 2025.
[86] vLLM Project. AIBrix: Cost-efficient and pluggable
infrastructure components for genai inference. https:
//github.com/vllm-project/aibrix, 2025. GitHub
repository, release v0.2.1, accessed 21 April 2025.
[87] Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu
Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xu-
anzhe Liu, and Xin Jin. Fast distributed inference
serving for large language models, 2024.
[88] Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xu-
anzhe Liu, and Xin Jin. dLoRA: Dynamically orches-
trating requests and adapters for LoRA LLM serving.
In 18th USENIX Symposium on Operating Systems De-
sign and Implementation (OSDI 24), pages 911–927,
Santa Clara, CA, July 2024. USENIX Association.
[89] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang,
Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah,
Ryen W White, Doug Burger, and Chi Wang. Autogen:
Enabling next-gen llm applications via multi-agent
conversation framework. 2023.
16

<!-- page 17 -->

[90] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen
Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie
Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang,
Limao Xiong, Yuhao Zhou, Weiran Wang, Chang-
hao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue
Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng,
Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu,
Xuanjing Huang, and Tao Gui. The rise and potential
of large language model based agents: A survey, 2023.
[91] Wenhan Xia, Hongxu Yin, and Niraj K. Jha. Efficient
synthesis of compact deep neural networks, 2020.
[92] Yifei Xia, Fangcheng Fu, Wentao Zhang, Jiawei Jiang,
and Bin Cui. Efficient multi-task llm quantization and
serving for multiple lora adapters. Advances in Neu-
ral Information Processing Systems, 37:63686–63714,
2024.
[93] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian
Guo, Shang Yang, Haotian Tang, Yao Fu, and Song
Han. Duoattention: Efficient long-context llm infer-
ence with retrieval and streaming heads, 2024.
[94] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. Efficient streaming language
models with attention sinks, 2024.
[95] Yingxuan Yang, Qiuying Peng, Jun Wang, and Weinan
Zhang. Multi-llm-agent systems: Techniques and busi-
ness perspectives. arXiv preprint arXiv:2411.14033,
2024.
[96] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. Hotpotqa: A dataset for diverse,
explainable multi-hop question answering, 2018.
[97] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yi-
hua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and
Junchen Jiang. Cacheblend: Fast large language model
serving for rag with cached knowledge fusion, 2024.
[98] Kai Yao, Penglei Gao, Lichun Li, Yuan Zhao, Xiaofeng
Wang, Wei Wang, and Jianke Zhu. Layer-wise impor-
tance matters: Less memory for better performance in
parameter-efficient fine-tuning of large language mod-
els, 2024.
[99] Catherine Yeh, Yida Chen, Aoyu Wu, Cynthia Chen,
Fernanda Viégas, and Martin Wattenberg. Attentionviz:
A global view of transformer attention, 2023.
[100] Yin Song and Chen Wu and Eden Duthie. amazon/Mis-
tralLite, 2023.
[101] Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li,
Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao,
Song Yun, Xuanjing Huang, and Zhongyu Wei. Disc-
lawllm: Fine-tuning large language models for intelli-
gent legal services, 2023.
[102] Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen.
Mammoth2: Scaling instructions from the web, 2024.
[103] Ceyao Zhang, Kaijie Yang, Siyi Hu, Zihao Wang,
Guanghe Li, Yihang Sun, Cheng Zhang, Zhaowei
Zhang, Anji Liu, Song-Chun Zhu, Xiaojun Chang,
Junge Zhang, Feng Yin, Yitao Liang, and Yaodong
Yang. Proagent: Building proactive cooperative agents
with large language models. Proceedings of the AAAI
Conference on Artificial Intelligence, 38(16):17591–
17599, Mar. 2024.
[104] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong
Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu,
and Chuang Gan. Building cooperative embodied
agents modularly with large language models, 2024.
[105] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, Zhangyang
Wang, and Beidi Chen. H 2o: Heavy-hitter oracle for
efficient generative inference of large language models,
2023.
[106] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu,
Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. Dist-
serve: Disaggregating prefill and decoding for goodput-
optimized large language model serving, 2024.
17

<!-- page 18 -->

Dataset Size Med. Std. P95
hotpotQA [96] 300 10933 5160 18650
2wikimQA [38] 200 7466 3976 10705
multifieldQA_en [10] 150 8084 3849 14680
multi_news [28] 200 7624 5923 17547
lcc [35] 200 12562 9220 26066
repobench-p [60] 200 14285 8665 30376
Table 2: Size, context lengths, and evaluation metrics of
datasets in the evaluation.
A Detail statistics of the datasets
Table 2 shows the size of the datasets and the lengths of the
contexts for these datasets.
B Dynamic Re-computation Configuration
Adaptation
3436384042
00.20.40.60.81
Accuracy
Violation Rate
multifieldQACacheSync w/o adaCacheSyncFull prefill
Figure 21: DroidSpeak reduces SLO violation rate over
DroidSpeak without adaptation and full prefill. Plotted with
ultrachat-8B and fingpt model pair.
DroidSpeak initially checks the current workload inten-
sity by monitoring the running and waiting requests in the
vLLM engine [51]. If there are requests that are waiting to
be executed, it indicates that the current workload is high and
triggers an increase in the reuse ratio. In this case, Droid-
Speak chooses a re-computation configuration that has the
lowest number of re-computed layers above an accuracy tar-
get. When there are no queuing requests in the system, Droid-
Speak estimates prefill latency based on the number of tokens
of each request. The parameters for estimation can be ob-
tained through the profiling phase. DroidSpeak then selects
the highest recomputation ratio satisfying the latency SLO
for each request to maximize accuracy.
18
