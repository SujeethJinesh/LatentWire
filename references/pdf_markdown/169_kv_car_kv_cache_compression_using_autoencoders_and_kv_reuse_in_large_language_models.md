# references/169_kv_car_kv_cache_compression_using_autoencoders_and_kv_reuse_in_large_language_models.pdf

<!-- page 1 -->

KV-CAR: KV Cache Compression using A utoencoders and KV R euse in
Large Language Models
Sourjya Roy, Shrihari Sridharan, Surya Selvam, Anand Raghunathan
Elmore Family School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN
Abstract—As Large Language Models (LLMs) scale in size
and context length, the memory requirements of the key–
value (KV) cache have emerged as a major bottleneck during
autoregressive decoding. The KV cache grows with sequence
length and embedding dimension, often exceeding the memory
footprint of the model itself and limiting achievable batch sizes
and context windows.
To address this challenge, we present KV-CAR, a unified
and architecture-agnostic framework that significantly reduces
KV-cache storage while maintaining model fidelity. KV-CAR
combines two complementary techniques. First, a lightweight
autoencoder learns compact representations of key and value
tensors along the embedding dimension, compressing them before
they are stored in the KV cache and restoring them upon
retrieval. Second, a similarity-driven reuse mechanism identifies
opportunities to reuse KV tensors of specific attention heads
across adjacent layers. Together, these methods reduce the
dimensional and structural redundancy in KV tensors without
requiring changes to the transformer architecture. Evaluations
on GPT-2 and TinyLLaMA models across Wikitext, C4, PIQA,
and Winogrande datasets demonstrate that KV-CAR achieves
up to 47.85% KV-cache memory reduction with minimal impact
on perplexity and zero-shot accuracy. System-level measurements
on an NVIDIA A40 GPU show that the reduced KV footprint
directly translates into longer sequence lengths and larger batch
sizes during inference. These results highlight the effectiveness
of KV-CAR in enabling memory-efficient LLM inference.
I. INTRODUCTION
Large Language Models (LLMs) have achieved remarkable
performance across a wide range of natural language and
multimodal tasks due to their ability to capture long-range
dependencies and generate contextually rich outputs. This
capability, however, is accompanied by substantial compu-
tational and memory requirements. Modern LLMs routinely
contain billions of parameters, and empirical scaling laws
indicate that model quality continues to improve as models
grow larger—placing increasing pressure at inference time on
compute and memory subsystems.
Most state-of-the-art LLMs use a decoder-only transformer
and perform inference in two phases:prefillanddecode. In
the prefill phase, the model processes the full input prompt
and produces the key and value vectors for all tokens across
all layers. These vectors are stored in a structure known as
theKV cache. In the decode phase, tokens are generated
autoregressively: each new token attends to all previously
cached representations, and its own key and value vectors are
appended to the cache.
Although these key and value tensors could, in principle,
be recomputed at every decoding step, doing so would require
repeating the full sequence of KV projections across multiple
layers for every previously generated token. As sequence
length and batch size increase, the amount of repeated compu-
tation becomes prohibitively large. To avoid this cost, current
systems cache and retrieve the KV tensors during decoding.
While this significantly reduces compute, it introduces a major
memory burden: the KV cache grows in proportion to the
product of the sequence length, batch size and embedding
dimension of each attention layer. As a result, the KV
cache often becomes the dominant contributor to inference-
time memory consumption on GPU systems, and it directly
limits themaximum supported batch sizeand themaximum
achievable context length. In practice, the cache can reach
tens of gigabytes even for moderate sequence lengths and
batch sizes, causing out-of-memory failures despite available
compute headroom.
As LLMs scale toward longer context lengths and high-
throughput batched decoding, managing KV-cache memory
has become a central challenge for efficient inference. Several
strategies have been adopted in attempts to mitigate this bottle-
neck. KV quantization reduces precision of entries in the KV
cache, while token-pruning removes less influential tokens.
The model can be redesigned to share attention projections
across heads or layers. Offloading solutions move portions
of the KV cache to host memory at the cost of additional
latency during retrieval. While these approaches have proven
somewhat effective, the KV cache remains an ongoing bot-
tleneck with the drive towards very long context lengths.
Critically, prior approaches do not reduce the embedding
dimensionality of key and value vectors—a key factor in KV-
cache growth. This observation motivates our work, which
represents an orthogonal axis of compression that targets the
internal structure of KV representations themselves.
In this paper, we propose KV-CAR, a framework that
directly reduces KV-cache storage without requiring model
redesign or extensive training. Our contributions are threefold:
•Layer-wise autoencoder compression.We introduce
lightweight per-layer autoencoders that compress full-
dimensional key and value vectors from dimensionD
to a compact latent dimensiond≪Dbefore storage
in the KV cache. A decoder reconstructs the vectors
upon retrieval before they are used in attention com-
putation. This learned nonlinear mapping adapts to the
statistical structure of each layer, offering substantially
greater memory savings than fixed linear projections
while maintaining accuracy.
•Similarity-guided attention-head reuse.We identify
redundant attention heads across adjacent layers using an
arXiv:2512.06727v1  [cs.LG]  7 Dec 2025

<!-- page 2 -->

L1-norm similarity metric and reuse their key and value
tensors when similarity is high. This reduces the number
of stored KV tensors without degrading performance
and complements embedding compression by eliminating
redundancy at the head level.
•Lightweight autoencoder training methodology.We
design a stable training pipeline that first trains layer-
specific autoencoders independently while keeping the
base model frozen, followed by a joint optimization
stage for selected compressed layers using a hybrid
reconstruction–cross-entropy objective. This staged pro-
cess ensures convergence and preserves downstream task
accuracy.
Evaluations on the GPT-2 and TinyLLaMA models across
the Wikitext, C4, PIQA, and Winogrande datasets show that
KV-CAR achieves up to47.85%KV-cache memory reduction
with minimal impact on perplexity and zero-shot accuracy.
System-level analysis on an NVIDIA A40 GPU confirms
that shrinking the KV cache directly enableslonger context
lengthsandlarger batch sizesbefore memory exhaustion.
These results suggest that KV-CAR can significantly extend
the scalability of LLM inference on memory-constrained GPU
systems.
II. BACKGROUND
In this section, we present an overview of multi-head
attention followed by a description of the KV cache and key
parameters that impact its memory requirements.
A. Multi-head Attention
Initially developed for natural language processing (NLP),
attention mechanisms have transformed deep learning for
various modalities by enabling models to focus on key parts of
input sequences dynamically. Among these, multi-head atten-
tion (MHA) is pivotal, especially in transformer architectures,
as it allows models to attend to multiple aspects of an input
simultaneously. MHA operates through several independent
"heads," each focusing on different subspaces of the input,
thus capturing a wider range of contextual relationships.
Mathematically, MHA consists of multiple scaled dot-
product attention heads. Each head computes the attention
scores by computing dot producs between queries (Q) and
keys (K). These are scaled based on the key dimension, passed
through a softmax operation, and multiplied by the values
(V). These outputs are concatenated and linearly transformed,
yielding a final representation that integrates various depen-
dencies across the sequence. The attention formula for a single
head is given as:
Attention(Q,K,V) =softmax

QKT
√dk
!
V(1)
and for MHA:
MultiHead(Q,K,V) =Concat(head 1, . . . ,headh)WO (2)
whereW O is a learnable weight matrix. MHA enhances the
ability of models to capture both long-range dependencies and
local interactions, making it a key component of transformers
and all modern LLMs.
B. KV-cache
LLMs present significant computational challenges during
inference, primarily due to the high cost of repeated attention
computations. To mitigate this, transformer architectures em-
ploy theKey–Value (KV) cachemechanism, which stores the
key (K) and value (V) matrices for each attention layer and
reuses them in subsequent decoding steps. This caching strat-
egy eliminates redundant matrix computations for previously
processed tokens, thereby improving inference efficiency.
By reusing the stored K and V matrices, the model reduces
inference time complexity fromO(n 2)toO(n), wheren
denotes the sequence length, since attention is computed
only for newly generated tokens. However, this optimization
introduces a significantmemory overhead, as the KV cache
grows in proportion to the product of several parameters. The
total memory required to store the KV cache can be expressed
as:
KV_Cache_Size= 2×P×N layers ×d model ×L seq ×B(3)
where the factor of 2 accounts for the need to store both
the key and value tensors,Prepresents the number of bytes
per element (e.g., 2 bytes for FP16),N layers is the number of
transformer layers,d model is the embedding dimension,L seq is
the sequence length, andBis the batch size. This equation
highlights the dependency of KV cache size on sequence
length, model width, and batch size, illustrating why long-
context inference quickly becomes memory-bound.
To provide a concrete example, consider the GPT-2 Medium
model, which hasN layers = 24andd model = 1024. Assuming
FP16 precision (2 bytes), a sequence lengthL seq = 2048, and
a batch sizeB= 8, the total KV cache size is:
KV_Cache_Size= 2×2bytes×24×1024×2048×8≈1.61GB.
By comparison, the model itself has approximately 345 million
parameters (roughly 690 MB in FP16). Thus, during long-
context inference, the KV cache alone can occupy around
2.33×the memory of the model parameters. Efficient man-
agement of the KV cache is therefore crucial to balancing
inference speed, memory usage, and overall performance as
LLMs continue to scale. We discuss prior efforts to address
this challenge in the next section and place our work in their
context.
III. RELATEDWORK
Existing methods to address the KV cache bottleneck dur-
ing LLM inference include KV quantization, token pruning,
architectural changes to lower the number of KV heads, and
memory offloading techniques. We outline these approaches
and discuss how our approach differs from and complements
them below.
Quantization-Based KV Compression.Quantization re-
duces KV-cache memory by lowering the precision of stored
tensors without altering the tensor dimensions [1], [2], [3], [4].

<!-- page 3 -->

Prior work has shown that KV tensors can often be represented
at low bitwidths with minimal impact on downstream accuracy.
While useful, the push towards longer context lengths and
batch sizes that are sufficiently large to utilize the compu-
tational resources of the inference hardware platform implies
that KV cache remains a bottleneck. We note that quantization
operates purely along the precision axis and does not reduce
the dimensionality of the KV vectors themselves. Thus, our
approach is complementary to and can be combined with
quantization.
Dynamic and Attention-Based Pruning.These works
reduces KV-cache memory by pruning tokens based on their
estimated importance during decoding. Attention-driven ap-
proaches such as Keyformer [5] retain only those tokens that
account for the majority of attention mass, discarding the
rest to shrink the cache along the sequence dimension. More
recent adaptive schemes, including [6], vary pruning budgets
across layers and time by using relevance aware heuristics
to remove tokens dynamically. Other methods such as [7]
apply unstructured sparsity directly to KV entries, selecting
a compact subset of tokens that contribute meaningfully to
downstream predictions. These techniques reduce memory by
reducing the sequence length, whereas our method compresses
the representation of each token. As a result, pruning-based
strategies and our embedding-dimension compression are com-
plementary and can be combined for additional KV-cache
savings.
Architectural Modifications.Another class of techniques
reduces KV-cache memory by altering the attention architec-
ture itself. Multi-Query Attention (MQA) [8] and Grouped-
Query Attention (GQA) [9] share key and value projections
across multiple query heads, lowering the number of KV
heads that must be stored during decoding. More recent
variants [10] extend this idea across layers, allowing deeper
layers to reuse KV projections computed earlier in the network
and thereby reducing the total number of distinct KV tensors.
Other works [11] restructure the transformer into hierarchical
or pyramidal forms that allocate larger KV budgets to early
layers while aggressively compressing deeper ones. These
architectural approaches reduce memory by decreasing the
quantity of KV heads generated in the model. Our method
differs in that we compress the dimensionality of each KV
vector. As a result, our technique is compatible with these
modified architectures and provide additional benefits.
Memory Offloading techniques.These approaches address
KV-cache memory by offloading some portions to CPU or
host memory, paging segments in and out as needed during
decoding. Methods such as KV-cache paging and streaming
systems [12] maintain a working set while keeping the remain-
der in lower-cost memory, enabling longer contexts without
modifying the model. More recent works integrate token
importance estimation to guide which KV blocks remain on-
device, improving bandwidth utilization during long-sequence
generation [13]. These approaches trade increased transfer
latency for substantial memory savings. Offloading reduces
the placement cost of KV tensors rather than their size. Our
method is complementary in that the embedding-dimension
compression can be applied before offloading, decreasing both
memory pressure and the volume of data transferred across
devices.
IV. METHODOLOGY
This section outlines the KV cache compression method-
ology for two different compression techniques. Figure 1
illustrates the multi-head attention mechanism during the de-
coding phase. Each decoder layer is composed of a multi-head
attention module and a feedforward network (FFN) module,
with the same decoder structure replicated across multiple
layers in the network. Only the multi-head attention module,
which is relevant to KV cache quantization, is depicted in
Fig 1. In the decoding stage, the output of the attention
module is passed to the FFN, typically consisting of fully
connected layers, activation functions like GELU, and batch
normalization. The output of the FFN is then propagated to
subsequent layers in the network. At each time stept, the
multi-head attention module receives a decoder input of size
D×1, whereDrepresents the embedding dimension. This
input is projected through three different weight matrices:W Q,
WK, andW V , each of sizeD×D, to produce corresponding
query, key, and value vectors. These vectors are then split into
hheads, where each head has a dimension of D
h ×1.The key
vectors from the current time step are concatenated with those
from the previousl−1time steps (dimension(l−1)×D)
and then multiplied with the query vectors, followed by a
softmax operation to compute attention scores for each head.
The attention outputs are then multiplied by the current and
previous value vectors (also of size(l−1)×D). Finally,
the attention outputs from all heads are concatenated and
passed through a projection matrix of sizeD×D. This
result is normalized using a layer normalization operation,
producing an output of sizeD×lfor the multi-head attention
module. The changes in the network Architecture and the
training methodology to support the KV cache compression
is highlighted in the following subsections.
A. Network Architecture
A small modification has been introduced to the model
architecture, as shown in Fig. 1. Specifically, an encoder layer
is added after the generation of key and value tokens. This
encoder processes the key and value vectors, each originally
having dimensions of1×D, and reduces them to a lower
dimensiond, whered < D. The resulting key and value
vectors now have dimensions1×dand are stored in the
key-value (KV) cache located in the high-bandwidth memory
(HBM). At each time stept, a1×dvector is appended to the
KV cache for both key and value data structures.
When retrieving information from the cache, a decoder is
applied just before concatenating the previously generated key
and value vectors, restoring them to their original dimensions.
As a result, at each time stepl, the(l−1)×ddimensional
key and value matrices are passed through two decoders,

<!-- page 4 -->

TABLE I
COMPARISON OF KEY-VALUE(KV)CACHE COMPRESSION AND OPTIMIZATION TECHNIQUES FOR TRANSFORMER INFERENCE. OUR WORK FOCUSES ON
LEARNED EMBEDDING-DIMENSION COMPRESSION AND INTRA-LAYERKVREUSE,DISTINCT FROM STATIC,QUANTIZATION,OR
ARCHITECTURE-DEPENDENT APPROACHES.
Category Method Compression Strategy Limitations
Quantization-BasedGPTQ, AWQ [14], [1] Quantize KV tensors to int8/fp8 preci-
sion
Limited gain over fp16; sensitive to
rounding error
Architectural Com-
pression
Grouped Query Attention
(GQA) [15]
Share K/V across query heads (2–4×
reduction)
Requires architectural modification
Low-Rank / Linear
Projection
Mistral [16] Project KV tensors into a low-rank sub-
space
Fixed-rank bottleneck may reduce con-
text fidelity
Dynamic / Attention-
Based
KV Pruning [17] Drop low-attention tokens to reduce
storage
Risk of quality degradation on long-
context reasoning
Memory OffloadingKV Paging / Offloading [12] Store and prefetch KV tensors between
host and device
No compression; performance is
hardware-dependent
This Work: Learned
Compression +
Reuse
Autoencoder-based KV
compression with inter-
layer reuse
Learned embedding-dimension re-
duction plus structural KV reuse
guided by L1-distance and CE-based
finetuning
First joint approach combining KV
embedding compression with cross-
layer cache reuse
Fig. 1. Block diagram of the KV-CAR framework. For each transformer layer, key and value representations are compressed using a learned autoencoder
before insertion into the KV cache. Additionally, structurally redundant attention heads are identified and reused across adjacent layers, jointly minimizing
KV-cache storage requirements while preserving inference fidelity.
expanding them to(l−1)×D. This expanded matrix is then
concatenated with the current1×Dkey and value vectors.
The autoencoder used in this work consist of an encoder and
decoder. The encoder layer consists of two fully connected
layers, starting with theD-dimensional input. The first fully
connected layer may have a different number of neurons than
the second layer, which hasdneurons. Between these layers, a
batch normalization and a Leaky ReLU activation function are
applied. The decoder mirrors the encoder, but with the inverse
structure: it takes ad-dimensional vector as input and outputs
Ddimensions.
Additionally, a second optimization has been incorporated
into the data flow, which can be used independently or in
conjunction with the autoencoder compression. In this opti-
mization, certain key and value heads in layerNreuse key
and value heads from the previous layer N-1. The heads to be
replaced are identified by computing the L1 norm between
consecutive layers, which captures the absolute differences
between their key and value matrices. A threshold is em-
pirically determined to identify the heads that can be safely
replaced without degrading performance. This approach is
depicted in the second layer in Fig. 1, and represents an inter-
layer optimization. Ideally, replacing all the key and value
heads between consecutive layers could halve the KV cache
requirements. However, in practice, this leads to a drop in
application-level accuracy, as discussed in the Results section.
B. Training Methodology
This section describes the training methodology. The au-
toencoder training process is detailed in Algorithm 1. The

<!-- page 5 -->

Algorithm 1Finetuning Procedure for using autoencoders
1:Initialize model parametersθtaking a pretrained model
2:Set learning rateα, batch sizeB, number of epochsE
and number of layers L
3:Freeze the gradients for all the layers except the autoenc-
doer being trained
4:forlayer= 1toLdo
5:forepoch= 1toEdo
6:Shuffle training data
7:foreach mini-batchb= 1toN/Bdo
8:Sample mini-batch of data{x i, yi}B
i=1
9:Compute predictionsˆy i =f θ(xi)
10:Compute extra loss as L1 norm of real
11:and predicted key and value for layer
12:Scale the extra loss
13:Compute lossL= 1
B
PB
i=1 ℓ(ˆyi, yi)+extraloss
14:Compute gradients∇ θL
15:Update the autoencoder parameters
16:θ=θ−α∇ θL
17:end for
18:end for
19:end for
20:Return trained parametersθ
21:The individual autoencoder for all layers have been stored
separately
22:Include autoencoder in the desired layers and initialize
with the stored weights
23:Initialize model parametersθ
24:Set learning rateα, batch sizeB, number of epochsE
25:Freeze the gradients for all the layers except all the
encoders
26:Repeat the finetuning process by adding sum of scaled L1
norm loss for all encoders to the final cross entropy loss
27:Return trained parameters
training begins by initializing the weights of the autoencoders
across different layers. Starting with a pre-trained model, an
autoencoder is integrated into the key and value structures at
one layer at a time. All the weights except the autoencoder
of the layer of interest is frozen during the training process.
Key hyperparameters, such as learning rate, batch size B, and
number of epochs E, are set. For each mini-batch, predictions
are generated, and the loss is calculated based on the cross-
entropy loss of the model’s output . Additionally, an L1
norm loss is computed, reflecting the difference between
the true and predicted outputs of the autoencoder. This L1
loss, scaled by an empirical value, is added to the original
loss. This is highlighted in Algorithm 1. Backpropagation
is then performed using this combined loss, updating only
the autoencoder weights in the specific layer based on the
gradients. This process repeats for each mini-batch across all
epochs, and the entire procedure is applied iteratively for every
layer in the network having an autoencoder.
Once the initial parameters are obtained, they are loaded
into the autoencoders in the designated layers for compression.
Algorithm 2Finetuning Procedure for inter layer key and
value vector reuse
1:Collect the key and value heads for different batches over
1 epoch for different layers
2:Calculate inter layer L1 norm between heads of adjacent
layers
3:Based on L1 norm, the key and value heads in the
particular layers are reused by the key and value in the
previous layer
4:Initialize model parametersθ
5:Set learning rateα, batch sizeB, number of epochsE
and number of layers L
6:Initialize model parametersθ
7:Set learning rateα, batch sizeB, number of epochsE
8:forepoch= 1toEdo
9:Shuffle training data
10:foreach mini-batchb= 1toN/Bdo
11:Sample mini-batch of data{x i, yi}B
i=1
12:Compute predictionsˆy i =f θ(xi)
13:Compute lossL= 1
B
PB
i=1 ℓ(ˆyi, yi)
14:Compute gradients∇ θL
15:Update parametersθ=θ−α∇ θL
16:end for
17:end for
18:Return trained parametersθ
After the weights are applied, the model is fine-tuned over a
set number of epochs, freezing all layers parameters except
the autoencoder weights. During fine-tuning, the L1 norm
is recalculated between the actual and predicted values from
the autoencoder in the specific layers, scaled by an empirical
value. The sum of all the scaled L1 losses are added to the
final loss for back propagation higlighted in Algorithm 1.
Upon completing the fine-tuning process, the final autoencoder
weights can be used in the selected layers for further down-
stream zero shot tasks which are not finetuned.
The training methodology for key and value optimization
is outlined in Algorithm 2. During each training epoch,
the dataset samples are processed, and in each mini-batch
iteration, the key and value heads from different layers are
recorded. After capturing these values, the L1 norm between
consecutive layers is computed. The L1 norm is then averaged
across mini-batches. Based on this, an empirical threshold is
established to determine which key and value heads should
be replaced in their respective layers. After fixing the key and
value heads to be replaced from respective layers based on the
threshold, a fine tuning process is done on the dataset over a
certain number of training epochs. An L1 norm between the
actual KV values and the reused KV values for the key and
value data structure is computed during each mini batch. The
L1 norm loss is scaled and is added to the Cross Entropy loss
at the end. This final loss is used in the backpropagation step.
The final parameters of the model are saved at the end of
finetuning and the model with the updated parameters is used
further for zero shot downstream tasks.

<!-- page 6 -->

C. Quantization
Quantization can be applied on top of the previously men-
tioned optimization to achieve additional benefits in terms of
memory compression, as shown in Eq. 4. One approach is to
quantize the full floating-point values to int8 after compression
by the encoder. When the key and value data structures are
later retrieved from the KV cache, they can be dequantized be-
fore passing through the decoder. This process further reduces
memory requirements by compressing the values. However,
the trade-off is the overhead introduced by the quantization
and dequantization processes. This method is most beneficial
when memory compression is the primary objective.
scale= 255/(max(x)−min(x))
zeropoint=−round(scale∗min(x))−128
Xquant=round(scale∗x+zeropoint)
Xdequant= (Xquant−zeropoint)/scale(4)
V. RESULTS
This section outlines the results of our model evaluations.
We utilized two models for functional assessment: GPT-2, a
774-million-parameter model, and TinyLlama, a 1.1-billion-
parameter model. The models were evaluated on a range of
benchmarks, including WikiText, C4, PIQA, and Winogrande.
WikiText is a language modeling dataset comprising 100
million tokens curated from high-quality, featured Wikipedia
articles. C4 (Colossal Clean Crawled Corpus) is a large-scale
dataset derived from a cleaned version of the Common Crawl
web corpus. It is often used in pretraining large language
models. Due to computational limitations, we employed only
a small subset of C4 for both training and evaluation in
this study. PIQA (Physical Interaction Question Answering)
is a commonsense reasoning benchmark that focuses on the
physical world, specifically testing a model’s ability to reason
about how everyday physical tasks are accomplished. The
task consists of multiple-choice questions where the model
is required to choose the more plausible answer for perform-
ing a given task. Winogrande is a benchmark designed to
evaluate commonsense reasoning with a focus on resolving
ambiguous pronouns in sentences. It extends the original
Winograd Schema Challenge by increasing both the dataset
size and the variety of linguistic structures. Each question
presents a context followed by a pronoun reference ambiguity
and two answer options. The model must infer the correct
antecedent based on commonsense understanding.Both PIQA
and Winogrande are zero-shot tasks, meaning that the models
were evaluated without fine-tuning on these specific datasets.
The ability to generalize to these tasks demonstrates the
model’s capacity for applying pre-existing knowledge to new,
unseen scenarios. The fine-tuning for the C4 dataset is based
on the saved weights from separately fine-tuning autoencoders
on the Wikitext dataset.
A. Accuracy Evaluation
Table II presents the application-level results of the two
models across multiple benchmarks. It reports perplexity for
Wikitext and C4, and accuracy for PiQA and Winogrande,
each compared against their respective baselines. Lower per-
plexity indicates better language modeling ability, while higher
accuracy reflects stronger task performance. The reported
memory savings correspond to the reduction in KV cache size
obtained through compression of the key and value embedding
vectors.
As shown in Table II, different models and datasets exhibit
varying tolerance to compression, often with minimal loss in
performance. For the TinyLlama model, autoencoders can be
applied to up to 11 layers, compressing the key and value
vectors by a factor of two without significant change in
perplexity. In contrast, for the C4 dataset, only up to 6 layers
can be compressed before the perplexity begins to increase
noticeably.
In zero-shot evaluation tasks, PiQA maintains stable accu-
racy with compression applied to 5 layers, while Winogrande
tolerates compression across all 22 layers, achieving nearly
50% reduction in KV cache memory. Similarly, for GPT-2,
up to 10 layers can be compressed for Wikitext, PiQA, and
Winogrande with negligible accuracy loss, whereas for C4,
only 4 layers can be compressed effectively.
Overall, these results demonstrate a clear trade-off between
accuracy retention and achievable memory savings. The toler-
ance to compression depends strongly on the dataset and task
characteristics, emphasizing that optimal compression levels
should be selected based on application-specific accuracy
requirements.
The second optimization focuses on the replacement of
key and value heads within the attention layers. This tech-
nique is first evaluated independently to analyze its individual
contribution to model accuracy. Table III summarizes the
results for GPT-2 on the WikiText dataset, illustrating how
the model’s accuracy changes as the proportion of replaced
heads increases. When nearly half of the layers have their
key and value heads replaced, corresponding to around 50%
compression, a noticeable increase in perplexity is observed.
This indicates a clear trade-off between compression and
predictive quality, as excessive replacement begins to degrade
model performance.
The last three columns of Table III show cases where
only a limited number of heads—19 key heads, 25 value
heads, and 36 key–value pairs—are replaced. In these selective
configurations, replacement is guided by similarity checks
performed during the initial analysis. By targeting only the
most redundant heads, the model maintains high accuracy
while still achieving moderate compression. The perplexity
increase remains minimal, demonstrating that selective head
replacement can preserve the model’s representational capacity
even under compression.
Building on these results, we evaluate two configurations in
Table IV: (1) head replacement only, and (2) head replace-
ment combined with autoencoder-based compression. Head

<!-- page 7 -->

TABLE II
RESULTS FORGPT-2ANDTINYLLAMAEVALUATED ONWIKITEXT, C4, PIQA,ANDWINOGRANDE ACROSS MULTIPLE
METRICS. THE TABLE REPORTS BASELINE PERFORMANCE,COMPRESSED PERFORMANCE USING AUTOENCODER-BASED
KVCOMPRESSION,AND THE CORRESPONDINGKV-CACHE MEMORY SAVINGS.
Model Benchmark Metric Baseline Compressed Memory Savings
Tiny LLaMA
Wikitext Perplexity 10.29 12.33 (11 layers) 25%
C4 Perplexity 15.6874 16.02 (6 layers) 13.63%
Piqa (zero shot) Accuracy 0.6485 0.6322 (5 layers) 11.36%
Winogrande (zero shot) Accuracy 0.5241 0.513 (22 layers) 50%
GPT-2
Wikitext Perplexity 21.4 23.3 (10 layers) 41.6%
C4 Perplexity 34.61 37.3 (4 layers) 25%
Piqa (zero shot) Accuracy 0.6262 0.6055 (10 layers) 41.6%
Winogrande (zero shot) Accuracy 0.5083 0.5067 (10 layers) 41.6%
TABLE III
PERPLEXITY RESULTS FORGPT-2ONWIKITEXT UNDER
DIFFERENT LEVELS OF KEY AND VALUE HEAD
REPLACEMENT. THE TABLE REPORTS BASELINE
PERPLEXITY,COMPRESSED PERPLEXITY,AND THE
CORRESPONDINGKV-CACHE MEMORY SAVINGS ACHIEVED
BY THE REPLACEMENT STRATEGY.
Baseline Compressed heads replaced savings
21.4 30.8 All key and value 50%
21.4 26.4 all key 25%
21.4 26.4 all value 25%
21.4 21.8 19 key 6.59%
21.4 23.32 25 value 8.68%
21.4 23.9 36 key and value 12.5%
TABLE IV
GPT-2RESULTS ONWIKITEXT FOR DIFFERENT AMOUNTS
OF KEY AND VALUE HEAD REPLACEMENT,BOTH ALONE
AND IN COMBINATION WITH AUTOENCODER-BASEDKV
COMPRESSION. THE TABLE REPORTS BASELINE AND
COMPRESSED PERPLEXITY VALUES ALONG WITH THE
RESULTINGKV-CACHE MEMORY SAVINGS,SHOWING THAT
COMBINING HEAD REPLACEMENT WITH AUTOENCODER
COMPRESSION YIELDS THE HIGHEST REDUCTION IN
MEMORY FOOTPRINT.
Dataset Baseline Compressed Memory Savings
wikitext 21.4 (Perpl) 23.9 12.5% (heads)
wikitext 21.4 (Perpl) 23.9 47.85% (aut+heads)
piqa 0.6262 (Acc) 0.5892 12.5% (heads)
piqa 0.6262 (Acc) 0.5936 47.85% (aut+heads)
replacement alone provides approximately 12.5% memory
savings, while integrating autoencoders increases total savings
to 47.85% for GPT-2 on WikiText. These reductions come with
only marginal changes in perplexity or accuracy across both
WikiText and PIQA. The results show that combining selective
TABLE V
ACCURACY ONPIQAFORGPT-2ANDTINYLLAMAUNDER BASELINE,
AUTOENCODER(AE),AND AUTOENCODER+QUANTIZATION(AE+Q)
COMPRESSION.
Model / Task Base AE AE+Q
GPT-2 PIQA (10L) 0.6262 0.6055 0.6039
TinyLLaMA PIQA
(5L)
0.6485 0.6322 0.6219
head replacement with strategically placed autoencoders offers
substantially greater compression than either approach alone,
while preserving model quality relative to the baseline.
We further evaluate the impact of incorporating quantization
into this hybrid compression pipeline. In this setup, an Int8
quantization was applied using the formulation given in Eq. 4.
Table V reports the baseline performance, the autoencoder-
compressed performance, and the combined autoencoder +
Int8 quantized results. The results show that Int8 quantization
introduces only negligible accuracy degradation for both GPT-
2 and TinyLLaMA on PIQA. This demonstrates that quantiza-
tion can be effectively stacked on top of autoencoder-based KV
compression to provide additional memory reduction without
compromising model quality.
These results confirm that the hybrid approach can achieve
significant memory reduction while maintaining model accu-
racy. By selectively choosing specific heads and layers for
autoencoder placement, the model achieves efficient compres-
sion without any degradation in performance. This hybrid
compression framework can also be extended to integrate other
techniques such as quantization or pruning. Together, these
methods offer a flexible and scalable solution for enabling
memory-efficient inference across different transformer archi-
tectures.
B. System Evaluation
A NVIDIA A40 GPU was used for the system-level eval-
uation of two transformer models, GPT-2 and TinyLlama.
Figure 2 presents the maximum sequence length achievable
for different batch sizes when running GPT-2 on the A40 GPU
before encountering an out-of-memory (OOM) condition. The

<!-- page 8 -->

Fig. 2. System-level analysis of maximum achievable sequence length as a
function of batch size for GPT-2 on an NVIDIA A40 GPU under different
KV-cache compression levels. Higher compression rates enable either longer
context lengths at fixed batch size or larger batch sizes at fixed sequence
length before running out of GPU memory.
Fig. 3. System-level analysis of maximum achievable sequence length as a
function of batch size for Tiny Llama on an NVIDIA A40 GPU under different
KV-cache compression levels. Higher compression rates enable either longer
context lengths at fixed batch size or larger batch sizes at fixed sequence
length before running out of GPU memory.
four curves in the figure represent different levels of compres-
sion applied to the key–value (KV) tensors. The lowest curve
corresponds to the baseline configuration without compression,
whereas the upper curves correspond to progressively higher
compression ratios.
At a fixed batch size, applying a higher compression ratio
enables the model to process longer sequences before reaching
the GPU memory limit. Similarly, at a fixed sequence length,
increasing the compression ratio allows for a larger batch
size, leading to better GPU utilization and higher overall
throughput. For GPT-2, the results show a clear improvement:
the maximum sequence length increases by 5248 tokens for a
batch size of 64 with 75% compression, by 2752 tokens for a
batch size of 64 with 50% compression, and by 1920 tokens
for a batch size of 32 with 25% compression compared to the
baseline.
A similar trend is observed for TinyLlama, as shown
in Figure 3. With 75% compression, the model achieves a
maximum sequence length gain of 3776 tokens at a batch
size of 32. For 50% compression, the gain is 2880 tokens
at a batch size of 16, and for 25% compression, it is 1728
tokens at a batch size of 16. These results confirm that higher
KV-cache compression consistently improves the achievable
sequence length and batch size across both models, without
compromising accuracy or model quality.
VI. CONCLUSION
As LLMs continue to scale in size and context length, the
KV cache has become a dominant source of memory consump-
tion during autoregressive decoding. This growing footprint
limits both sequence length and batch size, creating practical
constraints for efficient inference on modern hardware.
To address this challenge, we proposed two complemen-
tary techniques: a learned autoencoder that compresses key
and value vectors along the embedding dimension, and a
similarity-based KV head reuse mechanism that removes re-
dundant structures across layers. These methods operate with-
out architectural modification and can be integrated directly
into existing transformer models.
Across GPT-2 and TinyLLaMA, our approach yields up to
47.85% KV-cache memory reduction with minimal degrada-
tion in perplexity or zero-shot accuracy. System measurements
further show that the reduced KV footprint translates into
longer achievable sequence lengths and larger batch sizes
before OOM on GPU hardware.
Overall, embedding-dimension compression combined with
selective head reuse offers a practical and general solution for
memory-efficient LLM inference, and can be used alongside
quantization, pruning, or architectural variants for additional
gains.
REFERENCES
[1] Ji Lin, Yuxin Tang, Zechun Yuan, and Song Han. Awq: Activation-
aware weight quantization for llm compression and acceleration.arXiv
preprint arXiv:2306.00978, 2024.
[2] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. Gptq:
Accurate post-training quantization for generative pretrained transform-
ers. InTransactions on Machine Learning Research, 2023.
[3] Ji Lin, Yuxin Tang, Zechun Yuan, and Song Han. Awq: Activation-
aware weight quantization for llm compression and acceleration.arXiv
preprint arXiv:2306.00978, 2023.
[4] Rui Xu, Xia Fan, Zhenyu Liang, Yufan Wang, Ziyu Chen, Yitan Li,
Bokai Shi, Hailong Xiao, and Gelei Wang. Rotatekv: Maximizing quan-
tization robustness for llm kv caches.arXiv preprint arXiv:2408.10417,
2024.
[5] Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J. Nair,
Ilya Soloveychik, and Purushotham Kamath. Keyformer: Kv cache
reduction through key tokens selection for efficient generative inference.
InProceedings of the 7th MLSys Conference, 2024.
[6] Hui Zeng, Daming Zhao, Pengfei Yang, Wenxuan Hou, Tianyang Zheng,
Hui Li, Weiye Ji, and Jidong Zhai. Lethe: Layer- and time-adaptive
kv cache pruning for reasoning-intensive llm serving.arXiv preprint
arXiv:2511.06029, 2025.
[7] Anonymous. Mustafar: Promoting unstructured sparsity for kv cache
pruning in llm inference.arXiv preprint arXiv:2505.22913, 2025.
[8] Noam Shazeer. Fast transformer decoding: One write-head is all you
need. InICLR, 2019.
[9] Joshua Ainslie, Santiago Ontanon, et al. Gqa: Training generalized
multi-query transformers. InInternational Conference on Machine
Learning (ICML), 2023.
[10] Samuel Brandon, Animesh Rao, Rohan Patel, and Aakash Kumar. Cross-
layer key-value sharing for memory-efficient transformer inference.
arXiv preprint arXiv:2405.06789, 2024.
[11] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming
Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. Pyramidkv: Dy-
namic kv cache compression based on pyramidal information funneling.
arXiv preprint arXiv:2406.02069, 2024.
[12] Ming Zhao, Yuhong Chen, Jie Zhou, et al. Efficient key-value cache
paging and offloading for long-context transformer inference.arXiv
preprint arXiv:2402.01030, 2024.

<!-- page 9 -->

[13] Woosuk Lee, Jaeho Lee, Juhyung Seo, and Jae Sim. Infinigen: Efficient
generative inference of large language models with dynamic KV cache
management. In18th USENIX Symposium on Operating Systems Design
and Implementation (OSDI 24), pages 93–110. USENIX Association,
2024.
[14] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. Gptq:
Accurate post-training quantization for generative pretrained transform-
ers.Transactions on Machine Learning Research, 2024.
[15] Noam Shazeer, Joshua Ainslie, Kevin Hsu, and et al. Efficient trans-
formers with grouped query attention. InInternational Conference on
Machine Learning (ICML), 2023.
[16] Yaniv Leviathan, Jérémy Lin, Daniel Bikel, et al. Mistral 7b: Open-
weights language models at scale.arXiv preprint arXiv:2310.06825,
2024.
[17] Zeyu Li, Wei Zhang, Peng Wang, et al. Dynamic kv cache pruning for
efficient transformer inference.arXiv preprint arXiv:2403.11250, 2024.
