# references/107_headinfer_memory_efficient_llm_inference_by_head_wise_offloading.pdf

<!-- page 1 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Cheng Luo 1 Zefan Cai 2 Hanshi Sun 3 Jinqi Xiao 4 Bo Yuan4 Wen Xiao 5 Junjie Hu 2 Jiawei Zhao 1
Beidi Chen 3 Anima Anandkumar 1
Abstract
Transformer-based large language models (LLMs)
demonstrate impressive performance in long con-
text generation. Extending the context length has
disproportionately shifted the memory footprint
of LLMs during inference to the key-value cache
(KV cache). In this paper, we propose HEAD -
INFER , which offloads the KV cache to CPU
RAM while avoiding the need to fully store the
KV cache for any transformer layer on the GPU.
HEAD INFER employs a fine-grained, head-wise
offloading strategy, maintaining only selective at-
tention heads’ KV cache on the GPU while com-
puting attention output dynamically. Through
roofline analysis, we demonstrate that HEAD IN-
FER maintains computational efficiency while sig-
nificantly reducing memory footprint. We evalu-
ate HEAD INFER on the Llama-3-8B model with
a 1-million-token sequence, reducing the GPU
memory footprint of the KV cache from 128 GB
to 1 GB and the total GPU memory usage from
207 GB to 17 GB, achieving a 92% reduction
compared to BF16 baseline inference. Notably,
HEAD INFER enables 4-million-token inference
with an 8B model on a single consumer GPU with
24GB memory (e.g., NVIDIA RTX 4090) without
approximation methods.
1. Introduction
Modern Large Language Models (LLMs) increasingly sup-
port extremely long inputs: Llama-3 (Dubey et al., 2024)
handles up to 128K tokens, Claude (Anthropic, 2024) sup-
ports up to 1 million tokens, while Gradient AI (Pekelis
et al., 2024) extends Llama-3 to 4 million tokens. These
extended context lengths improve performance on tasks
such as book summarization (Pal et al., 2023) and video
generation (Liu et al., 2024b), requiring millions of tokens.
1California Institute of Technology 2University of
Wisconsin-Madison 3Carnegie Mellon University 4Rutgers
University 5Microsoft. Correspondence to: Cheng Luo
<chengluo@caltech.edu>, Anima Anandkumar <an-
ima@caltech.edu>.
Preprint. Work in Progress
0 50 100 150 200
Memory Footprint(GB)
HeadInfer
4-bit KV-quant
Layer-wise Offload
8-bit KV-quant
Chunked Prefill
Baseline(BF16)
RTX-4090(24GB)
Weight Activation KV-cache
Figure 1. Estimated memory consumption of inference a Llama-3-
8B model with 1 million token on a single GPU.
As context lengths increase, memory usage and latency grow
significantly due to self-attention in transformers. To im-
prove inference efficiency (Zhou et al., 2024), most LLM
inference consists of two phases: prefill for input processing
and decoding for token generation, with key and value states
from attention cached for reuse (KV cache). However, as
the context length increases, the KV cache memory grows
rapidly, posing significant challenges for storage and effi-
ciency. For example, generating 1 million tokens using a
Llama-3-8B model requires 207 GB of memory (15 GB for
pre-trained parameters, 64 GB for activation, and 128 GB
for KV cache 1) as shown in Figure 1. This makes long-
context inference unfeasible on consumer-level GPUs such
as the NVIDIA RTX-4090 with 24 GB of memory.
To address this memory challenge, modern LLM serving
systems have introduced offloading strategies that transfer
data to CPU memory, enabling efficient LLM inference
within hardware constraints (Sheng et al., 2023; Lee et al.,
2024; Aminabadi et al., 2022). By moving portions of the
KV cache to the CPU, these methods allow for much longer
context generation than would otherwise fit in GPU memory.
However, these methods can hardly be applied to consumer-
level GPUs such as the RTX-4090, where only 8GB of
memory remain for the KV cache and activations. This is
due to two factors: (1) offloading does not reduce activation
memory usage, and (2) because offloading is applied at a
coarse granularity, the remaining KV cache on the GPU
ends up larger than the rest of the memory usage.
1The calculation is based on standard inference generation
using Hugging Face with BF16 precision and a batch size of 1.
Details of how it is calculated can be referred to in Appendix A.3.
1
arXiv:2502.12574v1  [cs.LG]  18 Feb 2025

<!-- page 2 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Layer
Head
CPU
Non-offloadKV Cache Sample Layer-offload Proposed: Head-offload
GPU
Figure 2. Demonstrations of KV cache policies in inference. Full KV cache contains two main dimensions: layer and head. Layer-wise
offloads KV cache in the layer’s dimension, with a cache budget of all heads per layer.HEAD INFER further reduces GPU memory by
adaptively reallocating cache budgets in the head’s dimension, with a cache budget of one head.
Our approach : We propose head-wise offload
(HEAD INFER ), an inference framework that drasti-
cally reduces the GPU memory needed for LLM inference
by offloading the KV cache at the level of individual
attention heads. The key idea is to leverage attention head
independence to decompose the attention computation in a
head-wise manner, requiring only one head of the KV cache
on the GPU at a time and offloading the rest to the CPU.
As shown in Figure 2, HEAD INFER frees the GPU system
from storing the KV cache of the entire model or layer and
instead stores only a single head at any given time. This
fine granularity slashes the GPU memory footprint while
maintaining exact mathematical equivalence.
To enable efficient long-context inference on consumer
GPUs, HEAD INFER integrates several key optimization
techniques: adaptive heads grouping, chunked prefill, and
asynchronous data transfer. Adaptive head grouping dy-
namically adjusts the number of attention heads retained
on the GPU, progressively reducing the memory footprint
as the context length increases. Chunked prefill reduces
peak memory usage by splitting long input sequences into
smaller segments, processing them incrementally to avoid
excessive activation memory consumption. Asynchronous
data transfer overlaps KV cache movement between GPU
and CPU and ensures computation proceeds without stalls
from memory transfers.
Through roofline analysis (Williams et al., 2009), we demon-
strate that HEAD INFER maintains high compute efficiency
while significantly reducing GPU memory consumption. We
implement HEAD INFER on the Hugging Face framework
and evaluate it on representative LLMs, including Llama,
Qwen, Mistral, and Gemma, with varying model sizes and
sequence lengths. More than the 1 million token inferences
shown in Figure 1, our results demonstrate thatHEAD INFER
extends the Llama-3-8B model’s context length from 25K
(standard inference) and 45K (standard offload) to 4 mil-
lion tokens, achieving around 100x improvement in context
length extension using NVIDIA RTX 4090.
In summary, this paper presents the following contributions.
• HEAD INFER enables inference with context length with
over a million tokens on a single consumer GPU like
NVIDIA RTX 4090.
• In-depth analysis using the roofline model: high com-
pute efficiency is achieved during prefill without shifting
computation into the memory-bound regime.
• Fully general and implementation-agnostic attention:
HEAD INFER supports dense as well as sparse attention,
and it works with head-wise sparse attention such as duo-
attention (Xiao et al., 2024a).
• Support for massive model inference:HEAD INFER collab-
orates with pipeline parallelism to support larger models
like Llama3-70B with 1 million tokens.
• Easy-to-use and portable, requiring minimal code changes
to the existing inference frameworks. The code example
can be referred to in Appendix E.
2. Related Work
2.1. Generative Inference and KV Caching
Generative LLM inference typically involves two main
stages: the prefill stage and the decoding stage. In the
prefill stage, the model processes the initial input prompt
by computing attention scores for all tokens in the input
sequence. For long-context input, it is common to adopt
chunked prefill (Agrawal et al., 2024), which divides the
prompt into fixed-length chunks to incrementally build the
KV cache. This strategy significantly reduces peak memory
usage by lowering linear layers’ peak intermediate activation
size from the entire sequence to just the smaller chunk. In
the subsequent decoding stage, each newly generated token
from the prefill stage is fed back into the model, creating an
autoregressive generation process. The LLM produces one
new token at a time, and each token attends to all previous
KV cache.
2

<!-- page 3 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
2.2. Lossy KV Cache Management
Evit KV cache can reduce memory usage and computa-
tional complexity. One direction is to identify and retain
only the most ’valuable’ tokens within the KV cache. Repre-
sentative methods include Sliding Window Attention (Belt-
agy et al., 2020), Heavy Hitter (Zhang et al., 2023), and
StreamingLLM (Xiao et al., 2024b). Another direction is to
identify and retain the attention heads. Wu et al. (Wu et al.,
2024) find a way to evaluate the importance of attention
heads. Head-wise sparsity such as duo-attention (Xiao et al.,
2024a), HeadKV (Fu et al., 2024), and Razorattention (Tang
et al., 2024) start to divide up KV cache budgets based on
the importance of each head, which is usually determined
by the need for retrieval or reasoning. Minference (Jiang
et al., 2024a) takes this idea further by applying distinct
sparse patterns to different heads.
2.3. Offloading KV Cache
Offloading the KV cache from the GPU memory to CPU
DRAM is another memory-efficient strategy. For instance,
LayerKV (Xiong et al., 2024) implements efficient layer-
wise KV offloading and overlapping data transfers to im-
prove the context length. FastDecode (He & Zhai, 2024)
and NEO (Jiang et al., 2024b) also offload parts of the KV
cache to the CPU and perform attention computations on
the CPU. ShadowKV (Sun et al., 2024) combines SVD
decomposition with offloading to reduce communication
overhead. FlexInfer (Xu et al., 2024) introduces the vTen-
sor abstraction to better manage heterogeneous memory
resources. Infinigen (Lee et al., 2024) introduces dynamic
KV cache management with offloading systems.
3. HEAD INFER : Head-wise Offload
3.1. Background:
Regular Inference with KV cache Generation. At each
time step t, the input token xt is transformed into an embed-
ding vector E(xt) by embedding the token. Then, a linear
projection generates the key Kt and the value Vt, which can
be written as follows:
Kt = WKE(xt), Vt = WV E(xt) (1)
Here WK and Vt are projection weights, and the current Kt
and Vt are appended to the existing key cache Kcache and
value cache Vcache.
Kcache = [K1, K2, ..., Kt], Vcache = [V1, V2, ..., Vt] (2)
The state of the cache can be memory-intensive. Together
with sequence length S and hidden length D, Kt, Vt ∈
RS×H. This takes 2 × S × D memory.
Attention with KV Cache. During the computation of
self-attention at time step t, the model utilizes the entire
sequence of keys and values from time steps 1 to t (stored
in the KV cache) alongside the new query Qt:
At = Sof tmax( QtK T
cache√dk
)Vcache (3)
where dk is the dimensionality of the keys.
Offload KV Cache.
When the context length S grows on a million scale, or
the GPU’s on-device High-Bandwidth Memory (HBM, or
MHBM ) is small on the consumer GPU, it may become
insufficient to store the entire key-value cache. In such sce-
narios, KV cache offloading provides a practical approach
to handling memory limitations. Offloading can temporarily
move parts of the KV cache to CPU RAM or other external
resources (e.g., NVMe storage and CPU disk). However,
each offloading strategy introduces potential communica-
tion overheads that must be carefully weighed against the
gains in usable context length.
For a batch of size B, a transformer with L layers, and the
KV cache with the bytes per element sizeof (datatype),
the total KV cache size is:
Size KV cache = 2 × B × L × S × D × sizeof (datatype)
(4)
If Size KV cache > M HBM , the system can offload some
portion of the KV cache to external memory to avoid out-of-
memory errors. Let α(0 ≤ α ≤ 1) be the fraction of the KV
cache that remains on the GPU, and 1 − α be the fraction
offloaded to external memory. The memory footprint on the
GPU Size on−GP U can be expressed as follows:
Size on−GP U = α × Size KV cache (5)
Therefore, we require:
Size on−GP U ≤ MHBM , α ≤ MHBM
Size KV cache
(6)
3.2. Head-wise Offload (HEAD INFER )
Head-wise KV Cache Generation. In transformer architec-
tures, each self-attention layer is split into multiple attention
heads H. Each head has its own set of key-value (KV)
tensors:
K (h) ∈ RS×Dh , V (h) ∈ RS×Dh (7)
where D is divided by H so that each head handles a sub-
dimension Dh = D/H. Therefore, instead of a single
large KV cache, a head-wise approach organizes cache inH
separate memory spaces. Formally, at time step t, we have:
K (h)
cache = [K (h)
1 , ..., K(h)
t ], V (h)
cache = [V (h)
1 , ..., V (h)
t ] (8)
3

<!-- page 4 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Standard Inference Chunked Prefill Layer-wise Offload HeadInfer
H
S L
H
S L
H
S L
H
S L
H
S L
H
S L
H
S L
Figure 3. Granularity of different methods. Each cube represents the entire attention process along three dimensions: Sequence (S), Layers
(L), and Heads (H). Standard inference puts everything on the GPU. Chunked-prefill fetches only a part of the sequence dimension of all
tokens on the GPU at a time. Layer-wise offloading fetches a subset of layers on the GPU, offloading the rest. HEAD INFER introduces an
even finer approach that maintains only selected heads within a layer.
As a result, each head stores its keys and values in a contigu-
ous memory space, enabling selective offloading of certain
heads’ cache when memory constraints emerge.
During self-attention at the time step t, we calculate the
attention output for each head h independently, using:
A(h)
t = Sof tmax( Q(h)
t K T
cache(h)
√dk
)Vcache(h) (9)
Finally, the outputs A(h)
t
H
h=1 are concatenated to form the
final output of attention for that layer.
Head-wise Offload. Since the attention computation has a
head-wise independence, if we can keep the KV cache of
a single head rather than the entire layer, then the memory
consumption can be reduced substantially. This leads to our
proposed head-wise offload (HEAD INFER ) strategy.
HEAD INFER is designed to minimize the fraction of on-
GPU data α (the fraction of the total KV cache stored on
GPU). Using Llama-3-8b as an example, we define Hall as
the total number of attention heads of a given model, which
equals the number of heads per layer times the number of
layers H × L. Hon is the number of heads h retained on the
GPU, and Hof f is the number of heads offloaded to external
memory (CPU); obviously, we have Hon + Hof f = Hall
Define α as the fraction of the KV cache that remains on
the GPU. We can store all KV cache on the GPU if α = 1
or layer-wise offload if α = 1/L. In our head-wise scheme:
α = Hon
Hall
= 1
L × H (10)
Here we keep only a single head on the GPU, and the frac-
tion of the total KV cache that occupies GPU memory is
reduced by α = 1/(L × H), with a total size of:
Son−GP U = 2 × B × S × DH × sizeof (datatype) (11)
By reducing α, we can preserve GPU memory capacity
for extended context inference and make the million-level
inferences on consumer GPUs possible.
3.3. Granularity: Sequences, Layers, and Heads
HEAD INFER differs from traditional offload in terms of
granularity. When deploying large language models, each
dimension of the model can become a potential bottleneck
in GPU memory. Naively offloading the entire KV cache
or entire layers can be too coarse, leading to suboptimal
memory usage.
As shown in Figure 3, HeadInfer addresses this challenge
by introducing a hierarchical set of techniques, including
chunked-prefill, layer-wise offload, and head-wise offload
that operate at increasingly fine-grained levels of sequence
(S), layers (L), and heads (H).
• Chunked-Prefill (S) : Rather than processing all se-
quences of input tokens at once, HEAD INFER divides
the sequence into smaller chunks, each processed sepa-
rately during the prefill stage. This partition helps reduce
the activation memory usage.
• Layer-Wise Offload (L): Instead of storing the entire KV
cache on the GPU, HEAD INFER offloads it to the GPU
and fetches the KV cache of specific layers on demand.
Consequently, only the relevant portion of the KV cache
resides on the GPU at any given time.
• Head-Wise Offload (H): Within each layer, HEAD IN-
FER can selectively offload the KV cache for all attention
heads, fetching certain attention heads on demand. This
offers the finest granularity by focusing on a subset of
heads within each layer.
By combining these techniques, HEAD INFER allows precise
control over which parts of the activation and KV cache
remain on the GPU. Because these dimensions nest naturally
as chunks are subsets of the sequence, layers are subsets
of the model, and heads are subsets of the attention layer;
HEAD INFER can flexibly adapt to a wide range of hardware
constraints, such as available GPU memory, while meeting
performance requirements.
4

<!-- page 5 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
GPU Model State
#Layer 0
KV Cache
#Layer n
GPU Model State
P0~nP0~n
CPU
Parameter0(P0)Parameter0(P0)
Pn
Non-offload: HeadInfer:
……
…
…
Memory
SavedLayer0
Head0
(H0,0)
Layer0
Head0
(H0,0)
H0,1H0,1 H0,jH0,j
Hn,0Hn,0 Hn,1Hn,1 … Hn,jHn,j
Hn,k-1 Hn,k
PCIEPCIE
…H0,0H0,0 H0,jH0,j
…Hn,0Hn,0 Hn,jHn,j
K-1,K∈[0,j]
…Hn,1Hn,1…Hn,1
…H0,1H0,1…H0,1
Figure 4. HEAD INFER snapshot. All parameters are stored on the
GPU. Head-wise partitioned KV cache is moved across GPU and
CPU with the ping-pong memory.
4. HEAD INFER for Memory-Efficient
Inference
For the offloading strategy, the potential communication
overheads must be carefully considered against the gains
in context length extension. This is because the offloading
communication cost grows along with the context length.
Fortunately, for this, HEAD INFER can overlap the commu-
nication cost with evening growing attention computation.
This is achieved by a ping-pong memory design (Figure 4)
for asynchronous offloading and prefetch. We present the
implementation on algorithm 1.
Overlapping Head-wise KV Cache Movement and GPU
Compute. The overlapping mechanism is critical for of-
fload performance, especially for long-context inference
when the computation time can completely overlap the of-
fload communication. While the GPU computes attention
for the current head A(h)
t , it concurrently prefetches the next
head’s KV cache from the CPU and offloads the current
head’s cache back to the CPU. Ping-pong memory enables
non-blocking PCIe transfers to synchronize memory move-
ment with computation. This mechanism is presented in
Algorithm 1 lines 10-11 and 25-26, as async prefetch and
async update.
Efficient Management of KV Cache. Effective KV cache
management guarantees the feasibility of long context head-
wise offload. Key strategies here include head-wise parti-
tioning and pre-allocation. Head-wise partitioning makes
sure that each head has its own KV cache K (h), V (h), and
allows selective offloading and retention based on memory
availability and head importance. HEAD INFER pre-allocates
the CPU’s KV cache memory and the GPU’s ping-pong
memory to avoid runtime memory allocation overheads.
Adaptive Head-wise Offloading. Adaptive head-wise of-
floading reduces kernel launch overheads caused by head-
wise partitioning, especially for small context sizes. This
involves fusing multiple attention heads into a single ker-
nel, reducing the number of kernel launches at the cost of
larger KV cache fetches. For instance, we fuse all heads
when processing the LLAMA3 inference 1-500K context,
a process that is identical to the standard attention calcula-
tion. We divide the attention heads into two groups when
processing 500K-1M, four groups when processing 1M-2M,
and eight groups when processing 2M-4M. Specifically for
eight groups, HEAD INFER performs head-wise offload with
finest granularity, where only one head is left on the GPU.
Extension: Combining with Head-wise Sparsity. HEAD -
INFER is compatible with existing sparse optimization tech-
niques. Our work focuses on integrating HEAD INFER
with head-wise sparsity, such as duo-attention (Xiao et al.,
2024a). This head-wise sparsity reduces memory by trun-
cating less important heads to a fixed length (typically under
1k). We designate these heads as on-GPUHon heads without
offloading since they consume minimal GPU memory, and
offloading KV cache smaller than 1k lead to performance
degradation, as analyzed in section 5. The detailed design
for the extension can be found in Appendix D
Algorithm 1 HEAD INFER : Head-Wise Inference
Require: Transformer model with L layers, H attention heads
per layer, input sequence length S
# Initialization:
1: Allocate GPU memory for Hon heads.
2: Allocate CPU memory for Hoff heads.
# Prefill Phase:
3: for each chunk of input tokens S do
4: for each layer l in Transformer Layers L do
5: for each head h in H do
6: Compute key K (h), value V (h) on GPU
7: if h ∈ Hon (GPU-resident heads) then
8: Update GPU KV cache: K (h)
GPU, V (h)
GPU
9: else
10: Async Prefetch next KV cache K (h+1), V (h+1)
11: Async Update CPU KV cache: K (h)
CPU, V (h)
CPU
12: end if
13: Compute attention output A(h)
t on GPU
14: end for
15: Concatenate A(h)
t for all heads to form layer output
16: end for
17: end for
# Decoding Phase:
18: for each token t to be generated do
19: for each layer l in Transformer Layers L do
20: for each head h in H do
21: Compute key K (h), value V (h) on GPU
22: if h ∈ Hon (GPU-resident heads) then
23: Update GPU KV cache: K (h)
GPU, V (h)
GPU
24: else
25: Async Prefetch next KV cache K (h)
CPU, V (h)
CPU
26: Async Updated CPU KV cache: K (h)
CPU, V (h)
CPU
27: end if
28: Compute attention output A(h)
t on GPU
29: end for
30: Concatenate A(h)
t for all heads to form layer output
31: end for
32: end for
5

<!-- page 6 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
10
3
10
3
10-310-3
Performance[TFLOP/s]
Arithmetic Intensity[FLOP/Byte]
Roofline Analysis with RTX-4090
10
2
10
2
10
1
10
1
10
0
10
0
10
-1
10
-1
10
-2
10
-2
10-210-2 10-110-1 10 010 0 10 110 1 10 210 2
Decode
Prefill 1k Prefill 10k
Prefill 1k
HeadInfer/Offload
Prefill 10k
HeadInfer/Offlload
Saturation: BF16 Tensor Core(165.2 GFLOPs)
Decode HeadInfer(Overlapped with Decode Offload) Offload
BaselineBaseline
HeadInferHeadInfer
10
3
10-3
Performance[TFLOP/s]
Arithmetic Intensity[FLOP/Byte]
Roofline Analysis with RTX-4090
10
2
10
1
10
0
10
-1
10
-2
10-2 10-1 10 0 10 1 10 2
Decode
Prefill 1k Prefill 10k
Prefill 1k
HeadInfer/Offload
Prefill 10k
HeadInfer/Offlload
Saturation: BF16 Tensor Core(165.2 GFLOPs)
Decode HeadInfer(Overlapped with Decode Offload) Offload
Baseline
HeadInfer
Figure 5. Flashattention in the roofline plot analysis using the RTX-4090 device setting.
5. Analysis
Although KV cache offload is proposed to reduce memory
usage, it remains an open question whether offloading harms
overall performance, especially when the context lengthS is
large and works with chunked-prefill. This section analyzes
the theoretical peak performance for a given GPU under
constrained high-bandwidth memory (HBM) and peripheral
component interconnect express (PCIe).
Performance Model. We consider a GPU characterized
by HBM capacity MHBM , memory bandwidth Bmem, and
compute throughput FGP U (measured in FLOPS). We also
incorporate the slower PCIe bandwidth Bpcie into the per-
formance model to account for the offload.
Memory Bound vs. Compute Bound. GPU operators can
be classified as compute bound or memory bound, which is
determined by the time spent in arithmetic operations and
the time spent accessing HBM. Two primary bottlenecks
define the system’s performance regime:
• Memory-Bound: When Bmem (memory bandwidth) is
insufficient to transfer the KV cache quickly enough, in-
ference operates below the GPU’s peak FLOPS capacity.
• Compute-Bound: When Bmem is sufficient, high compute
efficiency is achieved and the throughput is determined
by the GPU’s peak computation rateFGP U.
Peak Performance. Assume, for simplicity, that we can
approximate the per-token inference time T as follows:
T ≈ max(Tcomp, Tmem) (12)
During the prefill stage, Tcomp ∝ D×S2×L×H
FGP U
captures
the compute part, and Tmem ∝ D×S×L×H
Bmem
captures the
memory part. When scaling S, Tcomp grows faster than
Tmem, associated with the quadratic growth in sequence
length, which makes compute throughput the limiting factor.
During the decoding stage, each new token must attend
to all previously generated tokens. Generating a single
token incurs costs is roughly: Tcomp ∝ D×S×L×H
FGP U
and
Tmem ∝ D×S×L×H
Bmem
. When scaling S, Tcomp grows lin-
early as Tmem, but Bmem, Bpcie are usually slower than
FGP U, making the memory bandwidth the main bottleneck.
Roofline Model. The roofline plot typically displays a ker-
nel’s computation throughput and arithmetic intensity, pro-
viding a visual representation of its hardware performance.
We present the roofline analysis for the FlashAttention ker-
nel (Dao et al., 2022; Dao, 2023; Shah et al., 2024) executed
on an RTX-4090 (Figure 5), with details in Appendix C.
We believe that the roofline model can analyze both GPU
performance and heterogeneous systems performance.
Our key observations from roofline analysis are:
• Prefill (compute-bound behavior). ”Prefill HEAD IN-
FER/Offload” has higher arithmetic intensity than ”Prefill”
as it only offloads KV cache, and the arithmetic intensity
grows as context length increases. For context lengths
S ≥ 10k, prefill remains compute-bound, even when of-
floading via slower PCIe bandwidth. In contrast, for short
context lengths (S ≤ 1k), “HEAD INFER /Offload” shifts
to the memory-bound regime, leading to offload-induced
slowdowns. The turning point is achieved at 2K, and 10K
can ensure high computational efficiency.
• Decoding (memory-bound behavior). Decoding perfor-
mance is primarily memory-bound. Consequently, rely-
ing on PCIe bandwidth during offloading substantially
degrades overall throughput.
• HEAD INFER (unchanged behavior). HEAD INFER ’s
head-wise offloading during both chunked-prefill and de-
coding does not alter the position of the roofline plot due
to the independence of the attention heads. Although head-
wise computation impacts GPU utilization, the adaptive
head strategy can preserve performance.
6

<!-- page 7 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Table 1. Performance(benchmark score) of different methods on LongBench v2 on a single RTX-4090 GPU, under different task
difficulties (Easy/Hard) and context lengths (Short/Medium/Long). Overall performance is average scores on all questions.
LongBench V2 Overall Easy Hard Short Medium Long
Standard 25K 28.4 30.2 27.3 33.9 25.1 25.9
Chunked Prefill 30K 28.2 27.1 28.9 32.8 25.6 25.9
Layer-wise offload 45K 29.0 29.2 28.9 36.1 24.2 26.9
HEAD INFER 1024K 30.2 31.2 29.6 33.9 27.0 30.6
Table 2. Performance(benchmark score) of different methods on SCBench on a single RTX-4090 GPU. kv and prefix-suffix are string
retrieval in key-value and prefix-suffix scenarios. vt is variable tracking. qa-chn, qa-eng, and choice-eng are English/Chinese question
answering. mf is finding the math answer. many-shot is finding multiple shots in context. summary is document summarization.
SCBench kv prefix-suffix vt qa-chn qa-eng choice-eng mf many-shot summary
Standard 25K 15.8 9.6 4.6 9.4 13.3 36.5 2.6 16.3 32.3
Chunked Prefill 30K 21.4 10.4 6.9 9.4 15.5 38.6 2.2 25.2 33.5
Layer-wise offload 45K 22.6 12.8 8.4 10.4 15.7 37.8 2.2 25.9 33.6
HEAD INFER 1024K 28 17.2 42 11.9 23.0 59.8 9.4 25.9 37.1
6. Performance Evaluation
6.1. Experimental Setup
We conduct our experiments on RTX-4090 GPUs, which
are configured with 24.5 GB of HBM3, 4 × AMD EPYC
7B13 with 64 CPUs each (a total of 256 cores), and 2× Gen4
NVMe of 512 GB each and 1 TB of DDR5 RAM in total.
The GPUs are independently connected to the host with
16 PCIe 4.0 interfaces, providing 25.0 GB/s unidirectional
D2H and H2D throughput for pinned host memory.
6.2. Long-Context Benchmarks
We evaluate HEAD INFER using the LongBench v2 (Bai
et al., 2024) and SCBench (Li et al., 2024) benchmarks, and
other long-context benchmark results such as Needle-in-a-
Haystack (NIAH) (Kamradt, 2023) and Ruler (Hsieh et al.,
2024) are shown in the Appendix A.2. We use the Llama-
3-8B-Instruct-Gradient-1024k model, which supports up to
1 million context lengths. All lossless methods, including
standard inference, chunked prefill, layer-wise offload, and
HEAD INFER , are running on a single RTX-4090 GPU, with
a maximal context length achieved within 24 GB. The long-
context benchmark results aim to reveal the performance
gain with HEAD INFER ’s context length extensions.
LongBench v2 is a comprehensive suite of long-context
datasets designed to assess long-context problems that re-
quire deep understanding and reasoning. It comprises 503
difficult multiple-choice questions within two difficulties,
including ”Easy/Hard” and word lengths, including ”Short”
of 0-32k words, ”Medium” of 32k-128k words, and ”Long”
of 128k-2M words. As shown in Table 1, we measure their
overall scores for different task categories. HEAD INFER
achieves the highest benchmark score for ”Medium” (27.70)
and ”Long” (30.6). Layer-wise offload performs best on
”Short” (36.1) for truncation side effects.
SCBench is a comprehensive suite of datasets encompass-
ing multiple tasks, designed to assess long-context under-
standing capabilities. It contains subsets with different
prompt lengths up to 500K tokens and 227K on average.
As shown in Table 2, we measure their overall scores for
different task categories. HEAD INFER outperforms all other
methods on all 9 tasks, with superior 1M context length.
6.3. Efficiency Results of Memory and Throughput
We evaluate the maximum context length supported under
RTX-4090 memory constraints, as well as prefill/decoding
throughput. Our experiments use the Llama-3-8B, Llama-3-
70B, Mistral (Jiang et al., 2023), Qwen (Bai et al., 2023a),
and Gemma-2 (Team et al., 2024) models. The default
number format for weights and activations is BFloat16, and
4-bit KV-quant is deployed with KIVI (Liu et al., 2024c).
The chunk size is set to 10K based on our roofline analysis.
LLM Inference on Consumer GPUs with 24G memory.
We measure the GPU memory consumption of Llama3-8B
inference with HEAD INFER and 1 million context length.
HEAD INFER uses 17GB during prefill and 16.4GB during
decoding; in contrast, other methods are unable to run at
this context scale using 24GB RTX-4090. Accordingly, we
measure the maximum achievable context length to assess
memory efficiency. As shown in Table 3, HEAD INFER
outperforms other system optimization approaches, scaling
from thousand-level contexts (10K-70K) to million-level
contexts (1,024K–4,200K). Compared to layer-wise offload,
HEAD INFER can maintain stable activation memory with
chunked prefill and minimize KV cache GPU memory with
head-wise offload. Note that with HEAD INFER , the maxi-
mum context lengths for Llama3, Llama2, and Mistral are
constrained by CPU RAM (512GB for KV cache), while
the other methods are limited by GPU memory. This means
we can use larger CPU RAM or offload to disk for a more
extended context. We leave this exploration for future work.
7

<!-- page 8 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Table 3. Comparison of maximum context length with system optimization methods on various models inference. All experiments within
this table run on a single RTX-4090 GPU with 24GB and AMD EPYC 7V12 CPUs with 1TB memory.
Context Length Llama-3-8B Llama-2-7B Mistral-7B Qwen2-7B Gemma-2-9b
Standard 25K 10K 30K 35K 10K
Chunked Prefill 30K 20K 40K 70K 10K
4-bit KV-quant 45K 30K 40K 50K 20K
Layer-wise offload 45K 60K 45K 50K 35K
HEAD INFER 4096K 1024K 4096K 4200K 1300K
Table 4. Llama3-70B Inference with long context input.
Context Length Llama-3-70B
Standard 10K
HEAD INFER + 10k chunk size 950K
HEAD INFER + 5k chunk size 1000K
Scaling up to Llama3 70B Architecture. Scaling ability to
70B models is a key factor for demonstrating ifHEAD INFER
is effective for larger-scale model inference. We evaluate
HEAD INFER on the Llama-3-70B model. The model infer-
ence is conducted using pipeline parallelism (Narayanan
et al., 2019) across 8 RTX-4090 GPU nodes. As shown in
table 4, HEAD INFER achieves efficient inference of 1 mil-
lion sequences with the 70B model, outperforming standard
methods. Additionally, due to Llama-3-70B’s higher arith-
metic intensity compared to Llama-3-8B, we can employ a
smaller chunk size without performance degradation. This
reduced chunk size allows for better memory utilization,
enabling increased KV cache allocation and extending the
maximum context length from 950k to 1M.
Prefill and Decoding Throughput of HEAD INFER We
evaluate the prefill and decode throughput of the inference
Llama-3-8B model using HEAD INFER with adaptive head
grouping. Our HEAD INFER implementation achieves 516
tokens/second during prefill for a 1 million-token input and
0.15 tokens/second during decoding with a 1 million-token
KV cache. A context length of 1 million cannot be achieved
by other methods. For comparison, with a 20K-token input,
HEAD INFER achieves 7210 tokens/second prefill and 6 to-
kens/second decoding. Standard inference achieves 7235
tokens/second prefill and 33 tokens/second decoding, while
layer-wise offloading matches HEAD INFER at 7210 token-
s/second prefill and 6 tokens/second decoding. Additional
latency results for different context lengths and different
methods are provided in Appendix A.1.
7. Ablation Study
How do different levels of granularity affect memory?
To evaluate the impact of different levels of granularity on
memory, we conducted a detailed ablation study using the
Llama-3-8B model. From the memory perspective, three
Table 5. Ablation study of HEAD INFER on Llama-3-8B.
Context Length Llama-3-8B
Standard 25K
Layer-wise Offload 45K
Chunked Prefill 30K
HEAD INFER Head=8 Group = 1 550K
HEAD INFER Head=4 Group = 2 1100K
HEAD INFER Head=2 Group = 4 2100K
HEAD INFER 4096K
dimensions of sequences S, layer L, and head H are man-
aged through chunked-prefill, layer-wise offloading, and
head-wise offloading, respectively. As shown in Table 5,
HEAD INFER with S × L × H supports a 4096K context
length, far exceeding the 25K of standard methods, 45K
of layer-wise offload with L, and chunked prefill with S.
Adjusting head-group granularity shows a trade-off, with
context lengths ranging from 2100K to 550K. Note that
HEAD INFER with (Head=8 Group=1) works the same as
layer-wise offloading combined with chunked-prefill for
S × L. HEAD INFER achieves 8× context extension than
(Head=8 Group=1), showing its memory efficiency.
8. Conclusion
In this paper, we introducedHEAD INFER , a novel head-wise
KV cache management framework designed to enable effi-
cient long-context large language model (LLM) inference
on consumer GPUs. HEAD INFER dynamically offloads
KV cache components to CPU memory, employing head-
wise and asynchronous offloading strategies. Our Roofline
analysis highlights HEAD INFER ’s ability to preserve GPU
computational efficiency while reducing memory footprint,
making long-context inference feasible on consumer-grade
GPUs. Our evaluations demonstrated HEAD INFER ’s abil-
ity to achieve over 1-4 million context tokens on a single
consumer GPU with mathematical equivalency.
We hope that our work will inspire future research in
memory-efficient inference from the perspective of head-
wise offload. We believe thatHEAD INFER will be a valuable
tool for the community, enabling the inference of long con-
text on consumer-grade hardware with limited resources.
8

<!-- page 9 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Impact Statement: Democratizing Access to
Advanced AI
Artificial Intelligence (AI) has the potential to transform in-
dustries, revolutionize education, and empower individuals.
However, the deployment of cutting-edge models, particu-
larly large language models (LLMs), is often hindered by
significant resource requirements, creating barriers to entry
for smaller organizations and underserved communities.
In this work, we introduce HEAD INFER , a memory-efficient
inference framework designed to bridge this gap. By lever-
aging head-wise offloading strategies, HEAD INFER enables
resource-constrained devices to process unprecedentedly
long contexts, achieving capabilities typically reserved for
high-performance systems. For instance, HEAD INFER al-
lows a consumer-grade GPU to handle over 1 million con-
text tokens, democratizing access to advanced LLM func-
tionalities.
Ultimately, HEAD INFER aligns with the broader vision
of democratizing AI, ensuring that technological advance-
ments benefit humanity as a whole rather than a select few.
Through innovations like HEAD INFER , we hope to inspire a
future where AI is universally accessible, fostering creativ-
ity, knowledge sharing, and inclusive progress.
References
Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra,
N., Gulavani, B., Tumanov, A., and Ramjee, R. Taming
throughput-latency tradeoff in llm inference with sarathi-
serve. In 18th USENIX Symposium on Operating Sys-
tems Design and Implementation (OSDI 24), pp. 117–134,
2024.
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y .,
Lebron, F., and Sanghai, S. Gqa: Training generalized
multi-query transformer models from multi-head check-
points. In The 2023 Conference on Empirical Methods in
Natural Language Processing, 2023.
Aminabadi, R. Y ., Rajbhandari, S., Awan, A. A., Li, C.,
Li, D., Zheng, E., Ruwase, O., Smith, S., Zhang, M.,
Rasley, J., et al. Deepspeed-inference: enabling efficient
inference of transformer models at unprecedented scale.
In SC22: International Conference for High Performance
Computing, Networking, Storage and Analysis, pp. 1–15.
IEEE, 2022.
Anthropic, A. The claude 3 model family: Opus, sonnet,
haiku. Claude-3 Model Card, 1, 2024.
Badri, H. and Shaji, A. Half-quadratic quantization of large
machine learning models, 2023.
Bai, J., Bai, S., Chu, Y ., Cui, Z., Dang, K., Deng, X., Fan,
Y ., Ge, W., Han, Y ., Huang, F., et al. Qwen technical
report. arXiv preprint arXiv:2309.16609, 2023a.
Bai, Y ., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z.,
Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A
bilingual, multitask benchmark for long context under-
standing. arXiv preprint arXiv:2308.14508, 2023b.
Bai, Y ., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X.,
Cao, S., Xu, J., Hou, L., Dong, Y ., Tang, J., and Li,
J. Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks. arXiv
preprint arXiv:2412.15204, 2024.
Beltagy, I., Peters, M. E., and Cohan, A. Long-
former: The long-document transformer. arXiv preprint
arXiv:2004.05150, 2020.
Chen, S., Wong, S., Chen, L., and Tian, Y . Extending
context window of large language models via positional
interpolation. arXiv preprint arXiv:2306.15595, 2023.
Dao, T. Flashattention-2: Faster attention with bet-
ter parallelism and work partitioning. arXiv preprint
arXiv:2307.08691, 2023.
Dao, T., Fu, D., Ermon, S., Rudra, A., and R´e, C. Flashat-
tention: Fast and memory-efficient exact attention with
io-awareness. Advances in Neural Information Process-
ing Systems, 35:16344–16359, 2022.
De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-
Muraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y .,
Srinivasan, S., et al. Griffin: Mixing gated linear recur-
rences with local attention for efficient language models.
arXiv preprint arXiv:2402.19427, 2024.
Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W.,
Zheng, N., and Wei, F. Longnet: Scaling transformers to
1,000,000,000 tokens. arXiv preprint arXiv:2307.02486,
2023.
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,
A., et al. The llama 3 herd of models. arXiv preprint
arXiv:2407.21783, 2024.
Fu, Y ., Cai, Z., Asi, A., Xiong, W., Dong, Y ., and Xiao, W.
Not all heads matter: A head-level kv cache compression
method with integrated retrieval and reasoning. arXiv
preprint arXiv:2410.19258, 2024.
Gu, A. and Dao, T. Mamba: Linear-time sequence
modeling with selective state spaces. arXiv preprint
arXiv:2312.00752, 2023.
He, J. and Zhai, J. Fastdecode: High-throughput gpu-
efficient llm serving using heterogeneous pipelines.arXiv
preprint arXiv:2403.11421, 2024.
9

<!-- page 10 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney,
M. W., Shao, Y . S., Keutzer, K., and Gholami, A.
Kvquant: Towards 10 million context length llm in-
ference with kv cache quantization. arXiv preprint
arXiv:2401.18079, 2024.
Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D.,
Jia, F., Zhang, Y ., and Ginsburg, B. Ruler: What’s the
real context size of your long-context language models?
arXiv preprint arXiv:2404.06654, 2024.
Hu, E. J., Shen, Y ., Wallis, P., Allen-Zhu, Z., Li, Y ., Wang,
S., Wang, L., and Chen, W. Lora: Low-rank adaptation of
large language models. arXiv preprint arXiv:2106.09685,
2021.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G.,
Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint
arXiv:2310.06825, 2023.
Jiang, H., LI, Y ., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han,
Z., Abdi, A. H., Li, D., Lin, C.-Y ., et al. Minference 1.0:
Accelerating pre-filling for long-context llms via dynamic
sparse attention. In The Thirty-eighth Annual Conference
on Neural Information Processing Systems, 2024a.
Jiang, X., Zhou, Y ., Cao, S., Stoica, I., and Yu, M. Neo:
Saving gpu memory crisis with cpu offloading for online
llm inference. arXiv preprint arXiv:2411.01142, 2024b.
Kamradt, G. Needle in a haystack-pressure testing llms.
Github Repository, pp. 28, 2023.
Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.
Transformers are rnns: Fast autoregressive transformers
with linear attention. In International conference on ma-
chine learning, pp. 5156–5165. PMLR, 2020.
Langley, P. Crafting papers on machine learning. In Langley,
P. (ed.),Proceedings of the 17th International Conference
on Machine Learning (ICML 2000), pp. 1207–1216, Stan-
ford, CA, 2000. Morgan Kaufmann.
Lee, W., Lee, J., Seo, J., and Sim, J. Infinigen: Efficient gen-
erative inference of large language models with dynamic
kv cache management. In 18th USENIX Symposium on
Operating Systems Design and Implementation (OSDI
24), pp. 155–172, 2024.
Li, Y ., Jiang, H., Wu, Q., Luo, X., Ahn, S., Zhang, C., Abdi,
A. H., Li, D., Gao, J., Yang, Y ., et al. Scbench: A kv
cache-centric analysis of long-context methods. arXiv
preprint arXiv:2412.10319, 2024.
Liu, G., Li, C., Zhao, J., Zhang, C., and Guo, M. Clus-
terkv: Manipulating llm kv cache in semantic space for
recallable compression. arXiv preprint arXiv:2412.03213,
2024a.
Liu, H., Yan, W., Zaharia, M., and Abbeel, P. World model
on million-length video and language with blockwise
ringattention. CoRR, 2024b.
Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V .,
Chen, B., and Hu, X. Kivi: A tuning-free asymmetric
2bit quantization for kv cache. In Forty-first International
Conference on Machine Learning, 2024c.
Loeschcke, S. B., Toftrup, M., Kastoryano, M., Belongie, S.,
and Snæbjarnarson, V . Loqt: Low-rank adapters for quan-
tized pretraining. In The Thirty-eighth Annual Conference
on Neural Information Processing Systems, 2024.
Luo, C., Zhao, J., Chen, Z., Chen, B., and Anandkumar,
A. Mini-sequence transformers: Optimizing intermediate
memory for long sequences training. In The Thirty-eighth
Annual Conference on Neural Information Processing
Systems, 2024.
Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V .,
Devanur, N. R., Ganger, G. R., Gibbons, P. B., and Za-
haria, M. Pipedream: Generalized pipeline parallelism for
dnn training. In Proceedings of the 27th ACM Symposium
on Operating Systems Principles, pp. 1–15, 2019.
Pal, A., Karkhanis, D., Roberts, M., Dooley, S., Sundarara-
jan, A., and Naidu, S. Giraffe: Adventures in expanding
context lengths in llms. arXiv preprint arXiv:2308.10882,
2023.
Pekelis, L., Feil, M., Moret, F., Huang, M., and Peng, T.
Llama 3 gradient: A series of long context models, 2024.
Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho,
S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella,
M., et al. Rwkv: Reinventing rnns for the transformer era.
arXiv preprint arXiv:2305.13048, 2023.
Shah, J., Bikshandi, G., Zhang, Y ., Thakkar, V ., Ramani, P.,
and Dao, T. Flashattention-3: Fast and accurate attention
with asynchrony and low-precision. In The Thirty-eighth
Annual Conference on Neural Information Processing
Systems, 2024.
Sheng, Y ., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen,
B., Liang, P., R´e, C., Stoica, I., and Zhang, C. Flexgen:
High-throughput generative inference of large language
models with a single gpu. In International Conference
on Machine Learning, pp. 31094–31116. PMLR, 2023.
Su, J., Ahmed, M., Lu, Y ., Pan, S., Bo, W., and Liu, Y .
Roformer: Enhanced transformer with rotary position
embedding. Neurocomputing, 568:127063, 2024.
Sun, H., Chang, L.-W., Bao, W., Zheng, S., Zheng, N., Liu,
X., Dong, H., Chi, Y ., and Chen, B. Shadowkv: Kv
cache in shadows for high-throughput long-context llm
inference. arXiv preprint arXiv:2410.21465, 2024.
10

<!-- page 11 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Tang, H., Lin, Y ., Lin, J., Han, Q., Hong, S., Yao, Y ., and
Wang, G. Razorattention: Efficient kv cache compression
through retrieval heads. arXiv preprint arXiv:2407.15891,
2024.
Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju,
S., Pathak, S., Sifre, L., Rivi`ere, M., Kale, M. S., Love,
J., et al. Gemma: Open models based on gemini research
and technology. arXiv preprint arXiv:2403.08295, 2024.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At-
tention is all you need. Advances in neural information
processing systems, 30, 2017.
Williams, S., Waterman, A., and Patterson, D. Roofline:
an insightful visual performance model for multicore
architectures. Communications of the ACM, 52(4):65–76,
2009.
Wu, W., Wang, Y ., Xiao, G., Peng, H., and Fu, Y . Re-
trieval head mechanistically explains long-context factu-
ality. arXiv preprint arXiv:2404.15574, 2024.
Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H.,
Fu, Y ., and Han, S. Duoattention: Efficient long-context
llm inference with retrieval and streaming heads. arXiv
preprint arXiv:2410.10819, 2024a.
Xiao, G., Tian, Y ., Chen, B., Han, S., and Lewis, M. Ef-
ficient streaming language models with attention sinks.
In The Twelfth International Conference on Learning
Representations, 2024b.
Xiong, Y ., Wu, H., Shao, C., Wang, Z., Zhang, R., Guo, Y .,
Zhao, J., Zhang, K., and Pan, Z. Layerkv: Optimizing
large language model serving with layer-wise kv cache
management. arXiv preprint arXiv:2410.00428, 2024.
Xu, J., Zhang, R., Guo, C., Hu, W., Liu, Z., Wu, F., Feng,
Y ., Sun, S., Shao, C., Guo, Y ., et al. vtensor: Flexible
virtual tensor management for efficient llm serving.arXiv
preprint arXiv:2407.15309, 2024.
Zhang, Z., Sheng, Y ., Zhou, T., Chen, T., Zheng, L., Cai,
R., Song, Z., Tian, Y ., R´e, C., Barrett, C., et al. H2o:
Heavy-hitter oracle for efficient generative inference of
large language models. Advances in Neural Information
Processing Systems, 36:34661–34710, 2023.
Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A.,
and Tian, Y . Galore: Memory-efficient llm training by
gradient low-rank projection. In Forty-first International
Conference on Machine Learning, 2024.
Zhou, Z., Ning, X., Hong, K., Fu, T., Xu, J., Li, S., Lou,
Y ., Wang, L., Yuan, Z., Li, X., et al. A survey on effi-
cient inference for large language models. arXiv preprint
arXiv:2404.14294, 2024.
Zhu, D., Yang, N., Wang, L., Song, Y ., Wu, W., Wei, F.,
and Li, S. Pose: Efficient context window extension of
llms via positional skip-wise training. In The Twelfth
International Conference on Learning Representations,
2023.
11

<!-- page 12 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
A. Experiment Details
A.1. Prefill and Decoding Overhead of HEAD INFER
We evaluate the computational overhead ofHEAD INFER compared to baseline approaches across different context lengths
using the Llama-3-8B model. Our analysis focuses on two key phases: prefill and decoding.
Table 6. Prefill overhead (in seconds) of Llama3-8B under different context lengths.
Prefill Latency(s) 1K 10K 20K 40K 100K 200K 400K 1M 2M 4M
Standard 0.11 1.23 2.83 - - - - - - -
Chunked Prefill 0.11 1.23 2.83 - - - - - - -
Layer-offload 0.12 1.24 2.84 6.93 - - - - - -
HEAD INFER (head=8/group=1) 0.12 1.24 2.84 7.11 30.2 100 357 - - -
HEAD INFER (head=4/group=2) 0.13 1.23 2.89 7.26 30.2 99 351 2033 - -
HEAD INFER (head=2/group=4) 0.14 1.23 2.94 7.54 30.5 100 353 2035 7952 -
HEAD INFER (head=1/group=8) 0.21 1.27 3.06 7.77 31.2 101 356 2054 7975 27114
HEAD INFER Adaptive 0.13 1.24 2.84 7.11 30.2 99 351 2033 7952 27114
Table 7. Decoding overhead (in seconds per generated token) of Llama3-8B under different KV cache (context) sizes.
Decoding Latency(s) 1K 10K 20K 40K 100K 200K 400K 1M 2M 4M
Standard 0.03 0.03 0.03 0.04 - - - - - -
Chunked Prefill 0.03 0.03 0.03 0.04 - - - - - -
Layer-offload 0.03 0.09 0.17 0.28 0.66 1.3 2.58 - - -
HEAD INFER (head=8/group=1) 0.03 0.09 0.17 0.28 0.66 1.3 2.58 - - -
HEAD INFER (head=4/group=2) 0.04 0.10 0.16 0.28 0.67 1.31 2.58 6.41 - -
HEAD INFER (head=2/group=4) 0.06 0.11 0.17 0.30 0.68 1.32 2.59 6.46 13.7 -
HEAD INFER (head=1/group=8) 0.10 0.14 0.21 0.33 0.71 1.33 2.61 6.51 13.8 27.2
HEAD INFER Adaptive 0.03 0.09 0.17 0.28 0.66 1.73 3.03 6.41 13.7 27.2
For prefill operations, HEAD INFER demonstrates similar performance to standard approaches for shorter context lengths (up
to 20K tokens). Beyond this range, HEAD INFER scales efficiently with longer contexts, outperforming Layer-offload due to
its fine-grained head-wise KV cache management. Notably, HEAD INFER enables inference for 4M tokens on a single GPU,
which is otherwise infeasible.
In decoding, HEAD INFER maintains low latency at short and medium context lengths. For extended contexts (e.g., 1M
and beyond), HEAD INFER introduces manageable latency while supporting unprecedented context lengths, with adaptive
configurations optimizing performance further.
The performance of HEAD INFER relies on its ability to dynamically adapt to varying hardware constraints and workload
requirements by controlling the granularity of offloading. Specifically, HeadInfer Adaptive achieves optimal performance by
selectively choosing the most suitable head group size based on the context length and memory limitations.
HeadInfer (Head = 8 / Group = 1). This configuration aggregates all attention heads within a layer into a single group for
offloading, effectively making the KV cache management layer-wise. As such,HeadInfer (Head Group = 8) is functionally
equivalent to Layer-Offload, where all KV cache for a layer is stored on either the GPU or offloaded to the CPU in a single
operation. However, HeadInfer provides the flexibility to adjust the granularity of KV cache management beyond this
layer-wise approach.
HeadInfer (Head = 1 / Group = 8). At the other extreme, this configuration offloads each attention head individually,
offering the finest level of granularity for KV cache control. While this achieves the highest memory savings, frequent PCIe
transfers and kernel launches introduce overhead, especially for shorter context lengths, which can impact throughput.
HeadInfer Adaptive dynamically selects the optimal head group size (h) to balance memory efficiency and computational
12

<!-- page 13 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Figure 6. HEAD INFER provides equal accuracy as standard inference on the Needle-in-a-Haystack benchmark
throughput. This adaptability allows it to achieve superior performance across diverse hardware and context length
requirements:
• Shorter Contexts: For shorter context lengths, PCIe transfer and kernel launch overheads dominate. Larger head groups
(e.g., h = 4 or h = 8) are preferred to minimize overhead while maintaining memory efficiency.
• Longer Contexts: For longer context lengths, memory usage becomes the primary bottleneck. Smaller head groups (e.g.,
h = 2 or h = 1) are chosen to maximize context length while reducing GPU memory usage.
This dynamic strategy enables HeadInfer Adaptive to consistently deliver the best performance by adapting to both
memory and compute-bound regimes, as observed in performance and overhead evaluations.
In summary:
• HeadInfer Adaptive balances memory usage and performance by dynamically adjusting head group sizes, providing the
best of both worlds: memory efficiency and scalability.
• HeadInfer (Head Group = 8) is equivalent to Layer-Offload, serving as a robust baseline for layer-level KV cache
management.
• The head-wise approach of HEAD INFER enables extended context lengths during prefill, maintaining high throughput
without requiring specialized hardware or approximation techniques.
This adaptability positions HeadInfer Adaptive as an essential method for large-scale language model inference in
memory-constrained environments.
A.2. Long-Context Benchmarks Details
Needle-in-a-Haystack (Kamradt, 2023) is a challenging pressure test designed to assess the ability of models to accurately
identify and retrieve relevant information from a lengthy context. Figure 6 presents the results on Needle In A Haystack. We
use 1024K for Llama-3-8B-Instruct-Gradient-1024k. HEAD INFER can accurately recall the information queried within the
context in question.
Ruler (Hsieh et al., 2024). Designed for a comprehensive evaluation of long context, the Ruler benchmark is a recent
synthetic benchmark suite that includes 13 complex tasks in four main categories. Each context length variation includes
2,600 examples, with tests conducted at 4K, 8K, 16K, 32K, 64K, and 128K tokens. The benchmark comprises four key
categories. The retrieval category includes various Needle-in-a-Haystack tasks: Single (S-NIAH) for finding individual
key-value pairs in noisy text, Multi-keys (MK-NIAH) for retrieving specific values among hard distractors, Multi-values
(MV-NIAH) for finding all values linked to a single key, and Multi-queries (MQ-NIAH) for retrieving values across multiple
keys. The Multi-hop Tracing category features Variable Tracking (VT), requiring models to trace and return all variable
names pointing to the same value through variable bindings. For aggregation, the benchmark includes Common Words
Extraction (CWE) for identifying top-K common words from mixed sets and Frequent Words Extraction (FWE) for finding
the most frequent words from a Zeta distribution. The Question Answering category extends traditional QA datasets by
adding distracting paragraphs, testing models’ ability to locate and utilize relevant information amid distractors.
13

<!-- page 14 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Table 8. Maximum achievable sequence lengths for different inference methods
Method Supported Sequence Length within Ruler
Standard Inference 16K
Chunked-Prefill 32K
Layer-wise Offload 32K
HEAD INFER 128K
Table 8 demonstrates the maximum achievable sequence lengths for different inference methods on the Ruler benchmark.
Standard inference, while straightforward, is limited to 16K tokens due to memory constraints. Both Chunked-Prefill and
Layer-wise Offload methods double this capacity to 32K tokens through their respective optimization strategies.HEAD INFER
shows a significant advancement by enabling processing of sequences up to 128K tokens - a 4x improvement over other
offloading methods and a 8x improvement over standard inference. This extension in sequence length is achieved through
HEAD INFER ’s novel head-wise offloading strategy, which more efficiently manages GPU memory utilization.
Table 9. Performance on Ruler benchmark tasks across different context lengths
Context NIAH MK-2 MK-3 VT CWE FWE QA-1 QA-2
Length (%) (%) (%) (%) (%) (%) (%) (%)
4K 100.0 99.6 100.0 99.20 99.38 94.53 84.6 59.8
8K 100.0 99.8 99.6 99.08 94.68 84.93 79.2 56.2
16K 100.0 100.0 99.4 98.72 56.90 90.60 79.6 53.2
32K 100.0 99.6 99.8 97.32 2.78 93.20 77.2 50.4
64K 100.0 97.4 97.8 92.48 0.10 84.27 76.0 49.4
128K 100.0 75.2 56.6 54.68 0.10 74.8 71.8 41.2
HEAD INFER only
Table 9 presents a comprehensive analysis of performance across different tasks and context lengths. Since the ruler designs
different tasks for different context lengths, the ruler’s Multi-hop Tracing, Aggregation, and Question Answering become
more difficult to complete in long sequence scenarios. This is different from scbench, which directly truncates long sequence
tasks. However, even in this case, HEAD INFER can still show the same performance as standard inference, which confirms
its mathematical consistency.
A.3. Memory Analysis of Llama-3-8B Inference
In this work, we compare several inference configurations, each targeting different trade-offs between memory efficiency
and inference speed. While Figure 1 gives the theoretical benefit of HEAD INFER compared to other methods in terms of
memory usage, we also give how the theoretical memory is calculated here. Table 10 provides a comparative breakdown
of memory usage across different inference strategies. These methods balance GPU memory consumption for weights,
KV-cache, and activations while addressing the challenges of scaling to long-context inputs. Each strategy is outlined below:
• Standard: The standard inference method keeps all weights, KV-cache, and activations entirely in GPU memory. With
a context length of S, the KV-cache scales as S × L × Dh × H, where L is the number of layers, Dh is the hidden
dimension per head, and H is the number of attention heads. Activations require additional memory ofS × D + 2× S × I,
where I is the intermediate MLP dimension. While this approach achieves baseline performance, it is constrained by GPU
memory limits (e.g., 128 GB for the KV-cache and 207GB for toal).
• Chunked-Prefill: By dividing the input into smaller chunks of size chunk, this method reduces activation memory
from S to chunk. The memory footprint for activations is chunk×D + 2×chunk×I, where only part of the sequence
resides in GPU memory during processing. Although the KV-cache size remains S × L × Dh × H, this technique
significantly lowers activation memory requirements, reduce total memory from 207GB to 143GB.
• 4-bit KV-Quant: This method compresses the KV-cache from fp16/bf16 (16-bit floating point) to a 4-bit representation,
reducing its size by a factor of 4. The memory usage becomesS×L×Dh×H/4, while activations remainS×D+2×S×I.
14

<!-- page 15 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Table 10. Memory usage comparison for Llama3-8B with 1 million context length
Method Weight KV-cache Activation Total Total KV cache ∗
Standard 15.08 128 64 207 128
Chunked Prefill 15.08 128 0.625 143 128
4bit-KV-quant 15.08 32 64 111 32
layer-wise-offload 15.08 8 64 87 128
HEAD INFER 15.08 1 0.625 16.7 128
∗Total KV cache includes both GPU and CPU memory for offloading methods
• Layer-wise Offload: This strategy offloads the KV-cache for entire layers to CPU memory as soon as computation for
that layer is complete. On GPU, the memory required for KV-cache is reduced to S × Dh × H × 2, the final 2 due to
the ping-pong memory mechanism. However, offloading incurs communication overhead, making this approach more
suitable for scenarios with sufficient PCIe bandwidth.
• HEAD INFER : Our proposed approach achieves fine-grained control over memory by offloading KV-cache at the attention
head level. With a ping-pong memory mechanism, the on-GPU KV-cache is reduced to S × Dh × 2, significantly
lowering memory requirements. Activations are further minimized by combining chunked prefill with selective offloading.
HEAD INFER enables unprecedented scaling to 1M tokens and beyond, allowing context lengths of up to 4 million tokens
with minimal GPU memory usage. Also, we observe the total estimated memory of 16.7 GB is close to the real measured
memory of 17.0 GB, demonstrating the effectiveness of our estimation.
Table 11 summarizes the GPU memory usage for weights, KV cache, and activations, as well as the total usage for in-device
experiment. A detailed inference setting for each model and a comprehensive memory estimation of each method (including
key hyperparameters, chunk sizes, and offloading policies) are provided here. As shown in table 11, we focus on five
representative strategies for Llama-3-8B inference under varying context lengths:
Table 11. Memory consumption analysis for different inference methods (in GB) on Llama-3-8B
Method Weights KV cache Activation Total Total KV cache ∗
Standard-25K 15.08 3.13 1.56 19.77 3.13
Chunked-Prefill-30K 15.08 3.75 0.63 19.46 3.75
4bit-KV-Quant-45K 15.08 1.41 2.81 19.30 1.41
Layer-Wise-Offload-45K 15.08 0.35 2.81 18.25 5.63
HEAD INFER -4000K 15.08 3.91 0.63 19.61 500
∗Total KV cache includes both GPU and CPU memory for offloading methods
• Standard-25K: Baseline approach with unmodified inference. All model weights, activations, and the entire KV cache
stay on the GPU. Context length is limited by GPU memory.
• Chunked-Prefill-30K: Splits the prompt into sequential chunks to reduce activation overhead. KV cache remains on
GPU. We use a chunk size (e.g., 4K or 8K tokens) such that each partial forward pass does not exceed GPU capacity.
• 4bit-KV-Quant-45K: Applies a 4-bit quantization technique to KV cache, shrinking cache size by approximately 4×
compared to FP16/BF16. However, the method can introduce additional on-GPU overhead in activation or conversion
operations. In our tests, we adopt a standard quantization library (Hooper et al., 2024) for uniform 4-bit KV representation.
• Layer-Wise-Offload-45K: Offloads each layer’s KV cache to CPU memory as soon as possible. During inference, only
the KV cache for the currently processing layer is kept on the GPU. Once attention computation for that layer completes,
its KV cache is swapped out to the CPU. This approach significantly lowers on-GPU KV cache usage but may incur
additional offload overhead at each layer boundary.
• HEAD INFER -4000K: Our proposed head-wise offloading approach ( HEAD INFER ) that partitions the KV cache by
attention heads. While some heads remain fully on GPU, the cache for other heads (or tokens) are immediately offloaded
15

<!-- page 16 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
to CPU memory. Despite a very large total KV cache (reported as “500 GB” in Table 11), only a small fraction resides on
the GPU at any time. This enables context lengths in the order of millions of tokens if sufficient CPU RAM is available.
For total memory we define:
Mtotal; =; Mweights + MKV cache + Mactivation.
Weights (e.g., 15.08GB in Llama-3-8B) remain constant for all methods. The KV cache size grows with the number of
tokens processed (sequence length) but can be reduced or offloaded depending on the method. Activation memory arises
from forward-pass intermediate tensors (attention blocks, MLP layers, etc.) and is partially minimized by chunking or
parallelization strategies.
In Table 11, KV cache and Activation columns refer to the approximate GPU memory usage during inference; any data
offloaded to CPU memory (or disk) is not included in these columns but is counted in Total KV cache if applicable. For
instance, HEAD INFER has a GPU-side KV cache usage of about 3.9GB at any instant, whereas the overall KV cache across
CPU and GPU is up to 500GB, enabling very long context windows.
B. Motivation and Additional Related Work
In this section, we first explain that the KV cache size becomes a critical issue for long-text generation in LLM inference,
and it becomes more problematic when deploying modern offloading-based inference systems. We then discuss why the
existing KV cache management methods cannot fundamentally address the problem in an offloading-based system.
B.1. Memory Requirements for Inference
This section characterizes the memory requirements for transformer inference. It can be categorized into two components:
i) Model Parameters and ii) Activation memory primarily referring to KV cache. The memory requirement for model
parameters primarily depends on the hidden dimension D, the number of attention heads, and the number of Transformer
layers L. Nearly all the parameters in a Transformer block come from linear layers within L attention blocks, L multilayer
perceptron (MLP) blocks, and one language modeling head (LM-head) block. Take Llama-3-8B as an example; the total
parameters in a transformer-based model can be approximated as 14 × L × HD 2 with 15GB of memory. The memory
required for activation memory primarily consists of the KV cache, which depends on the model architecture, batch size B,
and sequence length S, and it can be pretty significant. The memory can be estimated as 2 × B × S × H × D. For instance,
in the Llama-3-8B (Dubey et al., 2024) model architecture, serving with FP16 KV cache for 1 million tokens would require
at least 207 GB of memory—exceeding the capacity of a single 80GB GPU.
B.2. KV Cache in LLM Inference Systems
As discussed in the previous section, today’s LLM serving systems exploit KV caching to avoid redundant computation of
key and value projections during the chunked-prefill decoding stage. While this is an effective solution for short sequence
generation with a single client request, the KV cache quickly becomes a key memory consumer when we generate long
sequences or employ modern request batching techniques (Sheng et al., 2023).
In the Llama-3-8B (Dubey et al., 2024) model architecture, serving with FP16 KV cache for 1 million tokens would require
at least 256 GB of memory—exceeding the capacity of a single 80GB GPU. Additionally, the latencies of pre-filling and
decoding with such large contexts are significant, posing substantial challenges to the effective use of LLMs in long-context
scenarios.
The rapidly expanding KV cache leads to an urgent need and numerous efforts for KV cache compression, particularly in
scenarios with limited GPU memory. Architectural modifications, such as Grouped-Query Attention (Ainslie et al., 2023),
Pose (Zhu et al., 2023), Rope (Su et al., 2024), PI (Chen et al., 2023), LongNet (Ding et al., 2023), MST (Luo et al., 2024),
LoQT (Loeschcke et al., 2024), Lora (Hu et al., 2021) and Galore (Zhao et al., 2024) require expensive model pre-training.
One direction is non-Transformer architecture design, such as Mamba (Gu & Dao, 2023), Linear Attention (Katharopoulos
et al., 2020), RWKV (Peng et al., 2023), Griffin (De et al., 2024). However, the transformer is still the most widely used
model structure, and in this paper, we focus on KV cache reduction for typical transformers. KV cache token-drop methods,
such as H2O (Zhang et al., 2023), StreamingLLM (Xiao et al., 2024b), InfiniGen (Lee et al., 2024), ClusterKV (Liu et al.,
2024a) often compromise accuracy in long-context applications and are incompatible with essential KV cache optimization
16

<!-- page 17 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Input Context(This is a very long story book):
…[repeating of long paragraph]…
HeadInfer’s favorite number is 42
…[repeating of long paragraph]...
       Question:                           What is HeadInfer’s favorite number?
       Original model:                   HeadInfer’s favorite number is 42
       StreamLLM:                        HeadInfer’s favorite number is not provided
       H2O:                                     zzz…(repeating meaningless sign)
       HeadInfer:                             HeadInfer’s favorite number is 42
       HeadInfer + 50% Sparsity:  HeadInfer’s favorite number is 42
J

L
J
J
L
Figure 7. Token eviction methods cannot work when querying the less relevant information to the main theme. Here, we use a 10K
document from LongBench (Bai et al., 2023b) and add one sentence that is not relevant to the main theme. In this case, H2O discards
tokens less relevant to the main theme, leading to error generation. StreamingLLM discards tokens based on the query but remaining
question tokens, making it Hallucinations. HEAD INFER can successfully output the exact information from the lengthy input, even when
we compress 75% of the KV cache
techniques like GQA. KV cache quantization (Hooper et al., 2024; Liu et al., 2024c; Badri & Shaji, 2023), although practical,
does have the upper limit of memory saving by 4 − 8×.
LLM Inference Systems with Offloading. In modern GPU systems, there is a significant disparity between CPU and GPU
memory capacities and costs. CPU RAM typically offers much larger capacity at a lower cost than GPU memory. For
example, modern server-grade systems can easily accommodate 1-2TB of CPU RAM, while even high-end GPU cards like
the NVIDIA A100 are limited to 80GB of memory. The cost difference is also substantial—server-grade DDR4/DDR5
RAM typically costs a fraction per gigabyte compared to specialized GPU memory.
This observation is supported by several prominent works offloading the KV cache to the CPU memory: FlexGen (Sheng
et al., 2023) leverages this hardware characteristic by utilizing CPU memory as an extension of GPU memory, allowing for
efficient LLM inference even with limited GPU resources. DeepSpeed (Aminabadi et al., 2022) implements sophisticated
offloading strategies that take advantage of the larger CPU memory capacity to handle model weights and KV cache that
would not fit in GPU memory alone. Infinitigen (Lee et al., 2024) builds on these foundations by introducing dynamic KV
cache management that works synergistically with offloading systems, but its efficiency is highly related to token eviction.
However, this memory hierarchy presents a trade-off: while CPU memory provides larger capacity at lower cost, data
transfer between CPU and GPU over PCIe becomes a potential bottleneck due to limited bandwidth compared to GPU
memory access speeds. This necessitates careful data movement management and strategic offloading decisions to maintain
efficient inference performance.
B.3. Challenges in KV Cache Management
In this context, several recent works propose reducing the KV cache size through retrieval head evictions. However, all the
prior works assume the persistence of attention patterns across layers, that is, if a head is deemed a retrieval head.
KV token eviction affects long-context performance. Figure 7 shows significant performance degradation since the actual
information required by the query might be discarded if considered unimportant, which uses the KV cache of all prior tokens
for computing attention results, and the KV cache management method of H2O with a KV cache budget of 2000 tokens.
H2O (Zhang et al., 2023) is a state-of-the-art technique that retains only a small percentage of important tokens in the KV
cache to reduce its size. It assesses the importance of each token in every iteration and removes unimportant ones before the
next iteration to keep the KV cache size in check.
The figure indicates that this is not the practice case, despite H2O-like approaches assuming that the attention pattern does
17

<!-- page 18 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
not change across heads. The tokens deemed unimportant in the one-head iteration could become important in other heads.
Consequently, H2O exhibits high similarity until around 200 iterations (i.e., within the KV cache budget). However, as the
sequence length extends beyond the KV cache budget, it struggles with the attention pattern’s dynamic nature, resulting in
more error generation than the optimal case. Note that while we only show the scenario of a KV cache budget of 2000 out
of a total sequence length of 10000 tokens for brevity, this issue would become more pronounced as the sequence length
surpasses it.
Prior works aiming to reduce the KV cache size through token eviction inherently have some challenges. Given the dynamic
attention pattern across iterations, permanently excluding evicted tokens from retinal head token generation can result in a
non-negligible drop in accuracy. Instead, we must keep the full attention tokens from the retrieval head while selectively
evicting less important heads. Furthermore, prior works’ iterative allocation of KV cache memory leads to inefficient KV
cache management. The number of key/value tokens required increases during chunked-prefill, and each extended context
inference demands effective memory management. Failing to account for this allocation may result in ineffective KV cache
management. Thus, we need to adjust the memory of key/value token pre-allocation while considering the variances between
retrieval and full head.
C. Roofline Model for head-wise flash attention
The Roofline model (Williams et al., 2009) serves as an effective theoretical framework to assess the potential performance
of deploying a model on particular hardware. Here we evaluate hardware performance of memory access and processing
unit capabilities.
Table 12. Performance comparison of different attention mechanisms under RTX-4090 setting
Operator Regular Offload
Ops Memory Arithmetic
Intensity FLOPS Bound Memory
(KV cache)
Arithmetic
Intensity FLOPS Bound
Prefill
flashattention (1k) 17G 21M 820 165T compute 4.2M 4100 102T memory
flashattention (10k) 1.7T 209M 8200 165T compute 42M 41000 165T compute
flashattention (100k) 172T 2.1G 82000 165T compute 419M 410000 165T compute
head-wise (1k) 2.1G 2.6M 820 165T compute 0.5M 4100 102T memory
head-wise (10k) 215G 26M 8200 165T compute 5.2M 41000 312T compute
head-wise (100k) 21T 262M 82000 165T compute 52M 410000 312T compute
Decode
flashattention (1k) 17M 17M 1 1T memory 17M 1 13G memory
flashattention (10k) 168M 168M 1 1T memory 168M 1 13G memory
flashattention (100k) 1.7G 1.7G 1 1T memory 1.7G 1 13G memory
head-wise (1k) 2.1M 2.1M 1 1T memory 2.1M 1 13G memory
head-wise (10k) 21M 21M 1 1T memory 21M 1 13G memory
head-wise (100k) 210M 210M 1 1T memory 210M 1 13G memory
Table 12 presents the analysis of layers in Llama-3-8b. From the table, we observe that during the prefill stage, the majority
of computations are compute-bound, leading to high performance. Conversely, in the decode stage, all computations are
memory-bound, resulting in performance significantly below the computational capacity of the GPU’s computation units.
Moreover, offload would make small context prefill memory-bound. Head-wise, the roofline model performs the same
arithmetic intensity and peak performance as standard FlashAttention.
We also show the roofline analysis on other GPUs, such as the A100, to demonstrate the generality of this analysis. Figure 8
shows that the prefill 1K offload is also positioned on memory-bound while the prefill 10K offload is on the compute-bound.
The details data is listed on table 13.
D. Extension: H EAD INFER Implementation with Head-wise Sparsity
Figure 9 shows our memory management framework. HEAD INFER , which enables offloading the head-wise KV cache with
head-wise sparsity. The key design principle behind HEAD INFER is to exploit the redundancy of CPU memory capacity
to increase the context size after identifying the important heads in the KV cache. As such, most of the heads for the KV
cache are kept in the CPU memory as we generate new tokens, not discarding them like previous work. However, we do not
18

<!-- page 19 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Table 13. Performance comparison of different attention mechanisms under A100 setting
Operator Regular Offload
Ops Memory Arithmetic
Intensity FLOPS Bound Memory
(KV cache)
Arithmetic
Intensity FLOPS Bound
Prefill
flashattention (1k) 17G 21M 820 312T compute 4.2M 4100 102T memory
flashattention (10k) 1.7T 209M 8200 312T compute 42M 41000 312T compute
flashattention (100k) 172T 2.1G 82000 312T compute 419M 410000 312T compute
head-wise (1k) 2.1G 2.6M 820 312T compute 0.5M 4100 102T memory
head-wise (10k) 215G 26M 8200 312T compute 5.2M 41000 312T compute
head-wise (100k) 21T 262M 82000 312T compute 52M 410000 312T compute
Decode
flashattention (1k) 17M 17M 1 1.4T memory 17M 1 23G memory
flashattention (10k) 168M 168M 1 1.4T memory 168M 1 23G memory
flashattention (100k) 1.7G 1.7G 1 1.4T memory 1.7G 1 23G memory
head-wise (1k) 2.1M 2.1M 1 1.4T memory 2.1M 1 23G memory
head-wise (10k) 21M 21M 1 1.4T memory 21M 1 23G memory
head-wise (100k) 210M 210M 1 1.4T memory 210M 1 23G memory
10
3
10
3
10-310-3
Performance[TFLOP/s]
Arithmetic Intensity[FLOP/Byte]
Roofline Analysis with A100
10
2
10
2
10
1
10
1
10
0
10
0
10
-1
10
-1
10
-2
10
-2
10-210-2 10-110-1 10 010 0 10 110 1 10 210 2
Prefill 1k Prefill 10k
Prefill 1k
HeadInfer/Offload
Prefill 10k
HeadInfer/Offlload
Saturation: BF16 Tensor Core(312 GFLOPs)
Decode HeadInfer(Overlapped with Decode Offload) Offload
BaselineBaseline
HeadInferHeadInfer
Decode
Figure 8. Flashattention in the roofline plot analysis using A100 device setting.
bring the entire KV cache to the GPU for attention but load and compute only the retrieval head of keys and values, leaving
other non-retrieval ones staying on the GPU without offloading. To do so, we maintain the head-wise cache pool in the CPU
memory and iteratively load the necessary data.
In detail, we use the pre-trained attention input head to speculate the important retrieval head. The speculation is done by
processing a customized dataset and analyzing the output. This reduces the waste of PCIe bandwidth by only transferring
retrieval heads critical for attention computation. In addition, although the data is offloaded to CPU memory, which is much
cheaper and larger than GPU memory, we manage the KV cache pool size so as not to fully utilize the CPU memory.
D.1. Design Principles
Interleaved KV cache Updates Across GPU and CPU. The uneven memory consumption and low PCIe link utilization
(studied in section 3) during different attention head generation provide an opportunity to exploit the idle GPU memory
and PCIe link during the KV cache update phase. To exploit this opportunity, during the attention processing, a head of
the attention KV cache can be dynamically fetched on the GPU to compute the attention weight output in parallel while
the CPU prefetches the next head. A key requirement to generate the attention output for a given attention head is to stage
its parameters (p), query (q), key (k), and value (v), and the attention weight generation is scheduled for each head. In
case the key (k) and value (v) of the head are not present on the GPU, the generation operation will trigger a prefetch read
from the CPU memory where the head is offloaded, causing I/O operations in the critical execution path of updates. By
leveraging the fact that multiple head attention, such as MHA (Vaswani et al., 2017) and GQA (Ainslie et al., 2023), are
19

<!-- page 20 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
time
......
...
Full Head KV-cache Sliding Window KV-cache
H0
Full-head
H0
Full-head
H1
Sliding
H1
SlidingGPU
HeadInfer + 50% Sparsity example:
#Only offload Full-head KV-cache
CPU Load H2Load H2
H2
Full-head
H2
Full-head
H3
Sliding
H3
Sliding
Load H4Load H4 ...
Figure 9. Demonstrations of KV cache policies in inference from the head-wise view. Upper plots illustrate symbolic plots of an attention
map deploying different policies in LLM generation. Lower: the overview of HEAD INFER .
embarrassingly parallel, and HEAD INFER partitions the attention into smaller subgroups, we can perform fine-grained
attention generation scheduling across both GPU and CPU without impacting the consistency of generation or introducing
computational dependencies between different subgroups. Furthermore, interleaving does not incur memory allocation and
deallocation overheads because on the GPU, memory allocation is handled by PyTorch through lightweight memory pools,
and on the host, the memory for all subgroups (except static GPU subgroups) is already pre-allocated and pre-pinned (if
enabled) during initialization.
As illustrated in Figure 10, the attention is partitioned into 4 sub-heads, out of which the first head is statically placed on
the GPU for the entire inference lifetime; the KV cache corresponding to the head resides in the GPU memory. Therefore,
the interleaved offloading adopted by HEAD INFER scheduled all the heads to be updated on the GPU, i.e., for every head
updated on the GPU, the host-to-device transform for the next head and the device-to-host transform for the previous head
would occur in a non-blocking fashion. This interleaved offloading makes sure that only two heads are maintained on the
GPU while most KV cache are offloaded on the CPU. This interleave-centric design allows for efficient overlap between
GPU computations and asynchronous head-wise KV cache movement across the PCIe link, which we will detail next.
Overlapping Head-wise KV cache Movement and GPU Compute The data movement observed when the state-of-the-art
framework enabling layer-wise KV cache offload (e.g., Flexgen(Sheng et al., 2023) and Deepspeed(Aminabadi et al., 2022))
runs an inference process. We observe that after the updates to the KV cache generation to a given layer i are computed on
the GPU, the updated KV cache is H2D transferred to the CPU to continue inference in the subsequent chunked-prefill and
decoding. Only when all the KV cache of next later are transferred to the GPU can the subsequent iteration begin. Given the
parallel nature of attention, the sub-heads can be updated and transferred out of order and do not impact the accuracy of the
inference. On contract, using the existing offloading solutions can be slow with head-wise sparsity attributed to (a) the KV
cache of all heads within one layer staying on the GPU, which would occupy large memory for long context, and (b) not all
attention layers are discretized uniformly, which causes blocking H2D transfer of KV cache.
To mitigate the aforementioned challenges, we propose a head-centric design illustrated in Figure 10 for efficient offloading
interleaving of H2D transfor and GPU compute. It works as follows: while the GPU computes the generation of the initial
head (H 1), the KV cache corresponding to the next head (H 2), including key (k) and value (v), are being prefetched using
asynchronous H2D transfers, thereby overlapping GPU computations with next-head prefetching. Meanwhile, the CPU
update for previous H 0 is being uploaded using asynchronous D2H transfers. After this, three operations happen in parallel:
(1) H2D transfer of the next head and prefetching of the next KV cache to be updated on the GPU (2) Updating of the
previous head with KV cache from GPU; and (3) GPU generation of current head outputs, thereby exploiting full-duplex
D2H and H2D transfers and parallel GPU computations.
20

<!-- page 21 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
Time
...
H2D
D2H
H0,2H0,2 H0,3H0,3 H1,0H1,0H0,0H0,0 H0,1H0,1
CPU(Memory)
H0,2
H0,2H0,2
H0,3
H0,3H0,3
H1,0
H1,0H1,0
H0,1H0,1H0,0H0,0 H0,2H0,2 H0,3H0,3 H1,0H1,0
H0,0H0,0
H0,1H0,1
H0,1
H0,1H0,1
H0,2H0,2
H0,3
H1,0
GPU
(2 head size
one time)
GPU Memory
GPU Compute
...
...
...
...
...
...
...
...
H2D: Host(CPU) to Device(GPU) Transfer D2H: Device(GPU) to Host(CPU) Transfer
...
Prefetch Next
layer’s First Head
Figure 10. Workflow of HEAD INFER generating a model with (n+1) layers and (j+1) attention heads.
Layer
Head
KV Cache Sample 1-Head Group4-Head Group 2-Head Group
Figure 11. Demonstrations of adaptive head-wise offloading.
The attention phase is executed in a parallel fashion by multiple heads of the mechisan. Consequently, our proposed
overlapping of GPU computations and PCIe transfers does not incur any blocking overheads, as the computation speed
on the GPU is slower than the PCIe throughput to transfer subheads back and forth between the GPU and CPU using
high-throughput PCIe.
Efficient Management of KV cache State-of-the-art hybrid KV cache offloading solutions (e.g., FlexFLow) by default
retain the layer-wise KV cache corresponding to the statically GPU-resident head (h1 to h4) on the GPU during the attention
computation, and for the remainder of the layer, KV cache are offloaded to the host memory. We extend this design with
head-wise KV cache management to incorporate head-wise sparsity. Head-wise sparsity divides attention heads into two
categories: important retrieval heads retain all tokens, while unimportant The streaming head only retains the most recent
token. Since the memory usage of the retrieval head will be much larger than that of the streaming head, especially under
long context inference, HEAD INFER will selectively offload the retrieval head to the CPU and keep the streaming head in
the GPU.
Adaptive selecting head-wise granularity. In transformer-based architectures, attention heads often operate in parallel.
However, processing these heads individually—or in small groups—can incur repeated kernel launches and excessive PCIe
transfers, particularly if the context size for each head is small. This overhead can quickly dominate total inference time,
undermining the benefits of parallelism. Adaptive head-wise offloading addresses these inefficiencies by merging multiple
heads into a single “HeadGroup.” As shown on Figure 11, with a reduced number of HeadGroups (e.g., 2-HeadGroup
or 4-HeadGroup), fewer offloading operations are required, lowering both the latency and the bandwidth usage on the
CPU–GPU boundary. Conceptually, this “batching” of heads takes advantage of the fact that many computations within
attention heads are structurally similar, thus allowing shared memory transfers and kernel calls. Layer-offload can be viewed
as the extreme case of adaptive head-wise offloading. Instead of grouping only a fraction of heads at a time, layer-offload
groups every head in a given layer into one offload operation. While it can drastically reduce overhead further, the trade-off
21

<!-- page 22 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
is less granularity and potentially higher intermediate memory requirements. In practice, the decision to use 2-head, 4-head,
or full-layer offloading depends on available hardware resources, batch size, and the typical context length being processed.
By carefully tuning the grouping strategy—ranging from small, flexible head groups to large, layer-wide groups—adaptive
head-wise offloading makes it possible to significantly optimize inference time in transformer-based models, particularly in
latency-sensitive scenarios involving small context.
D.2. Extension Experiment with Head-wise Sparsity
We evaluate our head-wise extension ofHEAD INFER using 50% sparsity, which means half of the attention heads are sparse,
on Llama3-8B models and compare against HEAD INFER . For the prefill and decoding latency, the extension achieves close
to 2× speedup. The results of this evaluation are shown in Table 14.
Table 14. Prefill 1M, Decoding with 1M KV cache performance comparison
Latency Prefill 1M (s) Decoding with 1M KV cache(s)
HeadInfer 2054 6.51
HeadInfer+duoattention 50% sparsity 1152 3.28
E. Easy-to-use and Portable Implementation
HEAD INFER is designed for straightforward integration with existing inference or training frameworks, such as Hugging
Face Transformers, requiring only minimal modifications. Below, we illustrate how one can adapt a standard Llama attention
module to enable head-wise KV offloading with minimal impact on the rest of the code.
E.1. Overview of Required Modifications
Our changes largely center around intercepting or replacing the attention’s forward pass so that:
• Heads are processed individually or in groups, rather than as a single large multi-head block.
• Key and value states can be transparently stored in, or fetched from, CPU memory (or other off-device storage) before
each head’s attention is computed.
• Asynchronous transfers are used where possible to overlap CPU–GPU data movement with on-GPU computation.
E.2. Annotated Code Snippet
In the listing below, we demonstrate key functions that illustrate how to integrateHEAD INFER into a transformers-style
codebase. These snippets show how the standard forward method is patched with head-wise logic. We also provide
HeadwiseOffloadedCache, a class that manages CPU–GPU memory movement, and a small helper function for simulating
or preparing decoding with large context windows.
The implementation primarily requires modifying the attention mechanism class in transformer models. The key modifica-
tions are:
1. Attention Class Modification: Update the forward pass of the attention mechanism to support head-wise offloading:
Listing 1. Modified LlamaAttention forward pass
def f o r w a r d ( s e l f , h i d d e n s t a t e s , a t t e n t i o n m a s k =None ,
p a s t k e y v a l u e =None ) :
# O r i g i n a l a t t e n t i o n c o m p u t a t i o n
q u e r y s t a t e s = s e l f . q p r o j ( h i d d e n s t a t e s )
k e y s t a t e s = s e l f . k p r o j ( h i d d e n s t a t e s )
v a l u e s t a t e s = s e l f . v p r o j ( h i d d e n s t a t e s )
22

<!-- page 23 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
# Head−w i s e p r o c e s s i n g
b a t c h s i z e = q u e r y s t a t e s . s h a p e [ 0 ]
num heads = s e l f . num heads
head dim = s e l f . head dim
# Reshape f o r head−w i s e p r o c e s s i n g
q u e r y s t a t e s = q u e r y s t a t e s . view (
b a t c h s i z e , −1 , num heads , head dim )
k e y s t a t e s = k e y s t a t e s . view (
b a t c h s i z e , −1 , num heads , head dim )
v a l u e s t a t e s = v a l u e s t a t e s . view (
b a t c h s i z e , −1 , num heads , head dim )
# P r o c e s s each head i n d e p e n d e n t l y
o u t p u t s = [ ]
f o r h e a d i d x i n range ( num heads ) :
h e a d q = q u e r y s t a t e s [ . . . , h e a d i d x , : ]
h e a d k = k e y s t a t e s [ . . . , h e a d i d x , : ]
h e a d v = v a l u e s t a t e s [ . . . , h e a d i d x , : ]
# Update or f e t c h from s p e c i a l i z e d Cache ( p a s t k e y v a l u e )
head k , h e a d v = p a s t k e y v a l u e . u p d a t e ( headk , head v , l a y e r i d x , h e a d i d x )
h e a d o u t p u t = c o m p u t e h e a d a t t e n t i o n (
head q , head k , head v ,
a t t e n t i o nm a s k ,
p a s t k e y v a l u e
)
o u t p u t s . append ( h e a d o u t p u t )
return t o r c h . c a t ( o u t p u t s , dim = −1)
2. KV Cache Management: Implement the OffloadedCache class for head-wise cache management:
Listing 2. HeadwiseOffloadedCache implementation
c l a s s HeadwiseOffloadedCache :
def i n i t ( s e l f , h e a d g r o u p s =1) :
s e l f . k e y c a c h e = [ ]
s e l f . v a l u e c a c h e = [ ]
s e l f . h e a d g r o u p s = h e a d g r o u p s
s e l f . p r e f e t c h s t r e a m = t o r c h . cuda . Stream ( )
s e l f . e v i c t s t r e a m = t o r c h . cuda . Stream ( )
def u p d a t e ( s e l f , k e y s t a t e s , v a l u e s t a t e s , l a y e r i d x , h e a d i d x ) :
””” Updates cache f o r s p e c i f i c head , managing CPU o f f l o a d ”””
w i t h t o r c h . cuda . s t r e a m ( s e l f . p r e f e t c hs t r e a m ) :
# P r e f e t c h n e x t head i f needed
s e l f . p r e f e t c hh e a d ( l a y e r i d x , h e a d i d x + 1 )
# Update c u r r e n t head cache
i f l e n ( s e l f . k e y c a c h e ) <= l a y e r i d x :
s e l f . k e y c a c h e . append ( [ ] )
s e l f . v a l u e c a c h e . append ( [ ] )
w h i l e l e n( s e l f . k e y c a c h e [ l a y e ri d x ] ) <= h e a d i d x :
s e l f . k e y c a c h e [ l a y e ri d x ] . append ( None )
s e l f . v a l u e c a c h e [ l a y e ri d x ] . append ( None )
# S t o r e on a p p r o p r i a t e d e v i c e
d e v i c e = s e l f . g e t h e a d d e v i c e ( l a y e ri d x , h e a d i d x )
s e l f . k e y c a c h e [ l a y e ri d x ] [ h e a d i d x ] = k e y s t a t e s . t o ( d e v i c e )
s e l f . v a l u e c a c h e [ l a y e ri d x ] [ h e a d i d x ] = v a l u e s t a t e s . t o ( d e v i c e )
23

<!-- page 24 -->

HEAD INFER : Memory-Efficient LLM Inference by Head-wise Offloading
# E v i c t p r e v i o u s head i f needed
s e l f . e v i c t h e a d ( l a y e r i d x , h e a d i d x − 1 )
return k e y s t a t e s , v a l u e s t a t e s
def p r e f e t c hh e a d ( s e l f , l a y e r i d x , h e a d i d x ) :
””” A s y n c h r o n o u s l y moves n e x t head t o GPU i f needed ”””
i f h e a d i d x >= s e l f . num heads :
return
c a c h e k = s e l f . k e y c a c h e [ l a y e ri d x ] [ h e a d i d x ]
c a c h e v = s e l f . v a l u e c a c h e [ l a y e ri d x ] [ h e a d i d x ]
i f c a c h e k i s not None and c a c h e k . d e v i c e != t o r c h . d e v i c e ( ’ cuda ’ ) :
c a c h e k = c a c h e k . cuda ( n o n b l o c k i n g = True )
c a c h e v = c a c h e v . cuda ( n o n b l o c k i n g = True )
def e v i c t h e a d ( s e l f , l a y e r i d x , h e a d i d x ) :
”””Moves p r e v i o u s head t o CPU i f needed ”””
i f h e a d i d x < 0 :
return
w i t h t o r c h . cuda . s t r e a m ( s e l f . e v i c t s t r e a m ) :
c a c h e k = s e l f . k e y c a c h e [ l a y e ri d x ] [ h e a d i d x ]
c a c h e v = s e l f . v a l u e c a c h e [ l a y e ri d x ] [ h e a d i d x ]
i f c a c h e k i s not None and c a c h e k . d e v i c e == t o r c h . d e v i c e ( ’ cuda ’ ) :
c a c h e k = c a c h e k . cpu ( )
c a c h e v = c a c h e v . cpu ( )
Our implementation shows that these modifications can be made with minimal changes to existing model architectures while
maintaining full compatibility with standard transformer frameworks huggingface. The complete implementation would be
available at https://github.com/wdlctc/headinfer.
24
