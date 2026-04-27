# references/74_fouriercompress_layer_aware_spectral_activation_compression.pdf

<!-- page 1 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
FourierCompress: Layer-Aware Spectral Activation Compression for
Efficient and Accurate Collaborative LLM Inference
Jian Ma, Xinchen Lyu, Jun Jiang, Longhao Zou, Chenshan Ren, Qimei Cui, and Xiaofeng Tao
Abstract—Collaborative large language model (LLM) inference
enables real-time, privacy-preserving AI services on resource-
constrained edge devices by partitioning computational work-
loads between client devices and edge servers. However, this
paradigm is severely hindered by communication bottlenecks
caused by the transmission of high-dimensional intermediate
activations, exacerbated by the autoregressive decoding structure
of LLMs, where bandwidth consumption scales linearly with
output length. Existing activation compression methods struggle
to simultaneously achieve high compression ratios, low recon-
struction error, and computational efficiency. This paper proposes
FourierCompress, a novel, layer-aware activation compression
framework that exploits the frequency-domain sparsity of LLM
activations. We rigorously demonstrate that activations from the
first Transformer layer exhibit strong smoothness and energy
concentration in the low-frequency domain, making them highly
amenable to near-lossless compression via the Fast Fourier Trans-
form (FFT). FourierCompress transforms activations into the
frequency domain, retains only a compact block of low-frequency
coefficients, and reconstructs the signal at the server using conju-
gate symmetry, enabling seamless hardware acceleration on DSPs
and FPGAs. Extensive experiments on Llama 3 and Qwen2.5
models across 10 commonsense reasoning datasets demonstrate
that FourierCompress preserves performance remarkably close
to the uncompressed baseline, outperforming Top-k, QR, and
SVD. FourierCompress bridges the gap between communication
efficiency (an average7.6×reduction in activation size), near-
lossless inference (less than0.3%average accuracy loss), and
significantly faster compression (achieving over32×reduction in
compression time compared to Top-kvia hardware acceleration)
for edge-device LLM inference.
Index Terms—LLM, 6G networks, Collaborative LLM Infer-
ence, FFT.
I. INTRODUCTION
The integration of artificial intelligence (AI) and com-
munication systems stands as one of the six key scenarios
for sixth-generation (6G) wireless networks [1]. A central
vision of 6G is to deliver ubiquitous intelligent connectivity,
enabling real-time, intelligent services across a vast ecosystem
of resource-constrained devices expanding from smartphones
and wearables to Internet of Things (IoT) sensors [2], [3]. At
the forefront of AI advancements are Large Language Models
(LLMs), which have demonstrated remarkable capabilities in
natural language understanding and generation. However, the
This work was supported in part by Mobile Information Networks National
Science and Technology Major Project (Grant No. 2025ZD1303100), and
in part by the National Natural Science Foundation of China under Grant
62371059.
Corresponding authors: Xinchen Lyu; Jun Jiang.
J. Ma, X. Lyu, Q. Cui and X. Tao are with Beijing University of Posts and
Telecommunications, China, and also with Pengcheng Laboratory, China. J.
Jiang and L. Zou is with Pengcheng Laboratory, China. C. Ren is with Minzu
University of China, China.
deployment on mobile devices faces fundamental limitations
due to the immense computational, memory, and energy re-
quirements inherent to billion-parameter LLMs [4], [5].
While on-device inference preserves data privacy and
minimizes latency, executing state-of-the-art LLMs entirely
on mobile platforms remains impractical [6], [7]. Although
techniques such as model distillation [8], quantization [9]
, and KV-Cache optimization [10] have been proposed to
create lightweight LLMs, the resource demands after model
lightweighting still substantially exceed the capabilities of typ-
ical mobile devices. This necessitates a distributed inference
paradigm (also known as collaborative/split LLM inference)
[11], where the LLM is partitioned into the device portions and
edge portions with only intermediate activations transmitted
via the wireless channel.
Collaborative LLM inference has garnered significant re-
search interest for enabling real-time, privacy-preserving in-
telligent, and energy-efficient LLM services on mobile de-
vices [12]–[15]. However, as shown in Figure 1,the critical
challenge of collaborative LLM inference lies in bandwidth
consumption of activation transmission arising from the recur-
sive decoding structure of LLMs.Unlike conventional machine
learning models that process inputs in a single forward pass,
LLMs generate text token-by-token in an autoregressive man-
ner [16], [17]. Each new token requires processing the entire
history of previously generated tokens, creating a cumulative
communication burden that scales linearly with output length.
Taking Qwen3-235B as an example, each in-depth thinking
conversation consumes approximately 81,920 tokens and re-
quires an activation size of around 1.25 GB.
Although existing research has explored model partitioning
strategies [18], resource allocation mechanisms [19]–[21], and
system optimization techniques [13], [15] for collaborative
LLM inference, the fundamental bandwidth bottleneck of
activation transmission remains inadequately addressed.This
paper aims to address the activation bandwidth bottleneck (es-
sential for both collaborative LLM inference and fine-tuning)
by designing an computational efficient compression technique
that achieves substantial bandwidth reduction while preserving
inference accuracy even under aggressive compression ratios.
A. Technical Challenges of Activation Compression
Traditional compression techniques, such as quantization
[22], [23], sparsification (e.g., Top-k) [24], and low-rank
approximation (e.g., SVD) [25]–[27], have been primarily
developed for static model weights rather than the transient,
data-dependent activations in collaborative LLM inference.
This mismatch arises because LLM activations are dynamic,
arXiv:2510.16418v1  [cs.DC]  18 Oct 2025

<!-- page 2 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
Mobile
devices
 Split LLM
Channel
Split LLM
Server
Activation
Token LLM
iteration 1
Is grape a
fruit?
LLM
iteration 2
KV
cache
LLM
iteration 3
KV
cache
Recursive Decoding Structure of LLM
...
Yes
 it
Fig. 1: The bandwidth bottleneck of collaborative LLM inference due to recursive decoding structure.
data-driven signals with rich spatial and spectral structure,
unlike the relatively stable statistical distributions of model
weights [28], [29]. Consequently, directly applying weight-
centric compression to activations results in semantic distor-
tion, where critical information for downstream reasoning is
lost despite high compression ratios. Two critical challenges
must be addressed to enable efficient and accurate collabora-
tive LLM inference:
(1) Layer-Specific Compressibility: Where to Split?A crit-
ical yet underexplored question is: which layer serves as
the optimal split point for minimizing communication while
preserving inference accuracy? Existing approaches typically
propose dynamic splitting strategies based on system resources
(e.g., ALS [30], Splitwise [31], and EdgeShard [5]). However,
they overlook the intrinsic mathematical properties of activa-
tions across different layers. As we demonstrate empirically
in Sec. III, activations from different layers exhibit vastly
different compressibility. Selecting an appropriate LLM split
point is the prerequisite of designing an efficient compression
technique.
(2) Activation-Aware Compression: How to Compress?
Even with an optimal split point, the choice of compression
method is crucial. Which compression method can achieve
high ratios while preserving the semantic integrity of acti-
vations to maintain inference accuracy? Current approaches
primarily target model weights rather than activations, and
treat activations as unstructured tensors, leading to suboptimal
performance. This creates a critical gap: a lack of activation-
specific, layer-aware compression methods for collaborative
LLM inference.
B. Contributions and Paper Organization
We propose FourierCompress, a novel activation compres-
sion framework via Fast Fourier Transform (FFT) [32] to
exploit the spectral sparsity of early-layer LLM activations.
We find that activations in the first Transformer layer exhibit
strong smoothness and rapid spectral decay with most energy
concentrated. We find that in low-frequency components, thus
perfectly suitable for compression. By transforming activations
into the frequency domain and retaining only the dominant
low-frequency coefficients, we achieve high compression ra-
tios with near-lossless reconstruction. FourierCompress ad-
dresses both technical challenges:
(1) Where to Split?We establish, for the first time, that the
first Transformer layer is the optimal split point for activa-
tion compression. We provide empirical evidence that early-
layer activations are uniquely compressible due to their local
attention patterns and structured spatial correlations, whereas
deeper layers develop complex, high-entropy representations
that resist aggressive compression.
(2) How to Compress?We propose a frequency-domain
compression framework: transform activations via FFT, retain
low-frequency coefficients, and reconstruct using conjugate
symmetry. We show that FourierCompress achieves a tighter
reconstruction error than its counterparts at the same compres-
sion ratio.
From extensive experiments on Llama 3 and Qwen 2.5
across 10 datasets, FourierCompress demonstrate state-of-the-
art performance in accuracy, compression ratio, and compres-
sion speed.
•High Compression Efficiency with Minimal Accuracy
Loss.By splitting at the first layer, FourierCompress
achieves an average 7.6×reduction in activation size
across 10 commonsense reasoning tasks, with less than
0.3% average accuracy drop, effectively solving the re-
cursive bandwidth problem in autoregressive generation.
•Near-Lossless Inference.FourierCompress consistently
outperforms Top-k, QR, SVD, maintaining performance
remarkably close to the uncompressed baseline, even

<!-- page 3 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
TABLE I: Summary of Existing Model Partitioning and Activation Compression Studies.
Problem Literature Model Type Details
FcaNet [33] Small Model Partitions feature maps by channel to enrich channel attention.
ALS [30] LLM Dynamically splits an LLM by layer between a device and an edge server.
Splitwise [31] LLM Separates prompt computation and token generation onto specialized hardware.
EdgeShard [5] LLM Partitions an LLM layer-wise across multiple heterogeneous edge devices.
Model Partitioning
FFSplit [34] LLM Splits the Feed-Forward Network (FFN) based on neuron output norms.
COBLA [35] Small Model Uses constrained optimization to find the optimal low-rank approximation for layers.
Dynamic Pruning [36] Small Model Prunes unimportant coefficients in the frequency domain for CNN compression.
SpinQuant [37] LLM Reduces activation outliers using rotation matrices for better low-bit quantization.
Split fine-tuning [24] LLM Compresses activations using Top-ksparsification during distributed training.
AWQ [22] LLM Protects important weights by scaling them before quantization.
Agile-Quant [23] LLM Prunes tokens to reduce activation outliers before applying quantization.
Atom [38] LLM Reorders activation channels to isolate outliers for mixed-precision quantization.
FWSVD [25] LLM Applies a weighted low-rank factorization based on parameter importance.
Asvd [26] LLM Transforms weights based on activations for improved low-rank decomposition.
Activation Compression
Svd-llm [27] LLM Uses an activation-based data whitening technique to guide weight decomposition.
Ours (FourierCompress) LLM Our method addresses both where to split and how to compress by leveraging
FFT for high activation compression ratios and near-lossless reconstruction.
surpassing it in some cases.
•Hardware-Accelerated Speed.The use of FFT enables
seamless integration with existing hardware accelerators.
This reduces compression time by up to 32×compared
to Top-k, making it ideal for edge deployment.
The rest of this paper is structured as follows. Section
II reviews related work in model partitioning and activation
compression. Section III provides a layer-specific analysis
of activation compressibility to motivate our approach, then
details the proposed FourierCompress framework and its hard-
ware integration. Section IV presents extensive experimental
results and ablation studies that validate the performance of
our method. Finally, Section V concludes the paper.
II. RELATEDWORK
This section reviews the recent efforts for collaborative
LLM inference, which can be broadly categorized into two
complementary directions: (1) where to split the model across
edge and server, and (2) how to compress the intermediate ac-
tivations to reduce communication overhead. As summarized
in Table I, existing work typically addresses the two questions
separately, and none leverages the signal structure of LLM
activations for efficient compression.
A. Model Partitioning: Where to Split?
Some research has focused on determining the optimal
split point in a layered LLM to balance computational load,
memory usage, and end-to-end latency. These methods rely
on system-level metrics such as network bandwidth, device
compute capability, and power constraints: (1) ALS [11]
dynamically selects the split layer based on real-time network
conditions and node resources; (2) Splitwise [12] uses a phase-
splitting strategy to optimize for latency and throughput across
multiple inference stages; (3) EdgeShard [13] adaptively par-
titions the model across heterogeneous edge devices, consid-
ering bandwidth and model size; (4) FFSplit [14] proposes
splitting within feedforward networks to improve the accuracy-
efficiency trade-off.
While these approaches provide valuable system-level opti-
mizations, they overlook the intrinsic mathematical properties
of LLM activations. As will be shown in Section III, activation
compressibility varies drastically across layers. Splitting at
a deep layer can make compression ineffective and degrade
accuracy. Our work fills this gap by introducing a signal-aware
splitting strategy, establishing for the first time that the first
Transformer layer is optimal due to its unique spectral sparsity
and spatial smoothness.
B. Activation Compression: How to Compress?
(1) Quantization.Quantization reduces the bit-width of
activation values to lower transmission costs. However, quanti-
zation alone often provides limited compression ratios without
significant accuracy loss, especially in the presence of activa-
tion outliers. Many state-of-the-art quantization methods are
primarily designed for model weights or require complex cal-
ibration. For example, AWQ (Activation-aware Weight Quan-
tization) [22] protects salient weights from quantization errors
by observing activation magnitudes but is fundamentally a
weight-only method. Other techniques, like Agile-Quant [23],
address activations by pruning tokens that cause large outliers
before applying quantization. While effective, these methods
either do not directly compress activations for transmission
or rely on removing information (pruning), which can be
suboptimal compared to transforming the signal to a more
compressible basis.
(2) Low-Rank Approximation.Low-rank approximation
methods aim to compress LLM parameter matrices by iden-
tifying and preserving the most significant components via
Singular Value Decomposition (SVD). However, these tech-
niques are predominantly designed for static model weights,
not for the dynamic, data-dependent activations transmitted
during collaborative inference. For example, prominent meth-
ods like FWSVD [25], SVD-LLM [27], and ASVD [26] all
leverage activation characteristics to improve the compression
of weight matrices. FWSVD applies a weighted factorization
based on parameter importance, SVD-LLM uses an activation-
based whitening technique to guide weight decomposition, and
ASVD transforms weights based on activation distributions.
(3) Sparsification.Sparsification methods reduce commu-
nication by transmitting only a small subset of the most

<!-- page 4 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
significant activation values. The most prominent technique
is Top-ksparsification, which is applied to activations in
frameworks likeSplit fine-tuning [24]. This method retains the
Kactivation values with the largest absolute magnitudes while
setting the rest to zero.
(4) Frequency-Domain Methods.Few studies have explored
frequency-domain analysis for model compression, and these
have been confined to smaller, non-LLM architectures. A
notable example is Dynamic Pruning [36], which prunes
unimportant frequency coefficients ofweightsin Convolutional
Neural Networks (CNNs) to reduce model size.
Distinctively different from existing work, FourierCompress
is the first activation-specific, layer-aware compression ap-
proach that jointly considers the split point selection and model
compression. By identifying the first Transformer layer as the
split point (with energy concentrated in the low-frequency
domain), we show that FFT is superior in terms of accuracy,
compression ratio, and compression speed. FourierCompress
establishes a new paradigm for efficient, nearly-lossless col-
laborative inference framework with aggressive compression
ratio for empowering edge-device LLM services.
III. LAYER-AWARESPECTRALACTIVATION
COMPRESSION: MOTIVATION ANDFRAMEWORKDESIGN
This section presents the motivation and technical frame-
work of our proposed approach. We first establish the criti-
cal insight regarding layer-specific activation compressibility
through empirical analysis, and present the FourierCompress
framework for efficient collaborative inference.
A. Motivation: Layer-Aware Compressibility Analysis
We start by presenting our key insights regarding the
compressibility characteristics of LLM activations, which form
the foundation of FourierCompress. We demonstrate through
empirical evidence and analysis why early-layer activations
exhibit unique properties that make them ideal candidates for
spectral compression. This section also provides an overview
of the workflow of the FourierCompress framework.
(1) Analysis of Activation Patterns.Our investigation begins
with an observation that LLM activations exhibit dramatically
different compressibility characteristics across network layers
[39]–[41], necessitating a layer-aware compression strategy.
Figure 2(a) provides compelling visual evidence of this phe-
nomenon through a comparative analysis of activation patterns
and reconstruction errors at different layer depths. As shown
in the top row and left column of Figure 2(a), activations from
the first Transformer layers display smooth, structured spatial
patterns characterized by gradual value transitions and consis-
tent vertical patterns across tokens. These vertical structures
indicate that the same neurons activate consistently across
different input tokens, reflecting a shared feature extraction
mechanism that operates across diverse inputs. In contrast,
the bottom row reveals that deeper layer activations exhibit
chaotic, high-frequency noise with abrupt value changes and
inconsistent activation patterns across tokens.
The smoothness of early-layer activations stems from funda-
mental properties of the Transformer architecture, particularly
the phenomenon of shared feature amplification in early layers.
As reported in [39], when an LLM processes different input
tokens, the neuron activation patterns in its first few layers
(especially those around the first layer) all show higher sim-
ilarity. This strongly indicates that the early layers activate a
common set of parameters to perform a foundational analysis
of the language, identifying its shared, underlying properties.
In summary,We find that early-layer activations are
uniquely compressible due to their local attention patterns
and structured spatial correlations, whereas deeper layers
develop complex, high-entropy representations that resist ag-
gressive compression.The compression error (achieved by
FourierCompress and existing Top-k/SVD benchmarks) in
right columns of Figure 2(a) validates that reconstruction
errors in early layers remain confined to low-magnitude re-
gions, suggesting that the essential semantic information is
preserved despite compression. Conversely, deep layers exhibit
widespread, high-magnitude errors (intense red patches) that
disrupt critical semantic information, even when applying the
same compression ratio. This asymmetry demonstrates that a
uniform compression approach across all layers is not practical
and establishes the necessity for layer-aware compression
strategies. Our empirical evidence also demonstrates that the
first Transformer layer serves as the optimal split point for
activation compression, as its activations maintain maximum
structural redundancy while preserving downstream reasoning
capabilities.
(2) Quantitative Analysis of Activation Similarity.To quan-
titatively validate our observation, Figure 2(b) presents acti-
vation similarity measurements across four diverse datasets
(PiQA, ARC-Easy, CommonsenseQA, and OpenBookQA) as
a function of layer depth. The results reveal a consistent pattern
across all datasets: activation similarity remains high in early
layers (peaking around Layers 1-3) but declines sharply as we
progress through the network. This trend indicates that early
layers function as general, task-agnostic feature extractors,
maintaining consistent activation patterns across different in-
puts and tasks. The high similarity values (approaching 0.8 in
some cases) confirm that these layers concentrate information
along a few principal directions with significant structural
redundancy. As we move to deeper layers, the similarity
metrics decrease substantially (often below 0.4), reflecting a
transition to complex, highly contextualized representations
that are unique to each token’s role within the sequence.
Figure 2(b) reveals two critical insights: (1) Early layers
exhibit strong cross-dataset consistency in activation patterns,
confirming their role as general feature extractors; (2) Deeper
layers develop increasingly task-specific representations with
lower inherent redundancy, making them inherently less com-
pressible. This transition from general to specific representa-
tions explains why early layers are uniquely suitable for high-
ratio compression while preserving inference accuracy.
B. Rationale of First-Layer Spectral Activation Compression
Activations from early Transformer layers, particularly the
first layer, exhibit strong spectral concentration in the low-
frequency domain. As illustrated in Figure 2(c), the 2D Fourier

<!-- page 5 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
Original Activations FC reconstruction(ours) FC Error: 2.1 Top-k Error: 2.5 SVD Error: 2.4
Layer1
Original Activations FC reconstruction(ours) FC Error: 5.8 Top-k Error: 6.0 SVD Error: 6.9
Layer 8
Original Activations FC reconstruction(ours) FC Error: 19.0 Top-k Error: 23.5 SVD Error: 20.0
Layer 15
0.4
0.3
0.2
0.1
0.0
-0.1
-0.2
-0.3
-0.4
(a) Rows (from top to bottom) show results for different split layers. From left to right is the original activation, the activation reconstructed
by FourierCompress, and the errors generated by different compression methods. Coordinate values are color coded (positive, negative). As
the split layer increases, activation becomes less smooth (more chaotic) and errors also increase.
(b) Activation similarity across different datasets(PiQA, ARC-Easy, CommonsenseQA, and OpenBookQA) with increasing layers.
Energy Distribution
Low-Frequency Energy:  52.2%
…
…
…
…
…
…
Frequency
Frequency
Low Frequency
Frequency Distribution
Original Activation FC reconstruction(ours)
Tokens
Tokens
Hidden Size Hidden Size
(c) Certain components of the activation, specifically the low-frequency information, exhibit a higher energy distribution, allowing the entire
activation to be reconstructed from them.
Fig. 2: Activations across different LLM layers exhibit significant differences. Activations at Layer 1 show high similarity in
information and energy distribution across different tokens, observed with Llama 3-1B. This property enables FourierCompress
to achieve significantly higher compression ratios.
spectrum of the activation tensor from Layer 1 of Llama 3-
1B reveals that the majority of its energy is tightly clustered
around the origin (i.e., low-frequency components). This con-
centration implies that the essential structural and semantic
information of the activation can be accurately captured using
only a small subset of low-frequency coefficients.
This phenomenon is a direct consequence of the smooth spa-
tial patterns observed in early-layer activations in Figure 2(a),
which reflect consistent neuron responses across input tokens.
As a result, by retaining only the dominant low-frequency
block submatrix in the top-left corner of the 2D spectrum),
spectral compression achieves high compression ratios while
preserving the core signal structure. The reconstruction via
inverse FFT of this truncated spectrum yields activations that
closely approximate the original, as confirmed by the low
reconstruction errors in Figure 2(a). This spectral sparsity
provides a principled, signal-theoretic justification for applying
FFT-based compression specifically at the first Transformer
layer, establishing both the layer-awareness and near-lossless
fidelity that distinguish FourierCompress from generic com-
pression baselines.

<!-- page 6 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
Mobile
devices Split LLM
Channel
Split LLM
Server
Low Frequency
Token
①
③
…
…
…
…
…
…
Frequency
Frequency
Frequency Domain
Spatial Domain
FFT
iFFT
Low
Frequency
Activation Compression
with FourierCompress
…
…
…
…
…
…
Hidden size
Tokens
…
…
…
…
…
…
Hidden size
Tokens
②
① Frequency-Domain Transformation
③ Conjugate Symmetry-Aware Reconstruction
② Low-Frequency Coefficient Retention
Fig. 3: The FourierCompress framework for efficient LLM inference, where the activation is compressed via Fourier transform
at the device before transmitting to the edge server.
C. Proposed FourierCompress Framework
As shown in Figure 3, the core of our method is to treat the
entire activation tensor from the first layer, a matrix whose
dimensions correspond to the sequence length (S) and the
hidden size (D), as a single 2D signal. This approach allows
us to simultaneously capture correlations both across neurons
(the hidden dimension) and between tokens (the sequence
dimension). By transforming this 2D signal into the frequency
domain, we can isolate and preserve the structurally significant
low-frequency components that represent the shared, smooth
patterns. The framework consists of three sequential stages:
(1) a 2D frequency-domain transformation, (2) low-frequency
coefficient retention for compression, and (3) a conjugate
symmetry-aware reconstruction. More details are as follows:
(1) Frequency-Domain Transformation.For a given input,
the activation output of the first Transformer layer is a real-
valued matrixA∈R S×D, whereSis the sequence length and
Dis the model’s hidden size [16]. We interpret this matrix as
a 2D signal, where one axis represents the token sequence
and the other represents the neuron activations. The first step
is to apply a 2D Fast Fourier Transform (FFT) [32] to this
entire matrix. This converts the spatio-temporal signalAinto
its 2D frequency-domain representation, a matrix of complex
numbersA ∈C S×D, computed as:
A[u, v] =
S−1X
s=0
D−1X
d=0
A[s, d]·e −j2π( us
S + vd
D )
where(u, v)are the frequency coordinates. This transfor-
mation decomposes the activation patterns into constituent
frequencies along both the sequence and hidden dimensions.
The low-frequency components (whereuandvare small),
located near the origin of the 2D spectrum, represent the
smooth, dominant patterns that are consistent across both
tokens and neurons, effectively capturing the shared structure.
(2) Low-Frequency Coefficient Retention.With the activa-
tions transformed into a 2D frequency spectrum, compression
is achieved by retaining a rectangular block of the most in-
formative, low-frequency coefficients. We select cutoff points
KS andK D (whereK S ≪SandK D ≪D) based on
the target compression ratio. We then preserve only the top-
leftK S ×K D block of coefficients from the spectrumA.
This truncation strategy is principled: it keeps the coefficients
corresponding to low frequencies in both dimensions, which
encode the essential shared structure, while discarding the
high-frequency coefficients that represent less critical, token-
specific and neuron-specific variations. By transmitting only
this smallerK S ×K D matrix of complex numbers, the data
volume is substantially reduced.
(3) Conjugate Symmetry-Aware Reconstruction.On the re-
ceiving end, the originalS×Dactivation matrix is recon-
structed from the receivedK S ×K D block of coefficients.
This process relies on the fact that the 2D Fourier trans-
form of a real-valued matrix exhibits conjugate symmetry
(A[u, v] =A ∗[S−u, D−v]), making the high-frequency
components inherently redundant. In practice, the reconstruc-
tion is performed efficiently by creating a newS×Dmatrix,
placing the receivedK S ×KD block in the top-left corner, and
padding the rest of the matrix with zeros. This zero-padded
matrix,A padded, effectively serves as a low-pass filtered version
of the original spectrum. Finally, the server applies the 2D
Inverse Fast Fourier Transform (IFFT) toA padded to obtain the
reconstructed activation matrixA ′ ∈R S×D:
A′[s, d] = 1
SD
S−1X
u=0
D−1X
v=0
Apadded[u, v]·e j2π( us
S + vd
D )
This method provides a highly efficient, metadata-free re-

<!-- page 7 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
construction, as the position of the coefficients is implicitly
known.
D. Hardware Integration and Performance Analysis
FourierCompress is designed not only for high compres-
sion fidelity but also for practical deployment on resource-
constrained edge platforms. Its core operation FFT benefits
from decades of hardware optimization and is natively sup-
ported across a wide range of edge accelerators, including
DSPs and FPGAs. This section details our hardware-aware
implementation strategy and analyzes its impact on system-
level performance.
(1) Hardware Acceleration.The computational complex-
ity of FourierCompress is dominated by the 2D FFT and
its inverse, both of which scale asO(SDlog(SD)), where
Sis the sequence length andDis the hidden dimension.
This is substantially more efficient than theO(SD 2)or
higher complexity of low-rank methods like SVD or QR
decomposition. More importantly, FFT is highly amenable to
parallelization and fixed-function hardware acceleration, and
particularly suitable for edge deployment. We implemented
and evaluated FourierCompress in two representative edge
hardware environments:
•GPU/DSP Acceleration:On NVIDIA Jetson platforms,
we leveraged cuFFT, a highly optimized CUDA library, to
perform real-time FFT/IFFT operations. This yields sig-
nificant speedups over general-purpose CPU execution.
•FPGA Implementation:On the Alinx AXU15EGB FPGA
platform, we deployed a custom FFT core using a
pipelined architecture that exploits dedicated DSP slices.
This design achieves over 10×throughput improvement
compared to software baselines by maximizing data reuse
and minimizing memory access latency.
(2) Performance Evaluation Metrics.The hardware-aware
design of FourierCompress translates into dramatic reductions
in compression latency. As will shortly shown in Section IV ,
our hardware-accelerated implementation reduces activation
compression time by up to 32×compared to Top-ksparsi-
fication and by over two orders of magnitude compared to
SVD-based approaches. Crucially, this acceleration incurs only
a 0.3% average overhead in total end-to-end inference latency,
making it suitable for real-time edge applications.
This combination of algorithmic efficiency, near-lossless
reconstruction, and seamless hardware integration establishes
FourierCompress as a practical and scalable solution for mit-
igating the communication bottleneck in collaborative LLM
inference, particularly in emerging 6G edge AI scenarios
where bandwidth, latency, and energy are tightly constrained.
IV. EXPERIMENTALRESULTS
This section presents a comprehensive empirical evaluation
of FourierCompress (FC) across four different LLM models,
ten popular datasets, and FPGA/Jetson hardware platforms.
We assess its performance in terms of inference accuracy,
compression ratio, computational overhead, and system-level
scalability under realistic collaborative inference scenarios.
Fig. 4: Comparison of Split Layer and Accuracy in Llama 3.
Fig. 5: Comparison of Compression Ratio and Accuracy in
Llama 3.
A. Experimental Setup
Models and Datasets:We conducted experiments using
two families of LLMs: Llama 3 [42] (specifically, Llama 3-
1B and Llama 3-3B) and Qwen2.5 [43] (Qwen2.5-1.5B and
Qwen2.5-3B). To assess the generalization capabilities of the
compression methods, we evaluated them on a comprehensive
suite of 10 commonsense reasoning datasets: OpenBookQA
[44] (OA), ARC-Easy [45] (A-e), ARC-Challenge [45] (A-c),
PIQA [46] (PA), SIQA [47] (SA), WinoGrande [48] (WG),
CommonsenseQA [49] (CQ), QASC [50] (QC), LogiQA [51]
(LA), and CosmosQA [52] (CA). The primary evaluation
metric is accuracy (reported in percentages).
Hardware and software:The experimental environment
featured eight NVIDIA RTX 4090 GPUs for general simula-
tion tasks and establishing performance baselines. Jetson-DSP
and Zynq-FPGA were used for the accelerated computation of
FourierCompress, simulating its performance on edge devices.
Core software included Python 3.11, PyTorch 2.7.0 (built with

<!-- page 8 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
TABLE II: Different datasets have different FourierCompress ratios. Other methods were also evaluated using the same
compression ratios as FourierCompress for fair comparison.
Model Compression RatioOA A-e A-c PA SA WG CQ QC LA CA
10 36.5 53.7 37.4 74.6 35.4 54.3 40.9 20.7 30.6 23.5
9 37.4 55.2 38.1 74.6 38.7 55.2 41.1 21.2 32.1 25.3
8 37.4 56.0 39.5 74.6 41.8 56.1 41.2 21.2 34.6 26.5
7 37.4 56.2 39.7 74.6 42.5 56.3 42.3 22.3 34.1 27.1
6 37.4 56.4 39.8 74.6 44.2 57.8 42.3 22.4 33.3 27.3
Llama 3-1B
Baseline 37.4 56.5 39.8 74.6 44.0 57.9 42.5 22.4 33.3 27.3
10 38.5 58.7 39.4 76.5 40.1 56.5 42.3 22.2 35.2 26.8
9 40.8 59.3 40.3 76.4 42.9 56.9 43.4 23.6 37.4 28.4
8 41.3 62.1 42.1 76.4 45.8 59.7 43.1 24.2 39.5 30.5
7 41.2 62.5 42.5 76.4 46.9 60.4 44.3 25.8 40.1 31.4
6 41.6 62.5 42.8 76.4 46.6 63.5 44.3 26.0 40.5 31.4
Llama 3-3B
Baseline 41.6 62.5 42.8 76.4 46.6 63.5 44.5 26.0 40.7 31.4
10 39.6 60.8 40.2 73.7 44.3 56.7 30.8 21.2 40.2 20.7
9 39.1 61.6 41.3 73.8 44.7 56.8 30.4 21.2 40.5 21.5
8 38.4 62.5 42.8 73.8 46.2 57.2 31.7 21.6 40.7 22.1
7 38.4 63.5 44.1 73.8 48.4 57.2 33.0 21.9 41.1 22.6
6 38.4 63.7 44.2 73.8 50.2 57.7 33.5 22.3 41.8 22.6
Qwen2.5-1.5B
Baseline 38.4 63.7 44.2 73.8 49.1 58.8 33.7 22.3 42.0 22.6
10 38.7 62.1 42.2 75.7 43.1 59.2 32.3 23.5 35.7 23.7
9 40.4 64.4 42.6 75.7 45.7 60.4 32.4 23.8 36.6 25.4
8 40.4 64.0 43.5 75.6 48.3 62.5 34.7 25.6 37.0 26.6
7 40.4 64.0 43.5 75.7 48.6 62.8 34.2 26.7 35.8 27.2
6 40.4 64.0 43.5 75.7 49.2 64.0 34.2 27.9 35.9 27.6
Qwen2.5-3B
Baseline 40.4 64.0 43.5 75.7 49.8 63.9 34.2 27.9 35.8 27.6
Avg. Compression Ratio 8.7 8.4 8.3 10.3 6 5.8 7.5 6.8 8.0 6.6
CUDA 11.8).
Compared Methods:We compare our proposed Fourier-
Compress framework against a comprehensive suite of rep-
resentative compression techniques, all applied directly to
the intermediate activation tensors for a fair evaluation. The
performance upper bound is established by a Baseline, where
no compression is applied. We evaluate the following methods:
1)Top-k: [24] retains only thekactivation values with the
largest absolute magnitudes for sparsification.
2)FWSVD[25]: A state-of-the-art SVD variant which
applies a weighted factorization.
3)ASVD[26]: A state-of-the-art SVD variant which trans-
forms weights based on activation statistics.
4)SVD-LLM[27]: A state-of-the-art SVD variant which
uses data whitening to guide decomposition.
5)QR Decomposition (QR)[53]: A classical low-rank
factorization method included alongside the SVD vari-
ants.
6)Baseline: No compression (upper-bound performance).
B. Accuracy and Compression Performance
(1) Validation of Early-Layer Splitting.Figure 4 validates
our core design principle: splitting at the first Transformer
layer is essential for high-fidelity activation compression.
On Llama 3-1B, we compare the performance of Fourier-
Compress against other methods across four datasets (PIQA,
OpenBookQA, CommonsenseQA, and ARC-Easy) at their
respective optimal compression ratios (ranging from 7.5×to
10.3×; more details in Table II). At the first layer, all methods
achieve relatively high accuracy, with FourierCompress being
the most accurate. However, as the split layer moves to deeper
layers (e.g., layer 5 or 15), the accuracy of all methods drops
sharply. For instance, on PIQA, FourierCompress’s accuracy
falls from 74.6% at Layer 1 to 48% at Layer 15. This sharp
degradation confirms that deeper activations lack the structural
redundancy required for aggressive compression, reinforcing
the necessity of layer-aware splitting.
(2) Dataset-Adaptive Near-Lossless Compression Ratios.
Table II presents a fine-grained ablation to identify the max-
imum compression ratio that incurs less than 0.3% accuracy
loss relative to the uncompressed baseline—our definition of
“near-lossless.” Results reveal significant dataset-dependent
variability: PIQA supports up to 10.3×compression due to
its high spectral redundancy, whereas WinoGrande is more
sensitive, tolerating only 5.8×.The average compression
ratio across all 10 datasets is 7.6×, which we adopt as the
standard evaluation setting for fair comparison in Table III.
(3) Accuracy Comparison at Fixed Compression Ratios.
Table III presents a comparison of inference accuracy, with
each method evaluated at the compression ratios determined in
Table II.FourierCompress consistently preserves accuracy
to within 0.3% of the uncompressed baseline across all
four models and ten datasets.In stark contrast, competing
methods suffer substantial degradation under the same aggres-
sive compression ratios. For instance, on Llama 3-1B with the
CommonsenseQA dataset, FourierCompress achieves 42.3%
accuracy (a negligible 0.2-point drop), whereas SVD-LLM
and Top-kplummet to 33.4% and 31.0%, respectively—a
performance loss of over 9 points. This highlights a funda-
mental limitation of methods like Top-k, which disrupt the
spatial coherence of the activation tensor, and SVD-based
approaches, which are ill-suited for capturing the complex
structures inherent in activations. Remarkably, on Qwen2.5-
3B, FourierCompress’s average accuracy (46.4%) slightly ex-
ceeds the baseline (46.3%). We attribute this to a beneficial
regularization effect; by design, FourierCompress acts as a
low-pass filter, discarding high-frequency components that

<!-- page 9 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
TABLE III: Comparison of Accuracy at the Same Compression Ratio.
Model Method OA A-e A-c PA SA WG CQ QC LA CA Avg.
FWSVD [25] 27.0 44.5 29.8 67.0 41.4 52.9 31.5 19.9 32.3 24.0 37.0 (↓6.6)
ASVD [26] 27.4 45.4 31.1 67.0 41.1 53.9 31.4 20.1 32.8 23.7 37.4 (↓6.2)
SVD-LLM [27] 28.5 47.0 32.0 69.3 43.9 55.6 33.4 20.9 33.2 25.1 38.9 (↓4.7)
QR [53] 28.1 44.6 33.3 64.0 37.2 49.2 28.5 18.6 31.6 22.5 35.8 (↓7.8)
Top-k[24] 29.8 47.2 29.8 65.0 40.6 53.0 31.0 19.3 32.1 23.3 37.1 (↓6.5)
FC 37.4 56.0 39.5 74.6 44.2 57.8 42.3 22.3 34.6 27.1 43.6 (↓0.0)
Llama 3-1B
Baseline37.4 56.5 39.8 74.644.057.9 42.5 22.433.327.3 43.6
FWSVD [25] 27.8 52.7 38.1 73.7 44.3 62.1 43.6 23.9 39.2 30.4 43.6 (↓4.0)
ASVD [26] 28.1 52.8 38.1 73.6 44.3 62.4 43.9 24.1 39.2 30.4 43.7 (↓3.9)
SVD-LLM [27] 28.3 53.3 38.7 73.4 44.8 62.2 43.5 24.3 40.9 30.9 44.0 (↓3.6)
QR [53] 31.5 53.4 37.7 74.3 41.5 59.1 41.5 23.2 40.2 31.5 43.4 (↓4.2)
Top-k[24] 38.6 64.2 41.1 75.3 45.0 62.8 44.1 24.0 40.7 32.3 46.8 (↓0.8)
FC 40.8 62.1 42.1 76.5 46.9 63.5 44.3 25.8 39.5 31.4 47.3(↓0.3)
Llama 3-3B
Baseline41.6 62.5 42.876.4 46.663.5 44.5 26.0 40.7 31.4 47.6
FWSVD [25] 30.2 48.9 31.6 72.6 49.1 57.6 32.4 21.0 39.2 22.0 40.5 (↓4.3)
ASVD [26] 31.1 50.2 32.2 72.6 49.6 57.1 32.2 20.1 39.9 22.4 40.7 (↓4.1)
SVD-LLM [27] 31.9 51.5 33.3 72.8 50.0 57.6 33.3 21.1 40.7 22.3 41.4 (↓3.4)
QR [53] 31.6 47.4 41.5 70.5 46.4 55.1 31.5 19.9 37.1 22.1 40.3 (↓4.5)
Top-k[24] 38.0 62.2 42.5 72.8 48.1 56.9 32.0 21.3 38.5 21.9 43.4 (↓1.4)
FC 39.6 62.5 42.8 73.7 50.2 57.7 33.0 21.9 40.7 22.6 44.5(↓0.3)
QWen2.5-1.5B
Baseline 38.463.7 44.2 73.849.158.8 33.7 22.3 42.0 22.6 44.8
FWSVD [25] 29.3 50.5 36.9 73.6 48.9 62.8 33.5 26.4 33.1 26.5 42.1 (↓4.2)
ASVD [26] 29.2 50.3 36.9 73.2 48.6 62.2 33.2 26.4 33.1 26.3 41.9 (↓4.4)
SVD-LLM [27] 29.7 51.2 37.5 74.3 49.8 63.8 34.0 26.9 33.7 26.8 42.8 (↓3.5)
QR [53] 27.5 54.9 38.5 73.3 48.2 62.7 32.9 26.9 32.6 25.9 42.3 (↓4.0)
Top-k[24] 39.2 62.6 42.2 74.1 48.9 63.4 33.6 27.7 33.3 26.6 45.2 (↓1.1)
FC 40.4 64.4 43.5 75.7 49.2 64.0 34.7 27.9 37.0 27.2 46.4↑0.1
QWen2.5-3B
Baseline40.464.043.5 75.7 49.863.9 34.227.935.827.646.3
TABLE IV: Total time for activation compression and decompression (s).
LLM Hidden Size FWSVD ASVD SVD-LLM QR Top-kFC(software) FC (hardware)
Llama 3-1B 2048 90.4 197.8 94.2 2354.0 22.8 6.9 1.0
Llama 3-3B 3072 327.6 212.7 141.8 2076.4 30.9 6.7 0.7
Qwen2.5-1.5B 1536 141.8 113.4 81.0 2027.5 15.6 6.9 0.6
Qwen2.5-3B 2048 11.5 171.5 89.6 2057.9 20.4 4.9 0.5
Avg. — 142.8 173.9 101.7 2129.0 22.46.4 0.7
may represent noise. This implicit denoising can improve the
signal-to-noise ratio for subsequent layers, thereby enhancing
generalization.
(4) Robustness Across Compression Ratios.Figure 5 exam-
ines the accuracy–compression trade-off on Llama 3. Fouri-
erCompress exhibits graceful degradation. In stark contrast,
SVD-based methods collapse rapidly—SVD-LLM’s accuracy
drops by more than 8 points at the same ratio—highlighting
FourierCompress’s superior ability to retain semantically criti-
cal information under aggressive compression. This robustness
stems directly from its exploitation of the strong low-frequency
energy concentration in early-layer activations (Figure 2(c)),
which ensures that discarded coefficients contribute minimally
to overall signal fidelity.
C. Compression Efficiency and Hardware Acceleration
To assess the practical viability of FourierCompress for real-
time edge deployment, we evaluate both the absolute compres-
sion/decompression latency and its contribution to end-to-end
inference time. As shown in Table IV, traditional low-rank
methods incur prohibitive computational overhead. For in-
stance, QR decomposition requires an average of 2129.0s, and
SVD-LLM takes 101.7 s to compress activations from input
Fig. 6: Proportion of compression time in the response time.
across four LLMs (Llama 3-1B/3B and Qwen2.5-1.5B/3B).
Even the relatively lightweight Top-ksparsification demands
22.4s on average, which remains impractical for interactive
applications.
In contrast, the software implementation of FourierCom-
press achieves a compression time of only 6.4s—already 3.5×
faster than Top-kand>15×faster than SVD-LLM. This
efficiency stems from the near-linear complexity of the 2D
FFT (O(SDlog(SD)) ) and its highly parallelizable structure.

<!-- page 10 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
(a) The server has an NVIDIA GeForce RTX 4090. As the number of clients increases, the average response time grows
rapidly for both Original Activations and FC. This response time is not reduced by improving network speed.
(b) The server has 8 NVIDIA GeForce RTX 4090s. As the number of clients increases, the average response time for Original
Activations grows rapidly, while FC reduces the average response time. With higher network speeds, FC can support more
clients.
Fig. 7: Comparison of Average Client Response Time with Different Server Computational Capabilities and Network Speeds
(Gbps).
More importantly, FourierCompress is uniquely suited for
hardware acceleration. By leveraging dedicated FFT engines
on edge platforms—such as cuFFT on NVIDIA Jetson GPUs
or custom pipelined FFT cores on FPGAs—the compression
time drops dramatically to an average of 0.7 s.FC represents
32×speedup over Top-kand over two orders of magnitude
improvement compared to SVD-based approaches.
Figure 6 contextualizes this gain within the full inference
pipeline, showing the proportion of compression time relative
to total response latency (including client-side processing,
transmission, and server-side computation). Low-rank methods
dominate the latency budget: QR and SVD-LLM consume
66% and 53% of total time, respectively. While Top-kand
software-based FourierCompress introduce modest overheads
(3% and 4%), hardware-accelerated FourierCompress reduces
this to just 0.3%. This near-negligible overhead confirms that
FourierCompress is not only accurate and communication-
efficient but also computationally lightweight—making it ideal
for real-time, resource-constrained edge AI systems.
D. Multi-Client Scalability Under Varying 6G Network Con-
ditions
To evaluate the practical impact of FourierCompress in real-
world collaborative inference scenarios, we simulate a multi-
client edge computing environment under different 6G wire-
less data rates (1 Gbps, 3 Gbps, 5 Gbps, and 10 Gbps), using
the Llama 3 model and the PIQA dataset. We analyze system
behavior under two distinct resource regimes to identify the
dominant performance bottleneck, as illustrated in Figure 7.
(1) Computation-constrained Regime.In Figure 7(a), the
edge server is equipped with a single NVIDIA RTX 4090
GPU. Here, computational capacity—not communication
bandwidth—limits system throughput. As the number of con-
current clients increases beyond approximately 10, the server
becomes saturated, causing average response time to rise
sharply for both the uncompressed baseline and FourierCom-
press. Crucially, improving network speed yields negligible
latency reduction in this regime, confirming that when com-
putation is the bottleneck, activation compression provides
minimal benefit.
(2) Bandwidth-constrained Regime.In contrast, Figure 7(b)
depicts a bandwidth-constrained scenario where the server is
scaled to eight RTX 4090 GPUs, providing ample compute
resources. Under this configuration, communication becomes
the primary bottleneck. FourierCompress dramatically reduces
average client response time across all tested network speeds
by shrinking activation payloads by an average of 10.3×.
Moreover, it significantly enhances system scalability: at 10
Gbps, the system supports over 1,500 concurrent clients be-
fore saturation, compared to only 150 clients with uncom-
pressed activations. This near 10×increase in client capacity
demonstrates that FourierCompress is not just an incremental
improvement but a critical enabling technology. It effectively
shifts the system bottleneck away from the communication
channel and back to the more easily scalable domain of
computation. This allows system operators to fully exploit the
high-throughput capabilities of future 6G networks, making ef-
ficient and large-scale collaborative LLM inference a practical
reality.
V. CONCLUSION
FourierCompress effectively addresses the critical trade-off
between communication efficiency, inference accuracy, and
compression overhead. By strategically exploiting the unique
spectral characteristics of early-layer LLM activations, it offers

<!-- page 11 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
a robust and efficient pathway towards enabling scalable
and privacy-preserving collaborative LLM inference on edge
devices.
REFERENCES
[1] I. WP5D, “Future technology trends of terrestrial international mobile
telecommunications systems towards 2030 and beyond,”International
Telecommunication Union, Report M, pp. 2516–0, 2022.
[2] F. Guo, F. R. Yu, H. Zhang, X. Li, H. Ji, and V . C. Leung, “Enabling
Massive IoT Toward 6G: A Comprehensive Survey,”IEEE Internet of
Things Journal, vol. 8, no. 15, pp. 11 891–11 915, 2021.
[3] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, D. Niyato,
O. Dobre, and H. V . Poor, “6G Internet of Things: A Comprehensive
Survey,”IEEE Internet of Things Journal, vol. 9, no. 1, pp. 359–383,
2021.
[4] Z. Yu, Z. Wang, Y . Li, R. Gao, X. Zhou, S. R. Bommu, Y . Zhao,
and Y . Lin, “EDGE-LLM: Enabling Efficient Large Language Model
Adaptation on Edge Devices via Unified Compression and Adaptive
Layer V oting,” inProceedings of the 61st ACM/IEEE Design Automation
Conference, 2024, pp. 1–6.
[5] M. Zhang, X. Shen, J. Cao, Z. Cui, and S. Jiang, “EdgeShard: Efficient
LLM Inference via Collaborative Edge Computing,”IEEE Internet of
Things Journal, 2024.
[6] P. P. Ray and M. P. Pradhan, “LLMEdge: A Novel Framework for
Localized LLM Inferencing at Resource Constrained Edge,” in2024
International Conference on IoT Based Control Networks and Intelligent
Systems (ICICNIS). IEEE, 2024, pp. 1–8.
[7] J. Li, B. Han, S. Li, X. Wang, and J. Li, “CoLLM: A Collabora-
tive LLM Inference Framework for Resource-Constrained Devices,” in
2024 IEEE/CIC International Conference on Communications in China
(ICCC). IEEE, 2024, pp. 185–190.
[8] C. Yang, Y . Zhu, W. Lu, Y . Wang, Q. Chen, C. Gao, B. Yan, and Y . Chen,
“Survey on Knowledge Distillation for Large Language Models: Meth-
ods, Evaluation, and Application,”ACM Transactions on Intelligent
Systems and Technology, 2024.
[9] X. Zhu, J. Li, Y . Liu, C. Ma, and W. Wang, “ASurvey on Model Com-
pression for Large Language Models,”Transactions of the Association
for Computational Linguistics, vol. 12, pp. 1556–1577, 2024.
[10] A. Liu, J. Liu, Z. Pan, Y . He, G. Haffari, and B. Zhuang, “MiniCache:
KV Cache Compression in Depth Dimension for Large Language
Models,”Advances in Neural Information Processing Systems, vol. 37,
pp. 139 997–140 031, 2024.
[11] A. Mudvari, Y . Jiang, and L. Tassiulas, “SplitLLM: Collaborative
Inference of LLMs for Model Placement and Throughput Optimization,”
arXiv preprint arXiv:2410.10759, 2024.
[12] X. Chen, W. Wu, L. Li, and F. Ji, “LLM-Empowered IoT for 6G
Networks: Architecture, Challenges, and Solutions,”IEEE Internet of
Things Magazine, 2025.
[13] X. He, Y . Jiang, X. Xu, H. Cui, Y . Liu, M. Chen, Y . Hong, and
J. Zhang, “Large Language Model Offloading using Active Inference
in 6G Symbiotic IoT,”IEEE Internet of Things Journal, 2025.
[14] D. Cao, J. Wu, and A. K. Bashir, “Multimodal Large Language Models
Driven Privacy-Preserving Wireless Semantic Communication in 6G,”
in2024 IEEE International Conference on Communications Workshops
(ICC Workshops). IEEE, 2024, pp. 171–176.
[15] S. Long, F. Tang, Y . Li, T. Tan, Z. Jin, M. Zhao, and N. Kato, “6G
comprehensive intelligence: Network operations and optimization based
on Large Language Models,”IEEE Network, 2024.
[16] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention Is All You Need,”Advances in
neural information processing systems, vol. 30, 2017.
[17] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
Technical Report,”arXiv preprint arXiv:2303.08774, 2023.
[18] I. Ong, “Efficient Distributed LLM Inference with Dynamic Partition-
ing,”California, Berkeley, Technical Report UCB/EECS-2024-108, May,
2024.
[19] C. Liu and J. Zhao, “Resource Allocation in Large Language Model
Integrated 6G Vehicular Networks,” in2024 IEEE 99th Vehicular
Technology Conference (VTC2024-Spring). IEEE, 2024, pp. 1–6.
[20] L. Qian and J. Zhao, “User Association and Resource Allocation in
Large Language Model Based Mobile Edge Computing System over
6G Wireless Communications,” in2024 IEEE 99th Vehicular Technology
Conference (VTC2024-Spring). IEEE, 2024, pp. 1–7.
[21] M. Haider, I. Ahmed, Z. Hassan, K. Hasan, and H. V . Poor, “LLM-
Integrated Digital Twins for Hierarchical Resource Allocation in 6G
Networks,”arXiv preprint arXiv:2506.18293, 2025.
[22] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao,
X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quanti-
zation for llm compression and acceleration,”Proceedings of machine
learning and systems, vol. 6, pp. 87–100, 2024.
[23] X. Shen, P. Dong, L. Lu, Z. Kong, Z. Li, M. Lin, C. Wu, and Y . Wang,
“Agile-quant: Activation-guided quantization for faster inference of llms
on the edge,” inProceedings of the AAAI Conference on Artificial
Intelligence, vol. 38, no. 17, 2024, pp. 18 944–18 951.
[24] S. Zhang, G. Cheng, W. Wu, X. Huang, L. Song, and X. Shen, “Split
fine-tuning for large language models in wireless networks,”IEEE
Journal of Selected Topics in Signal Processing, pp. 1–16, 2025.
[25] Y . Hsu, T. Hua, S. Chang, Q. Lou, Y . Shen, and H. Jin, “Language
model compression with weighted low-rank factorization,” inThe Tenth
International Conference on Learning Representations, ICLR 2022,
Virtual Event, April 25-29, 2022. OpenReview.net, 2022. [Online].
Available: https://openreview.net/forum?id=uPv9Y3gmAI5
[26] Z. Yuan, Y . Shang, Y . Song, Q. Wu, Y . Yan, and G. Sun, “Asvd:
Activation-aware singular value decomposition for compressing large
language models,”arXiv preprint arXiv:2312.05821, 2023.
[27] X. Wang, Y . Zheng, Z. Wan, and M. Zhang, “SVD-LLM:
truncation-aware singular value decomposition for large language
model compression,” inThe Thirteenth International Conference
on Learning Representations, ICLR 2025, Singapore, April 24-28,
2025. OpenReview.net, 2025. [Online]. Available: https://openreview.
net/forum?id=LNYIUouhdt
[28] H. Lin, H. Xu, Y . Wu, J. Cui, Y . Zhang, L. Mou, L. Song, Z. Sun, and
Y . Wei, “Duquant: Distributing outliers via dual transformation makes
stronger quantized llms,”Advances in Neural Information Processing
Systems, vol. 37, pp. 87 766–87 800, 2024.
[29] Y . An, X. Zhao, T. Yu, M. Tang, and J. Wang, “Systematic
outliers in large language models,” inThe Thirteenth International
Conference on Learning Representations, ICLR 2025, Singapore,
April 24-28, 2025. OpenReview.net, 2025. [Online]. Available:
https://openreview.net/forum?id=rLX7Vyyzus
[30] Y . Chen, R. Li, X. Yu, Z. Zhao, and H. Zhang, “Adaptive layer splitting
for wireless large language model inference in edge computing: a
model-based reinforcement learning approach,”Frontiers of Information
Technology & Electronic Engineering, vol. 26, no. 2, pp. 278–292, 2025.
[31] P. Patel, E. Choukse, C. Zhang, A. Shah, ´I. Goiri, S. Maleki, and
R. Bianchini, “Splitwise: Efficient generative llm inference using phase
splitting,” in2024 ACM/IEEE 51st Annual International Symposium on
Computer Architecture (ISCA). IEEE, 2024, pp. 118–132.
[32] J. W. Cooley and J. W. Tukey, “An algorithm for the machine calculation
of complex fourier series,”Mathematics of computation, vol. 19, no. 90,
pp. 297–301, 1965.
[33] Z. Qin, P. Zhang, F. Wu, and X. Li, “Fcanet: Frequency channel attention
networks,” inProceedings of the IEEE/CVF international conference on
computer vision, 2021, pp. 783–792.
[34] Z. Liu, Q. Song, Q. C. Xiao, S. K. Selvaraj, R. Mazumder, A. Gupta,
and X. Hu, “Ffsplit: Split feed-forward network for optimizing accuracy-
efficiency trade-off in language model inference,”arXiv preprint
arXiv:2401.04044, 2024.
[35] C. Li and C. Shi, “Constrained optimization based low-rank approx-
imation of deep neural networks,” inProceedings of the European
Conference on Computer Vision (ECCV), 2018, pp. 732–747.
[36] Z. Liu, J. Xu, X. Peng, and R. Xiong, “Frequency-domain dynamic prun-
ing for convolutional neural networks,”Advances in neural information
processing systems, vol. 31, 2018.
[37] Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary, R. Krishnamoorthi,
V . Chandra, Y . Tian, and T. Blankevoort, “Spinquant: LLM
quantization with learned rotations,” inThe Thirteenth International
Conference on Learning Representations, ICLR 2025, Singapore,
April 24-28, 2025. OpenReview.net, 2025. [Online]. Available:
https://openreview.net/forum?id=ogO6DGE6FZ
[38] Y . Zhao, C.-Y . Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Kr-
ishnamurthy, T. Chen, and B. Kasikci, “Atom: Low-bit quantization for
efficient and accurate llm serving,”Proceedings of Machine Learning
and Systems, vol. 6, pp. 196–209, 2024.
[39] Y . Wang, D. Dai, Z. Yang, J. Ma, and Z. Sui, “Exploring activation
patterns of parameters in language models,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 39, no. 24, 2025, pp. 25 416–
25 424.
[40] W. Li, L. Li, M. G. Lee, and S. Sun, “Adaptive layer sparsity for
large language models via activation correlation assessment,” inThe

<!-- page 12 -->

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
Thirty-eighth Annual Conference on Neural Information Processing
Systems, 2024. [Online]. Available: https://openreview.net/forum?id=
Jup0qZxH7U
[41] F. Zhang, Y . Liu, W. Li, J. Lv, X. Wang, and Q. Bai, “Towards
superior quantization accuracy: A layer-sensitive approach,”CoRR, vol.
abs/2503.06518, 2025. [Online]. Available: https://doi.org/10.48550/
arXiv.2503.06518
[42] AI@Meta, “Llama 3 Model Card,” 2024. [Online]. Available:
https://github.com/meta-llama/llama3/blob/main/MODEL CARD.md
[43] Qwen Team, “Qwen2.5: A party of foundation models,” September
2024. [Online]. Available: https://qwenlm.github.io/blog/qwen2.5/
[44] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a Suit of
Armor Conduct Electricity? A New Dataset for Open Book Question
Answering,” inEMNLP, 2018.
[45] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick,
and O. Tafjord, “Think you have solved question answering? Try ARC,
the AI2 reasoning challenge,”arXiv:1803.05457v1, 2018.
[46] Y . Bisk, R. Zellers, R. L. Bras, J. Gao, and Y . Choi, “PIQA: Reasoning
about physical commonsense in natural language,” inThirty-Fourth
AAAI Conference on Artificial Intelligence, 2020.
[47] M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y . Choi, “SocialIQA:
Commonsense reasoning about social interactions,” 2019.
[48] “WinoGrande: An adversarial Winograd schema challenge at scale,”
2019.
[49] A. Talmor, J. Herzig, N. Lourie, and J. Berant, “CommonsenseQA: A
question answering challenge targeting commonsense knowledge,” in
Proceedings of the 2019 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers). Minneapolis,
Minnesota: Association for Computational Linguistics, Jun. 2019, pp.
4149–4158. [Online]. Available: https://aclanthology.org/N19-1421
[50] T. Khot, P. Clark, M. Guerquin, P. Jansen, and A. Sabharwal, “QASC:
A dataset for question answering via sentence composition,” inThe
Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020,
The Thirty-Second Innovative Applications of Artificial Intelligence
Conference, IAAI 2020, The Tenth AAAI Symposium on Educational
Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA,
February 7-12, 2020. AAAI Press, 2020, pp. 8082–8090. [Online].
Available: https://doi.org/10.1609/aaai.v34i05.6319
[51] J. Liu, L. Cui, H. Liu, D. Huang, Y . Wang, and Y . Zhang,
“Logiqa: A challenge dataset for machine reading comprehension with
logical reasoning,” inProceedings of the Twenty-Ninth International
Joint Conference on Artificial Intelligence, IJCAI 2020, C. Bessiere,
Ed. ijcai.org, 2020, pp. 3622–3628. [Online]. Available: https:
//doi.org/10.24963/ijcai.2020/501
[52] L. Huang, R. Le Bras, C. Bhagavatula, and Y . Choi, “Cosmos
QA: Machine reading comprehension with contextual commonsense
reasoning,” inProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP).
Hong Kong, China: Association for Computational Linguistics, Nov.
2019, pp. 2391–2401. [Online]. Available: https://www.aclweb.org/
anthology/D19-1243
[53] Z. Zhuang, Z. Zhuang, and T. Wang, “Medical image encryption algo-
rithm based on a new five-dimensional multi-band multi-wing chaotic
system and QR decomposition,”Scientific Reports, vol. 14, no. 1, p.
402, 2024.
BIOGRAPHIES
JIAN MA(mj@bupt.edu.cn) is currently pursuing
the Ph.D. degree with the School of Cyberspace
Security at the Beijing University of Posts and
Telecommunications (BUPT). His research interests
include edge intelligence, split learning, and artificial
intelligence.
XINCHEN LYU(lvxinchen@bupt.edu.cn) received
the B.E. degree from BUPT in 2014, and the dual
Ph.D. degrees from BUPT and the University of
Technology Sydney in 2019. He is currently an
Associate Professor with the National Engineering
Research Center of Mobile Network Technologies,
BUPT. His research interests include resource man-
agement and security of edge intelligence and its
applications in future wireless networks.
JUN JIANG(jiangj@pcl.ac.cn) received the Ph.D.
degrees from the Harbin Institute of Technology
(HIT), Harbin, China, in 2009. He is currently a
Senior Engineer with the Pengcheng Laboratory
(PCL), Shenzhen, China. His current research inter-
ests include 3-D computer vision, SLAM, and deep
learning.
LONGHAO ZOU(zoulh@pcl.ac.cn) received the
BEng and PhD degrees from Beijing University of
Posts and Telecommunications, Beijing, China, and
Dublin City University (DCU), Dublin, Ireland, in
2011 and 2016, respectively. He was a postdoctoral
researcher with the European Union’s Horizon 2020
NEWTON Project at DCU, Dublin Ireland. Now he
is an associate researcher with Peng Cheng Lab-
oratory, Shenzhen, China, and also with Southern
University of Science and Technology, Shenzhen,
China. His research interests include mobile and
wireless communications, holographic communication, multi-sensorial inter-
action, resource allocation, and user quality of experience.
CHENSHAN REN(renchenshan06@163.com) re-
ceived the B.E. degree from Zhengzhou University,
Henan, China, in 2013, the first Ph.D. degree from
the Beijing University of Posts and Telecommunica-
tions, and the second Ph.D. degree from the Univer-
sity of Technology Sydney in 2019. She is currently
a Lecturer with the Minzu University of China. Her
research interests include fog computing, software-
defined networking, and radio resource management.
QIMEI CUI(cuiqimei@bupt.edu.cn) received the
B.E. and M.S. degrees from Hunan University in
2000 and 2003, respectively, and the Ph.D. degree
from the Beijing University of Posts and Telecom-
munications (BUPT) in 2006. She is currently a
Full Professor with the School of Information and
Communication Engineering, BUPT. Her research
interests include energy-efficient transmission theory
and networking technology for wireless and green
communications.
XIAOFENG TAO(taoxf@bupt.edu.cn) received the
B.S. degree in electrical engineering from Xi’an
Jiaotong University, Xi’an, China, in 1993, and
the M.S. and Ph.D. degrees in telecommunication
engineering from the Beijing University of Posts
and Telecommunications (BUPT), Beijing, China, in
1999 and 2002, respectively. He is currently a Full
Professor with BUPT, a fellow of the Institution of
Engineering and Technology, and the Chair of the
IEEE ComSoc Beijing Chapter. He is also with the
Pengcheng Laboratory (PCL), Shenzhen, China.
