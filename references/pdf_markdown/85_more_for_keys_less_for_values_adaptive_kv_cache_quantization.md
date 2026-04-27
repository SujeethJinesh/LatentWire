# references/85_more_for_keys_less_for_values_adaptive_kv_cache_quantization.pdf

<!-- page 1 -->

Quantize What Counts: More for Keys, Less for Values
Mohsen Hariri1, Alan Luo1, Weicong Chen1, Shaochen Zhong2, Tianyi Zhang2, Qifan Wang3,
Xia Hu2,Xiaotian Han 1,Vipin Chaudhary 1
1Case Western Reserve University; 2Rice University; 3Meta
mohsen.hariri@case.edu
Abstract
Large Language Models (LLMs) suffer
inference-time memory bottlenecks dominated
by the attention Key-Value (KV) cache, which
scales with model size and context length.
While KV-cache quantization alleviates this
cost, bit allocation between keys and values
is often tuned heuristically, lacking theoreti-
cal grounding and generalizability. This pa-
per proposes two theorems that anchor mixed-
precision KV quantization in the intrinsic ge-
ometry of Transformer models. First, key pro-
jections systematically have larger spectral and
Frobenius norms than value matrices, implying
higher information density along the key path.
Second, for any given memory budget, prior-
itizing precision for keys over values strictly
reduces quantization error and better preserves
accuracy. Empirical evaluations across vari-
ous prominent LLMs and benchmarks show
that key-favored allocations (e.g., 4-bit keys,
2-bit values) retain up to 98.3% accuracy com-
pared to uniform allocations (e.g., 4-bit for
both), while conserving memory. These results
transform bit allocation from ad hoc tuning
into a theoretically grounded, geometry-driven
design principle for efficient LLM inference.
Source code is available at https://github.
com/mohsenhariri/spectral-kv.
1 Introduction
Large Language Models (LLMs) have rapidly
scaled in recent years, driving major advances
in generative capabilities and reasoning perfor-
mance (Vaswani et al., 2017; Sutskever et al., 2014).
Model size has increased by several orders of mag-
nitude: GPT has grown from 117M parameters
in GPT-1 (Radford et al., 2018) to 1.5B in GPT-
2 (Radford et al., 2019), 175B in GPT-3 (Brown
et al., 2020), and 1.8T in GPT-4 (OpenAI et al.,
2024). Open-source models have followed a sim-
ilar trajectory, with Llama reaching 2T parame-
ters (Meta AI, 2025b), Mistral Large scaling to
0 20 40 60 80
Layer Index
0
200
400
600
800
1000
1200Spectral norm
Key
Value
Size K 2V4 K4V2
1B 0.06 0.34
8B 0.55 0.75
14B 0.78 0.91
70B 0.76 0.87
Figure 1:Key cache needs more bits.(Left): Spectral
norms of the key cache (blue) and value cache (orange)
across layers in Llama3.3-70B show that key caches
consistently exhibit higher norms. (Right): GSM8k
accuracy for two schemes: K2V4, representing 2-bit
allocation for the K cache and 4-bit allocation for the V
cache and K4V2, representing 4-bit allocation for the K
cache and 2-bit allocation for the V cache, demonstrates
that allocating more bits to the key cache maintains
strong performance, confirming the efficacy of norm-
aware, mixed-precision quantization.
123B (Mistral AI, 2024), and DeepSeek V3 to
671B (DeepSeek-AI et al., 2025b).
However, this rapid growth has introduced se-
vereinference-time memory bottlenecks, primar-
ily due to the Key-Value (KV) cache (Wei et al.,
2022). As parameter counts increase, context
lengths must also grow to support more complex
reasoning, which further expands the KV cache
and strains GPU memory (Sheng et al., 2023a).
Modern systems already support extremely long
contexts (Google DeepMind, 2025; OpenAI, 2024,
2025; DeepSeek-AI et al., 2025a), reaching up
to 10 million tokens (Meta AI, 2025a), making
memory-efficient methods essential.
KV cache quantizationrefers to reducing the
precision of KV tensors (e.g., BF16 to INT4),
which can provide substantial memory savings
while maintainingcontrolled accuracy degradation,
provided applied strategically (Han et al., 2016;
Nagel et al., 2021). Although many KV quantiza-
tion methods have been proposed, most determine
key–value bit splits through ad hoc hyperparame-
1
arXiv:2502.15075v3  [cs.LG]  15 Oct 2025

<!-- page 2 -->

ter tuning on cache statistics (e.g., inference-time
activations
) rather than grounding them in intrinsic model
properties (e.g., model weights). This raises a fun-
damental question:How should bits be allocated
in a principled and generalizable way?
To answer, Figure 1 illustrates two key observa-
tions. First, key caches (K) consistently have larger
spectral norms than value caches (V). Second, as-
signing more bits to keys (e.g., K4V2) improves
accuracy across multiple models compared to key-
underprovisioned allocations (K2V4). These moti-
vate a deeper investigation into the distinct roles of
key-value weights (i.e., WK and WV) in attention
mechanisms and their implications for quantization
performance. Toward this, our contributions are:
• We propose theKey-V alue Norm Disparitythe-
orem, proving that the expected spectral and
Frobenius norms of WK predominantly exceed
those of WV across prominent LLM models
(i.e., Llama3 and Mistral herds). We then de-
rive theKey-Prioritized Quantizationtheorem,
establishing the theoretical foundation on why
assigning higher precision to K than V strictly
reduces quantization error, enabling greater KV-
cache compression while maintaining accuracy.
• We corroborate and operationalize these theo-
rems across a diverse set of models (Llama-3.2-
1B/3B/8B, Llama-3.3-70B, Phi-4-14B, Qwen3-
0.6B/1.7B/4B/8B, DeepSeek-R1, Mistral-0.3-
7B), datasets (C4, MMLU, GSM8K, EQ-Bench,
CoQA, and LongBench1), and two quantization
backends (Optimum Quanto and HQQ).
Notably, K4V2 retains98.3%(1-shot) and
94.1%(5-shot) accuracy of K4V4 accuracy
while reducing KV-cache memory by25%,
demonstrating both the theoretical soundness
and practical effectiveness of the proposed strat-
egy.
• Owing to its efficient one-off tunability, we
show that our geometry-driven mixed-precision
strategy isorthogonalto existing inference-time
KV quantization methods and can be seam-
lessly integrated to yield synergistic gains. In
a case study with rotation-based outlier redis-
tribution, combining a key-prioritized quantiza-
tion (K4V2) withkey-only rotationoutperforms
K4V4 by4.4-18%in accuracy across tasks. In
1We purposefully select generative tasks rather than com-
monsense reasoning tasks to better isolate quantization effects;
see Section C.1 for details.
contrast, rotating value caches is unnecessary
and sometimes detrimental
2 Background and Related Work
KV Quantization.Quantization methods for
LLMs can be categorized by timing:training-time
andpost-training(PTQ) (Gholami et al., 2021).
Training-time quantization integrates quantization
into model training, typically achieving higher ac-
curacy by quantizing weights or activations dur-
ing the optimization process. However, it requires
labeled data and incurs significant training over-
head. PTQ applies quantization after training,
avoiding retraining costs and labeled data require-
ments (Nagel et al., 2021), but sometimes yields
lower accuracy.
PTQ can target different model components:
weights, activations, or the KV cache. Weight-
only quantization (Frantar et al., 2023a,b; Lin et al.,
2024) achieves strong accuracy but does not reduce
activation or KV memory. Weight-activation quan-
tization (Dettmers et al., 2022; Xiao et al., 2023;
Shao et al., 2024) reduces overall memory but of-
ten sacrifices accuracy. KV quantization offers
the best of both: it targets the rapidly growing KV
cache (Pope et al., 2023), providing activation-level
memory savings while maintaining the accuracy of
weight-only methods (Yue et al., 2024).
Existing KV Quantization Schemes.KV quan-
tization methods can be categorized by how they
treat keys and values (Li et al., 2025a):
• Outlier redistribution.These methods smooth
or relocate outliers (i.e., unusually large activa-
tion values that dominate quantization ranges)
in KV tensors, e.g., SmoothQuant (Xiao et al.,
2023), AWQ (Lin et al., 2024), and Omni-
Quant (Shao et al., 2024).
• Fixed-precision.A single bit-width is used for
both keys and values, ignoring their different
roles and statistical properties (Yao et al., 2022;
Sheng et al., 2023b; Liu et al., 2024).
• Mixed-precision.Different bit-widths are as-
signed to different parts of the cache (Hooper
et al., 2024; Yue et al., 2024; Li et al., 2024;
Dong et al., 2024; Duanmu et al., 2024; Zhang
et al., 2024; Li et al., 2025b).
While mixed-precision schemes are the most
flexible, existing methods have not systematically
exploredasymmetric bit allocation between keys
and values. KVQuant (Hooper et al., 2024), for
2

<!-- page 3 -->

example, focuses on vector-wise outlier handling
rather than analyzing keys and values separately.
Only a few methods have attempted to address this
issue, and even then, only superficial insights have
been gained.
Needs and Gaps.Despite rapid progress in KV
quantization, several needs from the LLM research
community remain unmet.First, there is a need for
principled strategiesto guide bit allocation. Cur-
rent frameworks either treat the key-value split as a
hyperparameter tuned through grid search (Li et al.,
2025b) or rely on heuristics derived from cache
statistics (e.g., activation ranges or distributions
collected at inference time) (Zhang et al., 2023; Liu
et al., 2025; Yang et al., 2024). These approaches
are costly, model- or data-specific, and provide lit-
tle theoretical insight into the inherent differences
between keys and values.Second, there is a need to
understand and exploit key-value asymmetry. Ex-
isting works such as KVTuner (Li et al., 2025b),
SKVQ (Duanmu et al., 2024), and QAQ (Dong
et al., 2024) briefly observe that allocating more
bits to keys can preserve accuracy, but none explain
whyor propose a generalizable strategy. KVTuner
reports differences in attention vs. perplexity errors
across bit pairs without analysis; SKVQ finds asym-
metric allocations (e.g., 2-bit keys, 1.5-bit values)
through hyperparameter search rather than model
structure; and QAQ focuses on different data types
for keys and values rather than bit-width asymme-
try.Third, there is a need forlightweight, modular
methods that integrate seamlessly with existing KV
quantization frameworks. The studies mentioned
above often involve complex, runtime-dependent
procedures that are difficult to generalize or com-
bine, limiting their drop-in applicability and com-
posability with other techniques.
Our Perspective.We address these needs by de-
riving bit allocation strategiesdirectly from model
weights, analyzing spectral and Frobenius norms
of key and value matrices. Our approach is
lightweight, incurring only a one-off analytical
cost per model without any inference-time intro-
spection;generalizable, since weight statistics are
invariant across inputs and tasks; andprincipled,
grounding allocation in linear algebraic properties
rather than heuristic search. Moreover, our mix-
precision quantization oracle isorthogonalto exist-
ing KV quantization techniques, paving a founda-
tion that others can build upon. For example, pair-
ing our mixed-precision allocation with rotation-
based outlier redistribution techniques yields com-
plementary improvements in both accuracy and
memory savings. In this way, we elevate bit alloca-
tion from an ad hoc design choice to a geometry-
informed building block for future KV quantization
frameworks.
3 Norm Dynamics of KV Weights
We now establish a theoretical foundation for
mixed-precision KV-cache quantization by analyz-
ing the intrinsic geometry of thekeyandvalue
projection weights. Our analysis proceeds in two
steps. First, we prove that key weights systemati-
cally exhibit larger spectral and Frobenius norms
than value weights (Key-Value Norm Disparity),
a property that is preserved in the resulting key
and value caches. Second, we demonstrate that
this norm gap directly impacts quantization error,
providing a formal guarantee that assigning higher
bit precision to keys yields strictly lower distortion
and higher inference accuracy than symmetric or in-
verted allocations (Key-Prioritized Quantization).
3.1Key-V alue Norm Disparity: Key Weights
Dominate in Norm
The key weight matrix WK maps hidden states
into the key cache, whereas the value weight ma-
trix WV determines the representations retrieved
during attention. Because quantization error scales
approximately with the dynamic range of the signal,
the relative magnitudes of ∥WK∥ and ∥WV ∥, e.g.,
their Frobenius or spectral norms, indicate which
cache is more sensitive to quantization.
2 4 6 8 10 12 14
10
15
20
25
30
35
40Frobenius Norm
Llama3.2-1B-it
Key Weight
Value Weight
0 3 6 9 12 15 18 21 24 27
20
30
40
50
60
Llama3.2-3B-it
Key Weight
Value Weight
0 4 8 12 16 20 24 28 32
Layer Index
20
30
40
50Frobenius Norm
Llama3.1-8B-it
Key Weight
Value Weight
0 10 20 30 40 50 60 70 80
Layer Index
20
40
60
80
100
120
Llama3.3-70B-it
Key Weight
Value Weight
Figure 2:Frobenius norms of key and value weight
matrices across the Llama 3 family. ∥W K∥F consis-
tently exceeds ∥W V ∥F across nearly all layers, with the
exception occurring in early layers of the 70B variant.
Empirical measurements across four Llama-3
3

<!-- page 4 -->

herds (Figure 2) show that the Frobenius norm of
WK consistently exceeds that of WV in nearly
every layer; the same ordering holds for spectral
norms. This persistent gap motivates the following
theorem.
Theorem 1 (Key-Value Norm Disparity)
Let WK and WV denote the key and value pro-
jection matrices in a Transformer. Then
E

∥WK∥F

>E

∥WV ∥F

.
The detailed proof is provided in Appendix A.2.
The core idea here is to examine how the Frobenius
norms of the key and value weight matrices evolve
during training. The analysis begins with Xavier
initialization (Glorot and Bengio, 2010), where all
projection matrices have identical expected norms,
and tracks their evolution under stochastic gradient
descent (SGD). Intuitively, WK play a dual role:
they shape the attention map and determine what
representations are stored in the cache. Each input
is multiplied by WK to produce keys that interact
multiplicatively with the queries derived from WQ
(i.e., the query projection). As training proceeds,
WQ typically grows to sharpen attention, amplify-
ing the gradient signals backpropagated into WK.
By contrast, WV only influences post-attention rep-
resentations, so its gradients lack this multiplicative
amplification. This architectural asymmetry causes
WK to receive systematically larger updates, lead-
ing to persistently larger norms over time.
This phenomenon is ubiquitous. Appendix C.2
offers analogous results for Mistral family (Mis-
tral AI, 2024), demonstrating the generality of the
pattern. We next show that this norm disparity has
direct consequences for quantization.
3.2 Key-Prioritized Quantization: Key-Favored
Allocation Minimizes Quantization Error
Theorem 1 implies that, on average, WK and their
resulting activations,K, have larger magnitude than
their value counterparts. Since quantization error
under uniform scalar quantization scales with the
signal energy, assigning equal bit precision to both
is sub-optimal.
Consider an additive-residual Transformer block
(layer normalization omitted for clarity (Elhage
et al., 2021)):
hl+1 =h l +W lhl.
Quantizing Wl to ˜Wl =W l + ∆l introduces an
error bounded by
∥hl+1 − ˜hl+1∥2 =∥∆ lhl∥2 ≤ ∥∆ l∥2∥hl∥2.
After L layers, the worst-case deviation accumu-
lates multiplicatively:
∥hL − ˜hL∥2 ≤
 LY
i=1
∥Wi∥2

max
1≤i≤L
∥∆i∥2.
Because the expected norms satisfy
E

∥W K∥2
F

>E

∥W V ∥2
F

,
any quantization noise injected along the key path
is amplified more strongly through the network.
Let X∈R seq×dmodel be the hidden-state matrix
at a given layer. The same input generates both
caches via
K=XW K, V=XW V .
Multiplying the previous inequality by X and ap-
plying sub-multiplicativity of the Frobenius norm
yields
E

∥K∥2
F

>E

∥V∥ 2
F

,
i.e., the norm gap persists in the caches.
Quantization Error and Norm Magnitude.For
a matrix M∈R m×n, the squared Frobenius
norm equals the total signal energy and the sum of
squared singular values. When quantized with b-bit
uniform scalar quantization, the expected mean-
square error satisfies
E

∥M− ˜M∥ 2
F

= Θ
 
∥M∥ 2
F 2−2b
,
with constants depending only on the quantizer.
Since key and value caches have identical shape,
minimizing quantization error reduces to allocating
bits in proportion to their energy. When ∥K∥F ≫
∥V∥ F and equal bit-widths are used, key-cache
error dominates:
E

∥K− ˜K∥2
F

≫E

∥V− ˜V∥ 2
F

.
This asymmetry in quantization error directly
motivates an asymmetric bit allocation strategy,
formalized as the following theorem:
Theorem 2 (Key-Prioritized Quantization)
Let (bK, bV ) denote the bit allocations for key
and value caches under a uniform scalar quan-
tizer. For any pair withbK > b V , the expected
inference accuracy is strictly higher than for
the swapped allocation (bV , bK), provided that
E[∥K∥ 2
F ]>E[∥V∥ 2
F ].
4

<!-- page 5 -->

0
10
20Magnitude
Keys, Layer 0
0
50
100
Keys, Layer 11
0
50
100
Keys, Layer 22
0
50
100
Keys, Layer 33
0
100
200
Keys, Layer 45
0
50
100
150
Keys, Layer 56
0
100
200
Keys, Layer 67
0
50
100
150
Keys, Layer 79
0 50 100
Singular Value Index
0.0
0.1
0.2
0.3Magnitude
Values, Layer 0
0 50 100
Singular Value Index
0
2
4
Values, Layer 11
0 50 100
Singular Value Index
0.0
2.5
5.0
7.5
Values, Layer 22
0 50 100
Singular Value Index
0.0
2.5
5.0
7.5
Values, Layer 33
0 50 100
Singular Value Index
2
4
6
Values, Layer 45
0 50 100
Singular Value Index
2.5
5.0
7.5
Values, Layer 56
0 50 100
Singular Value Index
5
10
Values, Layer 67
0 50 100
Singular Value Index
0
5
10
Values, Layer 79
Figure 3:Singular value spectra of key and value activations in Llama 3.3-70B on C4 benchmark dataset.
The x-axis shows singular value indices, ordered from the 5th largest onward for cleaner illustration, and the y-axis
shows their magnitudes. Shaded regions mark the minimum-maximum range across attention heads within each
layer, while dashed lines indicate the mean at each index. Beyond the top singular value (i.e., the spectral norm),
key activations consistently exhibit larger singular values than value activations across the spectrum, highlighting
their greater representational capacity. Full spectra are provided in Figure 6 of Appendix C.3.
Figure 3 provides empirical evidence for this
effect. For Llama 3.3-70B on C4, the singular
value spectra of the key caches consistently exceed
those of the value caches beyond the top singular
mode, indicating higher representational signifi-
cance throughout the spectrum. Appendix C.3 ex-
tends this analysis to the full singular value range,
where Figure 7 shows layer-wise Frobenius and
spectral norms, all consistently revealing larger
magnitudes for keys than for values.
The theoretical underpinnings of this phe-
nomenon are established in Theorems 3 of Ap-
pendix B.2 and 4 of Appendix B.3, which derive
a norm-dependent upper bound on the quantiza-
tion error of an arbitrary matrix M. Appendix B.4
then applies this bound to the key and value caches,
showing that the larger norms of the key caches
translate into proportionally higher quantization
errors under equal bit allocations.
4 Results
4.1 Experimental Setup
Three evaluations are conducted:(i)a quantization-
error analysis, which measures reconstruction error
across different bit-widths;(ii)a downstream task
accuracy evaluation, which assesses how mixed-
precision KV cache quantization affects model ac-
curacy on practical benchmarks; and(iii)an inte-
gration case study with rotation-based outlier dis-
tribution methods, which investigates the effect of
combining mixed precisions with rotation strate-
gies. Full details on compute resources and soft-
ware configurations are supplied in Appendix D.
Quantization-Error Evaluation.PyTorch’s
weight-packing quantization is applied with-
out residual buffers or activation grouping to
isolate quantization effects. Ten random se-
quences are sampled from C4 (Dodge et al.,
2021), MMLU (Hendrycks et al., 2021), and
GSM8K (Cobbe et al., 2021), padded to the
longest length, and generated autoregressively for
up to 1,000 tokens. Both per-layer and per-head
reconstruction errors are computed, along with
global averages across all heads, layers, and to-
kens, to offer a comprehensive view of bit-width
sensitivity. This experiment spans seven models,
including Llama-3.2-1B, Llama-3.1-8B (Grattafiori
et al., 2024), Phi-4-14B (Abdin et al., 2024b),
Mistral-0.3-7B (Mistral AI, 2024), Qwen-2.5-14B,
Llama-3.3-70B, DeepSeek-R1-Qwen-14B, Phi-3-
Medium-128K (Abdin et al., 2024a), and Llama-
3.1-Nemotron-70B (Nvidia et al., 2024).
Downstream Task Accuracy.Two repre-
sentative quantization backends are employed.
Optimum Quantoapplies token-wise (per-row)
quantization with mixed 2/4-bit precision on
Llama-3.2-1B, Llama-3.1-8B, Phi-4-14B, and
DeepSeek-R1-Qwen-14B.HQQ(Badri and Shaji,
2023) applies channel-wise (per-column) quan-
tization with bit-widths {1,2,4,6,8} on Llama-
3.1-8B, Llama-3.2-1B, Llama-3.2-3B, and Qwen3-
0.6B/1.7B/4B/8B (Qwen Team, 2025). A 64-
token residual buffer (which stores the most re-
cent tokens in full precision before being periodi-
cally flushed) and 128-element activation grouping
(which quantizes activations in fixed-size blocks
to reduce overhead) are adopted to mirror prac-
tical decoding configurations used in KIVI (Liu
et al., 2024), Flash-Decoding (Dao et al., 2023),
and Marlin (Frantar et al., 2025), ensuring that
the evaluation reflects realistic deployment rather
5

<!-- page 6 -->

Table 1:Quantization error (lower is better) of K and V caches under matched bit-widths.Each cell reports
the mean ± standard deviation of the reconstruction MSE between thedequantizedcache and its BF16 reference,
averaged over layers, heads, and tokens (10 sequences per dataset up to 1,000 tokens each). Both caches are
quantized to thesameprecision, i.e., K2V2, K3V3, and K4V4, to isolate their intrinsic sensitivity at equal bit
budgets. Across model families (Llama, Phi, Mistral, Qwen, DeepSeek) and datasets (C4, MMLU, GSM8K), K
consistently exhibits larger reconstruction error than V at the same bit-width, and the error decreases monotonically
with increasing precision, indicating that keys are the dominant source of quantization distortion.
Dataset Model 2-bit 3-bit 4-bit
K2 V2 K3 V3 K4 V4
MMLU
Llama3.2-1B 4.851± 1.037 0.127± 0.1011.037± 0.265 0.021± 0.0150.227± 0.059 0.005± 0.003
Llama3.1-8B-it 6.003± 1.782 0.187± 0.1271.082± 0.244 0.028± 0.0190.235± 0.055 0.006± 0.004
Llama3.3-70B-it 4.883± 1.106 0.112± 0.0930.942± 0.198 0.016± 0.0120.206± 0.043 0.003± 0.003
Phi4 5.929± 1.545 0.657± 0.4721.306± 0.231 0.103± 0.0700.286± 0.050 0.022± 0.015
Mistral0.3-7B 4.718± 1.340 0.398± 0.4050.941± 0.240 0.059± 0.0590.206± 0.053 0.013± 0.013
Qwen2.5-14B 5.184± 2.241 1.270± 1.5471.005± 0.288 0.182± 0.2210.223± 0.067 0.040± 0.052
R1Q-14B 5.126± 2.375 1.406± 1.6090.900± 0.269 0.198± 0.2260.199± 0.062 0.044± 0.052
C4
Llama3.2-1B 4.885± 1.056 0.207± 0.1661.074± 0.289 0.030± 0.0240.233± 0.062 0.006± 0.005
Llama3.1-8B-it 6.262± 1.789 0.254± 0.1851.128± 0.249 0.036± 0.0260.247± 0.056 0.008± 0.005
Llama3.3-70B-it 4.391± 1.027 0.121± 0.0970.847± 0.175 0.017± 0.0130.186± 0.038 0.004± 0.003
Phi4 5.715± 1.442 0.850± 0.6841.316± 0.245 0.124± 0.0930.291± 0.056 0.027± 0.020
Mistral0.3-7B 5.027± 1.332 0.543± 0.4931.014± 0.269 0.079± 0.0680.223± 0.060 0.017± 0.015
Qwen2.5-14B 4.382± 2.170 1.544± 1.8720.846± 0.250 0.220± 0.2650.187± 0.060 0.048± 0.060
DeepSeekR1Q-14B 4.832± 2.354 1.651± 1.9140.927± 0.283 0.232± 0.2670.201± 0.061 0.051± 0.060
GSM8K
Llama3.2-1B 5.703± 1.557 0.179± 0.1361.213± 0.352 0.026± 0.0200.266± 0.078 0.005± 0.004
Llama3.1-8B-it 6.445± 1.837 0.213± 0.1611.184± 0.268 0.030± 0.0220.257± 0.060 0.007± 0.005
Llama3.3-70B-it 4.967± 1.127 0.113± 0.0910.978± 0.203 0.016± 0.0120.214± 0.044 0.004± 0.003
Phi4 6.610± 1.624 0.785± 0.5981.498± 0.293 0.116± 0.0820.330± 0.064 0.025± 0.017
Mistral0.3-7B 5.308± 1.367 0.461± 0.4341.065± 0.288 0.067± 0.0610.232± 0.061 0.015± 0.013
Qwen2.5-14B 4.829± 2.179 1.736± 2.6590.979± 0.264 0.241± 0.3720.214± 0.061 0.051± 0.077
DeepSeekR1Q-14B 4.477± 2.176 1.424± 1.7520.830± 0.256 0.200± 0.2420.181± 0.058 0.044± 0.056
than idealized settings. Accuracy is measured on
three generative benchmarks: GSM8K (Cobbe
et al., 2021), COQA (Reddy et al., 2019), and EQ-
BENCH(Paech, 2024), which collectively probe
mathematical reasoning, conversational QA, and
structured long-form generation.
Integration with Rotation-Based Methods.
QuaRot (Ashkboos et al., 2024) is selected for
this case study. It applies structured randomized
Hadamard rotations to activations before quanti-
zation, effectively dispersing outliers and improv-
ing the uniformity of the quantization distribu-
tion. A three-dimensional design space is ex-
plored, spanningbit-width allocation,group size
configuration, androtation strategies. Specifically,
mixed-precision settings K2V2, K2V4, K4V2, and
K4V4; key and value group sizes in {32,64,128} ;
and four rotation strategies (no rotation, key-only,
value-only, and both). The evaluation encompasses
generative tasks including COQA, GSM8K, EQ-
BENCH, and LONGBENCH, utilizing the same
seven models as in quantization-error evaluation.
4.2 Mixed-Precision Quantization Error
Table 1 reports reconstruction errors at 2-, 3-, and 4-
bit precision for seven representative models span-
ning multiple model families (Llama, Phi, Mistral,
Qwen, DeepSeek) and datasets; full results appear
in Appendix C.4. Figure 8 shows the complete er-
ror curves for Llama-3.3-70B on C4, and Figure 9
provides a per-layer breakdown for Llama-3.1-8B.
Across all models, datasets, and bit-widths, key
caches consistently incur larger reconstruction er-
rors than value caches, and this gap remains stable
across precision levels. These findings empirically
support the theoretical prediction that key represen-
tations have higher energy and are therefore more
sensitive to quantization.
4.3 Mixed-Precision Downstream Accuracy
Table 2 summarizes the Optimum Quanto results
on GSM8K under both 1-shot and 8-shot Chain-of-
Thought (CoT) prompting. Although CoT prompt-
ing can sometimes reduce reasoning accuracy, it is
included here to assess the impact of longer con-
texts on quantized decoding. Across four repre-
sentative models, including Llama-3.2-1B, Llama-
3.1-8B, Phi-4-14B, and DeepSeek-R1-Qwen-14B,
a consistent pattern emerges: allocating higher pre-
cision to thekeycache ( K4V2) preserves accuracy
substantially better than the inverse ( K2V4). On
average, K4V2 recovers approximately94%of the
6

<!-- page 7 -->

full-precision baseline, whereas value-favored allo-
cations incur losses of up to 30 percentage points
(pp). The performance gap widens with model
scale; for instance, under 1-shot GSM8K, the
K4V2 configuration outperforms K2V4 by 30 pp
on Llama-3.2-1B and by 16 pp on Phi-4-14B. No-
tably, K4V2 nearly matches the symmetric K4V4
baseline despite halving the value bit budget, indi-
cating that downstream performance is primarily
constrained by key precision.
Table 2:GSM8K – Downstream accuracy with Opti-
mum Quanto (token-wise) mixed-precision KV quan-
tization.Token-wise quantization is applied with sup-
ported precisions i, j∈ {2,4} , where KiVj denotes
i-bit keys and j-bit values. Results are shown for both 1-
shot and 8-shot settings. Across models, the key-favored
K4V2 consistently outperforms the value-favoredK2V4
and approaches the K4V4 baseline, demonstrating the
benefits of prioritizing key precision.
Model ShotsK 2V2 K2V4 K4V2 K4V4
Llama
3.2-1B-it
1-shot 0.033 0.035 0.338 0.357
8-shot 0.031 0.031 0.289 0.369
Llama
3.1-8B-it
1-shot 0.511 0.547 0.752 0.754
8-shot 0.408 0.441 0.770 0.782
Phi
4-14B
1-shot 0.759 0.783 0.913 0.923
8-shot 0.771 0.815 0.927 0.931
DeepSeek
R1Q-14B
1-shot 0.772 0.775 0.865 0.867
8-shot 0.763 0.792 0.876 0.875
The HQQ results, shown in Table 3, extend
these observations to a broader range of bit-widths,
model scales, and tasks. The analysis system-
atically compares KiVx against KxVi for i∈
{1,2,4,6,8} , where x denotes the mean accuracy
over all bit-widths of the other cache. This di-
rectly addresses the question: “If i bits are avail-
able, should they be allocated to keys or val-
ues?” Across all models (0.6B-32B) and datasets
(GSM8K, CoQA, EQ-Parseable), the answer is
consistently “keys”. At ultra-low precision (1-2
bits), the advantage is especially pronounced; for
instance, on GSM8K, allocating a single extra bit to
keys with 1-bit quantization yields gains of +48 pp
for Qwen3-8B. Similar trends hold on EQ-Bench
and CoQA: for Qwen3-0.6B on EQ-Parseable, pri-
oritizing keys delivers up to +62 pp, and key-first
allocations never underperform value-first ones in
any configuration. Even at moderate precisions
(4-6 bits), key-centric allocation continues to of-
fer 7-12 pp improvements for 8-14B models, in-
dicating that the advantage persists well beyond
extreme compression.On average, K4V2 retains
98.3% accuracy of K4V4 (CoQA: 99.2%, EQ-
Bench: 99.35%, GSM8K: 97.7%; worst: 88.3%,
best: 103.5%).
K2V2 K2V4 K4V2 K4V4
NoRotationKey-onlyValue-onlyBoth
Rotation Strategy
42.92 45.30 54.94 55.73
46.95 48.11 62.75 63.54
43.46 45.42 58.98 59.42
46.84 48.20 63.05 63.48
CoQA
K2V2 K2V4 K4V2 K4V4
NoRotationKey-onlyValue-onlyBoth
31.50 33.83 79.78 79.62
35.59 39.43 93.98 93.90
29.91 33.92 76.78 79.45
35.00 38.68 91.73 93.82
Eq Bench
K2V2 K2V4 K4V2 K4V4
Quantization Config
NoRotationKey-onlyValue-onlyBoth
Rotation Strategy
4.91 7.06 57.79 58.92
19.39 23.05 63.65 65.13
5.44 7.05 58.23 59.11
19.96 22.70 64.64 65.32
GSM8K
K2V2 K2V4 K4V2 K4V4
Quantization Config
NoRotationKey-onlyValue-onlyBoth
12.66 13.02 20.49 20.63
12.86 13.64 21.54 21.98
12.65 12.97 20.15 20.41
12.91 13.51 22.23 21.89
LongBench
45.0
47.5
50.0
52.5
55.0
57.5
60.0
62.5
30
40
50
60
70
80
90
10
20
30
40
50
60
14
16
18
20
22
Figure 4:Integration of rotation and mixed-precision
quantization.Downstream accuracy is shown for
four quantization configurations (K2V2, K2V4, K4V2,
K4V4) combined with four rotation strategies (none,
key-only, value-only, both), using a fixed group size
of 64 for both keys and values. Results are reported
on COQA, GSM8K, EQ-BENCH, and LONGBENCH,
enabling a controlled comparison of precision-rotation
interactions.
More detailed downstream accuracy results are
provided in Appendix C.5. Overall, downstream
accuracy is far more sensitive to key precision than
to value precision. Across both token-wise and
channel-wise quantization schemes, model scales,
and task types, assigning the higher bit-width to
K consistently yields near-baseline accuracy while
substantially reducing KV memory. This estab-
lishes a simple, backend-agnostic design principle
for mixed-precision KV cache quantization:More
for keys, less for values.
4.4 Integrating with Rotation-Based Methods
Figure 4 visualizes how rotation interacts with key-
value bit allocations. Across all tasks, applying
rotation tokeysconsistently yields larger gains than
applying it to values. Notably, applying key-only
rotation to the K4V2 configuration achieves accu-
racy that closely matches the full K4V4 baseline,
indicating that the primary benefits of rotation arise
from mitigating key outliers. These trends are con-
sistent across tasks and model scales, reinforcing
7

<!-- page 8 -->

Table 3:Downstream accuracy with HQQ (channel-wise) mixed-precision KV quantization across GSM8K,
EQ-Parseable, and CoQA.Each cell compares KiVx and KxVi for i∈ {1,2,4,6,8} , where x represents the
mean accuracy averaged over the other cache’s bit-widths B={1,2,4,6,8} . This corresponds to contrasting
“allocate i bits to keys (values averaged over B)” versus “allocate i bits to values (keys averaged over B).” Across
model scales from 0.6B to 32B and multiple tasks, key-favored allocations consistently yield higher accuracy, with
more pronounced gains observed at lower precision and persistent benefits at higher precision, demonstrating that
the key-first advantage generalizes beyond a single model, task, and quantization backend.
GSM8KK 1Vx vs.K xV1 K2Vx vs.K xV2 K4Vx vs.K xV4 K6Vx vs.K xV6 K8Vx vs.K xV8
Qwen3-0.6B0.00
+3%
<0.030.00
+16%
<0.160.04
+13%
<0.17 0.34
+16%
>0.180.34
+16%
>0.18
Llama-3.2-1B0.00
+5%
<0.050.01
+18%
<0.19 0.27
+7%
>0.200.28
+8%
>0.200.28
+7%
>0.21
Llama-3.2-3B0.00
+19%
<0.190.27
+17%
<0.44 0.57
+11%
>0.460.58
+12%
>0.460.59
+13%
>0.46
Qwen3-4B0.00
+41%
<0.410.01
+49%
<0.50 0.80
+29%
>0.510.82
+31%
>0.510.82
+31%
>0.51
Llama-3.1-8B0.00
+31%
<0.310.35
+17%
<0.52 0.70
+16%
>0.540.70
+16%
>0.540.70
+16%
>0.54
Qwen3-8B0.00
+48%
<0.480.08
+46%
<0.54 0.86
+31%
>0.550.87
+32%
>0.550.86
+31%
>0.55
Qwen3-32B0.00
+42%
<0.420.25
+23%
<0.48 0.73
+23%
>0.500.71
+21%
>0.500.72
+21%
>0.51
EQ-Parseable
Qwen3-0.6B0.00
+10%
<0.100.00
+53%
<0.530.51
+2%
<0.53 0.84
+32%
>0.530.85
+33%
>0.52
Llama-3.2-1B0.00
+34%
<0.340.26
+37%
<0.62 0.87
+22%
>0.650.90
+24%
>0.660.90
+25%
>0.66
Llama-3.2-3B0.00
+36%
<0.350.68
+5%
<0.73 0.90
+14%
>0.760.89
+13%
>0.760.90
+14%
>0.76
Qwen3-4B0.00
+54%
<0.480.29
+33%
<0.58 0.84
+28%
>0.600.86
+31%
>0.590.85
+29%
>0.60
Llama-3.1-8B0.00
+61%
<0.61 0.78
+2%
>0.760.96
+19%
>0.770.97
+20%
>0.770.97
+21%
>0.77
Qwen3-8B0.00
+62%
<0.580.54
+17%
<0.70 0.95
+25%
>0.710.96
+27%
>0.710.96
+26%
>0.71
Qwen3-32B0.00
+60%
<0.470.56
+7%
<0.62 0.80
+22%
>0.620.80
+23%
>0.620.80
+22%
>0.63
CoQA
Qwen3-0.6B0.20
+33%
<0.440.32
+29%
<0.52 0.65
+16%
>0.530.69
+23%
>0.530.69
+23%
>0.53
Llama-3.2-1B0.22
+29%
<0.420.48
+13%
<0.57 0.67
+13%
>0.580.68
+14%
>0.580.68
+14%
>0.58
Llama-3.2-3B0.21
+50%
<0.60 0.73
+6%
>0.680.79
+14%
>0.680.79
+14%
>0.680.79
+15%
>0.67
Qwen3-4B0.34
+34%
<0.620.68
+3%
<0.70 0.81
+12%
>0.710.81
+12%
>0.710.81
+13%
>0.70
Llama-3.1-8B0.21
+50%
<0.60 0.72
+7%
>0.660.78
+14%
>0.670.78
+15%
>0.670.78
+14%
>0.67
Qwen3-8B0.38
+37%
<0.68 0.73
+1%
>0.720.82
+12%
>0.720.82
+12%
>0.720.82
+12%
>0.72
Qwen3-32B0.32
+44%
<0.68 0.79
+9%
>0.720.82
+12%
>0.730.82
+12%
>0.730.82
+12%
>0.73
that rotation is most effective when applied in con-
junction with key-favored bit allocation.
Appendix C.6 presents detailed downstream re-
sults illustrating the synergistic effects of inte-
grating rotation with mixed-precision quantization
across tasks and models. It also examines the im-
pact of key and value group sizes, showing that
smaller sizes are beneficial for K due to higher
information density, whereas larger group sizes suf-
fice forVgiven lower sensitivity to quantization.
5 Conclusion
As large language models increasingly devote most
of their inference cost to KV-cache storage and
access, effective cache compression has become
essential for practical deployment. This work pro-
vides a theoretically grounded justification and so-
lution to the bit-allocation problem by bridging
model geometry and quantization design. We theo-
retically establish that key projections consistently
carry higher information density than value projec-
tions. Building on this, we show that allocating
higher precision to keys and lower precision to val-
ues minimizes quantization error. Extensive experi-
ments across nine model families, six benchmarks,
and two hardware-aligned backends validate this
principle: a K4V2 precision split reliably recov-
ers up to 98.3% of K4V4 accuracy while signifi-
cantly reducing memory consumption. Moreover,
we demonstrate that our geometry-driven strategy
isorthogonalto rotation-based outlier redistribu-
tion methods, enabling seamless integration and
further accuracy gains. These findings elevate bit
allocation from empirical tuning to a theoretically
grounded geometry-driven design principle, pro-
viding clear guidance for efficient deployment and
future hardware-algorithm co-design.
8

<!-- page 9 -->

Limitations
While our geometry-driven mixed-precision quan-
tization framework demonstrates both theoretical
soundness and practical effectiveness through rig-
orous analysis and extensive experiments, several
limitations remain. First, all evaluations are con-
ducted with a maximum context length of 2,000
tokens, which reflects common inference-time con-
figurations but does not fully capture the behavior
of models operating at much larger context win-
dows. Extending the approach to longer contexts
may expose additional challenges, including in-
creased quantization sensitivity and higher memory
management overheads. Addressing these factors
is an important direction for future work.
Ethical Considerations
This research aims to reduce the memory and com-
putational costs of large language model infer-
ence by utilizing efficient KV-cache quantization.
Such improvements have the potential to lower en-
ergy consumption and broaden access to language
models in resource-constrained settings, promot-
ing more sustainable and inclusive deployment.
However, any compression technique entails accu-
racy trade-offs, which must be carefully monitored
to avoid disproportionate impacts in high-stakes
domains such as healthcare, law, or finance. Re-
sponsible deployment requires thorough evaluation
of model behavior under mixed-precision settings,
particularly for safety-critical applications.
Acknowledgment
This research was supported in part by NSF awards
2117439, 2112606, and 2320952.
References
Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat
Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck,
Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav
Chaudhary, Dong Chen, Dongdong Chen, and 110
others. 2024a. Phi-3 technical report: A highly capa-
ble language model locally on your phone.Preprint,
arXiv:2404.14219.
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien
Bubeck, Ronen Eldan, Suriya Gunasekar, Michael
Harrison, Russell J. Hewett, Mojan Javaheripi, Piero
Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi
Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen,
Eric Price, Gustavo de Rosa, Olli Saarikivi, and 8
others. 2024b. Phi-4 technical report.Preprint,
arXiv:2412.08905.
Saleh Ashkboos, Amirkeivan Mohtashami, Maximil-
ian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi,
Dan Alistarh, Torsten Hoefler, and James Hensman.
2024. Quarot: Outlier-free 4-bit inference in rotated
llms.Advances in Neural Information Processing
Systems, 37:100213–100240.
Hicham Badri and Appu Shaji. 2023. Half-quadratic
quantization of large machine learning models.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, and 12 others. 2020. Language
models are few-shot learners. InProceedings of the
34th Conference on Neural Information Processing
Systems (NeurIPS 2020). Vancouver, Canada.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.Preprint, arXiv:2110.14168.
Tri Dao. 2024. FlashAttention-2: Faster attention with
better parallelism and work partitioning. InInter-
national Conference on Learning Representations
(ICLR).
Tri Dao, Daniel Y . Fu, Stefano Ermon, Atri Rudra,
and Christopher Ré. 2022. FlashAttention: Fast and
memory-efficient exact attention with IO-awareness.
InAdvances in Neural Information Processing Sys-
tems (NeurIPS).
Tri Dao, Daniel Haziza, Francisco Massa, and Grig-
ory Sizov. 2023. Flash-decoding for long-context
inference. Accessed: 2025-05-15.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, and 181 others.
2025a. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning.Preprint,
arXiv:2501.12948.
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai,
and 181 others. 2025b. Deepseek-v3 technical report.
Preprint, arXiv:2412.19437.
Tim Dettmers, Mike Lewis, Younes Belkada, and
Luke Zettlemoyer. 2022. Llm.int8(): 8-bit matrix
multiplication for transformers at scale.Preprint,
arXiv:2208.07339.
9

<!-- page 10 -->

Jesse Dodge, Maarten Sap, Ana Marasovi ´c, William
Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret
Mitchell, and Matt Gardner. 2021. Documenting
large webtext corpora: A case study on the colossal
clean crawled corpus.Preprint, arXiv:2104.08758.
Shichen Dong, Wen Cheng, Jiayu Qin, and Wei Wang.
2024. Qaq: Quality adaptive quantization for llm kv
cache.Preprint, arXiv:2403.04643.
Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei
Duan, Xingcheng Zhang, and Dahua Lin. 2024.
Skvq: Sliding-window key and value cache quan-
tization for large language models. InProceedings
of COLM 2024.
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom
Henighan, Nicholas Joseph, Ben Mann, Amanda
Askell, Yuntao Bai, Anna Chen, Tom Conerly, and 1
others. 2021. A mathematical framework for trans-
former circuits.Transformer Circuits Thread.
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and
Dan Alistarh. 2023a. Gptq: Accurate post-training
quantization for generative pre-trained transformers.
InProceedings of the International Conference on
Learning Representations (ICLR 2023).
Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten
Hoefler, and Dan Alistarh. 2025. Marlin: Mixed-
precision auto-regressive parallel inference on large
language models. InProceedings of the 30th ACM
SIGPLAN Annual Symposium on Principles and
Practice of Parallel Programming, PPoPP ’25, page
239–251, New York, NY , USA. Association for Com-
puting Machinery.
Elias Frantar, Sidak Pal Singh, and Dan Alistarh. 2023b.
Optimal brain compression: A framework for ac-
curate post-training quantization and pruning. In
Proceedings of the 36th Conference on Neural Infor-
mation Processing Systems (NeurIPS 2022).
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Bider-
man, Sid Black, Anthony DiPofi, Charles Foster,
Laurence Golding, Jeffrey Hsu, Alain Le Noac’h,
Haonan Li, Kyle McDonell, Niklas Muennighoff,
Chris Ociepa, Jason Phang, Laria Reynolds, Hailey
Schoelkopf, Aviya Skowron, Lintang Sutawika, and
5 others. 2024. The language model evaluation har-
ness.
Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao,
Michael W. Mahoney, and Kurt Keutzer. 2021. A
survey of quantization methods for efficient neural
network inference.Preprint, arXiv:2103.13630.
Xavier Glorot and Yoshua Bengio. 2010. Understand-
ing the difficulty of training deep feedforward neural
networks. InProceedings of the Thirteenth Interna-
tional Conference on Artificial Intelligence and Statis-
tics, volume 9 ofProceedings of Machine Learning
Research, pages 249–256, Chia Laguna Resort, Sar-
dinia, Italy. PMLR.
Google DeepMind. 2025. Gemini 2.0 flash. Accessed:
2025-05-15.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.Preprint, arXiv:2407.21783.
Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp
Schmid, Zachary Mueller, Sourab Mangrulkar, Marc
Sun, and Benjamin Bossan. 2022. Accelerate: Train-
ing and inference at scale made simple, efficient and
adaptable.
Song Han, Huizi Mao, and William J. Dally. 2016. Deep
compression: Compressing deep neural networks
with pruning, trained quantization and huffman cod-
ing. InProceedings of the International Conference
on Learning Representations (ICLR 2016). Oral pre-
sentation.
Mohsen Hariri, Amirhossein Samandar, Michael
Hinczewski, and Vipin Chaudhary. 2025. Don’t
pass@k: A bayesian framework for large language
model evaluation.arXiv preprint arXiv:2510.04265.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021. Measuring massive multitask language under-
standing. InProceedings of the International Confer-
ence on Learning Representations (ICLR 2021).
Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh,
Michael W. Mahoney, Yakun Sophia Shao, Kurt
Keutzer, and Amir Gholami. 2024. Kvquant: To-
wards 10 million context length llm inference with
kv cache quantization. InProceedings of the 38th
Conference on Neural Information Processing Sys-
tems (NeurIPS 2024).
Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang,
Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong,
Qing Li, and Lei Chen. 2025a. A survey on large
language model acceleration based on kv cache man-
agement.Preprint, arXiv:2412.19442.
Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-Ling
Zhen, Wulong Liu, Yiwu Yao, Sinno Jialin Pan, and
Mingxuan Yuan. 2025b. Kvtuner: Sensitivity-aware
layer-wise mixed precision kv cache quantization for
efficient and nearly lossless llm inference.Preprint,
arXiv:2502.04420.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. Snapkv:
Llm knows what you are looking for before gener-
ation. InProceedings of the 38th Conference on
Neural Information Processing Systems (NeurIPS
2024).
10

<!-- page 11 -->

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-
Ming Chen, Wei-Chen Wang, Guangxuan Xiao,
Xingyu Dang, Chuang Gan, and Song Han. 2024.
Awq: Activation-aware weight quantization for llm
compression and acceleration. InProceedings of the
7th MLSys Conference 2024, Santa Clara, CA, USA.
Tengxuan Liu, Shiyao Li, Jiayi Yang, Tianchen Zhao,
Feng Zhou, Xiaohui Song, Guohao Dai, Shengen
Yan, Huazhong Yang, and Yu Wang. 2025. Pm-kvq:
Progressive mixed-precision kv cache quantization
for long-cot llms.Preprint, arXiv:2505.18610.
Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong,
Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and
Xia Hu. 2024. Kivi: A tuning-free asymmetric 2bit
quantization for kv cache. InProceedings of the
41st International Conference on Machine Learning
(ICML 2024), PMLR 235, Vienna, Austria.
Meta AI. 2025a. The llama 4 herd: The be-
ginning of a new era of natively multimodal
ai innovation. https://ai.meta.com/blog/
llama-4-multimodal-intelligence/ . Official
announcement of Llama 4. Accessed 2025-10-06.
Meta AI. 2025b. The llama 4 herd: The beginning of
a new era of natively multimodal intelligence. Ac-
cessed: 2025-05-15.
Mistral AI. 2024. Large enough: Announcing mistral
large 2. Accessed: 2025-05-15.
Markus Nagel, Marios Fournarakis, Rana Ali Amjad,
Yelysei Bondarenko, Mart van Baalen, and Tijmen
Blankevoort. 2021. A white paper on neural network
quantization.Preprint, arXiv:2106.08295.
Nvidia, :, Bo Adler, Niket Agarwal, Ashwath Aithal,
Dong H. Anh, Pallab Bhattacharya, Annika Brun-
dyn, Jared Casper, Bryan Catanzaro, Sharon Clay,
Jonathan Cohen, Sirshak Das, Ayush Dattagupta,
Olivier Delalleau, Leon Derczynski, Yi Dong, Daniel
Egert, Ellie Evans, and 64 others. 2024. Nemotron-4
340b technical report.Preprint, arXiv:2406.11704.
OpenAI. 2024. Openai o1 system card. Accessed via
OpenAI.
OpenAI. 2025. Openai o3-mini. Accessed via OpenAI.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report.Preprint,
arXiv:2303.08774.
Samuel J. Paech. 2024. Eq-bench: An emotional
intelligence benchmark for large language models.
Preprint, arXiv:2312.06281.
Reiner Pope, Sholto Douglas, Aakanksha Chowdhery,
Jacob Devlin, James Bradbury, Anselm Levskaya,
Jonathan Heek, Kefan Xiao, Shivani Agrawal, and
Jeff Dean. 2023. Efficiently scaling transformer in-
ference. InProceedings of the 6th MLSys Conference
2023, Miami Beach, FL, USA. Copyright 2023 by
the author(s).
Qwen Team. 2025. Qwen3: Think deeper, act faster.
Accessed: 2025-05-15.
Alec Radford, Karthik Narasimhan, Tim Salimans, and
Ilya Sutskever. 2018. Improving language under-
standing by generative pre-training. Accessed via
OpenAI.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners. Ac-
cessed via OpenAI.
Siva Reddy, Danqi Chen, and Christopher D. Manning.
2019. Coqa: A conversational question answering
challenge.Preprint, arXiv:1808.07042.
Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng
Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng
Gao, Yu Qiao, and Ping Luo. 2024. Omniquant:
Omnidirectionally calibrated quantization for large
language models. InProceedings of the Inter-
national Conference on Learning Representations
(ICLR 2024).
Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan
Li, Max Ryabinin, Beidi Chen, Percy Liang, Christo-
pher Re, Ion Stoica, and Ce Zhang. 2023a. FlexGen:
High-throughput generative inference of large lan-
guage models with a single GPU. InProceedings of
the 40th International Conference on Machine Learn-
ing, volume 202 ofProceedings of Machine Learning
Research, pages 31094–31116. PMLR.
Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuo-
han Li, Max Ryabinin, Daniel Y . Fu, Zhiqiang Xie,
Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy
Liang, Christopher Ré, Ion Stoica, and Ce Zhang.
2023b. Flexgen: High-throughput generative infer-
ence of large language models with a single gpu. In
Proceedings of the 40th International Conference
on Machine Learning (ICML 2023), PMLR 202,
Honolulu, Hawaii, USA. Copyright 2023 by the
author(s).
Ilya Sutskever, Oriol Vinyals, and Quoc V . Le. 2014.
Sequence to sequence learning with neural networks.
InProceedings of the 27th Conference on Neural
Information Processing Systems (NeurIPS 2014).
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. InProceedings of the 31st Conference
on Neural Information Processing Systems (NeurIPS
2017), Long Beach, CA, USA.
11

<!-- page 12 -->

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and
Denny Zhou. 2022. Chain-of-thought prompting elic-
its reasoning in large language models. InProceed-
ings of the 36th Conference on Neural Information
Processing Systems (NeurIPS 2022).
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, and 3 others. 2020. Trans-
formers: State-of-the-art natural language processing.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations, pages 38–45, Online. Association
for Computational Linguistics.
Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu,
Julien Demouth, and Song Han. 2023. Smoothquant:
Accurate and efficient post-training quantization for
large language models. InProceedings of the 40th In-
ternational Conference on Machine Learning (ICML
2023), PMLR 202, Honolulu, Hawaii, USA.
June Yong Yang, Byeongwook Kim, Jeongin Bae,
Beomseok Kwon, Gunho Park, Eunho Yang, Se Jung
Kwon, and Dongsoo Lee. 2024. No token left be-
hind: Reliable kv cache compression via importance-
aware mixed precision quantization.Preprint,
arXiv:2402.18096.
Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang,
Xiaoxia Wu, Conglong Li, and Yuxiong He. 2022.
Zeroquant: Efficient and affordable post-training
quantization for large-scale transformers. InProceed-
ings of the 36th Conference on Neural Information
Processing Systems (NeurIPS 2022).
Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan
Zhou, Jianlong Wu, and Liqiang Nie. 2024.
Wkvquant: Quantizing weight and key/value cache
for large language models gains more.Preprint,
arXiv:2402.12065.
Hongxuan Zhang, Yao Zhao, Jiaqi Zheng, Chenyi
Zhuang, Jinjie Gu, and Guihai Chen. 2024.
Csr:achieving 1 bit key-value cache via sparse repre-
sentation.Preprint, arXiv:2412.11741.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Re, Clark Barrett, Zhangyang
Wang, and Beidi Chen. 2023. H2o: Heavy-hitter ora-
cle for efficient generative inference of large language
models. InThirty-seventh Conference on Neural In-
formation Processing Systems.
12

<!-- page 13 -->

Appendix
A Dynamics of the Norms of Key and
Value Weight Matrices
A.1 Preliminaries and Notation
cSequence length
dm Embedding dimension
dv =d k =d m/nh Single head dimension
nh Number of attention heads
XInput matrix(c×d m)
X O Output matrix(c×d m)
W V Value weights(d m ×d m)
W K Key weights(d m ×d m)
W Q Query weights(d m ×d m)
K=XW K (c×d m)
Q=XW Q (c×d m)
V=XW V (c×d m)
S=QK ⊤/√dm (c×c)
A= softmax(S) (c×c)
Lloss function.
A.2 Training Dynamics of Frobenius Norms
We compare the long-time behavior of the Frobe-
nius norms of the key ( W K) and value ( W V )
weight matrices of a single-head self-attention layer
trained with stochastic gradient descent (SGD). We
show that, under standard isotropic assumptions,
∥W K∥F grows faster than∥W V ∥F .
Update rule.At stepi, the weights follow
W m
i+1 =W m
i −η ∂L
∂W m
i
, m∈ {K, V},
with learning rate η. Squaring the Frobenius norm
of both sides gives
∥W m
i+1∥2
F =∥W m
i ∥2
F +η 2



 ∂L
∂W m
i



2
F
−2η
D
W m
i , ∂L
∂W m
i
E
F
.
where
∥A∥F ≡
q
tr(A⊤A),
⟨A, B⟩F ≡tr(A ⊤B).
Expectation over mini-batches.Taking an expec-
tation over mini-batches yields
∆E

∥W m
i ∥2
F

≡E

∥W m
i+1∥2
F

−E

∥W m
i ∥2
F

=η 2 E
h


 ∂L
∂W m
i



2
F
i
−2ηE
h
W m
i , ∂L
∂W m
i

F
i
.
In high-dimensional weight space, the gradient
is almost orthogonal to the current weights, making
the second term negligible. Hence,
∆E

∥W m
i ∥2
F

≈η 2 E
h


 ∂L
∂W m
i



2
F
i
.
Throughout, we assume Xavier initialization:
each entry of W K and W V is drawn i.i.d. from
a zero-mean distribution whose variance preserves
the input scale (Glorot and Bengio, 2010).
The attention outputs are
X O =A V W O,
where A= softmax
 
QK⊤/√dm

and V=
XW V . Differentials give
dL(X O) = tr

(∂L/∂X O)⊤ AX dW V W O
= tr

(AX) ⊤ ∂L
∂X O (W O)⊤ dW V
.
so that
∂L
∂W V = (AX) ⊤ ∂L
∂X O (W O)⊤.
Similarly, writingS=QK ⊤/√dm,
dL(S) = tr

(∂L/∂S)⊤ dS

= 1√dm
tr

X ⊤ (∂L/∂S)⊤ Q dW K
.
yielding
∂L
∂W K = 1√dm
X ⊤ (∂L/∂S)⊤ Q.
Squaring the Frobenius norm of(A.2) and taking
an expectation under the isotropic-input assump-
tion,
E

∥∂L/∂W V ∥2
F

=c σ 2
x σ2
o,
where E[X ⊤X] =cσ 2
xI and
E
h
A⊤ ∂L
∂X O (W O)⊤

A⊤ ∂L
∂X O (W O)⊤
⊤i
=
σ2
oI.
13

<!-- page 14 -->

For the key weights,
E

∥∂L/∂W K∥2
F

= σ2
x σ2
s
dm
∥Q∥2
F ,
withσ 2
s the entry-wise variance of∂L/∂S.
Substituting these expectations into (A.2) shows
that
E

∥W K∥2
F

grows as η2 σ2
xσ2
s
dm ∥Q∥2
F , whereas
E

∥W V ∥2
F

grows asη 2 cσ2
xσ2
o.
Because ∥Q∥2
F itself increases during training,
∥W K∥F eventually dominates:
E

∥W K∥F

>E

∥W V ∥F

.
As shown in Equation A.2, the expectation of
the Frobenius norm of W k is greater than that of
W v.2
B Quantization Error Bounds
We establish upper bounds on the quantization er-
ror incurred when a matrix is represented using a
finite bit-width in two’s complement format. Two
theorems are presented: one that characterizes the
error in terms of the spectral norm and another in
terms of the Frobenius norm.
The first theorem demonstrates that the spectral
norm of the quantization error is bounded by
∥A− bA∥2 ≲
√m n
2b ∥A∥2.
This result implies that, for a given bit-width b,
matrices with larger spectral norms incur propor-
tionally larger quantization errors. Consequently, a
matrix that exhibits a larger spectral norm is more
susceptible to quantization errors. To control error
propagation in such matrices, a higher bit width is
necessary.
The second theorem provides an analogous
bound for the Frobenius norm,
∥A− bA∥F ≲
√m n
2b ∥A∥F .
Similar to the spectral norm result, this bound in-
dicates that the quantization error, measured in
the Frobenius norm, is directly proportional to the
norm of the original matrix. Hence, matrices with
larger Frobenius norms are also more vulnerable to
quantization errors and would benefit from a higher
precision during quantization.
2The analysis considers a single-head decoder-only Trans-
former without loss of generality; the conclusions extend di-
rectly to multi-head, group-query, and multi-query attention.
B.1 Preliminaries
Let A∈R m×n denote a real matrix whose entries
are to be stored using a fixed number of bits in
two’s complement representation. The following
notation is adopted:
• Bit Depth.Given b bits in two’s-complement
format, each representable integer q lies in the
interval
q∈ {−2 b−1,−2 b−1 + 1, . . . ,2b−1 −1}.
•Maximum Entry Magnitude.Define
M= max
1≤i≤m,1≤j≤n
|Aij|.
•Scale Factor.Set
α= M
2b−1 −1 .
This choice ensures that the scaled entries
Aij/αlie within the representable range.
• Quantization.Define the integer matrix Q∈
Zm×n by
Qij = round
 Aij
α

,
withQ ij ∈ {−2 b−1, . . . ,2b−1 −1}.
• Dequantization (Reconstruction).The recon-
structed matrix bAis given by
bAij =α Q ij.
Objective.The aim is to bound the errors
∥A− bA∥2 and∥A− bA∥F ,
in terms ofb,m,n, and the norms ofA.
14

<!-- page 15 -->

B.2 Spectral Norm Error Bound
Theorem 3 (Spectral Norm Error Bound for
Uniform Quantization)
Let A∈R m×n and b∈N be given. Consider
two’s complement quantization with a scale fac-
tor of
α= M
2b−1 −1 , M= max
i,j
|Aij|.
Define
Qij = round
 Aij
α

and bAij =α Q ij.
Then, the following bound holds:
M≤ ∥A− bA∥2 ≤ √m n M
2(2b−1 −1)
≤ √m n ∥A∥2
2(2b−1 −1) .
In approximate form for largeb,
∥A− bA∥2 ≲
√m n
2b ∥A∥2.
Proof.Entrywise Bound.By construction,

Aij
α −Q ij
 ≤ 1
2 .
Multiplying byαgives
|Aij − bAij| ≤ α
2 = M
2 (2b−1 −1) .
Thus,
max
i,j
|Aij − bAij| ≤ M
2 (2b−1 −1) .
Conversion to the Spectral Norm.Using the
inequality
∥B∥2 ≤ √m nmax
i,j
|Bij|,
withB=A− bA, it follows that
∥A− bA∥2 ≤ √m n M
2 (2b−1 −1) .
RelatingMto∥A∥ 2.Since
M= max
i,j
|Aij| ≤ ∥A∥ 2,
the bound can be written as
∥A− bA∥2 ≤ √m n ∥A∥2
2 (2b−1 −1) .
For large b, where 2b−1 −1≈2 b−1, the bound
becomes
∥A− bA∥2 ≲
√m n
2b ∥A∥2.
B.3 Frobenius Norm Error Bound
Theorem 4 (Frobenius Norm Error Bound for
Uniform Quantization)
Under the same setup as Theorem 3, the Frobe-
nius norm of the quantization error satisfies
∥A− bA∥F ≤ √m n M
2(2b−1−1)
≤ √m n ∥A∥F
2(2b−1−1) .
In approximate form,
∥A− bA∥F ≲
√m n
2b ∥A∥F .
Proof.Entrywise Bound.As established,
|Aij − bAij| ≤ M
2 (2b−1 −1) for alli, j.
Conversion to the Frobenius Norm.By defini-
tion,
∥A− bA∥2
F =
mX
i=1
nX
j=1
(Aij − bAij)2,
which yields
∥A− bA∥2
F ≤m n
 M
2 (2b−1 −1)
2
.
Taking square roots leads to
∥A− bA∥F ≤ √m n M
2 (2b−1 −1) .
RelatingMto∥A∥ F .Since
M 2 ≤
mX
i=1
nX
j=1
A2
ij =∥A∥ 2
F ,
it follows thatM≤ ∥A∥ F and hence
∥A− bA∥F ≤ √m n ∥A∥F
2 (2b−1 −1) .
15

<!-- page 16 -->

For largeb, this simplifies to
∥A− bA∥F ≲
√m n
2b ∥A∥F .
Remark.The results indicate that both the spectral
norm and Frobenius norm errors satisfy similar
approximate bounds:
∥A− bA∥2 ≲
√m n
2b ∥A∥2,
∥A− bA∥F ≲
√m n
2b ∥A∥F .
B.4 Implications for KV Cache Quantization
Consider the key and value cache
V∈R L×dhead,K∈R L×dhead,
with quantization bit-widths denoted by bV and bK,
respectively. LetbV and bK denote the dequantized
matrices, and define the quantization errors as
EV =V− bV,E K =K− bK.
An empirical observation is that
∥V∥∗ <∥K∥ ∗,
where ∗ denotes either the spectral norm ( ∥ · ∥ 2)
or the Frobenius norm ( ∥ · ∥ F ). In practice, K
typically exhibits a larger norm thanV.
Spectral-Norm Perspective.Standard quantiza-
tion error bounds yield
∥EV ∥2 ≲
√L dhead
2bV
∥V∥2,
∥EK∥2 ≲
√L dhead
2bK
∥K∥2.
To achieve comparable spectral-norm errors (i.e.,
∥EV ∥2 ≈ ∥E K∥2), it is necessary that
√L dhead
2bV
∥V∥2 ≈
√L dhead
2bK
∥K∥2.
Cancelling the common factor √L dhead yields
2bV ∥V∥2 ≈2 bK ∥K∥2,
or equivalently,
2 bK −bV ≈ ∥V∥2
∥K∥2
.
Since∥V∥ 2 <∥K∥ 2, it follows thatb K > b V .
Frobenius Norm (MSE) Perspective.The Frobe-
nius norm of the quantization error corresponds
directly to the mean-squared error (MSE) when
normalized by the number of elements. Specifi-
cally, for a matrix A with quantized approximation
bA, the MSE is
MSE(A,bA) = 1
nnz(A) ∥A− bA∥2
F ,
where nnz(A) denotes the number of entries. Thus,
controlling the Frobenius norm is equivalent to
controlling the MSE up to a scaling factor.
The quantization error bounds under the Frobe-
nius norm are given by
∥EV ∥F ≲
√L dhead
2bV
∥V∥F ,
∥EK∥F ≲
√L dhead
2bK
∥K∥F ,
where L is the sequence length and dhead is the
head dimension.
To ensure comparable Frobenius (or MSE) errors
between keys and values, we require
√L dhead
2bV
∥V∥F ≈
√L dhead
2bK
∥K∥F ,
which simplifies to
2bV ∥V∥F ≈2 bK ∥K∥F ,
and therefore
2 bK −bV ≈ ∥V∥F
∥K∥F
.
Since typically ∥V∥F <∥K∥ F , it follows that
bK > b V . This reinforces the earlier spectral-norm
result: keys should be allocated more bits than
values to achieve balanced quantization error under
an MSE criterion.
C Supplemental Results
C.1 Focus on Generative Tasks
We validate and operationalize our theorems across
a diverse set of datasets, including C4, MMLU,
GSM8K, EQ-Bench, CoQA, and LongBench. Our
evaluation purposefully focuses ongenerative
tasks, i.e., open-ended generation and free-form
responses, because KV-cache quantization primar-
ily affects thedecoding phaserather than theprefill
phase. To assess quantization error in isolation,
16

<!-- page 17 -->

0 4 8 12 16 20 24 28 32 36
5
10
15
20
25Frobenius Norm
Ministral2410-8B-it
Key Weight
Value Weight
0 10 20 30 40 50 60 70 80 90
4
6
8
10
12
Mistral2411-Large-it
Key Weight
Value Weight
0 4 8 12 16 20 24 28 32
Layer Index
4
6
8
10
12Frobenius Norm
Mistral0.3-7B-it
Key Weight
Value Weight
0 4 8 12 16 20 24 28 32
Layer Index
4
6
8
10
12
 Mistral0.1-7B
Key Weight
Value Weight
Figure 5:Frobenius norm plot for Mistral Family.
The x-axis represents the layer index in the model, while
the y-axis represents the Frobenius norm magnitude.
The spectral norms are higher for the key weights than
for the value weights across layers.
we use C4 under a free-form decoding setup that
mirrors pretraining usage, enabling us to analyze
how compression directly distorts activations. To
quantify downstream impact, we select GSM8K,
CoQA, EQ-Bench, and LongBench, all of which
rely on generate_until-style evaluation, where
the model must produce extended, structured re-
sponses. In contrast, commonsense reasoning
benchmarks (e.g., BoolQ, PIQA, HellaSwag, or
DoRA) use log-likelihood scoring over candidate
options and do not engage KV-cache quantization
unless the input prompt itself is compressed. Con-
sequently, such discriminative evaluations are or-
thogonal to our focus and were intentionally ex-
cluded.
Within LongBench, we surveyed all subtasks
(gov_report, lcc, lsht, multi_news, narrativeqa,
qasper, qmsum, repobench-p, samsum) and found
that only gov_report and qmsum require substan-
tial generation, with gov_report involving the
longest outputs. Throughout this work, “Long-
Bench” refers specifically togov_report.
C.2 Key-Value Weight Norm
Figure 5 presents the Frobenius norms of key and
value weights for the Mistral family, exhibiting
the same pattern (keys consistently having higher
norms than values) as shown in Figure 2 for the
Llama family, further corroborating the Key-Value
Norm Disparity theorem established in Section 3.1.
C.3 Singular Value Distributions
Figure 6 illustrates the full-spectrum singular value
distribution of key and value caches across layers
of the Llama 3.3 70B model on the C4 dataset. The
horizontal axis indexes singular values in descend-
ing order, starting from the largest (i.e., the spectral
norm), while the vertical axis shows their magni-
tudes. The shaded region represents the minimum-
maximum range of singular values across attention
heads within each layer, and the solid curves indi-
cate the mean singular value at each rank.
C.4 Quantization Error
By using quantization error, we refer to evaluating
how closely the quantized-then-dequantized key/-
value (KV) caches reconstruct their full-precision
counterparts, measured as the reconstruction mean-
squared error (MSE), which is exactly the Frobe-
nius norm error normalized by the number of ele-
ments. This serves as a practical proxy for down-
stream quality because uniform b-bit quantization
admits norm-based bounds in which reconstruc-
tion error scales with the cache norm and decays
roughly like 2−b; thus, at a fixed bit budget, higher-
norm caches incur larger distortion, and bit alloca-
tions that minimize reconstruction error are directly
targeting the quantity most affected by quantiza-
tion.
C.5 Downstream Accuracy Across
Quantization Precisions
Comprehensive downstream accuracy results for
KV cache quantization using the HQQ backend
are provided for CoQA (F1 scores), EQ-Parseable
(parseable-pass rates and exact-match scores), and
GSM8K (flexible and strict accuracy) in Tables 7,
8, 9, and 10, respectively.
C.6 Rotation and Grouping Results
Tables 11-14 present the full set of downstream
evaluation results for integrating rotation-based out-
lier redistribution with mixed-precision KV quanti-
zation across multiple models and tasks, including
GSM8K (flexible and strict exact match), COQA
(F1), and COQA (exact match). Each table reports
accuracy for four key-value bit allocations (K2V2,
K2V4, K4V2, K4V4) under four Rotations: none,
key-only, value-only, and both key and value. The
group size is fixed at 64 for both keys and values in
these experiments to isolate the effect of rotation.
The results reveal several consistent trends
across model scales and tasks. First, applying rota-
17

<!-- page 18 -->

0
50
100Magnitude
Keys, Layer 0
0
250
500
750
Keys, Layer 11
0
250
500
750
Keys, Layer 22
0
250
500
750
Keys, Layer 33
0
500
Keys, Layer 45
0
500
Keys, Layer 56
0
500
1000
Keys, Layer 67
0
250
500
750
Keys, Layer 79
0 50 100
Singular Value Index
0.0
0.5
1.0Magnitude
Values, Layer 0
0 50 100
Singular Value Index
0
20
40
Values, Layer 11
0 50 100
Singular Value Index
0
20
40
60
Values, Layer 22
0 50 100
Singular Value Index
0
20
40
60
Values, Layer 33
0 50 100
Singular Value Index
0
20
40
Values, Layer 45
0 50 100
Singular Value Index
0
50
100
Values, Layer 56
0 50 100
Singular Value Index
0
25
50
75
Values, Layer 67
0 50 100
Singular Value Index
0
50
100 Values, Layer 79
Figure 6:Complete singular value distribution of key and value cachefor Llama 3.3-70B on the C4 dataset.
The x-axis denotes the singular value indices, ordered from the largest (spectral norm) to the smallest, while the
y-axis represents the corresponding magnitudes. The shaded region illustrates the range between the minimum and
maximum singular values across attention heads within each layer, and the solid lines indicate the mean singular value
magnitude at each index. This full-spectrum view highlights that key matrices consistently maintain significantly
higher singular values throughout the entire distribution, further reinforcing their dominant representational capacity
compared to value matrices.
Table 4: Quantization error (mean ± std) for the key and value caches ( Ki and Vi) at 2-bit, 3-bit, and 4-bit
quantization, evaluated on theMMLUdataset.
K2 V2 K3 V3 K4 V4
Llama3.2-1B 4.851± 1.0370.127± 0.101 1.037± 0.2650.021± 0.015 0.227± 0.0590.005± 0.003
Llama3.2-1B-it 4.373± 1.0340.124± 0.090 0.879± 0.2180.019± 0.013 0.192± 0.0470.004± 0.003
Llama3.2-3B 3.943± 0.9240.193± 0.096 0.849± 0.1500.030± 0.015 0.183± 0.0310.007± 0.003
Llama3.2-3B-it 4.487± 1.1800.202± 0.100 0.894± 0.1800.030± 0.015 0.193± 0.0370.007± 0.003
Llama2-7B 3.190± 0.7830.259± 0.184 0.769± 0.1940.042± 0.030 0.168± 0.0420.009± 0.006
Llama3.1-8B-it 6.003± 1.7820.187± 0.127 1.082± 0.2440.028± 0.019 0.235± 0.0550.006± 0.004
Llama3.3-70B-it 4.883± 1.1060.112± 0.093 0.942± 0.1980.016± 0.012 0.206± 0.0430.003± 0.003
Nemotron3.1-it 5.125± 1.2840.114± 0.094 0.985± 0.2070.016± 0.012 0.216± 0.0460.003± 0.003
Phi3-Medium-128K-it 5.063± 1.9140.584± 0.559 1.000± 0.3190.087± 0.083 0.217± 0.0680.019± 0.018
Phi4 5.929± 1.5450.657± 0.472 1.306± 0.2310.103± 0.070 0.286± 0.0500.022± 0.015
Mistral0.3-7B 4.718± 1.3400.398± 0.405 0.941± 0.2400.059± 0.059 0.206± 0.0530.013± 0.013
Qwen2.5-14B 5.184± 2.2411.270± 1.547 1.005± 0.2880.182± 0.221 0.223± 0.0670.040± 0.052
DeepSeekR1L-8B 5.502± 1.5490.189± 0.118 0.955± 0.2040.028± 0.017 0.209± 0.0460.006± 0.004
DeepSeekR1Q-14B 5.126± 2.3751.406± 1.609 0.900± 0.2690.198± 0.226 0.199± 0.0620.044± 0.052
18

<!-- page 19 -->

Table 5: Quantization error (mean ± std) for the key and value caches ( Ki and Vi) at 2-bit, 3-bit, and 4-bit
quantization, evaluated on theC4dataset.
K2 V2 K3 V3 K4 V4
Llama3.2-1B 4.885± 1.0560.207± 0.166 1.074± 0.2890.030± 0.024 0.233± 0.0620.006± 0.005
Llama3.2-1B-it 4.524± 1.1080.193± 0.137 0.925± 0.2350.028± 0.020 0.201± 0.0500.006± 0.005
Llama3.2-3B 3.885± 0.7770.282± 0.150 0.909± 0.1680.042± 0.023 0.194± 0.0320.009± 0.005
Llama3.2-3B-it 4.135± 1.0880.274± 0.137 0.912± 0.1760.039± 0.020 0.195± 0.0360.009± 0.004
Llama2-7B 6.337± 1.7100.456± 0.247 1.054± 0.2630.071± 0.038 0.213± 0.0520.015± 0.008
Llama3.1-8B-it 6.262± 1.7890.254± 0.185 1.128± 0.2490.036± 0.026 0.247± 0.0560.008± 0.005
Llama3.3-70B-it 4.391± 1.0270.121± 0.097 0.847± 0.1750.017± 0.013 0.186± 0.0380.004± 0.003
Nemotron3.1-it 5.367± 1.3320.127± 0.105 1.049± 0.2220.018± 0.013 0.231± 0.0490.004± 0.003
Phi3-Medium-128K-it 4.831± 1.7590.788± 0.726 1.022± 0.3060.109± 0.097 0.220± 0.0640.023± 0.021
Phi4 5.715± 1.4420.850± 0.684 1.316± 0.2450.124± 0.093 0.291± 0.0560.027± 0.020
Mistral0.3-7B 5.027± 1.3320.543± 0.493 1.014± 0.2690.079± 0.068 0.223± 0.0600.017± 0.015
Qwen2.5-14B 4.382± 2.1701.544± 1.872 0.846± 0.2500.220± 0.265 0.187± 0.0600.048± 0.060
DeepSeekR1L-8B 4.575± 1.1220.204± 0.134 0.817± 0.1410.030± 0.019 0.179± 0.0330.006± 0.004
DeepSeekR1Q-14B 4.832± 2.3541.651± 1.914 0.927± 0.2830.232± 0.267 0.201± 0.0610.051± 0.060
Table 6: Quantization error (mean ± std) for the key and value caches ( Ki and Vi) at 2-bit, 3-bit, and 4-bit
quantization, evaluated on the GSM8K dataset.
K2 V2 K3 V3 K4 V4
Llama3.2-1B 5.703± 1.5570.179± 0.136 1.213± 0.3520.026± 0.020 0.266± 0.0780.005± 0.004
Llama3.2-1B-it 5.002± 1.3830.171± 0.130 1.024± 0.2870.025± 0.020 0.223± 0.0610.006± 0.004
Llama3.2-3B 4.840± 1.3960.261± 0.136 1.045± 0.2110.038± 0.021 0.226± 0.0440.008± 0.005
Llama3.2-3B-it 3.604± 0.8500.226± 0.129 0.790± 0.1350.034± 0.019 0.171± 0.0280.007± 0.004
Llama2-7B 5.081± 1.3960.405± 0.231 0.969± 0.2380.065± 0.037 0.205± 0.0500.014± 0.008
Llama3.1-8B-it 6.445± 1.8370.213± 0.161 1.184± 0.2680.030± 0.022 0.257± 0.0600.007± 0.005
Llama3.3-70B-it 4.967± 1.1270.113± 0.091 0.978± 0.2030.016± 0.012 0.214± 0.0440.004± 0.003
Nemotron3.1-it 4.752± 1.1240.113± 0.089 0.940± 0.1940.016± 0.012 0.206± 0.0420.004± 0.003
Phi3-Medium-128K-it 4.940± 1.8340.605± 0.579 1.042± 0.3200.088± 0.082 0.227± 0.0690.019± 0.018
Phi4 6.610± 1.6240.785± 0.598 1.498± 0.2930.116± 0.082 0.330± 0.0640.025± 0.017
Mistral0.3-7B 5.308± 1.3670.461± 0.434 1.065± 0.2880.067± 0.061 0.232± 0.0610.015± 0.013
Qwen2.5-14B 4.829± 2.1791.736± 2.659 0.979± 0.2640.241± 0.372 0.214± 0.0610.051± 0.077
DeepSeekR1L-8B 5.547± 1.5170.193± 0.129 1.000± 0.2120.028± 0.018 0.218± 0.0490.006± 0.004
DeepSeekR1Q-14B 4.477± 2.1761.424± 1.752 0.830± 0.2560.200± 0.242 0.181± 0.0580.044± 0.056
19

<!-- page 20 -->

Table 7:Downstream Accuracy on COQA (Word-Overlap F1) Across Quantization Precisions. (K xVy) denotes
x-bit keys and y-bit values; higher is better. Results show the keys are the bottleneck while values can be compressed
more aggressively: moving from 1-bit to 2-bitkeyswithvalues fixed at 1-bityields large gains (e.g., Qwen3-32B:
0.231 → 0.732; Llama-3.2-3B: 0.128 → 0.577), whereas atfixed keys ≥4-bit, sweeping values from 1→8 bits
changes F1 only marginally (e.g., Qwen3-8B (K6V1) 0.813 vs. (K6V8) 0.819; Qwen3-32B (K8V1) 0.818 vs. (K8V8)
0.827). With sufficiently precise keys (6-8 bits), even2-bit valuesnearly match BF16: Llama-3.2-1B (K 6V2) 0.700
vs. BF16 0.701; Qwen3-32B (K6V2) 0.826 vs. BF16 0.826. In contrast, raisingvaluesat 1-bit keys barely helps
(e.g., Qwen3-4B (K1V2) 0.361 vs. (K1V8) 0.364).
Qwen3 Llama-3.2 Llama-3.2 Qwen3 Llama-3.1 Qwen3 Qwen3
0.6B 1B 3B 4B 8B 8B 32B
K1V1 0.130 0.148 0.128 0.222 0.127 0.359 0.231
K1V2 0.227 0.236 0.223 0.361 0.207 0.384 0.328
K1V4 0.226 0.237 0.229 0.376 0.242 0.386 0.349
K1V6 0.215 0.243 0.230 0.379 0.221 0.379 0.352
K1V8 0.220 0.240 0.222 0.364 0.242 0.382 0.355
K2V1 0.253 0.190 0.577 0.464 0.565 0.599 0.732
K2V2 0.334 0.526 0.760 0.721 0.757 0.767 0.809
K2V4 0.333 0.559 0.766 0.732 0.763 0.761 0.809
K2V6 0.340 0.565 0.764 0.730 0.764 0.766 0.807
K2V8 0.332 0.568 0.764 0.733 0.766 0.770 0.804
K4V1 0.497 0.575 0.769 0.803 0.773 0.815 0.818
K4V2 0.666 0.693 0.797 0.806 0.786 0.822 0.820
K4V4 0.692 0.701 0.798 0.807 0.788 0.819 0.825
K4V6 0.688 0.702 0.797 0.809 0.784 0.817 0.824
K4V8 0.687 0.700 0.797 0.807 0.782 0.818 0.823
K6V1 0.650 0.597 0.771 0.801 0.765 0.813 0.817
K6V2 0.693 0.700 0.798 0.803 0.785 0.821 0.826
K6V4 0.702 0.705 0.796 0.807 0.786 0.821 0.825
K6V6 0.701 0.706 0.795 0.808 0.790 0.819 0.825
K6V8 0.694 0.703 0.796 0.807 0.789 0.819 0.825
K8V1 0.651 0.599 0.769 0.802 0.767 0.815 0.818
K8V2 0.691 0.696 0.798 0.803 0.788 0.823 0.825
K8V4 0.701 0.704 0.795 0.807 0.789 0.821 0.826
K8V6 0.699 0.705 0.795 0.809 0.787 0.819 0.826
K8V8 0.697 0.704 0.795 0.808 0.786 0.821 0.827
BF160.708 0.701 0.795 0.808 0.785 0.820 0.826
20

<!-- page 21 -->

Table 8:Downstream Accuracy on EQ-BENCH PARSEABLEAcross Quantization Precisions. (K xVy) denotes
x-bit keys and y-bit values; the higher the value, the better.Parseable accuracyis the share of prompts where the
model’s output follows the required structured format so the scorer can automatically extract the four 0-10 emotion
ratings. Results consistently show that keys are the bottleneck, while values can be compressed more aggressively.
With 1-bit keys, outputs are never parseable across all models regardless of value precision ((K1Vy=0) everywhere).
Upgrading to2-bit keysyields large jumps even with very low-precision values. For instance, Llama-3.2-3B rises
from 0 to80.702at (K 2V2), Llama-3.1-8B from 0 to85.965, and Qwen3-32B from 0 to67.836, while further
increasingvaluesfrom 2 →8 bits at fixed (K=2) brings only modest gains (e.g., Llama-3.2-3B80.702 →84.211,
Llama-3.1-8B85.965 →90.059, Qwen3-8B65.497 →68.421). Oncekeys are ≥4-6 bits, even1-2-bit valuesnearly
saturate parseability and often match or exceed BF16 (e.g., Qwen3-8B (K4V1)98.830vs. BF1694.737; Qwen3-32B
(K6V1)79.532≈BF1679.532; Llama-3.2-1B (K 6V2)97.076vs. BF1697.661).
Qwen3 Llama-3.2 Llama-3.2 Qwen3 Llama-3.1 Qwen3 Qwen3
0.6B 1B 3B 4B 8B 8B 32B
K1V1 0 0 0 0 0 0 0
K1V2 0 0 0 0 0 0 0
K1V4 0 0 0 0 0 0 0
K1V6 0 0 0 0 0 0 0
K1V8 0 0 0 0 0 0 0
K2V1 0 0 8.187 0 35.673 0 2.339
K2V2 0 16.374 80.702 27.485 85.965 65.497 67.836
K2V4 0 34.503 84.795 39.766 88.889 70.175 70.175
K2V6 0 39.181 83.041 36.842 90.059 67.836 69.591
K2V8 0 38.012 84.211 40.936 88.889 68.421 70.760
K4V1 0 47.953 58.480 77.778 83.626 98.830 77.778
K4V2 63.743 97.076 95.322 87.719 97.661 95.322 80.702
K4V4 65.497 97.076 98.830 84.795 99.415 95.322 80.702
K4V6 64.328 97.076 98.830 85.380 98.830 93.567 80.702
K4V8 62.573 97.076 98.830 84.795 98.830 93.567 80.702
K6V1 23.392 60.234 53.801 81.287 90.643 96.491 79.532
K6V2 99.415 97.076 94.737 87.719 97.661 94.737 80.702
K6V4 99.415 97.661 99.415 87.135 98.830 95.906 80.702
K6V6 99.415 97.661 98.830 86.550 98.830 96.491 80.702
K6V8 99.415 97.661 98.830 87.135 98.830 97.661 80.702
K8V1 26.901 60.234 56.725 78.947 92.983 97.076 77.193
K8V2 99.415 97.076 94.737 87.135 97.661 96.491 79.532
K8V4 99.415 97.661 99.415 87.719 98.830 95.322 80.702
K8V6 99.415 97.661 99.415 86.550 98.830 95.906 80.702
K8V8 99.415 97.661 99.415 86.550 98.830 95.906 80.702
BF16100 97.661 99.415 87.719 98.830 94.737 79.532
21

<!-- page 22 -->

Table 9:Downstream Accuracy on GSM8K (FLEXIBLE) Across Quantization Precisions. (K xVy) denotes
x-bit keys and y-bit values; the higher the value, the better.Flexible accuracycounts a prediction as correct if the
gold final numeric answer appears anywhere in the model’s output (ignoring extra formatting), rather than requiring
an isolated exact-match box. Results consistently show that keys are the bottleneck, while values can be compressed
more aggressively. With 1-bit keys, performance is essentially zero across all models (max= 0.028). Upgrading
to2-bit keysyields large jumps even with low-precision values—for example, Llama-3.2-3B rises from (0.015)
at (K1V2) to 0.278 at (K2V2), Llama-3.1-8B from (0.020) to 0.385, and Qwen3-32B from (0.018) to 0.385. At
fixedK( ≥6), even2-bit valuesalready match or beat BF16 (e.g., Qwen3-8B (K 6V2) 0.880 vs. BF16 0.877;
Llama-3.1-8B 0.762 vs. 0.760; Qwen3-4B 0.830 vs. 0.845), and increasing values from (2→8) bits yields only
modest gains (Llama-3.2-3B 0.644(→)0.665, Qwen3-8B 0.880(→)0.886). In several cases, quantized caches
evenexceedBF16 (e.g., Qwen3-8B (K 6V8) 0.886> 0.877; Llama-3.2-3B (K4V6) 0.668> 0.663; Qwen3-32B
(K4V1)0.721>0.603).
Qwen3 Llama-3.2 Llama-3.2 Qwen3 Llama-3.1 Qwen3 Qwen3
0.6B 1B 3B 4B 8B 8B 32B
K1V1 0.011 0.012 0.016 0.010 0.016 0.011 0.020
K1V2 0.008 0.014 0.015 0.010 0.020 0.013 0.018
K1V4 0.009 0.020 0.025 0.014 0.020 0.008 0.011
K1V6 0.010 0.011 0.027 0.017 0.020 0.008 0.016
K1V8 0.008 0.015 0.028 0.009 0.015 0.014 0.011
K2V1 0.003 0.018 0.015 0.008 0.050 0.013 0.014
K2V2 0.003 0.029 0.278 0.027 0.385 0.129 0.385
K2V4 0.003 0.020 0.351 0.043 0.455 0.171 0.434
K2V6 0.004 0.026 0.344 0.040 0.446 0.169 0.419
K2V8 0.004 0.028 0.355 0.035 0.460 0.167 0.449
K4V1 0.023 0.073 0.281 0.577 0.517 0.779 0.721
K4V2 0.040 0.294 0.639 0.808 0.757 0.877 0.649
K4V4 0.040 0.333 0.653 0.830 0.778 0.877 0.629
K4V6 0.055 0.332 0.668 0.832 0.763 0.881 0.619
K4V8 0.042 0.341 0.658 0.826 0.758 0.878 0.632
K6V1 0.093 0.096 0.308 0.632 0.522 0.795 0.701
K6V2 0.368 0.312 0.644 0.830 0.762 0.880 0.598
K6V4 0.405 0.334 0.654 0.837 0.770 0.878 0.610
K6V6 0.420 0.337 0.663 0.844 0.772 0.884 0.596
K6V8 0.414 0.342 0.665 0.841 0.771 0.886 0.608
K8V1 0.088 0.096 0.304 0.637 0.530 0.798 0.708
K8V2 0.390 0.313 0.647 0.826 0.750 0.882 0.606
K8V4 0.416 0.340 0.660 0.832 0.770 0.875 0.621
K8V6 0.419 0.342 0.667 0.843 0.767 0.880 0.608
K8V8 0.413 0.334 0.666 0.843 0.759 0.880 0.603
BF160.412 0.328 0.663 0.845 0.760 0.877 0.603
22

<!-- page 23 -->

Table 10:Downstream Accuracy on GSM8K (STRICT) Across Quantization Precisions. (K xVy) denotes
x-bit keys and y-bit values; higher is better. Same dataset as Table 9, butStrict accuracyonly counts a prediction
as correct when the scorer can extract a single final numeric answer that exactly matches the gold. As with the
Flexible metric, keys are the bottleneck, while values can be compressed more aggressively. With 1-bit keys,
accuracy is essentially zero across all models (max = 0.002). Upgrading to2-bit keysyields large jumps even
with low-precision values—for example, Llama-3.2-3B rises from (0) to 0.276 at (K2V2), Llama-3.1-8B from
(0) to 0.380, and Qwen3-32B from (0) to 0.234. Oncekeys are ≥4-6 bits, even2-bit valuesare near or above
BF16 (e.g., Qwen3-8B (K6V2) 0.879 vs. BF16 0.874; Llama-3.2-3B (K8V6) 0.661>0.657 ; Qwen3-32B (K8V6)
0.723>0.718 ). At fixed high key precision, increasing values from (2→8) bits brings only modest gains (e.g.,
Qwen3-8B at (K= 6):0.879(→)0.885; Llama-3.2-3B at (K= 6):0.639(→)0.657).
Qwen3 Llama-3.2 Llama-3.2 Qwen3 Llama-3.1 Qwen3 Qwen3
0.6B 1B 3B 4B 8B 8B 32B
K1V1 0.000 0.000 0.000 0.000 0.000 0.000 0.000
K1V2 0.000 0.000 0.000 0.000 0.000 0.000 0.000
K1V4 0.000 0.000 0.000 0.000 0.000 0.000 0.000
K1V6 0.000 0.000 0.002 0.000 0.000 0.001 0.000
K1V8 0.000 0.000 0.000 0.000 0.000 0.000 0.000
K2V1 0.000 0.001 0.005 0.000 0.028 0.000 0.002
K2V2 0.000 0.011 0.276 0.004 0.380 0.067 0.234
K2V4 0.000 0.010 0.350 0.009 0.444 0.104 0.334
K2V6 0.000 0.012 0.344 0.010 0.439 0.115 0.330
K2V8 0.000 0.014 0.353 0.005 0.449 0.109 0.347
K4V1 0.006 0.061 0.293 0.647 0.500 0.785 0.701
K4V2 0.042 0.297 0.628 0.825 0.751 0.876 0.726
K4V4 0.044 0.331 0.643 0.844 0.759 0.876 0.734
K4V6 0.055 0.331 0.657 0.839 0.748 0.880 0.733
K4V8 0.049 0.342 0.649 0.835 0.745 0.877 0.737
K6V1 0.083 0.083 0.326 0.698 0.506 0.803 0.702
K6V2 0.369 0.308 0.639 0.839 0.750 0.879 0.709
K6V4 0.405 0.334 0.644 0.845 0.756 0.876 0.719
K6V6 0.422 0.335 0.656 0.853 0.751 0.882 0.717
K6V8 0.414 0.341 0.657 0.848 0.753 0.885 0.721
K8V1 0.085 0.084 0.324 0.704 0.513 0.805 0.695
K8V2 0.388 0.314 0.640 0.829 0.739 0.882 0.717
K8V4 0.417 0.340 0.652 0.845 0.756 0.875 0.721
K8V6 0.419 0.340 0.661 0.848 0.749 0.879 0.723
K8V8 0.415 0.332 0.660 0.851 0.744 0.876 0.723
BF160.413 0.328 0.657 0.852 0.741 0.874 0.718
23

<!-- page 24 -->

0 10 20 30 40 50 60 70
Layer Index
0
250
500
750
1000
1250
1500Frobenius Norm
Key Matrix (Mean)
Value Matrix (Mean)
(a) Frobenius Norm
0 10 20 30 40 50 60 70
Layer Index
0
200
400
600
800
1000
1200Spectral Norm
Key Matrix (Mean)
Value Matrix (Mean)
(b) Spectral Norm
Figure 7:Frobenius and spectral norms of key and
value caches across layers for the Llama 3.3 70B
model on the C4 dataset.The x-axis represents the
layer index. The shaded regions indicate the min-max
range across attention heads within each layer, while
the solid curves represent the mean norm per layer. In
both cases, key matrices consistently exhibit substan-
tially higher norms than value matrices, reflecting their
stronger representational capacity.
2 3 4 5 6 7 8
Bit Width
10 6
10 4
10 2
100
Quantization Error (MSE)
key (min-max)
key (mean)
value (min-max)
value (mean)
Figure 8:MSE of KV quantization for Llama 3.3-70B
on the C4 dataset.We use quantization bit-widths rang-
ing from 2 to 8. The x-axis represents the quantization
bit-width, while the y-axis shows the MSE on a loga-
rithmic scale. A logarithmic scale is used to highlight
differences at higher bit-widths, where MSE values de-
crease significantly and approach zero, particularly at
8-bit. Solid lines indicate the mean MSE across layers,
while the shaded regions represent the min-max range
of errors.
tion tokeysconsistently improves performance at
lower bit-widths (K2V2 and K2V4), significantly
narrowing the gap to higher-precision baselines. In
contrast, rotatingvaluesalone yields marginal or
even negative effects, especially at low precisions.
Notably, applying rotation tokeys onlyunder the
K4V2 configuration achieves accuracy that closely
matches the full K4V4 baseline, indicating that the
dominant benefits of rotation stem from mitigating
key outliers rather than value outliers. Applying
rotation to both keys and values provides little ad-
ditional gain beyond key-only rotation, reinforcing
that keys are the primary target for rotation-based
improvements.
Figure 11 examines the effect ofgroup size con-
figurationunder a fixed K4V2 mixed-precision set-
ting without rotation. Group size (gs) determines
the block granularity of quantization: smaller
groups offer finer scaling at the cost of increased
metadata and computation. The results reveal a
clear asymmetry between keys and values. Re-
ducing group size forkeysconsistently improves
downstream accuracy across COQA, GSM8K,
EQ-BENCH, and LONGBENCH, with gsK = 32
achieving the best overall performance. This re-
flects the higher sensitivity of key caches to quan-
tization distortion. In contrast, increasing value
group size to 64 or 128 has a negligible impact on
accuracy while reducing overhead, consistent with
their lower sensitivity.
Together, these findings provide practical guid-
ance for combining geometry-driven bit alloca-
tion with rotation and grouping strategies: rotation
should primarily targetkeys, while keys benefit
from finer group granularity and higher precision;
values, on the other hand, can use coarser grouping
and lower precision with minimal accuracy loss.
D Reproducibility and Resources
Inference was performed using the Hugging Face
Transformers (Wolf et al., 2020) and Accelerate
(Gugger et al., 2022) with FlashAttention (Dao
et al., 2022; Dao, 2024). We integrated both Quanto
and HQQ into the Language Model Evaluation Har-
ness (Gao et al., 2024) to enable systematic and re-
producible evaluation of model performance under
varying quantization schemes. Quantization error
is measured by MSE, and confidence intervals use
Bayesian variance. (Hariri et al., 2025). Evalu-
ations were executed on two High-performance
Computing (HPC) clusters, detailed in Table 16.
24

<!-- page 25 -->

Table 11:Rotation-based Outlier Redistribution Inte-
gration on GSM8K - Exact Match (Flexible).Down-
stream accuracy under different key-value bit allocations
and rotation scopes. Group size is fixed to 64 for both
keys and values.
Model RotationK 2V2 K2V4 K4V2 K4V4
Llama
3.1-8B
K+V 52.99 55.80 76.50 76.95
K only 52.92 57.32 75.44 76.50
V only 20.70 25.70 75.28 76.95
none 18.65 24.41 75.21 76.95
Llama
3.2-1B
K+V 1.97 2.43 30.93 33.13
K only 2.05 3.03 28.43 33.13
V only 2.05 1.29 28.58 29.57
none 1.82 2.20 27.22 29.57
Llama
3.2-3B
K+V 40.11 44.12 62.85 64.37
K only 38.06 45.26 61.18 63.91
V only 16.60 19.26 63.46 63.91
none 14.86 21.08 62.93 63.53
Qwen
0.6B
K+V 0.76 0.61 37.83 36.47
K only 1.52 1.06 36.92 37.68
V only 1.52 1.67 1.36 2.12
none 0.83 1.90 1.36 2.05
Qwen
32B
K+V 31.69 33.81 66.49 64.22
K only 30.10 33.51 63.99 65.28
V only 6.90 11.07 65.81 63.68
none 6.67 10.92 64.82 63.76
Qwen
4B
K+V 0.91 0.76 83.78 85.37
K only 1.14 0.68 82.26 84.76
V only 0.83 1.14 82.56 83.70
none 1.14 0.83 81.80 83.40
Qwen
8B
K+V 29.34 36.01 88.17 88.17
K only 29.57 35.56 87.64 87.95
V only 1.97 1.21 86.73 87.34
none 2.27 1.67 86.58 88.02
Table 12:Rotation-based Outlier Redistribution In-
tegration on GSM8K - Exact Match (Strict).Down-
stream accuracy under different key-value bit allocations
and rotation scopes. Group size is fixed to 64 for both
keys and values.
Model RotationK 2V2 K2V4 K4V2 K4V4
Llama
3.1-8B
K+V 51.71 55.27 74.37 74.45
K only 51.93 56.48 73.24 73.92
V only 19.33 24.49 73.01 74.60
none 17.97 22.74 72.93 74.60
Llama
3.2-1B
K+V 0.83 1.21 31.01 32.98
K only 0.68 1.29 29.04 33.06
V only 0.38 0.61 28.51 29.57
none 0.45 0.61 27.22 29.34
Llama
3.2-3B
K+V 39.65 43.67 61.94 63.84
K only 36.47 44.96 60.27 63.46
V only 16.00 19.26 63.00 63.38
none 14.40 21.00 62.02 62.85
Qwen
0.6B
K+V 0.00 0.00 38.14 36.92
K only 0.00 0.00 37.38 37.83
V only 0.00 0.00 0.15 0.53
none 0.00 0.00 0.30 0.08
Qwen
32B
K+V 21.99 28.35 74.75 75.59
K only 21.46 26.38 74.75 75.28
V only 2.20 5.00 74.00 75.13
none 1.52 4.93 73.77 74.37
Qwen
4B
K+V 0.23 0.08 84.53 85.60
K only 0.00 0.08 83.40 84.99
V only 0.00 0.00 82.49 83.70
none 0.00 0.08 81.80 83.78
Qwen
8B
K+V 25.32 30.33 87.72 87.87
K only 25.17 32.15 87.49 87.34
V only 0.15 0.00 86.43 86.88
none 0.00 0.08 86.50 87.41
25

<!-- page 26 -->

Table 13:Rotation-based Outlier Redistribution In-
tegration on COQA - F1 Score.Downstream accuracy
under different key-value bit allocations and rotation
scopes. Group size is fixed to 64 for both keys and val-
ues.
Model RotationK 2V2 K2V4 K4V2 K4V4
Llama
3.1-8B
K+V 77.70 77.83 78.88 78.45
K only 77.39 78.10 77.93 78.75
V only 75.46 76.71 79.34 78.76
none 74.72 76.57 78.10 79.18
Llama
3.2-1B
K+V 50.66 53.55 70.44 70.47
K only 50.18 53.46 69.40 70.27
V only 39.50 44.93 69.58 70.52
none 38.82 44.33 69.53 70.31
Llama
3.2-3B
K+V 78.45 78.78 79.24 79.13
K only 78.24 78.88 78.71 79.20
V only 73.25 74.48 78.86 79.27
none 74.10 74.55 79.14 79.37
Qwen
0.6B
K+V 33.99 34.06 68.92 70.09
K only 33.88 33.70 69.62 70.08
V only 29.00 27.39 48.19 48.06
none 28.89 28.04 46.49 48.41
Qwen
32B
K+V 64.85 67.39 82.46 82.59
K only 67.16 67.18 82.62 82.55
V only 77.82 78.55 82.35 82.50
none 76.80 78.35 82.60 82.77
Qwen
4B
K+V 62.43 63.82 80.16 80.56
K only 62.09 64.35 80.46 80.63
V only 56.73 59.26 80.07 80.16
none 54.93 59.68 80.41 80.19
Qwen
8B
K+V 79.33 79.65 82.32 81.98
K only 78.27 79.89 81.15 82.09
V only 67.51 70.41 82.23 82.01
none 66.86 69.83 81.49 81.99
Table 14:Rotation-based Outlier Redistribution In-
tegration on COQA - Exact Match.Downstream
accuracy under different key-value bit allocations and
rotation scopes. Group size is fixed to 64 for both keys
and values.
Model RotationK 2V2 K2V4 K4V2 K4V4
Llama
3.1-8B
K+V 62.43 63.37 63.25 63.40
K only 62.12 63.32 63.03 63.65
V only 58.13 60.12 64.48 63.75
none 58.12 59.65 62.92 64.45
Llama
3.2-1B
K+V 33.38 37.87 54.78 56.02
K only 33.72 37.30 54.22 55.32
V only 25.47 30.37 53.97 55.65
none 24.42 30.23 54.30 55.45
Llama
3.2-3B
K+V 62.47 62.28 62.83 63.40
K only 62.12 62.72 62.93 63.53
V only 56.07 57.48 62.70 63.73
none 56.45 57.62 63.42 63.77
Qwen
0.6B
K+V 19.67 19.57 56.07 56.73
K only 18.72 18.18 55.53 56.73
V only 13.97 14.27 28.18 28.17
none 14.25 13.80 28.22 28.82
Qwen
32B
K+V 46.97 49.95 70.18 70.77
K only 50.28 49.42 70.48 70.87
V only 61.77 64.08 70.02 70.37
none 61.12 63.92 69.90 70.50
Qwen
4B
K+V 41.28 42.45 65.68 66.28
K only 41.65 43.45 66.40 66.47
V only 39.22 40.03 65.43 66.20
none 37.70 40.22 65.52 66.07
Qwen
8B
K+V 61.70 61.88 68.55 67.73
K only 60.07 62.40 66.62 68.23
V only 49.62 51.58 68.10 68.07
none 48.40 51.65 67.07 68.00
26

<!-- page 27 -->

Table 15:Performance by rotation strategy across tasks.Values report the mean performance across seven
models (0.6B–32B parameters) and four quantization configurations (K2V2, K2V4, K4V2, K4V4). The ± values
indicate the variability range across model scales and quantization settings, with larger ranges reflecting higher
sensitivity to these factors.
Task Metric No Rotation Value-Only Key-Only Both
CoQa Exact Match50.10±17.65 51.82±17.22 55.34±14.21 55.39±14.10
Eq Bench Parseable56.18±42.57 55.01±41.93 65.73±39.70 64.81±39.19
GSM8K Exact Match32.73±33.42 32.96±33.48 43.46±28.98 43.80±29.28
Longbench ROUGE Score16.70±11.75 16.54±11.82 17.50±12.83 17.63±12.81
Table 16: Specifications of Two High-Performance Computing (HPC) Clusters Used in This Study
Cluster Cluster A Cluster B
ProcessorAMD EPYC 7742×2 Intel Xeon Platinum 8468×2
RAM2048 GB 2048 GB
GPUNVIDIA A100×8 NVIDIA H200×8
VRAM80 GB HBM2e 141 GB HBM3e
Scale5 nodes (40 A100) 5 nodes (40 H200)
27

<!-- page 28 -->

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
1.031
3.933
6.834
9.736
12.637
15.539
18.440
21.342
k2
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.001
0.388
0.775
1.162
1.549
1.936
2.323
2.710
v2
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.041
0.143
0.245
0.347
0.450
0.552
0.654
0.756
k4
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.000
0.010
0.020
0.029
0.039
0.049
0.059
0.068
v4
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
(a) C4
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.842
3.462
6.083
8.704
11.325
13.946
16.567
19.188
k2
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.001
0.221
0.442
0.662
0.883
1.103
1.324
1.544
v2
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.030
0.118
0.205
0.293
0.380
0.468
0.555
0.643
k4
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.000
0.007
0.015
0.022
0.030
0.037
0.045
0.052
v4
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
(b) MMLU
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
1.159
3.898
6.636
9.375
12.113
14.852
17.590
20.328
k2
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.001
0.342
0.683
1.023
1.364
1.705
2.046
2.386
v2
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.046
0.156
0.266
0.376
0.486
0.595
0.705
0.815
k4
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
0.000
0.009
0.019
0.028
0.037
0.047
0.056
0.065
v4
head=0
head=1
head=2
head=3
head=4
head=5
head=6
head=7
(c) GSM8k
Figure 9:Quantization error (MSE) of key and value caches in the Llama 3.1 8B model across 32 layers for
(a) C4, (b) MMLU, and (c) GSM8K.The top row shows 2-bit quantization (K2, V2), and the bottom row shows
4-bit quantization (K4, V4). Each point corresponds to an attention head within the respective layer. The x-axis
denotes the layer index, and the y-axis indicates the quantization error (MSE).
28

<!-- page 29 -->

0.0
0.1
0.2
0.3
0.4
0.5
0.6Performance
0.580 0.579 0.582
0.633 0.635 0.636
0.593 0.595 0.597
CoQA
0
20
40
60
80
70.92770.59371.094
89.55789.47489.891
79.36579.44979.365
EQ-Bench
Kgs=128, Vgs=128
Kgs=128, Vgs=32
Kgs=128, Vgs=64
Kgs=32, Vgs=128
Kgs=32, Vgs=32
Kgs=32, Vgs=64
Kgs=64, Vgs=128
Kgs=64, Vgs=32
Kgs=64, Vgs=64
Group Size
0.0
0.1
0.2
0.3
0.4
0.5
0.6Performance
0.559 0.561 0.559
0.602 0.597 0.599 0.591 0.589 0.587
GSM8K
Kgs=128, Vgs=128
Kgs=128, Vgs=32
Kgs=128, Vgs=64
Kgs=32, Vgs=128
Kgs=32, Vgs=32
Kgs=32, Vgs=64
Kgs=64, Vgs=128
Kgs=64, Vgs=32
Kgs=64, Vgs=64
Group Size
0.00
0.05
0.10
0.15
0.20 0.189 0.188 0.190
0.207 0.208 0.205 0.203 0.206 0.206
LongBench
Figure 11:Effect of group size configurations on downstream accuracy.Each bar shows performance on
COQA, GSM8K, EQ-BENCH, and LONGBENCHfor different key-value grouping combinations ( gsK,gs V ∈
{32,64,128} ) under the K4V2 mixed-precision setting. Smaller group sizes forkeysconsistently improve accuracy
across all tasks, while value group size has a smaller but non-negligible effect. Configurations with gsK = 32
achieve the best overall performance, highlighting the benefit of finer quantization granularity for key caches, which
are more sensitive to quantization distortion. In contrast, larger group sizes for values (gsV = 64 or 128) preserve
performance while reducing overhead, aligning with their lower sensitivity.
29
