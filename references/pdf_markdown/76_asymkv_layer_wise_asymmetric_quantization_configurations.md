# references/76_asymkv_layer_wise_asymmetric_quantization_configurations.pdf

<!-- page 1 -->

Proceedings of the 31st International Conference on Computational Linguistics , pages 2316–2328
January 19–24, 2025. ©2025 Association for Computational Linguistics
2316
AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise
Asymmetric Quantization Configurations
Qian Tao1, Wenyuan Yu1, Jingren Zhou2
1Tongyi Lab, Alibaba Group 2Alibaba Cloud Computing, Alibaba Group
Correspondence: qian.tao@alibaba-inc.com
Abstract
Large language models have shown exceptional
capabilities in a wide range of tasks, such as
text generation and video generation, among
others. However, due to their massive parame-
ter count, these models often require substantial
storage space, imposing significant constraints
on the machines deploying LLMs. To over-
come this limitation, one research direction
proposes to compress the models using inte-
ger replacements for floating-point numbers, in
a process known as Quantization. Some recent
studies suggest quantizing the key and value
cache (KV Cache) of LLMs, and designing
quantization techniques that treat the key and
value matrices equivalently.
This work delves deeper into the asymmetric
structural roles of KV Cache, a phenomenon
where the transformer’s output loss is more sen-
sitive to the quantization of key matrices. We
conduct a systematic examination of the atten-
tion output error resulting from key and value
quantization. The phenomenon inspires us to
propose an asymmetric quantization strategy.
Our approach allows for 1-bit quantization of
the KV cache by implementing distinct config-
urations for key and value matrices. We carry
out experiments across a variety of datasets,
demonstrating that our proposed model allows
for the quantization of up to 75% decoder lay-
ers with 1 bit, while simultaneously maintain-
ing performance levels comparable to those of
the models with floating parameters.
1 Introduction
Large language models (LLMs) have gained con-
siderable interest of late due to their remarkable per-
formance in various directions (McKeown, 1992;
Wang et al., 2022; Taylor et al., 2022; Ji et al., 2021;
Gruver et al., 2024). However, to achieve a high
level of expressiveness, LLMs typically require bil-
lions of parameters, which necessitates substantial
storage space and poses challenges for deployment
on machines with limited resources.
A line of research has been dedicated to enabling
the deployment of these models on machines with
less available space through model compression
techniques. One such technique, model quanti-
zation, aims to represent the parameter matrices
in LLMs using fewer bits ( e.g., integer, binary),
thereby making them more suitable for deployment
on hardware with limited storage capacity (Kim
et al., 2023). More recently, the Key-Value cache
(KV cache) in LLMs has been shown to occupy a
large proportion of space (Pope et al., 2023; Mo-
htashami and Jaggi, 2023), especially when the
length of context increases, and numerous works
have focused on the quantization of KV cache (Liu
et al., 2024a,c; Kang et al., 2024). Nonetheless,
these studies typically employ the same quantiza-
tion configuration for both key and value matrices.
In this paper, we cast a spotlight on the asymmet-
ric structural roles of key and value matrices. Our
analysis reveals that, while a quantization method
could yield a quantized matrix with a commen-
surate loss for both key and value matrices, the
multiplication of query and application of the acti-
vation function to the key matrix results in a larger
loss of key matrix in the transformer’s output as
compared to the value matrix.
Drawing on this observation, this paper intro-
duces a simple yet efficacious quantization strategy,
which entails the use of asymmetric and layer-wise
quantization configurations for key and value matri-
ces. Specifically, during the next token’s inference,
we employ a higher-bit quantization strategy (for in-
stance, a 4-bit strategy) for the firstl decoder layers,
whilst a lower-bit strategy (i.e., the 1-bit strategy)
is applied for the remaining decoder layers. For
key and value matrices, we choose different l to
account for their asymmetric structural positions.
Our extensive experiments reveal that the adoption
of an asymmetric and layer-wise quantization strat-
egy allows us to quantize a subset of layers using a
1-bit approach, resulting in a strategy that is both

<!-- page 2 -->

2317
space and computationally efficient.
In summary, the primary contributions of this
paper can be outlined as follows:
• We conduct the exploration of the asymmetric
structural roles of the key and value matrices.
Through practical and theoretical demonstra-
tions, we show that the loss derived from the
key matrix’s quantization will be magnified
relative to that of the value matrix, owing to
the multiplication of the query and activation
function applied specifically to the key matrix.
• To counteract the impact of asymmetric struc-
tural roles, this paper proposes AsymKV, a
simple yet effective approach that combines
varied degrees of quantization configurations
at the layer level. AsymKV applies differ-
ent quantization strategies to the key and
value matrices, striking a balance between
consumed memory and model performance.
• We conduct experiments on various datasets
to substantiate the effectiveness of AsymKV.
Our results validate the asymmetric roles of
the key and value matrices and demonstrate
that by applying distinct quantization strate-
gies to the key and value matrices, LLMs can
be equipped with the extreme 1 bit quantiza-
tion while ensuring performance on par with
the models utilizing floating-point parameters.
In the remainder of this paper, we first outline the
basic definitions of transformers and KV cache in
Sec. 2, then we highlight the observed asymmetric
structural roles in Sec. 3, and present the design
of AsymKV in Sec. 4. The evaluation and related
works of AsymKV are discussed in Sec. 5 and
Sec. 6, respectively. We finally conclude in Sec. 7.
2 Preliminaries
2.1 Attention Mechanism and KV Cache
Given the input embeddings of an attention mech-
anism, X ∈ R t×h, where t represents the num-
ber of tokens already generated and h is the di-
mension of attention head, an attention mechanism
M (Vaswani, 2017; Ainslie et al., 2023; Shazeer,
2019) obtains the hidden states as follows:
Q = XWq,K = XWk, V = XWv
Aw = sm( QKT
√
h
)
Ao = AwV
Here, Wq, Wk and Wv are the weight matrices for
the query, key, and value, respectively, andsm(·)
signifies the softmax function. Aw and Ao are
typically referred to as the attention weights and
attention output, respectively.
As an LLM generates tokens, the embeddings of
the newly produced token are appended to the end
of X, necessitating the generation of query, key,
and value matrices. Consequently, we can store the
embeddings of K and V from previous tokens and
only generate the corresponding segments for the
new token in K and V. Specifically, by partition-
ing X into the embeddings of previous tokens, i.e.,
X1:t−1, and the embeddings of the current token,
Xt, we can leverage the key and value cache to
enhance LLM’s computational efficiency.
xq = XtWq, xk = XtWk, xv = XtWv
K1:t = cat(K1:t−1, xk), V1:t = cat(V1:t−1, xv)
Aw = xqKT
1:t√
h
(1)
Aw = sm(Aw) (2)
Ao = AwV1:t (3)
Here, the key and value matrices, K1:t−1 and
K1:t−1 are cached while generating the last token.
2.2 KV Cache Quantization
Round-To-Nearest Quantization. While enhanc-
ing computational efficiency, the KV cache de-
mands considerable memory, particularly as more
tokens are generated. To mitigate this, previous
studies propose quantizing the key and value ma-
trices into integers to accommodate more tokens
using a Round-To-Nearest (RTN) methodology.
Formally, given a key or value matrix, M ∈
Rt×h, an RTN quantization breaks down M into
the quantized matrix MQ, the scaling matrix s, and
the zero-point matrix z as follows.
Quantization Phase:
z = min
i
(M), s = maxi(M) − mini(M)
2b − 1 (4)
MQ = ⌊ M − z
s ⌉ (5)
Dequantization Phase:
M∗ = (MQ + z) ∗ s (6)
Here, b represents the required bit of quantization,
and mini (respectively maxi) is a function that

<!-- page 3 -->

2318
Figure 1: Squared error in the inference of attention.
retrieves the minimum (respectively maximum)
tensor of the input in relation to the i-th dimen-
sion. i may be chosen from {1, 2}, representing
per-channel or per-token quantization respectively.
Measurement of Quantization. Given a quanti-
zation method, a natural question would be how to
measure the effectiveness of the proposed method.
Recent works (Frantar et al., 2022; Dong et al.,
2024) proposed using the squared error of the
output between the quantized weights and full-
precision weights to measure the effectiveness or
optimize the strategies. Formally, the error is
e = |||f (M∗) − f (M)||2
2 (7)
where f (·) could be a linear layer or the whole
attention layer (i.e.,Equ. 1-Equ. 3). Following these
works, we use the squared error to study how the
structure of attention affects the effectiveness.
3 Asymmetric Attention Sensitivity of KV
Cache Quantization
As shown in Equ. 1-Equ. 3, the key matrix and
value matrix perform distinct roles in transform-
ers. While existing studies have proposed intri-
cate quantization methods to mitigate the loss from
quantization and some studies (Dong et al., 2024)
have recognized the disparate roles of the key ma-
trix and value matrix, an important question still
lingers: provided that the key matrix and value ma-
trix play different roles from various perspectives,
for instance, the multiplication of xq and the opera-
tion of softmax function on key matrix, what factor
truly contributes to the loss of the transformer?
Observation. For the key (respectively value) ma-
trix, we hold the value (respectively key) matrix in
floating type, and evaluate the accumulated mean
squared error between the output with key (respec-
tively value) matrix in floating type and that with
2-bit quantization at different stages of the atten-
tion. Fig. 1 illustrates the average loss per element
during the inference of the Llama-2 model of size
7b. Here, the green (respectively red) line denotes
the MSE between the attention output with floating
type and the 2-bit quantization of the key (respec-
tively value) matrix in different stages of the atten-
tion. The number on the lines depicted in Fig. 1
represents the ratio between the MSE that arises
from the key matrix quantization and the MSE that
arises from the value matrix quantization.
Interestingly, even though the quantization strat-
egy results in a comparative loss ( i.e., the MSE
after Equ. 6) on the key matrix and value matrix,
there emerges marginal gap loss for key matrices
after the multiplication of xq, i.e., after Equ. 1. The
gap is further amplified after the softmax function,
i.e., after Equ. 2. This indicates that even though
the quantization methods can guarantee a similar
loss for key and value matrices, the multiplication
of xq and the activation function makes the MSE of
the attention output for the key matrix significantly
larger than that of the value matrix.
MSE Amplification. Next, we analyze why the
multiplication of xq and the softmax function ex-
acerbates the MSE of the key matrix. Consider
a matrix M and its quantization matrices, MQ, z,
and s. M could be either the key matrix or the value
matrix. Assume that the deviation of each element
between M and the quantized matrix follows the
distribution P, i.e., |Mi,j − M∗
i,j| ∼ P. We aim
to understand how the error of an element in the
matrix varies after being multiplied by a vector.
Proposition 1. Consider a matrix M and its esti-
mation M∗. Denote the error by E = M − M∗.
Upon left multiplying by a matrix A, the error ma-
trix becomes AE. Correspondingly, a right multi-
plication of A results in the error EA.
Proof. Consider the (s, r)-th element of AM. We
could obtain its error
As,·M·,r − As,·M∗
·,r
=
X
i
As,i(Mi,r − M∗
i,r)
=
X
i
As,iϵi,r (8)
which precisely corresponds to the (s, r)-th ele-
ment of AE. Similarly, the right multiplication of

<!-- page 4 -->

2319
(a) Layer 21 Head 13
 (b) Layer 30 Head 25
 (c) Layer 31 Head 22
Figure 2: Statistics of the error from key matrix quantization and value matrix quantization.
A results in an error matrix EA.
Based on Proposition 1, we can deduce the error
stemming from the value matrix’s quantization.
Proposition 2. Given a value matrix V and its
quantization V∗, with a quantization error Ev =
V − V∗, the error in the attention output is AwEv.
Proposition 2 can be derived from Equ. 3 and
Proposition 1. On the other hand, it is also feasible
to derive the error resulting from the quantization
of key matrices, although this process is complex
due to the involvement of softmax functions.
Theorem 1. Given a key matrix K and its quantiza-
tion K∗, with a quantization error Ek = K − K∗,
the error of the attention output is given by (Aw ⊙
(1 − sr · e
Eq
√
h ) · V, where Eq = xqEk, min Eq
and max Eq are the smallest and largest elements
of Eq respectively, and sr = sf t/sf t∗ such that
sf t = P
j e
P
i qiKi,j√
h and sf t∗ = P
j e
P
i qiK∗
i,j√
h
are the dominator in the softmax function for the
key matrix K and K∗ respectively.
Proof. Consider the error in the 1, r-th element of
Aw. It is given by
e
P
i qiKi,r√
h
sf t − e
P
i qiK∗
i,r√
h
sf t∗
= e
P
i qiKi,r√
h
sf t (1 − sf t
sf t∗
e
P
i qiK∗
i,r√
h
e
P
i qiKi,r√
h
)
=Aw
1,r(1 − sf t
sf t∗ e
P
i qi(K∗
i,r −Ki,r )
√
h )
=Aw
1,r(1 − sf t
sf t∗ e
−xq Ekr√
h )
=Aw
1,r(1 − sf t
sf t∗ e
Eq
√
h ) (9)
This can be reformulated in matrix form as Aw
1,r ⊙
(1−sr ·e
Eq
√
h ). Since Aw is subsequently multiplied
by V, in accordance with Proposition 1, the error
in the attention output is given by
(Aw
1,r ⊙ (1 − sr · e
Eq
√
h )) · V. (10)
To demonstrate the difference in error caused by
the quantization of the key and value matrix, we se-
lect three decoder layers and plotted the error from
Equ. 8 and Equ. 10 in Fig. 2. The results indicate
that the distribution of the key matrix quantization
error is more sparse around0 compared to the value
matrix quantization, which consequently leads to a
larger MSE for the key matrix.
Discussion: Why does the key matrix quanti-
zation leads to a larger error than the value
matrix? For the value matrix, it is not influenced
by the softmax function, making its error straight-
forward to compute and directly tied to the quanti-
zation error. In contrast, for the key matrix quan-
tization as shown in Equ. 10, sf t = P
j e
P
i qiKi,j√
h
and sf t∗ = P
j e
P
i qiK∗
i,j√
h are relatively large, and
they are nearly equivalent becauseK∗ is the quanti-
zation of K. This suggests that sr ≈ 1 and the key
discrepancy between the errors of the key matrix
quantization and value matrix quantization arises
in the Hadamard product of 1 − sr · e
Eq
√
h . (1) Mul-
tiplication of xq. Observe that the first dimension
of xq is consistently set to 1. Thus, the multiplica-
tion by xq results in each element accumulating the
error from quantization multiple times. This is il-
lustrated in Equ. 10, where each element of Eq has
a comparatively larger error than the error distri-
bution of P, given that Eq = qEk. (2) Utilization

<!-- page 5 -->

2320
Figure 3: Workflow of AsymKV .
of the softmax function. In the key matrix error
obtained in Equ. 10, the original error from the key
matrix quantization is situated in the exponentia-
tion of e. As the proof of Theorem 1 demonstrates,
this replacement stems from the utilization of the
softmax function, in which all elements are treated
as the exponentiation of e. In consideration of the
super-linear growth rate of the power function, the
softmax function further exacerbates the loss in-
duced by the key matrix quantization.
4 AsymKV: Layer-wise Quantization with
Asymmetric Configuration
From the discussion in Sec. 3, it is evident that
the quantization of the key matrix could potentially
result in a more significant loss for the attention
output due to the specific role of the key matrix.
In response to this, our study introduces AsymKV,
a simple yet efficacious quantization strategy that
blends various degrees of quantization for the key
and value matrix based on their respective impacts
on the loss of the attention mechanism.
Basic Idea. AsymKV applies various degrees of
quantization to the key and value matrix at the
layer level. Specifically, AsymKV introduces two
parameters, lk and lv, to control the degree of quan-
tization for the key and value matrix, respectively.
During the inference of the model, for the key (re-
spectively value) matrix, the initial lk (respectively
lv) attention layers utilize a quantization method
with a higher number of bits ( e.g., 4-bit or 2-bit).
In contrast, the remaining attention layers employ
a quantization method with fewer bits (i.e., 1-bit).
Fig. 3 illustrates the design of AsymKV where
green, blue, and red blocks symbolize the matrices
in full-precision, higher-bit quantization, and lower-
bit quantization, respectively. For each attention
layer, its key and value matrices are cached with
different quantization bits based on the layer index.
As demonstrated in Fig. 3, those layers with a layer
index i ≤ lk (respectively i ≤ lv) will cache the
quantized key (respectively value) matrices with
higher bits, while the other layers will use lower
bits. After generating the query, key, and value
matrix of the current token, i.e., Kt, Qt, and Vt,
the LLM will produce the output of the attention
Ao, as illustrated in Equ. 1-Equ. 3. Given that
AsymKV chooses lk and lv such that lv ≤ lk, those
decoder layers with indices in range [lv, lk] will
contain a blend of higher bits for key matrix and
lower bits for value matrix.
The design of AsymKV relies on the observa-
tions in Sec. 3 as well as certain intuitive insights.
(1) Asymmetric Configuration. In light of our
observation in Sec. 3, we decide to independently
configure the degree of quantization for key and
value matrices by defining the configuration pa-
rameters lk and lv, respectively. Besides, since the
quantization error for the key matrix results in a
larger error for the attention output, we generally
choose a larger lk than lv to achieve performance
comparable to the models with full precision.
(2) Layer-wise Quantization. While generating a
token, the quantization error is accumulated as the
number of attention layers increases. Therefore, by
choosing the later attention layers to be quantized

<!-- page 6 -->

2321
with fewer bits, we can mitigate the error caused by
the quantization from being amplified, while con-
currently allowing the KV cache to be quantized
with a less number of bits.
Discussion. Generally speaking, the design of
AsymKV is not dependent on any specific quanti-
zation techniques. Our findings indicate that the
performance of a LLM model is more significantly
impacted by the quantization of key matrices. Con-
sequently, the propose AsymKV can be applied to
various quantization techniques for KV cache to
achieve a better balance between space efficiency
and performance.
5 Evaluation
5.1 Experimental Setup
Tested Models. We examine the performance
of AsymKV using the widely used LLM fam-
ily Llama (Touvron et al., 2023), which includes
Llama-2-7b and Llama-2-13b. All models are de-
ployed based on the LLM implementation from
Huggingface1 with the default implementation of
quantization selected from (Liu et al., 2024c).
Tasks and Baselines. In terms of model per-
formance, we evaluate AsymKV on tasks with
a standard context length, including CoQA and
TruthfulQA from LM-Eval (Gao et al., 2024),
as well as tasks with long context length from
LongBench (Bai et al., 2023), including TriviaQA,
TREC, SAMSum, RepoBench-P, and Qasper. Re-
garding model efficiency, we assess the memory
usage of AsymKV under various quantization con-
figurations, comparing it with previous works that
handle the key and value matrices uniformly, in-
cluding the original floating implementation, and
KIVI (Liu et al., 2024c) with 2-bit quantization.
Inference Settings. Following KIVI (Liu et al.,
2024c), AsymKV employs per-channel quantiza-
tion for key matrices and per-token quantization
for value matrices. Consequently, both KIVI and
AsymKV store the key matrices of a limited num-
ber of tokens in floating-point types, a parameter
referred to as residual length. We choose a residual
length of 128 for tasks with normal context length,
while for tasks with long context length, we opt
for a residual length of 512. For the peak memory
usage experiments, we standardized the generation
1https://huggingface.co/
Model Type TruthfulQA CoQA
Llama-2-7b
float 30.76 63.88
KIVI-2bit 33.95 63.05
AsymKV-0/16 12.81 34.18
AsymKV-16/0 38.77* 58.12*
Llama-2-13b
float 29.53 66.37
KIVI-2bit 29.84 66.23
AsymKV-0/20 9.52 43.13
AsymKV-20/0 28.44* 61.42*
Table 1: Evaluation on tasks with normal context length
(bold: Higher bits for key matrix better than lower bits
for key matrix, *: AsymKV achieves at least 90% per-
formance of floating-type models).
length of tokens to 4096.
Implementation. AsymKV is implemented us-
ing PyTorch and is built upon the Huggingface
codebase. All experiments are executed on a ma-
chine equipped with 200GB memory and an A800
GPU with 80GB memory. Each decoder layer
in AsymKV adheres to the quantization scheme
outlined in KIVI (Liu et al., 2024c), that is, per-
channel quantization for the key matrix and per-
token quantization for the value matrix, with a
group size of 32. AsymKV utilizes a combina-
tion of higher 2-bit quantization and lower 1-bit
quantization. To validate our analysis concerning
the diverse errors instigated by the key matrix quan-
tization and value matrix quantization, we also ex-
amine AsymKV under various quantization config-
urations.
5.2 Evaluation Results
5.2.1 Tasks with Normal Context Length
Table 1 presents the experimental results for tasks
with normal context length, namely CoQA and
TruthfulQA. In this case, the model AsymKV-lk/lv
represents AsymKV where the key and value matri-
ces in the first lk and lv attention layers are respec-
tively quantized with 2-bit, while those in other
layers are quantized with 1 bit.
Upon examining AsymKV with various quanti-
zation configurations, we observe that AsymKV-
16/0 (respectively AsymKV-20/0) performs better
than AsymKV-0/16 (respectively AsymKV-0/20)
for Llama-7b (respectively Llama-13b). This find-
ing aligns with our observation and analysis in
Sec. 3, where the quantization of key matrices
results in a higher loss than that of value matri-
ces. Therefore, even though AsymKV-16/0 and

<!-- page 7 -->

2322
Model Type TriviaQA TREC SAMSum RepoBench-P Qasper
Llama-2-7b
float 87.72 66.0 41.69 59.82 9.52
KIVI-2bit 87.64 66.0 41.62 56.81 9.73
AsymKV-0/32 11.6 25.0 3.79 23.9 3.18
AsymKV-32/0 85.27* 65.50* 38.28* 43.35 8.96 *
Llama-2-13b
float 87.87 70.00 43.55 56.42 9.32
KIVI-2bit 87.31 69.50 43.52 53.66 8.27
AsymKV-0/40 24.57 28.5 5.25 25.33 3.33
AsymKV-40/0 86.70* 67.50* 41.90* 46.92 8.78 *
Table 2: Evaluation on LongBench tasks (bold: Higher bits for key matrix better than lower bits for key matrix, *:
AsymKV achieves at least 90% performance of floating-type models).
AsymKV-0/16 occupy the same space in GPU
memory, a quantization strategy that employs
higher bits for the key matrix and lower bits for
the value matrix enhances performance.
Besides, AsymKV yields performance compa-
rable to Llama and KIVI while using less GPU
memory, achieved by implementing asymmetric 1-
bit quantization. In particular, AsymKV-16/0 and
AsymKV-20/0 assures a minimum performance of
91.0% that of Llama and 92.2% that of KIVI. In
contrast to KIVI, which quantizes both key and
value matrices with 2 bits, AsymKV allows for
75% decoder layers quantized with the extreme 1
bit, which is more efficient in peak memory.
5.2.2 Tasks with Long Context Length
Table 2 presents the experimental results for tasks
with long context lengths. Mirroring the tasks with
normal context length, AsymKV with a higher
bit count in the key matrix ( i.e.,AsymKV-32/0
for Llama-7b and AsymKV-40/0 for Llama-13b)
once more surpasses AsymKV with value matrices
quantized with higher bits (i.e.,AsymKV-0/32 and
AsymKV-0/40). This aligns with the reasons as
illustrated in Sec. 5.2.1.
Besides, in the case of long context lengths,
AsymKV necessitates more decoder layers quan-
tized with higher bits to attain performance com-
parable to the baselines ( lk = 32 /40 for long
context length vs. lk = 16 /20 for normal con-
text length). When contrasted with the baselines,
AsymKV could assure performance levels of at
least 91.8% and 92.0% relative to Llama and KIVI
across 4 out of 5 datasets, except for RepoBench-P.
5.2.3 Ablation Analysis
Evaluation of Varying Higher Bits. We choose
different higher bit configurations to analyze the
ablation performance of AsymKV. In Table 3, we
Type TruthfulQA CoQA
float 30.76 63.88
KIVI-2bit 33.95 63.05
AsymKV-2/1-0/16 12.81 34.18
AsymKV-2/1-16/0 38.77* 58.12*
AsymKV-4/1-0/16 8.72 32.67
AsymKV-4/1-16/0 41.63* 56.32
Table 3: Ablation evaluation on varying higher bits for
Llama-7b (bold: Higher bits for key matrix better than
lower bits for key matrix, *: AsymKV achieves at least
90% performance of floating-type models).
use “AsymKV-hi/lo-lk/lv” to represent AsymKV
with higher bit hi, a lower bit lo, and the degrees
of quantization for key and value matrices, denoted
as lk and lv, respectively.
From Table 3, when we set higher and lower bits
to 4 and 1 respectively, AsymKV adheres to the pat-
tern established in our paper. Specifically, a model
with more key matrices quantized to lower bits
tends to exhibit poorer performance, as evidenced
by the comparison between AsymKV-4/1-0/16 and
AsymKV-4/1-16/0. Besides, when we increase the
higher bit from 4 to 2, 3 out of 4 cases, namely
AsymKV-4/1-0/16 on both datasets and AsymKV-
4/1-16/0 on CoQA, demonstrate worsened perfor-
mance, which may seem counterintuitive. This
phenomenon could be attributed to the larger dis-
parity between higher and lower bits, which may
disrupt the harmony in the correlation between key
and value matrices.
Evaluation of Varying lk and lv. We further eval-
uate the performance of AsymKV by varying lk
and lv, which represent the number of key or value
matrices quantized with the higher bit. In our eval-
uation, we quantize all key (resp. value) matrices
using the lower bit and vary the number of value
(resp. key) matrices quantized with the higher bit,

<!-- page 8 -->

2323
Model Type TruthfulQA CoQA
Llama-2-7b
float 30.76 63.88
KIVI-2bit 33.95 63.05
AsymKV-0/12 7.37 28.92
AsymKV-0/16 12.81 34.18
AsymKV-0/22 12.23 35.60
AsymKV-12/0 29.17* 48.02
AsymKV-16/0 38.77* 58.12*
AsymKV-22/0 40.14* 59.83*
Table 4: Ablation evaluation on varying lk and lv (*:
AsymKV achieves at least 90% performance of floating-
type models).
selecting from the set {12, 16, 22}.
From Table 4, it is evident that as lk or lv in-
creases, the performance of AsymKV on both
datasets improves. Notably, when lk is reduced
to 12, there is a significant performance gap be-
tween AsymKV-12/0 and the floating-type model.
Based on our evaluation, the quantization configu-
ration of lk = 16 and lv = 0 strikes a good balance
between performance and efficiency for tasks with
normal length. Additional experimental results can
be found in the appendix.
5.2.4 Peak Memory
Fig. 4 reports the experimental results of the peak
memory in GPU for AsymKV. We choose a batch
size of 48 for Llama-7b and 36 for Llama-13b, and
report the peak storage consumption by varying the
quantization configurations lk and lv. Specifically,
we first set lv = 0, implying all value matrices of
the decoder layers are quantized with 1 bit, and in-
crease the number of key matrices quantized with
2 bits, i.e., lk, from 0 to the maximum number of
decoder layers, illustrated in the left part of Fig. 4a
and Fig. 4b. Then, we keep all key matrices quan-
tized with 2 bits and further increase the number
of value matrices quantized with 2 bits, i.e., lv, as
shown in the right part of Fig. 4a and Fig. 4b. It
is noteworthy that when both lk and lv achieve the
maximum number of layers, the results correspond
to the performance of KIVI.
From Fig. 4, as more attention layers are quan-
tized with higher bits, the consumed space in GPU
increases almost linearly until all attention layers
employ a quantization configuration with higher
bits. The locations where AsymKV achieves com-
parable performance to the floating-point model
on tasks with normal and long context lengths are
highlighted. For Llama-7b, AsymKV can ensure
similar performance while saving 9.0 GB and 6.0
GB of space for the tasks with normal and long con-
text lengths respectively, compared to KIVI. For
Llama-13b, the memory saved increases to 10.4GB
and 7.0GB space for tasks with normal and long
context lengths respectively.
6 Related Works
Large language models have gained considerable
attention since their inception. Despite their impres-
sive performance, these models are constrained by
their immersive quantity of parameters, which re-
sults in hardware limitations and poor throughput.
To address these issues, recent research trends
are centered on reducing the size of LLMs (Kim
et al., 2023). Among these methods, quanti-
zation techniques target the transformation of a
portion of the model’s parameters into integers,
which reduces the space of LLMs. For instance,
llm.int8 (Dettmers et al., 2022) suggests quantiz-
ing the query, key, and value weights of LLMs
using the round-to-nearest method, i.e.,, mapping
each floating-point number to its closest integer.
AWQ (Lin et al., 2024) and SmoothQuant (Xiao
et al., 2023) further introduce an amplifying scale
prior to quantization to prevent extremely large out-
liers during the process. Omniquant (Shao et al.,
2023) devises a quantization algorithm by imple-
menting a learnable scale and learnable clipping
during quantization. GPTQ (Frantar et al., 2022)
perceives quantization as a problem of minimizing
square error and designing the quantization algo-
rithms using an approximation of the second-order
information. These studies mainly focus on the
quantization of the model weights.
On the other hand, to mitigate redundant com-
putations across token generation, LLMs utilizes
KV cache. While KV cache enhances inference
efficiency, it consumes significant space, particu-
larly when generating long contexts. Consequently,
another line of research focuses on the compression
of KV cache (Zhang et al., 2024; Kwon et al., 2023;
Jin et al., 2023; Liu et al., 2024b). Among these
approaches, quantization techniques have garnered
much attention and have emerged as a popular tool
for KV cache compression.
Previous works have applied consistent quan-
tization techniques for both model weights and
KV cache. SmoothQuant (Xiao et al., 2023) also
quantizes the query, key, and value matrices to
further minimize memory usage. In contrast, Flex-
gen (Sheng et al., 2023) structures the problem of

<!-- page 9 -->

2324
(a) AsymKV for Llama-7b (batch size 48).
 (b) AsymKV for Llama-13b (batch size 36).
Figure 4: Memory Variation of AsymKV.
quantization in an environment conprising GPU,
CPU, and memory. ATOM (Zhao et al., 2024) uti-
lizes the quantized weights and re-quantizes the
key and value cache matrices into integer types.
More recently, several works have examined the
distriution of the KV cache and formulated quan-
tization algorithms specifically tailored for it. For
example, ATOM (Zhao et al., 2024) discovered that
the key matrix contains more outliers than the value
matrix. KIVI (Liu et al., 2024c) extends on this
observation and suggesting quantizing the key and
value matrices from different perspectives (employ-
ing per-channel quantization for key matrix and
per-token quantization for value matrix). Along-
side this, KVQuant employs the per-channel quan-
tization to the key matrices and introduces a non-
uniform quantization technique for KV cache. In-
tactKV (Liu et al., 2024a) identifies outliers caused
by common tokens, including the punctuations and
split tokens. Meanwhile, WKVQuant (Yue et al.,
2024) proposes a two-dimensional quantization
strategy to smoothly handle the outliers across dif-
ferent channels. Other studies seek to combine
the quantization techniques with other techniques
to approach a fine-grained KV cache compression.
GEAR (Kang et al., 2024) establishes the residual
matrix and sparse matrix to capture the residual and
individual outliers during the quantization. (Yang
et al., 2024) proposes a mix-precision quantiza-
tion scheme and quantizes the crucial KV cache
with higher bits. In contrast to the aforementioned
studies, this paper concentrates on the asymmetric
roles of the key and value matrices. We attribute
this phenomenon to the multiplication of xq and
the application of the softmax function. We pro-
pose a simple yet effective solution: employing
distinct quantization configurations for the key and
value matrices at the layer level. This approach is
designed to accommodate the asymmetric role of
the key and value matrices.
7 Conclusions
This paper primarily concentrates on the asymmet-
ric roles of the key and value matrices in the quan-
tization of the KV cache. We analyze why quan-
tizing the key matrix leads to a more significant
performance drop than quantizing the value matrix
and attribute it to the multiplication of xq and the
implementation of the softmax function. Based on
this analysis, we introduce AsymKV, which applies
asymmetric and layer-wise quantization configu-
rations to the key and value matrices. AsymKV
facilitates a mixed quantization approach with 2
bits and 1 bit, while simultaneously ensuring per-
formance comparable to the floating-type model.
Extensive experiments validate our analysis of the
asymmetric roles of the key and value matrices.
8 Limitations
Despite AsymKV facilitating the quantization of
1 bit for KV cache in LLMs, it still depends on
exhaustive testing to identify the optimal configu-
rations for different LLMs, i.e., configurations that
yield performance close to models in floating types.
This approach is relatively inefficient. A potential
futural direction could involve efficiently identify-
ing the optimal configurations for LLMs. Besides,
AsymKV maintains a consistent quantization con-
figuration for a decoder layer during the generation
of new tokens. However, it might prove more flexi-
ble and efficient if we consider a mixture of higher
and lower bit quantizations at the token level.

<!-- page 10 -->

2325
References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
2023. Gqa: Training generalized multi-query trans-
former models from multi-head checkpoints. arXiv
preprint arXiv:2305.13245.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, et al. 2023. Longbench:
A bilingual, multitask benchmark for long context
understanding. arXiv preprint arXiv:2308.14508.
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke
Zettlemoyer. 2022. Gpt3. int8 (): 8-bit matrix multi-
plication for transformers at scale. In NIPS.
Shichen Dong, Wen Cheng, Jiayu Qin, and Wei Wang.
2024. Qaq: Quality adaptive quantization for llm kv
cache. arXiv preprint arXiv:2403.04643.
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and
Dan Alistarh. 2022. Gptq: Accurate post-training
quantization for generative pre-trained transformers.
arXiv preprint arXiv:2210.17323.
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li,
Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, An-
ish Thite, Ben Wang, Kevin Wang, and Andy Zou.
2024. A framework for few-shot language model
evaluation.
Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew G
Wilson. 2024. Large language models are zero-shot
time series forecasters. In NeurIPS.
Shaoxiong Ji, Tianlin Zhang, Luna Ansari, Jie Fu,
Prayag Tiwari, and Erik Cambria. 2021. Mentalbert:
Publicly available pretrained language models for
mental healthcare. arXiv preprint arXiv:2110.15621.
Yunho Jin, Chun-Feng Wu, David Brooks, and Gu-Yeon
Wei. 2023. s3: Increasing gpu utilization during gen-
erative inference for higher throughput. In NeurIPS.
Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa
Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao.
2024. Gear: An efficient kv cache compression
recipefor near-lossless generative inference of llm.
arXiv preprint arXiv:2403.05527.
Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen
Dong, Xiuyu Li, Sheng Shen, Michael W Ma-
honey, and Kurt Keutzer. 2023. Squeezellm:
Dense-and-sparse quantization. arXiv preprint
arXiv:2306.07629.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In SOSP.
Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-
Ming Chen, Wei-Chen Wang, Guangxuan Xiao,
Xingyu Dang, Chuang Gan, and Song Han. 2024.
Awq: Activation-aware weight quantization for on-
device llm compression and acceleration. In MLSys.
Ruikang Liu, Haoli Bai, Haokun Lin, Yuening Li, Han
Gao, Zhengzhuo Xu, Lu Hou, Jun Yao, and Chun
Yuan. 2024a. Intactkv: Improving large language
model quantization by keeping pivot tokens intact.
arXiv preprint arXiv:2403.01241.
Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao
Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyril-
lidis, and Anshumali Shrivastava. 2024b. Scis-
sorhands: Exploiting the persistence of importance
hypothesis for llm kv cache compression at test time.
In NeurIPS.
Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong,
Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and
Xia Hu. 2024c. Kivi: A tuning-free asymmetric 2bit
quantization for kv cache. In ICML.
Kathleen McKeown. 1992. Text generation. Cambridge
University Press.
Amirkeivan Mohtashami and Martin Jaggi. 2023.
Landmark attention: Random-access infinite con-
text length for transformers. arXiv preprint
arXiv:2305.16300.
Reiner Pope, Sholto Douglas, Aakanksha Chowdhery,
Jacob Devlin, James Bradbury, Jonathan Heek, Kefan
Xiao, Shivani Agrawal, and Jeff Dean. 2023. Effi-
ciently scaling transformer inference. In ICML.
Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng
Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng
Gao, Yu Qiao, and Ping Luo. 2023. Omniquant:
Omnidirectionally calibrated quantization for large
language models. arXiv preprint arXiv:2308.13137.
Noam Shazeer. 2019. Fast transformer decoding:
One write-head is all you need. arXiv preprint
arXiv:1911.02150.
Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan
Li, Max Ryabinin, Beidi Chen, Percy Liang, Christo-
pher Ré, Ion Stoica, and Ce Zhang. 2023. Flexgen:
High-throughput generative inference of large lan-
guage models with a single gpu. In ICML.
Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas
Scialom, Anthony Hartshorn, Elvis Saravia, Andrew
Poulton, Viktor Kerkez, and Robert Stojnic. 2022.
Galactica: A large language model for science. arXiv
preprint arXiv:2211.09085.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288.

<!-- page 11 -->

2326
A Vaswani. 2017. Attention is all you need. In NeurIPS.
Haifeng Wang, Hua Wu, Zhongjun He, Liang Huang,
and Kenneth Ward Church. 2022. Progress in ma-
chine translation. Engineering, 18:143–153.
Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu,
Julien Demouth, and Song Han. 2023. Smoothquant:
Accurate and efficient post-training quantization for
large language models. In ICML.
June Yong Yang, Byeongwook Kim, Jeongin Bae,
Beomseok Kwon, Gunho Park, Eunho Yang, Se Jung
Kwon, and Dongsoo Lee. 2024. No token left be-
hind: Reliable kv cache compression via importance-
aware mixed precision quantization. arXiv preprint
arXiv:2402.18096.
Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan
Zhou, Jianlong Wu, and Liqiang Nie. 2024.
Wkvquant: Quantizing weight and key/value cache
for large language models gains more. arXiv preprint
arXiv:2402.12065.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, et al. 2024.
H2o: Heavy-hitter oracle for efficient generative in-
ference of large language models. In NeurIPS.
Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn
Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy,
Tianqi Chen, and Baris Kasikci. 2024. Atom: Low-
bit quantization for efficient and accurate llm serving.
In MLSys.

<!-- page 12 -->

2327
Model Type TruthfulQA CoQA
Llama-2-7b
float 30.76 63.88
KIVI-2bit 33.95 63.05
AsymKV-0/6 4.11 26.90
AsymKV-0/12 7.37 28.92
AsymKV-0/16 12.81 34.18
AsymKV-0/22 12.23 35.60
AsymKV-6/0 7.64 36.00
AsymKV-12/0 29.17* 48.02
AsymKV-16/0 38.77* 58.12*
AsymKV-22/0 40.14* 59.83*
Llama-2-13b
float 29.53 66.37
KIVI-2bit 29.84 66.23
AsymKV-0/5 4.81 37.53
AsymKV-0/10 4.16 39.70
AsymKV-0/20 9.52 43.03
AsymKV-0/30 10.24 45.20
AsymKV-5/0 15.35 41.25
AsymKV-10/0 19.43 45.40
AsymKV-20/0 28.44* 61.42*
AsymKV-30/0 29.50* 64.92*
Table 5: Evaluation on tasks with normal context length
(*: AsymKV achieves at least 90% performance of
floating-type models).
A Supplemental Experiments
In this section, we present the complete experimen-
tal setup and the corresponding results.
A.1 Experimental Results
A.1.1 Results on Tasks with Normal Context
Length
Table 5 proposes the performance of AsymKV
with varying lk and lv values for tasks with nor-
mal context length. For Llama-7b, we choose
lk, lv ∈ {6, 12, 16, 20} and for Llama-13b, we con-
sider lk, lv ∈ {5, 10, 20, 30}.
As the number of decoder layers quantized with
higher bits increases, the performance of AsymKV
improves until it reaches performance levels com-
parable to the floating-point model and KIVI. Be-
sides, we observe that AsymKV with value matri-
ces quantized using lower bits, i.e.,AsymKV-l/0,
consistently outperforms AsymKV with key matri-
ces quantized using lower bits, i.e.,AsymKV-0/l,
and the difference is substantial. This observa-
tion confirms that choosing a configuration with
lk > l v can enhance the performance of AsymKV.
AsymKV can achieve at least 90% of the perfor-
mance of floating-point models when a quantiza-
tion configuration that follows AsymKV-16/0 for
Llama-7b and AsymKV-20/0 for Llama-13b is uti-
lized.
A.1.2 Results on Tasks with Long Context
Length
Table 6 presents the experimental results for tasks
with long context length. For key and value matri-
ces, we set aside one type of matrices quantized
with higher bits (i.e., lk/lv = 32/40) and vary the
number of the other type of matrices that are quan-
tized with lower bits.
Similar to the tasks with normal context lengths,
the performance of AsymKV augments as more
key and value matrices are quantized with higher
bits. Besides, AsymKV with key matrices quan-
tized with higher bits (AsymKV-32/lv for Llama-
7b and AsymKV-40/ lv for Llama-13b) outper-
forms AsymKV with value matrices quantized with
higher bits, despite them occupying the same GPU
memory.

<!-- page 13 -->

2328
Model Type TriviaQA TREC SAMSum RepoBench-P Qasper
Llama-2-7b
float 87.72 66.0 41.69 59.82 9.52
KIVI-2bit 87.64 66.0 41.62 56.81 9.73
AsymKV-0/32 11.6 25.0 3.79 23.9 3.18
AsymKV-6/32 19.02 29.0 5.53 28.46 4.04
AsymKV-12/32 22.96 42.50 8.77 32.34 5.13
AsymKV-32/0 85.27* 65.50* 38.28* 43.35 8.96*
AsymKV-32/6 86.36* 66.50* 39.75* 49.93 9.04*
AsymKV-32/12 86.62* 66.00* 40.93* 52.46 9.64*
Llama-2-13b
float 87.87 70.00 43.55 56.42 9.32
KIVI-2bit 87.31 69.50 43.52 53.66 8.27
AsymKV-0/40 24.57 28.5 5.25 25.33 3.33
AsymKV-10/40 42.30 41.00 12.64 28.65 5.10
AsymKV-15/40 48.14 50.00 17.82 31.37 5.73
AsymKV-40/0 86.70* 67.50* 41.90* 46.92 8.78*
AsymKV-40/10 86.80* 69.00* 42.23* 50.68 7.56*
AsymKV-40/15 87.39* 69.00* 42.45* 50.25 8.58*
Table 6: Evaluation on LongBench tasks (*: AsymKV achieves at least 90% performance of floating-type models).
