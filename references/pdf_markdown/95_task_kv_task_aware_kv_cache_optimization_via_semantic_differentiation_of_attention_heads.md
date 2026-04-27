# references/95_task_kv_task_aware_kv_cache_optimization_via_semantic_differentiation_of_attention_heads.pdf

<!-- page 1 -->

1
Task-KV: Task-aware KV Cache Optimization via
Semantic Differentiation of Attention Heads
Xingyang He, Jie Liu, Shaowei Chen
Abstract—KV cache is a widely used acceleration technique for
large language models (LLMs) inference. However, its memory
requirement grows rapidly with input length. Previous studies
have reduced the size of KV cache by either removing the
same number of unimportant tokens for all attention heads or
by allocating differentiated KV cache budgets for pre-identified
attention heads. However, due to the importance of attention
heads varies across different tasks, the pre-identified attention
heads fail to adapt effectively to various downstream tasks. To
address this issue, we propose Task-KV , a method that leverages
the semantic differentiation of attention heads to allocate differ-
entiated KV cache budgets across various tasks. We demonstrate
that attention heads far from the semantic center (called hetero-
geneous heads) make an significant contribution to task outputs
and semantic understanding. In contrast, other attention heads
play the role of aggregating important information and focusing
reasoning. Task-KV allocates full KV cache budget to hetero-
geneous heads to preserve comprehensive semantic information,
while reserving a small number of recent tokens and attention
sinks for non-heterogeneous heads. Furthermore, we innovatively
introduce middle activations to preserve key contextual informa-
tion aggregated from non-heterogeneous heads. To dynamically
perceive semantic differences among attention heads, we design a
semantic separator to distinguish heterogeneous heads from non-
heterogeneous ones based on their distances from the semantic
center. Experimental results on multiple benchmarks and differ-
ent model architectures demonstrate that Task-KV significantly
outperforms existing baseline methods. Notably, in scenarios
requiring full-context processing, such as summarization and
synthetic tasks, Task-KV achieves performance comparable to
the full KV cache while utilizing only 40% of the memory.
Index Terms—Large language models, KV cache optimization,
Long-context, Inference acceleration
I. I NTRODUCTION
L
LMS are widely utilized in long-context scenarios such
as in-context learning [1], [2], multi-turn conversations
[3], [4], and retrieval-augmented [5], [6] tasks. To improve
inference speed and efficiency, LLMs reduce redundant com-
putations by caching the Key and Value states (KV cache) of
all tokens across all attention heads [7], [8]. However, as the
length of the input sequence increases, the storage require-
ment of KV cache expands dramatically, posing significant
challenges to memory capacity and inference efficiency.
To address this issue, researchers have proposed various KV
cache compression methods, mainly starting from two dimen-
sions: token-level and head-level, as shown in Fig. 1. Token-
Xingyang He is with the College of Artificial Intelligence, NanKai Univer-
sity, Tianjin 300350, China (e-mail: xingyanghe@mail.nankai.edu.cn).
Jie Liu is with the College of Artificial Intelligence, NanKai University,
Tianjin 300350, China (e-mail: jliu@nankai.edu.cn).
Shaowei Chen is with the College of Artificial Intelligence, NanKai
University, Tianjin 300350, China (e-mail: shaoweichen@mail.nankai.edu.cn).
Head1
Head2
(a) Token-Level (b) Head-Level
Head1
Head2
Task1
(c) Task-Level (Ours)
Task2
Fig. 1. Illustration of Task-KV compared with existing KV cache compression
methods. (a) Token-level methods allocate the same KV cache budget to each
attention head. (b) Head-level methods pre-identify important attention heads,
but the KV cache budget among these heads remains fixed regardless of the
task. (c) Our method identifies important attention heads based on the specific
task and dynamically adjusts the KV cache budget among attention heads
according to task semantics.
level [9]–[13] compression methods evict a fixed number of
unimportant tokens for each attention head, aiming to reduce
the KV cache size while preserving generation quality as much
as possible. Head-level [14]–[17] compression methods, on the
other hand, pre-identify important attention heads [17], [18]
(e.g., retrieval heads) through experiments and allocate KV
cache budgets based on their significance during inference,
thereby further optimizing KV cache compression. However,
the importance of attention heads varies across different tasks
[19], [20], meaning that the pre-identified important attention
heads may not be universally critical for all tasks. A key chal-
lenge remains: how to adaptively identify and select critical
attention heads based on task-specific requirements.
In this paper, we propose a novel approach called Task-KV ,
which dynamically allocates KV cache budgets by leverag-
ing task-aware semantic differences among attention heads.
Through theoretical analysis and empirical validation, we
demonstrate that attention heads far from the semantic center,
referred to as heterogeneous heads, make particularly signifi-
cant contributions to task outputs. These heterogeneous heads
capture the semantic information of the task from different
perspectives, which is crucial for LLMs to fully understand
the task semantics. The remaining non-heterogeneous heads,
on the other hand, are mainly responsible for information ag-
gregation and inference, and tend to process similar semantic
information. Based on these findings, Task-KV allocates the
full KV cache budget to heterogeneous heads to preserve the
completeness of multi-perspective semantic information. For
non-heterogeneous heads, we retain only a small number of
recent tokens and attention sinks to maintain basic inference
capabilities. However, limiting storage to these tokens alone
may lead to significant information gaps. To address this, we
selectively retain a small subset of tokens with high attention
scores from intermediate positions, referred to as middle
arXiv:2501.15113v1  [cs.CL]  25 Jan 2025

<!-- page 2 -->

2
activations, which effectively capture the critical contextual
information aggregated by non-heterogeneous heads. To dy-
namically perceive semantic differences among attention heads
based on task requirements, we design a simple yet efficient
semantic separator. This separator calculates the semantic
vectors of attention heads by selecting task-relevant tokens and
distinguishes heterogeneous heads from non-heterogeneous
ones based on their distances from the semantic center.
We conduct extensive experiments across multiple bench-
mark tasks [21], [22] and different model architectures [23]–
[25] to validate the effectiveness of Task-KV . The results
demonstrate that Task-KV significantly outperforms existing
baseline methods in a variety of long-context tasks. Notably,
in scenarios requiring processing of complete context, such
as summarization and synthetic tasks, Task-KV achieves per-
formance comparable to a full KV cache while utilizing only
40% of the KV cache budget.
In summary, our contributions are as follows:
• We identify that attention heads far from the semantic
center (heterogeneous heads) have a substantial impact
on task outputs and validate this conclusion through both
theoretical analysis and experimental evidence.
• We propose the Task-KV method, which dynamically dis-
tinguishes between heterogeneous and non-heterogeneous
heads based on task-aware semantic differences among
attention heads. By allocating differentiated KV cache
budgets for different categories of attention heads, Task-
KV effectively balances inference efficiency and genera-
tion quality.
• We demonstrate the superiority of Task-KV through
comprehensive experiments on multiple benchmarks and
different model architectures and conduct ablation studies
to analyze the effectiveness of its individual components.
II. R ELATED WORK
A. Token-level KV compression methods
Optimizing the KV cache has become a critical strategy
for managing long sequences and reducing memory usage
[8], [26]. Prior research primarily focuses on selecting sig-
nificant tokens and caching only their KV states to minimize
KV cache size while maintaining model performance. For
instance, Xiao et al. [9] retains only attention sinks and recent
tokens, restoring the sliding window mechanism to handle
long contexts effectively. Li et al. [12] enhances efficiency by
compressing KV caches through the selection of significant
KV positions based on attention scores. Zhang et al. [7] em-
ploys a dynamic eviction policy that balances the retention of
recent and historically significant tokens, optimizing memory
usage while preserving essential information. Liu et al. [13]
leverages similarities in KV caches across layers, enabling
compression by caching KV states for only a subset of layers
and reconstructing the states for other layers during decoding.
Zhang et al. [10] allocates progressively reduced KV cache
budgets across layers following a pyramid structure, further
optimizing information transmission during KV cache com-
pression. However, these methods allocate the same KV cache
budget to all attention heads, which may lead to the omission
of crucial information in key attention heads. In contrast, our
method assigns differentiated KV cache budgets to different
types of attention heads, effectively reducing information loss
during KV cache compression.
B. Head-level KV compression methods
Recent research has begun to explore head-level methods
for compressing the KV cache. For example, Ge et al. [27]
applies various fixed compression strategies based on the
characteristics of the attention head, but it relies on atten-
tion weights rather than semantic information. Feng et al.
[15] optimizes the Top-k selection algorithm by identifying
important tokens from a global perspective, but it still risks
overlooking critical attention heads. Tang et al. [14], Xiao et al.
[16], and Fu et al. [17] pre-identify important attention heads
(e.g., retrieval or retrieval-reasoning heads) and allocate KV
cache budgets according to their importance. Although these
methods are highly effective, they may not fully optimize the
KV cache allocation for downstream tasks. In contrast, our
method exhibits task-awareness by recognizing the semantic
differences among attention heads and allocates KV cache
budgets based on the specific semantic requirements of each
task.
III. M OTIVATION
In this section, we first explore the semantic heterogene-
ity among attention heads and empirically demonstrate that
heterogeneous heads are critical for maintaining model’s per-
formance (Section III-A). Next, we experimentally verify
that there are significant differences in heterogeneous heads
activated by different tasks (Section III-B). Finally, we provide
a theoretical analysis, establishing that heterogeneous heads
are the key factor determining the upper bound of the LLMs’
output contribution, further highlighting their pivotal role in
LLMs inference (Section III-C).
A. Heterogeneous Heads
Although Q, K, and V in the attention mechanism all
contain semantic information, the attention mechanism itself
functions as a weighted average of V , meaning that only the
semantic information in V is propagated to the next layer
and contributes to the model’s output. Consequently, we focus
on analyzing the V in the attention heads to investigate how
the semantic differences between attention heads influence the
model’s output. Specifically, for each attention head, after
computing the attention weight matrix A, we average by
columns to obtain the weight distribution of the attention head
to the current context, and then weighted sum over V to derive
the semantic vector of the attention head. The formula for this
process is as follows:
A = Sof tmax(QK T /
√
d + M) (1)
v =
PN
i=1 A[i, :]
N · V (2)
where Q, K, V ∈ RN ×d are query states, key states, value
states respectively, M ∈ RN ×N is the mask matrix, v ∈ R1×d

<!-- page 3 -->

3
2.0
 1.5
 1.0
 0.5
 0.0
0.5
0.0
0.5
1.0
1.5
/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b
/uni0000001c
/uni00000014/uni00000013
/uni00000014/uni00000014/uni00000014/uni00000015
/uni00000014/uni00000016
/uni00000014/uni00000017/uni00000014/uni00000018
/uni00000014/uni00000019
/uni00000014/uni0000001a/uni00000014/uni0000001b/uni00000014/uni0000001c/uni00000015/uni00000013/uni00000015/uni00000014
/uni00000015/uni00000015/uni00000015/uni00000016/uni00000015/uni00000017/uni00000015/uni00000018/uni00000015/uni00000019/uni00000015/uni0000001a
/uni00000015/uni0000001b
/uni00000015/uni0000001c
/uni00000016/uni00000013
/uni00000016/uni00000014
PCA On Layer 5
2
 1
 0 1 2
1.0
0.5
0.0
0.5
1.0
1.5
2.0
2.5
/uni00000013/uni00000014
/uni00000015
/uni00000016
/uni00000017
/uni00000018
/uni00000019
/uni0000001a/uni0000001b
/uni0000001c
/uni00000014/uni00000013
/uni00000014/uni00000014
/uni00000014/uni00000015
/uni00000014/uni00000016
/uni00000014/uni00000017
/uni00000014/uni00000018
/uni00000014/uni00000019/uni00000014/uni0000001a
/uni00000014/uni0000001b
/uni00000014/uni0000001c/uni00000015/uni00000013
/uni00000015/uni00000014/uni00000015/uni00000015/uni00000015/uni00000016/uni00000015/uni00000017
/uni00000015/uni00000018/uni00000015/uni00000019
/uni00000015/uni0000001a
/uni00000015/uni0000001b
/uni00000015/uni0000001c
/uni00000016/uni00000013
/uni00000016/uni00000014
PCA On Layer 9
1
 0 1 2 3
2
1
0
1
2
3
/uni00000013
/uni00000014
/uni00000015
/uni00000016
/uni00000017
/uni00000018
/uni00000019/uni0000001a
/uni0000001b
/uni0000001c/uni00000014/uni00000013/uni00000014/uni00000014
/uni00000014/uni00000015
/uni00000014/uni00000016/uni00000014/uni00000017/uni00000014/uni00000018/uni00000014/uni00000019
/uni00000014/uni0000001a
/uni00000014/uni0000001b
/uni00000014/uni0000001c/uni00000015/uni00000013/uni00000015/uni00000014/uni00000015/uni00000015/uni00000015/uni00000016/uni00000015/uni00000017/uni00000015/uni00000018/uni00000015/uni00000019/uni00000015/uni0000001a/uni00000015/uni0000001b
/uni00000015/uni0000001c
/uni00000016/uni00000013
/uni00000016/uni00000014
PCA On Layer 13
0 2 4 6
2
0
2
4
6
/uni00000013/uni00000014
/uni00000015/uni00000016/uni00000017
/uni00000018
/uni00000019/uni0000001a
/uni0000001b
/uni0000001c/uni00000014/uni00000013/uni00000014/uni00000014/uni00000014/uni00000015/uni00000014/uni00000016/uni00000014/uni00000017
/uni00000014/uni00000018/uni00000014/uni00000019/uni00000014/uni0000001a/uni00000014/uni0000001b /uni00000014/uni0000001c/uni00000015/uni00000013/uni00000015/uni00000014/uni00000015/uni00000015/uni00000015/uni00000016
/uni00000015/uni00000017
/uni00000015/uni00000018/uni00000015/uni00000019/uni00000015/uni0000001a
/uni00000015/uni0000001b
/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014
PCA On Layer 17
0 2 4 6
3
2
1
0
1
2
3
4
/uni00000013/uni00000014/uni00000015/uni00000016
/uni00000017
/uni00000018/uni00000019/uni0000001a
/uni0000001b
/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000014/uni00000014/uni00000015
/uni00000014/uni00000016/uni00000014/uni00000017
/uni00000014/uni00000018
/uni00000014/uni00000019
/uni00000014/uni0000001a
/uni00000014/uni0000001b/uni00000014/uni0000001c
/uni00000015/uni00000013
/uni00000015/uni00000014/uni00000015/uni00000015/uni00000015/uni00000016/uni00000015/uni00000017
/uni00000015/uni00000018
/uni00000015/uni00000019
/uni00000015/uni0000001a/uni00000015/uni0000001b/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014
PCA On Layer 21
0 1 2 3 4
1
0
1
2
3
4
/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017
/uni00000018
/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013/uni00000014/uni00000014
/uni00000014/uni00000015/uni00000014/uni00000016/uni00000014/uni00000017/uni00000014/uni00000018/uni00000014/uni00000019
/uni00000014/uni0000001a
/uni00000014/uni0000001b/uni00000014/uni0000001c/uni00000015/uni00000013
/uni00000015/uni00000014
/uni00000015/uni00000015
/uni00000015/uni00000016/uni00000015/uni00000017/uni00000015/uni00000018/uni00000015/uni00000019/uni00000015/uni0000001a /uni00000015/uni0000001b/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014
PCA On Layer 25
8
 6
 4
 2
 0 2
8
6
4
2
0
2
/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c
/uni00000014/uni00000013
/uni00000014/uni00000014
/uni00000014/uni00000015
/uni00000014/uni00000016/uni00000014/uni00000017
/uni00000014/uni00000018
/uni00000014/uni00000019/uni00000014/uni0000001a/uni00000014/uni0000001b/uni00000014/uni0000001c/uni00000015/uni00000013/uni00000015/uni00000014/uni00000015/uni00000015
/uni00000015/uni00000016
/uni00000015/uni00000017
/uni00000015/uni00000018/uni00000015/uni00000019/uni00000015/uni0000001a/uni00000015/uni0000001b/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014
PCA On Layer 29
0 10 20
5
0
5
10
15
20
/uni00000013/uni00000014/uni00000015/uni00000016
/uni00000017
/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013/uni00000014/uni00000014/uni00000014/uni00000015/uni00000014/uni00000016/uni00000014/uni00000017/uni00000014/uni00000018/uni00000014/uni00000019/uni00000014/uni0000001a/uni00000014/uni0000001b/uni00000014/uni0000001c/uni00000015/uni00000013/uni00000015/uni00000014/uni00000015/uni00000015/uni00000015/uni00000016/uni00000015/uni00000017/uni00000015/uni00000018/uni00000015/uni00000019
/uni00000015/uni0000001a
/uni00000015/uni0000001b/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014
PCA On Layer 31
Fig. 2. For each specific layer, we use PCA to reduce the semantic vectors of different attention heads to two dimensions for visualization, allowing us to
observe the differences between the semantic vectors.
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3Negative Log-Likelihood
Layer 5
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000018/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000014/uni00000013/uni0000000f/uni00000003/uni00000014/uni00000016/uni0000000f/uni00000003/uni00000014/uni00000018/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000018/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni0000000f/uni00000003/uni00000015/uni00000018/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3
Layer 9
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni0000001c/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni0000000f/uni00000003/uni00000014/uni00000018/uni0000000f/uni00000003/uni00000015/uni0000001a/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni0000001c/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni0000000f/uni00000003/uni00000015/uni00000018/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3
Layer 13
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000014/uni00000016/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni0000001b/uni0000000f/uni00000003/uni00000014/uni0000001a/uni0000000f/uni00000003/uni00000016/uni00000013/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000014/uni00000016/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni0000000f/uni00000003/uni00000015/uni00000018/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3
Layer 17
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000014/uni0000001a/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni0000001b/uni0000000f/uni00000003/uni00000014/uni0000001c/uni0000000f/uni00000003/uni00000015/uni00000017/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000014/uni0000001a/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000013/uni0000000f/uni00000003/uni00000015/uni00000014/uni0000000f/uni00000003/uni00000015/uni00000015/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3Negative Log-Likelihood
Layer 21
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni00000014/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000014/uni00000016/uni0000000f/uni00000003/uni00000014/uni0000001a/uni0000000f/uni00000003/uni00000015/uni00000018/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni00000014/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000015/uni0000000f/uni00000003/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3
 Layer 25
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni00000018/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000014/uni0000001a/uni0000000f/uni00000003/uni00000015/uni00000014/uni0000000f/uni00000003/uni00000015/uni0000001b/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni00000018/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000015/uni0000000f/uni00000003/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
Layer 29
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni0000001c/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000014/uni00000013/uni0000000f/uni00000003/uni00000014/uni00000018/uni0000000f/uni00000003/uni00000015/uni00000017/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000015/uni0000001c/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000013/uni0000000f/uni00000003/uni00000015/uni00000014/uni0000000f/uni00000003/uni00000015/uni00000015/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
500 1000 1500 2000 2500 3000
Seq Length
1.7
1.8
1.9
2.0
2.1
2.2
2.3
Layer 31
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000016/uni00000014/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000017/uni0000000f/uni00000003/uni00000015/uni0000001a/uni0000000f/uni00000003/uni00000016/uni00000014/uni00000040
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni0000003e/uni00000016/uni00000014/uni00000040/uni00000042/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000003e/uni00000015/uni00000016/uni0000000f/uni00000003/uni00000015/uni00000017/uni0000000f/uni00000003/uni00000015/uni00000018/uni00000040
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
Fig. 3. Results of comparative experiments in which only heterogeneous or non-heterogeneous heads were retained in different layers.
is the semantic vector that highly summarizes the semantic
information the current attention head is focusing on.
To more intuitively observe the semantic differences be-
tween different attention heads, we apply Principal Component
Analysis (PCA) [28] to downscale the semantic vectors of the
attention heads to two dimensions and perform visualization
analysis. As shown in Fig. 2, within the semantic space of each
layer, most attention heads cluster closely together, while a
smaller subset is positioned farther from the semantic center.
We hypothesize that these attention heads, distant from the
semantic center, encode semantic information from diverse
perspectives, and they are essential for the model’s compre-
hensive understanding of task semantics. We refer to them
as heterogeneous heads. To validate this hypothesis, we select
three attention heads from each of the heterogeneous and non-
heterogeneous heads for the control experiment. Specifically,
for layer 9, we retain only the selected three heterogeneous
heads or three non-heterogeneous heads while removing the
remaining attention heads in that layer. The attention heads
in all other layers are kept unchanged. We then calculate
the negative log-likelihood (NLL) to measure the divergence
between the outputs and the standard model’s outputs. As
shown in Fig. 3, the NLL curve is closer to the standard
output when the heterogeneous heads are retained, which
better preserves the model’s original performance compared
to retaining the non-heterogeneous heads.

<!-- page 4 -->

4
/uni00000013/uni00000018/uni00000014/uni00000013/uni00000014/uni00000018/uni00000015/uni00000013/uni00000015/uni00000018/uni00000016/uni00000013
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000044/uni0000004f
/uni00000013/uni00000018/uni00000014/uni00000013/uni00000014/uni00000018/uni00000015/uni00000013/uni00000015/uni00000018/uni00000016/uni00000013
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000036/uni00000058/uni00000050/uni00000050/uni00000044/uni00000055/uni0000004c/uni0000005d/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000013/uni00000018/uni00000014/uni00000013/uni00000014/uni00000018/uni00000015/uni00000013/uni00000015/uni00000018/uni00000016/uni00000013
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000026/uni00000052/uni00000047/uni00000048/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000013/uni00000011/uni00000013/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000013/uni00000015
/uni00000013/uni00000011/uni00000013/uni00000017
/uni00000013/uni00000011/uni00000013/uni00000019
/uni00000013/uni00000011/uni00000013/uni0000001b
/uni00000013/uni00000011/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000015
/uni00000013/uni00000011/uni00000014/uni00000017
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000013/uni00000015
/uni00000013/uni00000011/uni00000013/uni00000017
/uni00000013/uni00000011/uni00000013/uni00000019
/uni00000013/uni00000011/uni00000013/uni0000001b
/uni00000013/uni00000011/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000015
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
Fig. 4. Distribution of heterogeneous heads across different tasks within the Llama-2-7B-Chat model
/uni00000013/uni00000018
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000044/uni0000004f
/uni00000013/uni00000018
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000036/uni00000058/uni00000050/uni00000050/uni00000044/uni00000055/uni0000004c/uni0000005d/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000013/uni00000018
/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013
/uni00000018
/uni00000014/uni00000013
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000026/uni00000052/uni00000047/uni00000048/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013
/uni00000013/uni00000011/uni00000014/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000013/uni0000001b
/uni00000013/uni00000011/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000015
/uni00000013/uni00000011/uni00000014/uni00000017
/uni00000013/uni00000011/uni00000014/uni00000019
/uni00000013/uni00000011/uni00000014/uni0000001b
/uni00000013/uni00000011/uni00000015/uni00000013
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013
/uni00000013/uni00000011/uni00000014/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013
/uni00000013/uni00000011/uni00000015/uni0000001a/uni00000018
/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000044/uni00000051/uni00000046/uni00000048
Fig. 5. Distribution of heterogeneous heads across different tasks within the Mistral-7B-v0.2-Instruct model
The reason behind this phenomenon is clear: the seman-
tic distinctiveness of the heterogeneous heads enhances the
expressive and generalization capabilities of the model. Re-
taining these heads can take full advantage of the diverse
semantic information extracted by the multi-head attention
(MHA) mechanism [29], which is the original purpose of
the design of the MHA. In contrast, the semantic information
of non-heterogeneous heads is more homogeneous. Retaining
only non-heterogeneous heads results in information loss,
leading to greater deviation between the outputs and those
of the standard model. However, this does not mean that non-
heterogeneous heads are useless. They mainly serve the func-
tion of information aggregation and reasoning. Their absence
can seriously affect the inference ability of the model. We have
a detailed discussion in Section VI-A.
B. Task variability among attentional heads
To further investigate the distribution characteristics of het-
erogeneous heads in different tasks, we select three attention
heads furthest from the semantic center in each layer for
visualization across different tasks. As shown in Fig. 4, 5, we
analyze the distribution characteristics of activated heteroge-
neous heads in retrieval, summarization, and code completion
tasks across different model architectures. It can be clearly
seen that the distribution of heterogeneous heads is directly
influenced by the task objectives. The semantic representa-
tion requirements of different tasks determine the distribution
characteristics of heterogeneous heads in the semantic space.
C. Theoretical analysis
The contribution of a particular attention head to the model’s
output can be interpreted as the degree of change in the
model’s output after removing that attention head. Therefore,
the contribution of the j-th attention head to the model’s output
y can be defined as:
∥∆yj∥2 =


y − yH\{hj }


2
(3)
where H denotes the set of attention heads, hj denotes the
j-th attention head, H \ {hj} denotes the removal of hj from
H, ∥·∥ denotes the L2 norm.
To simplify the analysis, we study the case of one layer of
LLM and consider only the output of the MHA. The output of
MHA is obtained by splicing the outputs of different attention
heads and then transforming them linearly, so y and yH\{hj }
can be expressed as:
y = [v1, ..., vn] Wo
=
X
1≤i≤n
viWo,i (4)
yH\{hj } =
X
1≤i≤n,i̸=j
viWo,i (5)
where vi ∈ RN ×d is the result of the calculation of the i-
th attention head, Wo ∈ Rnd×N is a linear projection layer,
Woi ∈ Rd×N is the i-th block matrix of wo, n is the number
of attention heads.

<!-- page 5 -->

5
Average
By Column
 V
Weight Distribution
Observation
Windows
Sink Tokens
Middle Activations
Recent Tokens
Evicted Tokens
Semantic Space
Full Cache Budget
Heterogeneous Heads
Non-heterogeneous Heads
Selective Cache Budget
Semantic Separator
Top-t Tokens
 KV Cache Allocation Strategy
Attention Weights
Fig. 6. Overview of Task-KV .
Therefore, ∥∆yj∥2 can be formulated as:
∥∆yj∥2 = ∥vj · Wo,j∥2 (6)
Denote vj as the sum of the mean vector ev and the offset
δj:
vj =ev + δj (7)
ev =
Pn
i=1 vi
n (8)
Substituting into Equation (6) expands it:
∥∆yj∥2 = ∥(ev + δj) · Wo,j∥2
= ∥ev · Wo,j∥2 + ∥δj · Wo,j∥2 + 2 ⟨ev · Wo,j, δj · Wo,j⟩
(9)
In practice, the elements in Wo,j are usually a finite number,
so it can be assumed to be a bounded matrix. Therefore, Wo,j
satisfies:
∥Wo,j∥ ≤ C, ∀j ∈ {1, 2, ..., n} (10)
where C is a bounded constant.
So ∥∆yj∥2 can be deflated as:
∥∆yj∥2 ≤

∥ev∥2 + ∥δj∥2 + 2 ∥ev∥ ∥δj∥

C 2 (11)
As shown in Equation (11), for different attention heads,
bothev and C remain constant. The factor that truly influences
the upper bound of the contribution to the model’s output is the
offset δj. This observation provides a theoretical explanation
for the higher contribution of heterogeneous heads to the
model’s output.
IV. T ASK -KV
Motivated by the above insights, we propose a novel method
called Task-KV , designed to dynamically allocate KV cache
budgets by leveraging task-aware semantic differences among
attention heads. As illustrated in Fig. 6, Task-KV comprises
two key components: (1) a semantic separator, which effi-
ciently and accurately distinguishes heterogeneous heads from
non-heterogeneous heads based on their semantic differences
(Section IV-A); and (2) a KV cache allocation strategy, which
allocates differentiated KV cache budgets to different types of
attention heads and determines the critical KV states to retain
for each head (Section IV-B).
A. Semantic separator
Normally, the semantic vectors of attention heads should be
computed according to Equation (1)(2). However, calculating
the complete attention weight matrix introduces significant
computational costs, which is detrimental to inference accel-
eration. Inspired by Li et al. [12], we adopt a more efficient
approach by using only a small portion of the segment at the
end of the input sequence as the observation window. This
allows us to compute a local weight matrix to approximate the
semantic information. The local weight matrix A′ ∈ RL×N is
calculated as follows:
A′ = Sof tmax(Q[−L :, :] · K T /
√
d + M ′) (12)
where L denotes the observation window size, M ′ ∈ RL×L is
the mask matrix.
Subsequently, we average the weight matrices by columns
and compute a weighted sum with the corresponding value
states to generate the semantic vectors for each attention head.

<!-- page 6 -->

6
TABLE I
DETAILS OF LONG BENCH AND LOOGLE.
Source Task Task Type Eval metric Avg Len Language Nums
LongBench
Qasper Single-Doc. QA F1 3,619 EN 200
MultiFieldQA-en Single-Doc. QA F1 4,559 EN 150
HotpotQA Multi-Doc. QA F1 9,151 EN 200
2WikiMultihopQA Multi-Doc. QA F1 4,887 EN 200
GovReport Summarization Rouge-L 8,734 EN 200
QMSum Summarization Rouge-L 10,614 EN 200
TREC Few-shot Learning Accuracy 5,177 EN 200
TriviaQA Few-shot Learning F1 8,209 EN 200
PassageCount Synthetic Task Accuracy 11,141 EN 200
PassageRetrieval-en Synthetic Task Accuracy 9,289 EN 200
LCC Code Completion Edit Sim 1,235 Python/C++/Java 200
RepoBench-P Code Completion Edit Sim 4,206 Python/Java 500
LonGLE
Computation Long-Dep. QA F1 17,001 EN 100
Multiple Information Retrieval Long-Dep. QA F1 14,808 EN 100
Long Dependency Summarization Long-Dep. Sum. Rouge-L 20,887 EN 100
However, the computational costs incurred by this process
remains unacceptable when the input sequence is long. Based
on previous studies [9], [10], [30], a small number of tokens
often account for the majority of attention scores, we select
only the top t tokens with the highest attention scores to com-
pute the semantic vectors. This approach significantly reduces
computational costs while maintaining results comparable to
those obtained using the full sequence. The specific formula
is as follows:
C =
PL
i=1 A′ [i, :]
L (13)
I = T opk(C, t) (14)
v′ = C[I, :] · V [I, :] (15)
where I denotes the index of the top t score selected from
C, v′ is the semantic vector of the current attention head.
With these two optimization steps, we significantly reduce the
computational costs of semantic vectors.
Next, we rank the attention heads based on the distance
from the semantic center, selecting a certain number of heads
from farthest to nearest as heterogeneous heads. The remaining
heads are classified as non-heterogeneous. While the hetero-
geneous heads capture diverse semantic features, they lack
the aggregated semantic information typically provided by
the non-heterogeneous heads. To address this, we select the
attention head closest to the semantic center from the non-
heterogeneous set and incorporate it into the heterogeneous
head set. This ensures that the heterogeneous heads can cover
all types of semantic information.
Moreover, as observed in Fig. 2, the number of hetero-
geneous heads decreases progressively across layers as the
model depth increases. To accommodate this trend, we select
a larger number of heterogeneous heads in lower layers and
fewer in higher layers. Specifically, we define the parameter
β to represent the proportion of heterogeneous heads in the
bottom layer and the parameter m to denote the number of
heterogeneous heads in the top layer. For intermediate layers,
the number of heterogeneous heads is determined through
linear interpolation. The number of heterogeneous heads in
the r-th layer is:
f(r) = nβ − nβ − m
R − 1 · r, r = 0, 1, ..., R − 1 (16)
where n is the number of attention heads and R is the number
of transformer layers of the model.
B. KV Cache allocation strategy
For heterogeneous heads, we allocate the full KV cache
budget to ensure the completeness of diverse semantic infor-
mation. For non-heterogeneous heads, we adopt a selective
retention strategy by preserving a small number of the most
recent tokens and attention sinks to maintain basic inference
capabilities. Additionally, we select a small set of tokens with
the highest attention scores from the intermediate portion of
the sequence. These tokens, referred to as middle activations,
aggregate critical contextual information and provide precise
guidance for model inference (a detailed analysis is presented
in Section VI-B). The number of middle activations k, is
determined by the following formula:
k = B − N · f(r)
n − f(r) − s1 − s2 (17)
where B denotes the total KV cache budget of the current
layer, N denotes the sequence length, s1 denotes the number
of sink tokens, s2 denotes the number of recent tokens.
V. E XPERIMENT
In this section, we first introduce the baselines (Section
V-A), evaluation datasets (Section V-B), and backbone LLMs
(Section V-C), followed by a detailed description of the
experimental setup for Task-KV (Section V-D). Finally, we
compare the performance of Task-KV with the baselines in the
following three aspects: (1) a comprehensive evaluation of the
model’s ability to handle various long-context tasks (Section
V-E1); (2) an assessment of its performance in long-context
retrieval and reasoning using the Reasoning-in-a-Haystack task
[17] (Section V-E2); and (3) an evaluation of the model’s
memory footprint and computational efficiency in long-context
scenarios (Section V-E3).

<!-- page 7 -->

7
TABLE II
PERFORMANCE COMPARISON ON THE LONG BENCH AND LOOGLE BENCHMARKS FOR LLAMA -2-7B-C HAT AND MISTRAL -7B- V0.2-I NSTRUCT .
Method
LongBench LoogGLE
Single-Doc Multi-Doc Summarization Few-shot Synthetic Code Avg. Computation Multi-Info Long-Dep Avg.QA QA Learning Tasks Completion Retrieval Sum.
Llama-2-7B-Chat, KV Cache Budget=100%
FullKV 27.25 28.69 22.71 73.76 6.00 54.30 35.45 10.88 11.38 2.76 8.34
Llama-2-7B-Chat, KV Cache Budget=40%
StreamingLLM 19.16 23.40 18.53 71.97 2.25 53.74 31.51 8.19 9.46 2.37 6.67
SnapKV 26.99 28.99 21.20 73.59 5.75 54.16 35.11 9.58 10.87 2.36 7.60
PyramidKV 27.70 28.24 21.29 73.67 5.50 54.09 35.08 9.84 11.04 2.41 7.76
HeadKV-R2 27.77 29.03 21.34 73.58 5.25 54.22 35.20 10.01 10.96 2.44 7.80
Task-KV 26.70 29.08 21.35 73.92 6.75 54.12 35.32 10.23 11.12 2.48 7.94
Llama-2-7B-Chat, KV Cache Budget=60%
StreamingLLM 22.71 26.43 19.62 73.80 2.75 53.75 33.17 8.39 9.31 2.44 6.71
SnapKV 27.36 28.41 22.01 73.59 6.00 54.13 35.25 10.73 10.82 2.59 8.05
PyramidKV 27.76 28.42 22.05 73.76 5.25 54.22 35.24 10.36 10.89 2.43 7.89
HeadKV-R2 27.78 28.86 22.13 73.83 5.00 54.11 35.29 10.54 10.81 2.48 7.94
Task-KV 26.78 28.91 22.27 73.71 6.25 54.20 35.35 10.78 11.02 2.59 8.13
Mistral-7B-v0.2-Instruct, KV Cache Budget=100%
FullKV 41.14 35.56 27.33 78.62 47.45 48.71 46.47 12.25 18.22 3.23 11.23
Mistral-7B-v0.2-Instruct, KV Cache Budget=40%
StreamingLLM 29.09 31.17 23.89 76.83 26.59 47.29 39.14 9.92 15.02 3.17 9.37
SnapKV 40.89 35.02 25.92 77.56 47.20 48.71 45.88 9.89 17.78 3.25 10.31
PyramidKV 40.06 34.32 25.81 78.37 47.45 48.25 45.71 10.64 17.35 3.35 10.45
HeadKV-R2 40.91 34.96 25.76 78.24 47.32 48.44 45.94 10.66 17.82 3.41 10.63
Task-KV 40.73 34.89 26.01 78.33 47.45 48.49 45.98 10.94 17.74 3.50 10.73
Mistral-7B-v0.2-Instruct, KV Cache Budget=60%
StreamingLLM 32.81 32.64 25.08 77.87 31.46 47.61 41.24 9.70 16.03 3.13 9.62
SnapKV 40.69 34.90 26.83 78.28 47.20 48.69 46.10 10.80 17.90 3.18 10.63
PyramidKV 41.11 35.45 27.39 78.28 47.12 48.63 46.33 11.56 17.64 3.10 10.77
HeadKV-R2 41.24 35.36 27.25 78.39 47.15 48.65 46.34 11.65 17.81 3.20 10.89
Task-KV 41.31 35.12 27.44 78.54 47.45 48.65 46.42 11.78 18.14 3.14 11.02
A. Baselines
We select StreamingLLM [9] as the KV cache compression
method based on attention sinks, while SnapKV [12] and
PyramidKV [10] are chosen as baselines for token-level KV
cache compression. Additionally, HeadKV [17] is used as the
baseline for head-level KV cache compression. By comparing
compression techniques across these different levels, we aim to
provide a more comprehensive evaluation of the effectiveness
of each approach.
B. Datasets
We choose two benchmarks for comprehensively evaluating
the model’s capabilities on various long context tasks: Long-
Bench [21] and LooGLE [22]. LongBench covers multiple
types of long-context tasks, including single-document QA
[31], multi-document QA [32], [33], summarization [34], [35],
few-shot learning [36], [37], synthetic tasks [38] and code
completion [39], [40]. LooGLE [22] covers a variety of long
dependency tasks, and we choose computation, multiple in-
formation retrieval, and long dependency summarization tasks
to complement LongBench. The details of LongBench and
LooGLE are shown in Table I.
C. Backbone LLMs
In this experiment, we utilize two distinct types of open-
source LLMs: Llama-2-7B-Chat [24] and Mistral-7B-v0.2-
Instruct [25], to comprehensively compare the performance
differences between Task-KV and baselines. Llama-2-7B-Chat
employs a MHA mechanism, where Q, K, V have a one-
to-one correspondence. In contrast, Mistral-7B-v0.2-Instruct
model adopts a grouped query attention (GQA) mechanism
[23], where each group of KV pairs can correspond to multiple
queries.
D. Experiment setup
We set t = 256 for computing semantic vectors. For
heterogeneous heads, we configure β = 0 .25, m = 4 for
Llama-2-7B-Chat, and β = 0.3, m = 1 for Mistral-7B-v0.2-
Instruct. For non-heterogeneous heads, we set the number of
sink tokens to 16 and recent tokens to 256 for both models.
To ensure a fair comparison, we follow Zhang et al. [10] and
set an observation window size of 32 and an average pooling
kernel size of 7 across all baselines and our method.
E. Main results
1) Long-context understanding tasks: In Table II, we
present a comprehensive evaluation of various long-context

<!-- page 8 -->

8
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000013/uni00000011/uni0000001b/uni00000013/uni00000011/uni0000001c/uni00000014/uni00000011/uni00000013
/uni0000002e/uni00000039/uni00000003/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048/uni00000003/uni00000025/uni00000058/uni00000047/uni0000004a/uni00000048/uni00000057
/uni00000014/uni00000015/uni00000011/uni00000018
/uni00000014/uni00000016/uni00000011/uni00000013
/uni00000014/uni00000016/uni00000011/uni00000018
/uni00000014/uni00000017/uni00000011/uni00000013
/uni00000014/uni00000017/uni00000011/uni00000018/uni00000024/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004a/uni00000048/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000015/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000026/uni0000004b/uni00000044/uni00000057
/uni00000036/uni00000057/uni00000055/uni00000048/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni0000002f/uni0000002f/uni00000030
/uni00000036/uni00000051/uni00000044/uni00000053/uni0000002e/uni00000039
/uni00000033/uni0000005c/uni00000055/uni00000044/uni00000050/uni0000004c/uni00000047/uni0000002e/uni00000039
/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000002e/uni00000039/uni00000010/uni00000035/uni00000015
/uni00000037/uni00000044/uni00000056/uni0000004e/uni00000010/uni0000002e/uni00000039
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000013/uni00000011/uni0000001b/uni00000013/uni00000011/uni0000001c/uni00000014/uni00000011/uni00000013
/uni0000002e/uni00000039/uni00000003/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048/uni00000003/uni00000025/uni00000058/uni00000047/uni0000004a/uni00000048/uni00000057
/uni00000016/uni00000018/uni00000011/uni00000013
/uni00000016/uni00000018/uni00000011/uni00000018
/uni00000016/uni00000019/uni00000011/uni00000013
/uni00000016/uni00000019/uni00000011/uni00000018
/uni00000016/uni0000001a/uni00000011/uni00000013
/uni00000016/uni0000001a/uni00000011/uni00000018
/uni00000016/uni0000001b/uni00000011/uni00000013/uni00000024/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004a/uni00000048/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000059/uni00000013/uni00000011/uni00000015/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000036/uni00000057/uni00000055/uni00000048/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni0000002f/uni0000002f/uni00000030
/uni00000036/uni00000051/uni00000044/uni00000053/uni0000002e/uni00000039
/uni00000033/uni0000005c/uni00000055/uni00000044/uni00000050/uni0000004c/uni00000047/uni0000002e/uni00000039
/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000002e/uni00000039/uni00000010/uni00000035/uni00000015
/uni00000037/uni00000044/uni00000056/uni0000004e/uni00000010/uni0000002e/uni00000039
Fig. 7. Experimental results on summarization tasks and synthetic tasks under different KV cache budget conditions. The final experimental results are the
average score of the two tasks
TABLE III
REASONING -IN-A-H AYSTACK TEST RESULTS WITH KV CACHE BUDGET = 50%
Method Llama-2-7B-Chat, KV Cache Budget=50% Mistral-7B-v0.2-Instruct, KV Cache Budget=50%
0k 1k 2k 4k Avg. 0k 1k 2k 4k 8k 16k 32k Avg.
FullKV 35.40 37.20 41.80 34.40 37.20 59.60 50.80 44.80 37.80 34.40 27.60 29.80 40.69
StreamingLLM 36.00 35.80 31.80 29.20 33.20 50.60 47.60 39.20 33.40 29.60 25.80 26.80 37.43
SnapKV 35.60 36.60 41.20 33.20 36.65 59.60 50.60 44.80 38.20 34.60 27.20 30.80 40.83
PyramidKV 35.60 37.40 40.40 33.20 36.65 59.40 50.60 45.00 38.60 34.60 27.20 29.60 40.71
HeadKV-R2 36.00 36.80 41.20 33.40 36.85 59.60 50.80 45.00 38.20 34.40 27.20 30.20 40.77
Task-KV 35.20 38.20 41.20 34.00 37.15 59.40 51.20 45.40 38.40 34.40 27.40 30.40 40.94
tasks from the LongBench and LooGLE benchmarks, compar-
ing the performance under two resource-constrained scenarios:
KV cache budgets of 40% and 60%. These scenarios represent
different levels of resource limitations. Experimental results
demonstrate that our method significantly outperforms all
baselines in terms of average scores on both benchmarks.
Efficient allocation of resources is critical under resource-
constrained conditions. StreamingLLM [9] retains only atten-
tion sinks and recent tokens, resulting in substantial informa-
tion loss. PyramidKV [10] and SnapKV [12] allocate identical
KV cache budgets to all attention heads, potentially overlook-
ing critical information while retaining irrelevant or redundant
content. HeadKV [17], which relies on pre-identified key
attention heads, exhibits limitations in adapting to the diversity
of long-context tasks. In contrast, our approach adaptively
allocates differentiated KV cache budgets to various types of
attention heads based on task characteristics. This enables it to
efficiently handle diverse long-context tasks even in resource-
constrained scenarios.
Notably, Task-KV exhibits superior performance in task
scenarios that require spanning the complete context (e.g.,
summarization tasks and synthetic tasks). As illustrated in
Fig. 7, we evaluate the Llama-2-7B-Chat and Mistral-7B-v0.2-
Instruct models under varying KV cache budget conditions
for summarization and synthetic tasks. In resource-constrained
settings, Task-KV significantly outperformed existing base-
lines. This superiority is primarily attributed to the fact that
Task-KV allocates a full KV cache budget for heterogeneous
heads, enabling a comprehensive understanding of global
semantic information.
2) Reasoning-in-a-Haystack: We adopt the experimental
setup proposed by Fu et al. [17] to perform the Reasoning-in-
a-Haystack evaluation. Unlike the Needle-in-a-Haystack test,
this test inserts multiple needles into the haystack, requiring
the model to retrieve and reason through them to extract the
correct answer. We perform the evaluation on LLaMA2-7B-
Chat and Mistral-7B-v0.2-Instruct under a fixed KV cache
budget of 50%. As shown in Table III, Task-KV achieves
higher average scores on both models compared to baseline
methods. This demonstrates that Task-KV exhibits efficient
retrieval and reasoning capabilities across various context
length ranges in resource-constrained scenarios.
3) Memory and latency: We evaluate the computational
efficiency of our Task-KV using the Mistral-7B-Instruct model
and set KV cache budget to 40% for all methods. To assess the
decoding latency of each method, we use 30K-length data as
input and set various generation lengths (1, 512, 1024, 2048,
4096) for comparison. As shown in the Decoding Latency
of Fig. 8, our proposed method achieves the same decoding
latency as other KV cache compression methods. Notably,
the decoding time includes both the pre-filling time and the
decoding time. Therefore, we can conclude that the pre-filling

<!-- page 9 -->

9
/uni00000013 /uni00000014/uni00000013/uni00000013/uni00000013/uni00000015/uni00000013/uni00000013/uni00000013/uni00000016/uni00000013/uni00000013/uni00000013/uni00000017/uni00000013/uni00000013/uni00000013
/uni0000002a/uni00000048/uni00000051/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000013
/uni00000014/uni00000013/uni00000013
/uni00000015/uni00000013/uni00000013
/uni00000016/uni00000013/uni00000013
/uni00000017/uni00000013/uni00000013
/uni00000018/uni00000013/uni00000013/uni00000027/uni00000048/uni00000046/uni00000052/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c
/uni00000027/uni00000048/uni00000046/uni00000052/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
/uni00000036/uni00000057/uni00000055/uni00000048/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni0000002f/uni0000002f/uni00000030
/uni00000036/uni00000051/uni00000044/uni00000053/uni0000002e/uni00000039
/uni00000033/uni0000005c/uni00000055/uni00000044/uni00000050/uni0000004c/uni00000047/uni0000002e/uni00000039
/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000002e/uni00000039/uni00000010/uni00000035/uni00000015
/uni00000037/uni00000044/uni00000056/uni0000004e/uni00000010/uni0000002e/uni00000039
/uni00000017/uni0000004e/uni0000001b/uni0000004e/uni00000014/uni00000019/uni0000004e/uni00000016/uni00000015/uni0000004e/uni00000017/uni0000001b/uni0000004e/uni00000019/uni00000017/uni0000004e/uni0000001b/uni00000013/uni0000004e/uni0000001c/uni00000019/uni0000004e/uni00000014/uni00000014/uni00000015/uni0000004e
/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000014/uni00000018
/uni00000015/uni00000013
/uni00000015/uni00000018
/uni00000016/uni00000013
/uni00000016/uni00000018
/uni00000017/uni00000013/uni00000033/uni00000048/uni00000044/uni0000004e/uni00000003/uni00000030/uni00000048/uni00000050/uni00000052/uni00000055/uni0000005c/uni00000003/uni0000000b/uni0000002a/uni00000025/uni0000000c
/uni00000033/uni00000048/uni00000044/uni0000004e/uni00000003/uni00000030/uni00000048/uni00000050/uni00000052/uni00000055/uni0000005c/uni00000003/uni00000038/uni00000056/uni00000044/uni0000004a/uni00000048
/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000002e/uni00000039
/uni00000036/uni00000057/uni00000055/uni00000048/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni0000002f/uni0000002f/uni00000030
/uni00000036/uni00000051/uni00000044/uni00000053/uni0000002e/uni00000039
/uni00000033/uni0000005c/uni00000055/uni00000044/uni00000050/uni0000004c/uni00000047/uni0000002e/uni00000039
/uni0000002b/uni00000048/uni00000044/uni00000047/uni0000002e/uni00000039/uni00000010/uni00000035/uni00000015
/uni00000037/uni00000044/uni00000056/uni0000004e/uni00000010/uni0000002e/uni00000039
Fig. 8. The Decoding Latency and Peak Memory Usage results.
/uni00000014/uni0000004e/uni00000015/uni0000004e/uni00000017/uni0000004e/uni0000001b/uni0000004e/uni00000014/uni00000019/uni0000004e/uni00000016/uni00000015/uni0000004e
/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013
/uni00000014/uni00000011/uni00000015
/uni00000014/uni00000011/uni00000017/uni00000029/uni00000014/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000033/uni00000044/uni00000056/uni00000056/uni0000004e/uni00000048/uni0000005c/uni00000003/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000044/uni0000004f
/uni0000005a/uni00000012/uni00000003/uni00000051/uni00000052/uni00000051/uni00000010/uni0000004b/uni00000048/uni00000057/uni00000048/uni00000055/uni00000052/uni0000004a/uni00000048/uni00000051/uni00000048/uni00000052/uni00000058/uni00000056/uni00000003/uni0000004b/uni00000048/uni00000044/uni00000047/uni00000056
/uni0000005a/uni00000012/uni00000052/uni00000003/uni00000051/uni00000052/uni00000051/uni00000010/uni0000004b/uni00000048/uni00000057/uni00000048/uni00000055/uni00000052/uni0000004a/uni00000048/uni00000051/uni00000048/uni00000052/uni00000058/uni00000056/uni00000003/uni0000004b/uni00000048/uni00000044/uni00000047/uni00000056
/uni00000014/uni0000004e/uni00000015/uni0000004e/uni00000017/uni0000004e/uni0000001b/uni0000004e/uni00000014/uni00000019/uni0000004e/uni00000016/uni00000015/uni0000004e
/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000015/uni00000018
/uni00000016/uni00000013
/uni00000016/uni00000018
/uni00000017/uni00000013
/uni00000017/uni00000018
/uni00000018/uni00000013/uni00000029/uni00000014/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000035/uni00000048/uni00000044/uni00000056/uni00000052/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000010/uni0000004c/uni00000051/uni00000010/uni00000044/uni00000010/uni0000002b/uni00000044/uni0000005c/uni00000056/uni00000057/uni00000044/uni00000046/uni0000004e
/uni0000005a/uni00000012/uni00000003/uni00000051/uni00000052/uni00000051/uni00000010/uni0000004b/uni00000048/uni00000057/uni00000048/uni00000055/uni00000052/uni0000004a/uni00000048/uni00000051/uni00000048/uni00000052/uni00000058/uni00000056/uni00000003/uni0000004b/uni00000048/uni00000044/uni00000047/uni00000056
/uni0000005a/uni00000012/uni00000052/uni00000003/uni00000051/uni00000052/uni00000051/uni00000010/uni0000004b/uni00000048/uni00000057/uni00000048/uni00000055/uni00000052/uni0000004a/uni00000048/uni00000051/uni00000048/uni00000052/uni00000058/uni00000056/uni00000003/uni0000004b/uni00000048/uni00000044/uni00000047/uni00000056
Fig. 9. Results of ablation of non-heterogeneous heads on Passkey Retrieval and Reasoning-in-a-Haystack experiments.
time for our method and other baselines is almost negligible.
In addition to decoding latency, we also provide the Peak
Memory Usage results, as shown in the Peak Memory Usage
of Fig. 8. Our proposed method achieves performance compa-
rable to other KV cache compression baselines, significantly
reducing memory usage compared to the Full KV cache.
VI. A BLATION STUDY
In this section, we first analyze the role of non-
heterogeneous attention heads during model inference (Section
VI-A). Next, we compare the differences between middle acti-
vations and other information compensation methods (Section
VI-B). Finally, we provide a detailed explanation of the hyper-
parameter selection strategy (Section VI-C). All experiments
are conducted using the Mistral-7B-v0.2-Instruct model under
a fixed KV cache budget of 50%.
A. Effect of non-heterogeneous heads
To validate the role of non-heterogeneous heads in infor-
mation reasoning, we conduct two sets of experiments. First,
we perform the Passkey Retrieval experiment, designed to
evaluate the model’s ability to retrieve random passkeys from
long documents, focusing solely on retrieval without involving
reasoning. As shown on the left in Fig. 9, we conduct an
ablation study on the non-heterogeneous heads of the top 12
layers of the model. The results indicate that removing the
non-heterogeneous heads has minimal impact on the model’s
retrieval performance, suggesting that these heads do not
contribute to retrieval functionality.
Next, we conduct the Reasoning-in-a-Haystack experiment,
which evaluates both retrieval and reasoning capabilities. As
illustrated on the right in Fig. 9, the model’s performance
significantly declines when the non-heterogeneous heads are
removed. Since retrieval performance is unaffected by the
absence of non-heterogeneous heads, this decline can be at-
tributed to weakened reasoning ability. These findings suggest
that non-heterogeneous heads play a critical role in enabling
information reasoning within the model.
B. Importance of middle activations
We evaluate the impact of three approaches to information
compensation on model performance. The distinctions among

<!-- page 10 -->

10
0.8 0.30.5
Evicted Token Compensation Token Middle Activations
(a) No Cache (b) Compressed Cache (c) Selective Cache
Evict all Average
compression Top select
Fig. 10. Differences between the three information compensation methods. (a) No Cache, which retains only sink tokens and recent tokens; (b) Compressed
Cache, which averages intermediate tokens into a single compensation token; and (c) Selective Cache, which represents our approach using middle activations.
/uni00000014/uni00000019/uni00000016/uni00000015/uni00000019/uni00000017/uni00000014/uni00000015/uni0000001b/uni00000015/uni00000018/uni00000019
/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048
/uni00000016/uni00000019
/uni00000016/uni0000001b
/uni00000017/uni00000013
/uni00000017/uni00000015
/uni00000017/uni00000017
/uni00000017/uni00000019
/uni00000017/uni0000001b/uni00000029/uni00000014/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000031/uni00000052/uni00000003/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048
/uni00000026/uni00000052/uni00000050/uni00000053/uni00000055/uni00000048/uni00000056/uni00000056/uni00000048/uni00000047/uni00000003/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048
/uni00000036/uni00000048/uni0000004f/uni00000048/uni00000046/uni00000057/uni0000004c/uni00000059/uni00000048/uni00000003/uni00000026/uni00000044/uni00000046/uni0000004b/uni00000048
Fig. 11. Results of three information compensation methods on the Multi-
FieldQA dataset. In the figure, Cache Size represents the number of additional
intermediate tokens introduced. To ensure fairness, we keep the number of
sink tokens and recent tokens consistent across all methods and observe how
different Cache Sizes affect model performance. Specifically, When Cache
Size = 16, the No Cache method adds 16 additional tokens to the recent
tokens. The Compressed Cache method divides the intermediate tokens into
16 groups, generating one compensation token for each group. The Selective
Cache method selects the 16 intermediate tokens with the highest attention
scores.
these methods are illustrated in Fig. 10. To ensure fairness, we
allocate the same cache size to all three methods and conduct
experiments on the MultiFieldQA [21] dataset.
As shown in Fig. 11, the results show that incorporating
middle activations effectively mitigates information loss and
improves the F1 score of the model compared to the other
two methods. The No Cache [9], [16] method discards all
intermediate information, resulting in a significant information
gap that cannot be compensated for by increasing the number
of recent tokens. The Compressed Cache method [14], while
partially compensating for information loss by compressing
intermediate information into a single token, introduces noise
and blurs critical information. In contrast, our Selective Cache
method, which incorporates middle activations, achieves high
F1 score even with a small cache size (e.g., cache size = 16).
This result indicates that non-heterogeneous heads aggregate
key information for reasoning, and this critical information is
stored in the middle activations. Retaining these key elements
is essential to fully leveraging the reasoning capabilities of
non-heterogeneous heads.
C. Hyper parameter selection
1) Analysis of top t tokens in semantic vector computation:
We use 30k-length data as input and analyze the proportion
of top t tokens in attention weights during the computation of
semantic vectors. As shown in Fig. 12, even when selecting
only a small number of tokens with the highest attention scores
(e.g., t = 16 ), these tokens still account for a substantial
proportion of the attention weights (e.g., Layer 3). The overall
trend indicates that when t = 256 , the proportion reaches
a turning point. Beyond this threshold, the rate of increase
in the weight proportion progressively slows as t increases
further. Consequently, we set t = 256 for semantic vector
computation to effectively balance computational cost with
accuracy requirements.
2) Analysis of β and m in layer-wise heterogeneous heads
allocation: When the values of β and m are set too high, the
KV cache budget for non-heterogeneous heads is compressed,
which negatively impacts the model’s reasoning capability.
Conversely, if these values are too low, the model may fail
to fully understand the task semantics. Therefore, we first fix
m = 1 , and then select different values of β from the set
{0.2, 0.25, 0.3, 0.35, 0.4 } to observe the changes in model F1
score on the MultiFieldQA [21] dataset. As shown in the left
two figures in Fig. 13, when β = 0 .25 and β = 0 .3, Task-
KV achieves the highest score on the Llama-2-7B-Chat and
Mistral-7B-v0.2-Instruct models, respectively. For the Llama
model, we fix β = 0 .25 and test the results for m values
in {1, 2, 3, 4, 5, 6 }, while for the Mistral model, we fix
β = 0 .3 and test the results for m values in {0, 1, 2 }. As
shown in the right two figures in Fig. 13, when m = 4
and m = 1 , Task-KV demonstrates optimal performance on
two models, respectively. Furthermore, Fig. 13 show that the
score remains relatively stable in the middle range, while
larger fluctuations occur at the extremes. This suggests that
as long as the number of heterogeneous heads is within a

<!-- page 11 -->

11
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni00000013/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000018/uni00000013
/uni00000013/uni00000011/uni0000001a/uni00000018
/uni00000014/uni00000011/uni00000013/uni00000013/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000003/uni00000035/uni00000044/uni00000057/uni0000004c/uni00000052
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000013
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000016
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000017
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000018
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000019
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000001a
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni00000013/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000018/uni00000013
/uni00000013/uni00000011/uni0000001a/uni00000018
/uni00000014/uni00000011/uni00000013/uni00000013/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000003/uni00000035/uni00000044/uni00000057/uni0000004c/uni00000052
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000001b
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni0000001c
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000013
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000014
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000015
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000016
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000017
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000018
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni00000013/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000018/uni00000013
/uni00000013/uni00000011/uni0000001a/uni00000018
/uni00000014/uni00000011/uni00000013/uni00000013/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000003/uni00000035/uni00000044/uni00000057/uni0000004c/uni00000052
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni00000019
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni0000001a
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni0000001b
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000014/uni0000001c
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000013
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000014
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000015
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000016
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni00000013/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013/uni00000011/uni00000018/uni00000013
/uni00000013/uni00000011/uni0000001a/uni00000018
/uni00000014/uni00000011/uni00000013/uni00000013/uni00000024/uni00000057/uni00000057/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000003/uni00000035/uni00000044/uni00000057/uni0000004c/uni00000052
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000017
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000018
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni00000019
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni0000001a
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni0000001b
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000015/uni0000001c
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000016/uni00000013
/uni00000014/uni00000019/uni00000015/uni00000018/uni00000019/uni00000018/uni00000014/uni00000015/uni00000014/uni00000013/uni00000015/uni00000017
/uni00000057
/uni0000002f/uni00000044/uni0000005c/uni00000048/uni00000055/uni00000003/uni00000016/uni00000014
/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000013/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000014/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000015/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000016/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000017/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000018/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni00000019/uni0000002b/uni00000048/uni00000044/uni00000047/uni00000003/uni0000001a
Fig. 12. Analysis of different numbers of top t tokens in attention weights
/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000016/uni00000018/uni00000013/uni00000011/uni00000017
/uni00000016/uni00000015
/uni00000016/uni00000016
/uni00000016/uni00000017
/uni00000016/uni00000018
/uni00000016/uni00000019
/uni00000016/uni0000001a
/uni00000016/uni0000001b
/uni00000016/uni0000001c/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000015/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000026/uni0000004b/uni00000044/uni00000057/uni00000003/uni0000000b/uni0000000c
/uni00000050/uni00000020/uni00000014
/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000016/uni00000018/uni00000013/uni00000011/uni00000017
/uni00000017/uni00000013
/uni00000017/uni00000015
/uni00000017/uni00000017
/uni00000017/uni00000019
/uni00000017/uni0000001b
/uni00000018/uni00000013
/uni00000018/uni00000015
/uni00000018/uni00000017/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000059/uni00000013/uni00000011/uni00000015/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000003/uni0000000b/uni0000000c
/uni00000050/uni00000020/uni00000014
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019
/uni00000050
/uni00000016/uni00000015
/uni00000016/uni00000016
/uni00000016/uni00000017
/uni00000016/uni00000018
/uni00000016/uni00000019
/uni00000016/uni0000001a
/uni00000016/uni0000001b
/uni00000016/uni0000001c/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000015/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000026/uni0000004b/uni00000044/uni00000057/uni00000003/uni0000000b/uni00000050/uni0000000c
/uni00000020/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000013 /uni00000014 /uni00000015
/uni00000050
/uni00000017/uni00000013
/uni00000017/uni00000015
/uni00000017/uni00000017
/uni00000017/uni00000019
/uni00000017/uni0000001b
/uni00000018/uni00000013
/uni00000018/uni00000015
/uni00000018/uni00000017/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001a/uni00000025/uni00000010/uni00000059/uni00000013/uni00000011/uni00000015/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000003/uni0000000b/uni00000050/uni0000000c
/uni00000020/uni00000013/uni00000011/uni00000016
Fig. 13. Ablation experiments on the Llama-2-7B-Chat and Mistral-7B-v0.2-Instruct models for β and m.
reasonable range, it has little impact on the final performance
of the model and is robust to changes in parameters. However,
when the parameters fall within extreme ranges, it may lead
to performance instability.
VII. C ONCLUSION
In this study, we theoretically and experimentally demon-
strate the significance of heterogeneous heads for model’s
outputs. Heterogeneous heads capture semantic information
from diverse perspectives, which helps enhance the model’s
representational and generalization abilities. Furthermore, our
experiments confirm that the heterogeneous heads activated by
different types of tasks exhibit significant differences. Based
on these insights, we propose a novel KV cache compression
method, called Task-KV , which dynamically allocates KV
cache budgets by leveraging task-aware semantic differences
among attention heads. Task-KV consists of two key compo-
nents: a semantic separator and a KV cache allocation strategy.
The semantic separator efficiently computes the semantic
vectors of attention heads through a two-stage optimization
process, and selects task-relevant heterogeneous and non-
heterogeneous heads based on the distance from the semantic
center. The KV cache allocation strategy assigns the full
KV cache budget to the heterogeneous heads, ensuring the
completeness of multi-perspective semantic information. For
non-heterogeneous heads, our extensive experiments confirm
their critical role in information aggregation and reasoning.
Consequently, we allocate a small number of sink tokens and
recent tokens to non-heterogeneous heads to maintain their
basic reasoning capabilities, while introducing middle activa-
tions to retain crucial aggregated information. We comprehen-
sively evaluate Task-KV across multiple benchmarks, models,
and long-context tasks. The overall results demonstrate that
our method achieves superior performance while maintaining
computational efficiency.
REFERENCES
[1] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu,
Z. Wu, T. Liu et al. , “A survey on in-context learning,” arXiv preprint
arXiv:2301.00234, 2022.
[2] Y . Qin, S. Hu, Y . Lin, W. Chen, N. Ding, G. Cui, Z. Zeng, Y . Huang,
C. Xiao, C. Han et al. , “Tool learning with foundation models,” arXiv
preprint arXiv:2304.08354, 2023.

<!-- page 12 -->

12
[3] A. R. Fabbri, I. Li, T. She, S. Li, and D. R. Radev, “Multi-news: A large-
scale multi-document summarization dataset and abstractive hierarchical
model,” arXiv preprint arXiv:1906.01749 , 2019.
[4] J.-N. Li, Q. Tu, C. Mao, Z. Yu, J.-R. Wen, and R. Yan, “Streamingdi-
alogue: Prolonged dialogue learning via long context compression with
minimal losses,” arXiv preprint arXiv:2403.08312 , 2024.
[5] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,” arXiv preprint arXiv:2312.10997 , 2023.
[6] C. Wang, Q. Long, M. Xiao, X. Cai, C. Wu, Z. Meng, X. Wang,
and Y . Zhou, “Biorag: A rag-llm framework for biological question
reasoning,” arXiv preprint arXiv:2408.01107 , 2024.
[7] Z. Zhang, Y . Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y . Tian, C. R´e, C. Barrett et al. , “H2o: Heavy-hitter oracle for efficient
generative inference of large language models,” Advances in Neural
Information Processing Systems , vol. 36, 2024.
[8] Y . Lu, X. Zhou, W. He, J. Zhao, T. Ji, T. Gui, Q. Zhang, and X. Huang,
“Longheads: Multi-head attention is secretly a long context processor,”
arXiv preprint arXiv:2402.10685 , 2024.
[9] G. Xiao, Y . Tian, B. Chen, S. Han, and M. Lewis, “Efficient streaming
language models with attention sinks,” arXiv preprint arXiv:2309.17453,
2023.
[10] Y . Zhang, B. Gao, T. Liu, K. Lu, W. Xiong, Y . Dong, B. Chang, J. Hu,
W. Xiao et al. , “Pyramidkv: Dynamic kv cache compression based
on pyramidal information funneling,” arXiv preprint arXiv:2406.02069 ,
2024.
[11] D. Yang, X. Han, Y . Gao, Y . Hu, S. Zhang, and H. Zhao, “Pyramidinfer:
Pyramid kv cache compression for high-throughput llm inference,”arXiv
preprint arXiv:2405.12532, 2024.
[12] Y . Li, Y . Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai,
P. Lewis, and D. Chen, “Snapkv: Llm knows what you are looking for
before generation,” arXiv preprint arXiv:2404.14469 , 2024.
[13] A. Liu, J. Liu, Z. Pan, Y . He, G. Haffari, and B. Zhuang, “Minicache:
Kv cache compression in depth dimension for large language models,”
arXiv preprint arXiv:2405.14366 , 2024.
[14] H. Tang, Y . Lin, J. Lin, Q. Han, S. Hong, Y . Yao, and G. Wang,
“Razorattention: Efficient kv cache compression through retrieval heads,
2024,” URL https://arxiv. org/abs/2407.15891 .
[15] Y . Feng, J. Lv, Y . Cao, X. Xie, and S. K. Zhou, “Ada-kv: Optimizing kv
cache eviction by adaptive budget allocation for efficient llm inference,”
arXiv preprint arXiv:2407.11550 , 2024.
[16] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y . Fu, and S. Han,
“Duoattention: Efficient long-context llm inference with retrieval and
streaming heads,” arXiv preprint arXiv:2410.10819 , 2024.
[17] Y . Fu, Z. Cai, A. Asi, W. Xiong, Y . Dong, and W. Xiao, “Not all
heads matter: A head-level kv cache compression method with integrated
retrieval and reasoning,” arXiv preprint arXiv:2410.19258 , 2024.
[18] W. Wu, Y . Wang, G. Xiao, H. Peng, and Y . Fu, “Retrieval
head mechanistically explains long-context factuality,” arXiv preprint
arXiv:2404.15574, 2024.
[19] W. Ma, K. Zhang, R. Lou, L. Wang, and S. V osoughi, “Contributions
of transformer attention heads in multi-and cross-lingual tasks,” arXiv
preprint arXiv:2108.08375, 2021.
[20] J. Zhang, G. De Melo, H. Xu, and K. Chen, “A closer look at transformer
attention for multilingual translation,” in Proceedings of the Eighth
Conference on Machine Translation , 2023, pp. 496–506.
[21] Y . Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu,
A. Zeng, L. Hou et al. , “Longbench: A bilingual, multitask benchmark
for long context understanding,” arXiv preprint arXiv:2308.14508, 2023.
[22] J. Li, M. Wang, Z. Zheng, and M. Zhang, “Loogle: Can long-
context language models understand long contexts?” arXiv preprint
arXiv:2311.04939, 2023.
[23] J. Ainslie, J. Lee-Thorp, M. de Jong, Y . Zemlyanskiy, F. Lebr ´on, and
S. Sanghai, “Gqa: Training generalized multi-query transformer models
from multi-head checkpoints,” arXiv preprint arXiv:2305.13245 , 2023.
[24] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y . Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al. , “Llama
2: Open foundation and fine-tuned chat models,” arXiv preprint
arXiv:2307.09288, 2023.
[25] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier et al. ,
“Mistral 7b,” arXiv preprint arXiv:2310.06825 , 2023.
[26] C. Xiao, P. Zhang, X. Han, G. Xiao, Y . Lin, Z. Zhang, Z. Liu,
S. Han, and M. Sun, “Infllm: Unveiling the intrinsic capacity of llms
for understanding extremely long sequences with training-free memory,”
arXiv preprint arXiv:2402.04617 , 2024.
[27] S. Ge, Y . Zhang, L. Liu, M. Zhang, J. Han, and J. Gao, “Model tells
you what to discard: Adaptive kv cache compression for llms,” arXiv
preprint arXiv:2310.01801, 2023.
[28] A. Ma ´ckiewicz and W. Ratajczak, “Principal components analysis (pca),”
Computers & Geosciences , vol. 19, no. 3, pp. 303–342, 1993.
[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in
neural information processing systems , vol. 30, 2017.
[30] C. Han, Q. Wang, W. Xiong, Y . Chen, H. Ji, and S. Wang, “Lm-infinite:
Simple on-the-fly length generalization for large language models,”
arXiv preprint arXiv:2308.16137 , 2023.
[31] P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner,
“A dataset of information-seeking questions and answers anchored in
research papers,” arXiv preprint arXiv:2105.03011 , 2021.
[32] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,” arXiv preprint arXiv:1809.09600 , 2018.
[33] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps,”
arXiv preprint arXiv:2011.01060 , 2020.
[34] M. Zhong, D. Yin, T. Yu, A. Zaidi, M. Mutuma, R. Jha, A. H. Awadallah,
A. Celikyilmaz, Y . Liu, X. Qiu et al. , “Qmsum: A new benchmark
for query-based multi-domain meeting summarization,” arXiv preprint
arXiv:2104.05938, 2021.
[35] L. Huang, S. Cao, N. Parulian, H. Ji, and L. Wang, “Efficient attentions
for long document summarization,” arXiv preprint arXiv:2104.02112 ,
2021.
[36] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension,”
arXiv preprint arXiv:1705.03551 , 2017.
[37] X. Li and D. Roth, “Learning question classifiers,” in COLING 2002:
The 19th International Conference on Computational Linguistics , 2002.
[38] J. Q. Li, Y . Zhao, and B. Liu, “Exploiting semantic resources for large
scale text categorization,” Journal of Intelligent Information Systems ,
vol. 39, no. 3, pp. 763–788, 2012.
[39] T. Liu, C. Xu, and J. McAuley, “Repobench: Benchmarking repository-
level code auto-completion systems,” arXiv preprint arXiv:2306.03091 ,
2023.
[40] D. Guo, C. Xu, N. Duan, J. Yin, and J. McAuley, “Longcoder: A long-
range pre-trained language model for code completion,” in International
Conference on Machine Learning . PMLR, 2023, pp. 12 098–12 107.
