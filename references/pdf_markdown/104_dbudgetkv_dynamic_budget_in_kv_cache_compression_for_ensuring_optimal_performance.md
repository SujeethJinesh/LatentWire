# references/104_dbudgetkv_dynamic_budget_in_kv_cache_compression_for_ensuring_optimal_performance.pdf

<!-- page 1 -->

Towards Threshold-Free KV Cache Pruning
Xuanfan Ni*♣ Liyan Xu*‡r Chenyang Lyuq Longyue Wangq
Mo Yur Lemao Liur Fandong Mengr Jie Zhour Piji Li♠
♣Nanjing University of Aeronautics and Astronautics
rWeChat AI, Tencent qIndependent Researcher
xuanfanni@gmail.com liyanlxu@tencent.com
Abstract
To reduce memory consumption during LLM
inference, prior works have proposed numer-
ous methods that focus on KV cache pruning
based on various criteria. While these tech-
niques often accomplish lossless memory re-
duction on many datasets, they often rely on an
under-emphasized condition: a dataset/domain-
specific budget size threshold needs to be
pre-determined to achieve the optimal perfor-
mance. However, such input-specific tuning
may be considerably limited in real-world sce-
narios, as open-domain inputs span diverse do-
mains, lengths and difficulty levels, without
clear boundaries for pre-tuning. Thus, the de-
pendence of an input-sensitive threshold can
be an inherent limitation that may cause large
degradation on arbitrary inputs.
In this work, we propose a new objective that
lifts the threshold constraints for robust KV
pruning, calling for “threshold-free” methods
that automatically adjust budget sizes while
ensuring full-cache performance. We then pro-
pose a novel method ReFreeKV as the first
solution fulfilling this objective, validated by
intensive experiments on 13 datasets of diverse
context lengths, task types, and model sizes.
1 Introduction
As current Transformer-based large language mod-
els (LLMs) employs autoregressive generation,
they produceKV Cacheto store intermediate states
during inference (Chang et al., 2023; OpenAI,
2023; Minaee et al., 2024), where they access the
cached key and value vectors of past tokens that
typically reside within GPU memory for attention
calculation (Vaswani et al., 2017; Shazeer, 2019;
Ainslie et al., 2023). Consequently, managing the
KV cache efficiently has become crucial to miti-
gate the overall memory consumption and compu-
* Equal contribution. Partial work done during Xuanfan’s
internship at WeChat AI.
‡ Project lead: Liyan Xu<liyanlxu@tencent.com>
tational overhead during LLM inference, as they
can become massive and grow proportionally when
the model size and the input length increase. For
instance, Llama3-8B model (Dubey et al., 2024)
requires 1GB of KV cache memory for 2K-token
inputs, while its 70B counterpart demands a gigan-
tic memory up to 50GB for 20K tokens.
Focusing on this research topic, numerous meth-
ods have been recently proposed to effectively re-
duce the KV footprint after LLM prefilling. Lever-
aging thesparsityof attention, retaining full KV
cache is not always necessary. Several optimiza-
tion methods, such as H2O (Zhang et al., 2023),
ScissorHands (Liu et al., 2023), SnapKV (Li et al.,
2024), FastGen (Ge et al., 2024), CAKE (Qin
et al., 2025), etc., discard the less critical cache
positions based on certain pruning criteria. Other
paradigms such as KVMerger (Wang et al., 2024)
and D2O (Wan et al., 2024) resort to merge or
compress KV vectors instead of hard-pruning for
achieving the memory reduction effects.
While these studies have confirmed on a number
of datasets that a decent portion of KV cache can
be pruned, they hinge on a fundamental yet often
under-emphasized condition: athreshold, often
beingdataset-specific, is typically involved for se-
lective tuning to achieve satisfactory results. For
instance, D2O takes a pre-defined KV cache bud-
get ratio of 20% to reach full cache performance
on LongBench (Bai et al., 2024), whereas the re-
quired budget jumps to 80% on GSM8K (Cobbe
et al., 2021); likewise, a pre-selected 1024 cache
positions is enough for CAKE on LongBench, yet
the same budget underperforms on the needle-in-
the-haystack test (Kamradt, 2023).
The existence of such threshold serves fine in
idealized settingsgiven dedicated datasets. How-
ever, its applicability may be considerably limited
inreal-world scenarios, where the inputs are inter-
mixed across different domains, lengths and diffi-
culty levels without explicit separation. As a re-
1
arXiv:2502.16886v3  [cs.CL]  6 Jan 2026

<!-- page 2 -->

MethodsGSM8K GPQA CoQAA VG.
Budget=90%
H2O 99.1% 96.9% 100.2%98.7%
SnapKV 93.3% 96.1% 99.9% 96.4%
CAKE 98.6% 93.8% 97.0% 96.5%
Budget=20%
H2O 3.2% 8.5% 96.5% 36.1%
SnapKV 1.7% 3.9% 98.6% 34.7%
CAKE 2.2% 15.4% 96.1% 37.9%
Table 1: KV pruning methods that depend on a preset
KV budget threshold can exhibit inconsistent perfor-
mance across domains (percentage relative to full-cache
scores using Llama3), making input-specific threshold
selection unavoidable for achieving optimal inference.
sult, optimal thresholds cannot be pre-determined
for diverse real-world inputs, potentially making
the system less robust and prone to significant
performance degradation. For demonstration, Ta-
ble 1 presents preliminary experiments using H2O,
SnapKV and CAKE where inputs from different
datasets are mixed to preclude dataset-specific
threshold tuning. By varying the KV cache budget
ratios, the performance across datasets is shown
inconsistent, with a trivial gap under 90% budget
but drastically escalating under 20% budget.
In this work, we propose anew objectivethat
lifts the threshold dependency in KV cache prun-
ing, thereby enabling the system to robustly han-
dle arbitrary inputs without input-specific tuning.
Specifically, our objective prioritizes two princi-
ples:1)the method operates with a universal thresh-
old insensitive to inputs, effectively rendering it
“threshold-free”; and2)the method should consis-
tently achieve performance comparable to its full-
cache counterpart, by automatically adjusting KV
cache budgets. Only after satisfying these two crite-
ria should the method pursue the best possible com-
pression ratio. As shown by our full experimental
results in Table 2, while prior methods may achieve
strong compression on certain datasets, none satis-
fies both critical aspects by our new objective.
Towards our goal, we then propose a novel
method,ReFreeKV, featuring a th Reshold-Free
KV cache pruning. Our method adopts a two-step
process with a novel metric to dynamically control
the KV cache budget. Concretely, it first ranks all
KV cache positions based on their positional im-
portance, a signal proven useful by StreamingLLM
(Xiao et al., 2024); then, it progressively retains
key-value vectors in order, and discards the remain-
ing KV cache once the stopping criterion of our
proposed input-insensitiveUni-Metric. For min-
imal overhead, ReFreeKV is implemented with
Torch parallel operators, and our analysis in Sec-
tion 3.3 shows that its latency is on par with prior
efficient KV pruning methods varying batch sizes.
The “threshold-free” aspect stems from the em-
pirical validation of ourUni-Metricdescribed in
Section 2.2, whose design is shown to be consis-
tent and insensitive to variations in input domains
and sequence lengths. Specifically, it adopts a
Universal threshold leveraging the Frobenius norm
of attention matrix, such that halting by the same
threshold empirically incurs minimal degradation
relative to the full-cache capacity. By design, Re-
FreeKV naturally yields higher compression ratios
for simpler tasks, while allocating more cache re-
sources to more complex ones.
To examine its efficacy, our experiments adopt
13 datasets varying diverse context lengths and
tasks, e.g. mathematical, commonsense reason-
ing, reading comprehension and coding, demon-
strating that ReFreeKV fulfills our objective: with-
out an input-specific threshold, the resulting infer-
ence is highly comparable or even surpasses the
full-cache performance, evaluated with multiple
LLMs of different sizes. Using Llama3-8B, it au-
tomatically allocates an average KV budget ratio
of 63.7%, while retaining 100% full-cache perfor-
mance across 13 datasets. Extensive experiments
and ablation studies empirically demonstrate that
ReFreeKV effectively accomplishes our proposed
objective. Moreover, as ReFreeKV does not rely on
specific model architectures, it has the potentials to
further combine with other orthogonal memory op-
timization paradigms such as quantization (Hooper
et al., 2024; Du et al., 2025; Li et al., 2025b).
2 Methodology
In this section, we first elaborate our motivation
and the new objective for KV cache compression
that differs from prior works. We then delineate
our proposed approach ReFreeKV, along with its
key implementation details.
2.1 Threshold-Free: A New Objective
As aforementioned, previous KV cache compres-
sion methods require a pre-selected budget thresh-
old. While these methods can perform well on
numerous datasets, it is forthcoming that the de-
pendence on a threshold can become an inherent
limitation. As the optimal threshold can vary across
different inputs, it may not be feasible for such sys-
2

<!-- page 3 -->

Layer 0
Prompt Encoding
Layer 2
Layer 31
×
...
Position-based
Token Importance Attention Matrix
Set the
Value to 0
Threshold
Compressed KV Cache
Evict the Least Important Token
Original KV Cache
Reduced Matrix
Model Inference
Layer 1
Figure 1: The overall workflow of ReFreeKV in Section 2.2. After prefilling, tokens are initially ranked based
on their positions, followed by the eviction of the least significant tokens (per layer), whose halting condition is
determined by the norm value of the reduced attention matrix. The KV cache for the remaining tokens are then
preserved to subsequent generation.
tems to pick appropriate thresholds or maintain
stable performance in practical scenarios involving
arbitrary inputs and open-domain instructions. Cer-
tain inputs may necessitate a relatively higher mem-
ory budget, such as tasks with multi-step or math-
ematical reasoning, whereas certain inputs such
as straightforward QA queries need only a small
set of KV cache. Real-world inputs, however, are
mixed and unpredictable, with no clear boundary
by difficulty or domain.
Our propose objective is exactly to remove input-
specific threshold constraints, motivating threshold-
free methods that ensure consistent performance
comparable to full-cache regardless of inputs, ob-
viating the need of tuning for optimal thresholds.
Pursuit of the best compression ratio is prioritized
only after satisfying the objective. To the best of
our knowledge, we are the first to propose an effec-
tive solution that fulfills such objective.
2.2 ReFreeKV
ReFreeKV consists of two stages implemented
with efficient parallel operators. Conceptually, it
first ranks all KV cache positions per layer and
per attention head. It then sequentially retains
key–value vectors until a stopping condition, de-
termined by our input-insensitive metric with a
universal threshold, is met; the KV cache at the
remaining positions is subsequently discarded. In
line with most prior KV compression works (Li
et al., 2025a), our method is applied once after in-
put prefilling, with the primary goal of reducing
memory consumption and an additional benefit of
improved throughput (Appendix F).
Initial RankingThe first stage ranks all KV
cache, such that the beginning of the sequence may
likely contain more critical information than its lat-
ter parts, which forms the basis for downstream
sequential eviction.
Building on the positional bias reported in
StreamingLLM (Xiao et al., 2024), we also ob-
serve that positions at the beginning and end of
the input generally play a more critical role in
subsequent generation. We exploit this property
in the first stage, and rank KV cache by token
positions as follows. Denote a LLM input se-
quence as X = {x1,x 2, . . . ,x n}, where each Trans-
formers layer originally consists of n positions
of KV vectors per attention head. The initial
ranking takes the first m positions and reversely
takes the remaining n−m positions, denoted by
bX = {x1,x 2, . . . ,x m,x n,x n−1, . . . ,x m+1}. Note that
m is a chosen hyperparameter that works well re-
gardless of specific input sequences.
Our early experiments also conduct exploration
on other ranking strategies, reported in Section 3.4.
By empirical validation, the position-based rank-
ing is shown not only effective but also particularly
advantageous in terms of computational overhead,
constituting the first stage for ReFreeKV . However,
pruning solely by positions does not fulfill our ob-
jective, as our experiments in Appendix G show
that StreamingLLM can exhibit inconsistent perfor-
mance varying budget sizes, highlighting the need
for a more robust eviction strategy.
Eviction byUni-MetricWith the initial ranking
on KV cache, ReFreeKV then sequentially retains
KV vectors, and halts upon the stopping condition
3

<!-- page 4 -->

by an input-insensitiveUni-Metric, after which the
remaining cache is effectively evicted.
The design ofUni-Metricis then at the core of
this process, which requires to signal the degra-
dation level after removing KV cache of certain
positions. we propose a metric that empirically cor-
relates well with the performance change when dis-
carding a position, which couldserve as a bridge
to ensure a minimal degradation upon the full
cache performance. Inspired by Devoto et al.
(2024), we utilize the Frobenius norm (L2 norm)
of the attention matrix A∈R n×n as theUni-Metric,
denoted as ||A||F =
qPn
i=1
Pn
j=1 |Ai,j |2. For each
position i in the initially ranked sequence, we com-
pare the Frobenius norm of the original attention
matrix, ||A||F, with that of a curated attention ma-
trix, || eAi||F, in which scores to all positions >i in
the ranked sequence are masked out, replicating the
effect of discarding all KV cache beyond positioni.
Once the norm difference reaches a thresholdT at
position iprune, the entire process terminates, retain-
ing only the KV cache up to iprune and discarding
the remainder, denoted as:
iprune =argmin n
j=1(1− || eA j||F
||A||F
<T) (1)
The Universal ThresholdTo fulfill our objec-
tive, the threshold T should ensure near lossless
pruning invariant to inputs. Upon empirical search,
we identify T =1% could serve well for this pur-
pose. Figure 2 illustrates how performance varies
with changes in the norm across different domains.
Preliminary studies indicate that when T< 1%,
performance remains comparable to the full-cache
version robustly. We selectT =1% to balance min-
imal degradation with maximal cache eviction. The
efficacy ofUni-Metricand its universal threshold
is validated at full scale in the main experiments
presented in Section 3.2 and Figure 3.
Reducing OverheadAs the input sequence
length n increases, the time and space overhead
for norm calculation on the attention matrix grows
by O(n2). To reduce the computational scale, we
seek to use an approximate norm calculation by
O(n). Instead of using the full attention A, we re-
duce A by taking the average of its last k rows to a
single attention vector A′ ∈R 1×n. The score si for
a positioni∈[1,n] inA ′ is denoted as:
si =
Pn
j=k Ai,j
Pn
j=k 1{Ai,j ,0}
(2)
0.1 1.0 2.0 3.0 4.0 5.0
Threshold (%)
0.4
0.5
0.6
0.7
0.8
0.9
1.0Normalized Performance
Model Performance vs. Threshold
Llama3-8B (GSM8K)
Llama3-8B (NQA)
Mistral-7B (GSM8K)
Mistral-7B (NQA)
Qwen2.5-7B (GSM8K)
Qwen2.5-7B (NQA)
Figure 2: Performance trends of Llama3-8B, Mistral-7B,
and Qwen2.5-7B across varyingUni-Metricthresholds.
The x-axis represents the threshold percentage (0.1%
to 5%), and the y-axis denotes the performance score
normalized by the full-cache performance.
LogisticsThe rationale behind ReFreeKV ’s two-
stage process is that finding the optimal pruning
positions in a sequence is acombinatorialproblem.
By combining ranking with a linear search, the
problem becomes tractable. The entire procedure
is further optimized through parallel operations, as
described as follows.
2.3 Implementation Details
As ReFreeKV is conceptually a sequential search
process, the design of ReFreeKV allows efficient
implementation by PyTorch’s operators, such that
the stopping positions of all layers are directly iden-
tified in parallel without explicit looping operations.
Latency and throughput analyses in Section 3.3 and
Appendix F demonstrate that ReFreeKV matches
prior popular KV pruning approaches, with negli-
gible latency overhead and improved throughput
compared to the baselines.
Specifically, the pruning position iprune can be
determined directly by the combination of Torch
cumulative-sum and where operators. We com-
pute the cumulative square-sum of each element in
A′
rank, obtained after the initial ranking onA ′:
|| eA′
i||F =A ′
cumsum[i]=
vut iX
k=1
(A′
rank[k])2 (3)
such that A′
cumsum[i] represents the Frobenius Norm
of the attention matrix after removing all cache
to the right of position i. We then divide || eA′
i||F
by ||A′||F to determine the norm difference as in
Eq (1). The torch where operation allows us to
directly identify the leftmost position that satisfies
the 1% universal threshold, ultimately yielding the
set of positions for which the KV cache is called to
4

<!-- page 5 -->

retain. The overall pruning process of ReFreeKV
is further presented in Algorithm 1.
Retaining Bottom LayersIn our experiments,
we discovered that the LLM’s first two Transform-
ers have a relatively uniform attention distribution,
usually requiring to retain most of the cache po-
sitions. For simplicity and robustness, we always
retain all KV cache of the first two layers in our
implementation. More studies on bottom layers are
further provided in Appendix B.
Running at ScaleSupporting batch sizes > 1,
ReFreeKV is able to perform the entire pruning
process for each sample in parallel. To achieve
this, we pad the shorter cache segments and update
the attention masks accordingly, allowing the LLM
to ignore the padded KV positions. The padding
operation has a negligible impact on overall per-
formance. Meanwhile, in popular LLM inference
engines, e.g. vLLM (Kwon et al., 2023), it is pos-
sible to allocate separate KV cache size for each
sample, which aligns well with ReFreeKV . Integra-
tion with LLM inference engines such as vLLM is
under planning for future development.
3 Experiments
3.1 Experimental Settings
BackbonesOur experiments are conducted with
three LLM families of different model sizes:
Llama3-Instruct with size of 8B/70B, Mistral-7B-
Instruct-V0.3, and Qwen2.5-Instruct with size of
7B/32B/72B. We implement our ReFreeKV upon
the released codebase of SnapKV*. For the reduced
attention matrix A′, we set k =1 in practice (abla-
tion provided in Section 3.4), and set m =4 for the
initial cache ranking stage.
DatasetsFor comprehensive evaluation, we eval-
uate ReFreeKV on datasets of both short and long
context length of different domains, including math-
ematics, science, and commonsense reasoning on
GSM8K (Cobbe et al., 2021), GPQA (Rein et al.,
2023), TheoremQA (Chen et al., 2023), Truth-
fulQA (Lin et al., 2022), and CoQA (Reddy et al.,
2019). We also include tasks from Longbench (Bai
et al., 2024) with 8 datasets spanning document
comprehension, summarization and coding. Ap-
pendix D provides a detailed description and statis-
tics for all13 datasets, along with how they are
utilized in our experiments.
*https://github.com/FasterDecoding/SnapKV
Evaluation ProtocolOur evaluation primarily
assesses ReFreeKV ’s ability to preserve full-cache
performance with its automatic pruning budgets.
Accordingly, we compare ReFreeKV against its
full-cache scores and also report the average com-
pression ratio for each dataset.
Additionally, we compare with five prior KV
cache pruning methods with varied budget sizes, in-
cluding: Heavy Hitter Oracle (H2O) (Zhang et al.,
2023), StreamingLLM (SLM) (Xiao et al., 2024),
SnapKV(Li et al., 2024),PyramidKV(Cai et al.,
2024) andCAKE(Qin et al., 2025). By evaluat-
ing performance consistency under fixed KV cache
budgets of 90%, 50%, and 20%, we further illus-
trate the limitations arising from the input-specific
threshold dependence.
Lastly, we also include a concurrent workTwi-
light(Lin et al., 2025), which does not rely on a
budget threshold but employs a top-p-inspired met-
ric for adaptive token selection. It is worth noting
that the p value remains a hyperparameter to be
tuned across models and inputs.
3.2 Main Results
The main experimental results are shown in Table 2.
Due to space limitations, the results for the 50%
budget size are placed in Appendix C. For compar-
ison with Twilight, we separately report the results
in Table 4 due to its different hyperparameter type.
Based on Table 2, we can draw the following ob-
servations.
•ReFreeKV is able to fulfill our objective, ca-
pable of performing near-lossless dynamic com-
pression across different models, varying input
lengths, and diverse task types. Interestingly, with
Llama3-8B and Qwen2.5-7B, ReFreeKV even sur-
passes the full-cache performance by 0.12% and
2.63% respectively, utilizing an average of 63.68%
and 76.02% KV cache. With Mistral, ReFreeKV
also manages to achieve near 15% compression
with a relatively small 1.5% performance reduction.
These results indicate that our proposed method can
be suited for real-world scenarios with no bother
by input-specific budget thresholds.
•In stark contrast, previous methods achieve a
consistent full-cache performance only when man-
ually determined a high budget ratio, e.g. 90%.
However, when the budget is reduced, e.g. 50%
and 20%, the degradation can become severe on
certain datasets, distinct from ReFreeKV that auto-
matically adjusts the pruning to always prioritize
full-cache performance. Theoretically, one could
5

<!-- page 6 -->

Methods
Math&Science CR Single-Doc QA Multi-Doc QA Summarization FSL Code
Avg.
GSM8KGPQATheoQAThQACoQANrtvQAQasper2WkMQAMusiqueQMSumM-NewsTriviaQALcc
Llama3-8B-Instruct
Full 75.28 29.02 21.29 25.5952.74 24.06 43.91 35.33 14.77 22.27 27.37 70.26 19.16 100%Oursk=1 76.50 30.13 23.03 26.0952.86 23.4437.38 36.21 15.96 21.95 27.88 64.24 19.31+0.12%Budget 93.2%86.7%92.8%78.7%81.5%48.7%46.4%45.8% 43.7%15.0%76.4%41.0%78.0%63.68%H2O0.9 74.60 28.13 21.69 25.41 52.8324.5541.73 33.95 15.2322.5127.64 69.77 19.13 -0.39%SLM0.9 72.78 28.79 20.75 25.15 52.83 23.75 43.63 32.68 15.86 22.48 27.21 69.9719.66-0.59%SnapKV0.9 70.20 27.90 20.08 25.38 52.72 24.0443.9333.92 15.56 22.49 27.6570.5819.05 -1.12%PyramidKV0.9 75.44 28.7924.4624.72 52.16 23.64 43.39 32.10 14.11 22.22 23.70 70.48 19.05 -1.59%CAKE0.9 74.23 27.23 23.16 22.87 51.18 21.32 43.78 34.65 14.20 21.19 22.33 69.22 19.22 -4.17%H2O0.2 2.43 2.46 6.56 20.67 50.91 23.54 42.00 32.50 15.67 22.16 23.68 69.88 19.13 -23.33%SLM0.2 3.21 3.35 8.82 18.02 40.11 20.12 36.45 29.91 15.01 21.11 24.68 66.17 18.69 -28.21%SnapKV0.2 1.29 1.12 5.89 20.36 52.00 23.10 42.78 32.98 14.01 22.45 24.14 69.63 19.34 -22.89%PyramidKV0.2 1.36 1.79 4.82 20.24 51.71 24.31 41.90 33.81 12.56 21.11 27.01 68.43 18.19 -25.33%CAKE0.2 1.67 4.46 7.93 21.17 50.66 22.01 42.18 33.02 14.93 20.15 24.48 69.92 19.10 -23.47%
Mistral-7B-Instruct
Full 33.36 29.24 6.83 20.8239.68 28.74 37.80 33.87 22.88 22.19 22.94 86.87 16.08 100%Oursk=1 31.54 29.02 6.71 20.8139.80 27.20 38.93 33.46 22.15 22.39 22.67 86.81 15.33-1.50%Budget 89.6%90.5%84.2%92.3%84.1%78.0%97.9%86.7% 74.4%84.3%89.1%87.2%89.4%86.75%H2O0.9 19.18 24.11 6.96 20.61 39.74 27.36 38.38 35.23 23.24 22.13 23.41 86.46 16.01 -4.29%SLM0.9 31.16 27.68 5.89 19.88 39.15 27.26 37.66 35.06 21.94 22.31 21.73 86.63 13.67 -4.44%SnapKV0.9 27.60 25.677.1020.23 39.76 27.43 38.27 35.66 22.99 22.16 23.31 86.46 16.07 -1.90%PyramidKV0.9 22.74 26.75 5.35 20.34 38.4327.6038.2737.62 25.2822.1824.6485.5316.37-3.15%CAKE0.9 25.02 24.10 6.12 20.21 38.82 27.49 39.19 32.96 23.55 22.22 24.51 86.36 15.80 -4.14%H2O0.2 1.52 4.46 4.02 18.45 35.55 27.28 33.45 34.36 23.21 21.80 20.93 85.99 14.82 -23.55%SLM0.2 1.21 1.79 0.54 18.35 25.45 24.89 28.39 29.53 16.74 20.60 16.75 80.12 14.27 -35.58%SnapKV0.2 1.44 0.20 3.35 17.81 37.44 26.76 36.02 33.90 23.05 22.13 20.88 84.27 16.20 -24.28%PyramidKV0.2 1.14 0.58 1.40 19.24 31.20 26.19 34.53 37.25 24.55 22.24 22.18 86.58 15.76 -23.75%CAKE0.2 4.55 1.03 2.09 18.21 30.07 26.25 36.03 32.39 22.79 21.53 23.96 86.61 15.00 -24.05%
Qwen2.5-7B-Instruct
Full 88.02 31.70 29.85 24.5561.43 20.81 43.17 47.15 30.70 23.64 24.24 87.64 2.44 100%Oursk=1 88.02 31.25 30.39 24.4860.01 20.66 42.74 47.20 29.56 22.64 24.04 87.65 3.58 +2.63%Budget 90.7%88.8%86.5%69.2%84.1%65.1%69.2%73.3% 64.4%70.6%85.4%56.6%84.4%76.02%H2O0.9 83.47 26.56 30.25 24.4061.2920.85 43.2647.90 30.7323.4224.3587.57 2.45 -1.46%SLM0.9 88.9330.80 28.25 24.30 60.91 19.86 42.54 46.02 28.75 23.06 23.42 87.03 3.05 -0.41%SnapKV0.9 80.14 30.13 28.11 24.30 61.26 20.9043.3147.81 30.6923.9024.29 87.57 2.55 -1.01%PyramidKV0.9 84.15 27.68 26.78 23.51 50.34 27.60 42.27 37.62 25.28 22.18 21.65 87.53 3.37 -2.74%CAKE0.9 85.67 30.23 29.28 23.52 58.2729.2733.73 46.60 30.32 23.91 23.89 90.14 2.41 -0.06%H2O0.2 5.91 3.57 17.40 21.70 55.39 21.41 40.30 45.84 29.72 23.04 20.84 87.99 2.15 -13.89%SLM0.2 1.06 4.24 4.82 23.85 40.92 18.17 29.22 39.27 21.71 20.32 18.94 77.80 1.46 -31.66%SnapKV0.2 2.58 7.37 15.13 21.52 57.06 21.40 40.93 46.96 29.87 23.58 20.69 87.10 2.44 -13.26%PyramidKV0.2 9.70 8.57 9.60 26.42 51.19 26.42 39.48 39.54 25.73 21.02 21.65 85.31 1.76 -23.47%CAKE0.2 7.58 9.11 5.56 23.66 49.19 30.20 43.04 46.35 29.00 23.33 22.94 85.45 2.89 -16.98%
Table 2:Performance of ReFreeKV and its comparison with five KV pruning methods on 13 datasets.Boldnumbers indicate
the best resultsaside from full-cache.Italicsrepresent the real budget utilized by ReFreeKV. The correspondence between
abbreviations and their full names of datasets can be found in Appendix.Avg.calculates the mean ratio of the model’s
performance using different KV cache compression methods to its performance with the full cache. All average results (except
for budget) are adjusted by subtracting 1 to provide a more intuitive understanding of the effectiveness of different methods.
tune the budget for each dataset that achieves mini-
mal degradation, but this is generally infeasible for
real-world open-domain instructions.
•ReFreeKV naturally reflects thedifficultyof
the generation task. As in Table 2, the dynamic bud-
get ratio is high on Math&Science datasets (over
90%), while much lower on QA or Summarization
datasets (as low as 15%). This observation is in
line with our intuition, where inference on concise
but hard tasks, such as math problems, requires
more context and more precise calculation, result-
ing in higher budget allocation. From this aspect,
our method design serves beyond for memory effi-
ciency, but could be potentially leveraged for input
analysis in a broader scope.
•Besides the dynamic compression, ReFreeKV
also outperforms the three 90%-budget baselines,
while itself uses less than 90% budget. Though,
It is worth noting that the goal of this work is not
to propose yet another KV cache pruning method
that targets the best possible compression ratio un-
der specific conditions. Instead, we seek to lift
the threshold constraints and advocate for robust
KV pruning that generalizes to arbitrary inputs.
Through our proposed objective, we hope to fur-
ther advance research in this direction.
As separately reported in Table 4, we compare
ReFreeKV with Twilight’s reported performance
across the GSM8K, NarrativeQA, 2WikiMQA, and
Musique datasets. Regarding the hyperparameter
for Twilight, we adopt their reported optimal p
value: p =0 .95 for Llama3-8B and p =0 .85
for Mistral. As shown in the results, ReFreeKV
achieves performance comparable to Twilight on
6

<!-- page 7 -->

GSM8K CoQA NarrativeQA Musique QMSum TriviaQA Avg.
Overall Prune Overall Prune Overall Prune Overall Prune Overall Prune Overall Prune Overall Prune
Llama3-8B-Instruct
Full 4.693 —0.289— 3.441 — 3.659 — 5.458 — 2.717 — 3.376 —
Oursk=1 4.6380.034 0.331 0.0333.159 0.255 3.6580.2645.3300.241 2.684 0.2113.3000.173
H2O0.5 4.686 0.020 0.2900.0183.243 0.264 3.679 0.266 5.398 0.2492.6120.237 3.318 0.176
SLM0.5 6.0710.0060.301 0.030 3.230 0.257 3.7840.2595.4540.2342.6280.2083.5780.166
SnapKV0.5 5.874 0.021 0.285 0.019 3.307 0.265 3.721 0.266 5.519 0.264 2.650 0.265 3.559 0.183
Llama3-70B-Instruct
Full 17.504 — 1.167 — 6.285 —6.565— 13.993 — 5.302 — 8.469 —
Oursk=1 15.9750.154 1.253 0.2906.0422.345 7.412 2.35213.8353.0285.2241.6028.2901.623
H2O0.5 19.0620.114 1.0590.240 6.788 2.298 7.082 2.307 14.360 3.712 5.566 1.308 8.986 1.663
SLM0.5 22.079 0.150 1.1380.2296.8042.0697.1072.05314.734 3.700 5.576 1.367 9.573 1.595
SnapKV0.5 16.517 0.117 1.116 0.241 6.795 2.474 7.085 2.468 13.8462.7935.5621.0838.4871.529
Table 3: The average inference time and pruning time of Llama3 with size of 8B/70B across six datasets, measured
in seconds, with lower values indicating better performance.
Methods GSM8K NQA Qasper 2WQA Musique
Llama3-8B 75.28 24.06 43.91 35.33 14.77
+Oursk=1 76.50 23.4437.3836.21 15.96
+Twip=0.95 76.04 23.37 43.08 36.18 14.92
Mistral-7B 33.36 28.74 37.80 33.87 22.88
+Oursk=1 31.54 27.20 38.93 33.4622.15
+Twip=0.85 30.33 27.17 38.90 33.23 22.92
Table 4: Performance comparison between ReFreeKV
and Twilight across five datasets using Llama3-8B-
Instruct and Mistral-7B-V0.3.Boldnumbers indicate
the best results aside from full-cache.
both models. Notably, both methods frequently sur-
pass the full-cache baseline across these datasets.
Though, as Twilight requires different optimal p
values for each model, tuning may still be neces-
sary. Nonetheless, both ReFreeKV and Twilight
share the same principle of adaptive pruning, which
we believe can further advance research in this area.
3.3 Efficiency Analysis
In this section, we conduct a quantitative study on
the latency of ReFreeKV and the overall impact on
the inference time, apart from the space reduction
from KV cache pruning. We compare the time
usage on six datasets with Llama3-8B/70B using
ReFreeKV, along with three baselines with the 50%
budget setting. Table 3 reports the time latency
needed to apply pruning (Prune), as well as the
average generation time after the prefilling stage of
each sample (Overall).
From the results, the latency by ReFreeKV is
on par with prior pruning methods. Notably, Re-
FreeKV achieves the best performance in 8 out of
12 comparisons of overall generation time, suggest-
ing that the total generation speed of ReFreeKV
is advantageous. The trend also remains consis-
tent across different model sizes, underscoring the
efficiency aspect of ReFreeKV .
For batch process, we further conduct analysis
on throughput with different batch sizes. The full
details of which are provided in Appendix F. The
results confirm that ReFreeKV improves through-
put upon the naive generation by 10-20%, espe-
cially it retains performance edges and robustness
with increasing batch sizes.
3.4 Ablation Studies
We conduct ablation studies to investigate the im-
pact of various configurations of ReFreeKV . We
use Llama-3-8B-Instruct and perform experiments
across five datasets, with results presented in Ta-
ble 5. Appendix E provides additional results and
analysis from our ablation studies.
Attention Matrix ReductionThe reduced atten-
tion matrix A′ in Section 2.2 aggregates attention
scores from the last k rows. The upper part of Ta-
ble 5 illustrates the model’s performance and the
actual budget when setting k =1, 1% n, 5%n, and
10%n (n being the number of input tokens). It is
clear that setting k as 1 achieves a significantly
reduced budget, thus a higher compression ratio,
with almost no change in performance compared
to 1%n and 5%n. On the other hand, while 10%n
can compress more KV cache, it fails to maintain
performance (for instance, on NarrativeQA, the for-
mer achieves a performance of 9.74 using 32% of
the budget, whereas the latter scores 21.44 using
48.7% of the budget). What’s even better is that
since k =1 requires the least amount of computa-
tion, relying solely on the scores from the last row,
the complexity of obtaining A′ becomes O(1), inde-
7

<!-- page 8 -->

0.1% 1% 2% 3% 4% 5% 6% 7% 8% 9% 10%
Pruning Threshold
20
40
60
80
100
120
Performance Retention (%)
(a) Relative Performance
0.1% 1% 2% 3% 4% 5% 6% 7% 8% 9% 10%
Pruning Threshold
0
20
40
60
80
100
Budget Consumption (%)
(b) Budget Consumption
GSM8K GPQA NQA Musique QMSum
Figure 3:Performance vs. Efficiency Trade-off.(a) Performance retention across five datasets (solid lines). (b)
Computational budget consumption (dashed lines) relative to the dense baseline. The shared legend indicates the
datasets. Results show that setting the universal threshold to 1% could well balance between performance and
memory budget, as it maintains robust full-cache performance while substantially reducing KV cache.
Methods GSM8K CoQA NQA Musique QMSum
Full 75.28 52.74 24.06 14.77 22.27
Post=1% Performance Using Different k
k=1 76.50 52.86 23.44 15.96 21.95
Budget 93.2% 81.5% 48.7% 43.7% 15.0%
k=1%n76.19 52.87 23.17 15.40 21.94
Budget 95.1% 94.8% 77.3% 77.2% 59.5%
k=5%n75.59 52.75 21.62 13.71 21.43
Budget 98.6% 96.2% 45.1% 30.0% 33.1%
k=10%n76.72 52.85 9.74 13.67 21.11
Budget 96.3% 96.8% 32.0% 19.7%29.3%
k=1Performance with Different Ranking Method
Attnt=1% 2.35 43.68 13.37 9.58 17.62
Budget 20.0% 9.7% 6.4% 6.4% 6.4%
Attnt=0.01% 54.66 51.58 17.45 14.79 20.90
Budget 61.7% 28.6% 6.7% 7.9% 7.1%
Table 5: Performance comparison for ablation studies in
Section 3.4, with differentk values for attention matrix
reduction and initial ranking alternatives.
pendent of the sequence length. The advantages of
both high efficacy and low overhead makek =1 a
solid design choice for calculating the norm metric.
Initial Ranking StrategiesApart from the
position-based ranking described in Section 2.2,
we also investigate other alternatives, such as rank-
ing by each token’s average attention score received
from other tokens, similar to previous approaches
such as H2O. The results are shown in the bottom
of Table 5: under the same experimental settings,
the attention-based ranking struggles to maintain a
robust final performance. Empirical validation sup-
ports that position ranking is an appealing choice,
offering both superior efficacy and efficiency.
The Universal ThresholdAs we adopt the uni-
versal threshold as 1% for the attention norm dif-
ference, we comprehensively study the effects of
smaller or larger thresholds, as shown in Figure 3.
Intuitively, a larger threshold allows for a higher
compression ratio but potentially more degradation,
while a smaller threshold does the opposite. It is
clear that when the threshold is set to a smaller
value, such as 0.1%, the average budget increases
as expected, yet the model’s performance sees little
improvement. Conversely, when the threshold is set
to 10%, it prunes more KV cache, but the model’s
performance significantly deteriorates. Thus, we
deem 1% as a reasonable and robust threshold for
universal uses across models and tasks.
GeneralizationWe further conduct experiments
to examine whether our design and hyperparam-
eters can be generalized to LLMs of more sizes.
On five datasets in Appendix Table 12, ReFreeKV
with Llama3-70B and Qwen2.5-32B/72B demon-
strates consistent near full-cache performance with
the same ReFreeKV setting. Notably, the averaged
compression ratio increases on datasets of longer
context, achieving nearly a 50% compression.
4 Conclusion
In this study, we introduce a new KV cache com-
pression objective that lifts the threshold depen-
dency, so to achieve input-insensitive pruning for
robust inference performance. Towards this objec-
tive, we propose a novel method, termed ReFreeKV ,
which employs a straightforward yet effective two-
stage KV cache pruning process. Comprehensive
experiments conducted across diverse datasets, en-
compassing a variety of tasks and context lengths,
demonstrate that ReFreeKV achieves nearly loss-
less compression without input-specific tuning or
thresholds, while with notable memory reduction.
8

<!-- page 9 -->

Limitations
The main limitation of ReFreeKV stems from the
gap between its current compression budget and
the true optimal budget. This can be seen in Ta-
ble 2 that for certain scenarios, e.g. QMSum
with Mistral-7B, ReFreeKV reaches 84.3% budget
while 50% budget is also viable with no perfor-
mance degradation. We regard this gap as room
for improvement in future work, facilitating more
aggressive KV cache compression while ensuring
the full-cache objective.
Another limitation of ReFreeKV is that though
it demonstrates almost lossless compression, we
rely on empirical experiments, and there is no hard
guarantee on the degree of degradation. In Ta-
ble 2, while both Llama3-8B and Qwen2.5-7B even
surpass the full-cache performance, ReFreeKV
with Mistral-7B shows trivial degradation of 1.5%.
More robust methods could be developed as future
work to further strengthen the robustness.
References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
2023. GQA: training generalized multi-query trans-
former models from multi-head checkpoints. InPro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, EMNLP 2023,
Singapore, December 6-10, 2023, pages 4895–4901.
Association for Computational Linguistics.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. Longbench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (V olume 1:
Long Papers), ACL 2024, Bangkok, Thailand, Au-
gust 11-16, 2024, pages 3119–3137. Association for
Computational Linguistics.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu
Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao
Chang, Junjie Hu, and Wen Xiao. 2024. Pyramidkv:
Dynamic KV cache compression based on pyramidal
information funneling.CoRR, abs/2406.02069.
Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-
Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi
Huang, Luis Ceze, Mohamed S. Abdelfattah, and Kai-
Chiang Wu. 2025. Palu: Kv-cache compression with
low-rank projection. InThe Thirteenth International
Conference on Learning Representations, ICLR 2025,
Singapore, April 24-28, 2025. OpenReview.net.
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu,
Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi,
Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang,
Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie.
2023. A survey on evaluation of large language mod-
els.CoRR, abs/2307.03109.
Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan,
Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony
Xia. 2023. Theoremqa: A theorem-driven question
answering dataset. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, EMNLP 2023, Singapore, December 6-
10, 2023, pages 7889–7901. Association for Compu-
tational Linguistics.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.CoRR, abs/2110.14168.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, NAACL-HLT 2021, On-
line, June 6-11, 2021, pages 4599–4610. Association
for Computational Linguistics.
DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingx-
uan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng,
Chong Ruan, Damai Dai, Daya Guo, Dejian Yang,
Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli
Luo, Guangbo Hao, Guanting Chen, and 83 others.
2024. Deepseek-v2: A strong, economical, and ef-
ficient mixture-of-experts language model.CoRR,
abs/2405.04434.
Alessio Devoto, Yu Zhao, Simone Scardapane, and
Pasquale Minervini. 2024. A simple and effective
l2 norm-based strategy for KV cache compression.
CoRR, abs/2406.11430.
Dayou Du, Shijie Cao, Jianyi Cheng, Ting Cao, and
Mao Yang. 2025. Bitdecoding: Unlocking tensor
cores for long-context llms decoding with low-bit
KV cache.CoRR, abs/2503.18773.
Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei
Duan, Xingcheng Zhang, and Dahua Lin. 2024.
SKVQ: sliding-window key and value cache quan-
tization for large language models.CoRR,
abs/2405.06219.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
Archi Mitra, Archie Sravankumar, Artem Korenev,
Arthur Hinsvark, Arun Rao, Aston Zhang, and 82
others. 2024. The llama 3 herd of models.CoRR,
abs/2407.21783.
9

<!-- page 10 -->

Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, and
Dragomir R. Radev. 2019. Multi-news: A large-scale
multi-document summarization dataset and abstrac-
tive hierarchical model. InProceedings of the 57th
Conference of the Association for Computational Lin-
guistics, ACL 2019, Florence, Italy, July 28- August
2, 2019, V olume 1: Long Papers, pages 1074–1084.
Association for Computational Linguistics.
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang,
Jiawei Han, and Jianfeng Gao. 2024. Model tells you
what to discard: Adaptive KV cache compression
for llms. InThe Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024. OpenReview.net.
Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Ju-
lian J. McAuley. 2023. Longcoder: A long-range pre-
trained language model for code completion. InIn-
ternational Conference on Machine Learning, ICML
2023, 23-29 July 2023, Honolulu, Hawaii, USA, vol-
ume 202 ofProceedings of Machine Learning Re-
search, pages 12098–12107. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020, pages 6609–6625. International Committee on
Computational Linguistics.
Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh,
Michael W. Mahoney, Yakun Sophia Shao, Kurt
Keutzer, and Amir Gholami. 2024. Kvquant: To-
wards 10 million context length LLM inference with
KV cache quantization. InAdvances in Neural In-
formation Processing Systems 38: Annual Confer-
ence on Neural Information Processing Systems 2024,
NeurIPS 2024, V ancouver , BC, Canada, December
10 - 15, 2024.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics, ACL
2017, V ancouver , Canada, July 30 - August 4, V olume
1: Long Papers, pages 1601–1611. Association for
Computational Linguistics.
Gregory Kamradt. 2023. Needle In A Haystack - pres-
sure testing LLMs.Github.
Tomás Kociský, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The narrativeqa reading
comprehension challenge.Trans. Assoc. Comput.
Linguistics, 6:317–328.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang,
Zhanchao Xu, Xuejia Chen, Nicole HU, Wei Dong,
Li Qing, and Lei Chen. 2025a. A survey on large
language model acceleration based on KV cache
management.Transactions on Machine Learning
Research.
Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-
Ling Zhen, Wulong Liu, Yiwu Yao, Sinno Jialin Pan,
and Mingxuan Yuan. 2025b. Kvtuner: Sensitivity-
aware layer-wise mixed precision KV cache quantiza-
tion for efficient and nearly lossless LLM inference.
CoRR, abs/2502.04420.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. Snapkv:
LLM knows what you are looking for before genera-
tion. InAdvances in Neural Information Processing
Systems 38: Annual Conference on Neural Informa-
tion Processing Systems 2024, NeurIPS 2024, V an-
couver , BC, Canada, December 10 - 15, 2024.
Chaofan Lin, Jiaming Tang, Shuo Yang, Hanshuo Wang,
Tian Tang, Boyu Tian, Ion Stoica, Song Han, and
Mingyu Gao. 2025. Twilight: Adaptive attention
sparsity with hierarchical top-ppruning.CoRR,
abs/2502.02770.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
Truthfulqa: Measuring how models mimic human
falsehoods. InProceedings of the 60th Annual Meet-
ing of the Association for Computational Linguistics
(V olume 1: Long Papers), ACL 2022, Dublin, Ireland,
May 22-27, 2022, pages 3214–3252. Association for
Computational Linguistics.
Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Ye-
long Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian
Jiao, Nan Duan, and Weizhu Chen. 2024. Rho-1: Not
all tokens are what you need.CoRR, abs/2404.07965.
Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao
Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyril-
lidis, and Anshumali Shrivastava. 2023. Scis-
sorhands: Exploiting the persistence of importance
hypothesis for LLM KV cache compression at test
time. InAdvances in Neural Information Processing
Systems 36: Annual Conference on Neural Informa-
tion Processing Systems 2023, NeurIPS 2023, New
Orleans, LA, USA, December 10 - 16, 2023.
Shervin Minaee, Tomás Mikolov, Narjes Nikzad,
Meysam Chenaghlu, Richard Socher, Xavier Am-
atriain, and Jianfeng Gao. 2024. Large language
models: A survey.CoRR, abs/2402.06196.
OpenAI. 2023. GPT-4 technical report.CoRR,
abs/2303.08774.
Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shix-
uan Fan, Ke Cheng, Weiyao Lin, and Jianguo Li.
10

<!-- page 11 -->

2025. CAKE: cascading and adaptive KV cache
eviction with layer preferences. InThe Thirteenth In-
ternational Conference on Learning Representations,
ICLR 2025, Singapore, April 24-28, 2025. OpenRe-
view.net.
Siva Reddy, Danqi Chen, and Christopher D. Manning.
2019. Coqa: A conversational question answering
challenge.Trans. Assoc. Comput. Linguistics, 7:249–
266.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
GPQA: A graduate-level google-proof q&a bench-
mark.CoRR, abs/2311.12022.
Noam Shazeer. 2019. Fast transformer decoding: One
write-head is all you need.CoRR, abs/1911.02150.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics, 10:539–554.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. InAdvances in Neural Information Pro-
cessing Systems 30: Annual Conference on Neural
Information Processing Systems 2017, December 4-9,
2017, Long Beach, CA, USA, pages 5998–6008.
Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan
Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong,
and Mi Zhang. 2024. D2O: dynamic discriminative
operations for efficient generative inference of large
language models.CoRR, abs/2406.13035.
Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia
Zhang. 2024. Model tells you where to merge: Adap-
tive KV cache merging for llms on long-context tasks.
CoRR, abs/2407.08454.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. 2024. Efficient streaming lan-
guage models with attention sinks. InThe Twelfth
International Conference on Learning Representa-
tions, ICLR 2024, Vienna, Austria, May 7-11, 2024.
OpenReview.net.
Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan
Zhou, Jianlong Wu, and Liqiang Nie. 2024.
Wkvquant: Quantizing weight and key/value cache
for large language models gains more.CoRR,
abs/2402.12065.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher Ré, Clark W. Barrett,
Zhangyang Wang, and Beidi Chen. 2023. H2O:
heavy-hitter oracle for efficient generative inference
of large language models. InAdvances in Neural
Information Processing Systems 36: Annual Confer-
ence on Neural Information Processing Systems 2023,
NeurIPS 2023, New Orleans, LA, USA, December 10
- 16, 2023.
Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia
Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli
Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir R.
Radev. 2021. Qmsum: A new benchmark for query-
based multi-domain meeting summarization. InPro-
ceedings of the 2021 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, NAACL-
HLT 2021, Online, June 6-11, 2021, pages 5905–
5921. Association for Computational Linguistics.
11

<!-- page 12 -->

A Full Algorithm of ReFreeKV
Algorithm 1ReFreeKV
Input:Prompt, Thresholdt
Output:Compressed KV Cache
Create Empty ListK c,V c
forTransformer Layer L i in LLMdo
Qi,K i,V i ←L i(Prompt)
Ri ←Postion-Based Importance Rank
Ai
last ←Attention(Q i[. . . ,−1,:],K iT )
Fi
b ←Frobenius(A i
last)
Ai
last ←Square(A i
last)
ReorderA i
last by RankR
Ai
cumsum ←Cumsum(A i
last)
Ai
cumsum ←Sqrt(A i
cumsum)
Ai
ratio ←(F i
b −A i
cumsum)/F i
b
Indexi ←Max(Where(A i
ratio <=t))
Ki
c ←CompressK ibyRi[I:]
V i
c ←CompressV ibyRi[I:]
AppendK i
c,V i
c toK c,V c
end for
returnK c,V c
B Retain the first two layers of LLM
In this section, we demonstrate the design choice
of ReFreeKV to compress the KV cache starting
from the third layer of LLMs. We apply ReFreeKV
with Llama2-7B-Chat and conduct experiments on
GSM8K and CoQA. We first perform a case study,
followed by a comparison of the effects of not re-
taining full KV cache of the first two layers.
As shown in Table 6, when the threshold is set
to 1% and no layers are frozen, the outputs for two
examples on GSM8K and CoQA are incorrect and
lack logical coherence. However, the budget ex-
ceeds 90% in both cases. By observing the actual
budget of each layer, we can see that for these two
examples, the budgets for layer 1 and 2 are rela-
tively low, while the budgets from layer 3 to layer
31 are the same and very high, with the last layer
being low again. We analyze that since the model,
in the first two layers, is not yet able to identify
truly important tokens, the attention distribution
is relatively uniform. This may lead to the model
discarding truly important tokens during eviction,
resulting in subsequent generations being unable
to access this information, ultimately causing the
output to fail.
Based on the above case study, we attempt to re-
tain certain layers of the model and explore the op-
GSM8K
Input
A robe takes 2 bolts of blue fiber a-
nd half that much white fiber. How
many bolts in total does it take?
Budget
Layer 1: 67.71
Layer 2: 82.29
Layer 3∼31:95.83
Layer 32: 37.50
Avg.: 93.90
Output
I have determined by answering the
answer to the format of bolts bolts b-
olts...(repeat)
Ground-Truth 3
CoQA
Input
You are given a story and a question.
Answer the question as concisely as
you can...Question: What color was
Cotton?
Budget
Layer 1: 42.14
Layer 2: 42.54
Layer 3∼31: 99.19
Layer 32: 79.64
Avg.: 95.03
Output Question: Question: What is the que-
stion: What is the question:
Groud-Truth White
Table 6: Case Study.
Datasets Layer None 0 0,1 0,1,2 0,1,2,3 0,1,31
GSM8K Oursk=1 0.014 0.250 0.264 0.252 0.264 0.252Budget 93.1% 97.0% 95.4% 97.5% 97.7% 98.2%
CoQA Oursk=1 1.47 52.80 53.58 53.58 53.32 53.46Budget 92.7% 94.7% 95.5% 98.3% 98.4% 99.2%
Table 7: Performance of Llama2-7B-Chat on GSM8K
and CoQA using ReFreeKV with different frozen layers.
timal configuration. We continue to test the results
of retain different layers. The results are shown in
Table 7. We can observe that freezing the first two
layers achieves a balance between model perfor-
mance and budget. Moreover, retaining the 31st
layer does not significantly enhance the model’s
performance and instead leads to an increase in
budget. Consequently, ReFreeKV ultimately opts
to begin KV cache compression from the 3rd layer
of the model.
C Main Results with 50% Budget
We show peformance comparison between Re-
FreeKV and different eviction-based methods with
50% cache size. The results are show in Table 8.
12

<!-- page 13 -->

Methods
Math&Science CR Single-Doc QA Multi-Doc QA Summarization FSL Code
Avg.
GSM8KGPQATheoQAThQACoQANrtvQAQasper2WkMQAMusiqueQMSumM-NewsTriviaQALcc
Llama3
Full 75.28 29.02 21.29 25.59 52.74 24.06 43.91 35.33 14.77 22.27 27.37 70.26 19.16 100%
Oursk=1 76.50 30.13 23.03 26.09 52.86 23.44 37.38 36.21 15.96 21.95 27.88 64.24 19.31+0.12%
Budget 93.2%86.7%92.8%78.7%81.5%48.7%46.4% 45.8% 43.7% 15.0%76.4% 41.0%78.0%63.68%
H2O0.5 31.08 2.90 16.47 17.99 42.37 23.14 43.05 32.79 15.77 22.79 26.24 69.83 19.18 -17.63%
SLM0.5 3.41 6.03 8.84 18.75 44.68 20.94 37.73 31.20 15.29 21.80 25.93 67.79 19.12 -24.73%
SnapKV0.5 12.96 7.81 16.87 19.52 42.84 23.40 43.93 33.98 15.94 22.67 26.07 69.66 19.19 -17.03%
Mistral
Full 33.36 29.24 6.83 20.82 39.68 28.74 37.80 33.87 22.88 22.19 22.94 86.87 16.08 100%
Oursk=1 31.54 29.02 6.71 20.81 39.80 27.20 38.93 33.46 22.15 22.39 22.67 86.81 15.33 -1.50%
Budget 89.6%90.5%84.2%92.3%84.1%78.0%97.9% 86.7% 74.4% 84.3%89.1% 87.2%89.4%86.75%
H2O0.5 2.50 8.26 5.89 20.28 38.53 27.62 36.97 34.31 23.17 22.32 22.60 86.52 16.45 -14.31%
SLM0.5 2.43 6.47 1.14 18.27 32.76 24.91 33.80 32.73 20.68 21.48 18.54 86.45 13.58 -27.61%
SnapKV0.5 3.03 8.93 6.83 19.03 39.22 27.15 38.27 34.60 22.94 21.93 22.68 86.31 16.18 -13.41%
Qwen2.5
Full 88.02 31.70 29.85 24.55 61.43 20.81 43.17 47.15 30.70 23.64 24.24 87.64 2.44 100%
Oursk=1 88.02 31.25 30.39 24.48 60.01 20.66 42.74 47.20 29.56 22.64 24.04 87.65 3.58 +2.63%
Budget 90.7%88.8%86.5%69.2%84.1%65.1%69.2% 73.3% 64.4% 70.6%85.4% 56.6%84.4%76.02%
H2O0.5 34.12 14.73 20.88 23.10 59.69 21.59 43.37 47.60 30.81 21.00 22.95 85.54 2.24 -13.47%
SLM0.5 3.26 1.34 10.58 23.75 49.87 19.29 36.45 42.42 25.33 21.24 22.77 87.33 3.40 -23.56%
SnapKV0.5 20.77 12.28 22.76 22.25 60.21 20.93 43.22 47.45 30.61 23.76 22.70 87.57 2.21 -14.39%
Table 8:Performance of ReFreeKV and its comparison with five KV cache pruning models on 13 datasets. The cache size is set
to 50%.
MethodsGSM8K CoQA NQA Musique QMSum
Full 75.28 52.74 24.06 14.77 22.27
Post=1% Performance Using Different k
k=1 76.50 52.86 23.44 15.96 21.95
Budget 93.2% 81.5% 48.7% 43.7% 15.0%
k=1%n 76.19 52.87 23.17 15.40 21.94
Budget 95.1% 94.8% 77.3% 77.2% 59.5%
k=2%n 76.19 52.82 22.87 14.34 21.74
Budget 96.0% 96.1% 69.8% 62.0% 47.1%
k=3%n 75.82 52.80 23.03 13.61 21.56
Budget 98.6% 95.7% 60.3% 46.0% 39.5%
k=4%n 75.21 52.76 22.96 13.45 21.47
Budget 98.8% 95.8% 51.9% 35.2% 35.4%
k=5%n 75.59 52.75 21.62 13.71 21.43
Budget 98.6% 96.2% 45.1% 30.0% 33.1%
k=6%n 75.66 52.81 21.95 14.33 20.94
Budget 98.5% 96.6% 40.4% 25.8% 31.6%
k=7%n 76.57 52.88 21.18 13.66 20.94
Budget 98.0% 96.7% 37.3% 22.5% 30.6%
k=8%n 76.35 52.86 20.36 13.75 21.11
Budget 97.4% 96.7% 35.1% 20.9% 30.1%
k=9%n 76.65 52.84 19.99 13.66 20.76
Budget 96.8% 96.8% 33.4% 19.9% 29.5%
k=10%n 76.72 52.85 9.74 13.67 21.11
Budget 96.3% 96.8% 32.0% 19.7%29.3%
Table 9: Performance comparison with differentk, ex-
panded from Table 5.
From the table, we can observe the same conclu-
sions with Section 3.
D Datasets Used in Experiments
In this section, we provide a comprehensive
overview of all the tasks and datasets utilized in the
experiments in this paper.
Math&ScienceThis task evaluates the model’s
ability to tackle mathematical and scientific prob-
lems. By directly inputting questions and compar-
ing the model’s output with the correct answers, we
calculate the model’sAccuracyon these datasets:
GSM8Kis a dataset for evaluating model’s math-
solving skills, featuring 8,000 elementary-level
math word problems requiring basic arithmetic and
reasoning.GPQAtests model’s understanding of
physics concepts and problem-solving across vari-
ous topics, assessing scientific reasoning abilities.
TheoremQAevaluates model’s grasp and appli-
cation of mathematical theorems, ranging from
simple applications to complex proofs, testing ad-
vanced math skills.
Commonsense Reasoning (CR)This task eval-
uates model’s ability to make deductions and un-
derstand everyday situations using implicit knowl-
edge and logical inference.TruthfulQA(ThQA)
evaluates model’s ability to generate accurate and
truthful responses, testing models on distinguishing
fact from fiction, especially in areas prone to mis-
conceptions. We useBLEUas the metric.CoQA
assesses model’s ability to understand and respond
to questions in a conversational context, focusing
on maintaining coherence and context throughout
a dialogue. We useF1 Scoreas the metric.
Single Document QA (Single-Doc QA)This
task assesses the model’s reading comprehension
skills when dealing with a single, extended doc-
ument.NarrativeQA(Kociský et al., 2018) is a
dataset designed to evaluate model’s ability to com-
13

<!-- page 14 -->

BS=1 BS=2 BS=8 BS=16
[Input, Output][4K, 8K] [8K, 16K] [512, 4K] [4K, 8K] [512,512] [4K, 4K] [512, 512] [2K, 2K]
Latency (s/100tokens)
HF Accelerate 3.61 5.29 3.50 5.26 3.48 12.23 4.71 13.01
Ours 3.19 4.08 2.84 3.60 3.06 10.42 4.25 10.64
Throughput (token/s)
HF Accelerate 27.71 18.89 57.08 38.01 230.00 65.41 339.43 122.98
ReFreeKV 31.35 24.51 70.30 55.56 261.83 76.81 376.89 150.40
Budget 73.3% 55.9% 62.0% 34.0% 76.7% 65.3% 82.8% 78.2%
Table 10: Performance comparison across various batch sizes and sequence lengths. We report latency (s/100 tokens,
lower is better) and throughput (tokens/s, higher is better). In the header, ’k’ denotes 1024 tokens (e.g., 4K=4096).
prehend and answer questions based on narrative
texts, focusing on understanding stories and their
underlying themes.Qasper(Dasigi et al., 2021) is
a dataset aimed at assessing model’s capability to
extract and answer questions from academic papers,
emphasizing understanding complex scientific in-
formation. We employF1 Scoreas the metric for
above two datasets.
Multi-Document QA (Multi-Doc QA)This task
evaluates the model’s reading comprehension ca-
pabilities across multiple extended documents.
2WikiMultiHopQA(2WKMQA) (Ho et al., 2020)
is a dataset designed to test model’s ability to
perform multi-hop reasoning and answer com-
plex questions using information from multiple
Wikipedia articles.MuSiQue(Trivedi et al., 2022)
evaluates model’s skill in integrating and reasoning
over information from multiple sources to answer
comprehensive questions accurately. We leverage
F1 Scoreas the metric for above two datasets.
SummarizationThis task examines the model’s
ability to comprehend and summarize lengthy doc-
uments.QMSum(Zhong et al., 2021) is a dataset
for evaluating model’s ability to generate concise
summaries of meeting transcripts, focusing on cap-
turing the key points from multi-party discussions.
Multi-News(M-News) (Fabbri et al., 2019) is a
dataset that challenges models to create coherent
summaries by synthesizing information from multi-
ple news articles on the same topic. We useRouge-
Las the metric for above two datasets.
Few-Shot Learning (FSL)This task assesses
the model’s few-shot learning capabilities.Trivi-
aQA(Joshi et al., 2017) is a dataset designed to
assess model’s ability to retrieve and answer ques-
tions based on large collections of trivia, emphasiz-
ing comprehension and factual recall. We useF1
Scoreas the metric.
CodeThis task evaluates the model’s ability to
complete and generate code.LCC(Guo et al.,
2023) is a dataset focused on evaluating models’
ability to understand and generate code by consid-
ering extended code contexts, enhancing the ability
to reason over complex programming structures.
We useEdit Simas the metric.
E Ablation
In this section, we present additional ablation study
results for Section 3.4. By setting various values
for k, we expand upon the results shown in Ta-
ble 5. The expanded results are displayed in Ta-
ble 9. These experiments facilitate a deeper under-
standing of how different parameter settings impact
model performance and provide a basis for optimiz-
ing parameter selection.
As shown in Table 9, setting k =1 not only
conserves pruning time but also achieves better
model performance with a reduced budget.
F Analysis of Latency and Throughput
We perform following experiments for a compre-
hensive analysis of the latency and throughput. We
follow the setup in FastGen using inputs of Narra-
tiveQA, and conduct end-to-end latency compar-
ison experiments on a single 80GB A100 GPU,
using the standard HuggingFace (HF) Accelerate
library as the baseline. The results are shown in
Table 10. From the table, we can observe that
ReFreeKV consistently outperforms the baseline in
both latency and throughput. Moreover, ReFreeKV
holds its advantage when the batch size increases,
and its throughput remains robust. It is notewor-
thy that Table 4 in Section 4.2 also demonstrates
the trivial overhead of ReFreeKV due to its simple
14

<!-- page 15 -->

GSM8K GPQA CoQA NrtvQA QMSum TriviaQA
Full 75.28 29.02 52.74 24.06 22.27 70.26
Oursk=1 76.50 30.13 52.86 23.44 21.95 64.24
Budget 93.2% 86.7% 81.5% 48.7% 15.0% 41.0%
SLM0.9 72.78 28.79 52.83 23.75 22.48 69.97
SLMF0.9 76.42 29.24 52.89 22.95 22.47 69.10
SLM0.5 3.41 6.03 44.68 20.94 21.80 67.79
SLMF0.5 4.55 3.13 44.59 21.31 21.88 68.04
SLM0.2 3.21 3.35 40.11 20.12 21.11 66.17
SLMF0.2 1.21 1.12 38.44 19.52 20.61 65.69
Table 11: Performance comparison between ReFreeKV and StreamingLLM with the first two layers frozen (SLMF)
under difference KV cache budgets (0.9, 0.5 and 0.2).
MethodsGSM8K CoQA NQA Musique QMSum
Llama3-70B-Instruct
Full 89.69 60.36 27.15 29.31 22.52
Oursk=1 89.76 60.38 26.94 28.88 22.30
Budget 93.3% 77.2% 58.7% 60.8% 52.9%
Qwen2.5-32B-Instruct
Full 91.36 58.03 24.78 40.04 22.84
Ours 91.81 57.29 22.65 40.54 22.52
Budget 91.6% 85.0% 68.2% 74.4% 76.8%
Qwen2.5-72B-Instruct
Full 90.22 54.14 24.36 42.13 23.93
Ours 90.30 54.19 24.10 41.70 23.31
Budget 90.9% 87.6% 63.4% 65.0% 68.6%
Table 12: Performance of LLMs with scales of 70B,
34B, and 72B across five datasets using ReFreeKV.
matrix computation during its two-stage pruning
process.
G Comparison with StreamingLLM
We perform additional experiments on six datasets
using StreamingLLM (with Llama3-8B) that has
the first two layers frozen, as an ablation study on
the effects of the proposed pruning and halting pro-
cess. As shown in Table 11, despite having the first
two layers frozen, StreamingLLM still falls short of
achieving the objective proposed in this work, not
able to remain optimal consistently across datasets
like GSM8K and GPQA as the budget varies. It
is demonstrated that solely leveraging the position
bias as in StreamingLLM is not able to maintain the
full performance; the proposed pruning and halting
mechanism that integrates the attention norm met-
ric is necessary in achieving the goal for lossless
pruning.
H Related Work
KV Cache CompressionTo mitigate the large
memory footprint of the KV cache during LLM
inference, a prominent line of work has focused
on pruning, which selectively discards less impor-
tant tokens. Methods like Scissorhands (Liu et al.,
2023) and SnapKV (Li et al., 2024) retain tokens
based on high attention scores. Others employ
strategies based on token recency and historical
importance, such as H2O (Zhang et al., 2023) or
StreamingLLM (Xiao et al., 2024). FastGen (Ge
et al., 2024) further refines this by adapting reten-
tion strategies on a per-head basis.
A common limitation of these pruning methods
is their reliance on a pre-defined memory budget,
which can lead to inconsistent performance across
different domains and datasets. In contrast, our
work lifts this constraint, aiming for near full-cache
performance with a dynamically sized budget with-
out requiring task-specific tuning.
Beyond pruning, other compression paradigms
have been explored. These include quantization-
based methods, which reduce the numerical preci-
sion of cached values (Duanmu et al., 2024; Hooper
et al., 2024; Du et al., 2025; Yue et al., 2024; Li
et al., 2025b), and low-rank approximation tech-
niques that apply matrix decomposition to com-
press the KV cache (Chang et al., 2025; DeepSeek-
AI et al., 2024).
Attention PatternsOur approach is grounded
in research on the structural patterns of attention
mechanisms. Studies have shown that attention
heads often specialize, with some exhibiting pre-
dictable, position-biased behaviors such as con-
sistently focusing on adjacent tokens (Shazeer,
2019). This indicates a degree of redundancy in
15

<!-- page 16 -->

their learned patterns. Furthermore, the importance
of individual heads can vary significantly across
tasks (Lin et al., 2024), suggesting opportunities
for dynamic optimization.
16
