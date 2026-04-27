# references/149_thin_keys_full_values_reducing_kv_cache_via_low_dimensional_attention_selection.pdf

<!-- page 1 -->

Thin Keys, Full Values:
Reducing KV Cache via Low-Dimensional Attention Selection
Hengshuai Yao1,2 Xing Chen1 Ahmed Murtadha1 Guan Wang1
1Sapient Intelligence
2Department of Computing Science, University of Alberta
hengshu1@ualberta.ca, raincchio@gmail.com,
murtadha20@gmail.com, imonenext@gmail.com
March 31, 2026
Abstract
Standard transformer attention uses identical dimensionality for queries, keys, and values,
yet these components serve different roles: queries and keys produce scalar attention weights
(selection), while values carry rich representations (value transfer). We show that selection
requires onlyO(logN )dimensions to distinguish among N relevant token categories (e.g.,
syntactic roles, semantic clusters, positional patterns)—far fewer than value transfer needs.
We introducefactored keys, which exploit this asymmetry to physically shrink the KV
cache ofany pretrained model without retraining from scratch—unlike GQA and MLA, which
must be designed into the architecture before pretraining. We factorize each key projection
WK≈Ad×rBr×dvia truncated SVD (wherer =dselect), setW′
K =A as the new key projection
producing compactr-dimensional keys for the cache, and absorbB⊤into the query projection
(W′
Q =WQB⊤) at zero cost—since queries are never cached. At 7B scale, training from scratch
with dselect = dmodel/4matches full-attention perplexity (9.24vs9 .25PPL after 20B tokens,
mean over 2 seeds) while using 12% fewer parameters and training 8% faster. For existing models,
SVD + QK fine-tuning (3 epochs,<1% of pretraining data) achieves 75% key cache savings
at∼2% quality cost on both GPT-2 and Mistral-7B. The approach composes with GQA and
quantization for up to 16×combined key cache compression. For a 7B model serving 128K
context, factored keys save 25GB of KV cache per user, enabling∼60% more concurrent users
on identical hardware.
1 Introduction
The self-attention mechanism (Vaswani et al., 2017) projects inputs into queries, keys, and values
(Q, K, V) and computes softmax(QK⊤/√dk)V. In virtually all modern transformers—GPT
(Radford et al., 2018), LLaMA (Touvron et al., 2023), Mistral (Jiang et al., 2023)—the projection
dimensionality is identical:dq =dk =dv =dmodel. This symmetry is a design convention, not a
necessity.
As models scale to longer contexts, the KV cache becomes the dominant memory bottleneck
during autoregressive inference: a 7B model serving 128K context requires over 60GB of KV cache
per user. Existing approaches to this bottleneck reduce thenumberof KV heads—Multi-Query
Attention (Shazeer, 2019) and Grouped-Query Attention (Ainslie et al., 2023)—or compress the
jointKV state into a low-rank latent. These methods have been widely adopted (GQA in LLaMA-2
(Touvron et al., 2023) onward; MLA in DeepSeek-V2 (Liu et al., 2024)), but they all maintain
dk =d head within each head’s attention computation.
1
arXiv:2603.04427v4  [cs.LG]  28 Mar 2026

<!-- page 2 -->

We take a complementary approach, motivated by a structural observation: attention performs
two functionally distinct operations with fundamentally different complexity requirements. (1)Se-
lection( QK⊤): determiningwhichtokens are relevant via scalar attention weights. (2)Value
transfer( attn·V): aggregating information from selected tokens. Our key insight is that selection
is arankingproblem, not a representation problem: by the Johnson–Lindenstrauss lemma (Johnson
and Lindenstrauss, 1984), distinguishing amongN items in a dot-product space requires only
O(logN )dimensions. Value transfer, by contrast, must preserve the full representational capacity
of the model. Recent theoretical work on KV cache compression lower bounds provides independent
support: Haris and Onak (2025) show that any attention-based token generation algorithm requires
Θ(nd)space with d = Ω( logn ), establishing that the KV cache cannot be compressed belowlogn
dimensions per token without loss. This asymmetry implies that queries and keys are drastically
overparameterized atd k =d model.
We proposeasymmetric attention, where queries and keys project todselect≪dmodel while
values retain full dimensionality. The attention computation is unchanged—QK⊤/
√
dQK
head produces
scalar weights applied to V of any dimensionality. This yields three benefits: (i) parameter
reduction (up to 75% for QK whendselect = dmodel/4), (ii) KV cache reduction (the key cache
stores dselect-dimensional vectors, saving 25GB per user at 128K context for 7B models), and
(iii) attention compute reduction (O(n2·dselect)instead ofO(n2·dmodel)). Unlike GQA or MLA,
this approach reduces theper-head dimensionalityof keys while keeping the number of heads and
value dimensionality intact—and composes with both for further savings.
A central contribution of this work is that asymmetric attention can be applied toany existing
pretrained modelwithout retraining from scratch—in contrast to GQA and MLA, which must be
built into the architecture before pretraining or require expensive conversion (Ma et al., 2025).
We introducefactored keys, an inference primitive that physically shrinks the key cache of
already-deployed models: factorize each layer’sWK≈Ad×rBr×dvia truncated SVD, useA as the
new key projection producing compactr-dimensional cached keys, and absorbB⊤into the query
projection at zero memory cost—exploiting the fundamental asymmetry that keys accumulate in
memory while queries are ephemeral. With optional lightweight QK fine-tuning (3 epochs on<1%
of pretraining data), this recovers nearly all quality. This is related to recent work on low-rank KV
cache compression (Li et al., 2025; Wang et al., 2024; Kang et al., 2024), but differs in a key respect:
we compressonlykeys and absorb the cost into queries, which are never cached. A striking finding
is that keys are far more compressible than queries: at rank 192 on GPT-2,K-only SVD degrades
by 26% whileQ-only degrades by 181%—a 7×asymmetry consistent with recent observations that
key vectors lie in a significantly lower-dimensional space (Kang et al., 2024).
We validate across nine experiments from algorithmic tasks to 7B-scale training (Experiments 1–4
and 6 appear in Sections 8–9; the four largest-scale experiments are presented first). Controlled
experiments confirm that positional selection requires only 1 dimension per head, and content-
based selection requireslog2N dimensions (Section 8). At 7B scale, training from scratch with
dselect =dmodel/4matchesfull-attentionperplexityafter20Btokenswhileusing12%fewerparameters
and training 8% faster. For existing models, SVD + QK fine-tuning on Mistral-7B achieves 75%
key cache savings at∼2% quality cost.
2

<!-- page 3 -->

2 Method
2.1 Asymmetric Attention
In multi-head attention withh heads, standard transformers setdhead =dmodel/h for all ofQ,K, and
V. We decouple this by introducingdselect: the per-head QK dimension becomesdQK
head =dselect/h
while the value dimension staysdV
head =d model/h. For each headi:
Qi =XW (i)
Q , W (i)
Q ∈Rdmodel×dQK
head (1)
Ki =XW (i)
K , W (i)
K ∈Rdmodel×dQK
head (2)
Vi =XW (i)
V , W (i)
V ∈Rdmodel×dV
head (3)
headi =softmax

 QiK⊤
i√
dQK
head

Vi (4)
No projection is needed between attention weights and value aggregation: the weights∈Rn×n
are scalars regardless ofdQK
head. Whendselect =dmodel, this reduces to standard multi-head attention.
2.2 Theoretical Motivation
We argue that selection requiresO(logN )dimensions where N is the number of distinct patterns
the attention mechanism must distinguish. The attention weightαij∝exp(q⊤
i kj/
√
dQK
head)assigns
scalar relevance scores—a ranking problem. By the Johnson–Lindenstrauss lemma,N points can be
embedded intoO(logN )dimensions while preserving pairwise distances, soO(logN )dimensions
suffice to maintain the relative ordering of dot-product similarities. In language modeling, the
effectiveN is the number of distinct selection patterns (syntactic roles, semantic clusters, recency
patterns)—empirically in the hundreds, much smaller than vocabulary size, suggestingdQK
head≈10–20
dimensions per head. Value transfer, by contrast, must preserve the complete representational
content across alldmodel dimensions.
2.3 Factored Keys: An Inference Primitive for Pretrained Models
Given a pretrainedWK∈Rdmodel×dmodel, we seek a thin-key factorization of the form:


WK


  
d×d
≈


W′
K


  
thin keys(d×r)
[
B
]
  
r×d(absorbed intoWQ)
(5)
Conveniently, truncated SVD provides exactly this factorization, withr =dselect≪dmodel. We then
replace the original projections:
W′
K =A=U rΣ r∈Rdmodel×r (thin keyprojection — cached) (6)
W′
Q =W QB⊤=W QVr∈Rdmodel×r (absorbed query projection — ephemeral) (7)
Keys becomek′
j =xjW′
K∈Rr—the “thin keys” stored in the cache at a fraction of the original size.
Queries becomeq′
i = xiW′
Q∈Rr, preserving attention scores exactly:q′
ik′⊤
j = xiWQB⊤A⊤x⊤
j =
3

<!-- page 4 -->

xiWQW⊤
Kx⊤
j =qik⊤
j . ComputingW′
Q is a one-time matrix multiplication—no training or fine-tuning
is required. This is the beauty of SVD: we obtain thin keys from any pretrained full-key modelfor
free, simply by repartitioning the existing weights. The result: key cache entries shrink fromdmodel
to r dimensions while all compression cost is absorbed by the ephemeral query side (queries are
computed fresh each step and never cached).
Crucially,nothing else in the network changes— WV, WO, the FFN, and all normalization layers
remain untouched. The only modification is replacingWK and WQ with their thin counterparts,
computed from a single SVD. Despite this simplicity, training from scratch with thin keys at 7B
scale achieves 9.2 vs 9.3 validation PPL after 20B tokens—matching full attention—while using
12% fewer parameters and training 8% faster (Experiments 7–7b, Tables 3 and 4).
This zero-cost property distinguishes factored keys from all existing KV compression methods.
GQA (Ainslie et al., 2023) and MLA (Liu et al., 2024) must be designed into the architecture before
pretraining—they cannot be applied to already-deployed models. LRQK (Li et al., 2025) performs
QK decomposition at prefill time, adding latency to every inference call. SVD-LLM (Wang et al.,
2024) requires calibration data and a whitening step to guide truncation. By contrast, factored
keys require only a one-time offline SVD per layer—no calibration data, no prefill overhead, no
retraining—and the resulting model runs with standard attention kernels. We validate this in
Experiments 5 and 8: zero-cost SVD at rankdmodel/2immediately achieves 50% key cache savings
at +2% PPL with no fine-tuning at all. To compress more aggressively, optional lightweight QK
fine-tuning (3 epochs) recovers quality at lower ranks, enabling 75% savings at rankdmodel/4with
only +2% PPL (Tables 2 and 7), with decode throughput gains increasing with batch size as the
KV cache fraction of memory bandwidth grows (Table 11).
2.4 KV Cache Implications
During autoregressive generation, the KV cache forLlayers at context lengthnis:
Standard KV cache= 2·n·dmodel·L·bbytes (8)
Asymmetric KV cache=n·(dselect +d model)·L·bbytes (9)
whereb is bytes per parameter. Withdselect =dmodel/4, the K cache shrinks by 75%, yielding 37.5%
total KV cache reduction.
3 Experiments
We validate asymmetric attention through nine experiments of increasing complexity. Controlled
experiments on algorithmic tasks and small-scale language modeling (Experiments 1–4) are in
Section 8; architecture generalization at 125M scale (Experiment 6) is in Section 9. We first
present the four largest-scale experiments: post-training SVD compression of GPT-2 (Experiment 5),
from-scratch training at 7B scale (Experiments 7 and 7b), and SVD + fine-tuning of Mistral-7B
(Experiment 8).
3.1 Experiment 5: Post-Training Compression of GPT-2
Setup.We apply SVD compression to pretrained GPT-2 (124M parameters,dmodel = 768, 12
heads, 12 layers) without retraining, testing three modes: bothWQ andW K,K-only, andQ-only.
4

<!-- page 5 -->

Results.Table 1 reveals a striking asymmetry. Compressing bothWQ and WK is catastrophic—
errors compound through the softmax. However,K-only compression is far more forgiving: rank
384 (dmodel/2) incurs only 2.0% degradation.K is substantially more compressible thanQ at every
rank: at rank 192,K-only degrades by 26% whileQ-only degrades by 181%—a 7×asymmetry.
Table 1: SVD compression of GPT-2 projections. Baseline PPL: 24.91.∆is relative PPL increase.
Rankr r/head BothQ+K K-onlyQ-only
128 10 67,132 57.37 (+130%) 149.04 (+498%)
192 16 90.95 (+265%) 31.32 (+26%) 70.05 (+181%)
256 21 46.15 (+85%) 27.49 (+10%) 40.42 (+62%)
384 32 — 25.40 (+2.0%) 27.58 (+11%)
512 42 25.69 (+3.1%) 25.23 (+1.3%) 25.47 (+2.2%)
Deployment via factored keys.The SVD factorization provides a direct path to KV cache
reduction: the key cache storesr-dimensional vectors whileB⊤is absorbed into the query projection.
Our K-only PPL measurements directly reflect deployment performance: rank 384 (dmodel/2) saves
50% of the key cache (25% total KV) at +2.0% PPL; rank 512 saves 33% of keys at +1.3%.
Recovery via QK fine-tuning.We compress WK via SVD, then fine-tune only QK projections
(∼21M of 124M parameters) on WikiText-103 for 3 epochs over 10M tokens. Table 2 shows that
fine-tuning nearly eliminates the SVD quality loss: at rank 192 (dmodel/4), the gap shrinks from
+27.6% to just +1.8% relative to an identically fine-tuned control.
Table 2: SVD compression + QK fine-tuning on WikiText-103. The “vs control” column measures
the residual gap relative to the uncompressed model after both receive identical fine-tuning.
RankrBefore FT After FT Control vs Control K cache saved
768 (none) 29.51 19.07 — baseline 0%
384 (dmodel/2) 30.15 (+2.2%) 19.14 19.07 +0.4% 50%
256 (dmodel/3) 32.61 (+10.5%) 19.28 19.07 +1.1% 67%
192 (dmodel/4) 37.64 (+27.6%) 19.42 19.07 +1.8% 75%
128 (dmodel/6) 67.26 (+128%) 19.77 19.07 +3.7% 83%
3.2 Experiment 7: From-Scratch Training at 7B Scale
Setup.We train two LLaMA-7B models from random initialization on OpenWebText (2B tokens):
a full-attention baseline and a thin-keys variant withdselect = 1024(dmodel/4). Both usedmodel =
4096, 32 heads, 32 layers,dff = 11008, with identical hyperparameters. We run each configuration
with two random seeds and report mean±std.
Results.Thin keysoutperformfull attention at 7B scale with 2B tokens (Table 3): 5.1% better
OWT perplexity while using 12% fewer parameters and training 9.4% faster. This inverts the
∼4% cost observed at smaller scales (Sections 8, 9), consistent with a regularization effect in the
overparameterized regime (tokens-to-parameters ratio∼0.3). The training trajectory (Figure 1)
shows thin keys leading at every checkpoint.
5

<!-- page 6 -->

Table 3: 7B LLaMA trained from scratch on OpenWebText (2B tokens), mean±std over 2 seeds.
Full Attention Thin Keys (dselect =d model/4)
Parameters 6.74B 5.93B (−12%)
OWT val PPL13.82±0.0913.12±0.04(−5.1%)
WT103 val PPL20.79±0.3519.60±0.46(−5.7%)
Wall-clock time 25.9h 23.4h (−9.4%)
Step Full Thin
5K 26.62 25.05
10K 20.81 19.21
15K 17.68 16.51
20K 15.66 14.74
25K 14.37 13.59
30K 13.79 13.12
Step Full Thin
5K 44.23 41.34
10K 31.99 29.20
15K 26.43 25.36
20K 24.20 22.73
25K 22.05 20.83
30K 21.19 19.96
Figure 1: Training trajectory for 7B from-scratch models (seed 137). Left: OWT validation PPL.
Right: WikiText-103 validation PPL. Thin keys lead throughout.
3.3 Experiment 7b: Extended Training at 7B Scale (20B Tokens)
Setup.We extend training to 20B tokens (3 epochs over OpenWebText, 305,175 steps), increas-
ing the tokens-to-parameters ratio to∼3.0. Architecture and hyperparameters are identical to
Experiment 7.
Table 4: 7B LLaMA on OpenWebText (20B tokens), mean±std over 2 seeds. Both architectures
converge to the same quality; thin keys train 8% faster.
Full Attention Thin Keys (dselect =d model/4)
Parameters 6.74B 5.93B (−12%)
OWT val PPL9.25±0.009.24±0.00(−0.1%)
WT103 val PPL13.26±0.1213.00±0.07(−2.0%)
Wall-clock time261.5±0.1h240.7±0.9h (−8.0%)
Results.At 20B tokens, both models converge to nearly identical perplexity (9.24±0.00vs
9.25±0.00OWT, mean over 2 seeds), confirming the earlier advantage was a regularization effect.
Thin keys are slightly better on WT-103 (13.00vs13 .26). Crucially, thin keysnever fall behind.
Figure 2 shows PPL vs training step (curves merge by∼150K steps) and vs wall-clock time (thin
keys reach any target PPL∼20 hours sooner). Downstream evaluation (Table 5) confirms task
parity: Hellaswag, ARC-Challenge, and WinoGrande scores match within noise (<1.5%). Combined
with 8% speedup and 75% key cache savings, thin keys are a strictly dominant choice at this scale.
Comparison with GQA and MLA at 7B scale.Table 6 compares thin keys against GQA
and MLA analytically at the LLaMA-7B configuration (dmodel = 4096, 128K context, bf16). GQA
and MLA compress both K and V, achieving larger total savings (75–93%) than thin keys alone
(37.5%). However, thin keys composes with GQA: applyingdselect =dmodel/4to GQA-8 yields 84.4%
6

<!-- page 7 -->

0 100 200 300
10
15
20
25
30
Step (K)
OWT val PPL
Full training (305K steps)
Full attn
Thin keys
0 100 200
10
15
20
25
30
Wall-clock time (hours)
PPL vs wall-clock time
Full attn
Thin keys
Figure 2: Training trajectory for 7B models over 305K steps (20B tokens).Left:PPL vs step; thin
keys lead early but converge by∼150K steps.Right:PPL vs wall-clock time; thin keys reach any
target PPL∼20 hours sooner.
Table 5: Downstream evaluation of 7B from-scratch models (20B tokens, seed 137). Scores are low
in absolute terms (OWT-only training) but thin keys matches full attention on all tasks.
Task Full Attention Thin Keys∆
Hellaswag (acc_norm) 55.6 55.7+0.1
ARC-Challenge (acc_norm) 29.8 29.4−0.4
WinoGrande (acc) 60.6 59.1−1.5
MMLU (acc) 23.4 23.3−0.1
GSM8K (exact_match) 0.0 0.0 0.0
total KV savings—approaching MLA’s 93% without learned up/down-projections or decoupled
RoPE. MLA’s joint latent means thin-keys composition is a no-op, but the thin-keys insight is
already embeddedin MLA: DeepSeek-V2’sdc = 512for 128 heads implies an effective per-head key
dimension of 4, far belowdhead = 128. For practitioners, factored keys offer the simplest drop-in
path for existing models; for new architectures, GQA + thin keys or MLA achieve more aggressive
compression (Ma et al., 2025).
3.4 Experiment 8: SVD + Fine-tuning at Scale (Mistral 7B)
Setup.We apply SVD + QK fine-tuning to Mistral-7B (7.2B parameters, GQA with 32 query
heads, 8 KV heads,dmodel = 4096). We compressWK via SVD to ranks 512, 256, and 128, then
fine-tune only QK projections on WikiText-103 for 3 epochs (10M tokens).
Results.The pipeline scales to 7B with consistent quality (Table 7). At rank 256 (75% K cache
saved), the residual gap is +2.0%—matching GPT-2 results exactly.
Downstream task evaluation.We evaluate on five benchmarks: Hellaswag (10-shot), ARC-
Challenge (25-shot), WinoGrande (5-shot), MMLU (5-shot), and GSM8K (5-shot CoT). Table 8
7

<!-- page 8 -->

Table 6: Analytical KV cache comparison at LLaMA-7B scale (128K context, bf16). MLA stores a
shared latentdc plus a decoupled RoPE key (dR
h). Thin keys compose with GQA by reducingdQK
head
independently.
Method K cache (GB) V cache (GB) KV total (GB) KV saved
MHA (baseline) 32.0 32.0 64.0 —
Thin keys (dselect =d model/4) 8.0 32.0 40.0 37.5%
GQA-8 8.0 8.0 16.0 75.0%
MLA (dc =512,d R
h =64) 4.5 (joint) 4.5 93.0%
GQA-8 + thin keys 2.0 8.0 10.0 84.4%
Table 7: Mistral-7B: SVD compression + QK fine-tuning on WikiText-103.
RankrBefore FT After FT Control vs Control K cache saved
1024 (none) 6.69 5.91 — baseline 0%
512 (dK/2) 6.81 (+1.7%) 5.93 5.91 +0.3% 50%
256 (dK/4) 7.84 (+17.1%) 6.03 5.91 +2.0% 75%
128 (dK/8) 12.34 (+84.4%) 6.10 5.91 +3.2% 88%
shows the results.
Table 8: Downstream evaluation of SVD-compressed Mistral-7B. “Ctrl+FT” is the uncompressed
model with identical fine-tuning.
Task Metric Baseline r512 +FT r256 +FT Ctrl+FT∆ 512 ∆ 256
Hellaswag acc_norm 81.2 81.4 80.7 81.3+0.1−0.7
ARC-Challenge acc_norm 54.0 54.1 53.4 54.4−0.5−1.7
WinoGrande acc 75.4 73.2 72.1 73.2+0.0−1.6
MMLU acc 60.1 55.2 54.4 55.7−0.9−2.3
GSM8K exact_match 38.4 27.7 25.8 29.9−7.4−13.7
At rank 512, compression is effectively lossless on knowledge and commonsense tasks (<1%
vs control). At rank 256, gaps remain modest (0.7–2.3%) except for GSM8K, where multi-step
math reasoning is more sensitive. Domain-matched fine-tuning recovers GSM8K quality: fine-
tuning on GSM8K’s own training split closes the compression gap from−13.7% to just−1.2% for
r=256 (Section 11). Generalization to held-out math benchmarks (Minerva Algebra, AGIEVAL
AQuA-RAT) confirms transferable reasoning recovery with<2.5% compression gaps.
4 Analysis
4.1 Practical Benefit: KV Cache
The primary practical benefit is inference-time KV cache reduction. For a representative 7B
configuration (dmodel = 4096, 32 layers, fp16), Table 10 shows savings at different context lengths.
8

<!-- page 9 -->

Table 9: Math generalization of GSM8K-fine-tuned models on held-out benchmarks.
Benchmark Metric Control r512 r256∆ 512 ∆ 256
GSM8K exact_match 53.7 52.1 51.2−1.6−2.5
Minerva Algebra math_verify 14.2 14.1 12.4−0.2−1.9
Minerva Pre-Algebra math_verify 21.0 19.1 19.1−2.0−2.0
AGIEVAL AQuA-RAT acc 15.7 17.3 17.3+1.6+1.6
Table 10: KV cache memory per user (dmodel = 4096, 32 layers, fp16).
Standardd select =d model/2d select =d model/4
(SVD, no retrain) (train or SVD+FT)
128K context
K cache 33.6 GB 16.8 GB 8.4 GB
V cache 33.6 GB 33.6 GB 33.6 GB
Total 67.2 GB 50.4 GB 42.0 GB
Savings/user — 16.8 GB (25%) 25.2 GB (37.5%)
1M context
Total524 GB 393 GB 328 GB
Savings/user — 131 GB (25%) 196 GB (37.5%)
These savings compound with concurrent users and context length. Even the conservative SVD
path saves 131GB per user at 1M context.
4.2 Decode Throughput: Roofline Analysis
The KV cache savings translate directly to decode throughput gains.
Bandwidth model.Autoregressive decode is deeply memory-bandwidth-bound: on H100 SXM
(3.35TB/s peak), each decode step reads model weightsW (shared across the batch) plus per-
sequence KV cache. For batch sizeb and context lengthn, total bytes read per step areW +b·Ckv,
where Ckv = 2Ln kvdheadn·βand β= 2for bf16. Factored keys reduceboth W (thinner WQ, WK
projections) andC kv (thinner K cache). The predicted speedup is:
Speedup(b) = W+b·Ckv
W′+b·C′
kv
(10)
This increases monotonically withb: atb=0the gain comes only from smaller projections (W/W′);
asb→∞the cache term dominates and the speedup approachesCkv/C′
kv.
Mistral-7B numbers.For Mistral-7B ( W= 14.2GB, Ckv=537MB at n=4096), factored r256
givesW′=13.2GB (1.0GB saved from thinner Q/K projections) andC′
kv=336MB (K cache shrunk
4×). Table 11 reports both the bandwidth-model prediction from Eq. 10 and measured throughput
on a single H100 SXM.
As predicted by the bandwidth model, measured speedups increase monotonically with batch
size, since the KV cache fraction of total bandwidth grows from∼4%at b=1to ∼55%at b=32.
9

<!-- page 10 -->

Table 11: Decode throughput for Mistral-7B with factored keys on H100 SXM (context 4096, 128
generated tokens). “Predicted” is the bandwidth roofline from Eq. 10.
Batch size
1 4 8 16 32
Measured throughput (tokens/s)
Baseline (dk =128) 49.5 124.0 160.6 181.5 194.7
r512 (dk =64) 53.5 142.9 192.8 223.2 243.0
r256 (dk =32) 56.1 155.7 216.4 254.3 280.2
Measured speedup
r512 1.08×1.15×1.20×1.23×1.25×
r256 1.13×1.26×1.35×1.40×1.44×
Predicted speedup (Eq. 10)
r512 1.06×1.08×1.10×1.14×1.19×
r256 1.09×1.12×1.17×1.23×1.31×
Measured speedups slightly exceed predictions (e.g.1.44×vs1 .31×at b=32for r256), likely due
to improved cache locality from smaller tensors. At batch size 1, the 14.2GB model-weight read
dominates and KV savings are modest; at large batch sizes the cache term dominates and factored
keys approach their theoretical maximum ofCkv/C′
kv = 1.60×for r256. Standard SDPA suffices; no
custom Flash Attention kernels are required. Prefill roofline analysis is in Section 12.
5 Related Work
Multi-Query, Grouped-Query, and Multi-Latent Attention.MQA (Shazeer, 2019) and
GQA (Ainslie et al., 2023) reduce KV cache by sharing KV heads across query groups. MLA (Liu
et al., 2024) projects the joint KV state into a low-dimensional latent. These methods reduce head
count or joint KV dimensionality; our approach reducesper-head key dimensionalityand composes
with all three (Table 17 in Section 9). Chen et al. (2025) show commonly used GQA configurations
are suboptimal for long contexts.
Low-Rank Attention and KV Cache Compression.Linformer (Wang et al., 2020) reduces
the sequence dimension; we reduce the feature dimension. Loki (Kang et al., 2024) exploits low-rank
key structure for sparse attention. LRQK (Li et al., 2025) jointly decomposes QK matrices during
prefill. SVD-LLM (Wang et al., 2024) applies truncation-aware SVD for post-training compression.
KVQuant (Hooper et al., 2024) and AsymKV (Tao et al., 2024) reduce bit width; ZACK (Zhang et al.,
2025) achieves zero-overhead dimensionality compression. Haris and Onak (2025) proveΩ(logn )
space lower bounds for attention. Our dimensionality reduction is orthogonal to quantization and
composes multiplicatively.
Efficient Attention.Flash Attention (Dao et al., 2022; Shah et al., 2024) optimizes memory
access patterns; sparse attention methods (Child et al., 2019; Beltagy et al., 2020; Mazaré et al.,
2025) reduce the attended set. These are complementary to reducing QK dimensionality.
10

<!-- page 11 -->

6 Discussion
Limitations.We validate up to 7B parameters over 20B tokens (tokens-to-parameters∼3). At
truly Chinchilla-optimal budgets (∼140B tokens), a modest cost may emerge as observed at smaller
scales. While we evaluate perplexity and five downstream tasks, other capabilities (in-context
learning, instruction following, long-context retrieval) may exhibit different sensitivity to QK
dimensionality.
Flash Attention and composability.Most Flash Attention implementations assumedQK
head =
dV
head, but standard SDPA already achieves 25–44% decode speedups (Section 12)—decode is
bandwidth-bound, so standard kernels suffice. Thin keys compose with KV quantization (Liu et al.,
2024; Tao et al., 2024): dimensionality reduction removes low-rank redundancy, then quantization
compresses remaining elements, yielding up to16×combined key cache compression (4×from thin
keys×4×from INT4).
Deployment paths.We identify three complementary paths, in order of increasing investment.
(1) K-only SVD atdmodel/2: 50% key cache savings, zero retraining. (2) SVD + QK fine-tuning
to dmodel/4: 75% savings at ∼2% cost. (3) Training from scratch with thin keys for future
architectures—analogous to how GQA was adopted into LLaMA-2 onward.
7 Conclusion
We identify a fundamental asymmetry in attention—selection requires onlyO(logN )dimensions
while value transfer needs fulldmodel—and derivefactored keys, a zero-cost inference primitive
that shrinks the key cache of any deployed model via SVD: no retraining, no calibration data, no
prefill overhead. At 7B scale over 20B tokens (2 seeds), thin keys match full-attention quality (9.24
vs9 .25PPL) with task parity on downstream benchmarks, while training 8% faster. For existing
models, SVD + QK fine-tuning achieves 75% key cache savings at∼2% cost across a 58×scale
range (GPT-2 to Mistral-7B), with 25–44% faster decode on H100. The approach composes with
GQA and quantization for up to16×combined compression.
References
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. GQA: Training
generalized multi-query transformer models from multi-head checkpoints. InEMNLP, 2023.
Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer.arXiv
preprint arXiv:2004.05150, 2020.
Chen, Y., Wu, Y., Song, C., Thai, Z. L., Shen, X., Han, X., Liu, Z., and Sun, M. Cost-optimal
grouped-query attention for long-context modeling. InEMNLP, 2025.
Child, R., Gray, S., Radford, A., andSutskever, I. Generatinglongsequenceswithsparsetransformers.
arXiv preprint arXiv:1904.10509, 2019.
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. FlashAttention: Fast and memory-efficient
exact attention with IO-awareness. InNeurIPS, 2022.
11

<!-- page 12 -->

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional
transformers for language understanding. InNAACL, 2019.
Frankle, J. and Carlin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks.
InICLR, 2019.
Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami,
A. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In
NeurIPS, 2024.
Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami,
A. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In
NeurIPS, 2024.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA:
Low-rank adaptation of large language models. InICLR, 2022.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand,
F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7B.arXiv preprint arXiv:2310.06825,
2023.
Johnson, W. B. and Lindenstrauss, J. Extensions of Lipschitz mappings into a Hilbert space.
Contemporary Mathematics, 26:189–206, 1984.
Kang, J., et al. Loki: Low-rank keys for efficient sparse attention.arXiv preprint arXiv:2406.02542,
2024.
Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The efficient transformer. InICLR, 2020.
Li, T., Zhou, G., Zhao, X., Qiu, Y., and Zhao, Q. Efficient low rank attention for long-context
inference in large language models. InNeurIPS, 2025.
Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., and Hu, X. KIVI: A
tuning-free asymmetric 2bit quantization for KV cache. InICML, 2024.
Liu, A., Feng, B., Wang, B., et al. DeepSeek-V2: A strong, economical, and efficient mixture-of-
experts language model.arXiv preprint arXiv:2405.04434, 2024.
Ma, J., Zhou, H., Lin, J., Wang, Z., Hu, J., and Han, S. Towards economical inference: Enabling
DeepSeek’s MLA in any transformer.arXiv preprint arXiv:2502.14837, 2025.
Mazaré, P.-E., Szilvasy, G., Lomeli, M., Massa, F., Murray, N., Jégou, H., and Douze, M. Inference-
time sparse attention with asymmetric indexing.arXiv preprint arXiv:2502.08246, 2025.
Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. InICLR, 2017.
Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by
generative pre-training. 2018.
Shah, J., et al. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision.
arXiv preprint arXiv:2407.08608, 2024.
Shazeer, N. Fast transformer decoding: One write-head is all you need.arXiv preprint
arXiv:1911.02150, 2019.
12

<!-- page 13 -->

Tao, Q., et al. AsymKV: Enabling 1-bit quantization of KV cache with layer-wise asymmetric
quantization configurations.arXiv preprint arXiv:2410.13212, 2024.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal,
N., Hambro, E., Azhar, F., et al. LLaMA: Open and efficient foundation language models.arXiv
preprint arXiv:2302.13971, 2023.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and
Polosukhin, I. Attention is all you need. InNeurIPS, 2017.
Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear
complexity.arXiv preprint arXiv:2006.04768, 2020.
Wang, X., et al. SVD-LLM: Truncation-aware singular value decomposition for large language
model compression. InICLR, 2025.
Zhang, Y., Hu, Y., Zhao, R., Lui, J. C. S., and Chen, H. Unifying KV cache compression for large
language models with LeanKV.arXiv preprint arXiv:2412.03131, 2024.
Zhang, Z., et al. ZACK: Zero-overhead LLM inference acceleration via dimensionality compression
of the key-value cache.arXiv preprint arXiv:2408.04107, 2025.
Haris, A. and Onak, K. Compression barriers for autoregressive transformers.arXiv preprint
arXiv:2502.15955, 2025.
8 Small-Scale Experiments
Experiments 1–4 use a standard transformer decoder with pre-norm layer normalization, GELU
activations, and learned positional embeddings. The only variable across configurations isdselect.
8.1 Experiment 1: Positional Selection (Copy-Back Task)
Setup.We design a task where each token must copy the token fromK = 8positions earlier:
yt = xt−K. The source position is fixed regardless of token content, isolating purelypositional
selection. Sequences of length 64 are generated with random tokens from a vocabulary of 16. The
model usesdmodel = 64, 4 heads, 2 layers.
Results.Table 12 shows that even dselect = 4(1 dimension per head) achieves 100% accuracy,
though convergence is slightly slower. This confirms that positional selection—learning to attend to
a fixed offset—requires minimal dimensionality.
8.2 Experiment 2: Content-Based Selection (Key-Value Retrieval)
Setup.We construct sequences of 8 random key-value pairs from a vocabulary of 16 tokens,
followed by a query key. The model must output the value associated with the queried key. Positions
are randomized each batch, so positional information is useless; the model must match content.
Architecture:d model = 64, 4 heads, 4 layers.
13

<!-- page 14 -->

Table 12: Copy-back task: accuracy and convergence bydselect.
dselect dselect/head Best Accuracy Converge Epoch
4 1 100.0% 300
8 2 100.0% 200
16 4 100.0% 200
32 8 100.0% 200
64 16 100.0% 200
Results.Table 13 reveals a sharp transition betweendselect = 4and dselect = 8. With 1 dimension
per head, keys map to scalars and cannot be reliably separated via dot product, achieving only
65.2% accuracy. With 2 dimensions per head, keys can point in distinct angular directions, enabling
perfect matching. TheO(log2N)prediction gives a lower bound: 16 keys requirelog2(16) = 4
dimensions total, but the model needs a small constant factor above this minimum to learn reliable
separation via gradient descent. In practice,dselect≈2 log2Nappears sufficient.
Table 13: Key-value retrieval: accuracy and convergence bydselect.
dselect dselect/head Best Accuracy Converge Epoch
4 1 65.2% did not converge
8 2 100.0% 1900
16 4 100.0% 1900
32 8 100.0% 1400
64 16 100.0% 1000
8.3 Experiment 3: WikiText-2 Language Modeling
Setup.We train causal language models on WikiText-2 (Merity et al., 2017) ( 2M training tokens)
withdmodel = 256, 8 heads, 6 layers,dff = 1024, sweepingdselect∈{8, 16, 32, 64, 128, 256}. Word-level
tokenization with vocabulary size∼29K (min_freq = 2). Training uses AdamW with cosine learning
rate schedule for 30 epochs.
Results.Table 14 shows that dselect = 64(dmodel/4) achieves a test perplexity of 122.24, essentially
identical to the baseline of 122.22. Even dselect = 32loses only 1.3%. Notably, dselect = 128
outperformsthe baseline (120.76 vs 122.22), likely because reducing QK capacity acts as a regularizer
in this heavily overfitting regime (train PPL 37 vs validation PPL 127).
Connecting to theO(logN )prediction.The vocabulary contains ∼29K words, yetdselect = 64
(8 dimensions per head) suffices. This implies that the effective number of selection patternsN is far
smaller than the vocabulary size. Intuitively, attention does not need to distinguish every individual
word from every other—it needs to distinguishcategoriesof tokens relevant to different attention
patterns: syntactic roles (subject vs. object vs. modifier), semantic clusters (topic-related words),
positional patterns (recent vs. distant tokens), and punctuation or structural cues. The number of
such categories is in the hundreds, not the tens of thousands. With 8 heads each operating in 8
dimensions, the model can represent∼28 = 256distinguishable patterns per head, which appears to
14

<!-- page 15 -->

be sufficient for the selection demands of language modeling. This is consistent with the observation
that attention heads tend to specialize into a modest number of interpretable roles (positional,
syntactic, rare-word) rather than implementing vocabulary-sized lookup tables.
Table 14: WikiText-2 results. Model:dmodel = 256, 8 heads, 6 layers. Baselinedselect = 256.
dselect dselect/head Val PPL Test PPL∆PPL QK Params QK Saved
8 1 133.78 126.48 +3.5% 24,672 97%
16 2 132.67 125.49 +2.7% 49,344 94%
32 4 130.51 123.78 +1.3% 98,688 87%
64 8 129.34 122.24 +0.0% 197,376 75%
128 16 126.42 120.76−1.2% 394,752 50%
256 32 126.95 122.22 — 789,504 0%
8.4 Experiment 4: WikiText-103 Language Modeling
Setup.To eliminate overfitting as a confound, we repeat the experiment on WikiText-103 (Merity
et al., 2017) ( 100M training tokens) with the same architecture. Word-level tokenization with
min_freq = 200yields a vocabulary of∼22K. We train for 10 epochs with batch size 64 and 2000
warmup steps.
Results.Table 15 shows the clean comparison. With 50 ×more training data, the model is
capacity-limited (train PPL≈val PPL), eliminating the regularization confound. Heredselect = 128
(dmodel/2) incurs only a 2.1% perplexity increase with 50% QK savings, anddselect = 64( dmodel/4)
incurs 4.3% for 75% savings—larger than on WikiText-2, confirming that the WikiText-2 result
was partly masked by overfitting. Nevertheless, the tradeoffs remain compelling: the relationship
between dselect and perplexity is smooth and monotonic, allowing practitioners to choose their
operating point on a clear Pareto frontier.
Table 15: WikiText-103 results. Model:dmodel = 256, 8 heads, 6 layers. Baselinedselect = 256.
dselect dselect/head Val PPL Test PPL∆PPL QK Saved
32 4 38.38 — +7.6% 87%
64 8 37.22 — +4.3% 75%
128 16 36.42 35.80 +2.1% 50%
256 32 35.67 — — 0%
9 Architecture Generalization at 125M Scale
9.1 Experiment 6: Architecture Generalization (LLaMA 125M)
Motivation.All training experiments so far use 10M-parameter vanilla transformers. We validate
on a faithful LLaMA implementation at 125M parameters—12.5×larger.
15

<!-- page 16 -->

Setup.We implement a standard LLaMA architecture with RMSNorm, SwiGLU FFN, Rotary
Position Embeddings (RoPE), no bias terms, pre-norm residuals, and tied embeddings (Touvron
et al., 2023). Theonlymodification: WQ and WK project todselect dimensions; WV remains dmodel.
Whendselect =dmodel, the model is exactly standard LLaMA. Configuration:dmodel = 768, 12 heads,
12 layers,dff = 2048,∼101.7M parameters at baseline. Training: WikiText-103 with vocabulary
truncated to 22K tokens (min frequency 200), 5 epochs, batch size 64, sequence length 512, cosine
schedule with 2000-step warmup.
Table 16: LLaMA 125M with asymmetric attention on WikiText-103. Comparison with 10M vanilla
transformer from Experiment 4.
dselect dselect/head Params Val PPL∆PPL QK saved
768 (full) 64 101.7M 22.80 — 0%
192 (dmodel/4) 16 91.1M 23.77 +4.3% 75%
96 (dmodel/8) 8 89.3M 24.45 +7.2% 87%
48 (dmodel/16) 4 88.4M 25.30 +11.0% 94%
Results.The degradation ratios are remarkably consistent across architectures and scales:
dselect 10M Vanilla (Exp. 4) 125M LLaMA (Exp. 6)
dmodel/4+4.3% +4.3%
dmodel/8+7.6% +7.2%
The +4.3% cost atdselect =dmodel/4is identical across architectures despite a 12.5×scale differ-
ence and completely different design choices (LayerNorm vs RMSNorm, GELU vs SwiGLU, learned
positions vs RoPE). This strongly suggests the QK dimensionality requirement is a fundamental
property of the attention mechanism.
Comparison with alternative KV compression methods.We train the same 125M LLaMA
architecture with Grouped-Query Attention (GQA) (Ainslie et al., 2023) and Multi-Latent Attention
(MLA) (Liu et al., 2024). All models are trained from scratch with identical hyperparameters.
Table 17: 125M LLaMA: comparison of KV compression methods trained from scratch on WikiText-
103. KV budget = per-token, per-layer cache size (K + V dimensions stored).
Method Config Params KV budget KV saved Test PPL
MHA 12 heads 101.7M 1536 0% 23.07
Thin keysd select = 38494.6M 1152 25% 23.22 (+0.7%)
Thin keysd select = 19291.1M 960 37.5% 24.09 (+4.4%)
GQA 6 KV heads 94.6M 768 50% 23.15 (+0.3%)
GQA 4 KV heads 92.3M 512 66.7% 23.32 (+1.1%)
MLAd c = 768108.8M 768 50% 23.11 (+0.2%)
MLAd c = 512101.7M 512 66.7% 23.23 (+0.7%)
16

<!-- page 17 -->

Table 17 shows that all three approaches achieve strong compression with modest quality costs.
Thin keys withdselect = 384achieves 23.22 test PPL—matching GQA-4 and MLA-512 quality
despite compressing only keys. The practical implication is that thin keyscomposewith GQA or
MLA for further savings.
10 Additional Analysis
10.1 Scaling of Minimumd select
Our experiments reveal a consistent pattern in the minimumdselect required for different selection
tasks:
Table 18: Minimum effectivedselect scales with task complexity.
TaskN effective Mind select/head Prediction (log 2N)
Positional (copy-back)∼10 offsets 1log 2 10≈3
Content (16 keys) 16 keys 2log 2 16 = 4(total)
Language (WikiText)∼256 patterns 8log 2 256 = 8
The empirical results are consistent with theO(logN )prediction. For language modeling, the
effectiveN appears to be approximately 256, suggesting that attention selection operates over a few
hundred distinct semantic/syntactic categories rather than the full vocabulary space. This aligns
with recent findings that key vectors naturally lie in a significantly lower-dimensional space than
the model dimension (Kang et al., 2024).
10.2 Overfitting Masks the Effect
The WikiText-2 vs WikiText-103 comparison reveals an important methodological point: on small
datasets, reducingdselect canappearcostless (or even beneficial) because the model is overfitting.
The WikiText-2 baseline has a train/val PPL ratio of 3.4×, indicating massive overfitting. Removing
QK capacity acts as implicit regularization. On WikiText-103, where the model is underfitting
(train PPL>val PPL), the true cost of reduceddselect becomes visible.
This suggests that results reported only on small benchmarks may overstate the losslessness of
QK compression, and that large-scale experiments are essential.
11 GSM8K Fine-tuning Progression
Table 19 shows the full progression of GSM8K recovery across fine-tuning experiments. Fine-tuning
on out-of-domain data (WikiText-103, C4) yields GSM8K scores in the 22–32% range with large
compression gaps. Domain-matched fine-tuning on GSM8K’s own training split (Experiment F3,
7,473 chain-of-thought examples,∼1.5M tokens) more than doubles scores and closes the compression
gap for r=256 from−13.7% to just−1.2%.
This demonstrates that the GSM8K degradation was a fine-tuning data problem, not a funda-
mental compression limitation. Data quality and domain match matter far more than data volume:
1.5M tokens of in-domain CoT outperform 10M tokens of generic web text.
17

<!-- page 18 -->

Table 19: GSM8K exact-match accuracy across fine-tuning experiments. All models use the same
QK-only fine-tuning protocol (3 epochs, lr=5×10−5). “Control” = uncompressed model with
identical fine-tuning.
Exp FT Data Control r512 r256∆ 512 ∆ 256
– None (baseline) 38.4 33.8 16.5 – –
A WikiText-103 29.9 27.7 25.8−7.4−13.7
F C4 (10M tok) 30.6 29.4 22.8−3.9−25.5
F2 C4 + Math (10M tok) 31.7 28.2 24.1−3.5−7.6
F3 GSM8K CoT (1.5M tok)53.7 52.5 52.0−0.7−1.2
12 Prefill Roofline and Flash Attention
Prefill roofline.Prefill differs from decode: the fullQK⊤product is computed over the entire
prompt. At context length s = 4096with Mistral-7B, a single layer’s attention FLOPs are
∼137GFLOPs while KV reads are∼2MB—yielding arithmetic intensity of∼68,000 FLOP/byte,
well above the H100’s ridge point. Prefill attention is thus compute-bound. Reducingdk from 128
to 32 cutsQK⊤FLOPs by4×per head. Standard FlashAttention-2 assumesdk =dv for shared tile
sizes; FlashAttention-3 (Shah et al., 2024) introduces more flexible tiling that could accommodate
asymmetric dimensions. Standard SDPA withenable_math=True already achieves 6–12% prefill
speedups.
18
