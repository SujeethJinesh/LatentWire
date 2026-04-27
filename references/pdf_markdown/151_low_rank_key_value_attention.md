# references/151_low_rank_key_value_attention.pdf

<!-- page 1 -->

Low-Rank Key Value Attention
James O’Neill†1 Robert Clancy 1 Mariia Matskevichus 1 Fergal Reid 1
Abstract
The key-value (KV) cache is a primary memory
bottleneck in Transformers. We propose Low-
Rank Key-Value (LRKV) attention, which re-
duces KV cache memory by exploiting redun-
dancy across attention heads, while being com-
pute efficient. Each layer uses a shared full-
rank KV projection augmented with low-rank,
head-specific residuals, providing a continuous
trade-off between complete sharing and full in-
dependence. After pretraining models of size
128M to 6.3B parameters, LRKV consistently
achieves the lowest test loss among standard
MHA, MQA/GQA, and MLA while using only
45-53% of MHA’s KV cache. LRKV reaches
equivalent baseline quality 18-25% faster (mea-
sured in training steps). After supervised midtrain-
ing, LRKV achieves the highest downstream task
performance across ARC-Easy, ARC-Challenge,
MMLU, GSM8K, and HumanEval benchmarks.
1. Introduction
Transformers are the dominant architecture for large-scale
sequence modeling in language, vision, and multimodal do-
mains (Vaswani et al., 2017; OpenAI, 2023), but as their size,
sequence length, and context window grow, so does, rapidly,
their computational and memory costs. KV-caching, which
stores the attention key and value representations, is a pri-
mary contributor to this overhead, as it spans every attention
layer, and scales linearly with sequence length and count of
heads. The cumulative KV footprint of modern models with
tens of billion parameters can exceed the parameter mem-
ory itself, especially for long-context inference (Dao et al.,
2024; Child et al., 2019). To illustrate the scale of this chal-
lenge, consider a 2.5B parameter model (18 layers, 18 heads,
dh=128) serving 8K-token contexts: the KV cache alone
requires 2×18×18×128×8192×2bytes= 648MB per re-
quest. At batch size 4, KV cache memory (2.6 GB) exceeds
1 AI Group, Intercom
124 St Stephen’s Green, Dublin 2, D02 C628, Ireland . Correspon-
dence to: James O’Neill<james.oneill@intercom.io>.
Preprint. April 9, 2026.
the model’s parameter memory (5 GB in bfloat16), consum-
ing more memory than the model itself. This forces prac-
titioners into an impossible trilemma: use smaller batches
(sacrificing throughput), shorter contexts (limiting capa-
bility), or frequent cache eviction (adding latency)—each
option directly degrading production utility. Reducing KV
cache size thus directly translates to 2× longer contexts or
2×larger batches at fixed memory budgets.
A range of approaches have been proposed to alleviate the
growing KV cache cost. Multi-Query Attention (Shazeer,
2019, MQA) and Grouped-Query Attention (Ainslie et al.,
2023, GQA) reduce memory and latency by sharing key-
value (KV) projections across heads or groups of heads,
and are now standard in large-scale models such as PaLM
and LLaMA (Touvron et al., 2023; 2024). However, ag-
gressive KV sharing introduces a fundamental trade-off:
while memory is reduced, head-level representational diver-
sity is constrained, even though distinct attention heads are
known to encode complementary syntactic and semantic
patterns (Clark et al., 2019; Michel et al., 2019).
At the same time, empirical studies and recent spectral anal-
yses show that attention heads are not fully independent:
head-specific KV projections are highly correlated and oc-
cupy overlapping subspaces, indicating substantial redun-
dancy (Yunis et al., 2024). Crucially, this redundancy is
structured rather than uniform—small, head-specific varia-
tions remain important for capturing nuanced dependencies.
This raises a natural question:can KV memory be reduced
by exploiting redundancy across heads, without collapsing
the specialization that makes multi-head attention effective?
Beyond KV sharing, efficiency research has pursued com-
plementary directions such as sparse or kernelized atten-
tion (Wang et al., 2020; Beltagy et al., 2020; Child et al.,
2019), architectural optimizations (Dao et al., 2024), and
latent compression methods including Multi-Latent Atten-
tion (Liu et al., 2024, MLA). However, none of these ap-
proaches explicitly resolve the duplication of per-head key
and value representations that dominates the KV memory
footprint in large models.
In this work, we proposeLow-Rank Key-Value Atten-
tion (LRKV), a simple modification of multi-head atten-
tion that exploits structured redundancy across heads while
preserving head specialization. Each Transformer layer
1
arXiv:2601.11471v3  [cs.LG]  7 Apr 2026

<!-- page 2 -->

Low-Rank Key Value Attention
Figure 1.Comparison of attention mechanisms.Standard MHA:Uses H independent projections per head, requiring high cache cost
(2LHd h).MQA/GQA:Shares projections across all heads, reducing cache but losing head-specific detail.LRKV (Ours):Combines
shared full-rank parameters with head-specific low-rank residuals (UhBT
h , rank r≪d h), achieving low cache cost 2L(dh +Hr) while
preserving head diversity.
maintains a shared, full-rank key and value projection that
serves as a global basis, while each head learns a compact,
trainable low-rank residual that captures head-specific de-
viations. The shared component encodes global relational
structure, and the residuals restore the localized diversity
otherwise lost under aggressive KV sharing. This factoriza-
tion substantially reduces KV memory while retaining the
expressivity of multi-head attention.
LRKV differs fundamentally from prior KV-efficient de-
signs in where compression is applied. Unlike LoRA (Hu
et al., 2022), which introduces low-rank structure post hoc
during fine-tuning, LRKV learns head-specific low-rank
residuals jointly during pretraining. Unlike factorized QKV
methods that reduce rank along the token or hidden di-
mension (Wang et al., 2020; Saxena et al., 2024; Chang
et al., 2025), LRKV preserves full token resolution and com-
presses only across attention heads. We now discuss the
main contributions of this paper.
Contributions.(1) LRKV reduces KV cache by 45-53%
while achieving lowest test loss across 128M-6.3B mod-
els, reaching baseline performance 18-25% faster in train-
ing steps. (2) LRKV achieves highest downstream perfor-
mance across ARC, MMLU, GSM8K, demonstrating better
pretraining translates to improved capabilities. (3) Gauge-
invariant analysis shows LRKV effectively uses low-rank
structure by preserving 93.5% head diversity vs 94% MHA.1
2. Related Work
KV-Sharing Mechanisms.MQA and GQA reduce KV
cache by sharing projections across heads, but sacrifice
1Code and trained models will be released upon acceptance.
head-level expressivity. MQA (Shazeer, 2019) uses a sin-
gle shared KV projection for all heads, achieving maximal
cache reduction but constraining diversity. GQA (Ainslie
et al., 2023) groups heads to share KV projections within
groups, interpolating between MQA and full MHA. While
both methods reduce memory, they fundamentally limit the
representational capacity available to each head. LRKV pre-
serves shared projection efficiency while restoring diversity
through structured low-rank head residuals, achieving better
quality at comparable cache sizes.
Latent Compression Approaches.MLA (Liu et al., 2024)
takes a complementary approach: rather than sharing across
heads, it compresses across the token dimension by project-
ing inputs into a low-dimensional latent space ( dc ≪d )
before caching. While effective for extreme compression,
this bottleneck constrains all heads to operate through the
same latent representation and requires T-dependent recon-
struction overhead during generation (see Appendix F).
LRKV instead preserves full token-level resolution and com-
presses along the head dimension via additive factorization
(Wh =W shared +U hB⊤
h ), avoiding latent bottlenecks
while maintaining per-head specialization. These represent
orthogonal design choices: MLA optimizes for minimal
cache size via aggressive token compression, while LRKV
balances memory efficiency with head-level expressivity.
Low-Rank Parameterization.LoRA (Hu et al., 2022)
introduces low-rank updates for parameter-efficient fine-
tuning, intervening on the optimization pathway rather than
the cached representations. LRKV applies the low-rank prin-
ciple directly to KV projections during pretraining, shaping
the structure of cached features from the start. Recent work
explores low-rank QKV factorizations (Xie et al., 2023;
Khalaf et al., 2025; Lv et al., 2024) to reduce parameters or
2

<!-- page 3 -->

Low-Rank Key Value Attention
computation; LRKV differs by preserving a full-rank shared
base (capturing global structure) while using low-rank resid-
uals only for head-specific deviations. Concurrent work
on factorized KV mechanisms (MFA/MFA-KR (Hu et al.,
2025), TPA (Zhang et al., 2025)) explores related structured
compression ideas.
Unlike joint-head low-rank decompositions such as J-
LRD/Palu, which compress projections post hoc, LRKV is
a pretraining-time architectural reparameterization that pre-
serves a full-rank shared base while learning head-specific
low-rank residuals. This enables exact associative decoding
and reduces KV cache without post-hoc approximation.
Complementary Efficiency Methods.LRKV is orthogo-
nal to other efficiency techniques: sparse/kernelized atten-
tion (Child et al., 2019; Beltagy et al., 2020; Wang et al.,
2020) reduce computational complexity but not KV cache
size; architectural optimizations like FlashAttention (Dao
et al., 2024) improve throughput through better memory
access patterns; quantization and pruning reduce precision
or parameters. For head diversity analysis, we extend prior
metrics (Michel et al., 2019; Kornblith et al., 2019; Wang
& Wang, 2025) using gauge-invariant bilinear forms with
centered Gram matrix analysis (see Appendix A).
3. Methodology
Transformers represent each token in a sequence through
queries, keys, and values that interact via scaled dot-product
attention. For an input matrix X∈R L×d, the h-th attention
head computes Qh =XW Q
h , Kh =XW K
h , and Vh =
XWV
h , where WQ,K,V
h ∈R d×dh and dh =d/H for H
heads. The head output is
Oh = softmax
 QhK⊤
h√dh

Vh,(1)
and outputs from all heads are concatenated and projected
to form the layer output.
KV caching.During autoregressive decoding, the KV cache
stores Kh and Vh for all previous tokens and heads, incur-
ring per-layer memory Mstandard = 2LHd h = 2Ld. Equiva-
lently, per token the cache stores2Hd h floating-point values
(keys and values acrossH heads). Over N layers in bfloat16
precision, this is2N·(2LHd h)bytes of KV storage.
Motivation.Empirical studies show attention heads within
a layer are often correlated (Clark et al., 2019; Michel et al.,
2019), suggesting that per-head key/value features contain
substantial redundancy. Existing KV-sharing methods such
as MQA and GQA reduce cache size by sharing K/V across
heads, but can reduce head diversity and modeling capacity.
Our goal is to reduce redundant KV storage while preserving
head-specific flexibility.
Low-Rank KV Attention.We parameterize each head’s
key/value projection as a shared dense (unconstrained) base
plus a head-specific low-rank residual, while keeping the
resulting projection shaped×d h:
WK
h =W K
shared +U K
h BK
h
⊤
,(2)
WV
h =W V
shared +U V
h BV
h
⊤
,(3)
where WK,V
shared ∈R d×dh are dense projection matrices with
no rank constraint, UK,V
h ∈R d×r, BK,V
h ∈R dh×r, and
r≪d h. The effective keys and values are Kh =XW K
h
and Vh =XW V
h . When r= 0 , LRKV reduces to com-
plete KV sharing (MQA-style) within a layer; increasing r
interpolates continuously toward standard MHA.
Training and initialization.All parameters in Equation 3
are optimized jointly during pretraining. Gradients from
the shared and residual paths are additive, allowing the
model to learn how much per-head variation is required.
We initialize WK,V
shared using standard Kaiming initializa-
tion for attention projections. The per-head factors UK,V
h
and BK,V
h are initialized to small random values scaled by
1/√r such that initial residuals UhBh
⊤ have magnitude
≈0.1× ∥W shared∥F , ensuring the model starts close to
the shared baseline and gradually learns head specialization.
During training, we materialize fullKh and Vh matrices be-
fore attention computation for simplicity; the factored form
is used only during inference to realize memory savings.
This is an exact reparameterization, so the factorized and
unfactorized forms are mathematically equivalent. We apply
low-rank factorization only to key and value projections, as
these are cached during inference; query projections remain
independent per head at full rank.
LRKV caching scheme.During decoding, the bottleneck
is storing Kh,V h ∈R L×dh for every head. LRKV instead
caches shared features once per layer,
Kshared =XW K
shared,V shared =XW V
shared ∈R L×dh ,
and compact per-head latents
RK
h =XU K
h ,R V
h =XU V
h ∈R L×r.
The full per-head features implied by Equation 3 are
Kh =K shared +R K
h BK
h
⊤
,(4)
Vh =V shared +R V
h BV
h
⊤
.(5)
Importantly, the matrices BK,V
h are model parameters and
arenotcached per token.
Attention computation without explicit reconstruction.
Naively materializing Kh,V h for all cached tokens would
cost O(Lrdh) memory and compute. Instead, we exploit
3

<!-- page 4 -->

Low-Rank Key Value Attention
associativity to compute attention exactly without forming
full per-head KV tensors. For a decoding step with query
qh ∈R dh,
qhK⊤
h =q hK⊤
shared + (qhBK
h ) (RK
h )⊤,(6)
and for attention weightsa h ∈R 1×L,
ahVh =a hVshared + (ahRV
h )B V
h
⊤
.(7)
Equations 6 and 7 compute theexactattention logits and
outputs implied by Equation 4–Equation 5, but avoid explicit
reconstruction of Kh,V h. This form can be implemented
inside fused attention kernels. See Table 4 for a detailed
mathematical comparison to existing mechanisms.
Compatibility with Positional Embeddings.Positional
embeddings, such as RoPE, are applied after projection and
distribute linearly over the LRKV decomposition:
RoPE(Kh) = RoPE(Kshared) + RoPE(Rh)B⊤
h .
Both components are rotated once at caching time, as in stan-
dard KV caching, and reused during decoding. This intro-
duces no additional memory or sequence-length-dependent
computation. In contrast to latent-attention approaches such
as MLA, which require partial RoPE to preserve projection
absorption, LRKV supports full-dimension RoPE without
modification.
KV cache memory complexity (during decoding).Stan-
dard attention stores per-head keys and values:
Mstandard = 2LHd h.
LRKV stores shared features plus per-head latents:
MLRKV = 2Ld h|{z}
shared(K,V)
+ 2LHr|{z}
per-head latents
= 2L(dh +Hr).(8)
Thus the memory ratio is
MLRKV
Mstandard
= dh +Hr
Hd h
= 1
H + r
dh
.(9)
Per token, LRKV stores 2dh + 2Hr values versus 2Hd h
for standard attention.
Compute overhead during decoding (FLOPs).Per decod-
ing step, standard attention has dominant per-head computa-
tional cost O(Ldh) from query–key dot products and value
aggregation. Using the distributed forms in Equation 6 and
Equation 7, LRKV adds an extra ∆FLOPs =O(Lr+rd h)
per head per decoding step. For long contexts (L≫1 ), the
dominant additional term scales as O(Lr), yielding a rela-
tive overhead of approximately r/dh compared to standard
attention.
Memory bandwidth.Modern inference is often bandwidth-
bound rather than compute-bound (Dao et al., 2024). Per
decoding step, standard MHA reads2H cached tensors (Kh
and Vh for each head). LRKV reads two shared tensors plus
2H per-head latent tensors. Since the shared tensors (L×dh)
are reused across heads and the per-head latents (L×r ) are
substantially smaller (r≪d h), the total bytes transferred is
2L(dh +Hr) for LRKV versus 2LHd h for standard MHA,
matching the cache-size reduction and directly translating
memory savings into reduced bandwidth pressure.
Parameter complexity.Standard per-layer K/V parameters
scale as PMHA,KV = 2Hdd h. LRKV uses a shared base plus
low-rank factors:
PLRKV = 2dd h|{z}
shared(K,V)
+ 2Hr(d+d h)| {z }
per-head low-rank factors
,(10)
which may be lower or higher than standard depending on
(H, r, dh). In practice, we choose r to prioritize KV-cache
reduction with minimal quality impact; in our experiments,
LRKV remains competitive or superior even when the total
K/V parameter count is below that of standard MHA.
Rank as a control for diversity.The residual rank r
controls a continuous spectrum between fully shared KVs
(MQA-style, r= 0 ) and fully independent per-head projec-
tions (standard MHA, large r). The shared base WK,V
shared
captures globally useful features, while low-rank residu-
als enable head specialization within a constrained budget.
Empirically, we find r≈0.36 -0.43×d h provides a criti-
cal threshold: LRKV achieves 93.5% PCA-based effective
rank versus 94.0% for standard MHA at 2.5B scale (subsec-
tion 4.3), preserving nearly all head diversity while achiev-
ing substantial cache reduction. See Appendix B for detailed
discussion of spectral structure and low-rank factorization.
4. Experiments
We evaluate LRKV through large-scale pretraining exper-
iments across four model sizes (128M, 1.2B, 2.5B, 6.3B
parameters), comparing against standard MHA and state-
of-the-art KV-efficient baselines. We measure pretraining
loss, training efficiency, downstream task performance, and
provide detailed analysis of why LRKV preserves modeling
quality despite substantial cache reduction.
Experimental setup.We pretrain decoder-only Transform-
ers at four scales (128M, 1.2B, 2.5B, 6.3B) on FineWeb-
Edu (Penedo et al., 2024) for 100B tokens (128M) and 50B
tokens (others), comparing LRKV against Standard MHA,
GQA, MQA, and MLA. Models use 2048-token context,
Muon+AdamW optimizers (Sardana & Havens, 2024), and
are pretrained on 8×H200 GPUs in bfloat16. After pretrain-
ing, we perform supervised midtraining on 568K examples
from SmolTalk (Allal et al., 2024), MMLU (Hendrycks
4

<!-- page 5 -->

Low-Rank Key Value Attention
Table 1.Pretraining performance across model scales.LRKV achieves competitive test loss across all model scales (128M, 1.2B, 2.5B,
6.3B) while maintaining efficient KV cache usage.
Arch. KV Cache % 128M 1.2B 2.5B 6.3B
Model KV Heads 128M 1.2B 2.5B 6.3B CE↓BPB↓ CE↓BPB↓ CE↓BPB↓ CE↓BPB↓
Standard MHA 6/12/18/32 100 100 100 100 2.903 0.878 2.530 0.765 2.389 0.723 2.319 0.701
GQA 3/4/6/2 50 33 33 6 2.918 0.883 2.536 0.767 2.397 0.725 2.354 0.712
MQA 1 17 8 6 3 2.929 0.886 2.573 0.778 2.408 0.729 2.304 0.697
MLA – 8.3 8.3 8.3 12.5 2.901 0.876 2.564 0.775 2.392 0.724 2.377 0.719
LRKV – 53 48 48 45 2.893 0.872 2.509 0.758 2.376 0.719 2.288 0.692
KV Heads show values for 128M/1.2B/2.5B/6.3B scales. KV Cache % indicates memory relative to Standard MHA baseline (see Ap-
pendix G for profiling details). Test metrics on held-out FineWeb-Edu (100B tokens for 128M, 50B for 1.2B/2.5B/6.3B).
0 10 20 30 40 50 60 70
Training Compute (ExaFLOPs)
0 20 40 60 80 100
Training Tokens (Billions)
2.85
2.90
2.95
3.00
3.05
3.10
3.15
Test Cross-Entropy Loss
(a) 128M Parameters
Standard MHA
LRKV
MQA
GQA
MLA
0 50 100 150 200 250
Training Compute (ExaFLOPs)
0 10 20 30 40 50
Training Tokens (Billions)
2.50
2.55
2.60
2.65
2.70
2.75
2.80
2.85
2.90
Test Cross-Entropy Loss
(b) 1B Parameters
0 100 200 300 400 500 600 700
Training Compute (ExaFLOPs)
0 10 20 30 40 50
Training Tokens (Billions)
2.35
2.40
2.45
2.50
2.55
2.60
2.65
Test Cross-Entropy Loss
(c) 2.5B Parameters
0 250 500 750 1000 1250 1500 1750
Training Compute (ExaFLOPs)
0 10 20 30 40 50
Training Tokens (Billions)
2.3
2.4
2.5
2.6
2.7
2.8
Test Cross-Entropy Loss
(d) 6.3B Parameters
Figure 2.Cross-scale pretraining curves with dual-axis compute metrics.Test cross-entropy loss across four model scales (128M,
1.2B, 2.5B, 6.3B) plotted against training tokens (bottom x-axis) and cumulative training compute in ExaFLOPs (top x-axis). LRKV
demonstrates competitive performance across all scales, achieving the lowest test loss at 128M and 2.5B scales. Final results in Table 1.
et al., 2021), and GSM8K (Cobbe et al., 2021), then evalu-
ate on ARC-Easy, ARC-Challenge, MMLU, GSM8K, and
HumanEval. LRKV uses 45-53% of Standard MHA’s KV
cache (see Appendix D for details).
4.1. Pretraining Results
Final pretraining performance.Table 1 shows LRKV
achieves competitive test loss across all scales (128M, 1.2B,
2.5B, 6.3B), with the best performance at 128M and 2.5B
scales. At 6.3B scale, LRKV reaches 2.288 CE (0.692 BPB),
outperforming MHA (2.319 CE), MQA (2.304 CE), GQA
(2.354 CE), and MLA (2.377 CE). At 1.2B scale, LRKV
achieves 2.509 CE (0.758 BPB) with only 48% of MHA’s
cache—outperforming all baselines while using half the
memory. At 2.5B, LRKV achieves 0.719 BPB with only
48.4% of MHA’s cache—a strictly better accuracy-memory
tradeoff. Cache efficiency improves at larger scales (52.6%
→ 48% → 48.4% → 45.1%), making LRKV increasingly
attractive for large models.
Cross-scale training efficiency and compute normaliza-
tion.Figure 2 presents test performance across all four
model scales, measured in cross-entropy loss with dual-axis
compute metrics (training tokens and cumulative FLOPs).
LRKV demonstrates competitive performance across scales,
achieving the lowest test loss at 128M and 2.5B scales while
remaining highly competitive at 1.2B and 6.3B, showing
consistent advantages across three orders of magnitude in
model size. The top x-axis shows cumulative training com-
pute (ExaFLOPs), accounting for mechanism-specific per-
token costs (MQA -6%, GQA -3.2%, MLA -3.7%, LRKV
+0.8% at 2.5B scale). 2 When normalized by total com-
pute rather than token count, LRKV maintains its perfor-
mance advantage: at every ExaFLOP milestone, LRKV
achieves lower test loss than all baselines. At 2.5B scale,
LRKV reaches equivalent baseline performance 18–30%
faster (23.6% average across all methods), while no baseline
reaches LRKV’s final performance even after exhausting the
full compute budget. This establishes LRKV as dominant
in both sample efficiency and final quality. At the largest
6.3B scale, the performance gap widens: LRKV achieves
2.288 CE versus Standard MHA’s 2.319 CE (1.3% improve-
ment), demonstrating that LRKV’s architectural advantages
strengthen rather than diminish with scale.
Training efficiency analysis.Beyond superior converged
performance, LRKV demonstrates remarkable sample effi-
ciency at 2.5B scale (Figure 3). LRKV reaches each base-
line’s final validation performance 18-30% faster, averaging
23.6% training compute savingsacross all baselines while
2The reported per-token costs reflectinferenceFLOPs. During
training, LRKV materializes full K/V matrices (as do all methods),
so training cost per step is comparable to MHA. The 18-25%
“training compute savings” refer to sample efficiency (fewer steps
to reach target loss), not per-step speedup.
5

<!-- page 6 -->

Low-Rank Key Value Attention
Table 2.Downstream task performance after midtraining (128M, 2.5B, and 6.3B scales).LRKV achieves the highest combined
accuracy across all three scales (18.9%, 37.9%, 40.2%) on five diverse benchmarks, demonstrating that superior pretraining performance
translates to stronger downstream capabilities. Combined score (Comb.) is the average across all five benchmarks. HE = HumanEval.
128M 2.5B 6.3B
Model ARC-E ARC-C MMLU GSM8K HEComb.ARC-E ARC-C MMLU GSM8K HEComb.ARC-E ARC-C MMLU GSM8K HEComb.
Standard MHA26.4 26.0 27.2 1.0 8.8 17.9 66.6 47.1 39.3 10.213.4 35.3 72.7 53.6 42.7 10.6 14.6 38.8GQA 28.0 25.4 25.7 0.5 4.6 16.8 65.4 49.6 40.5 10.013.4 35.8 69.0 48.2 40.9 9.9 13.4 36.3MQA 28.0 27.1 27.8 1.3 2.4 17.3 65.2 47.3 40.2 10.3 3.7 33.3 67.2 47.0 40.5 8.2 4.3 33.4MLA 27.3 27.8 26.3 1.1 7.7 18.0 67.5 51.5 41.9 10.8 12.8 36.9 69.4 51.3 40.9 9.5 13.4 36.9LRKV 26.7 30.2 28.40.3 9.1 18.9 70.7 53.8 42.2 11.311.7 37.9 75.0 58.0 44.5 11.512.8 40.2
0 20 40 60 80 100
KV Cache Size (% of Standard MHA)
0.718
0.720
0.722
0.724
0.726
0.728
0.730Final Validation BPB (lower is better)
LRKV
(Best performance,
moderate memory)
(a) Memory vs Performance
0 20 40 60 80 100
KV Cache Size (% of Standard MHA)
0
5
10
15
20
25
30
Training Compute Savings (%)
(to reach baseline performance)
29.7%
21.4%
24.5%
18.7%
LRKV reaches each baseline's
final performance 18-30% faster
(b) Training Efficiency Advantage
MQA MLA GQA LRKV Standard MHA
Figure 3.LRKV achieves superior training efficiency alongside
best performance (2.5B scale). (a) Memory vs Performance:
Test BPB versus KV cache percentage for all methods. LRKV
achieves optimal trade-off with lowest BPB at 48.4% cache usage
(2.5B scale).(b) Training Efficiency Advantage:LRKV reaches
each baseline’s final test loss, quantifying training compute savings.
LRKV reaches all baselines’ performance earlier.
achieving better final performance. Critically, this reveals
an asymmetric advantage: LRKV reaches any baseline’s
performance target early in training, but no baseline reaches
LRKV’s final performance (0.719 BPB) even after the full
50B token budget.
Long Context Pretraining.We extend evaluation to 8192
token sequences using 512M parameter models trained for
50B tokens (153.6 ExaFLOPs). Figure 4 shows that LRKV
achieves 2.67 test loss, outperforming MHA (2.74) by 2.7%
while using only 48.2% of its KV cache. LRKV also sur-
passes GQA (2.70, -1.3%), MLA (2.71, -0.9%), and MQA
(2.73, -0.4%). Notably, all KV-efficient methods outperform
Standard MHA at long context, suggesting compression
provides implicit regularization benefits, though LRKV’s
architectural advantages remain pronounced. These results
validate LRKV’s effectiveness at extended sequence lengths.
4.2. Downstream Task Performance
To evaluate whether LRKV’s pretraining advantages trans-
late to practical capabilities, we perform supervised
midtraining on a diverse instruction-following dataset
(568K examples from SmolTalk (Allal et al., 2024),
MMLU (Hendrycks et al., 2021), and GSM8K (Cobbe et al.,
2021)) and evaluate on five standard benchmarks. Table 2
shows final downstream performance across three scales
(128M, 2.5B, 6.3B).
20 40 60 80 100 120 140
Training Compute (ExaFLOPs)
5 10 15 20 25 30 35 40 45 50
Training Tokens (Billions)
2.65
2.70
2.75
2.80
2.85
2.90
2.95
3.00
Test Cross-Entropy Loss
Standard MHA
LRKV
MLA
GQA
MQA
Figure 4.Long context pretraining for 512M parameter models.
Test cross-entropy loss curves for models trained with 8192-token
sequence length on 50B tokens (153.6 ExaFLOPs). LRKV outper-
forms Standard MHA by 2.7% while using 48.2% cache.
Scale-consistent superiority across task categories.
LRKV achieves the highest combined accuracy at all three
scales: 18.9% (128M), 37.9% (2.5B), and 40.2% (6.3B),
demonstrating consistent advantages across three orders of
magnitude in model size. At 2.5B scale, LRKV outperforms
Standard MHA (35.3%), GQA (35.8%), MQA (33.3%), and
MLA (36.9%), with particularly strong gains on knowledge-
intensive and reasoning tasks: ARC-Easy (+4.1pp over
MHA), ARC-Challenge (+6.7pp), MMLU (+2.9pp), and
GSM8K (+1.1pp). At 6.3B scale, LRKV achieves 40.2%
combined versus MHA’s 38.8%, with the largest improve-
ments on reasoning benchmarks (ARC-Challenge: 58.0%
vs 53.6%, MMLU: 44.5% vs 42.7%). One exception is
HumanEval at 6.3B scale, where MHA is slightly better,
possibly suggesting code generation may be more sensitive
to head specialization.
Notably, MQA shows catastrophic degradation on code
generation (HumanEval: 2.4% at 128M, 3.7% at 2.5B, 4.3%
at 6.3B versus 13.4%, 11.7-13.4%, and 12.8-14.6% for other
methods), confirming that complete KV sharing particularly
harms structured generation tasks requiring precise long-
range dependencies. LRKV avoids this pathology through
per-head residuals that preserve specialization.
Pretraining quality predicts downstream performance.
The strong correlation between pretraining BPB and down-
6

<!-- page 7 -->

Low-Rank Key Value Attention
20 40 60 80 100
KV Cache Size (% of Standard MHA)
2.87
2.88
2.89
2.90
2.91
2.92
2.93
2.94
2.95
Final Test CE Loss
r=8
r=16
r=32
r=64
d=128
d=192
d=384
Standard MHA
GQA
(3 groups)
MQA
(a) LRKV
MLA
Standard MHA
GQA (3 groups)
MQA
0 20 40 60 80 100
Training Tokens (Billions)
2.90
2.95
3.00
3.05
3.10
3.15
Test Cross-Entropy Loss
(b) LRKV Rank
r=8
r=16
r=32
r=64
r=128
Figure 5.LRKV rank ablation with performance-memory tradeoff analysis (128M, 100B tokens). (a)Final test cross-entropy loss
versus KV cache size (relative to Standard MHA) for LRKV rank ablations (r∈ {8,16,32,64,128} ), MLA latent dimension ablations
(d∈ {128,192,384} ), and baselines (Standard MHA, GQA with 3 groups, MQA). LRKV dominates the performance-memory tradeoff
space.(b)LRKV training dynamics across ranks, showing consistent convergence and monotonic improvement with increasing rank.
0 5 10 15
Head Index
0
2
4
6
8
10
12
14
16 Head Index
Standard MHA
0 5 10 15
Head Index
0
2
4
6
8
10
12
14
16 Head Index
LRKV (r=64)
0 5 10 15
Head Index
0
2
4
6
8
10
12
14
16 Head Index
GQA
0 5 10 15
Head Index
0
2
4
6
8
10
12
14
16 Head Index
MQA
0 5 10 15
Head Index
0
2
4
6
8
10
12
14
16 Head Index
MLA
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
Figure 6.Gauge-invariant head similarity matrices (2.5B scale).Heatmaps show pairwise similarities sij =
tr((WK
i )⊤WK
j (WQ
j )⊤WQ
i ) normalized by Frobenius norms for all 18 heads. Red indicates high similarity, blue indicates inde-
pendence. LRKV exhibits similarity structure nearly identical to Standard MHA with predominantly dark off-diagonal regions.
stream accuracy (R²=0.786 at 2.5B scale, see Appendix E)
confirms that architectural improvements generalizing
across pretraining data translate directly to task-specific
capabilities. LRKV’s 2.6 percentage point downstream ad-
vantage over Standard MHA at 2.5B scale (37.9% vs 35.3%)
stems directly from its superior pretraining performance
(0.719 vs 0.723 BPB). This validates that LRKV’s low-
rank factorization provides fundamental capacity gains that
manifest across both language modeling and downstream
evaluation, rather than overfitting to pretraining objectives.
4.3.Why Low-Rank Key-V alue Attention Works
This subsection provides an empirical analysis explain-
ing why LRKV preserves or exceeds the modeling qual-
ity of standard attention despite substantially reducing KV-
cache memory. LRKV’s design is motivated by a practi-
cal constraint: standard MHA duplicates K/V representa-
tions across H heads, causing memory cost to scale linearly
with head count. While prior analyses suggest attention
heads exhibit substantial redundancy (Michel et al., 2019;
Clark et al., 2019), heads also specialize for distinct syn-
tactic and semantic patterns (Zhang et al., 2023), imply-
ing that complete KV sharing (MQA) may degrade quality.
LRKV addresses this tension through additive factorization:
Wh =W shared +U hB⊤
h , which separates a full-rank
shared basis from compact per-head residuals. We now
examine three questions: (1) whether appropriate rank se-
lection is critical for quality, (2) whether LRKV preserves
architectural head diversity, and (3) whether the learned
factorization approaches mathematical optimality.
Analysis setup.For a given Transformer layer, we analyze
head-specific projection matrices {WQ
h ,W K
h ,W V
h }H
h=1
through their gauge-invariant bilinear forms on the final
pretrained checkpoint. Our novelty is not the bilinear form
itself, but its use for quantifying head diversity and the
query-compensation effect under KV sharing. For LRKV ,
we quantify geometric separation using cosine similarity
between shared and residual projections, and measure sub-
7

<!-- page 8 -->

Low-Rank Key Value Attention
0 2 4 6 8 10
Layer
1
2
3
4
5
6Effective Rank
128M Scale
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
Fully redundant
0 5 10 15 20 25 30 35
Layer
2.5
5.0
7.5
10.0
12.5
15.0
17.5Effective Rank
2.5B Scale
Figure 7.LRKV preserves head diversity across scales.Gauge-invariant effective rank shows LRKV with sufficient rank matches
Standard MHA at 128M, while at 2.5B LRKV achieves 98.3% vs 98.9% for MHA using 48.4% of KV cache.
space overlap via principal-angle-based metrics. To assess
head diversity, we use gauge-invariant similarity metrics
based on attention bilinear forms Ah =W Q
h (WK
h )⊤,
which are invariant to per-head rotations. We extend this
analysis using PCA in bilinear form space by centering
the Gram matrix G (where Gij =⟨A i, Aj⟩F ) via the
kernel PCA transformation (Smola & Sch ¨olkopf, 1998):
Gcentered =G−G row −G col +G mean. This removes the
mean bilinear form and reveals the intrinsic dimensionality
of head specialization. To assess optimality, we compare
LRKV’s learned decomposition to the mathematically op-
timal rank-r truncated SVD of standard MHA projections,
computed post-hoc.
Rank selection determines capacity and performance.
We first examine how residual rank r affects model-
ing quality through a systematic ablation study. Fig-
ure 5(b) shows training curves for LRKV with ranks r∈
{8,16,32,64,128} on 128M parameter models with 100B
tokens, demonstrating monotonic improvement as rank in-
creases: r= 128 achieves the best performance (CE=2.877,
BPB=4.156), while r= 8 shows the worst (CE=2.908,
BPB=4.201). The performance gap of approximately 1.06%
confirms that residual rank is a critical capacity control. Fig-
ure 5(a) contextualizes this ablation by plotting final perfor-
mance against KV cache size for all evaluated methods, re-
vealing that LRKV achieves superior performance-memory
tradeoffs across the rank spectrum compared to MLA latent
dimension ablations and baselines. When comparing LRKV
and MLA across full sweeps of r and dc, respectively, we
see that LRKV dominates the memory-performance frontier
even against MLA settings with larger latent dimensions.
The constrained r= 16 configuration underperforms stan-
dard MHA (0.881 vs 0.878 BPB at 128M), lacking capacity
to capture head-specific variation. In contrast, sufficient rank
(r≥64 ) outperforms MHA (0.875 BPB at 128M, 0.719
at 2.5B vs 0.878 and 0.723), demonstrating that representa-
tional capacity is the limiting factor. In our experiments, we
find that r≈0.36 -0.43×d h provides the necessary capacity
for LRKV to exceed standard attention performance and
MQA, GQA and MLA baselines (see Appendix G for scale-
specific rank values). This empirical regularity across scales
suggests that this rank range aligns with the intrinsic rank
of attention projections while optimizing the performance-
memory tradeoff (see Appendix B for spectral analysis).
LRKV preserves functional head diversity.Figure 6 con-
firms LRKV exhibits nearly identical similarity patterns to
Standard MHA. Quantitatively, we measure head diversity
using gauge-invariant metrics based on attention bilinear
forms Ah =W Q
h (W K
h )⊤, computing effective rank via
eigenvalue entropy ( Appendix B). LRKV (r=64) achieves
98.3% effective rank at 2.5B scale versus 98.9% for Stan-
dard MHA—a negligible 0.6pp difference (Figure 7). In
contrast, MQA achieves only 86.2% and GQA 95.4%.
Interpreting uncentered vs. PCA-based effective rank.
The distinction between uncentered (98.3%) and PCA-based
(93.5%) effective rank reveals LRKV’s factorization struc-
ture. Uncentered analysis measures total variance including
the shared mean direction—the global structure captured
by Wshared. PCA-based analysis centers the Gram matrix,
isolating variance around this mean and measuring true
head independence. The modest 4.8pp gap indicates LRKV
achieves diversity primarily through genuine per-head spe-
cialization rather than merely perturbing a dominant shared
structure. For comparison, MQA shows dramatic improve-
ment from uncentered (86.2%) to centered (91.0%)—acom-
pensation effectwhere forced KV sharing creates a strong
mean direction, but heads recover diversity by aggressively
diversifying query projections around this baseline (see sub-
8

<!-- page 9 -->

Low-Rank Key Value Attention
0 2 4 6 8 10
Layer
500
1000
1500
2000
2500
3000Frobenius norm (Keys)
 LRKV/Standard: 4.2×
(a) 128M
Standard MHA
LRKV (total)
LRKV (shared)
LRKV (residual)
0 5 10 15 20 25 30 35
Layer
200
400
600
800
1000
1200
1400Frobenius norm (Keys)
 LRKV/Standard: 5.1×
(b) 2.5B
Standard MHA
LRKV (total)
LRKV (shared)
LRKV (residual)
Figure 8.Magnitude scaling in LRKV is absorbed by post-projection normalization.Frobenius norms of key projections comparing
standard MHA with LRKV’s shared, residual, and total (shared + residual) components for(a)128M and(b)2.5B models. LRKV
projections operate in a higher-magnitude regime than standard MHA, with both shared and residual components individually exceeding
standard MHA magnitudes.
section C.4 for detailed analysis).
PCA-based analysis reveals compensation mechanisms.
Applying PCA in bilinear form space ( subsection C.4),
LRKV achieves 93.5% PCA-based effective rank at 2.5B
versus 94.0% for Standard MHA—within 0.5% despite rank-
64 factorization. Remarkably, PCA reveals acompensation
effectin MQA: its centered effective rank (91.0%)increases
versus uncentered (86.2%), opposite to other architectures.
This indicates MQA heads compensate for forced KV shar-
ing by diversifying query projections more aggressively.
Magnitude scaling and post-projection normalization.
Figure 8 reveals LRKV projections operate in a higher-
magnitude regime than standard MHA ( 3.1–6.7×), with
both shared and residual components exceeding MHA
norms. This arises naturally from the additive structure,
allowing independent growth during optimization. Cru-
cially, this scaling does not affect attention patterns because
RMSNorm is appliedafterprojection, making attention
effectively use cosine similarity.
Why magnitude scaling doesn’t degrade quality.The
higher-magnitude regime emerges from unconstrained opti-
mization of the additive factorization: gradients can indepen-
dently scale Wshared and residuals UhB⊤
h without affecting
their sum’s direction. RMSNorm applied after projection
normalizes representations before attention computation,
making the attention mechanism operate ondirectionalin-
formation (cosine similarity) rather than absolute magni-
tudes. This architectural property ensures that magnitude
scaling affects only the internal parameterization, not the
functional behavior. The magnitude difference serves as a
diagnostic: it indicates that optimization successfully allo-
cated capacity between shared and residual pathways, with
both contributing substantively to the final projection rather
than one dominating.
5. Discussion
The empirical analyses collectively reveal why LRKV
achieves superior performance despite KV cache reduction:
(1) Appropriate rank selection is critical.The range
r≈0.36 -0.43×dh (46-55 for dh = 128) provides sufficient
degrees of freedom for per-head specialization while con-
straining the factorization to exploit structured redundancy.
Below this threshold (e.g., r= 16 , 0.881 BPB), capacity be-
comes limiting; above it (r≥64 , 0.875 BPB), performance
exceeds MHA, confirming that representational capacity
(not geometric properties) determines quality.
(2) LRKV preserves architectural head diversity.PCA-
based analysis shows LRKV achieves 93.5% effective
rank versus 94.0% for standard MHA at 2.5B scale (Fig-
ure 7)—within 0.5% despite rank-64 factorization and
48.4% cache size. This near-perfect preservation validates
that low-rank residuals provide sufficient capacity for heads
to occupy independent dimensions in bilinear form space.
The consistency across all 36 layers demonstrates depth-
invariant factorization quality.
(3) Compensation mechanisms explain baseline behavior.
The PCA-based methodology reveals phenomena invisible
to prior metrics: MQA’s 4.8pp improvement from uncen-
tered to centered effective rank quantifies query compensa-
tion - heads recover diversity by specializing queries around
forced shared KVs. This explains why MQA remains vi-
able despite complete KV sharing, and positions LRKV’s
approach (preserving KV and query diversity) as superior.
(4) Emergent properties validate factorization quality.
9

<!-- page 10 -->

Low-Rank Key Value Attention
Geometric properties like moderate shared-residual orthog-
onality (cosine similarity 0.2-0.4) and magnitude scaling
(4-5× MHA) emerge as consequences of effective optimiza-
tion, not design constraints. These indicators confirm that
end-to-end training discovers factorizations that efficiently
allocate capacity between global structure (shared base) and
local specialization (residuals).
These properties collectively enable LRKV to achieve strong
pretraining performance (0.692 BPB at 6.3B) and down-
stream accuracy (40.2% combined at 6.3B) while reducing
KV cache to 45-53% of standard attention. The analyses
confirm LRKV exploits structured redundancy without sac-
rificing the head specialization that makes MHA effective.
6. Conclusion
We introduced Low-Rank Key-Value attention, which de-
composes key/value projections into a shared dense com-
ponent and compact per-head low-rank residuals. LRKV
achieves(1)45-53% KV cache reduction while attaining
the best pretraining loss (0.692 BPB at 6.3B) and 18-25%
training compute savings;(2)the highest downstream per-
formance across 5 benchmarks for varying model sizes; and
(3)near-complete preservation of head diversity give suf-
ficiently lower rank (93.5% vs. 94% for standard MHA),
confirming that LRKV exploits structured redundancy with-
out sacrificing specialization. Our gauge-invariant analy-
sis and rank ablations show that residual rank r≈0.36 -
0.43×d h provides a critical threshold for preserving head
specialization while enabling substantial KV cache reduc-
tion, positioning LRKV as a practical drop-in replacement
for standard attention under memory constraints.
References
Ainslie, J., Onta ˜n´on, S., Saxena, V ., et al. Gqa: Training
generalized multi-query transformer models from multi-
head checkpoints. InarXiv:2305.13245, 2023.
Allal, L. B., Lozhkov, A., von Werra, L., and Wolf, T.
Smoltalk: A conversational dataset for instruction tuning.
arXiv preprint arXiv:2408.00833, 2024.
Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The
long-document transformer.arXiv:2004.05150, 2020.
Bhojanapalli, S., Chakrabarti, A., Veit, A., Lukasik, M.,
Jain, H., Liu, F., Chang, Y .-W., and Kumar, S. Leveraging
redundancy in attention with reuse transformers.arXiv
preprint arXiv:2110.06821, 2021.
Chang, C.-C., Lin, W.-C., Lin, C.-Y ., Chen, C.-Y ., Hu, Y .-
F., Wang, P.-S., Huang, N.-C., Ceze, L., Abdelfattah,
M. S., and Wu, K.-C. Palu: Kv-cache compression with
low-rank projection. InThe Thirteenth International
Conference on Learning Representations, 2025.
Chari, V ., Qin, G., and Van Durme, B. Kv-distill: Nearly
lossless learnable context compression for llms.arXiv
preprint arXiv:2503.10337, 2025.
Chen, B., Zhang, F., Nguyen, A., Zan, D., Lin, Z., Lou, J.-
G., and Chen, W. Codet: Code generation with generated
tests.arXiv preprint arXiv:2207.10397, 2022.
Child, R., Gray, S., Radford, A., and Sutskever, I. Generat-
ing long sequences with sparse transformers. InICML,
2019.
Choromanski, K., Likhosherstov, V ., et al. Rethinking atten-
tion with performers. InICLR, 2021.
Clark, K., Khandelwal, U., Levy, O., and Manning, C. D.
What does bert look at? an analysis of bert’s attention. In
Proceedings of the 58th Annual Meeting of the Associa-
tion for Computational Linguistics (ACL), pp. 276––286,
2019.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
arXiv preprint arXiv:1803.05457, 2018.
Cobbe, K., Kosaraju, V ., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., Hesse, C., and Schulman, J. Training verifiers to solve
math word problems.arXiv preprint arXiv:2110.14168,
2021.
Dao, T., Fu, D. Y ., Ermon, S., Rudra, A., and Re, C.
Flashattention-3: Fast and memory-efficient exact atten-
tion with io-awareness. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS), 2024.
Fang, R. and Xu, Y . Addressing spectral bias of deep neural
networks by multi-grade deep learning. InAdvances in
Neural Information Processing Systems (NeurIPS) 2024,
pp. –, 2024.
Ge, S. et al. Longnet: Scaling transformers to 1,000,000
tokens.arXiv:2307.02486, 2023.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multitask
language understanding.Proceedings of the International
Conference on Learning Representations (ICLR), 2021.
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,
Welbl, J., Clark, A., et al. Training compute-optimal
large language models.arXiv preprint arXiv:2203.15556,
2022.
10

<!-- page 11 -->

Low-Rank Key Value Attention
Hu, E. J., Shen, Y ., Wallis, P., Allen-Zhu, Z., Li, Y ., Wang,
S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation
of large language models.ICLR, 1(2):3, 2022.
Hu, J., Li, H., Zhang, Y ., Wang, Z., Zhou, S., Zhang, X.,
Shum, H.-Y ., and Jiang, D. Multi-matrix factorization
attention.arXiv preprint arXiv:2412.19255, 2025.
Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.
Transformers are rnns: Fast autoregressive transformers
with linear attention. InICML, 2020.
Khalaf, M., Shamshoum, Y ., Hodos, N., Sieradzki, Y ., and
Schuster, A. Qkv projections require a fraction of their
memory.arXiv preprint arXiv:2506.02939, 2025.
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Similar-
ity of neural network representations revisited. InInterna-
tional Conference on Machine Learning, pp. 3519–3529.
PMLR, 2019.
Li, Y ., Huang, Y ., Yang, B., Venkitesh, B., Locatelli, A., Ye,
H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows
what you are looking for before generation.Advances
in Neural Information Processing Systems, 37:22947–
22970, 2024.
Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C.,
Dengr, C., Ruan, C., Dai, D., Guo, D., et al. Deepseek-v2:
A strong, economical, and efficient mixture-of-experts
language model.arXiv preprint arXiv:2405.04434, 2024.
Lv, X., Ding, N., Zhang, K., Hua, E., Cui, G., and Zhou,
B. Scalable efficient training of large language models
with low-dimensional projected attention.arXiv preprint
arXiv:2411.02063, 2024.
Michel, P., Levy, O., and Neubig, G. Are Sixteen Heads Re-
ally Better than One? InAdvances in Neural Information
Processing Systems (NeurIPS), 2019.
OpenAI. GPT-4. https://openai.com/research/
gpt-4, 2023. Accessed: 2025-10-02.
Penedo, G., Kydl´ıˇcek, H., Lozhkov, A., Mitchell, M., Raffel,
C. A., V on Werra, L., Wolf, T., et al. The fineweb datasets:
Decanting the web for the finest text data at scale.Ad-
vances in Neural Information Processing Systems, 37:
30811–30849, 2024.
Peng, Y ., Wang, Y ., Fang, Z., Zhu, L., Deng, Y ., and Duan, Y .
Revisiting lora: A smarter low-rank approach for efficient
model adaptation. In2025 5th International Conference
on Artificial Intelligence and Industrial Technology Ap-
plications (AIITA), pp. 1248–1252. IEEE, 2025.
Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J.
Svcca: Singular vector canonical correlation analysis for
deep learning dynamics and interpretability. InAdvances
in Neural Information Processing Systems, volume 30,
2017.
Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M.,
Hamprecht, F., Bengio, Y ., and Courville, A. On the spec-
tral bias of neural networks. InInternational conference
on machine learning, pp. 5301–5310. PMLR, 2019.
Sardana, N. and Havens, Z. Muon: An optimizer for hidden
layers. https://github.com/KellerJordan/
modded-nanogpt, 2024. Accessed: 2024-12-22.
Saxena, U., Saha, G., Choudhary, S., and Roy, K. Eigen
attention: Attention in low-rank space for kv cache com-
pression.arXiv preprint arXiv:2408.05646, 2024.
Sch¨olkopf, B., Smola, A., and M ¨uller, K.-R. Nonlinear
component analysis as a kernel eigenvalue problem. In
Neural computation, volume 10, pp. 1299–1319, 1998.
Shazeer, N. Fast transformer decoding: One write-head is
all you need. InarXiv:1911.02150, 2019.
Smola, A. J. and Sch¨olkopf, B. On a kernel-based method
for pattern recognition, regression, approximation, and
operator inversion.Algorithmica, 22(1):211–231, 1998.
Sun, Z., Liu, J., Dong, L., Wang, S., Huang, S., Chen, X.,
Zhou, Y ., Wang, X.-L., and Zhang, F. Retnet: Retentive
network for efficient sequence modeling. InAdvances in
Neural Information Processing Systems (NeurIPS), 2024.
Touvron, H., Martin, L., Stone, K., Albert, P., and et al.
Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288, 2023.
Touvron, H., Martin, L., Lavril, T., Albert, P., and et al. The
llama 3 herd of models. Meta AI Technical Report, 2024.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser,Ł., and Polosukhin, I. Attention
is All You Need. InAdvances in Neural Information
Processing Systems, 2017.
V oita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I.
Analyzing multi-head self-attention: Specialized heads
do the heavy lifting, the rest can be pruned.arXiv preprint
arXiv:1905.09418, 2019.
Wang, H. and Wang, K. Complete characterization of gauge
symmetries in transformer architectures. InNeurIPS
2025 Workshop on Symmetry and Geometry in Neural
Representations, 2025.
Wang, S., Zaheer, M., Katiyar, N., Salakhutdinov, R., and
Ahmed, A. Linformer: Self-attention with linear com-
plexity. InNeurIPS, 2020.
11

<!-- page 12 -->

Low-Rank Key Value Attention
Xie, Y . et al. Gptq: Accurate post-training quantization for
generative pretrained transformers.arXiv:2303.09035,
2023.
Yao, D., Shen, B., Lin, Z., Liu, W., Luan, J., Wang, B., and
Wang, W. Tailorkv: A hybrid framework for long-context
inference via tailored kv cache optimization.arXiv
preprint arXiv:2505.19586, 2025.
Yunis, D., Patel, K. K., Wheeler, S., Savarese, P., Vardi, G.,
Livescu, K., Maire, M., and Walter, M. R. Approaching
deep learning through the spectral dynamics of weights.
arXiv preprint arXiv:2408.11804, 2024.
Zhang, X., Ghosh, A., Liu, G., and Wang, R. Improving gen-
eralization of complex models under unbounded loss us-
ing pac-bayes bounds.arXiv preprint arXiv:2305.19243,
2023.
Zhang, Y ., Liu, Y ., Yuan, H., Qin, Z., Yuan, Y ., Gu, Q., and
Yao, A. C. Tensor product attention is all you need. In
ICML 2025 Workshop ES-FoMo-III (OpenReview), 2025.
OpenReview publication.
12

<!-- page 13 -->

Low-Rank Key Value Attention
A. Related Work (Full Version)
The memory footprint of the key-value (KV) cache has become a central bottleneck in deploying large autoregressive
Transformers. While early work improved thecomputationalefficiency of attention via sparsity, kernelization, or low-rank
approximations of the attention matrix (Child et al., 2019; Beltagy et al., 2020; Wang et al., 2020; Katharopoulos et al.,
2020; Choromanski et al., 2021), these methods do not reduce KV-cache size and therefore provide limited benefit for
decoding-heavy LLMs where memory, rather than compute, dominates inference cost.
Sharing K/V Projections.A widely adopted strategy for reducing KV memory is to share key and value projections
across attention heads. Multi-Query Attention (MQA) (Shazeer, 2019) uses a single shared set of K/V projections, and
Grouped-Query Attention (GQA) (Ainslie et al., 2023) extends this by sharing within small head groups. These approaches
achieve substantial memory savings but sacrifice head-level expressivity. Analyses of pretrained models (Clark et al., 2019;
Michel et al., 2019) consistently show that attention heads specialize for different syntactic or semantic functions; collapsing
their projections can therefore degrade modeling quality. LRKV is motivated by this tension: we preserve the efficiency of
shared projections while reintroducing diversity through low-rank head-specific residuals.
Low-Rank Parameterization of Attention Weights.Our formulation is related to low-rank reparameterization techniques
such as LoRA (Hu et al., 2022), which add low-rank updates to weight matrices for parameter-efficient fine-tuning. Follow-
up work extends this idea to training-time regularization or improved factorization schemes (Peng et al., 2025). However,
these methods intervene on theoptimization pathway, not on the structure of stored activations. LRKV applies the same
low-rank principle directly to attention projections, introducing low-rankrepresentationaldeviations that shape the K/V
features encoded in the cache.
Recent work has also explored low-rank or factored Q/K/V projections to improve parameter efficiency or computational
throughput (Xie et al., 2023; Khalaf et al., 2025). The concurrent work of Lv et al. (2024) investigates low-rank param-
eterizations of attention projections more broadly. These methods typically replace full-rank projections with low-rank
ones to reduce computation. LRKV differs conceptually: we preserve a full-rank shared projection and add only low-rank
head-specific residuals, enabling memory reduction while maintaining full-rank representational capacity.
KV-Cache Compression and Long-Context Modeling.A complementary body of research aims to compress or restructure
the KV cache itself. Approaches include clustering or distilling K/V states (Chari et al., 2025), exact hybrid and compressed
buffers (Yao et al., 2025; Li et al., 2024), token selection or pooling (Ge et al., 2023), and architectural changes that
replace attention with recurrent or state-space alternatives (Sun et al., 2024). Unlike these activation-level methods, LRKV
reduces the amount of information that must be stored by changing how each head generates its K/V features. Our hybrid
long-context cache builds on this: exact short-range KVs are retained, while long-range information is reconstructed
efficiently via low-rank residuals.
Comparison to Multi-Latent Attention.Multi-Latent Attention (MLA) (Liu et al., 2024) reduces KV cache memory by
compressing token representations into a shared low-dimensional latent space before caching. During attention, interaction
with the cache therefore requires projecting queries and/or values through this latent bottleneck. This achieves strong memory
compression but constrains all heads to operate through the same latent representation, limiting per-head expressivity and
restricting positional encoding choices (e.g., requiring partial RoPE).
LRKV addresses a different source of redundancy. Rather than compressing information across tokens, LRKV preserves
full token-level resolution and reduces redundancyacross headsby factorizing each head’s KV projection into a shared
full-rank base plus low-rank head-specific residuals. This additive structure retains the original feature dimensionality,
supports arbitrary positional encodings, and preserves head specialization while substantially reducing KV memory. As a
result, LRKV and MLA represent complementary approaches: MLA compresses token space via a shared latent bottleneck,
whereas LRKV compresses head space via structured sharing.
Factorized or Shared KV Mechanisms.Concurrent work explores structured KV-cache reduction via factorized repre-
sentations (e.g., MFA/MFA-KR; TPA (Hu et al., 2025; Zhang et al., 2025)) explores structured compression across heads.
LRKV provides a principled middle ground between fully shared (MQA/GQA) and fully independent projections: the
full-rank shared base captures global structure, while low-rank residuals preserve head-level variability. This structured
additive decomposition is key for achieving memory savings without collapsing representational diversity.
Analyzing attention head diversity.Prior work has measured head redundancy using attention pattern similarity (Michel
et al., 2019; V oita et al., 2019), activation-based metrics such as CKA (Kornblith et al., 2019), or SVCCA (Raghu et al.,
13

<!-- page 14 -->

Low-Rank Key Value Attention
2017). Recent work explicitly characterizes gauge symmetries in attention parameterizations (Wang & Wang, 2025),
recognizing that per-head rotations preserve attention function. However, these approaches either compare raw weights (not
gauge-invariant) or analyze activation space rather than the functional operators WQ(WK)⊤ that determine attention. We
introduce a principled method combining gauge-invariant bilinear forms with centered Gram matrix analysis (kernel PCA)
to measure intrinsic head diversity independent of parameterization choices.
B. Shared Subspaces, Spectral Bias, and Information Structure
Prior empirical analyses of pretrained Transformers have shown that attention heads exhibit substantial redundancy and
occupy overlapping subspaces (Michel et al., 2019; Clark et al., 2019; Bhojanapalli et al., 2021; Rahaman et al., 2019). To
assess this structure in a gauge-invariant manner, we analyze attention bilinear forms Ah =W Q
h (WK
h )⊤ and measure head
diversity via PCA on their centered Gram matrix (subsection 4.3). We find that LRKV achieves 93.5% effective rank versus
94.0% for standard MHA at 2.5B scale, confirming that heads occupy nearly independent dimensions in bilinear form space
despite using shared key-value bases. This validates that the functional behavior of attention projections lies near a shared
low-dimensional manifold, with LRKV’s additive structure explicitly parameterizing this geometry. Importantly, while
heads remain functionally diverse, theindividualweight matrices WK
h exhibit low intrinsic rank—they concentrate energy
in a small number of dominant singular directions.
Such spectral concentration is consistent with thespectral biasof neural networks (Rahaman et al., 2019; Fang & Xu, 2024),
whereby gradient-based optimization preferentially amplifies smooth, high-variance modes before fitting higher-frequency
residual structure. As a result, attention projections tend to concentrate energy in a small set of globally shared directions.
LRKV makes this implicit organization explicit: the shared projections WK,V
shared capture dominant spectral components,
while per-head low-rank residuals model localized refinements aligned with lower-variance directions. Each head therefore
operates within an affine subspace centered on WK,V
shared, with a tangent space spanned by a compact set of learned low-rank
factors.
Residual rank as a control knob for diversity.In LRKV , the residual rank r controls a continuous spectrum between
fully shared keys/values and fully independent per-head projections. At a high level, the shared base WK,V
shared serves as a
common coordinate system capturing features that are broadly useful across heads, while the low-rank residuals provide a
budgeted mechanism for head specialization. When r is small, heads are strongly coupled through the shared representation;
as r grows, the model can allocate additional degrees of freedom to capture head-specific patterns. This view motivates
treating r as an architectural hyperparameter with a clear interpretation: it directly controls the efficiency-diversity trade-off
in the KV cache and per-head specialization. Empirically, we find that r≈0.36 -0.43×d h provides the critical threshold:
LRKV achieves 93.5% PCA-based effective rank versus 94.0% for standard MHA at 2.5B scale (subsection 4.3), confirming
that moderate rank suffices to preserve nearly all head diversity while achieving substantial cache reduction.
Decomposition geometry and (non-)orthogonality.The decomposition Wh =W shared +U hB⊤
h is not identifiable:
for any ∆, we can shift Wshared ←W shared + ∆ and UhB⊤
h ←U hB⊤
h −∆ without changing Wh. In unconstrained
Euclidean parameter space there is therefore noa priorireason to expect strict orthogonality between the shared and residual
terms. Instead, we use overlap (e.g., cosine similarity or subspace angles) as a diagnostic: low overlap indicates that the
residual contributes directions not already captured by the shared component, reducing redundant parameterization. In
practice, any tendency toward reduced overlap is an emergent outcome of end-to-end training and the model’s incentives to
allocate capacity efficiently, rather than an explicit constraint or training objective.
Connection to low-rank approximation.LRKV can also be interpreted as learning a structured low-rank approximation
of the family of per-head projections. In classical matrix approximation, the optimal rank- r representation is given by
truncated SVD with respect to a chosen error metric. LRKV differs in two important ways: (i) it learns the shared and
residual factorsjointlywith the rest of the network under the task loss, and (ii) the residual factors are head-specific, enabling
specialization without requiring each head to store full-rank keys/values. This perspective suggests that end-to-end training
can discover factorizations that are close to classical optima while remaining aligned with the functional requirements of
attention.
Scaling, normalization, and stability.Because LRKV is additive, the norms of the shared and residual components
can evolve independently during training. However, attention depends primarily ondirectionalstructure and relative
14

<!-- page 15 -->

Low-Rank Key Value Attention
alignment, and common transformer normalization (e.g., RMSNorm) reduces sensitivity to absolute scale. From this
perspective, changes in parameter magnitudes are best interpreted as a redistribution of representational capacity between
shared and residual pathways rather than as a direct indicator of instability. This motivates analyzing LRKV through
geometry (alignment, overlap) and function (loss/perplexity), rather than through norms alone.
C. Principled Head Diversity Analysis via PCA in Bilinear Form Space
This section presents our gauge-invariant PCA-based methodology for analyzing attention head diversity and provides
comprehensive results across model scales. To our knowledge, this is the first work to combine (i) gauge-invariant bilinear
form comparison, (ii) centered Gram matrix analysis (kernel PCA), and (iii) variance-explained interpretation for measuring
head independence in transformers.
C.1. Methodology: PCA in the Space of Bilinear Forms
Gauge invariance motivation.Comparing attention heads via raw weight matrices WK
h or WQ
h is not meaningful
because attention is invariant to coupled per-head rotations: for any orthogonal Rh, the transformations WQ
h ←W Q
h Rh
and WK
h ←W K
h Rh leave attention outputs unchanged. The gauge-invariant object is the bilinear formAh =W Q
h (WK
h )⊤,
which determines attention logits.
Inner product on bilinear forms.We define similarity between heads i and j using the Frobenius inner product on their
bilinear forms:
⟨Ai,A j⟩F =tr(A ⊤
i Aj) =tr((W K
i )⊤WK
j (WQ
j )⊤WQ
i ).(11)
Normalized by individual norms, this yields a similarity scores ij ∈[−1,1]forming the Gram matrixG.
Centering for proper PCA.The key methodological contribution is recognizing that the Gram matrix G can be centered
without materializing the mean bilinear form:
Gcentered[i, j] =G[i, j]− 1
H
X
k
G[i, k]− 1
H
X
k
G[k, j] + 1
H 2
X
k,ℓ
G[k, ℓ].(12)
This is the kernel PCA centering trick (Sch¨olkopf et al., 1998), enabling PCA in the abstract space of bilinear forms using
only their inner products.
Interpretation as variance decomposition.The eigenvalues {λi} of Gcentered represent variance explained by each
principal component in bilinear form space. We compute:
•Variance explained:v i =λ i/P
j λj (fraction of total variance ini-th PC)
•Cumulative variance: Pk
j=1 vj (variance captured by firstkPCs)
•Effective rank (PCA-based):exp(− P
i vi logv i)via Shannon entropy
High effective rank indicates heads occupy many independent dimensions (low sharing); low effective rank indicates
clustering in few PCs (high sharing).
Comparison to uncentered analysis.Prior work typically uses uncentered Gram matrices, whose eigenvalues include
both the ”mean direction” (shared structure) and variance around the mean. The centered analysis isolates the latter, revealing
head independence independent of shared baselines. This distinction is critical: MQA’s complete KV sharing creates a
dominant mean direction (lowering uncentered effective rank) but heads compensate via query specialization (maintaining
centered effective rank). The PCA-based metric reveals this compensation effect.
15

<!-- page 16 -->

Low-Rank Key Value Attention
Table 3.PCA-based effective rank comparison.Centered Gram matrix analysis reveals LRKV preserves head diversity within 1% of
Standard MHA at both scales, while MQA shows surprising resilience through query compensation.
128M (6 heads) 2.5B (18 heads)
Model Uncentered PCA-based Uncentered PCA-based
Standard MHA 95.4% 82.4% 98.9%94.0%
LRKV (r=64) 96.2%83.0% 98.3%93.5%
GQA 90.5%* 79.6%* 95.4% 91.7%
LRKV (r=16) 88.8% 81.5% – –
MQA 72.6% 78.4% 86.2% 91.0%
MLA 83.9% 78.5% 92.7% 91.6%
*GQA at 128M estimated from 2.5B patterns. PCA-based percentages are relative to maximum possible effective rank (number of heads).
Note MQA’s 4.8pp improvement from uncentered to PCA-based at 2.5B scale, revealing query compensation.
0 2 4 6 8 10
Layer
1
2
3
4
5
6Effective Rank
Head Diversity via Effective Rank (128M)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
Fully redundant
Figure 9.Head diversity at 128M scale (PCA-based analysis).Effective rank computed from centered Gram matrices shows consistent
patterns: LRKV (r=64) achieves 83.0% effective rank versus 82.4% for Standard MHA, demonstrating that sufficient residual capacity
(r≈0.36 -0.43×d h in deployed models) preserves head specialization. LRKV (r=16) achieves 81.5%, while MQA shows 78.4%.
The PCA-based metric reveals that even aggressive compression methods maintain substantial head diversity through compensation
mechanisms.
C.2. Complete Results Across Scales
C.3. Effective Rank at 128M Scale
At 128M scale with 6 attention heads, LRKV (r=64) achieves effective rank (83.0%) nearly matching Standard MHA
(82.4%), demonstrating that the low-rank factorization does not inherently degrade head diversity when rank is appropriately
chosen. The moderately constrained LRKV (r=16) configuration shows only slightly reduced diversity (81.5%), while MQA
maintains 78.4% effective rank despite complete KV sharing—revealing the query compensation effect at smaller scale.
C.4. PCA Eigenvalue Spectra
The eigenvalue spectra provide an alternative view of head diversity, complementing the aggregate effective rank metric. The
PCA-based effective rank summarizes the entire spectrum via entropy, while these plots show the full eigenvalue distribution.
Key observations:
• LRKV matches Standard MHA spectra:At 2.5B scale (Figure 10), LRKV’s eigenvalue curves closely track Standard
MHA across early, middle, and late layers. This indicates not just similar effective rank, but nearly identical head
16

<!-- page 17 -->

Low-Rank Key Value Attention
2.5 5.0 7.5 10.0 12.5 15.0 17.5
PC Index
0
1
2
3
4
5
6
7Variance (PCA Eigenvalue)
Early Layer (L=0)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
2.5 5.0 7.5 10.0 12.5 15.0 17.5
PC Index
0.0
0.5
1.0
1.5
2.0
2.5Variance (PCA Eigenvalue)
Middle Layer (L=18)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
2.5 5.0 7.5 10.0 12.5 15.0 17.5
PC Index
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4Variance (PCA Eigenvalue)
Late Layer (L=35)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
PCA Eigenvalue Spectra in Bilinear Form Space (2.5B)
Figure 10.PCA eigenvalue spectra reveal variance structure in bilinear form space (2.5B).Eigenvalues of centered Gram matrices at
early, middle, and late layers show how variance is distributed across principal components. LRKV’s spectra closely match Standard
MHA across all depth regimes, with similar leading eigenvalues and comparable tail decay, indicating nearly identical head correlation
structure. MQA shows slightly elevated later eigenvalues, consistent with query compensation: while the first few PCs capture shared KV
structure, remaining PCs capture query-driven diversity. Early layers show slightly higher leading eigenvalues across methods, suggesting
a stronger shared component (greater alignment in bilinear form space) in lower layers; deeper layers exhibit more uniform spectra,
consistent with variance being spread across more modes.
1 2 3 4 5 6
PC Index
0.0
0.2
0.4
0.6
0.8
1.0
1.2Variance (PCA Eigenvalue)
Early Layer (L=0)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
1 2 3 4 5 6
PC Index
0.0
0.2
0.4
0.6
0.8
1.0Variance (PCA Eigenvalue)
Middle Layer (L=6)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
1 2 3 4 5 6
PC Index
0.0
0.2
0.4
0.6
0.8
1.0
1.2Variance (PCA Eigenvalue)
Late Layer (L=11)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
PCA Eigenvalue Spectra in Bilinear Form Space (128M)
Figure 11.PCA eigenvalue spectra at 128M scale.With 6 heads, the eigenvalue structure is more pronounced. LRKV (r=64) tracks
Standard MHA closely across all layers, while LRKV (r=16) shows similar patterns with slightly elevated leading eigenvalues in later
layers. MQA exhibits comparable spectra to other methods, with its eigenvalue distribution revealing that query compensation maintains
diversity despite complete KV sharing. The consistency of spectra shapes across architectures suggests fundamental constraints on head
organization.
correlation structure in the space of bilinear forms.
• MQA’s compensation appears in eigenvalue distribution:MQA shows slightly elevated later eigenvalues compared
to its uncentered analysis would suggest, consistent with query compensation. While early PCs capture the shared KV
structure (mean direction), later PCs reveal query-driven diversity that maintains head specialization.
• Depth-dependent patterns:Early layers show slightly higher leading eigenvalues across all methods, suggesting that
heads capture more structured relationships (greater specialization) for low-level features. Middle and late layers show
more uniform spectra, consistent with increasingly abstract representations where heads become more similar.
• Scale consistency:The 128M results (Figure 11) mirror the 2.5B patterns despite different head counts (6 vs 18),
confirming that LRKV’s preservation of head diversity and MQA’s compensation mechanism are scale-invariant
phenomena.
C.5. Cumulative Variance Explained
The cumulative variance plots directly answer the question: ”How many principal components capture most of the head
diversity?” This provides an intuitive measure of intrinsic dimensionality that complements the entropy-based effective rank.
17

<!-- page 18 -->

Low-Rank Key Value Attention
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Early Layer (L=0)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
90% variance
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Middle Layer (L=18)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
90% variance
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Late Layer (L=35)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
90% variance
Cumulative Variance Explained by PCs (2.5B)
Figure 12.Cumulative variance explained quantifies dimensionality of head diversity (2.5B).Plots show how many principal
components are needed to capture X% of total variance in bilinear form space at early, middle, and late layers. Standard MHA and
LRKV require ∼16-17 PCs for 90% variance (out of 18 total heads), demonstrating heads occupy nearly all available dimensions with
minimal redundancy. GQA requires ∼15-16 PCs, showing modest clustering from group-based sharing. MQA and MLA require ∼14-15
PCs, indicating moderate concentration in fewer dimensions but still substantial spread. The 90% threshold (dashed line) serves as a
practical measure of intrinsic dimensionality. All methods show relatively linear cumulative variance curves, indicating no single PC
dominates—even MQA maintains distributed variance.
1 2 3 4 5 6
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Early Layer (L=0)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
90% variance
1 2 3 4 5 6
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Middle Layer (L=6)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
90% variance
1 2 3 4 5 6
Number of PCs
0.0
0.2
0.4
0.6
0.8
1.0Cumulative Variance Explained
Late Layer (L=11)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
90% variance
Cumulative Variance Explained by PCs (128M)
Figure 13.Cumulative variance explained at 128M scale.With 6 heads, the variance structure is more visible. Standard MHA and
LRKV (r=64) require ∼5 PCs for 90% variance, demonstrating that heads remain largely independent. LRKV (r=16) shows similar
patterns despite constrained rank. MQA requires ∼4-5 PCs, confirming that query compensation maintains distributed variance even with
complete KV sharing. The steeper curves at 128M compared to 2.5B suggest slightly more variance concentration at smaller scale.
Key findings:
• LRKV preserves full dimensionality:At both scales, LRKV requires nearly all available PCs to capture 90% variance,
matching Standard MHA. This confirms that low-rank residuals provide sufficient degrees of freedom for heads to
occupy independent dimensions in bilinear form space.
• No dominant principal component:The relatively linear cumulative variance curves indicate that no single PC
captures disproportionate variance. Even the first PC accounts for only 15-20% of total variance, confirming that heads
do not collapse to a single dominant direction.
• MQA’s distributed compensation:Despite complete KV sharing, MQA’s cumulative variance curve shows substantial
spread across PCs rather than concentration in the first few components. This quantifies the compensation effect: heads
diversify their queries across many dimensions rather than collapsing to a low-dimensional subspace.
• Scale-dependent behavior:The gap between methods narrows from 128M to 2.5B scale. At 128M, methods are more
separated in the cumulative variance plot; at 2.5B, curves converge more closely, suggesting larger models develop
more sophisticated compensation mechanisms.
18

<!-- page 19 -->

Low-Rank Key Value Attention
0 5 10 15 20 25 30 35
Layer
6
8
10
12
14
16
18Effective Rank
Original (Uncentered Gram Matrix)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
0 5 10 15 20 25 30 35
Layer
9
10
11
12
13
14
15
16
17Effective Rank (PCA)
PCA-based (Centered Gram Matrix)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
Effective Rank Comparison (2.5B)
Figure 14.Comparing uncentered vs PCA-based effective rank reveals compensation mechanisms (2.5B).Layer-by-layer comparison
shows consistent patterns across all 36 layers.Left panel (uncentered):Standard MHA and LRKV maintain high effective rank (17-18),
while MQA shows substantial degradation (15-16), suggesting significant diversity loss from complete sharing.Right panel (PCA-
based):After centering, all methods converge to narrower range (16-17), with MQA showing dramatic improvement. This reveals the
compensation effect: MQA’s shared KV structure creates a strong mean direction (visible in uncentered analysis), but heads compensate
by diversifying queries around this mean (revealed by centering). LRKV consistently tracks Standard MHA in both metrics, confirming
true head independence. GQA and MLA show intermediate behavior with modest gaps between uncentered and centered metrics.
C.6. Comparison: Uncentered vs PCA-based Effective Rank
The side-by-side comparison of uncentered versus PCA-based effective rank reveals why centering is essential for under-
standing head diversity:
The MQA compensation effect (quantified).At 2.5B scale, MQA improves from 86.2% (uncentered) to 91.0% (PCA-
based)—a 4.8 percentage point gain. This quantifies how much diversity MQA recovers through query specialization
after accounting for its shared KV baseline. Without centering, we would conclude MQA loses 12.7pp of diversity versus
Standard MHA (98.9% → 86.2%); with centering, the loss is only 3.0pp (94.0% → 91.0%), revealing MQA is much less
constrained than raw metrics suggest.
LRKV’s consistency across metrics.LRKV shows minimal gap between uncentered (98.3%) and PCA-based (93.5%)
effective rank—a 4.8pp difference similar to Standard MHA’s 4.9pp gap. This indicates LRKV achieves diversity through
genuinely independent head specialization rather than compensation around a shared baseline. The low-rank residuals
provide real degrees of freedom, not just perturbations of a dominant mean structure.
Scale-dependent centering effects.The gaps between uncentered and PCA-based metrics are larger at 128M scale (LRKV:
13.2pp gap) than at 2.5B (4.8pp gap). This suggests smaller models rely more on shared structure with local compensation,
while larger models develop more truly independent head specializations. The PCA-based metric, by removing the mean
direction, reveals this scale-dependent transition.
C.7. Eigenvalue Spectra of Uncentered Similarity Matrices
For completeness, we also show the eigenvalue spectra of uncentered head similarity matrices, which have been used in
prior work for head diversity analysis.
C.8. Head Similarity Heatmaps
The heatmaps provide direct visualization of head relationships in the uncentered similarity space, complementing the
PCA-based aggregate metrics. Key observations:
• LRKV preserves correlation structure:At both scales, LRKV’s similarity matrices closely match Standard MHA’s
19

<!-- page 20 -->

Low-Rank Key Value Attention
0 2 4 6 8 10
Layer
2.0
2.5
3.0
3.5
4.0
4.5
5.0
5.5
6.0Effective Rank
Original (Uncentered Gram Matrix)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
0 2 4 6 8 10
Layer
4.0
4.2
4.4
4.6
4.8
5.0Effective Rank (PCA)
PCA-based (Centered Gram Matrix)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
Effective Rank Comparison (128M)
Figure 15.Effective rank comparison at 128M scale.Similar patterns emerge with 6 heads: uncentered metric shows larger separation
between methods, while PCA-based metric reveals closer clustering. MQA’s improvement from uncentered (4.36/6 = 72.6%) to PCA-
based (4.70/6 = 78.4%) demonstrates compensation effect at smaller scale. The larger percentage gaps at 128M versus 2.5B suggest
smaller models rely more heavily on shared structure with compensation, while larger models develop more independent specialization.
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Eigenvalue Index
0
2
4
6
8
10Eigenvalue
Early Layer (L=0)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Eigenvalue Index
0.5
1.0
1.5
2.0
2.5
3.0
3.5Eigenvalue
Middle Layer (L=18)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
2.5 5.0 7.5 10.0 12.5 15.0 17.5
Eigenvalue Index
0.6
0.8
1.0
1.2
1.4
1.6
1.8Eigenvalue
Late Layer (L=35)
Standard MHA
LRKV (r=64)
GQA
MQA
MLA
Eigenvalue Spectra of Head Similarity Matrices (2.5B)
Figure 16.Uncentered eigenvalue spectra at early, middle, and late layers (2.5B scale).The eigenvalue distribution of uncentered head
similarity matrices S shows broader separation between methods than PCA-based analysis. LRKV maintains eigenvalue spectra nearly
identical to Standard MHA across all depth regimes. MQA shows elevated leading eigenvalues and faster tail decay, indicating stronger
head correlation in the uncentered view—but PCA analysis reveals much of this is due to the shared mean direction rather than loss of
diversity. Early layers show slightly higher leading eigenvalues across all methods, indicating a stronger shared component in bilinear
form space at lower layers; deeper layers exhibit more uniform spectra, consistent with variance being distributed across more modes.
patterns. Off-diagonal entries (head-to-head similarities) show comparable magnitudes and spatial distribution,
indicating that low-rank factorization does not artificially increase or decrease head correlations in the uncentered
space.
• No emergent clustering:Standard MHA shows diffuse correlations without strong block structure (aside from diagonal
dominance). LRKV maintains this property, suggesting heads remain independently specialized rather than forming
redundant groups, even before centering removes shared mean structure.
• GQA shows group structure:At 2.5B scale with 6 KV groups of 3 heads each, GQA’s heatmap reveals visible
within-group structure—heads sharing KV projections exhibit stronger mutual similarity. This validates that partial KV
sharing induces measurable correlation visible in raw similarity space.
• Quantitative confirmation:The mean off-diagonal similarity for LRKV (r=64) is within 0.02–0.04 of Standard
MHA across layers, confirming the visual similarity is not merely qualitative. This tight correspondence holds in both
uncentered and PCA-based analysis.
20

<!-- page 21 -->

Low-Rank Key Value Attention
1 2 3 4 5 6
Eigenvalue Index
0
1
2
3
4
5Eigenvalue
Early Layer (L=0)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
1 2 3 4 5 6
Eigenvalue Index
0.50
0.75
1.00
1.25
1.50
1.75
2.00
2.25Eigenvalue
Middle Layer (L=6)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
1 2 3 4 5 6
Eigenvalue Index
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0Eigenvalue
Late Layer (L=11)
Standard MHA
LRKV (r=16)
LRKV (r=64)
MQA
MLA
GQA
Eigenvalue Spectra of Head Similarity Matrices (128M)
Figure 17.Uncentered eigenvalue spectra at 128M scale.With 6 heads, the eigenvalue structure is more pronounced. LRKV (r=64)
tracks Standard MHA closely, while LRKV (r=16) shows slightly elevated leading eigenvalues in later layers, consistent with reduced
effective rank. MQA exhibits the most concentrated spectra in the uncentered view, with a dominant leading eigenvalue suggesting strong
head redundancy—but this is largely due to the shared KV mean direction rather than true loss of diversity, as PCA analysis reveals.
0 1 2 3 4 5
Head Index
0
1
2
3
4
5 Head Index
Standard MHA
0 1 2 3 4 5
Head Index
0
1
2
3
4
5 Head Index
LRKV (r=16)
0 1 2 3 4 5
Head Index
0
1
2
3
4
5 Head Index
LRKV (r=64)
0 1 2 3 4 5
Head Index
0
1
2
3
4
5 Head Index
MQA
0 1 2 3 4 5
Head Index
0
1
2
3
4
5 Head Index
MLA
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
Figure 18.Head similarity matrices at 128M scale.With fewer heads (6 vs 18), individual head relationships are more visible. LRKV
(r=64) shows similarity structure nearly identical to Standard MHA, with moderate positive correlations but no dominant clusters. LRKV
(r=16) exhibits slightly stronger correlations, consistent with its reduced effective rank. The mean off-diagonal similarity for LRKV
(r=64) is within 0.02–0.04 of Standard MHA across layers, providing quantitative confirmation of the visual similarity. These uncentered
heatmaps complement the PCA-based analysis by showing the raw pairwise similarities before mean removal.
C.9. Key Findings from PCA Analysis
1. LRKV preserves head independence within 1% of standard attention.At 2.5B scale, LRKV achieves 93.5%
PCA-based effective rank versus 94.0% for Standard MHA (0.5pp gap). This near-perfect preservation occurs despite 52.6%
KV cache size, validating that rank-64 residuals provide sufficient capacity for head specialization. The consistency across
all 36 layers demonstrates depth-invariant factorization quality.
2. The MQA compensation effect.MQA’s 4.8pp improvement from uncentered (86.2%) to PCA-based (91.0%) effective
rank at 2.5B scale reveals a previously unknown mechanism: forced KV sharing creates a strong mean bilinear form, but
heads compensate by diversifying query projections around this mean. The centered analysis isolates this true independence,
showing MQA is less constrained than prior metrics suggest. This effect is consistent at 128M scale (5.8pp improvement),
confirming it is a fundamental property of MQA rather than a scale-specific artifact.
3. MLA sits between GQA and MQA in diversity space.MLA achieves 91.6% PCA-based effective rank at 2.5B,
placing it between GQA (91.7%) and MQA (91.0%). This makes architectural sense: MLA’s latent bottleneck constrains
heads more than GQA’s group sharing but less than MQA’s complete sharing, with per-head decompression allowing
specialization after the bottleneck. At 128M scale, MLA shows 78.5% effective rank, slightly below MQA (78.4%),
suggesting the latent bottleneck is more constraining at smaller scales where bottleneck capacity is limited.
4. Scale-dependent behavior.The gap between uncentered and PCA-based metrics shrinks from 128M to 2.5B (e.g.,
LRKV: 13.2pp gap at 128M vs 4.8pp at 2.5B), suggesting larger models develop stronger head specialization around shared
structure rather than purely independent representations. This transition indicates that model scale affects not just capacity
but the fundamental organization principle of attention heads.
21

<!-- page 22 -->

Low-Rank Key Value Attention
5. Rank selection determines capacity.Comparing LRKV configurations reveals that appropriate rank is critical. At
128M scale, r=16 (12.5% of head dimension) achieves 81.5% PCA-based effective rank, while r=64 (50%) achieves
83.0%—a modest 1.5pp improvement in diversity but substantial performance difference (0.881 vs 0.875 BPB). This
confirms that representational capacity, not geometric properties, determines quality: even moderate diversity can yield
strong performance if the rank provides sufficient degrees of freedom.
C.10. Comparison to Prior Diversity Metrics
Our PCA-based approach differs from prior work in several key respects:
CKA/SVCCA (Kornblith et al., 2019; Raghu et al., 2017).These methods compare activation similarities (repre-
sentations), not functional operators. While CKA uses centered Gram matrices (providing the ”C” in ”Centered Kernel
Alignment”), it analyzes activation space rather than weight-space bilinear forms. Our contribution is applying the centering
principle to the space of attention operators themselves, revealing compensation mechanisms invisible when analyzing
activations.
Attention pattern similarity (Michel et al., 2019; Voita et al., 2019).These behavioral metrics measure which tokens
heads attend to on specific inputs. They depend on input data distribution and don’t directly measure the representational
capacity encoded in the parameterization. Our approach analyzes the parameterization itself, providing a data-independent
measure of potential diversity.
Raw weight comparison.Directly comparing WK
h or WQ
h matrices is not gauge-invariant: arbitrary per-head rotations
that preserve attention function can make identical heads appear different or vice versa. The bilinear form WQ(WK)⊤ is
the minimal gauge-invariant object for comparison.
Uncentered Gram matrices.Several prior analyses(Zhang et al., 2023) form head similarity matrices (often using
uncentered correlations or kernel similarities) and interpret their spectra directly; our contribution is to apply explicit kernel
centering in the space of gauge-invariant attention operators, which separates shared mean structure from variance around
the mean. This conflates mean structure (shared baselines) with spread (true diversity), masking compensation effects like
MQA’s query specialization. Our centered analysis isolates intrinsic dimensionality independent of mean direction.
The combination of (i) gauge-invariant bilinear forms, (ii) centered Gram matrix (kernel PCA), and (iii) PCA interpretation
as variance explained provides, to our knowledge, the first principled operator-space analysis of transformer head diversity.
This methodology reveals phenomena invisible to prior approaches, such as the MQA compensation effect and the precise
quantification of LRKV’s near-perfect diversity preservation.
C.11. Methodological Limitations and Extensions
Limitations.Our analysis focuses on final trained checkpoints and does not track how the factorization evolves during
training. The PCA-based metric measures potential diversity encoded in the parameterization but not behavioral diversity on
specific inputs. The gauge-invariant similarity metric is one of many possible inner products on bilinear forms; alternative
metrics might reveal additional structure.
Future extensions.Several directions merit exploration: (1) Analyzing PCA eigenvalue dynamics during pretraining to
understand how shared and head-specific structure co-evolve. (2) Comparing parameterization-based diversity (this work)
with input-dependent behavioral diversity to understand their relationship. (3) Extending the framework to cross-attention in
encoder-decoder models, where query and key-value heads may have different sizes. (4) Investigating whether PCA-guided
initialization or regularization can improve training dynamics by explicitly encouraging high-variance factorizations.
D. Experimental Setup (Full Details)
D.1. Pretraining Configuration
We pretrain from scratch a family of decoder-only Transformer language models with parameter counts ranging from128M
to 6.3Bon theFineWeb-Edudataset (Penedo et al., 2024). Unless otherwise stated, model scale (e.g., 128M, 2.5B, 6.3B)
22

<!-- page 23 -->

Low-Rank Key Value Attention
Table 4.Mathematical comparison of attention mechanisms.We compare how different mechanisms parameterize key/value projections
and where compression is applied. Here X∈R T×d is the token sequence, dh =d/H is the head dimension, H is the number of heads,
andr, d c ≪d h.
Method KV projection form Rank constraint Compression axis
Standard MHAW K,V
h ∈R d×dh (independent per head) None (full rank) None
MQAW K,V
h =W K,V
shared ∀hrank(W K,V
h ) = rank(Wshared)Across heads (complete sharing)
GQAW K,V
h =W K,V
g(h) for groupg(h)Full rank within group Across heads (group-wise sharing)
MLAW K,V
h =W downWK,V
up,h rank(WK,V
h )≤d c Across tokens (latent bottleneck)
LRKV (ours)W K,V
h =W K,V
shared+U K,V
h (BK,V
h )⊤ rank(WK,V
h −W K,V
shared)≤r Across heads (additive low-rank de-
viations)
refers to the parameter count of thebase architecture with standard multi-head attention (MHA). Alternative attention
mechanisms (LRKV , MQA, GQA, MLA) replace only the attention module while keeping all other architectural components
fixed, which will result in reduced total parameter count due to reparameterized key-value projections.
Model Architecture Summary.Table 5 provides complete architectural specifications for all model scales evaluated in
this work.
Table 5.Complete model architectural specifications.All models use dhead = 128 and FFN expansion ratio of 4 (i.e.,dFFN = 4×d model).
Scale Layers Headsd model dFFN Vocab Size
128M 12 6 768 3072 50304
512M 24 12 1536 6144 50304
1.2B 24 12 1536 6144 50304
2.5B 18 18 2304 9216 50304
6.3B 32 32 4096 16384 50304
Training Configuration Summary.Table 6 provides complete training configuration details across all model scales.
Table 6.Training configuration across model scales.Batch size refers to number of sequences; total tokens per batch = batch size ×
context length. All models trained on FineWeb-Edu with Muon+AdamW optimization.
Scale Context Batch Size Tokens/Batch Total Tokens Compute (EF) Purpose
128M 2048 16 32,768 100B 31.0 Ablations
512M 8192 4 32,768 50B 153.6 Long context
1.2B 2048 8 (GA=2) 32,768 50B 304.7 Scaling study
2.5B 2048 8 (GA=2) 32,768 50B 635.2 Main experiments
6.3B 2048 8 (GA=2) 32,768 50B 1603.3 Scaling study
GA=gradient accumulation steps. EF=ExaFLOPs. 128M overtrained (100B tokens) for low-variance ablations; others use compute-near-
optimal budgets (Hoffmann et al., 2022).
Optimization Details.We use a hybrid optimizer strategy: theMuon optimizer(Sardana & Havens, 2024) for Transformer
weight matrices andAdamWfor embeddings, with component-specific learning rates (matrix: 0.02, embedding: 0.2,
unembedding: 0.004). All models are optimized using cosine learning rate decay with linear warmup (2000 steps), weight
decay 0.1, and gradient clipping at 1.0.
Hardware and Precision.All experiments are run on a single8 ×H200 GPUnode in mixed-precision (bfloat16) mode.
Full architectural and optimization hyperparameters, including learning rate schedules and detailed optimizer settings, are
provided in Appendix F.
Evaluation Metrics.Training progress is monitored using thecross-entropy lossandbits-per-byte (BPB)on held-out
validation split of FineWeb-Edu. These metrics allow consistent comparison across model scales and serve as the primary
indicators of data efficiency and convergence quality.
23

<!-- page 24 -->

Low-Rank Key Value Attention
D.2. Mid-training Configuration
After pretraining, we performsupervised fine-tuning(mid-training) on a curated mixture of instructional data to improve
downstream task performance. The mid-training dataset consists of three components:
1.SmolTalk(Allal et al., 2024) with 460K conversational examples for general instruction-following
2. MMLU auxiliary train(Hendrycks et al., 2021) with 100K multiple-choice questions spanning diverse academic
subjects
3.GSM8K(Cobbe et al., 2021) with 8K grade-school math problems including calculator tool use
This yields a total training mixture of approximately568K examples. Validation is performed on held-out test splits using
proportional sampling (24K SmolTalk, 5.2K MMLU, 420 GSM8K examples).
We train forone full epochover the mid-training mixture with a batch size of524,288 tokensand sequence length of 2048.
We use a decayed linear schedule for the learning rate, using AdamW for the embeddings and Muon for the remaining
parameters. Models are evaluated every 150 steps using bits-per-byte on the validation set. All mid-training runs use the
same hyperparameters across different attention mechanisms to ensure fair comparison.
D.3. Attention Mechanism Configurations
To evaluate the effectiveness of low-rank KV factorization, we compare LRKV against four baseline attention mechanisms
across all model scales. Table 7 provides complete configuration details for all methods and scales.
Table 7.Attention mechanism configurations across model scales.KV cache percentages are relative to Standard MHA. All models
used head = 128and 2048-token context (except 512M long-context with 8192 tokens).
Standard MQA / GQA MLA / LRKV
Scale Heads KV Heads Cache MQA KV GQA KV GQA Cache MLAd c LRKVrLRKV Cache
128M 6 6 100% 1 (16.7%) 3 (50%) 50% 128 (8.3%) 46 52.6%
512M 12 12 100% 1 (8.3%) 4 (33.3%) 33.3% 256 (8.3%) 51 48.2%
1.2B 12 12 100% 1 (8.3%) 4 (33.3%) 33.3% 256 (8.3%) 51 48.2%
2.5B 18 18 100% 1 (5.6%) 6 (33.3%) 33.3% 384 (8.3%) 55 48.5%
6.3B 32 32 100% 1 (3.1%) 2 (6.3%) 6.3% 1024 (12.5%) 54 45.3%
512M-8K† 12 12 100% 1 (8.3%) 4 (33.3%) 33.3% 256 (8.3%) 51 48.2%
†512M-8K denotes the long-context configuration with 8192-token sequences. All other models use 2048-token context.
KV Cache Calculation.All KV cache percentages are reported relative to Standard MHA’s memory usage. For LRKV , the
reported cache size assumes an optimized implementation that caches the shared projection Wshared and per-head low-rank
latents RK,V
h rather than the fully reconstructed key-value matrices, following the theoretical analysis in Section 3. The
cache ratio follows: dh+Hr
Hd h
= 1
H + r
dh
.
This configuration balances memory efficiency with model expressiveness: LRKV uses approximately half the KV cache
of Standard MHA while maintaining full representational capacity through low-rank adaptation, outperforming both
memory-minimal approaches (MQA, MLA) and full attention in downstream task performance.
D.4. Rank Ablation Experiments
To systematically evaluate how residual rank affects LRKV’s performance-memory tradeoff, we conduct comprehensive
ablation studies at the 128M scale. These experiments isolate the effect of rank selection while controlling for all other
architectural and training factors.
LRKV Rank Ablation.We train five 128M parameter models with LRKV using ranks r∈ {8,16,32,64,128} on
100B tokens of FineWeb-Edu. All models use identical architecture (6 heads, dmodel = 768, dhead = 128), optimization
hyperparameters, and training schedule. Table 8 presents complete results.
24

<!-- page 25 -->

Low-Rank Key Value Attention
Table 8.LRKV rank ablation and MLA latent dimension comparison (128M scale, 100B tokens).All models use 6 heads with
dh = 128. Cache percentages relative to Standard MHA baseline.
Configuration Rank/Latent KV Cache CE Loss↓BPB↓Notes
LRKV Rank Ablation
LRKV-8 r= 822.9% 2.908 0.881 Insufficient capacity
LRKV-16 r= 1629.2% 2.905 0.880 Below MHA
LRKV-32 r= 3241.7% 2.896 0.877 Approaching MHA
LRKV-64 r= 6466.7% 2.888 0.875 Exceeds MHA
LRKV-128 r= 128116.7% 2.877 0.873 Best, but>100% cache
MLA Latent Dimension Ablation
MLA-128 dc = 1288.3% 2.901 0.876 Minimal memory
MLA-192 dc = 19212-5% 2.898 0.876 –
MLA-384 dc = 38425.0% 2.894 0.875 Comparable to LRKV-64
Baselines
Standard MHA 6 KV heads 100% 2.903 0.878 Baseline
GQA 3 groups 50.0% 2.918 0.883 –
MQA 1 KV head 16.7% 2.929 0.886 Severe quality loss
Key Findings.The ablation reveals that performance improves monotonically with rank, confirming that representational
capacity (not geometric properties of the factorization) determines modeling quality. Ranks around r≈0.36 –0.43×d h
(46–55 for dh = 128 ) provide the optimal tradeoff: sufficient capacity to exceed Standard MHA performance while
achieving substantial cache reduction.
Comparing methods at matched cache budgets reveals LRKV’s architectural advantage: LRKV at 41.7% cache (r= 32 ,
BPB=0.877) slightly underperforms MLA at 50.0% cache (dc = 384, BPB=0.875), but LRKV at 66.7% cache (r= 64 ,
BPB=0.875) matches MLA-384 performance. However, LRKV avoids the sequence-length-dependent reconstruction
overhead inherent to MLA’s latent compression (see Appendix F), providing better inference latency characteristics at long
context lengths.
All ablation models are trained for 100B tokens (rather than compute-optimal allocation) to reduce variance and enable
high-confidence comparison of architectural capacity at fixed training budget.
D.5. Long Context Experiments
To evaluate LRKV’s effectiveness at extended sequence lengths, we conduct 8192-token context experiments using 512M
parameter models trained on 50B tokens (153.6 ExaFLOPs).
Configuration.The 512M models use 12 attention heads with dmodel = 1536 and dhead = 128, with 24 layers matching
the 1.2B scale architecture. We compare LRKV (r= 51 , 48.2% cache), GQA (4 groups, 33.3% cache), MLA (dc = 256,
8.3% cache), and MQA (8.3% cache) against Standard MHA baseline.
Batching Strategy.To maintain computational throughput while accommodating 4 × longer contexts, we adjust batch size
to keep total tokens per batch constant. Standard 2048-context runs use batch size 16 (32,768 tokens per batch); 8192-context
runs use batch size 4 (32,768 tokens per batch). This ensures comparable gradient noise and optimizer dynamics while
scaling to long contexts.
Results.Table 9 presents validation performance at 8192-token context length after 50B tokens (153.6 ExaFLOPs).
LRKV achieves the strongest performance, outperforming Standard MHA by 2.7% while using less than half the KV cache
(48.2% vs 100%). All KV-efficient methods (LRKV , GQA, MLA, MQA) outperform the full Standard MHA baseline at 8K
context, suggesting that some degree of KV compression may provide implicit regularization benefits at longer sequences.
However, LRKV’s 2.7% advantage over MHA (compared to GQA’s 1.3%, MLA’s 0.9%, and MQA’s 0.4%) demonstrates
that head-specific low-rank residuals are particularly effective for maintaining quality at extended context lengths.
These results validate LRKV’s applicability to long-context scenarios where memory efficiency is critical, demonstrating that
25

<!-- page 26 -->

Low-Rank Key Value Attention
Table 9.Long context performance (512M models, 8K context, 50B tokens).Degradation computed relative to MHA baseline.
Method KV Cache CE Loss↓BPB↓vs. MHA
Standard MHA 100% 2.740 0.494 baseline
LRKV48.2%2.665 0.481 -2.7%
GQA 33.3% 2.704 0.488 -1.3%
MLA 8.3% 2.714 0.489 -0.9%
MQA 8.3% 2.728 0.492 -0.4%
low-rank factorization preserves head diversity more effectively than aggressive sharing (MQA, GQA) or latent compression
(MLA) approaches at extended sequence lengths.
D.6. Downstream Evaluation Tasks
After midtraining, we evaluate all models on five standard benchmarks:
•ARC-Easy and ARC-Challenge(Clark et al., 2018): Question answering requiring scientific reasoning
•MMLU(Hendrycks et al., 2021): Multi-task language understanding across 57 subjects
•GSM8K(Cobbe et al., 2021): Grade-school math word problems
•HumanEval(Chen et al., 2022): Python code generation from docstrings
All evaluations follow standard zero-shot or few-shot protocols as specified in the original benchmark papers.
E. Pretraining-Downstream Correlation Analysis
0.7200.7220.7240.7260.728
Pretraining BPB (lower is better)
34
35
36
37
38Downstream Performance (%)
Pretraining vs Downstream Performance
MLA
MQA
GQA
Standard MHA
LRKV
Trend (R²=0.786)
Figure 19.Pretraining quality predicts downstream performance.Scatter plot relating pretraining validation BPB (x-axis, inverted)
to downstream combined performance (y-axis) across all attention mechanisms at 2.5B scale. A strong positive correlation emerges
(R= 0.786 ), demonstrating that models with better pretraining performance consistently achieve higher downstream scores. LRKV
occupies the optimal position with both the lowest pretraining BPB (0.719) and highest downstream performance (37.9%), while Standard
MHA achieves 0.723 BPB and 35.3% downstream.
To validate the relationship between pretraining quality and downstream performance, we analyze the correlation between
validation BPB and combined benchmark scores across all architectures after midtraining (Figure 19). The strong positive
26

<!-- page 27 -->

Low-Rank Key Value Attention
correlation (R²=0.786) demonstrates that models achieving lower pretraining loss consistently deliver superior downstream
performance. LRKV achieves the lowest pretraining BPB (0.719) and highest downstream score (37.9%), while Standard
MHA attains 0.723 BPB and 35.3% downstream.
This correlation suggests that architectural improvements that enhance language modeling performance are not orthogonal
to, but rather aligned with, practical task capabilities. The consistent 2.6 percentage point downstream advantage LRKV
maintains over Standard MHA directly stems from its BPB pretraining improvement, confirming that low-rank KV
factorization provides fundamental capacity gains that manifest across both pretraining and downstream evaluation regimes.
This relationship validates that LRKV’s architectural improvements during pretraining directly translate to enhanced
downstream capabilities, confirming the effectiveness of low-rank KV factorization for both language modeling and
task-specific adaptation.
F. Additional Experimental Details and Computational Analysis
F.1. Batching and Optimization Hyperparameters
For models up to 256M parameters, we use a global batch size of16sequences (2048 tokens each). For the larger 1B–6.3B
models, we reduce the physical batch size to8sequences and applygradient accumulation of 2 stepsto maintain an
equivalent effective batch size. This setup ensures comparable gradient noise scale and optimizer dynamics across all model
sizes. All runs employ weight decay of 0.1, Adamβ 1 = 0.9,β 2 = 0.95, and gradient clipping at 1.0.
F.2. Memory-Computation Trade-offs in Attention Mechanisms
Modern attention mechanisms present a fundamental trade-off between cache memory (storage) and per-token computation
(latency). We analyze this trade-off across standard attention, query-sharing variants, and compression-based approaches.
The Compression Paradigm: Lossy vs. Lossless.Attention mechanisms can be understood through the lens of
compression theory:
• Multi-Latent Attention (MLA)implementslossy compression: it compresses K/V representations into a low-
dimensional latent space Z=XW down, caching only Z. During generation, full K/V need not be explicitly materialized;
instead, interaction with the cached latent requires per-step projection overhead, e.g., qK ⊤ = (qW K
up )Z ⊤ and
aV= (aZ)W V
up . This introduces additional per-token computation proportional to the latent dimension, and scales
linearly with sequence length through dot products withZ ⊤.
• LRKVimplementsnear-lossless compression with additive structure: it caches both shared full-rank features ( ¯K, ¯V )
and compact per-head latents (RK
h , RV
h ). It avoids explicitly materializing per-head Kh, Vh tensors; instead, attention
can be computed using associative forms (Eq. 6–7). Beyond the unavoidableO(T) scan over cached tokens in attention,
LRKV introduces onlyO(Hrd h)additional per-step projection work that does not grow withT.
This distinction is critical for autoregressive generation: both MLA and LRKV incur the unavoidable O(T) cost to compare
against cached tokens, but MLA introduces additional per-step projection work tied to the latent bottleneck, whereas LRKV’s
extra work depends only onrand does not increase withT.
Quantitative Analysis: Memory vs. Latency.We compare the memory footprint and inference-time computational
characteristics of different attention mechanisms during autoregressive generation. Table 10 reports (1) the KV cache size
stored per token and (2) theadditional reconstruction overheadrequired per generation step,beyondthe standard O(T)
attention computation inherent to cached autoregressive decoding.
Key observations.
1. Standard attention, MQA, and GQA incur no additional reconstruction cost at inference time, as per-head or shared
key/value tensors are directly cached.
2. MLA achieves the smallest cache footprint but introduces an additional latent-to-head expansion step whose cost
27

<!-- page 28 -->

Low-Rank Key Value Attention
Table 10.Memory and inference-time characteristics of attention mechanisms. Cache size is measured per token. Reconstruction overhead
denotesadditionalcomputation beyond standardO(T)cached attention.
Method Cache / Token Extra Cost / StepT-Dependence
Standard Attn.2Hd h O(1)None
MQA2d h O(1)None
GQA (G=3)2Gd h O(1)None
MLA (dc=128)d c O(T d cdh)Linear
LRKV(r=16)2(d h+Hr)O(Hrd h)None
Extra cost refers to reconstruction needed to obtain per-head K/V representations. MLA requires latent-to-head expansion over the cached
sequence unless expanded K/V are stored or fused kernels are used.
scales with the cached sequence length T , unless expanded representations are stored (which increases memory) or
specialized fused kernels are used.
3. LRKV trades a modest increase in cache size relative to MLA for sequence-length-independent reconstruction
overhead. This avoids T -dependent expansion during generation while preserving per-head expressivity through
low-rank residuals.
4. As a result, LRKV occupies an intermediate point in the memory–latency design space: it substantially reduces KV
memory relative to full attention while avoiding the sequence-dependent reconstruction overhead introduced by more
aggressive compression schemes.
G. KV Cache Memory Measurements
This appendix presents detailed measurements of KV cache memory usage during inference for all attention mechanisms
evaluated in this work.
G.1. Measurement Methodology
All memory measurements are obtained by instantiating cache tensors with the exact shapes used during inference and
measuring their memory footprint using PyTorch’stensor.numel() and element size() methods. Measurements
use bfloat16 precision (2 bytes per element) with batch size 1 and sequence length 2048 tokens, matching our experimental
configuration.
G.2. Cache Structure by Method
Table 11 summarizes the cache structure for each attention mechanism, showing tensor shapes and memory formulas for
inference with batch size B, sequence length T , number of layers L, number of heads H, head dimension dh, group size G,
latent dimensiond c, and rankr.
Table 11.KV cache structure by attention mechanism.All measurements use bfloat16 precision (2 bytes per element).
Method Cached Tensor/s Total Memory (bytes)
Standard MHA Shape:(L, B, H, T, d h) 2×L×B×H×T×d h ×2
MQA Shape:(L, B,1, T, d h) 2×L×B×T×d h ×2
GQA Shape:(L, B, H/G, T, d h) 2×L×B×(H/G)×T×d h ×2
MLA Shape:(L, B, T, d c)(latents)L×B×T×d c ×2
LRKV (Optimized) Shared:(L, B, T, d h)
Latents:(L, B, H, T, r)
2×L×B×T×(d h +H×r)×2
28

<!-- page 29 -->

Low-Rank Key Value Attention
G.3. Measured Results
Table 12 presents measured memory usage across all three model scales. The optimized LRKV implementation (using
LRKVCache from lrkv attention fused.py) achieves substantial memory savings compared to standard MHA
while maintaining full representational capacity.
Table 12.Measured KV cache memory during inference.All values measured for T=2048 tokens, batch size 1, bfloat16 precision,
using rankr= 64for LRKV and latent dimensions as specified in main paper.
Method 128M 2.5B 6.3B
Standard MHA 72.0 MB 648.0 MB 1152.0 MB
MQA 12.0 MB (16.7%) 36.0 MB (5.6%) 36.0 MB (3.1%)
GQA 36.0 MB (50.0%) 216.0 MB (33.3%) 72.0 MB (6.2%)
MLA 24.0 MB (33.3%) 216.0 MB (33.3%) 288.0 MB (25.0%)
LRKV 48.0 MB (52.6%) 360.0 MB (48.4%) 612.0 MB (45.1%)
G.4. Scale-Dependent Efficiency
LRKV’s cache percentage relative to standard MHA improves with model scale:
Ratio= dh +H·r
H·d h
= 1
H + r
dh
128M (H=6): 1
6 + 64
128 = 0.526 = 52.6%
2.5B (H=18): 1
18 + 64
128 = 0.484 = 48.4%
6.3B (H=32): 1
32 + 64
128 = 0.451 = 45.1%
This demonstrates that LRKV’s efficiencyincreaseswith model scale: larger models with more heads benefit more from the
shared component, achieving greater relative memory savings while maintaining fixed rankr= 64.
H. Limitations and Future Work
H.1. Limitations
Our experiments focus on decoder-only language models at scales up to 6.3B parameters. The design space for large-scale
Transformer training is extremely large, spanning model size, training data, optimization schedules, attention mechanisms,
and hardware/software configurations, many of which interact in nontrivial ways. As a result, it is infeasible to exhaustively
benchmark all combinations or to claim that any single configuration is universally optimal across setups.
Our empirical evaluation therefore samples this space using widely adopted architectures, training recipes, and system
implementations on modern accelerator hardware. While this provides a representative and practically relevant assessment,
different choices of model scale, data regime, training duration, or system optimizations may shift the precise efficiency–
performance trade-offs.
To facilitate independent validation and extension, we will release our training code and configuration details, enabling
replication and exploration across alternative setups. We also note that the relative efficiency of different KV compression
strategies may vary with hardware characteristics such as memory bandwidth, kernel fusion, and cache behavior.
29
