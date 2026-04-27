# references/178_the_residual_stream_is_all_you_need_on_the_redundancy_of_the_kv_cache_in_transformer_inference.pdf

<!-- page 1 -->

The Residual Stream Is All You Need: On the Redundancy of the KV
Cache in Transformer Inference
Kaleem Ullah Qasima, Jiashu Zhanga,∗, Muhammad Kafeel Shaheena, Razan Alharitha and
Heying Zhanga
aSchool of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, 611756, China
ARTICLE INFO
Keywords:
KV cache
Residual stream
Transformer inference
Bounded memory
KV-Direct
Mechanistic interpretability
Attention redundancy
ABSTRACT
Thekey-value(KV)cacheiswidelytreatedasessentialstateintransformerinference,andalargebody
of work engineers policies to compress, evict, or approximate its entries. We prove that this state is
entirelyredundant:keysandvaluesateverylayeraredeterministicprojectionsoftheresidualstream,
andrecomputingthemfromasingleresidualvectorpertokenincursexactlyzeroreconstructionerror,
notapproximately,butbit-identically.Weverifythisacrosssixmodelsfromfourarchitecturefamilies
(135M to 4B parameters). Cross-task residual patching at every layer produces𝐷KL = 0between
patched and original output distributions, confirming that the residual stream satisfies a Markov
property and is the sole information-carrying state. Removing the cache entirely and recomputing
from scratch yields token-identical output under greedy decoding on all models tested. We build on
this result with KV-Direct, a bounded-memory inference scheme that checkpoints residual vectors
(5 KB per token on Gemma 3-4B) instead of full KV pairs (136 KB), recomputing keys and values
on demand. Over 20 conversation turns, KV-Direct holds peak memory at 42 MB while the standard
cache grows past 103 MB. Against five eviction baselines (H2O, StreamingLLM, SnapKV, TOVA,
window-only),KV-Directmaintains100%tokenmatchateverycachebudget;allbaselinesdegradeto
5–28%.Aper-operationlatencyanalysisshowsrecomputationrunsupto5×fasterthanreadingcached
tensorsatmoderatebatchsizes.Codeisavailableathttps://github.com/Kaleemullahqasim/KV-Direct.
1. Introduction
The key-value (KV) cache is a primary memory bottle-
neck in large language model inference. During autoregres-
sive decoding, the standard approach stores precomputed
keys and values for every past token at every layer. For a
4-billion parameter model, each token adds 136 KB to the
cache;a20-turnconversationaccumulatesover100MB,and
at the 12B-parameter scale this approaches a gigabyte.
This cost has driven a large body of work on KV cache
compression: eviction policies [1, 2], quantization [3, 4],
grouped-query attention [5], and paged memory manage-
ment [6]. All of these treat the KV cache as containing
information that must be preserved or approximated. This
paper challenges that assumption.
Keysandvaluesateachlayeraredeterministicfunctions
of the residual stream: they are obtained by applying frozen
weightmatrices(and,forkeys,adeterministicpositionalro-
tation)tothenormalisedresidualvector.Thecachetherefore
stores derived quantities rather than unique state. That this
relationship is implicit in the transformer specification has
not prevented the field from treating the cache as a primary
information store. The practical consequence has not been
systematically examined:the KV cache can be eliminated
entirely without changing a single output token. We verify
this empirically: under greedy decoding, generating 30 to-
kens with and without the cache yields 100% token identity
acrossallsixmodelstested(fourarchitecturefamilies,135M
∗Corresponding author
kaleem@my.swjtu.edu.cn(K.U. Qasim);jszhang@home.swjtu.edu.cn
(J. Zhang);kafeel@my.swjtu.edu.cn(M.K. Shaheen);
razanalharith@my.swjtu.edu.cn(R. Alharith);hey_zhang@qq.com(H. Zhang)
ORCID(s):
to4Bparameters).Theidentityholdsforfull-attentionlayers
universally; for sliding-window layers, value reconstruction
remains exact while key reconstruction requires additional
position state (Section 5.1). The cache provides a speed
advantage but carries no additional information.
This redundancy extends beyond individual projections
to the full computational state. The residual stream satisfies
a Markov property: future outputs depend on the input
history only through the current residual vectors. Cross-
task patching experiments confirm this at every layer, with
𝐷KL = 0.0between the patched and original output distri-
butions (Section 3). Zero-shot HellaSwag accuracy [7] and
WikiText-2 perplexity under KV-Direct exactly match full
caching. Because the property follows from the pre-norm
transformer architecture, it holds regardless of model scale;
our experiments span 135M to 4B parameters across four
architecture families.
Theseresultsleadtoapracticalinferencescheme.Rather
than caching K and V at every layer, the residual stream
can be checkpointed instead (one vector per token, shared
across all layers) and KV entries recomputed on demand.
Because the residual vector is shared across layers while
KV entries are per-layer, each checkpoint is substantially
smaller:onGemma3-4B-IT,5KBpertokenversus136KB
for the full KV pair (27×reduction). We call this scheme
KV-Directandevaluateitover20conversationturns,where
thestandardcachegrowsto103MBwhileKV-Directholds
at42MB(2.5×peakmemoryreduction).Aper-operationla-
tencyanalysisshowsthatrecomputingKVfromcheckpoints
takes0.2–0.3×the time of reading cached tensors at 500
evicted tokens, as memory bandwidth rather than computa-
tion becomes the bottleneck. Against five eviction baselines
Qasim et al.:Preprint submitted to ElsevierPage 1 of 13
arXiv:2603.19664v1  [cs.LG]  20 Mar 2026

<!-- page 2 -->

The Residual Stream Is All You Need
(H2O[1],StreamingLLM[8],SnapKV[9],TOVA[10],and
window-only eviction), KV-Direct preserves 100% token
match at every cache budget while all baselines degrade to
5–28% match with KL divergences of 7–14.
Our contributions:
1 Empirical proof that KV cache entries are exactly re-
constructible from the residual stream (zero error across
six models spanning four architecture families: LLaMA,
Qwen2, Qwen3, Gemma 3), with a precise characterisation
ofthesliding-windowboundary:valuereconstructionisuni-
versal, while key reconstruction in window-relative RoPE
layers requires the local position index (Section 5.1).
2 Verification that removing the KV cache entirely yields
token-identical output under greedy decoding on all full-
attention models, that cross-task residual patching produces
𝐷KL = 0.0at every layer, and that zero-shot HellaSwag ac-
curacy and WikiText-2 perplexity are fully preserved (Sec-
tions 5.2–5.4).
3 Analysis of the per-token memory ratio across model
scales (ranging from6.9×on Qwen2.5-0.5B to56×on
Qwen3-0.6B)alongsideeffectiverankmeasurementsreveal-
ing strong head-level heterogeneity (median rank 70 of 256
at 90% spectral energy on Gemma 3-4B-IT) (Section 5.7).
4 KV-Direct, a bounded-memory inference scheme that
reduces peak KV memory by2.5×over 20 conversation
turns (103 MB→42 MB on Gemma 3-4B-IT) while the
standard cache grows without bound, and a latency analysis
showingthatper-operationKVrecomputationfromresidual
checkpoints is up to5×faster than reading the equivalent
cached tensors (Sections 5.5–5.6).
The remainder of the paper is organised as follows.
Section 2 surveys related work across five categories of KV
cacheoptimization.Section3formalisestheresidualstream
hypothesis and derives the theoretical foundations for KV
reconstruction. Section 4 describes models, baselines, and
experimental settings. Section 5 answers our four research
questions with empirical evidence and discusses practical
implications. Section 6 addresses limitations and future di-
rections.
2. Related Work
The KV cache is the primary memory bottleneck in
autoregressive transformer inference, and a large body of
work targets its reduction. Token-level eviction methods
select which entries to retain based on attention impor-
tance [1, 2, 9], attention-sink patterns [8], RNN-style state
compression [10], layer-dependent budgets [11, 12], or
query-aware and head-aware scoring [13, 14, 15, 16]. All
treat eviction as permanent information loss. Quantization
reduces the per-entry footprint through asymmetric [17],
sensitivity-weighted [18], error-decomposed [19], coupled-
channel [20], or MPO-based [21] schemes (see [4] for a
survey), while low-rank methods decompose KV projec-
tions into learned factors [22], exploit the low-dimensional
structureofkeyvectors[23],orcomputeattentiondirectlyin
SVD-reduced spaces [24]. Both families introduce approxi-
mation error by design.
Architectural modifications take a different route by
sharing KV heads across query groups [25, 5], across lay-
ers [26, 27, 28], or through learned latent bottlenecks such
as DeepSeek-V2’s Multi-head Latent Attention [29], which
achieves 93.3% memory reduction by compressing KV into
a low-dimensional representation and up-projecting on de-
mand. MiniCache [30] interpolates KV states between ad-
jacent layers. These approaches require model retraining or
architecture changes.
Recomputation-basedinferencetradescomputeformem-
orywithoutmodifyingthemodel.KVPR[31]transfersacti-
vation checkpoints between CPU and GPU and recomputes
partial KV tensors on-device. HybridServe [32] adaptively
balancescachingagainston-the-flyreconstruction.FlashAt-
tention [33, 34] recomputes intermediate attention matri-
ces within fused kernels, and gradient checkpointing [35]
applies the same principle during training. KVPR and Hy-
bridServe are the most closely related systems, but neither
formalises a zero-information-loss guarantee nor identifies
the sliding-window boundary where exact reconstruction
breaks down.
Onthetheoreticalside,thecircuitsframework[36]treats
the residual stream as a shared communication channel
between attention and MLP blocks. Induction head analy-
sis[37]establishedthatstructuredinformationflowsthrough
this channel across positions and layers. Shai et al. [38]
proved that belief states are linearly encoded in the residual
stream,Heetal.[39]showedthat50%ofattentioncomputa-
tion can be pruned without affecting outputs, and causal in-
terventionmethods[40,41]providedtheactivation-patching
methodology we build on. This prior work uses the residual
stream to explain what transformers compute; we use it
to eliminate redundant state and build a bounded-memory
inference scheme that preserves exact output fidelity.
3. Method
Weworkwiththestandarddecoder-onlytransformer[42]:
an embedding layer followed by𝐿identical blocks, each
containingmulti-headself-attentionandafeed-forwardnet-
work. Both sub-layers use residual connections [43], form-
ing theresidual stream𝐡(𝓁) that flows through the network.
Modern variants apply RMSNorm before each sub-layer
(pre-norm) and rotary position embeddings (RoPE) [44].
During autoregressive generation, the KV cache stores key
andvaluevectorsfrompreviousstepstoavoid𝑂(𝑡 2)recom-
putation, requiring2⋅𝐿⋅𝑛 kv ⋅𝑑 head parameters per token.
Following the circuits framework [36], we treat the residual
streamastheprimaryobjectofcomputation:attentionheads
and MLPs read from and write to this stream, and every
intermediate quantity is a function of𝐡(𝓁) at the relevant
layer.
This section formalises the claim that the KV cachecar-
riesnoinformationbeyondtheresidualstream.Westatethe
Markov property, derive reconstruction identities, analyse
Qasim et al.:Preprint submitted to ElsevierPage 2 of 13

<!-- page 3 -->

The Residual Stream Is All You Need
the projection geometry, and describe a bounded-memory
inference scheme built on these results.
3.1. The Residual Markov Property
We first define the normalisation used throughout. For a
vector𝐱∈ℝ 𝑑, RMSNorm computes
RMSNorm(𝐱) = 𝐱
‖𝐱‖RMS
⊙𝜸,(1)
‖𝐱‖RMS =
√
1
𝑑
∑𝑑
𝑖=1 𝑥2
𝑖 ,(2)
where𝜸∈ℝ 𝑑 is a learned scale vector frozen at inference
time.UnlikeLayerNorm,RMSNormomitsmean-centering,
making it a positive-homogeneous function of degree zero:
RMSNorm(𝛼𝐱) =RMSNorm(𝐱)for any𝛼 >0.
Definition3.1(ResidualMarkovProperty).Let𝐡 (𝓁)
𝑝 denote
theresidualstreamatlayer𝓁andposition𝑝.Thetransformer
satisfies theresidual Markov property at layer𝓁if the
output distribution over next tokens is fully determined by
thecollection{𝐡 (𝓁)
𝑝 ∶𝑝= 1,…, 𝑡},independentofhowthat
state was produced.
Derivation.Consider layer𝓁of a pre-norm transformer.
Let ̄𝐡(𝓁)
𝑝 =RMSNorm(𝐡 (𝓁)
𝑝 )denote the normalised residual.
The attention sub-layer first computes, for each headℎ∈
{1,…, 𝑛 ℎ},
𝐐(ℎ)
𝑝 =RoPE(̄𝐡(𝓁)
𝑝 𝐖(𝓁,ℎ)
𝑞 , 𝑝 ),(3)
𝐊(ℎ)
𝑗 =RoPE(̄𝐡(𝓁)
𝑗 𝐖(𝓁,ℎ)
𝑘 , 𝑗 ),(4)
𝐕(ℎ)
𝑗 = ̄𝐡(𝓁)
𝑗 𝐖(𝓁,ℎ)
𝑣 ,(5)
where𝐖 (𝓁,ℎ)
𝑞 ,𝐖 (𝓁,ℎ)
𝑘 ,𝐖 (𝓁,ℎ)
𝑣 are frozen weight matrices
mappingℝ 𝑑hidden →ℝ 𝑑head, and RoPE is a deterministic
position-dependent rotation defined in Section 3.2. The
attention output at position𝑝for headℎis
𝛼(ℎ)
𝑝𝑗 = sof tmax𝑗
⎛
⎜
⎜⎝
𝐐(ℎ)
𝑝 𝐊(ℎ)
𝑗
⊤
√
𝑑head
⎞
⎟
⎟⎠
,(6)
head(ℎ)
𝑝 =
𝑡∑
𝑗=1
𝛼(ℎ)
𝑝𝑗 𝐕(ℎ)
𝑗 .(7)
Themulti-headoutputisconcatenatedandprojectedthrough
the output matrix𝐖(𝓁)
𝑜 ∈ℝ 𝑛ℎ𝑑head×𝑑hidden:
̂𝐡(𝓁)
𝑝 =𝐡 (𝓁)
𝑝 + [head(1)
𝑝 ; … ;head (𝑛ℎ)
𝑝
]𝐖(𝓁)
𝑜 .(8)
The MLP sub-layer operates position-wise:
𝐡(𝓁+1)
𝑝 = ̂𝐡(𝓁)
𝑝 +MLP (𝓁)(RMSNorm(̂𝐡(𝓁)
𝑝 )).(9)
Every operation in (3)–(9) takes the set{𝐡(𝓁)
𝑝 }𝑡
𝑝=1 as input
andusesonlyfrozenparameters.Thefinaltokendistribution
is𝑝(𝑥 𝑡+1) = sof tmax(𝐡 (𝐿)
𝑡 𝐖vocab).Byinductionoverlayers
𝓁,𝓁+1,…, 𝐿, we obtain:
Proposition 3.1(Residual Sufficiency).For a pre-norm
transformer with𝐿layers, the output distribution𝑝(𝑥 𝑡+1 ∣
𝑥≤𝑡)is a deterministic function of{𝐡 (𝓁)
𝑝 }𝑡
𝑝=1 for any𝓁∈
{0,1,…, 𝐿}. It follows that the KV cache carries zero
additional information:
𝐼(𝐊(1∶𝐿),𝐕 (1∶𝐿);𝑥 𝑡+1 ∣ {𝐡 (𝓁)
𝑝 }𝑡
𝑝=1
)= 0.(10)
3.2. KV Reconstruction from the Residual Stream
Rotary position encoding.RoPE encodes position𝑝by
rotating consecutive pairs of the projected vector. For𝐱∈
ℝ𝑑head:
RoPE(𝐱, 𝑝)2𝑖−1 =𝑥 2𝑖−1 cos𝜃 (𝑝)
𝑖
−𝑥 2𝑖 sin𝜃 (𝑝)
𝑖 ,(11)
RoPE(𝐱, 𝑝)2𝑖 =𝑥 2𝑖−1 sin𝜃 (𝑝)
𝑖
+𝑥 2𝑖 cos𝜃 (𝑝)
𝑖 ,(12)
where𝜃 (𝑝)
𝑖 =𝑝⋅𝑏 −2𝑖∕𝑑head and𝑏is a fixed base (typically
10,000). In matrix form, RoPE(𝐱, 𝑝) =𝐑 𝑝𝐱, where𝐑 𝑝 ∈
ℝ𝑑head×𝑑head is an orthogonal block-diagonal rotation matrix
satisfying𝐑 ⊤
𝑝 𝐑𝑝 =𝐈. The key property for reconstruction is
that𝐑 𝑝 is a deterministic function of the absolute position
index𝑝alone.
Reconstruction identity.The KV cache stores𝐊 (𝓁) and
𝐕(𝓁) for every past token at every layer. Using the notation
̄𝐡(𝓁)
𝑝 =RMSNorm(𝐡 (𝓁)
𝑝 )fromSection3.1,thecachedentries
can be reconstructed exactly:
𝐊(𝓁)
recon, 𝑝 =𝐑 𝑝 ̄𝐡(𝓁)
𝑝 𝐖(𝓁)
𝑘 ,(13)
𝐕(𝓁)
recon, 𝑝 = ̄𝐡(𝓁)
𝑝 𝐖(𝓁)
𝑣 .(14)
Thevalueprojectionin(14)involvesnopositionalencoding
and holds universally across all architectures.
Proposition 3.2(Exact KV Reconstruction).For any full-
attention layer𝓁using absolute RoPE, the cached and re-
constructed KV entries are identical:
𝐊(𝓁)
cached, 𝑝 ≡𝐊 (𝓁)
recon, 𝑝,
𝐕(𝓁)
cached, 𝑝 ≡𝐕 (𝓁)
recon, 𝑝,(15)
for all positions𝑝and layers𝓁. The reconstruction error is
exactly zero, not approximately.
Proof.Both the cached and reconstructed paths apply the
same sequence of deterministic operations to𝐡 (𝓁)
𝑝 : RM-
SNorm (1), linear projection (𝐖(𝓁)
𝑘 or𝐖 (𝓁)
𝑣 ), and for keys,
rotation𝐑 𝑝 at absolute position𝑝(11)–(12). All parameters
are frozen at inference. The two computation paths are al-
gebraically identical, yielding zero error under any floating-
point precision that preserves operation ordering.
Qasim et al.:Preprint submitted to ElsevierPage 3 of 13

<!-- page 4 -->

The Residual Stream Is All You Need
Standard KV Cache
Unbounded memory, full accuracy
Input Token
RMSNorm
Wq
Attention
⨁ → MLP → ⨁
Next Token
Wk Wv
KV Cache
Pos 1 ∶ 𝑲1, 𝑽1
Pos 2 ∶ 𝑲2, 𝑽2
Pos 3 ∶ 𝑲3, 𝑽3
136 KB/token
𝒙𝒕
Residual Stream h
Memory: grows without bound
𝑂(𝑇 ×  2𝐿 ×  nkv × dhead)
Q
Sliding Window Eviction
Bounded memory, lossy
Window Cache (Budget B)
Pos T − B + 2 ∶ 𝑲, 𝑽
Pos T: 𝑲, 𝑽
Keeps last B tokens only
from evicted tokens
Memory: bounded
But 5-28% token match, KL 7 -14
discard
Pos 1 ∶ 𝑲1, 𝑽1
Pos 2 ∶ 𝑲2, 𝑽2
Pos 3 ∶ 𝑲3, 𝑽3
Evicted - Permanently Lost
Pos T − B + 1 ∶ 𝑲, 𝑽
…
… Attention
Input Token
RMSNorm
Wq
⨁ → MLP → ⨁
Next Token
Wk Wv
𝒙𝒕
Residual Stream h
Q
KV-Direct (Ours)
Bounded memory, lossless
Attention
Input Token
RMSNorm
Wq
⨁ → MLP → ⨁
Next Token
Wk Wv
𝒙𝒕
QResidual Stream h
Memory: bounded
100% token match, KL ≈0
Active Cache
 Fixed (Budget B)
Recent B token: K,V
𝑲all, 𝑽all
Evict store 𝒉𝒍
Residual Checkpoints
h1
5 KB / token
h2 h𝑇⋯
Recompute K, V
RMSNorm (h) ∙ Wk, Wv
Figure 1:Three inference regimes compared.(Left)Standard KV cache: stores all K/V pairs, memory grows as𝑂(𝑇)with
sequence length.(Centre)Sliding window eviction: bounds memory to the last𝐵tokens but permanently discards evicted KV
entries, yielding 5–28% token match and high KL divergence.(Right)KV-Direct: evicted KV entries are replaced by residual
stream checkpoints (5 KB/token for Gemma3-4B), from which exact K and V are recomputed on the fly, achieving bounded
memory with 100% token match and𝐷KL ≈ 0.
Sliding-window boundary.In sliding-window attention,
the key at position𝑗within a window starting at offset𝑤
is rotated by𝐑𝑗−𝑤 rather than𝐑 𝑗. Reconstruction from the
residual uses absolute position𝑗, producing a mismatch:
‖𝐊recon −𝐊 cached‖=‖(𝐑 𝑗 −𝐑 𝑗−𝑤) ̄𝐡(𝓁)
𝑗 𝐖(𝓁)
𝑘 ‖,(16)
which is non-zero whenever𝑤≠0. Value reconstruction is
unaffected because𝐕involves no rotation. On Gemma 3-
4B-IT, this boundary affects 29 of 34 layers (the sliding-
window layers), while all 5 global-attention layers satisfy
Proposition 3.2 exactly.
Corollary 3.3(Zero Conditional Entropy).Since the map-
ping𝐡 (𝓁)
𝑝 ↦(𝐊 (𝓁)
𝑝 ,𝐕 (𝓁)
𝑝 )is deterministic for full-attention
layers,
𝐻(𝐊(𝓁),𝐕 (𝓁) ∣𝐡 (𝓁))= 0 ∀𝓁.(17)
Themutualinformationequalsthefullentropyofthecache:
𝐼(𝐊 (𝓁),𝐕 (𝓁);𝐡 (𝓁)) =𝐻(𝐊 (𝓁),𝐕 (𝓁)). The residual stream
captures the complete information content of the KV cache.
3.3. Bilinear Attention Form and Effective Rank
In standard scaled dot-product attention, the score be-
tween query position𝑖and key position𝑗at headℎcan be
written as a bilinear form over the residual stream:
𝑎(ℎ)
𝑖𝑗 =
(𝐡𝑖𝐖(ℎ)
𝑞 ) (𝐡𝑗𝐖(ℎ)
𝑘 )⊤
√
𝑑head
=
𝐡𝑖 𝐌(ℎ) 𝐡⊤
𝑗
√
𝑑head
,(18)
where𝐌 (ℎ) =𝐖 (ℎ)
𝑞 𝐖(ℎ)
𝑘
⊤
∈ℝ 𝑑hidden×𝑑hidden and we omit
RMSNorm and RoPE for clarity. The matrix𝐌(ℎ) deter-
mines which directions in the residual stream produce high
attention scores.
Architecturally, rank(𝐌(ℎ))≤𝑑 head because both𝐖 (ℎ)
𝑞
and𝐖 (ℎ)
𝑘 mapfromℝ 𝑑hidden toℝ 𝑑head.Let𝐌 (ℎ) =𝐔𝚺𝐕 ⊤ de-
notethesingularvaluedecomposition,where𝚺=diag(𝜎 1,…, 𝜎 𝑑head )
with𝜎 1 ≥⋯≥𝜎 𝑑head ≥0. We define thespectral energy
fractioncaptured by the top𝑟components as
𝐸(𝑟) =
∑𝑟
𝑖=1 𝜎2
𝑖
∑𝑑head
𝑖=1 𝜎2
𝑖
,(19)
and theeffective rankat threshold𝜏as
𝑟∗(𝜏) = min { 𝑟∶𝐸(𝑟)≥𝜏 } .(20)
Rank-truncated attention approximation.A rank-𝑟ap-
proximation𝐌 (ℎ)
𝑟 = ∑𝑟
𝑖=1 𝜎𝑖 𝐮𝑖𝐯⊤
𝑖 yields approximate atten-
tion scores
̃ 𝑎(ℎ)
𝑖𝑗 =
𝐡𝑖 𝐌(ℎ)
𝑟 𝐡⊤
𝑗
√
𝑑head
,(21)
with per-entry error bounded by
|||𝑎(ℎ)
𝑖𝑗 −̃ 𝑎(ℎ)
𝑖𝑗
||| ≤
𝜎𝑟+1 ‖𝐡𝑖‖ ‖𝐡𝑗‖
√
𝑑head
.(22)
Qasim et al.:Preprint submitted to ElsevierPage 4 of 13

<!-- page 5 -->

The Residual Stream Is All You Need
Algorithm 1KV-Direct Inference (Single Decoding Step)
def KV_DIRECT ( x_t , C , K , B , L ) :
# C : residual checkpoints , K : KV cache ( B slots / layer )
h = EMBED ( x_t )
for l in range (1 , L +1) :
h_norm = RMSNORM ( h )Eq. 1
# Current token projections
Q = h_norm * W_q [ l ]Eq. 3
K_t = h_norm * W_k [ l ]
V_t = h_norm * W_v [ l ]Eqs. 4, 5
# Recompute evicted KV from checkpoints
K_old , V_old = RECOMPUTE_KV (C , l )Eqs. 13, 14
# Assemble full KV sequence
K_all = CONCAT ( K_old , K [ l ]. keys , K_t )
V_all = CONCAT ( V_old , K [ l ]. vals , V_t )
# Attention + residual updates
out = ATTENTION (Q , K_all , V_all )Eq. 8
h = h + out
h = h + MLP ( RMSNORM ( h ) )Eq. 9
# Eviction policy
if LEN ( K [ l ]) > B :
EVICT_OLDEST ( K [ l ])
STORE_RESIDUAL (C , l )Eq. 24
return SOFTMAX ( h * W_vocab )
When𝑟 ∗(0.9)≪ 𝑑 head, the attention computation con-
centrates along a small number of spectral directions. This
provides a geometric account of why eviction methods that
select tokens by attention score [1] can preserve generation
quality: residual components along low-energy singular di-
rections contribute minimally to the attention pattern.
3.4. KV-Direct: Bounded Inference via Residual
Checkpointing
TheprecedingidentitiessuggestreplacingtheKVcache
withresidualcheckpointsandon-the-flyrecomputation.We
propose KV-Direct, summarised in Algorithm 1. When a
token’s KV entry is evicted from the cache, we retain its
residualvector𝐡 (𝓁)
𝑝 (asinglevectorofdimension𝑑 hidden)and
recompute𝐊and𝐕when the token is needed for attention.
The cost is one matrix multiply plus a normalisation per
evicted token per layer.
Per-token memory.For a model with𝐿layers,𝑛 kv KV
heads, head dimension𝑑head, and𝑏bytes per element, the
standard KV cache stores
KV per token= 2𝐿 𝑛kv 𝑑head 𝑏bytes,(23)
while the residual checkpoint costs only
Residual per token=𝑑hidden ⋅𝑏bytes.(24)
A single residual vector servesall𝐿layers; downstream𝐊
and𝐕atanydepthcanberecomputedfromit.Theper-token
compression ratio is
𝜌= 2𝐿 𝑛kv 𝑑head
𝑑hidden
.(25)
For Gemma 3-4B-IT with𝐿=34,𝑛 kv=4,𝑑 head=256, and
𝑏=2(bfloat16):theKVcostis2×34×4×256×2 = 139,264
bytes≈ 136KB per token, versus2560 × 2 = 5,120bytes
= 5KB for the residual (𝜌= 27.2).
Recomputation cost.Reconstructing K and V for𝑁
evictedtokensatasinglelayerrequirestwomatrixmultipli-
cations of shape(𝑁, 𝑑hidden) × (𝑑hidden, 𝑑head)per KV head,
plus𝑁RMSNorm and𝑁RoPE operations. The dominant
cost in floating-point operations is
𝐶recomp = 4𝑁⋅𝑛 kv ⋅𝑑 hidden ⋅𝑑 head,(26)
where the factor of 4 accounts for two projections (K and
V),eachcosting2𝑁𝑑 hidden𝑑headmultiply-addoperationsper
head. By contrast, reading𝑁cached KV entries transfers
𝐵read = 2𝑁⋅𝑛 kv ⋅𝑑 head ⋅𝑏bytes (27)
over the memory bus. Whether recomputation or cache
reading is faster depends on the hardware’s compute-to-
bandwidth ratio (arithmetic intensity), which we measure
empirically in Section 5.8.
Total memory bound.For a sequence of𝑇tokens with
cache budget𝐵, KV-Direct stores𝐵recent KV entries per
layerandcheckpointstheremaining𝑇−𝐵residuals(shared
across all layers). The total memory is
(𝑇 , 𝐵) = 2𝐵𝐿𝑛kv𝑑head𝑏
+ (𝑇−𝐵)𝑑 hidden 𝑏.(28)
Unbounded caching costs𝑇⋅2𝐿𝑛 kv𝑑head𝑏, growing𝜌times
faster in𝑇. For any fixed budget𝐵, KV-Direct memory
grows at rate𝑑hidden ⋅𝑏per token regardless of model depth
or head count.
4. Experiments
To systematically investigate the redundancy hypothe-
sis, we organise the experimental evaluation around four
research questions:
RQ1 Can K and V tensors at every layer be exactly re-
constructed from residual stream vectors across different
architectures, precisions, and sequence lengths?
RQ2 Does the residual stream at any given layer constitute
asufficientstatisticforallsubsequentcomputationsintrans-
former inference?
RQ3 Can residual checkpointing with aggressive memory
budgetsmatchtheoutputfidelityofunboundedKVcaching,
and how does this compare to existing eviction strategies?
RQ4 AtwhatpointdoesrecomputingKandVfromcheck-
pointedresidualsbecomefasterthanreadingcachedtensors
from memory?
We test each component on six models spanning four
architecture families, from 135M to 4B parameters. All
experimentsrunonAppleM3Max(64GBunifiedmemory)
using the MLX framework with bfloat16 precision. Table 1
summarises the model architectures and per-token memory
costs. The theoretical compression ratio𝜌(25) ranges from
6.9×(Qwen2.5-0.5B, 2 KV heads) to56×(Qwen3-0.6B,
8 KV heads), demonstrating that the memory advantage of
residualcheckpointinggrowswiththeproduct𝐿⋅𝑛 kv ⋅𝑑head
relative to𝑑hidden.
Qasim et al.:Preprint submitted to ElsevierPage 5 of 13

<!-- page 6 -->

The Residual Stream Is All You Need
Table 1
Model architectures and per-token memory footprint under standard KV caching vs. KV-Direct (bfloat16,𝑏=2bytes). KV-Direct
stores one residual vector (𝑑hidden ⋅𝑏bytes) instead of2𝐿KV vectors, yielding𝜌= 2𝐿𝑛 kv𝑑head∕𝑑hidden compression (Eq. 25). Attn:
G=global, S=sliding window. All models use pre-norm (RMSNorm) and RoPE.▾denotes memory reduction by KV-Direct.
Per-token memory
Model Family𝐿 𝑛 kv 𝑑head 𝑑Quant Attn KV cache→KV-Direct𝜌Saving
SmolLM2-135M [45] LLaMA 30 3 64 576 Full G 22.5 KB→1.1 KB20.0×▾95%
Qwen2.5-0.5B [46] Qwen2 24 2 64 896 4-bit G 12.0 KB→1.8 KB6.9×▾85%
Qwen3-0.6B Qwen3 28 8 128 1024 Full G 112.0 KB→2.0 KB56.0×▾98%
DS-R1-Distill-1.5B [47] DeepSeek 28 2 128 1536 Full G 28.0 KB→3.0 KB9.3×▾89%
Qwen2.5-1.5B Qwen2 28 2 128 1536 4-bit G 28.0 KB→3.0 KB9.3×▾89%
Gemma 3-4B-IT [48] Gemma3 34 4 256 2560 4-bit 5G/29S 136.0 KB→5.0 KB27.2×▾96%
4.1. Baselines
For RQ1–RQ2, we compare against full recomputation
from scratch (no cache) and standard KV-cached decod-
ing. For RQ3, we benchmark KV-Direct against five preva-
lent cache eviction strategies: H2O [1], StreamingLLM [8],
SnapKV [9], TOVA [10], and window-only eviction. We
evaluate on two models (Qwen2.5-0.5B-Instruct 4-bit and
Qwen2.5-1.5B-Instruct4-bit)acrossfivecachebudgetsfrom
32 to 384 tokens out of a 512-token context, generating 50
tokens per passage over 5 diverse prompts. For RQ4, we
measurerecomputationlatencyagainstmemory-buscopyof
cached tensors across batch sizes from 1 to 500 tokens. All
experimentsusegreedy(argmax)decodingunlessotherwise
noted.
5. Results
5.1. KV Reconstruction Verification (RQ1)
We verify (13)–(14) by direct computation. During a
forward pass we capture both the cached KV entries and
theresidualstream𝐡 (𝓁) enteringeachlayer,thenreconstruct
𝐊(𝓁)
recon and𝐕 (𝓁)
recon from the residual and compare element-
wise.
Results.Table 2 reports the maximum absolute difference
across all layers for each model. Every full-attention ar-
chitecture achievesexactzero: not approximately, but bit-
identically.Thisholdsforbothfull-precisionand4-bitquan-
tisedmodels,confirmingthatquantisationdoesnotbreakthe
structural identity. For Gemma’s 29 sliding-window layers,
𝐕remains exactly zero (the value projection involves no
positionalencoding),while𝐊showsnon-zeroerrorbecause
the window-relative RoPE offset diverges from the absolute
position (Eq. 16).
Sequence-length invariance.We measure KV recon-
struction error at sequence lengths{16,32,64,128,256}
tokens on SmolLM2-135M and Gemma 3-4B-IT. Both
max|Δ𝐾|andmax|Δ𝑉|remain indistinguishable from
zero (<10 −17) at all lengths. This is expected: reconstruc-
tion is a per-token matrix multiplication at each layer, so its
correctness cannot depend on sequence length. Confirming
Table 2
KV reconstruction error across six models and four architecture
families. Max absolute difference between cached and recom-
puted K/V over all layers. Every full-attention architecture
achievesexact zero. For Gemma’s sliding-window layers, V
remains zero; K is non-zero due to window-relative RoPE.
Model𝐿Attnmax|Δ𝐾|max|Δ𝑉|
SmolLM2-135M 30 Full 0.00 0.00
Qwen2.5-0.5B 24 Full 0.00 0.00
Qwen3-0.6B 28 Full 0.00 0.00
DS-R1-1.5B 28 Full 0.00 0.00
Qwen2.5-1.5B (4-bit) 28 Full 0.00 0.00
Gemma 3-4B (Global) 5 Global 0.00 0.00
Gemma 3-4B (Sliding) 29 Sliding>00.00
it empirically rules out subtle caching artefacts at short or
long contexts.
Numerical precision invariance.We repeat the recon-
struction under four dtype regimes: native bfloat16, float32,
float16, and explicit bfloat16 cast. In all casesmax|Δ𝐾|=
max|Δ𝑉|= 0exactly,confirmingthattheresultreflectsthe
algebraic identity𝐕cached ≡RMSNorm(𝐡)𝐖 𝑣 rather than a
numerical coincidence of a particular precision.
5.2. Token-Identical Generation (RQ1)
Reconstruction from residuals alone does not rule out
subtle state-accumulation effects in the autoregressive loop.
We test this by generating 30 tokens two ways on all six
models:Method Auses standard KV-cached decoding;
Method Bfeeds the entire sequence from scratch at every
step, with no cache. Both use greedy (argmax) decoding.
Table 3 shows the result. All six models produce 30/30
token-identicaloutputunderbothmethods.MethodBis1.7–
3.8×slower due to𝑂(𝑛 2)recomputation, but produces the
sametokensfromthesamelogitvalues.Thecacheisaspeed
optimisation and nothing more.
5.3. Cross-Task Residual Patching (RQ2)
We test whether the residual stream encodes thefull
computational state, not just KV entries. Following Geiger
et al. [40] and Conmy et al. [41], we perform activation
Qasim et al.:Preprint submitted to ElsevierPage 6 of 13

<!-- page 7 -->

The Residual Stream Is All You Need
Table 3
Generation comparison: standard KV-cached decoding vs. full
recomputation from scratch. All models use greedy (argmax)
decoding and produce identical output under both methods.
Model Match Cache Recomp. Speed
(s) (s)
SmolLM2-135M 30/30 0.11 0.191.7×
Qwen2.5-0.5B 30/30 0.18 0.341.9×
Qwen3-0.6B 30/30 0.20 0.402.0×
DS-R1-1.5B 30/30 0.41 0.701.7×
Qwen2.5-1.5B (4-bit) 30/30 0.20 0.703.5×
Gemma 3-4B 30/30 0.82 3.143.8×
patching: a donor prompt (“What is the capital of Aus-
tralia?”) and a recipient prompt (“What language is spoken
inFrance?”)areeachrunthroughSmolLM2-135M.Ateach
layer𝓁∈ {0,…,29}, we replace the recipient’s residual
with the donor’s and continue the forward pass.
The result is𝐷KL
(𝑝patched ‖𝑝donor
)= 0.0at every layer:
exactly zero, not approximately. The patched model outputs
“Canberra”(thedonoranswer)regardlessofwhichlayerwe
inject at. We verified this across all 30 layers of SmolLM2-
135M andall 24layers ofQwen2.5-0.5B:𝐷 KL = 0at every
injection point, with zero exceptions. The residual stream is
a complete Markov state at every depth of the network.
Remark5.1.Thiszero-KLresultisexactbecausethesame
modelweightsprocessthecontinuation.Theresidualstream
determines all subsequent computation; there is nowhere
else for information to reside.
5.4. Downstream Task Evaluation (RQ2)
While zero KL divergence guarantees identical output
distributions, we verify this parity empirically on standard
benchmarks. Table 4 reports results on 0-shot HellaSwag
(𝑁=500)andWikiText-2perplexity,whereKV-Directisin-
dependently measured via a separate layer-by-layer forward
pass that recomputes K and V from the residual stream at
each layer (cache=None), rather than copied from the full-
cache baseline.
On HellaSwag, KV-Direct matches full-cache accuracy
exactlyonallfivestandard-attentionmodelswith100%pre-
diction agreement, confirming that distribution-level equiv-
alence translates to task-level equivalence. On WikiText-2
perplexity, KV-Direct achieves identical perplexity to full
caching across all models (e.g., 26.46 on SmolLM2-135M,
24.63 on Qwen2.5-0.5B), with zero numerical difference.
Window-only baselines degrade sharply: perplexity rises
from 38.2 at𝐵=128to 65.3 at𝐵=64and 135.4 at𝐵=32.
This confirms that KV-Direct preserves complete model
quality regardless of cache budget, whereas naive eviction
destroys it.
Sliding-windowlimitation.OnGemma-3-4B(29/34sliding-
windowlayers),thecache-freerecomputepathdegradesdra-
matically: HellaSwag accuracy drops from 49.2% to 25.0%
(nearrandomchance)andWikiText-2perplexitydivergesby
Table 4
Downstream task evaluation. HellaSwag 0-shot accuracy (%,
𝑁=500)andWikiText-2perplexity.KV-Directisindependently
measured via layer-by-layer recompute (not copied from full
cache). On all five standard-attention models, KV-Direct
matches full-cache outputs exactly.
Model Method HellaSwag PPL
SmolLM2-135M
Full cache 39.4 26.46
KV-Direct 39.4 26.46
Window-128 — 38.2
Window-64 39.4 65.3
Window-32 — 135.4
Qwen2.5-0.5B
Full cache 41.0 24.63
KV-Direct 41.0 24.63
Window-64 41.0 —
Qwen3-0.6B
Full cache 44.0 18.63
KV-Direct 44.0 18.63
Window-64 44.0 —
DS-R1-1.5B
Full cache 42.0 50.37
KV-Direct 42.0 50.37
Window-64 42.0 —
Qwen2.5-1.5B
Full cache 47.5 17.04
KV-Direct 47.5 17.04
Window-64 47.5 —
orders of magnitude. This occurs because sliding-window
layers require a rotating KV buffer to enforce window-
relative position encoding and local attention masking; a
simple cache-free recompute bypasses these constraints.
ThisresultempiricallyconfirmsthatKV-Directinitscurrent
formislimitedtostandard(global)attentionlayers,asnoted
in Section 5.9.
5.5. Memory and Multi-Turn Evaluation (RQ3)
Figure2visualisestheper-tokenmemoryratioacrossall
six models. Storing one residual vector (𝑑hidden floats) costs
1.1–5.0KB, while the corresponding KV pair (2⋅𝐿⋅𝑛kv ⋅
𝑑head floats) costs12–136KB. The ratio ranges from6.9×
(Qwen2.5-0.5B, 2 KV heads) to56×(Qwen3-0.6B, 8 KV
heads) and grows with the product𝑛kv ⋅𝑑 head relative to
𝑑hidden.
Multi-turn experiment.We ran a 20-turn conversation
benchmarkboundingthecachetoa150MBaggregatedbud-
get across different models. Figure 3 illustrates the memory
divergence. On smaller models scaling up to DS-R1-1.5B
(4.7×memoryratio),KV-Directlimitspeakmemoryexactly
to the initial bounds without sacrificing latency, yielding an
extremely consistent0.07–0.26s generation time matching
the unbounded baseline down to the millisecond.
Across the conversation benchmark, the conventional
unbounded cache steadily accrues megabytes (growing lin-
early),yetunderKV-Direct,thecachelimitsperfectlybound
memory.Residualstreamvectorsactasscalablereplacement
markersthattriggerinstantaneousrematerialisationonpass-
through when necessary.
Qasim et al.:Preprint submitted to ElsevierPage 7 of 13

<!-- page 8 -->

The Residual Stream Is All You Need
22 KB
1.1 KB
12 KB
1.8 KB
112 KB
2.0 KB
28 KB
3.0 KB
28 KB
3.0 KB
136 KB
5.0 KB
SmolLM2-135M
20x
Qwen2.5-0.5B
7x
Qwen3-0.6B
56x
DS-R1-1.5B
9x
Qwen2.5-1.5B
9x
Gemma3-4B
27x
KV cache per token Residual per token
Figure 2:Proportional square visualisation of per-token memory anatomy. The outer grey square represents the full KV cache
footprint; the inner blue square represents the residual stream checkpoint, sized proportionally by area. The visual disparity
between the two directly encodes the memory inflation ratio (shown in red above each model).
Figure 3:Multi-turn inference evaluation.(a)Memory growth over 20 conversation turns: standard KV cache grows to 103 MB
while KV-Direct stabilises at 42 MB.(b)Latency per turn: both methods track nearly identically, confirming zero inference penalty
from residual checkpointing.(c)Per-token memory across all six models: the KV cache costs7–27×more than a single residual
checkpoint.
At the 12B-parameter scale [49], the divergence is more
dramatic:thestandardcachereaches∼978MBover20turns
whilea150MBKV-Directbudgetmaintainsstable4-second
turns versus 13 seconds under unbounded caching.
5.6. Compression Baselines (RQ3)
Figure 4 presents the full performance matrix across all
seven methods (five eviction baselines, KV-Direct, and full
cache) at five cache budgets on both evaluation models.
The gap between methods is large. KV-Direct achieves
100%tokenmatchandnear-zeroKLdivergence(<10 −5)at
everybudgetonbothmodels,matchingthefull(unbounded)
KV cache exactly. All five eviction baselines, by contrast,
degradeseverelyevenatthemostgenerousbudget(𝐵=384,
75%retention):tokenmatchrangesfrom6%to28%andKL
divergencefrom7.5to14.1.Thegapisnotmarginal;itspans
orders of magnitude on KL and 70–95 percentage points
on token match. At the most aggressive budget (𝐵=32,
6% retention), baselines produce essentially random output
while KV-Direct remains lossless.
KV budget sweep.We also isolate the effect of cache
window size without any eviction baseline. Holding only
the last𝐵tokens in cache and evicting the rest without
recomputation, we measure token match on two models
acrosssixwindowsizes(𝐵∈ {8,16,32,64,128,256})with
250-token generation. At𝐵=256, Qwen2.5-0.5B recovers
88% of tokens while SmolLM2-135M recovers only 34%,
reflecting model-specific sensitivity to context truncation.
At the smallest window (𝐵=8), both models produce near-
random output (0–2% match). With residual recomputation
enabled, KV-Direct recovers 100% match at every budget,
becauseevictedtokensarerecomputedfromresidualcheck-
points rather than discarded.
5.7. Effective Rank Analysis (RQ4)
Wecompute𝐌 (ℎ) =𝐖 (ℎ)
𝑞 𝐖(ℎ)
𝑘
⊤
foreveryattentionhead
inthreemodelsandmeasureeffectiverankfromthesingular
valuespectrum.Figure5showstheresultasadual-encoded
dot matrix across all heads and layers: colour indicates the
fraction of architectural rank used at 90% spectral energy
(blue=lowrank/highlycompressible,red=nearfullrank),
Qasim et al.:Preprint submitted to ElsevierPage 8 of 13

<!-- page 9 -->

The Residual Stream Is All You Need
32
(6%)
64
(12%)
128
(25%)
256
(50%)
384
(75%)
KV-Direct (Ours)
Full KV Cache
H2O
StreamingLLM
SnapKV
TOV A
Window-Only
100 100 100 100 100
100 100 100 100 100
13 5 6 6 10
7 8 7 8 12
5 6 6 19 9
6 8 4 8 9
5 6 6 6 6
(A1)  Qwen2.5-0.5B — Token Match (%)
32
(6%)
64
(12%)
128
(25%)
256
(50%)
384
(75%)
100 100 100 100 100
100 100 100 100 100
6 24 24 27 28
22 23 24 10 8
10 22 24 23 22
6 24 24 27 28
10 10 10 7 7
(A2)  Qwen2.5-1.5B — Token Match (%)
0
25
50
75
100
Token Match (%)
32
(6%)
64
(12%)
128
(25%)
256
(50%)
384
(75%)
Cache Budget (tokens / retention %)
KV-Direct (Ours)
Full KV Cache
H2O
StreamingLLM
SnapKV
TOV A
Window-Only
~0 ~0 ~0 ~0 ~0
~0 ~0 ~0 ~0 ~0
8.2 9.0 8.5 8.9 7.9
11.4 10.7 12.2 8.3 7.6
13.7 9.8 10.2 7.2 8.0
10.8 9.3 9.5 8.7 8.1
13.7 13.0 13.3 8.4 8.7
(B1)  Qwen2.5-0.5B — KL Divergence
32
(6%)
64
(12%)
128
(25%)
256
(50%)
384
(75%)
Cache Budget (tokens / retention %)
~0 ~0 ~0 ~0 ~0
~0 ~0 ~0 ~0 ~0
12.2 9.6 9.2 8.4 8.2
9.0 9.3 8.6 7.5 8.3
12.6 9.7 9.1 8.9 8.4
12.2 9.6 9.2 8.4 8.2
12.6 13.0 14.1 9.0 9.0
(B2)  Qwen2.5-1.5B — KL Divergence
~0
1e-3
1
10
KL Divergence
Figure 4:Performance matrix across seven methods, five cache budgets, and two models.Top row:Token match percentage
(higher is better; darker blue=higher match).Bottom row:KL divergence from the full-cache output distribution (lower is
better; blue=near-zero divergence, red=high divergence). KV-Direct and full KV cache achieve 100% token match and≈0 KL
divergence at every budget, while all five eviction baselines degrade severely (5–28% match, KL 7–14). The blue-bordered row
highlights KV-Direct.
while dot size encodes the same fraction as area. Dashed
outlines mark Gemma’s five global-attention layers.
Thethreearchitecturesdifferinheaddimension(𝑑 head =
64vs. 256) but share a common pattern visible in Figure 5:
the mean effective rank at 90% energy is 27–33% of𝑑head
(meanranksof21.2,20.0,and70.2forSmolLM2,Qwen2.5-
0.5B,andGemmarespectively),yielding3.0–3.6×compres-
sionratios.Alongtailofnear-rank-1headsispresentacross
all models; on Gemma, 3 of 136 heads (2%) have effective
rank≤10, including one rank-1 head at layer 0 consistent
with the attention-sink phenomenon.
Low-rank approximation fails at generation.Despite
the low effective rank, truncating the KV projections to
rank𝑟 < 𝑑 head destroys output quality. Figure 6 plots
token match and KL divergence against projection rank for
SmolLM2-135M and Qwen2.5-0.5B. At full rank (𝑟=64),
token match is 100% with𝐷 KL = 5 × 10 −4. At𝑟=32
(50% of𝑑 head), match drops to 15% with𝐷 KL = 10.9.
Below𝑟=15, output is essentially random (5–10% match,
𝐷KL >11). The spectral energy captured by the top 32
components exceeds 95%, yet discarding the remaining 5%
of energy produces catastrophic output degradation. This
exposes a separation: the low-rank structure explainswhy
attention works (computation concentrates on a subspace),
but it cannot be exploited for lossy compression without
degrading generation. Lossless recomputation from the full
residual stream, as in KV-Direct, is the only approach that
preserves exact output fidelity.
Qasim et al.:Preprint submitted to ElsevierPage 9 of 13

<!-- page 10 -->

The Residual Stream Is All You Need
Figure 5:Effective rank of𝐌 (ℎ) =𝐖 (ℎ)
𝑞 𝐖(ℎ)
𝑘
⊤
at 90% spectral energy across three models. Each dot is one KV head at one layer.
Colour: rank as a fraction of𝑑head (blue=compressible, red=near full rank).Size: same fraction (larger=higher rank). Dashed
outlines on Gemma mark global-attention layers. Layer 0 consistently shows near-rank-1 heads across all models, consistent with
the BOS-focus phenomenon [8]. Rank heterogeneity is visible both within and across architectures.
Geometric account of eviction robustness.This low-
rank structure provides a geometric account of why token-
importance eviction methods [1, 2] preserve generation
quality at moderate budgets: attention computation concen-
trates along a small subspace of the residual stream, so
tokenswhoseprojectionslieoutsidethissubspacecontribute
minimally to the attention pattern. Rank-based compression
ismosteffectivewhenappliedselectivelytothenear-rank-1
minority rather than uniformly across all heads.
5.8. Recomputation Latency (RQ4)
A primary assumption driving unbounded KV caches
is that recomputing state is invariably slower than reading
it from memory. To test this, we benchmarked the time
requiredtoreconstruct𝑁KVvectorsfromresidualmatrices
versus copying identical cached tensors over the memory
bus.
Figure 7 reveals a surprising crossover: memory band-
width becomes the overriding bottleneck. For small batches
ofevictedtokens(𝑁=1),recomputationholdsaslightover-
head (1.1×ratio). However, as𝑁scales, dense matrix mul-
tiplicationfromresidualsfullyoutstripsmemoryfetches.At
𝑁=500,reconstructingthematricesfromresidualsoperates
in roughly0.2×to0.3×the time required to read pre-
computedcachestructuresfrommemory.Checkpointingthe
residual is not merely a memory optimization; for moder-
atelysizedtokenwindows,itacceleratesdatadeliverytothe
attention compute units.
Allfivemodelscrossbelowparityby𝑁=50.At𝑁=100,
ratiosrangefrom0.46×(Qwen2.5-1.5B)to0.68×(SmolLM2-
135M). At𝑁=500, the largest model reconstructs KV
in0.17×the time required to read cached tensors. The
crossover point scales with model size: larger models have
proportionally more compute per byte of cache, so recom-
putation amortises faster.
5.9. Discussion
Our results recast the KV cache as derived state. Serv-
ing systems [6] currently treat KV entries as irreplace-
able, engineering memory allocation, garbage collection,
and CPU/disk swapping around them. Recognising KV as
recoverable from the residual stream turns the cache into
a true cache in the computer-science sense: a performance
optimisationthatcanbeevictedandregeneratedwithoutloss
of correctness.
This shift has direct implications for edge deployment
and long conversations. On a memory-constrained device,
Qasim et al.:Preprint submitted to ElsevierPage 10 of 13

<!-- page 11 -->

The Residual Stream Is All You Need
Figure 6:Token match (%) and KL divergence vs. KV projection rank𝑟on two models. At full rank (𝑟=64), both models achieve
100% match. Truncating to𝑟=32(50% of𝑑head, capturing>95% spectral energy) causes catastrophic degradation: 5–15% match
and𝐷 KL >10. The shaded region marks ranks where lossy compression fails.
Figure 7:Recompute-to-cache-read latency ratio across five
model architectures. Each curve traces one model as the
eviction batch size𝑁increases from 1 to 500 tokens. The
teal-shaded region marks where recomputation is faster than
cache reading. All models cross below parity by𝑁=50.
the KV cache is often the binding constraint on context
length;residualcheckpointingrelaxesitbytradingcompute
formemory.Forlongconversations,KV-Directoffersathird
option beyond truncation and unbounded growth: retain all
context in a fixed memory budget, recomputing KV for
evicted tokens as needed. Combined with attention sparsity
methods [1, 13], the recomputation cost drops in proportion
to the sparsity. The approach also composes naturally with
FlashAttention [33, 34], which optimises within a single
attention call while residual checkpointing optimises across
the sequence dimension.
6. Limitations and Future Work
Our experiments cover six models from 135M to 4B
parameters. The theoretical argument applies to any pre-
norm transformer with standard attention, but we have not
verified exact-zero reconstruction on models with Layer-
Norm (instead of RMSNorm), mixture-of-experts routing,
or parameters above 4B. On Gemma 3-4B-IT, cache-free
recompute degrades HellaSwag accuracy from 49.2% to
25.0% and perplexity by orders of magnitude, confirm-
ingthatsliding-windowarchitecturesrequireposition-aware
cache management that simple recompute does not provide
(Section5.4).Themulti-turnbenchmarkuses20turns;real-
world deployment at 100K+ token contexts would face
additional recomputation latency.
The bounded inference prototype uses full recomputa-
tion (budget𝐵=0), the extreme case. A practical system
would set𝐵 >0, caching recent tokens and recomputing
only evicted entries. The optimal budget depends on the
hardware’s compute-to-memory bandwidth ratio, a system-
level optimisation we leave to future work.
Several directions warrant investigation. First, integrat-
ing residual checkpointing into production serving frame-
works (e.g., vLLM) to measure end-to-end throughput at
scale. Second, combining KV-Direct with weight quantisa-
tion and per-head mixed-precision caching guided by the
rankheterogeneityobservedinSection5.7.Third,extending
the Markov property analysis to architectures with cross-
layer KV sharing [26, 27] and latent-space attention [29],
where the residual-to-KV mapping takes different algebraic
forms.
7. Conclusion
We have shown, through theory and experiment, that
the KV cache in transformer inference is a computational
shortcut, not an information store. Keys and values at every
Qasim et al.:Preprint submitted to ElsevierPage 11 of 13

<!-- page 12 -->

The Residual Stream Is All You Need
layer are deterministic projections of the residual stream.
Removingthecacheandrecomputingfromscratchproduces
identical tokens. Replacing the residual stream wholesale at
any layer produces the donor’s output distribution with zero
KL divergence, confirming the Markov property.
The bilinear attention form reveals strong rank hetero-
geneity across heads and models. The mean effective rank
at 90% spectral energy is 27–33% of𝑑head on all three
architecturestested,withasmallsubset(<2%)ofnear-rank-
1 attention-sink heads. This structure motivates per-head
mixed-precision caching and provides a geometric account
ofwhytoken-evictionheuristicspreservegenerationquality.
These results reframe KV cache management. Instead
of designing eviction policies that try to preserve the “most
important”cachedentries,wecantreattheresidualstreamas
groundtruth andrecomputeKV entriesasneeded.Memory
becomesatunableknob:morecachemeansfasterinference,
less cache means lower memory, but correctness is guar-
anteed regardless. For memory-constrained settings (edge
devices,longconversations,high-concurrencyserving),this
guarantee changes what is possible.
Acknowledgments
The authors extend their appreciation to the National
Science Foundation of China under grants (No.:62471411).
References
[1] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y.Tian,C.Ré,C.Barrett,etal., H2O:Heavy-hitteroracleforefficient
generativeinferenceoflargelanguagemodels,in:AdvancesinNeural
InformationProcessingSystems,volume36,2023,pp.34661–34710.
[2] Z. Liu, A. Desai, F. Liao, W. Wang, V. Xie, Z. Xu, A. Kyrillidis,
A. Shrivastava, Scissorhands: Exploiting the persistence of impor-
tance hypothesis for LLM KV cache compression at test time, Ad-
vances in Neural Information Processing Systems 36 (2023) 52342–
52364.
[3] A. Devoto, Y. Zhao, S. Scardapane, P. Minervini, A simple and
effective L2 norm-based strategy for KV cache compression, arXiv
preprint arXiv:2406.11430 (2024).
[4] R. Gong, Y. Ding, Z. Wang, C. Lv, X. Zheng, J. Du, Y. Yong, S. Gu,
H. Qin, J. Guo, D. Lin, M. Magno, X. Liu, A survey of low-bit large
language models: Basics, systems, and algorithms, Neural Networks
192 (2025) 107856.
[5] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón,
S. Sanghai, GQA: Training generalized multi-query transformers
frommulti-headcheckpoints, in:Proceedingsofthe2023Conference
on Empirical Methods in Natural Language Processing, 2023.
[6] W.Kwon,Z.Li,S.Zhuang,Y.Sheng,L.Zheng,C.H.Yu,J.Gonzalez,
H.Zhang,I.Stoica, Efficientmemorymanagementforlargelanguage
model serving with PagedAttention, in: Proceedings of the 29th
Symposium on Operating Systems Principles, 2023.
[7] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, Y. Choi, HellaSwag:
Can a machine really finish your sentence?, in: Proceedings of the
57th Annual Meeting of the Association for Computational Linguis-
tics, 2019, pp. 4791–4800.
[8] G.Xiao,Y.Tian,B.Chen,S.Han,M.Lewis, Efficientstreaminglan-
guage models with attention sinks, arXiv preprint arXiv:2309.17453
(2024).
[9] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai,
P. Lewis, D. Chen, SnapKV: LLM knows what you are looking for
before generation, in: Advances in Neural Information Processing
Systems, volume 37, 2024.
[10] M. Oren, M. Hassid, Y. Adi, R. Schwartz, Transformers are multi-
state RNNs, in: Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, 2024.
[11] Z. Zhang, Z. Yang, et al., PyramidKV: Dynamic KV cache com-
pression based on pyramidal information funneling, arXiv preprint
arXiv:2406.02069 (2024).
[12] D. Yang, X. Han, Y. Gao, Y. Hu, S. Zhang, H. Zhao, PyramidInfer:
Pyramid KV cache compression for high-throughput LLM inference,
in: Findings of the Association for Computational Linguistics: ACL
2024, 2024.
[13] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, S. Han, Quest:
Query-awaresparsityforefficientlong-contextLLMinference, arXiv
preprint arXiv:2406.10774 (2024).
[14] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, J. Gao, Model tells
you what to discard: Adaptive KV cache compression for LLMs, in:
International Conference on Learning Representations, 2024.
[15] H.Tang,Y.Lin,J.Lin,Q.Han,S.Hong,Y.Yao,G.Wang, RazorAt-
tention: Efficient KV cache compression through retrieval heads, in:
International Conference on Learning Representations, 2025.
[16] F.Yuan,J.Lv,J.Zhou,etal., Ada-KV:OptimizingKVcacheeviction
by adaptive budget allocation for efficient LLM inference, arXiv
preprint arXiv:2407.11550 (2024).
[17] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen,
X. Hu, KIVI: A tuning-free asymmetric 2bit quantization for KV
cache, in: International Conference on Machine Learning, 2024.
[18] C. Hooper, S. Kim, H. Mohber, T. Wattanawong, M. W. Mahoney,
Y. S. Shao, K. Keutzer, A. Gholami, KVQuant: Towards 10 million
context length LLM inference with KV cache quantization, in:
Advances in Neural Information Processing Systems, volume 37,
2024.
[19] H. Kang, Q. Zhang, S. Kundu, G. Jeong, Z. Liu, T. Krishna, T. Zhao,
GEAR: An efficient KV cache compression recipe for near-lossless
generative inference of LLM, arXiv preprint arXiv:2403.05527
(2024).
[20] T.Zhang,J.Yi,Z.Xu,A.Shrivastava, KVcacheis1bitperchannel:
Efficient large language model inference with coupled quantization,
in: Advances in Neural Information Processing Systems, volume 37,
2024.
[21] J.-Q. Wang, X.-Q. Han, P.-J. Guo, R.-Q. He, Z.-F. Gao, Z.-Y. Lu,
Enablingefficientlow-bitquantizationbasedonmatrixproductopera-
torsforKVcachecompression, NeuralNetworks197(2025)108467.
[22] C.-C. Chang, W.-C. Lin, C.-Y. Lin, C.-Y. Chen, Y.-F. Hu, P.-H.
Wang, N.-C. Huang, L. Ceze, M. S. Abdelfattah, K.-C. Wu, Palu:
Compressing KV-cache with low-rank projection, in: International
Conference on Learning Representations, 2025.
[23] P. Singhania, S. Singh, S. He, S. Feizi, A. Bhatele, Loki: Low-rank
keysforefficientsparseattention, in:AdvancesinNeuralInformation
Processing Systems, volume 37, 2024.
[24] U.Saxena,G.Saha,S.Choudhary,K.Roy, Eigenattention:Attention
in low-rank space for KV cache compression, in: Findings of the
Association for Computational Linguistics: EMNLP 2024, 2024.
[25] N. Shazeer, Fast transformer decoding: One write-head is all you
need, arXiv preprint arXiv:1911.02150 (2019).
[26] W.Brandon,M.Mishra,A.Nrusimha,R.Panda,J.R.Kelly,Reducing
transformer key-value cache size with cross-layer attention, arXiv
preprint arXiv:2405.12981 (2024).
[27] H. Wu, K. Tu, Layer-condensed KV cache for efficient inference of
large language models, in: Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics, 2024.
[28] Y. Sun, L. Dong, Y. Zhu, S. Huang, W. Wang, S. Ma, Q. Zhang,
J.Wang,F.Wei, Youonlycacheonce:Decoder-decoderarchitectures
for language models, in: Advances in Neural Information Processing
Systems, volume 37, 2024.
[29] DeepSeek-AI, DeepSeek-V2: A strong, economical, and efficient
mixture-of-expertslanguagemodel,arXivpreprintarXiv:2405.04434
(2024).
Qasim et al.:Preprint submitted to ElsevierPage 12 of 13

<!-- page 13 -->

The Residual Stream Is All You Need
[30] A. Liu, J. Liu, et al., MiniCache: KV cache compression in depth
dimension for large language models, in: Advances in Neural Infor-
mation Processing Systems, volume 37, 2024.
[31] C.Jiang,L.Gao,H.E.Zarch,M.Annavaram, KVPR:EfficientLLM
inference with I/O-aware KV cache partial recomputation, arXiv
preprint arXiv:2411.17089 (2025).
[32] S. Lee, H. Kim, S. Hwang, G. Heo, M. Noh, J. Huh, Efficient LLM
inference with activation checkpointing and hybrid caching, arXiv
preprint arXiv:2501.01792 (2025).
[33] T. Dao, D. Fu, S. Ermon, A. Rudra, C. Ré, FlashAttention: Fast and
memory-efficient exact attention with IO-awareness 35 (2022).
[34] T.Dao, FlashAttention-2:Fasterattentionwithbetterparallelismand
work partitioning, arXiv preprint arXiv:2307.08691 (2023).
[35] T. Chen, B. Xu, C. Zhang, C. Guestrin, Training deep nets with
sublinear memory cost, arXiv preprint arXiv:1604.06174 (2016).
[36] N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann,
A. Askell, Y. Bai, A. Chen, T. Conerly, et al., A mathematical
framework for transformer circuits, Transformer Circuits Thread
(2021).
[37] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma,
T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, et al., In-context
learning and induction heads, Transformer Circuits Thread (2022).
[38] A. Shai, S. Marzen, L. Teixeira, A. G. Oldenziel, P. M. Riechers,
Transformers represent belief state geometry in their residual stream,
in: Advances in Neural Information Processing Systems, volume 37,
2024.
[39] S. He, G. Sun, Z. Shen, A. Li, What matters in transformers? not all
attention is needed, arXiv preprint arXiv:2406.15786 (2024).
[40] A. Geiger, H. Lu, T. Icard, C. Potts, Causal abstractions of neural
networks, Advances in Neural Information Processing Systems 34
(2021) 9574–9586.
[41] A. Conmy, A. N. Mavor-Parker, A. Lynch, S. Heimersheim,
A. Garriga-Alonso, Towards automated circuit discovery for mech-
anistic interpretability, Advances in Neural Information Processing
Systems 36 (2023) 16318–16352.
[42] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in:
Advances in Neural Information Processing Systems, volume 30,
2017.
[43] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image
recognition (2016) 770–778.
[44] J.Su,M.Ahmed,Y.Lu,S.Pan,W.Bo,Y.Liu, RoFormer:Enhanced
transformer with rotary position embedding, Neurocomputing 568
(2024) 127063.
[45] L.B.Allal,A.Lozhkov,G.Penedo,T.Wolf,L.vonWerra,SmolLM2:
Whensmolgoesbig–data-centrictrainingofasmalllanguagemodel,
arXiv preprint arXiv:2502.02737 (2025).
[46] Q.Team, Qwen2.5technicalreport, arXivpreprintarXiv:2412.15115
(2025).
[47] D.Guo,D.Yang,H.Zhang,J.Song,R.Zhang,R.Xu,Q.Zhu,S.Ma,
P.Wang,X.Bi,etal.,Deepseek-r1:Incentivizingreasoningcapability
in llms via reinforcement learning, arXiv preprint arXiv:2501.12948
(2025).
[48] Gemma Team, Gemma 3 technical report, arXiv preprint
arXiv:2503.19786 (2025).
[49] C. Hay, We don’t need KV cache anymore? KV-Direct: Bounded-
memory inference via residual checkpointing,https://github.com/
chrishayuk/chuk-lazarus, 2025.
A. Experimental Details
Hardware.AllexperimentswererunonanAppleM3Max
with 64 GB unified memory. We use the MLX framework
for model loading and inference, with bfloat16 precision
throughout.
Models.
•SmolLM2-135M-Instruct[45]: LLaMA-family, 30
layers,𝑑 hidden = 576, 9 query heads, 3 KV heads,
𝑑head = 64.
•Qwen2.5-0.5B-Instruct[46]:Qwen2,24layers,𝑑 hidden =
896, 14 query heads, 2 KV heads,𝑑head = 64. 4-bit
quantised.
•Qwen3-0.6B-Base: Qwen3, 28 layers,𝑑hidden = 1024,
16 query heads, 8 KV heads,𝑑head = 128. Full preci-
sion.
•DeepSeek-R1-Distill-Qwen-1.5B[47]: Qwen2 archi-
tecture (reasoning-distilled), 28 layers,𝑑hidden = 1536,
12 query heads, 2 KV heads,𝑑head = 128.
•Qwen2.5-1.5B-Instruct: Qwen2, 28 layers,𝑑 hidden =
1536, 12 query heads, 2 KV heads,𝑑head = 128. 4-bit
quantised.
•Gemma 3-4B-IT[48]: Gemma3, 34 layers,𝑑 hidden =
2560, 8 query heads, 4 KV heads,𝑑head = 256. 29/34
sliding window; 5 global. 4-bit quantised.
All models use pre-norm (RMSNorm) and RoPE. Exper-
iments run on Apple M3 Max via MLX with bfloat16
precision.
Prompts.For Experiment 1 (KV reconstruction): “The
residual stream in a transformer is the central information
highway.AllattentionandMLPoutputsareadditiveupdates
to it.” (24 tokens after tokenization.)
For Experiment 2 (generation match): “Explain why the
skyisblueinsimpleterms.”Greedy(argmax)decodingwith
no temperature or sampling.
ForExperiment3(multi-turn):Systemprompt“Youare
ahelpful,conciseAIassistant.”followedbyuserturnsabout
France, the Eiffel Tower, etc. 30 tokens generated per turn.
Rankcomputation.Thebilinearform𝐌 (ℎ) =𝐖 (ℎ)
𝑞 𝐖(ℎ)
𝑘
⊤
was computed in float32 to avoid precision loss. Singular
values were obtained vianumpy.linalg.svd. Effective rank
was defined as the smallest𝑟such that ∑𝑟
𝑖=1 𝜎2
𝑖 ≥0.90⋅
∑𝑑head
𝑖=1 𝜎2
𝑖.
B. Additional Analysis
This section presents two supplementary analyses that
complement the main results. Figure 9 examines how to-
ken match degrades under window-only caching as the KV
budgetvaries,confirmingthatKV-Directmaintainslossless
reconstruction at all budget levels. Figure 8 visualises the
cross-taskresidualpatchingexperimentacrossalllayersand
models,providinglayer-granularityevidencefortheMarkov
property established in Section 3.1.
Qasim et al.:Preprint submitted to ElsevierPage 13 of 13

<!-- page 14 -->

The Residual Stream Is All You Need
Figure 8:Cross-task residual patching across all layers for four models. Each block represents one layer where the recipient’s
residual stream is replaced with the donor’s. All tested layers produce𝐷KL = 0.0across all four architectures, confirming that the
residual stream is a sufficient Markov state at every depth.
Figure 9:KV budget sweep across six window sizes
(𝐵∈ {8,16,32,64,128,256}) with 250-token generation. (a)
Window-only token match degrades sharply as the cache
budget shrinks, while KV-Direct maintains 100% match at all
budgets. (b) Averaged across budgets, window-only caching
achieves 16–23% match versus KV-Direct’s perfect recovery.
Qasim et al.:Preprint submitted to ElsevierPage 14 of 13
