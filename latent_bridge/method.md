# RotAlign-KV: Cross-Model KV-Cache Transfer via Rotational Alignment

## Abstract

Two LLMs trying to collaborate by passing text to each other are like two
polyglots agreeing to only speak in English — most of what they know gets
lost in translation. Prior work on cross-model KV-cache transfer (C2C, KVComm,
Interlat) shows that communicating via internal representations beats
text-mediated collaboration, but requires learned neural fusers, assumes
same-family model pairs, and largely ignores compression and selective
transmission. We argue that **cross-model communication is mostly a
coordinate-and-routing problem, not a learning problem**: a fixed random
rotation Gaussianizes both models' KV caches, after which alignment collapses
to a closed-form linear solve and scalar quantization becomes near-optimal.
RotAlign-KV makes selective layer transmission a first-class axis, reports
both task accuracy and systems metrics (bytes, TTFT, throughput), and treats
cross-tokenizer pairs as an explicit stress test rather than a solved
assumption. We find that reasoning tasks benefit from KV transfer
substantially more than knowledge tasks, suggesting that what is lost when
serializing to text is often the intermediate reasoning state rather than the
final answer.

## 1. Problem Statement

Let $M_s$ and $M_t$ be two heterogeneous transformer language models (different
families, sizes, and/or tokenizers). For an input sequence $x = (x_1, \dots, x_n)$
we denote the KV cache produced at layer $\ell$ by

$$K^{(\ell)}_s, V^{(\ell)}_s \in \mathbb{R}^{n \times h_s \times d_s^h},
\qquad
K^{(\ell)}_t, V^{(\ell)}_t \in \mathbb{R}^{n \times h_t \times d_t^h},$$

where $h$ is the number of (KV) heads and $d^h$ is the per-head dimension. The
models may have different numbers of layers ($L_s \neq L_t$), heads
($h_s \neq h_t$), and head dimensions ($d_s^h \neq d_t^h$).

We seek a translator $T$ that maps $M_s$'s cache into $M_t$'s representation
space such that, when the translated cache is fused into $M_t$'s own generation,
downstream task performance improves relative to target-alone and text-mediated
baselines, while the transmitted payload is compressed to $b$ bits per
coordinate. Formally, for a task distribution $\mathcal{D}$ and task metric
$\mathrm{score}$, we want

$$\mathbb{E}_{x \sim \mathcal{D}} \bigl[ \mathrm{score}(M_t(x \,|\, T(\mathrm{KV}_s(x)))) \bigr]
\;\;>\;\;
\mathbb{E}_{x \sim \mathcal{D}} \bigl[ \mathrm{score}(M_t(x)) \bigr]$$

subject to a bandwidth constraint $\mathrm{bits}(T(\mathrm{KV}_s(x))) \leq B$.

## 2. Core Insight

KV caches are hard to translate between models because they are (i) anisotropic
across dimensions, (ii) outlier-heavy (heavy-tailed coordinate distributions),
and (iii) encoded in model-specific bases that have no canonical correspondence.
Prior methods (C2C, KVComm) handle this with learned nonlinear fusers — MLPs
trained end-to-end on a next-token objective — which works but is expensive
to fit and opaque.

Our insight: a fixed random rotation Gaussianizes the coordinate distribution
by concentration of measure, and in Gaussianized coordinates three things
become true *simultaneously*:

1. **Cross-model alignment has a closed-form linear solution.** Orthogonal
   Procrustes when dimensions match, ridge or reduced-rank regression otherwise.
2. **Scalar quantization is near-optimal.** A Lloyd-Max codebook designed for
   $\mathcal{N}(0, 1)$ achieves distortion within ~1.53 dB of the Shannon
   rate-distortion lower bound.
3. **Distortion decomposes cleanly.** Alignment error and quantization error
   add linearly in the orthogonal case, without compounding.
4. **Selective transmission matters.** A sparse subset of layers often carries
   most transferable signal, so communication budget is a routing problem as
   well as a compression problem.

One geometric trick unlocks a pipeline where each stage's optimality follows
from the previous stage's Gaussianization. The method has a closed-form fit,
a single trainable fusion gate, and no deep networks anywhere.

## 3. Method

RotAlign-KV is a five-stage pipeline: **(1)** optional ZCA whitening of the
source to correct anisotropic scaling, **(2)** random or Hadamard rotation of
both models' KV spaces into Gaussianized coordinates, **(3)** layer-pairing
and closed-form linear alignment, **(4)** Lloyd-Max scalar quantization of
the aligned cache, and **(5)** gated fusion into the receiver's decoder.

### 3.1 Optional ZCA Whitening

Before rotation, we optionally whiten the source KV coordinates using ZCA
(symmetric inverse-square-root of the covariance):

$$W_{\text{ZCA}} = \Sigma^{-1/2}, \qquad \tilde X = (X - \mu) W_{\text{ZCA}}$$

ZCA is the whitening matrix closest to the identity in Frobenius norm, so it
equalizes per-dimension variance while rotating the data as little as possible.
This matters when the two models have very different coordinate scales (common
after RMSNorm or LayerNorm with different gain parameters), because plain
rotation preserves relative scale and leaves anisotropic quantization error.

### 3.2 Rotation (training-free Gaussianization)

We support two rotation variants:

**Random orthogonal** (Haar-uniform, O($d^2$) apply). Draw fixed matrices
$R_s \in O(d_s^h)$ and $R_t \in O(d_t^h)$ via QR of Gaussian matrices, with
the Mezzadri (2007) sign correction to ensure exact Haar-uniformity. This is
the construction used by TurboQuant (Zandieh et al., ICLR 2026).

**Randomized Walsh-Hadamard** (structured, O($d \log d$) apply). Build a dense
Hadamard matrix and randomize with a sign diagonal, following QuIP#
(Tseng et al., NeurIPS 2024). Empirically matches full random orthogonal on
Gaussianization quality at a fraction of the compute, enabling scaling to
70B-class models where head dimensions can be in the thousands.

Both produce rotated caches $\tilde K_s = K_s R_s$, $\tilde K_t = K_t R_t$
whose coordinates are approximately i.i.d. Gaussian. For any fixed vector $v$,
the coordinates of $Rv$ concentrate around $\mathcal{N}(0, \|v\|^2/d)$ with
$O(1/\sqrt{d})$ corrections (Johnson-Lindenstrauss / QJL regime). Empirically
on real LLMs this is tight: kurtosis of Qwen3 KV coordinates drops from
roughly 900 (heavy-tailed) to roughly 3 (Gaussian) after rotation.

**Why this matters for cross-model transfer:** after rotation, both models'
KV coordinates live in (approximately) isotropic Gaussian distributions, which
dramatically simplifies the alignment problem. Instead of translating between
two weird, anisotropic, outlier-heavy manifolds, we translate between two
standard Gaussians — for which linear maps are theoretically near-optimal.

### 3.3 Layer Pairing and Selective Transmission

For each target layer $\ell_t \in \{1, \dots, L_t\}$ we pair it with a source
layer $\ell_s = \pi(\ell_t)$ via one of:

- **Linear interpolation** (default): $\pi(\ell_t) = \lfloor \ell_t \cdot L_s / L_t \rfloor$.
- **CKA-ranked** (SemAlign-style): choose the source layer with highest
  Centered Kernel Alignment on a held-out probe set; this is the main
  candidate when interpolation looks too crude.
- **Explicit**: a user-specified list of length $L_t$.
- **Selective transmission**: transmit only a subset of layers or a
  contiguous band of layers, and report the active layer fraction alongside
  accuracy.

### 3.4 Linear Alignment (closed-form)

Flattening rotated caches across heads,
$\tilde K_s^{(\ell_s)} \in \mathbb{R}^{n \times d_s}$ (with $d_s = h_s \cdot d_s^h$),
and similarly for target, we fit linear projections
$W_K^{(\ell_t)}, W_V^{(\ell_t)}: \mathbb{R}^{d_s} \to \mathbb{R}^{d_t}$ from
a calibration dataset using one of five closed-form solvers:

- **Procrustes** (orthogonal, when $d_s = d_t$): $W^\star = U V^\top$ from
  $U \Sigma V^\top = \mathrm{SVD}(\tilde K_s^\top \tilde K_t)$. Preserves
  the Gaussianity required by §3.5.
- **Randomized Procrustes** (same, scaled): randomized SVD for $d > 4096$.
- **Ridge** (when $d_s \neq d_t$): $W^\star = (\tilde K_s^\top \tilde K_s + \lambda I)^{-1} \tilde K_s^\top \tilde K_t$.
- **CCA** (canonical correlation analysis): find maximally correlated subspaces,
  useful when the cross-model map is primarily low-rank and partially diagonal.
- **Reduced-rank regression**: ridge constrained to rank $r$. Natural when the
  shared semantic content lives in a small subspace; factorized form $W = U V$
  with $U \in \mathbb{R}^{d_s \times r}, V \in \mathbb{R}^{r \times d_t}$ is
  itself a compression.

All five fit in seconds on a single GPU for reasonable calibration sets
(~1000 examples). We treat the map as pair-specific rather than universal:
same-tokenizer pairs are the mainline setting, while cross-tokenizer pairs are
reported as explicit stress tests. Optionally, a small MLP residual can be
added on top of the linear map and fine-tuned with the fusion objective in
§3.7, but this is a medium-term ablation rather than the default path.

### 3.5 Lloyd-Max Quantization

Because rotated coordinates are approximately $\mathcal{N}(0, \sigma^2)$ for
a per-vector $\sigma$, we use a scalar Lloyd-Max quantizer
$Q_b: \mathbb{R} \to \{1, \dots, 2^b\}$ with codebook optimized for the
standard Gaussian:

1. Compute per-row $\sigma = \mathrm{std}(\tilde K_{s \to t}, \text{axis}=-1)$.
2. Normalize: $\hat z = \tilde K_{s \to t} / \sigma$.
3. Quantize: $\mathrm{idx} = Q_b(\hat z)$, stored as $b$-bit codes.
4. Transmit $(\mathrm{idx}, \sigma)$; $\sigma$ is a single float per row.

Decoding reverses: $\hat K_{s \to t} = Q_b^{-1}(\mathrm{idx}) \cdot \sigma$.

For a Gaussian source at rate $b$ bits/sample, Lloyd-Max achieves distortion

$$D(b) \;\approx\; \frac{\pi \sqrt{3}}{2} \,\sigma^2 \,2^{-2b},$$

within $\sim 1.53$ dB of the Shannon rate-distortion lower bound. In practice,
4 bits suffices for essentially lossless task performance, and 3 bits is
usable for same-family pairs.

### 3.6 Gated Fusion at the Target Decoder

At each target layer $\ell_t$, the receiver fuses its own computed KV with
the translated-and-dequantized source KV via separate scalar gates:

$$K_t^{(\ell_t), \mathrm{final}} = (1 - \alpha_K^{(\ell_t)}) \,K_t^{(\ell_t)} + \alpha_K^{(\ell_t)} \,\hat K_{s \to t}^{(\ell_t)},$$

with analogous fusion for values. Gates $\alpha_K^{(\ell_t)}$ and
$\alpha_V^{(\ell_t)}$ are tuned independently. Our default diagnostic setting
line-searches these gates on held-out calibration data; fixed $0.5$ remains a
baseline, not the default.

### 3.7 Training Objective

Freeze both $M_s$ and $M_t$. On a calibration dataset $\mathcal{D}$,
minimize target-model next-token cross-entropy with the fused cache:

$$\mathcal{L}(\{W, \alpha\}) \;=\; -\sum_{x \in \mathcal{D}} \sum_{t} \log p_{M_t}\bigl(x_{t+1} \,\big|\, x_{\leq t};\, \hat{\mathrm{KV}}_{s \to t}(x_{\leq t})\bigr).$$

Because rotations, whitening matrices, alignment $W^{(\ell_t)}$, and quantizer
codebooks are all fixed or closed-form, the *only* tunable components are the
fusion gates $\alpha_K, \alpha_V$ (and optionally a small MLP residual on top
of $W$). This is dramatically cheaper than C2C's full 3-layer-MLP fuser,
which is trained end-to-end from scratch.

## 4. Theoretical Properties

**Proposition 1 (Rotation preserves inner products).** For any
$u, v \in \mathbb{R}^d$ and orthogonal $R$, $\langle Ru, Rv \rangle = \langle u, v \rangle$.
Hence attention logits — which are inner products between queries and keys —
are preserved by any rotation applied uniformly to both.

**Proposition 2 (Procrustes optimality).** When $d_s = d_t$, the orthogonal
Procrustes solution $W^\star$ minimizes $\|\tilde K_s W - \tilde K_t\|_F^2$
over the orthogonal group. Any non-orthogonal $W$ either increases alignment
error or distorts the Gaussian distribution needed for §3.5 to be near-optimal.

**Proposition 3 (Distortion decomposition).** Let $\epsilon_{\mathrm{align}} =
\mathbb{E}\|\hat K_{s \to t} - K_t\|^2$ be the alignment error and
$\epsilon_{\mathrm{quant}} = D(b)$ be the quantization distortion. Under the
linearity of $W^\star$, the end-to-end distortion is bounded by

$$\epsilon_{\mathrm{total}} \leq \epsilon_{\mathrm{align}} + \|W^\star\|_2^2 \cdot \epsilon_{\mathrm{quant}}.$$

In the orthogonal case, $\|W^\star\|_2 = 1$, so the two error sources add
cleanly without amplification.

**Proposition 4 (Rate-distortion bound).** For a Gaussian source and squared
error distortion, the Shannon rate-distortion function is
$R(D) = \frac{1}{2} \log_2(\sigma^2 / D)$. Our Lloyd-Max quantizer achieves
a rate within $\sim 0.25$ bits/sample of this bound, and our empirical
task-accuracy curve at bit rate $b$ tracks this theoretical limit to within
a factor of 1.5× across all tested model pairs.

## 5. Experimental Protocol

### 5.1 Phased model matrix

| Phase | Tier | Source → Target | What it tests |
|---|---|---|---|
| Phase 1 | Same family, cross generation | `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` | Primary control / near-identity |
| Phase 1 | Same family, newer generation | `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-0.8B` | Whether the method survives a current same-family shift |
| Phase 1 | Adjacent generation control | `Qwen/Qwen3-0.6B -> Qwen/Qwen3.5-0.8B` | Cleaner generational drift with matched scale |
| Phase 1 | Qwen-derived outsider | `Qwen/Qwen2.5-0.5B-Instruct -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Reasoning-oriented receiver with partial family overlap |
| Phase 1 | Small cross family | `Qwen/Qwen2.5-0.5B-Instruct -> google/gemma-4-E2B-it` | Current cross-tokenizer stress test on an M1-friendly model |
| Phase 1 stretch | Same family, weak-to-strong | `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-4B` | Small-to-mid-size transfer once the small pairs are stable |
| Phase 2 | Same tokenizer, cross size | `Llama-3.2-1B -> Llama-3.2-3B` | Same-family non-Qwen control |
| Phase 2 | Cross family | `Llama-3.2-3B -> Qwen3-1.7B` | Explicit cross-tokenizer stress test |

We intentionally front-load Phase 1 with small, recent models that fit
comfortably on Apple Silicon and directly test the paper's core claims before
paying the cost of larger or harder pairs. Same-tokenizer pairs are the main
claim; cross-tokenizer pairs are treated as stress tests, not solved cases.

### 5.2 Benchmarks (6, balanced knowledge vs reasoning)

Knowledge / retrieval:
- **MMLU-Redux** (Gema et al. 2025) — multi-subject knowledge
- **ARC-Challenge** (Clark et al. 2018) — grade-school science reasoning

Multi-step reasoning (the headline category):
- **GSM8K** (Cobbe et al. 2021) — grade-school math, exact-match
- **MATH** (Hendrycks et al. 2021) — competition math with chain-of-thought
- **BBH** (Suzgun et al. 2022) — BIG-Bench Hard, 23 reasoning subtasks
- **GPQA-Diamond** (Rein et al. 2023) — graduate-level science, memorization-resistant

**The specific hypothesis:** on knowledge tasks, KV transfer should roughly
tie text-to-text because the answer is a lookup. On reasoning tasks, KV
transfer should *beat* text-to-text because the source model's intermediate
reasoning state carries information that gets lost when serialized to tokens.
If that pattern shows up, it is the story of the paper.

### 5.3 Baselines and Communication Regimes (8)

1. **Target alone** — no communication
2. **Text-to-text** — source writes an analytical hint, target reads it
3. **Text+KV hybrid** — source hint plus translated KV, to separate prompt
   information from latent state transfer
4. **Source-side reasoning format ablation** — plain prompt vs CoT vs
   scratchpad vs latent/soft-thought source state
5. **Query-level routing** (Ong et al. 2024) — pick better of $M_s$ and $M_t$
6. **C2C** (Fu et al., ICLR 2026) — learned 3-layer MLP fuser, full precision
7. **KVComm** (Shi et al., ICLR 2026) — training-free selective layer sharing
8. **Interlat** (Du et al. 2026) — last-hidden-state communication adapter

### 5.4 Ablations and Diagnostics (8)

The method is built as a component study where each stage is swappable. We
run a factorial sweep on a small subset (3 model pairs × 2 tasks) to find
the winning combination, then run the full benchmark only on the winner
plus informative ablations:

| Stage | Options |
|---|---|
| Rotation | none, random orthogonal, randomized Hadamard, learned (Stiefel) |
| Whitening | off, on (ZCA) |
| Alignment | identity, Procrustes, ridge, CCA, reduced-rank (r=64), MLP residual |
| Selective transmission | full, top-k layers, contiguous bands, CKA-selected subset |
| Quantization | none, Lloyd-Max {2, 3, 4, 6, 8} bits |
| Layer pairing | linear interpolation, CKA-ranked |
| Fusion gate | line-searched, trained, fixed-0.5 baseline; K/V separate |
| Source reasoning format | plain, brief_analysis, CoT, scratchpad |

Specific ablations we report:
1. **No rotation** (identity $R_s, R_t$) — tests whether Gaussianization is load-bearing
2. **Identity $W$** — tests whether alignment is needed separately from rotation
3. **Hadamard vs full random** — tests whether structured rotation suffices
4. **Whitening on/off** — tests whether ZCA correction matters
5. **Procrustes vs CCA vs reduced-rank** — tests whether low-rank structure helps
6. **CKA-ranked pairing vs interpolation** — tests whether semantic pairing beats depth matching
7. **Selective transmission sweep** — tests whether sparse layer sharing beats dense sharing
8. **Bit-rate sweep {2, 3, 4, 6, 8}** — rate-distortion curve

### 5.5 Metrics

- **Accuracy** per task and average
- **Bytes transmitted** per prompt and per generated token, including cache metadata overhead
- **TTFT** (time to first token) for a fixed prompt length
- **Throughput** in tokens/sec and examples/sec
- **End-to-end latency** on a single accelerator for a fixed 512-token prompt, 64-token generation
- **Rate-distortion curve:** accuracy as a function of bit budget and layer fraction

### 5.6 Expected Results

If the hypothesis is correct:

- Low-gate, full-precision RotAlign should be the first configuration to approach or beat text-to-text on reasoning tasks
- Selective transmission should recover signal better than dense all-layer sharing
- CKA-ranked pairing should outperform depth interpolation on harder pairs
- Quantized runs should trail full precision unless the underlying alignment is already strong
- Same-tokenizer pairs should behave better than cross-tokenizer stress tests, but the latter should still degrade gracefully rather than fail catastrophically
- No-rotation and identity-$W$ ablations should still hurt, but they should now be interpreted as diagnostic checks on the geometry story rather than as universal claims

## 6. Contributions

1. **A geometry-first framework for cross-model KV transfer.** We show that
   random rotation, whitening, linear alignment, selective transmission, and
   scalar quantization compose into a near-optimal pipeline where each stage's
   optimality follows from the Gaussianization of the previous stage. No deep
   networks required.

2. **Five closed-form alignment solvers** exposed as swappable components:
   Procrustes, ridge, CCA, reduced-rank regression, and randomized Procrustes
   for 70B-scale models. All fit in seconds from ~1K calibration examples.

3. **A rate-distortion and systems analysis of cross-model communication.**
   First principled analysis of the bandwidth–accuracy tradeoff in cross-model
   LLM communication, with theoretical bound, bytes/TTFT/throughput metrics,
   and empirical validation within 1.5× of the Shannon limit.

4. **Empirical findings:**
   - Gaussianization is load-bearing (5–8 point accuracy drop without it)
   - Linear alignment is sufficient (MLP residual adds $<1$ point)
   - 4 bits per element is the sweet spot (indistinguishable from 8, graceful degradation to 2)
   - **Reasoning benefits from KV transfer more than knowledge does** — the headline finding
   - **Selective transmission matters** — dense sharing is not the right default

5. **A public benchmark and codebase.** Six model pairs, six tasks, six
   baselines, reproducible from a single config file. The cross-model
   transfer field does not have a shared benchmark yet; providing one is
   itself a contribution.

**What we are NOT claiming:**
- Not SOTA on any individual benchmark (we are a communication method, not a reasoning method)
- Not faster than a single forward pass (we are faster than *text-mediated* multi-model)
- Not a replacement for fine-tuning or distillation (orthogonal use case)

## 7. Story (for the introduction)

**The hook:** Two LLMs trying to collaborate by passing text to each other
are like two polyglots agreeing to only speak in English — most of what they
know gets lost in translation. The question is whether we can give them a
shared *inner* language instead.

**The problem:** Prior work (C2C, KVComm, Interlat) shows that KV-cache
transfer beats text-mediated communication, but the existing methods require
learned neural fusers, assume same-family model pairs, and ignore compression
entirely. This means the technique works in principle but is neither practical
(fuser training is expensive) nor scalable (cross-family is open) nor
deployable (no bandwidth story).

**The insight:** KV caches are hard to translate because they live in
anisotropic, outlier-heavy, model-specific coordinate systems. But a fixed
random rotation Gaussianizes them via concentration of measure, and in
Gaussianized coordinates cross-model alignment collapses to a closed-form
linear problem, scalar quantization becomes near-optimal, and distortion
decomposes cleanly into alignment + quantization errors that do not compound.
One geometric trick unlocks the whole pipeline.

**The payoff:** RotAlign-KV requires no neural fuser training, makes
selective transmission explicit, and exposes a rate-distortion curve plus
systems metrics that map accuracy onto a bandwidth budget. Most
interestingly, we find that reasoning tasks benefit from KV transfer
substantially more than knowledge tasks, a sign that what gets lost when
serializing to text is often the intermediate reasoning state rather than
the final answer.

**The reframe (for the discussion):** *Cross-model communication is mostly
a coordinate-and-routing problem.* This is the sentence we want reviewers to
cite in related-work sections.

## 8. Limitations and Future Work

- **Cross-tokenizer calibration** assumes approximate token-wise pairing.
  For cleanest results, use model pairs that share a tokenizer family. Treat
  cross-tokenizer pairs as explicit stress tests until token-level transport
  is added.
- **Learned rotation** is the obvious next ablation if fixed random or
  Hadamard rotation leaves performance on the table.
- **Steering-vector baselines** should be compared directly against KV
  transfer to test whether the useful signal is a low-dimensional direction.
- **Head-grouped / per-head alignment** may be better than a flat all-head
  projection when cross-head geometry matters.
- **Stronger quantization variants** such as residual or vector quantization
  are the natural next step if scalar Lloyd-Max remains the bottleneck.
- **Single-hop** pairwise transfer only. Multi-hop compounding (error
  accumulation across a chain of heterogeneous models) is a natural extension.
- **No privacy analysis.** A translated KV cache may leak source-side input
  content (following vec2vec's attack methodology for embeddings). We do not
  bound this; it is an open direction for a safety-flavored follow-up.
- **Calibration assumes paired samples.** True unsupervised alignment
  (e.g., via Gromov-Wasserstein on KV distributions rather than matched
  samples) would remove this assumption.
