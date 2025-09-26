# LatentWire: A Shared Soft‑Token Interlingua for Heterogeneous LLM Cooperation

---

## Update (2025-09-18): Token‑level encoder + learned query pooling (STQueryEncoder)

### Why this change? (one paragraph, plain-English)

Our earlier **SimpleEncoder** collapsed the whole source text into a single sentence embedding before “spreading” it across the M latent slots. That’s fast but loses **positional structure** from the source. The new **STQueryEncoder** preserves order by first computing **token‑level features** with a small, **frozen** HF encoder (default: `microsoft/MiniLM‑L6‑v2`) and then using **learned queries** to **cross‑attend** over those tokens to build the `M×d_z` latent. Inside Llama/Qwen, **RoPE** continues to assign positions to the latent prefix, so the LLMs still see an ordered prefix of length `M`.

### What the pipeline looks like now

```mermaid
flowchart LR
    A[Text: Question + Context] --> B[STQueryEncoder\n(HF token encoder, frozen)]
    B --> C[Cross‑Attention Pooler\nLearned queries + slot sinusoid]
    C -->|Z \in R^{M×d_z}| D1[Adapter_L\nLayerNorm→Linear→tanh]
    C -->|Z \in R^{M×d_z}| D2[Adapter_Q\nLayerNorm→Linear→tanh]
    D1 --> E1[Llama (frozen)\ninputs_embeds: [P_L, E_L(answer[:-1])]]
    D2 --> E2[Qwen (frozen)\ninputs_embeds: [P_Q, E_Q(answer[:-1])]]
    E1 --> F1[Answer 1]
    E2 --> F2[Answer 2]
    F1 --> G[Joint rescoring\nsum logp under both models]
    F2 --> G
    G --> H[Final answer]
```

**Yes—we are still:** `text → encoder (Z) → model‑specific adapter → LLM`. We’ve **replaced the encoder** with one that keeps token‑level structure and added small systems fixes (anchor normalization, dataloader speedups, optimizer‑state device alignment).

### Why this should outperform the previous encoder at the same M (4× compression target)

1. **Token‑level evidence is preserved**: learned queries attend over **all tokens** (not a single pooled vector), so `Z` retains fine‑grained cues for extraction.
2. **Better early‑token acceptance**: slot sinusoid + LayerNorm/tanh adapters keep statistics close to real token embeddings; plus stronger first‑token loss (`λ_first`) aligns the first decode step.
3. **Frozen, off‑the‑shelf backbone**: MiniLM‑L6 is tiny, fast, and general—no task‑specific tuning required.
4. **Interlingua story intact**: LLMs remain **frozen**; only the small encoder/adapters learn—so we can honestly claim a shared wire, not per-model fine-tuning.

## Update (2025-09-25): Single-model warm-up scaffolding

We now have a focused path for iterating on Llama in isolation. The trainer respects `--models` so we can skip Qwen entirely (smaller footprint, faster spin-up), Stage B keeps the first warm-up epochs purely in text teacher-forcing mode, and `scripts/run_llama_single.sh` wires those pieces together into a reproducible Stage A→B→C loop. During warm-up we match the first few gold answer embeddings (default 4 tokens) against the adapter output and include a teacher-forced CE term. The adapters are residual two-layer MLPs with dropout, we reserve a 16-vector private latent block, and we keep tail text batches (50% probability) after the three-epoch warm-up so the encoder/adapter stack continues to absorb the teacher signal before we fold Qwen back in.

### Controlled experiment we will run (baked into `run_hero_stq.sh`)

- **Models:** Llama‑3.1‑8B‑Instruct, Qwen‑2.5‑7B‑Instruct (NF4 for memory).
- **Encoder:** `encoder_type=stq`, MiniLM‑L6‑v2 (frozen), `max_enc_tokens=1024`.
- **Latent budget:** `M=32` (`≈4×` compression on SQuAD prompts), `d_z=256`, split `shared=24`, `private=4`.
- **Training:** 12 epochs, `B=2`, `grad_accum=32`, `λ_first=3.0`, `K=8`, `τ=1.25`.
- **Eval each epoch:** Text vs Latent vs Token‑budget‑32; EM/F1, NLL/token, FirstTok@k, wall‑clock.

### Minimal math (unchanged objective)

- `Z = Attn(Q=learned_queries, K=Proj(HF_tokens), V=Proj(HF_tokens))`
- `P_ℓ = A_ℓ(Z)`, `inputs_embeds = [P_ℓ, E_ℓ(y[:-1])]`, labels `[mask… , y[1:]]`
- Loss: `L = 0.5·(CE_L + CE_Q)` with auxiliary first-token / first-K terms.

### Ablations (guardrails)

- `M ∈ {24, 32, 48}`, `d_z ∈ {192, 256, 320}`; compare **Latent vs Token-budget** at same `M`.
- Encoder swap: `stq` vs `simple-st` to quantify the positional benefit.

## Update (2025-09-25): Latent adaptation diagnostics

- Introduced per-loss gradient diagnostics in `latentwire/train.py` (`--grad_diag_interval`, `--grad_diag_components`). Every N steps we now log gradient norms for teacher-forced CE, first-token CE, k-step KD, hidden-state KD, and alignment terms, so stalled objectives are obvious without digging through tensorboard dumps.
- `scripts/run_llama_single.sh` understands sweep lists (`LATENT_LEN_LIST`, `D_Z_LIST`, `REFINER_LAYERS_LIST`, `REFINER_HEADS_LIST`) and enables the diagnostics by default (Stage A every 100 steps, Stage B every 50). This gives us a one-command latent/refiner grid on the 4×H100 node with gradient health indicators baked into the logs.
- Added a gist reconstruction head (`--use_gist_head`) that projects the latent wire back into prompt embeddings via learned queries. We mask targets with `--gist_mask_prob` to simulate gist attention and score an embedding MSE (`--gist_weight`). The additional loss should ensure the latent preserves enough prompt information for downstream adapters.
- Diagnostics can be tee'd into JSONL (`--diagnostic_log`), giving us structured per-step records (first-token acc, gist loss, grad norms) for the controlled SQuAD smoke runs before scaling.
- Hero preparation: `run_scoped_softprompt_multi.sh --hero` increases Stage B duration (8→14 epochs), doubles sample counts, and writes the same diagnostics stream so hero sweeps share the exact acceptance gate we use during smoke testing.

## Update (2025-09-25): Deep prefix injection is live

- Added `DeepPrefixGenerator` modules per backend. Each maps the calibrated latent prefix into layer-wise key/value caches (prompt dropout → LayerNorm → residual MLP → per-layer projections). The feature is gated by `--use_deep_prefix`, with `--deep_prefix_len` / `--deep_prefix_dropout` to sweep capacity.
- Training links the deep prefix through every prefix-aware path: teacher-forced loss, first-token CE, KD (logits + hidden states), and chunked latent generation now accept an optional `deep_prefix_past`. Gradients flow into the generator, so we can co-train shallow embeddings and deep prompts.
- Checkpoints persist `deep_prefix_{llama,qwen}.pt` alongside adapters; `config.json.deep_prefix = {enabled,len,dropout}` keeps eval in sync, and resume restores generator weights automatically.
- Evaluation reconstructs the same deep prefix caches before scoring. Latent NLL, first-token accuracy, joint rescoring, and decode chunks slice/cache the per-layer K/V tensors so A/B against the legacy shallow-only path is faithful.

---

**Project type:** Systems + Methods (MLSys)  
**Target venue:** MLSys / systems-for-ML workshop track (or main)  
**Project window:** 8 weeks (Week 8 reserved for writing and polish)

---

## Abstract

Large Language Models (LLMs) such as Llama and Qwen are trained with different tokenizers and internal representations. Today, if we want two heterogeneous LLMs to “share” context or collaborate, we serialize information as **text**, retokenize it for each model, and pay the full **prefill** (prompt) cost in compute, latency, and bandwidth. This proposal develops **LatentWire**, a compact **interlingua**—a short sequence of continuous vectors ("soft tokens")—that multiple, frozen LLMs can consume directly via `inputs_embeds`. **Unlike existing soft prompt methods (single model) or multi-model systems (text-based), LatentWire is the first to enable different LLM families to communicate via learned continuous embeddings.** The interlingua replaces a long textual prompt with **M vectors of dimension `d_z`** (e.g., `M=8`, `d_z=256`), mapped into each model’s embedding space by tiny **adapters**. We train a small **encoder** (bytes or sentence-embedding based) and the model-specific **adapters** while keeping the LLMs frozen. Training uses a standard next-token loss on gold answers; evaluation compares **text vs latent vs token‑budget** baselines for quality (EM/F1), conditioning (NLL/token), efficiency (compression and latency), and **two‑model synergy** (joint rescoring).

We hypothesize that at moderate compression (≥4×), a shared interlingua matches or closely approaches text prompting while enabling measurable two‑model gains. We outline a minimal but thorough plan: architecture, metrics, ablations, risks (including information bottlenecks and model‑space mismatch), and concrete milestones. The result is an end‑to‑end, reproducible system demonstrating that **heterogeneous LLMs can consume the same compact continuous context** and **work better together** than either alone—all without fine‑tuning the LLMs themselves.

---

## 1. Introduction

**Problem.** LLMs are excellent reasoners when given a rich prompt, but prompts are long, textual, and **model‑specific** (tokenizer‑dependent). In multi‑model systems—e.g., a code specialist and a math specialist—transferring context between models means **restringifying everything as text** and paying full prompt costs **twice** (retokenize + prefill). This is wasteful, slow, and brittle.

**Goal.** Learn a **compact, model‑agnostic interlingua** that both Llama and Qwen can consume **directly**—a small number of **continuous vectors** instead of hundreds of tokens—while preserving downstream task quality and enabling **two‑model cooperation**.

**Why now?** Modern toolchains expose `inputs_embeds` for direct prefix seeding, and lightweight adapters make it practical to learn small front‑ends while keeping giant LLMs frozen. The community has shown soft prompt/prefix tuning for single models; we extend the idea to **cross‑model shared prefixes** with **frozen heterogeneous LLMs**.

**Scope.** We target question answering (HotpotQA) to measure facts + reasoning under strong baselines. We start with **TinyLlama‑1.1B** and **Qwen2‑0.5B** for feasibility on commodity hardware, and our code supports larger variants (e.g., Llama‑3.1‑8B, Qwen2‑7B).

**Novelty.** While soft prompts have been explored extensively since 2021 for single models, and multi-model ensembles exist, **no prior work demonstrates heterogeneous LLMs consuming the same learned continuous prefix**. This positions LatentWire as the first practical wire protocol for embedding-level LLM communication.

---

## 2. Background (for readers with ML‑101)

> This section anticipates common questions (from undergrads to advisors).

### 2.1 Tokens, embeddings, and transformer prefill vs decode

- **Tokens** are integer IDs from a tokenizer’s vocabulary. Text → tokens is model‑specific (Llama’s tokenizer ≠ Qwen’s).
- **Embeddings** are vectors the model looks up for each token (think: numerical meaning).
- **Prefill** means processing the **prompt** (context) to build the model’s internal state (KV cache). **Decode** is generating new tokens one by one using that state.
- **Cost:** Prefill cost scales with prompt length. If your prompt is 500 tokens, prefill is expensive. Decode cost is per generated token.

### 2.2 What is “the M vector”?

We call `M` the **latent length**—the number of **soft tokens** we feed as the prefix interlingua. A normal text prompt might be 300–1000 tokens; we replace it with **M vectors** (e.g., 8, 12, 16). Each vector has dimension `d_z` (e.g., 256). After a tiny linear adapter, those become each model’s prefix embeddings.

### 2.3 What is backpropagation (“backprop”)?

Backprop is the algorithm used to update parameters in neural networks. We compute a **loss** (how wrong the model is), then use the chain rule to compute gradients for each parameter and step them to reduce the loss.

### 2.4 What does “freezing the weights” mean?

We **do not** update the LLM’s parameters (billions). We only update our **small encoder** and **adapters** (millions or less). Freezing keeps compute/memory manageable and isolates whether a **universal prefix** can condition heterogeneous LLMs without changing them.

### 2.5 What is an adapter? What is LoRA? Why not LoRA now?

- An **adapter** is a small module that maps from our latent space (`d_z`) into a specific model’s embedding size (`d_model`). Here it’s a simple LayerNorm + Linear + tanh clip.
- **LoRA** (Low‑Rank Adapters) inserts trainable low‑rank matrices into the LLM’s attention/MLP layers. It changes the LLM’s behavior. We **do not** use LoRA initially because we want to prove a **frozen‑model interlingua** works; LoRA is a **stability fallback** if frozen acceptance is too brittle.

### 2.6 What is “joint rescoring”?

We let **both** models generate answers from the same latent prefix and then **score each candidate under both models**; we pick the answer with the higher **sum of log‑probabilities**. This is a simple way to get **two‑model synergy** without complex message passing.

---

## 3. State of the Art (SoTA)

- **Soft prompts / prefix tuning (Li & Liang, ACL 2021; Lester et al., EMNLP 2021).** Learn continuous task-specific vectors for **single models**. Gist Tokens (Mu et al., 2023) achieve 26x compression but within one model family. These assume fixed tokenizer/model space.
- **Byte-level models (ByT5, Xue et al. 2022; CANINE, Clark et al. 2022; BLT, Meta 2024).** Process text at byte level to avoid tokenization, but don't enable cross-model communication via continuous embeddings.
- **Multi-model ensembles (DeePEn, April 2024; Co-LLM, ACL 2024; MoA, June 2024).** Achieve collaboration through probability fusion or text intermediaries. No direct embedding-level communication between heterogeneous models.
- **Emergent communication (Lazaridou & Baroni 2020; Discrete Messages, Dec 2023).** Theoretical work on agents developing protocols, but not implemented for commercial LLM families.
- **Prompt compression (AutoCompressors, Chevalier et al. 2023; ICAE, Ge et al. 2024).** Compress within model families, not across heterogeneous architectures.
- **Cross-model communication (Exchange-of-Thought, Dec 2023; DroidSpeak, Dec 2024).** Use natural language or KV-cache sharing respectively, not learned continuous prefixes.

**Gap:** There is no simple, end‑to‑end, task‑measurable demonstration that a **single continuous interlingua** can condition **heterogeneous, frozen LLMs** comparably to text while reducing prefill and enabling measurable two‑model gains.

---

## 4. Research Gaps We Tackle

1. **Tokenizer heterogeneity:** A prompt that works for Llama must be retokenized for Qwen. We bypass text entirely with a **shared continuous prefix**.
2. **Efficiency:** Long prompts are expensive. A short `M`‑vector interlingua reduces prefill **length** and **payload** (bytes over a wire). **No existing work demonstrates >4x compression while maintaining cross-model compatibility.**
3. **Synergy:** Show that two models **jointly** outperform either alone under the same prefix.
4. **Fairness:** Compare against a **token‑budget baseline** (truncate text to `M` tokens) to ensure gains are not just from more budget. **Critical for proving learned compression > naive truncation.**
5. **Stability with frozen LLMs:** Ensure prefixes are **accepted** by frozen models (statistics, normalization, small adapters).
6. **Wire protocol innovation:** Unlike all existing work that uses text or probability distributions for inter-model communication, we establish **continuous embeddings as a wire protocol** between different LLM families.

---

## 5. Our Contributions (What We Will Do)

### 5.1 System overview

**Key innovation:** While prefix tuning exists for single models and byte-level models exist for tokenizer-free processing, **no prior work combines these to enable heterogeneous LLMs to consume the same compressed continuous representation**. This is the first demonstration of Llama and Qwen communicating via learned soft tokens rather than text.

```
              +-------------------+
   text ----> |   Interlingua     | -- Z ∈ R^{M × d_z} --> [adapters] --> Llama
 (Q+Context)  |   Encoder         |                              \--> Qwen
              +-------------------+
                          ^
                          |  (two variants)
                    (1) ByteEncoder + Cross-Attn Pooler
                    (2) SimpleEncoder (MiniLM) + Learned Queries
```

- **Interlingua Encoder** (two variants):
  - **ByteEncoder**: byte embeddings → tiny Transformer → **LatentPooler** (cross‑attention) → `Z` (M×d_z).
  - **SimpleEncoder**: sentence embedding (MiniLM, frozen) → linear projection + **learned query bank** → `Z`.
- **Adapters:** `A_L: R^{d_z}→R^{d_L}` and `A_Q: R^{d_z}→R^{d_Q}` (LayerNorm→Linear→tanh) map `Z` into each model’s embedding space.
- **Frozen LLMs:** Llama and Qwen weights are not updated.
- **Training loss:** next‑token loss on the gold **answer** with the prefix masked out of labels (teacher‑forcing on the suffix only).
- **Evaluation:** Text vs Latent vs Token‑budget; NLL/token; compression & latency; joint rescoring; agreement; oracle bound.

### 5.2 Notation and training objective

- Let `x = "Question: …\nContext: …\nAnswer:"` and `y = gold answer` (tokenized per model).
- **Encoder** produces `Z = [z₁,…,z_M] ∈ R^{M×d_z}`.
- **Adapters** produce `P_L = A_L(Z) ∈ R^{M×d_L}`, `P_Q = A_Q(Z) ∈ R^{M×d_Q}`.
- For a model `ℓ ∈ {L, Q}` we set `inputs_embeds = [P_ℓ, E_ℓ(y[:-1])]` and labels `[-100,…,-100, y[1:]]`.
- Loss is standard cross‑entropy over the answer positions. Total loss: `L = 0.5 * (L_Llama + L_Qwen)`.
- **Why mask the prefix?** We do not want to punish the prefix for not “predicting itself”—it is **context**, not a target.

### 5.3 Decoding

- We **seed** the model with `P_ℓ` (one forward pass) to build the KV cache (short prefill, length `M`), then generate tokens step‑by‑step as usual.
- This preserves decode behavior; only prefill length is reduced.

### 5.4 Two‑model synergy (joint rescoring)

- Generate one answer from each model under the same latent prefix.
- **Rescore** both answers under **both** models’ log‑prob and pick the larger sum.
- This provides a **deterministic, simple** ensemble that often beats either single model, especially when the models disagree.

### 5.5 Why this design should work

- **Soft prompts already work** in single‑model settings: models can be conditioned by continuous prefixes.
- **Adapters + normalization** make `Z` resemble valid token embeddings statistically, so frozen LLMs accept them.
- **Encoder capacity (`M × d_z`)** is enough to carry the **gist** needed to answer questions—especially when the question and a small context are summarized by MiniLM (SimpleEncoder) or byte‑Transformer features (ByteEncoder).
- **Two‑model diversity** improves robustness: reranking across heterogeneous models reduces idiosyncratic errors.

### 5.6 What we will test (explicit hypotheses)

- **H1 (Quality @ Compression):** At `M ≤ 16`, Latent F1 ≥ Text F1 − 2.0 pts with **≥4× compression**.
- **H2 (Better than token budget):** At the same `M`, Latent F1 > Text‑truncated‑to‑M F1 by ≥ **+3.0** pts.
- **H3 (Conditioning):** Average NLL/token on gold answers under Latent ≤ **1.05×** Text.
- **H4 (Synergy):** Joint rescoring ≥ **+1.5** F1 over the better single‑latent model.
- **H5 (Efficiency):** Prefill time and payload bytes reduced materially (≥4× in tokens; payload measured in bytes).

### 5.7 Engineering choices (and alternatives)

- **Start with SimpleEncoder on Mac (MPS)** for speed, then confirm ByteEncoder on GPU. (Same training loop.)
- **Frozen LLMs first**, LoRA only if frozen acceptance is unstable.
- **Equal‑weight joint rescoring** first; learned aggregator (logistic regression) only if needed.
- **Asymmetric interlingua** (shared + model‑specific sub‑prefixes) is a follow‑up if shared‑only underperforms.

---

## 6. Experimental Design

### 6.1 Dataset & splits

- **HotpotQA** (`fullwiki` preferred; `distractor` fallback for MPS sanity). We use a **subsample** for feasibility on laptops and larger sets on GPU.
- Typical subsets: `train: 10k`, `val: 1k`, `test: 1k` (for GPU runs). On MPS, smaller (`train: 256–512`, `val: 200`).
- We evaluate on HotpotQA (multi‑hop) and SQuAD (single‑hop) to demonstrate generality. SQuAD’s single‑paragraph structure yields clearer systems metrics (prefill time vs M), while HotpotQA stresses multi‑document reasoning.

### 6.2 Models

- **TinyLlama/TinyLlama‑1.1B‑Chat‑v1.0** and **Qwen/Qwen2‑0.5B‑Instruct** (baseline feasibility). Larger (8B/7B) with `--load_4bit` later.

### 6.3 Metrics (Figures of Merit)

1. **Task quality:** EM/F1 on HotpotQA (standard SQuAD‑style normalization).
2. **Conditioning quality:** **Average NLL/token** on gold answers under text vs latent (per model).
3. **Efficiency:**
   - **Compression ratio**: avg prompt tokens / `M`.
   - **Payload bytes** for `Z`: `M × d_z × dtype_bytes`.
   - **Wall‑clock** time for text vs latent (prefill+decode; same hardware).
4. **Two‑model synergy:**
   - **Joint pick F1**, **agreement rate**, and **oracle upper bound**.
5. **Fairness:** **Token‑budget baseline** (text prompt truncated to `M` tokens).

### 6.4 Baselines

- **Text baseline** (full prompt).
- **Token‑budget baseline** at the same `M`.
- **Single‑model latent** (no joint rescoring).
- **Oracle** (upper bound if we magically pick the better of Llama/Qwen outputs).

### 6.5 Ablations (minimal but decisive)

- `M ∈ {6, 8, 12, 16}` (pick the smallest satisfying H1–H5).
- **Encoder type:** SimpleEncoder vs ByteEncoder.
- **d_z ∈ {128, 256, 384}`** (capacity sensitivity).
- **Joint rescoring** vs best single latent.
- (If time) **Asymmetric** latent vs shared‑only.
- (If needed) **LoRA‑early** (Q/V of first layers) vs frozen.

### 6.6 Training details (frozen LLMs)

- Batch size 1 (fits everywhere).
- Optimizer: AdamW (`1e‑4`).
- Adapters: LayerNorm → Linear → tanh clip (stability).
- MPS: `--sequential_models` + `--grad_ckpt` + optional `--fp16_mps`.
- GPU (Linux/CUDA): optional `--load_4bit` for larger models.

### 6.7 Statistical treatment

- Report **mean ± 95% CI** (bootstrap over examples) for F1.
- Fix random seeds for splits; keep configs in JSON for reproducibility.

### §6.8 Benchmarks & Collaboration Patterns

#### SQuAD (single‑hop extraction; MLSys‑friendly)

We add SQuAD v1.1/v2 as a clean single‑paragraph QA benchmark. It offers consistent prompt lengths (100–300 tokens), making our prefill vs latent story crisp.

Metrics: EM/F1; Efficiency: prefill wall‑clock, compression, payload bytes.

Expectation: Text F1 ~85–90% (8B/7B); Latent aims ≥75–80% at ≥10× compression (M=8–16).

Why for MLSys: recognizable, stable lengths, easy apples‑to‑apples systems metrics.

#### MS MARCO v2.1 (two‑stage: ranking → answer)

Pipeline: (1) Llama reranks top‑k passages from BM25/ANCE; (2) Qwen extracts/answers. LatentWire replaces long query+context prompts with M‑vector interlingua fed to both models.

Metrics: MRR@10 (ranking), F1 (QA).

Systems win: Heavy prefill avoided in both stages; show ≥10× prompt compression and ≥40% latency reduction at similar quality.

#### MMLU (routing)

Specialize by subject: STEM → Llama; Humanities/SocSci → Qwen. Optionally compute a shared latent from question and let both models propose an answer; pick via joint rescoring.

Metrics: Accuracy per subject; cost per query.

Systems win: Routing reduces cost by avoiding always calling both, while shared latent enables a cheap joint check.

#### TruthfulQA (factuality)

Use cross‑validation: both models decode; we penalize candidates with low mutual likelihood under the other model’s prefix. Evaluate TruthfulQA accuracy gains at similar or less prefill.

Metrics: TruthfulQA accuracy; self‑consistency score.

Systems win: reliability improvements via model diversity with minimal prompt cost (shared latent).

---

## 7. Risks & Mitigations (top 10)

1. **Information bottleneck (too little capacity).**  
   _Mitigate:_ Increase `M`, use SimpleEncoder (richer global features), and compare to token‑budget baseline. Target `M ≤ 16`.
2. **Model‑space mismatch (Llama vs Qwen).**  
   _Mitigate:_ Start with a **shared** latent; if necessary, add **asymmetric** sub‑prefixes `[shared || model‑specific]` under the same total `M`.
3. **Frozen LLM rejection.**  
   _Mitigate:_ Adapter normalization + tanh clipping (already in code). LoRA‑early only if absolutely needed.
4. **Optimization conflicts.**  
   _Mitigate:_ **Sequential backprop** through each LM (your `--sequential_models`), gradient checkpointing, small LR.
5. **Byte front‑end inefficiency.**  
   _Mitigate:_ Use **SimpleEncoder** on Mac for speed; validate ByteEncoder later on GPU.
6. **Pooling redundancy (queries look at same bytes).**  
   _Mitigate:_ (Optional) diversity/coverage regularizers; in practice, SimpleEncoder sidesteps this early.
7. **Same prefix too restrictive.**  
   _Mitigate:_ Asymmetric variant if shared‑only misses H1 by >2 pts.
8. **No real synergy.**  
   _Mitigate:_ Joint rescoring is already implemented; if flat, lightly calibrate (temperature scaling) or learn a tiny logistic combiner.
9. **Calibration mismatch.**  
   _Mitigate:_ Temperature scaling per model on val split if needed.
10. **Wrong bottleneck (bandwidth vs compute).**  
    _Mitigate:_ Measure **payload bytes** and **wall‑clock** prefill+decode separately and show where we win.
11. **dataset loader format discrepancies (e.g., Hotpot dict‑of‑lists vs list‑of‑pairs) yielding empty contexts.**
    _Mitigate:_: robust indexing code; unit test that Context: length > 0 for sampled examples.

---

## 8. Implementation Plan & Timeline (8 weeks)

**Week 1–2 (MPS feasibility):**

- Bring‑up with **SimpleEncoder**; collect **text vs latent vs token‑budget** metrics at `M={8,12}` on small subsets.
- Verify NLL/token closeness and prefill reduction.
- Produce **metrics.json / metrics.csv** from `eval.py` (done).

**Week 3–4 (GPU confirmation):**

- Repeat at larger scale; optionally switch to **ByteEncoder**.
- Run `M` sweep; pick smallest `M` meeting H1–H5.
- Joint rescoring; optional calibration.

**Week 5–6 (Ablations & systems):**

- Encoder type ablation; `d_z` sensitivity; token‑budget fairness.
- Efficiency profiling; payload/latency tables.

**Week 7 (Stability/fallbacks only if needed):**

- Asymmetric interlingua; LoRA‑early; light aggregator if synergy lagging.

**Week 8 (Writing):**

- Final tables, plots, paper text, reproducibility checklist.

---

## 9. Reproducibility Checklist

- Code versioned with `requirements.txt` and model IDs.
- `eval.py` emits **metrics.json/metrics.csv**.
- Seeds fixed for splits; configs stored in `config.json`.
- Training/eval logs include step time (`sec/step`) and final metrics.
- Minimal scripts to replicate tables.

---

## 10. Expected Outcomes & Impact

- **Demonstrate feasibility**: short, continuous prefixes can condition **frozen heterogeneous LLMs** comparably to text on QA.
- **Efficiency**: ≥4× prompt compression with non‑worse wall‑clock on prefill; concrete payload savings over a wire.
- **Synergy**: simple joint rescoring yields measurable gains over either model alone.
- **Foundation**: A clean, open codebase others can extend (multi‑round comms, asymmetric latents, learned aggregators, LoRA variants).
- We report wire bytes for text (UTF‑8) vs interlingua (fp32/fp16) and show that a single latent payload can be broadcast to multiple models, unlike text which is retokenized per model.

---

## 11. Glossary (for quick reference)

- **M vector / latent length (`M`)**: number of soft tokens in the interlingua prefix (e.g., 8, 12).
- **`d_z`**: latent vector dimension (e.g., 256).
- **Adapter**: small module mapping from `d_z` to a model’s embedding dimension.
- **Freeze weights**: do not update the LLM parameters during training.
- **Backpropagation**: algorithm to compute gradients of the loss w.r.t. parameters.
- **Prefill vs Decode**: prefill processes the prefix; decode generates tokens.
- **NLL/token**: negative log‑likelihood per token—lower is better conditioning.
- **Joint rescoring**: combine both models’ log‑prob scores to pick the better answer.
- **Token‑budget baseline**: text prompt truncated to `M` tokens (fairness control).

---

## 12. Frequently Asked PI Questions (and answers)

**Q1: Why not just fine‑tune the LLM (LoRA) and skip interlingua?**  
A1: We want to prove **frozen** LLMs can consume a shared compact prefix—showing a practical path for multi‑model systems **without touching vendor models**. LoRA is a fallback for stability, not our story.

**Q2: Isn’t `M × d_z` too small to replace 300–1000 text tokens?**  
A2: We aren’t reproducing _every_ token—only the **conditioning signal** needed to answer. Empirically, soft prompts carry task information efficiently; we validate via H1–H5 and compare to the **token‑budget** control.

**Q3: Why should Llama and Qwen agree on the same prefix?**  
A3: They don’t need to “agree”; each gets its own **adapter** mapping the shared latent into its own embedding space. If still restrictive, we try **asymmetric** latents under the same total budget.

**Q4: What if frozen models reject continuous inputs?**  
A4: Our adapters normalize and clip to match embedding statistics. If instability remains, we use **LoRA‑early** (few low‑rank params in first layers) as a surgical fix.

**Q5: Isn’t this just ensembling?**  
A5: Yes—and that’s the point. But unlike typical ensembling, **both models consume the same compact context**, which saves bandwidth/prefill while increasing accuracy through diversity.

**Q6: How do we know we’re not solving the wrong bottleneck?**  
A6: We measure **payload bytes**, **compression**, and **wall‑clock** separately, and present a clear efficiency narrative alongside quality.

---

## 13. What to Run on Your Mac (and What to Send)

**Train (MPS, SimpleEncoder, small subset):**

```bash
source .venv/bin/activate
export RUN="mps_m8_simple_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"; mkdir -p "$OUT"

PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python latentwire/train.py \
  --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
  --samples  512 \
  --epochs   1 \
  --batch_size 1 \
  --latent_len 8 \
  --d_z 256 \
  --encoder_type simple-st \
  --hotpot_config distractor \
  --sequential_models \
  --grad_ckpt \
  --fp16_mps \
  --save_every 100 \
  --save_dir "$OUT/ckpt" \
  2>&1 | tee "$OUT/train.log"

tail -f "$OUT/train.log"   # watch loss and sec/step
```

**Evaluate (writes JSON/CSV):**

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --samples 200 \
  --max_new_tokens 16 \
  --hotpot_config distractor \
  --out_dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"

# Quick extract
echo "==== STEP_TIMES ===="
grep -E "sec/step" "$OUT/train.log" | tail -n 5
echo "==== EVAL_JSON ===="
cat "$OUT/metrics.json"
```

**Send me:** the **metrics.json** plus the last few lines with `sec/step` from the train log.

---

## 14. Quiz (30 questions + answers)

**1) What is the “interlingua” in LatentWire?**  
_A short sequence of continuous vectors (“soft tokens”) produced by a small encoder and consumed by multiple LLMs via `inputs_embeds`._

**2) What does `M` represent?**  
_The number of soft tokens in the interlingua prefix (latent length). Lower `M` = higher compression._

**3) What is `d_z`?**  
_The dimension of each soft token vector in the interlingua._

**4) Why do we mask the prefix in the training labels?**  
_The prefix is context; we don’t want the loss to penalize it for not predicting itself._

**5) What does “freezing the LLM” mean and why do we do it?**  
_We don’t update LLM weights; we only train the small encoder and adapters. It isolates the interlingua’s effectiveness and reduces compute/memory._

**6) What is `inputs_embeds`, and why is it central here?**  
_It lets us feed embeddings directly instead of token IDs, enabling continuous soft‑prefixes._

**7) How does joint rescoring work?**  
_Let both models generate answers; score each answer under both models’ log‑probs; choose the higher sum._

**8) What is “token‑budget baseline,” and why is it necessary?**  
_Text prompt truncated to `M` tokens; it ensures fairness—same prefix budget as our latent._

**9) What is NLL/token and why do we report it?**  
_Average negative log‑likelihood per gold token; indicates conditioning quality independent of decoding randomness._

**10) What does `--sequential_models` change?**  
_Backprop through Llama then Qwen sequentially, reducing peak memory by not holding both graphs simultaneously._

**11) When and why would we enable `--grad_ckpt`?**  
_To reduce activation memory (at extra compute), helpful on MPS or limited GPUs._

**12) Why start with SimpleEncoder on Mac and not ByteEncoder?**  
_Byte sequences are longer and slower to process; SimpleEncoder (MiniLM) gives a fast feasibility signal._

**13) What is the expected compression ratio target and how is it computed?**  
_≥4×, computed as avg text prompt tokens divided by `M`._

**14) How do we compute the interlingua payload bytes?**  
_`M × d_z × bytes_per_value`, where bytes_per_value depends on dtype (e.g., fp16 ~2 bytes)._

**15) Why do we apply LayerNorm + tanh in the adapters?**  
_To match expected embedding statistics and clip extremes so frozen LLMs accept the prefix._

**16) What if the latent underperforms the token‑budget baseline?**  
_Increase `M`, improve the encoder (ByteEncoder on GPU), or use an asymmetric latent (shared + model‑specific) under the same total budget._

**17) Why might two models together outperform either alone under joint rescoring?**  
_Model diversity: different failure modes; rescoring aggregates evidence and reduces idiosyncratic errors._

**18) What is the difference between prefill and decode cost here?**  
_Prefill depends on prefix length (`M` vs hundreds of tokens for text); decode cost is the same._

**19) Why is LoRA not used by default?**  
_It modifies the LLM; we want to demonstrate a **frozen‑model** interlingua first. LoRA is reserved as a stability fallback._

**20) What is the “agreement rate,” and why care?**  
_Fraction of examples where Llama and Qwen produce the same normalized answer; signals diversity._

**21) What is the “oracle upper bound”?**  
_Accuracy if we magically always choose the better of the two model outputs; shows headroom for ensembling._

**22) How do we ensure results are not cherry‑picked?**  
_Fix splits and seeds; write metrics to JSON/CSV; report mean ± CI across test sets._

**23) What might cause “frozen LLM rejection” and how do we detect it?**  
_Feeding out‑of‑distribution embeddings; symptoms include repetitive outputs and poor NLL/token. Our adapter normalization mitigates this._

**24) What is the rationale for having both SimpleEncoder and ByteEncoder?**  
_SimpleEncoder accelerates feasibility on laptops; ByteEncoder gives model‑agnostic byte‑level evidence on GPUs._

**25) What is the main risk with very small `M`?**  
_Information bottleneck: prefix can’t carry enough context; F1 drops or NLL/token rises._

**26) Why compare to text truncated to `M` tokens?**  
_To ensure we are not just “cheating” by using more prefix budget than a fair text baseline._

**27) How do we compute text vs latent wall‑clock fairly?**  
_Same hardware, same batch, measure end‑to‑end prefill + decode for the same number of samples._

**28) What does “asymmetric interlingua” mean?**  
_Split the budget into `[shared || model‑specific]` parts so each model gets a small bespoke slice while sharing most content._

**29) What would justify enabling a learned aggregator over equal‑weight joint rescoring?**  
_If calibration differs and equal‑weight underperforms; a simple logistic combiner on features (log‑prob, length, repetition) may help._

**30) How do we decide the final `M`?**  
_Run a sweep; pick the smallest `M` satisfying H1–H5 on the validation set._

**31) Why does “frozen acceptance” matter beyond this project?**  
_If frozen LLMs accept continuous prefixes, we can ship multi‑model systems without vendor fine‑tunes or heavyweight adapters._

**32) What evidence will convince a skeptical advisor?**  
_Strong tables: Latent within 2 F1 of Text at ≥4× compression; Latent > Token‑budget; Joint > best single latent; NLL/token close; clear wall‑clock and payload wins; ablations and CIs._

---

## 15. References (informal pointers)

- Soft prompt / prefix tuning; p‑tuning and prompt tuning work (single‑model conditioning).
- Ensembling / reranking practices in QA and generation.
- Practical guidance on KV cache, `inputs_embeds`, and gradient checkpointing in HF Transformers.

_We will include formal references in the paper draft._

---

_Prepared for the LatentWire project team. This proposal is intentionally detailed and pedagogical so a motivated ML‑101 reader can follow every design choice and its justification._

## Addendum A: Prompt & Context Realization

### 1. Prompt Realization for Instruct LLMs

We will apply each model’s apply_chat_template to construct prompts for text baselines and text scoring, and optionally to construct the encoder’s source string for latent training. This aligns inputs with the distribution the checkpoints expect.

### 2. Context Construction for HotpotQA

We will include more sentences per context item (k=4) and up to 3 items, with a 2k‑char cap. On GPU we will switch to supporting‑fact extraction.

### 3. Mac Feasibility vs Realistic Benchmarks

On Mac MPS we use TinyLlama‑1.1B and Qwen2‑0.5B to validate mechanics (compression/latency, frozen acceptance). For quality claims we will run Llama‑3.1‑8B + Qwen2‑7B (4‑bit) on GPU.

### 4. Sanity Dataset

If HotpotQA remains too hard for tiny models even after prompt/context changes, we will add SQuAD v1.1 runs to validate Text vs Latent vs Token‑budget behavior, then return to Hotpot on larger models.

### 5. Updated Milestones

Week 1–2: Chat templates + context shaping on Mac; establish non‑zero Text F1; show Latent within ΔF1 vs Text on SQuAD; report prefill speedups.

Week 3–4: GPU runs on Hotpot with 8B/7B models; M sweep (8/12/16); joint rescoring.

Remaining weeks unchanged.

### 6. Reporting

We will add predictions.jsonl dumps for qualitative error analysis, and we will report both: (i) chat-templated text baselines, and (ii) latent trained with/without templated encoder input.

### 7. Recent Implementation Note (2025-09-22)

- Stage-C evaluation crashed because the chat-template path stripped the `Answer:` literal but did not restore it before constructing the assistant turn. This left both text and latent baselines with empty prefills and produced near-zero EM/F1.
- **Fix:** Always reattach the answer literal for chat-mode anchors (without adding extra BOS tokens) so training and evaluation share the same prompt prefill. The latent prefix remains unchanged, and evaluation now completes with non-zero text baselines.

# Addendum (Sept 2025): Scoped Interlingua via Tiny LoRA + Deep Prefix Injection with Strict Chat-Template Compliance

## What changed and why

Our hero runs showed that a purely frozen two-model interlingua (shared soft tokens + minimal linear adapters) did not reliably "take" as a prefix authority inside different LLM families. We observed (a) near-zero first-token acceptance in latent mode, (b) large gaps to the text baseline, and (c) sensitivity to BOS/anchor handling. The takeaway is consistent with the literature: parameter-efficient adaptation that touches early attention pathways (keys/queries/values) or injects prefixes across layers is often necessary for stable conditioning while keeping models mostly frozen. We therefore adopt a scoped plan that maintains the spirit of a shared latent "wire" but adds just enough per-model trainable capacity to make the signal usable:

- **Tiny per-model LoRA** (rank 8–16) on early attention blocks to improve acceptance of soft-prompted context without editing knowledge broadly. LoRA is a low-rank, parameter-efficient method that empirically matches or exceeds full fine-tuning with orders-of-magnitude fewer trained parameters.

- **Deep prefix injection** (a.k.a. prefix/"deep prompt" tuning) that inserts learned key/value "virtual tokens" in every layer—a proven approach to steer frozen decoders and close the gap to full fine-tuning at scale.

- **Prompt/soft-prompt compatibility at scale** (P-Tuning / P-Tuning v2): with the right optimization, deep prompts can be universally competitive with fine-tuning, especially as models grow. This supports our design to favor prefixes + small per-model adapters over heavier tuning.

- **Strict chat-template compliance** for Llama/Qwen via `tokenizer.apply_chat_template(...)` so our prefixes sit in the expected conversation scaffolding (system/instructions/assistant turns). This removes formatting drift as a confounder.

## Revised objective

Maintain a shared latent interlingua (the M-vector soft prompt we encode from the task input) while equipping each model with just-enough trainable structure (tiny LoRA + deep prefix) to interpret the wire. We target:

- **Compression**: ≥ 4× prompt reduction (e.g., average 240–260 text tokens → M ≈ 48 soft tokens).
- **Quality**: ≥ 80% of each model's text baseline F1/EM on QA-style evaluation (and ROUGE on summarization).
- **Cross-model portability**: one latent wire consumed by both Llama and Qwen with per-model PEFT heads.

## Architecture (concise)

```mermaid
flowchart LR
  A[Task input (QA/sum)] -->|encode| E[Latent Encoder (frozen or lightly tuned)]
  E -->|M x d_z| Z[(Shared latent "wire")]

  subgraph Llama branch
    Z --> P1[Deep Prefix (per-layer KV, learnable)]
    P1 --> L1[LoRA r=8-16 on early attn (per-model)]
    L1 -->|inputs_embeds + chat template| LLM1[Llama (frozen base)]
  end

  subgraph Qwen branch
    Z --> P2[Deep Prefix (per-layer KV, learnable)]
    P2 --> L2[LoRA r=8-16 on early attn (per-model)]
    L2 -->|inputs_embeds + chat template| LLM2[Qwen (frozen base)]
  end

  LLM1 --> O1[Answer/summary]
  LLM2 --> O2[Answer/summary]
```

**Key differences vs. the original**: We still learn a shared latent Z, but we augment the per-model front-end with (1) deep prefixes and (2) small LoRA deltas on the first N attention blocks to ensure the frozen decoders can read Z. Prefix-to-chat alignment is enforced by building the prompt strictly via each model's chat template.

## Training recipe (high level)

- **Backbone**: two frozen instruct LLMs (e.g., Llama-3.1-8B-Instruct, Qwen-2.5-7B/8B-Instruct).

- **Per-model PEFT head**:

  - LoRA on attention projections (q_proj, k_proj, v_proj, o_proj) in early K layers (e.g., first 8–12) with ranks r ∈ {8, 16}, α ≈ 16–32, small dropout.
  - Prefix tuning (deep) with prefix length p ∈ {8–16} per layer (KV prompts).

- **Latent encoder**: sentence-embedding or byte encoder producing M×d_z soft tokens (M≈48, d_z≈256).

- **Losses**:

  - Teacher-forced NLL on gold,
  - First-token CE (stabilize BOS acceptance),
  - Short-horizon K-token CE,
  - (Optional) KD from text-prompted runs for early tokens.

- **Prompt construction**: Always use `tokenizer.apply_chat_template([...], add_generation_prompt=True)` for both Llama and Qwen; the latent prefix and anchor (e.g., Assistant:) are inserted at the assistant preamble location defined by each tokenizer's template.

## Why this should work (evidence & intuition)

- **LoRA** provides a small, targeted path to adapt early attention so the model can interpret non-textual prefixes without touching its knowledge broadly—shown to match or beat full fine-tuning with 10³–10⁴× fewer trained params.

- **Deep prefix tuning** inserts virtual tokens in every layer, giving the model layer-wise hooks to propagate the latent signal—an effect known to achieve near-fine-tuning performance with frozen weights when scaled properly (P-Tuning v2).

- **Prompt/soft-prompt scaling**: As models get larger, prompt/soft-prompt methods close the gap to full fine-tuning, which aligns with our 7B–8B regime.

- **Template correctness** removes a frequent source of failure: inconsistent BOS/assistant headers and spacing differences across Llama/Qwen. HF's chat templates are the canonical way to serialize conversational inputs.

## Evaluation plan

We will report, per model and jointly:

- Text baseline (chat-templated) vs Latent+Prefix vs Latent+Prefix+LoRA (our main),
- Compression (text tokens vs M), latency (prefill time), acceptance (first-token top-k acc),
- Quality: EM/F1 for QA, ROUGE-1/2/L for summarization,
- Ablations: (a) LoRA off, (b) shallow vs deep prefix, (c) M sensitivity, (d) with/without KD.

## Addendum (2025-09-25): Debugging deep prefix rollout

- **What we tried.** We enabled deep prefix generation for both Llama and Qwen with grouped KV caches, combined with the gist reconstruction head and the new gradient diagnostics (Milestones 1–4). Initial smoke runs on the 4×H100 cluster exposed cache shape mismatches (32 attention heads vs 8 KV heads) and cross-device `torch.cat` failures when Accelerate sharded layers across GPUs.

- **What broke.** Prefix projections that assumed `d_z` divisible by the full attention head count produced tensors that could not concatenate with Hugging Face’s grouped KV cache (`transformers` ≥ 4.46) [7]. Even after fixing the head count, the KV tensors remained on GPU0 while later decoder blocks lived on GPU1–3, causing multi-device runtime errors during Stage A.

- **What we changed.** `DeepPrefixGenerator` now projects to `num_kv_heads × head_dim` and `_prepare_deep_prefix` moves each layer’s cache to the layer’s actual device before dispatch. Loss assembly skips the latent-prefix segment when the cache is used, so logits/labels align. The gist head loss is now a masked MSE (normalised by embedding dimension) with a lighter default weight, Stage A reintroduces a text↔latent warm-up, and we activate tiny LoRA adapters (r=8, first 8 layers) during both training and evaluation to help the frozen LM read the latent wire. Both single- and multi-model runners stream the resulting metrics to `diagnostics.jsonl` for post-mortem analysis, and the text baseline was repaired (no double chat templates, cleaner strips "Answer:").

- **What’s next / expected.** With the head/device alignment fixes, we expect Stage A smoke runs to stably report positive first-token accuracy and decreasing gist loss within the first epoch. Hero runs (`run_scoped_softprompt_multi.sh --hero`) should extend those gains to the full Llama+Qwen setting, providing the acceptance metrics we outlined in Milestone 5 (EM/F1, latency, compression) while keeping the structured diagnostics for regression tracking. We will validate against SQuAD (smoke) and HotpotQA (hero) in line with prior work on cross-model latent prompting [1,2,4].

## Risks & mitigations

- **Over-fitting to one model**: Keep the latent shared, but constrain per-model PEFT heads to be tiny (LoRA r≤16, short prefixes) and regularize with early-token KD.

- **Template drift**: Treat chat template as non-negotiable; add CI checks that refuse to run if a template isn't applied.

- **Under-capacity at 4× compression**: If acceptance remains low, we will (1) increase M to 64 temporarily, (2) raise prefix length modestly, or (3) expand LoRA to +4 layers—while tracking parameter budget.

## Implementation notes (PEFT & Transformers)

- Use PEFT for both LoRA and Prefix-Tuning; keep the base model in 4-bit/bfloat16 as appropriate, with gradient checkpointing to fit 4×H100.

- Always wrap inputs with `apply_chat_template(...)` for both models; ensure `add_generation_prompt=True` so the assistant turn header is in place when our latent/prefix is injected.

## References

1. **LoRA** (Hu et al., 2021) — low-rank adapters for efficient fine-tuning. [arXiv](https://arxiv.org/abs/2106.09685)

2. **Prefix-Tuning** (Li & Liang, 2021) — continuous prefixes for frozen LMs. [arXiv](https://arxiv.org/abs/2101.00190)

3. **Prompt Tuning** (Lester et al., 2021) — soft prompts scale well with model size. [arXiv](https://arxiv.org/abs/2104.08691)

4. **P-Tuning v2** (Liu et al., 2021/2022) — deep prompt tuning competitive with fine-tuning. [arXiv](https://arxiv.org/abs/2110.07602)

5. **PEFT docs** — LoRA & Prefix-Tuning implementations. [Hugging Face](https://huggingface.co/docs/peft)

6. **Transformers docs** — chat templates & correct serialization. [Hugging Face](https://huggingface.co/docs/transformers)

7. **Transformers KV cache guide** — grouped-key/value cache API and device placement notes. [Hugging Face](https://huggingface.co/docs/transformers/kv_cache)
