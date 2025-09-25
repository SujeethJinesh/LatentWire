# LatentWire — 8B_clean_answer_ftce — Experiment Log

**Run ID:** `8B_clean_answer_ftce`  
**Start:** Sun Sep 14 23:54:43 PDT 2025  
**Backbones:** - Llama: `meta-llama/Meta-Llama-3.1-8B-Instruct`  
- Qwen:  `Qwen/Qwen2.5-7B-Instruct`  
**Dataset:** SQuAD (`train` for training subsample, `validation` for eval)  
**Seeds:** train seed = 42; deterministic eval seed = 12345  
**Encoder:** `byte` interlingua (token-level input) → `M=32`, `d_z=256`, `BYTE_MAX=2048`  
**Adapters:** 2× linear + scale (to each LM) with RMS calibration to input embeddings  
**Eval mode:** Sequential (per‑LM), `fresh_eval=1` (recompute Z), deterministic first step

---

## 0) Global Flags / Script (for reproducibility)

From `run_pipeline.sh` at time of the baseline and the current re‑run (unless otherwise noted):

- **Training knobs**
  - `EPOCHS=24`, `BATCH_SIZE=64`, `TRAIN_SAMPLES=87599`
  - `ENCODER_TYPE=byte`, `LATENT_LEN=32`, `D_Z=256`, `BYTE_MAX=2048`
  - `LR=5e-5`, `SCALE_L2=0.05`, `ADAPTER_RMS_L2=0.0`, `MAX_GRAD_NORM=1.0`
  - `WARM_ANCHOR_TEXT="Answer: "`
  - `FIRST_TOKEN_CE=0.5` (λ for first‑token CE)
  - `TRAIN_APPEND_BOS="yes"` (BOS appended after prefix+anchor for the **first‑token** objective)
  - `SEQUENTIAL_MODELS=1` (train both LMs in the same step, shared encoder)

- **Eval knobs**
  - `DATASET=squad`, `SMOKE_SAMPLES=200`, `SAMPLES=200`
  - `MAX_NEW_TOKENS=12`, `CHUNK_SIZE=8`
  - `SEQUENTIAL_EVAL=1`, `FRESH_EVAL=1`, `LOAD_4BIT=0`
  - **Anchors & BOS:** `LATENT_ANCHOR_MODE="text"`, `LATENT_ANCHOR_TEXT="Answer: "`, `APPEND_BOS_AFTER_PREFIX="yes"`  
  - **Calibration:** `CALIBRATION="embed_rms"`, `PREFIX_GAIN=1.0`
  - **Decode hardening:** `FIRST_TOKEN_TOP_P=1.0`, `FIRST_TOKEN_TEMPERATURE=0.0`  
    (deterministic first token), `min_new_tokens=3`, `eos_ban_steps=6`.

- **Bookkeeping**
  - Saving `state.pt`, `encoder.pt`, `adapter_{llama,qwen}.pt`, `config.json`, and `training_stats.json` every epoch (end step).

---

## 1) Baseline Observations (before PAD fixes)

### 1.1 High‑level pattern

- **Text prompting** is strong (F1 ≈ 0.80–0.85).
- **Latent prompting** collapses: F1 ≈ 0.006–0.022; **first‑token top‑1 ≈ 0.055–0.075**.
- **Debug generations** show filler loops (“the the the …”) despite RMS calibration and early EOS ban.

> **Key insight:** Training loss looked reasonable, but gradients were dominated by **left‑padded tokens** in the teacher‑forced path (PAD/EOS transitions), not by the actual answer tokens.

### 1.2 Concrete snapshots (from eval logs you posted)

| Epoch | Group     | EM   | F1      | NLL/token (gold) | FirstTok@1 | FirstTok@5 |
|------:|-----------|-----:|---------|------------------:|-----------:|-----------:|
| 14    | **Llama (latent)** | 0.000 | **0.006** | 11.370 | **0.060** | 0.105 |
| 14    | **Qwen  (latent)** | 0.000 | **0.022** | 8.226  | **0.065** | 0.145 |
| 20    | **Llama (latent)** | 0.000 | **0.010** | 11.513 | **0.055** | 0.130 |
| 20    | **Qwen  (latent)** | 0.000 | **0.017** | 8.150  | **0.065** | 0.160 |
| 21    | **Llama (latent)** | 0.000 | **0.008** | 11.240 | **0.060** | 0.110 |
| 21    | **Qwen  (latent)** | 0.000 | **0.015** | 8.194  | **0.075** | 0.165 |

**Text baseline** (constant across these epochs):  
- *Llama:* EM 0.58, F1 ~0.799  
- *Qwen:* EM 0.68, F1 ~0.853

**Selected debug generations (latent, first few; representative):**

- *Llama (e.g., ep20):* - `two years after the first successful use of the vaccine, the`  
  - `the 20th century, the 20th century`  
  - `the 1960s and 1970s. The`  
  - `the the the the the the the the the the the the`  
- *Qwen (e.g., ep20):* - `by the of the of thej`  
  - `the of the of the of theJ`  
  - `the of the and the of the and and`  

**Diagnostics showing RMS/scale were *not* the issue** (ep20 excerpts):  
- `prefix_std ≈ embed_rms` (e.g., Llama: 0.01057 vs 0.01057)  
- `adapter.scale ≈ 1` (e.g., 0.988–1.000)  
- So **amplitude/calibration looked healthy**; the problem lay elsewhere.

---

## 2) Root‑Cause Diagnosis

- We globally set the tokenizer to **left padding** (typical for decoder LMs).  
- During training, we formed TF sequences from the **answers** but did **not**:
  1. **Mask PAD tokens** out of the labels (`-100`), **and**
  2. **Zero their attention** so the model wouldn’t attend to left‑pad noise.
- Result: the CE focused on trivial PAD/EOS transitions instead of content tokens.  
  The model then failed to learn a strong **first token** from the latent prefix, and free‑run decoding collapsed into high‑frequency fillers.

This matches the empirical signals:
- Low first‑token accuracy (~5–7%),  
- “the …” loops despite early EOS ban and good RMS calibration.

---

## 3) Changes Applied (today)

> ✅ **All implemented; optional items are listed in §4 but not turned on yet.**

### 3.1 PAD‑aware losses (code only; no flag changes)

**File:** `latentwire/models.py` (inside `LMWrapper`)

- **`forward_with_prefix_loss(...)`**
  - Mask labels where `label == pad_token_id` → `-100`.
  - Build **attention masks** that **zero out padded TF positions**.
  - Keep ignoring the positions for `[latent prefix]` and optional `[anchor]`.

- **`loss_with_text_prompt(...)`** (used for NLL diagnostics)
  - Same masking for PAD labels.
  - Zero attention at padded TF positions after the prompt.

**Why it should work:** Now the CE is dominated by **real answer tokens**, not padding, so gradients will align the latent prefix + (optional) anchor with the **first content token** and subsequent answer tokens. This is the most common and decisive fix for latent‑prefix training collapse.

### 3.2 Right‑pad **answers** only when building TF labels (code only)

**File:** `latentwire/train.py`  
- Temporarily set `tokenizer.padding_side="right"` just for **answer tokenization** (teacher forcing labels). Everything else stays the same.  
- Rationale: prevents a wall of left PADs at the beginning of TF sequences, further reducing the chance of PAD dominating the loss.

**Why it should work:** Right‑padding ensures the earliest supervised steps correspond to **actual answer tokens**, aligning the loss with what we want the prefix to control (the start of the answer).

---

## 4) Optional ablations (not applied yet)

These are **off** right now. Enable only if needed after observing the post‑fix epoch.

1) **BOS after prefix+anchor (A/B)** - **Flag:** `APPEND_BOS_AFTER_PREFIX="no"` (eval) and `TRAIN_APPEND_BOS="no"` (for first‑token CE)  
   - **Why:** For many chat LMs, a BOS **after** `"Answer: "` can be unnatural and push toward generic fillers. Removing BOS often increases first‑token @1.  
   - **Metric to watch:** first_token_top1 ↑, latent F1 ↑.

2) **Increase first‑token supervision (short boost)** - **Flag:** `FIRST_TOKEN_CE=1.0` (temporarily)  
   - **Why:** Once PAD masking is correct, a slightly stronger first‑step CE can accelerate alignment.  
   - **Metric:** first_token_top1 should move noticeably (>0.10–0.15 in a couple of epochs).

3) **Mild prefix gain at eval** - **Flag:** `PREFIX_GAIN=1.25`  
   - **Why:** Gives the latent prefix slightly more influence at decode time; keep within 1.0–1.5.  
   - **Metric:** latent F1 ↑ without weird phrasing; if outputs over‑shoot or get erratic, roll back.

4) **First‑token nucleus sampling (if greedy remains sticky)** - **Flags:** `FIRST_TOKEN_TOP_P=0.9`, `FIRST_TOKEN_TEMPERATURE=0.7`  
   - **Why:** Adds small stochasticity only to the **first** token; often enough to break filler ties. Determinism remains repeatable under fixed seed.  
   - **Metric:** first_token_top1 ↑; inspect first five generations.

5) **Anchor mode A/B** - **Flag:** switch `LATENT_ANCHOR_MODE="text" ↔ "chat"` (keep text `"Answer: "` vs. letting the model’s chat template drive)  
   - **Why:** If an LM strongly expects its chat formatting, aligning the anchor mode can help.  
   - **Metric:** first_token_top1 & latent F1.

---

## 5) What we expect **after the fixes in §3** (acceptance criteria)

These are *expectations*, not guarantees, to decide next actions:

- **First‑token acc (top‑1)** should rise substantially above chance, typically into the **0.15–0.30** range after 1–2 epochs.  
- **Latent F1** should move off the floor (no longer ~0.01); any **monotonic** improvement across epochs is the signal we want.
- **Qualitative**: the “the the the …” loops should mostly disappear in the first few debug generations.

**If, after one epoch with the fixes, first_token_top1 is still < 0.10**, apply ablation **(1)** (BOS=no). If still flat, try **(2)** FIRST_TOKEN_CE=1.0 for an epoch.

---

## 6) Evidence the issue wasn’t amplitude/calibration

- Logs consistently showed `prefix_std ≈ embed_rms` and `adapter.scale ≈ 1`.  
- CE loss numbers (1.1–1.6) were **much** lower than the **latent NLL/token** at eval (8–11), consistent with CE being dominated by easy PAD/EOS.  
- Early EOS was already banned for the first steps (`eos_ban_steps=6`, `min_new_tokens=3`), so sampling wasn’t the root cause.

---

## 7) Current status

- ✅ **Code fixes applied**: PAD‑aware CE + right‑padded answers for TF (train + eval loss paths).  
- 🚫 **Not applied (yet)**: BOS=no, FIRST_TOKEN_CE bump, PREFIX_GAIN>1, first‑token sampling tweaks.

**Next action:** run the provided script unchanged (keeps `APPEND_BOS_AFTER_PREFIX="yes"`, `FIRST_TOKEN_CE=0.5`) to **isolate** the effect of the PAD fixes. Then review:
- `eval_epoch*/metrics.json` → `latent.first_token_top1/top5`, `latent.f1`  
- `eval_epoch*/predictions.jsonl` → quick scan of first 5 predictions per LM.

---

## 8) Notes, warnings, and environment quirks

- HF Transformers >=4.46 warning: *“`logits` model output will have the same type as the model …”* — informational only.  
- KV cache deprecation: *“`past_key_values` as a tuple of tuples … will be removed in v4.47”*. Our usage is fine for now; unrelated to the collapse.  
- We record `training_stats.json` with prefix RMS stats per LM; these confirm RMS calibration is behaving as intended.

---

## 9) Minimal checklist to avoid running in circles

- [x] Mask PAD in **labels** (train + eval losses)  
- [x] Zero **attention** on padded TF positions  
- [x] **Right‑pad** answers when constructing TF labels  
- [ ] (If needed) BOS after prefix+anchor **OFF** (`APPEND_BOS_AFTER_PREFIX="no"`, `TRAIN_APPEND_BOS="no"`)  
- [ ] (If needed) Temporarily **increase** `FIRST_TOKEN_CE` to `1.0`  
- [ ] (If needed) `PREFIX_GAIN=1.25` at eval  
- [ ] (If needed) First‑token `top_p=0.9`, `temperature=0.7`  
- [ ] (If needed) Anchor mode A/B: `text` ↔ `chat`

**Stop criteria for each ablation:** keep one change for 1–2 epochs; if no improvement in `first_token_top1` and latent F1, revert and try the next.

---

## 10) Appendix — representative flags & their *why*

- `LATENT_ANCHOR_TEXT="Answer: "`: provides a short, stable context to bias the LM toward concise answers.
- `CALIBRATION="embed_rms"` + `PREFIX_GAIN=1.0`: matches latent amplitude to the LM’s input embedding RMS (prevents blown logits while keeping signal).
- `FIRST_TOKEN_CE=0.5`: adds explicit supervision on the first step; we may tune this after PAD fixes if first‑token acc is still low.
- `APPEND_BOS_AFTER_PREFIX="yes"`: kept **on** initially for continuity with earlier runs; we will A/B `no` if needed.
- `min_new_tokens=3`, `eos_ban_steps=6`: bans early EOS / chat EOT tokens; ensures we observe a proper first token and short continuation.
- `SEQUENTIAL_EVAL=1`, `FRESH_EVAL=1`: recompute Z per model (text alignment) and avoid stale caches; crucial when encoders or wrappers change.

---

### 2025‑09‑15 — Run 8B_clean_answer_ftce (SQuAD)
**Goal:** make latent prompting usable by fixing loss target hygiene and first‑token alignment, while holding capacity at M=32 (vs prior runs at M=16).

#### Hardware / Models
- **GPUs:** `CUDA_VISIBLE_DEVICES=0,1`
- **LLMs:** `meta-llama/Meta-Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- **Encoder:** `byte` (`BYTE_MAX=2048`)
- **Latent shape:** `LATENT_LEN=32`, `D_Z=256`

#### Common eval settings (Epoch 1–2)
- **Dataset:** `squad`, `samples=200`, `max_new_tokens=12`
- **Latent anchor:** `mode=text`, `text="Answer: "`
- (As run in Epoch 1–2) `APPEND_BOS_AFTER_PREFIX="yes"` (training matched eval)
- **Calibration:** `embed_rms`, `prefix_gain=1.0`
- **First step decode:** `first_token_top_p=1.0`, `first_token_temperature=0.0` (greedy first token)
- Sequential eval with fresh Z: `--sequential_eval --fresh_eval`

#### Training knobs (Epoch 1–2)
- `EPOCHS=24`, `BATCH_SIZE=64`, `TRAIN_SAMPLES=87599`
- `LR=5e-5`, `SCALE_L2=0.05`, `ADAPTER_RMS_L2=0.0`, `MAX_GRAD_NORM=1.0`
- **First‑token CE:** `first_token_ce_weight=0.5`
- (As run) `train_append_bos_after_prefix="yes"`
- **Save cadence:** end of each epoch; smoke eval each epoch (200 samples)

#### What we changed before this run (code hygiene)
- Cross‑entropy masking & right‑padding fixes in `train.py`/`models.py`
  - **Why:** avoid training on pad/garbage; align targets with real tokens.
  - **Expected effect:** immediate drop in latent NLL; steadier training curves.
- Anchor consistency `Answer: ` used in both train and eval.
  - **Why:** reduce train/eval mismatch at the first step.
  - **Expected effect:** lower variance in first‑token logits; better NLL.

#### Results so far (Epoch 1 → Epoch 2)
- **Text baseline** (reference, unchanged across epochs)
  - Llama F1 **0.799**, Qwen F1 **0.853**
- **Latent path** (shared interlingua)
| Metric | Epoch 1 | Epoch 2 | Δ |
| :--- | :--- | :--- | :--- |
| **Llama NLL/token (gold)** | 8.1683 | 7.8636 | –0.3047 (–3.73%) |
| **Qwen NLL/token (gold)** | 7.7830 | 7.4624 | –0.3206 (–4.12%) |
| **Llama F1** | 0.0205 | 0.0312 | +0.0107 |
| **Qwen F1** | 0.0035 | 0.0095 | +0.0060 |
| **Llama FirstTok@1** | 0.030 | 0.025 | –0.005 |
| **Llama FirstTok@5** | 0.040 | 0.075 | +0.035 |
| **Qwen FirstTok@1** | 0.060 | 0.055 | –0.005 |
| **Qwen FirstTok@5** | 0.125 | 0.140 | +0.015 |

- **Calibration / amplitude (debug)**
  - `Z.std`: 0.606 → 0.662 (encoder “using the space” more)
  - `adapter.scale`: ~1.0 (calibrator doing its job)
  - `rms_mean_raw` (train): Llama 0.632 → 0.696, Qwen 0.618 → 0.692 (pre‑calibration scale rose; OK with `embed_rms`)
- **Qualitative:** First generations still dominated by function‑word loops ("the of the …"), indicating the first‑token decision is still under‑aligned despite the NLL gains.

#### Interpretation:
The NLL/F1 improvements are coming from the target hygiene + anchor consistency changes; the bottleneck is first‑token alignment. Greedy first step (temp=0.0) plus a BOS inserted after the anchor makes the LM default to high‑frequency function words when the latent signal isn’t yet strong.

#### Decision after Epoch 2
Proceed to Epoch 3 to capture one more checkpoint under the “Stage‑A” settings, then stop and restart with a first‑token–focused configuration (“Stage‑B”) aimed at breaking the "the/of/and" failure mode.

#### Stage‑B configuration (to apply after Epoch 3)
- **Exact flag deltas (A → B):**
| Old Setting | New Setting |
| :--- | :--- |
| `APPEND_BOS_AFTER_PREFIX="yes"` | `APPEND_BOS_AFTER_PREFIX="no"` |
| `TRAIN_APPEND_BOS="yes"` | `TRAIN_APPEND_BOS="no"` |
| `FIRST_TOKEN_CE=0.5` | `FIRST_TOKEN_CE=1.0` |
| `PREFIX_GAIN=1.0` | `PREFIX_GAIN=1.15` |

- **Rationale:**
  - **Remove BOS after the anchor (train+eval):** keeps the latent+anchor in a single continuous stream so the very next token is conditioned by the latent, not reset toward generic sentence starts.
    - **Hypothesis:** should lift FirstTok@1/@5 noticeably within the next couple of epochs.
  - **Double first‑token CE weight:** increases gradient pressure on the first decision.
    - **Hypothesis:** pushes the latent to create a clear margin on the correct first word.
  - **Mild PREFIX_GAIN at decode:** gives the latent a small nudge without destabilizing longer‑range decoding.
- **What stays the same:** `LATENT_LEN=32`, `LR=5e-5`, `SCALE_L2=0.05`, deterministic first step for now (`top_p=1.0`, `temp=0.0`). We’ll revisit decode sampling only if first‑token accuracy remains flat after these changes.

#### Measurement plan for Stage‑B
Track, per epoch (200‑sample smoke eval):
- FirstTok@1/@5 (primary success signal)
- Latent NLL/token (should continue trending down or hold)
- Latent F1 (should move up along with FirstTok metrics)
- Debug first generations (expect function‑word loops to fade)
- **Guardrail:** if FirstTok@1 does not improve meaningfully after 1–2 epochs on Stage‑B, switch eval first‑step to `first_token_top_p=0.9`, `first_token_temperature=0.7` and sweep `PREFIX_GAIN` in `[1.10, 1.25]`.

#### Artifacts & paths (for reproducibility)
- **Epoch 1 eval:** `runs/8B_clean_answer_ftce/eval_epoch1/metrics.json`
  - Llama latent: F1 0.021, NLL 8.168; Qwen latent: F1 0.003, NLL 7.783
- **Epoch 2 eval:** `runs/8B_clean_answer_ftce/eval_epoch2/metrics.json`
  - Llama latent: F1 0.031, NLL 7.864; Qwen latent: F1 0.009, NLL 7.462
- Debug snippets show first generations dominated by "the/of/and" patterns in both epochs.
- **Next action:** Stop after Epoch 3 checkpoint is written, then restart training with the Stage‑B script above (resume from latest ckpt).

### 2025‑09‑15 — Latent prompting stalled at first token; fix plan

#### What went wrong (evidence)
- **Latent F1/EM remain near zero** across two successive epoch evals on SQuAD (M=32):
  - *Epoch 1:* Llama EM 0.000 / F1 0.025, Qwen EM 0.000 / F1 0.009
  - *Epoch 2:* Llama EM 0.000 / F1 0.025, Qwen EM 0.000 / F1 0.013
- **First‑token accuracy is flat/very low** despite more training:
  - Llama Top‑1 2.5% → 4.0%, Qwen ~6.0%; Top‑5 stays <16%.
- **Oracle upper bound is also tiny** (F1 ≈ 0.025–0.028), meaning errors are systematic at the first decode steps, not sampling.
- **Degenerate first generations at eval** (debug): e.g., "the of …", numeric runs ("1919…")—typical when the model can’t read the latent evidence and falls into function‑word attractors.
- **Amplitude calibration looks fine** (RMS near targets; adapter.scale ≈ 1.0), so the issue is semantic alignment, not scale.

**Diagnosis:** We are supervising only the `t=0` decision (first‑token CE) and relying on scalar RMS calibration. That does not provide enough signal for steps 0–3 to land on the same distribution the model uses under text prompting. As a result, decoding enters a generic basin and never recovers within a 12‑token budget.

#### Attempted solution (what we will change)
We are adding early‑step guidance + a slightly more expressive prefix mapping, plus a guardrail check.

1.  **K‑token teacher‑forced CE (K=4) after the "Answer: " anchor**
    - Supervise the first 4 answer tokens under the latent prefix (teacher forcing).
    - Keep the existing first‑token CE; fold it into this K‑step average.
    - Loss weights to start: `λ_first = 1.0`, `λ_kce = 0.5`.
2.  **Prefix knowledge distillation (KD) for `t=0..K-1` from the text‑prompted teacher**
    - Run the same LLM with the text prompt and teacher‑force `t=0..K-1` to get teacher logits.
    - Minimize `KL(teacher || latent‑student)` over those steps.
    - Loss weight to start: `λ_kd = 0.5` (lower to 0.25 if unstable).
3.  **Per‑channel affine calibration on the prefix (γ, β)**
    - After RMS calibration, apply a learnable element‑wise scale and bias on the injected prefix to correct directional mismatch (not just magnitude).
    - L2‑regularize `(γ−1, β)` with weight ≈ 1e‑4.
4.  **Upgrade the adapter to a tiny 2‑layer MLP (GELU)**
    - `Linear(d_z → 4·d_model) → GELU → Linear(4·d_model → d_model)` with WD ≈ 1e‑4.
    - This gives the encoder a small nonlinearity to map latent space into the LLM’s prefix manifold.
5.  **Eval‑only nudges (temporary, to reflect progress sooner)**
    - *First token decode:* `top_p=0.9`, `temperature=0.7` (`t=0` only), then deterministic.
    - *Prefix gain schedule:* `gain@t0=1.25`, `gain@t1=1.10`, then 1.0.
    - Reduce `eos_ban_steps` from 6 → 0–1 to avoid forced babbling on short answers.
    - *(Optional demo‑only)* light stop‑list at `t=0` for `the, of, and, to, in, a, is, was` to remove the most common attractors.
6.  **Sanity check: anchor/label alignment assertion (both tokenizers)**
    - Verify the first gold token after `"Answer: "` is the same id used as `y_gold[:,0]` for each model (Llama/Qwen). An off‑by‑one here would exactly produce the observed flat first‑token CE.

#### Why we believe this will work
- **Multi‑step supervision (K‑token CE)** gives the model a short guided runway so it learns not just which token to start with, but also how to stay on the answer manifold through steps 1–3—precisely where we collapse today.
- **Prefix KD** forces the latent‑prompted distribution at early steps to match the text‑prompted distribution, directly transferring the text baseline’s behavior (our text F1 is good: Llama ≈ 0.80, Qwen ≈ 0.85).
- **Per‑channel affine + tiny MLP** add just enough expressiveness to correct directional/shape mismatches that scalar RMS cannot fix; this is a common failure mode behind “function‑word first token” degeneration.
- **Eval nudges** remove decode‑time headwinds so training gains show up immediately, improving stakeholder confidence while the new losses converge.

#### Expected acceptance signals
- **FirstTok@1** should move from ~3–6% into the teens (Top‑5 into the 30–40% range).
- Degenerate "the/of/and" first tokens largely disappear in the debug print.
- Latent F1/EM increase materially above the token‑budget baseline (currently ~0.04 F1 for Llama), trending toward the text counterpart.

#### Implementation notes (concise)
- **K-step CE under latent prefix (teacher forcing)**
  ```python
  K = 4
  loss_kce = sum(F.cross_entropy(logits_latent[:, t, :], y_gold[:, t]) for t in range(K)) / K
  loss = loss_main + λ_first*first_token_ce + λ_kce*loss_kce    ```

### 2025-09-22 — Stage C eval crash (chat literal)

- **Error:** `UnboundLocalError: local variable 'strip_literal' referenced before assignment` during Stage C evaluation.
- **Cause:** The chat-mode prompt path stripped the `Answer: ` literal and attempted to reattach it before the literal was initialised in the anchor loop.
- **Fix:** Initialise the literal once (from `config.json` or the default) before building `anchor_info`, then reuse it when constructing prompts and anchors. Evaluation now completes and text baselines recover.

### 2025-09-22 — Stage A warm-up & chat-template baseline repair

- **Pipeline update:** `run_scoped_softprompt_multi.sh` now performs a Stage A latent fit (encoder + adapters unfrozen) before the scoped Stage B prefix training, saving the first pass to `ckpt/stageA` and resuming from it with the encoder frozen. This prevents Stage B from starting with random latents.
- **Training sanity:** `_assert_t0_alignment` skips its check when chat templates are active, eliminating false warnings about first-token mismatches under templated prompts.
- **Evaluation fix:** `format_with_chat_template` always routes through the tokenizer’s own chat template and appends `"Answer: "` afterward, so text baselines retain model-specific headers instead of falling back to plain “Assistant:” scaffolds.
- **Post-mortem:** The initial Stage C rerun still showed zero text EM/F1 because we reloaded prefix-tuning adapters *before* computing text baselines. Evaluation now measures text prompts using the raw base checkpoints and only attaches prefix adapters afterwards for latent runs.

### 2025-09-22 — Stage A instability (fix)

- Stage A gradients were spiking into the 500–800 range, starving the latent encoder of real progress. We made clipping the default (`--max_grad_norm=1.0`) in `latentwire/train.py` and reduced the Stage A/Stage B first-token + K-token weights in `scripts/run_scoped_softprompt_multi.sh` to stabilise optimisation. These knobs apply automatically for future runs; setting `--max_grad_norm <= 0` still disables clipping for experiments.
- Stage B now keeps the encoder trainable while prefix-tuning so the warmed-up latent model can continue improving instead of freezing at a random initialisation.
- Enabled a gentle cosine schedule for the first-token CE (peaks capped at 2.5/3.0) and turned on KD for the first K steps in both Stage A and Stage B. This keeps gradients in check while distilling the text baseline into the latent path during smoke runs, giving the latent wire a fighting chance before the hero sweep.
- Stage B now resumes from Stage A weights with `--reset_epoch`, so we reuse the learned latent encoder without inheriting Stage A's epoch counter; each stage now cleanly runs its own four epochs.
- Stage B no longer freezes the encoder; instead we resume from Stage A, reset the epoch counter, drop the first-token peak slightly (2.2), and lower the LR (5e-5) so the encoder and prefix continue to improve together without blowing up gradients.
- Both stages now add light state-KD (`state_kd_weight=0.1`) and use a lower LR (`5e-5`) so the latent prefix is nudged toward the text teacher’s early-layer activations during smoke runs; this should move first-token losses faster and reduce the need for ad-hoc tuning before the hero sweep.
- Default smoke runs now keep Stage A at 4 epochs but extend Stage B to 6 (hero: 6/10), export `TOKENIZERS_PARALLELISM=false`, and disable `use_cache` in eval, which clears the repeated tokenizer/past-key warnings in the logs.
- Stage B now trains on a larger sampled subset (default 1.3k vs 640) while Stage A keeps the smaller 640 batch; the extra data plus longer epoch budget should help the prefix/encoder continue to improve during smoke runs before we scale to hero configurations.
- Stage C now evaluates with a mild prefix gain (`1.1`) to counteract under‑scaling during decode; this will be our default until the latent first-token accuracy stabilises.
- Stage A starts with latent dropout (`keep_start=0.7`) and Stage B starts even lower (`0.5`), annealing to 1.0; combined with state KD it mixes teacher tokens into the latent path early on so first-token learning no longer stalls.
- **Next intervention plan (latent acceptance):**
  1. **Mixed text/latent warm-up.** For the first Stage B epoch alternate batches between text teacher forcing and latent-prefix teacher forcing. This injects clean gold scaffolds at the moment the encoder/adapters are most fragile, which should push first-token top‑1 into double digits and kick latent F1 off the floor.
  2. **Shared + per-model latent slices w/ deeper adapters.** Split `latent_len` into `[shared || llama_private || qwen_private]` (e.g., 32→20/6/6) and upgrade adapters to 2-layer MLPs with residual. This gives each model enough dedicated bandwidth to interpret the shared wire without fighting the other, particularly important because Qwen’s first-token acceptance remains 0%.
  3. **Tiny LoRA fallback.** If the above still leaves latent F1 >10 points behind text, attach r=4 LoRA to the first 4 attention blocks on each LLM. This keeps the story scoped while letting the models learn how to read the latent prefix instead of being purely frozen.
  4. **Parallel Llama/Qwen passes.** Once latent learning is healthy, run both LLM updates concurrently (Accelerate or manual threading) so all four GPUs are busy; that roughly halves turn-around time for smoke sweeps and hero runs.
- **Next steps:** Re-run Stage A→Stage B→Stage C to confirm text EM/F1 recover, then inspect latent metrics with the warmed-up wire.

### 2025-09-25 — Single-model warm-up + runner (today)

- Added optional model selection to `latentwire/train.py` (`--models` now honours `llama`/`qwen` subsets) so we can train a single backend without loading the other 7B checkpoint. Checkpoint loading/saving now adapts to whichever adapters are present.
- Implemented the Stage B text↔latent warm-up (controlled via `--warmup_text_latent_steps` / `--warmup_text_latent_epochs`). When enabled we alternate full-text and latent teacher forcing for the initial steps; logging now tags each batch `L` (latent) or `T` (text) so we can verify the schedule.
- Updated `scripts/run_scoped_softprompt_multi.sh` to enable a one-epoch warm-up during Stage B, and added `scripts/run_llama_single.sh` for the Llama-only pipeline (Stage A/B/C). The new runner defaults to smoke-sized budgets and accepts `--hero` for longer sweeps.
- Known issue: `pytest -q` currently fails on this workstation because Torch cannot locate `libtorch_cpu.dylib` in the host Anaconda env; rerun inside the project venv/conda env before publishing results.
- Fixed a regression spotted in the latest smoke logs where Stage A aborted with an `IndentationError` (`state_blob` block in `latentwire/train.py`). The periodic checkpoint save now has the correct indentation and we only emit the warm-anchor metadata once per checkpoint record.
- Warm-up now includes an explicit embedding-alignment term: during text-mode steps we match the first few gold answer embeddings (default 4 tokens, weight 0.5) against the adapter output. Both `scripts/run_scoped_softprompt_multi.sh` and `scripts/run_llama_single.sh` wire the new `--warmup_align_tokens/--warmup_align_weight` knobs so the gradient actually reaches the encoder/adapters instead of only exercising the frozen teachers.
- Alignment now skips any leading BOS tokens when computing the warm-up loss so single-token answers still contribute signal; the warm-up path also adds a teacher-forced cross-entropy term during text batches and logs those warm-up steps so we can track `align`/`text_tf` in real time. Stage C summary reports “joint” metrics as `n/a` when only one model is active.
- Upgraded the per-model adapters to a residual two-layer MLP and bumped the single-model runner defaults (`adapter_hidden_mult=4`, `adapter_dropout=0.1`, `latent_private_len=12`). Warm-up now runs for two epochs with stronger alignment/teacher weights (`warmup_text_latent_epochs=2`, `warmup_align_weight=1.0`, `warmup_text_teacher_weight=1.5`) so the adapter can actually absorb the teacher signal before we switch to latent-only batches.
- Default device maps in `run_llama_single.sh` and the multi-model runner stay on HuggingFace's `auto` setting; to encourage a more even split across the listed GPUs set `GPU_MEM_GIB` (e.g., `GPU_MEM_GIB=60`) before launching or override `LLAMA_DEVICE_MAP`/`QWEN_DEVICE_MAP` manually.
- Evaluation now respects the active model subset when loading the encoder (fixes STQuery checkpoints produced with private latent slices for single-model runs).
