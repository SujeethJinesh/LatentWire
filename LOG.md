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