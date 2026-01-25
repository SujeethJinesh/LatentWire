# Implementation Plan + Experiment Matrix (1x H100 per milestone)

Goal: deliver a workshop‑ready paper on **Quantized Cache‑to‑Cache** with communication‑budget curves, and a clear path to a main‑conference submission (QAT + mixed precision + heterogeneity scaling). Each milestone is designed to run on a single H100.

## Data Capture Contract (applies to every milestone)
- Each run creates `quantization/data/step_X_<name>/<run_tag>/` with (folder names remain `step_X` for now to match scripts):
  - `configs/` (the exact eval/train configs used)
  - `logs/` (stdout/stderr logs)
  - `results/` (JSON outputs)
  - `manifests/` (checkpoint + environment provenance)
- Commit `quantization/data/step_*` to git after each milestone so results are reviewable.
- Cleanup hygiene: remove failed runs that do not contain `results/` or have `status=failed` in manifests (keep only validated runs).

## C2C Fork Strategy (for Milestone 1+ changes)
**Decision**  
Fork C2C and point the submodule at the fork. Keep all quantization changes on a dedicated branch (e.g., `quant-kv`).

**Why**  
This is the cleanest path for reproducibility and publication: exact diffs are visible, and upstream sync is straightforward.

**How**  
- Create fork on GitHub, add `upstream` remote for the original repo.  
- Update submodule remote to the fork and pin to the branch.  
- Keep changes minimal and well‑scoped (quantization utilities + hook points only).

## Milestone 0: Baseline + Environment Sanity
**What**  
Run the official C2C baselines on OpenBookQA and ARC‑C with the released 0.6B+0.5B fuser.

**Why**  
Establish a trustworthy baseline before introducing quantization or cache‑budget changes.

**How**  
Use `quantization/scripts/run_step0_baselines.py` (supports `--prep-only` on login nodes).  
The script:
- Ensures the C2C submodule is initialized.
- Verifies the `rosetta` conda env and installs deps if missing.
- Downloads the published fuser checkpoint to scratch.
- Writes `env_info.json` and checkpoint manifests.
- Runs OpenBookQA + ARC‑C and logs everything into `quantization/data/step_0_baselines/<run_tag>/`.

**Commands (from repo root)**  
- Login node prep: `python quantization/scripts/run_step0_baselines.py --prep-only`  
- GPU node eval: `python quantization/scripts/run_step0_baselines.py`

**Workshop/Main‑conf connection**  
Baseline accuracy and latency anchors the whole paper.

## Milestone 1: Implement KV PTQ Utilities
**What**  
Add post‑training quantization (INT8, INT4/NF4, optional FP8) for KV caches.

**Why**  
Quantization is not covered in C2C; it is the core novelty for the workshop paper.

**How**  
Add a small quantization module (e.g., `rosetta/utils/quant.py`) and hook it before projection in `rosetta/model/wrapper.py`.  
Expose a `kv_quant_config` block (per‑head / per‑layer) under `model.rosetta_config`:  
```yaml
kv_quant_config:
  enabled: true
  scheme: int8   # int8 | int4
  axis: head     # head | layer
  eps: 1e-6
  collect_stats: false
```  
Follow the Milestone 0 pattern with a minimal, well‑scoped runner:
- `quantization/scripts/run_step1_kv_ptq.py` with GPU as default; `--mode local` for Mac validation.  
- Local smoke test: single prompt with the same C2C eval template + `extract_answer_from_content` check.  
- GPU eval: identical to Step 0, but with quantization flags enabled.  

**Workshop/Main‑conf connection**  
PTQ results + bandwidth reductions are sufficient for a workshop paper.

**Milestone 1 sub‑steps (phased)**  
- **Phase 0 (Plan update)**: Document decisions, sub‑steps, and extensions (this section).  
- **Phase 1 (Fork changes)**: Add `rosetta/utils/quant.py` + a single hook in `rosetta/model/wrapper.py` for source‑KV quantization.  
- **Phase 2 (Runner)**: Add `quantization/scripts/run_step1_kv_ptq.py` mirroring Step 0, with GPU default and local dry‑run support.  
- **Phase 3 (Local validation)**: Run a **single dataset sample** (OpenBookQA or ARC‑C) to validate prompt formatting, quantization path, and logging.  
- **Phase 4 (GPU eval)**: Run full OpenBookQA + ARC‑C with quantization enabled; compare accuracy vs Step 0.  

**Milestone 1 Phase 1 implementation details (applied)**  
- **Quantization module**: `rosetta/utils/quant.py` implements symmetric fake‑quant for KV (INT8/INT4), per‑head or per‑layer.  
- **Hook point**: `rosetta/model/wrapper.py` quantizes **source KV slices** immediately before projection.  
- **Config plumbing**: `rosetta/utils/evaluate.py` passes `kv_quant_config` into `RosettaModel`.  
- **Design choice**: fake‑quant only (quantize → dequantize), no bit‑packing yet. This preserves execution path while modeling quantization noise.
- **Interpretability**: optional `collect_stats` aggregates min/max/mean scale stats per run (low overhead when enabled).

**Decisions + justification**  
- **Start with INT8 + INT4 only**: These are the most standard, hardware‑agnostic PTQ baselines. They deliver a clear accuracy/bytes trade‑off and are sufficient for a workshop paper.  
  - **Extensions (later)**: NF4 and FP8. NF4 can improve INT4 accuracy; FP8 is H100‑friendly and could strengthen a main‑conf submission.  
- **Quantize source KV only**: This models the actual communication channel (teacher → base) and isolates the quantization effect to the transmitted cache.  
  - **Extensions (later)**: quantize base KV, quantize both source+base, or quantize only a subset of layers.  
- **Default per‑head axis**: Per‑head scales typically preserve accuracy better than per‑layer with modest overhead.  
  - **Extensions (later)**: per‑layer (cheaper), per‑token (more accurate but expensive), or mixed granularity by layer depth.  

**Extensions to Milestone 1 (backburner)**  
- **NF4 / FP8**: add additional schemes for improved INT4 accuracy or H100‑optimized precision.  
- **Bit‑packing**: actual KV compression for realistic bandwidth measurements.  
- **Source+base quantization**: quantify whether quantization noise on both sides compounds or cancels.  
- **Layer‑selective quantization**: higher precision in late layers, lower in early layers.  
- **Group‑wise quantization**: group size tuning for accuracy/overhead trade‑offs.  
- **Per‑run scale histograms**: richer quantization diagnostics beyond min/max/mean.  

**Technical notes / acronyms (Milestone 1)**  
- **KV cache**: Attention Keys/Values stored for past tokens to avoid recomputation.  
- **C2C**: Cache‑to‑Cache projection from a source model’s KV to a target model’s KV.  
- **PTQ**: Post‑Training Quantization; quantize activations without retraining.  
- **INT8/INT4**: 8‑bit / 4‑bit integer quantization (smaller = less bandwidth).  
- **NF4**: Normalized 4‑bit quantization (non‑uniform 4‑bit, often higher accuracy than plain INT4).  
- **FP8**: 8‑bit floating point (hardware‑dependent, often H100‑friendly).  
- **Per‑head vs per‑layer**: granularity of scales/zero‑points; per‑head is higher accuracy, per‑layer is cheaper.  

## Milestone 2: PTQ Evaluation
**What**  
Evaluate INT8 and INT4/NF4 on OpenBookQA + ARC‑C.

**Why**  
Quantifies accuracy drop vs bandwidth savings.

**How**  
Run the same eval pipeline as Step 0 with quantization flags enabled.  
Store results under `quantization/data/step_2_ptq/<run_tag>/`.

## Milestone 3: Cache‑Length Reduction
**What**  
Prune KV tokens (e.g., keep top‑50%, 25%, 10%) before projection.

**Why**  
Adds a second “budget” axis (length) beyond precision, enabling stronger trade‑off curves.

**How**  
Implement a simple selection policy (magnitude‑based or attention‑norm) and evaluate with INT8.

## Milestone 4: Communication‑Budget Curves
**What**  
Report accuracy vs transmitted bytes (precision × KV length).

**Why**  
This reframes C2C under a realistic communication budget, a key paper contribution.

**How**  
Log bytes in evaluation outputs and produce a single “accuracy vs bytes” plot.

## Milestone 5: QAT Recovery (Main‑conf extension)
**What**  
Quantization‑aware training of the projector under INT8 noise.

**Why**  
Shows that low‑precision transfer can be learned, not just approximated.

**How**  
Train on a small subset (10–50k samples), then re‑evaluate ARC‑C + OpenBookQA.

## Milestone 6: Mixed Precision by Layer (Main‑conf extension)
**What**  
Use higher precision in later layers and lower precision in early layers.

**Why**  
Leverages layer sensitivity to improve the accuracy‑per‑byte curve.

**How**  
Add per‑layer precision configs; evaluate a small grid (e.g., last 4 layers FP16).

## Milestone 7: Heterogeneity Scaling (Main‑conf extension)
**What**  
Test at least one cross‑family pair (e.g., Qwen3 ← Llama3.2).

**Why**  
Demonstrates that the method generalizes beyond a single model family.

**How**  
Reuse the PTQ + pruning settings from Steps 2–3.

---

# Experiment Matrix (Merged)

## A. Baselines (Milestone 0)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| A1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C | FP16 | full | no |
| A2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C | FP16 | full | no |

## B. PTQ (Milestones 1–2)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| B1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + PTQ | INT8 | full | no |
| B2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | full | no |
| B3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + PTQ | INT4/NF4 | full | no |
| B4 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT4/NF4 | full | no |

## C. Cache‑Length Reduction (Milestone 3)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| C1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 50% | no |
| C2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 25% | no |
| C3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 10% | no |

## D. QAT / Mixed Precision (Milestones 5–6)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| D1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + QAT | INT8 | full | yes |
| D2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + QAT | INT8 | full | yes |
| D3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | Mixed precision | FP16/INT8/INT4 | full | yes |

## E. Heterogeneity (Milestone 7)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| E1 | Qwen3‑0.6B ← Llama3.2‑1B | ARC‑C | C2C + PTQ | INT8 | full | no |
| E2 | Qwen3‑0.6B ← Gemma‑1B | ARC‑C | C2C + PTQ | INT8 | full | no |

---

## Workshop vs Main‑Conference Path
- **Workshop**: Milestones 0 → 1 → 2 → 3 → 4 (baseline + PTQ + pruning + budget curve).
- **Main‑conf**: Workshop path + Milestones 5 → 6 → 7 (QAT + mixed precision + heterogeneity).

## Potential Improvements / Follow‑ups
- **Byte accounting**: include per‑head scale metadata in “bytes transferred” so INT8/INT4 are not under‑counted.
- **Milestone 3 simplification**: use `proportion/order_mode` in `kv_cache_index` for cheap cache‑length ablations before adding top‑k pruning.
- **Budget curves**: report accuracy vs *effective* bytes (precision × cache length + metadata).
- **QAT scope**: keep projector‑only training (freeze base/teacher weights) to stay within 1×H100 budget.
- **Mixed precision**: start with “last‑N layers FP16, rest INT8” as the minimal schedule.
- **Heterogeneity**: plan for tokenizer alignment (`is_do_alignment`) when crossing model families.

## Time Budget (rough)
- Workshop path: ~1 day on 1 H100 (assuming caches are warm).
- Main‑conf path: ~3–5 days on 1 H100.
