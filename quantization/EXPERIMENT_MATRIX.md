# C2C Quantization + Communication-Budget Experiments

Goal: produce a strong workshop paper with a clear path to a main-conference submission, using 1–2 H100s.

This plan assumes you will start from the official C2C repo at `quantization/C2C` and keeps training minimal
(PTQ-only where possible). It explicitly avoids overlap with existing C2C contributions (gated fusion, ablations,
layer mapping, multi-sharer support, long-context evaluation, etc.).

---

## What C2C Already Covers (Do Not Duplicate)

- KV-cache projection + fusion with gating and learned weights (`rosetta/model/projector.py`, `rosetta/model/ablation_projector.py`).
- Token alignment across different tokenizers (`rosetta/model/aligner.py`).
- Layer mapping strategies (`rosetta/train/model_utils.py`).
- Multi-sharer fusion modes (parallel/sequential) (`rosetta/model/wrapper.py`).
- Oracle experiments + ablations + scaling/venn/consistency analyses (`script/analysis/*`, `script/train/oracle_train*.py`).
- Unified evaluation across OpenBookQA, ARC-C, MMLU-Redux, C-Eval, LongBench, etc. (`script/evaluation/unified_evaluator.py`).

Notably missing: any quantization of KV caches, explicit bandwidth accounting, cache pruning/length reduction, or
precision-aware projection training (QAT).

---

## Proposed Contributions (Non-overlapping)

### Core Workshop Contribution (Low Compute)
1) **KV-cache PTQ for C2C**: quantize K/V caches and report accuracy vs bytes.
2) **Cache-length reduction**: keep only top-k KV tokens (or a fixed ratio) per layer/head to reduce bandwidth.
3) **Communication-budget curves**: accuracy vs bytes (cache length x precision) for C2C vs text relay.

### Main-Conference-Ready Extension (Moderate Compute)
4) **Precision-aware training (QAT)**: train projector with quantization noise to recover accuracy.
5) **Selective quantization**: per-layer / per-head mixed precision (e.g., last layers FP16, mid layers INT8, early layers INT4).
6) **Heterogeneity scaling**: show gains persist across model families and sizes (Qwen, Llama, Gemma).

---

## Experimental Matrix (Minimal but Publishable)

### A. Baselines (Run First)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train | Notes |
|----|------------|---------|--------|-----------|--------------|-------|-------|
| A1 | Qwen3-0.6B ← Qwen2.5-0.5B | OpenBookQA | C2C (FP16) | FP16 | full | no | Reproduce official baseline |
| A2 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C (FP16) | FP16 | full | no | Main comparison point |

### B. PTQ Quantization (Workshop Core)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train | Notes |
|----|------------|---------|--------|-----------|--------------|-------|-------|
| B1 | Qwen3-0.6B ← Qwen2.5-0.5B | OpenBookQA | C2C + PTQ | INT8 (per-head) | full | no | Quantize source KV before projection |
| B2 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | INT8 (per-head) | full | no | Same as B1 |
| B3 | Qwen3-0.6B ← Qwen2.5-0.5B | OpenBookQA | C2C + PTQ | INT4/NF4 | full | no | Blockwise quantization |
| B4 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | INT4/NF4 | full | no | Blockwise quantization |
| B5 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | FP8 | full | no | Optional if tooling is easy |

### C. Cache-Length Reduction (Workshop Core)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train | Notes |
|----|------------|---------|--------|-----------|--------------|-------|-------|
| C1 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | INT8 | top-50% KV tokens | no | Token selection per head |
| C2 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | INT8 | top-25% KV tokens | no | Aggressive pruning |
| C3 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + PTQ | INT8 | top-10% KV tokens | no | Stress test |

### D. QAT / Mixed Precision (Main-Conf Extension)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train | Notes |
|----|------------|---------|--------|-----------|--------------|-------|-------|
| D1 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + QAT | INT8 | full | yes (10–50k) | Projector only |
| D2 | Qwen3-0.6B ← Qwen2.5-0.5B | OpenBookQA | C2C + QAT | INT8 | full | yes (10–50k) | Compare to PTQ |
| D3 | Qwen3-0.6B ← Qwen2.5-0.5B | ARC-C | C2C + QAT | mixed (INT4/8/FP16) | full | yes | Layer-wise precision |

### E. Heterogeneity (Main-Conf Extension)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train | Notes |
|----|------------|---------|--------|-----------|--------------|-------|-------|
| E1 | Qwen3-0.6B ← Llama3.2-1B | ARC-C | C2C + PTQ | INT8 | full | no | Cross-family robustness |
| E2 | Qwen3-0.6B ← Gemma3-1B | ARC-C | C2C + PTQ | INT8 | full | no | Cross-family robustness |

---

## Metrics to Report (for all experiments)

- Accuracy (primary) on OpenBookQA + ARC-C.
- Latency + throughput (ms/sample, samples/sec).
- Bytes transmitted = (KV tokens * head_dim * heads * 2 * precision bytes).
- Quality vs bandwidth curves (accuracy vs bytes).
- Optional: layer-wise sensitivity plots (which layers tolerate INT4/INT8).

---

## Where to Implement (Repo Map)

- Quantization hooks: `rosetta/model/wrapper.py` (right before/after projector calls in `forward()`).
- Utility functions: add `rosetta/utils/quant.py` for quantize/dequant + calibration.
- Config flags: update `recipe/train_recipe/*.json` and `recipe/eval_recipe/*.yaml`.
- Bandwidth logging: extend `script/evaluation/unified_evaluator.py` output metrics.

---

## Execution Steps (Order)

1) **Reproduce baseline** (A1–A2) with official configs. Verify expected accuracy.
2) **Add PTQ utilities** (INT8 + INT4/NF4), run B1–B4.
3) **Add cache-length reduction** (C1–C3) and report accuracy vs bytes.
4) Draft workshop paper with PTQ + pruning + budget curves.
5) If time: **QAT + mixed precision** (D1–D3) for main-conference quality.
6) Optional heterogeneity scaling (E1–E2) if extra budget.

---

## Training Budget Estimates (1–2 H100s)

- Baseline eval (A1–A2): 1–2 hours total.
- PTQ + pruning (B/C): 2–6 hours total (mostly eval).
- QAT (D1–D3): 3–12 hours total (projector-only training).
- Heterogeneity (E1–E2): 2–4 hours total (eval).

Total workshop track: **~1 day** of GPU time.
Total main-conference track: **~2–4 days** of GPU time.

---

## Notes on Novelty

This plan focuses on **precision-aware C2C** and **communication-budget evaluation**, which are not in the current
C2C repo. It avoids duplicating their gate/ablation/multi-sharer work and instead answers:

- How low can we push KV precision before accuracy collapses?
- How many KV tokens do we actually need if we optimize the budget?
- Can QAT recover accuracy at low precision?

These questions directly target deployment constraints (memory + bandwidth) and are strong enough for a workshop
paper, with a clear path to a main-conference submission if QAT + mixed precision shows gains.
