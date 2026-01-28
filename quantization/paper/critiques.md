# ICLR-Style Reviews (3-reviewer committee)

Paper: **Quantized Cache-to-Cache: Communication-Budgeted KV Transfer for Heterogeneous LLMs**

Date: 2026-01-28

---

## Reviewer 1 (Score: **6 / 10** — Weak Accept)

**Summary**
The paper reframes Cache-to-Cache (C2C) under a communication-budget lens and shows that PTQ (INT8/INT4) is nearly lossless and that front/back cache truncation yields asymmetric behavior. The work is practical and timely, but the current submission is incomplete: the main novelty (token-level selective transfer) is only partially evaluated, and “heterogeneous” claims are not backed by cross-family results.

**Strengths**
- Clear motivation: practical bandwidth limits for cross-LLM communication.
- Clean baseline story: INT8/INT4 PTQ is nearly lossless, cache-length pruning shows strong asymmetry.
- Solid empirical pipeline with reproducible artifacts.

**Weaknesses / Concerns**
- **Heterogeneity claim not substantiated**: results are within-family (Qwen↔Qwen) only. Cross-family (Llama/Gemma) results are missing.
- **Main novelty incomplete**: selective token transfer (SparseC2C) is only partially evaluated (INT4 subset and single INT8 run). This is insufficient for ICLR main track.
- **No system measurements**: the paper lacks measured bandwidth/latency/fuser FLOPs, only analytical bytes. Reviewers will ask for real speed/throughput numbers.
- **Limited datasets (2)**: OpenBookQA + ARC-C only. The paper should justify this choice or add more tasks.
- **Comparisons** to KVComm/Q-KVComm or KV cache compression work are mostly narrative; no empirical comparison.

**Questions**
- How do results change for true heterogeneous pairs (Llama/Gemma)?
- Is the apparent INT4 robustness an artifact of evaluation settings (prompt template, answer extraction)?
- What is the actual throughput/bandwidth gain (in wall-clock or bytes) versus the uncompressed baseline?

**Recommendation**
Accept if the final version includes heterogeneity results and a clearer system evaluation (even a small timing study). Otherwise, this is borderline.

---

## Reviewer 2 (Score: **4 / 10** — Weak Reject)

**Summary**
This paper presents quantized C2C and cache-length pruning. While practical, it currently reads as an incremental evaluation paper rather than a strong method paper. The claimed main-track contributions (selective token transfer, mixed precision, QAT) are either incomplete or not demonstrated.

**Strengths**
- Good experimental engineering and reproducibility.
- The cache-length asymmetry (front vs back) is a useful observation.

**Weaknesses / Concerns**
- **Novelty risk**: At the method level, the paper’s contributions overlap with KVComm/Q-KVComm and KV compression literature. Without a full SparseC2C evaluation, it’s unclear what is truly new.
- **Evidence gap**: Main conference contributions (QAT recovery, mixed precision, heterogeneity, SparseC2C) are described but not actually shown. This is closer to a workshop report.
- **Insufficient ablations**: no error bars, no receiver-only baseline comparison, no calibration against C2C paper’s exact settings.
- **Presentation issues**: bibliographic entries are incomplete for some citations, and some numerical claims appear to be partial.

**Questions**
- Why do INT4 results sometimes match or exceed FP16? Is this noise or extraction bias?
- Are there repeated runs to assess stability?

**Recommendation**
Reject unless the paper demonstrates the full method (SparseC2C) and heterogeneity results, plus a system-level analysis.

---

## Reviewer 3 (Score: **5 / 10** — Borderline)

**Summary**
The paper is promising and relevant, and the initial results are encouraging. However, the submission is missing key experiments that are necessary to justify the claims (heterogeneous models and selective token transfer). The methodology section could also be more rigorous about quantization details and byte accounting.

**Strengths**
- Concrete evaluation pipeline with publicly reproducible artifacts.
- Clear framing of communication budget as the core axis.

**Weaknesses / Concerns**
- **Incomplete methodology**: no details on per-head/per-layer quantization or how scales/metadata are accounted in bytes.
- **Missing heterogeneity**: the “heterogeneous” title doesn’t match the current results.
- **SparseC2C lacks full evaluation**: the strongest idea is under-tested.
- **Limited scope**: only two datasets, no latency results.

**Questions**
- Can the method be shown to generalize beyond OpenBookQA/ARC-C?
- How sensitive are results to template / answer extraction?

**Recommendation**
Borderline; needs stronger completeness and more convincing comparisons.

---

# Cross-review Summary

**Main blockers for ICLR:**
1) Missing **true heterogeneity** results (Llama/Gemma).
2) Incomplete **SparseC2C** evaluation (INT8 grid, token selection variants).
3) **No system measurements** (latency/bandwidth beyond analytic bytes).
4) Limited datasets and lack of variance / stability analysis.

---

# Actionable Fix List (mapped to experiments)

1) **Run heterogeneity (M7)** with Llama + Gemma and alignment on/off.
2) **Complete M8 INT8 grid** (front/vnorm/proj/random/knorm × p={1.0,0.5,0.25,0.10}).
3) **Add system-level metrics** (end-to-end wall time and bytes including index/scale overhead; optionally fuser FLOPs).
4) **Add stability checks**: repeat a subset (2–3 runs) or use different random seeds.
5) **Expand datasets** or justify why these two are sufficient (and align to C2C’s evaluation).

---

# Scores (Summary)
- Reviewer 1: **6 / 10** (Weak Accept)
- Reviewer 2: **4 / 10** (Weak Reject)
- Reviewer 3: **5 / 10** (Borderline)

Average: **5.0 / 10**

---

# Notes on Paper Quality (line-level themes)
- **Abstract:** Clear but claims main-track extensions not yet demonstrated.
- **Introduction:** Good motivation; needs explicit statement of what is already completed vs in-progress.
- **Method:** Quantization details underspecified (symmetric? per-head? scale overhead?).
- **Experiments:** Missing heterogeneity; tables should clarify sample counts, standard deviations, and whether results are partial.
- **Related Work:** Should explicitly contrast with KVComm/Q-KVComm and KV compression in terms of *token vs layer selection* and *cross-model projection*.

