# 02 — SinkAware: Sink-as-Static-Prior Attention Kernel

## TL;DR
Attention sinks at positions 0–3 absorb a large fraction of attention mass nearly regardless of content. Existing kernels still compute scores against sink tokens and let softmax allocate the mass. We build the first attention kernel that treats sinks as a *statically precomputed prior* (closed-form bias) and skips the score computation entirely.

## Hypothesis (testable)
At long context (≥32K), score computation against sink positions accounts for ≥10% of attention kernel time on Qwen3.6-27B and GPT-OSS-20B. Treating sinks as a precomputed prior recovers ≥80% of that overhead with no quality loss on long-context benchmarks (RULER, Needle, PG-19).

## Why this is novel
- **BLASST (Dec 2025)** does *dynamic* threshold-based block-skipping, not static sink handling.
- **Block-Sparse Flash Attention (Dec 2025)** has generic sparse patterns, no sink specialization.
- **EARN (July 2025)** uses dual sinks at architectural level (recommendation systems), not as a kernel optimization.
- **OrthoRank** uses sink orthogonality for token *selection*, not kernel-level fusion.
- **StreamingLLM** introduced sinks but kept them as runtime-computed mask entries.
- **DeepSeek Sparse Attention (DSA, V3.2-Exp Sept 2025 / V4 April 2026)** does *dynamic content-based* top-k via lightning indexer; static sink prior is orthogonal and complementary.
- **NSA (Native Sparse Attention, Feb 2025)** does block-wise sparsity, also dynamic.
- **GPT-OSS-20B (Aug 2025)** uses "alternating dense and locally banded sparse attention patterns" — banded, not sink-aware.
- **No published kernel** treats sinks as a closed-form static bias.

## Competitive landscape note (May 2026)
Production sparse attention is now standard (DSA in V3.2/V4, GLM-5.1; banded sparse in GPT-OSS). The "static sink prior" wedge is *narrower* than 6 months ago but still distinct: every dynamic scheme above re-decides per-query which tokens to attend to. Static sink prior precomputes the contribution from *fixed* positions whose membership doesn't depend on the query. Phase 1 must explicitly tabulate this distinction.

## Why it fits the lab
Sinks are a structural outlier phenomenon — direct fit with the lab's outlier and quantization research. Pure GPU kernel artifact. MLSys / ICLR-shaped.

## Risk profile
**Medium–high.** Field is hot. The exact "sink-as-static-prior" framing is unpublished but the territory is being mapped fast. **Phase 1 audit is the most important gate of the three projects** — if the idea is hiding in FlashInfer's mask handling, we need to know in week 1.

---

## Folder
`experimental/sinkaware/`

---

## Phase 0: Setup (½ day, Macbook, $0)

- [ ] Create `colm-sink` env per `00_setup.md`
- [ ] Clone repos: BLASST, Block-Sparse-Flash-Attention, FlashInfer, StreamingLLM, EARN (if released), OrthoRank, SinkTrack, DeepSeek-V3.2-Exp DSA kernels, FlashMLA
- [ ] Pull configs for:
  - `Qwen/Qwen3.6-27B` (April 2026 dense, GQA, primary target — 256K context, fits 5090 in BF16 with KV pressure)
  - `openai/gpt-oss-20b` (16GB MXFP4, GQA-8, **alternating dense/banded-sparse pattern** — interesting comparison)
  - `Qwen/Qwen3-8B` (smaller dense reference for ablations)
  - `Qwen/Qwen3-14B` (mid-size reference)

**Deliverable**: `phase0/setup_complete.md`.

---

## Phase 1: Critical literature + codebase audit (2–3 days, Macbook, $0)

**This is the highest-value gate in the entire sprint.** The goal is to determine whether any existing kernel already handles sinks the way we propose. Read code, not just papers.

For each, find the lines that handle position 0–3 (or the first N tokens) and document what the kernel does with them.

- [ ] **BLASST** (Dec 2025): how does the dynamic threshold treat early positions?
- [ ] **Block-Sparse Flash Attention** (Dec 2025): does the generic sparse pattern API allow precomputed sink handling?
- [ ] **FlashInfer**: search for "sink", "bos", "first_token", static mask handling, prefill/decode paths
- [ ] **DeepSeek Sparse Attention (DSA)**: check V3.2-Exp + V4 kernels — does the lightning indexer special-case early positions?
- [ ] **DeepSeek V4 c4a/c128a token-wise compressed attention**: does compression treatment of early tokens differ from later ones?
- [ ] **NSA (Native Sparse Attention)**: block-wise sparsity, check sink block handling
- [ ] **FlashMLA**: MLA kernel used by V3.2/V4
- [ ] **EARN**: dual-sink architecture — is there a kernel?
- [ ] **OrthoRank**: how does it handle sinks?
- [ ] **SinkTrack**: algorithm-only, but check for kernel companion
- [ ] **StreamingLLM** original kernel impls + community forks
- [ ] **FlashAttention 3** code: attention-sink optional path
- [ ] **GPT-OSS-20B reference impl**: how does its alternating dense/banded-sparse handle position 0–3?
- [ ] **Triton tutorials and kernels**: any reference to sink fusion
- [ ] Arxiv search: "attention sink kernel", "static sink fusion", "precomputed attention bias", "BOS token kernel"

**Deliverable**: `phase1/lit_review.md` with:
- Per-kernel: file path + line numbers + what they do at sink positions
- Comparison table: dynamic vs static; runtime-computed vs precomputed; mask-based vs bias-based
- Conclusion paragraph: precise statement of what our kernel does that no existing kernel does

### KILL CRITERION
Any existing kernel implements sink-as-precomputed-prior in the form: `output += sink_bias_precomputed; for non_sink_tokens: compute_scores(); softmax_with_bias()`. If found: project killed.

### PIVOT TRIGGERS
- Found related-but-different angle (e.g., sink + register fused, sink as KV slot rather than score, per-head adaptive sink count): pivot framing accordingly.
- Found that sink handling is buried in a generic mask system but not optimized as a static path: confirms the wedge — proceed.

---

## Phase 2: Sink-prior math + theory (1 day, Macbook, $0)

- [ ] Derive the sink-as-precomputed-prior formulation: given fixed sink positions S and per-head sink mass distribution, compute the contribution to output that depends only on V[S] and a precomputed weight, independent of Q
- [ ] Identify when this is exact vs lossy: when does sink mass depend only on Q's *norm* vs Q's *direction*?
- [ ] Show conditions for numerical equivalence to standard attention with mask
- [ ] Identify the approximation error bound

**Deliverable**: `phase2/theory.md` (~3–4 pages, math notation OK), `phase2/notation.tex`.

### KILL CRITERION
If the math shows the sink contribution depends meaningfully on Q in a way that prevents precomputation: kill, the static-prior framing is wrong. Pivot would be necessary.

---

## Phase 3: Sink mass empirical validation (1 day, Macbook, $0)

- [ ] Pull cached attention statistics from prior work (StreamingLLM release, SinkTrack stats, public attention dumps)
- [ ] If unavailable: run Qwen3-1.7B or Qwen3-4B on Mac MPS, dump attention scores at multiple contexts, save
- [ ] Quantify sink mass at positions 0–3 across heads, layers, models, contexts
- [ ] Cross-check on GPT-OSS-20B traces (its banded-sparse pattern may already strongly weight early tokens)
- [ ] Validate: is sink mass predictable enough to precompute? Variance across queries?

**Deliverable**: `phase3/sink_statistics.ipynb` with plots showing:
- Sink mass distribution per layer / head
- Variance of sink mass across queries (is it actually static?)
- Per-model differences

### Pivot trigger
If sink mass variance across queries is high (>20% relative): the static-prior approach is lossy enough to hurt quality. Reframe as "low-rank sink prior" or kill.

---

## Phase 4: Reference + Triton interpreter skeleton (2 days, Macbook, $0)

- [ ] NumPy reference: sink-prior attention vs standard attention, equivalence (or bounded approximation) test on synthetic data
- [ ] Triton kernel skeleton: precomputed-bias path + branch-free non-sink path
- [ ] Run `TRITON_INTERPRET=1` correctness tests against the CPU reference
- [ ] If local Triton is unavailable, keep CPU tests passing and mark Phase 4 blocked, not complete
- [ ] Integration plan with vLLM's FlashInfer backend

**Deliverable**: `phase4/reference/sink_attention.py`, `phase4/kernel/sink_triton.py`, `phase4/integration_plan.md`.

### KILL CRITERION
If the Triton skeleton requires more kernel launches than baseline FlashAttention to handle the sink path cleanly: the fusion is not net positive. Kill.

---

## ── GATE TO 5090 ──

**All must hold:**
- [ ] Phase 1: explicit confirmation no existing kernel implements sink-as-static-prior
- [ ] Phase 2: math shows static prior is exact or has bounded error
- [ ] Phase 3: empirical sink mass variance is low enough to support static treatment
- [ ] Phase 4: reference passes equivalence test, Triton interpreter skeleton matches the CPU reference or is explicitly blocked by missing local Triton

**Sign-off**: human review.

---

## Phase 5: 5090 baseline (½ day, ~$5)

- [ ] FlashAttention 3 baseline on Qwen3.6-27B at 32K, 64K context (BF16; KV pressure tight on 32GB)
- [ ] FA3 baseline on GPT-OSS-20B (MXFP4) at 32K, 64K
- [ ] BLASST baseline on Qwen3.6-27B at same configs
- [ ] DSA reference on a model that supports it (V3.2-Exp small variant if available; otherwise a custom impl)
- [ ] Measure: ms in attention kernel, fraction attributable to score computation at sink positions
- [ ] Use NCU (Nsight Compute) for fine-grained kernel analysis

**Deliverable**: `phase5/baseline_profile.md`.

### PIVOT TEST
- ≥10% of attention kernel time at sink positions → **continue**
- 5–10% → **scope down**: target 64K+ context where it's bigger
- <5% → **kill**

---

## Phase 6: Prototype kernel (4–5 days, ~$60)

- [ ] Implement sink-prior attention kernel in Triton on the 5090
- [ ] Numerical correctness vs FlashAttention 3 (fp16/bf16 tolerance, end-to-end model output match)
- [ ] Initial perf characterization at 32K, 64K context on Qwen3.6-27B and GPT-OSS-20B (128K if memory allows)

**Deliverable**: `phase6/kernel/sink_attn.py`, `phase6/correctness.md`, `phase6/perf.md`.

### Continue criteria
- Correctness: model output cosine sim ≥0.999 vs FA3 on a held-out set of 100 prompts
- Perf: ≥1.2× wall-clock over BLASST at iso-quality

---

## Phase 7: Optimization + evaluation (1 week, ~$250)

- [ ] Optimize: precomputed bias caching, branch-free non-sink path, vectorized loads
- [ ] Quality preservation eval:
  - PG-19 perplexity at long context
  - Needle-in-Haystack at 32K, 64K
  - RULER at 32K, 64K
  - LongBench
- [ ] Comparison sweep: vs FlashAttention 3, vs BLASST, vs Block-Sparse-FA, vs StreamingLLM kernel, vs DSA (where applicable), vs GPT-OSS's native banded-sparse
- [ ] Ablations: sink count (1 vs 4 vs 8), per-head adaptive, register-token combination

**Deliverable**: `phase7/results/` with all benchmark CSVs, plots, ablation tables.

---

## Phase 8: Paper (1 week, mostly Macbook + ~$80 H100 burst)

- [ ] COLM workshop paper draft (4–8 pages)
- [ ] H100 burst for datacenter numbers (~$60–80)
- [ ] Polish, internal review, submit

---

## Total estimated cost
- Phase 5: $5
- Phase 6: $60
- Phase 7: $250
- Phase 8: $80
- **Total: ~$395**

## Risk factors and mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Existing kernel does this | Medium-high | Phase 1 codebase audit |
| Sink mass too variable for static prior | Medium | Phase 3 empirical validation |
| BLASST already gives most of the speedup | Medium | Phase 5 baseline + ablations |
| Quality degrades at extreme contexts | Low–medium | Phase 7 quality eval |
| Scooped during the 7 weeks | Medium | Be fast; aim COLM-then-extend |

## Follow-up paper (MLSys/ICLR Sept–Oct)
Extension: per-head adaptive sink count, sink+register joint fusion, training-time sink prior (learned), evaluation on Qwen3.6-27B / GPT-OSS-120B / Llama 4 / DeepSeek-V4-Flash. The COLM submission carves the niche; the ICLR paper closes it. Bonus angle: combining sink-prior with DSA's lightning indexer to see whether the static prior reduces the indexer's search space.
