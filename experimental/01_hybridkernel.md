# 01 — HybridKernel: Fused Attention↔SSM Boundary Kernel

> **Current 2026-05-07 authority.** This is a historical project brief. The
> live HybridKernel gate is now the native vLLM/Nsight profiler packet described
> in `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md` and
> `experimental/native_gpu_handoff_20260506.md`. Do not implement a fused
> kernel, Phase 6 prototype, or Phase 7 benchmark suite until that profiler
> packet clears the preregistered `>=3%` recoverable-gain gate with primary,
> same-family, and cross-family control rows.

## TL;DR
Hybrid LLMs interleave attention layers and SSM layers. Current targets (May 2026): Granite-4.0-H (9 Mamba : 1 attention, Oct 2025), Nemotron-H (8B/47B/56B), Nemotron-3-Nano-30B-A3B (Dec 2025, hybrid MoE with reasoning budget), Mamba-3 (March 2026), Apriel-H1-15B-Thinker (hybrid reasoner), Qwen3-Next-80B-A3B. The transitions may involve memory format conversions and separate kernel launches. This project tests whether a narrow boundary-fusion primitive remains novel and useful after source audit and Macbook correctness gates.

## Hypothesis (testable)
In Granite-4.0-H-Small and Nemotron-3-Nano-30B-A3B-FP8 decode at typical inference sizes (4K–32K context, 256-token output), ≥5% of decode latency is attributable to layer-type transitions (memory traffic + kernel launch overhead). A fused boundary kernel recovers ≥60% of this overhead, yielding ≥3% end-to-end TPS improvement.

## Why this is novel
- vLLM V1 added "first-class" hybrid support but explicitly notes Triton launch overhead as a problem.
- Mamba-3 ships its own kernels; FlashInfer adds Mamba ops; **neither fuses across the boundary**.
- The July 2025 hybrid characterization paper documents the problem but doesn't build a kernel.
- No published kernel-level work targets the seam itself.
- DeepSeek V4 (April 2026) ships token-wise compressed attention but is *not* a hybrid model — orthogonal contribution.

## Why it fits the lab
Hybrid architectures explicitly on Tambe's stated research roadmap. Pure kernel-level systems contribution. Single-GPU artifact. MLSys-shaped.

## Risk profile
**Low–medium.** Architectures are mainstream enough that someone will notice this gap within 6–12 months. The boundary may turn out to be too small to fuse on a 5090's massive bandwidth (caught at Phase 5).

---

## Folder
`experimental/hybridkernel/`

---

## Phase 0: Setup (½ day, Macbook, $0)

- [ ] Create `colm-hybrid` env per `00_setup.md`
- [ ] Clone repos: vLLM, mamba-ssm, mamba (Mamba-3), flash-linear-attention, Bamba v2 reference
- [ ] Pull model configs (no full weights yet):
  - `ibm-granite/granite-4.0-h-tiny` (primary target — small enough for fast iteration)
  - `ibm-granite/granite-4.0-h-small` (secondary target — production-relevant size)
  - `nvidia/Nemotron-3-Nano-30B-A3B` (FP8 variant — fits 5090 with reasoning budget control)
  - `nvidia/Nemotron-H-8B-Base` (proven hybrid reference)
  - `ServiceNow-AI/Apriel-H1-15B-Thinker` (hybrid reasoner; bridges to ThoughtFlow)
  - `Qwen/Qwen3-Next-80B-A3B` (reference; too big for 5090, use config only)

**Deliverable**: `phase0/setup_complete.md` confirming env + repos + configs.

---

## Phase 1: Literature audit (2 days, Macbook, $0)

For each item: read carefully and note (a) kernel architecture, (b) how transitions are handled, (c) whether transitions are fused or separate launches.

- [ ] Mamba-3 paper (ICLR 2026, March 2026) + kernel code — MIMO design over Mamba-2
- [ ] Bamba v2 paper + reference impl
- [ ] Granite-4.0 architecture report (IBM, Oct 2025) — 9:1 Mamba:transformer ratio
- [ ] Nemotron-H papers (8B/47B/56B, March 2025) — 92% attention layers replaced with Mamba2
- [ ] Nemotron-3 Nano technical report (Dec 2025) — hybrid MoE Mamba-Transformer with 1M context, FP8 native
- [ ] Apriel-H1-15B-Thinker tech report (Nov 2025) — hybrid reasoning, 30/50 SSM/MHA variants
- [ ] Qwen3-Next-80B-A3B technical report (Sep 2025)
- [ ] Hymba paper
- [ ] vLLM hybrid model V1 design doc + relevant GitHub PRs (search for "hybrid", "mamba", "boundary")
- [ ] FlashInfer Mamba integration code
- [ ] July 2025 hybrid characterization paper (find exact title)
- [ ] Arxiv search: "fused attention mamba kernel", "hybrid model boundary", "ssm transformer transition"

**Deliverable**: `phase1/lit_review.md` with:
- One paragraph per work
- Explicit table: which works fuse boundary ops, which don't
- Conclusion: "no published kernel for layer-type boundary fusion"

### KILL CRITERION
Existing fused boundary kernel found in any production codebase or published paper. If found: kill the project, reallocate effort.

---

## Phase 2: Architecture mapping (1 day, Macbook, $0)

For each of Granite-4.0-H-Tiny, Granite-4.0-H-Small, Nemotron-3-Nano-30B-A3B, Apriel-H1-15B-Thinker:

- [ ] Identify number and type of layer transitions per forward pass
- [ ] Inventory tensor shapes crossing each boundary (attention KV/output → SSM input; SSM state/output → attention input)
- [ ] Compute theoretical bytes-per-token moved at each boundary (model dim × dtype × directions)
- [ ] Roofline-style theoretical bound on fusion benefit (bandwidth-saving × boundaries-per-token)

**Deliverable**: `phase2/architecture_map.md` with per-model boundary inventory + theoretical fusion bound. Table format preferred.

### Pivot trigger
If theoretical fusion benefit < 3% across all models: pivot to focused model (whichever has the highest theoretical bound) or reconsider hypothesis.

---

## Phase 3: Reference implementation (2 days, Macbook, $0)

- [ ] CPU/NumPy reference of attention output → SSM input boundary (Granite-4 style: 9 Mamba blocks per attention block — pick the dominant transition)
- [ ] CPU/NumPy reference of SSM state/output → attention input boundary
- [ ] Equivalence test: run a Granite-4-style mini-model on Mac (CPU or MPS, ≤500M params) using both your reference and the canonical implementation; outputs match to fp32 tolerance
- [ ] Unit tests with synthetic tensors at multiple shapes (d_model ∈ {128, 256, 512}, d_state ∈ {16, 64, 128 — Granite-4 uses larger states})

**Deliverable**: `phase3/reference/boundary.py`, `phase3/tests/test_boundary.py`, passing pytest output captured to `phase3/test_output.txt`.

### KILL CRITERION
If the reference implementation diverges from the canonical implementation by >1e-3 relative error and the cause cannot be diagnosed within 1 day: the boundary semantics are more complex than expected — kill or scope down.

---

## Phase 4: Triton interpreter kernel skeleton (1 day, Macbook, $0)

- [ ] Write Triton kernel skeleton for the fused boundary (attention output → SSM input; the dominant case)
- [ ] Add a plain PyTorch CPU reference
- [ ] Run `TRITON_INTERPRET=1` correctness tests against the CPU reference
- [ ] If local Triton is unavailable, keep CPU tests passing and mark Phase 4 blocked, not complete
- [ ] Confirm tile shapes are compatible across attention and SSM portions
- [ ] Document integration plan with vLLM's attention backend: which file, which class, where the fused kernel slots in

**Deliverable**: `phase4/kernel/boundary_triton.py`, `phase4/integration_plan.md` (~1 page).

### KILL CRITERION
If Triton cannot express the boundary as a single fused kernel without launching a separate kernel for the SSM scan: the fusion is mechanically impossible at the kernel level. Document why and kill or pivot to a smaller fusion (e.g., just the format conversion, not the scan).

---

## ── GATE TO 5090 ──

**All must hold:**
- [ ] Phase 1: open niche confirmed
- [ ] Phase 2: theoretical fusion benefit ≥3%
- [ ] Phase 3: reference passes all tests
- [ ] Phase 4: Triton interpreter skeleton matches the CPU reference or is explicitly blocked by missing local Triton
- [ ] `experimental/hybridkernel/progress.md` summarizes all of above

**Sign-off**: human review.

---

## Phase 5: 5090 baseline (½ day, ~$5 on spot)

- [ ] Spin up RunPod 5090 spot pod with persistent volume
- [ ] Profile Granite-4.0-H-Small in vLLM at decode lengths: 1K, 4K, 16K context, 256-token output
- [ ] Profile Nemotron-3-Nano-30B-A3B (FP8) same configs
- [ ] Profile Apriel-H1-15B-Thinker (BF16 — fits 30GB) — bonus: bridges to ThoughtFlow framing if both projects survive
- [ ] Use NSight Systems + PyTorch profiler
- [ ] Measure: kernel launches per token, ms in transitions vs attention vs SSM scan

**Deliverable**: `phase5/baseline_profile.md` with per-model breakdown.

### PIVOT TEST
- ≥5% of decode latency attributable to transitions → **continue to Phase 6**
- 2–5% → **scope down**: focus on the model with highest boundary cost; prepare a smaller paper
- <2% → **kill**, reallocate to surviving project(s)

---

## Phase 6: Prototype kernel (3–4 days, ~$50)

- [ ] Implement the fused boundary kernel in Triton on the 5090
- [ ] Numerical correctness check vs the Phase 3 reference on Granite-4.0-H-Small (fp16 tolerance)
- [ ] Initial perf characterization: time-per-boundary vs unfused baseline
- [ ] Wall-clock end-to-end TPS comparison on a representative workload

**Deliverable**: `phase6/kernel/boundary.py`, `phase6/correctness.md`, `phase6/perf.md`.

### Continue criteria
- Correctness: relative error <1e-2 in fp16
- Perf: ≥1.2× speedup at the boundary itself in microbenchmark

---

## Phase 7: Optimization + multi-model evaluation (1 week, ~$200–300)

- [ ] Optimize: tile sizes, shared memory usage, vectorization, async copy where applicable
- [ ] Benchmark across Granite-4.0-H-Tiny, Granite-4.0-H-Small, Nemotron-3-Nano-30B-A3B-FP8, Apriel-H1-15B-Thinker, Nemotron-H-8B (if memory allows; tensor-parallel options if needed)
- [ ] vLLM end-to-end TPS measurements at multiple batch sizes and context lengths
- [ ] Quality preservation: GSM8K, MMLU, AIME 2025 (for Apriel-H1-Thinker) on each model — confirm no accuracy regression
- [ ] Ablations: fusion benefit per boundary type, sensitivity to context length

**Deliverable**: `phase7/results/` with all benchmark CSVs, plots, ablation tables.

---

## Phase 8: Paper (1 week, mostly Macbook + ~$80 H100 burst)

- [ ] Draft COLM workshop paper (4–8 pages) on Macbook
- [ ] Final eval burst on rented H100 for datacenter numbers (~24 hours, ~$60)
- [ ] Comparison plots vs Mamba-3 baseline, FlashInfer baseline
- [ ] Polish, internal review, submit

**Deliverable**: `phase8/paper.tex`, `phase8/figures/`, COLM submission.

---

## Total estimated cost
- Phase 5: $5
- Phase 6: $50
- Phase 7: $250
- Phase 8: $80
- **Total: ~$385**

## Risk factors and mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Existing fusion subsumes ours | Medium | Phase 1 audit |
| Boundary too small on 5090 | Medium | Phase 5 pivot test |
| Triton fusion mechanically infeasible | Low-medium | Phase 4 interpreter skeleton |
| Quality regression from fusion | Low | Phase 7 quality eval |
| Reviewers want H100 numbers | Medium | Phase 8 H100 burst |

## Follow-up paper (MLSys/ICLR Sept–Oct)
Extension to all hybrid architectures (training too?), data-center evaluation on 8×H100, integration upstream into vLLM. Single COLM workshop paper → expanded MLSys submission is a natural arc.
