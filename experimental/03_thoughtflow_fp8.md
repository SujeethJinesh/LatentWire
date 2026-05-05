# 03 — ThoughtFlow-FP8: Reasoning KV Eviction with Anchor Protection

## TL;DR
Reasoning models generate tens of thousands of tokens; KV cache pressure is the dominant cost. Existing methods (LongFlow, ThinKV, R-KV) compress aggressively but suffer accuracy drops because they evict tokens uniformly across reasoning phases. We build a single fused Triton kernel that combines: (a) FP8 KV quantization, (b) sink-anchor protection, and (c) reasoning-phase-aware eviction. Targets: GPT-OSS-20B, Qwen3.6-27B (thinking mode), Apriel-H1-15B-Thinker, Nemotron-3-Nano.

## Hypothesis (testable)
At iso-throughput vs LongFlow on AIME-2025 long-decode with GPT-OSS-20B (high reasoning effort) or Qwen3.6-27B (thinking mode), ThoughtFlow-FP8 preserves ≥3 percentage points more accuracy. The benefit comes from: (a) FP8 keeps more bytes of context within budget; (b) anchor protection avoids the failure mode behind LongFlow's review issues; (c) phase-aware eviction matches reasoning structure.

## Why this is novel
- **LongFlow** (withdrawn from ICLR 2026) fuses FlashAttention + importance + eviction, but uniform-importance and not phase-aware.
- **ThinKV** is thought-adaptive but operates at the algorithm level, not as a fused kernel.
- **PM-KVQ** does precision reduction but no anchor protection or phase awareness.
- **R-KV, RaaS, LazyEviction, ForesightKV** — various eviction policies, none combine all three ingredients in a fused kernel.
- **DeepSeek V4's c4a/c128a + DSA** (April 2026) is the new production benchmark for KV compression: token-wise compression (4× or 128×) + sliding window + DSA top-k. But it's a *training-time architectural change* requiring the model to be retrained; not retrofitted onto existing reasoning models. **We retrofit.**
- **"Pitfalls of KV Cache Compression"** (also ICLR 2026 withdrawal) explicitly argues these methods have hidden quality costs — our anchor protection directly targets the documented failure mode.

## Competitive landscape note (May 2026)
The bar shifted in April 2026: DeepSeek V4 ships token-wise compressed attention as a *production* feature. Our angle is no longer "fuse compression with attention" — V4 already does that. Our angle is **retrofitting reasoning-aware KV compression onto existing open-weight reasoning models without retraining**. This is the practical question for everyone who can't afford to pretrain a new model with V4-style attention but needs to deploy GPT-OSS-20B / Qwen3.6 / Apriel-H1 efficiently. Frame the paper this way.

**Deprecation alert**: DeepSeek R1 line retires July 24, 2026. Do NOT use R1 distills as headline experimental targets. Use them only for cached offline analysis if needed.

## Why it fits the lab
FP8 quantization with structural anchor preservation = direct extension of the lab's outlier and quantization work. Reasoning-phase awareness adds the systems contribution. Both ICLR and MLSys-shaped.

## Risk profile
**Medium–high**, the most crowded of the three projects. The wedge depends on (a) identifying the specific failure mode that tanked LongFlow's review and (b) clearly distinguishing from V4's production approach. **Phase 1 (forensics + V4 analysis) is the highest-leverage work in the project**.

---

## Folder
`experimental/thoughtflow_fp8/`

---

## Phase 0: Setup (½ day, Macbook, $0)

- [ ] Create `colm-thought` env per `00_setup.md`
- [ ] Clone repos: LongFlow (find via OpenReview / arxiv), ThinKV, R-KV, RaaS, LazyEviction, ForesightKV, PM-KVQ, "Pitfalls of KV Cache Compression", Open-R1
- [ ] Pull configs for current reasoning targets:
  - `openai/gpt-oss-20b` (16GB MXFP4, three reasoning effort levels — primary target, fits easily)
  - `Qwen/Qwen3.6-27B` (April 2026, hybrid thinking/non-thinking, 256K context — primary)
  - `ServiceNow-AI/Apriel-H1-15B-Thinker` (hybrid reasoning, BF16 fits 30GB — bonus: bridges to HybridKernel)
  - `nvidia/Nemotron-3-Nano-30B-A3B` (FP8, has reasoning budget control — secondary)
  - Smaller for Phase 2 trace generation: `Qwen/Qwen3-4B` (with /think mode), `Qwen/Qwen3-1.7B`
- [ ] Pull AIME-2024, AIME-2025, GPQA-Diamond, MATH-500 datasets
- [ ] Pull cached reasoning traces from Open-R1 (still useful for offline analysis even though R1 retiring)

**Deliverable**: `phase0/setup_complete.md`.

---

## Phase 1: Forensics + literature audit (2 days, Macbook, $0)

**This is the highest-leverage work in the project.** If we can identify the specific failure mode that tanked LongFlow's review, we have a clean wedge. If we can't, we have a much weaker pitch.

- [ ] Find LongFlow OpenReview reviews (search ICLR 2026 withdrawn submissions; also Google Scholar / Semantic Scholar for citing papers that critique it)
- [ ] If reviews unavailable: read papers that cite LongFlow critically; reconstruct likely failure modes
- [ ] Read "The Pitfalls of KV Cache Compression" (ICLR 2026 withdrawal) carefully — this is your map of what reviewers care about
- [ ] **Analyze DeepSeek V4's c4a/c128a + DSA + sliding-window architecture** (April 2026): document exactly what V4 does and why it requires retraining (versus our retrofit approach)
- [ ] Audit:
  - [ ] LongFlow paper + code (if released)
  - [ ] ThinKV paper + code
  - [ ] R-KV paper + code
  - [ ] RaaS paper
  - [ ] LazyEviction paper
  - [ ] ForesightKV paper
  - [ ] PM-KVQ paper
  - [ ] DeepSeek V3.2-Exp DSA paper + V4 attention mechanism (vLLM blog, github)
- [ ] Build competitive matrix: per-method, what is fused vs separate? FP8 yes/no? Anchor protection yes/no? Phase awareness yes/no? Retrofit-able to existing models yes/no?

**Deliverable**:
- `phase1/lit_review.md`
- `phase1/competitive_matrix.md` (ours vs LongFlow vs ThinKV vs R-KV vs RaaS vs LazyEviction vs ForesightKV vs PM-KVQ vs DeepSeek V4)
- `phase1/longflow_failure_hypothesis.md`: the specific failure mode we believe sank LongFlow's review, with evidence
- `phase1/v4_differentiation.md`: precise statement of what our retrofit approach does that V4's production approach does not

### KILL CRITERION
If after 2 days we cannot articulate a specific, concrete failure mode of LongFlow that our method addresses: the wedge is too thin. Kill the project.

### PIVOT TRIGGERS
- LongFlow numbers are robust under our scrutiny: pivot framing to "LongFlow + FP8 + anchor" as a pure extension paper (weaker novelty).
- We find LongFlow is buggy in a specific way: that's the entire paper — focused critique + fix.

---

## Phase 2: Reasoning trace analysis (1–2 days, Macbook, $0)

- [ ] Pull cached AIME-2024 / AIME-2025 / GPQA traces from Open-R1 (HuggingFace datasets)
- [ ] **Generate fresh traces** from current reasoning models: GPT-OSS-20B (high effort), Qwen3-4B (/think mode) — runnable on Macbook MPS at small batch sizes; falls back to using API or 5090 if too slow. We need traces from current-gen models, not just R1, since we're targeting current models for evaluation.
- [ ] Algorithmic identification of planning vs execution phases:
  - [ ] String-marker heuristic: "let me think", "wait", "actually", "let me verify", "the answer is"
  - [ ] Token-position heuristic: early-trace = planning, late-trace = execution
  - [ ] Validate heuristic on a held-out sample by manual labeling (~20 traces from each model)
- [ ] Quantify: how often would phase-aware eviction save tokens that uniform eviction drops?
- [ ] **Cross-model robustness**: does the phase-marker heuristic transfer between R1, GPT-OSS-20B, Qwen3.6, and Apriel-H1? If not, we need either a model-agnostic signal (entropy? token confidence?) or per-model classifiers.
- [ ] Cache the labeled traces for downstream use

**Deliverable**: `phase2/phase_classifier.ipynb`, `phase2/labeled_traces/`, `phase2/phase_eviction_analysis.md`, `phase2/cross_model_robustness.md`.

### Pivot trigger
If phase-aware eviction is no better than uniform on cached traces (in terms of which tokens it would protect): the phase-awareness component is dropped. Project becomes "FP8 + anchor only" — still viable but weaker.

---

## Phase 3: FP8 + anchor + phase-aware eviction design (1 day, Macbook, $0)

- [ ] FP8 quantization scheme for KV: per-channel? per-token? per-block? Justify with prior work
- [ ] Anchor list: positions 0–3 (sink) + last N tokens (recent) + phase-transition tokens (from Phase 2 classifier) — formal definition
- [ ] Phase-aware eviction policy: planning-phase keeps coarser tokens; execution-phase keeps fine-grained tokens
- [ ] Importance scoring function compatible with the fused kernel pass

**Deliverable**: `phase3/eviction_policy.md` (~3 pages with formal definitions).

---

## Phase 4: Reference impl + Triton skeleton (2 days, Macbook, $0)

- [ ] CPU reference of FP8 quantization (PyTorch float8 dtype on CPU; or simulate via int8 + scale)
- [ ] CPU reference of anchor + phase-aware eviction policy
- [ ] CPU reference of full eviction step: importance score → quantize → evict → attention pass
- [ ] Test on cached labeled traces from Phase 2: which tokens get evicted, what does the resulting context look like
- [ ] Triton kernel skeleton: fused importance + quantize + eviction; compile and inspect IR

**Deliverable**: `phase4/reference/eviction.py`, `phase4/kernel/thoughtflow_triton.py`, `phase4/integration_plan.md`.

### KILL CRITERION
If the Triton fusion turns out to require multiple kernel launches per step (not actually fused): the systems contribution evaporates. Pivot to algorithm-only paper or kill.

---

## ── GATE TO 5090 ──

**All must hold:**
- [ ] Phase 1: LongFlow failure hypothesis articulated; competitive matrix shows our combination is unique
- [ ] Phase 2: phase classifier validated; phase-aware eviction shown to differ from uniform on cached traces
- [ ] Phase 3: eviction policy formally defined
- [ ] Phase 4: reference passes correctness on cached traces; Triton skeleton compiles

**Sign-off**: human review. **This gate is the most important decision point** — given the field crowding, only proceed if the wedge is sharp.

---

## Phase 5: LongFlow reproduction + baseline (1 day, ~$10)

- [ ] Run LongFlow (or best available re-implementation) on **GPT-OSS-20B** (high reasoning effort) with AIME-2025
- [ ] Reproduce headline numbers — or document where they fail to replicate (LongFlow was originally evaluated on smaller R1 distills; expect adapter work)
- [ ] Run ThinKV same configs
- [ ] Vanilla full-KV baseline
- [ ] Optional: also run on Qwen3.6-27B (BF16 with KV pressure)

**Deliverable**: `phase5/baseline_results.md`, `phase5/reproducibility.md`.

### PIVOT TEST (most decisive in the project)
- **LongFlow numbers don't replicate** → very strong wedge: paper becomes "rigorous re-eval + fix"
- **LongFlow replicates but our analysis identifies clear failure mode** → moderate wedge: continue with our combination
- **LongFlow replicates and is robust against our hypothesis** → kill or major pivot (no clear room)

---

## Phase 6: Prototype kernel (4–5 days, ~$70)

- [ ] Implement fused ThoughtFlow-FP8 kernel in Triton on 5090
- [ ] Numerical correctness: KV quantization round-trip, eviction policy matches Phase 4 reference
- [ ] Initial accuracy comparison: ThoughtFlow-FP8 vs LongFlow on AIME-2025 with GPT-OSS-20B

**Deliverable**: `phase6/kernel/thoughtflow.py`, `phase6/correctness.md`, `phase6/initial_results.md`.

### Continue criteria
- Correctness: KV quantization within expected fp8 tolerance; eviction matches reference exactly
- Initial accuracy: ThoughtFlow-FP8 ≥ LongFlow at iso-throughput on AIME-2025 with GPT-OSS-20B

---

## Phase 7: Optimization + evaluation (1 week, ~$250)

- [ ] Optimize the fused kernel: tile sizes, FP8 conversion fast paths
- [ ] Evaluate at scale on multiple reasoning models (all fit on 5090 at sensible quantization):
  - [ ] **GPT-OSS-20B** (MXFP4 native — primary target; the reasoning effort knob lets us study eviction at varying CoT lengths)
  - [ ] **Qwen3.6-27B** (BF16 thinking mode; KV pressure is the real test on 32GB)
  - [ ] **Apriel-H1-15B-Thinker** (BF16 hybrid reasoner — bonus eval; may behave differently due to SSM layers)
  - [ ] **Nemotron-3-Nano-30B-A3B-FP8** (with reasoning budget control)
- [ ] Benchmarks: AIME-2025 (accuracy + tokens-per-second), GPQA-Diamond, MATH-500, LongBench
- [ ] Iso-throughput comparison: ThoughtFlow-FP8 vs LongFlow vs ThinKV vs vanilla
- [ ] Iso-quality comparison: same accuracy, lower KV memory
- [ ] Ablations: FP8 only / anchor only / phase only / all three
- [ ] **Cross-model ablation**: does the phase-marker heuristic hurt on hybrid Apriel-H1? This is a paper-level finding either way.

**Deliverable**: `phase7/results/`.

---

## Phase 8: Paper (1 week, mostly Macbook + ~$80 H100 burst)

- [ ] COLM workshop paper draft (4–8 pages); the "Pitfalls of KV Compression" framing helps; the V4 differentiation is essential
- [ ] H100 burst for larger-model evaluation: GPT-OSS-120B (fits 80GB H100) — strong reviewer signal that we scale beyond 5090-class models
- [ ] Polish, internal review, submit

---

## Total estimated cost
- Phase 5: $10
- Phase 6: $70
- Phase 7: $250
- Phase 8: $80
- **Total: ~$410**

## Risk factors and mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Field crowding makes novelty thin | High | Phase 1 forensics + V4 differentiation |
| LongFlow is robust | Medium | Phase 5 reproduction is decisive |
| V4-style attention becomes the de facto standard quickly | High | Frame as "retrofit for existing open-weight models" — that's the practical question |
| Phase classifier too crude / fails on hybrid models | Medium | Phase 2 cross-model robustness check |
| FP8 quantization quality cost | Medium | Phase 7 ablations, anchor protection mitigates |
| Multiple competitors publish first | High | Move fast; submit to COLM workshop early |
| R1 retirement makes our trace data look dated | Low | Phase 2 generates fresh traces from current models |

## Follow-up paper (MLSys/ICLR Sept–Oct)
Full kernel + larger-model evaluation (GPT-OSS-120B, DeepSeek-V4-Flash) + production vLLM integration + datacenter eval (8×H100). The COLM submission stakes the claim; the MLSys submission delivers the artifact and full evaluation suite. Combined arc: "we identified a class of failure mode in reasoning KV compression and built the kernel that retrofits the fix onto any existing reasoning model."
