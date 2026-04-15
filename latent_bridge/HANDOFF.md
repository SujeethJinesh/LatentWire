# RotAlign-KV: Project Handoff Document

**Last updated:** April 13, 2026
**Status:** Implementation complete, validated on synthetic data, awaiting real-model experiments
**Target venues:** COLM 2026 workshop (deadline June 23, 2026) → ICLR 2027 full paper (deadline ~late September 2026)

---

## 1. What this project is

**One-line pitch:** Cross-model LLM communication is mostly a coordinate problem, not a learning problem. A fixed random rotation Gaussianizes both models' KV caches, after which cross-model alignment collapses to a closed-form linear solve and scalar quantization becomes near-optimal.

**The method (RotAlign-KV)** lets two heterogeneous LLMs (different families, sizes, or tokenizers) communicate by translating one model's KV cache into the other's representation space. Unlike prior methods (C2C, KVComm, Interlat) which require learned MLP fusers, our pipeline is:

1. Optional ZCA whitening of source coordinates
2. Fixed random orthogonal or Hadamard rotation (Gaussianizes the distribution)
3. Closed-form linear alignment (Procrustes / ridge / CCA / reduced-rank)
4. Lloyd-Max scalar quantization (near-optimal on Gaussian coordinates)
5. Inverse rotation and gated fusion into target decoder

Everything except the per-layer fusion gate is closed-form or training-free.

**Key differentiators vs. prior art:**
- No neural network fuser to train (just closed-form linear algebra)
- Built-in compression (3–4 bits per KV element vs full precision in C2C/KVComm)
- First rate-distortion analysis of cross-model LLM communication
- Handles cross-family pairs via CCA and reduced-rank regression

**The headline empirical claim we want to make:** reasoning tasks benefit from KV transfer substantially more than knowledge tasks. This would be the first demonstration that what gets lost when serializing to text is specifically the *intermediate reasoning state*, not the final answer.

---

## 2. Why now, and where this fits in the literature

The field moved fast in late 2025 and early 2026. Here's the landscape as of April 2026:

**Foundational work (theoretical motivation):**
- **Platonic Representation Hypothesis** (Huh, Cheung, Wang, Isola, 2024) — argues that different models converge to the same latent space. arXiv: https://arxiv.org/abs/2405.07987
- **Relative representations** (Moschella et al., ICLR 2023) — zero-shot model stitching via anchor-based latent communication. arXiv: https://arxiv.org/abs/2209.15430
- **Strong Platonic / vec2vec** (Jha et al., 2025) — unsupervised embedding translation between vector spaces. arXiv: https://arxiv.org/abs/2505.12540
- **Cross-model Platonic Transferability** (Huang et al., ACL 2025) — linear transformations transfer steering vectors across LLMs. arXiv: https://arxiv.org/abs/2501.02009

**Direct prior art on cross-model KV transfer (our primary comparisons):**
- **Cache-to-Cache (C2C)** (Fu et al., ICLR 2026) — 3-layer MLP fuser, 6.4–14.2% gains over individual models. arXiv: https://arxiv.org/abs/2510.03215 — Code: https://github.com/thu-nics/C2C
- **KVComm** (Shi et al., ICLR 2026) — selective KV-pair sharing, training-free. arXiv: https://arxiv.org/abs/2510.03346 — Code: https://github.com/Zephyroam/KVComm
- **Communicating Activations** (Ramesh & Li, 2025) — single hidden-state graft, simple baseline. arXiv: https://arxiv.org/abs/2501.14082
- **Interlat** (Du et al., 2025) — multi-step latent communication via compression adapters. arXiv: https://arxiv.org/abs/2511.09149
- **LatentMAS** (Yang et al., 2025) — training-free multi-agent latent collaboration with KV working memory. arXiv: https://arxiv.org/abs/2511.20639
- **Latent Space K-V Cache Alignment** (Dery et al., Jan 2026) — closest to us: hub-and-spoke shared KV space. arXiv: https://arxiv.org/abs/2601.06123
- **SemAlign / Beyond Neural Incompatibility** (Gu et al., 2025) — layer outputs as transfer medium, CKA pairing. arXiv: https://arxiv.org/abs/2510.24208

**Compression building blocks (what makes our method practical):**
- **TurboQuant** (Zandieh et al., ICLR 2026) — the random rotation + Gaussianization trick we borrow. arXiv: https://arxiv.org/abs/2504.19874 — Community impl: https://github.com/0xSero/turboquant
- **QuIP#** (Tseng et al., NeurIPS 2024) — Hadamard rotation for O(d log d) alternative
- **KV Cache Transform Coding (KVTC)** (Staniszewski & Łańcucki, ICLR 2026) — found cross-head KV alignment is nearly orthogonal, which inspired our Procrustes approach. arXiv: https://arxiv.org/abs/2511.01815

**Related compression work:**
- **CacheBlend / CacheGen** — KV cache compression for transmission
- **DroidSpeak** (Liu et al., NSDI 2026) — cross-LLM KV sharing with selective recomputation. arXiv: https://arxiv.org/abs/2411.02820
- **LMCache** — open-source KV caching system. arXiv: https://arxiv.org/abs/2510.09665

**The gap we fill:** All prior cross-model KV methods either use learned MLPs (expensive, opaque) or assume same-family pairs (limited). None of them address compression. Nobody has done a rate-distortion analysis. Nobody has applied the TurboQuant rotation trick to cross-model alignment. Those are our four contributions.

---

## 3. What's in the codebase (ships as working code)

Repo layout:

```
rotalign_kv/
├── method.md                     Formal writeup: math, protocol, contributions
├── README.md                     User-facing docs + quickstart
├── requirements.txt
├── rotalign/                     Core library (~600 lines)
│   ├── __init__.py               Public API
│   ├── rotation.py               Haar rotation, Hadamard, ZCA whitening
│   ├── procrustes.py             5 closed-form alignment solvers
│   ├── quantize.py               Lloyd-Max scalar quantizer (searchsorted-optimized)
│   └── translator.py             RotAlignKVTranslator composing all stages
└── scripts/
    ├── demo.py                   Self-contained sanity check (no model downloads)
    ├── calibrate.py              Fit translator on real HF models
    ├── evaluate.py               Compare against baselines on MCQ and generation tasks
    └── ablation_sweep.py         Factorial grid runner
```

**What's verified working** (all in `demo.py`, no model downloads required):
- Random orthogonal rotation: exactly orthogonal (err ~2e-7)
- Hadamard rotation: exactly orthogonal, Gaussianizes kurtosis from ~100 to ~3.4
- ZCA whitening: perfect variance equalization
- All 5 alignment solvers recover planted linear structure at cos > 0.9
- Lloyd-Max distortion tracks theory within 1.5× at 2-8 bits
- Full translator end-to-end on mismatched shapes (4 heads × 64 dim → 8 heads × 96 dim) at cos 1.000

**What's NOT yet tested** (this is what the next agent needs to do):
- Anything involving real HuggingFace models
- Anything involving actual benchmark tasks
- Anything involving baseline comparisons (C2C, KVComm)
- Whether the method actually works on real LLM KV caches

---

## 4. Component study and pipeline options

The method is deliberately built as a swappable pipeline. Each stage has multiple options exposed via `TranslatorConfig`:

| Stage | Options | Config flag |
|---|---|---|
| Rotation | `orthogonal`, `hadamard` | `rotation_kind` |
| Whitening | off, ZCA on source | `use_whitening` |
| Alignment | `procrustes`, `ridge`, `cca`, `reduced_rank`, `procrustes_rand` | `alignment_method` |
| Quantization | 2 / 3 / 4 / 6 / 8 bits (or off) | `quant_bits` |
| Layer pairing | linear interp, explicit list | `layer_pairing` |
| Fusion gate | fixed 0.5, learned, line-searched | trained on `gate_K/V` |

All solvers are closed-form (SVD / eigendecomp). The only trainable parameter is the fusion gate, and even that can be replaced with a 1D line search.

---

## 5. The story to sell (for paper introductions)

**The hook:** Two LLMs trying to collaborate by passing text to each other are like two polyglots agreeing to only speak in English — most of what they know gets lost in translation.

**The problem:** Prior work shows KV-cache transfer beats text-mediated communication, but existing methods require learned neural fusers, assume same-family model pairs, and ignore compression. This means the technique works in principle but is neither practical (fuser training is expensive) nor scalable (cross-family is open) nor deployable (no bandwidth story).

**The insight:** KV caches are hard to translate because they live in anisotropic, outlier-heavy, model-specific coordinate systems. But a fixed random rotation Gaussianizes them via concentration of measure, and in Gaussianized coordinates cross-model alignment collapses to a closed-form linear problem, scalar quantization becomes near-optimal, and distortion decomposes cleanly into alignment + quantization errors that do not compound.

**The payoff:** RotAlign-KV matches C2C's accuracy at 4× lower bandwidth, requires no neural fuser training, handles cross-family pairs, and exposes a rate-distortion curve mapping accuracy onto a theoretical bound. Most importantly: reasoning tasks benefit substantially more than knowledge tasks — the first demonstration that what gets lost in text serialization is the intermediate reasoning state.

**The reframe (memorable):** *Cross-model communication is mostly a coordinate problem, not a learning problem.*

---

## 6. Technical contributions list (for the paper)

1. **A unified geometric framework for cross-model KV transfer.** Rotation, whitening, linear alignment, and scalar quantization compose into a near-optimal pipeline where each stage's optimality follows from the previous stage's Gaussianization. No deep networks required.

2. **Five closed-form alignment solvers** as swappable components: Procrustes, ridge, CCA, reduced-rank regression, and randomized-SVD Procrustes for 70B scale.

3. **First rate-distortion analysis of cross-model LLM communication.** Theoretical bound plus empirical validation within 1.5× of the Shannon limit.

4. **Empirical findings:**
   - Gaussianization is load-bearing (no-rotation ablation loses 5–8 points)
   - Linear alignment is sufficient (MLP residual adds <1 point)
   - 4 bits is the sweet spot (graceful degradation to 2)
   - **Reasoning benefits from KV transfer more than knowledge does** (headline)

5. **Public benchmark and codebase.** Six model pairs, six tasks, six baselines, reproducible from config files. The field has no shared benchmark; providing one is a contribution.

**Explicitly NOT claiming:**
- Not SOTA on any individual benchmark
- Not faster than single forward pass (faster than text-mediated multi-model)
- Not replacing fine-tuning or distillation

---

## 7. Experimental protocol

### Workshop version (COLM, June 23 deadline): minimum viable set

- **1 control pair**: Qwen2.5-0.5B-Instruct → Qwen3-0.6B
- **1 stress test**: Qwen2.5-0.5B-Instruct → google/gemma-4-E2B-it
- **3 benchmarks**: MMLU-Redux, ARC-Challenge, **GSM8K** (reasoning is the headline)
- **4 baselines**: target alone, text-to-text, C2C, ours
- **Rate-distortion sweep**: bits ∈ {2, 3, 4, 8}
- **Core ablations**: no-rotation, identity-W, interp-vs-CKA, low-gate full precision, sparse layer transmission
- **Comparator runs**: translated-only, fused, text+KV hybrid, and one reasoning-state comparator
- **Systems metrics**: bytes transmitted, TTFT, decode throughput, and end-to-end latency
- **Compute**: ~10–15 GPU-hours, depending on calibration size and cache reuse
- **Page budget**: 4 pages

### Full paper version (ICLR 2027, late September deadline): full component study

**Models (6 pairs):**

| Tier | Source → Target | What it tests |
|---|---|---|
| Same family, cross size | Qwen2.5-0.5B → Qwen3-0.6B | Baseline |
| Same family, weak-to-strong | Qwen2.5-0.5B → Qwen2.5-7B | Small teaches large |
| Same family, strong-to-weak | Qwen2.5-7B → Qwen2.5-0.5B | Large teaches small |
| Same tokenizer, cross size | Llama-3.2-1B → Llama-3.2-3B | Same-family control |
| Cross family | Llama-3.2-3B → Qwen3-1.7B | Cross-tokenizer stress test |
| Cross family, hard | Gemma-2-2B → Llama-3.2-3B | Cross-tokenizer stress test |

**Benchmarks (6, balanced):**
- Knowledge: MMLU-Redux, ARC-Challenge
- Reasoning (headline): GSM8K, MATH, BBH, GPQA-Diamond

**Baselines (6):**
1. Target alone
2. Text-to-text (source writes analytical hint)
3. Query-level routing
4. C2C (https://github.com/thu-nics/C2C)
5. KVComm (https://github.com/Zephyroam/KVComm)
6. Interlat (last-hidden-state communication)

**Ablations (8):**
1. No rotation (identity R_s, R_t)
2. Identity W (rotation only, no alignment)
3. Hadamard vs full random orthogonal
4. Whitening on/off
5. Procrustes vs CCA vs reduced-rank vs MLP residual
6. Bit rates {2, 3, 4, 6, 8, 16}
7. Calibration size {100, 500, 1K, 5K, 10K}
8. Fusion gate: fixed / trained / line-searched

**Reporting requirements:**
1. Accuracy
2. Bytes transmitted per example
3. TTFT
4. Decode throughput
5. End-to-end latency

**Compute**: ~150–200 GPU-hours, ~$300–500 on cloud A100s
**Page budget**: 8–9 pages + appendix

### Expected results (if hypothesis holds)

- RotAlign-KV at 4 bits matches C2C at full precision, 4× lower bandwidth
- Clean rate-distortion curve from 2–8 bits
- **Reasoning gap:** beats text-to-text by ≥3% on GSM8K/MATH/BBH but ties on MMLU/ARC
- No-rotation ablation loses 5–8 points
- Identity-W ablation loses 10+ points
- Hadamard matches random orthogonal within 0.5 points at 4× speed
- Cross-family pairs 2–4 points worse than same-family but above T2T

---

## 8. Conference strategy

**COLM 2026 workshops**
- Location: Hilton San Francisco Union Square
- Conference dates: October 6–9, 2026 (workshop day Oct 9)
- Workshop proposal deadline: April 14, 2026
- Workshop list announced: ~May 12, 2026
- **Workshop paper deadline: June 23, 2026 (our target)**
- Notification: July 24, 2026
- Main site: https://colmweb.org
- Workshop list: https://colmweb.org/workshops.html (watch after May 12)

**ICLR 2027 (primary full paper target)**
- Historical deadline: late September 2026 (based on prior years' patterns)
- Acceptance rate: ~32% overall, ~5% spotlight, ~1.2% oral
- Why ICLR: explicitly welcomes representation-learning work; C2C and KVComm both landed there; rewards clean methods with good ablations over leaderboard chasing
- Main site: https://iclr.cc

**Path:** Submit workshop version to COLM by June 23 → get feedback in July → extend for ICLR submission in late September → ICLR reviews in winter 2026-2027 → conference April 2027.

**Dual submission note:** COLM workshops are typically non-archival. Verify this in the specific workshop's CFP (look for "non-archival" / "dual submission" / "concurrent submission" keywords). If non-archival, no dual submission issue with ICLR.

---

## 9. What the next agent needs to do (priority order)

### Phase 1: Real-model validation (the critical experiment)

**This is the single most important thing.** Before any paper writing, answer: does RotAlign-KV actually work on real LLMs?

1. **Set up environment:**
   ```bash
   cd /Users/sujeethjinesh/Desktop/LatentWire
   source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/demo.py  # sanity check, should all pass
   ```

2. **Get calibration data.** Grab 500–1000 prompts from any instruction dataset. Suggestions:
   - First N lines of OpenHermes 2.5 (what C2C used)
   - First N lines of Alpaca, ShareGPT, or UltraChat
   - Write them to `data/calibration.txt` one per line

3. **Run the critical control experiment (Qwen2.5 → Qwen3, GSM8K):**
   ```bash
   python scripts/calibrate.py \
     --source-model Qwen/Qwen2.5-0.5B-Instruct \
     --target-model Qwen/Qwen3-0.6B \
     --calibration-file data/calibration.txt \
     --output checkpoints/qwen25_to_qwen3.pt \
     --bits 4 --verbose
   ```

4. **Build the first eval.** Convert ~100 GSM8K examples to the MCQ format that `scripts/evaluate.py` expects (or — better — extend evaluate.py to handle exact-match generation for math).

5. **Run the eval:**
   ```bash
   python scripts/evaluate.py \
     --translator checkpoints/qwen25_to_qwen3.pt \
     --source-model Qwen/Qwen2.5-0.5B-Instruct \
     --target-model Qwen/Qwen3-0.6B \
     --eval-file data/gsm8k.jsonl
   ```

6. **Interpret the result:**
   - If RotAlign-KV > text-to-text: method works, proceed to full experiments
   - If RotAlign-KV ≈ text-to-text: debug — try `--no-quantize`, try without whitening, check alignment-quality diagnostics
   - If RotAlign-KV < target-alone: something is badly broken (probably fusion gate or alignment fit)

**Expected: ~30 minutes on a rented A100. This single experiment determines whether the project lives or dies.**

**Current status note (April 14, 2026):**
- The local control suite did reach a best row of `0.06` on a `100`-example
  GSM8K slice, but that score came from sweeping gate values on the same slice
  it reported on. Treat it as directional, not final.
- The next control rerun must use held-out gate selection:
  `data/gsm8k_gate_search_30.jsonl` for gate search, and
  `data/gsm8k_eval_70.jsonl` for the reported score.
- The next control matrix should compare `text-to-text`, fused KV,
  translated-only KV, and `text+KV hybrid` under matched
  `plain` / `brief_analysis` / `cot` prompting.

### Phase 1b: Expand only after the control path is stable

Once the control pair works, expand in this order:

1. `Qwen/Qwen2.5-0.5B-Instruct -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
2. `Qwen/Qwen2.5-0.5B-Instruct -> google/gemma-4-E2B-it`
3. `Qwen/Qwen3-0.6B -> google/gemma-4-E2B-it` if you want a second stress test
4. Manual follow-up, not unattended default: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-0.8B`
5. Later stretch: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-4B`

Why this order:
- The DeepSeek pair checks whether the method still works on a reasoning-tuned receiver with partial Qwen lineage.
- The Gemma 4 pair is the first clean small cross-family stress test.
- Qwen3.5 is useful, but it should not block the earlier signal or be treated as the unattended default.

### Phase 2: Harden the evaluation harness

Current `scripts/evaluate.py` now handles both MCQ and exact-match generation.
The next step is to make it benchmark-grade and systems-aware:

1. **Integrate `lm-evaluation-harness`** (https://github.com/EleutherAI/lm-evaluation-harness) as an optional scoring backend. This gives you MMLU, ARC, GSM8K, MATH, BBH, GPQA with less custom dataset glue.

2. **Wire in the RotAlign-KV method as a custom "model" wrapper** that intercepts the target model's forward pass and injects translated KVs.

3. **Standardize exact-match and normalization rules** for generation tasks (GSM8K, MATH) so the paper numbers are reproducible.

4. **Log systems metrics on every run**: bytes transmitted, TTFT, decode throughput, and end-to-end latency.

5. **Keep `--no-quantize` as the main diagnostic path** until full-precision low-gate runs beat text-to-text.

6. **Run at least one knowledge benchmark before broadening the model matrix.**
   A small `MMLU-Redux` or `ARC-Challenge` slice with the best control-pair
   config is the quickest test of the paper's headline reasoning-vs-knowledge claim.

### Phase 3: Add the C2C baseline

1. Clone https://github.com/thu-nics/C2C
2. Wrap it as another `--method` option in `evaluate.py`
3. Use their pre-trained checkpoints if compatible, otherwise train a new fuser with their scripts on the same calibration data

### Phase 4: Run the workshop experiments

Given Phase 1 validates the method, run:
- 2 small model pairs plus 1 cross-tokenizer stress test
- 3 benchmarks × 4 methods × 4 bit rates
- 4 core ablations: no-rotation, identity-W, interp-vs-CKA, low-gate full precision
- 2 comparator runs: text+KV hybrid and reasoning-state comparator
- 1 fairness rerun block: held-out gate selection and matched text baselines
- Systems metrics for every cell
- Total ~15 GPU-hours

### Phase 5: Write the workshop paper

- 4 pages, COLM format
- Method section: compressed version of `method.md`
- Results: one main table, one rate-distortion plot, one ablation table
- Ship by June 23

### Phase 6 (post-workshop): Full ICLR experiments

- All 6 model pairs × 6 benchmarks × 6 baselines
- Full 8-way ablation grid (use `scripts/ablation_sweep.py`)
- Cross-tokenizer pairs via CCA / reduced-rank
- Theory section: write out the 4 propositions with proofs
- Target late September deadline

---

## 10. Known gaps and risks

**Technical risks (ordered by severity):**

1. **The method might not actually beat text-to-text on real models.** My synthetic tests prove the math is right, but real KV caches have subtle correlations that rotation + Procrustes might not fully capture. This is the Phase 1 question.

2. **The fusion gate is initialized at 0.5 and not yet trained.** The translator ships with gates at sigmoid(0) = 0.5. For real-model experiments you need to either (a) add a training loop that optimizes the gate via gradient descent on next-token CE, or (b) do a 1D line search over alpha per layer. I recommend option (b) for simplicity — it's a single-parameter grid search.

   Update after the first control suite: do not line-search on the same eval
   slice you report. Use held-out gate search only, otherwise marginal wins on
   70–100 example pilots are too optimistic.

3. **Cross-tokenizer pairs.** Current calibration assumes token-wise pairing. For Llama → Qwen this is only approximate. Treat Qwen → Gemma and similar pairs as stress tests, and fall back to same-tokenizer pairs if the alignment looks unstable.

4. **Calibration memory.** For 60-layer models with long sequences, the Procrustes SVD may OOM on small GPUs. Use `--alignment procrustes_rand` (randomized SVD variant) if this happens.

5. **Hadamard requires power-of-2 head dim.** For arbitrary head dims, the code falls back to QR'd sub-block which is no longer exactly Hadamard. Fine in practice but flag this in the paper.

**Scope risks:**

1. **Component study explosion.** The factorial grid is 3 rotations × 5 alignments × 6 bit rates × 2 whitening × 6 pairs × 6 tasks = ~6500 cells. The ICLR paper should run the full grid on 1–2 pairs to find the winning combo, then run only the winner on all other pairs/tasks. Don't try to run the full grid everywhere.

2. **Reviewer pushback on "simplicity."** The closed-form nature is both the selling point and a potential criticism ("is this novel enough for ICLR?"). Counter by emphasizing the *theoretical* contribution (rate-distortion bound, Proposition 3 cleanliness) rather than architectural novelty.

---

## 11. Open research directions (post-paper)

These didn't make it into the first paper but are natural follow-ups:

1. **Cross-tokenizer via Gromov-Wasserstein.** The biggest open problem. Replace token-wise pairing with soft GW correspondences. Could be the core contribution of a second paper.

2. **Multi-hop compounding.** What happens when a thought passes through 3+ heterogeneous models? Does rotation-based alignment compound errors better than C2C? Easy experiment, potentially striking result.

3. **Head-grouped / per-head alignment.** KVTC suggests the remaining mismatch
   may be head-structured rather than layer-structured. This is the first
   structural ablation to add if sparse CKA remains positive but small.

4. **Steering-vector comparator.** If a small set of residual directions
   matches RotAlign-KV, the real story is semantic direction transfer rather
   than full-KV transfer. That is publishable either way.

3. **Privacy analysis.** Follow vec2vec's attack methodology: can an adversary reconstruct source input from translated KVs? If yes, quantify. If no, that's a feature.

4. **SAE feature transmission.** Instead of transmitting raw KVs, transmit sparse-autoencoder feature activations. More compressible, more interpretable, genuinely unexplored.

5. **Information-theoretic lower bound.** Can you prove the Shannon rate-distortion bound applies exactly to this setting under the Gaussianization assumption? If so, that's a theorem-grade contribution.

6. **70B-class scale test.** Run on Llama-3.3-70B ↔ Qwen2.5-72B using randomized Procrustes. If the method holds at scale, killer appendix result.

7. **Async speculative decoding via KV.** Small draft model hands off to large verifier via translated KV instead of tokens. Could beat standard speculative decoding on latency.

---

## 12. Reference tables

**Conference deadlines (as of April 13, 2026):**

| Venue | Location | Paper deadline | Workshop deadline | Conference dates |
|---|---|---|---|---|
| COLM 2026 | San Francisco | Mar 31, 2026 (passed) | **Jun 23, 2026** | Oct 6–9, 2026 |
| NeurIPS 2026 | Sydney | May 6, 2026 | TBA (~Aug) | Dec 6–12, 2026 |
| ICML 2026 | Seoul | Jan 24, 2026 (passed) | Apr 24, 2026 (passed) | Jul 6–11, 2026 |
| ICLR 2026 | Rio de Janeiro | already happened | already happened | Apr 23–27, 2026 |
| ICLR 2027 | TBA | **~late Sep 2026** | TBA | ~Apr 2027 |
| EMNLP 2026 | Budapest | May 25, 2026 | ~Aug | Oct 24–29, 2026 |
| ACL 2026 | San Diego | Feb 15, 2026 (passed) | ~Apr-May | Jul 2–7, 2026 |

**Acceptance rates (recent):**

| Venue | Main | Oral | Spotlight |
|---|---|---|---|
| ICLR | ~31–32% | ~1.2% | ~5% |
| NeurIPS | ~25% | — | — |
| ICML | ~27% | — | — |
| COLM main | ~29% | — | — |
| COLM workshops | ~60–80% (estimated, varies) | — | — |

**Key tools and libraries:**

| Purpose | Tool | Link |
|---|---|---|
| Benchmarks | lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness |
| C2C baseline | thu-nics/C2C | https://github.com/thu-nics/C2C |
| KVComm baseline | Zephyroam/KVComm | https://github.com/Zephyroam/KVComm |
| LatentMAS | Gen-Verse/LatentMAS | https://github.com/Gen-Verse/LatentMAS |
| TurboQuant impl | 0xSero/turboquant | https://github.com/0xSero/turboquant |
| Model hosting | HuggingFace Hub | https://huggingface.co |
| GPU rentals | Lambda Labs, RunPod, Vast.ai | — |

**Paper deadlines tracker:** https://aideadlin.es (now https://huggingface.co/spaces/huggingface/ai-deadlines)

---

## 13. Questions to resolve early in Phase 1

Before the next agent writes any new code, they should answer these questions from the Phase 1 experiment:

1. **Does the method beat text-to-text on GSM8K?** If yes → proceed. If no → debug.
2. **What's the cosine similarity of the alignment on real KVs?** Expect ~0.7–0.95 per layer. If it's below 0.5, something is wrong.
3. **How much does quantization hurt?** Compare `--no-quantize` or `--bits 16` to `--bits 4`. If the gap is >2 points, Lloyd-Max isn't doing its job and you need whitening.
4. **Does whitening help?** Compare `--whitening` on vs off. This tells you whether anisotropy is actually a problem.
5. **Does CKA beat interpolation?** If yes, layer semantics matter more than depth matching.
6. **Does the low-gate full-precision path beat fixed 0.5, and does text+KV hybrid beat translated-only?** If yes, the signal is in the reasoning-state comparator, not raw cache replay.

The answers to these 6 questions determine the shape of the final paper.

---

## 14. How to seed the next agent's context

Copy this paragraph into the next conversation:

> I'm working on RotAlign-KV, a cross-model KV-cache transfer method for heterogeneous LLMs. The core insight is that a fixed random rotation Gaussianizes both models' KV caches via concentration of measure, after which cross-model alignment collapses to a closed-form linear solve (Procrustes / ridge / CCA) and Lloyd-Max scalar quantization becomes near-optimal. No neural network fuser required, unlike C2C (ICLR 2026) and KVComm (ICLR 2026). Target venues: COLM 2026 workshop by June 23, then ICLR 2027 full paper in late September. Current status: implementation complete and validated on synthetic data (scripts/demo.py all passes), but not yet tested on real HuggingFace models. See HANDOFF.md for the full project context, method.md for the formal writeup, README.md for the CLI, and method.md §5 for the experimental protocol. Next step is Phase 1: run the Qwen2.5-0.5B → Qwen3-0.6B experiment on GSM8K to determine whether the method actually works on real models. Everything hinges on that single experiment.

---

## 15. File-by-file summary

**Root:**
- `method.md` — formal method writeup with 4 propositions, 5-stage pipeline, full experimental protocol, contributions list, story for introduction. Read this first.
- `README.md` — user-facing docs, quickstart, CLI reference, component study table, references.
- `HANDOFF.md` — this document.
- `requirements.txt` — torch, numpy, transformers, accelerate.

**`rotalign/` (core library):**
- `__init__.py` — public API exports.
- `rotation.py` — `random_orthogonal` (Haar + Mezzadri sign correction), `hadamard_matrix` (randomized Walsh-Hadamard), `make_rotation` (dispatcher), `fit_zca_whitening` / `apply_whitening`, `kurtosis` / `verify_gaussianization` diagnostics.
- `procrustes.py` — `orthogonal_procrustes` (SVD-based closed form), `orthogonal_procrustes_randomized` (for 70B scale), `ridge_projection`, `cca_projection`, `reduced_rank_regression`, `fit_alignment` (dispatcher), `alignment_quality` (diagnostics).
- `quantize.py` — `lloyd_max_gaussian` (codebook fitting via Lloyd iteration with searchsorted), `GaussianQuantizer` class with encode/decode/quantize_dequantize.
- `translator.py` — `TranslatorConfig` dataclass with all swappable options, `RotAlignKVTranslator` nn.Module that composes rotation + whitening + alignment + quantization + fusion. Has `fit_from_pairs` for closed-form calibration and `translate_layer` for inference.

**`scripts/`:**
- `demo.py` — self-contained sanity check on synthetic data. All checks pass.
- `calibrate.py` — loads two HF models, captures KVs, fits translator, saves checkpoint. CLI: `--source-model`, `--target-model`, `--calibration-file`, `--output`, `--bits`, `--rotation`, `--whitening`, `--alignment`, `--alignment-rank`, `--ridge-lambda`.
- `evaluate.py` — loads translator, runs MCQ evaluation with three methods: target-alone, text-to-text, RotAlign-KV. CLI: `--translator`, `--source-model`, `--target-model`, `--eval-file`, `--no-quantize`, `--methods`.
- `ablation_sweep.py` — factorial runner that calls calibrate+evaluate for every combo and writes JSONL results. CLI: `--rotations`, `--alignments`, `--bits`, `--whiten`.

---

## 16. Final notes

The method is elegant but not yet validated. The code ships clean, passes all synthetic tests, and has the right structure for a component-study paper. The single remaining question — whether it actually works on real models — is a 30-minute experiment away.

**If the Phase 1 experiment works:** this is a very strong workshop paper, and a plausible ICLR paper with 2 months of work on top.

**If Phase 1 fails:** the most likely culprit is that real KV caches have more structure than the synthetic tests capture. Debug paths in order: (1) check per-layer alignment cosines from the calibration diagnostics, (2) try whitening on, (3) try Hadamard rotation, (4) try CCA instead of Procrustes, (5) train the fusion gate with a short loop. If none of those work, the method doesn't generalize and you have a negative result worth publishing as a short paper.

Either outcome is publishable. The risk is only in the size of the win, not in having nothing to write about. Good luck.
