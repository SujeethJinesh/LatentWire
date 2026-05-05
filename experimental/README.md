# COLM 2026 Sprint: Three-Project Triage Plan

## Timeline
- **Today**: May 4, 2026
- **COLM workshop submission target**: ~June 25, 2026
- **MLSys / ICLR submission**: September–October 2026
- **Budget cap**: ~$0.50/hr on GPU; total ~$700–1,000 for the sprint

## Strategy
Three projects run in parallel inside this single LatentWire checkout through Phase 4. Gates determine which survive to the 5090. **No GPU spend until all pre-GPU gates pass.**

Project docs:
1. `01_hybridkernel.md` — fused attention↔SSM boundary kernel for hybrid LLMs
2. `02_sinkaware.md` — attention sink as static precomputed prior
3. `03_thoughtflow_fp8.md` — fused FP8-KV + sink-anchor + phase-aware eviction for reasoning

Shared setup: `00_setup.md`.

Project folders:
1. `hybridkernel/`
2. `sinkaware/`
3. `thoughtflow_fp8/`

## Model landscape (May 2026, current as of doc creation)
- **Hybrid LLMs**: Granite-4.0-H (Oct 2025), Nemotron-H-8B/47B/56B, Nemotron-3-Nano-30B-A3B (Dec 2025), Mamba-3 (March 2026), Apriel-H1-15B-Thinker (Nov 2025), Qwen3-Next-80B-A3B
- **Dense/MoE LLMs**: Qwen3.6-27B (April 2026), GPT-OSS-20B/120B, Llama 4 family, Gemma 4, Mistral Large 3
- **Reasoning models**: DeepSeek V4-Pro/Flash (April 24 2026), GPT-OSS-20B (configurable reasoning effort), Qwen3.6 thinking mode, Apriel-H1-Thinker, Nemotron-3-Nano (reasoning budget control), GLM-5.1
- **Production sparse attention now standard**: DeepSeek V3.2/V4 use DSA + c4a/c128a token-wise KV compression; GLM-5.1 also uses DSA. **This raises the bar for SinkAware and ThoughtFlow novelty claims** — both projects must explicitly differentiate vs DSA in Phase 1.
- **Deprecation alert**: DeepSeek R1 line retires July 24, 2026 — do not anchor experiments on R1 or its distills.

## Gate philosophy
Cheap → expensive. Each phase has explicit deliverables and kill criteria. Failure at a Macbook phase = pivot or kill *before* spending GPU time.

| Phase | Where | Cost | What |
|---|---|---|---|
| 0 | Macbook | $0 | Setup, repos, env |
| 1 | Macbook | $0 | Literature audit (this is the highest-value gate) |
| 2 | Macbook | $0 | Theory / architecture mapping |
| 3 | Macbook | $0 | NumPy/PyTorch CPU reference impl |
| 4 | Macbook | $0 | Triton kernel skeleton (compiles, doesn't run) |
| **GATE** | — | — | **All Phase 0–4 deliverables present + reviewed** |
| 5 | 5090 | ~$5 | Empirical baseline + project-specific pivot test |
| 6 | 5090 | ~$50 | Prototype kernel |
| 7 | 5090 | ~$200–300 | Optimization + multi-model eval |
| 8 | Macbook + H100 burst | ~$100 | Paper draft + final eval |

## Master schedule

| Week | Activity | Parallelism |
|---|---|---|
| 1 (May 4–10) | Phase 0–1 all projects | 3 agents in parallel |
| 2 (May 11–17) | Phase 2–4 all projects | 3 agents in parallel |
| **End wk 2 (~May 17)** | **GATE DECISION** | Pick survivors |
| 3 (May 18–24) | Phase 5 + start Phase 6 | Survivors only |
| 4 (May 25–31) | Phase 6–7 | |
| 5 (Jun 1–7) | Phase 7 | |
| 6 (Jun 8–14) | Paper draft + ablations | |
| 7 (Jun 15–21) | H100 burst + polish | |
| ~Jun 25 | **Submit COLM** | |

## Decision rules at end of week 2
- **3 survive**: pick the cleanest story for COLM, queue others as ICLR/MLSys follow-ups.
- **2 survive**: parallelize on shared 5090 with queue; pick stronger for COLM.
- **1 survives**: full sprint on it. Bigger paper. Better results.
- **0 survive**: pivot to backup ideas (SpecKernel reasoning-FPGA, MLA-FPGA — see prior conversation).

## Agent allocation
3 agents can work on Macbook, one per project folder, inside this single repo checkout. Each agent:
- Owns its project's Phase 0–4 files under `experimental/<project>/`
- Has read access to other agents' progress through the shared repo
- Reports gate status at each phase end in that project's `progress.md`
- Cannot launch GPU work without explicit gate sign-off

## Cost projections (post-gate, assuming 1–2 projects survive)

| Scenario | Phase 5–7 5090 cost | H100 burst | Total |
|---|---|---|---|
| 1 project, conservative | $250 | $80 | ~$330 |
| 1 project, intensive | $500 | $80 | ~$580 |
| 2 projects, parallel (shared GPU) | $400 | $120 | ~$520 |
| 2 projects, sequential | $700 | $120 | ~$820 |

Add ~$70 for storage (500GB × $0.07/GB × 2 months).

## Critical rules
1. **No GPU spend before gate.** Period.
2. **Kill criteria are non-negotiable.** If a kill criterion triggers, the project ends. No "let me just try one more thing" — those reflexes burn weeks.
3. **Macbook phases are timeboxed.** Phase 1 = 2–3 days, Phase 2 = 1 day, Phase 3 = 2 days, Phase 4 = 1 day. If a phase overruns by more than 50%, escalate / kill.
4. **Deliverables must exist as files in the repo.** Verbal "it's almost done" doesn't count. The gate checks for files.
