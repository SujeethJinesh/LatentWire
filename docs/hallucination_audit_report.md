# COLM LLM Policy and Hallucination Audit

Date: 2026-05-26

This audit verifies the paper's LLM-use disclosure, bibliography, attributed prior-work claims, headline numbers, and named entities. Sources were checked against primary URLs when possible: arXiv abstracts, USENIX/OpenReview pages, Hugging Face model cards, project pages, and publisher pages. No unverified bibliography entries remain after the fixes listed below.

## LLM Use Disclosure

Status: PASS

The paper now includes an explicit `LLM Use Disclosure` section. It discloses substantive use of Anthropic Claude and Codex for paper drafting, figure-code generation, simulated peer review, artifact summarization, method ideation/prioritization, experiment/release code assistance, and verification scripting. The statement also records that final authorization, numerical interpretation, and submission framing remain the human author's responsibility.

## Reference Verification

| Citation | Status | Source | Notes |
|---|---|---|---|
| `arai2025qep` | PASS | https://arxiv.org/abs/2504.09629 | Title/authors/arXiv ID verified. |
| `wiggers2025reasoningcosts` | FIX_APPLIED | https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/ | Corrected author from Julie Bort to Kyle Wiggers and updated key. |
| `gartner2026inferencecost` | PASS | https://www.gartner.com/en/newsroom/press-releases/2026-03-25-gartner-predicts-that-by-2030-performing-inference-on-an-llm-with-1-trillion-parameters-will-cost-genai-providers-over-90-percent-less-than-in-2025 | Title/date/source verified. |
| `marketsandmarkets2025aiinference` | FIX_APPLIED | https://www.marketsandmarkets.com/Market-Reports/ai-inference-market-189921964.html | Corrected title to `AI Inference Market Size, Share & Trends, 2025 To 2030`. |
| `marketsandmarkets2025edgeaihardware` | PASS | https://www.marketsandmarkets.com/PressReleases/edge-ai-hardware.asp | Title/date/source verified. |
| `wang2026mobiquant` | PASS | https://arxiv.org/abs/2602.20191 | Title/authors/arXiv ID verified. |
| `park2025decdec` | PASS | https://www.usenix.org/conference/osdi25/presentation/park-yeonhong | Title/authors/OSDI source verified. |
| `liu2024kivi` | PASS | https://arxiv.org/abs/2402.02750 | Title/authors/venue verified. |
| `liu2025pmkvq` | PASS | https://arxiv.org/abs/2505.18610 | Title/authors/arXiv ID verified. |
| `yang2025attentionpredictor` | PASS | https://arxiv.org/abs/2502.04077 | Title/authors/arXiv ID verified. |
| `chen2025pmpd` | PASS | https://arxiv.org/abs/2410.13461 | Title/authors/arXiv ID verified. |
| `li2025qmamba` | PASS | https://arxiv.org/abs/2501.13624 | Title/authors/arXiv ID verified. |
| `ramachandran2025ouromamba` | PASS | https://arxiv.org/abs/2503.10959 | Title/authors/arXiv ID verified. |
| `pierro2024mambaptq` | PASS | https://arxiv.org/abs/2407.12397 | Title/authors/arXiv ID verified. |
| `chiang2024quamba` | PASS | https://arxiv.org/abs/2410.13229 | Title/authors/arXiv ID verified. |
| `chiang2025quamba2` | PASS | https://arxiv.org/abs/2503.22879 | Title/authors/arXiv ID verified. |
| `choi2026laquant` | PASS | https://arxiv.org/abs/2605.08755 | Title/authors/arXiv ID verified. |
| `xu2025mambaquant` | PASS | https://arxiv.org/abs/2501.13484 | Title/authors/arXiv ID verified. |
| `chen2026quambase` | PASS | https://arxiv.org/abs/2601.09451 | Reference exists; no main-body claim currently depends on it. |
| `xiao2023smoothquant` | PASS | https://arxiv.org/abs/2211.10438 | Title/authors/ICML context verified. |
| `lin2023awq` | PASS | https://arxiv.org/abs/2306.00978 | Title/authors/MLSys context verified. |
| `ashkboos2024quarot` | PASS | https://arxiv.org/abs/2404.00456 | Title/authors/arXiv ID verified. |
| `hooper2024kvquant` | PASS | https://arxiv.org/abs/2401.18079 | Title/authors/NeurIPS context verified. |
| `jang2025blockdialect` | PASS | https://arxiv.org/abs/2501.01144 | Title/authors/ICML context verified. |
| `helcig2026slq` | PASS | https://arxiv.org/abs/2605.02404 | Title/authors/arXiv ID verified. |
| `liang2026paroquant` | FIX_APPLIED | https://arxiv.org/abs/2511.10645 | Source verified; venue/year label corrected to ICLR 2026. |
| `zlab2026paroquantproject` | PASS | https://z-lab.ai/projects/paroquant/ | Project page and cited AIME-24 numbers verified. |
| `yi2024rrs` | PASS | https://arxiv.org/abs/2409.20361 | Title/authors/arXiv ID verified. |
| `dong2026outlierdynamics` | PASS | https://arxiv.org/abs/2602.02047 | Title/authors/arXiv ID verified. |
| `koikeakino2026ttq` | FIX_APPLIED | https://arxiv.org/abs/2603.19296 | Added explicit citation for test-time TTQ disambiguation. |
| `liao2026chanmix` | PASS | https://openreview.net/forum?id=yjr2jX41qO | OpenReview title/authors/ICLR 2026 poster verified. |
| `qwen36hf` | PASS | https://huggingface.co/Qwen/Qwen3.6-35B-A3B | Model-card entity verified. |
| `xu2026activationsensitivity` | PASS | https://arxiv.org/abs/2601.11663 | Title/author/arXiv ID verified. |
| `yang2023gla` | PASS | https://arxiv.org/abs/2312.06635 | Title/authors/arXiv ID verified. |
| `yang2024gdn` | PASS | https://arxiv.org/abs/2412.06464 | Title/authors/arXiv ID verified. |
| `kimilinear2025` | PASS | https://arxiv.org/abs/2510.26692 | Title and Kimi Team authorship verified. |
| `zhang2025mixkvq` | PASS | https://arxiv.org/abs/2512.19206 | Title/authors/arXiv ID verified. |

## Attributed Claim Verification

| Claim area | Status | Action |
|---|---|---|
| Quamba2 channel-order/channel-persistence claim | FIX_APPLIED | Reworded to source-aligned claim: channel-order preservation in selective-SSM computation and channel/state persistence for SSM activations, used for sort-and-cluster reordering and per-state-group quantization. |
| DecDEC static-recall claim | PASS | Source supports dynamic per-step salient-channel selection and low static recall near 20% over short decode. |
| ParoQuant rotation/reasoning claim | FIX_APPLIED | Reworded as 4-bit weight-only PTQ with scaled independent Givens/pairwise rotations plus channel-wise scaling; retained project-page AIME-24 numbers. |
| Rotated Runtime Smooth adjacency | FIX_APPLIED | Reworded as runtime channel/group-wise activation maxima used as smoothing scales in fused GEMM, not per-token magnitude broadcasting. |
| HCP/CHON hot-channel terminology | PASS | Source supports persistently extreme channels during NVFP4 pretraining. |
| test-time TTQ disambiguation | FIX_APPLIED | Added a real citation to arXiv:2603.19296. |
| SLQ counter-result | FIX_APPLIED | Reworded to task-lossless below 4 bits per parameter and distribution-lossless at roughly 5--6 bits per parameter under its searched quantization regime. |
| Quamba-SE | PASS | Reference exists; no active paper claim uses it beyond bibliography context. |

## Numerical Claim Verification

| Number or claim | Status | Source artifact | Action |
|---|---|---|---|
| Four-model final set-leaving values | PASS | Phase 1/2/5-prime/7 migration decomposition artifacts | Exact values round to Granite 0.566, Nemotron 0.534, DeepSeek 0.671, Falcon 0.674. |
| “Increasing strict set-leaving across decode position” | FIX_APPLIED | Raw activation packets via figure-generation script | Reworded to “not monotone at every intermediate point, but large by the final measured position.” |
| Method medians/CIs/control margins | PASS | Phase 9 checker/control JSON artifacts | Main method table values match source artifacts within rounding. |
| Nemotron top-10 recovery | PASS | Nemotron static-1% salvage checker result | Reported as 0.815 in tables/text; exact value is 0.814739798903. |
| Granite M11b and ParoQuant values | PASS | Granite M11b and ParoQuant checker/metrics JSON | Exact medians and margins match rounded paper values. |
| No-gap fractions | PASS | Per-trace metrics JSON | Granite static-top-1% statement reworded to exclude M10's different static-SmoothQuant denominator. |
| KL means and AR decays | PASS | Dense KL summary/fits JSON | Means 0.150/0.133/0.131 and AR decays 0.51/0.45/0.44 match source artifacts. |
| KL sampling grid | FIX_APPLIED | `kl_positions.json` and preregistration | Replaced incorrect “every 100 positions to 10K” wording with the actual 921-position fallback grid through 20K. |
| Layer-type drift/shuffling values | PASS | `phase3/results/layer_stratified_migration.md` | Paper table and figure match the rounded layer-stratified artifact. |
| Model snapshot commits | PASS | Model provenance JSON files | Prefixes match paper reproducibility table. |

## Entity Verification

| Entity | Status | Action |
|---|---|---|
| Granite-4.0-H-Small | PASS | Official spelling retained. |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | FIX_APPLIED | Full checkpoint name added at first use; later `Nemotron-3` shorthand retained. |
| DeepSeek-R1-Distill-Qwen-1.5B | PASS | Official spelling retained. |
| Falcon-H1 / Falcon-H1-0.5B-Instruct | PASS | Family shorthand retained; checkpoint appears in reproducibility table. |
| Qwen3.6-35B-A3B | PASS | Official Hugging Face model-card spelling retained. |
| AIME / MATH-500 / GPQA Diamond | PASS | No unsupported main-body claims added in this pass. |
| Quamba2, MambaQuant, ParoQuant, DecDEC, Rotated Runtime Smooth | PASS | Official spellings retained. |

## Outcome

All bibliography entries resolve to real sources after the applied fixes. Attributed prior-work wording was tightened where the previous prose was broader than the source. Headline experimental numbers were rechecked against local artifacts; the audit changed wording and rounding, not the underlying results.
