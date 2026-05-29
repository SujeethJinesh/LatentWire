# References Audit

Date: 2026-05-29

Scope: active inline `thebibliography` in `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`. The paper build uses this inline bibliography. `experimental/outlier_migrate/paper/bibliography.bib` remains a broader source manifest and is not used by the build.

## Summary

- Active cited references: 32
- Cited-but-missing keys: 0
- Orphaned active bibliography entries: 0 after this pass
- URL/source resolution: all active source URLs resolved or were confirmed by browser/search. Gartner blocks script HEAD requests but the public source page was verified through browser search.
- Safe fixes applied:
  - Removed eight uncited inline bibliography entries from the active paper bibliography.
  - Corrected the DecDEC prose claim from `roughly 20% recall` to `static-channel recall around 30% or below`.
  - Kept ParoQuant as a prior-work baseline, not as our method.

## Active Reference Rows

| Key | Status | Source checked | Notes |
|---|---|---|---|
| `arai2025qep` | VERIFIED | https://arxiv.org/abs/2504.09629 | Title, authors, arXiv ID, and 2025 venue label checked against arXiv/source page. |
| `wiggers2025reasoningcosts` | VERIFIED | https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-models-is-making-benchmarking-more-expensive/ | Byline/title/date match current citation. |
| `gartner2026inferencecost` | VERIFIED | https://www.gartner.com/en/newsroom/press-releases/2026-03-25-gartner-predicts-that-by-2030-performing-inference-on-an-llm-with-1-trillion-parameters-will-cost-genai-providers-over-90-percent-less-than-in-2025 | Browser/search verification confirms title, date, and over-90% inference-cost forecast. |
| `wang2026mobiquant` | VERIFIED | https://arxiv.org/abs/2602.20191 | arXiv ID resolves and matches title. |
| `park2025decdec` | VERIFIED_FIX_APPLIED | https://www.usenix.org/system/files/osdi25-park-yeonhong.pdf | OSDI 25 source verified. Static-channel recall wording changed to `around 30% or below` to match the paper more safely than the previous `roughly 20%`. |
| `liu2024kivi` | VERIFIED | https://arxiv.org/abs/2402.02750 | arXiv/ICML citation resolves. |
| `liu2025pmkvq` | VERIFIED | https://arxiv.org/abs/2505.18610 | arXiv ID resolves and title matches. |
| `yang2025attentionpredictor` | VERIFIED | https://arxiv.org/abs/2502.04077 | arXiv ID resolves and title matches. |
| `li2025qmamba` | VERIFIED | https://arxiv.org/abs/2501.13624 | arXiv ID resolves and title matches. |
| `ramachandran2025ouromamba` | VERIFIED | https://arxiv.org/abs/2503.10959 | arXiv ID resolves and title matches. |
| `pierro2024mambaptq` | VERIFIED | https://arxiv.org/abs/2407.12397 | arXiv ID resolves and title matches. |
| `chiang2024quamba` | VERIFIED | https://arxiv.org/abs/2410.13229 | arXiv ID resolves and title matches. |
| `chiang2025quamba2` | VERIFIED | https://arxiv.org/abs/2503.22879 | Load-bearing claim checked: source uses channel-order preservation / activation-persistence language. Our prose now scopes this to measured block-output top-K protection. |
| `choi2026laquant` | VERIFIED | https://arxiv.org/abs/2605.08755 | arXiv ID resolves and title matches. |
| `xu2025mambaquant` | VERIFIED | https://arxiv.org/abs/2501.13484 | arXiv ID resolves and title matches. |
| `xiao2023smoothquant` | VERIFIED | https://arxiv.org/abs/2211.10438 | arXiv/ICML source resolves. |
| `lin2023awq` | VERIFIED | https://arxiv.org/abs/2306.00978 | arXiv/MLSys source resolves. |
| `ashkboos2024quarot` | VERIFIED | https://arxiv.org/abs/2404.00456 | arXiv ID resolves and title matches. |
| `chen2025cmpq` | VERIFIED | https://arxiv.org/abs/2410.13056 | Load-bearing claim checked: channel-wise mixed precision uses calibration-time activation distributions; our distinction is decode-time drift. |
| `hooper2024kvquant` | VERIFIED | https://arxiv.org/abs/2401.18079 | arXiv/NeurIPS source resolves. |
| `jang2025blockdialect` | VERIFIED | https://arxiv.org/abs/2501.01144 | arXiv/ICML source resolves. |
| `helcig2026slq` | VERIFIED | https://arxiv.org/abs/2605.02404 | Load-bearing claim checked: task-lossless below 4 bits and distribution-lossless roughly 5--6 bits are supported by source abstract/prose. |
| `liang2026paroquant` | VERIFIED | https://arxiv.org/abs/2511.10645 | arXiv ID resolves; ParoQuant is cited as prior-work baseline. |
| `zlab2026paroquantproject` | VERIFIED | https://z-lab.ai/projects/paroquant/ | Load-bearing Qwen3-4B AIME-24 numbers checked: AWQ 62.2, ParoQuant 73.3, FP16 75.6. |
| `yi2024rrs` | VERIFIED | https://arxiv.org/abs/2409.20361 | Load-bearing distinction checked: runtime smoothing via activation maxima/scales, not per-layer top-K EMA protection. |
| `saxena2025resq` | VERIFIED | https://arxiv.org/abs/2412.14363 | Load-bearing claim checked: low-rank principal-component subspace retained at higher precision. |
| `salfati2026kvquantization` | VERIFIED | https://arxiv.org/abs/2604.11501 | Load-bearing claim checked: source argues projection/direction dropping dominates quantization damage under softmax Fisher metric; prose avoids overstating beyond this. |
| `dong2026outlierdynamics` | VERIFIED | https://arxiv.org/abs/2602.02047 | Training-step outlier persistence claim is used as complementary time-axis framing. |
| `koikeakino2026ttq` | VERIFIED | https://arxiv.org/abs/2603.19296 | Used only for naming disambiguation of test-time TTQ. |
| `liao2026chanmix` | VERIFIED | https://openreview.net/forum?id=yjr2jX41qO | OpenReview entry resolves. Current citation uses public author/title metadata; if the camera-ready page changes anonymity metadata, update before submission. |
| `xu2026activationsensitivity` | VERIFIED | https://arxiv.org/abs/2601.11663 | Load-bearing claim checked: activation sensitivity is a formal principle for PTQ. |
| `zhang2025mixkvq` | VERIFIED | https://arxiv.org/abs/2512.19206 | arXiv ID resolves and title matches. |

## Attributed-Claim Checks

| Claim | Status | Action |
|---|---|---|
| Quamba2 channel-order preservation and activation persistence | VERIFIED | Prose scopes comparison to block-output top-K protection and avoids saying Quamba2 is globally wrong. |
| DecDEC static recall | FIX_APPLIED | Changed from `roughly 20%` to `around 30% or below` against per-step ground truth. |
| ParoQuant Qwen3-4B AIME numbers | VERIFIED | Project page supports AWQ 62.2 / ParoQuant 73.3 / FP16 75.6. |
| SLQ task-lossless/distribution-lossless bit claims | VERIFIED | Current prose says `task-lossless below 4 bits` and `distribution-lossless at roughly 5--6 bits`, matching source scope. |
| Salfati Fisher-metric direction dropping claim | VERIFIED | Current prose uses the claim only as support for hard direction dropping being worse than quantization under that metric. |
| ResQ PCA/low-rank subspace claim | VERIFIED | Current prose differentiates PCA-basis subspace preservation from channel-basis top-K drift. |
| Activation Sensitivity formalization | VERIFIED | Current prose cites it as a formalization of perturbation impact, not as a long-decode drift result. |

No claim-support problem requiring human decision was found in this pass.
