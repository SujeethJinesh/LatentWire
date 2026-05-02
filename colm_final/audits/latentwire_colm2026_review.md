# LatentWire COLM 2026 Draft Review

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM is close after this draft; ICLR remains blocked by cross-family and native systems evidence.
- Current story: fixed-byte source-private same-family packets mostly preserve the source-selected candidate and improve ARC-Challenge and OpenBookQA receivers while strict cross-family and learned cached repairs fail.
- Exact gap: before submission, do one human author pass for emphasis, anonymity, and final venue-specific metadata; the main scientific gap is a direct source-choice/index baseline.

## Template and build review

- Used the provided COLM 2026 zip template under `colm/2026/`.
- Draft source: `latentwire_colm2026.tex`.
- Bibliography: `latentwire_colm2026.bib`.
- Figures: `figures/protocol_diagram.pdf`, `figures/accuracy_overview.pdf`, `figures/systems_boundary.pdf`.
- Compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`.
- Recompiled after regenerating figures with opaque white backgrounds to avoid black-page rendering under Ghostscript.
- Final checked build: `latentwire_colm2026.pdf`, 9 pages including references and appendix.
- Final log check: no natbib warnings, undefined citations, undefined references,
  or overfull boxes. The only remaining typography notes are underfull boxes
  from long path/bibliography wrapping.

## Claim review

Safe main claims:

- ARC-Challenge test: 8B payload / 11B framed packet, 10/10 seeds, matched 0.344, target 0.265, same-byte text 0.300, minimum paired CI95 lower bound vs target +0.038.
- OpenBookQA test: 3B payload / 6B framed packet, 5/5 seeds, matched 0.378, target 0.276, same-byte text 0.350, minimum paired CI95 lower bound vs target +0.038.
- ARC destructive coordinate controls fail on test in 0/10 seeds.
- Phi-3 strict cross-family replacement fails; Qwen-disagreement packet is 0.200 vs Qwen-substituted packet 0.340.
- Systems artifact supports byte/exposure accounting only: 6-11 framed packet bytes versus a 768B one-token 1-bit-per-KV-element accounting floor; no native GPU speed claim.

Claims intentionally avoided:

- No universal latent language claim.
- No solved cross-family transfer claim.
- No hidden-state, KV-cache, or source-memory transfer claim.
- No C2C/KVComm/TurboQuant/QJL superiority claim.
- No claim to beat explicit source-index/source-choice communication.
- No native GPU throughput, TTFT, TPOT, goodput, or HBM claim.

## Citation review

Primary/official citation pages were checked for the works used in the draft:

- Relative Representations, OpenReview ICLR 2023.
- ARC arXiv page and OpenBookQA ACL Anthology page.
- Qwen2.5 and Phi-3 arXiv pages.
- Prefix-Tuning, Gist Tokens, BLIP-2/Q-Former.
- C2C, KVComm, QJL, KIVI, KVQuant, TurboQuant, vLLM/PagedAttention, and SGLang.
- Slepian-Wolf and Wyner-Ziv are stable information-theory background citations.

## Remaining submission risks

- The paper is credible as a conservative COLM submission, but not as an ICLR full-paper claim.
- Reviewers may ask whether the same-family positive is too narrow or simply source-choice transfer. The answer in the draft is to frame the method as a positive packet protocol plus falsification ladder, not general latent communication.
- Native systems rows are explicitly pending. A systems-focused reviewer may want NVIDIA serving measurements; this is the largest missing item for a stronger later version.
- The Qwen2.5-1.5B diagnostic is encouraging but validation-incomplete; it is correctly kept as diagnostic rather than headline.

## Re-review decision

The current PDF is compile-ready and reviewer-honest for COLM after one human author pass. The draft should not be expanded with more speculative cross-family language unless new experiments land.
