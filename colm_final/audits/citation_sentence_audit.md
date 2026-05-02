# Citation and Sentence Audit

Date: 2026-05-02

## Summary

The paper's citations were checked against primary sources where available:
OpenReview, ACL Anthology, PMLR, NeurIPS proceedings, arXiv, and IEEE metadata
for the information-theory papers. Several BibTeX entries were corrected in
`paper/latentwire_colm2026.bib` and mirrored in `colm/2026`.

No checked citation currently supports an over-broad claim such as universal
latent language, solved cross-family communication, or native serving
throughput. The paper text explicitly avoids those claims.

## Corrected Metadata

| Key | Correct source | What was verified |
|---|---|---|
| `Fu2025C2C` | `https://openreview.net/forum?id=LeatkxrBCi` | ICLR 2026 poster; authors Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang. |
| `Shi2025KVComm` | `https://openreview.net/forum?id=F7rUng23nw` | ICLR 2026 poster; authors Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., Dejan Kostic. |
| `Zandieh2025TurboQuant` | `https://openreview.net/forum?id=tO3ASKZlok` | ICLR 2026 poster; authors Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. |
| `Zheng2023SGLang` | `https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html` | NeurIPS 2024 proceedings; DOI 10.52202/079017-2000. |
| `Li2021Prefix` | `https://aclanthology.org/2021.acl-long.353/` | ACL-IJCNLP 2021 long paper; pages 4582-4597; DOI 10.18653/v1/2021.acl-long.353. |
| `Mu2023Gist` | `https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html` | NeurIPS 2023 proceedings. |
| `Cunningham2023SAE` | `https://openreview.net/forum?id=F76bwRSLeK` | ICLR 2024 OpenReview record; author order Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, Lee Sharkey. |
| `Li2023BLIP2` | `https://proceedings.mlr.press/v202/li23q.html` | ICML/PMLR 2023; pages 19730-19742. |
| `Liu2024KIVI` | `https://proceedings.mlr.press/v235/liu24bz.html` | ICML/PMLR 2024; pages 32332-32344. |
| `Hooper2024KVQuant` | `https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html` | NeurIPS 2024 proceedings, DOI 10.52202/079017-0040, and author spelling. |
| `Karvonen2025SAEBench` | `https://proceedings.mlr.press/v267/karvonen25a.html` | ICML/PMLR 2025; full title, author list, and pages 29223-29264. |
| `Engels2024SAEUniversality` | `https://arxiv.org/abs/2410.06981` | Title and author list for SAE feature-space universality. |
| `Kwon2023vLLM` | `https://dl.acm.org/doi/10.1145/3600006.3613165` | SOSP 2023 proceedings; pages 611-626; DOI 10.1145/3600006.3613165. |
| `Mihaylov2018OpenBookQA` | `https://aclanthology.org/D18-1260/` | EMNLP 2018 paper; pages 2381-2391; DOI 10.18653/v1/D18-1260. |

## Sentence-Level Claim Boundaries

| Paper section | Citation use | Audit result |
|---|---|---|
| Introduction, cache communication | C2C and KVComm are cited only as cache/KV communication systems. | Correct. The paper does not claim to outperform them natively. |
| Introduction, side information | Slepian-Wolf and Wyner-Ziv motivate decoder-side-information framing. | Correct as analogy; paper does not claim a formal rate-distortion theorem. |
| Datasets/model families | ARC, OpenBookQA, Qwen2.5, Phi-3 cite benchmark and model-family sources. | Correct. |
| Public-coordinate packets | Relative Representations supports the anchor/coordinate-system prior. | Correct; paper explicitly says it does not claim zero-shot latent stitching. |
| Related work, SAE/common features | SAE papers motivate future common-feature interfaces; SAEBench is cited only as evaluation context. | Correct; paper does not claim an SAE result or SAE universality from SAEBench. |
| Prompt compression | Prefix-Tuning and Gist Tokens support continuous/prompt-compression comparison. | Correct; paper distinguishes byte packets from soft prompts. |
| Query bottleneck | BLIP-2 supports Q-Former/query-bottleneck inspiration only. | Correct; paper says this branch is not closed. |
| Systems boundary | vLLM/SGLang and KV quantizers are serving/cache-state neighbors. | Correct; paper states the systems result is accounting, not throughput. The 768B row is worded as an internal 1-bit-per-KV-element accounting floor, not a QJL-native claim. |

## Desk-Reject Risks Checked

- Author spellings were checked for the cited high-risk recent works.
- Conference years/venues were corrected for recent 2024-2026 proceedings.
- The BibTeX compiles with the COLM bibliography style after corrections.
- The audit intentionally does not rely on tertiary summaries for final
  metadata.
- Remaining scientific risk is not citation metadata; it is claim scope. The
  paper now explicitly says the packet mostly preserves source choice and does
  not beat explicit source-index communication.
- A stale internal comparator memo (`references/442_benchmark_pairing_uncertainty_refs.md`)
  was corrected to use the OpenReview titles for C2C and KVComm.
