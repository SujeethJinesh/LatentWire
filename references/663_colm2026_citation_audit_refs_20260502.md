# COLM 2026 Citation Audit References

Date: 2026-05-02

## Purpose

This memo records the paper-facing citation set used in
`colm/2026/latentwire_colm2026.tex` and the claim boundary each source is
allowed to support.

## Datasets and model families

- ARC-Challenge: `https://arxiv.org/abs/1803.05457`
  - Supports the benchmark description and citation for ARC-Challenge.
- OpenBookQA: `https://aclanthology.org/D18-1260/`
  - Supports the benchmark description and citation for OpenBookQA.
- Qwen2.5 technical report: `https://arxiv.org/abs/2412.15115`
  - Supports the Qwen2.5 same-family model-family citation.
- Phi-3 technical report: `https://arxiv.org/abs/2404.14219`
  - Supports the Phi-3 cross-family/source-family falsification citation.

## Representation and connector prior work

- Relative Representations: `https://openreview.net/forum?id=SrC-nwieGJ`
  - Supports the anchor/relative-coordinate prior-art boundary.
  - Does not support claiming LatentWire is zero-shot model stitching.
- Sparse Autoencoders: `https://openreview.net/forum?id=F76bwRSLeK`
  - Supports feature dictionary motivation only.
  - Verified against the ICLR 2024 OpenReview record: Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, Lee Sharkey.
- SAE universality: `https://arxiv.org/abs/2410.06981`
  - Supports common-feature plausibility only.
  - Verified title and author list in BibTeX: Michael Lan, Philip Torr, Austin Meek, Ashkan Khakzar, David Krueger, Fazl Barez.
- SAEBench: `https://proceedings.mlr.press/v267/karvonen25a.html`
  - Supports sparse-feature evaluation caution only.
  - Verified full BibTeX author list from the ICML/PMLR 2025 record after audit correction.
- Prefix-Tuning: `https://aclanthology.org/2021.acl-long.353/`
  - Supports the continuous-prefix baseline family.
- Gist Tokens: `https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html`
  - Supports prompt-compression/soft-token baseline framing.
  - Verified proceedings author spelling: Jesse Mu, Xiang Li, Noah Goodman.
- BLIP-2 / Q-Former: `https://proceedings.mlr.press/v202/li23q.html`
  - Supports query-bottleneck connector inspiration; not claimed as a completed LatentWire result.

## Systems and communication competitors

- Cache-to-Cache: `https://openreview.net/forum?id=LeatkxrBCi`
  - Supports cache/KV communication competitor framing.
  - Does not support claiming LatentWire beats C2C natively.
- KVComm: `https://openreview.net/forum?id=F7rUng23nw`
  - Supports selective KV-sharing competitor framing.
  - Does not support claiming LatentWire beats KVComm natively.
- QJL: `https://arxiv.org/abs/2406.03482`
  - Supports the idea that one-bit KV quantization/sketching is an aggressive KV-state compression neighbor.
  - In the paper, the 768B row is our internal one-token 1-bit-per-KV-element accounting floor, not a claim about QJL's end-to-end native performance.
- KIVI: `https://proceedings.mlr.press/v235/liu24bz.html`
  - Supports a 2-bit KV-cache quantization comparator floor.
- KVQuant: `https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html`
  - Supports sub-4-bit KV-cache quantization comparator framing.
  - Verified author list after audit correction: Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami.
- TurboQuant: `https://openreview.net/forum?id=tO3ASKZlok`
  - Supports vector/KV quantization comparator framing.
- vLLM/PagedAttention: `https://dl.acm.org/doi/10.1145/3600006.3613165`
  - Supports native serving substrate context and KV-cache memory motivation.
  - Verified against SOSP 2023 proceedings: pages 611--626 and DOI 10.1145/3600006.3613165.
- SGLang: `https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html`
  - Supports structured LLM serving/KV-reuse substrate context.

## Metadata corrections made in this audit pass

- C2C, KVComm, and TurboQuant were updated from preprint-style metadata to their ICLR 2026 OpenReview records.
- Prefix-Tuning was updated to the ACL-IJCNLP 2021 anthology record, including pages and DOI.
- Gist Tokens was updated to the NeurIPS 2023 proceedings record.
- BLIP-2 and KIVI were updated to their PMLR ICML records.
- KVQuant and SGLang were updated to NeurIPS 2024 proceedings records.
- OpenBookQA was updated to the EMNLP 2018 ACL Anthology record, including pages and DOI.
- SAEBench was updated to the ICML/PMLR 2025 record.
- Sparse Autoencoders was updated to the ICLR 2024 OpenReview record, with author order corrected to Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, Lee Sharkey.
- Gist Tokens author spelling was corrected to Xiang Li in the NeurIPS 2023 proceedings entry.
- vLLM/PagedAttention was updated to the SOSP 2023 proceedings entry, including pages 611--626 and DOI 10.1145/3600006.3613165.
- The QJL-related systems row was reworded as an internal 1-bit-per-KV-element accounting floor rather than a QJL-native performance row.

## Information theory

- Slepian-Wolf, 1973.
  - Supports distributed source coding of correlated sources.
- Wyner-Ziv, 1976.
  - Supports rate-distortion with decoder side information.

## Claim boundary

The citation set supports a conservative COLM story:

- fixed-byte source-private packet evidence transfer;
- public-coordinate packet relation to relative representations;
- systems byte/exposure accounting against KV/cache-state objects;
- negative cross-family and connector evidence.

It does not support claiming:

- universal latent language;
- solved cross-family communication;
- native GPU throughput gains;
- superiority over C2C, KVComm, QJL, KIVI, KVQuant, TurboQuant, vLLM, or SGLang.
