# Universal Sparse Autoencoders

- Title: `Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment`
- Date: 2025-02-06
- Link: https://arxiv.org/abs/2502.03714
- Why it matters here:
  - strongest direct reference for learning one shared sparse concept space across multiple models
  - keeps the sparse-dictionary lane alive even after dense grouped transport and dense canonicalization variants plateau

Most transplantable mechanism:
- fit one overcomplete sparse dictionary and align models in that sparse feature basis rather than raw dense head space

Immediate use in our setting:
- use it as the main anchor if the next branch becomes a shared SAE or sparse dictionary KV bridge
