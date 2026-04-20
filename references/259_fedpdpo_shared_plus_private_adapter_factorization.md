# FedPDPO

- Title: `FedPDPO: Federated Personalized Direct Preference Optimization for Large Language Model Alignment`
- Date: 2026-03-20
- Link: https://arxiv.org/abs/2603.19741
- Why it matters here:
  - clean recent example of a frozen backbone with a globally shared adapter plus private low-rank personalization modules
  - useful support for treating cross-model communication as a shared-plus-private interface problem rather than one monolithic bridge

Most transplantable mechanism:
- factor the interface into a reusable shared adapter and small private residual modules that capture pair-specific behavior

Immediate use in our setting:
- cite it as a modular-interface reference if the next dense bridge step keeps a shared bridge plus pair-specific residual heads
