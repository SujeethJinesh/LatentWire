## MOSA: Mixtures of Simple Adapters Outperform Monolithic Approaches in LLM-based Multilingual ASR

- Title: `MOSA: Mixtures of Simple Adapters Outperform Monolithic Approaches in LLM-based Multilingual ASR`
- Link: https://arxiv.org/abs/2508.18998
- Why it matters here:
  - useful if the core issue is not teacher quality alone but the brittleness of one monolithic low-rank bridge
  - directly supports the hypothesis that a small bank of simple adapters can beat one more complicated adapter

Most transplantable mechanism:
- mixture of simple low-rank adapters with a lightweight router

Immediate use in our setting:
- upgrade the current bridge family from one residual adapter or one routed bank to a slightly richer modular mixture with simple experts
