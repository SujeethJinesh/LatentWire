## AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components

- Title: `AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components`
- Link: https://arxiv.org/abs/2403.00745
- Why it matters here:
  - useful interpretability reference for paired-flip and component-attribution artifacts in the paper
  - supports moving beyond aggregate accuracy into causal evidence about which translated layers or heads actually change outcomes

Most transplantable mechanism:
- component-level localization on paired successes and failures instead of only reporting overall accuracy deltas

Immediate use in our setting:
- add paired-flip analyses on matched examples
- use per-layer or per-head interventions as reviewer-facing evidence for where the transport or bridge actually helps or hurts
