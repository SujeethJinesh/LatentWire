# Transport and Merge

- Title: `Transport and Merge: Cross-Architecture Merging for Large Language Models`
- Date: 2026-02-05
- Link: https://arxiv.org/abs/2602.05495
- Why it matters here:
  - strongest recent direct precedent for using transport plans to align heterogeneous model components before a stronger merge/interface step
  - useful if the next LatentWire pivot becomes a more explicit correspondence module over grouped KV bundles rather than another tiny bridge

Most transplantable mechanism:
- use optimal-transport-style cross-component correspondences as the explicit interface object, then learn a small module on top of that aligned correspondence space

Immediate use in our setting:
- anchor a next branch that replaces ad hoc grouped matching with an explicit learned correspondence step before any bridge/projector
