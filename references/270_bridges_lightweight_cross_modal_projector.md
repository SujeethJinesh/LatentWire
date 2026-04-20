# BRIDGES

- Title: `BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks`
- Date: 2025-04-07
- Link: https://arxiv.org/abs/2504.05180
- Why it matters here:
  - clean recent example of solving a hard modality/interface mismatch with a lightweight projector on top of a frozen backbone
  - useful if the next LatentWire branch becomes a tiny projector module after frozen grouped transport rather than another residual bridge

Most transplantable mechanism:
- insert a small trainable projector between frozen source-side representations and the target model interface, and supervise it directly rather than only patching residuals

Immediate use in our setting:
- anchor a next branch that adds a small post-transport projector trained against attention/logit/readout targets
