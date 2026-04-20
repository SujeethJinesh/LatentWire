# Mass-Editing Memory with Attention

- Title: `Mass-Editing Memory with Attention in Transformers: A cross-lingual exploration of knowledge`
- Date: 2025-02-04
- Link: https://arxiv.org/abs/2502.02173
- Why it matters here:
  - recent evidence that attention-side editing can move model behavior more directly than small residual corrections in latent space
  - useful support for a future LatentWire branch that treats the bridge as a module/interface replacement rather than another additive repair layer

Most transplantable mechanism:
- learn a compact attention-side edit that directly rewrites the effective retrieval pathway instead of only nudging the post-transport representation

Immediate use in our setting:
- keep as a supporting reference for the next explicit module-replacement / attention-editing branch after the current residual, projector, and small xattn lanes saturated
