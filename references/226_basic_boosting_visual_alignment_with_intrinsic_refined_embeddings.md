Title: BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models

Date:
- 2025-08-09

Link:
- https://arxiv.org/abs/2508.06895

Why it matters here:
- This is a useful bridge/projector reference because it trains a small
  alignment module with stronger internal supervision than plain latent
  regression.
- The transferable idea for LatentWire is to keep the main backbone frozen and
  supervise a tiny bridge with both direct internal-state alignment and a
  distribution-matching objective.

Most transplantable mechanism:
- Pair a small projector with direct representation alignment plus logit-style
  distillation, rather than relying on hidden-state regression alone.

Immediate use in our setting:
- Keep the transport path frozen.
- Replace the current latent-only bridge loss with a two-part target:
  internal interaction alignment plus a teacher/student output-matching term.

