Title: IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model with Multimodal Capabilities

Date:
- 2024-08-23

Link:
- https://arxiv.org/abs/2408.12902

Why it matters here:
- This is a useful “tiny adapter on a frozen backbone” reference for our
  bridge lane.
- The transferable idea is that a small learned module can correct a frozen
  model if it is inserted at the right internal interface instead of trying to
  retrain the whole stack.

Most transplantable mechanism:
- Use lightweight internal adapters rather than replacing the main transport
  path, and let those adapters learn the residual mismatch that the frozen map
  cannot remove.

Immediate use in our setting:
- Keep grouped-subspace transport fixed.
- Let a tiny learned residual adapter or projector correct only the remaining
  cross-model KV mismatch at the translator interface.
