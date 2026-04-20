# Vision-Language Distillation via Partial Information Decomposition

- Title: `Vision Language Model Distillation Using Partial Information Decomposition`
- Date: 2025-06-11
- Link: https://openreview.net/forum?id=caBO989n7l

Why it matters here:

- It is a useful recent reference for distilling **interaction structure** and
  synergistic information rather than only matching pointwise logits or hidden
  states.
- That maps directly onto the current LatentWire blocker: our small
  prediction-level teachers still look too local and too marginal, even when
  the bridge itself is weakly alive.
- It strengthens the case for a next branch that supervises token-token or
  span-span interaction structure after dynamic remapping rather than another
  top-k KL variant.

Most transplantable mechanism:

- Distill shared and synergistic information between teacher and student
  representations, not just marginal token outputs, so the bridge is pushed to
  preserve higher-order predictive structure.

Immediate use in our setting:

- Keep as a supporting reference for a future dynamic token/span remapping
  branch with an added affinity / interaction loss on top of aligned
  likelihoods.
