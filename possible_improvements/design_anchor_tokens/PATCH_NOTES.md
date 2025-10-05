
# Patch Notes (Anchor Tokens)
- Insert learned anchor tokens at segment boundaries.
- Wrap attention mask construction with anchor-aware masking.
- At inference, compress KV by selecting only anchors.
- Compare F1 @ equal byte budgets vs token-budget baseline.
