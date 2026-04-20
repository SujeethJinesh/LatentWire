# Mapping Visual Representations to Language Feature Space

- Title: `How Visual Representations Map to Language Feature Space in Multimodal LLMs`
- Date: 2025-06-13
- Link: https://arxiv.org/abs/2506.11976

Why it matters here:

- It is a direct frozen-backbone reference for learning a **small adapter into
  an existing language feature space** rather than training a large fusion
  stack.
- That is closely aligned with the current LatentWire question of whether the
  bridge should predict free dense KV outputs or instead land in a more
  target-native basis.
- It also supports the current token-basis and LangBridge-style intuition that
  a portable interface may need to be anchored to the target model's native
  basis rather than only to geometric transport.

Most transplantable mechanism:

- Freeze the backbone models and learn a compact adapter that lands foreign
  representations directly into the target model's language-native feature
  manifold.

Immediate use in our setting:

- Keep as a reference for the token-basis and vocabulary-grounded interface
  lane, especially if we pivot from local KV correction to a target-native
  basis or vocab-anchored replacement module.
