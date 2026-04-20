## AsymLoRA: Unlocking the Power of Multimodal LLMs via Asymmetric LoRA

- Title: `AsymLoRA: Unlocking the Power of Multimodal LLMs via Asymmetric LoRA`
- Link: https://openreview.net/forum?id=E2T8wulSb9
- Why it matters here:
  - useful modular-adapter reference when one monolithic bridge looks too brittle and we want a shared bridge plus smaller asymmetric correction paths
  - supports a design where most transport is shared but a small residual branch specializes by role or layer

Most transplantable mechanism:
- shared projection plus task- or role-specific low-rank residual branches

Immediate use in our setting:
- keep grouped transport frozen
- replace one monolithic query-conditioned bridge with a shared base bridge plus a few smaller role-specific residual adapters
