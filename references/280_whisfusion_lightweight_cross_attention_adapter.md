# Whisfusion

- Title: `Whisfusion: Parallel ASR Decoding via a Diffusion Transformer`
- Date: 2025-08-09
- Link: https://arxiv.org/abs/2508.07048
- Why it matters here:
  - recent frozen-backbone example of a lightweight cross-attention adapter trained with PEFT between mismatched upstream and downstream modules
  - useful if the next LatentWire branch should become a fuller cross-attention transfer module instead of another local bridge tweak

Most transplantable mechanism:
- attach a compact cross-attention adapter between frozen components and train only that interface, keeping the main backbone untouched

Immediate use in our setting:
- good support for a future Attention Editing / LLM Modules style module-replacement branch built around a stronger attention-side transfer interface
