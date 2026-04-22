# Query Resampler / Connector References

Date: `2026-04-22`

## Why This Memo Exists

The current same-pair story keeps pointing in one direction:

- direct basis surgery is narrow and fragile
- the only real lift is still dynalign-specific
- tokenization and interface mismatch become real immediately on broader pairs

That makes learned connectors the clearest lateral branch.

## Strongest Learned Connector References

- [BLIP-2](https://arxiv.org/abs/2301.12597)
  Canonical frozen-backbone + learned Q-Former bridge; best first blueprint.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Perceiver Resampler plus gated cross-attention; strongest classic latent
  bottleneck precedent.
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
  Instruction-aware query bridge; relevant if the connector should react to the
  reasoning prompt itself.
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
  General latent interface for arbitrary structured input/output; useful
  precursor for a learned transport space.
- [Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)
  Strong recent example of a richer learned connector that uses multi-layer
  features.
- [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](https://arxiv.org/abs/2405.20797)
  Useful if the connector must align more than one embedding or layer family.
- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)
  Strong vocabulary-adaptation reference when tokenizer mismatch is the visible
  blocker.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
  Cleanest tokenizer-agnostic transport reference; strongest shared byte-level
  interface argument.
- [SelfCP: Compressing Over-Limit Prompt via the Frozen Large Language Model Itself](https://arxiv.org/abs/2405.17052)
  Good analogue for turning long or dense context into a short learned memory
  interface.
- [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948)
  Clean bridge-model reference if one-shot projection proves too brittle.

## Smallest Decisive Experiment

Build a frozen 16-query connector:

1. source side stays frozen
2. target side stays frozen
3. a Q-Former or Perceiver-style resampler reads source states and emits a
   fixed-size query bottleneck
4. the target consumes those query tokens through one learned connector path
5. compare against the current live `dynalign_module_replace_residrank16` row
   on the larger frozen same-pair campaign, then on one matched cross-family
   pair

Keep the branch only if it:

- beats or clearly stabilizes the live dynalign residual lane
- survives `3` seeds
- does not collapse under the first cross-family pair

## Why This Branch Still Looks Alive

1. It is output-aware by construction.
2. It does not require a globally shared latent basis.
3. It naturally supports tokenizer-agnostic or byte-level transport.
4. It fits the frozen-backbone regime better than repeated analytic basis
   tweaks.
