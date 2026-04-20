# HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling

- Title: `HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling`
- Date: `2026-03-10`
- Link: `https://arxiv.org/abs/2603.16917`

## Why it matters

- It replaces discrete token matching with a continuous tokenizer-free
  interface.
- That is directly relevant to LatentWire because the strongest remaining live
  lane now looks like it needs a more tokenizer-agnostic supervision target.

## Transplantable idea

- Use a shared continuous interface before the bridge:
  - distill source and target behavior into a tokenizer-independent latent
    target,
  - then evaluate whether the current transport and bridge stack works better
    on top of that shared space.

## Use in our stack

- Best fit as a longer-horizon pivot after the byte/span control:
  - treat it as evidence that a tokenizer-agnostic target is plausible,
  - use it to justify a byte/shared-interface teacher branch before any larger
    architectural pivot.
