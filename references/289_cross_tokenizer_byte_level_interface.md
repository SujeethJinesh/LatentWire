# Cross-Tokenizer LLM Distillation through a Byte-Level Interface

- Title: `Cross-Tokenizer LLM Distillation through a Byte-Level Interface`
- Date: `2026-04-08`
- Link: `https://arxiv.org/abs/2604.07466`

## Why it matters

- It proposes a tokenizer-agnostic shared interface by distilling through
  byte-level probabilities instead of forcing teacher and student token IDs to
  match.
- That is directly relevant to the current blocker in LatentWire: the live
  lane now looks upstream of local bridge parameterization and increasingly
  tokenization-sensitive.

## Transplantable idea

- Build a shared byte- or character-span teacher before the local bridge:
  - align source and target supervision in a tokenizer-invariant space,
  - then fit the bridge on top of that shared interface.

## Use in our stack

- Best fit as a next-step extension of `dynalign`:
  - replace or augment token-overlap scoring with byte/span-level supervision,
  - keep the current bridge/module path fixed,
  - test whether tokenizer-agnostic supervision turns the current smoke gain
    into a controlled held-out gain.
