# Tokenizer, Connector, Quantization, and Harness References

Scope: tokenizer/vocab alignment, multimodal connector bottlenecks, quantization-inspired communication, and runnable benchmark harnesses that help us sanity-check LatentWire-style communication paths without editing non-reference code.

## Why this memo exists

The current positive-method loop is converging on a few recurring implementation questions:

1. Can we reduce interface entropy with tokenizer/vocab alignment instead of adding more adapter depth?
2. If the interface must be narrow, should the bottleneck look more like a Q-Former, Perceiver resampler, or learned projector bank?
3. Can quantization ideas from AWQ / EXL2 / KV compression be reused as communication rules, not just memory tricks?
4. Which external benchmark harnesses can we clone under `references/repos/` for future smoke tests and table generation?

## Source stack

- `Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching`  
  https://www.alphaxiv.org/overview/2503.20083v4  
  Use as the strongest cross-tokenizer alignment hypothesis: approximate likelihood matching, not just embedding cosine.
- `AdaptBPE`  
  https://arxiv.org/abs/2601.21665  
  Use as a tokenizer adaptation reference for shared-vocabulary experiments.
- `Tokenizer-Aware Cross-Lingual Adaptation`  
  https://arxiv.org/abs/2604.07466  
  Use as a “tokenizer/vocab changes can be a first-class adaptation parameter” reference.
- `Model-Aware Tokenizer Transfer`  
  https://arxiv.org/abs/2504.17013  
  Use as a transfer rule reference when one side has a fixed base tokenizer and the other side must adapt.
- `BLIP-2 / Q-Former`  
  https://arxiv.org/abs/2301.12597  
  Use as the canonical “frozen backbones + narrow learned connector” template.
- `Perceiver IO` implementation  
  https://github.com/krasserm/perceiver-io  
  Use as the narrow resampler / latent bottleneck analogue.
- `AWQ`  
  https://github.com/mit-han-lab/llm-awq  
  Use as a saliency-preserving channel-protection analogue, not just a weight quantizer.
- `ExLlamaV2 / EXL2`  
  https://github.com/turboderp-org/exllamav2  
  Use as a mixed-bit / outlier-protection / practical packing reference.
- `KVzip`  
  https://arxiv.org/abs/2505.23416  
  Use as query-agnostic compression and reconstruction ceiling reference.
- `KVPress`  
  https://github.com/NVIDIA/kvpress  
  Use as a cheap local compression harness for same-model communication controls.
- `lm-evaluation-harness`  
  https://github.com/EleutherAI/lm-evaluation-harness  
  Use as a stable generic benchmark harness for future table generation and smoke tests.
- `OpenCompass`  
  https://github.com/open-compass/opencompass  
  Use as a broader benchmark orchestration reference when we need a second harness perspective.

## Runnable repos to clone under `references/repos/`

- `git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness references/repos/lm-evaluation-harness`
- `git clone --depth 1 https://github.com/open-compass/opencompass references/repos/opencompass`

These are useful even if we never run full suites immediately:

- `lm-evaluation-harness` gives us a compact, reproducible adapter for sanity-checking prompts, tasks, and generation contracts.
- `OpenCompass` gives us a broader evaluation control when we need a second harness shape for comparison tables.

## Concrete ablations to prioritize next

1. **Tokenizer/vocab alignment ablation**
   - Compare baseline adapter routing against a shared-tokenizer or tokenizer-transfer variant.
   - Log token overlap, effective sequence length, and answer exactness under the same byte budget.
   - Add a byte-level fallback when tokenizer disagreement is the main failure mode.

2. **Narrow connector bottleneck ablation**
   - Compare a plain projector, a Q-Former-style query bottleneck, and a Perceiver-style resampler.
   - Keep encoder/decoder frozen and vary only bottleneck width, query count, and latent depth.
   - Measure whether performance tracks bottleneck size or whether a small learned query bank saturates early.

3. **Quantization-inspired communication ablation**
   - Compare dense communication against mixed-bit / outlier-protected channels.
   - Test AWQ-style salient-channel preservation, EXL2-like mixed-bit packing, and KV compression-inspired budget splits.
   - Log route entropy, outlier coverage, byte budget, and exact-answer accuracy together.

## Telemetry we should keep

- tokenizer overlap ratio
- effective bytes per example
- connector latent count and query count
- salient-channel retention / outlier mask coverage
- route entropy and route stability
- answer exactness after compression or bottlenecking
- whether the harness itself is run by a local clone or wrapper

## Anti-loop rules

- Do not add more adapter depth before trying the shared-tokenizer / shared-bottleneck variants.
- Do not claim improvement from quantization-inspired ideas unless byte budget and exactness are both reported.
- Use harness clones for smoke tests first; only then scale to larger sweeps.

