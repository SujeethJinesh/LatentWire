# Disagreement-Prototypes and Soft-Prefix Receiver References

Date: 2026-05-03

## Purpose

This memo records the literature boundary after the negative HellaSwag
disagreement-prototype receiver gate and motivates the next ARC target-loss
soft-prefix/query-bottleneck gate.

## What Is Not Novel By Itself

- Prefix-tuning:
  https://arxiv.org/abs/2101.00190
  - Continuous learned prefix vectors for frozen models are established prior
    art. LatentWire must distinguish source-private per-example communication
    from static task prefix tuning.
- Prompt tuning:
  https://arxiv.org/abs/2104.08691
  - Learned prompt embeddings are also not a novel contribution by themselves.
- Gist tokens:
  https://arxiv.org/abs/2304.08467
  - Prompt compression into reusable tokens is a baseline/comparison boundary,
    not LatentWire's novelty.
- Perceiver IO:
  https://arxiv.org/abs/2107.14795
  - Learned latent query bottlenecks are established architectures.
- BLIP-2 / Q-Former:
  https://arxiv.org/abs/2301.12597
  - Querying one frozen module into another with a small learned connector is
    established, especially in multimodal transfer.
- Flamingo:
  https://arxiv.org/abs/2204.14198
  - Frozen language-model cross-attention connectors are a relevant precedent.

## Query-Conditioned Compression And Systems Boundary

- Quest query-aware KV selection:
  https://arxiv.org/abs/2406.10774
- SnapKV:
  https://arxiv.org/abs/2404.14469

These motivate receiver-query-conditioned source selection but are KV/cache
systems methods, not proof of fixed-byte source-private reasoning packets.

## Sparse/Common-Feature Boundary

- SAE feature universality:
  https://arxiv.org/abs/2410.06981
- SAE seed instability:
  https://arxiv.org/abs/2501.16615
- Crosscoders:
  https://arxiv.org/abs/2504.02922
- Cross-architecture crosscoders:
  https://arxiv.org/abs/2602.11729

Sparse/common-feature methods are plausible regularizers and diagnostics, but
the current evidence does not justify claiming a universal sparse latent
language. They should support interpretability and controls after a downstream
target-loss connector works.

## Cache/Quantization Competitor Boundary

- C2C / Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
- QJL:
  https://arxiv.org/abs/2406.03482
- TurboQuant:
  https://arxiv.org/abs/2504.19874

These remain byte/exposure and native-systems competitors. LatentWire should
not claim to beat them natively until matched NVIDIA serving rows exist.

## Diffusion/Repair Inspiration

- Denoising diffusion probabilistic models:
  https://arxiv.org/abs/2006.11239
- Consistency models:
  https://arxiv.org/abs/2303.01469

Use only as inspiration for one-step denoising/repair of corrupted source
innovations. Do not turn this into a separate contribution until a concrete
gate exists.

## Decision Boundary

The HellaSwag prototype gate fails:

- default prototype receiver: `+0.000100` over packet-only, CI low
  `-0.000896`;
- best diagnostic prototype: `+0.001195` over packet-only, CI low
  `-0.000199`;
- oracle headroom remains `+0.067616`.

Therefore the next Mac-local gate should stop shallow selector/prototype work
and run a target-loss query/soft-prefix repair. The uniqueness claim, if it
passes, is not "soft prompts"; it is per-example source-conditioned,
fixed-byte, source-private communication that survives source-destroying and
Qwen-substituted controls.
