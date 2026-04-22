# Q-Former Connector Follow-Up References

Date: `2026-04-22`

## Why This Memo Exists

The direct latent-basis story is narrow and fragile. A learned connector still
looks like the cleanest lateral branch if anchor-preserving codecs stall.

## Strongest References

- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)
- [Ovis](https://arxiv.org/abs/2405.20797)
- [SelfCP](https://arxiv.org/abs/2405.17052)
- [TokAlign](https://arxiv.org/abs/2506.03523)
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
- [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948)

## Smallest Decisive Experiment

- frozen source and frozen target
- learned query bottleneck with a small fixed query count
- minimal rank / bank budget
- evaluate first on the larger frozen same-pair slice, then on one matched
  cross-family pair

Keep the branch only if it beats or clearly stabilizes the live dynalign
residual lane across seeds.

## Diagnostics That Separate Connector Gains From Repair Gains

- run with and without repair heads
- compare latent reconstruction and downstream exact match separately
- log source / target / communicated / oracle on same IDs
- run the first cross-family pair immediately after the same-pair slice
