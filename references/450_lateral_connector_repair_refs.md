# Lateral Connector And Repair References

Date: `2026-04-22`

## Why This Memo Exists

The current direct latent-basis story is still too narrow. If the codec-side
anchor-tail branch stalls, the cleanest next pivot is a frozen-backbone learned
connector or a conditional repair mechanism.

## Strongest References

- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
- [Re-ViLM](https://arxiv.org/abs/2302.04858)
- [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948)
- [Attractor Patch Networks](https://arxiv.org/abs/2602.06993)
- [S'MoRE](https://arxiv.org/abs/2504.06426)
- [LongMem](https://arxiv.org/abs/2306.07174)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [ResQ](https://arxiv.org/abs/2412.14363)

## What Maps Cleanly Onto The Frozen-Backbone Setting

- learned query bottlenecks / resamplers
- small memory or side-network connectors
- routed patch experts
- conditional repair heads

The strongest immediate fit is still:

- frozen source
- frozen target
- small learned query bottleneck
- optional conditional tail repair on top

## Smallest Decisive Ablations

1. learned connector:
   - small fixed query bottleneck, `8-16` learned queries
   - compare directly against the live dynalign residual lane
2. conditional repair:
   - gate tail correction by uncertainty or margin instead of always-on repair
3. routed patch repair:
   - replace the single repair head with a tiny `2`-expert patch router
4. tiny iterative bridge:
   - two-step denoising bridge on the tail only

## Practical Read

If anchor-preserving selective precision cannot preserve the live row on the
larger frozen slice, the best next architectural pivot is no longer another
geometry tweak. It is a small learned connector or a conditional repair layer.
