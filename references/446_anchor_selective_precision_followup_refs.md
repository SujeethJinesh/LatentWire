# Anchor Selective Precision Follow-Up References

Date: `2026-04-22`

## Why This Memo Exists

The live same-pair row still points to one codec-side branch that is not yet
fully tested: preserve a small anchor set exactly, then quantize or code only
the tail.

## Strongest References

- [Preserve-Then-Quantize](https://openreview.net/forum?id=cHUXlmsSXK)
- [ResQ](https://openreview.net/forum?id=4qIP1sXcR1)
- [SERQ](https://openreview.net/forum?id=nFjj8NEBqv)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [OWQ](https://arxiv.org/abs/2306.02272)
- [QuIP#](https://arxiv.org/abs/2402.04396)
- [AQLM](https://arxiv.org/abs/2401.06118)
- [VPTQ](https://arxiv.org/abs/2409.17066)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [MatryoshkaKV](https://arxiv.org/abs/2410.14731)
- [TurboQuant](https://arxiv.org/abs/2504.19874)
- [KV Cache Transform Coding](https://arxiv.org/abs/2511.01815)

## Ranked Ablations

1. Preserve `V` only vs `K` only vs `K+V`
2. Saliency-selected anchors vs magnitude-selected anchors
3. Preserve rank `4 / 8 / 16`
4. Tail precision `4-bit / 3-bit`
5. Preserve-only vs tail-only vs preserve+tail

## Telemetry That Makes This Interpretable

- protected mass / saliency coverage
- anchor error, tail error, and total error separately
- bytes, latency, and peak memory
- source / target / communicated / oracle on the same IDs
- paired bootstrap interval and seed spread

## Smallest Live Branch

Start with `V`-only anchor preservation on top of
`dynalign_module_replace_residrank16`, keep a small saliency-selected anchor
exact, and quantize only the tail to `4` bits. If that preserves `0.1250`
while reducing bytes, the codec story remains alive.
