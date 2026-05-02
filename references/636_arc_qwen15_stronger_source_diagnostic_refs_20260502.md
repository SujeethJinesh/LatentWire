# ARC Qwen-1.5B Stronger-Source Diagnostic References

Date: 2026-05-02

## Local Evidence

- Gate:
  `results/source_private_arc_challenge_source_family_cache_falsification_20260502_qwen15_cpu/source_family_cache_falsification.json`
- Decision: Qwen-1.5B is a promising same-family stronger-source diagnostic,
  not yet a cross-family ICLR claim.
- Frozen test full-slice matched/target/text: `0.442/0.265/0.401`.
- Frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.482/0.184/0.456/0.296`.
- Frozen test minimum matched-minus-Qwen-substituted and CI95 low:
  `+0.294`, `+0.216`.
- Validation Qwen-disagreement pass: `0/5`, with CI95 low versus
  Qwen-substituted packets `-0.021` on `95` rows.

## Related Work Boundary

- Relative representations. Use to frame the shared-coordinate/public-basis
  packet as an anchor-coordinate method, not raw hidden transfer:
  `https://arxiv.org/abs/2209.15430`.
- C2C. Boundary: C2C transfers and fuses source KV cache state, while this
  diagnostic transmits a 12-byte public-basis packet:
  `https://arxiv.org/abs/2510.03215`.
- KVComm. Boundary: selective KV sharing is a high-rate source-state baseline:
  `https://arxiv.org/abs/2510.03346`.
- KVCOMM. Boundary: offset-aligned cache reuse is a systems/cache baseline,
  not a fixed packet:
  `https://arxiv.org/abs/2510.12872`.
- TurboQuant. Boundary: useful for byte/distortion floors against KV/cache
  quantization, not a duplicate of stronger-source packet transfer:
  `https://arxiv.org/abs/2504.19874`.
- Prefix tuning. Boundary: learned virtual prompts are not per-example
  source-private packets:
  `https://arxiv.org/abs/2101.00190`.

## Paper Implication

This is a positive diagnostic showing source strength matters and that the
packet can carry useful stronger-source decisions on frozen ARC test rows.
Because it is same-family and validation-gate-incomplete, the next paper-safe
claim still requires a true cross-family source, a larger validation/control
surface, or a learned hidden/query common-basis connector.
