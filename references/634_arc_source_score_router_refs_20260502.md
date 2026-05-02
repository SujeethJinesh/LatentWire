# ARC Source-Score Router References

Date: 2026-05-02

## Local Evidence

- Gate: `results/source_private_arc_challenge_source_score_router_gate_20260502/source_score_router_gate.json`
- Decision: source-side scalar confidence overfits validation and fails frozen
  ARC TinyLlama-vs-Qwen disagreement test.
- Validation-selected rule: `source_index_pair_lookup`.
- Validation router/Qwen: `0.451/0.389`, delta `+0.063`, CI95 low `+0.010`.
- Test router/Qwen/oracle: `0.315/0.317/0.586`, router-minus-Qwen `-0.002`,
  CI95 low `-0.031`.

## Confidence Routing And Uncertainty

- RACER: confidence-based routing for LLMs. Use as the closest routing
  baseline family for "choose the model to trust" framing:
  `https://arxiv.org/abs/2603.06616`.
- Confidence-Driven LLM Router. Use for recent confidence-router positioning:
  `https://arxiv.org/abs/2502.11021`.
- Semantic uncertainty. Use for uncertainty features beyond scalar logit
  margins and entropy:
  `https://arxiv.org/abs/2302.09664`.
- Semantic entropy. Use for uncertainty-aware generation and hallucination
  detection context:
  `https://www.nature.com/articles/s41586-024-07421-0`.

## Common-Basis And Representation Alignment

- Relative representations. Use to motivate comparing model representations
  in coordinates defined by shared anchors:
  `https://arxiv.org/abs/2209.15430`.
- Semantic channel equalization / relative representations. Use as a direct
  common-basis communication reference:
  `https://arxiv.org/abs/2411.19719`.
- Sparse autoencoder features. Use as motivation for feature-level common
  dictionaries rather than raw hidden-state transfer:
  `https://arxiv.org/abs/2309.08600`.
- SAE universality. Use cautiously as motivation for cross-model feature
  overlap, not as evidence that this gate works:
  `https://arxiv.org/abs/2410.06981`.

## Boundaries Against Nearby Systems

- Prefix tuning. Boundary: prefix methods optimize continuous prompt vectors
  for one model; the current packet is a fixed source-private communication
  object and the source-score sidecar is not a prompt prefix:
  `https://arxiv.org/abs/2101.00190`.
- C2C cache-to-cache communication. Boundary: C2C moves/fuses source KV/cache
  state and requires internals, while this gate tests fixed packets plus a
  one-byte score sidecar:
  `https://arxiv.org/abs/2510.03215`.
- KVComm. Boundary: KV-cache communication/quantization is a high-rate
  source-state baseline, not the same low-byte packet threat model:
  `https://arxiv.org/abs/2510.03346`.
- KVCOMM. Boundary: same systems family as KVComm but with a different cache
  communication design point:
  `https://arxiv.org/abs/2510.12872`.
- TurboQuant. Boundary: TurboQuant is an efficient KV-cache quantization
  systems baseline; it helps frame byte/HBM comparisons but is not a
  source-private packet method:
  `https://arxiv.org/abs/2504.19874`.

## Paper Implication

This gate should be cited as a negative method-ablation, not as a contribution
claim by itself. It rules out the simplest confidence-router repair and
promotes the next branch: a learned common-basis connector, SAE/cross-feature
dictionary, or stronger alternate-source run under the same frozen
disagreement-row protocol.
