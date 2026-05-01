# References: HellaSwag Dense Residual Sketch Scout

## Purpose

This memo records the prior-work boundary for the dense residual sign-sketch
scout. The branch is demoted on the current HellaSwag decision surface.

## Primary Sources

- QJL studies Quantized Johnson-Lindenstrauss sketches for low-bit vector/KV
  information. This motivates the sign-sketch control but is not new here.
  Source: https://arxiv.org/abs/2406.03482

- TurboQuant combines vector quantization with a 1-bit QJL residual correction
  for low-bit vector/KV compression. LatentWire should cite it as a systems
  neighbor, not as something beaten by the current Mac scout.
  Source: https://arxiv.org/abs/2504.19874

- C2C communicates/fuses source KV cache state across LLMs. LatentWire differs
  by transmitting only a fixed-byte candidate/confidence packet, not source KV.
  Source: https://arxiv.org/abs/2510.03215

- KVComm selectively shares KV pairs/layers. This is a high-rate cache-state
  communication comparator, not the same threat model as a source-private packet.
  Source: https://arxiv.org/abs/2510.03346

- Prefix tuning optimizes target-side continuous prefix state. It is not the
  same as a sender-side private packet selected from hidden evidence.
  Source: https://arxiv.org/abs/2101.00190

## Non-Claim Boundary

Do not claim a new sign sketch, new JL transform, new quantizer, or new KV-cache
compression method. The only valid claim would be that a sketch can serve as a
local sender front end for selecting LatentWire's fixed-byte source-private
packet. The current result does not support that claim.

## Experimental Outcome

Best variant: `qjl_norm_sign128`.

- accuracy: `0.501953`
- best label-copy: `0.500000`
- delta vs label-copy: `+0.001953`
- score-only: `0.497070`
- delta vs score-only: `+0.004883`
- scout pass: false

Decision: demote simple data-oblivious dense residual sketches for this cycle.
