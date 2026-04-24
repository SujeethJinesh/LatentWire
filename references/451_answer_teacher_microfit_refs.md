# Answer-Teacher Microfit And Source-Control Connector References

Date: `2026-04-24`

## Why This Memo Exists

The current SVAMP32 blocker is not a lack of matched-only generation rows; it is
the absence of teacher-forced source-specific answer signal. This memo records
the primary references that motivate the remaining controlled connector options
after the answer-teacher calibration proxy failed.

## Primary Sources

- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Frozen-backbone learned query bottleneck; closest multimodal blueprint for a
  small connector trained to make one frozen model consumable by another.
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
  General learned latent-query interface for mapping arbitrary input structure
  to task-conditioned outputs.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Perceiver resampler plus gated cross-attention; useful for low-interference
  integration of external states into a frozen language model.
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
  Instruction-aware query bridge; relevant if the connector must condition on
  the downstream reasoning prompt rather than source cache alone.
- [Cache-to-Cache Communication](https://arxiv.org/abs/2510.03215)
  Closest LLM-to-LLM cache-fusion prior; motivates C2C residual distillation
  rather than source-answer copying.
- [Information Bottleneck](https://arxiv.org/abs/physics/0004057)
  Rate-relevance framing for learned bottlenecks.
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
  Trainable bottleneck objective if a future connector needs explicit capacity
  pressure.
- [Deep Joint Source-Channel Coding](https://arxiv.org/abs/1809.01733)
  End-to-end communication framing for noisy or mismatched channels.
- [Slepian-Wolf Coding](https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf)
  Decoder-side-information framing for latent syndrome sidecars.
- [Wyner-Ziv Coding](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf)
  Lossy source coding with decoder side information; closest information theory
  analogue for transmitting only the target-missing residual.

## Smallest Decisive Experiments

- Standalone answer-margin sidecar:
  - train only connector tensors against matched-vs-control answer margins
  - promote only if `>=2/6` clean residual IDs become matched-only positive
- C2C-residual fuser:
  - distill C2C cache/text teacher signal into a query fuser
  - require zero/shuffle/target-only collapse before generation
- Latent syndrome sidecar:
  - transmit numeric residue checks from source residual states
  - use target candidate pool as decoder side information

## Practical Read

The answer-teacher calibration proxy failed because target-only and learned-slot
controls still reproduced the positive margins. The next branch must optimize
source necessity directly or test whether the source contains any recoverable
numeric side information at all.
