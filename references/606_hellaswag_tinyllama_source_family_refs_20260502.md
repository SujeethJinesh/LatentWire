# HellaSwag TinyLlama Source-Family Stress References

This memo supports
`paper/source_private_hellaswag_tinyllama_source_family_stress_20260502.md`.

## Local Claim

TinyLlama is the first non-Qwen source-family stress row for the current
HellaSwag hidden-innovation packet. It passes a frozen heldout validation slice
with the same `2B` raw / `5B` framed packet contract and destructive controls.
This weakens the Qwen-only-geometry critique, but it is not yet a full
cross-family ICLR claim.

## Primary Related Work Boundaries

- HellaSwag defines the commonsense continuation benchmark used here.
  Source: https://arxiv.org/abs/1905.07830
- Prefix tuning and prompt tuning learn continuous/persistent conditioning
  vectors. LatentWire sends a per-example discrete source-private packet, not a
  learned soft prompt. Sources: https://aclanthology.org/2021.acl-long.353/ and
  https://arxiv.org/abs/2104.08691
- LoRA and adapters are parameter-efficient receiver adaptation methods.
  LatentWire does not adapt receiver weights in this gate. Sources:
  https://openreview.net/forum?id=nZeVKeeFYf9 and
  https://arxiv.org/abs/1902.00751
- C2C and KVComm communicate or fuse source-side KV/cache state. LatentWire
  differs by transmitting only a tiny task-evidence packet, but native
  comparisons remain pending. Sources: https://arxiv.org/abs/2510.03215 and
  https://arxiv.org/abs/2510.03346
- QJL, TurboQuant, KIVI, and KVQuant define vector/KV-cache compression floors.
  They are mandatory systems comparators, not duplicates of this fixed-byte
  source-private packet. Sources: https://arxiv.org/abs/2406.03482,
  https://arxiv.org/abs/2504.19874, https://arxiv.org/abs/2402.02750, and
  https://arxiv.org/abs/2401.18079
- Relative representations, SAE universality, and sparse crosscoders remain the
  relevant common-basis literature. The TinyLlama pass is evidence against a
  Qwen-only story, not proof of a solved shared basis. Sources:
  https://arxiv.org/abs/2209.15430,
  https://arxiv.org/abs/2410.06981, and
  https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
- vLLM and SGLang define native serving baselines for future systems rows.
  Current Mac-local evidence supports byte/exposure accounting only. Sources:
  https://arxiv.org/abs/2309.06180 and https://arxiv.org/abs/2312.07104

## Citation Use

Use this memo for the narrowed claim:

> The source-private hidden-innovation packet is not restricted to Qwen2.5:
> TinyLlama source hidden evidence also yields a positive HellaSwag heldout
> slice under the same packet budget and destructive controls.

Use this memo to avoid overclaiming:

> The result is a source-family stress test, not a full universal
> cross-family-transfer result. Full non-Qwen validation, true receiver-family
> transfer, and native systems comparisons remain open gates.

## Next Branch

If the full TinyLlama or Phi source-family run fails, the next method branch is
a sparse residual dictionary/common-basis packet: train-only residual atoms,
atom-ID/value shuffles, wrong-example residuals, candidate-roll residuals, and
the same `2B` raw / `5B` framed packet budget.
