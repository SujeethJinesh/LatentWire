# HellaSwag Terminal-Tail Stress References

This memo supports
`paper/source_private_hellaswag_full_validation_tail_20260501.md` and the
refreshed ICLR evidence bundle. The local result is a positive-but-soft-failing
terminal-tail diagnostic, not a full-validation claim.

## Local Claim

The `2B` raw / `5B` framed dense hidden-innovation packet remains
source-private and preserves a positive overall terminal-tail margin over
label-copy and score-only controls. It fails the strict terminal-tail gate
because one jackknife subbag has a non-positive lower confidence bound, so the
paper must not claim a full HellaSwag validation pass.

## Primary Related Work Boundaries

- Prefix tuning, prompt tuning, P-Tuning, and adapters learn persistent
  conditioning parameters or model-side adaptation modules. LatentWire does not
  transmit learned prompt tokens, soft prompts, or adapters per request.
  Sources: https://arxiv.org/abs/2101.00190,
  https://arxiv.org/abs/2104.08691, https://arxiv.org/abs/2110.07602, and
  https://arxiv.org/abs/1902.00751
- Sparse autoencoders, universal sparse autoencoders, sparse crosscoders, and
  relative representations remain the right literature for any future
  common-basis/shared-dictionary branch. The current terminal-tail diagnostic
  does not establish a universal latent language. Sources:
  https://arxiv.org/abs/2309.08600, https://arxiv.org/abs/2502.03714,
  https://transformer-circuits.pub/2024/crosscoders/index.html, and
  https://arxiv.org/abs/2209.15430
- C2C, KVComm, KVCOMM, and Q-KVComm are the closest non-text communication
  competitors because they communicate or reuse source-side cache/state.
  LatentWire differs by exposing no source KV cache or raw source state, but
  native matched comparisons remain pending. Sources:
  https://arxiv.org/abs/2510.03215, https://openreview.net/forum?id=F7rUng23nw,
  https://arxiv.org/abs/2510.12872, and https://arxiv.org/abs/2512.17914
- QJL, TurboQuant, KIVI, and KVQuant define systems-side source-state
  compression floors rather than source-private packet methods. These are
  important comparison rows for the native systems table. Sources:
  https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, and https://arxiv.org/abs/2401.18079
- Diffusion transformers and token-free/latent reasoning motivate possible
  iterative repair or denoising mechanisms, but the current result is still a
  fixed-byte per-example communication protocol. Sources:
  https://arxiv.org/abs/2212.09748, https://arxiv.org/abs/2502.12134, and
  https://arxiv.org/abs/2412.06769
- Selective prediction and calibration are relevant to a next top-2
  trust-or-switch branch, because the packet must decide when to preserve the
  target answer and when to switch to a source-informed alternative. Source:
  https://arxiv.org/abs/1705.08500

## Citation Use

Use these references to keep the terminal-tail story precise:

> LatentWire has a strong contiguous `0:9216` HellaSwag result and a
> positive-but-soft-failing terminal-tail diagnostic. The current contribution
> is a source-private fixed-byte repair protocol with strict controls, not a
> prompt-tuning, adapter, SAE/common-basis, KV-cache communication,
> quantization, diffusion-transformer, or latent-CoT method.

## Next Method Branch

If the full-validation gate remains required before submission, the most
promising Mac-local branch is a sparse top-2 trust-or-switch packet. It should
reuse the same frozen train/eval split, keep the `5B` framed budget if possible,
and report label-copy, score-only, zero-hidden, wrong-hidden, candidate-roll,
same-byte visible text, and full-validation aggregate controls.
