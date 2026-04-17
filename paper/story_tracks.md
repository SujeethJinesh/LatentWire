# RotAlign-KV Story Tracks

## Current read

- The strongest current positive signal is on held-out GSM8K with `k_only` fused KV.
- `k_only + cosine_shifted` reached `0.0571`, beating `target alone` at `0.0429`.
- `translated-only k_only` collapsed to `0.0000`, which suggests translated keys help only when the target model supplies its own values.
- ARC is still a warning sign: zero-byte target attenuation beat real translated KV there.

## COLM workshop path

Goal: a careful, control-heavy short paper with one sharp claim.

Proposed claim:
- Cross-model latent transfer is not uniformly helpful.
- The useful signal appears concentrated in translated **keys**, while translated **values** are unstable or harmful.
- Strong zero-byte and random-source controls are necessary because naive cache perturbations can look like communication gains.

What must be true:
- Replicate the `k_only` GSM8K result on one more reasoning split or one more reasoning task.
- Keep paired comparisons against:
  - target alone
  - text-to-text
  - zero-byte attenuation
  - random translated KV
  - translated-only `k_only`

## ICLR full-paper path

Goal: a broader method paper with a real mechanism story.

Target claim:
- Cross-model latent communication works when the transmitted state is structured, selective, and source-dependent.

What we still need:
- better than `target alone` on held-out reasoning more than once
- stronger gap over zero-byte controls
- one second reasoning benchmark
- a clearer mechanism section built around:
  - `k_only` vs `v_only`
  - head selection
  - position selection
  - quantized vs no-quantized
  - static vs source-dependent fusion

## Immediate next experiments

1. Sweep `k_only` gates at `0.05`, `0.10`, `0.15`, `0.20`.
2. Keep comparing `static` vs `cosine_shifted`.
3. Add one second reasoning task before widening model families.
4. If `k_only` keeps holding, make it the default paper branch.
