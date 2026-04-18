# RotAlign-KV Story Tracks

## Current read

- The broad “cross-model KV transfer works” story is still not supported.
- The strongest surviving regime is **sparse `k_only` transport on GSM8K**, not dense all-layer transfer.
- The cleanest same-pair positive signal is now on Qwen2.5-0.5B -> Qwen3-0.6B with sparse key import under matched controls.
- The strongest new mechanistic lead is even narrower: on that same Qwen pair, a **fixed calibration-derived per-head budget prior** beat both the old uniform sparse baseline and a budget-matched shuffled-prior null, and it held directionally on `gsm8k_100`.
- The saved-prior transfer matrix is now clearly **asymmetric**:
  - Qwen prior -> Qwen is strong
  - Qwen prior -> DeepSeek collapses
  - DeepSeek prior -> DeepSeek stays weak
  - DeepSeek prior -> Qwen partially transfers only up to the old uniform-sparse level
- The key failure boundaries still matter:
  - `translated-only` collapses
  - `v_only` is mostly harmful
  - ARC remains mixed or confounded by zero-byte controls
  - cross-family transfer is weaker than same-family transfer
  - the fixed per-head prior did not carry over cleanly to DeepSeek
- The next mechanism question is no longer “keys or values?” We answered that directionally. It is now:
  - **which heads get the sparse key budget**
  - **whether that budget should be live, calibrated, or query-aware**

## COLM workshop path

Goal: a careful, control-heavy short paper with one sharp claim.

Proposed claim:
- Cross-model latent transfer is not uniformly helpful.
- The only reliable gains so far come from **selective sparse key import**, not generic KV fusion.
- The newest same-pair gain may come from **calibrated head identity**, not just live query-aware sparsity.
- The transfer story is not “universal head priors.” It is currently **pair-conditioned and asymmetric**.
- Strong zero-byte, random-source, and query-blind selector controls are necessary because naive cache perturbations or blind sparsity can look like communication gains.

What must be true:
- Replicate the sparse `k_only` GSM8K result on one more reasoning split or one more reasoning task.
- Keep paired comparisons against:
  - target alone
  - text-to-text
  - zero-byte attenuation
  - random translated KV
  - translated-only `k_only`
  - blind selector controls such as shuffled attention, random selector, and fixed priors when relevant

## ICLR full-paper path

Goal: a broader method paper with a real mechanism story.

Target claim:
- Cross-model latent communication works only when the transmitted state is **structured, selective, and source-dependent**.

What we still need:
- better than `target alone` on held-out reasoning more than once
- stronger gap over zero-byte controls
- one second reasoning benchmark
- one second model pair with at least directional support
- a clearer mechanism section built around:
  - `k_only` vs `v_only`
  - live query-aware position selection vs blind priors
  - head selection vs per-head budgets
  - fixed calibrated head priors vs shuffled-prior nulls
  - asymmetric prior transfer across target models
  - quantized vs no-quantized
  - static vs source-dependent fusion
  - selector-specific failure cases

## Immediate next experiments

1. Rerun the fixed per-head prior branch on a larger GSM slice so the same-pair gain is not one-example fragile.
2. Keep the shuffled-prior null beside it, because that is now the main reviewer-proof control.
3. Keep the DeepSeek pair as the main transfer stress test instead of widening to many models too early.
4. Preserve the negative controls and failure cases in the main paper, not just the appendix.
