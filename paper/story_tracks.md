# RotAlign-KV Story Tracks

## Current read

- The broad “cross-model KV transfer works” story is still not supported.
- The strongest surviving regime is **sparse `k_only` transport on GSM8K**, not dense all-layer transfer.
- The cleanest same-pair positive signal is now on Qwen2.5-0.5B -> Qwen3-0.6B with sparse key import under matched controls.
- The strongest new mechanistic lead is even narrower: on that same Qwen pair, a **fixed calibration-derived per-head budget prior** beat both the old uniform sparse baseline and a budget-matched shuffled-prior null, and it held directionally on `gsm8k_100`.
- The fixed-prior budget sweep now sharpens that story: the branch is best at the **middle budget** (`50%`), while `25%` and `75%` keep only a smaller directional edge over shuffled and uniform baselines.
- The second reasoning-task check is mixed: on SVAMP, the same fixed head-prior branch beats the shuffled-prior null but only climbs back to **target-alone**, so it is a boundary condition rather than a second positive benchmark.
- Two simple follow-up tweaks did **not** beat the fixed prior:
  - prior/live blends help somewhat but top out below the pure fixed prior
  - entropy-based fixed priors are positive but weaker than peak-based priors
- Shrinkage-regularized priors also did **not** rescue the branch:
  - global shrinkage hurts GSM and leaves SVAMP / DeepSeek unchanged
  - uniform shrinkage only ties the original fixed prior
- The new seed repeat is the most important negative update:
  - `seed0 = 0.0200`
  - `seed1 = 0.0700`
  - `seed2 = 0.0100`
  on `gsm8k_100`, so the fixed-prior branch is currently **high variance and not stable enough to headline**.
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
  - **how to stabilize that budget without diluting the useful calibrated prior**
  - **whether that budget should be shrinkage-regularized, retrieval-head-specific, or attention-logit-preserving**
  - **whether the useful structure is retrieval-head-specific or attention-logit-preserving**

## COLM workshop path

Goal: a careful, control-heavy short paper with one sharp claim.

Proposed claim:
- Cross-model latent transfer is not uniformly helpful.
- The only reliable gains so far come from **selective sparse key import**, not generic KV fusion.
- The newest same-pair gain may come from **calibrated head identity**, not just live query-aware sparsity.
- The current best workshop-safe statement is: calibrated sparse key head budgets help on GSM8K for a compatible same-family pair, but the effect weakens on SVAMP and across target families.
- The fixed-prior story is now best used as a **mechanism clue**:
  there is structure in which heads matter, but the current fixed prior is not
  stable enough across seeds to present as the final method.
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
- better than `target alone` on more than one reasoning task, not just GSM8K
- better than `target alone` across seeds, not just on one calibration seed
- stronger gap over zero-byte controls
- one second reasoning benchmark
- one second model pair with at least directional support
- a clearer mechanism section built around:
  - `k_only` vs `v_only`
  - live query-aware position selection vs blind priors
  - head selection vs per-head budgets
  - fixed calibrated head priors vs shuffled-prior nulls
  - fixed calibrated head priors vs seed repeats
  - asymmetric prior transfer across target models
  - SVAMP as an explicit boundary case for the calibrated-prior branch
  - quantized vs no-quantized
  - static vs source-dependent fusion
  - selector-specific failure cases

## Immediate next experiments

1. Run the **3-seed repeat on the stronger live query-aware sparse `k_only` branch**, because the fixed-prior branch is now too unstable to carry the story by itself.
2. Keep the DeepSeek pair as the main transfer stress test instead of widening to many models too early.
3. Implement the next method pivots suggested by the literature:
   - OT / permutation head matching
   - causal head scoring
   - retrieval-head-only routing
   - attention-logit-preserving head ranking
4. Preserve the negative controls and failure cases in the main paper, not just the appendix.
