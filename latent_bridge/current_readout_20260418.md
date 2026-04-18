# RotAlign-KV Current Readout — 2026-04-18

## Headline

We are **not ICLR-main-paper ready yet**.

The strongest surviving result is now a **pair-conditioned calibrated sparse
key-head budget prior** on the Qwen2.5-0.5B -> Qwen3-0.6B control pair. That
result survives a matched shuffled-prior null and beats the older uniform
sparse baseline on GSM8K, but it does **not** transfer cleanly to DeepSeek and
it does **not** replicate as a clear win on SVAMP. The new multi-seed repeat
also shows that the current fixed-prior branch is **not stable enough yet**:
`seed0=0.0200`, `seed1=0.0700`, `seed2=0.0100` on `gsm8k_100`.
The live query-aware sparse `k_only` branch also fails the same stability test
on `gsm8k_100`: `seed0=0.0200`, `seed1=0.0400`, `seed2=0.0200`.
A direct retrieval-head-style heuristic, `retrieval_peak`, also fails to
rescue the method:
- Qwen -> Qwen, `gsm8k_100`: `0.0300`
- Qwen -> Qwen, `svamp_eval_70`: `0.071429`
- Qwen -> DeepSeek, `gsm8k_eval_70`: `0.014286`
So simple retrieval-head scoring is currently another **negative boundary**,
not a stable new branch.

## Strongest Positive Regime

Qwen2.5-0.5B -> Qwen3-0.6B, sparse `k_only`, `gate=0.10`,
`position_selection_ratio=0.5`, fixed calibration-derived per-head budget prior:

- `gsm8k_eval_70`: `0.085714`
- `gsm8k_100`: `0.070000`
- `gsm8k_100` shuffled-prior null: `0.040000`
- `gsm8k_100` uniform sparse baseline: `0.050000`
- `gsm8k_100` target alone: `0.040000`
- `gsm8k_100` text-to-text: `0.100000`

Interpretation:

- The gain is **not** explained by generic sparsity or any uneven head budget.
- The specific calibrated head assignment matters.
- The effect is still narrower than text-to-text and therefore not yet a
  replacement for token communication.
- The new seed repeat materially weakens this branch as a headline result:
  across `seed0/1/2`, the fixed-prior mean is only `0.0333`, below the
  target-alone baseline `0.0400`.

## Budget Sweep

Same Qwen pair, same fixed-prior branch, `gsm8k_100`, matched `k_only` /
`gate=0.10` / quantized setup:

- `25%` budget:
  - fixed prior: `0.050000`
  - shuffled prior: `0.040000`
  - uniform sparse baseline: `0.040000`
- `50%` budget:
  - fixed prior: `0.070000`
  - shuffled prior: `0.040000`
  - uniform sparse baseline: `0.050000`
- `75%` budget:
  - fixed prior: `0.050000`
  - shuffled prior: `0.040000`
  - uniform sparse baseline: `0.040000`

Interpretation:

- The fixed prior stays directionally better than the matched shuffled and
  uniform baselines at every tested budget.
- The effect still has a clear **middle-band sweet spot** at `50%`.
- Tightening to `25%` or loosening to `75%` keeps the direction but loses most
  of the gain.

## Transfer Read

Saved-prior transfer on `gsm8k_eval_70`:

- Qwen prior -> Qwen target: `0.085714`
- Qwen prior -> DeepSeek target: `0.014286`
- DeepSeek prior -> DeepSeek target: `0.014286`
- DeepSeek prior -> Qwen target: `0.057143`

Interpretation:

- The fixed-prior story is **asymmetric**.
- The strong Qwen-native prior does not transfer to DeepSeek.
- The DeepSeek-native prior is weak on DeepSeek and only partially lifts Qwen.
- The defensible claim is **pair-conditioned calibrated head budgeting**, not a
  universal transferable prior.

## Second Reasoning Task

SVAMP, same Qwen pair and same fixed head-prior branch:

- fixed head prior: `0.071429`
- shuffled-prior null: `0.042857`
- prior vs shuffled: delta `+0.0286`, method-only wins `4`, baseline-only wins
  `2`, bootstrap `[-0.0286, +0.1000]`, McNemar `0.6831`
- previous target-alone baseline on the same SVAMP split:
  `0.071429`
- previous text-to-text baseline on the same SVAMP split:
  `0.414286`

Interpretation:

- On SVAMP, the calibrated prior is better than a matched blind null.
- But it only recovers to **target-alone**, not beyond it.
- So SVAMP is currently a **boundary condition**, not a second positive
  replication.

## Live-Blend And Entropy Ablations

Same Qwen pair, `gsm8k_100`, `50%` budget:

- fixed peak-based prior: `0.070000`
- prior/live blend, `alpha=0.10`: `0.040000`
- prior/live blend, `alpha=0.25`: `0.060000`
- prior/live blend, `alpha=0.50`: `0.060000`
- fixed entropy-based prior: `0.050000`

Interpretation:

- A little live correction helps over the weakest baselines, but it still does
  **not** beat the pure fixed prior.
- Entropy-based head priors are directionally positive, but they underperform
  the peak-based calibrated prior.
- Right now the best branch is still the **pure fixed peak-based prior** rather
  than a live-corrected or entropy-derived variant.

## Shrinkage Ablations

Same protocol, with shrinkage applied to the fixed head prior before use:

- GSM8K-100, shrinkage `0.25`, target `global`: `0.050000`
- GSM8K-100, shrinkage `0.25`, target `uniform`: `0.070000`
- SVAMP-70, shrinkage `0.25`, target `global`: `0.071429`
- DeepSeek GSM8K-70 transfer, shrinkage `0.25`, target `global`: `0.014286`

Interpretation:

- Global shrinkage hurts the main GSM branch.
- Uniform shrinkage ties the unshrunken fixed prior rather than improving it.
- Shrinkage does not rescue SVAMP or cross-pair transfer.
- So shrinkage is currently a **null or weak regularization result**, not a new
  positive method branch.

## Retrieval-Head Routing Check

Same sparse `k_only` Qwen protocol, but replacing the previous head-budget
heuristics with `--per-head-position-budget-mode retrieval_peak`:

- Qwen -> Qwen, `gsm8k_100`: `0.030000`
- Qwen -> Qwen, `svamp_eval_70`: `0.071429`
- Qwen -> DeepSeek, `gsm8k_eval_70`: `0.014286`

Interpretation:

- The simple retrieval-head heuristic does **not** stabilize the current method.
- On `gsm8k_100`, it falls below target-alone (`0.0400`), below the older live
  sparse branch (`0.0400`), and well below the fixed-prior branch (`0.0700`).
- On SVAMP it only ties the existing boundary level rather than improving it.
- On DeepSeek it remains weak, so it does not improve transfer either.
- This means the next meaningful structure-aware pivots should target
  **head matching or QK geometry**, not just a sharper retrieval heuristic.

## Seed Stability

Fixed peak-based prior on `gsm8k_100`, same branch, same budget, recalibrated
translator seeds:

- `seed0`: `0.020000`
- `seed1`: `0.070000`
- `seed2`: `0.010000`

Interpretation:

- The current fixed-prior result is **high variance across calibration seeds**.
- That makes it unsuitable as the main paper headline in its current form.
- The fixed-prior branch remains interesting as a mechanism clue, but not yet
  as a stable method claim.

## What Survives

- `k_only` matters more than `v_only`.
- Sparse selection matters more than dense transport.
- The strongest current branch is head-budgeted and calibration-derived.
- Reviewer-proof nulls matter:
  - zero-byte attenuation
  - random translated
  - shuffled selector / shuffled prior
  - translated-only
  - text-to-text
- Cross-family transfer is still weak.
- Simple retrieval-head-style scoring is not enough on its own.

## What This Means For The Paper

Best honest claim today:

> Calibrated sparse key-head budgets can improve cross-model latent transport on
> a compatible same-family pair under matched bandwidth, but the effect is
> pair-conditioned, asymmetric, and not yet stable enough across calibration
> seeds for a broad reasoning-time method claim.

The live query-aware sparse branch no longer rescues that story on the larger
held-out GSM slice, because it is also unstable across seeds and averages below
target-alone.

That is strong enough for a tighter workshop story and promising for a main
paper only if the next replication steps succeed.

## Next Highest-Value Steps

1. Budget sweep for the fixed-prior branch: `0.25 / 0.50 / 0.75`, each with
   shuffled-prior and uniform baselines.
2. Treat the fixed-prior branch as a mechanism clue, not the headline, until it
   is stabilized across seeds.
3. Next method pivots from the new literature:
   - OT / permutation or gauge-aware head matching across models
   - attention-logit-preserving or QK-geometry head ranking
   - causal head scoring once the matching space is less noisy
   - only then revisit retrieval-head routing with a stronger structure-aware score
4. Keep SVAMP, ARC, cross-family transfer, and now seed instability in the
   paper as explicit failure boundaries.
5. Treat the live query-aware sparse branch as another mechanism clue unless a
   stronger retrieval-head or logit-preserving variant survives the same seed
   repeat.
