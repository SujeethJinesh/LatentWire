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

## Attention-Margin Check

Same Qwen sparse `k_only` protocol, but using the new last-token top-1 vs
top-2 attention gap as the head score:

- GSM8K-100, fixed `attention_margin` prior: `0.050000`
- GSM8K-100, matched shuffled null: `0.040000`
- GSM8K-100, live `attention_margin` budget: `0.040000`
- GSM8K-100, fixed prior vs shuffled:
  - delta `+0.0100`
  - prior-only wins `1`
  - null-only wins `0`
  - bootstrap `[0.0000, 0.0300]`
  - McNemar `1.0000`
- SVAMP-70, fixed `attention_margin` prior: `0.071429`
- DeepSeek GSM8K-70 transfer, fixed `attention_margin` prior: `0.014286`

Interpretation:

- The `attention_margin` prior is directionally cleaner than its shuffled null.
- It does **not** beat the older peak-based fixed prior (`0.0700` on
  `gsm8k_100`).
- The live `attention_margin` branch collapses back to target-alone.
- SVAMP and DeepSeek remain unchanged boundary cases.
- So `attention_margin` is a useful bounded ablation, not a new headline method.

## Grouped-CCA Structural Pivot

New checkpoint:

- `checkpoints/cca_pivot_20260418/qwen25_to_qwen3_grouped_cca_headhalf_affine.pt`

Calibration read:

- `K cos = 0.023`, `K rel_err = 1.040`
- `V cos = 0.058`, `V rel_err = 3.042`

Held-out results:

- GSM8K-100, grouped-CCA + fixed peak prior: `0.030000`
- GSM8K-100, grouped-CCA + shuffled-prior null: `0.060000`
- SVAMP-70, grouped-CCA + fixed peak prior: `0.171429`
- SVAMP-70, grouped-CCA + shuffled-prior null: `0.128571`

Paired reads:

- GSM8K-100, fixed prior vs shuffled:
  - delta `-0.0300`
  - prior-only wins `1`
  - shuffled-only wins `4`
  - bootstrap `[-0.0800, +0.0100]`
- SVAMP-70, fixed prior vs shuffled:
  - delta `+0.042857`
  - prior-only wins `7`
  - shuffled-only wins `4`
  - bootstrap `[-0.042857, +0.128571]`
- SVAMP-70, grouped-CCA prior vs old peak-prior branch:
  - old branch: `0.071429`
  - grouped CCA: `0.171429`
  - delta `+0.1000`
  - grouped-only wins `11`
  - old-only wins `4`
  - bootstrap `[0.0000, +0.2000]`

Interpretation:

- Grouped CCA is **bad on GSM8K-100** under the same control ladder.
- But it is the first structural pivot that **materially lifts SVAMP**, even
  though the matched shuffled null also rises.
- The real story is now more specific:
  useful latent structure may be **task-conditioned subspace geometry**, not a
  single routing rule that transfers cleanly across all reasoning sets.

## Grouped-CCA Expected-Attention Follow-Up

Same grouped-CCA checkpoint, but replacing the live position selector with
fixed query-blind position priors:

- GSM8K-100, calibration expected-attention prior: `0.030000`
- GSM8K-100, uniform prior null: `0.030000`
- SVAMP-70, calibration expected-attention prior: `0.171429`
- SVAMP-70, uniform prior null: `0.171429`

Interpretation:

- On the grouped-CCA branch, the **position prior is not the lever**.
- The task split remains unchanged under both calibration-derived and uniform
  query-blind priors.
- That means the grouped-CCA behavior is being driven much more by the
  checkpoint / head-subspace geometry than by better fixed position routing.

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
- A logit-gap-style attention proxy is also not enough on its own.
- A structural subspace pivot can help on one reasoning boundary (SVAMP) while
  hurting another (GSM8K), so task-conditioned geometry is now a live hypothesis.
- Fixed query-blind position priors do not explain that split.
- Head-level expected-attention scoring on top of grouped CCA improves the weak
  GSM fixed-prior branch from `0.0300` to `0.0600`, but that only ties the old
  grouped-CCA shuffled-null result `0.0600`, so it is another bounded ablation
  rather than a new source-specific win.
- A simple permutation-invariant head-prior shortcut also failed cleanly:
  `attention_match` on the exact Qwen GSM70 branch scored `0.042857`, below the
  old fixed head-prior branch `0.085714` and equal to the old shuffled-prior
  null `0.042857`.
- The competitor baseline path is now real:
  `C2C` ran end to end on the exact Qwen pair and scored `0.128571` on
  `data/gsm8k_eval_70.jsonl`, above our current best same-pair GSM70 branch
  `0.085714`.

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

There is now a stronger constraint on the paper than before:

> even in the favorable same-family Qwen setting, our best current branch still
> trails a published baseline (`C2C`) on the held-out GSM70 split.

And a second structural constraint:

> simple permutation-aware rank matching is not enough to rescue the same
> branch, so the remaining blockers are more likely QK-geometry / attention
> fidelity or a richer symmetry problem than plain head-order mismatch.

## Next Highest-Value Steps

1. Budget sweep for the fixed-prior branch: `0.25 / 0.50 / 0.75`, each with
   shuffled-prior and uniform baselines.
2. Treat the fixed-prior branch as a mechanism clue, not the headline, until it
   is stabilized across seeds.
3. Next method pivots from the new literature:
   - OT / permutation or gauge-aware head matching across models
   - attention-fidelity-preserving head ranking after expected-attention ties the shuffled null
   - extend the grouped CCA branch on SVAMP-like tasks before treating it as a general method
   - causal head scoring once the matching space is less noisy
   - only then revisit retrieval-head routing with a stronger structure-aware score
   - use `C2C` as the first external bar and try to beat it on the exact Qwen GSM split
4. Keep SVAMP, ARC, cross-family transfer, and now seed instability in the
   paper as explicit failure boundaries.
5. Treat the live query-aware sparse branch as another mechanism clue unless a
   stronger retrieval-head or logit-preserving variant survives the same seed
   repeat.
