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

## Query-Conditioned Bridge-Bank Follow-Ups

After the fair Qwen3 prompt/thinking control was locked in with shared chat
serialization and `enable_thinking=False` on both sides, I tested two
query-conditioned bridge-bank variants on top of the live grouped-subspace
transport family:

- `bridge_low_rank_bank`
- `bridge_ridge_residual_bank`

Both used the same `64`-prompt calibration slice, the same sparse `K-only`
protocol, and the same fixed head prior from `.debug/head_prior_64.txt`.
Both were clean negatives on the first controlled `gsm8k_5` smoke:

- `bridge_low_rank_bank`: `0.0000`, `722,107.7` bytes
- `bridge_ridge_residual_bank`: `0.0000`, `722,107.7` bytes

Interpretation:

- replacing the stable bridge with an attention-template-routed expert bank is
  too aggressive
- even preserving the full `bridge_ridge` base and routing only a low-rank
  residual bank is still not enough when the routing signal is just a mean
  attention template
- so the bridge lane is still alive, but the next serious version has to use a
  **richer query-conditioned signal** such as QK/retrieval features or a
  richer interaction-level / distillation target, not another attention-template
  gate or bank

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
- A cheap attention-fidelity / QK-geometry proxy is directionally better than
  `attention_match`, but still bounded:
  `attention_fidelity` on the same Qwen GSM70 branch scored `0.057143`, which
  is still below the old fixed head-prior branch `0.085714` and below `C2C`
  at `0.128571`.
- A richer calibration-time template transport branch also failed on the same
  Qwen GSM70 path:
  `attention_template_transport` scored `0.042857`, so upgrading the fixed
  prior from scalar head scores to full per-head attention templates still did
  not beat the old fixed prior or `C2C`.
- A direct gauge-aware Procrustes overlap score also failed cleanly on the same
  Qwen GSM70 path:
  `attention_procrustes` scored `0.028571`, which is below the old fixed
  head-prior branch `0.085714` and far below `C2C` at `0.128571`.
- The paired gap is not small:
  compared with the old fixed prior, `attention_procrustes` is `-0.057143`
  with `0` Procrustes-only wins and `4` fixed-prior-only wins; compared with
  `C2C` it is `-0.100000` with `2` Procrustes-only wins and `9` C2C-only wins.
- A stronger post-quantization linear correction layer is still only a bounded
  repair:
  the new `ridge` quantization-correction checkpoint scored `0.057143` on the
  same Qwen GSM70 split, better than `attention_procrustes` but still below
  the old fixed prior `0.085714` and below `C2C` `0.128571`.
- Paired against the old fixed prior, the ridge-correction branch is
  `-0.028571` with `1` ridge-only win and `3` fixed-prior-only wins; paired
  against `C2C` it is `-0.071429` with `3` ridge-only wins and `8` C2C-only
  wins.
- A lightweight Sinkhorn-style soft transport probe is also bounded:
  `attention_sinkhorn` on the same Qwen GSM70 split scored `0.042857`, which
  is below the old fixed prior `0.085714`, below the ridge-correction branch
  `0.057143`, and below `C2C` `0.128571`.
- Paired against the old fixed prior, the Sinkhorn branch is `-0.042857` with
  `0` Sinkhorn-only wins and `3` fixed-prior-only wins; paired against
  ridge-correction it is `-0.014286` with `0` Sinkhorn-only wins and `1`
  ridge-only win; paired against `C2C` it is `-0.085714` with `3`
  Sinkhorn-only wins and `9` C2C-only wins.
- `KVComm` is now partially bootstrapped on the exact same Qwen GSM70 split,
  but only with a clearly labeled heterogeneous-geometry compatibility lift:
  stock `KVComm` could not run on `Qwen/Qwen2.5-0.5B-Instruct ->
  Qwen/Qwen3-0.6B` because the pair mismatches both KV-head count (`2 -> 8`)
  and per-head dimensionality (`64 -> 128`).
- Under that compatibility lift, the held-out `KVComm` replay on
  `data/gsm8k_eval_70.jsonl` scored `0.000000`, so it is not competitive on
  this pair even after a fair held-out layer-selection pass.
- The held-out calibration sweep still found a weak dev-side signal:
  on `data/gsm8k_gate_search_30.jsonl`, `KVComm` peaked at `0.033333` with a
  `0.50` top-layer fraction and `14` selected layers, while `0.25`, `0.75`,
  and `1.00` all fell back to `0.000000`.
- Paired against our old fixed-prior branch (`0.085714`), the compatibility-
  lifted `KVComm` replay is `-0.085714`, with `0` KVComm-only wins and `6`
  fixed-prior-only wins.
- Paired against published `C2C` (`0.128571`), the compatibility-lifted
  `KVComm` replay is `-0.128571`, with `0` KVComm-only wins and `9`
  `C2C`-only wins.
- The competitor baseline path is now real:
  `C2C` ran end to end on the exact Qwen pair and scored `0.128571` on
  `data/gsm8k_eval_70.jsonl`, above our current best same-pair GSM70 branch
  `0.085714`.
- That external gap is not just a small-split artifact:
  `C2C` also scored `0.110000` on `data/gsm8k_100.jsonl`, above our current
  best same-pair GSM100 branch `0.070000`.
- The SVAMP external bar is even stronger:
  `C2C` scored `0.442857` on `data/svamp_eval_70.jsonl`, which is far above
  our best current SVAMP branch (`0.171429` from grouped CCA) and above the
  older text-to-text reference (`0.414286`).

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

And a stronger external-task constraint:

> on SVAMP, the published `C2C` baseline is not just ahead of our sparse or
> grouped-CCA branches; it also clears the older text-to-text reference, so the
> next method class has to compete with a stronger reasoning bar than our
> internal control ladder alone.

And a second structural constraint:

> simple permutation-aware rank matching is not enough to rescue the same
> branch, so the remaining blockers are more likely QK-geometry / attention
> fidelity or a richer symmetry problem than plain head-order mismatch.

And a third bounded update:

> a cheap attention-fidelity proxy partially repairs the failed permutation
> shortcut, but it still does not beat the old fixed prior or the external
> `C2C` baseline, so this alone is not the missing ingredient either.

And a fourth constraint:

> even richer calibration-time head templates still collapse back to
> `0.042857` on Qwen GSM70, so the selector family itself is likely saturated
> on this branch.

And a fifth structural constraint:

> a cheap gauge-aware Procrustes overlap score is both slower and weaker on the
> same split, so simple orthogonal-invariant head scoring is not the missing
> ingredient either.

And a sixth method-class constraint:

> even a stronger full linear post-quantization correction layer only climbs to
> `0.057143` on Qwen GSM70, so post-transport cleanup alone is not enough to
> close the gap to either the old fixed prior or `C2C`.

And a seventh transport constraint:

> a lightweight Sinkhorn-style soft transport score still collapses to
> `0.042857` on Qwen GSM70, so evaluator-level soft matching alone is not
> enough either.

And an eighth external-baseline constraint:

> even a compatibility-lifted `KVComm` replay collapses to `0.000000` on the
> exact heterogeneous Qwen GSM70 pair, so training-free raw selective KV
> sharing appears to be extremely brittle once KV-head geometry itself stops
> matching.

And a ninth canonicalization constraint:

> target-only whitening is clearly the wrong quotient-space shortcut on the
> Qwen pair: its calibration quality collapses to roughly `K cos 0.284 / V cos
> 0.274` with relative error around `0.965`.

And a tenth symmetric-canonicalization constraint:

> even symmetric source+target whitening only reaches `0.071429` on Qwen
> GSM70, which is better than some failed routing probes but still below the
> old fixed prior `0.085714` and below `C2C` `0.128571`.

Paired reads for the symmetric-whitening branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.014286`
  - symwhite-only wins `3`
  - fixed-prior-only wins `4`
  - bootstrap `[-0.085714, +0.057143]`
  - McNemar `1.0000`
- vs `C2C`:
  - delta `-0.057143`
  - symwhite-only wins `3`
  - `C2C`-only wins `7`
  - bootstrap `[-0.128571, +0.028571]`
  - McNemar `0.3428`

And an eleventh transport-fit constraint:

> a much stronger grouped transport map with residual rank `64` improves
> calibration quality sharply (`K cos 0.925`, rel err `0.370`) but still
> collapses to `0.014286` on Qwen GSM70.

Paired reads for the grouped-transport branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.071429`
  - grouped-transport-only wins `0`
  - fixed-prior-only wins `5`
  - bootstrap `[-0.128571, -0.014286]`
  - McNemar `0.0736`
- vs `C2C`:
  - delta `-0.114286`
  - grouped-transport-only wins `1`
  - `C2C`-only wins `9`
  - bootstrap `[-0.200000, -0.028571]`
  - McNemar `0.0269`

Interpretation:

- better offline transport fit is not enough
- the next missing ingredient looks more like runtime or example-conditioned
  correction / fusion than another static transport map

And a twelfth tiny-correction constraint:

> the first learned example-conditioned diagonal affine fuser also fails
> cleanly: on the same Qwen GSM70 split it collapses to `0.000000`, below the
> old fixed prior `0.085714` and below `C2C` `0.128571`.

Paired reads for the learned-affine branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.085714`
  - learned-affine-only wins `0`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.157143, -0.028571]`
  - McNemar `0.0412`
- vs `C2C`:
  - delta `-0.128571`
  - learned-affine-only wins `0`
  - `C2C`-only wins `9`
  - bootstrap `[-0.214286, -0.057143]`
  - McNemar `0.0077`

Interpretation:

- a tiny diagonal example-conditioned correction is not enough either
- the next missing ingredient is likely a **stronger transport map plus**
  correction, not correction alone

And a thirteenth stronger-correction constraint:

> a richer per-head ridge fuser over `[translated, target]` also collapses to
> `0.000000` on the same Qwen GSM70 split, matching the failure of the smaller
> diagonal learned-affine branch and staying below the old fixed prior
> `0.085714` and below `C2C` `0.128571`.

Paired reads for the learned-head-ridge branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.085714`
  - learned-head-ridge-only wins `0`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.157143, -0.028571]`
  - McNemar `0.0412`
- vs `C2C`:
  - delta `-0.128571`
  - learned-head-ridge-only wins `0`
  - `C2C`-only wins `9`
  - bootstrap `[-0.214286, -0.057143]`
  - McNemar `0.0077`

Interpretation:

- a stronger fusion family still does not rescue the current transport path
- the next credible move is **transport-first**: improve the transport map
  itself, then reconsider learned correction on top

And a fourteenth hard-matching transport constraint:

> a translator-side hard grouped permutation map reaches only `0.028571` on
> Qwen GSM70. That is better than the total collapse of some other transport
> probes, but still below the old fixed prior `0.085714` and below `C2C`
> `0.128571`.

Paired reads for the grouped-permutation branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.057143`
  - grouped-permutation-only wins `0`
  - fixed-prior-only wins `4`
  - bootstrap `[-0.114286, -0.014286]`
  - McNemar `0.1336`
- vs `C2C`:
  - delta `-0.100000`
  - grouped-permutation-only wins `1`
  - `C2C`-only wins `8`
  - bootstrap `[-0.185714, -0.028571]`
  - McNemar `0.0455`

Interpretation:

- hard matching is directionally less bad than soft grouped transport, but it
  still does not recover the main Qwen reasoning signal
- that suggests naive permutation recovery alone is not the whole symmetry fix

And a fifteenth geometry-aware transport update:

> adding a small grouped spectral-signature penalty to the soft transport cost
> lifts the same Qwen GSM70 setting to `0.042857`. That is still below the old
> fixed prior `0.085714` and below `C2C` `0.128571`, but it is materially
> better than grouped transport `0.014286` and grouped permutation `0.028571`.

Paired reads for the grouped-signature-transport branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.042857`
  - grouped-signature-only wins `0`
  - fixed-prior-only wins `3`
  - bootstrap `[-0.100000, +0.000000]`
  - McNemar `0.2482`
- vs `C2C`:
  - delta `-0.085714`
  - grouped-signature-only wins `3`
  - `C2C`-only wins `9`
  - bootstrap `[-0.185714, +0.000000]`
  - McNemar `0.1489`
- vs grouped permutation:
  - delta `+0.014286`
  - grouped-signature-only wins `2`
  - grouped-permutation-only wins `1`
  - bootstrap `[-0.028571, +0.057143]`
  - McNemar `1.0000`

Interpretation:

- geometry-aware transport cost is the first transport-only change that helps
  directionally on the main Qwen split
- but it is still a bounded gain, not a competitive result against the old
  fixed prior or `C2C`

And a sixteenth subspace-aware transport update:

> replacing the spectral-signature penalty with a principal-subspace mismatch
> penalty leaves the exact Qwen GSM70 result unchanged at `0.042857`. So
> subspace-aware grouped transport does not improve on the earlier
> grouped-signature branch in the current regime.

Paired reads for the grouped-subspace-transport branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.042857`
  - grouped-subspace-only wins `0`
  - fixed-prior-only wins `3`
  - bootstrap `[-0.100000, +0.000000]`
  - McNemar `0.2482`
- vs `C2C`:
  - delta `-0.085714`
  - grouped-subspace-only wins `3`
  - `C2C`-only wins `9`
  - bootstrap `[-0.185714, +0.000000]`
  - McNemar `0.1489`
- vs grouped signature transport:
  - delta `+0.000000`
  - grouped-subspace-only wins `0`
  - grouped-signature-only wins `0`
  - bootstrap `[+0.000000, +0.000000]`
  - McNemar `1.0000`

Interpretation:

- geometry-aware transport is still the right direction, but the current
  grouped transport family seems saturated on exact Qwen GSM70
- moving from coarse spectral signatures to principal-subspace mismatch does
  not change held-out behavior here
- the remaining viable internal branch is now richer transport itself
  (stronger OT / canonicalization), not another small tweak to the current
  grouped cost family

And a seventeenth canonical-subspace transport update:

> fitting each grouped block in a shared low-rank canonical basis drops the
> exact Qwen GSM70 result to `0.028571`. So the first clean low-rank
> canonical-subspace shortcut is also a negative result.

Paired reads for the grouped-canonical-transport branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.057143`
  - grouped-canonical-only wins `1`
  - fixed-prior-only wins `5`
  - bootstrap `[-0.128571, +0.014286]`
  - McNemar `0.2207`
- vs `C2C`:
  - delta `-0.100000`
  - grouped-canonical-only wins `1`
  - `C2C`-only wins `8`
  - bootstrap `[-0.185714, -0.014286]`
  - McNemar `0.0455`
- vs grouped signature transport:
  - delta `-0.014286`
  - grouped-canonical-only wins `1`
  - grouped-signature-only wins `2`
  - bootstrap `[-0.071429, +0.028571]`
  - McNemar `1.0000`

Interpretation:

- low-rank canonicalization by itself does not rescue the transport map
- the likely remaining gap is now richer transport or transport-plus-correction,
  not another small grouped or canonical-basis shortcut

And an eighteenth transport-plus-correction update:

> adding a **rank-4 residual** on top of grouped-subspace transport lifts the
> exact Qwen GSM70 result to `0.057143`. That is still below the old fixed
> prior `0.085714` and below `C2C` `0.128571`, but it is the first
> transport-plus-correction branch that clearly improves over the pure
> transport-only grouped family.

Paired reads for the grouped-subspace-plus-rank4-residual branch on Qwen GSM70:

- vs grouped subspace transport:
  - delta `+0.014286`
  - grouped-subspace-resid4-only wins `1`
  - grouped-subspace-only wins `0`
  - bootstrap `[+0.000000, +0.042857]`
  - McNemar `1.0000`
- vs old fixed prior:
  - delta `-0.028571`
  - grouped-subspace-resid4-only wins `0`
  - fixed-prior-only wins `2`
  - bootstrap `[-0.071429, +0.000000]`
  - McNemar `0.4795`
- vs `C2C`:
  - delta `-0.071429`
  - grouped-subspace-resid4-only wins `4`
  - `C2C`-only wins `9`
  - bootstrap `[-0.171429, +0.028571]`
  - McNemar `0.2673`

Interpretation:

- transport-plus-correction is directionally better than transport alone
- but the first small residual still does not clear the old fixed prior or
  the `C2C` baseline
- that keeps the paper in blocker/mechanism territory, while pointing to the
  next live method lane more clearly than before

And a nineteenth covariance-aware transport-plus-correction update:

> replacing the subspace-aware cost with a covariance-aware cost and keeping
> the same rank-4 residual drops exact Qwen GSM70 to `0.014286`. So covariance
> geometry is **not** the next shortcut in the current transport family.

Paired reads for the grouped-covariance-plus-rank4-residual branch on Qwen GSM70:

- vs grouped subspace transport + rank-4 residual:
  - delta `-0.042857`
  - grouped-covariance-resid4-only wins `0`
  - grouped-subspace-resid4-only wins `3`
  - bootstrap `[-0.100000, +0.000000]`
  - McNemar `0.2482`
- vs old fixed prior:
  - delta `-0.071429`
  - grouped-covariance-resid4-only wins `0`
  - fixed-prior-only wins `5`
  - bootstrap `[-0.128571, -0.014286]`
  - McNemar `0.0736`
- vs `C2C`:
  - delta `-0.114286`
  - grouped-covariance-resid4-only wins `1`
  - `C2C`-only wins `9`
  - bootstrap `[-0.200000, -0.028571]`
  - McNemar `0.0269`

Interpretation:

- richer covariance geometry is not helping in this regime
- the best internal transport-plus-correction branch remains grouped-subspace
  transport plus a rank-4 residual at `0.057143`

## Next Highest-Value Steps

1. Treat the fixed-prior branch as a mechanism clue, not the headline, until it
   is stabilized across seeds.
2. Next method pivots from the new literature:
   - OT / permutation or gauge-aware head matching across models
   - deeper transport maps in the translation path rather than evaluator-only
     soft scores
   - attention-fidelity-preserving head ranking after expected-attention ties
     the shuffled null
   - extend the grouped CCA branch on SVAMP-like tasks before treating it as a general method
   - move toward transport plus tiny correction layers once pure routing stops improving against `C2C`
   - after grouped transport failed despite much better calibration fit, move
     specifically toward example-conditioned or learned correction on top of a
     transport map rather than more static transport variants
   - deprioritize standalone correction-only variants if they stay below the old fixed prior on GSM70
   - after the learned-affine collapse to `0.000000`, treat diagonal
     correction-only fusers as ruled out on the main Qwen GSM70 split
   - after the per-head ridge collapse to the same `0.000000`, treat
     richer correction-only fusers as ruled out too until the transport map
     itself improves
   - after grouped permutation only reaches `0.028571`, treat simple
     transport-only symmetry fixes as bounded too; the remaining viable branch
     is likely richer OT/canonicalized transport rather than simple hard
     assignment
   - deprioritize evaluator-level soft-transport variants if they also stay below the old fixed prior on GSM70
   - deprioritize whitening-only or symmetric-canonicalization-only pivots if
     they stay below the old fixed prior on GSM70
   - causal head scoring once the matching space is less noisy
   - only then revisit retrieval-head routing with a stronger structure-aware score
3. Keep `C2C` as the main external bar on the exact Qwen pair; treat the
   compatibility-lifted `KVComm` result as evidence that heterogeneous raw KV
   sharing is itself a hard blocker, not yet as a clean apples-to-apples
   leaderboard baseline.
4. Keep SVAMP, ARC, cross-family transfer, and now seed instability in the
   paper as explicit failure boundaries.
5. Treat the live query-aware sparse branch as another mechanism clue unless a
   stronger retrieval-head or logit-preserving variant survives the same seed
   repeat.

And a twentieth attention-template transport-plus-correction update:

> adding calibration-time grouped last-token attention templates to the
> grouped transport cost, then keeping the same rank-4 residual, reaches
> `0.042857` on exact Qwen GSM70. This first template run used a practical
> `64`-prompt calibration slice because full-attention template extraction over
> all `600` prompts is too slow for the current inner loop. It ties the earlier
> transport-only plateau and stays below the old fixed prior `0.085714`, below
> the best internal transport-plus-correction branch `0.057143`, and below
> `C2C` `0.128571`.

Paired reads for the grouped-template-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.042857`
  - grouped-template-resid4-only wins `1`
  - fixed-prior-only wins `4`
  - bootstrap `[-0.114286, +0.014286]`
  - McNemar `0.3711`
- vs `C2C`:
  - delta `-0.085714`
  - grouped-template-resid4-only wins `2`
  - `C2C`-only wins `8`
  - bootstrap `[-0.171429, +0.000000]`
  - McNemar `0.1138`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.014286`
  - grouped-template-resid4-only wins `1`
  - grouped-subspace-resid4-only wins `2`
  - bootstrap `[-0.071429, +0.028571]`
  - McNemar `1.0000`

Interpretation:

- behavior-matched head transport is more principled than the earlier grouped
  geometry probes, but on the main held-out GSM70 split it still does not
  recover the missing reasoning signal
- adding templates to the grouped transport cost is therefore not enough by
  itself; the next live lane is still richer transport-plus-correction rather
  than another light grouped penalty

And a twenty-first hybrid transport update:

> combining the grouped attention-template penalty and the grouped subspace
> penalty in one transport score, then keeping the same rank-4 residual,
> drops exact Qwen GSM70 to `0.014286`. So naively stacking the two best
> partial transport hints does **not** improve the branch; it makes it worse.

Paired reads for the grouped-template-subspace-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.071429`
  - grouped-template-subspace-resid4-only wins `1`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.142857, +0.000000]`
  - McNemar `0.1306`
- vs `C2C`:
  - delta `-0.114286`
  - grouped-template-subspace-resid4-only wins `0`
  - `C2C`-only wins `8`
  - bootstrap `[-0.185714, -0.042857]`
  - McNemar `0.0133`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.042857`
  - grouped-template-subspace-resid4-only wins `1`
  - grouped-subspace-resid4-only wins `4`
  - bootstrap `[-0.100000, +0.014286]`
  - McNemar `0.3711`
- vs grouped template transport + rank-4 residual:
  - delta `-0.028571`
  - grouped-template-subspace-resid4-only wins `0`
  - grouped-template-resid4-only wins `2`
  - bootstrap `[-0.071429, +0.000000]`
  - McNemar `0.4795`

Interpretation:

- the grouped subspace penalty and grouped template penalty are not additive
  in the current solver
- the best internal lane is still the simpler grouped-subspace-plus-rank4
  branch at `0.057143`
- so the next serious method change has to be a different transport class,
  not another grouped-penalty combination

And a twenty-second broadcast transport update:

> to break the grouped `gcd(2, 8) = 2` bottleneck directly, I fit a new
> `broadcast_template_transport` branch with a true rectangular `2 -> 8`
> head-transport plan built from per-head calibration-time attention
> templates, then kept the same rank-4 residual correction on top. Offline
> calibration fit looked directionally promising on the same `64`-prompt
> slice: average `K` cosine `0.868` with relative Frobenius error `0.470`.
> But on exact Qwen GSM70 the held-out result collapsed to `0.000000`.

Paired reads for the broadcast-template-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.085714`
  - broadcast-template-resid4-only wins `0`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.157143, -0.028571]`
  - McNemar `0.0412`
- vs `C2C`:
  - delta `-0.128571`
  - broadcast-template-resid4-only wins `0`
  - `C2C`-only wins `9`
  - bootstrap `[-0.214286, -0.057143]`
  - McNemar `0.0077`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.057143`
  - broadcast-template-resid4-only wins `0`
  - grouped-subspace-resid4-only wins `4`
  - bootstrap `[-0.114286, -0.014286]`
  - McNemar `0.1336`

Interpretation:

- escaping the grouped `2 x 2` transport bottleneck is **not** enough by
  itself; a finer rectangular `2 -> 8` head map still does not recover the
  missing reasoning signal
- better offline transport fit is again not predictive of held-out reasoning
  behavior
- the remaining live positive-method lane is now narrower: richer OT /
  retrieval-template / QK-fidelity transport, not more grouped or lightly
  behavior-matched transport variants

And a twenty-third OT transport update:

> I then replaced the broadcast row-softmax plan with a true rectangular
> Sinkhorn-style OT plan so each target head receives a normalized mixture over
> source heads while the two source heads carry balanced load across the eight
> target heads. This `broadcast_template_ot_transport` branch fit even better
> offline on the same `64`-prompt slice: average `K` cosine `0.883` with
> relative Frobenius error `0.447`. But on exact Qwen GSM70 it still collapsed
> to `0.000000`.

Paired reads for the broadcast-template-OT-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.085714`
  - broadcast-template-ot-resid4-only wins `0`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.157143, -0.028571]`
  - McNemar `0.0412`
- vs `C2C`:
  - delta `-0.128571`
  - broadcast-template-ot-resid4-only wins `0`
  - `C2C`-only wins `9`
  - bootstrap `[-0.214286, -0.057143]`
  - McNemar `0.0077`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.057143`
  - broadcast-template-ot-resid4-only wins `0`
  - grouped-subspace-resid4-only wins `4`
  - bootstrap `[-0.114286, -0.014286]`
  - McNemar `0.1336`
- vs broadcast template transport + rank-4 residual:
  - delta `+0.000000`
  - broadcast-template-ot-resid4-only wins `0`
  - broadcast-template-resid4-only wins `0`
  - bootstrap `[+0.000000, +0.000000]`
  - McNemar `1.0000`

Interpretation:

- in the current calibration-time attention-template space, richer many-to-many
  OT is **not** enough; it exactly matches the `0.000000` collapse of the
  simpler broadcast branch
- so if OT still lives as the final positive-method lane, it likely has to
  live in a different representation space: retrieval-template or QK-fidelity
  transport, not the current attention-template space alone

And a twenty-fourth peak-template OT update:

> I then changed only the broadcast OT template representation, replacing mean
> attention mass with a simple peak-location histogram per head across the
> same `64`-prompt calibration slice. This `broadcast_peak_template_ot_transport`
> branch keeps the same rectangular Sinkhorn-style `2 -> 8` plan and the same
> rank-4 residual. On exact Qwen GSM70 it improves from `0.000000` to
> `0.014286`.

Paired reads for the broadcast-peak-template-OT-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.071429`
  - broadcast-peak-template-ot-resid4-only wins `1`
  - fixed-prior-only wins `6`
  - bootstrap `[-0.142857, +0.000000]`
  - McNemar `0.1306`
- vs `C2C`:
  - delta `-0.114286`
  - broadcast-peak-template-ot-resid4-only wins `0`
  - `C2C`-only wins `8`
  - bootstrap `[-0.185714, -0.042857]`
  - McNemar `0.0133`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.042857`
  - broadcast-peak-template-ot-resid4-only wins `1`
  - grouped-subspace-resid4-only wins `4`
  - bootstrap `[-0.100000, +0.014286]`
  - McNemar `0.3711`
- vs broadcast template OT transport + rank-4 residual:
  - delta `+0.014286`
  - broadcast-peak-template-ot-resid4-only wins `1`
  - broadcast-template-ot-resid4-only wins `0`
  - bootstrap `[+0.000000, +0.042857]`
  - McNemar `1.0000`

Interpretation:

- the template representation does matter a little: retrieval-like peak
  templates are directionally better than mean attention templates in the same
  OT solver
- but the gain is still tiny and far below the fixed-prior branch and `C2C`
- so the remaining positive-method lane is now even narrower:
  retrieval-template or QK-fidelity transport may still be alive, but the
  current simple peak-template proxy is not enough to make the paper a
  positive-method result

And a twenty-fifth retrieval-spectrum-OT update:

> I then replaced the broadcast OT attention template with a retrieval-weighted
> per-head key-spectrum descriptor, keeping the same rectangular Sinkhorn-style
> `2 -> 8` plan and the same rank-4 residual on the same `64`-prompt
> calibration slice. Offline fit improved materially (`K` cosine `0.931`,
> relative Frobenius error `0.350`). The first dense replay collapsed to
> `0.000000`, but that was not the matched-budget protocol. Under the fair
> sparse `K-only` evaluation used for the other transport branches, exact Qwen
> GSM70 recovered only to `0.014286`.

Paired reads for the broadcast-retrieval-spectrum-OT-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.071429`
  - broadcast-retrieval-spectrum-ot-resid4-only wins `0`
  - fixed-prior-only wins `5`
  - bootstrap `[-0.128571, -0.014286]`
  - McNemar `0.0736`
- vs `C2C`:
  - delta `-0.114286`
  - broadcast-retrieval-spectrum-ot-resid4-only wins `1`
  - `C2C`-only wins `9`
  - bootstrap `[-0.200000, -0.028571]`
  - McNemar `0.0269`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.042857`
  - broadcast-retrieval-spectrum-ot-resid4-only wins `0`
  - grouped-subspace-resid4-only wins `3`
  - bootstrap `[-0.100000, +0.000000]`
  - McNemar `0.2482`

Interpretation:

- “use a richer calibration-time key descriptor” is also not enough in this
  simple spectral form
- better offline fit is still not predictive of held-out reasoning utility
- the dense replay was too pessimistic, but the fair matched sparse replay is
  still only tied with the peak-template OT branch and remains far below the
  fixed-prior branch and `C2C`
- the remaining positive-method lane is now extremely narrow:
  transport in a genuinely different query-conditioned representation space,
  such as QK-fidelity or richer retrieval templates, or else a pivot to a
  blocker/mechanism paper

And a twenty-sixth QK-template-OT update:

> I then replaced the retrieval-spectrum descriptor with a last-token QK-logit
> template, still keeping the same rectangular Sinkhorn-style `2 -> 8` plan
> and the same rank-4 residual on the same `64`-prompt calibration slice.
> Offline fit stayed strong (`K` cosine `0.931`, relative Frobenius error
> `0.350`; `V` cosine `0.613`, relative Frobenius error `0.781`). Under the
> fair matched sparse `K-only` evaluation used for the other transport
> branches, exact Qwen GSM70 again recovered only to `0.014286`.

Paired reads for the broadcast-QK-template-OT-plus-rank4-residual branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.071429`
  - broadcast-qk-template-ot-resid4-only wins `0`
  - fixed-prior-only wins `5`
  - bootstrap `[-0.128571, -0.014286]`
  - McNemar `0.0736`
- vs `C2C`:
  - delta `-0.114286`
  - broadcast-qk-template-ot-resid4-only wins `1`
  - `C2C`-only wins `9`
  - bootstrap `[-0.200000, -0.028571]`
  - McNemar `0.0269`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.042857`
  - broadcast-qk-template-ot-resid4-only wins `0`
  - grouped-subspace-resid4-only wins `3`
  - bootstrap `[-0.100000, +0.000000]`
  - McNemar `0.2482`
- vs broadcast retrieval-spectrum OT + rank-4 residual:
  - delta `+0.000000`
  - broadcast-qk-template-ot-resid4-only wins `0`
  - broadcast-retrieval-spectrum-ot-resid4-only wins `0`
  - bootstrap `[+0.000000, +0.000000]`
  - McNemar `1.0000`

Interpretation:

- moving into a simple last-token QK/logit template space is still not enough
- in this current broadcast OT family, QK templates do not outperform the
  retrieval-spectrum descriptor; they tie it exactly on both accuracy and
  paired behavior
- that makes the remaining positive-method lane narrower again:
  the next live idea would have to be a genuinely query-conditioned
  QK-fidelity or retrieval-template transport, not another static calibration-
  time descriptor inside the same transport family

And a twenty-seventh query-conditioned-budget update:

> I then stopped changing the translator and changed only the live sparse
> budget rule on top of the best current internal transport-plus-correction
> checkpoint, `grouped_subspace_transport + rank-4 residual`. Using
> `--per-head-position-budget-mode attention_qk_fidelity` on the exact same
> Qwen GSM70 setup gave `0.042857` at `157,989.2` average bytes.

Paired reads for the query-conditioned QK-fidelity budget branch on Qwen GSM70:

- vs old fixed prior:
  - delta `-0.042857`
  - qk-fidelity-budget-only wins `1`
  - fixed-prior-only wins `4`
  - bootstrap `[-0.100000, +0.014286]`
  - McNemar `0.3711`
- vs grouped subspace transport + rank-4 residual:
  - delta `-0.014286`
  - qk-fidelity-budget-only wins `1`
  - grouped-subspace-resid4-only wins `2`
  - bootstrap `[-0.071429, +0.028571]`
  - McNemar `1.0000`
- vs `C2C`:
  - delta `-0.085714`
  - qk-fidelity-budget-only wins `3`
  - `C2C`-only wins `9`
  - bootstrap `[-0.171429, +0.014286]`
  - McNemar `0.1489`

Interpretation:

- a genuinely query-conditioned QK-fidelity budget is a real branch, not a
  crash or a null replay
- but it is still below the best current grouped-subspace-plus-residual branch
  and still far below the fixed-prior branch and `C2C`
- it is also less byte-efficient than the best sparse internal branches
- so live query-conditioning at evaluation time alone is still not the missing
  ingredient

And a first runtime-head-gating update:

> I then added a new runtime path that keeps the same grouped-subspace-plus-
> rank4 checkpoint frozen and uses live head scores only to modulate the
> fusion gate per head. The first two smoke tests,
> `attention_qk_fidelity` and `attention_fidelity`, both collapsed to
> `0.000000` on the held-out `gsm8k_eval_10` slice.

Interpretation:

- the first soft query-conditioned gate variants do not look like the rescue
  path either
- that makes the next serious positive-method try narrower again:
  a richer query-conditioned transport cost is still more plausible than
  another gate-only variant on top of the same frozen transport map

And a twenty-eighth grouped-contrastive-template update:

> I then added a prompt-indexed grouped attention-template bank and a
> contrastive transport bonus inside the grouped transport family, still
> keeping the same `64`-prompt calibration slice and the same rank-4 residual.
> Calibration fit stayed strong (`K` cosine `0.932`, relative Frobenius error
> `0.351`; `V` cosine `0.608`, relative Frobenius error `0.791`). But the
> first held-out smoke on `gsm8k_eval_10`, under the same sparse `K-only`
> protocol used for the live grouped branches, collapsed to `0.000000` at
> `150,097.85` average bytes.

Interpretation:

- grouped prompt-contrastive templates are not enough to rescue the grouped
  transport lane
- this is another case where better offline calibration fit still does not
  translate into held-out reasoning gains
- it narrows the remaining positive-method lane again:
  if we keep pushing, the next transport cost has to be genuinely
  query-conditioned in a richer retrieval/QK space, not another grouped
  calibration-time template variant

And a twenty-ninth query-conditioned QK-template update:

> I then added `attention_qk_template_transport` as a new evaluator-side
> query-conditioned per-head budget metric. It builds fixed calibration-time
> QK templates from `--runtime-head-prior-file`, then soft-transports the fixed
> head-prior mass onto the live heads using the current example's last-token
> QK distributions. On the best current grouped-subspace-plus-rank4
> checkpoint, under the matched sparse `K-only` protocol, the first held-out
> `gsm8k_5` smoke scored `0.000000` at `142,353.225` average bytes.

Interpretation:

- adding fixed QK templates to evaluator-side budgeting is not enough to
  rescue the best current transport checkpoint
- query-conditioning at evaluation time keeps looking weaker than changing the
  transport itself
- if the positive-method lane gets another serious try, it should move
  query-conditioning into the transport cost or translator, not another
  evaluator-side budget rule

And a thirtieth grouped QK-retrieval transport update:

> I then moved the same idea into the transport path itself with
> `grouped_qk_retrieval_transport + rank-4 residual`. This branch keeps the
> grouped transport family but swaps the calibration-time template from mean
> attention to a grouped last-token QK retrieval profile. On the exact
> `Qwen2.5-0.5B-Instruct -> Qwen3-0.6B` pair with the standard `64`-prompt
> calibration slice, offline fit again looked respectable (`K` cosine `0.881`,
> relative Frobenius error `0.452`), but the first held-out matched sparse
> `K-only` smoke on `gsm8k_5` still collapsed to `0.000000` at
> `630,701.475` average bytes.

Interpretation:

- grouped retrieval-shaped QK templates are still too static when averaged over
  the calibration slice
- this is another case where better offline transport fit does not imply
  held-out reasoning transfer
- the branch is also too byte-heavy to be a plausible efficiency win even if it
  later recovers slightly
- if the positive-method lane stays alive, the next transport hypothesis has to
  be genuinely query-conditioned at runtime, not another calibration-time
  averaged descriptor

And a thirty-first prompt-conditioned QK-bank update:

> I then tried the lighter evaluator-side version of the same hypothesis on the
> best current internal checkpoint, `grouped_subspace_transport + rank-4
> residual`, using a prompt-indexed QK template bank instead of one averaged
> template. This new per-head budget metric,
> `attention_qk_bank_transport`, still collapsed on the first matched sparse
> `gsm8k_5` smoke to `0.000000` at `143,636.825` average bytes.

Interpretation:

- prompt-conditioning at the evaluator overlay level is still not enough to
  rescue the live checkpoint
- this is evidence that the remaining query mismatch probably lives in the
  transport or fusion path itself, not only in how we spend the sparse budget
- the evaluator-overlay lane now looks close to saturated

And a thirty-second tokenwise QK-gating update:

> I then moved the same query signal into the fusion path itself, without
> changing the frozen translator checkpoint, by adding
> `attention_qk_fidelity_tokenwise` as a runtime per-head, per-position gate
> override on top of the best current checkpoint,
> `grouped_subspace_transport + rank-4 residual`. This branch keeps the same
> sparse `K-only` protocol, the same fixed head prior from
> `.debug/head_prior_64.txt`, and the same `attention_prior` per-head position
> budget. On the first matched sparse `gsm8k_5` smoke, it still scored
> `0.000000` at `146,756.475` average bytes.

Interpretation:

- even a tokenwise query-conditioned fusion override is not enough, by itself,
  to rescue the best frozen transport checkpoint
- this is the strongest evidence so far that the remaining mismatch is not
  just “where to spend the budget” or “how hard to gate,” but the transport map
  itself
- if the positive-method lane stays alive, the next real shot has to be a
  query-conditioned transport or a tiny learned bridge, not another evaluator-
  side overlay on a frozen map

And a thirty-third low-rank bridge-correction update:

> I then added a tiny decoder-side low-rank bridge correction after
> quantize/dequantize, keeping the same grouped-subspace transport, the same
> rank-4 residual, and the same head-selection ratio. This new branch uses a
> reduced-rank linear correction (`rank=8`) in rotated target space instead of
> the older affine or full-ridge decoder repair. On the `64`-prompt
> calibration slice (`.debug/head_prior_64.txt`), the checkpoint fit looked
> similar to the old grouped-subspace family (`K` cosine `0.881`, relative
> Frobenius error `0.452`; `V` cosine `0.399`, relative Frobenius error
> `0.915`). The first matched sparse `gsm8k_5` smoke then reached `0.200000`
> at `298,063.425` average bytes, but the follow-up matched sparse
> `gsm8k_eval_10` slice fell back to `0.000000` at `297,233.538` average
> bytes.

Interpretation:

- this is the first adapter-style branch in a while with any nonzero held-out
  smoke signal at all, but it is not yet stable
- the low-rank bridge lane is therefore weakly alive, but currently too
  byte-heavy and too unstable to count as a real method win
- if the positive-method lane stays alive, a tiny learned bridge is still more
  promising than another evaluator overlay, but it likely needs either
  query-conditioning or a better training target to stabilize

And a thirty-fourth low-rank bridge stacking update:

> I then tried to stabilize that weak low-rank bridge signal by stacking the
> best existing live routing knobs on top of the same low-rank bridge
> checkpoint, all under the matched sparse `K-only` protocol on
> `gsm8k_eval_10`: retrieval-head-style runtime head selection
> (`runtime_head_selection_metric=retrieval_peak`, ratio `0.5`), direct
> QK-fidelity runtime head selection (`attention_qk_fidelity`, ratio `0.5`),
> and a prior-plus-live blend selector (`attention_blend`, ratio `0.5`,
> prior-alpha `0.25`). All three collapsed back to `0.000000` at the same
> reduced byte budget, `156,889.600` average bytes.

Interpretation:

- the current selector stack does not stabilize the weak low-rank bridge clue
- live routing can cut the byte cost of the bridge substantially, but it does
  not yet turn that bridge into a real reasoning improvement
- if the positive-method lane stays alive, the next step should be a
  query-conditioned bridge or projector, not just a more selective runtime mask

And a thirty-fifth low-rank bridge plus learned-fusion update:

> I then stacked one more existing small fix on top of that same bridge:
> grouped-subspace transport, rank-4 residual, low-rank bridge correction, and
> the stronger `learned_head_ridge` fusion head fit from the same `64`-prompt
> calibration slice. On the first matched sparse `gsm8k_5` smoke it still
> collapsed to `0.000000` at `298,063.425` average bytes.

Interpretation:

- simply stacking the current tiny linear fixes is not enough
- the next bridge-style attempt probably needs a better query-conditioned or
  interaction-level training target, not just another static linear layer on
  top of the existing map

And a thirty-sixth bridge-affine correction update:

> I then swapped the low-rank bridge for a simpler decoder-side
> `bridge_affine` correction that sees both the dequantized translated tensor
> and the pre-quant translated prediction, while keeping the same
> grouped-subspace transport, the same rank-4 residual, and the same `64`-
> prompt calibration slice. The fit again looked like the old grouped-subspace
> family (`K` cosine `0.881`, relative Frobenius error `0.452`; `V` cosine
> `0.399`, relative Frobenius error `0.915`). The held-out smokes reproduced
> the same weak adapter pattern as the low-rank bridge:
> - `gsm8k_5`: `0.200000` at `298,063.425` average bytes
> - matched `gsm8k_eval_10`: `0.000000` at `297,233.538` average bytes

Interpretation:

- giving the bridge both the pre-quant and post-quant translated states is not
  enough, by itself, to stabilize the weak adapter signal
- the bridge-affine lane therefore ties the low-rank bridge clue rather than
  improving it
- the adapter lane is still weakly alive, but the next bridge-style step
  probably has to be **query-conditioned or trained against a richer
  interaction target**, not another static linear repair

And a thirty-seventh bridge-ridge correction update:

> I then widened that bridge one more step, replacing the coordinatewise
> `bridge_affine` repair with a full `bridge_ridge` correction over both the
> dequantized translated tensor and the pre-quant translated prediction, while
> keeping the same grouped-subspace transport, the same rank-4 residual, and
> the same `64`-prompt calibration slice. Calibration fit again looked the same
> as the older grouped-subspace family (`K` cosine `0.881`, relative Frobenius
> error `0.452`; `V` cosine `0.399`, relative Frobenius error `0.915`), but
> the held-out behavior finally stabilized beyond a one-off smoke:
> - `gsm8k_5`: `0.400000` at `298,063.425` average bytes
> - matched `gsm8k_eval_10`: `0.100000` at `297,233.538` average bytes
> - `gsm8k_gate_search_30`: `0.066667` at `307,244.558` average bytes
> - exact `gsm8k_eval_70`: `0.042857` at `295,614.896` average bytes

Interpretation:

- this is the **first bridge-style branch that survives multiple held-out
  slices** instead of collapsing immediately after the first smoke
- it still does **not** beat the best internal transport-plus-correction branch
  (`grouped_subspace + rank-4 residual = 0.0571`) and it still stays well
  below the old fixed-prior bar (`0.0857`)
- the bridge lane is therefore now a **real live method family**, but not yet a
  positive-method win
- the strongest next bridge step is now likely **query-conditioned or
  distillation-shaped**, not another static linear variant

And a thirty-eighth Qwen3 prompt/thinking control update:

> I then finished the end-to-end prompt-control patch in both calibration and
> evaluation so we can run shared chat-template / thinking-mode controls
> cleanly. On the cheapest held-out fairness slice (`/tmp/gsm8k_eval_10.jsonl`)
> with shared chat serialization on both sides and `enable_thinking=False` for
> both tokenizers, the controlled results were:
> - `target-alone`: `0.100000`
> - `grouped_subspace + rank-4 residual + bridge_ridge`: `0.100000`
>   at `340,375.975` average bytes

Interpretation:

- the Qwen3 serialization / thinking-mode control is worth keeping because it
  removes a real fairness confound in the evaluation path
- but it does **not** rescue the bridge method: under the controlled prompt
  regime, `bridge_ridge` ties `target-alone` on `gsm8k_eval_10` instead of
  opening a new bridge margin
- prompt/thinking mismatch is therefore not the main explanation for why the
  bridge lane still trails the live internal bars
- future fair comparisons on this exact Qwen pair should keep the same prompt
  serialization and `enable_thinking=False`, but the next method step still has
  to be a **query-conditioned bridge / projector**, not another serialization
  tweak

And a thirty-ninth query-conditioned bridge-gating update:

> I then implemented the cheapest dynamic bridge variant that still changes the
> method class: `bridge_ridge_query`, which reuses the same decoder-side
> `bridge_ridge` correction but gates that correction by live target
> attention-template agreement with a calibration-time mean template. This was
> calibrated under the same fair prompt regime as the new Qwen control:
> shared chat serialization and `enable_thinking=False` on both sides. The
> resulting checkpoint fit looked like the old grouped-subspace family
> (`K` cosine `0.864`, relative Frobenius error `0.476`; `V` cosine `0.381`,
> relative Frobenius error `0.915`). Held-out controlled smokes were then:
> - `gsm8k_5`: `0.200000` at `722,107.700` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `720,487.313` average bytes

Interpretation:

- a simple live template-agreement gate on top of `bridge_ridge` is **not**
  enough; it makes the bridge less stable, not more stable
- this rules out the cheapest “Activated-LoRA-style” bridge-on/off variant in
  the current form
- the next bridge lane, if it stays alive, has to be **richer than a single
  scalar agreement gate**:
  likely a query-conditioned bridge bank, a low-rank dynamic projector, or a
  bridge trained against a richer interaction target

And a fortieth QK-routed residual-bank update:

> I then kept the stable `bridge_ridge` base but swapped the attention-template
> router for a live QK/retrieval-profile router:
> `bridge_ridge_qk_residual_bank`. This still uses a `4`-expert low-rank
> residual bank on top of grouped-subspace transport + rank-4 residual, and it
> was calibrated under the same fair shared-chat / `enable_thinking=False`
> regime. Calibration fit remained essentially unchanged from the grouped-
> subspace family (`K` cosine `0.864`, relative Frobenius error `0.476`; `V`
> cosine `0.381`, relative Frobenius error `0.915`), but the first held-out
> controlled smoke still collapsed:
> - `gsm8k_5`: `0.000000` at `722,107.700` average bytes

Interpretation:

- simply improving the router signal from mean attention to live QK/retrieval
  profiles is **not** enough
- the blocker is now more likely the **supervision target** than the routing
  signal by itself
- if the bridge lane stays alive, the next step should change what the bridge
  is trained to preserve, not just how experts are selected

And a forty-first retrieval-weighted bridge-fit update:

> I then kept the same global `bridge_ridge` form but changed the calibration
> objective itself: `bridge_ridge_qk_weighted` fits the bridge with aligned
> calibration samples reweighted by the target model's last-token QK retrieval
> importance instead of plain uniform latent regression. The checkpoint still
> sits on top of grouped-subspace transport + rank-4 residual and the same
> `64`-prompt calibration slice under the fair shared-chat /
> `enable_thinking=False` regime. Calibration fit again looked like the older
> grouped-subspace family (`K` cosine `0.864`, relative Frobenius error
> `0.476`; `V` cosine `0.381`, relative Frobenius error `0.915`). Held-out
> behavior was:
> - `gsm8k_5`: `0.200000` at `722,107.700` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `720,487.313` average bytes

Interpretation:

- changing the bridge **supervision target** is directionally more plausible
  than another router-only tweak, because it reproduced a nonzero smoke
- but this first retrieval-weighted version still does **not** stabilize on the
  larger controlled slice
- the bridge lane remains weakly alive, but the next serious attempt should
  probably move from weighted latent regression to a **richer interaction /
  affinity distillation target**

And a forty-second query-feature projector update:

> I then tried the first genuinely query-conditioned bridge inside the
> translator itself: `bridge_ridge_qk_projector`. Instead of using live query
> structure only as a gate, bank router, or calibration weight, this branch
> feeds aligned target query features directly into the decoder-side bridge by
> fitting a correction over both the translated state and the elementwise
> query-conditioned translated state. The checkpoint still sits on top of the
> same grouped-subspace transport + rank-4 residual and the same fair shared-
> chat / `enable_thinking=False` regime. Calibration fit again matched the
> older grouped-subspace family (`K` cosine `0.864`, relative Frobenius error
> `0.476`; `V` cosine `0.381`, relative Frobenius error `0.915`), but the
> first held-out fair smoke still collapsed:
> - `gsm8k_5`: `0.000000` at `722,107.700` average bytes

Interpretation:

- simply injecting live query features into a closed-form bridge projector is
  **not** enough in this first form
- that makes the blocker more precise: the missing ingredient is likely **not**
  just richer routing or richer live query features
- the next plausible bridge step now has to change the **supervision target**
  more substantially, likely toward token-interaction / affinity distillation,
  rather than another latent-regression projector variant

And a forty-third learned query-conditioned adapter update:

> I then tried the first tiny learned query-conditioned bridge on top of the
> same grouped-subspace transport + rank-4 residual checkpoint:
> `bridge_ridge_qk_adapter`. This branch keeps the closed-form `bridge_ridge`
> base but learns a low-rank residual adapter over query-conditioned translated
> features during calibration, under the same fair shared-chat /
> `enable_thinking=False` Qwen control. The checkpoint fit stayed in the same
> family as the older bridge branches (`K` cosine `0.864`, relative Frobenius
> error `0.476`; `V` cosine `0.381`, relative Frobenius error `0.915`). Held-
> out behavior was:
> - `gsm8k_5`: `0.200000` at `722,107.700` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `720,487.313` average bytes

Interpretation:

- this is the first **learned** query-conditioned bridge residual that stays
  nonzero on the cheapest fair smoke, so the adapter lane is not dead-on-
  arrival
- but it still fails to stabilize on the next controlled held-out slice, so it
  is not yet a real positive-method result
- the best next step is now sharper than before: keep the fair control on, but
  stop spending cycles on more latent-regression routing tricks and move the
  bridge supervision target toward **attention / affinity / token-interaction
  distillation**

And a forty-fourth affinity-distilled adapter update:

> I then kept the same learned query-conditioned residual adapter but changed
> the training objective from plain latent regression to a cheap
> query-conditioned affinity loss over calibration samples:
> `bridge_ridge_qk_affinity_adapter`. This branch still sits on top of the same
> grouped-subspace transport + rank-4 residual checkpoint and the same fair
> shared-chat / `enable_thinking=False` control. Calibration fit again matched
> the older bridge family (`K` cosine `0.864`, relative Frobenius error
> `0.476`; `V` cosine `0.381`, relative Frobenius error `0.915`). Held-out
> behavior was:
> - `gsm8k_5`: `0.200000` at `722,107.700` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `720,487.313` average bytes

Interpretation:

- a cheap affinity-matching term is **not** enough to stabilize the learned
  bridge lane; it exactly ties the old learned adapter on the smallest smoke
  and still collapses on the next controlled slice
- this closes the cheapest “interaction-shaped but still local” target we
  could add without changing the calibration data path
- the next serious bridge branch now has to use a **stronger teacher target**:
  explicit attention-behavior distillation, richer affinity supervision, or a
  prediction-level distillation term, not just another local residual loss

And a forty-fifth attention-KL adapter update:

> I then tried the strongest teacher target we could still graft onto the same
> calibration path without collecting new artifacts:
> `bridge_ridge_qk_attnkl_adapter`. This keeps the same learned
> query-conditioned residual adapter, but adds a sampled attention-logit KL
> loss over calibration query/key tensors. It still sits on top of the same
> grouped-subspace transport + rank-4 residual checkpoint and the same fair
> shared-chat / `enable_thinking=False` control. Calibration fit again matched
> the older bridge family (`K` cosine `0.864`, relative Frobenius error
> `0.476`; `V` cosine `0.381`, relative Frobenius error `0.915`). The first
> held-out fair smoke then collapsed immediately:
> - `gsm8k_5`: `0.000000` at `722,107.700` average bytes

Interpretation:

- a sampled attention-logit KL target is **not** enough in this local bridge
  form; it is strictly worse than the plain learned adapter on the cheapest
  fair smoke
- that closes the current cheap “stronger local teacher” family
- the next serious method step should now be either a materially stronger
  distillation target or a fair external comparator lane, not another small
  residual loss variant

And a forty-sixth Expected Attention-style comparator update:

> I then switched from method branches to the fastest fair extra comparator
> lane: an in-repo **Expected Attention-style** sparse selector on top of the
> same grouped-subspace transport + rank-4 residual checkpoint, still under the
> shared-chat / `enable_thinking=False` Qwen control. This is an approximation
> to the Expected Attention / KVPress family, not exact KVPress parity, because
> it reuses our in-repo `attention_expected` scoring path rather than the
> external library implementation. Held-out behavior was:
> - `gsm8k_5`: `attention_expected = 0.200000` at `729,580.825` average bytes
> - `gsm8k_5`: `attention_expected_shuffled = 0.200000` at `722,050.950`
>   average bytes
> - controlled `gsm8k_eval_10`: `attention_expected = 0.100000` at
>   `727,855.375` average bytes
> - controlled `gsm8k_eval_10`: `attention_expected_shuffled = 0.100000` at
>   `720,900.538` average bytes

Interpretation:

- the in-repo Expected Attention-style comparator is fair enough to keep in the
  paper as a query-aware sparse-control baseline
- but it does **not** separate from its matched shuffled null on either the
  cheap smoke or the controlled held-out slice
- that makes it a useful **negative-boundary comparator**, not a new positive
  method result
- the right next step is still to move the internal method lane toward a
  materially stronger teacher signal rather than another selector or local
  bridge-loss variant

And a forty-seventh CAB-style bridge-distillation update:

> I then tried the first bridge branch in this repo that changes the teacher
> signal in a materially more faithful way: `bridge_ridge_qk_cab_adapter`.
> This keeps the same learned query-conditioned residual adapter on top of the
> same grouped-subspace transport + rank-4 residual checkpoint and the same
> fair shared-chat / `enable_thinking=False` Qwen control, but replaces the
> old global/local residual losses with a prompt-local **causal attention
> behavior** target inspired by CAB. The checkpoint fit again matched the older
> bridge family (`K` cosine `0.864`, relative Frobenius error `0.476`; `V`
> cosine `0.381`, relative Frobenius error `0.915`). Held-out behavior was:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `681,668.400` average bytes

Interpretation:

- this is a more principled teacher signal than the earlier local affinity or
  global attention-KL variants, and it is modestly cheaper in bytes than the
  earlier learned-adapter family
- but it still does **not** stabilize beyond the cheapest fair smoke, so even
  prompt-local causal attention behavior is not enough in the current tiny
  bridge form
- that pushes the next serious lane one step further: if we keep the bridge
  framing, the teacher target probably has to move beyond local attention
  behavior alone toward richer affinity or prediction-level distillation, or we
  need a more expressive routed bridge on top of the same frozen transport

And a forty-eighth routed CAB-bank update:

> I then tried the next natural escalation of the same idea:
> `bridge_ridge_qk_cab_bank`. This keeps the same fair shared-chat /
> `enable_thinking=False` control and the same grouped-subspace transport +
> rank-4 residual checkpoint, but replaces the single learned
> query-conditioned residual bridge with a **QK-routed bank of query-conditioned
> bridge experts**. The routing uses the same QK template-bank machinery as the
> earlier residual-bank branches, while each expert is trained with the same
> prompt-local causal attention teacher used in the single-expert CAB branch.
> Calibration fit again matched the older bridge family (`K` cosine `0.864`,
> relative Frobenius error `0.476`; `V` cosine `0.381`, relative Frobenius
> error `0.915`). The first held-out fair smoke was:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes

Interpretation:

- more expressive bridge capacity alone is **not** enough here: the routed
  mixture exactly ties the single-expert CAB branch on both the cheap smoke and
  byte cost
- that makes the current bridge-bank lane look saturated in its present form
- if we keep the positive-method lane alive, the next move should probably be a
  richer **teacher target** or a stronger canonicalization step, not just more
  bridge experts

And a forty-ninth EM-KD-style interaction-distillation update:

> I then tried `bridge_ridge_qk_emkd_adapter`. This keeps the same grouped-
> subspace transport + rank-4 residual checkpoint, the same fair shared-chat /
> `enable_thinking=False` Qwen control, and the same learned
> query-conditioned residual bridge form as the earlier adapter family, but it
> replaces the local CAB-style teacher with a prompt-local **token-interaction
> distribution** target inspired by EM-KD / interaction distillation. The
> checkpoint fit again matched the recent bridge family (`K` cosine `0.864`,
> relative Frobenius error `0.476`; `V` cosine `0.381`, relative Frobenius
> error `0.915`). Held-out behavior was:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `681,668.400` average bytes

Interpretation:

- the richer prompt-local interaction target is more principled than plain
  latent regression, cheap affinity matching, or local CAB-style attention
  behavior alone
- but in the current tiny bridge form it still does **not** stabilize beyond
  the cheapest fair smoke
- this makes the current local bridge-distillation family look saturated:
  richer **local** teachers are no longer buying held-out gains by themselves
- the next serious positive-method move should probably be either a
  materially stronger teacher signal still closer to prediction space, or a
  stronger canonicalization / transport step before the bridge

And a fiftieth rotational-canonicalized transport update:

> I then tried the first stronger geometry-side follow-up after the saturated
> local bridge-distillation family: `grouped_rotational_transport`. This keeps
> the same fair shared-chat / `enable_thinking=False` Qwen control and the
> same grouped soft-transport + rank-4 residual structure, but changes the
> grouped block fit itself. Each grouped source/target block is first
> covariance-normalized into its own canonical rotational gauge, then a shared
> orthogonal map is fit in that quotient space before the transport map is
> assembled. Calibration fit was actually *worse* than grouped-subspace
> transport (`K` cosine `0.827`, relative Frobenius error `0.530`; `V` cosine
> `0.285`, relative Frobenius error `0.958`). Held-out behavior was:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- this is the first recent “new method class” branch that does **not** fall
  straight to `0.0000` on the controlled `gsm8k_eval_10` slice
- but it only climbs to the same `0.1000` level as the controlled
  `target_alone` read, so it is still **not** a positive method result
- that means the geometry / canonicalization lane is still weakly alive, but
  the current covariance-normalized rotational fit is not yet creating a real
  gain over the controlled target baseline
- the strongest read now is that richer geometry may matter, but not in a way
  that simple local bridge losses were able to capture

And a fifty-first exact KVPress comparator update:

> I then closed the exact external Expected Attention comparator with the
> vendored KVPress pipeline itself instead of our in-repo approximation. I
> added a reusable harness, `scripts/run_kvpress_eval.py`, which loads the
> vendored `references/repos/kvpress` package, patches its cache API
> compatibility for the current `transformers` stack at runtime, and replays it
> on our JSONL generation slices under the same fair shared-chat /
> `enable_thinking=False` Qwen control. The exact held-out reads were:
> - `gsm8k_5`, no-press: `0.200000`
> - `gsm8k_5`, `ExpectedAttentionPress`: `0.200000`
> - controlled `gsm8k_eval_10`, no-press: `0.100000`
> - controlled `gsm8k_eval_10`, `ExpectedAttentionPress`: `0.100000`

Interpretation:

- the exact external KVPress / Expected Attention baseline reproduces the same
  **negative-boundary comparator** story we already saw in the in-repo
  approximation
- on this exact Qwen3-0.6B setup, Expected Attention survives the held-out
  slices but it does **not** improve over its own no-press floor
- so it should stay in the paper as an honest external null / boundary
  comparator, not as a live positive baseline and not as the next method lane

And a fifty-second fitted-gauge transport update:

> I then pushed the geometry lane one step further with
> `grouped_fitted_rotation_transport`. This keeps the same fair shared-chat /
> `enable_thinking=False` control and the same grouped soft-transport +
> rank-4 residual structure, but it replaces the generic covariance-normalized
> rotational quotient with a **calibration-fit per-group ZCA-whitened
> rectangular orthogonal gauge map** before assembling the grouped transport.
> Offline alignment improved modestly over `grouped_rotational_transport`
> (`K` cosine `0.853`, relative Frobenius error `0.493`; `V` cosine `0.355`,
> relative Frobenius error `0.923`). Held-out behavior, however, was unchanged:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- a calibration-fit grouped gauge fix is more principled and does improve
  offline fit a bit
- but it exactly ties the earlier `grouped_rotational_transport` read on both
  held-out slices
- that is another useful blocker result: stronger grouped canonicalization
  alone still does **not** clear the controlled target floor

And a fifty-third shared-basis transport update:

> I then tried the first explicit shared-basis / dictionary-style geometry
> follow-up: `grouped_shared_basis_transport`. This again keeps the same fair
> shared-chat / `enable_thinking=False` control and the same grouped
> soft-transport + rank-4 residual structure, but it replaces the grouped
> rotational gauge fit with a **shared low-rank cross-covariance basis** per
> grouped block. Each grouped block is ZCA-whitened, projected into a shared
> source/target coefficient basis from the block cross-covariance SVD, and the
> transport map is fit in that coefficient space. Offline fit again improved
> slightly over the rotational baseline (`K` cosine `0.854`, relative
> Frobenius error `0.491`; `V` cosine `0.354`, relative Frobenius error
> `0.926`). Held-out behavior was still unchanged:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- this is the first geometry branch explicitly shaped by the shared-basis /
  dictionary-learning references rather than only by a rotational gauge fit
- but in this simple coefficient-space form it still exactly ties the two
  earlier geometry probes on the held-out slices that matter
- so even stronger shared-basis canonicalization is **not** enough by itself in
  the current grouped transport family

And a fifty-fourth stronger-teacher bridge update:

> I then implemented `bridge_ridge_qk_readout_adapter` on top of the same fair
> shared-chat / `enable_thinking=False` Qwen control and the same
> `grouped_subspace_transport + rank-4 residual` base. This keeps the tiny
> query-conditioned residual bridge framing, but it changes the calibration
> teacher from local CAB / EM-KD-style structural losses to a prompt-local
> **attention readout** target, so the bridge is trained against a signal
> closer to layer-level prediction behavior. While wiring that branch I also
> fixed a real implementation bug in the `bridge_ridge_qk_*adapter` family:
> those modes had only been fitting the K-side residual, leaving the V-side
> query residual at zero. The new branch now fits both K and V query residuals.
> Offline fit on the 64-prompt calibration slice was:
> - `K` cosine `0.864`, relative Frobenius error `0.476`
> - `V` cosine `0.381`, relative Frobenius error `0.915`
> Held-out behavior under the same fair controlled regime was still negative:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.000000` at `681,668.400` average bytes

Interpretation:

- the stronger readout-style teacher is more principled than the earlier local
  bridge losses, and the K-side offline fit does improve
- but even after fixing the missing V-side bridge fit, this tiny
  query-conditioned bridge still does **not** survive the controlled
  `gsm8k_eval_10` slice
- that narrows the live lane further: the next bridge teacher likely has to be
  even closer to prediction space, such as approximate likelihood / logit
  matching, not another prompt-local structural loss

And a fifty-fifth stronger-teacher bridge update:

> I then implemented `bridge_ridge_qk_predkl_adapter`, the first explicit
> prediction-level bridge teacher in this repo. This keeps the same
> `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but it replaces the
> prompt-local structural teachers with a calibration-time **top-k next-token
> teacher**. Concretely, the bridge now sees aligned target query features plus
> a prompt-local approximate likelihood target built from the target model's
> top-k next-token log-probabilities and output-embedding rows. On the full
> 64-prompt calibration slice, offline fit was:
> - `K` cosine `0.869`, relative Frobenius error `0.469`
> - `V` cosine `0.393`, relative Frobenius error `0.908`
> The first fair held-out smoke was still a clean negative:
> - `gsm8k_5`: `0.000000` at `722,107.700` average bytes

Interpretation:

- the repo now has a real prediction-level / likelihood-style bridge-teacher
  branch rather than only local structural teachers
- but even that stronger teacher still died immediately on the first fair
  held-out slice
- so the stronger-teacher lane is still conceptually the right lane, but the
  current **tiny local residual bridge** looks saturated even under a more
  ambitious teacher target

And a fifty-sixth richer-bridge update:

> I then tried the smallest higher-capacity follow-up to that prediction-level
> bridge: `bridge_ridge_qk_predkl_bank`. This keeps the same
> `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but it replaces the
> single prediction-level residual bridge with a **QK-routed bank of
> query-conditioned bridge experts** trained under the same top-k next-token
> teacher. On the 16-prompt smoke calibration slice, offline fit matched the
> single prediction-level bridge branch:
> - `K` cosine `0.887`, relative Frobenius error `0.437`
> - `V` cosine `0.480`, relative Frobenius error `0.862`
> The first fair held-out smoke was still negative:
> - `gsm8k_5`: `0.000000` at `722,107.700` average bytes

Interpretation:

- adding a small routed bank on top of the prediction-level teacher did **not**
  revive the bridge lane
- that makes the current tiny modular bank family look close to saturated too
- the next live bridge step should likely be a more materially different
  modular interface, not another small residual-bank tweak

And a fifty-seventh paper-artifact update:

> I then stopped pushing another tiny bridge variant and hardened the reviewer-
> facing paper artifacts around the **current live bars**. I added
> `scripts/build_reviewer_artifacts.py`, which rebuilds a concise current
> bytes/accuracy frontier plus a paired-flip table directly from the live
> result JSONL/meta files without importing the heavy training stack. The new
> outputs are:
> - `paper/bytes_accuracy_frontier_20260420.json`
> - `paper/bytes_accuracy_table_20260420.md`
> - `paper/paired_flip_table_20260420.jsonl`
> - `paper/paired_flip_table_20260420.md`
>
> The current artifact read is:
> - exact `gsm8k_eval_70`: `fixed prior = 0.0857`, `grouped_subspace + rank-4 residual = 0.0571`, `bridge_ridge = 0.0429`, `C2C = 0.1286`
> - controlled `gsm8k_eval_10`: `target-alone = 0.1000`, `bridge_ridge = 0.1000`, `grouped_rotational_transport = 0.1000`, `grouped_fitted_rotation_transport = 0.1000`, `grouped_shared_basis_transport = 0.1000`, exact `KVPress no-press = 0.1000`, exact `ExpectedAttentionPress = 0.1000`
> - paired flips: `fixed_prior` still beats its shuffled null, `grouped_subspace_resid4` still loses to `fixed_prior`, `bridge_ridge` still does not recover that gap, and none of the controlled survivors beat the controlled target floor

And a fifty-eighth modular-interface update:

> I then implemented `bridge_ridge_qk_asym_adapter`, the first explicit
> **shared-plus-private modular bridge** in this repo. It keeps the same
> `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but it replaces the
> fully separate K-side and V-side query adapters with one shared
> query-conditioned bottleneck plus private K and V residual heads. This is the
> closest branch here to an AsymLoRA-style interface rather than another
> monolithic tiny residual.
>
> On the 64-prompt calibration slice, offline fit was:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- the first materially different shared-plus-private modular interface is
  **weakly alive**
- but it still only ties the controlled `target_alone` floor rather than
  beating it
- so the modular-interface lane remains plausible, but this first AsymLoRA-
  style bridge is not yet a positive method result

And a fifty-ninth stacked-interface update:

> I then stacked the strongest output-side teacher we currently have on top of
> that same shared-plus-private interface: `bridge_ridge_qk_asym_predkl_adapter`.
> This keeps the same one-shared-plus-two-private low-rank bridge structure as
> `bridge_ridge_qk_asym_adapter`, but adds the same calibration-time top-k
> next-token teacher used by `bridge_ridge_qk_predkl_adapter`.
>
> On the same 64-prompt calibration slice, offline fit was unchanged:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were also
> unchanged:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- stacking the current prediction-level teacher on top of the first
  shared-plus-private dense interface did **not** improve the method
- the modular-interface lane is still weakly alive, but the first dense stack
  of “better interface + better teacher” still only ties the controlled floor
- that pushes the next live pivots toward either a **more materially different
  interface module** or a **shared sparse dictionary / SAE bridge**, not
  another dense tiny residual stack

Interpretation:

- the repo now has a machine-readable frontier and paired-flip layer that is
  aligned to the live methods rather than stale early transport branches
- that makes the current paper story easier to defend: the main live bar is
  still `fixed prior`, the best internal mechanism branch is still
  `grouped_subspace + rank-4 residual`, and the whole recent controlled family
  is best read as blocker evidence rather than as a hidden positive method

And a sixtieth sparse-interface update:

> I then implemented `bridge_ridge_qk_sae_adapter`, the first explicit
> **shared sparse codebook bridge** in this repo. It keeps the same
> `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but replaces the dense
> shared bottleneck with a small top-k sparse latent code that is decoded
> separately for K and V. This is the cleanest SAE-style shared interface we
> can test without rewriting the rest of the transport stack.
>
> On the same 64-prompt calibration slice, offline fit was:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the first held-out read was:
> - `gsm8k_5`: `0.000000` at `686,026.600` average bytes

Interpretation:

- the first shared sparse / SAE-style bridge is a **clean negative** on the
  first fair held-out smoke
- that means the simple sparse-code interface did not rescue the dense bridge
  family’s failure mode
- the sparse-interface lane is still worth citing and possibly revisiting in a
  richer form, but this first lightweight SAE-style bridge is not the positive
  method we need
- comparator priority also shifted again: after exact KVPress, the next
  highest-value external control day is now **KVzip** first and **Quest**
  second, with **KVComm** dropping below them in immediate value

And a sixty-first generated-interface update:

> I then implemented `bridge_ridge_qk_generated_adapter`, the first explicit
> **generated / instance-specific bridge** in this repo. It keeps the same
> `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but it replaces the
> fixed bridge residual with a continuous query-conditioned mixture over a
> shared bank of low-rank bridge atoms, in the MoRA / SHINE direction rather
> than the fixed residual or routed-bank direction.
>
> On the same 64-prompt calibration slice, offline fit was unchanged:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> The first held-out fair smoke was:
> - `gsm8k_5`: `0.000000` at `686,026.600` average bytes

Interpretation:

- the first generated / instance-specific bridge is also a **clean negative**
  on the first fair held-out smoke
- that means the current bridge family is not rescued simply by moving from a
  fixed adapter to a continuous generated mixture over low-rank atoms
- the strongest remaining live pivots are now even narrower:
  - a more materially different **module replacement** in the Attention Editing
    direction, or
  - a **dynamic output-alignment teacher** beyond static next-token KL

And a sixty-second dynamic-teacher interface update:

> I then implemented `bridge_ridge_qk_asym_dynmap_adapter`, the first branch
> here that keeps the stronger **shared-plus-private modular interface** from
> `bridge_ridge_qk_asym_adapter` but replaces the static top-k next-token KL in
> `bridge_ridge_qk_asym_predkl_adapter` with a **context-reweighted dynamic
> teacher** over the same top-k target rows.
>
> On the same 64-prompt calibration slice, offline fit was still:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- adding a **dynamic output-alignment teacher** on top of the first
  shared-plus-private interface does keep the branch weakly alive
- but it still only ties the controlled `target_alone` floor instead of
  improving on the plain asym interface or the asym-plus-static-predKL variant
- so the current evidence says the dynamic-teacher lane is conceptually better
  motivated, but the present tiny bridge family is still too weak to convert
  that into a positive result

And a sixty-third projector-interface update:

> I then implemented `bridge_ridge_qk_asym_projector`, the first explicit
> **post-transport projector** in the current shared-plus-private bridge
> family. It keeps the paired K/V modular interface from
> `bridge_ridge_qk_asym_adapter`, but it upgrades the base bridge into a full
> query-conditioned projector before the low-rank shared/private refinement.
>
> On the same 64-prompt calibration slice, offline fit again stayed at:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> The first fair held-out smoke was:
> - `gsm8k_5`: `0.000000` at `686,026.600` average bytes

Interpretation:

- the first shared-plus-private projector interface is a **clean negative**
  on the first held-out smoke
- simply turning the bridge base into a small query-conditioned projector is
  not enough to rescue the current transport family
- that strengthens the case that the next live branch has to be a **more
  explicit attention/module replacement** or a richer dynamic output-alignment
  mechanism, not another small projector or residual variation
