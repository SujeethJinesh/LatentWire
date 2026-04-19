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
