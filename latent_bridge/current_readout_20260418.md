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

And a sixty-fourth attention-interface update:

> I then implemented `bridge_ridge_qk_xattn_adapter`, the first explicit
> **attention-side transfer module** in the current bridge family. It keeps the
> same `grouped_subspace_transport + rank-4 residual` base and the same fair
> shared-chat / `enable_thinking=False` Qwen control, but it replaces the
> small residual-style shared bridge with a tiny query-conditioned
> cross-attention module over live K/V-side transport signals
> (`x`, `aux_input`, `paired_input`, `paired_aux_input`).
>
> On the same 64-prompt calibration slice, offline fit again stayed at:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- the first explicit attention-side transfer module is **weakly alive**
- but it still only ties the controlled `target_alone` floor rather than
  beating it
- that means the current modular-interface lane is still alive only in the
  weakest possible sense: new interface classes can survive the controlled
  slice, but they are still not producing a positive method result
- the next serious method pivots should now be even more concrete:
  - an actual **module replacement** in the Attention Editing / LLM Modules
    direction, or
  - a richer **dynamic output-alignment teacher** with explicit contextual
    remapping, not another small reweighting of the same top-k rows

And a sixty-fifth stacked-xattn update:

> I then stacked the strongest current contextual output-side teacher directly
> on top of that same explicit xattn interface as
> `bridge_ridge_qk_xattn_dynmap_adapter`.
>
> This keeps the same grouped-subspace transport + rank-4 residual base and
> the same tiny query-conditioned cross-attention bridge over the live K/V-side
> transport signals, but it adds the same context-reweighted top-k teacher used
> by `bridge_ridge_qk_asym_dynmap_adapter`.
>
> On the same 64-prompt calibration slice, offline fit again stayed at:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- stacking the dynamic contextual teacher on top of the weakly-alive xattn
  interface does **not** improve it
- the xattn branch and the xattn-plus-dynmap branch now sit at exactly the
  same weak `0.2000 / 0.1000` pattern as the other surviving modular
  interfaces
- that means the current local-teacher lane is saturated even when attached to
  the current explicit attention-side bridge
- the next serious positive-method shot should now be either:
  - a fuller **module replacement** in the Attention Editing / LLM Modules
    direction, or
  - a richer dynamic remapping teacher with explicit token alignment /
    interaction structure, not another small contextual reweighting of the
    same top-k rows

And a sixty-sixth module-replacement update:

> I then implemented `bridge_ridge_qk_module_adapter`, a fuller
> **attention-side transfer module** on top of the same
> `grouped_subspace_transport + rank-4 residual` base.
>
> Instead of a tiny residual bridge or a single small xattn block, this branch
> adds learned bridge slots, a query-conditioned cross-attention over the live
> K/V-side transport signals plus those slots, and a nonlinear readout trained
> with calibration-time top-k prediction distillation.
>
> On the same 64-prompt calibration slice, offline fit again stayed at:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- even the first fuller slotted module-replacement style bridge only ties the
  same weak controlled floor as the other surviving modular interfaces
- that strengthens the case that **local interface elaboration is saturated**
  in the current transport family
- the next serious positive-method shot should be more literal:
  - a fuller **Attention Editing / LLM Modules** style module replacement, or
  - a richer **dynamic token/output remapping** teacher with explicit
    contextual alignment rather than another local top-k reweighting

And a sixty-seventh direct-module-replacement update:

> I then implemented `bridge_ridge_qk_module_replace`, which keeps the same
> slotted attention-side module shape as `bridge_ridge_qk_module_adapter` but
> trains that module to predict the full corrected K/V directly rather than
> only a residual on top of the fixed bridge.
>
> On the same 64-prompt calibration slice, offline fit again stayed at:
> - `K` cosine `0.870`, relative Frobenius error `0.468`
> - `V` cosine `0.397`, relative Frobenius error `0.907`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- even a more literal direct-output module replacement still only ties the
  same weak controlled floor as the other surviving modular interfaces
- that strengthens the current read that **local module elaboration is
  saturated**, not just residual correction capacity
- the next serious positive-method shot likely needs:
  - an even more literal **Attention Editing / LLM Modules** style
    replacement that changes the interface more globally, or
  - a richer **dynamic token/output remapping** teacher with explicit
    contextual alignment rather than another local top-k target

And a first layer-localization artifact update:

> I added a reviewer-facing layer localization artifact from the live
> `selector_trace` telemetry on the controlled `gsm8k_eval_10` slice:
> `paper/layer_localization_20260420.{jsonl,md}`.
>
> Across the current weakly-alive modular family
> (`shared_plus_private_asym_adapter`, `shared_plus_private_dynmap_adapter`,
> `xattn_adapter`, `xattn_dynmap_adapter`, `module_adapter`,
> `module_replace`, `tokenbasis_replace`), the same top target-layer pattern
> repeats:
> - `L27 <- S23`
> - `L5 <- S4`
> - `L23 <- S20`
> - `L22 <- S19`
> - `L8 <- S7`

Interpretation:

- the surviving modular variants are not changing the runtime layer-selection
  story in any meaningful way under the fair control
- that is new blocker evidence that the current saturation point sits earlier
  than the exact local bridge parameterization
- this makes upstream pivots like **token/span remapping** or a more global
  **module replacement** even more plausible than further local bridge edits

And a sixty-eighth token-basis interface update:

> I then implemented `bridge_ridge_qk_tokenbasis_replace`, which keeps the
> same slotted attention-side module shape as `bridge_ridge_qk_module_replace`
> but constrains its direct K/V outputs to a basis distilled from target
> next-token output rows rather than a free dense output map.
>
> On the same 64-prompt calibration slice, offline fit remained:
> - `K` cosine `0.869`, relative Frobenius error `0.469`
> - `V` cosine `0.393`, relative Frobenius error `0.908`
>
> Under the matched-bytes fair controlled regime, the held-out reads were:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- even a target-native basis constraint on the direct-output module still only
  ties the same weak controlled floor as the other surviving modular branches
- that is stronger evidence that the current saturation point is **upstream of
  local bridge parameterization**, not just caused by free dense outputs
- the next serious positive-method shots should now prioritize:
  - explicit **token/span remapping** or vocabulary-side alignment before the
    local bridge, or
  - a more global **Attention Editing / LLM Modules** style replacement that
    changes the interface beyond the current local module family

And a sixty-ninth upstream token-remapping update:

> I then implemented `bridge_ridge_qk_spanalign_module_replace`, which keeps
> the same slotted attention-side module shape as
> `bridge_ridge_qk_module_replace` but changes the calibration data itself:
> source and target samples are no longer paired by truncated absolute token
> position. Instead, both formatted prompts are mapped back onto the raw user
> prompt span and aligned by a monotone character-span overlap pass before the
> direct-output module is fit.
>
> On the same 64-prompt calibration slice, that upstream remapping changed the
> sample geometry materially:
> - aligned calibration pairs dropped to `2702`
> - `K` cosine improved to `0.937`, relative Frobenius error `0.334`
> - `V` cosine improved to `0.632`, relative Frobenius error `0.763`
>
> Under the matched-bytes fair controlled regime, the held-out reads were
> still:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- simple raw-prompt token/span remapping improves **offline fit** sharply, so
  the old same-position pairing really was a live upstream problem
- but this first remapping pass still does **not** move held-out behavior
  above the same weak controlled floor
- that means the bottleneck is upstream of the local bridge family, but not
  solved by plain monotone raw-span overlap alone
- the next serious remapping branch likely needs:
  - **contextual** token/span alignment rather than raw-span overlap alone, or
  - a more global **Attention Editing / LLM Modules** style replacement on top
    of the better-aligned interface

And a seventieth contextual-remapping update:

> I then implemented `bridge_ridge_qk_ctxalign_module_replace`, which keeps
> the same direct-output slotted attention-side module as
> `bridge_ridge_qk_module_replace` but upgrades the upstream pairing again:
> each source token is now matched to a small **context-weighted mixture of
> target tokens** instead of a single hard target position.
>
> On the same 64-prompt calibration slice, that preserved the stronger
> upstream fit geometry:
> - contextual calibration samples: `2702`
> - mean target tokens per source sample: `2.96`
> - `K` cosine `0.937`, relative Frobenius error `0.334`
>
> The fair smoke read was a clean negative:
> - `gsm8k_5`: `0.000000`
> - average bytes: `686,026.600`

Interpretation:

- upstream remapping is still the right place to push, because the fit signal
  remains materially stronger there than in the old hard-pairing setup
- but a simple local **soft mixture over nearby target tokens** is still not
  enough to create a held-out gain
- so the next live remapping branch should be more explicitly
  alignment-aware:
  - dynamic token/span alignment or span-level likelihood matching,
  - multi-view remapping losses,
  - or a more global module replacement after the remapped interface

And a seventy-first dynamic-alignment update:

> I then implemented `bridge_ridge_qk_dynalign_module_replace`, which keeps
> the same direct-output slotted attention-side module as
> `bridge_ridge_qk_module_replace` but upgrades the upstream remapping again:
> candidate target tokens are now scored by both **local span/context
> agreement** and **next-token output overlap** before forming the
> source-to-target token mixture.
>
> On the same 64-prompt calibration slice:
> - dynamic calibration samples: `2702`
> - mean target tokens per source sample: `3.00`
> - `K` cosine `0.937`, relative Frobenius error `0.334`
> - `V` cosine `0.633`, relative Frobenius error `0.763`
>
> Under the fair held-out reads:
> - `gsm8k_5`: `0.400000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- this is the first upstream remapping branch that clearly beats the recent
  smoke floor, so **output-aware dynamic alignment is a live lane**
- but it still only ties the controlled `target-alone` floor on the harder
  10-example slice
- so the next step is not another local bridge tweak:
  - strengthen the dynamic alignment teacher itself,
  - add span-level / likelihood-style alignment losses,
  - and keep the same fair controlled readout to test whether the smoke lift
    becomes real held-out signal

And a seventy-second dynamic-program alignment diagnostic:

> I then implemented `bridge_ridge_qk_dpalign_module_replace`, which keeps
> the same direct-output slotted attention-side module as
> `bridge_ridge_qk_module_replace` but swaps the local source-to-target
> mixtures for a **global monotone dynamic-program alignment** scored by the
> same local span/context agreement plus next-token output overlap.
>
> On a smaller 16-prompt diagnostic calibration slice:
> - dynamic-program pairs: `660`
> - mean pairs per prompt: `41.25`
> - `K` cosine `0.948`, relative Frobenius error `0.304`
> - `V` cosine `0.699`, relative Frobenius error `0.697`
>
> The first fair smoke read was a clean negative:
> - `gsm8k_5`: `0.000000` at `686,026.600` average bytes

Interpretation:

- making the remapping **global and monotone** does improve the offline fit
  geometry again
- but the first diagnostic version still dies on the first held-out smoke
- so the local scorer was not the only issue:
  - a stronger **teacher/loss** is still the next live lane,
  - not merely a better alignment solver over the same local scores

And a seventy-third DWA-KD-style teacher diagnostic:

> I then implemented `bridge_ridge_qk_dynalign_dwakd_module_replace`, which
> keeps the same `dynalign` source-to-target token mixtures but strengthens the
> teacher side during module fitting:
> - calibration samples are confidence-weighted from both alignment
>   concentration and prediction entropy,
> - and the module uses both the plain prediction KL and the dynamic
>   context-shaped teacher term already used in the dynmap bridge family.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic calibration samples: `660`
> - mean target tokens per source sample: `3.00`
> - DWA-KD-style weight range: `0.588` to `1.400`
> - `K` cosine `0.948`, relative Frobenius error `0.304`
> - `V` cosine `0.699`, relative Frobenius error `0.697`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.400000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- strengthening the **teacher** does preserve the live dynalign smoke signal
- but this first weighted-teacher version still only ties the controlled floor
- so the next live step should strengthen the supervision itself further:
  - span-level or likelihood-style targets,
  - token/span interaction supervision,
  - or a tokenizer-agnostic shared byte/span interface

And a seventy-fourth dynalign interaction-teacher diagnostic:

> I then implemented `bridge_ridge_qk_dynalign_interact_module_replace`,
> which keeps the same `dynalign` source-to-target token mixtures and the
> same direct-output slotted module shape as
> `bridge_ridge_qk_module_replace`, but adds prompt-local interaction
> distillation during module fitting.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic calibration samples: `678`
> - mean target tokens per source sample: `3.00`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- adding prompt-local interaction supervision on top of `dynalign` does
  preserve a nonzero smoke signal, but it is weaker than both plain
  `dynalign` and `dynalign_dwakd` on the same diagnostic setup
- the controlled slice is still unchanged at the same `0.1000` floor
- so a richer **local interaction** loss is still not enough to rescue the
  live remapping lane by itself:
  - the next step should change the supervision target more globally,
  - toward span-level / likelihood-style targets,
  - or a tokenizer-agnostic byte / shared interface

And a seventy-fifth byte-span shared-interface diagnostic:

> I then implemented `bridge_ridge_qk_bytespan_module_replace`, which keeps
> the same direct-output slotted module as `bridge_ridge_qk_module_replace`,
> but changes the upstream calibration pairing to a tokenizer-agnostic
> UTF-8 byte-span interface. For each source token, the fitter chooses the
> dominant overlapping target token by raw byte mass before fitting the same
> module-replacement bridge.
>
> On a 16-prompt diagnostic calibration slice:
> - byte-span pairs: `678`
> - mean pairs per prompt: `42.38`
> - prompts changed versus char-span `spanalign`: `0`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- the implementation is now useful as a byte/shared-interface control, but on
  the current GSM-style calibration set it did **not** change any token-pair
  lists relative to char-span alignment
- this means the existing telemetry is not actually stressing tokenizer
  boundary differences; byte-level ideas need byte-stress or tokenizer-mismatch
  calibration data before we can treat them as a live method lane
- the next useful tokenizer-side work should be:
  - add a byte/tokenizer-stress calibration slice with units, symbols,
    Unicode, and adversarial BPE splits,
  - compare spanalign versus bytespan pair changes directly,
  - and only then stack byte/shared-interface supervision with the stronger
    `dynalign` / `dynalign_dwakd` teacher rather than testing more same-data
    byte variants

I added the first audit harness as `scripts/analyze_byte_alignment.py` and ran
it on the default Qwen2.5 -> Qwen3 byte-stress prompts. It found `1 / 8`
changed prompts, which confirms the harness can expose tokenizer-boundary
cases even though the GSM calibration slice did not.

And a seventy-sixth matched dynalign context-only null:

> I then implemented `bridge_ridge_qk_dynalign_ctxonly_module_replace`, which
> keeps the same downstream direct-output slotted module and the same dynalign
> candidate window, but disables prediction-overlap scoring during dynamic
> source-to-target mixture construction.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic context-only samples: `678`
> - mean target tokens per source sample: `3.00`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- the matched null isolates the recent dynalign smoke: using the same candidate
  window without prediction-overlap scoring falls back to the byte/span floor
- the output-overlap term is therefore genuinely carrying the dynalign
  `gsm8k_5 = 0.4000` smoke signal
- the controlled slice still does not move above `0.1000`, so the live lane is
  not “span/context alignment plus module replacement”; it is specifically a
  stronger **output-aware alignment teacher** lane
- next useful work should strengthen that teacher with span-level likelihood,
  multi-view remapping, or target-side refinement supervision rather than
  spending more cycles on context-only or byte-only variants on the same GSM
  calibration distribution

And a seventy-seventh layer-localization knockout diagnostic:

> I added an evaluator-side layer knockout hook:
> `--drop-target-layers` replaces translated K/V with target K/V for selected
> target layers under fused runs, so it removes source communication for those
> layers without corrupting the target prompt cache.
>
> On the 64-prompt `bridge_ridge_qk_dynalign_module_replace` checkpoint and
> the `gsm8k_5` smoke slice:
> - no knockout baseline: `0.400000` at `686,026.600` average bytes
> - recurrent top-layer signature knockout `L27,L5,L23,L22,L8`: `0.200000`
>   at `563,521.850` average bytes
> - matched offset-layer knockout `L26,L4,L21,L20,L7`: `0.200000` at
>   `563,521.850` average bytes

Interpretation:

- removing five communicated target layers is enough to cut the dynalign smoke
  in half
- but the effect is not unique to the recurrent top layer-localization
  signature because the offset control drops by the same amount
- so the shared layer signature is useful telemetry, but not yet evidence for
  one uniquely causal layer circuit
- next layer work should use larger controlled slices or single-layer
  leave-one-out / add-one-back curves before claiming a layer-localized
  mechanism

And a seventy-eighth dynalign likelihood-teacher diagnostic:

> I then implemented `bridge_ridge_qk_dynalign_likelihood_module_replace`,
> which keeps the same dynalign source-to-target token mixtures and DWA-style
> confidence weights, but injects empirical target next-token likelihood mass
> into the aligned top-k prediction teacher before direct-output module
> fitting.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic remapping samples: `678`
> - mean target tokens per source sample: `3.00`
> - likelihood/DWA-style sample weights: min `0.706`, max `1.303`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- adding gold target-token likelihood mass does not rescue the controlled
  slice and loses the best `dynalign` / DWA `gsm8k_5 = 0.4000` smoke
- the useful output-aware signal is probably a soft remapping / overlap signal,
  not a naive observed-next-token boost
- next likelihood-style work should use approximate likelihood matching over
  aligned spans or tokenizer-aware remapping, not direct gold-token injection

And a seventy-ninth dynalign span-ALM teacher diagnostic:

> I then implemented `bridge_ridge_qk_dynalign_spanalm_module_replace`, which
> keeps the same dynalign source-to-target token mixtures and confidence
> weighting, but replaces hard observed-next-token boosting with a small
> span-window approximate-likelihood blend. For each aligned target position,
> it adds sparse mass to observed future span tokens in proportion to the
> target model's own probability on those tokens, decayed by span offset.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic remapping samples: `678`
> - mean target tokens per source sample: `3.00`
> - span-ALM/DWA-style sample weights: min `0.588`, max `1.470`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- softening the likelihood teacher from exact next-token mass to a span-window
  approximate-likelihood blend still does not rescue the controlled slice
- it also loses the best `dynalign` / DWA `gsm8k_5 = 0.4000` smoke, so the
  failure is not only that the previous likelihood teacher was too hard
- direct target-token likelihood mass, even when span-smoothed, is probably the
  wrong teacher for the current bridge interface
- the next serious positive-method branch should move away from direct
  likelihood mass and toward attention/interaction refinement, explicit
  query-conditioned routing, or a stronger module interface that changes how
  the target consumes the communicated cache

And an eightieth dynalign DWA-interaction stack diagnostic:

> I then implemented
> `bridge_ridge_qk_dynalign_dwainteract_module_replace`, which keeps the same
> dynalign source-to-target mixtures, DWA-style confidence weights, and
> dynamic prediction teacher as `dynalign_dwakd`, then adds the prompt-local
> interaction distillation term from `dynalign_interact`. This isolates whether
> the interaction teacher only failed because it was previously unweighted.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic remapping samples: `678`
> - mean target tokens per source sample: `3.00`
> - DWA-interaction sample weights: min `0.588`, max `1.400`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- confidence weighting and the dynamic prediction teacher do not rescue the
  prompt-local interaction term
- the stack regresses to the same `gsm8k_5 = 0.2000` smoke as interaction-only,
  likelihood, span-ALM, bytespan, and context-only variants
- the controlled slice again ties the `0.1000` target-alone floor
- this closes the current local teacher-stacking branch; the next positive
  attempt should change the correspondence/interface class, with priority on
  CTPD-style aligned-span preferences, query-conditioned route geometry, or a
  more global target-side module replacement rather than another local KL term

And an eighty-first aligned preference-distillation diagnostic:

> I then implemented
> `bridge_ridge_qk_dynalign_prefdist_module_replace`, which keeps the same
> dynalign source-to-target mixtures, DWA-style confidence weights, and dynamic
> prediction teacher, but replaces direct likelihood mass and prompt-local
> interaction KL with a pairwise preference objective over the teacher's
> aligned target output rows. This tests whether preserving the teacher's
> relative output ranking is less brittle across tokenizers than exact
> next-token or local-interaction supervision.
>
> On a 16-prompt diagnostic calibration slice:
> - dynamic remapping samples: `678`
> - mean target tokens per source sample: `3.00`
> - preference-distillation sample weights: min `0.588`, max `1.400`
> - `K` cosine `0.948`, relative Frobenius error `0.305`
> - `V` cosine `0.697`, relative Frobenius error `0.700`
>
> Held-out diagnostic reads:
> - `gsm8k_5`: `0.400000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes

Interpretation:

- pairwise aligned-output preferences preserve the best dynalign / DWA smoke
  behavior, unlike likelihood, span-ALM, interaction, or DWA-interaction
  teachers
- the controlled slice still ties the `0.1000` target-alone floor, so this is
  not a positive-method result yet
- the best current read is that output-ranking supervision is directionally
  less destructive than exact token likelihood or local interaction KL, but it
  does not solve the generalization bottleneck on its own
- next useful work should stack this preference objective with a materially
  different interface or geometry step: query-conditioned route atoms,
  global target-side module replacement, or tokenizer-agnostic byte/span
  interfaces only after stress data shows real tokenizer divergence

And a competitor/reference integration update:

> I then consolidated the 301 reference sweep and the KVPress sanity checks.
> The strongest new direction is not a larger residual bridge; it is an
> explicit communication interface: head-wise route atoms, query-pool latent
> slots, gated target-native modulation, or a tokenizer-independent byte/readout
> probe. The local KVPress wrapper and upstream Needle smoke both ran, but
> ExpectedAttention tied no-press on GSM5, controlled GSM10, and a one-row
> Needle smoke, while being slower on GSM.

Interpretation:

- `readout_adapter` is now an explicit negative boundary: GSM5 smoke `0.2000`,
  controlled GSM10 `0.0000`, so prompt-local readout supervision does not
  generalize.
- KVPress ExpectedAttention is runnable as an external same-model compression
  comparator, but it is not a stronger local bar on the tiny sanity checks.
- The next implementable method branch should be `headwise_route_atom` or
  `query_pool_transport`, with route entropy, dead-atom count, per-layer/head
  collision counts, paired flips, and byte/token-family metrics logged before
  any larger run.
- Do not spend more cycles on direct likelihood, span-ALM, or local-interaction
  variants without changing the interface; those are now saturated blockers.

And an attention-stratified selector diagnostic:

> I added `attention_stratified` as a runtime position-selection metric. It
> keeps the same attention score source, but selects through four prompt-region
> bins instead of pure global top-k. The trace now records full selected
> positions for small runs plus prefix/mid/suffix coverage fractions, so later
> collapse telemetry is not limited to the first 16 selected positions.
>
> On the same `dynalign_prefdist` checkpoint:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes
> - controlled paired delta versus target-alone: `+0.000000`
> - corrected selector coverage on controlled GSM10: prefix `0.3612`, suffix
>   `0.3292`, full trace fraction `1.0000`

Interpretation:

- broader prompt-region coverage is mechanically achieved, but it loses the
  `dynalign_prefdist` GSM5 smoke (`0.4000 -> 0.2000`)
- controlled GSM10 still ties the target-alone floor
- this rules out naive coverage balancing as the missing selector fix
- the next interface ablation should use target-query-conditioned query-pool
  slots or head-wise route atoms, because simple top-k reshaping does not add
  semantic selectivity

And a deterministic query-pool transport diagnostic:

> I added `query_pool_transport` as a runtime position-selection metric. It
> keeps the cache shape fixed, partitions the prefix into attention-scored
> bins, pools translated K/V inside each bin, and writes one pooled
> representative slot per bin. This gives us query-pool-style telemetry without
> changing `PrefixState`, prefix length, or generation masks.
>
> On the same `dynalign_prefdist` checkpoint:
> - `gsm8k_5`: `0.200000` at `686,026.600` average bytes
> - controlled `gsm8k_eval_10`: `0.100000` at `681,668.400` average bytes
> - controlled paired delta versus target-alone: `+0.000000`
> - telemetry now includes pool slot count, pooled-bin entropy, pooled-bin
>   top weight, bin span, and full representative positions for small runs

Interpretation:

- fixed pooled representative slots are safer to test than true cache
  compression, but they are not enough for a positive method
- the result matches attention-stratified coverage balancing: interpretable
  route coverage improves, but exact answer performance still ties the target
  floor and loses the best GSM5 smoke
- the next real query-pool attempt should be learned and target-query
  conditioned inside transport/fusion, or replaced by head-wise route atoms or
  byte/tokenizer-independent probes
- the new quantization reference sweep suggests diagonal scaling, orthogonal
  preconditioning, asymmetric K/V allocation, product-quantized route atoms,
  and low-rank sketch bridges as the highest-value additive ablations

And a synthetic query-pool benchmark:

> I added `scripts/run_toy_query_pool.py`, which generates latent states,
> source K/V slots, and target queries, then compares matched-budget `topk`
> routing against a learned `query_pool` readout under aligned, rotated,
> outlier-scaled, and slot-permuted stress settings.
>
> At budget `4`, task accuracy was:
> - aligned: `topk 0.2396`, `query_pool 0.3125`
> - rotated: `topk 0.2760`, `query_pool 0.3125`
> - outlier: `topk 0.2448`, `query_pool 0.2917`
> - slot-permuted: `topk 0.2604`, `query_pool 0.3594`

Interpretation:

- the toy gives a real reason to try learned query-pool slots or head-wise
  route atoms next, because the learned pool is more task-useful than direct
  top-k under controlled stress
- the query-pool branch usually has worse reconstruction MSE, so the real
  method must log reconstruction/consistency, dead-slot rate, route entropy,
  collision rate, and paired flips; accuracy alone could reward shortcuts
- this is still synthetic evidence only; the paper needs a paired controlled
  Qwen improvement before calling query-pool a positive method

And a synthetic route-atom follow-up plus real evaluator hook:

> I extended `scripts/run_toy_query_pool.py` with a matched-budget
> `route_atom` readout and atom-level telemetry. The route atoms use
> query-conditioned multiplicative routing, then log atom entropy, atom
> collision, dead-atom rate, and atom top-margin alongside the existing route
> entropy/collision metrics. I also added a real-model evaluator surface:
> `--runtime-head-selection-metric headwise_route_atom` and
> `--runtime-head-gate-metric headwise_route_atom`. This treats target heads
> as route atoms by ranking heads that are sharp, distinct from the layer mean,
> and differently oriented over prefix positions, without changing
> `PrefixState`, cache shape, or generation masks.
>
> New toy artifact:
> - `results/query_pool_toy_20260421/query_pool_route_atom_vs_topk.json`
> - `results/query_pool_toy_20260421/query_pool_route_atom_vs_topk.md`
>
> Task accuracy at budget `4`:
> - aligned: `topk 0.2396`, `query_pool 0.3385`, `route_atom 0.2969`
> - rotated: `topk 0.2760`, `query_pool 0.3750`, `route_atom 0.3698`
> - outlier: `topk 0.2448`, `query_pool 0.2552`, `route_atom 0.2448`
> - slot-permuted: `topk 0.2604`, `query_pool 0.2969`, `route_atom 0.2917`

Interpretation:

- route atoms are not the new headline method by themselves; learned
  query-pool remains the stronger toy task readout
- route atoms are still useful because they expose the exact failure mode we
  need for paper-quality analysis: atom entropy, dead atoms, route collision,
  and orientation diversity
- the real evaluator hook is intentionally thin and safe; the next controlled
  run should test whether headwise route atoms improve bytes/accuracy or only
  improve interpretability
- if the real run is negative, the constructive branch is a hybrid: query-pool
  task readout plus route-atom/collapse telemetry, not a pure atom router

And the first real `headwise_route_atom` smoke:

> I ran the new evaluator-side metric on the same `dynalign_prefdist`
> checkpoint and GSM5 setup, adding `--runtime-head-selection-ratio 0.5` and
> `--runtime-head-selection-metric headwise_route_atom` to the prior
> attention-selected sparse `K-only` protocol.
>
> Result:
> - `gsm8k_5`: `0.200000` at `346,263.4` average bytes
> - prior dense-head `dynalign_prefdist` GSM5 reference: `0.400000` at
>   `686,026.6` average bytes
> - metadata now aggregates `route_atom_keep_fraction`,
>   `route_atom_score_entropy`, `route_atom_score_gap`,
>   `route_atom_sharpness_mean`, `route_atom_js_divergence_mean`, and
>   `route_atom_orientation_span`

Interpretation:

- `headwise_route_atom` is a useful compression/control branch because it
  roughly halves bytes, but it is not a positive-method result yet
- the accuracy drop says head diversity alone is insufficient; useful heads
  are being removed, or position selection and head routing are not aligned
- the next real test should stack route-atom tracing with the stronger learned
  query-pool interface, or sweep route-atom ratios before spending on larger
  GSM/SVAMP runs

And a route-atom ratio sweep:

> I added runtime head-routing pass-throughs to `latent_bridge/control_suite.py`
> so `EvalSpec` can reproduce `runtime_head_selection_ratio`,
> `runtime_head_selection_metric`, `runtime_head_gate_metric`, and
> `runtime_head_gate_strength`. I then swept the new `headwise_route_atom`
> metric on GSM5 at `25%`, `50%`, and `75%` head budgets.
>
> Results:
> - dense-head reference: `0.400000` at `686,026.6` average bytes
> - ratio `0.25`: `0.400000` at `176,381.8` average bytes
> - ratio `0.50`: `0.200000` at `346,263.4` average bytes
> - ratio `0.75`: `0.200000` at `516,145.0` average bytes
> - controlled GSM10, ratio `0.25`: `0.100000` at `175,249.2` average bytes
> - controlled GSM10 paired delta versus dense-head `dynalign_prefdist`:
>   `+0.000000`, method-only `0`, baseline-only `0`
> - ratio-sweep artifact:
>   `results/headwise_route_atom_20260421/route_atom_ratio_sweep.md`

Interpretation:

- the best route-atom setting is non-monotonic; keeping only the sharpest 25%
  heads preserves the smoke result, while reintroducing more heads hurts it
- this is the first route-atom result that is worth scaling: it preserves the
  `dynalign_prefdist` GSM5 smoke while reducing bytes by roughly `3.9x`
- on controlled GSM10 it ties the dense-head branch exactly, so this is a
  bytes-frontier/control result, not an accuracy-positive method result
- the likely mechanism is interference from less selective heads, not a simple
  lack of capacity
- next blocker test: either pair this 25% route-atom pruning with a stronger
  learned query-conditioned interface, or scale it to GSM30 only as a
  compression-frontier control

And a quantization-inspired toy preconditioning diagnostic:

> I added `preconditioned_query_pool` to the toy benchmark. It applies a
> learned diagonal preconditioner before query-pool routing and logs condition
> proxy, cosine drift, norm ratio, and absolute scale ratio.
>
> At budget `4`, task accuracy was:
> - aligned: `query_pool 0.3177`, `preconditioned_query_pool 0.3958`
> - rotated: `query_pool 0.2969`, `preconditioned_query_pool 0.2656`
> - outlier: `query_pool 0.3594`, `preconditioned_query_pool 0.2240`
> - slot-permuted: `query_pool 0.3281`, `preconditioned_query_pool 0.2760`

Interpretation:

- diagonal preconditioning helps the easy aligned setting but makes the hard
  stress cases worse
- this supports adding quantization-inspired controls to the paper, but not
  making diagonal preconditioning the next headline branch
- the next quantization-style attempt should be constrained/orthogonal
  preconditioning or asymmetric K/V budgeting, not an unconstrained diagonal
  scale that collapses route entropy

And a constrained preconditioning toy follow-up:

> I added `constrained_preconditioned_query_pool`, a bounded diagonal
> preconditioner for the toy benchmark. It uses
> `scale = 1 + 0.25 * tanh(raw_scale)`, so the learned scale stays in
> `[0.75, 1.25]` and cannot become the same free diagonal collapse mechanism
> as `preconditioned_query_pool`.
>
> New artifact:
> `results/query_pool_toy_20260421/query_pool_constrained_preconditioned_vs_topk.{json,md}`
>
> At budget `4`, task accuracy was:
> - aligned: `query_pool 0.3177`, free preconditioned `0.3958`,
>   constrained preconditioned `0.3594`
> - rotated: `query_pool 0.2969`, free preconditioned `0.2656`,
>   constrained preconditioned `0.3906`
> - outlier: `query_pool 0.3594`, free preconditioned `0.2240`,
>   constrained preconditioned `0.2812`
> - slot-permuted: `query_pool 0.3281`, free preconditioned `0.2760`,
>   constrained preconditioned `0.3177`

Interpretation:

- bounding the scale fixes one important failure mode: the constrained variant
  beats the free diagonal preconditioner on rotated, outlier, and slot-
  permuted stress settings, and it is the best rotated result in the toy suite
- the constrained variant still does not dominate learned query-pool or route
  atoms on all stresses, so it is a control/ablation, not a headline
- the paper should keep this as quantization-inspired evidence for bounded
  gauge/preconditioning, while real-model work should prefer constrained
  rotations or asymmetric K/V allocation over free diagonal scaling

And a GSM30 route-atom scale check:

> I scaled the best route-atom ratio (`0.25`) from GSM5/GSM10 to a paired
> GSM30 check with `target_alone` in the same run.
>
> Results:
> - `target_alone`: `0.066667`
> - `headwise_route_atom=0.25`: `0.033333` at `172,643.3` average bytes
> - paired delta versus target-alone: `-0.033333`
> - method-only wins: `0`
> - target-only wins: `1`
> - both correct: `1`
> - both wrong: `28`
> - route telemetry: keep fraction `0.2500`, score entropy `1.9455`, score
>   gap `0.0460`, sharpness `0.8588`, JS divergence `0.0954`, orientation
>   span `0.2442`

Interpretation:

- the GSM30 check does **not** support route atoms as a positive method by
  themselves
- the selector is still byte-efficient and interpretable, but it loses to the
  target-only controlled baseline on this larger slice
- route atoms should stay in the paper as a compression-frontier diagnostic
  and as collapse telemetry for later learned query-conditioned interfaces
- the next positive-method branch should not be "scale route atoms"; it should
  be "use route-atom telemetry and bounded preconditioning to debug a learned
  query-conditioned transport/interface"

And a new reference sweep:

> I added two literature memos:
> - `references/307_multimodal_resampler_interface_refs.md`
> - `references/307_diffusion_iterative_refinement_refs.md`

Interpretation:

- multimodal resamplers reinforce the fixed latent-bank / query-conditioned
  interface framing: separate selection from projection, log latent occupancy,
  and regularize against atom collapse
- diffusion and refinement papers reinforce a different point: do not spend
  extra compute uniformly; make refinement uncertainty- or confidence-
  conditioned and log step-wise drift/entropy/calibration
- these references are worth citing in the related-work and ablation-planning
  sections, but the current empirical results do not justify adding another
  main-method component until it beats the target-controlled slice

And an asymmetric K/V budget toy follow-up:

> I added `asymmetric_kv_budget` to the toy query-pool benchmark. It separates
> the K-like route budget from the V-like value budget, then logs route/value
> entropy, overlap, Jaccard, KL, cosine, gap, and gate statistics.
>
> New artifacts:
> - `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route1_value3.{json,md}`
> - `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route2_value2.{json,md}`
> - `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_vs_topk.{json,md}`
> - `results/query_pool_toy_20260421/asymmetric_kv_budget_summary.md`
>
> At matched total budget `4`, task accuracy was:
> - aligned: best prior control `0.3594`, asym K/V `1+3` `0.4844`,
>   asym K/V `2+2` `0.4167`
> - rotated: best prior control `0.3906`, asym K/V `1+3` `0.4219`,
>   asym K/V `2+2` `0.4688`
> - outlier: best prior control `0.3594`, asym K/V `1+3` `0.4219`,
>   asym K/V `2+2` `0.4167`
> - slot-permuted: best prior control `0.3542`, asym K/V `1+3` `0.4427`,
>   asym K/V `2+2` `0.3906`

Interpretation:

- this is the strongest new toy lead in the current cycle: both matched-budget
  asymmetric K/V splits beat the previous controls across all four stress cases
- the telemetry says the mechanism is real budget asymmetry rather than just a
  larger selector: route/value overlap is low, Jaccard is low, KL is high, and
  cosine is low
- this still is **not** a paper-positive real-model result; it is a high-value
  clue for the next real evaluator branch
- the next real-model ablation should separate K-route and V-value retention
  ratios, log separate K/V distortion or attention-fidelity metrics, and test
  equal-byte uniform/shuffled controls before stacking with route atoms

And a second reference expansion:

> I added:
> - `references/308_kv_compression_competitor_refs.md`
> - `references/308_symmetry_geometry_alignment_refs.md`
> - `references/308_quantization_asymmetry_refs.md`

Interpretation:

- `C2C` and `KVComm` remain the direct cross-model communication comparators
- KV-cache compression papers are mostly not apples-to-apples baselines, but
  they are essential controls for matched bytes, routing, retention, and
  latency
- symmetry/geometry work reinforces that any positive method needs to report
  gauge stability, orientation/span metrics, and interference/collapse
  telemetry
- quantization work now points to asymmetric K/V budgeting, bounded rotations
  or preconditioning, outlier protection, and variable-byte frontiers as the
  next concrete ablation family

## 2026-04-21 Execution Addendum

I implemented the asymmetric K/V clue in the real evaluator/control-suite
plumbing and added a second toy branch, `codebook_remap`, inspired by
tokenizer/vocab-bridge and quantization codebook ideas.

New code/results artifacts:

- `latent_bridge/evaluate.py`: `--kv-route-selection-ratio` and
  `--kv-value-selection-ratio`, with separate K/V position masks, exact
  K/V-aware byte accounting, and route/value telemetry in prediction sidecars.
- `latent_bridge/control_suite.py`: new default spec
  `fused_quant_asym_kv_attention_sparse_brief`.
- `scripts/run_toy_query_pool.py`: `codebook_remap` method with codebook
  entropy, collision, dead-code, reconstruction, support, and remap-overlap
  telemetry.
- `results/query_pool_toy_20260421/query_pool_codebook_remap_vs_topk.{json,md}`
- `results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json`

New references:

- `references/309_recent_llm_architecture_inspiration_refs.md`
- `references/309_competitor_benchmark_bootstrap.md`

Toy result summary from `query_pool_codebook_remap_vs_topk.md`:

| Scenario | Top-k | Best older control | Asym K/V 2+4 | Codebook remap |
|---|---:|---:|---:|---:|
| aligned | 0.2396 | 0.3958 | 0.4062 | 0.4948 |
| rotated | 0.2760 | 0.3906 | 0.3906 | 0.5781 |
| outlier | 0.2448 | 0.3594 | 0.4792 | 0.5312 |
| slot-permuted | 0.2604 | 0.3542 | 0.4010 | 0.4688 |

Interpretation:

- `codebook_remap` is now the strongest toy clue: it wins across all four
  stress cases while using budget `4`, not the larger K/V split budget `6`.
- The codebook result is interpretable rather than just higher accuracy:
  codebook support stays near `4`, dead-code rate is `0`, and remap overlap is
  near `1.0` while Jaccard stays near `0.25`, consistent with a compact
  learned interface rather than plain top-k reuse.
- The real evaluator now has the asymmetric K/V machinery needed to test the
  previous toy clue on actual model pairs without corrupting byte accounting.
- The C2C bootstrap resolved the exact published Qwen pair artifact
  (`nics-efc/C2C_Fuser`, subdir
  `qwen3_0.6b+qwen2.5_0.5b_Fuser`) but did not download or run the heavy
  benchmark in this pass.

Paper decision:

- Do **not** add another main-method component to the paper yet.
- Do add these as ablation families and interpretation hooks: asymmetric K/V
  budgets, codebook/token remapping, route/value overlap, codebook occupancy,
  remap stability, and matched-byte competitor controls.
- The next positive-method attempt should stack the smallest real-model
  asymmetric K/V branch with a learned codebook/token interface only after each
  clears target-alone and shuffled/uniform controls independently.

## 2026-04-21 Second Execution Addendum

New references/results:

- `references/310_recent_breakthrough_ablation_refs.md`
- `references/310_competitor_execution_plan.md`
- `results/query_pool_toy_20260421/query_pool_residual_codebook_remap_vs_topk.{json,md}`
- `results/asym_kv_qwen_20260421/qwen_gsm5_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat.jsonl`
- `results/asym_kv_qwen_20260421/qwen_gsm5_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat.jsonl`
- `results/asym_kv_qwen_20260421/qwen_gsm10_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat.jsonl`

Evaluator update:

- Added `--kv-route-selection-metric` and `--kv-value-selection-metric`.
- This fixes the first real-run blocker where route and value masks were nested
  because both used the same attention score. The route/value sidecar now logs
  the effective metric names, overlap, Jaccard, and score entropy.
- The standard control-suite asym K/V spec now uses route-by-attention and
  value-by-energy, so it exercises separable selection instead of merely
  changing ratios.

Real-model smoke results:

| Split | Selector | Target | RotAlign | Delta | Route/value overlap | Jaccard |
|---|---|---:|---:|---:|---:|---:|
| GSM5 | route attention, value energy | 0.20 | 0.40 | +0.20 | 0.674 | 0.206 |
| GSM5 | random route, random value | 0.20 | 0.20 | 0.00 | 0.837 | 0.270 |
| GSM10 | route attention, value energy | 0.10 | 0.10 | 0.00 | 0.666 | 0.202 |

Toy result update:

| Scenario | Asym K/V 2+4 | Codebook remap | Residual codebook remap |
|---|---:|---:|---:|
| aligned | 0.4062 | 0.4948 | 0.5417 |
| rotated | 0.3906 | 0.5781 | 0.6406 |
| outlier | 0.4792 | 0.5312 | 0.5677 |
| slot-permuted | 0.4010 | 0.4688 | 0.5417 |

Interpretation:

- Separable K/V selection is a live real-model lead, but not yet paper-safe:
  it beats target and random on GSM5 but saturates to target-alone on GSM10.
- The sidecar confirms the implementation is doing the intended thing: route
  and value masks are no longer identical/nested, and value-energy scores are
  higher-entropy than route-attention scores.
- Residual codebook remap is now the strongest toy signal. It improves over
  single codebook remap in all four stress cases and logs residual gate plus
  query/slot residual energy ratios, which makes the mechanism diagnosable.
- The next high-leverage stack is not another adapter: it is
  `gauge-aware alignment + separable K/V routing + residual codebook/token
  bridge`, with random/shuffled/uniform controls at each step.

Paper decision:

- Add the separable K/V selector and residual-codebook toy as ablations and
  method candidates, not as settled claims.
- Keep the main paper positive-method bar unchanged: a claim requires GSM30/70
  replication and matched controls against target-alone, random/shuffled
  selectors, C2C where runnable, and same-model compression controls.
- The immediate blocker is scale stability: the GSM5 method-only win on Wendi's
  chicken-feed example does not persist on the first GSM10 slice, so the next
  debugging pass should stratify method-only vs unchanged failures by prompt
  length, tokenizer overlap, selector entropy, and route/value overlap.
