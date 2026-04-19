## Blocker Map

This is the current best decomposition of why cross-model latent communication
is not yet a stable positive result in this repo.

### Blocker 1: Head-space symmetry / permutation mismatch

Observed symptom:

- fixed head priors are pair-conditioned and unstable across seeds
- cross-pair transfer is asymmetric
- grouped CCA changes behavior by task, which suggests basis sensitivity

Interpretation:

- head identity is not canonical across models
- a useful head in one model may be represented in a permuted or mixed basis in
  another

Current status:

- not solved
- simple permutation-aware rank matching also failed on the exact Qwen GSM70
  branch (`0.042857`), so plain head-order remapping is too weak on its own
- a direct gauge-aware Procrustes head-overlap score also failed on the same
  branch (`0.028571`), so cheap orthogonal-invariant head scoring is too weak
  on its own as well
- the compatibility-lifted `KVComm` replay also collapsed on the same Qwen
  pair (`0.000000` on GSM70), which is consistent with head identity and raw
  cache geometry becoming too brittle once KV-head count and per-head
  dimensionality stop matching directly

Next fix:

- permutation- or OT-aware head matching before sparse routing
- lightweight evaluator-level soft transport is also too weak on its own, so a
  stronger transport map likely has to move deeper into the translation path

### Blocker 2: Attention geometry is more important than raw KV similarity

Observed symptom:

- several routing heuristics improve one weak branch but only back to matched
  null levels
- expected-attention position priors were a null on grouped CCA
- head-level expected attention also only tied the old shuffled-null branch

Interpretation:

- preserving raw cache values is not enough
- the effective invariant is likely the induced QK / attention-logit geometry

Current status:

- not solved

Next fix:

- attention-fidelity or QK-geometry-preserving ranking / correction
- a stronger linear correction layer is still only a bounded repair on GSM70,
  so correction likely has to sit on top of a better transport map rather than
  replace it

### Blocker 3: Head importance is query- and task-conditional

Observed symptom:

- live sparse routing works only in narrow regimes
- grouped CCA helps SVAMP more than GSM
- fixed priors do not generalize broadly

Interpretation:

- there may be no single globally correct sparse route
- the useful head and position budgets depend on the reasoning regime

Current status:

- partially identified, not stabilized

Next fix:

- query-aware routing after the head basis is made more canonical
- task-aware comparisons rather than pooled scores

### Blocker 4: Keys and values are not equally useful transport channels

Observed symptom:

- `k_only` is consistently more promising than `v_only`
- dense full-KV transport has repeatedly failed or become confounded

Interpretation:

- keys carry retrieval geometry
- values are noisier to transplant directly and may be better left target-side

Current status:

- strong mechanism clue, not a final method

Next fix:

- keep `k_only` central
- test structured corrections on top of sparse key transport, not symmetric KV

### Blocker 5: Heterogeneous KV-head geometry itself is a transport barrier

Observed symptom:

- `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` mismatches both KV-head
  count (`2 -> 8`) and per-head dimensionality (`64 -> 128`)
- stock `KVComm` could not run on this pair without an explicit compatibility
  lift
- even with that lift, held-out GSM70 collapsed to `0.000000`

Interpretation:

- some baselines implicitly rely on matched or near-matched KV geometry
- once KV-head layout itself differs, training-free raw selective sharing may
  fail before higher-level routing logic even has a chance to help

Current status:

- newly identified and partially validated by the lifted `KVComm` replay

Next fix:

- canonicalize or transport KV geometry before selective routing
- keep baseline notes explicit about which methods natively support
  heterogeneous head geometry and which require a lift or learned adapter

### Blocker 6: Symmetric canonicalization alone is too weak

Observed symptom:

- target-only whitening collapsed calibration quality on the Qwen pair
- symmetric source+target whitening recovered to a bounded GSM70 result
  (`0.071429`) but still stayed below the old fixed-prior branch (`0.085714`)
  and below `C2C` (`0.128571`)

Interpretation:

- plain covariance canonicalization is not the missing ingredient
- the problem is richer than anisotropy or mean-shift mismatch alone

Current status:

- newly checked and not promising as a standalone method class

Next fix:

- keep canonicalization only as a component inside a stronger transport map
- prioritize learned or OT-style transport plus correction over whitening-only
  pivots

### Blocker 7: Better calibration fit can still hurt reasoning

Observed symptom:

- grouped soft transport plus a rank-64 residual improves calibration quality
  sharply on the Qwen pair (`K cos 0.925`, rel err `0.370`)
- the same checkpoint then collapses to `0.014286` on held-out GSM70

Interpretation:

- offline reconstruction quality is not a sufficient target
- the remaining gap is likely runtime-conditional: the model needs
  example-conditioned correction or fusion, not just a better static map

Current status:

- newly identified and strongly supported by the grouped-transport probe

Next fix:

- keep transport, but add a tiny example-conditioned correction / fusion layer
- stop investing in more static transport-map variants unless they also change
  runtime conditioning

### Blocker 8: Tiny diagonal correction alone is too weak

Observed symptom:

- a learned diagonal affine fuser trained from calibration pairs with target
  dropout reaches only `0.000000` on Qwen GSM70
- the same split still has the old fixed prior at `0.085714` and `C2C` at
  `0.128571`

Interpretation:

- some runtime conditioning may matter, but a tiny coordinatewise correction
  alone is not the missing ingredient
- the likely missing structure is **joint transport plus correction**, not a
  correction layer bolted onto a weak transport path

Current status:

- newly checked and strongly negative

Next fix:

- stop investing in diagonal correction-only probes on the main Qwen split
- move to stronger translator-side transport plus correction, ideally closer
  to OT / permutation-aware transport with a lightweight learned residual

### Blocker 9: Richer fusion still cannot rescue a weak transport map

Observed symptom:

- a stronger per-head ridge fuser over `[translated, target]` also reaches only
  `0.000000` on Qwen GSM70
- it matches the failure of the smaller diagonal learned-affine branch rather
  than improving it

Interpretation:

- the issue is not just that the learned correction was too small
- if the transported state is wrong enough, even a richer small fuser cannot
  recover the reasoning signal

Current status:

- newly checked and strongly negative

Next fix:

- stop spending cycles on fusion-only upgrades before the transport map changes
- move to transport-first branches: OT / permutation / gauge-aware transport,
  then correction on top only if transport itself becomes competitive

### Blocker 10: Simple hard permutation recovery is not enough

Observed symptom:

- a translator-side grouped permutation map improves over the worst transport
  collapses, but still reaches only `0.028571` on Qwen GSM70
- that stays below the old fixed prior `0.085714` and below `C2C` `0.128571`

Interpretation:

- some symmetry mismatch is clearly real, but a simple hard assignment between
  grouped head blocks is not the full fix
- the remaining gap likely needs richer OT/canonicalized transport, not just a
  one-shot permutation

Current status:

- newly checked and directionally informative, but still negative as a method

Next fix:

- move beyond simple hard assignment toward richer OT / Sinkhorn transport
  with a better cost or canonicalized subspace basis
- only revisit learned correction after that transport becomes competitive

### Blocker 11: Better transport cost helps, but only a little

Observed symptom:

- adding a grouped spectral-signature penalty to the transport cost raises
  Qwen GSM70 from grouped transport `0.014286` and grouped permutation
  `0.028571` up to `0.042857`
- that still stays below the old fixed prior `0.085714` and below `C2C`
  `0.128571`

Interpretation:

- transport cost quality matters; the transport lane is not fully dead
- but lightweight geometry-aware OT alone is still not enough to close the
  main gap

Current status:

- newly checked and directionally positive, but still bounded

## Blocker 12: Subspace-aware grouped transport does not move past the earlier geometry-aware branch

Observed symptom:

- replacing the spectral-signature penalty with a principal-subspace mismatch
  penalty leaves exact Qwen GSM70 unchanged at `0.042857`
- that ties grouped-signature transport exactly and still stays below the old
  fixed prior `0.085714` and below `C2C` `0.128571`

Interpretation:

- better geometry in the grouped-transport cost is not enough by itself
- the current grouped family is likely saturated on the main same-pair GSM
  setting
- the remaining plausible method class is richer transport or transport plus
  correction, not more small grouped-cost variants

Current status:

- newly checked and negative

Next fix:

- stop spending time on small grouped-cost tweaks
- move to stronger OT / canonicalized transport or explicitly narrow the paper
  to a blocker/mechanism contribution if the next transport branch still fails

## Blocker 13: Low-rank canonical subspace fitting still does not clear the main bar

Observed symptom:

- fitting each grouped block in a shared low-rank canonical basis reaches only
  `0.028571` on exact Qwen GSM70
- that is below the earlier grouped-signature transport `0.042857`, below the
  old fixed prior `0.085714`, and below `C2C` `0.128571`

Interpretation:

- denoising the block map into a shared low-rank basis is not enough by itself
- this pushes the blocker away from “we just need a canonical basis” and
  toward “we still need a richer transport objective or a stronger post-map
  correction”

Current status:

- newly checked and negative

Next fix:

- if we stay on the positive-method path, move to richer OT / attention-template
  transport or transport-plus-correction
- otherwise start tightening the paper around a blocker/mechanism contribution

## Blocker 14: Small residual correction helps only after the transport map gets somewhat reasonable

Observed symptom:

- adding a rank-4 residual on top of grouped-subspace transport lifts exact
  Qwen GSM70 from `0.042857` to `0.057143`
- but that is still below the old fixed prior `0.085714` and below `C2C`
  `0.128571`

Interpretation:

- correction is not useless; it just does not rescue a weak map by itself
- the first positive sign in the transport-plus-correction lane appears only
  once the transport map is already one of the better internal transport
  variants

Current status:

- newly checked and directionally positive, but still bounded

Next fix:

- if we keep pushing the positive-method path, transport-plus-correction is now
  the best remaining internal lane
- but it should be judged directly against fixed prior and `C2C`, not against
  weaker transport-only branches

## Blocker 15: Covariance-aware transport does not improve on the best subspace-plus-correction branch

Observed symptom:

- grouped covariance transport plus the same rank-4 residual drops exact Qwen
  GSM70 to `0.014286`
- that is far below grouped subspace transport plus rank-4 residual
  (`0.057143`), below the old fixed prior (`0.085714`), and below `C2C`
  (`0.128571`)

Interpretation:

- covariance shape alone is not the right transport geometry shortcut here
- the best remaining evidence inside the current family is still:
  decent transport map first, then tiny correction

Current status:

- newly checked and negative

Next fix:

- stop spending time on covariance-aware grouped transport in this regime
- if we keep pushing internally, use the grouped-subspace-plus-rank4 branch as
  the baseline for any richer OT or attention-template transport

## Blocker 16: Behavior-matched grouped attention templates still plateau on the main split

Observed symptom:

- grouped template transport plus the same rank-4 residual reaches only
  `0.042857` on exact Qwen GSM70, using a practical `64`-prompt calibration
  slice for the attention templates
- that ties the earlier transport-only plateau, stays below grouped subspace
  transport plus rank-4 residual (`0.057143`), below the old fixed prior
  (`0.085714`), and below `C2C` (`0.128571`)

Interpretation:

- matching grouped heads by calibration-time last-token attention behavior is
  more principled than a blind grouped cost, but it is still not enough to
  recover the held-out reasoning signal
- the missing ingredient is likely richer transport plus correction, not a
  lighter behavior-matching penalty inside the current grouped solver

Current status:

- newly checked and negative

Next fix:

- stop treating calibration-time attention-template matching as an obvious
  shortcut
- if we keep pushing the positive-method lane, move to richer OT or
  retrieval-template transport and judge it directly against grouped-subspace
  plus rank-4 residual, fixed prior, and `C2C`

## Blocker 17: Naively stacking the best grouped penalties makes the transport worse

Observed symptom:

- grouped template-subspace transport plus the same rank-4 residual drops exact
  Qwen GSM70 to `0.014286`
- that is below grouped template transport plus rank-4 residual (`0.042857`),
  below grouped subspace transport plus rank-4 residual (`0.057143`), below
  the old fixed prior (`0.085714`), and below `C2C` (`0.128571`)

Interpretation:

- the grouped template penalty and grouped subspace penalty are not simply
  additive inside the current transport solver
- the remaining blocker is not “we need one more grouped penalty term”; the
  method class itself likely has to change

Current status:

- newly checked and negative

Next fix:

- stop stacking light grouped penalties
- if the positive-method lane stays alive at all, move to richer OT or
  retrieval-template transport rather than another local combination inside the
  current grouped transport family

## Blocker 18: Escaping the grouped `2 x 2` bottleneck still does not rescue transport

Observed symptom:

- a new `broadcast_template_transport` branch fit a true rectangular `2 -> 8`
  head transport on Qwen2.5-0.5B -> Qwen3-0.6B using per-head calibration-time
  attention templates and the same rank-4 residual correction
- offline fit looked materially better than several earlier grouped probes
  (`K` cosine `0.868`, relative Frobenius error `0.470` on the `64`-prompt
  calibration slice)
- but exact Qwen GSM70 still collapsed to `0.000000`
- that is below grouped subspace transport + rank-4 residual (`0.057143`),
  below the old fixed prior (`0.085714`), and below `C2C` (`0.128571`)

Interpretation:

- the grouped family was not failing **only** because `gcd(2, 8) = 2`
  restricted it to a coarse `2 x 2` plan
- finer per-head transport without a richer cost or stronger correction is
  still too brittle
- better calibration fit is again not the same thing as better reasoning-time
  communication

Current status:

- newly checked and negative

Next fix:

- stop treating “more granular head transport” as sufficient by itself
- if the positive-method lane stays alive, move to richer OT or
  retrieval-/QK-template transport costs and judge them directly against fixed
  prior, grouped-subspace-plus-rank4, and `C2C`

## Blocker 19: Richer many-to-many OT in the same attention-template space still fails

Observed symptom:

- a new `broadcast_template_ot_transport` branch replaced the broadcast
  row-softmax plan with a rectangular Sinkhorn-style OT plan so each target
  head receives a normalized mixture over source heads and each source head
  carries balanced mass across the larger target head set
- offline fit improved again on the same `64`-prompt calibration slice
  (`K` cosine `0.883`, relative Frobenius error `0.447`)
- but exact Qwen GSM70 still collapsed to `0.000000`, exactly tying the
  simpler `broadcast_template_transport` branch

Interpretation:

- the current failure is not just “we need true many-to-many transport”
- it is also not enough to apply richer OT **inside the same current
  attention-template representation**
- if OT still lives as the final positive-method lane, it likely has to move
  to a richer template space such as retrieval-template or QK-fidelity

Current status:

- newly checked and negative

Next fix:

- stop spending time on attention-template OT variants in the current
  representation space
- if the positive-method lane gets one more serious try, move to retrieval-
  template or QK-fidelity OT and judge it directly against fixed prior,
  grouped-subspace-plus-rank4, and `C2C`

Next fix:

- keep pushing transport-first, but only with richer costs or canonicalization
- if the next transport improvement is still small, the paper should present
  this as evidence that *some* geometry-aware transport helps while still
  falling short of a publishable positive method claim

## Immediate Plan

### Today

1. bootstrap `C2C` on the exact Qwen pair
2. record a fair baseline replay path for our GSM split
3. stop investing in weak expected-attention variants that only tie nulls
4. use the lifted `KVComm` failure as further evidence that deeper transport,
   not evaluator-side routing alone, is the next method class
5. treat the learned-affine collapse as evidence that correction-only is not
   enough; the next branch must change the transport itself

### Next 1-2 days

1. implement permutation-matched head prior
2. run it on:
   - `gsm8k_eval_70.jsonl`
   - `gsm8k_100.jsonl`
   - `svamp_eval_70.jsonl`
3. compare against:
   - target-alone
   - text-to-text
   - zero-byte attenuation
   - random translated
   - grouped CCA fixed prior
   - grouped CCA shuffled-prior null
   - C2C

### What would count as real progress

- beat grouped-CCA shuffled null by a nontrivial margin
- stay above target-alone on at least one reasoning split
- survive one repeat seed or one second split
- remain competitive with or above the first published baseline
