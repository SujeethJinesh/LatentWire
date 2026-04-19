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
- The live query-aware sparse `k_only` branch also fails the larger-slice seed repeat:
  - `seed0 = 0.0200`
  - `seed1 = 0.0400`
  - `seed2 = 0.0200`
  on `gsm8k_100`, so the earlier single-seed live-sparsity signal also does not currently scale into a robust headline result.
- A direct retrieval-head-style heuristic also fails to rescue the method:
  - Qwen -> Qwen, `gsm8k_100`: `0.0300`
  - Qwen -> Qwen, `svamp_eval_70`: `0.071429`
  - Qwen -> DeepSeek, `gsm8k_eval_70`: `0.014286`
  so simple retrieval-peak scoring is another **negative boundary**, not yet a stable mechanistic solution.
- A direct attention-logit / confidence proxy also fails to become the new lead:
  - Qwen -> Qwen, `gsm8k_100`, fixed `attention_margin` prior: `0.0500`
  - matched shuffled null: `0.0400`
  - live `attention_margin` budget: `0.0400`
  - SVAMP stays at `0.071429`
  - DeepSeek stays at `0.014286`
  so simple top-1 vs top-2 attention gap scoring is a useful bounded ablation, but not a replacement for the older peak-based prior.
- The first structural subspace pivot is **split by task**:
  - Qwen -> Qwen, `gsm8k_100`, grouped-CCA + fixed prior: `0.0300`
  - matched shuffled null: `0.0600`
  - Qwen -> Qwen, `svamp_eval_70`, grouped-CCA + fixed prior: `0.171429`
  - matched shuffled null: `0.128571`
  so grouped CCA is not a broad rescue, but it is the first branch that pushes the old SVAMP boundary meaningfully upward.
- A grouped-CCA follow-up with query-blind position priors shows that this is
  **not** a position-prior effect:
  - GSM100 expected-attention prior: `0.0300`
  - GSM100 uniform prior: `0.0300`
  - SVAMP expected-attention prior: `0.171429`
  - SVAMP uniform prior: `0.171429`
  so the grouped-CCA split is more likely about checkpoint / head-subspace geometry than better fixed position routing.
- A simple permutation-aware shortcut also failed cleanly:
  - Qwen -> Qwen, `gsm8k_eval_70`, `attention_match`: `0.042857`
  - old fixed per-head prior on the same split: `0.085714`
  - `C2C` on the same split: `0.128571`
  so rank-sorting the fixed prior onto live attention-ranked heads is not enough
  to rescue the branch.
- A cheap attention-fidelity proxy also remains bounded:
  - Qwen -> Qwen, `gsm8k_eval_70`, `attention_fidelity`: `0.057143`
  - old fixed per-head prior on the same split: `0.085714`
  - `C2C` on the same split: `0.128571`
  so a simple QK-geometry proxy is better than `attention_match`, but still
  not enough to create a new best branch.
- A richer calibration-time template-transport branch also fails on the same
  path:
  - Qwen -> Qwen, `gsm8k_eval_70`, `attention_template_transport`: `0.042857`
  - old fixed per-head prior on the same split: `0.085714`
  - `C2C` on the same split: `0.128571`
  so richer fixed head templates still do not rescue the current selector
  family.
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
  - **whether that budget should be shrinkage-regularized, permutation-matched, or attention-logit-preserving**
  - **whether the useful structure is retrieval-head-specific, permutation-matched, or attention-logit-preserving**
  - **whether a lighter subspace / CCA-style match is enough before trying full OT**
  - **whether the useful geometry is task-conditioned, with different subspace structure on GSM-style vs SVAMP-style reasoning**
  - **whether the next useful signal lives at the head level rather than the position level**

## COLM workshop path

Goal: a careful, control-heavy short paper with one sharp claim.

Proposed claim:
- Cross-model latent transfer is not uniformly helpful.
- The only reliable gains so far come from **selective sparse key import**, not generic KV fusion.
- The newest same-pair gain may come from **calibrated head identity**, not just live query-aware sparsity.
- The current best workshop-safe statement is: calibrated sparse key head budgets help on GSM8K for a compatible same-family pair, but the effect weakens on SVAMP and across target families.
- The external-bar version of that statement is tighter:
  our internal branches expose real structure and failure boundaries, but they
  still trail published `C2C` replays on both GSM and SVAMP for the same Qwen
  pair.
- The fixed-prior story is now best used as a **mechanism clue**:
  there is structure in which heads matter, but the current fixed prior is not
  stable enough across seeds to present as the final method.
- The grouped-CCA + head-level expected-attention follow-up is also best used
  as a **bounded mechanism clue**:
  it rescues GSM from `0.0300` to `0.0600`, but only back to the old
  grouped-CCA shuffled-null level rather than creating a new clean win.
- The first external baseline now tightens the claim further:
  on the exact Qwen GSM70 split, published `C2C` is at `0.128571`, above our
  current best same-pair branch `0.085714`.
- That baseline lead persists on the larger held-out GSM slice too:
  on `gsm8k_100`, `C2C` is at `0.110000`, above our current best same-pair
  branch `0.070000`.
- The SVAMP bar is stronger still:
  on `svamp_eval_70`, `C2C` is at `0.442857`, above our best current SVAMP
  branch `0.171429` and above the older text-to-text reference `0.414286`.
- The permutation-aware follow-up tightens it again:
  simple head-rank matching drops back to `0.042857`, so the remaining blocker
  is likely richer than plain head-order mismatch.
- The cheap attention-fidelity follow-up keeps that same conclusion:
  it partly repairs the failed permutation shortcut, but it still does not beat
  the old fixed prior or `C2C`.
- The richer template-transport follow-up sharpens it further:
  even upgrading the fixed prior to per-head calibration templates does not
  lift the branch, so the current selector family likely is not where the main
  gain will come from.
- The direct gauge-aware follow-up also fails:
  `attention_procrustes` drops to `0.028571` on the same Qwen GSM70 split,
  below the old fixed prior `0.085714` and far below `C2C` `0.128571`, so
  cheap orthogonal-invariant head scoring is not the rescue path either.
- The first tiny-correction follow-up is still bounded:
  the new ridge-corrected checkpoint reaches `0.057143` on the same Qwen
  GSM70 split, which is better than the Procrustes branch but still below the
  old fixed prior `0.085714` and below `C2C` `0.128571`.
- So the paper should not frame linear cleanup alone as the answer; the next
  credible path is still better transport, likely OT / gauge-aware matching,
  with correction only as a secondary add-on.
- The first symmetric-canonicalization follow-up is also bounded rather than a
  rescue:
  symmetric source+target whitening reaches `0.071429` on the same Qwen GSM70
  split, which is better than several failed routing probes but still below
  the old fixed prior `0.085714` and below `C2C` `0.128571`.
- So canonicalization by itself is not the story either; if we keep it, it
  should be framed as a possible component inside a stronger transport map.
- The first stronger transport-map follow-up sharpens the story further:
  grouped soft transport plus a rank-64 residual improves calibration fit a
  lot, but still collapses to `0.014286` on the same Qwen GSM70 split.
- That means better offline transport quality is not enough either; the likely
  missing ingredient is example-conditioned correction or fusion, closer to the
  learned-fuser behavior of `C2C`.
- The second external baseline is currently more useful as a blocker signal
  than a competitive bar:
  stock `KVComm` is not directly runnable on the same heterogeneous Qwen pair
  because the pair mismatches both KV-head count and per-head dimensionality,
  and a compatibility-lifted replay still collapses to `0.000000` on GSM70.
- That makes the next honest story sharper:
  some training-free raw KV sharing methods appear to depend on matched or
  near-matched KV geometry, while our own failure mode already points toward
  the same deeper blocker.
- The first lightweight OT-style follow-up is also bounded:
  `attention_sinkhorn` reaches only `0.042857` on the same Qwen GSM70 split,
  below the old fixed prior, below ridge correction, and below `C2C`.
- So the paper should not frame evaluator-level soft transport as the answer
  either; the next credible move is a heavier transport map or the next
  external baseline, not more score-level transport tweaks.
- The live query-aware sparse story is also now best used as a **mechanism clue**:
  query-aware sparsity matters directionally, but the current implementation is
  not stable enough across seeds or held-out slices to headline the paper.
- The first tiny learned-correction follow-up is also clearly bounded:
  a diagonal learned-affine fuser trained from calibration pairs with target
  dropout collapses to `0.000000` on Qwen GSM70, below the old fixed prior
  `0.085714` and far below `C2C` `0.128571`.
- A richer per-head ridge fuser over `[translated, target]` also collapses to
  `0.000000` on the same split, so the failure is not specific to a diagonal
  parameterization.
- So the paper should not frame “just add a small learned correction layer” as
  the answer; if correction matters, it likely has to sit on top of a
  **stronger transport map** rather than replace it.
- A translator-side hard grouped permutation map is also only a bounded move:
  it reaches `0.028571` on Qwen GSM70, which is better than the worst transport
  collapses but still far below the old fixed prior `0.085714` and below
  `C2C` `0.128571`.
- So the paper should not frame simple head reassignment as the missing
  symmetry fix either; if the transport lane still lives, it likely needs
  richer OT/canonicalized transport rather than one-shot permutation recovery.
- The first richer geometry-aware transport cost is directionally better:
  grouped signature-aware transport reaches `0.042857` on Qwen GSM70, beating
  grouped transport `0.014286` and grouped permutation `0.028571`.
- A follow-up grouped subspace-aware transport branch stays at the same
  `0.042857`, so the current grouped geometry-aware transport family now looks
  saturated on the main same-pair GSM split.
- A low-rank grouped canonical-subspace transport branch then falls back to
  `0.028571`, so canonical-basis fitting alone is not the rescue either.
- A first transport-plus-correction branch finally gives a directional lift:
  grouped subspace transport plus a rank-4 residual reaches `0.057143` on
  Qwen GSM70.
- A covariance-aware version of that same branch then falls back to `0.014286`,
  so covariance geometry is not the next shortcut in the current family.
- But it is still below the old fixed prior `0.085714` and below `C2C`
  `0.128571`, so this is still a bounded mechanistic gain rather than a
  publishable headline result.
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
  - live query-aware sparse routing vs seed repeats
  - asymmetric prior transfer across target models
  - SVAMP as an explicit boundary case for the calibrated-prior branch
  - quantized vs no-quantized
  - static vs source-dependent fusion
  - selector-specific failure cases

## Immediate next experiments

1. Treat both the fixed-prior and current live-sparse branches as mechanism clues, not final methods.
2. Keep the DeepSeek pair as the main transfer stress test instead of widening to many models too early.
3. Implement the next method pivots suggested by the literature:
   - OT / permutation or gauge-aware head matching
   - attention-fidelity-preserving routing after both head-level
     expected-attention and simple permutation-matching proved bounded on GSM
   - use grouped CCA as a task-conditioned branch to test on more SVAMP-like slices
   - retrieval-head routing only after the head space is made more canonical
   - causal head scoring
   - move from correction-only probes to stronger transport-plus-correction branches
   - treat `C2C` as the first real external bar on the exact Qwen pair before
     spending more time on weaker internal heuristics
4. Preserve the negative controls and failure cases in the main paper, not just the appendix.
