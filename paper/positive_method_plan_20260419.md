# Positive-Method Plan (2026-04-19)

This note consolidates the current subagent consensus after the latest
transport-first failures.

## Current Bar

To stay on an ICLR-style positive-method path, the next serious branch should
clear most of these on the exact Qwen same-pair setting:

1. Beat the old fixed-prior branch on `gsm8k_eval_70` (`0.0857`).
2. Narrow or beat the `C2C` gap on `gsm8k_eval_70` (`0.1286`).
3. Stay above target-alone and the shuffled / zero / random controls.
4. Survive one second held-out slice or one second reasoning task.
5. Keep bytes competitive with the current sparse branches.

## Status After GPA Sparse Dictionary Hub Toy

The new held-out-family toy changes the positive-method read in one specific
way:

- a **GPA-initialized sparse shared dictionary** is now the best non-oracle
  method at the hardest low-shot point, `1` shot/class, reaching `0.1171`
  MSE versus `0.1825` for direct held-out-family few-shot fitting and
  `0.2355` for canonical-only GPA
- but the same shared-dictionary lane loses back to direct held-out-family
  fitting as soon as `2+` paired shots/class are available
- and the first verifier-gated repair step is still inactive, with
  accept/help/harm all at `0.0000` on that held-out-family toy

That means the next positive-method lane should treat canonicalization plus a
shared sparse dictionary as a **low-shot interface initializer / regularizer**,
not as a full replacement for family-specific fitting. The cleanest next
branch is:

1. GPA-initialized shared hub
2. sparse shared dictionary / crosscoder on top of that hub
3. tokenizer or byte-level interface control when token mismatch is actually
   present on the evaluated pair
4. route/repair added only after the shared basis is fixed and the repair rule
   shows nonzero accepted help

## Status After Real Tokenizer Pair Sweep

The real-tokenizer follow-up narrows the story further:

- on the exact Qwen2.5->Qwen3 pair used in the main same-pair runs, tokenizer
  mismatch is effectively absent on the GSM30 slice: shared decoded-token rate
  `1.0000`, boundary F1 `1.0000`, fragmentation delta `0.0000`
- on genuinely cross-family pairs, tokenizer mismatch appears immediately:
  Qwen->Mistral reaches shared decoded-token rate `0.8174` and boundary F1
  `0.9496`, while Qwen->Phi3 reaches shared decoded-token rate `0.7972` and
  boundary F1 `0.9347`

That means tokenizer fixes are still a real robustness and future
cross-family-generalization lane, but they are **not** the most likely rescue
for the current same-pair Qwen positive-method attempt. For the exact active
pair, the live blockers remain route quality, shared-basis transfer, and
repair/control design.

## Status After Gauge-Fix Quotient Bridge Toy

The new symmetry-focused toy sharpens the low-shot read again:

- once the head-matching score is made gauge-invariant, `quotient_match_after_fix`
  becomes the best non-oracle method at the true `1`-shot held-out-family point,
  reaching `0.0796` MSE versus `0.0985` for direct held-out-family few-shot
  fitting, `0.1665` for no-match gauge-fix, and `1.7099` for the global
  seen-family ridge control
- the quotient-aware matcher also recovers the true head correspondence exactly
  in that toy (`head_match_accuracy = 1.0000`)
- but, again, direct held-out-family fitting retakes the MSE lead by `2+`
  paired shots/class (`0.0410` vs `0.0693` at `2` shots; `0.0141` vs `0.0505`
  at `4` shots)

That means quotient-aware gauge fixing is now a stronger **low-shot
initializer** than canonicalization alone, but still not a full method. The
cleanest additive order is now:

1. gauge-fix / quotient-aware matching
2. GPA-style shared hub
3. sparse shared dictionary / crosscoder
4. tokenizer or byte-level interface control when mismatch is actually present
5. route/repair only after the shared basis is stable and repair shows nonzero
   accepted help

## Status After Quotient + GPA Sparse Dictionary Toy

The composed low-shot follow-up is the strongest shared-basis result so far:

- `quotient_gpa_sparse_dictionary` is now the best non-oracle method at both
  `1` and `2` shots/class on the held-out-family toy, reaching `0.0568` and
  `0.0576` MSE
- that beats direct held-out-family few-shot fitting at the same points
  (`0.1003`, `0.0638`), and also beats the isolated `quotient_match_after_fix`
  and `quotient_gpa_canonical` branches
- the method preserves exact head recovery in the same toy
  (`head_match_accuracy = 1.0000`)
- but the boundary is still real: by `4` shots/class, direct held-out-family
  fitting retakes the lead (`0.0179` vs `0.0555`)
- the repair gate is still inactive, with accept/help/harm all `0.0000`

This is the first clean sign that the symmetry fix and the shared sparse basis
are **actually additive**. That does not make the paper ready, but it does
change the method-discovery priority. The new order should be:

1. quotient-aware matching
2. GPA canonicalization
3. sparse shared dictionary
4. tokenizer or byte-level interface control when mismatch is present
5. route/repair only after the first four pieces remain stable under stress

So the next serious branch should no longer ask whether symmetry or sparse
bases help individually. It should ask whether the composed lane survives:

- tokenizer/interface mismatch
- route-pool feature ids
- matched competitor contracts
- and one real held-out benchmark slice

## Status After Strong Interface Stress

The strong interface-stress follow-up makes the next interface claim more
precise:

- under explicit tokenizer-like corruption, the byte/span-remap version of the
  composed lane becomes the best shared-basis variant at the true low-shot
  points, reaching `0.0566` MSE at `1` shot/class and `0.0570` at `2`
  shots/class
- that beats the same composed lane with the raw token-id interface
  (`0.0582`, `0.0579`) and stays very close to the oracle-interface ceiling
  (`0.0568`, `0.0576`)
- but it still does **not** overturn the broader boundary: by `4`
  shots/class, direct held-out-family fitting with the same remap becomes the
  best method again (`0.0238` vs shared-basis oracle-interface `0.0555`)

That means the interface result is now useful, but narrow:

1. byte/span or vocab control is a **robustness amplifier**
2. it can protect the low-shot shared-basis lane when interface mismatch is
   real
3. it is **not** the main explanation for the current same-pair Qwen gap
4. it should stay behind the main method story, which is still quotient-aware
   symmetry plus a shared sparse basis

So the next additive order is still:

1. quotient-aware matching
2. GPA canonicalization
3. sparse shared dictionary
4. byte/span or vocab interface control when mismatch is present
5. route/repair only after the first four pieces remain stable on a real
   held-out slice

## Status After Byte Sidecar Follow-Up

The new byte-sidecar toy changes the interface story again in a useful way:

- under the same strong tokenizer-like corruption used in the remap follow-up,
  the tokenizer-agnostic byte-sidecar branch becomes the best shared-basis
  method at both `1` and `2` shots/class
- `quotient_gpa_sparse_dictionary_byte_sidecar_remap` reaches `0.0392` and
  `0.0394` MSE, beating the remap-only branch (`0.0566`, `0.0570`) and even
  the oracle-interface latent-only branch (`0.0568`, `0.0576`)
- the boundary is still there: by `4` shots/class, direct held-out-family
  fitting plus remap remains better (`0.0238` vs best shared-basis `0.0390`)

This is the first interface-side result that looks materially additive rather
than cosmetic. The story is now:

1. quotient-aware matching fixes the low-shot symmetry issue
2. GPA plus a sparse shared dictionary gives a shared latent basis
3. a tokenizer-agnostic byte sidecar is the strongest current interface add-on
4. direct family-specific fitting still dominates once enough paired data is
   available

So the next method order is now:

1. quotient-aware matching
2. GPA canonicalization
3. sparse shared dictionary
4. byte sidecar or sequence-aligned interface loss
5. route/repair only after the first four pieces survive a real benchmark
   smoke

## Status After Sequence-Aligned Sidecar Follow-Up

The new sequence-aligned sidecar follow-up sharpens the interface read one step
further:

- under the same strong tokenizer-like corruption, the
  `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar` branch is now the
  best shared-basis method at both `1` and `2` shots/class
- with the default held-out-family toy it reaches `0.0360` MSE at `1`
  shot/class and `0.0362` at `2` shots/class, beating the plain byte-sidecar
  branch (`0.0384`, `0.0384`) and the remap-only shared-basis branch
  (`0.0584`, `0.0599`)
- the boundary is still real: by `4+` shots/class, direct held-out-family
  fitting plus byte/span remap remains much better (`0.0169` at `4` shots,
  `0.0035` at `8` shots)
- importantly, the alignment-aware sidecar stays the best **shared-basis**
  branch even after the direct few-shot fit retakes the overall lead

That changes the live interface story again:

1. quotient-aware matching fixes the low-shot symmetry issue
2. GPA plus a sparse shared dictionary gives the shared latent basis
3. a byte sidecar is already directionally strong
4. adding sequence-aligned interface features improves that sidecar further
5. but the whole lane is still toy-backed and low-shot-bounded, not benchmark-ready

So the current best additive order is now:

1. quotient-aware matching
2. GPA canonicalization
3. sparse shared dictionary
4. byte sidecar plus sequence-aligned interface features
5. route/repair only after the first four pieces survive the frozen GSM8K smoke

## Benchmark Contract Status

The benchmark track is now frozen conceptually and the first exact same-pair
smoke has been executed.

- Main Table A should be cross-model reasoning only: `gsm8k`,
  `gpqa_diamond`, `arc_challenge`, with `C2C` as the main external
  comparator.
- Main Table B should be cross-model long-context QA only: `hotpotqa`,
  `qasper`, `multifieldqa_en`, `2wikimqa`, `musique`, with `KVComm` as the
  main external comparator.
- Same-backbone `LatentMAS` rows belong in an appendix table, not the main
  cross-model comparison.

That means we should stop thinking about “the competitor table” as one mixed
artifact. The next execution step is a smoke-tested, frozen contract for the
relevant suite once the method itself is stronger.

The smallest next execution step is now fixed too:

- a `32`-example held-out GSM8K smoke on the exact Qwen sender/receiver pair
- rows: `target_alone`, `text_to_text`, `rotalign_kv`, `c2c_generate`
- stop immediately if `rotalign_kv` loses to `target_alone` or if
  `c2c_generate` fails to beat `target_alone` by at least two examples,
  because that indicates a prompt/scorer mismatch rather than useful benchmark
  evidence

## Status After Frozen GSM8K32 Smoke

The frozen same-pair smoke now gives a real benchmark boundary:

- exact pair: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- exact slice: first `32` rows of `data/gsm8k_eval_70.jsonl`
- matched non-thinking greedy decode, `max_new_tokens = 64`
- rows:
  - `target_alone = 0.0625`
  - `text_to_text = 0.0312`
  - `rotalign_kv = 0.0625`
  - `c2c_generate = 0.1250`
- the contract itself looks valid:
  - identical ordered example IDs across all rows
  - target rerun is byte-identical
  - offline rescoring matches the sidecar summary
  - `c2c_generate` beats target by exactly `2/32`, so the smoke clears its
    minimum external-bar gate
- but the current sparse KV row is still not promotable:
  - `rotalign_kv` only ties target
  - `rotalign_kv` fails numeric extraction coverage (`28/32` vs the required
    `>= 31/32`)

This means the benchmark track is no longer hypothetical. The exact smoke
contract works, and it says our current same-pair sparse transport row is
still below paper standard. The next benchmark-worthy method row should be the
current best toy-backed lane:

1. quotient-aware matching
2. GPA canonicalization
3. sparse shared dictionary
4. byte sidecar plus sequence-aligned interface features

Only after that lane beats `target_alone` on this frozen contract and passes
the extraction/coverage checks should we widen to larger slices or build any
paper tables.

## Status After Frozen GSM8K32 Checkpoint Sweep

The next immediate question was whether any existing real-model proxy already
captures the new toy story. The answer is: partly, but not enough.

On the exact same frozen 32-example contract:

- `dynalign_module_replace = 0.09375`
- `spanalign_module_replace = 0.09375`
- `bytespan_module_replace = 0.03125`
- `sae_adapter = 0.00000`
- `target_alone = 0.06250`
- `c2c_generate = 0.12500`

The read is sharp:

1. stronger **output-aware alignment** is the only existing real proxy that
   clearly beats target on this exact same-pair contract
2. plain byte-only alignment does not help on this pair
3. the first real SAE-style sparse codebook proxy is still a clean negative
4. even the best current proxy still trails `C2C`

That means the next real method row should not be “run the SAE checkpoint
again” or “push byte-only harder.” The next real row should be one of:

1. output-aware alignment + low-rank residual correction
2. output-aware alignment + adaptive canonicalization / gauge-invariant match
3. only after those, a more mature shared sparse/codebook interface

So the paper-critical order is now:

1. keep the low-shot toy story centered on quotient-aware symmetry + shared
   sparse basis
2. keep the real same-pair benchmark story centered on output-aware alignment
3. connect those two with a residual/canonicalization upgrade, not with a
   first-pass byte-only or SAE-only pivot

## Status After Expanded Dynalign Sweep

The narrowed follow-up over the strongest existing teacher-side variants closes
that branch enough to change the next action:

- `dynalign_module_replace = 0.09375`
- `tokenbasis_replace = 0.09375`
- `dynalign_dwakd_module_replace = 0.06250`
- `dynalign_prefdist_module_replace = 0.03125`
- `dynalign_spanalm_module_replace = 0.03125`

That means:

1. the current real same-pair ceiling for this family is still `0.09375`
2. token-grounded outputs are as good as the plain dynalign teacher on this
   contract, but not better
3. richer dynalign-teacher supervision does **not** improve the exact frozen
   contract and often regresses

So the next real benchmark branch should no longer be “try another dynalign
teacher.” It should be:

1. `dynalign_module_replace + low-rank residual correction`
2. or `tokenbasis_replace + low-rank residual correction`
3. in parallel on the toy side, `adaptive canonicalization +
   gauge-invariant matching`

If neither residual-correction branch beats `0.09375` on the frozen contract,
then the paper direction should treat the current same-pair alignment family as
saturated and move the positive-method burden back to the shared-basis /
low-shot lane.

## Status After Dynalign Rank16 Residual Benchmark

The first recalibrated residual run changes the real same-pair story again:

- on the exact frozen GSM8K32 contract, `dynalign_module_replace_residrank16`
  now reaches `0.1250`
- that is up from the old same-pair ceiling of `0.09375` for plain `dynalign`,
  `tokenbasis`, and the reused `rank8` residual baselines
- the row keeps full numeric extraction coverage (`32/32`) and exact example-ID
  parity
- on this slice it now matches the current `C2C` smoke accuracy rather than
  trailing it

This is the first real sign that explicit residual correction can matter on the
benchmark side, not just in toy telemetries. But it is still not enough to
move into paper-writing mode:

1. it is a single same-pair `32`-example smoke result
2. the matched `tokenbasis_replace + rank16` control has to agree
3. we still need at least one broader held-out slice before claiming the lane
   is stable

So the next real benchmark order is now:

1. finish the matched `tokenbasis_replace + rank16` control
2. if it agrees or improves, widen to the next held-out slice on the same
   contract family
3. if it fails or ties back to the old ceiling, try a fixed gauge-fix /
   canonicalization wrapper on top of the live `dynalign + rank16 residual`
   lane

That means the same-pair benchmark story is no longer “teacher family
saturated, move on.” It is now:

1. output-aware alignment alone saturates around `0.09375`
2. residual correction is the first real additive lift
3. gauge/adaptive canonicalization is the next wrapper if the matched control
   does not confirm the lift
4. the paper is still not ready until the lift survives a matched control and a
   broader slice

## Best Next Method Lane

Subagent consensus was tight: the only credible internal positive-method lane
left is **transport-first**, not another selector tweak or another small
correction layer in isolation.

Best-ranked next branches:

1. **Retrieval-template OT on `K-only`**
   - Match source and target heads by calibration-time retrieval behavior
     rather than grouped geometry alone.
   - Use a rectangular entropic OT plan for heterogeneous head counts.

2. **QK-fidelity / attention-template OT**
   - Replace raw grouped penalties with a transport cost that preserves
     attention geometry or retrieval templates more directly.

3. **Transport plus tiny correction**
   - Only after transport itself becomes directionally competitive.
   - Small residual correction stays plausible; correction-only does not.

## Most Useful External Baselines

1. **`C2C`**
   - already the live external bar
2. **`KVComm`**
   - good direct comparator, but currently blocker evidence more than a
     leaderboard baseline on this heterogeneous Qwen pair
3. **LatentMAS**
   - fastest adjacent latent-collaboration comparator if we want another
     runnable external point soon
4. **Latent Space Communication via K-V Cache Alignment**
   - scientifically strong, but higher engineering cost

## If The Next OT Branch Fails

The paper should narrow explicitly to a blocker/mechanism contribution:

- heterogeneous head-space mismatch is real
- transport quality matters more than selector cleverness
- transport-plus-correction can help locally
- but simple canonicalization, grouped penalties, and light behavior matching
  do not recover a competitive positive method

## Status After Broadcast-Template OT

The latest `broadcast_template_ot_transport + rank-4 residual` branch still
collapsed to `0.0000` on exact Qwen GSM70, exactly tying the simpler
`broadcast_template_transport` branch. That means:

- richer many-to-many OT **inside the current attention-template space** is not
  enough
- if we keep the positive-method lane alive, the next branch should move to a
  **different representation space**, not just a richer solver:
  - retrieval-template OT
  - QK-fidelity OT
  - or a residual-stream bridge if we pivot more aggressively

The follow-up `broadcast_peak_template_ot_transport + rank-4 residual` branch
lifted that score slightly to `0.0143`. That means:

- representation does matter somewhat
- but a simple peak-location proxy is still far too weak
- if we give the positive-method lane one more serious try, it should be a
  **richer retrieval-template or QK-fidelity transport**, not more tweaks to
  mean-attention templates

The follow-up `broadcast_retrieval_spectrum_ot_transport + rank-4 residual`
branch then moved into a richer per-head key-geometry space, using
retrieval-weighted key spectra instead of attention templates. Offline fit
improved materially (`K` cosine `0.931`). Under the fair matched sparse
`K-only` protocol, exact Qwen GSM70 recovered only to `0.0143`, tying the
peak-template OT branch and still using far more bytes than the live sparse
branches.

That means:

- “use a richer calibration-time key descriptor” is also not enough in this
  simple spectral form
- better offline geometry fit still does not predict held-out reasoning gains
- if we give the positive-method lane one last serious try, it should be a
  **QK-fidelity or retrieval-template transport in a genuinely different
  representation space**, not another attention- or key-descriptor OT tweak

The follow-up `broadcast_qk_template_ot_transport + rank-4 residual` branch
then replaced the retrieval-spectrum descriptor with a last-token QK/logit
template while keeping the same rectangular `2 -> 8` OT solver and the same
rank-4 residual. Under the fair matched sparse `K-only` protocol, exact Qwen
GSM70 again recovered only to `0.0143`, tying the retrieval-spectrum branch
exactly and using the same high byte budget.

That means:

- a shallow move into last-token QK/logit space is also not enough inside the
  current broadcast OT family
- the next serious try, if any, has to be a **genuinely richer
  query-conditioned QK-fidelity or retrieval-template transport**, not another
  static calibration-time descriptor swap

The follow-up `attention_qk_fidelity` per-head budget branch then stopped
changing the translator and changed only the live sparse budget rule on top of
the best current internal transport-plus-correction checkpoint,
`grouped_subspace_transport + rank-4 residual`. On exact Qwen GSM70 it
recovered to `0.0429` at `157,989.2` average bytes.

That means:

- a genuinely query-conditioned budget is a real branch, not a crash or a null
- but it is still below grouped-subspace-plus-rank4 (`0.0571`), fixed prior
  (`0.0857`), and `C2C` (`0.1286`)
- query-conditioning at evaluation time alone is still not enough

The next follow-up added runtime per-head soft gate overrides on top of that
same frozen grouped-subspace-plus-rank4 checkpoint. The first two held-out
smokes, `attention_qk_fidelity` gating and `attention_fidelity` gating, both
collapsed to `0.0000` on `gsm8k_eval_10`.

That means:

- the first gate-only query-conditioned rescue path looks weak
- if the positive-method lane gets one more serious try, it should stay
  **transport-first**, not shift to another gate-only or selector-only branch

The next follow-up stayed transport-first but moved back into the grouped
family with a prompt-indexed contrastive template bank:
`grouped_contrastive_template_transport + rank-4 residual`. Calibration fit
again looked strong (`K` cosine `0.932`, relative Frobenius error `0.351`),
but the first held-out `gsm8k_eval_10` smoke still collapsed to `0.0000`.

That means:

- grouped prompt-contrastive templates are still not enough
- “better grouped behavior matching at calibration time” is now looking
  saturated
- if the positive-method lane gets one more serious try, it should move to a
  **genuinely query-conditioned retrieval/QK transport**, not another grouped
  static-template variant

The next follow-up then tested that same idea one layer later in the stack,
without changing the translator: `attention_qk_template_transport` as a new
query-conditioned per-head sparse budget on top of the best current internal
checkpoint, `grouped_subspace_transport + rank-4 residual`. It builds fixed
calibration-time QK templates from `--runtime-head-prior-file`, then scores the
live heads by how well their current last-token QK distributions match those
templates. On the first matched sparse `K-only` smoke, `gsm8k_5`, it still
collapsed to `0.0000` at `142,353.225` average bytes.

That means:

- fixed QK templates inside the evaluator are not enough to rescue the live
  checkpoint
- query-conditioning keeps helping less when it is applied only at evaluation
  time than when it is baked into the transport hypothesis
- if we keep the positive-method lane alive, the next real shot should be
  **transport-first and query-conditioned in the transport itself**, not
  another evaluator-side budget or gate variant

The next follow-up tested exactly that inside the grouped transport family:
`grouped_qk_retrieval_transport + rank-4 residual`. It replaced the old grouped
attention template with a grouped last-token QK retrieval profile built from the
same `64`-prompt calibration slice. Offline fit was again respectable (`K`
cosine `0.881`, relative Frobenius error `0.452`), but the first matched sparse
`K-only` held-out smoke on `gsm8k_5` still scored `0.0000` at `630,701.475`
average bytes.

That means:

- a more retrieval-shaped grouped descriptor is still not enough when it is
  averaged statically over the calibration prompts
- better offline grouped transport fit still does not predict held-out reasoning
  utility
- the next transport-first shot has to be **genuinely query-conditioned at
  runtime**, not another grouped calibration-time summary

I also tested the lighter evaluator-side version of the same idea on the best
current internal checkpoint, `grouped_subspace_transport + rank-4 residual`:
`attention_qk_bank_transport`, which replaces the single averaged QK template
with a prompt-indexed calibration bank and lets each live example soft-select a
template family before budgeting the sparse positions. On the first matched
sparse `K-only` smoke, `gsm8k_5`, it still scored `0.0000` at `143,636.825`
average bytes.

That means:

- the weak point is probably not just that we averaged the calibration templates
- the evaluator-overlay lane now looks close to saturated
- if we keep the positive-method lane alive, query-conditioning probably has to
  change the transport or fusion path itself, not only the per-head sparse
  budget metric

## Status After Randomized Verifier Labels

The first target-model listwise verifier failed by selecting target-alone on
all `30/30` GSM30 examples. I then reran the same verifier after shuffling the
candidate labels while logging the label order, target label, selected source,
and raw verifier response for every example.

Result: accuracy improved from `0.0667` to `0.1000`, but the verifier chose
letter `A` on `29/30` examples while target-alone was label `A` only `7/30`
times. Target selection dropped to `6/30`, so the old failure was not pure
target preference; it was mostly option-position collapse.

That means:

- raw listwise verifier prompts are not a reliable selector yet
- label randomization is mandatory for any future verifier ablation
- the next selector path should use pairwise/pointwise candidate checks,
  confidence calibration, or candidate repair rather than asking for one
  multiple-choice letter
- the telemetry fields now make this interpretable enough to audit when we
  scale beyond GSM30

## Status After Confidence-Gated Compute Toy

I added a toy ablation for adaptive candidate-budget allocation, motivated by
the current gap between stochastic-route oracle quality and selector quality.
The toy simulates heavy-tailed candidate pools and calibrates thresholds on a
train split before evaluating on a held-out test split.

Result: `confidence_gated` reaches `0.6510` accuracy at average budget
`2.0312`, compared with `fixed_budget_2` at `0.6406` and
`random_budget_matched` at `0.6250`. The oracle gap also improves from
`0.3385` for `fixed_budget_2` to `0.3281` for `confidence_gated`.

That means:

- adaptive compute allocation is a plausible selector-side primitive
- the effect is modest but positive under matched-ish average budget
- the next real-model version should expand from one to three to five route
  candidates only when confidence or agreement telemetry says the example is
  hard
- paper-facing claims still require the same idea to move from toy data into
  GSM30/GSM70 stochastic route pools

## Status After GSM30 Confidence-Gated Route Expansion

I promoted the confidence-gated compute idea from toy data into the existing
GSM30 stochastic-route pool. The script calibrates low/high thresholds on the
first `15` examples, evaluates on the remaining `15`, and logs selected source,
seed budget, target confidence proxy, full-candidate oracle correctness, and
subgroup breakdowns.

Result: `confidence_gated_route_expansion` reaches `0.2000` on the eval half,
versus eval-half target-alone `0.0667` and random matched budget `0.1333`.
However, it ties both `fixed_route_budget_1` and `fixed_route_budget_3`, while
spending average seed budget `2.3333`.

That means:

- adaptive route expansion is useful enough to keep, because it beats target
  and random matched compute
- it is not yet a selector method, because fixed route budgets tie it with less
  tuning risk
- the next real selector should gate on richer telemetry than target-only
  format/numeric/completion proxies, especially route disagreement and
  process-verification signals
- these logs are now interpretable enough to decide whether failures come from
  under-spending, wrong candidate choice, or candidate pool ceiling

## Status After Pairwise Verifier Tournament

I also added a pairwise verifier tournament to remove the listwise option-A
failure mode. Candidate order is shuffled by seed, each match randomizes
left/right orientation, and every comparison logs sources, raw response, parsed
winner, fallback, win counts, and target-side rates.

Result: full GSM30 accuracy stays at `0.0667`, matching target-alone. The
tournament selects target only `0.2000` of the time and seeds `0.8000` of the
time, so it successfully breaks the target/default collapse, but it does not
identify correct seeds.

That means:

- pairwise aggregation alone is not sufficient
- the current target model can be made less position-biased, but not yet a good
  verifier for route candidates
- the next verifier should require process-level arithmetic checks, answer
  repair, or confidence-calibrated pointwise scoring before tournament
  aggregation

## Status After Calibrated Feature Selector

I added a transparent feature selector that calibrates candidate-feature weights
on the first half of GSM30 and evaluates on the second half. Features include
format score, numeric consistency, completion, answer agreement, target/seed
identity, and seed index. Every candidate score and feature vector is logged.

Result: the selector fits the calibration half to `0.2000` accuracy but scores
`0.0000` on the eval half, below eval-half target-alone `0.0667`. It selects
seeds on every eval example.

That means:

- simple candidate metadata is not enough to solve selection
- calibration on tiny route pools can overfit badly even when every feature is
  interpretable
- this strengthens the case for process-aware verification, repair, or
  externally grounded checking rather than another scalar metadata score

## Status After Process-Aware Repair Toy

I added a toy arithmetic-trace repair ablation for the exact failure mode seen
in GSM30: rerankers often choose near-miss candidates whose final answer is
wrong because an intermediate step is inconsistent. The toy gives the selector
a process verifier that detects the first inconsistent arithmetic step and
repairs downstream state.

Result: `rerank_only` accuracy is `0.5469`, `process_aware_repair` reaches
`1.0000`, and oracle is `1.0000`. Repair is applied on `0.5417` of examples,
with false repair rate `0.0885`; repair application rises monotonically with
inconsistency severity.

That means:

- repair is the strongest next hypothesis, not another listwise or pairwise
  chooser
- the real-model next step should ask for structured arithmetic/process traces
  from the selected route, localize the first inconsistent step, and repair the
  answer before final scoring
- the paper can frame this as moving from selection over finished answers to
  communication plus process-level correction

I then pushed that same hypothesis one step deeper into the live fusion path,
without changing the frozen translator checkpoint, by adding
`attention_qk_fidelity_tokenwise` as a runtime per-head, per-position gate
override on top of `grouped_subspace_transport + rank-4 residual`. This keeps
the same sparse `K-only` protocol, the same fixed head prior from
`.debug/head_prior_64.txt`, and the same `attention_prior` per-head position
budget, but it stops averaging the live query down to one score per head.
Instead it uses last-token QK agreement to modulate the fusion gate across
positions within each active head. On the first matched sparse `gsm8k_5`
smoke, it still scored `0.0000` at `146,756.475` average bytes.

That means:

- moving the query signal into the fusion path is directionally more sensible
  than another static template, but this first tokenwise gate-only version is
  still not enough
- the remaining mismatch now looks more like a transport-map or lightweight
  bridge problem than a budget-allocation problem
- if the positive-method lane stays alive, the next real shot should be a
  query-conditioned transport or tiny learned bridge, not another evaluator-side
  overlay on a frozen map

I then tried exactly that tiny bridge lane: a decoder-side low-rank correction
after quantize/dequantize on top of the same grouped-subspace transport and the
same rank-4 transport residual. This new `low_rank` quantization correction is
a reduced-rank linear repair in rotated target space (`rank=8`), meant to act
as a small bridge adapter rather than a full learned projector. On the `64`-
prompt calibration slice, the resulting checkpoint was the first adapter-style
branch in a while to show any nonzero held-out smoke signal: it reached
`0.2000` on the first matched sparse `gsm8k_5` check. But the larger matched
`gsm8k_eval_10` follow-up dropped back to `0.0000`, while staying roughly
twice as expensive in bytes as the older grouped-subspace-plus-rank4 branch.

That means:

- the tiny learned bridge lane is still more promising than another evaluator
  overlay, because it at least produced the first weak positive smoke
- but the current low-rank bridge is not yet stable enough to claim a real
  method improvement
- if we keep pushing the positive-method story, the next adapter-style step
  should add either live query-conditioning or a better interaction-level
  training target, not just a static low-rank correction

## Status After GSM30 Process Repair

I promoted process-aware repair from toy data into the existing GSM30
stochastic-route pool. The script first selects a route with
`target_on_strict_format`, then asks the target model to audit and repair the
selected reasoning while logging raw repair text, pre/post normalized answers,
changed-answer flags, help/harm, and candidate oracle availability.

Result: `process_repair_selected_route` reaches `0.2333` on full GSM30,
compared with pre-repair strict selector `0.1667` and target-alone `0.0667`.
It changes answers on `0.4333` of examples, helps on `0.0667`, and has
observed repair harm `0.0000`; candidate oracle remains `0.3000`.

That means:

- this is now the strongest real-model GSM30 method lane
- the method is still dev-slice evidence and must be tested on held-out
  GSM70/SVAMP before becoming a method claim
- repair quality, not selection alone, is now the best blocker to attack
- next variants should reduce partial repair outputs with stricter final-answer
  contracts, longer decode budget, and structured step extraction

## Status After Held-Out Process-Repair Bootstrap

I added a reproducible held-out bootstrap for the process-repair lane. It writes
the exact GSM70 and SVAMP70 stochastic-route commands for salts `0`, `1`, and
`2`, then writes the corresponding strict-selector process-repair commands.
The route-pool generation remains sequential by design, because each full
held-out salt is expected to take several minutes on MPS and the local shell is
already near its process limit.

At the bootstrap stage, this removed the main operational blocker before adding
a result: the paper-critical run became a frozen command sequence rather than
an ad hoc manual launch. Full runs can become held-out evidence; any `--limit`
subset generated by the same script remains dev-smoke only.

I also ran a two-example GSM70 `--limit 2` smoke through that bootstrap to
validate the path end-to-end. The repair stage reached `1.0000` on `n=2` after
changing one answer, with `repair_help=0.5000` and observed `repair_harm=0.0000`;
this is plumbing evidence only and should not be cited as a method result.

I then ran the full frozen held-out manifest. Raw stochastic route pools are
not stable enough to claim by themselves: GSM70 salts landed at `0.0857`,
`0.0286`, and `0.0571` against the manifest target baseline `0.0571`; SVAMP70
salts landed at `0.3000`, `0.3000`, and `0.2571` against the manifest target
baseline `0.3000`. The repair stage is the actual positive method candidate:
GSM70 `process_repair_selected_route` reached `0.2000` vs target `0.0571`,
with pre-repair `0.1286`, help `0.0714`, harm `0.0000`, and oracle `0.1571`.
SVAMP70 reached `0.5429` vs target `0.3000`, with pre-repair `0.3571`, help
`0.1857`, harm `0.0000`, and oracle `0.5286`.

That means:

- process repair is now a real held-out positive-method candidate, not just
  GSM30 dev evidence
- raw stochastic route generation remains unstable/neutral; repair is carrying
  the gain
- observed zero harm is encouraging but must be stress-tested with target
  self-repair, target-protection, and matched compute controls
- route-pool generation, repair, and byte/accounting outputs are now tied to one
  auditable command artifact
- next variants should add step-level verifier/test-before-repair,
  sensitivity-aware route budgeting, and explicit token/byte accounting

## 2026-04-21 Fair Repair-Control Harness

I added same-prompt repair-control arms to `process_repair_routes.py` and the
held-out bootstrap. A repair run can now emit:

- `selected_route_no_repair`: the selector output without target-side repair
- `target_self_repair`: the target candidate repaired with the same prompt and
  decode budget
- `process_repair_selected_route`: the selected route repaired with the same
  prompt and decode budget

This directly attacks the current claim risk: held-out process repair beats the
raw `C2C` row, but it adds target-side generation. The next fair comparison is
not only "repair vs target-alone"; it is "selected-route repair vs target
self-repair and selected-route no-repair under identical prompts, token budget,
temperature, and serialization."

I validated the control harness on the existing GSM70 `n=2` smoke route pool.
The smoke is intentionally not paper evidence: `target_self_repair` and
`process_repair_selected_route` both reached `1.0000` from a `0.5000`
pre-repair baseline, while `selected_route_no_repair` stayed at `0.5000`. That
is useful because it proves the control can falsify attribution: on tiny slices,
the gain may be target self-repair rather than cross-model communication.

Next full-result requirement:

Completed full-control result:

- GSM70: `selected_route_no_repair = 0.1286`,
  `target_self_repair = 0.1714`, and
  `process_repair_selected_route = 0.2000`
- SVAMP70: `selected_route_no_repair = 0.3571`,
  `target_self_repair = 0.5000`, and
  `process_repair_selected_route = 0.5429`
- both splits still show zero observed repair harm
- selected-route repair beats same-prompt target self-repair on both splits,
  but the attribution margins are modest: `+0.0286` on GSM70 and `+0.0429` on
  SVAMP70

This keeps the positive-method lane alive after the strongest immediate
fairness control. It also clarifies the next blocker: most of the repair gain is
target-side self-correction, while the cross-model route contributes a smaller
but consistent increment. The next real-model variants should therefore
increase the route-specific margin rather than merely making target repair
stronger.

## 2026-04-21 Toy Test-Before-Repair Ablation

I added a toy `test-before-repair` bridge that models the failure mode from the
real route pools: the highest-surface candidate can be internally consistent but
semantically drifted, so output-only repair has no local arithmetic error to
fix. The toy compares:

- `repair_only`: immediately spend the repair pass on the highest-surface route
- `test_before_repair`: run discriminative checks over the pool before spending
  repair budget
- `oracle`: label-leaking upper bound

Result on the generated `192`-example toy: `repair_only = 0.0312`,
`test_before_repair = 0.9531`, `oracle = 1.0000`; help vs repair-only is
`0.9219`, harm is `0.0000`. The cost caveat is explicit: test-before-repair
uses about `368.6` bytes of synthetic test evidence versus `61.5` bytes for
repair-only. This does not create a paper claim, but it gives a concrete next
real-model ablation: before target repair, ask for discriminative checks that
can reject semantically drifted but locally consistent routes.

I then tried the obvious stacked follow-up on that weak bridge clue: keep the
same low-rank bridge checkpoint and add the best current runtime routing knobs
on top. Three variants all failed on the matched sparse `gsm8k_eval_10` slice:

- retrieval-head-style runtime head selection (`retrieval_peak`, ratio `0.5`)
- direct QK-fidelity runtime head selection (`attention_qk_fidelity`, ratio `0.5`)
- prior-plus-live blend head selection (`attention_blend`, ratio `0.5`, `alpha=0.25`)

All three fell to `0.0000`, though they did cut the bridge bytes from roughly
`297k` down to roughly `157k`.

## 2026-04-21 Plan Update

The newest GSM30 asymmetric K/V controls shift the plan away from claiming a
single deterministic selector. `random/random` was the best matched selector
row at salt `0`, and salt `1` replicated the aggregate `+0.0667` delta, but
salt `2` collapsed to `-0.0667`. The stochastic branch is therefore a generator
of candidate routes, not a standalone method.

Revised positive-method lane:

1. **Multi-route stochastic candidate generation**
   - Run 3 to 5 deterministic salts for the same budget.
   - Aggregate by majority vote, answer-normalized vote, or target verifier.
   - Report seed variance, answer entropy, paired flips, latency, and bytes.

2. **Uncertainty-gated routing**
   - Use target-alone confidence, route disagreement, and answer entropy to
     decide when to invoke stochastic routes.
   - Keep high-confidence examples deterministic to avoid salt-2-style harm.

3. **Task-aware protected subspace**
   - Extend the toy signal-aware protected-channel result into the real bridge:
     activation/QK/supervised-signal channel masks, then orientation alignment.
   - Compare fixed, PCA, signal-aware, and AWQ-style activation-aware masks.

4. **Mixed-budget / quantization-inspired transport**
   - Borrow AWQ/EXL2-style allocation: spend more bytes on layers/channels with
     high flip saliency or QK error.
   - Report accuracy per transmitted byte and per-layer allocation telemetry.

Immediate success bar:

- Beat target-alone on GSM30 across at least 3 stochastic salts or an aggregated
  3-route verifier.
- Show non-negative paired delta on every salt or route aggregate.
- Preserve the matched-control ladder: attention/energy, attention/attention,
  random/random, shuffled/null, and target-alone.
- Keep the method positive on at least one second slice before drafting a main
  claim.

The next bridge-style follow-up tested a stronger version of the same idea:
replace the scalar query gate with a **query-conditioned low-rank bridge bank**
selected by live target attention-template agreement. The first version,
`bridge_low_rank_bank`, replaced the global bridge with a 4-expert low-rank
bank on top of the same grouped-subspace transport checkpoint. It was a clean
negative on the fair `gsm8k_5` control: `0.0000` at `722,107.7` average bytes.

That means:

- query-conditioned routing is not enough if it discards the stable global
  bridge map
- the bridge lane is probably a **base-plus-residual** story rather than a
  pure mixture-of-experts story

I then tested exactly that stronger base-preserving variant:
`bridge_ridge_residual_bank`, which keeps the full `bridge_ridge` map and adds
a 4-expert low-rank residual bank selected by the same live target attention
profile. That also stayed at `0.0000` on the fair `gsm8k_5` smoke, with the
same heavy byte profile.

That means:

- mean attention-template routing is now looking saturated as the query signal
  for bridge specialization
- if the bridge lane stays alive, the next serious step should use a **richer
  query-conditioned signal** such as QK/retrieval features or a richer
  interaction/distillation target, not another attention-template gate or bank

That means:

- the weak low-rank bridge clue does not stabilize under the current selector
  stack
- runtime routing is still useful as a byte-control knob, but not yet as a
  method rescue
- if the positive-method lane stays alive, the next bridge step has to change
  the bridge itself, not only the runtime selector

I then tried one more “stack the small fixes” move: grouped-subspace transport,
rank-4 residual, low-rank bridge correction, and the stronger
`learned_head_ridge` fusion head fit from the same `64`-prompt calibration
slice. On the first matched sparse `gsm8k_5` smoke it also collapsed to
`0.0000`.

That means:

- we are close to saturating the current family of static linear stack-ups
- the next bridge-style attempt should probably be query-conditioned or trained
  against a richer token-interaction target, not just another fixed linear
  correction layer

I then tried the smallest bridge variant that could still use more signal than
the low-rank repair: a decoder-side `bridge_affine` correction that sees both
the dequantized translated state and the pre-quant translated prediction,
while keeping the same grouped-subspace transport, the same rank-4 residual,
and the same `64`-prompt calibration slice. It reproduced the same weak smoke
pattern exactly:

- matched sparse `gsm8k_5`: `0.2000`
- matched sparse `gsm8k_eval_10`: `0.0000`
- bytes on `gsm8k_eval_10`: `297,233.5`

That means:

- the bridge lane is still the first live internal adapter clue
- but using both pre-quant and post-quant translated states is still not
  enough, by itself, to stabilize the effect
- the next adapter-style attempt should probably be **query-conditioned** or
  fitted against a **richer interaction/distillation target**, not another
  static linear correction

The strongest sidecar consensus after this update is now:

1. `C2C` remains the main external bar.
2. Token/vocabulary mismatch is probably **not** the first-order blocker on the
   exact Qwen2.5-0.5B -> Qwen3-0.6B pair because both configs share the same
   `vocab_size = 151936`; serialization / thinking-mode mismatch is the
   cheaper control to test.
3. The most plausible positive stack is now:
   - gauge-fix / canonicalize head space
   - transport in that stabilized space
   - add a tiny query-conditioned bridge / projector
   - use a richer target such as token-interaction or target-refinement
     distillation if the bridge is trained

I then widened the bridge one more step with `bridge_ridge`: a full linear
bridge over both the dequantized translated state and the pre-quant translated
prediction, still on top of grouped-subspace transport + rank-4 residual. This
was the first bridge branch to survive beyond tiny smokes:

- `gsm8k_5`: `0.4000`
- matched `gsm8k_eval_10`: `0.1000`
- `gsm8k_gate_search_30`: `0.0667`
- exact `gsm8k_eval_70`: `0.0429`

That means:

- the bridge lane is now the first internal method family that looks **stable
  enough to keep pushing**
- but it still does **not** beat grouped-subspace + rank-4 residual (`0.0571`)
  or the old fixed-prior branch (`0.0857`)
- so the strongest next positive-method shot is now **query-conditioned
  bridge/projector on top of the live transport**, not another static transport
  descriptor and not another evaluator-only overlay

The cheapest missing fairness control is still:

- Qwen3 prompt serialization / `enable_thinking=False` alignment in the main
  evaluator, because the official Qwen3 docs say that non-thinking mode is the
  setting that aligns Qwen3 with earlier Qwen2.5-Instruct behavior

That control is now done on the cheapest held-out slice. With shared chat
serialization and `enable_thinking=False` on both sides:

- `target-alone` on `gsm8k_eval_10`: `0.1000`
- `bridge_ridge` on the same controlled slice: `0.1000`

So prompt/thinking alignment is a fairness control we should keep, but it does
**not** create a bridge advantage by itself. The bridge lane is still alive,
but the next positive-method attempt has to come from the method itself:

- query-conditioned bridge / projector
- richer token-interaction or distillation target
- or both together

I then tried the cheapest dynamic version of that bridge idea:
`bridge_ridge_query`. This keeps the same `bridge_ridge` correction but gates
the correction by live target attention-template agreement with a calibration
mean template. It was calibrated and evaluated under the fair shared-chat /
`enable_thinking=False` regime.

Controlled held-out results:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`

So the branch is negative. The cheapest query-conditioned bridge gate is not
enough, and in this form it is worse than the static `bridge_ridge` branch.

That sharpens the next move again:

- not a scalar bridge gate
- not another static bridge
- next serious bridge attempt should be a **query-conditioned bridge bank /
  projector** or a bridge fit against a richer token-interaction target

I then tried both of those “next bridge bank” variants under the same fair
shared-chat / `enable_thinking=False` regime:

- `bridge_low_rank_bank`, which swaps the whole bridge for a routed low-rank
  bank
- `bridge_ridge_residual_bank`, which keeps the stable `bridge_ridge` base and
  only routes a low-rank residual on top

Both collapsed immediately on the first `gsm8k_5` smoke (`0.0000` at
`722,107.7` bytes), so the bridge-bank lane was not rescued by attention-based
query routing alone.

I then tried the richer version of the same idea:
`bridge_ridge_qk_residual_bank`, which keeps the full `bridge_ridge` base but
routes the residual bank with live QK/retrieval profiles rather than mean
attention templates. That also collapsed immediately on the same fair
`gsm8k_5` smoke (`0.0000` at `722,107.7` bytes).

That means:

- richer routing signals are still **not** enough on top of a bridge trained
  only with plain latent regression
- the likely missing piece is now the **training target**, not just the router

I then changed the training target instead of the router:
`bridge_ridge_qk_weighted` keeps the same global `bridge_ridge` form but fits
it with calibration samples reweighted by target last-token QK retrieval
importance. This is still cheap and closed-form, but it stops treating all
positions equally during bridge fitting.

Held-out behavior under the same fair regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`

That means:

- changing the supervision target is more plausible than changing only the
  router, because it still reproduces a nonzero smoke
- but the first retrieval-weighted bridge fit is still **not stable**
- the next serious bridge attempt should likely move one step further toward a
  **token-interaction / affinity distillation target**, not just another
  weighted latent-regression variant

So the positive-method lane is now narrower:

1. keep the fair Qwen control on
2. stop spending cycles on attention-template or QK-routed residual banks
3. if we keep pushing, the next serious branch should be a **query-conditioned
   bridge/projector trained against richer interaction targets**

I then tried the cleanest remaining “query-conditioned projector” version:
`bridge_ridge_qk_projector`. This branch feeds aligned target query features
directly into the bridge itself, rather than using them only for routing or
sample weighting. Concretely, it fits the bridge over both the translated
state and the elementwise query-conditioned translated state.

That is the first genuinely query-conditioned bridge **inside the translator**
rather than another evaluator overlay or residual bank.

Held-out result under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.0000`

That means:

- richer live query features alone are still **not** enough when the bridge is
  still trained against plain latent targets
- the remaining positive-method lane is now even narrower than before:
  if we keep pushing, the next serious bridge attempt should change the
  **training target**, not only the featureization or routing

So the strongest next step is now:

1. keep the fair Qwen control on
2. stop spending cycles on more latent-regression bridge gates, banks, or
   query projectors
3. move to a **token-interaction / affinity / attention-behavior distillation**
   bridge target

I then tried one more bridge branch before fully conceding the latent-
regression family: `bridge_ridge_qk_adapter`. This keeps the same closed-form
`bridge_ridge` base but adds a tiny learned low-rank residual adapter over
query-conditioned translated features during calibration.

Held-out behavior under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`
- bytes on the controlled slice: `720,487.3`

That means:

- the first **learned** query-conditioned bridge residual is at least weakly
  alive on the cheapest fair smoke
- but it is still not stable enough to claim a positive method
- the next serious bridge step should therefore keep the tiny learned-adapter
  framing, but change the **training target** to an interaction/distillation
  target rather than another latent-regression variant

So the next highest-value branch is now:

1. keep the fair Qwen control on
2. keep the tiny learned bridge framing
3. train it against **attention / affinity / interaction distillation**, not
   plain hidden-state regression

I then tried the cheapest interaction-shaped version of that idea without
changing the calibration data path:
`bridge_ridge_qk_affinity_adapter`. This keeps the same learned
query-conditioned residual adapter, but adds a query-conditioned affinity
matching loss over calibration samples.

Held-out behavior under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`
- bytes on the controlled slice: `720,487.3`

That means:

- the cheap affinity target does **not** improve over the plain learned
  adapter; it ties the smallest smoke and still fails the next held-out slice
- so the next branch should not be another “local residual plus one more
  scalar loss” variant
- the next real move should be a **stronger teacher signal**:
  explicit attention-behavior distillation, richer token-affinity
  distillation, or a prediction-level teacher objective

So the highest-value next branch is now:

1. keep the fair Qwen control on
2. keep the tiny learned bridge framing
3. move to a **stronger distillation target**, not another local residual loss

I then tried the strongest teacher target we could add cheaply without
collecting new calibration artifacts:
`bridge_ridge_qk_attnkl_adapter`. This keeps the same learned
query-conditioned residual adapter, but adds sampled attention-logit KL over
calibration query/key tensors.

Held-out behavior under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.0000`
- bytes on the smoke: `722,107.7`

That means:

- even the strongest cheap local teacher target we could add inside the
  existing calibration path is still **not enough**
- this effectively closes the current “small residual plus one more local
  distillation loss” family
- the next rational move is now either:
  - a materially stronger teacher signal, or
  - the external comparator lane, especially Expected Attention / KVPress

I then closed the fastest fair external-comparator lane we could test in-repo:
an **Expected Attention-style** selector on top of the same grouped-subspace +
rank-4 residual checkpoint, still under shared chat serialization and
`enable_thinking=False`. The result is useful but negative:

- `gsm8k_5`: `attention_expected = 0.2000`
- `gsm8k_5`: `attention_expected_shuffled = 0.2000`
- controlled `gsm8k_eval_10`: `attention_expected = 0.1000`
- controlled `gsm8k_eval_10`: `attention_expected_shuffled = 0.1000`

That means:

- our current in-repo Expected Attention-style approximation is a **fair
  negative-boundary comparator**, not a live new baseline
- it should be reported explicitly as **Expected Attention-style**, not as
  exact KVPress parity
- the next highest-value method step is still a **stronger teacher signal**
  for the tiny bridge, not another selector tweak

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. report `attention_expected` with its shuffled null as a negative-boundary
   comparator
4. implement a **CAB / EM-KD / prediction-level distillation** bridge target
   next

I then tried the first materially stronger bridge teacher in that family:
`bridge_ridge_qk_cab_adapter`. This keeps the tiny learned
query-conditioned residual bridge, but supervises it with **prompt-local
causal attention behavior** instead of plain latent regression, cheap
affinity matching, or global attention KL.

Held-out behavior under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`
- bytes on the controlled slice: `681,668.4`

That means:

- CAB-style local attention supervision is **more principled** than the older
  local losses, and it is slightly lighter than the earlier learned-adapter
  family
- but it still does **not** stabilize on the next held-out slice
- so even this stronger local attention teacher is not enough in the current
  tiny bridge form

So the next highest-value stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep `attention_expected` plus its shuffled null as a negative-boundary
   comparator
4. if we keep pushing the bridge lane, move beyond local attention behavior
   alone to **richer affinity or prediction-level distillation**, or make the
   bridge itself more expressive via a routed expert mixture

I then tried the first real “make the bridge itself more expressive” variant:
`bridge_ridge_qk_cab_bank`. This keeps the same grouped-subspace transport +
rank-4 residual checkpoint and the same prompt-local causal attention teacher,
but replaces the single learned residual bridge with a QK-routed bank of
query-conditioned bridge experts.

Held-out behavior under the same fair shared-chat / `enable_thinking=False`
regime:

- `gsm8k_5`: `0.2000`
- bytes on the smoke: `686,026.6`

That means:

- a routed bridge mixture is **not** enough by itself to improve over the
  single-expert CAB branch in this first form
- the current bank lane looks saturated if the teacher target stays local
  and the bridge remains this small
- so the next branch should likely change the **teacher target** or the
  **canonicalization step**, not just add more bridge experts

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep `attention_expected` plus its shuffled null as a negative-boundary
   comparator
4. if we keep the bridge lane alive, move to **richer affinity or prediction-level distillation**
5. in parallel, test a **rotation-canonicalized grouped transport** shortcut

I then tried the first explicit prompt-local token-interaction target in this
bridge family: `bridge_ridge_qk_emkd_adapter`. This keeps the same grouped-
subspace transport + rank-4 residual checkpoint and the same tiny learned
query-conditioned residual bridge, but it swaps in a prompt-local
token-interaction distribution loss inspired by EM-KD rather than CAB-style
causal attention behavior alone.

Held-out behavior under the same fair shared-chat / `enable_thinking=false`
regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`
- bytes on the controlled slice: `681,668.4`

That means:

- a richer **local interaction** teacher is more principled, but still not
  enough to stabilize the bridge on the next held-out slice
- the current “tiny learned bridge + local teacher upgrade” family is now
  looking saturated
- the next highest-value method move is no longer another small local loss
  tweak; it should be either:
  - a **prediction-level / stronger teacher** branch, or
  - a **stronger canonicalization / transport** branch before the bridge

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep `attention_expected` plus its shuffled null as a negative-boundary comparator
4. if we keep the bridge lane alive, move beyond local interaction matching to a **stronger teacher signal**
5. in parallel, test a **rotation-canonicalized grouped transport** shortcut

I then tried that geometry-side shortcut directly: `grouped_rotational_transport`.
This keeps the same grouped soft-transport + rank-4 residual structure, but
covariance-normalizes each grouped source/target block into its own canonical
rotational gauge before fitting the shared grouped transport block.

Held-out behavior under the same fair shared-chat / `enable_thinking=false`
regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- bytes on the controlled slice: `681,668.4`

That means:

- this branch survives the controlled `gsm8k_eval_10` slice where the recent
  local bridge-distillation upgrades all collapsed
- but it only rises to the same `0.1000` level as the controlled
  `target_alone` read, so it still does **not** create a positive method gain
- the geometry lane is therefore still weakly alive, but not yet a paper win

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. add exact KVPress / Expected Attention when we want the next clean external comparator
4. if we keep the method lane alive, the two live paths are:
   - a **stronger teacher closer to prediction space**
   - a **stronger geometry / canonicalization step** than the current grouped rotational fit

I then closed that exact external comparator lane with the vendored KVPress
pipeline itself through `scripts/run_kvpress_eval.py`.

Exact held-out reads on `Qwen/Qwen3-0.6B` under the same fair shared-chat /
`enable_thinking=false` control were:

- `gsm8k_5`, no press: `0.2000`
- `gsm8k_5`, `ExpectedAttentionPress`: `0.2000`
- controlled `gsm8k_eval_10`, no press: `0.1000`
- controlled `gsm8k_eval_10`, `ExpectedAttentionPress`: `0.1000`

That means:

- the exact external Expected Attention baseline reproduces the same
  **negative-boundary comparator** read as our in-repo approximation
- so Expected Attention is now closed as an honest external null / boundary
  comparator on this pair, not a new live baseline
- the next highest-value work should go back to method design, not more
  Expected-Attention variants

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, the two live paths are:
   - a **stronger teacher closer to prediction space**
   - a **stronger geometry / canonicalization step** than the current grouped rotational fit

I then tried the next stronger geometry-side follow-up:
`grouped_fitted_rotation_transport`. This keeps the same grouped soft-transport plus rank-4 residual structure and the same fair shared-chat / `enable_thinking=false` Qwen control, but it replaces the generic rotational canonicalization with a **calibration-fit grouped gauge fix**: each grouped block is ZCA-whitened, a rectangular orthogonal map is fit directly in that whitened space, and the grouped transport is assembled from those fitted blockwise gauges.

Offline fit improved modestly over `grouped_rotational_transport`:

- `K` cosine: `0.853`
- `K` relative Frobenius error: `0.493`
- `V` cosine: `0.355`
- `V` relative Frobenius error: `0.923`

Held-out behavior under the same fair controlled regime was:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- bytes on the controlled slice: `681,668.4`

That means:

- a more explicit calibration-fit gauge fix is more principled and does improve
  offline alignment quality a bit
- but it exactly ties the earlier `grouped_rotational_transport` branch on the
  held-out slices that matter
- so stronger grouped canonicalization is still **not** a positive method
  result by itself

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, the two live paths are:
   - a **stronger teacher closer to prediction space**
   - a **shared-basis / dictionary-style canonicalization** beyond the current rotational and fitted-gauge probes

I then tried the first explicit shared-basis follow-up:
`grouped_shared_basis_transport`. This keeps the same grouped soft-transport plus rank-4 residual structure and the same fair shared-chat / `enable_thinking=false` Qwen control, but it replaces the grouped rotational fit with a **shared low-rank cross-covariance basis** per grouped block. Each block is ZCA-whitened, projected into a shared source/target coefficient basis from the cross-covariance SVD, and the grouped transport is fit in that coefficient space.

Offline fit again improved slightly over `grouped_rotational_transport`:

- `K` cosine: `0.854`
- `K` relative Frobenius error: `0.491`
- `V` cosine: `0.354`
- `V` relative Frobenius error: `0.926`

Held-out behavior under the same fair controlled regime was:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- bytes on the controlled slice: `681,668.4`

That means:

- a shared-basis / coefficient-space canonicalization is more faithful to the
  dictionary-style references than the earlier pure rotation probes
- but in this simple form it still exactly ties the earlier geometry branches
  on the held-out slices that matter
- so the current grouped-canonicalization family now looks close to saturated

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, the best next move is now more likely a **stronger teacher closer to prediction space** than another small canonicalization variant

I then tried one more explicit stronger-teacher bridge follow-up:
`bridge_ridge_qk_readout_adapter`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and
the same fair shared-chat / `enable_thinking=false` Qwen control, but it
changes the tiny bridge teacher from local CAB / EM-KD-style structural losses
to a prompt-local **attention readout** target. While wiring that branch I also
fixed a real implementation bug in the `bridge_ridge_qk_*adapter` family: the
earlier adapter modes had only been fitting the K-side query residual and were
leaving the V-side query residual at zero.

The result was still negative under the fair controlled regime:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.0000`
- bytes on the controlled slice: `681,668.4`

That means:

- moving the local bridge teacher from plain latent regression to local
  attention readouts is still not enough
- fixing the missing V-side adapter fit did not, by itself, rescue the bridge
  lane
- the stronger-teacher lane is still the best live lane, but the next bridge
  teacher should likely move **above prompt-local structural losses** toward a
  prediction-level target such as approximate likelihood / output-distribution
  matching

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, the best next move is now a **prediction-level / stronger-teacher bridge** rather than another local bridge-loss tweak or another small geometry variant

I then implemented the first explicit prediction-level bridge teacher:
`bridge_ridge_qk_predkl_adapter`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and
the same fair shared-chat / `enable_thinking=false` Qwen control, but it
replaces the prompt-local structural teachers with a calibration-time
**top-k next-token teacher**. The bridge now sees aligned target query
features plus an approximate likelihood target built from target next-token
log-probabilities and output-embedding rows. On the full 64-prompt calibration
slice, offline fit was:

- `K` cosine: `0.869`
- `K` relative Frobenius error: `0.469`
- `V` cosine: `0.393`
- `V` relative Frobenius error: `0.908`

The first fair held-out smoke was a clean negative:

- `gsm8k_5`: `0.0000`
- bytes on the smoke slice: `722,107.7`

That means:

- the repo now contains a real prediction-level / likelihood-style bridge
  branch rather than only local structural teachers
- but even with that stronger teacher, the current tiny local residual bridge
  still dies immediately on the first fair held-out slice
- so the stronger-teacher lane is still the right conceptual lane, but the
  next live step likely needs either a richer bridge family or a more direct
  output-side teacher than the current low-capacity residual form

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, move to a **richer prediction-level bridge**
   rather than another local bridge-loss tweak or another small geometry variant

I then tried the smallest richer-capacity follow-up in that same lane:
`bridge_ridge_qk_predkl_bank`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and
the same fair shared-chat / `enable_thinking=false` Qwen control, but it
replaces the single prediction-level residual bridge with a **QK-routed bank
of query-conditioned bridge experts** trained under the same top-k next-token
teacher.

On the 16-prompt smoke calibration slice, the first fair held-out smoke was
still negative:

- `gsm8k_5`: `0.0000`
- bytes on the smoke slice: `722,107.7`

That means:

- moving from one tiny prediction-level bridge to a small routed bridge bank
  did not revive the method
- the current small modular bridge-bank family now also looks close to
  saturated
- the next live bridge step likely needs a more materially different modular
  interface, not another small residual bank variant

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. if we keep the method lane alive, move to a **more materially different modular bridge**
   rather than another tiny residual or banked variant

I then hardened the paper-facing artifact layer instead of spending the next
cycle on another small bridge tweak. The new builder,
`scripts/build_reviewer_artifacts.py`, produces:

- `paper/bytes_accuracy_frontier_20260420.json`
- `paper/bytes_accuracy_table_20260420.md`
- `paper/paired_flip_table_20260420.jsonl`
- `paper/paired_flip_table_20260420.md`

Those artifacts make the current paper story much sharper:

- exact `gsm8k_eval_70`: `fixed prior = 0.0857`,
  `grouped_subspace + rank-4 residual = 0.0571`, `bridge_ridge = 0.0429`,
  `C2C = 0.1286`
- controlled `gsm8k_eval_10`: `target-alone = 0.1000`, `bridge_ridge = 0.1000`,
  `grouped_rotational_transport = 0.1000`,
  `grouped_fitted_rotation_transport = 0.1000`,
  `grouped_shared_basis_transport = 0.1000`, exact `KVPress no-press = 0.1000`,
  exact `ExpectedAttentionPress = 0.1000`
- paired flips: `fixed prior` still beats its shuffled null, but
  `grouped_subspace + rank-4 residual` still loses to `fixed prior`,
  `bridge_ridge` still does not close that gap, and the controlled survivor
  family still exactly ties the controlled target floor

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. use the new frontier + paired-flip artifacts as the main reviewer-facing evidence layer
5. if the method lane stays alive, move to a **materially different modular interface**
   or a stronger output-side teacher, not another tiny residual family

I then implemented the first explicit shared-plus-private modular bridge in the
repo: `bridge_ridge_qk_asym_adapter`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and the
same fair shared-chat / `enable_thinking=false` Qwen control, but it replaces
the fully separate K-side and V-side query adapters with one shared
query-conditioned bottleneck plus private K and V residual heads. This is the
closest branch here to an AsymLoRA-style interface rather than another
monolithic tiny residual.

On the 64-prompt calibration slice, offline fit was:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the first materially different shared-plus-private interface is **weakly
  alive**
- but it still only ties the controlled `target-alone` floor rather than
  beating it
- so the modular-interface lane remains plausible, but this first AsymLoRA-
  style bridge is not yet a positive method result

So the highest-value next stack is now:

1. keep the fair Qwen control on
2. keep `C2C` as the main external bar
3. keep exact KVPress / Expected Attention in the paper as a negative-boundary comparator
4. keep the new frontier + paired-flip artifacts as the reviewer-facing evidence layer
5. if the method lane stays alive, the next live pivots are:
   - a stronger output-side / likelihood-style teacher on top of a more
     materially different interface, or
   - a shared sparse dictionary / SAE bridge rather than another dense tiny residual

I then stacked those two plausible fixes directly: the shared-plus-private
interface plus the prediction-level teacher in
`bridge_ridge_qk_asym_predkl_adapter`.

This keeps the same one-shared-plus-two-private low-rank bridge structure as
`bridge_ridge_qk_asym_adapter`, but adds the same calibration-time top-k
next-token teacher used by `bridge_ridge_qk_predkl_adapter`.

On the same 64-prompt calibration slice, offline fit was unchanged:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were also
unchanged:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- stacking the current best output-side teacher on top of the first
  shared-plus-private dense interface did **not** improve the method
- the dense modular-interface lane is still weakly alive, but it is still tied
  to the controlled target floor
- the next live pivots are now narrower:
  - a more materially different interface module in the Attention Editing /
    LRAgent direction, or
  - a shared sparse dictionary / SAE bridge in the USAE / SPARC direction

I then implemented that shared sparse dictionary pivot directly as
`bridge_ridge_qk_sae_adapter`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and the
same fair shared-chat / `enable_thinking=false` Qwen control, but it replaces
the dense shared bottleneck with a small top-k sparse code that is decoded
separately for K and V.

On the same 64-prompt calibration slice, offline fit was still:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Held-out read:

- `gsm8k_5`: `0.0000`
- bytes: `686,026.6`

That means:

- the first lightweight SAE-style interface is a **clean negative**
- the dense-transport-plus-tiny-bridge family is still not rescued by simply
  moving to a sparse shared code
- the next real method pivots are now even narrower:
  - a more materially different module in the Attention Editing / LRAgent /
    MoRA direction, or
  - a stronger dynamic output-side teacher closer to prediction space

Comparator guidance also sharpened:

1. keep `C2C` as the main external bar
2. keep exact KVPress / Expected Attention as the negative-boundary comparator
3. if we spend another comparator day, do **KVzip** next
4. keep **Quest** as the next fallback comparator after KVzip

I then pushed the dynamic modular lane one step further with
`bridge_ridge_qk_generated_adapter`.

This keeps the same `grouped_subspace_transport + rank-4 residual` base and the
same fair shared-chat / `enable_thinking=false` Qwen control, but it replaces
the fixed residual bridge with a continuous query-conditioned mixture over a
shared bank of low-rank bridge atoms, in the SHINE / Text-to-LoRA / MoRA
direction.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Held-out read:

- `gsm8k_5`: `0.0000`
- bytes: `686,026.6`

That means:

- the first generated / instance-specific bridge is also a **clean negative**
- simply moving from a fixed bridge to a continuous generated low-rank mixture
  is not enough in the current transport family
- the next serious method pivots are now:
  - a more materially different attention/module replacement in the
    Attention Editing direction, or
  - a dynamic output-alignment teacher with contextual mapping / token
    interaction supervision

I then tested that dynamic-teacher idea directly as
`bridge_ridge_qk_asym_dynmap_adapter`.

This keeps the same shared-plus-private interface as
`bridge_ridge_qk_asym_adapter`, but it replaces the static top-k next-token KL
teacher in `bridge_ridge_qk_asym_predkl_adapter` with a context-reweighted
teacher over the same top-k candidate rows.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the first dynamic output-alignment teacher on top of the stronger modular
  interface is **weakly alive**
- but it still only ties the controlled `target-alone` floor and does not beat
  the plain asym interface
- so the next live method pivots are now even narrower:
  - a materially different projector / module-replacement interface in the
    Attention Editing / BRIDGES / Skywork direction
  - or a richer dynamic output-alignment teacher with explicit contextual
    token/output mapping rather than a top-k reweighting of the same rows

Comparator guidance also shifted again after the latest web-backed sidecars:

1. keep `C2C` as the main external bar
2. keep exact KVPress / Expected Attention as the negative-boundary comparator
3. if we spend another comparator day, **DeltaKV** is now the highest-value
   external control
4. `DapQ` is the strongest newer decoding-aligned control if a clean public
   repo appears
5. `KVzip` and then `Quest` remain the next already-cloned fallback controls

I then tested the smallest projector-side version of that module/interface idea
directly as `bridge_ridge_qk_asym_projector`.

This keeps the same shared-plus-private paired K/V interface as
`bridge_ridge_qk_asym_adapter`, but it upgrades the base bridge into a full
query-conditioned projector before the low-rank shared/private refinement.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

The first fair held-out smoke was:

- `gsm8k_5`: `0.0000`
- bytes: `686,026.6`

That means:

- the first projector-side interface branch is a **clean negative**
- the current evidence is that a small post-transport projector is still too
  close to the saturated bridge family to rescue the method
- the next serious method pivots are now narrower again:
  - an explicit attention/module-replacement interface in the Attention
    Editing / LLM Modules direction
  - or a richer contextual token/output mapping teacher on top of a genuinely
    different interface

I then tested the smallest explicit attention-side transfer-module version of
that idea directly as `bridge_ridge_qk_xattn_adapter`.

This keeps the same grouped-subspace transport + rank-4 residual base, but it
replaces the residual-style shared bridge with a tiny query-conditioned
cross-attention module over the live K/V-side transport signals
(`x`, `aux_input`, `paired_input`, `paired_aux_input`).

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the first explicit attention-side transfer module is **weakly alive**
- but it still only ties the controlled `target-alone` floor
- the current live methodological conclusion is now sharper:
  - tiny residual bridges are saturated
  - small projector variants are saturated
  - the first explicit attention-side transfer module still does not beat the
    floor
- so the next serious positive-method shot should be either:
  - a more complete **module replacement** in the Attention Editing / LLM
    Modules direction
  - or a stronger **dynamic output-alignment teacher** with explicit
    contextual remapping rather than another local top-k reweighting

I then tested that stronger contextual teacher directly on top of the explicit
xattn interface as `bridge_ridge_qk_xattn_dynmap_adapter`.

This keeps the same grouped-subspace transport + rank-4 residual base and the
same tiny query-conditioned cross-attention bridge over the live K/V-side
transport signals, but it adds the same context-reweighted top-k teacher used
by `bridge_ridge_qk_asym_dynmap_adapter`.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- stacking the current contextual dynamic teacher on top of the xattn
  interface does **not** improve it
- the current local-teacher lane is now saturated even on top of the
  weakly-alive explicit attention bridge
- the next serious positive-method shot should now move one step further away
  from the current tiny bridge family:
  - a fuller **module replacement** in the Attention Editing / LLM Modules
    direction
  - or a richer dynamic remapping teacher with explicit token alignment /
    interaction structure rather than another local reweighting of the same
    top-k target rows

I then tested the first fuller slotted module-replacement style bridge
directly as `bridge_ridge_qk_module_adapter`.

This keeps the same grouped-subspace transport + rank-4 residual base, but it
replaces the smaller residual / xattn bridge variants with a learned
query-conditioned cross-attention module over live K/V-side transport signals
plus learned bridge slots, followed by a nonlinear readout trained with
calibration-time top-k prediction distillation.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- even a fuller slotted attention-side transfer module still only ties the
  same controlled floor as the other weakly-alive modular branches
- so the current local interface-elaboration lane is now looking saturated,
  not just the tiny residual lane
- the next serious method pivots should now be even more concrete:
  - a more literal **Attention Editing / LLM Modules** style module
    replacement
  - or a richer **dynamic token/output remapping** teacher with explicit
    contextual alignment rather than another local top-k reweighting

I then tested the more literal direct-output version of that same idea as
`bridge_ridge_qk_module_replace`.

This keeps the same grouped-subspace transport + rank-4 residual base and the
same slotted attention-side module shape, but it trains that module to predict
the full corrected K/V directly rather than only a residual on top of the
fixed bridge.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.870`, relative Frobenius error `0.468`
- `V` cosine `0.397`, relative Frobenius error `0.907`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- even the direct-output module-replacement variant only ties the same weak
  controlled floor
- so the current local module-elaboration lane is looking saturated in a
  stronger sense than before: additive modules, contextual-teacher modules,
  and now direct-output modules all land on the same floor
- the next serious method pivots should now be:
  - a more global **Attention Editing / LLM Modules** style replacement
  - or a richer **dynamic token/output remapping** teacher with explicit
    contextual alignment rather than another local top-k target

I also built a first layer-localization artifact from the live
`selector_trace` telemetry on the controlled `gsm8k_eval_10` slice:
`paper/layer_localization_20260420.{jsonl,md}`.

The main read is that the current weakly-alive modular family
(`shared_plus_private_asym_adapter`, `shared_plus_private_dynmap_adapter`,
`xattn_adapter`, `xattn_dynmap_adapter`, `module_adapter`,
`module_replace`, `tokenbasis_replace`) all reuse the same top target-layer
pattern:

- `L27 <- S23`
- `L5 <- S4`
- `L23 <- S20`
- `L22 <- S19`
- `L8 <- S7`

That means:

- the weakly-alive branches are not moving the runtime layer-selection story
  in a meaningful way under the fair control
- so more local interface elaboration is even less likely to rescue the
  method than before
- this strengthens the case for a pivot that changes something **upstream** of
  the current local bridge fit, such as:
  - a more global **Attention Editing / LLM Modules** style replacement
  - or a **token/span remapping** or richer token-output alignment teacher

I then tested a target-native basis version of that same direct-output module
idea as `bridge_ridge_qk_tokenbasis_replace`.

This keeps the same grouped-subspace transport + rank-4 residual base and the
same slotted attention-side module shape as `bridge_ridge_qk_module_replace`,
but it constrains the direct K/V outputs to a basis distilled from target
next-token output rows instead of allowing a free dense output map.

On the same 64-prompt calibration slice, offline fit remained:

- `K` cosine `0.869`, relative Frobenius error `0.469`
- `V` cosine `0.393`, relative Frobenius error `0.908`

Under the matched-bytes fair controlled regime, the held-out reads were:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- anchoring the direct-output module to a target next-token basis is still not
  enough to move above the controlled floor
- so the current modular/interface lane now looks saturated in an even more
  specific sense: free dense outputs and target-native basis outputs both land
  on the same weakly-alive pattern
- the next serious method pivots should therefore move **upstream** of the
  current local bridge:
  - explicit **token/span remapping** or vocabulary-side alignment before the
    bridge
  - or a more global **Attention Editing / LLM Modules** style replacement

I then tested the first explicit upstream remapping branch as
`bridge_ridge_qk_spanalign_module_replace`.

This keeps the same grouped-subspace transport + rank-4 residual base and the
same slotted attention-side module shape as `bridge_ridge_qk_module_replace`,
but it changes calibration pairing itself: source and target samples are
aligned by monotone overlap on the raw prompt span rather than by truncated
absolute token position.

On the same 64-prompt calibration slice, that changed the calibration geometry
substantially:

- aligned token pairs: `2702`
- `K` cosine `0.937`, relative Frobenius error `0.334`
- `V` cosine `0.632`, relative Frobenius error `0.763`

Under the matched-bytes fair controlled regime, the held-out reads were still:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the old same-position pairing was indeed part of the upstream problem,
  because offline fit improves sharply once raw prompt content is aligned
- but raw-span overlap alone is still not enough to change held-out behavior
- so the next live remapping step should be:
  - a **contextual** token/span alignment teacher or dynamic remapping cost,
    not just raw prompt overlap, or
  - a more global **Attention Editing / LLM Modules** style replacement on
    top of the better-aligned interface

I then tested the first contextual-remapping follow-up as
`bridge_ridge_qk_ctxalign_module_replace`.

This keeps the same slotted direct-output module as
`bridge_ridge_qk_module_replace`, but it upgrades the upstream pairing from a
single hard target token to a small **context-weighted mixture of target
tokens** for each source token during calibration.

On the same 64-prompt calibration slice:

- contextual remapping samples: `2702`
- mean target tokens per source sample: `2.96`
- `K` cosine `0.937`, relative Frobenius error `0.334`

The fair smoke read was a clean negative:

- `gsm8k_5`: `0.0000`
- bytes: `686,026.6`

That means:

- upstream remapping is still the right live lane, because it keeps improving
  the calibration geometry relative to the older hard-pairing setup
- but a simple local **soft mixture over nearby target tokens** is still not a
  positive method
- so the next remapping step likely needs:
  - dynamic token/span alignment,
  - span-level or likelihood-style remapping teachers,
  - or multi-view remapping losses rather than one fixed local score

I then tested the first explicit output-aware dynamic-remapping branch as
`bridge_ridge_qk_dynalign_module_replace`.

This keeps the same slotted direct-output module as
`bridge_ridge_qk_module_replace`, but it upgrades the upstream pairing again:
candidate target tokens are scored by both **local span/context agreement**
and **next-token output overlap** before the source-to-target token mixture is
formed.

On the same 64-prompt calibration slice:

- dynamic remapping samples: `2702`
- mean target tokens per source sample: `3.00`
- `K` cosine `0.937`, relative Frobenius error `0.334`
- `V` cosine `0.633`, relative Frobenius error `0.763`

Held-out reads:

- `gsm8k_5`: `0.4000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- output-aware dynamic remapping is the strongest recent **live upstream
  method lane**, because it improves on the older remapping smokes
- but the first version still only ties the controlled `target-alone` floor
- so the next method step should keep this upstream dynamic-alignment framing
  and strengthen the teacher:
  - likelihood-style or span-level alignment,
  - multi-view remapping losses,
  - or a stronger dynamic matching rule before pivoting to a broader module
    replacement

I then ran a smaller diagnostic on the first explicit **global monotone**
alignment variant as `bridge_ridge_qk_dpalign_module_replace`.

This keeps the same direct-output slotted module as
`bridge_ridge_qk_module_replace`, but instead of fitting against local
token-mixtures it builds a **global monotone dynamic-program alignment** using
the same context-plus-output score that powered `dynalign`.

On a 16-prompt diagnostic calibration slice:

- dynamic-program pairs: `660`
- mean pairs per prompt: `41.25`
- `K` cosine `0.948`, relative Frobenius error `0.304`
- `V` cosine `0.699`, relative Frobenius error `0.697`

Held-out diagnostic read:

- `gsm8k_5`: `0.0000`

That means:

- improving the alignment **solver** alone is not enough
- the dynalign signal is therefore still best interpreted as a **teacher**
  improvement, not just an alignment-path improvement
- so the next method step should stay on:
  - span-level / likelihood-style teachers,
  - multi-view remapping losses,
  - or byte/token shared interfaces before another global solver pivot

I then tested the first DWA-KD-style teacher follow-up as
`bridge_ridge_qk_dynalign_dwakd_module_replace`.

This keeps the same `dynalign` token mixtures, but strengthens the teacher
inside the same direct-output module fit:

- confidence-weighted calibration samples from alignment concentration and
  prediction entropy,
- plus both plain prediction KL and the dynamic context-shaped teacher term.

On a 16-prompt diagnostic calibration slice:

- samples: `660`
- mean target tokens per source sample: `3.00`
- DWA-KD-style weight range: `0.588` to `1.400`
- `K` cosine `0.948`, relative Frobenius error `0.304`
- `V` cosine `0.699`, relative Frobenius error `0.697`

Held-out diagnostic reads:

- `gsm8k_5`: `0.4000`
- controlled `gsm8k_eval_10`: `0.1000`

That means:

- a stronger teacher does preserve the live dynalign smoke signal
- but the first weighted-teacher version still only ties the controlled floor
- so the next method step should continue in the teacher lane:
  - span-level / likelihood-style supervision,
  - token/span interaction losses,
  - or byte/token shared interfaces

I then tested the first explicit prompt-local interaction follow-up as
`bridge_ridge_qk_dynalign_interact_module_replace`.

This keeps the same `dynalign` token mixtures and the same direct-output
slotted module as `bridge_ridge_qk_module_replace`, but adds prompt-local
interaction distillation during module fitting.

On a 16-prompt diagnostic calibration slice:

- dynamic remapping samples: `678`
- mean target tokens per source sample: `3.00`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- adding a richer **local interaction** loss on top of `dynalign` is
  directionally alive enough to preserve a nonzero smoke
- but it is weaker than both plain `dynalign` and `dynalign_dwakd` on the
  same diagnostic setup
- and it still does not move the controlled slice above the same floor
- so the next serious step should not be another local interaction term:
  - move to span-level / likelihood-style supervision,
  - or a tokenizer-agnostic byte/span interface,
  - while keeping `dynalign` as the live remapping base

I then tested the first target-likelihood teacher follow-up as
`bridge_ridge_qk_dynalign_likelihood_module_replace`.

This keeps the same `dynalign` token mixtures and DWA-style confidence weights,
but injects empirical target next-token likelihood mass into the aligned
top-k prediction teacher before fitting the same direct-output module.

On a 16-prompt diagnostic calibration slice:

- dynamic remapping samples: `678`
- mean target tokens per source sample: `3.00`
- likelihood/DWA-style weight range: `0.706` to `1.303`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- naively injecting the observed target next token into the teacher
  over-anchors the dynamic teacher and loses the plain `dynalign` / DWA
  `gsm8k_5 = 0.4000` smoke signal
- the positive part of the live lane is not generic target-side likelihood
  sharpening; it is specifically the softer output-overlap remapping signal
- next likelihood-style attempts should be span-level approximate likelihood
  matching or tokenization-aware remapping, not a simple gold-token boost

I then tested the first byte/shared-interface control as
`bridge_ridge_qk_bytespan_module_replace`.

This keeps the same direct-output slotted module as
`bridge_ridge_qk_module_replace`, but replaces the upstream calibration pairing
with dominant UTF-8 byte-overlap matching: each source token is mapped to the
target token that overlaps it by the most raw-prompt byte mass.

On a 16-prompt diagnostic calibration slice:

- byte-span pairs: `678`
- mean pairs per prompt: `42.38`
- prompts changed versus char-span `spanalign`: `0`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- byte-span fitting is now wired and measurable, but the current GSM
  calibration prompts do not stress tokenizer/byte-boundary differences at all
- byte/shared-interface ideas should not be counted as a real method lane until
  we add calibration data where byte and char/token segmentation actually
  disagree
- the next tokenizer-side step should be a **byte-stress calibration and
  evaluation harness**, then stack the useful byte signal with `dynalign` or
  `dynalign_dwakd`; more byte-span runs on the same GSM calibration data are
  unlikely to move the paper

I added that first harness as `scripts/analyze_byte_alignment.py`. On the
default Qwen2.5 -> Qwen3 stress prompts it reports `1 / 8` prompts with changed
byte-dominant pairings versus char-span alignment, so the next tokenizer-side
model run should use a deliberately byte-stressed calibration slice rather than
the current GSM-only calibration prompts.

I then added a matched context-only null for the dynamic-alignment branch as
`bridge_ridge_qk_dynalign_ctxonly_module_replace`.

This keeps the later dynalign candidate window and the same direct-output
module-replacement fit, but sets the prediction-overlap score weight to zero
during source-to-target mixture construction.

On a 16-prompt diagnostic calibration slice:

- dynamic context-only samples: `678`
- mean target tokens per source sample: `3.00`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the plain span/context part of dynalign is not what produced the earlier
  `0.4000` smoke
- the prediction-overlap score is carrying the useful signal, because removing
  it drops the smoke back to the byte/span floor
- the controlled slice still ties the target floor, so this is not yet a
  positive method result
- the next serious method step should strengthen the output-aware alignment
  teacher itself, using span-level likelihood, multi-view token remapping, or
  target-side refinement supervision; context-only or byte-only variants on the
  same GSM calibration data are now lower priority

I also added a first layer-localization knockout hook and ran it on the live
`bridge_ridge_qk_dynalign_module_replace` smoke branch.

The hook replaces translated K/V with target K/V for selected target layers,
removing source communication for those layers while preserving the target
prompt cache.

On `gsm8k_5`:

- baseline dynalign module replace: `0.4000`
- recurrent top-layer signature knockout `L27,L5,L23,L22,L8`: `0.2000`
- matched offset-layer knockout `L26,L4,L21,L20,L7`: `0.2000`
- bytes for both knockouts: `563,521.9`

That means:

- the dynalign smoke is sensitive to removing five communicated layers
- the current repeated layer signature is not uniquely causal yet, because the
  offset control drops by the same amount
- layer telemetry remains useful for interpretability, but the next decisive
  layer ablation should be a larger-slice leave-one-out or add-one-back curve,
  not a top-five knockout alone

I then tested the span-level approximate-likelihood follow-up as
`bridge_ridge_qk_dynalign_spanalm_module_replace`.

This keeps the same `dynalign` token mixtures and confidence-weighted module
fit, but replaces the hard observed-next-token likelihood boost with a
span-window teacher: observed future target tokens receive sparse teacher mass
according to the target model's own probability on those tokens, with a
decay by span offset.

On a 16-prompt diagnostic calibration slice:

- dynamic remapping samples: `678`
- mean target tokens per source sample: `3.00`
- span-ALM/DWA-style weight range: `0.588` to `1.470`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- a softer span-window approximate-likelihood teacher still regresses the
  best `dynalign` / DWA `gsm8k_5 = 0.4000` smoke to `0.2000`
- it does not move the controlled slice above the `0.1000` target floor
- the next positive-method attempt should not be another direct likelihood-mass
  variant unless the interface changes; stronger candidates are attention or
  refinement distillation, query-conditioned routing inside the bridge, or a
  more global target-side module interface

I then tested the minimal local-teacher stack:
`bridge_ridge_qk_dynalign_dwainteract_module_replace`.

This keeps the same dynamic remapping and DWA confidence weighting as
`dynalign_dwakd`, keeps the dynamic prediction teacher, and adds the
prompt-local interaction distillation term from `dynalign_interact`.

On the same 16-prompt diagnostic calibration slice:

- dynamic remapping samples: `678`
- mean target tokens per source sample: `3.00`
- DWA-interaction weight range: `0.588` to `1.400`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.2000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- the interaction teacher is not just missing confidence weighting; even when
  stacked with DWA and dynamic prediction supervision, it regresses the DWA
  `0.4000` smoke to `0.2000`
- local teacher stacking is now saturated for the current module-replace
  interface
- the next additive paper method should change the interface or objective
  class: CTPD-style aligned-span preferences, query-conditioned transport/route
  atoms, or a more global target-side module replacement

I then tested the first CTPD-style objective-class change as
`bridge_ridge_qk_dynalign_prefdist_module_replace`.

This keeps dynalign, confidence weighting, and the dynamic top-k prediction
teacher, but adds a pairwise preference loss over the aligned target output
rows instead of injecting exact token likelihood mass or prompt-local
interaction KL.

On the same 16-prompt diagnostic calibration slice:

- dynamic remapping samples: `678`
- mean target tokens per source sample: `3.00`
- preference-distillation weight range: `0.588` to `1.400`
- `K` cosine `0.948`, relative Frobenius error `0.305`
- `V` cosine `0.697`, relative Frobenius error `0.700`

Held-out diagnostic reads:

- `gsm8k_5`: `0.4000`
- controlled `gsm8k_eval_10`: `0.1000`
- controlled bytes on `gsm8k_eval_10`: `681,668.4`

That means:

- aligned output-ranking preferences preserve the best recent dynalign smoke,
  unlike direct likelihood mass, span-ALM, local interaction, or their DWA stack
- the controlled slice still does not move above the `0.1000` target-alone
  floor, so this is not yet a positive method
- the preference objective is worth keeping as the least destructive stronger
  teacher, but the next paper-grade attempt has to pair it with a stronger
  correspondence/interface change rather than treating it as a standalone fix
- the next implementation should be one of: query-conditioned route atoms over
  aligned spans, a more global target-side module replacement, or a
  tokenizer-agnostic byte/span interface after byte-stress confirms useful
  tokenizer divergence

## 2026-04-20 301 Reference Synthesis And Next Ablation Stack

The new literature sweep points to an interface change, not another additive
local-teacher term. Recent MoE routing work specifically warns that routing
after multi-head concatenation can collapse separable head factors, while
cross-tokenizer distillation work treats tokenizer mismatch as a primary
supervision problem rather than a preprocessing nuisance. Multimodal projector
systems point to the same design pattern from another direction: a small
query/latent interface, staged target-space alignment, and gated target-native
injection are safer than unconditional K/V replacement.

Current status:

- Keep `dynalign_prefdist` as the least-destructive stronger teacher because
  it preserves the best `gsm8k_5 = 0.4000` smoke.
- Do not keep stacking direct likelihood, span-ALM, or local interaction KL on
  the current module-replace interface; those variants all regress smoke or tie
  the controlled target floor.
- Treat `readout_adapter` as a negative boundary: it survives GSM5 smoke
  (`0.2000`) but falls to `0.0000` on controlled GSM10, below the target-alone
  `0.1000` floor.
- Treat KVPress ExpectedAttention as a runnable external compression baseline,
  not a harder local bar yet: on GSM5/GSM10 and a one-row Needle smoke it ties
  no-press and is slower on GSM.

Additive paper direction:

- Add a positive method only if it changes the communication interface:
  head-wise route atoms, fixed query-pool transport, gated FiLM/adaLN
  modulation, or a byte/tokenizer-independent target-space probe.
- Keep the old negative ideas in the paper as blocker evidence, not as the
  final contribution: prediction-KL, readout, direct likelihood, span-ALM,
  local interaction, and module replacement all define the saturated baseline.
- Make results interpretable by logging route entropy, atom usage, dead atoms,
  per-layer/head collision counts, byte-family/token-family errors, paired
  flips, and target logit entropy shift before claiming a positive method.

Two-week execution schedule:

- Day 1: implement an offline `headwise_route_atom` diagnostic around the
  existing generated-adapter branch. Success means non-degenerate atom usage,
  lower mean-collapse/collision scores, and at least one positive paired flip
  versus target-alone on controlled GSM10.
- Day 2: implement a deterministic query-pool prefix diagnostic with 4/8/16
  slots and a fixed small gate. Success means slots attend to reasoning-bearing
  spans rather than boilerplate and do not degrade the target floor.
- Day 3: add a byte-probe or token-family readout diagnostic so tokenizer
  mismatch is measurable independently of exact GSM answer extraction.
- Days 4-5: run the best two diagnostics on GSM30 and compare against C2C,
  KVComm replay, target-alone, text-to-text, and KVPress no-press /
  ExpectedAttention.
- Week 2: stack only the diagnostics that pass an interpretable gate with
  `dynalign_prefdist`, then run larger controlled GSM/SVAMP slices.

Paper-readiness criterion:

- A method becomes the positive paper branch only if it beats target-alone on
  paired controlled GSM slices, improves or matches the bytes/accuracy frontier
  against C2C/KVComm where applicable, and has interpretable route/interface
  telemetry that explains where the gain comes from.

## 2026-04-20 Attention-Stratified Selector Result

I implemented the first route-collapse ablation as `attention_stratified`.
This is deliberately small: it keeps the same checkpoint and attention scores
as `dynalign_prefdist`, but selects communicated positions through four
prompt-region bins instead of a global top-k. The purpose is to test whether
the observed prefix-heavy selector collapse is fixable by coverage alone.

Results:

- `gsm8k_5`: `0.2000` at `686,026.6` average bytes
- controlled `gsm8k_eval_10`: `0.1000` at `681,668.4` average bytes
- paired controlled delta versus target-alone: `+0.0000`
- corrected controlled selector coverage: prefix `0.3612`, suffix `0.3292`,
  full trace fraction `1.0000`

Interpretation:

- coverage balancing works as an instrumentation/control change, but not as a
  positive method
- it loses the `dynalign_prefdist` smoke (`0.4000 -> 0.2000`) and still ties
  controlled target-alone
- this supports the subagent recommendation: implement deterministic
  query-pool transport next, with byte-probe diagnostics alongside it, and
  defer full `headwise_route_atom` until after we have a safer query-slot
  interface

## 2026-04-20 Query-Pool Transport And Quantization Inspiration

I implemented `query_pool_transport` as the lowest-risk version of the
query-pool idea: it keeps the Hugging Face cache length and `PrefixState`
contract unchanged, bins prefix positions, attention-pools translated K/V
inside each bin, and writes one pooled representative slot per bin. This is not
full cache compression or a learned Q-Former; it is a deterministic interface
diagnostic that tests whether pooled slots are less brittle than direct top-k
position replacement.

Results on the same `dynalign_prefdist` checkpoint:

- `gsm8k_5`: `0.2000` at `686,026.6` average bytes
- controlled `gsm8k_eval_10`: `0.1000` at `681,668.4` average bytes
- paired controlled delta versus target-alone: `+0.0000`
- trace now logs `query_pool_slots`, `query_pool_weight_entropy_mean`,
  `query_pool_top_weight_mean`, `query_pool_mean_bin_span`, and full
  representative positions for small runs

Interpretation:

- deterministic pooling is interpretable and preserves cache invariants, but
  it is not a positive method
- it ties the controlled target floor and loses the `dynalign_prefdist` smoke,
  matching the attention-stratified diagnostic pattern
- the next query-pool branch must be learned or query-conditioned in the
  transport/fusion path, not only a fixed pooling wrapper around the same
  selector

The new quantization/compression reference sweep points to three concrete
method branches worth testing next:

- SmoothQuant/AWQ-style bridge preconditioning: migrate outlier mass with
  diagonal scaling and allocate capacity to task-critical channels
- QuaRot/SpinQuant-style gauge fixing: use orthogonal or Hadamard rotations to
  reduce coordinate outliers before bridge fitting
- EXL2/KV-cache-style heterogeneous budgeting: allocate bits or route atoms by
  layer/head sensitivity, not uniformly across the bridge

The paper should add these as planned ablations only if they come with the
interpretability columns now supported by the artifacts: per-layer error,
route/slot entropy, dead atom rate, paired flips, bytes, and tokenizer-family
failure modes.

## 2026-04-21 Toy Query-Pool Benchmark

I added a standalone synthetic benchmark for the top-k versus query-pool
interface question. It is intentionally separate from the real-model evaluator:
it samples latent states, source K/V slots, and target queries; trains matched
readouts for `topk` and `query_pool`; then evaluates aligned, rotated,
outlier-scaled, and slot-permuted stress settings at the same budget.

Run artifact:

- `results/query_pool_toy_20260421/query_pool_vs_topk.json`
- `results/query_pool_toy_20260421/query_pool_vs_topk.md`

Task accuracy at budget `4`:

- aligned: `topk 0.2396`, `query_pool 0.3125`
- rotated: `topk 0.2760`, `query_pool 0.3125`
- outlier: `topk 0.2448`, `query_pool 0.2917`
- slot-permuted: `topk 0.2604`, `query_pool 0.3594`

Interpretation:

- the toy supports trying a learned query-pool or route-atom interface because
  pooled queries beat direct top-k under every synthetic stress condition
- the query-pool branch usually has worse reconstruction MSE, so a real-model
  version must include reconstruction/consistency telemetry and not optimize
  only answer accuracy
- this upgrades query-pool from a fixed-selector diagnostic to a plausible
  learned-interface branch, but it is not paper evidence until it beats
  target-alone on paired controlled model runs

Next implementation target:

- run the new evaluator-side `headwise_route_atom` metric on the controlled
  Qwen `dynalign_prefdist` branch, because it is now implemented without
  changing cache format or translator checkpoints
- keep learned query-pool as the higher-priority positive-method candidate if
  the real route-atom smoke only improves telemetry; the toy benchmark now
  shows query-pool is stronger on task accuracy while route atoms provide
  better collapse diagnostics
- then stack only the useful part: learned/query-conditioned pooled transport
  with route-atom entropy, dead-atom, collision, reconstruction, paired-flip,
  and bytes telemetry

## 2026-04-21 Route-Atom Toy And Evaluator Hook

I added a matched-budget `route_atom` baseline to the synthetic query-pool
benchmark and added `headwise_route_atom` as a real evaluator head-selection
and head-gate metric. The evaluator path is intentionally conservative: heads
are treated as atoms, scored by sharpness, layer-mean divergence, and
orientation over prefix positions, then routed through the existing runtime
head-selection machinery. It does not introduce a new cache representation.

Run artifact:

- `results/query_pool_toy_20260421/query_pool_route_atom_vs_topk.json`
- `results/query_pool_toy_20260421/query_pool_route_atom_vs_topk.md`

Task accuracy at budget `4`:

- aligned: `topk 0.2396`, `query_pool 0.3385`, `route_atom 0.2969`
- rotated: `topk 0.2760`, `query_pool 0.3750`, `route_atom 0.3698`
- outlier: `topk 0.2448`, `query_pool 0.2552`, `route_atom 0.2448`
- slot-permuted: `topk 0.2604`, `query_pool 0.2969`, `route_atom 0.2917`

Interpretation:

- route atoms are competitive on rotated and slot-permuted geometry but do not
  beat learned query-pool on this toy suite
- the atom telemetry is valuable even when the method is not a headline win:
  atom entropy, atom collision, dead atoms, route entropy, and route top-margin
  are now in the artifact
- the next real-model smoke should test `headwise_route_atom` as an
  evaluator-side control, not as the main positive claim
- the stronger paper branch remains a learned query-conditioned interface that
  uses route-atom telemetry to avoid collapse

First real-model smoke:

- checkpoint:
  `checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`
- protocol: GSM5, sparse `K-only`, `position_selection_metric=attention`,
  `position_selection_ratio=0.5`, `runtime_head_selection_ratio=0.5`,
  `runtime_head_selection_metric=headwise_route_atom`, fixed gate `0.10`
- result: `0.2000` at `346,263.4` average bytes
- reference: same checkpoint without runtime head pruning was `0.4000` at
  `686,026.6` average bytes

Conclusion:

- this is a bytes/interpretability branch, not a positive accuracy branch
- do not scale it directly until a ratio sweep or learned query-pool hybrid
  shows it preserves the `0.4000` smoke while reducing bytes
- keep it in the paper plan as a diagnostic/control because it now provides
  route-atom collapse and orientation telemetry in the result sidecar

Follow-up ratio sweep:

- `headwise_route_atom` ratio `0.25`: `0.4000` at `176,381.8` average bytes
- `headwise_route_atom` ratio `0.50`: `0.2000` at `346,263.4` average bytes
- `headwise_route_atom` ratio `0.75`: `0.2000` at `516,145.0` average bytes
- dense-head reference: `0.4000` at `686,026.6` average bytes
- controlled GSM10 ratio `0.25`: `0.1000` at `175,249.2` average bytes
- controlled GSM10 paired delta vs dense-head `dynalign_prefdist`: `+0.0000`

Conclusion:

- the `0.25` setting is now the best route-atom lead: it preserves the GSM5
  smoke and cuts bytes by roughly `3.9x`
- the effect is non-monotonic, so the paper hypothesis should be "selective
  sharp-head transport avoids interference", not "more heads are better"
- controlled GSM10 ties dense-head `dynalign_prefdist`, so this is a
  compression-frontier result rather than a positive accuracy result
- the next positive-method attempt should stack this selective sharp-head
  transport with a learned query-conditioned interface rather than scaling the
  selector alone

## 2026-04-21 Quantization-Preconditioned Toy Branch

I added `preconditioned_query_pool`, a toy query-pool variant that applies a
learned diagonal preconditioner before routing. This is the smallest
SmoothQuant/AWQ-style experiment we can run without changing the real
translator. It logs condition proxy, cosine drift, norm ratio, and absolute
scale ratio.

Task accuracy at budget `4`:

- aligned: `query_pool 0.3177`, `preconditioned_query_pool 0.3958`
- rotated: `query_pool 0.2969`, `preconditioned_query_pool 0.2656`
- outlier: `query_pool 0.3594`, `preconditioned_query_pool 0.2240`
- slot-permuted: `query_pool 0.3281`, `preconditioned_query_pool 0.2760`

Interpretation:

- diagonal preconditioning is not robust enough as-is
- it sharpens routing and helps aligned data, but it degrades rotated,
  outlier, and slot-permuted stresses
- if we add this family to the real method, prefer constrained orthogonal
  rotations, Hadamard/gauge fixing, or asymmetric K/V budgets over a free
  diagonal scale

## 2026-04-21 Control-Suite Runtime Head Routing

I added runtime head-routing fields to `EvalSpec` and `build_evaluate_cmd`:

- `runtime_head_selection_ratio`
- `runtime_head_selection_metric`
- `runtime_head_gate_metric`
- `runtime_head_gate_strength`

This turns route-atom/head-budget sweeps into reproducible control-suite runs
instead of one-off evaluator commands. Use this before scaling to GSM30/GSM70.

## 2026-04-21 Constrained Preconditioning And GSM30 Route-Atom Check

I added `constrained_preconditioned_query_pool` to the toy suite as the
bounded quantization-inspired control requested by the latest research sweep.
It uses a near-identity diagonal scale,
`scale = 1 + 0.25 * tanh(raw_scale)`, so it tests preconditioning without
allowing the free diagonal scale to become the whole solution.

Run artifact:

- `results/query_pool_toy_20260421/query_pool_constrained_preconditioned_vs_topk.json`
- `results/query_pool_toy_20260421/query_pool_constrained_preconditioned_vs_topk.md`

Task accuracy at budget `4`:

- aligned: `query_pool 0.3177`, free preconditioned `0.3958`,
  constrained preconditioned `0.3594`
- rotated: `query_pool 0.2969`, free preconditioned `0.2656`,
  constrained preconditioned `0.3906`
- outlier: `query_pool 0.3594`, free preconditioned `0.2240`,
  constrained preconditioned `0.2812`
- slot-permuted: `query_pool 0.3281`, free preconditioned `0.2760`,
  constrained preconditioned `0.3177`

Interpretation:

- bounded preconditioning is better behaved than the free diagonal branch and
  is especially strong under rotation stress
- it still does not dominate query-pool or route atoms across all stresses, so
  it should be an ablation/control, not a main method
- if this family moves into the real evaluator, prioritize constrained
  rotations, Hadamard/gauge-style transforms, or asymmetric K/V budgeting
  instead of free diagonal scaling

I also scaled the best route-atom ratio (`0.25`) to a paired GSM30 check:

- `target_alone`: `0.0667`
- `headwise_route_atom=0.25`: `0.0333` at `172,643.3` average bytes
- paired delta versus target-alone: `-0.0333`
- method-only wins: `0`
- target-only wins: `1`
- both correct: `1`
- both wrong: `28`

Interpretation:

- route atoms alone should **not** be scaled as the next positive-method lane
- the useful result is still compression/telemetry: `0.25` heads give a clear
  budgeted control surface with route entropy, score gap, sharpness, JS
  divergence, and orientation-span diagnostics
- the next method branch should stack route-atom diagnostics with a learned
  query-conditioned interface, not report route atoms as a standalone win

## 2026-04-21 Reference Expansion

I added two reference memos for the next ablation cycle:

- `references/307_multimodal_resampler_interface_refs.md`
- `references/307_diffusion_iterative_refinement_refs.md`

How they change the plan:

- Multimodal resamplers support a fixed latent-bank interface: Q-Former /
  Perceiver-style queries, explicit latent occupancy, entropy regularization,
  and selection-versus-projection separation.
- Diffusion/refinement work supports confidence-adaptive updates: only refine
  high-uncertainty heads/slots, log step-wise KL or drift, and keep a no-op
  refinement control so "extra compute" is not confused with better transport.
- Quantization work continues to support bounded gauge/preconditioning and
  asymmetric K/V budgets, but the toy results say free diagonal scaling is too
  brittle.

Paper decision:

- Do not add a new main-method component yet.
- Add the new references and toy results as ablations / motivation only.
- The next real positive-method attempt should be a learned query-conditioned
  transport/interface with explicit route-atom and preconditioning telemetry,
  evaluated first against target-alone on controlled GSM10/GSM30 before any
  larger benchmark spend.

## 2026-04-21 Asymmetric K/V Budget Toy Branch

I added `asymmetric_kv_budget`, a quantization-inspired toy branch that
separates route selection from value retention. This is motivated by K/V-cache
quantization and compression work where keys and values have different error
profiles and should not always share the same budget.

Matched total budget `4`:

| Scenario | Top-k | Query-pool | Route atom | Constrained precond. | Asym K/V 1+3 | Asym K/V 2+2 |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 0.2396 | 0.3177 | 0.3281 | 0.3594 | 0.4844 | 0.4167 |
| rotated | 0.2760 | 0.2969 | 0.3073 | 0.3906 | 0.4219 | 0.4688 |
| outlier | 0.2448 | 0.3594 | 0.3125 | 0.2812 | 0.4219 | 0.4167 |
| slot-permuted | 0.2604 | 0.3281 | 0.3542 | 0.3177 | 0.4427 | 0.3906 |

Run artifacts:

- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route1_value3.json`
- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route1_value3.md`
- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route2_value2.json`
- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_route2_value2.md`
- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_vs_topk.json`
- `results/query_pool_toy_20260421/query_pool_asymmetric_kv_budget_vs_topk.md`
- `results/query_pool_toy_20260421/asymmetric_kv_budget_summary.md`

Interpretation:

- this is the cleanest new positive toy signal: both matched-budget K/V splits
  beat the prior controls across all four stress cases
- `1+3` is strongest for aligned, outlier, and slot-permuted settings; `2+2`
  is strongest for rotated settings
- low route/value overlap and Jaccard, high KL, and low cosine indicate that
  route selection and value preservation are genuinely selecting different
  slots
- this should not be added as a paper method yet because it is toy-only, but it
  should become the next real-model ablation

Updated next branch:

1. Add a real evaluator/control-suite asymmetric K/V budget mode with separate
   K-route and V-value retention ratios.
2. Log separate K/V distortion or attention-fidelity metrics, route/value
   overlap, Jaccard, KL, cosine, and bytes.
3. Compare against equal-byte uniform, shuffled, and target-alone controls.
4. Only then stack the best asymmetric K/V split with `headwise_route_atom=0.25`
   and the learned query-conditioned interface lane.

## 2026-04-21 Reference Expansion

New memos:

- `references/308_kv_compression_competitor_refs.md`
- `references/308_symmetry_geometry_alignment_refs.md`
- `references/308_quantization_asymmetry_refs.md`

Paper decision:

- Keep `C2C` and `KVComm` as direct cross-model communication competitors.
- Use Quest, KVzip, H2O, StreamingLLM, SnapKV, PyramidKV, AdaKV, rKV, and
  DeltaKV as matched-byte compression/retention controls, not semantic-transfer
  competitors.
- Add symmetry/gauge telemetry to every serious branch: orientation span,
  singular-value entropy, CKA/SVCCA-style layer similarity where feasible,
  residual norm ratio, and expert/head collapse metrics.
- Add quantization-inspired controls as ablations: bounded preconditioning,
  orthogonal/Hadamard gauge fixing, outlier-protected routing, and mixed-byte
  K/V allocation.

## 2026-04-21 Execution Pass

Implemented next concrete pieces:

- Real evaluator asymmetric K/V support: `--kv-route-selection-ratio` and
  `--kv-value-selection-ratio` now build separate K and V position masks,
  preserve byte accounting, and log route/value overlap, Jaccard, entropy,
  gap, and selected-count telemetry.
- Control-suite coverage: added
  `fused_quant_asym_kv_attention_sparse_brief` so this branch is available in
  the standard sweep rather than only by ad hoc CLI.
- Toy `codebook_remap`: a learned low-cardinality codebook/token-remap branch
  with occupancy, collision, dead-code, reconstruction, support, and remap
  stability telemetry.
- Competitor bootstrap: added a C2C artifact resolution file and a benchmark
  memo mapping direct peers (`C2C`, `KVComm`) separately from matched-byte KV
  compression controls (`Quest`, `KVzip`, `kvpress`, `DeltaKV_sparse_vllm`).
- Reference expansion: added a recent architecture memo emphasizing
  tokenizer/vocab bridges, multimodal latent-bank interfaces, iterative
  refinement, and explicit symmetry handling.

Current toy evidence:

| Scenario | Top-k | Best older control | Asym K/V 2+4 | Codebook remap |
|---|---:|---:|---:|---:|
| aligned | 0.2396 | 0.3958 | 0.4062 | 0.4948 |
| rotated | 0.2760 | 0.3906 | 0.3906 | 0.5781 |
| outlier | 0.2448 | 0.3594 | 0.4792 | 0.5312 |
| slot-permuted | 0.2604 | 0.3542 | 0.4010 | 0.4688 |

Updated paper stance:

- Additively add **ablations and telemetry**, not another main method claim yet.
- The two highest-value experimental lanes are now asymmetric K/V retention and
  learned codebook/token interfaces.
- Keep older ideas alive only where they become controls or diagnostics:
  route atoms for collapse/occupancy, bounded preconditioning for gauge
  stability, diffusion/refinement for one-shot-vs-iterative repair, and
  tokenizer/vocab adaptation for interface mismatch.
- A positive-method claim needs a real-model win over target-alone, uniform,
  shuffled, and C2C/KVComm where runnable; toy-only wins are not enough.

Immediate next run ladder:

1. Run `fused_quant_asym_kv_attention_sparse_brief` on the exact Qwen GSM10
   smoke with matched target-alone, uniform sparse, and shuffled selector
   controls.
2. If nonzero, expand to GSM30/GSM70 and log sidecar route/value overlap,
   Jaccard, entropy, bytes, and per-example correctness deltas.
3. Implement the real-model codebook/token bridge as a small learned interface:
   start with a frozen learned codebook over translated K/V slots, then test a
   tokenizer-aware remap only if occupancy stays healthy.
4. Schedule competitor baselines in order: C2C Qwen pair first, KVPress
   expected-attention matched-byte control second, KVComm only after isolating
   or preserving the locally dirty compatibility patch.

## 2026-04-21 Separable K/V And Residual-Codebook Update

What changed:

- Added metric-separated asymmetric K/V selection:
  `--kv-route-selection-metric` and `--kv-value-selection-metric`.
- The current control-suite asym K/V branch now tests route-by-attention and
  value-by-energy instead of using a single score for both masks.
- Added toy `residual_codebook_remap`: a base codebook remap plus a second
  residual codebook and learned residual gate.
- Added 310-series reference and competitor execution memos, including recent
  2025/2026 architecture, token/vocab, diffusion/refinement, symmetry, and
  quantization inspirations.

Evidence:

| Result | Interpretation |
|---|---|
| GSM5 route-attention/value-energy: target 0.20, method 0.40 | Candidate real-model positive signal. |
| GSM5 random/random: target 0.20, method 0.20 | Gain is not explained by bytes alone on this slice. |
| GSM10 route-attention/value-energy: target 0.10, method 0.10 | Signal does not yet scale to the first 10-example slice. |
| Toy residual codebook wins all four stress cases | Strongest current toy lead for tokenizer/codebook bridging. |

Current hypothesis:

The separable K/V branch sometimes fixes local routing failures, but it is not
yet robust enough to be the paper method. The toy residual-codebook result
suggests the missing component is a discrete/shared interface that absorbs
tokenizer and gauge mismatch before K/V routing is applied. The likely positive
method stack is therefore:

1. Gauge-aware bridge initialization or calibration.
2. Residual codebook/token bridge to reduce tokenizer/interface mismatch.
3. Separable K/V budgets and metrics for route vs value retention.
4. Optional route-atom/head gating only after the first three components pass
   random/shuffled/equal-budget controls.

Next execution targets:

1. Run GSM30 for route-attention/value-energy, plus random/random and
   equal-metric attention/attention controls.
2. Add per-example stratification to the sidecar: tokenizer overlap,
   source/target token counts, selector entropy, route/value overlap, and
   method-only/baseline-only tags.
3. Promote residual-codebook remap from toy to a small frozen-prefix or K/V-slot
   bridge. First target is a non-generative reconstruction/fidelity diagnostic;
   only then run GSM.
4. Keep C2C as the first direct competitor and kvpress/KVzip/Quest as
   matched-byte compression controls, using `references/310_competitor_execution_plan.md`.

## 2026-04-21 Paired-Telemetry And Protected-Channel Update

Executed next targets:

- Added per-example paired sidecar telemetry for every non-target method
  against `target_alone`.
- Ran GSM30 route-attention/value-energy at gate `0.10`.
- Added toy `protected_channel_residual_codebook_remap`.
- Added 311/312/313 reference memos covering lateral cross-model
  communication, competitor smoke execution, quantization-inspired ablations,
  and tokenizer/latent-interface ablations.

New evidence:

| Evidence | Outcome | Paper implication |
|---|---|---|
| GSM30 route-attention/value-energy | target `0.0667`, method `0.0667`, delta `0.0000` | Separable K/V is not yet a positive method at GSM30. |
| Paired flips | `1` method-only, `1` baseline-only, `1` both-correct, `27` both-wrong | Neutral aggregate is now interpretable instead of opaque. |
| Protected-channel toy | improves aligned/outlier, hurts rotated/slot-permuted | Outlier protection needs gauge/permutation awareness. |
| Quantization references | AWQ/EXL2/TurboQuant/InnerQ suggest asymmetric budgets, outlier channels, angle/polar bases, residual correction | These are ablation mechanisms, not claims. |
| Tokenizer/latent-interface references | byte bridge, token remap, latent relay, iterative refinement, zero-init adapters | These become the next stacked bridge candidates. |

Updated blocker decomposition:

1. **Gauge/orientation mismatch:** protected identity channels fail under
   rotation and slot permutation, so any protected path must be learned after
   alignment or made invariant/equivariant.
2. **Interface discreteness:** residual codebooks remain the strongest toy
   path, suggesting the real bridge needs a compact shared interface rather
   than pure continuous K/V projection.
3. **Routing/value asymmetry:** route-attention/value-energy can move examples
   but does not scale alone; keep it as a stack component and measure
   route/value overlap and Jaccard on every run.
4. **Byte/runtime economics:** current GSM30 method costs about `1.41 MB` per
   example and slightly slower decoding than target-alone, so matched-byte and
   matched-latency controls are mandatory.
5. **Evaluation interpretability:** sidecars now have enough per-example
   evidence to stratify wins/losses by prompt length, selector entropy,
   token-count ratio, and route/value overlap.

Revised next ladder:

1. Run GSM30 matched controls: random/random and attention/attention at the
   same K/V ratios, using the new paired telemetry.
2. Promote residual codebook from toy to a frozen K/V-slot diagnostic with
   reconstruction, route/value overlap, and no-generation fidelity metrics.
3. Add gauge-aware protected-channel variants: protected after Procrustes,
   protected after learned orthogonal, and protected with learned/permuted
   channel masks.
4. Add tokenizer-interface variants: byte-level bridge, token remap, shared
   tokenizer bridge, and zero-init gated adapter, all first on reconstruction
   before GSM.
5. Run direct competitor smoke in the order from
   `references/311_competitor_smoke_matrix.md`: C2C, kvpress, KVzip, KVComm,
   then Quest where task-compatible.

## 2026-04-21 Control And Competitor Follow-Up

Executed from the revised ladder:

- GSM30 `random/random` and `attention/attention` matched controls.
- Gauge-aware protected-channel toy ablation.
- C2C native GSM5 smoke on the exact Qwen pair.
- KVPress none and expected-attention GSM5 smokes.
- New 314 references covering recent architecture inspirations and competitor
  bootstrap status.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM30 attention/energy | target `0.0667`, method `0.0667` | Neutral selector. |
| GSM30 attention/attention | target `0.0667`, method `0.1000` | Deterministic same-metric routing helps slightly. |
| GSM30 random/random | target `0.0667`, method `0.1333` | Strongest current real-model smoke; not selector-semantic. |
| Gauge-aware protected toy | improves slot-permuted vs fixed, fails rotated vs residual | Covariance gauge alignment is insufficient. |
| C2C GSM5 native | `0.0000` | Direct competitor runnable, negative smoke. |
| KVPress GSM5 | none `0.2000`, expected-attention `0.2000` | Same-model compression control runnable. |

Updated blocker decomposition:

1. **Selector semantics vs perturbation:** random/random beating
   attention/energy means we must isolate whether the improvement is useful
   stochastic perturbation, implicit ensemble behavior, or noise.
2. **Alignment basis:** PCA/gauge canonicalization is not enough. The protected
   subspace has to be selected by answer-relevant signal, not variance alone.
3. **Residual interface:** residual codebook remains the strongest toy method;
   it should be promoted before further deterministic selector tuning.
4. **Competitor readiness:** C2C and KVPress are runnable, so future paper
   tables can include real baselines rather than only planned comparisons.

Next execution ladder:

1. Repeat GSM30 random/random over at least `3` seeds and log flip overlap.
2. Add a signal-aware protected-channel toy: supervised Procrustes or learned
   orthogonal basis with an orthogonality penalty.
3. Promote residual codebook to a frozen K/V-slot diagnostic and measure
   reconstruction before generation.
4. Run KVPress GSM30 `none` and `expected_attention` as matched same-model
   controls.
5. If random/random seed repeats are stable, test a controlled stochastic
   ensemble variant: multiple random route/value masks with vote or confidence
   selection under a fixed byte/runtime budget.

## 2026-04-21 Stochastic Aggregation And Learned-Mask Follow-Up

Executed next:

- Aggregated the three GSM30 random route/value salt runs into paired
  per-example telemetry.
- Ran KVPress GSM30 `none` and `expected_attention` as same-model compression
  controls.
- Added a learned soft protected-channel residual-codebook toy method and
  compared it to fixed, gauge-aware, and signal-aware protected channels.
- Implemented a first non-oracle stochastic reranker over the existing GSM30
  salt012 candidate set.
- Added numeric-consistency/completion reranker diagnostics and a stricter
  target-fallback policy.
- Added multimodal/diffusion latent-connector references and stochastic
  verifier/reranker references.
- Added quantization/KV-communication references for outlier protection,
  K/V-asymmetric bit allocation, and rotation-before-transport ablations.
- Added tokenizer/vocab blocker references for byte/span canonical interfaces
  and vocab remapping before transport.
- Added symmetry/alignment references for permutation, gauge, CKA/SVCCA/GW,
  stitching, SAE, and shortcut-structure ablations.
- Added a quantization toy that separates outlier-channel protection from
  rotation-before-quantization.
- Added recent latent/multimodal reasoning references for continuous-thought
  loops, soft-token bridges, diffusion-style refinement, TokenPacker-style
  projectors, and latent cache communication.
- Added a tokenizer/vocab toy that compares token-ID relay, vocab remap, and
  byte/span canonical relay under matched synthetic arithmetic strings.
- Added a symmetry/orientation toy that separates permutation, orthogonal
  rotation, permutation-plus-rotation, and nonlinear/stitching failure modes.
- Ran C2C on the full GSM30 gate-search slice for the exact Qwen pair.
- Added verifier/test-time selection references and a target-model listwise
  verifier over the existing GSM30 stochastic candidate set.
- Added architecture inspiration references for MoD/MoR, recurrent memory,
  Mamba/hybrid schedules, and gated delta updates.
- Added a latent-refinement toy comparing one-shot bridge, iterative residual
  refinement, gated refinement, soft-token mixture, and coarse-to-fine query
  bank.
- Added a real tokenizer audit for the Qwen2.5/Qwen3 GSM30 slice.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Three-seed stochastic majority | `0.0333` vs target-alone `0.0667` | Raw voting is worse than target. |
| Three-seed target tie-break | `0.0333` | Target fallback does not rescue naive aggregation. |
| Three-seed stochastic oracle | `0.2667` | Random route masks contain useful answers. |
| Target-or-seed oracle | `0.3000` | Selection, not candidate generation, is now the core blocker. |
| Format-first reranker | `0.1333` vs target-alone `0.0667` | First positive non-oracle selection result. |
| Target-on-low-format reranker | `0.1333` | Conservative target fallback matches format-first. |
| Target-on-strict-format reranker | `0.1667` | Best current non-oracle selector; zero baseline-only losses. |
| Target-model listwise verifier | `0.0667`, fallback `0.0000`; selected target `30/30` | Naive self-verification collapses to target-alone; needs calibration/process rewards. |
| Numeric-consistency rerankers | `0.0667` | Useful telemetry, not sufficient as standalone selection. |
| KVPress GSM30 | none `0.0667`, expected-attention `0.0667` | Same-model compression control is neutral. |
| C2C GSM30 | `0.0667` on `gsm8k_gate_search_30.jsonl` | Direct peer is runnable on the exact pair, but this smoke ties target-alone and trails strict stochastic reranking. |
| Learned protected toy | wins outlier `0.6562`, near-recovers rotated `0.6302`, loses slot-permuted to signal-aware `0.5781` | Learned masks help, but need signal/orientation constraints. |
| Quantization toy | uniform MSE `0.2646`; protected-outlier MSE `0.0077`; Hadamard uniform MSE `0.0575` | Outlier protection and basis rotation are separable mechanisms. |
| Tokenizer toy | token-ID exact `0.0000`, vocab-remap `0.0677`, byte/span canonical `1.0000`, noisy byte/span `0.9010` | Canonical byte/span relay is the cleanest interface control before adding bridge capacity. |
| Real Qwen tokenizer audit | shared decoded-token rate `1.0000`, source/target fragmentation `0.2313/0.2313` | Tokenizer mismatch is not the active blocker for this exact Qwen pair/slice. |
| Symmetry toy | permutation-only wins pure permutation MSE `0.0014`; Procrustes wins rotations MSE `0.0018-0.0019`; ridge/stitch wins nonlinear MSE `0.0760` | Symmetry fixing and stitching should be evaluated as separate factors, not hidden inside one dense bridge. |
| Latent-refinement toy | one-shot acc `0.7604`; iterative residual `0.9792`; gated `0.9271`; coarse-to-fine `0.9062` | Recursive/gated refinement is a stronger next design than widening one-shot transport. |

Updated blocker decomposition:

1. **Candidate-selection blocker:** stochastic cross-model routes expose useful
   answers, and strict format-based fallback is now a stronger positive
   selector (`0.1667`), but it still recovers less than half of the oracle gap.
   The naive target-model listwise verifier selected target-alone on every
   example, so the next selector should be calibrated listwise/process scoring
   or confidence-gated sampling, not an uncalibrated "ask the target" prompt.
2. **Orientation-aware interface blocker:** learned masks and signal-aware
   masks each help different toy regimes. The likely stack is learned soft mask
   plus supervised signal alignment plus orientation/permutation regularization.
3. **Tokenizer/vocab interface blocker:** the toy now makes the failure mode
   explicit: token-ID relay collapses under mismatched segmentation, vocab
   remap barely helps, and byte/span canonical relay is lossless before noise.
   The real Qwen pair does not show this mismatch on GSM30, so tokenizer work
   should become a robustness/generalization control rather than the main
   explanation for the current Qwen failure.
4. **Symmetry/interface blocker:** the symmetry toy confirms that permutation,
   rotation, and nonlinear stitching are distinguishable regimes. Real model
   experiments should log which regime a bridge is solving before claiming a
   generic latent-transport gain.
5. **Compression math blocker:** quantization results suggest we must separate
   basis rotation, outlier-channel protection, and mixed K/V precision. A win
   from one should not be attributed to the others.
6. **Competitor baseline blocker:** KVPress GSM30 and C2C GSM30 are both
   neutral on the current small slice. C2C remains the main direct peer for
   GSM70/SVAMP paper tables, where it is still stronger than our older held-out
   branches.
7. **Latent reasoning architecture blocker:** continuous-thought, soft-token,
   diffusion-refinement, and multimodal projector papers all point to iterative
   latent refinement plus gated injection as a better next design than one-shot
   hard replacement. The latent toy supports this: iterative residual
   refinement is much stronger than one-shot transport under the same synthetic
   task.
8. **Interpretability blocker:** every stochastic run must log candidate set
   quality separately from selection quality: oracle correctness, vote entropy,
   vote margin, verifier score, selected seed, and paired flips.

Next execution ladder:

1. Replace the naive target-model verifier with calibrated selection:
   listwise-with-position-randomization, process-condition checks, confidence
   weighted voting, and a held-out threshold sweep against strict format
   fallback.
2. Add confidence-gated route expansion: target-alone or strict-reranker
   confidence decides whether to spend 1, 3, or 5 stochastic routes.
3. Add signal-regularized learned masks to the toy suite and sweep protected
   channels.
4. Promote the best learned/signal protected-channel diagnostic into a frozen
   K/V-slot reconstruction experiment before generation.
5. Promote tokenizer/interface controls to a robustness suite across genuinely
   mismatched tokenizers; keep Qwen2.5/Qwen3 tokenizer audit as the matched
   control.
6. Promote symmetry controls from the toy to real activations:
   identity, permutation-only, orthogonal-only,
   permutation-plus-orthogonal, gauge-fixed head bases, and stitching loss vs
   reconstruction loss.
7. Promote latent-refinement controls from toy to bridge experiments:
   iterative residual correction, gated latent injection, recurrent
   continuous-thought loop, and TokenPacker-style query bank at matched byte
   budget.
8. Add quantization-inspired bridge controls: outlier-channel protection,
   rotation-before-transport, and K/V-asymmetric mixed precision at matched
   bytes.
9. Keep KVPress as a same-model compression control and scale C2C/KVComm only
   when the run is exact-split, exact-pair, and parser-matched.

## 2026-04-21 Held-Out Attribution And Protected-Basis Quant Follow-Up

Executed next:

- Ran the full GSM70 and SVAMP70 held-out process-repair route pools with
  same-prompt attribution controls: selected-route no-repair and target
  self-repair.
- Added geometry/symmetry alignment references in
  `references/341_geometry_symmetry_alignment_refs.md`.
- Added competitor benchmark bootstrap references and local clone snapshots in
  `references/342_competitor_benchmark_bootstrap_refs.md`.
- Added toy `protected_basis_quant_bridge`, separating uniform low-bit
  transport, protected salient channels, incoherent basis preprocessing, and
  mixed-bit allocation under a near-matched byte band.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 selected-route no-repair | `0.1286` | Candidate route quality is better than target-alone but not enough by itself. |
| GSM70 target self-repair | `0.1714` | Target-side self-correction explains most of the raw repair gain. |
| GSM70 selected-route repair | `0.2000`, harm `0.0000` | Survives target self-repair control with a modest `+0.0286` route-specific margin. |
| SVAMP70 selected-route no-repair | `0.3571` | Candidate route quality is useful but below repair controls. |
| SVAMP70 target self-repair | `0.5000` | Self-repair is again the strongest attribution control. |
| SVAMP70 selected-route repair | `0.5429`, harm `0.0000` | Survives target self-repair control with a modest `+0.0429` route-specific margin. |
| Protected-basis quant toy | uniform `0.9740`, protected channels `0.9792`, incoherent preprocessing `0.9948`, mixed-bit `1.0000` | Quantization-inspired wins come from basis choice and bit allocation, not just channel protection. |

Updated paper read:

The positive-method candidate is alive after the strongest immediate
target-side fairness control, but it is not yet an efficiency claim. The current
paper story should be: stochastic cross-model routes expose useful candidates;
strict route selection plus target-side process repair turns those candidates
into held-out gains; same-prompt target self-repair explains most of the gain;
the remaining route-specific increment is consistent but too small to be the
final headline without stronger matched-budget evidence.

Next execution ladder:

1. Add test-before-repair and step-level verifier controls on GSM70/SVAMP70 to
   increase the route-specific margin, not just target self-correction.
2. Convert the protected-basis quant toy into a frozen K/V-slot diagnostic:
   uniform low-bit, protected channels, incoherent rotation, mixed K/V bits,
   and protected-after-alignment.
3. Add geometry telemetry to every bridge/repair run: CKA/SVCCA-style
   similarity, Procrustes residual, singular spectrum, route trajectory
   curvature, and selection entropy.
4. Run exact-split competitor comparisons against `C2C`, `KVComm`, `KVPress`,
   `KVzip`, `Quest`, `H2O`, and `SnapKV` only when prompt, token, repair,
   byte, and latency budgets are logged.
5. Treat tokenizer/vocab work as a robustness axis for mismatched-tokenizer
   model pairs; for the current Qwen2.5/Qwen3 pair, focus on selection,
   repair attribution, orientation, and K/V budget structure.

## 2026-04-21 Attribution CI And K/V Mixed-Precision Follow-Up

Executed next:

- Added `scripts/analyze_process_repair_attribution.py` so process-repair
  telemetry can be regenerated as JSON and Markdown with deterministic
  bootstrap intervals.
- Added `paper/competitor_benchmark_readout_20260421.md` to consolidate direct
  competitor and same-model compression artifacts into one auditable readout.
- Added recent latent/multimodal/refinement references in
  `references/343_recent_latent_multimodal_refinement_refs.md`.
- Added quantization/KV-compression math references in
  `references/344_quant_kv_compression_math_refs.md`.
- Added toy `kv_slot_mixed_precision_bridge`, which separates key-side route
  recovery from value-side answer reconstruction under low-bit transport.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 selected-route repair vs target self-repair | `+0.0286`, bootstrap CI `[-0.0429, 0.1143]` | Positive point estimate, but not yet a statistically robust route-specific margin. |
| SVAMP70 selected-route repair vs target self-repair | `+0.0429`, bootstrap CI `[0.0000, 0.1000]` | Stronger than GSM, still narrow; route-specific gains need amplification. |
| Competitor readout | C2C GSM70 `0.1286`, SVAMP70 `0.4429`; KVPress GSM30 none/expected-attention both `0.0667` | LatentWire repair beats C2C only with target-side repair compute; selected-route no-repair ties C2C on GSM70 and trails it on SVAMP70. |
| K/V slot mixed-precision toy | uniform answer `0.8672`; key-protected `0.9219`; value-protected `0.8672`; mixed `0.9219`; rotation `0.8906` | In this synthetic regime, K-side route preservation is the active lever; V-only protection is neutral. |
| Recent latent/refinement references | Coconut, SoftCoT, diffusion/refinement, multimodal projectors, activation/KV handoff | The next design stack should test recurrent latent refinement and soft-token/projector interfaces, not only one-shot adapters. |

Updated blocker decomposition:

1. **Route-specific attribution is positive but underpowered.** The method
   survives target self-repair controls, but the confidence intervals show that
   the current margin is too small for a final ICLR-ready method claim.
2. **K-side preservation is the live compression/math clue.** The K/V toy says
   protecting keys can improve route and answer accuracy, while protecting
   values alone does not help under the current synthetic setup.
3. **Competitor framing is clearer.** Direct cross-model peers (`C2C`,
   `KVComm`) must stay separate from same-model cache-compression controls
   (`KVPress`, `KVzip`, `Quest`, `H2O`, `SnapKV`).
4. **The next architecture bet should be iterative.** Recent latent-reasoning
   and diffusion/refinement work points toward recurrent latent updates,
   soft-token mixtures, blockwise denoising, and projector-style interfaces.

Next execution ladder:

1. Run a real K/V diagnostic that mirrors the toy: key-protected, value-
   protected, mixed K/V precision, and rotation-before-transport on frozen
   route pools before generation.
2. Add target-self comparison columns to every future process-repair summary,
   not just target-alone deltas.
3. Add a small recurrent latent-refinement bridge smoke: one-shot transport vs
   two-step residual refinement vs gated refinement at matched decode budget.
4. Add a test-before-repair selector on the held-out process-repair route pools
   so target repair is only spent when the candidate fails an explicit check.
5. Move the competitor readout into the final comparison harness once every row
   logs bytes, repair calls, generated tokens, and latency.

## 2026-04-21 Test-Before-Repair And Recurrent Refinement Follow-Up

Executed next:

- Added test-time verification / process reward references in
  `references/345_test_time_verification_process_reward_refs.md`.
- Added shared feature dictionary / interface references in
  `references/346_shared_feature_dictionary_interface_refs.md`.
- Added `scripts/analyze_test_before_repair_policy.py` to replay non-oracle
  test-before-repair gates over the held-out process-repair telemetry.
- Added toy `recurrent_latent_refinement_bridge`, comparing one-shot transport,
  two-step residual refinement, gated refinement, blockwise diffusion-style
  denoising, and an oracle upper bound.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 format-gated test-before-repair | accuracy `0.2000`, repair rate `0.7286`, saved repair `0.2714` | A cheap format gate preserves full repair accuracy while saving about a quarter of repair calls. |
| SVAMP70 format-delta test-before-repair | accuracy `0.5429`, repair rate `0.8429`, saved repair `0.1571` | Non-oracle gates can preserve full accuracy but save less compute on SVAMP. |
| SVAMP70 oracle precheck upper bound | accuracy `0.5429`, repair rate `0.6429`, saved repair `0.3571` | Better pre-repair tests could save substantial repair compute without losing accuracy. |
| Recurrent refinement toy | one-shot `0.6458`; gated `0.6562`; blockwise diffusion `0.7083`; naive two-step `0.5000` | Iterative refinement helps only when gated or blockwise; naive residual recurrence can improve MSE while hurting task accuracy. |
| Shared dictionary references | SAE, crosscoder, transcoder, stitching, ReFT, symmetry-aware merge | Shared feature bases are a plausible next interface, but need causal/task-level telemetry rather than reconstruction-only claims. |

Updated read:

Test-before-repair now looks like a paper-useful efficiency control, not yet a
route-quality amplifier. The current non-oracle gates can reduce repair calls
while preserving the repair-all result, but they do not increase accuracy over
repair-all. The recurrent-refinement toy gives a separate design clue: if we add
iteration, it should be gated or blockwise/refinement-style, not an ungated
residual loop optimized only for reconstruction.

Next execution ladder:

1. Promote the best test-before-repair gates into the main held-out summary and
   log repair-call savings next to accuracy.
2. Add a stronger pre-repair test: numeric consistency plus process-step
   verifier or generated test, then compare against the cheap format gate.
3. Promote recurrent refinement to a small bridge smoke: one-shot vs gated
   two-step vs blockwise refinement at matched bytes/latency.
4. Add a shared-feature diagnostic before training a full SAE bridge: CKA/SVCCA
   plus feature sparsity and matched-feature stability on existing route pools.
5. Keep any recurrent/shared-feature win tied to task-level accuracy; do not
   claim success from MSE or representation similarity alone.

## 2026-04-21 Repair-Gate Feature Audit

Executed next:

- Added `scripts/analyze_repair_gate_features.py` to audit which selected-route
  telemetry features predict safe repair skipping or missed repair-help cases.
- Generated `results/process_repair_holdout_20260421/repair_gate_feature_audit_20260421.md`
  and JSON sidecar.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 format score | selected-correct AUROC `0.8570`; best threshold `3.5000` preserves `0.2000` accuracy while saving `0.2714` repair calls | Format is the strongest cheap safe-skip signal on GSM70. |
| SVAMP70 format score | selected-correct AUROC `0.7880`; best threshold `4.5000` preserves `0.5429` accuracy while saving `0.1429` repair calls | Format also works on SVAMP, but with less budget savings. |
| SVAMP70 format delta | selected-correct AUROC `0.6649`; best threshold `3.0000` preserves `0.5429` accuracy while saving `0.1571` repair calls | Format delta is slightly more budget-efficient than raw format on SVAMP. |
| Numeric consistency fields | weak or inconsistent AUROC as safe gates | Numeric-count telemetry is not enough; the next precheck must be process/step-aware. |

Updated read:

The current cheap gate is interpretable: high format quality predicts selected
route correctness and lets us skip repair safely. It does not identify new
correct answers or amplify the route-specific margin. The next gate should be a
process-level verifier that targets the examples where selected route is wrong
but repair can help, rather than another numeric-count heuristic.

Next execution ladder:

1. Add a process-step verifier score or generated-test score to the selected
   route telemetry.
2. Compare `format_gate`, `process_gate`, and `format+process_gate` on the
   same held-out route pools.
3. Keep reporting missed-help count and repair-call savings, not just final
   accuracy.
4. Use feature AUROC and best-threshold repair savings as required telemetry
   for all future test-before-repair variants.

## 2026-04-21 Shared-Dictionary And Evaluation-Contract Follow-Up

Executed next:

- Added `references/347_modern_attention_memory_architecture_refs.md` to turn
  recent attention, memory, recurrence, and KV-sharing work into concrete
  LatentWire ablation axes.
- Added `references/348_iclr_method_framing_and_evaluation_refs.md` as the
  paper evaluation contract: paired tests, bootstrap CIs, compute ledgers,
  frozen prompt manifests, contamination controls, and model-card-style
  reporting.
- Added toy `shared_feature_dictionary_bridge`, separating raw residual
  transport, separate per-model dictionaries, a shared dictionary/crosscoder,
  symmetry-aware shared dictionaries, and an oracle upper bound.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Raw residual bridge toy | accuracy `0.3646`, MSE `1.1263` | Direct latent regression leaves large task and reconstruction gaps under private/shared feature mismatch. |
| Separate dictionaries | accuracy `0.4167`, MSE `0.2763` | Sparse feature coding helps reconstruction, but separate gauges underuse shared task features. |
| Shared dictionary crosscoder | accuracy `0.5417`, MSE `0.2642`, oracle `0.5938` | A learned shared feature interface is the strongest toy route and comes within `0.0521` accuracy of oracle. |
| Symmetry-aware shared dictionary | accuracy `0.4688`, MSE `0.3026` | Orthogonal gauge alignment is useful but not sufficient; it should be treated as a component, not the whole method. |
| Architecture reference sweep | SSM transport, test-time memory, retrieval memory, adaptive compute, KV sharing | The next positive-method search should vary communication architecture and compute policy, not only bridge loss. |
| Evaluation contract | paired CIs, compute ledger, frozen manifests, contamination checks | Any headline gain now needs paired significance and matched compute/repair accounting before it can become a paper claim. |

Updated read:

The most promising new additive idea is a shared sparse feature interface,
not another local latent-regression adapter. The toy says dictionary/crosscoder
structure can recover task signal that raw regression misses, but it also says
geometry alignment alone is not enough. For the real system, this should be
promoted as an interpretable diagnostic first: shared feature recovery,
dictionary alignment residual, sparsity, task accuracy, bytes, and compute all
need to be logged together.

Next execution ladder:

1. Promote shared-dictionary telemetry to existing route pools: CKA/SVCCA,
   Procrustes residual, matched-feature stability, sparsity, and task-level
   deltas.
2. Add a real shared-feature bridge smoke only after the diagnostic shows
   stable matched features; compare raw bridge, separate adapters, shared
   dictionary, symmetry-aware shared dictionary, and target self-repair.
3. Add architecture ablations in small controlled form: attention vs
   selective-SSM transport, writable memory vs sliding cache, adaptive compute
   vs fixed-depth bridge, and MQA/GQA-style KV sharing.
4. Upgrade every main held-out table with paired bootstrap CIs and compute
   ledgers before scaling more competitor comparisons.
5. Keep the paper claim positive-method but budget honest: show route-specific
   lift, repair-call savings, and exact cost against `C2C`, text-to-text,
   target self-repair, and no-repair selected routes.

## 2026-04-21 Process-Gate Text Audit

Executed next:

- Added `scripts/analyze_process_gate_features.py` to derive non-oracle
  process features directly from selected-route solution text: valid equation
  count, equation validity fraction, answer-marker presence, tail completeness,
  prediction/tail agreement, reasoning-step count, and a combined
  `format_plus_process_score`.
- Generated
  `results/process_repair_holdout_20260421/process_gate_feature_audit_20260421.md`
  and JSON sidecar on GSM70/SVAMP70 held-out repair telemetry.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 `format_plus_process_score` | selected-correct AUROC `0.8707`; preserves `0.2000` repair-all accuracy while saving `0.3286` repair calls | Text-derived process features improve the cheap GSM efficiency gate over format alone. |
| GSM70 equation features | valid-equation help AUROC `0.6231`; equation-valid-fraction help AUROC `0.6462` | Equations help identify repair-help opportunities, but not enough as standalone safe-skip gates. |
| SVAMP70 `process_completeness_score` | selected-correct AUROC `0.8307`; preserves `0.5429` while saving `0.2286` repair calls | Process-text features are more budget-efficient than the prior SVAMP format-delta gate. |
| SVAMP70 `format_plus_process_score` | selected-correct AUROC `0.9187`; saves `0.2143` repair calls | Combining surface format and process completeness is the best selected-correct predictor, but not the best saver on SVAMP. |

Updated read:

The next real gate should not be another pure metadata threshold. We now have
evidence that process-text structure improves safe repair skipping, while
equation-validity features are closer to identifying repair-help cases. This
still does not increase accuracy over repair-all, but it gives an interpretable
path to lower the target-side repair budget and a measurable target for a
learned/generated-test verifier.

Next execution ladder:

1. Add a stacked `format_or_process_gate` policy to the held-out
   test-before-repair replay and compare against format-only at identical
   missed-help constraints.
2. Turn equation-help features into a generated-test/process-verifier score:
   do not skip repair when equations are inconsistent, incomplete, or absent.
3. Log process-gate features in future telemetry at generation time so every
   row has selected correctness, repair help, missed help, and repair-call
   savings.
4. Treat any process-gate claim as an efficiency result until it raises
   accuracy over repair-all or increases the route-specific delta against
   target self-repair.

Follow-up replay with paired uncertainty and cost ledger:

| Policy | Accuracy CI | Repair saved | Avg extra repair chars | Avg extra repair tokens | Implication |
|---|---:|---:|---:|---:|---|
| GSM70 `format_plus_process_gate` | `[0.1143, 0.3000]` | `0.3286` | `676.9` | `42.2` | Best current non-oracle GSM repair-budget saver at unchanged accuracy. |
| GSM70 `format_gate` | `[0.1143, 0.3000]` | `0.2714` | `738.0` | `46.3` | Prior cheap gate is now dominated by the combined process gate on cost. |
| SVAMP70 `process_gate` | `[0.4286, 0.6571]` | `0.2286` | `661.1` | `49.3` | Best current non-oracle SVAMP repair-budget saver at unchanged accuracy. |
| SVAMP70 `format_delta_gate` | `[0.4143, 0.6571]` | `0.1571` | `713.9` | `53.5` | Prior SVAMP gate is now dominated on repair budget. |

## 2026-04-21 Route-Atom Codebook And Competitor-Gap Follow-Up

Executed next:

- Added `references/349_recent_projector_refinement_alignment_refs.md` with
  recent latent communication, multimodal projector, refinement, crosscoder,
  and representation-alignment ablation ideas.
- Added `references/350_competitor_benchmark_gap_plan.md` and bootstrapped a
  local `LLMLingua` clone under `references/repos/LLMLingua` as the prompt
  compression control lane.
- Added toy `route_atom_codebook_bridge`, inspired by quantization codebooks,
  route atoms, protected outliers, and route-conditioned atom banks.
- Updated `test_before_repair_policy_20260421` so format-only, process-only,
  and combined process gates are replayed in one held-out table.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Route-atom raw ridge | accuracy `0.7812`, MSE `0.0437` | Raw regression reconstructs well in this toy, but leaves task accuracy below codebook methods. |
| Learned shared codebook | accuracy `0.8438`, atom recovery `0.9231`, MSE `0.7225` | Task-relevant atom recovery can matter more than low MSE; reconstruction alone is not the right objective. |
| Protected outlier atoms | accuracy `0.8438`, MSE `0.2069` | Outlier protection preserves the codebook accuracy gain while reducing reconstruction damage. |
| Route-conditioned codebook | accuracy `0.7656`, compute proxy `12288.0` | Route conditioning cuts compute but needs a better router; lower compute alone is not enough. |
| Oracle route atoms | accuracy `1.0000` | The synthetic task has a large atom-recovery headroom if route identity can be preserved. |
| Competitor gap plan | adds `LatentMAS` and `LLMLingua` controls | The final benchmark suite should separate direct cross-model communication, same-model KV compression, latent multi-agent collaboration, and prompt compression. |

Updated read:

The two toy results now agree on a central paper constraint: a bridge can win
task accuracy while losing naive reconstruction metrics. The method search
should therefore optimize and report interpretable task-causal features:
shared-feature recovery, route-atom recovery, codebook entropy/perplexity,
repair-help/missed-help, and paired task deltas. The next real-system blocker is
not another static regression objective; it is preserving the right shared
features or route atoms under a strict byte/compute budget.

Next execution ladder:

1. Add codebook/atom telemetry to the real bridge diagnostics: atom entropy,
   assignment stability, protected-outlier rate, route-family entropy, and
   task delta vs reconstruction delta.
2. Promote a small real codebook smoke only if assignment stability is
   non-random on existing route pools; compare raw ridge, shared codebook,
   route-conditioned codebook, protected atoms, and target self-repair.
3. Add `LLMLingua` as a prompt-compression control for target-only runs before
   making communication-efficiency claims.
4. Build two separate competitor tables: direct heterogeneous communication
   (`C2C`, `KVComm`, `LatentMAS`, LatentWire) and same-model cache compression
   (`KVPress`, `KVzip`, `Quest`, `H2O`, `SnapKV`).
5. Keep the current positive-method paper path focused: route selection +
   process repair + test-before-repair savings + interpretable shared
   feature/atom diagnostics.

## 2026-04-21 Paired Gate CIs, Prompt Compression Control, And Feature-Atom Stack

Executed next:

- Upgraded `scripts/analyze_test_before_repair_policy.py` with paired bootstrap
  accuracy intervals and repair-cost proxies: repair call count, extra repair
  prompt chars, and extra repair generated tokens.
- Added `references/351_recent_verifier_agent_training_refs.md` for
  step-localized verification, verifier-guided frontier expansion, structured
  agent protocols, RL self-correction, debate, and causal evaluation.
- Added `references/352_llmlingua_prompt_compression_control.md` plus
  `scripts/analyze_prompt_compression_control.py`, a no-download
  LLMLingua-style lexical compression control for GSM70/SVAMP70 prompt budgets.
- Added toy `feature_atom_stack_bridge`, which stacks shared-feature
  dictionaries with route-atom codebooks and protected stacked variants.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 process gate with paired CI | accuracy `0.2000`, CI `[0.1143, 0.3000]`, repair saved `0.3286`, extra repair chars `676.9` | Best current GSM budget saver; still same accuracy as repair-all. |
| SVAMP70 process gate with paired CI | accuracy `0.5429`, CI `[0.4286, 0.6571]`, repair saved `0.2286`, extra repair chars `661.1` | Best current SVAMP budget saver; still same accuracy as repair-all. |
| LLMLingua-style lexical control | GSM70 number preservation `1.00`, bytes saved `123.5`; SVAMP70 number preservation `1.00`, bytes saved `71.5` | Prompt compression can be logged as a budget control without learned compressor downloads, but does not make accuracy claims. |
| Feature-atom stacked toy | raw `0.6458`; shared-only `0.4167`; atom-only `0.5833`; stacked `0.8542`; oracle `1.0000` | Single fixes can hurt, but stacked complementary interfaces can unlock a large task gain. |
| Protected stacked toy | protected stacked `0.8542` with slightly larger byte/compute proxy | Protection preserves the stack's task gain; not yet an efficiency win. |
| Verifier/agent references | ProcessBench, xVerify, A*-Decoding, protocol/debate/self-correction papers | The next route-quality amplifier should be step-localized verification or frontier pruning, not another scalar reranker. |

Updated read:

This is the strongest lateral evidence so far for the user's "multiple fixes
stacked together" hypothesis. In the stacked toy, either shared features or
route atoms alone underperform raw ridge, but the combined interface strongly
beats raw. That is a concrete warning against prematurely rejecting components
based on isolated ablations. For the real paper, the next method stack should
combine: strict route generation, process-aware test-before-repair, shared
feature/atom diagnostics, and a step-localized verifier.

Next execution ladder:

1. Add a held-out `format_plus_process_gate` method row to the main comparison
   table with CI, repair-call savings, extra prompt chars/tokens, and missed
   help.
2. Build a small step-localized verifier replay over existing selected-route
   text: scalar score vs first-error localization vs critique-plus-repair, all
   on the same candidate pool.
3. Promote feature+atom diagnostics to real route pools before training:
   shared-feature stability, atom assignment stability, entropy/perplexity,
   protected-outlier rate, and task delta vs MSE delta.
4. Add the LLMLingua lexical control as the cheap prompt-budget baseline; only
   run learned LLMLingua after we need a paper table row.
5. Keep the target ICLR narrative as a positive method stack, not a single
   adapter trick: route generation + process gate + target repair + shared
   feature/atom interface + matched-budget competitor table.

## 2026-04-21 Step-Localized Verifier, Tokenizer Interface, And Competitor Matrix

Executed next:

- Added `scripts/analyze_step_localized_verifier.py` to replay scalar metadata,
  critique-plus-repair, and step-localized repair gates over the held-out
  process-repair telemetry without new model calls.
- Added `references/353_tokenizer_byte_interface_refs.md` for byte-level
  bridges, cross-tokenizer distillation, vocabulary remapping, time-warped
  alignment, adaptive hypertokens, and retokenization controls.
- Added `references/354_competitor_next_runnable_matrix.md` to rank the next
  direct competitor rows: `C2C` GSM70/SVAMP70 first, `KVComm` GSM70 second,
  and `KVPress` same-model compression controls alongside them.
- Added toy `verifier_guided_atom_pruning`, comparing no pruning, scalar
  pruning, step-error-localized pruning, verifier-guided frontier pruning, and
  atom-oracle pruning with missed-help, false-prune, atom-recovery, byte, and
  compute proxies.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| GSM70 scalar metadata repair gate | accuracy `0.2000`, CI `[0.1143, 0.3000]`, repair saved `0.3714`, extra repair chars `641.2`, missed help `0` | Best current GSM repair-budget saver; step localization is not yet stronger on GSM. |
| SVAMP70 step-localized repair gate | accuracy `0.5429`, CI `[0.4286, 0.6571]`, repair saved `0.2286`, extra repair chars `661.1`, missed help `0` | Step localization is safe on SVAMP and matches the process-gate saving. |
| SVAMP70 aggressive scalar metadata gate | accuracy `0.5286`, repair saved `0.2714`, missed help `1` | Scalar gates need per-task calibration; more saved repair calls can silently lose accuracy. |
| Verifier-guided atom pruning toy | no-pruning `0.8047`; scalar pruning `0.5234`; step-localized pruning `0.9063`; verifier-guided frontier pruning `0.9609` at about half bytes/compute | Localized verifier signals can become a positive method component when used to prune harmful atoms rather than only rerank routes. |
| Tokenizer/byte reference sweep | BLT, byte-interface distillation, DWA-KD, CTPD, TokAlign, zip2zip, Length-MAX, FOCUS | The next tokenizer lane should be tested as an interface control: byte/patch bridge, explicit vocab remap, time-warped span alignment, adaptive compression, and length-optimal retokenization. |
| Competitor runnable matrix | C2C GSM70/SVAMP70, KVComm GSM70, KVPress none/expected-attention controls | The next benchmark batch should separate direct cross-model communication from same-model cache/prompt compression. |

Updated read:

The paper should add these things additively, but only as controlled method
components. The current strongest real-model method remains strict route
selection plus process repair, now with cheaper test-before-repair gates. The
new positive toy says verifier localization can be more than a budget gate if
it controls the communication frontier itself. The tokenizer references point
to an upstream alignment lane that is different from adapter and rotational
alignment work, but it needs a byte/token stress audit before it becomes a
large real-model run.

Next execution ladder:

1. Run the competitor batch from `references/354_competitor_next_runnable_matrix.md`:
   `C2C` on GSM70/SVAMP70, `KVComm` on GSM70, and `KVPress` none versus
   expected-attention on GSM70/SVAMP70.
2. Promote verifier-guided frontier pruning from toy to route/atom telemetry:
   atom recovery, missed-help, false-prune, bytes, compute, and task delta.
3. Add a generated-test or learned step verifier so GSM can use more than the
   scalar metadata gate while keeping SVAMP missed-help at zero.
4. Add tokenizer-interface controls before training a new bridge: byte/patch
   bridge, TokAlign-style remap, Soft-DTW/span alignment, adaptive hypertokens,
   and length-optimal retokenization.
5. Keep the ICLR claim stack narrow: route generation + process repair +
   test-before-repair + verifier-guided frontier control + feature/atom
   interface, all reported against `C2C`, `KVComm`, `KVPress`, text-to-text,
   target self-repair, and prompt-compression controls.

## 2026-04-21 Activation-Aware Quantization And Competitor Execution Check

Executed next:

- Added `references/355_quantization_compression_inspiration_refs.md`, covering
  AWQ, GPTQ, EXL2, SmoothQuant, QuaRot/SpinQuant, KIVI/KVQuant, BAQ, SpQR, and
  AQLM as concrete bridge-compression and atom-allocation inspirations.
- Added toy `activation_aware_atom_quant`, comparing full precision, uniform
  low-bit, random mixed precision, activation-aware mixed precision,
  protected-outlier mixed precision, and oracle mixed precision.
- Attempted the first full competitor batch rows from
  `references/354_competitor_next_runnable_matrix.md`; the status artifact is
  `results/competitor_next_runnable_20260421/competitor_batch_status_20260421.md`.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Uniform low-bit atom quantization | accuracy `0.9531`, bytes `16.0`, top-atom preservation `0.0000` | Low-bit compression is cheap but loses task accuracy and destroys salient atom tracking. |
| Random mixed precision | accuracy `0.9792`, bytes `29.0`, top-atom preservation `0.5000`, outlier protection `0.2500` | Mixed precision helps, but random allocation leaves avoidable harm. |
| Activation-aware mixed precision | accuracy `1.0000`, bytes `29.0`, top-atom preservation `1.0000`, outlier protection `1.0000` | AWQ/EXL2-style saliency allocation can preserve full-precision task behavior at less than half the full bytes in toy form. |
| Protected-outlier mixed precision | accuracy `1.0000`, bytes `29.0`, top-atom preservation `1.0000`, outlier protection `1.0000` | Outlier protection is not just a compression trick; it is a plausible route-atom preservation primitive. |
| Competitor full-row attempt | C2C reached model fetch completion but stalled in generation; KVPress reached MPS device setup but stalled | Competitor rows need explicit timeouts, `--limit` smokes, or CPU/GPU scheduling before we can fill the full benchmark table locally. |

Updated read:

The quantization analogy is now actionable: activation-aware bit allocation and
outlier protection should be stacked with the shared-feature/route-atom
interface rather than treated as a separate compression appendix. The method
hypothesis becomes: identify task-causal atoms/features, protect them at high
precision, compress the rest aggressively, then use process repair only when
the verifier says the transmitted frontier is unsafe.

Next execution ladder:

1. Add a real route-pool diagnostic for saliency and outlier atoms: activation
   energy, task-gradient or verifier saliency proxy, top-atom preservation,
   protected-byte share, and task delta.
2. Combine verifier-guided frontier pruning with activation-aware mixed-bit
   allocation in one toy stack before promoting either to real route pools.
3. Add rotation-before-compression controls: identity, Hadamard/random
   orthogonal, and learned rotation before mixed-bit atom quantization.
4. Rerun competitor rows with explicit `--limit` smokes and wall-clock
   timeouts before attempting full GSM70/SVAMP70 rows on MPS.
5. Keep the benchmark table honest: incomplete competitor rows are blockers,
   not hidden missing data.

## 2026-04-21 Limited Competitor Smokes And Verified Mixed-Precision Stack

Executed next:

- Added `--limit` support to `scripts/run_kvpress_eval.py` and
  `--calibration-limit` / `--eval-limit` support to `latent_bridge/kvcomm_eval.py`
  so competitor rows can be smoked before full GSM70/SVAMP70 runs.
- Completed paired KVPress GSM70 limit-1 smokes for `none` and
  `expected_attention` and recorded them in
  `results/competitor_next_runnable_20260421/competitor_batch_status_20260421.md`.
- Added `references/356_multimodal_diffusion_latent_interface_refs.md`, covering
  Q-Former/perceiver bottlenecks, simple projectors, routed connectors, soft
  belief-state refinement, trajectory-guided repair, latent-flow bridges, and
  expert fusion.
- Added toy `verified_mixed_precision_stack`, stacking verifier-guided pruning
  with low-bit and activation-aware quantization controls.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| KVPress GSM70 `none` limit-1 | completes in local smoke; accuracy `0.0000`, latency `2.1374s`, tokens/sec `1.8715` | Same-model competitor controls are runnable once split into bounded chunks. |
| KVPress GSM70 expected-attention limit-1 | completes in local smoke; accuracy `0.0000`, latency `2.1806s`, tokens/sec `2.7516` | The next fair step is limit-5 paired controls, not another full-row attempt. |
| Verified mixed-precision full precision | accuracy `0.9219`, bytes `772.0`, compute `768.0` | Baseline is strong but byte-heavy. |
| Prune then uniform quant | accuracy `0.9323`, bytes `118.0`, compute `730.3` | A verifier-prune plus simple low-bit compression stack can improve toy task behavior while cutting bytes sharply. |
| Prune then activation-aware quant | accuracy `0.8958`, bytes `184.0`, compute `730.3` | Activation-aware protection is not automatically additive after pruning; protected frontier selection can conflict with useful quantization noise. |
| Oracle stack | accuracy `0.9375`, bytes `184.0`, compute `726.5` | There is remaining headroom if protected atoms are selected correctly inside the pruned frontier. |
| Multimodal/diffusion refs | Q-Former, Perceiver Resampler, LLaVA projector, routed experts, diffusion refinement, flow matching | These suggest bottlenecked query connectors and iterative latent refinement as the next lateral architecture controls. |

Updated read:

The additive story is now sharper: some components stack, but not by default.
Verifier-guided pruning plus uniform quantization is a positive toy component;
activation-aware high-precision protection is positive alone but negative when
naively inserted after pruning. That is exactly the kind of interaction the
paper needs to report rather than hide. The next method stack should therefore
include interaction ablations, not just one-component deltas.

Next execution ladder:

1. Run KVPress GSM70 `none` and `expected_attention` at `--limit 5` with the
   same max-token budget, then scale only if both complete.
2. Add a protected-frontier selection ablation: global activation saliency,
   verifier saliency, quantization-error saliency, and oracle protected atoms.
3. Promote the positive compressed stack telemetry to real route pools:
   prune rate, missed-help, false-prune, bytes, compute, and task delta.
4. Add a Q-Former/perceiver-style bottleneck toy only after the route/atom
   stack has a real-pool diagnostic.
5. Keep all additive claims conditional on interaction tests: a component that
   helps alone can harm when inserted into the full stack.

## 2026-04-21 Protected Frontier Selection And Limit-5 Controls

Executed next:

- Added `references/357_frontier_attribution_routing_refs.md`, covering SAE /
  crosscoder selectors, attribution patching, causal tracing, sparse routing,
  uncertainty-aware frontier fallback, and saliency robustness checks.
- Added `references/358_recent_lateral_method_refs.md`, covering recent
  projector/adapters, diffusion-style refinement, cache controls,
  quantization geometry, tokenizer adaptation, and transport initialization.
- Added the deterministic `protected_frontier_selection` toy, which isolates
  the post-pruning question: after a verifier-pruned frontier is fixed, which
  atoms should receive high-precision protection?
- Added the deterministic `tokenizer_frontier_bridge` toy, which isolates
  tokenizer-boundary mismatch and compares naive token-id transfer, exact
  target-frontier regrouping, and a small learned remap table.
- Completed bounded KVPress limit-5 controls for GSM70 `none`,
  GSM70 `expected_attention`, SVAMP70 `none`, and SVAMP70
  `expected_attention` under the arm64 wrapper / CPU fallback.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Protected-frontier prune-uniform baseline | accuracy `0.6615`, MSE `0.2543`, bytes `89.6`, prune rate `0.3244` | The fixed pruned frontier is intentionally lossy under very low-bit quantization. |
| Global activation protection | accuracy `0.7917`, MSE `0.1905`, bytes `176.6`, protected-oracle overlap `0.4314` | Activation magnitude is a useful cheap selector once the frontier is fixed. |
| Quant-error protection | accuracy `0.8073`, MSE `0.1861`, bytes `176.6`, protected-oracle overlap `0.5434`, help `0.1458`, harm `0.0000` | Quantization-error saliency is the strongest cheap selector in this toy and ties the exact patch selector on task accuracy. |
| Exact patch-effect protection | accuracy `0.8073`, MSE `0.1698`, bytes `176.6`, patch-rank correlation `1.0000`, help `0.1458`, harm `0.0000` | Exact single-atom patch effect is the bounded compression oracle for this toy. |
| Utility-positive oracle protection | accuracy `0.7812`, MSE `0.2321`, bytes `176.6`, protected-oracle overlap `1.0000` | Utility-positive atom selection is not the same as compression-critical protection. |
| KVPress GSM70 `none` limit-5 | accuracy `0.2000`, latency `8.6682s`, tokens/sec `4.3839` | Same-model cache controls are runnable in small chunks and now have a nontrivial paired smoke. |
| KVPress GSM70 expected-attention `0.5` limit-5 | accuracy `0.2000`, latency `8.9888s`, tokens/sec `4.3832` | Expected-attention compression is not better than no compression on this tiny GSM smoke. |
| KVPress SVAMP70 `none` limit-5 | accuracy `0.4000`, latency `9.0091s`, tokens/sec `4.8396` | SVAMP bounded controls are runnable and now have a paired expected-attention row. |
| KVPress SVAMP70 expected-attention `0.5` limit-5 | accuracy `0.6000`, latency `7.9233s`, tokens/sec `5.0484` | Tiny positive smoke versus no-press on SVAMP, but far too small for a paper claim. |
| Tokenizer frontier naive token-id transfer | exact reconstruction `0.0000`, decoded-boundary F1 `0.3777`, source-target boundary F1 `0.7952`, bytes/example `19.23` | Token id compatibility is a false assumption when token boundaries differ. |
| Tokenizer learned remap | exact reconstruction `1.0000`, decoded-boundary F1 `1.0000`, bytes/example `11.74` | A small learned remap can beat target-frontier regrouping on bytes while preserving reconstruction in toy form. |

Updated read:

The protected-frontier result is now a real positive toy component: the
selector matters, and a compression-native score (`quant_error_protect`) beats
or matches the more semantic-looking selectors without adding harm relative to
the prune-uniform baseline. This supports the additive paper hypothesis:
route/atom interfaces should log protected-frontier selection quality directly
instead of treating quantization as a post-hoc byte setting.

The competitor path is also clearer. KVPress can be run locally if rows are
chunked; full monolithic rows remain an execution risk. Direct communication
baselines (`C2C`, `KVComm`, `LatentMAS`) should stay separate from same-model
cache controls (`KVPress`, `KVzip`, `Quest`, prompt compression), but both need
the same timing and byte ledgers.

Next execution ladder:

1. Scale KVPress GSM/SVAMP to `--limit 10` and `--limit 20` before full rows,
   keeping CPU fallback available when MPS setup fails.
2. Promote protected-frontier telemetry to real route pools: protected ids,
   quant-error score, activation score, attribution score, missed-help,
   false-prune, protected-oracle proxy, bytes, latency, and task delta.
3. Promote tokenizer-frontier telemetry to a real byte/token stress set:
   source-target boundary F1, remap coverage, exact reconstruction, byte budget,
   and downstream route delta.
4. Add an exact-ablation calibration slice for protected atoms so attribution /
   SAE / crosscoder selectors can be compared to real patch effect, not only
   saliency proxies.
5. Stack the best synthetic pieces cautiously: shared feature/atom interface,
   verifier pruning, quant-error protection, and test-before-repair, with full
   interaction ablations.
6. Keep lateral architecture work as controlled ablations: routed projectors,
   diffusion/refinement steps, tokenizer remap, and OT initialization should
   enter only when they have matched-budget toy or route-pool diagnostics.

## 2026-04-21 Exact Patch Frontier, Byte-Span Route Atoms, And Limit-10 Controls

Executed next:

- Added `references/360_recent_cross_model_interface_refs.md`, covering SAE /
  universal dictionaries, model stitching, tokenizer adaptation, activated
  adapters, decode-time refinement, and multimodal connector transfer.
- Tightened the protected-frontier toy with an `exact_patch_effect_protect`
  selector and an explicit `utility_oracle_protect` contrast, so the
  compression oracle is separated from semantic utility-positive atoms.
- Added the deterministic `shared_byte_span_route_atom_remap` toy, which tests
  whether a learned byte/span remap helps route-atom recovery across tokenizer
  boundaries before atom selection.
- Widened KVPress same-model controls from limit-5 to limit-10 on GSM70 and
  SVAMP70 using CPU fallback.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Exact patch-effect protection | accuracy `0.8073`, MSE `0.1698`, patch-rank correlation `1.0000` | The protected-frontier oracle is about marginal patch effect under compression, not generic task utility. |
| Quant-error protection | accuracy `0.8073`, MSE `0.1861`, patch-rank correlation `0.2712`, harm `0.0000` | Quantization-error saliency remains the best cheap proxy and is worth promoting to real route pools first. |
| Utility-positive oracle protection | accuracy `0.7812`, MSE `0.2321`, protected-oracle overlap `1.0000` | Selecting atoms that look semantically helpful can still miss compression-critical atoms. |
| Shared byte/span remap route atoms | accuracy `0.9583`, MSE `0.0028`, remap coverage `0.9167`, atom recovery `0.6111`, bytes `33.33` | A tokenizer-aware shared interface can improve task behavior and atom recovery over token-id and regroup baselines in toy form. |
| Token-id / regroup baselines | both accuracy `0.9167`, MSE about `0.0081`, bytes `34.92` | Boundary-compatible decoding alone does not recover the route atoms the bridge needs. |
| KVPress GSM70 limit-10 | `none` and expected-attention both accuracy `0.1000`; expected-attention latency `9.2177s` vs `11.3788s` | Same-model compression remains neutral on tiny GSM, but the harness now widens cleanly. |
| KVPress SVAMP70 limit-10 | `none` accuracy `0.2000`; expected-attention accuracy `0.5000`, latency `7.0471s` vs `8.6424s` | SVAMP remains the only positive KVPress smoke, still too small for a paper claim. |

Updated read:

The paper should add these results additively only as disciplined ablations,
not as headline claims yet. The positive-method lane remains strict route
selection plus target-side process repair, with the new toy stack supporting a
more interpretable bridge design: shared feature/atom interfaces, tokenizer
remapping, exact or proxy protected-frontier selection, and budget-aware
repair. The blocker is no longer "does any component help"; it is whether we
can promote the interacting components to route-pool evidence without hiding
negative interaction terms.

Next execution ladder:

1. Promote quant-error and exact-patch-style frontier labels to real route-pool
   telemetry before training another bridge.
2. Build a byte/token stress split where source-target boundary F1 is
   meaningfully below `1.0`, then rerun tokenizer remap, byte-span remap, and
   dynalign variants there.
3. Widen KVPress to limit-20 before comparing speed/accuracy against any
   LatentWire row; keep it in the same-model compression-control table.
4. Add SAE/universal-dictionary selectors as protected-frontier alternatives,
   but compare them against exact patch effect and quant-error, not only
   activation magnitude.
5. Stack only after interaction tests pass: shared feature/atom interface,
   byte-span remap, verifier pruning, quant-error protection, and repair gate.

## 2026-04-21 Universal Dictionaries, Iterative Refinement, And Limit-20 Controls

Executed next:

- Added `references/361_recent_refinement_quant_connector_refs.md`, covering
  LatentMAS, routed projector banks, universal dictionaries, model stitching,
  iterative latent refinement, tokenizer-aware bridges, outlier-aware
  protected frontiers, mixed-bit allocation, linear correction, and asymmetric
  K/V vector quantization.
- Added `universal_dictionary_frontier`, a deterministic toy that compares
  universal-dictionary feature persistence against raw activation, quant-error,
  exact patch-effect, random, and utility-oracle protected-frontier selectors.
- Added `iterative_latent_refinement`, a deterministic toy for one-pass bridge
  transfer versus two-step, four-step, confidence-gated, noisy diffusion-style,
  and oracle target-side latent refinement.
- Widened KVPress same-model controls to limit-20 on GSM70 and SVAMP70 using
  CPU fallback.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Universal dictionary selector | accuracy `1.0000`, MSE `0.0393`, feature persistence `0.3303`, patch correlation `0.6078`, stability `1.0000` | Shared feature persistence is a viable selector proxy, but it still trails exact patch-effect on MSE. |
| Exact patch-effect selector | accuracy `1.0000`, MSE `0.0318`, patch correlation `1.0000` | Exact patch-effect remains the oracle target for selector calibration. |
| Quant-error selector | accuracy `1.0000`, MSE `0.0324`, patch correlation `0.8675` | Quant-error remains the strongest cheap compression-native proxy in this toy. |
| Utility oracle selector | accuracy `0.7188`, MSE `0.0809`, patch correlation `-0.5753` | Utility-positive features can be actively misaligned with compression-critical frontier atoms. |
| Two-step refinement | accuracy `0.9563`, MSE `0.0449`, MSE-help `0.8250`, harm `0.0312` | Refinement improves latent quality but needs a task-aware stop rule. |
| Four-step refinement | accuracy `0.9125`, MSE `0.0673`, harm `0.0812` | Over-refinement is real; iterative methods cannot be added blindly. |
| Noisy diffusion refinement | accuracy `0.9500`, MSE `0.0468`, MSE-help `0.8500` | Denoising-style refinement is plausible, but not better than controlled two-step repair here. |
| Oracle refinement | accuracy `0.9750`, MSE `0.0048`, MSE-help `1.0000` | There is target-side repair headroom if the stop/selector policy improves. |
| KVPress GSM70 limit-20 | no-press `0.1000`; expected-attention `0.0500`, faster latency `8.9745s` vs `10.2290s` | Same-model compression can trade accuracy for speed; it is not a stable positive comparator on GSM. |
| KVPress SVAMP70 limit-20 | no-press `0.1500`; expected-attention `0.3000`, latency `8.7550s` vs `8.6173s` | Same-model compression can improve SVAMP accuracy, but not speed in this row; still smoke-scale. |

Updated read:

Do add the new ideas to the paper, but only as interpretable ablation axes:
feature-basis selectors, tokenizer remapping, protected mixed-bit frontier
allocation, and target-side refinement. Do not add them as a second headline
method yet. The route/repair result is still the only real-model positive
candidate; these toys explain how to make the next bridge attempt less blind.

Blockers to solve individually:

1. **Selector blocker:** choose compression-critical atoms, not merely
   semantically useful atoms. Exact patch effect is the audit target;
   quant-error and universal-dictionary persistence are the current proxies.
2. **Tokenizer blocker:** current Qwen slices do not stress boundaries enough.
   Build a byte/token stress split before spending real-model budget.
3. **Refinement blocker:** refinement has headroom but can over-correct. Add
   stop reasons, confidence calibration, and harm counters before scaling.
4. **Competitor blocker:** KVPress is runnable through chunking; LatentMAS is
   now the highest-priority direct latent-communication competitor to
   bootstrap next.
5. **Paper-readiness blocker:** every headline row still needs paired CIs,
   token/byte/latency ledgers, target self-repair, C2C, and same-model
   compression controls.

Near-term timeline:

1. **Next 1-2 runs:** route-pool telemetry for quant-error, exact-patch proxy,
   universal-dictionary persistence, protected byte share, and atom recovery.
2. **Next 2-4 runs:** tokenizer stress set plus byte/span remap and TokAlign /
   aligned-logit controls.
3. **Next 4-6 runs:** stop-rule refinement experiments using verifier/process
   features, with help/harm and compute matched against repair-all.
4. **Next benchmark batch:** LatentMAS bootstrap plus repeated KVPress
   limit-20 or bounded limit-50 controls.
5. **Submission gate:** promote only components that survive interaction tests
   on held-out GSM70/SVAMP70 with matched budgets and paired intervals.

## 2026-04-21 LatentMAS Bootstrap, Tokenizer Stress, Mixed-Bit Allocation, And Selector Telemetry

Executed next:

- Added `references/362_latentmas_competitor_bootstrap_refs.md`. The
  LatentMAS repo already exists at `references/repos/LatentMAS`, is clean, and
  points to `https://github.com/Gen-Verse/LatentMAS.git` at commit `b9b2095`.
  `references/repos/` is ignored, so the vendor repo remains unstaged.
- Added `tokenizer_stress_split`, a deterministic byte/token stress diagnostic
  with Unicode, math units, decimals, variables, punctuation, and multibyte
  spans.
- Added `mixed_bit_route_atom_allocator`, an EXL2/AWQ-style toy that keeps
  route atoms active while changing per-atom precision under a target average
  bits-per-weight budget.
- Added `frontier_selector_telemetry`, a lightweight schema normalizer for toy
  and future real route-pool selector rows.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| LatentMAS bootstrap | local clean clone at `references/repos/LatentMAS`, native GSM task, no native SVAMP | LatentMAS is the next direct latent-communication competitor, but needs a LatentWire-side wrapper for paper-grade telemetry and SVAMP. |
| Tokenizer stress split | boundary F1 `0.9463`, fragmentation delta `0.0448`, byte-span remap coverage `0.9354` | The stress set creates real tokenizer mismatch while retaining interpretable byte-span coverage. |
| Token-ID reconstruction proxy | exact reconstruction `0.0833` | Token-ID transfer remains brittle under stress, even when boundary F1 is fairly high. |
| Byte-span reconstruction proxy | exact reconstruction `1.0000` | Byte-span remapping is the right diagnostic baseline for tokenizer-aware bridge work. |
| Mixed-bit uniform 3-bit | accuracy `0.2250`, MSE `0.1269`, bpw `3.0000` | Flat low-bit compression destroys route-atom behavior. |
| Mixed-bit uniform 4-bit | accuracy `1.0000`, MSE `0.0303`, bpw `4.0000` | Full recovery is possible, but spends more bits uniformly. |
| Quant-error target-bpw allocator | accuracy `1.0000`, MSE `0.0314`, achieved bpw `3.9375`, patch corr `0.8886`, outlier protection `0.6000` | Quant-error allocation recovers uniform-4-bit task accuracy with fewer bits and strong interpretability. |
| Universal-feature allocator | accuracy `0.7438`, MSE `0.0334`, patch corr `0.8402` | Feature persistence can preserve reconstruction-like quality while missing task-critical atoms. |
| Frontier telemetry schema | normalized selector method, correlations, protected ids, bit allocation, help/harm, missed-help, false-prune, bytes, compute, stability | This is the schema to require before promoting selector evidence from toys to real route pools. |

Updated read:

The additive stack is now more precise: tokenizer stress should come before
real tokenizer-bridge claims; quant-error bit allocation is the best current
compression-native allocator; universal features are useful but need
task-critical calibration; and LatentMAS should be bootstrapped as a direct
competitor rather than treated as only related work. The paper should add these
as ablation and evaluation infrastructure, not as separate headline methods.

Next execution ladder:

1. Implement `scripts/run_latentmas_competitor_eval.py` outside the vendor repo
   to emit JSONL + `.meta.json` telemetry for native GSM and mapped SVAMP.
2. Run a small LatentMAS GSM10 baseline/text-MAS/latent-MAS smoke and compare
   against LatentWire target-alone and strict route/repair rows on matched
   limits.
3. Promote tokenizer stress to real tokenizers and real prompt slices:
   boundary F1, fragmentation delta, remap coverage, reconstruction proxy, and
   downstream route delta.
4. Promote mixed-bit allocation to the protected-frontier stack, with
   quant-error, exact-patch proxy, universal-feature, random, and oracle
   selectors under matched bpw.
5. Make `frontier_selector_telemetry` the required sidecar schema for all
   future route-pool selector experiments.

## 2026-04-21 Routed Projector Banks, Stop Rules, And LatentMAS Wrapper

Executed next:

- Added `references/363_recent_routed_refinement_reasoning_refs.md`, covering
  routed multimodal connectors, adaptive latent-depth reasoning, confidence
  refinement, latent diffusion residual denoising, LatentMAS/Interlat-style
  communication, and hub-versus-pairwise latent interfaces.
- Added `routed_projector_bank`, a deterministic toy for one monolithic
  bridge versus oracle, feature-routed, confidence-routed, and random
  projector banks under route-specific gauges.
- Added `refinement_stop_rules`, a deterministic toy for fixed-depth latent
  repair versus confidence, score-drift, verifier-harm, and oracle stop rules.
- Added `scripts/run_latentmas_competitor_eval.py`, a LatentWire-side wrapper
  for cloned LatentMAS methods that converts GSM/SVAMP JSONL rows and emits
  prediction JSONL plus `.jsonl.meta.json` telemetry without editing the
  ignored vendor repo.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Oracle routed projector bank | accuracy `0.9688`, MSE `0.0031`, route acc `1.0000` | Route-specific gauges have large headroom if routing is correct. |
| Feature-routed projector bank | accuracy `0.9187`, MSE `0.1387`, route acc `0.9187` | Cheap feature/centroid routing is a plausible next bridge selector. |
| Confidence-routed projector bank | accuracy `0.3000`, route acc `0.1437`, route-mismatch failures `112/160` | Target-head confidence alone is not a safe router. |
| Monolithic projector | accuracy `0.8687`, MSE `0.1002` | A single bridge is viable but leaves clear route-specific capacity unused. |
| Fixed 2-step refinement stop toy | MSE `0.0449` vs one-step `0.0559`, but harm `0.0312` | Latent repair improves reconstruction while already introducing task harm. |
| Fixed 4-step refinement stop toy | accuracy `0.9125`, over-refinement `0.9625` | Blind iterative repair is unsafe and should not be a headline component. |
| Verifier-harm stop | accuracy `0.9625`, harm `0.0188`, nonzero repair | Verifier-style stop rules can cap damage while preserving repair opportunities. |
| Oracle stop | accuracy `0.9688`, MSE `0.0406`, over-refinement `0.0000` | There is substantial stop-policy headroom. |
| LatentMAS wrapper | testable converter, lazy imports, JSONL/meta telemetry, compact trace hashes | Direct latent-communication competitor execution is now a runnable harness task, not a missing implementation. |

Updated read:

The paper should add two components to the ablation plan, not yet to the
headline method: routed projector/interface banks and stop-rule-governed
target-side latent repair. The toy evidence says both are structurally
important, but the route selector and repair halt policy are the actual
blockers. Confidence-only routing failed badly; fixed-depth repair over-refines
badly. The most credible positive-method stack is therefore not "more latent
steps" or "bigger bridge"; it is **byte/span-normalized route atoms + routed
projector bank + protected mixed-bit frontier + verifier/process stop rule +
matched LatentMAS/C2C/target-repair controls**.

Next execution ladder:

1. Run `scripts/run_latentmas_competitor_eval.py` on GSM/SVAMP limit smokes for
   `baseline`, `text_mas`, and `latent_mas`, capturing accuracy, latency,
   tokens, agent trace counts, and parse failures.
2. Promote feature-routed projector banks into the real route-pool harness:
   compare monolithic, source-layer routed, target-layer routed, feature
   routed, confidence routed, and random routed variants with matched bytes.
3. Promote stop-rule telemetry to the real repair lane: fixed 0/1/2/4 repair,
   confidence halt, score-drift halt, process-verifier halt, and oracle
   best-prefix, all with help/harm and over-refinement counters.
4. Stack only after interaction tests: tokenizer stress remap, quant-error
   mixed-bit allocation, feature route bank, and verifier-harm stop.
5. Paper gate remains positive-method only: promote a stack only if it beats
   target self-repair and direct competitors on matched examples with paired
   intervals and compute/token/byte ledgers.

## 2026-04-21 Hub Dictionaries, Stable Routers, And Matched Competitors

Executed next:

- Added `references/364_hub_router_tokenizer_verifier_refs.md`, covering hub
  latent interfaces, route stability, tokenizer adaptation, and verifier/search
  repair papers from the recent sweep.
- Added a hub-dictionary bridge toy: pairwise adapters versus a shared hub
  interface, random hub, oracle hub, and held-out-family transfer controls.
- Added a router-stability toy: hard feature routing, confidence routing, dense
  routing, load-balanced routing, sticky routing, and oracle/random controls.
- Added `scripts/build_matched_competitor_table.py`, which renders an explicit
  matched GSM70 comparison matrix while keeping missing competitor rows visible.
- Ran cached LatentMAS wrapper harness probes for `baseline` and `text_mas` on
  Qwen2.5-0.5B `N=1`; latent-MAS itself remains runtime-blocked locally.

New evidence:

| Result | Outcome | Implication |
|---|---|---|
| Hub dictionary bridge | accuracy `1.0000`, atom recovery `1.0000`, MSE `0.0199`, using `12` adapters | A shared hub can beat pairwise scaling when source/target families share route atoms. |
| Pairwise adapters | accuracy `0.6792`, MSE `0.3445`, using `20` adapters | Pairwise bridges scale badly and miss shared atom structure. |
| Random hub | accuracy `0.1875` | The hub result is not just extra capacity; atom semantics matter. |
| Held-out-family hub transfer | accuracy `1.0000` | The shared-interface hypothesis has generalization headroom in toy form. |
| Stable feature router | accuracy `0.9438`, MSE `0.0243`, perturb stability `0.9500` | Route assignment can be accurate and stable when it uses geometry instead of target confidence. |
| Confidence router | accuracy `0.3688`, MSE `1.5497` | Confidence-only routing is a saturated failure mode and should not be repeated as the sole selector. |
| Sticky router | accuracy `0.9438`, perturb stability `1.0000` | Stability regularization can remove small perturbation flips without losing task accuracy. |
| Matched GSM70 matrix | selected route + repair `0.2000`, target self-repair `0.1714`, C2C `0.1286`, KVComm `0.0000` | Current route-specific margin is positive but small; fair LatentMAS rows are still missing. |
| LatentMAS baseline/text-MAS probes | both `0.0000` on `N=1`, latencies `10.72s` and `15.37s` | Wrapper plumbing works with cached small models, but these are not competitor rows. |
| LatentMAS latent-MAS probe | import blocker fixed; MPS `linalg.solve` needed fallback; generation then failed in HF cache-position prep | The remaining direct competitor blocker is runtime compatibility, not missing wrapper telemetry. |

Updated read:

The strongest additive architecture clue is now **hub dictionary + stable
feature/sticky routing**, not another pairwise adapter or confidence selector.
This is still toy evidence, but it matches the paper's symmetry problem: many
model pairs appear to need a shared gauge/feature basis plus route-specific
experts, not `O(n^2)` pairwise maps. The real-model stack should therefore
promote hub/shared feature IDs, route stability, atom recovery, and matched
route-specific repair deltas together.

Immediate execution ladder:

1. Move hub dictionaries into real route-pool diagnostics: shared feature IDs,
   atom recovery, route-family transfer, dead-feature rate, and pairwise-versus
   hub parameter/byte scaling.
2. Move sticky/feature routing into the route-pool harness with random,
   confidence, dense, load-balanced, oracle, and perturbation-stability
   controls.
3. Patch or isolate LatentMAS latent-mode runtime on a bounded fair slice:
   cached Qwen2.5 smokes first, then Qwen3 only when the model is cached or the
   machine can sustain the fetch.
4. Keep the matched competitor matrix as a required paper artifact; missing
   rows must stay visible until real metrics exist.
5. Only stack hub dictionary, sticky routing, tokenizer remap, mixed-bit
   frontier, and verifier stop rules after interaction ablations confirm the
   components do not reverse each other.

## 2026-04-21 Leak-Free Stack Interaction Check

I added a deterministic composition toy for the exact stack we were tempted to
promote: hub dictionary, sticky routing, protected mixed-bit frontier, and
verifier stop. I first corrected the verifier stage to avoid using the held-out
target vector as the repair residual; the leak-free version uses calibration
class anchors and selects the logged stop step.

Result:

- raw pairwise bridge: accuracy `0.7344`, MSE `0.5842`, bytes `50,400`
- hub dictionary only: `0.6250`, below raw pairwise
- hub + sticky router + protected mixed-bit frontier: `0.5990`
- full hub + sticky + frontier + verifier stop: `0.5938`, over-refinement
  `0.4583`
- oracle router control: `0.8229`, MSE `0.2609`

That means:

- the components are not monotonically additive
- the route-quality ceiling is real, because oracle routing beats raw pairwise
- the naive stack should not be promoted to a paper method yet
- the next stack should isolate route assignment and stop-policy quality before
  adding mixed-bit frontier compression
- this result strengthens the paper's interpretability requirement: atom
  recovery, route accuracy, perturbation stability, bit histograms, stop
  reasons, help/harm, and over-refinement must travel together in the table

## 2026-04-21 Route-Conditioned Hub Frontier Sweep

I then decomposed the hub stack by router, frontier, and stop rule instead of
keeping only one composite row. The sweep evaluates the same hub decoder under
conditional-prior, feature, sticky, confidence, random, and oracle routing,
with and without the current protected frontier and verifier stop heuristic.

Result:

- raw pairwise bridge: `0.7344`
- oracle-routed hub base: `0.8229`
- oracle-routed hub + current frontier: `0.8125`
- oracle-routed hub + current frontier + current stop rule: `0.8073`
- best non-oracle hub base: conditional prior / confidence at `0.6250`
- best frontier gain anywhere in the sweep: only `+0.0104`
- best stop gain anywhere in the sweep: `0.0000`

That means:

- route assignment is still a real headroom source, because oracle routing
  lifts the hub base above raw pairwise
- but the current frontier and stop heuristics are also mis-specified, because
  they stay negative even under oracle routing
- the next stack should not simply “add frontier + verifier” after improving
  routing; it needs a route-aware protected set and a route-/class-calibrated
  stop policy
- this narrows the paper story further: shared hubs are plausible, but the
  positive method depends on solving both route assignment and later
  compression/repair control, not only the router

## 2026-04-21 Route-Class Patch Frontier Follow-Up

I then tested the most direct redesign of that frontier: use calibration-time
route/class-specific exact patch effects to choose the high-precision subset,
and use the modal best stop step per route/class as a lightweight calibrated
stop policy.

Result:

- conditional-prior base: `0.6250`
- conditional-prior quant-error frontier: `0.6354`
- conditional-prior route-class patch-protect: `0.6354`
- conditional-prior route-class frontier: `0.6146`
- oracle base: `0.8229`
- oracle quant-error frontier: `0.8125`
- oracle route-class patch-protect: `0.8125`
- oracle route-class frontier: `0.7969`

That means:

- calibration-aware protection can match the current quant-error frontier, but
  it does not beat it
- route-class frontier pruning is still actively harmful
- route-/class-calibrated stop selection is not the missing fix either
- the next positive-method shot should move up a level to multi-way canonical
  hubs, tokenizer/interface simplification, or a genuinely different pruning
  rule rather than more local score shaping

## 2026-04-21 Frozen Residual Harness Baseline

I added a dedicated residual-rank sweep runner for the frozen GSM8K32 contract:
`scripts/run_gsm8k_contract_residual_sweep.py`.

The first completed baseline pass reuses the existing `rank8` checkpoints only
and confirms that the harness itself is valid:

- `dynalign_module_replace_residrank8 = 0.0938`
- `tokenbasis_replace_residrank8 = 0.0938`
- both rows keep full numeric extraction coverage (`32/32`) and exact ID parity

That means:

- the residual benchmark path is now ready
- the next real benchmark action is not another contract check; it is the
  expensive `rank16` recalibration on top of `dynalign_module_replace` and
  `tokenbasis_replace`
- if neither `rank16` row beats `0.0938`, the same-pair live branch is likely
  saturated and the next benchmark move should be a gauge-fix /
  adaptive-canonicalization wrapper on top of the same real lane
