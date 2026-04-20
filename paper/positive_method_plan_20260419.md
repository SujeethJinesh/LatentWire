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

I then tried the obvious stacked follow-up on that weak bridge clue: keep the
same low-rank bridge checkpoint and add the best current runtime routing knobs
on top. Three variants all failed on the matched sparse `gsm8k_eval_10` slice:

- retrieval-head-style runtime head selection (`retrieval_peak`, ratio `0.5`)
- direct QK-fidelity runtime head selection (`attention_qk_fidelity`, ratio `0.5`)
- prior-plus-live blend head selection (`attention_blend`, ratio `0.5`, `alpha=0.25`)

All three fell to `0.0000`, though they did cut the bridge bytes from roughly
`297k` down to roughly `157k`.

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
