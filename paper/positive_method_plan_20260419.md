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
| Numeric-consistency rerankers | `0.0667` | Useful telemetry, not sufficient as standalone selection. |
| KVPress GSM30 | none `0.0667`, expected-attention `0.0667` | Same-model compression control is neutral. |
| Learned protected toy | wins outlier `0.6562`, near-recovers rotated `0.6302`, loses slot-permuted to signal-aware `0.5781` | Learned masks help, but need signal/orientation constraints. |

Updated blocker decomposition:

1. **Candidate-selection blocker:** stochastic cross-model routes expose useful
   answers, and strict format-based fallback is now a stronger positive
   selector (`0.1667`), but it still recovers less than half of the oracle gap.
   The next real method should be a stronger verifier/reranker, not more
   unselected random sampling.
2. **Orientation-aware interface blocker:** learned masks and signal-aware
   masks each help different toy regimes. The likely stack is learned soft mask
   plus supervised signal alignment plus orientation/permutation regularization.
3. **Competitor baseline blocker:** KVPress GSM30 is now a neutral control;
   C2C remains the direct competitor to scale beyond GSM5 when a CUDA-capable
   full replay is available.
4. **Interpretability blocker:** every stochastic run must log candidate set
   quality separately from selection quality: oracle correctness, vote entropy,
   vote margin, verifier score, selected seed, and paired flips.

Next execution ladder:

1. Add a target-model verifier prompt and a pairwise/listwise reranker over the
   existing GSM30 salt012 candidate set; compare against strict format fallback
   before adding more samples.
2. Add confidence-gated route expansion: target-alone or strict-reranker
   confidence decides whether to spend 1, 3, or 5 stochastic routes.
3. Add signal-regularized learned masks to the toy suite and sweep protected
   channels.
4. Promote the best learned/signal protected-channel diagnostic into a frozen
   K/V-slot reconstruction experiment before generation.
5. Add quantization-inspired bridge controls: outlier-channel protection,
   rotation-before-transport, and K/V-asymmetric mixed precision at matched
   bytes.
6. Keep KVPress as a same-model compression control and scale C2C only after
   the verifier lane has a positive GSM30 selection result.
