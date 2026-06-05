# Channel Sets Drift During Long Reasoning

## Abstract

W4A16 long-reasoning quantization often protects a fixed set of high-risk activation channels. Channel-Set asks whether that static unit is stable enough to support a positive method. The current evidence supports a bounded-negative regime result instead: top-channel sets drift substantially across reasoning horizons and model packets, with top-1% strict set-leaving of 0.634245 for Granite-Tiny, 0.538294 for Granite-Small, 0.533713 for Nemotron-3, and 0.670573 for the Phase 5' Transformer packet. Within-set rank shuffling is smaller but nontrivial, so the problem is not only threshold noise. The positive C-A1 clip/backfill path is not claimable because cached gate evidence was contaminated by confirmation-source rows; it must be rebuilt on fresh non-confirm same-row dev/gate materialization before GPU spend. The paper therefore contributes a measurement and regime map: static channel protection is a weak abstraction under long reasoning, and any positive Channel-Set method must be evaluated against ParoQuant/static baselines with strict split hygiene and OSC/DecDEC defenses.

## 1. Introduction

Long reasoning changes which activation channels matter. A W4A16 method that protects a static outlier set can work only if that set remains stable enough over the trace. The Channel-Set campaign tested a family of positive methods around per-model clipping, stable cores, warmup selectors, and drift-aware routing. The result is not yet a confirmed positive method. The robust result is a measurement: outlier channel membership itself moves.

This bounded negative is useful because it prevents an easy but fragile story. A method that wins by protecting a fixed early channel set may be solving the wrong unit if later tokens leave that set. Conversely, if a method adapts channel protection without beating ParoQuant and static controls under identical rows, the apparent gain is not a paper-strength contribution.

This draft makes four contributions.

1. We measure strict set-leaving across model packets and thresholds. At the top-1% threshold, strict set-leaving ranges from 0.533713 to 0.670573 across the reported packets.

2. We separate strict set-leaving from within-set rank shuffling. Both occur, but strict set-leaving dominates the story for the top-1% unit.

3. We audit the positive path. C-A1 is not claimable from cached evidence because confirm-contaminated source rows appear in the gate artifacts. The correct next step is fresh non-confirm same-row materialization.

4. We keep OSC/DecDEC and ParoQuant as defenses rather than optional baselines. No "beats static" result is sufficient unless it also survives the harder named baselines and parity checks.

## 2. Claim Boundary

The current claim is a bounded measurement/regime claim, not a positive method claim. The paper may state that static channel-set protection is unstable under long reasoning in these packets, and that the campaign did not establish a confirmed adaptive method. It may not state that C-A1 beats ParoQuant or that any GPU method is confirmed.

The positive claim requires:

- a registry card predating any confirmation access;
- Phase-A baseline lock for ParoQuant/static/EMA;
- fresh non-confirm dev/gate rows;
- same-row paired comparison across models and policies;
- pre-launch review records for the exact runner code hash;
- planted positive controls before promotion;
- held-out confirmation only after the above gates pass.

## 3. Measurement

The central metric is strict set-leaving. For a threshold such as top 1%, define the top-channel set at one trace segment and ask what fraction of later high-risk channels are outside that protected set. A value near zero would support static protection. Values above one half mean that the static set misses a large portion of the later outlier mass.

Within-set shuffling is measured separately: among channels that remain inside the protected set, how much does their rank/order change? This matters for policies that allocate different protection levels within a fixed set. It is not a replacement for strict set-leaving, because channels that leave the set entirely cannot be recovered by a better within-set ordering.

The threshold sensitivity analysis reports 0.5%, 1%, 2%, and 5% thresholds without selecting a threshold post hoc.

## 4. Results

### 4.1 Top-1% Strict Set-Leaving Is Large

| Packet | Top % | Gate-style migration | Strict set-leaving | Within-set shuffling |
| --- | ---: | ---: | ---: | ---: |
| Phase 0 Granite-Tiny | 1.0 | 0.817839 | 0.634245 | 0.175260 |
| Phase 1 Granite-Small | 1.0 | 0.843166 | 0.566235 | 0.270935 |
| Phase 2 Nemotron-3 | 1.0 | 0.820810 | 0.533713 | 0.269082 |
| Phase 5' Transformer | 1.0 | 0.839379 | 0.670573 | 0.165737 |

The top-1% set-leaving values are all above 0.53 and reach 0.67. This is too large for a simple static protection story. The high-risk channels at later horizons are often not the same channels that a static early set would protect.

### 4.2 The Finding Is Not A Single-Threshold Artifact

The threshold sensitivity scan shows the same qualitative behavior at nearby thresholds.

| Packet | 0.5% strict set-leaving | 1.0% | 2.0% | 5.0% |
| --- | ---: | ---: | ---: | ---: |
| Granite-Tiny | 0.583333 | 0.634245 | 0.669153 | 0.658009 |
| Granite-Small | 0.538294 | 0.566235 | 0.574123 | 0.689162 |
| Nemotron-3 | 0.520891 | 0.533713 | 0.520922 | 0.566002 |
| Phase 5' Transformer | 0.639323 | 0.670573 | 0.688028 | 0.680968 |

The exact value changes with threshold, but the regime does not. Static sets lose a large fraction of later high-risk channels across the reported thresholds.

### 4.3 Within-Set Shuffling Is Smaller But Still Relevant

Within-set shuffling at top 1% ranges from 0.165737 to 0.270935. This supports a two-level interpretation. First, many later high-risk channels are outside the static set at all. Second, even among channels that remain inside the set, their relative importance changes enough that fixed within-set allocation may also be fragile.

## 5. Positive Methods And Why They Are Not Yet Claims

C-A1, the per-model CVaR/EVT clip path, remains the nearest positive-method candidate. The current runbook correctly requires a three-model paired matrix and fresh non-confirm dev/gate materialization. Cached C-A1 gate evidence is invalid for selection because the confirm path audit found confirmation-source rows in the relevant gate artifacts. The required action is rebuild, not reinterpretation.

C-F and CE13 remain possible positive enablers only if they clear the same baseline and audit gates. C-D1 and CE21 are mandatory defenses: OSC/DecDEC stress and no-gap filters protect against overclaiming a policy that only wins in a narrow or contaminated trace regime.

Sidecar/kernel-free ideas are not claimable without a real systems path. A measurement result can be paper-strength; an unimplemented sidecar is not.

## 6. Baselines And Defenses

ParoQuant is the hardest named baseline for any adaptive protection claim. A method that beats static clipping but not ParoQuant is at best a diagnostic. The paper should report ParoQuant parity before any "beats ParoQuant" language.

OSC and DecDEC defenses are needed because long-reasoning failures can be reinterpreted as either output-space consistency issues or decode/decompose mismatch. A Channel-Set method should not win by exploiting an artifact that those defenses would remove.

The audit gate is part of the method, not bookkeeping. The C-A1 contamination finding is evidence that the gate matters: without it, the campaign would have been able to tell a premature positive story.

## 7. Related Work

This paper is closest to W4A16 quantization and outlier-channel protection work, including ParoQuant and related activation-aware policies. It is also adjacent to KL-Lens, ResQ, OSC, and DecDEC because the key question is not only how much an activation changes, but whether the change predicts a user-visible reasoning failure under a controlled decode.

The novelty of this bounded result is the long-reasoning channel-set view: rather than assuming a protected set and measuring final accuracy, it asks whether the protected set remains the same object over the trace.

## 8. Limitations

This draft is not a positive method paper yet. It needs final plotting, exact provenance tables, and the fresh C-A1 non-confirm matrix before any adaptive policy claim can be made.

The current measurement relies on cached decomposition packets. It is strong enough to motivate the static-set negative, but method claims require fresh paired rows, identical prompts, model/policy hashes, and write-once per-trace evidence.

The paper should avoid claiming that static channel sets are always wrong. The bounded claim is that in these long-reasoning packets, strict set-leaving is large enough that static protection is not a sufficient abstraction.

## 9. Conclusion

Channel-Set currently contributes a robust measurement and a discipline lesson. Long-reasoning outlier channel sets drift: at top 1%, more than half of later high-risk channels leave the static protected set in every reported packet. That makes static W4A16 protection a weak unit for long reasoning. A positive method may still emerge from C-A1 or related adaptive policies, but only after fresh non-confirm paired materialization and the full baseline/audit gate.
