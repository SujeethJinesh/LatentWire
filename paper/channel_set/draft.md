# Channel Sets Drift During Long Reasoning

## Abstract

W4A16 long-reasoning quantization often protects a fixed set of high-risk activation channels. Channel-Set asks whether that static unit is stable enough to support a positive method. The current evidence supports a measurement/regime result rather than a confirmed adaptive method: top-channel sets drift substantially across reasoning horizons and model packets, with top-1% strict set-leaving of `0.634245` for Granite-Tiny, `0.566235` for Granite-Small, `0.533713` for Nemotron-3, and `0.670573` for the Phase 5 Transformer packet. Within-set shuffling is smaller but nontrivial. C-A1 remains parked because the cached gate path was contaminated and the fresh three-model same-row ParoQuant-vs-tight-clip matrix is missing. The paper contributes a scoped regime claim: static channel-set protection is a weak abstraction for these long-reasoning traces, and any future positive method must beat ParoQuant/static baselines under split-clean same-row native evidence.

## Claim-Boundary Box

| Supported | Unsupported | Future-only |
| --- | --- | --- |
| Top-1% strict set-leaving is high across the four cached packets: `0.533713` to `0.670573`. | A confirmed-positive claim is unsupported. | C-A1 can become a positive only after fresh split-clean native paired materialization. |
| Drift is not a single-threshold artifact across 0.5%, 1%, 2%, and 5% thresholds. | C-A1 is not confirmed and does not beat ParoQuant in a claim-ready way. | A GPU/native systems card remains separate from this paper-finalization pass. |
| C_U1, C_S1, and C_Y5 blockers are real schema/floor/native-pairing blockers. | No held-out method confirmation is claimed. | OSC/DecDEC defenses can validate a future method after same-row pairing exists. |

## 1. Introduction

Long reasoning changes which activation channels matter. A static W4A16 outlier-protection method can work only if the protected channel set remains stable enough across the reasoning trace. The Channel-Set campaign tested adaptive possibilities, but the confirmed result is simpler and sharper: the set itself moves.

This matters for both method design and evaluation. If later high-risk channels leave the early protected set, then a static top-channel abstraction is incomplete. If an adaptive method appears to win without a paired ParoQuant/static baseline on the same rows, the win may be a cache artifact or a contaminated split artifact rather than a real method.

Contributions:

1. A long-reasoning channel-set drift measurement. At top 1%, strict set-leaving is above `0.53` for every reported packet (Figure 1).

2. A threshold sweep showing that the result is not a single chosen threshold (Figure 2).

3. A separation between strict set-leaving and within-set rank shuffling (Figure 3).

4. An audit-preserving regime map. C-A1 remains parked; C_U1, C_S1, and C_Y5 are blocked by missing schema, floor, or native-pairing requirements.

![Top-1 set leaving](figures/top1_set_leaving.png)

Figure 1: More than half of later high-risk top-1% channels leave the static protected set in every reported packet.

## 2. Measurement Setup

For a top-k channel threshold, define the protected set at one trace segment and ask how many later high-risk channels are outside that set. This is strict set-leaving. A value near zero would support static protection; a value above one half means the static set misses a large portion of later high-risk channels.

Within-set shuffling is measured separately: among channels that remain inside the protected set, their relative rank can still change. This affects allocation inside a fixed protected set but cannot recover channels that leave the set entirely.

The detailed provenance is in `tables/provenance.md`; the final regime checklist is in `tables/regime_checklist.md`.

## 3. Results

### 3.1 Top-1% Strict Set-Leaving Is Large

| Packet | Strict set-leaving | Within-set shuffling |
| --- | ---: | ---: |
| Granite-Tiny | `0.634245` | `0.175260` |
| Granite-Small | `0.566235` | `0.270935` |
| Nemotron-3 | `0.533713` | `0.269082` |
| Phase 5 Transformer | `0.670573` | `0.165737` |

The top-1% values all exceed `0.53`, and the largest reaches `0.670573`. Static protection is therefore not a sufficient abstraction for these traces.

### 3.2 The Result Is Stable Across Thresholds

![Threshold sensitivity](figures/threshold_sensitivity.png)

Figure 2: Strict set-leaving remains high across the 0.5%, 1%, 2%, and 5% thresholds.

The threshold sweep reports all thresholds rather than selecting one post hoc. The exact values move, but the regime does not: strict set-leaving stays high across nearby thresholds for all four packets.

### 3.3 Strict Leaving Dominates Within-Set Shuffling

![Within-set shuffling](figures/within_set_shuffling.png)

Figure 3: Within-set shuffling exists, but strict set-leaving is the larger effect.

Within-set shuffling at top 1% ranges from `0.165737` to `0.270935`. That is nontrivial, but it is smaller than strict set-leaving. The main failure mode for a static set is not only that the protected channels change rank; many later high-risk channels are absent from the protected set at all.

## 4. Why C-A1 Is Parked, Not Confirmed

![Cached policy screens](figures/paroquant_vs_channelset_methods.png)

Figure 4: Cached C-A1 and C-F screens are support evidence only. They are not native same-row ParoQuant comparisons and cannot carry a positive claim.

C-A1 remains the nearest positive-method candidate, but it is not a paper claim. The cached screen has only `6` C-A1 gate rows, with `5` positive medians and `1` nonpositive median. That is enough to motivate a fresh materialization, not enough to claim a method.

![C-A1 sentinel status](figures/c_a1_sentinel_status.png)

Figure 5: The split-clean C-A1 sentinel matrix is incomplete. DeepSeek and Falcon have `12` same-row pairs, while Granite has `0` valid same-row pairs in the safe manifest.

The blocker is concrete. A future claim needs a Granite/DeepSeek/Falcon same-row matrix with ParoQuant baseline and tight-clip C-A1 outputs. The safe manifest has DeepSeek and Falcon same-row pairs, but Granite is missing/invalid. C-A1 remains parked until this is rebuilt.

## 5. Defenses And Schema Blockers

![OSC DecDEC drift defense status](figures/osc_decdec_drift_defense.png)

Figure 6: Defense and router ideas are blocked by missing labels, insufficient rows, or missing native pairing.

The defense stack is useful because it prevents overclaiming:

- C_U1 materializes `500` KL trajectory rows, but lacks paired difficulty/policy-uplift labels.
- C_S1 finds only `198` eligible rows against a `500` row floor. The floor is not lowered.
- C_Y5 has defense inputs and audit checks, but the three-model native C-A1-vs-ParoQuant packet is missing.

These are honest blockers, not negative proof against every future method. They define the evidence required before a positive Channel-Set method can be claimed.

## 6. Related Work

Channel-Set is closest to W4A16 quantization, outlier-channel protection, rotation-based quantization, and residual correction methods such as ParoQuant, OSC, DecDEC, ResQ, OSCAR, and OScaR. Those systems are the correct baseline family for any future adaptive method.

The distinguishing view here is the long-reasoning channel-set object itself. Rather than assuming a protected channel set and measuring final answer accuracy, this paper measures whether the protected set remains the same object over the reasoning trace.

## 7. Limitations

This is a measurement/regime paper, not a confirmed positive-method paper. C-A1 remains parked. The paper does not claim to beat ParoQuant, does not claim a native GPU result, and does not claim held-out method confirmation.

The measurements are cached decomposition evidence. They are strong enough to establish the static-set drift regime, but a method claim needs fresh split-clean native paired rows, model/policy hashes, no-gap denominators, and write-once per-trace evidence.

The claim is also not that static channel sets are always wrong. It is that, in these long-reasoning packets, strict set-leaving is large enough that static protection alone is a weak abstraction.

## 8. Conclusion

Channel-Set contributes a scoped but useful result: long-reasoning outlier channel sets drift. At top 1%, more than half of later high-risk channels leave the static protected set in every reported packet. That turns static W4A16 protection from a stable object into a fragile assumption. A positive method may still emerge from C-A1 or related adaptive policies, but only after fresh split-clean native paired materialization and the full ParoQuant/audit gate.
