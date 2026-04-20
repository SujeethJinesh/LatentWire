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
