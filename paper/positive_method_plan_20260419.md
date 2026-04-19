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
