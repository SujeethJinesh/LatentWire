# ARC Cross-Family Failure Decomposition

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: the ARC `8B` payload / `11B` framed same-family
  Fourier/anchor-syndrome packet is seed-stable and uncertainty-stable, but
  strict Phi-3 cross-family transfer fails.
- Exact blocking gap: the next method must improve the cross-family source
  endpoint or learn a common-feature connector; the current 8B packet codec is
  not the dominant failure mode.

## Artifact

`results/source_private_arc_cross_family_failure_decomposition_20260502/`

Files:

- `arc_cross_family_failure_decomposition.json`
- `arc_cross_family_failure_decomposition.md`
- `family_failure_summary.csv`
- `manifest.json`

## Result

| Source family | Primary blocker | Full test source | Full packet | Packet follows source | Disagreement matched/Qwen-sub |
|---|---|---:|---:|---:|---:|
| Phi-3 Mini 4K | source endpoint quality | `0.246` | `0.244` | `0.997` | `0.200/0.340` |
| Qwen2.5-1.5B | mixed / validation incomplete | `0.445` | `0.442` | `0.996` | `0.482/0.184` |
| TinyLlama-1.1B | source-family mismatch | `0.326` | `0.325` | `0.996` | `0.269/0.317` |

The headline decomposition selects:

`common_feature_connector_with_stronger_source`

## Interpretation

The 8B packet is mostly faithful to whatever source choice it receives:
full-slice packet-follow-source rates are `0.996-0.997`, and packet accuracy is
within about `0.002` of the source-choice accuracy. That means the Phi-3
failure is not mainly a garbled-wire problem.

The failure is source/common-feature quality. Phi-3 chooses below the target on
test (`0.246` vs target `0.265`), and on rows where Phi-3 and Qwen disagree,
the Qwen-substituted packet is much stronger (`0.340` vs `0.200`). Qwen2.5-1.5B
shows the opposite: a stronger same-family source produces a large frozen-test
gain, but validation disagreement remains too small/incomplete for an ICLR
claim.

## Decision

Do not spend the next turn revising the 8B packet codec in isolation. The next
high-value branch is a target-conditioned sparse innovation/crosscoder or
protected-subspace consistency connector using a stronger source, with strict
source-necessity controls.

## Lay Explanation

This checks whether the tiny message got scrambled or whether the sender sent a
weak answer. The message usually carries the sender's answer correctly. The
problem is that the Phi-3 sender is not choosing useful answers often enough,
and the receiver still lacks a shared feature language for deciding when a
different source model knows something useful.

## Next Exact Gate

Implement a cheap `n32/n64` Phi-3/Qwen disagreement smoke for a
target-conditioned sparse innovation connector:

- encode source-only residuals after removing target-public candidate features;
- packetize top-k sparse atoms at `8B/16B/32B`;
- train a permutation-equivariant target residual scorer with accept/abstain;
- require matched packets to beat zero-source, candidate-roll, source shuffle,
  same-byte text, and Qwen-substituted controls before widening.
