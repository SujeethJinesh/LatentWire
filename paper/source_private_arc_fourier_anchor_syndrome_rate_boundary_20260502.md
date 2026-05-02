# Source-Private ARC Fourier/Anchor-Syndrome Rate Boundary, 2026-05-02

## Status

- current paper readiness: COLM is stronger; ICLR is still blocked by strict
  cross-family and native systems evidence.
- current story: LatentWire has a positive fixed-byte ARC packet row that uses
  a shared public Fourier/anchor-syndrome basis and destructive basis-mismatch
  controls.
- exact blocking gap: the row is still same-family/source-cache dependent and
  not yet backed by native GPU serving comparisons.

Primary artifact:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed_b2000/`

Prior 10-seed artifact:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed/`

Prior 5-seed artifact:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8/`

## What This Gate Does

This reruns the ARC Fourier/anchor-syndrome packet with the payload reduced
from `12B` to `8B`. With the packet header and CRC included, the framed record
is `11B`.

Lay explanation: we tried to shrink the tiny packet while keeping the same
receiver and the same destructive controls. If the packet still works, then the
method has a stronger systems story because less information crosses the
source-to-target boundary.

## Results

ARC-Challenge test, 10 packet seeds and 2000 paired bootstrap samples:

| Row | Result |
|---|---:|
| pass gate | `True` |
| matched Fourier/anchor-syndrome seed pass count | 10/10 |
| bootstrap samples | 2000 |
| payload bytes | 8B |
| framed bytes with header/CRC | 11B |
| matched mean / min accuracy | 0.344 / 0.342 |
| target-only accuracy | 0.265 |
| same-byte text accuracy | 0.300 |
| min lift over target | +0.077 |
| min lift over same-byte text | +0.042 |
| min CI95 low vs target | +0.038 |
| candidate derangement max | 0.217 |

Validation also passes `10/10` seeds. On test, all destructive mismatch controls
fail the pass gate:

- anchor-ID shuffle: `0/10`, matched mean `0.252`;
- anchor-value shuffle: `0/10`, matched mean `0.242`;
- spectral-bin permutation: `0/10`, matched mean `0.263`.

The random shared-anchor diagnostic still passes `10/10`. That confirms the same
claim boundary as the 12B row: the packet needs shared public coordinate
agreement, but the present artifact does not prove semantic anchor names are
special.

## Decision

Promote the ARC Fourier/anchor-syndrome headline from `12B` payload / `5`
seeds to `8B` payload / `10` seeds (`11B` framed) for the current paper draft
and systems accounting. This is a stronger positive method row than the
earlier 12B artifact because it preserves the same gate while improving rate
and seed stability.

Do not claim a lower minimum rate yet. Exploratory `4B` and `6B` probes were
interrupted after producing no artifacts so the 8B gate could complete on the
Mac; they are not evidence. The next exact scientific gate is a strict
cross-family ARC run with the unchanged `8B` payload / `11B` framed packet,
the same 10 seeds, and the same 2000-bootstrap paired uncertainty. A planned
overnight `4B/6B/8B/12B` resumable sweep is useful only after the cross-family
gate is settled.
