# Candidate-Local Competitor/Basis Table

- date: `2026-04-30`
- status: reviewer-facing same-slice competitor gap table for the live
  candidate-local residual receiver
- code: `scripts/build_source_private_candidate_local_competitor_basis_table.py`
- artifact:
  `results/source_private_candidate_local_competitor_basis_table_20260430/`
- references:
  `references/550_candidate_local_competitor_basis_table_refs_20260430.md`

## What Changed

The live branch already had n512 seed repeats and a Mac systems waterfall. This
gate turns the common-basis falsification into a paper-facing competitor table
that separates the things reviewers will ask about:

1. measured same-slice controls and common-basis ablations;
2. the live source-private packet receiver;
3. a measured Procrustes common-basis row;
4. measured ridge CCA/SVCCA-style rows;
5. measured LSTIRP-lite inverse-relative rows;
6. measured Sinkhorn OT/Gromov-Wasserstein public-transport rows;
7. two guarded RR repair probes;
8. baselines still required before an ICLR-complete claim.

The table deliberately does not mark C2C, KVComm, or broader native
cache/stitching baselines as defeated. Orthogonal Procrustes, ridge CCA,
LSTIRP-lite inverse-relative projection, Sinkhorn OT, and Gromov-Wasserstein
transport are measured and unsafe under controls. Relative Representations is
measured and alive as a partial clean competitor, while the two simple RR
innovation repairs are now pruned.

## Main Evidence

Summary artifact:
`results/source_private_candidate_local_competitor_basis_table_20260430/candidate_local_competitor_basis_table.md`

Headline:

- common-basis pass gate: `false`;
- live candidate-local residual-normalized n512 rows: `9/9` pass;
- global public-anchor dot-product rows: `0/9` pass;
- public-calibration orthogonal Procrustes rows: `0/9` pass;
- ridge CCA/SVCCA-style canonical-coordinate rows: `0/9` pass;
- ridge CCA/SVCCA-style local-stack rows: `0/9` pass;
- LSTIRP-lite inverse-relative rows: `0/9` pass;
- LSTIRP-lite inverse-relative local-stack rows: `0/9` pass;
- Sinkhorn OT public-transport rows: `0/9` pass;
- Sinkhorn OT local-stack rows: `0/9` pass;
- Gromov-Wasserstein public-transport rows: `0/9` pass;
- Gromov-Wasserstein local-stack rows: `0/9` pass;
- Relative Representations-style anchor-coordinate rows: `6/9` pass;
- RR-anchor local-stack rows: `0/9` pass;
- RR-anchor innovation rows: `3/9` pass;
- ranked RR-anchor innovation rows: `0/9` pass;
- no-normalization local residual rows: `3/9` pass;
- measured table rows: `20`;
- pending ICLR-required rows: `3`;
- ICLR competitor complete: `false`;
- COLM competitor table ready: `true`;
- packet record bytes: `11`;
- resident sparse decode p50: `5.231934 us/request`;
- source text/KV exposure for the live row: `false/false`.

Measured rows:

| Row | Status | Accuracy Range | Best Control Max | Exposure |
|---|---|---:|---:|---|
| target prior / no source | floor | `0.250-0.250` | - | no source text/KV |
| matched-byte structured text/log prefix | floor at 8B | `0.250-0.250` | - | source text exposed |
| random same-byte packet | floor | `0.234-0.260` | - | no source text/KV |
| candidate-local residual norm | passes controls | `0.500-0.625` | `0.260` | no source text/KV |
| global public-anchor dot product | fails controls | `0.625-0.875` | `0.625` | no source text/KV |
| public-calibration orthogonal Procrustes dot product | fails controls | `0.500-0.875` | `0.875` | no source text/KV |
| ridge CCA/SVCCA-style canonical-coordinate dot product | fails controls | `0.250-0.750` | `0.750` | no source text/KV |
| ridge CCA/SVCCA-style residual norm stack | fails controls | `0.500-0.750` | `0.750` | no source text/KV |
| LSTIRP-lite inverse-relative dot product | fails controls | `0.250-0.500` | `0.500` | no source text/KV |
| LSTIRP-lite inverse-relative residual norm stack | fails controls | `0.250-0.500` | `0.500` | no source text/KV |
| Sinkhorn OT public-calibration transport dot product | fails controls | `0.625-1.000` | `1.000` | no source text/KV |
| Sinkhorn OT transport with residual chart normalization | fails controls | `0.375-0.625` | `0.625` | no source text/KV |
| Gromov-Wasserstein public-calibration transport dot product | fails controls | `0.500-1.000` | `1.000` | no source text/KV |
| Gromov-Wasserstein transport with residual chart normalization | fails controls | `0.375-0.625` | `0.625` | no source text/KV |
| Relative Representations anchor-coordinate dot product | partial clean competitor | `0.250-1.000` | `0.264` | no source text/KV |
| RR-anchor local residual norm stack | fails controls | `0.375-0.750` | `0.375` | no source text/KV |
| RR-anchor innovation residual norm stack | fails controls | `0.250-0.750` | `0.375` | no source text/KV |
| ranked RR-anchor innovation residual norm stack | fails controls | `0.125-0.625` | `0.375` | no source text/KV |
| local residual without normalization | fails controls | `0.500-0.875` | `0.500` | no source text/KV |
| oracle candidate-atom packet | upper bound | `0.875-0.875` | - | oracle only |

Pending ICLR rows:

- C2C cache-to-cache fuser or scoped cache-access proxy;
- KVComm / Q-KVComm selective or compressed KV sharing;
- TurboQuant / KIVI / KVQuant / CacheGen byte-floor systems rows.

## Interpretation

This changes the common-basis story. A global shared dictionary, orthogonal
Procrustes rotation, ridge CCA canonical chart, LSTIRP-lite inverse-relative
projection, Sinkhorn OT/GW public transport, and no-normalization local chart
can score on matched packets, but their destructive controls rise or one
held-out direction collapses. A true Relative Representations-style
anchor-coordinate decoder is cleaner and passes `6/9`
rows, but fails every holdout-to-core row. The first naive RR plus
local-residual stack is worse (`0/9`) because controls rise again. The raw RR
innovation repair preserves only core-to-holdout (`3/9`) and leaks controls;
the ranked RR innovation repair passes `0/9` and drives holdout-to-core below
target. The
normalized candidate-local residual chart is still the only repeated
all-direction n512 survivor.

Layman explanation: rotating or correlating a shared dictionary can make both
real and fake clues look good, so it is not safe communication. One stronger
shared-coordinate system really works in some directions, so we should treat it
as a serious baseline. Our current method is less spectacular in one direction
but more stable across all directions.

## What Is Saturated

- Global public-anchor dot-product decoding is pruned as an unsafe headline
  method because controls leak on `9/9` n512 rows.
- Public-calibration orthogonal Procrustes is pruned as an unsafe headline
  method because the permuted-teacher control ties the matched row on `9/9`
  n512 rows.
- Ridge CCA/SVCCA-style canonical coordinates are pruned as unsafe headline
  methods because plain CCA passes `0/9`, and the local residual stack also
  passes `0/9` while leaking the permuted-teacher control.
- LSTIRP-lite inverse-relative projection is pruned as an unsafe headline
  method because plain and local-stack variants both pass `0/9` and leak the
  permuted-teacher control on the improved rows.
- Sinkhorn OT and Gromov-Wasserstein public transport are pruned as unsafe
  headline methods because dot and local-stack variants all pass `0/9`, with
  all `36` transport rows leaking through the permuted-teacher control.
- Candidate-local centering without row/payload normalization is also pruned as
  a headline method because only `3/9` rows pass.
- Simple RR residual repair is pruned: raw anchor-prior innovation passes only
  `3/9` rows and leaks controls, while rank-normalized innovation passes `0/9`
  and loses the holdout-to-core signal.
- Matched-byte 8B text/log prefixes remain at the target floor on this surface.

## What Is Alive

- Candidate-local residual normalization is the live positive receiver.
- Relative Representations-style anchor-coordinate decoding is alive as a
  partial clean competitor and possible stackable component.
- The naive RR-anchor local-residual stack is saturated/pruned because it
  reintroduces control leakage.
- Simple RR innovation residual stacks are saturated/pruned for the same reason:
  they do not rescue holdout-to-core under clean controls.
- Better packet encoders remain alive if they preserve the same strict control
  behavior.
- Native cache/KV systems baselines remain necessary, not solved.

## Remaining ICLR Gap

Comfortable ICLR still requires the RR holdout-to-core gap to be framed cleanly
or attacked with a materially different hypothesis, plus C2C/KVComm-style cache
baselines and KV-compression byte floors. The current table is a clean
COLM-facing baseline/limitation artifact, not a completed ICLR competitor
section.
