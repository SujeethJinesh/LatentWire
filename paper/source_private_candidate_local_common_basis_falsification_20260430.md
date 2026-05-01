# Candidate-Local Common-Basis Falsification

- date: `2026-04-30`
- status: same-slice reviewer ablation for the live candidate-local residual
  receiver
- code:
  `scripts/build_source_private_candidate_local_common_basis_falsification.py`
- artifact:
  `results/source_private_candidate_local_common_basis_falsification_20260430/`
- references:
  `references/549_candidate_local_common_basis_falsification_refs_20260430.md`

## What Changed

The live candidate-local residual receiver now has direct n512 ablations
against implemented common-basis decoders: global public semantic anchor
dot-product scoring, public-calibration orthogonal Procrustes scoring, a
ridge CCA/SVCCA-style canonical-coordinate decoder, LSTIRP-lite
inverse-relative projection, Sinkhorn OT/Gromov-Wasserstein public transport,
and a Relative Representations-style anchor-coordinate decoder. It now also
records two guarded RR repair probes: local anchor-prior subtraction and
rank-normalized anchor-prior subtraction. The ablations use the same
seeds (`47/53/59`), directions (`core_to_holdout`,
`holdout_to_core`, `same_family_all`), budget (`8B`), public calibration split,
and MiniLM feature model.

## Main Evidence

Summary artifact:
`results/source_private_candidate_local_common_basis_falsification_20260430/candidate_local_common_basis_falsification.md`

Headline:

- pass gate: `false`;
- live candidate-local residual-normalized rows: `9/9` pass;
- global common-basis dot-product rows: `0/9` pass;
- global common-basis control-leak rows: `9/9`;
- public-calibration orthogonal Procrustes rows: `0/9` pass;
- Procrustes control-leak rows: `9/9`;
- ridge CCA/SVCCA-style canonical-coordinate rows: `0/9` pass;
- ridge CCA/SVCCA-style control-leak rows: `6/9`;
- ridge CCA/SVCCA-style local residual stack rows: `0/9` pass;
- LSTIRP-lite inverse-relative rows: `0/9` pass;
- LSTIRP-lite inverse-relative control-leak rows: `6/9`;
- LSTIRP-lite inverse-relative local-stack rows: `0/9` pass;
- LSTIRP-lite inverse-relative local-stack control-leak rows: `6/9`;
- Sinkhorn OT public-transport rows: `0/9` pass;
- Sinkhorn OT public-transport control-leak rows: `9/9`;
- Sinkhorn OT local-stack rows: `0/9` pass;
- Sinkhorn OT local-stack control-leak rows: `9/9`;
- Gromov-Wasserstein public-transport rows: `0/9` pass;
- Gromov-Wasserstein public-transport control-leak rows: `9/9`;
- Gromov-Wasserstein local-stack rows: `0/9` pass;
- Gromov-Wasserstein local-stack control-leak rows: `9/9`;
- Relative Representations-style anchor-coordinate rows: `6/9` pass;
- Relative Representations-style control-leak rows: `0/9`;
- RR-anchor plus local residual-normalization stack rows: `0/9` pass;
- RR-anchor innovation residual rows: `3/9` pass, with `6/9` control-leak
  rows;
- ranked RR-anchor innovation residual rows: `0/9` pass, with `3/9`
  control-leak rows;
- candidate-local residual without row/payload normalization rows: `3/9` pass;
- max global matched-packet accuracy: `0.875`;
- max global best-control accuracy: `0.625`;
- max Procrustes matched-packet accuracy: `0.875`;
- max Procrustes best-control accuracy: `0.875`;
- max ridge CCA/SVCCA-style matched-packet accuracy: `0.750`;
- max ridge CCA/SVCCA-style best-control accuracy: `0.750`;
- max ridge CCA/SVCCA-style local-stack best-control accuracy: `0.750`;
- max LSTIRP-lite matched-packet accuracy: `0.500`;
- max LSTIRP-lite best-control accuracy: `0.500`;
- max LSTIRP-lite local-stack matched-packet accuracy: `0.500`;
- max LSTIRP-lite local-stack best-control accuracy: `0.500`;
- max Sinkhorn OT matched-packet accuracy: `1.000`;
- max Sinkhorn OT best-control accuracy: `1.000`;
- max Sinkhorn OT local-stack matched-packet accuracy: `0.625`;
- max Sinkhorn OT local-stack best-control accuracy: `0.625`;
- max Gromov-Wasserstein matched-packet accuracy: `1.000`;
- max Gromov-Wasserstein best-control accuracy: `1.000`;
- max Gromov-Wasserstein local-stack matched-packet accuracy: `0.625`;
- max Gromov-Wasserstein local-stack best-control accuracy: `0.625`;
- max relative-anchor matched-packet accuracy: `1.000`;
- max relative-anchor best-control accuracy: `0.264`;
- max RR-local-stack best-control accuracy: `0.375`;
- max RR-innovation matched-packet accuracy: `0.750`;
- max RR-innovation best-control accuracy: `0.375`;
- max ranked RR-innovation matched-packet accuracy: `0.625`;
- max ranked RR-innovation best-control accuracy: `0.375`;
- max live best-control accuracy: `0.260`.

Key interpretation: the global dot-product, Procrustes, and ridge CCA/SVCCA
decoders often get high matched accuracy, but none is a valid source-private
communication row because source-destroying controls rise with them or one
direction collapses. Procrustes is especially clear: the permuted-teacher
receiver ties the matched row in all `9/9` rows. Plain ridge CCA reaches
`0.750` in core-to-holdout but collapses to target in holdout-to-core; the
ridge CCA plus local residual stack recovers holdout-to-core accuracy but leaks
the permuted-teacher control in every row. The LSTIRP-lite inverse-relative
projection is weaker than RR here: it reaches only `0.500` matched accuracy and
the permuted-teacher control ties the improved rows (`6/9` control-leak rows),
even after local residual normalization. The Relative Representations-style
anchor-coordinate decoder is different: controls remain clean, and it is a real
partial competitor (`6/9` rows), but it collapses to the target floor in all
holdout-to-core rows. Sinkhorn OT and Gromov-Wasserstein transport are also
unsafe on this surface: dot-product transport reaches matched accuracy up to
`1.000`, but the permuted-teacher receiver ties it (`9/9` control-leak rows);
the local residual transport stacks reduce matched accuracy to at most `0.625`
and still leak. The live candidate-local residual-normalized receiver is still
the only repeated all-direction survivor. The naive RR-anchor plus local
residual-normalization stack is pruned (`0/9`) because destructive controls rise
again. The two guarded RR repair probes do not fix the RR gap: raw anchor-prior
innovation keeps core-to-holdout alive but fails all holdout-to-core rows and
leaks controls in `6/9` rows; rank-normalized innovation passes `0/9`, drives
holdout-to-core below target (`0.125`), and still leaks controls in
core-to-holdout.

The completed no-normalization ablation also fails the full repeated gate:
only `3/9` seed/direction rows pass. Local candidate centering alone is not the
method; row and payload normalization are part of the source-private control
barrier.

## Why This Matters

This directly addresses the "is this just a shared/common basis?" reviewer
concern, but it does not close it. A global anchor space and an orthogonal
public calibration rotation are unsafe because they also let permuted-teacher
and atom-deranged controls score well. Ridge CCA/SVCCA-style canonical
coordinates and this LSTIRP-lite inverse-relative projection are also unsafe on
this surface. Sinkhorn OT and GW public transport are unsafe for the same reason:
they align enough public geometry to score the matched packet, but also align
the permuted-teacher control. A true relative-anchor representation is much
stronger and should now be treated as a baseline or stackable component. The
current positive contribution is the all-direction normalized candidate-local
residual chart, not a universal latent basis or public transport map.

The first direct stack of RR coordinates with local residual normalization is
not enough: it improves some matched rows but reintroduces fake-clue leakage.
The two guarded RR innovation variants are also pruned: one leaks while
preserving only the easy direction, and the ranked version discards too much
useful magnitude information.

Layman explanation: a simple shared dictionary, rotation, or correlation map can
often guess the right answer, but fake clues also work on it. A stronger
anchor-coordinate dictionary is clean and sometimes excellent, but fails in one
transfer direction. OT/GW transport looks very strong until we ask fake
receivers to use the same public map; then the fake receiver works too. The live
method is more balanced across directions.

## Safe Claim

On the current n512 held-out surface, a global public-anchor decoder is an
unsafe common-basis baseline: it reaches high matched accuracy but fails all
`9/9` seed/direction rows because source-destroying controls leak. A
public-calibration orthogonal Procrustes decoder is also unsafe: it reaches
matched accuracy up to `0.875`, but its best control also reaches `0.875`. The
ridge CCA/SVCCA-style decoder is also unsafe: plain CCA passes `0/9`, and its
local residual stack passes `0/9` because permuted-teacher controls rise. The
LSTIRP-lite inverse-relative decoder is also unsafe on this packet surface:
plain and local-stack variants both pass `0/9`, with matched rows tied by
permuted-teacher controls. Sinkhorn OT and Gromov-Wasserstein transport are also
unsafe: both dot and local-stack variants pass `0/9`, and all `36` transport
rows leak through the permuted-teacher receiver. The
Relative Representations-style anchor-coordinate decoder is a partial clean
competitor with `6/9` pass rows. RR innovation repairs pass at most `3/9` rows
and do not solve holdout-to-core. The live candidate-local residual-normalized
receiver is the only repeated n512 row group in this comparison that passes all
directions.

## Non-Claims

- This is not full dense LSTIRP, C2C, or KVComm.
- The OT/GW rows are public atom-axis calibration transport baselines, not full
  dense hidden-state OT over token activations.
- The Procrustes row is an orthogonal public-calibration packet receiver, not a
  CCA row.
- The ridge CCA row is a canonical-coordinate packet receiver, not CKA-only
  representational similarity or a full model-stitching experiment.
- The LSTIRP-lite row is an inverse-relative packet receiver, not a full
  dense latent-transfer reproduction.
- This RR row is an anchor-coordinate public-calibration baseline for this
  packet benchmark, not a full dense latent-transfer reproduction.
- This does not prove the live row has higher raw matched accuracy than every
  unsafe baseline.
- This is a Mac-local falsification table, not an NVIDIA/vLLM systems result.

## Remaining ICLR Gap

The next method table should not spend more time on simple RR residual repairs.
The highest-value path is either a materially different RR-family hypothesis
that explains the holdout-to-core asymmetry before implementation, or a clean
framing that reports RR as a one-way partial competitor while the paper moves
to native/proxy C2C, KVComm, and KV-compression systems rows.

## COLM Workshop Use

This is strong enough for a COLM reviewer-facing ablation if the paper claim is
kept narrow and honest: source-private packet communication through receiver
side information, with RR anchor coordinates now reported as a serious partial
baseline rather than defeated prior work.
