# Source-Private ARC-Challenge Anchor-Relative Basis, 2026-05-01

## Status

- validation artifact:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation/`
- test artifact:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test/`
- code:
  `scripts/run_source_private_arc_challenge_fixed_packet_gate.py` and
  `scripts/build_source_private_arc_challenge_seed_stability.py`
- references:
  `references/572_arc_challenge_anchor_relative_basis_refs_20260501.md`
- control follow-up:
  `paper/source_private_arc_challenge_anchor_controls_20260501.md`

## Method

The source-choice cache is the same answer-key-forbidden Qwen2.5-0.5B
choice-log-likelihood cache used by the shared hashed-basis ARC endpoint. The
new receiver-side decision surface changes the basis: instead of using direct
hashed coordinates, each question/candidate text is represented by its
similarities to a deterministic public set of train-split question/candidate
anchors, then centered, normalized, projected, and packed into the same fixed
`12B` packet.

This tests the common-basis claim more directly. A direct hashed basis can be
criticized as lexical feature engineering; an anchor-relative basis asks
whether a public coordinate chart over examples preserves enough geometry for
the same fixed-byte source-private packet to work.

## Results

Projection-seed stability passes on both frozen ARC splits:

| Split | Seeds | Pass | Matched mean/min/max | Target | Same-byte text | Min lift vs target | Min lift vs best control | Min lift vs text | Min CI95 low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 5 | 5/5 | 0.386 / 0.381 / 0.388 | 0.244 | 0.348 | 0.137 | 0.127 | 0.033 | 0.065 |
| test | 5 | 5/5 | 0.344 / 0.344 / 0.345 | 0.265 | 0.311 | 0.078 | 0.067 | 0.033 | 0.039 |

The test result is slightly below the direct hashed shared-basis endpoint in
some seeds, but it preserves the important gates: all five seeds pass, same-byte
text stays lower, destructive controls stay lower, and the paired CI lower
bound versus target remains positive.

## Interpretation

This strengthens the common-basis story: LatentWire does not require a private
receiver-only embedding basis, and it also does not require direct lexical
hashed coordinates. A public anchor-relative coordinate chart is enough to carry
the fixed-byte packet on ARC.

This should not be overclaimed as an anchor-relative superiority result. The
direct hashed shared-basis endpoint is at least as strong on the current ARC
rows, so the correct claim is robustness to a more interpretable public
coordinate chart, not a new best-performing basis.

The follow-up anchor controls refine this further. Anchor-ID and anchor-value
mismatch controls collapse to target-level accuracy on validation and test,
showing that source and receiver must agree on the coordinate chart. Shared
random anchors still pass, showing that semantic train-anchor values are not
necessary on ARC.

The safe paper claim is still narrow: public-basis source-private
candidate-disambiguation packets on ARC, not universal cross-model latent
communication. The source decision remains a Qwen log-likelihood cache, so ICLR
still needs either a second public benchmark or a stronger learned hidden-state
endpoint.

## Next Gate

Run anchor-specific controls: shuffle anchor identities, shuffle anchor values,
replace train anchors with random anchors of the same count, and compare
directly against the hashed basis under paired uncertainty. Then run one strict
cross-family source/receiver falsification pair or repeat the public-basis
endpoint on a second public MCQ benchmark. Native NVIDIA/vLLM
TTFT/TPOT/goodput/HBM rows remain the systems blocker.
