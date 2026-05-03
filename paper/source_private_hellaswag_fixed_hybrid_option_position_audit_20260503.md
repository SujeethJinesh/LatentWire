# HellaSwag Fixed Hybrid Option-Position Audit

Date: 2026-05-03

## Status

Current paper readiness: COLM workshop plausible; ICLR full still blocked.
Current story: fixed-byte source-private packets can carry HellaSwag task
evidence under strict destructive controls. Exact ICLR blocker: the packet row
still needs stronger bias controls, learned receiver/common-basis evidence, and
native systems measurements before broad ICLR claims.

This audit addresses a reviewer-critical weakness in the current HellaSwag
packet story: a one-candidate packet could be mistaken for answer-position
exploitation in a multiple-choice benchmark.

## Artifact

- `results/source_private_hellaswag_fixed_hybrid_option_position_audit_20260503_validation0_10042/`
- `scripts/build_source_private_hellaswag_fixed_hybrid_option_position_audit.py`
- `tests/test_build_source_private_hellaswag_fixed_hybrid_option_position_audit.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_fixed_hybrid_option_position_audit.py \
  --run-date 2026-05-03
```

## Result

The cached option-position audit passes on the full cached HellaSwag validation
surface `0:10042`.

| Readout | Value |
| --- | ---: |
| eval rows | `10042` |
| candidate-only accuracy | `0.526688` |
| fixed hybrid accuracy | `0.532464` |
| overall delta vs candidate-only | `+0.005776` |
| overall CI95 low | `+0.002888` |
| answer-balanced delta vs candidate-only | `+0.005779` |
| answer-balanced CI95 low | `+0.002984` |
| positive answer positions | `4 / 4` |
| max fixed-hybrid prediction shift from answer distribution | `0.013045` |
| worst fixed-hybrid cyclic-roll accuracy | `0.157040` |
| best non-identity global packet permutation accuracy | `0.348636` |
| best rowwise derangement accuracy | `0.162517` |
| best rowwise random permutation accuracy | `0.259908` |
| max equivariance sanity difference | `0.000000` |

Per-answer-position readout:

| Gold answer slot | Rows | Candidate-only | Fixed hybrid | Delta | CI95 low |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `2515` | `0.531213` | `0.537972` | `+0.006759` | `+0.000398` |
| `1` | `2485` | `0.508652` | `0.516298` | `+0.007646` | `+0.002414` |
| `2` | `2584` | `0.538700` | `0.543344` | `+0.004644` | `-0.000774` |
| `3` | `2458` | `0.527665` | `0.531733` | `+0.004068` | `-0.002441` |

Fixed-hybrid cyclic-roll controls:

| Control | Accuracy | Delta vs fixed hybrid | CI95 high |
| --- | ---: | ---: | ---: |
| roll by `1` | `0.155148` | `-0.377315` | `-0.362378` |
| roll by `2` | `0.157040` | `-0.375423` | `-0.361673` |
| roll by `3` | `0.155348` | `-0.377116` | `-0.362771` |

Packet-ID permutation controls:

| Control | Accuracy | Delta vs fixed hybrid | CI95 high |
| --- | ---: | ---: | ---: |
| best non-identity global label permutation `0 3 2 1` | `0.348636` | `-0.183828` | `-0.173170` |
| best rowwise random derangement | `0.162517` | `-0.369946` | `-0.354802` |
| best rowwise random permutation | `0.259908` | `-0.272555` | `-0.259209` |

Same-permutation equivariance sanity checks over all `24` label permutations
have max absolute accuracy difference `0.0`, verifying that the audit code
preserves accuracy when answers and emitted packet IDs are permuted together.

Pass gate: `True`.

## Interpretation

This weakens the simplest option-position critique. The fixed hybrid gain is
not concentrated in a single answer slot: all four gold answer slots have
positive mean deltas, and a uniform-by-answer-position bootstrap remains
positive with CI95 low `+0.002984`. The prediction distribution is also close
to the gold answer distribution, with max slot shift `0.013045`. Direct cyclic
rolls of the fixed hybrid packet collapse to roughly `0.155`, non-identity
global packet-label permutations peak at only `0.348636`, and rowwise
derangements peak at `0.162517`, all far below the true fixed hybrid row.

This does not close the full candidate-order concern. The source model was not
rerun with the candidate texts physically permuted, so this audit should be
reported as cached option-position hardening rather than permutation-invariant
evaluation. The next stronger control is a smaller but real rerun where each
HellaSwag row is evaluated under several answer-option permutations and the
packet is remapped back to canonical option IDs.

Lay explanation: we checked whether the tiny hybrid hint only helps when the
right answer is in a particular position, like choice A. It helps a little for
all four answer positions. If we rotate or remap the hint to wrong option
numbers, accuracy collapses. That makes the result less likely to be just an
answer-slot trick, but we still need a future test that reruns the model after
actually shuffling the answer choices.

## Decision

- Promote this as cached option-position hardening for the full-validation
  fixed hybrid packet row.
- Do not claim true candidate-permutation invariance yet.
- Keep the current full-paper blocker: learned receiver/common-basis evidence
  and native systems rows remain missing.
- Next exact HellaSwag evaluation gate should be a small true candidate-text
  permutation rerun on a frozen slice if local compute permits; otherwise move
  to a stronger receiver/common-basis feature source and keep this limitation
  explicit.
