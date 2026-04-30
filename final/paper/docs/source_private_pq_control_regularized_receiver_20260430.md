# Source-Private PQ Control-Regularized Receiver

Date: 2026-04-30

## Status

Current readiness: COLM workshop remains strong; ICLR full is still blocked by
disjoint-safe learned receiver evidence.

Current paper story: source-private residual packets can carry conditional
evidence when the target has public decoder side information. The strongest
current systems contribution is the 7-byte PQ transport plus receiver
waterfall. The strongest method contribution remains the strict packet/control
protocol plus source-causal PQ/verifier evidence, not broad protocol-free
latent reasoning.

Exact blocker addressed here: reviewers can still object that the PQ receiver
is deterministic public-table decoding. This gate tests whether a learned
candidate scorer can preserve the deterministic PQ signal while explicitly
regularizing target-only, source-destroying, random, permuted, and deranged
public-table controls back to the target prior.

Layman explanation: the target learns a rule for when to trust the tiny source
code. A real code with the right public lookup table should help. A fake code,
code from the wrong example, or a correct code paired with a scrambled public
lookup table should be ignored.

## Method

Added:

- `scripts/run_source_private_pq_control_regularized_receiver.py`
- `scripts/summarize_source_private_pq_control_regularized_receiver.py`
- `tests/test_run_source_private_pq_control_regularized_receiver.py`
- `tests/test_summarize_source_private_pq_control_regularized_receiver.py`

The receiver is a ridge-trained candidate scorer over public candidate features
and PQ packet compatibility features. Training views:

- matched source PQ packet -> gold candidate,
- target-only -> target prior,
- label-shuffled source, constrained shuffled source, answer-masked source,
  permuted codes, random same-byte packets, and deranged public tables ->
  target prior.

The deranged public-table control uses the real source packet but permutes the
public candidate basis. It tests whether the receiver depends on the shared
packet/public-side-information contract rather than treating packet bytes as a
standalone answer ID.

## Runs

Strict first probe:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_pq_control_regularized_receiver.py \
  --output-dir results/source_private_pq_control_regularized_receiver_20260430/n256_remap101_utility_protected_hadamard \
  --train-examples 512 --eval-examples 256 --train-seed 30 --eval-seed 29 \
  --train-start-index 10000 --eval-start-index 0 --train-family-set all \
  --eval-family-set all --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --candidate-view slot --seed 30 --no-require-pass
```

Exact-overlap diagnostic matching the established PQ surface:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_pq_control_regularized_receiver.py \
  --output-dir results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap101_utility_protected_hadamard_lowcontrol \
  --train-examples 768 --eval-examples 500 --train-seed 29 --eval-seed 30 \
  --train-start-index 0 --eval-start-index 0 --train-family-set all \
  --eval-family-set all --diagnostic-table-mode legacy --feature-dim 512 \
  --budget-bytes 4 --variant utility_protected_hadamard --remap-slot-seed 101 \
  --candidate-view slot --seed 29 --matched-weight 12.0 --control-weight 0.25 \
  --target-weight 0.5 --deranged-weight 0.0 --random-rounds 0 --no-require-pass
```

Repeated for remaps `103/107`, plus a disjoint-ID low-control falsification.
Aggregate:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_pq_control_regularized_receiver.py \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n256_remap101_utility_protected_hadamard \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap101_utility_protected_hadamard \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap101_utility_protected_hadamard_lowcontrol \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap103_utility_protected_hadamard_lowcontrol \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n500_overlap_remap107_utility_protected_hadamard_lowcontrol \
  --run-dir results/source_private_pq_control_regularized_receiver_20260430/n500_disjoint_remap101_utility_protected_hadamard_lowcontrol \
  --output-dir results/source_private_pq_control_regularized_receiver_20260430/summary
```

## Results

Aggregate artifact:
`results/source_private_pq_control_regularized_receiver_20260430/summary/`

Headline:

- rows: `6`
- overlap pass rows: `3/4`
- disjoint pass rows: `0/2`
- min low-control overlap learned accuracy: `0.504`
- max low-control overlap best control accuracy: `0.298`
- min low-control overlap CI95 low versus best control: `+0.142`
- max disjoint deterministic L2 accuracy: `0.270`
- max disjoint learned accuracy: `0.264`

| Run | Disjoint | Remap | Learned | L2 | Target | Best control | Deranged | Pass |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| strict n256 plausible-decoy | yes | 101 | 0.250 | 0.270 | 0.250 | label-shuffled 0.250 | 0.250 | false |
| overlap default-regularized | no | 101 | 0.250 | 0.504 | 0.250 | label-shuffled 0.250 | 0.250 | false |
| overlap low-control | no | 101 | 0.504 | 0.504 | 0.250 | answer-masked 0.268 | 0.180 | true |
| overlap low-control | no | 103 | 0.504 | 0.504 | 0.250 | random 0.298 | 0.158 | true |
| overlap low-control | no | 107 | 0.516 | 0.516 | 0.250 | permuted 0.262 | 0.124 | true |
| disjoint low-control | yes | 101 | 0.264 | 0.264 | 0.250 | permuted 0.288 | 0.234 | false |

## Interpretation

Promote narrowly:

- A learned score adapter can preserve deterministic utility-protected-Hadamard
  PQ reception on the established n500 exact-overlap surface across remaps.
- Deranged public-table controls collapse below target or near target on those
  overlap rows, so the learned receiver is not simply accepting any packet
  bytes.
- This is a useful control-regularized receiver diagnostic and a clean
  implementation scaffold for the next branch.

Do not promote as an ICLR headline:

- Disjoint train/eval IDs collapse the underlying PQ signal itself
  (`L2 <= 0.270`), and learned reception cannot rescue it.
- The exact-overlap rows have train/eval ID intersections equal to eval size.
  They are legitimate diagnostics for packet/basis mechanics, but not enough
  for a comfortable ICLR claim.
- Strong default control regularization collapses the receiver to target-only
  even when deterministic PQ is positive. The linear score adapter is
  separability-limited.

## Readiness Impact

This sharpens the paper rather than closing the ICLR gate. The current strong
technical contributions remain:

1. strict source-private packet/control protocol,
2. frozen target-verifier packet consumption,
3. geometry-mitigated PQ residual packets and lookup-risk diagnostics,
4. 7-byte PQ transport plus exact receiver systems waterfall.

This branch adds:

5. a bounded learned PQ receiver diagnostic showing the receiver can preserve
   PQ only on exact-overlap surfaces, while disjoint IDs reveal the next
   scientific blocker.

COLM workshop: strengthened, because the paper can honestly report the learned
receiver boundary and avoid overclaiming.

ICLR full: still needs a disjoint-safe source-private packet or learned
connector. The next method must construct a packet basis that generalizes
across held-out IDs, not merely across remapped candidate slots.

## Next Gate

The next highest-value method branch is not more tuning of this linear adapter.
It should change the source encoder/interface:

- conditional innovation encoder: encode `source matched - source
  answer-masked` or `source - target-prior candidate` rather than the marginal
  source projection,
- TurboResidual PQ/QJL packet: PQ centroid plus residual signs, with
  random-sign and deranged-public controls,
- or a small Perceiver/Q-Former-style packet connector trained with explicit
  control loss and evaluated first on disjoint IDs.

Pass bar for the next branch:

- disjoint-ID n256/n500 learned source accuracy at least `target + 0.15`,
- best destructive/deranged control no higher than `target + 0.05`,
- paired CI95 low versus best control above `+0.10`,
- and receiver/runtime bytes reported against the existing 7-byte systems
  waterfall.

## Tests

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/run_source_private_pq_control_regularized_receiver.py \
  scripts/summarize_source_private_pq_control_regularized_receiver.py

./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_pq_control_regularized_receiver.py \
  tests/test_summarize_source_private_pq_control_regularized_receiver.py -q
```

Outcome: `4 passed in 0.31s`.
