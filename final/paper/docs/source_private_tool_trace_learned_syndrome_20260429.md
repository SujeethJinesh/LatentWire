# Source-Private Tool-Trace Learned Syndrome

- date: `2026-04-29`
- gate: `source_private_tool_trace_learned_syndrome_20260429`
- status: passed real-feature learned syndrome smoke on two seed pairs at a common 6-byte budget
- scale rung: strict-small learned-method confirmation on Mac-local CPU

## Current Readiness

This is the strongest new method contribution since the diagnostic-packet
protocol. The learned-syndrome branch now uses real hidden-repair tool-trace and
candidate-text features rather than synthetic candidate latents. It should not
yet replace the main result, but it is now a plausible second method
contribution if it survives a larger frozen slice and compression-native
baselines.

## Method

The source feature is a hashed bag-of-token vector over the private hidden-test
log. The target side has hashed candidate-text features from public patch
metadata. A ridge-fitted encoder maps private source features to the target
candidate feature space. The packet is a bit-packed random-hyperplane syndrome
over the predicted target feature; the decoder compares the packet against
candidate-side binary codes by Hamming distance.

The strict control fix matters: the answer-masked source now masks
`REPAIR_DIAG`, hidden input, expected/actual values, failure status, test name,
and `repair_family`. An earlier run failed because `repair_family=...` leaked
source-private family identity into the masked control.

## Results

Seed pair `29 -> 30`, train/eval `512/256`, all families:

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `0.500` | `0.250` | `0.379` | `+0.121` | `0.633` |
| 2 | `False` | `0.738` | `0.250` | `0.410` | `+0.328` | `0.922` |
| 4 | `False` | `0.895` | `0.250` | `0.352` | `+0.543` | `1.000` |
| 6 | `True` | `0.945` | `0.250` | `0.285` | `+0.660` | `1.000` |
| 8 | `True` | `0.980` | `0.250` | `0.250` | `+0.730` | `1.000` |
| 12 | `False` | `0.996` | `0.250` | `0.277` | `+0.719` | `1.000` |

Seed pair `31 -> 32`, train/eval `512/256`, all families:

| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full diag oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | `True` | `0.918` | `0.250` | `0.289` | `+0.629` | `1.000` |
| 12 | `True` | `0.977` | `0.250` | `0.250` | `+0.727` | `1.000` |

The common promoted row is 6 bytes: it passes on both seed pairs and stays
below the source-destroying control tolerance. Larger budgets are not
monotonic; some source-free binary controls collide with candidate codes.

## Controls

Rows include target-only, zero-source, shuffled-source, answer-masked source,
random same-byte, target-derived sidecar, answer-only, structured text matched,
wrong-projection source, and full diagnostic oracle.

Pass rule:

- matched learned syndrome must beat best no-source by at least `15` points;
- every source-destroying control must stay within `target_only + 0.05`;
- full diagnostic oracle must be at least `0.95`;
- exact-ID parity must hold.

## Interpretation

This result turns learned syndrome packets from synthetic smoke into a real
tool-trace/candidate-text method candidate. It is still not final ICLR evidence:
the feature extractor is a hashed text featurizer, and there are no
compression-native baselines yet. But it directly addresses the "hand-coded
packet only" criticism.

## Next Gate

Run a compression-native baseline on the same frozen feature tensors:

- random sign sketch;
- rotation plus scalar quantization;
- optional QJL-style sign residual;
- same 6-byte budget and same source-destroying controls.

Promote learned syndrome only if it beats these generic compression baselines
or shows a cleaner systems/interpretability tradeoff.

## Artifacts

- `scripts/run_source_private_tool_trace_learned_syndrome.py`
- `tests/test_run_source_private_tool_trace_learned_syndrome.py`
- `results/source_private_tool_trace_learned_syndrome_20260429/summary.json`
- `results/source_private_tool_trace_learned_syndrome_20260429_seed31/summary.json`
