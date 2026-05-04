# HellaSwag Qwen-To-Phi Protected Rival Packet Gate

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible; ICLR full remains
blocked.

Current story: source-private byte-scale packets can carry useful Qwen
decision information under strict controls, but the open ICLR problem is a
positive cross-family receiver that improves beyond the fixed Qwen hybrid on a
larger frozen slice.

Exact blocking gap: directly exposing Qwen's top rival still does not give Phi
a train/select-stable rule for choosing when to override the protected hybrid
default.

## Gate

Implemented and ran:

- `scripts/build_source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate.py`
- `results/source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate_20260504_validation1024_2048/`
- references:
  `references/692_hellaswag_qwen_to_phi_protected_rival_refs_20260504.md`

The gate uses the cached Qwen packet rows, Phi receiver-local target-score
caches, and aligned Qwen source score cache for HellaSwag validation
`1024:2048`. The packet names either:

- the fixed Qwen hybrid plus the strongest Qwen source-score rival; or
- Qwen source-score top-2.

The receiver sees only the byte-scale packet and Phi's local score simplex. It
does not see source text, source KV, raw hidden vectors, raw source scores, or
source logits.

Split:

- fit: first `64` rows per `512`-row cached slice;
- select: next `64` rows per slice;
- eval: remaining `384` rows per slice, `768` held-out rows total.

## Result

The gate fails.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
|---|---:|---:|---:|
| hybrid-rival oracle diagnostic | `0.678385` | `+0.210938` | `+0.182292` |
| source top-2 oracle diagnostic | `0.675781` | `+0.208333` | `+0.178385` |
| Phi target top-2 pair control | `0.511719` | `+0.044271` | `-0.003906` |
| fixed Qwen hybrid | `0.467448` | reference | reference |
| selected protected pair decoder | `0.462240` | `-0.005208` | `-0.013021` |
| fit+select diagnostic decoder | `0.462240` | `-0.005208` | `-0.013021` |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` |
| Qwen source-score top-1 | `0.411458` | `-0.055990` | `-0.079427` |
| Phi argmax within hybrid-rival pair | `0.329427` | `-0.138021` | `-0.177116` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.250000` |

The selected train/select decoder used `code16_hybrid_rival_policy`, `2B` raw /
`5B` framed, with ridge L2 `3000.0`. On eval it made `9` overrides: `2` helped
and `6` harmed relative to fixed hybrid.

Per-slice result:

| Slice | Rows | Decoder Acc. | Fixed Hybrid Acc. | Delta |
|---|---:|---:|---:|---:|
| `1024:1536` | `384` | `0.484375` | `0.486979` | `-0.002604` |
| `1536:2048` | `384` | `0.440104` | `0.447917` | `-0.007812` |

Controls are informative. Source-row shuffle, label permutation, code-value
permutation, random same-byte, and destructive pair controls collapse. However,
source-score-row shuffle and candidate-roll controls tie fixed hybrid because
the selected conservative model often defaults back to the hybrid. This is not
a leakage concern, but it means the method does not pass a positive source-use
gate.

## Interpretation

This is a stronger negative than the prior oracle-switch gate. The prior gate
could have failed because the rival candidate itself was not transmitted. Here
the packet directly exposes the protected decision frontier, and the
hybrid-rival oracle is very high: `0.678385`, with `162` possible helps and no
harms when defaulting to the fixed hybrid outside the pair oracle.

The failure therefore localizes the bottleneck: the shared candidate-ID basis
contains large source-private headroom, but a tiny fit/select receiver trained
on only `128` fit plus `128` select rows cannot identify when the rival should
replace the protected hybrid. This weakens another small-slice switcher family
and promotes a larger official-train source-code dictionary or a different
receiver interface as the next live branch.

## Contribution Status

Promote:

- protected top-rival packets as a well-controlled falsification and oracle
  decomposition;
- the candidate-ID shared-basis framing as the narrow novelty boundary;
- the systems analogy to unequal protection: spend protected bits on
  decision-frontier fields, not uniform reconstruction.

Weaken:

- small fit/select pairwise decoders;
- deterministic Phi argmax within source-provided pairs;
- Qwen source top-1/top-2 as a direct replacement for the fixed hybrid default;
- more shallow handcrafted switchers on the same cached surface.

Still alive:

- a larger official-train, cross-fitted source-code dictionary;
- receiver interfaces that use more calibration data or richer candidate
  features;
- a strict NVIDIA systems table once native GPU hardware exists.

## Lay Explanation

Qwen often has the right answer among its two favorite choices. We sent Phi the
safe Qwen answer plus Qwen's strongest alternative, then asked Phi to pick
between those two. The right answer is often in that pair, but Phi still picks
the wrong member of the pair too often. So the problem is not just "send the
rival"; we need a better way to teach the receiver when that rival should win.

## Next Gate

Do not continue small-slice handcrafted switchers on Qwen-to-Phi. The next
highest-value Mac-local method branch is an official-train source-code
dictionary: train cross-fitted pair/rival utility codes on a larger
official-train calibration surface, then freeze the dictionary and evaluate on
the same validation `1024:2048` and at least one disjoint validation slice with
source-row shuffle, source-score shuffle, candidate-roll, code permutation,
label permutation, target-derived, and same-byte controls.
