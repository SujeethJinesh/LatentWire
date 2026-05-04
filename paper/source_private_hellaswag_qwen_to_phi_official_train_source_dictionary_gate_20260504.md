# HellaSwag Qwen-To-Phi Official-Train Source Dictionary Gate

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible; ICLR full remains
blocked.

Current story: LatentWire has a strict source-private candidate-packet protocol
and large candidate-frontier oracle headroom, but the learned receiver/source
dictionary still fails to convert that headroom into a positive Qwen-to-Phi
method on a larger frozen slice.

Exact blocking gap: the official-train source-side dictionary does not beat the
fixed Qwen hybrid packet on held-out Qwen-to-Phi validation. We still need a
positive method that survives larger frozen slices, seed repeats, and at least
one strict cross-family falsification pair.

## Gate

Implemented and ran:

- `scripts/build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate.py`
- `results/source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate_20260504_validation1024_2048/`
- references:
  `references/693_hellaswag_qwen_to_phi_official_train_source_dictionary_refs_20260504.md`

The gate trains only on out-of-bag official-train Qwen rows. For each row, the
default candidate is the fixed Qwen hybrid. The rival is Qwen source-score
top-1 unless top-1 already equals the hybrid, in which case the rival is
top-2. The learned dictionary predicts whether the rival should replace the
hybrid from source-side features only: score z-values, source margins,
rival-minus-hybrid gaps, policy agreement bits, and candidate IDs.

The dictionary is frozen after official-train fit/dev selection, then evaluated
on cached Qwen-to-Phi HellaSwag validation `1024:2048`. Phi scores are used
only for target-only/comparator rows, not for training or selecting the source
dictionary.

## Result

The gate fails.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
|---|---:|---:|---:|
| hybrid-rival oracle diagnostic | `0.678385` | `+0.210938` | `+0.182292` |
| eval-label best threshold diagnostic | `0.468750` | `+0.001302` | `0.000000` |
| fixed Qwen hybrid | `0.467448` | reference | reference |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` |
| official-train source dictionary | `0.429688` | `-0.037760` | `-0.063802` |
| Qwen source-score top-1 | `0.411458` | `-0.055990` | `-0.079427` |
| official-train label permutation control | `0.360677` | `-0.106771` | `-0.147135` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.250000` |

Official-train selection looked weakly positive but statistically unsafe:
`0.518817` official-dev accuracy, delta `+0.008065`, CI95 low `-0.032258`.
When frozen and evaluated on Qwen-to-Phi, the dictionary made `194` overrides:
`41` helped and `70` harmed relative to fixed hybrid.

Per-slice result:

| Slice | Rows | Dictionary Acc. | Fixed Hybrid Acc. | Delta |
|---|---:|---:|---:|---:|
| `1024:1536` | `384` | `0.450521` | `0.486979` | `-0.036458` |
| `1536:2048` | `384` | `0.408854` | `0.447917` | `-0.039062` |

The source-private packet remains byte-scale: `2` raw payload bits, `1B` stored
payload, `4B` framed record, no source text, no source KV, no raw source hidden
state, and no raw source score/logit vector. The frozen dictionary has `464B`
of static weights in this implementation, amortized to `0.604B` per eval
request. Because quality is negative, these byte numbers are context only, not
a systems win.

## Interpretation

This result weakens the larger-data source-only utility dictionary branch. The
candidate-ID basis still has large headroom: the hybrid-rival oracle reaches
`0.678385`, `+0.210938` over fixed hybrid. But the official-train source
surface does not generalize to Qwen-to-Phi receiver behavior well enough to
choose the rival safely.

The failure localizes the blocker more sharply than the previous protected
rival gate. The problem is not only tiny fit/select data: even `1,487`
out-of-bag official-train calibration rows lead to a dictionary that overuses
the rival on the held-out cross-family surface. This suggests the next live
branch needs receiver-side calibration, richer side-information, or a different
interface, not another source-only scalar threshold over the same Qwen score
geometry.

## Contribution Status

Promote:

- strict source-private packet evaluation with destructive controls;
- candidate-ID decision-frontier communication as the shared basis;
- systems/exposure accounting that cleanly separates bytes from native serving
  speed claims.

Weaken:

- small fit/select switchers;
- protected-rival pair decoders;
- larger official-train source-only utility dictionaries;
- deterministic Qwen source top-1/top-2 replacement of the protected hybrid.

Still alive:

- receiver-side calibration with Phi train scores, if we can create a clean
  official-train Phi cache;
- richer candidate-pair interfaces that expose receiver-checkable evidence
  without leaking source scores;
- mixed-precision/error-correcting packet fields, but only if they improve
  held-out quality rather than just shrinking bytes;
- native NVIDIA systems measurements after a positive method exists.

Cut if necessary:

- Fourier score-contrast variants and shallow handcrafted switchers on the
  same Qwen-to-Phi cached rows;
- claims that the method beats KV/hidden-state quantization systems. With a
  negative quality row, QJL/KIVI/KVQuant/TurboQuant belong only in the threat
  model and byte-floor context.

## Lay Explanation

We gave Qwen many training questions and asked it to learn when its backup
answer should replace its safe answer. That looked a little better on Qwen's
own training-dev split. But when we froze the rule and used it for Qwen-to-Phi
communication, it swapped too often and made more bad changes than good ones.

## Next Gate

Do not keep widening source-only dictionaries on this same surface. The next
highest-value gate is a receiver-calibrated interface: generate or obtain a
clean Phi official-train score cache, then learn a Qwen-packet/Phi-side
selective receiver under official-train-only selection and rerun the same
validation `1024:2048` gate with label permutation, row shuffle, candidate
roll, code permutation, same-byte random packet, and target-derived controls.
