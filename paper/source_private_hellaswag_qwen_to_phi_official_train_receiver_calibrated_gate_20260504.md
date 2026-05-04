# HellaSwag Qwen-To-Phi Official-Train Receiver-Calibrated Gate

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible; ICLR full remains
blocked.

Current story: LatentWire has a strict source-private packet protocol, real
candidate-frontier oracle headroom, and clean byte/exposure accounting. The
missing ICLR evidence is still a positive learned receiver that improves over
the fixed Qwen hybrid packet on a larger frozen cross-family slice.

Exact blocking gap: adding Phi official-train receiver-side score calibration
almost ties fixed hybrid but still does not beat it with paired uncertainty or
slice stability.

## Gate

Implemented and ran:

- `scripts/build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.py`
- `results/source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate_20260504_validation1024_2048/`
- references:
  `references/694_hellaswag_qwen_to_phi_receiver_calibrated_refs_20260504.md`

This gate is the direct follow-up to the failed source-only dictionary. It
builds a Phi official-train score cache on the same `1,487` out-of-bag Qwen
calibration rows, then trains a frozen receiver-side selector. The receiver
chooses among:

- fixed Qwen hybrid;
- Qwen source-score rival;
- Phi target top-1.

The receiver sees Qwen packet features and Phi-local score features during
official-train calibration. At evaluation time, the transmitted source payload
is still byte-scale and source-private; raw source scores, hidden vectors, KV,
and text are not transmitted.

## Result

The gate fails, but it is a useful near miss.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
|---|---:|---:|---:|
| hybrid/rival/Phi oracle diagnostic | `0.766927` | `+0.299479` | `+0.266927` |
| eval-label best threshold diagnostic | `0.470052` | `+0.002604` | `0.000000` |
| fixed Qwen hybrid | `0.467448` | reference | reference |
| official-train receiver-calibrated packet | `0.466146` | `-0.001302` | `-0.007812` |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` |
| label-permutation receiver control | `0.333333` | `-0.134115` | `-0.175781` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.251302` |

Official-train selection:

- calibration rows: `1,487`;
- fit/dev: `1,115/372`;
- official-train Qwen hybrid accuracy: `0.523874`;
- official-train Phi target accuracy: `0.264291`;
- official-train hybrid/rival/Phi oracle: `0.806994`;
- selected dev delta: `+0.008065`, CI95 low `-0.002688`;
- selected model: L2 `10000.0`, threshold `-0.178019`.

Held-out eval:

- receiver-calibrated packet: `0.466146`;
- fixed hybrid: `0.467448`;
- delta: `-0.001302`, CI95 low `-0.007812`;
- overrides: `12`, helps `3`, harms `4`.

Per-slice result:

| Slice | Rows | Receiver Acc. | Fixed Hybrid Acc. | Delta |
|---|---:|---:|---:|---:|
| `1024:1536` | `384` | `0.489583` | `0.486979` | `+0.002604` |
| `1536:2048` | `384` | `0.442708` | `0.447917` | `-0.005208` |

The Phi official-train score cache was generated locally on MPS with
`float16`, `max_length=256`, `mean` normalization, and `continuation` prompt
mode. The first full cache build took `861.247s` inside the scorer. The
committed result reran cache-only and reports `phi_train_score_cache_hit=true`.

## Interpretation

This result weakens the first receiver-calibrated branch, but it is not the
same failure as the source-only dictionary. The source-only dictionary was
clearly harmful: `0.429688`. With Phi receiver-side calibration, the learned
selector nearly returns to fixed hybrid: `0.466146`. The destructive controls
collapse, and the eval-label best threshold diagnostic reaches `0.470052`.

The key blocker is now sharper: the current official-train linear receiver
features have only tiny held-out signal. Even a cheating eval threshold over
the trained scores clears fixed hybrid by only `+0.002604`, below the `+0.005`
promotion threshold and with CI low at zero. The oracle remains enormous:
`0.766927`. So the candidate set is not the problem; the receiver evidence for
which candidate to trust is too weak.

## Contribution Status

Promote:

- the source-private packet protocol and strict control suite;
- candidate-ID decision-frontier communication as the shared basis;
- receiver-side calibration as the right next framing, because it nearly
  repairs the harmful source-only dictionary;
- systems/exposure accounting: `2B` raw / `5B` framed source-private packet,
  no source text/KV/hidden/score-vector transfer.

Weaken:

- source-only official-train dictionaries;
- the first linear receiver-calibrated selector over Qwen/Phi score features;
- fixed-depth or shallow threshold refinement on the same feature family.

Still alive:

- richer receiver-checkable packet fields, such as calibrated utility bins,
  disagreement class, or error-correcting redundancy;
- conformal/selective accept rules with explicit harm control;
- iterative receiver refinement only if it has a calibrated stopping rule;
- true soft-prefix/resonance experiments, starting with target self-compression
  before cross-model transfer.

Cut if necessary:

- repeated shallow ridge switchers on the same cached rows;
- claims of a systems win over QJL/KIVI/KVQuant/TurboQuant. Until quality is
  positive, those remain byte-floor and threat-model context only.

## Lay Explanation

This time we let Phi practice too. Phi scored the same training questions, and
the learned rule saw both Qwen's safe/backup answers and Phi's own scores. On
new questions, that rule almost matched the safe Qwen packet but still made one
more bad change than good changes overall. So we learned that receiver
calibration helps, but this simple receiver is not strong enough.

## Next Gate

Do not continue the same linear receiver features. The next highest-value gate
is a harm-controlled receiver: use official-train calibration to learn an
accept/defer rule over explicit complementarity buckets, then promote only if
it improves fixed hybrid by at least `+0.005` with positive paired CI and
nonnegative slice deltas. A second live branch is target self-compression:
prove that short soft/latent tokens can reproduce Phi's own full-text state
before attempting cross-model latent resonance.
