# HellaSwag Qwen-To-Phi Oracle Switch Decomposition Gate

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible as a
source-private packet/evaluation/systems-boundary paper; ICLR full remains
blocked.

Current story: fixed byte-scale Qwen packets transfer useful candidate evidence
to Phi under strict source-private controls, but learned receiver-side repair
has not yet harvested Phi's complementary information.

Exact blocking gap: even a selective switcher over Qwen packet bits, Qwen
Fourier score-contrast bits, and Phi receiver-local scores does not beat the
fixed Qwen hybrid packet.

## Gate

Implemented and ran:

- `scripts/build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate.py`
- `results/source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate_20260504_validation1024_2048/`

The gate uses the same cached Qwen-to-Phi HellaSwag `1024:2048` surface as the
denoising gate. It adds the aligned Qwen source score cache and encodes
candidate-score contrasts in a 4-choice Helmert/Fourier basis. The receiver
sees only a byte-scale packet plus Phi's local score simplex; it does not see
source text, KV, hidden vectors, raw scores, or logits.

Split:

- fit: first `64` rows per cached `512`-row slice;
- select: next `64` rows per slice;
- eval: remaining `384` rows per slice, `768` total.

## Result

The gate fails.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
|---|---:|---:|---:|
| target+hybrid+Qwen-top2 oracle diagnostic | `0.776042` | `+0.308594` | `+0.276042` |
| Qwen score top-2 oracle diagnostic | `0.675781` | `+0.208333` | `+0.174479` |
| target-or-hybrid oracle | `0.604167` | `+0.136719` | `+0.113281` |
| eval-label best switcher diagnostic | `0.467448` | `0.000000` | `-0.007812` |
| fixed Qwen hybrid | `0.467448` | reference | reference |
| selected train/select switcher | `0.460938` | `-0.006510` | `-0.015625` |
| forced nonzero switcher | `0.460938` | `-0.006510` | `-0.015625` |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.250000` |

The selected switcher made `21` held-out overrides: `3` helped and `8` harmed
relative to fixed hybrid. Even the eval-label diagnostic selected from the same
switcher family only tied fixed hybrid, with `5` helps and `5` harms.

Oracle decomposition on eval rows:

| Bucket | Count |
|---|---:|
| both Qwen hybrid and Phi correct | `97` |
| Qwen hybrid only correct | `262` |
| Phi only correct | `105` |
| both wrong | `304` |
| Qwen hybrid and Phi disagree | `551` |

## Interpretation

This is a useful negative result. The oracle headroom is large, and Qwen score
top-2 evidence is especially strong, but the current low-rate switch features
do not expose a learnable high-precision frontier. The switcher loses because
there are many more possible harms than helps: Phi is uniquely correct on
`105` eval rows, but uniquely wrong against Qwen hybrid on `262` rows.

This weakens simple selective-deferral, Fourier score-contrast, and protected
frontier variants when they are trained only on the current `128` fit +
`128` select rows and constrained to this byte packet family. It does not kill
the broader rate-distortion framing because the source score top-2 oracle
shows a much richer candidate-contrast surface remains available.

## Contribution Status

Promote:

- the oracle decomposition as reviewer-facing evidence that we are not hiding
  the remaining headroom;
- the Qwen score top-2 diagnostic as motivation for richer source-code
  dictionaries;
- the selective-deferral controls as a stricter falsification surface.

Weaken:

- shallow target-score switchers;
- simple Fourier sign/magnitude packets;
- claims that Phi local scores can safely repair Qwen hybrid without a richer
  learned source code or larger calibration surface.

Still alive:

- a protected-frontier packet that transmits top-2/rival evidence more
  directly;
- an official-train cross-fitted source-code dictionary;
- learned query/resampler interfaces if Mac or NVIDIA compute allows.

## Lay Explanation

Qwen's tiny hint is usually better than Phi's own answer, but Phi is sometimes
right when Qwen is wrong. We tried to learn those moments using Qwen's answer
hint, a tiny score-pattern code, and Phi's own four answer scores. It still
made more bad switches than good ones. The important clue is that Qwen's top-2
score list often contains the answer, so the next method should send a better
"which rival matters" packet instead of just a switch flag.

## Next Gate

Do not continue with shallow switchers on this exact packet family. The next
highest-value Mac-local branch is a protected top-2/rival source-code packet:
default to fixed Qwen hybrid, transmit a compact top-rival or candidate-pair
frontier bitfield, and force pass/fail against source-row shuffle,
source-score shuffle, code permutation, candidate roll, target-derived code,
and label-permutation controls.
