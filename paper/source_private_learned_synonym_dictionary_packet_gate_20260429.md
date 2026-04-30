# Source-Private Learned Synonym Dictionary Packet Gate

- date: `2026-04-29`
- status: positive calibrated learned-dictionary gate with scoped claim
- synonym-stress result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429/`
- native result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429_native/`
- seed-repeat result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429_seed_repeat/`
- held-out family-B result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429_heldout_synonym/`
- held-out family-B calibrated diagnostic root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429_heldout_synonym_calibrated_oracle/`
- scale rung: medium learned-method confirmation on Mac CPU

## Current Readiness

This gate strengthens the paper's third technical contribution: an
interpretable learned/calibrated shared dictionary can recover the sparse packet
under synonym-stressed candidate wording, where the previous hand atom decoder
failed.

The claim must remain scoped. The dictionary is trained from public candidate
text and a frozen public synonym calibration into the existing atom space. It
is not a neural crosscoder over LLM activations and not proof of arbitrary
semantic latent transfer. It is useful because it turns the hand synonym
failure into an executable, reproducible, source-private communication method
with strict controls and causal feature knockout.

## Method

The source still sends a rate-capped source-private atom packet extracted from
hidden diagnostic evidence. The target-side decoder no longer relies only on
literal hand phrase rules. Instead, it fits a ridge-calibrated hashed
word/character dictionary from public candidate descriptions and their
synonym-stressed variants into the shared atom space, then decodes candidate
overlap using learned atom scores.

Important scoping details:

- candidate calibration: `all_public`
- no private source logs are used for target dictionary fitting
- no eval answers are used for packet generation
- answer-masked source is made strictly destructive in this gate
- promoted rows are the 4-byte rows, because 8-byte rows can trigger
  structured-text/random byte collisions on some surfaces

## Commands

Synonym stress:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_learned_synonym_dictionary_packet_gate_20260429 \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 256 \
  --seed 43 \
  --candidate-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 512 \
  --feature-dim 384 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.02
```

Native companion:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_learned_synonym_dictionary_packet_gate_20260429_native \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 256 \
  --seed 43 \
  --candidate-atom-view native \
  --candidate-calibration all_public \
  --calibration-examples 512 \
  --feature-dim 384 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.02
```

## Native Result

- pass gate: `true`
- cross-family pass: `true`
- direction pass: all `true`
- pass rows: `4`
- max learned packet accuracy: `1.000`
- max learned-target delta: `+0.750`
- minimum passing CI95 lower bound vs target: `+0.695`

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 256 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.699 | 1.000 |
| holdout -> core | 4 | 256 | `true` | 1.000 | 0.250 | 0.270 | +0.750 | +0.699 | 1.000 |
| same-family all | 4 | 256 | `true` | 1.000 | 0.250 | 0.254 | +0.750 | +0.695 | 1.000 |

The native sanity check confirms the learned dictionary does not regress the
agreed-ontology surface.

## Synonym-Stress Result

- pass gate: `true`
- cross-family pass: `true`
- direction pass: all `true`
- pass rows: `4`
- max learned packet accuracy: `1.000`
- max learned-target delta: `+0.750`
- minimum passing CI95 lower bound vs target: `+0.562`

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 256 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.699 | 1.000 |
| holdout -> core | 4 | 256 | `true` | 0.875 | 0.250 | 0.266 | +0.625 | +0.566 | 1.000 |
| holdout -> core | 8 | 256 | `true` | 0.875 | 0.250 | 0.270 | +0.625 | +0.562 | 1.000 |
| same-family all | 4 | 256 | `true` | 0.938 | 0.250 | 0.254 | +0.688 | +0.633 | 1.000 |

This directly addresses the previous hand sparse failure under synonym stress:
the hand decoder had `0` pass rows at `n=512`, while the calibrated learned
dictionary recovers cross-family pass at `n=256`.

## Controls And Caveats

Passing rows keep strict source-destroying controls within target + `0.03`.
Controls include zero-source, shuffled-source, strict answer-masked source,
public-only sidecar, target-derived sidecar, random same-byte, answer-only
text, structured text truncated to the same bytes, atom-ID derangement, and
feature knockout.

The 8-byte core -> holdout and same-family rows are not promoted because
matched structured text or random-byte controls rise above target + `0.03`.
The clean promoted operating point is 4 bytes.

## Seed Repeat

The promoted 4-byte synonym-stress operating point repeats at seed `47`.

- pass gate: `true`
- cross-family pass: `true`
- direction pass: all `true`
- pass rows: `3`
- max learned packet accuracy: `1.000`
- max learned-target delta: `+0.750`
- minimum passing CI95 lower bound vs target: `+0.562`

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 256 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.695 | 1.000 |
| holdout -> core | 4 | 256 | `true` | 0.875 | 0.250 | 0.258 | +0.625 | +0.562 | 1.000 |
| same-family all | 4 | 256 | `true` | 0.938 | 0.250 | 0.250 | +0.688 | +0.629 | 1.000 |

## Held-Out Synonym Family Boundary

The stricter held-out family-B gate separates calibration surface from
evaluation surface:

- calibration atom view: `synonym_stress`
- evaluation atom view: `heldout_synonym`
- exact transformed evaluation phrase overlap with calibration: `0`
- pass gate: `false`
- cross-family pass: `false`
- pass rows: `0`
- max learned packet accuracy: `0.500`
- max learned-target delta: `+0.250`

| Direction | Budget | N | Pass | Learned packet | Target | Best control | Delta target | Oracle | Controls OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 256 | `false` | 0.500 | 0.250 | 0.375 | +0.250 | 0.375 | `false` |
| holdout -> core | 4 | 256 | `false` | 0.375 | 0.250 | 0.375 | +0.125 | 0.375 | `false` |
| same-family all | 4 | 256 | `false` | 0.438 | 0.250 | 0.375 | +0.188 | 0.375 | `false` |

The learned-candidate oracle also falls to `0.375`, so this is not a packet
capacity failure. The current hashed ridge receiver does not map unseen
paraphrase-family B into the shared atom ontology.

As a diagnostic, allowing calibration to see family B itself restores the
method:

- calibration atom view: `heldout_synonym`
- evaluation atom view: `heldout_synonym`
- pass gate: `true`
- cross-family pass: `true`
- pass rows: `3`
- learned packet accuracy: `1.000` in all directions
- controls remain collapsed near target

This makes the claim boundary precise: the contribution is a calibrated
public-dictionary packet interface, not held-out semantic paraphrase transfer.

The largest remaining caveats:

- the dictionary is public-calibrated, not discovered from LLM activations;
- held-out synonym clusters fail unless the calibration dictionary sees the
  held-out surface family;
- no production GPU systems row is added by this gate.

## Interpretation

This is a real strengthening step, but not a license to overclaim. The paper can
now put forward a third technical contribution:

> A calibrated learned shared dictionary converts source-private sparse packets
> into a synonym-robust agreed-protocol interface, preserving byte-level
> source-private communication under strict controls.

It should not say:

> We solved general cross-model semantic latent transfer.

## Next Exact Gate

Build the next receiver that can plausibly generalize across held-out surfaces:
anchor-relative sparse features, an embedding receiver, or a contrastive
synonym-consistency dictionary trained without exact family-B phrases. Reuse
the same held-out command above as the decisive gate: calibration atom view
`synonym_stress`, candidate atom view `heldout_synonym`, `n=256`, budget `4`.

Promotion beyond calibrated ontology robustness requires passing the held-out
family-B split without exact transformed phrase overlap.
