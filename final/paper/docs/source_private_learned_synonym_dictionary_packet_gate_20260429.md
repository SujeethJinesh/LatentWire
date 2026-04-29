# Source-Private Learned Synonym Dictionary Packet Gate

- date: `2026-04-29`
- status: positive calibrated learned-dictionary gate with scoped claim
- synonym-stress result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429/`
- native result root:
  `results/source_private_learned_synonym_dictionary_packet_gate_20260429_native/`
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

The largest remaining caveats:

- the dictionary is public-calibrated, not discovered from LLM activations;
- synonym calibration uses the same frozen synonym stress family, so the next
  gate must hold out synonym clusters or use independently generated
  paraphrases;
- no seed repeat has been run yet;
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

Run a seed repeat and holdout-paraphrase variant:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_learned_synonym_dictionary_packet_gate_20260429_seed_repeat \
  --budgets 4 \
  --train-examples 512 \
  --eval-examples 256 \
  --seed 47 \
  --candidate-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 512 \
  --feature-dim 384 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.02
```

Promotion beyond calibrated ontology robustness requires a synonym cluster
split: calibration synonyms cannot include the exact paraphrase forms used in
evaluation.
