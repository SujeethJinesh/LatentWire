# Source-Private Coded-Label Risk Gate

- date: `2026-04-29`
- status: passed strict-small reviewer-risk gate
- scale rung: strict small gate
- result root: `results/source_private_coded_label_risk_gate_20260429/`

## Current Readiness

This strengthens the current scoped positive-method paper. It does not make the
paper a broad cross-family latent-transfer claim, but it directly addresses the
most important reviewer objection for the strongest 2-byte packet result:
whether the method is just a fixed coded-label lookup.

## Blocker Addressed

The prior codebook-remap gate changed diagnostic vocabularies while holding the
public surface fixed. A stricter reviewer can still ask whether the packet
depends on public candidate labels, candidate display order, or a single
diagnostic-code convention. This gate tests those failure modes together.

## Method

I added `scripts/run_source_private_coded_label_risk_gate.py` and ran:

```bash
./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py \
  --examples 160 \
  --candidates 4 \
  --family-set all \
  --seeds 29,31,37 \
  --budget 2 \
  --output-dir results/source_private_coded_label_risk_gate_20260429
```

Transforms:

- `baseline`: original hidden-repair packet surface.
- `label_rename`: opaque candidate labels with answer-label update.
- `diagnostic_code_remap`: new diagnostic codebook and hidden `REPAIR_DIAG`.
- `candidate_pool_permutation`: opaque labels plus deterministic non-identity
  candidate-order shuffle.
- `label_code_order_composed`: opaque labels, diagnostic-code remap, and
  candidate-order shuffle together.

Conditions:

- matched 2-byte repair packet
- target-only
- zero-source
- shuffled-source
- random same-byte
- answer-only
- answer-masked
- target-derived sidecar
- structured JSON/free-text matched relays
- diagnostic-masked full log
- full hidden-log and full diagnostic positive oracles

## Result

- pass gate: `true`
- examples per seed: `160`
- seeds: `29`, `31`, `37`
- budget: `2` bytes
- transforms: `5`

Across all `15` seed/transform rows:

- matched 2-byte packet accuracy: `1.000`
- target-only accuracy: `0.250`
- best reviewer-negative control: `0.250`
- positive oracle floor: `1.000`
- worst source-destroying control: `0.263`

Transform summary:

| Transform | Pass | Min matched | Max target | Max source control | Max reviewer negative |
|---|---|---:|---:|---:|---:|
| `baseline` | `true` | 1.000 | 0.250 | 0.256 | 0.250 |
| `label_rename` | `true` | 1.000 | 0.250 | 0.256 | 0.250 |
| `diagnostic_code_remap` | `true` | 1.000 | 0.250 | 0.263 | 0.250 |
| `candidate_pool_permutation` | `true` | 1.000 | 0.250 | 0.256 | 0.250 |
| `label_code_order_composed` | `true` | 1.000 | 0.250 | 0.263 | 0.250 |

Artifact hashes:

- `summary.json`: `d29b2ba829b8fc09cbe0204f80d34f9ec298b71d3fb71e7a0c7b6902eecd8240`
- `summary.md`: `914e359f13d03022e3ebe6388f80a4c691e6fd1163512bb243f1b5ecbcdb5bdd`
- `predictions.jsonl`: `7f9067b7a59d8133df59265695d0bebdae3a718a907d98409e725bc04301eb79`
- `manifest.json`: `f7537d59d4c47cbcbaf7337adb7e17384271fbcf4e53acd29f9b6f16372c737e`

## Interpretation

This rules out the strongest simple protocol-artifact explanation for the
deterministic hidden-repair packet row: the gain does not depend on fixed public
candidate labels, display order, or one diagnostic-codebook assignment. The
right paper claim remains precise:

> A source-private packet can communicate diagnostic evidence under target-side
> candidate side information, and the effect survives label, codebook, and
> order stress tests plus source-destroying controls.

This still does not prove natural-language semantic generalization or broad
cross-family latent transfer. It strengthens the scoped evidence-packet claim.

## Next Gate

Build a reviewer-facing reproduction/novelty bundle:

1. one-command final table/manifest regeneration;
2. novelty matrix against C2C, KVComm, prompt compression, KV/cache
   quantization, and source-coding theory;
3. if time allows, a larger `n=500` composed coded-label stress row.
