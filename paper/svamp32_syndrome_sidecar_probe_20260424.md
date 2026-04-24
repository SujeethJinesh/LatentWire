# SVAMP32 Syndrome Sidecar Probe - 2026-04-24

## Status

- status: `syndrome_sidecar_bound_clears_gate_not_method`
- readiness impact: positive for branch selection, not yet a paper method row
- gate: target-side candidate pools plus compact source/C2C syndrome must
  recover `>=2/6` clean residual IDs while zero/shuffle/target-only/slots-only
  controls recover none and the `14/32` target-self floor is preserved
- outcome: strict target-side candidate pool clears the bound with a `1` byte
  residue syndrome: matched `14/32`, target-only fallback `14/32`, clean
  source-necessary `2/6`

## Motivation

The stronger-source audit killed source-final/source-margin escalation. The
remaining live story is that C2C has cache-level headroom (`16/32`) while
target-self repair is a strong floor (`14/32`). The exact question this turn
was whether a low-rate sidecar can use target candidate pools as decoder side
information, or whether the candidate pool lacks enough clean answers to make
the branch worth training.

Reference anchors:

- `paper/svamp32_stronger_source_margin_audit_20260424.md`
- `paper/svamp32_source_oracle_bound_20260424.md`
- `references/451_answer_teacher_microfit_refs.md`
- `references/452_syndrome_sidecar_refs.md`

## Decision Surface

Top moves considered:

- C2C-derived syndrome sidecar bound. It matters because it directly tests the
  newly promoted decoder-side-information branch before training. It might fail
  if target-side candidates do not contain clean gold answers, or if controls
  can select them without source. It costs a small analyzer and helps same-pair,
  efficiency, interpretability, and reproducibility.
- C2C-residual learned-query connector training. It matters because C2C is the
  only strong live headroom signal. It might fail like the answer-teacher
  microfit by leaking to target-only or slots-only controls. It costs a larger
  translator change and should come after the sidecar bound.
- Strict cross-family source-informativeness. It matters for paper scope, but
  current source-final and source-margin signals are saturated on same-family
  stronger sources. It is lower value until a live same-pair sidecar clears.

I executed the syndrome bound because it is the cheapest decisive gate.

## Implementation

Added `scripts/analyze_svamp32_syndrome_sidecar_probe.py`.

The analyzer:

- validates exact ordered ID parity against the frozen target row
- builds target candidate pools from numeric spans in target-side predictions
- optionally adds source/text rows for sensitivity only
- decodes C2C-derived residue signatures under moduli sets
- evaluates matched, zero-source, shuffled-source, target-only, and slots-only
  controls
- reports candidate-pool gold coverage, bytes, clean source-necessary IDs, and
  target-self preservation

Added `tests/test_analyze_svamp32_syndrome_sidecar_probe.py`.

## Commands

Strict target-side pool:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_syndrome_sidecar_probe.py \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli-set 2,3 \
  --moduli-set 2,3,5 \
  --moduli-set 2,3,5,7 \
  --moduli-set 97 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --date 2026-04-24 \
  --output-json results/svamp32_syndrome_sidecar_probe_20260424/targetpool_syndrome_probe.json \
  --output-md results/svamp32_syndrome_sidecar_probe_20260424/targetpool_syndrome_probe.md
```

Augmented sensitivity pool adds source/text numeric spans:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_syndrome_sidecar_probe.py \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --candidate text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli-set 2,3 \
  --moduli-set 2,3,5 \
  --moduli-set 2,3,5,7 \
  --moduli-set 97 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --date 2026-04-24 \
  --output-json results/svamp32_syndrome_sidecar_probe_20260424/augmentedpool_syndrome_probe.json \
  --output-md results/svamp32_syndrome_sidecar_probe_20260424/augmentedpool_syndrome_probe.md
```

## Evidence

Strict target-side pool:

| Moduli | Bytes | Matched | Target-only | Target-self matched | Clean gold in pool | Clean matched | Clean necessary | Control clean union | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2,3` | 1 | 9/32 | 14/32 | 1/3 | 2/6 | 1/6 | 0/6 | 1/6 | none |
| `2,3,5` | 1 | 13/32 | 14/32 | 3/3 | 2/6 | 2/6 | 2/6 | 0/6 | `1d50b408c8f5cd2c`, `aee922049c757331` |
| `2,3,5,7` | 1 | 14/32 | 14/32 | 3/3 | 2/6 | 2/6 | 2/6 | 0/6 | `1d50b408c8f5cd2c`, `aee922049c757331` |
| `97` | 1 | 14/32 | 14/32 | 3/3 | 2/6 | 2/6 | 2/6 | 0/6 | `1d50b408c8f5cd2c`, `aee922049c757331` |

Augmented sensitivity pool:

- `[2,3,5,7]` gives matched `15/32`, clean matched `3/6`, clean necessary
  `3/6`, control clean union `0/6`
- the additional clean ID is `6e9745b37ab6fc45`, sourced from text-to-text
  numeric spans

Interpretation:

- promoted: target candidate pools contain enough clean gold answers for a
  compact syndrome sidecar to be worth training
- not promoted as method: the current probe uses C2C numeric answers as the
  syndrome source, so it is still an oracle/bound
- risk: target candidate generation could be doing too much work; therefore
  the next gate must train a matched source-latent syndrome predictor and keep
  the same controls

## Subagent Synthesis

- repo/repro agent recommended using existing SVAMP32 exact-ID target-set,
  teacher-forced diagnostics, and control machinery before any generation
  sweep
- ablation agent converged on the same source-necessity matrix: matched,
  zero-source, shuffled-source, target-only, slots-only, with
  `clean_source_necessary = matched_clean - union(control_clean)`
- literature/creative agent recommended exactly this Wyner-Ziv/Slepian-Wolf
  style target-candidate sidecar and warned that source-latent predictability
  is the next real risk

## Verification

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_syndrome_sidecar_probe.py -q`
- result: `2 passed in 0.02s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_syndrome_sidecar_probe.py`
- result: pass
- `./venv_arm64/bin/python -m pytest -q`
- result: `662 passed in 25.98s`
- JSON artifact validation
- result: pass
- `git diff --check`
- result: pass

## Artifacts

See `results/svamp32_syndrome_sidecar_probe_20260424/manifest.md`.

## Hypothesis Update

- promoted: latent syndrome sidecar as the next live branch
- revived: a low-rate, interpretable positive method may exist even though
  source-final and source-margin channels are weak
- weakened: dense connector training as the immediate next step; the sidecar
  first needs a source-latent syndrome predictor
- still saturated: source-final copying, stronger-source margin escalation,
  and same-pair calibration proxy microfits

## Next Exact Gate

Train the smallest source-latent syndrome predictor:

- input: frozen source hidden/cache features on the same SVAMP32 IDs
- target: residue compatibility for target-side candidate pools
- objective: matched residue selects gold over target-wrong candidate, with
  zero/shuffle/target-only/slots-only controls
- promotion: `>=2/6` clean source-necessary IDs, matched `>=14/32`, target-self
  `3/3`, controls `0/6` clean, exact ID parity, numeric coverage `>=31/32`
