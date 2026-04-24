# SVAMP32 Source-Latent Syndrome Probe

- date: `2026-04-24`
- status: `source_latent_syndrome_probe_fails_gate`
- paper readiness: not ICLR-ready; the live branch is still a bound, not a
  positive method
- current story: target candidate pools can decode a compact C2C-derived
  residue syndrome, but frozen source hidden summaries do not yet predict that
  syndrome
- blocking gap: replace oracle C2C numeric residues with source-derived
  residues under exact-ID controls

## Local Evidence Read First

- latest ledger: `paper/experiment_ledger_20260421.md`
- latest syndrome memo: `paper/svamp32_syndrome_sidecar_probe_20260424.md`
- latest reference manifest: `references/452_syndrome_sidecar_refs.md`
- latest strict bound artifacts:
  - `results/svamp32_syndrome_sidecar_probe_20260424/targetpool_syndrome_probe.json`
  - `results/svamp32_syndrome_sidecar_probe_20260424/augmentedpool_syndrome_probe.json`
- current source-latent artifacts:
  - `results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.json`
  - `results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_mid_last_targetpool_probe.json`

## Alive, Saturated, Blocked

- alive: target-candidate syndrome decoding as a bound; it still cleanly
  separates source-necessary IDs when the syndrome comes from C2C numeric
  answers.
- weakened: direct linear readout from Qwen2.5-0.5B frozen source hidden
  summaries to C2C residue classes.
- saturated: source-final copying, stronger-source source-margin escalation,
  answer-teacher microfits, and dense connector variants that fail
  source-necessity controls.
- blocked: no positive-method paper claim until a deployable source-latent
  signal clears the same strict source-destroying controls.

## Top 3 Moves Considered

1. Source-latent syndrome probe.
   - why it matters: directly replaces the oracle C2C residue with matched
     source features.
   - why it might fail: source hidden summaries may not encode the conditional
     residue, or ridge readout may collapse to target/candidate priors.
   - evidence gained: whether pooled source states can clear the strict
     `>=14/32`, `>=2/6` clean source-necessary gate.
   - cost: one small analyzer plus two Qwen0.5B frozen feature passes.
   - category: same-pair, controls, efficiency, interpretability,
     reproducibility.
2. Candidate-pool arithmetic closure expansion.
   - why it matters: tests whether the decoder side information is the actual
     bottleneck if source residue prediction has signal.
   - why it might fail: extra candidates can create target-only leakage or
     turn the row into a candidate oracle.
   - evidence gained: clean-ID headroom and collision sensitivity.
   - cost: small analyzer extension.
   - category: robustness, interpretability.
3. Learned query-bottleneck residue head.
   - why it matters: Perceiver/Q-Former-style query bottlenecks are a plausible
     frozen-backbone interface if linear pooled summaries are too weak.
   - why it might fail: high overfit risk on 32 rows and may repeat dense
     connector leakage unless controls are kept identical.
   - evidence gained: whether source states contain a nonlinear residue signal.
   - cost: moderate training and stricter cross-fit bookkeeping.
   - category: same-pair, efficiency, interpretability.

I executed move 1 because it is the cheapest decisive gate and preserves the
Wyner-Ziv/Slepian-Wolf candidate-pool framing from `references/452_syndrome_sidecar_refs.md`.

## Method

Added `scripts/analyze_svamp32_source_latent_syndrome_probe.py`.

The analyzer:

- extracts frozen Qwen2.5-0.5B source hidden summaries on the frozen SVAMP32
  exact-ID prompts
- trains leave-one-ID-out ridge classifiers for residue classes modulo
  `[2,3,5,7]` using C2C numeric predictions as labels
- decodes only through the previously frozen strict target candidate pool
- evaluates matched, zero-source, shuffled-source, label-shuffled, target-only,
  and slots-only controls with exact ordered ID parity
- fails if matched drops below `14/32`, if target-self is not preserved, if
  clean source-necessary IDs are below `2/6`, or if any source-destroying
  control recovers clean IDs

Added `tests/test_analyze_svamp32_source_latent_syndrome_probe.py` for the
source-necessity gate, LOO residue signatures, and feature/target ID parity.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_latent_syndrome_probe.py
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --shuffle-offset 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers last \
  --device mps \
  --dtype float32 \
  --date 2026-04-24 \
  --output-json results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.json \
  --output-md results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.md
```

The same command with `--feature-layers mid,last` produced the
`qwen25_05b_mid_last_targetpool_probe.*` artifacts.

## Results

Strict last-layer feature probe:

- status: `source_latent_syndrome_probe_fails_gate`
- teacher numeric coverage: `32/32`
- provenance issues: `0`
- matched: `9/32`
- target-only: `14/32`
- zero-source: `13/32`
- shuffled-source: `10/32`
- label-shuffled: `13/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- control clean union: `0/6`
- target-self preserved: `2/3`

Strict mid+last feature probe:

- status: `source_latent_syndrome_probe_fails_gate`
- teacher numeric coverage: `32/32`
- provenance issues: `0`
- matched: `9/32`
- target-only: `14/32`
- zero-source: `14/32`
- shuffled-source: `10/32`
- label-shuffled: `13/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- control clean union: `0/6`
- target-self preserved: `3/3`

## Subagent Synthesis

- repo/repro agent recommended a standalone analyzer over editing
  `translator.py`, because the existing connector path mixes source, target,
  delta, and slot-memory effects.
- ablation agent recommended the exact matrix used here: matched, zero-source,
  shuffled-source, label-shuffled, target-only, and slots-only with a hard
  `0/6` clean-control union rule.
- creative/literature agent recommended a future Syndrome-Q branch: a small
  Q-Former/Perceiver-style query bottleneck predicting factorized residue
  posteriors. Primary references are listed in
  `references/452_syndrome_sidecar_refs.md`, including C2C, Perceiver IO,
  BLIP-2/Q-Former, Slepian-Wolf, and Wyner-Ziv.

## Hypothesis Update

- weakened: frozen pooled Qwen2.5-0.5B source hidden states are not enough for a
  deployable linear syndrome sidecar on SVAMP32.
- still alive: the sidecar itself as a candidate-pool bound.
- promoted next: either a held-out learned query-bottleneck residue predictor
  with the same controls, or a C2C-residual distillation probe that predicts
  the syndrome from source cache transformations rather than pooled hidden
  summaries.
- killed for now: claiming a source-latent positive method from pooled hidden
  summaries.

## Artifacts

See `results/svamp32_source_latent_syndrome_probe_20260424/manifest.md`.

## Verification

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q`
- result: `3 passed in 0.02s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_latent_syndrome_probe.py`
- result: pass

## Next Exact Gate

Run the same strict target-candidate decode, but replace pooled hidden
summaries with a cross-fitted learned query bottleneck or C2C-residual
distillation target. Require matched `>=14/32`, target-self `3/3`, clean
source-necessary `>=2/6`, exact ID parity, numeric coverage `>=31/32`, and
zero/shuffle/label-shuffle/target-only/slots-only clean union `0/6`.
