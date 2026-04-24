# SVAMP32 Learned Syndrome Probe

- date: `2026-04-24`
- status: `learned_syndrome_probe_fails_gate`
- paper readiness: not ICLR-ready; the sidecar remains an oracle bound rather
  than a positive method
- current story: target candidate pools can decode compact C2C-derived
  residues, but neither pooled source hidden summaries nor source-token query
  bottlenecks recover those residues from Qwen2.5-0.5B source states
- blocking gap: a deployable source-derived residue predictor that clears the
  same exact-ID source-destroying controls

## Local Evidence Read First

- ledger: `paper/experiment_ledger_20260421.md`
- oracle sidecar memo: `paper/svamp32_syndrome_sidecar_probe_20260424.md`
- failed pooled source-hidden memo:
  `paper/svamp32_source_latent_syndrome_probe_20260424.md`
- reference memos:
  - `references/452_syndrome_sidecar_refs.md`
  - `references/453_learned_syndrome_probe_refs.md`
- current results:
  - `results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.json`
  - `results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q8_h64_f8_seed1_targetpool_probe.json`

## Alive, Saturated, Blocked

- alive: C2C-derived syndrome sidecar as a clean decoder-side-information
  bound.
- weakened: source-token learned query bottlenecks as direct residue predictors.
- saturated: source-final copying, stronger-source source-margin escalation,
  answer-teacher microfits, dense connector variants, linear pooled hidden
  syndrome readout, and now two source-token query-bottleneck sizes.
- blocked: no ICLR positive-method claim until the syndrome comes from a
  source-derived method rather than the C2C oracle.

## Top 3 Moves Considered

1. Cross-fitted source-token query bottleneck.
   - why it matters: tests whether pooled summaries failed because they
     averaged away sparse source-token signal.
   - why it might fail: 32 rows are small, and the head may learn target/candidate
     priors or no useful source residue signal.
   - expected evidence: matched-vs-zero/shuffle/label-shuffle/same-norm-noise
     separation under the same candidate-pool decode.
   - cost: one standalone analyzer plus two Qwen0.5B runs.
   - category: same-pair, controls, interpretability, reproducibility.
2. C2C-residual distillation.
   - why it matters: the successful oracle residue came from C2C, so source
     cache transformations may contain signal that source-only hidden states do
     not expose.
   - why it might fail: high risk of accidentally using C2C answer leakage or
     non-deployable teacher artifacts.
   - expected evidence: whether a bridge-derived residual compatibility signal
     can predict residues without target-cache leakage.
   - cost: higher instrumentation cost.
   - category: same-pair, interpretability, efficiency.
3. Verifier-gated repair.
   - why it matters: can preserve the target-only floor if a source-derived
     candidate signal exists.
   - why it might fail: if the source predictor has no clean hits, the verifier
     has nothing useful to accept.
   - expected evidence: harm-minimization only after a source signal exists.
   - cost: moderate, but lower priority before source-syndrome signal.
   - category: robustness, reproducibility.

I executed move 1 because it is the smallest decisive nonlinear readout test
after the pooled hidden probe failed.

## Method

Added `scripts/analyze_svamp32_learned_syndrome_probe.py`.

The analyzer:

- extracts frozen source token states from Qwen2.5-0.5B on the frozen SVAMP32
  exact-ID prompts
- trains a small query bottleneck over source tokens to predict C2C residue
  classes modulo `[2,3,5,7]`
- uses cross-fitted fold-held-out predictions for every evaluated ID
- decodes through the same strict target candidate pool used by the oracle
  sidecar
- evaluates matched, zero-source, example-shuffled source, label-shuffled,
  same-norm-noise, target-only, and slots-only conditions

Added `tests/test_analyze_svamp32_learned_syndrome_probe.py` for fold splitting,
cross-fit prediction shape, and factorized modulus logits.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_learned_syndrome_probe.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_learned_syndrome_probe.py
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_learned_syndrome_probe.py \
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
  --query-count 4 \
  --hidden-dim 16 \
  --epochs 80 \
  --outer-folds 8 \
  --seed 1 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers mid,last \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --date 2026-04-24 \
  --output-json results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.json \
  --output-md results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.md
```

The richer variant used `--query-count 8`, `--hidden-dim 64`, and
`--epochs 120`.

## Results

Strict `q=4`, `h=16`, `8`-fold learned query bottleneck:

- status: `learned_syndrome_probe_fails_gate`
- teacher numeric coverage: `32/32`
- provenance issues: `0`
- matched: `10/32`
- target-only: `14/32`
- zero-source: `11/32`
- shuffled-source: `10/32`
- label-shuffled: `13/32`
- same-norm-noise: `14/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- target-self: `2/3`

Strict `q=8`, `h=64`, `8`-fold learned query bottleneck:

- status: `learned_syndrome_probe_fails_gate`
- teacher numeric coverage: `32/32`
- provenance issues: `0`
- matched: `9/32`
- target-only: `14/32`
- zero-source: `14/32`
- shuffled-source: `10/32`
- label-shuffled: `13/32`
- same-norm-noise: `14/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- target-self: `2/3`

## Subagent Synthesis

- repo hook agent recommended a sibling analyzer around the existing syndrome
  decoder rather than editing `translator.py`, because the generation path is
  too entangled with source/target/delta/slot effects.
- ablation agent recommended the exact controls used here and a hard clean
  union rule across all source-destroying controls.
- literature agent recommended query-bottleneck and verifier-gated repair
  ideas, with primary references recorded in
  `references/453_learned_syndrome_probe_refs.md`.

## Hypothesis Update

- weakened: source-token query bottlenecks over frozen Qwen2.5-0.5B states can
  recover the C2C residue syndrome.
- promoted: C2C-residual distillation or another source signal, because the
  useful syndrome appears tied to the C2C mechanism rather than generic source
  hidden/token states.
- rejected for now: verifier-gated repair as the next gate, since no
  source-derived clean candidate signal exists for a verifier to preserve.
- still alive: the target-candidate syndrome bound as a benchmark for the next
  source-derived method.

## Artifacts

See `results/svamp32_learned_syndrome_probe_20260424/manifest.md`.

## Verification

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_learned_syndrome_probe.py -q`
- result: `3 passed in 0.95s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_learned_syndrome_probe.py`
- result: pass

## Next Exact Gate

Inspect C2C artifact availability and implement the smallest C2C-residual
distillation probe that predicts the same residue from deployable source/cache
signals, not C2C final answers. Keep the same strict decode and pass rule:
matched `>=14/32`, target-self `3/3`, clean source-necessary `>=2/6`, numeric
coverage `>=31/32`, exact ID parity, and all source-destroying controls `0/6`
clean.
