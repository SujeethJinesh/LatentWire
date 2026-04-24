# SVAMP32 Source Informativeness And Oracle Bound

- Date: 2026-04-24
- Status: diagnostic complete; no method promoted
- Gate: SVAMP32 exact-ID source/complementarity diagnostic after delta-memory failed

## Question

Before spending more compute on connectors, does the frozen SVAMP32 slice show
usable source/complementary headroom, and do any existing latent rows expose
clean residual IDs beyond `target_self_repair`?

## Result

There is oracle-selection headroom, but the existing rows expose too little
clean source-necessary signal for a paper claim.

Key readout:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- source_alone: `5/32`
- text_to_text: `2/32`
- source/text union clean residual correct: `0/6`
- existing candidate union clean residual correct: `1/6`
- best candidate-row oracle with target_self_repair: `idweighted_gate015`,
  `17/32`
- strict clean source-sidecar bound for `idweighted_gate015`: `15/32`,
  `+1` vs target_self_repair, `1` clean source-necessary ID

The only currently exposed clean residual ID is `aee922049c757331`, recovered
by `idweighted_gate015`. Source-alone and text-to-text are not exact-correct on
any of the six clean residual targets. This means a future connector should
distill C2C/cache-level source signal rather than source final answers.

## Candidate Readout

| Row | Correct | Teacher-only | Clean residual | Wins vs target | Losses vs target |
|---|---:|---:|---:|---:|---:|
| `query_pool_gate010` | 9/32 | 1 | 0 | 2 | 1 |
| `idweighted_gate015` | 10/32 | 2 | 1 | 3 | 1 |
| `targetmem_gate020` | 9/32 | 1 | 0 | 2 | 1 |
| `deltamem_gate017` | 9/32 | 1 | 0 | 2 | 1 |
| `control_contrast_gate012` | 9/32 | 1 | 0 | 1 | 0 |

## Oracle Bounds

Best target_self_repair oracle bounds:

| Row | Oracle correct | Delta vs self-repair | Clean residual added |
|---|---:|---:|---:|
| `idweighted_gate015` | 17/32 | +3 | 1 |
| `source_alone` | 17/32 | +3 | 0 |
| `deltamem_gate017` | 16/32 | +2 | 0 |
| `query_pool_gate010` | 16/32 | +2 | 0 |
| `targetmem_gate020` | 16/32 | +2 | 0 |
| `control_contrast_gate012` | 15/32 | +1 | 0 |
| `text_to_text` | 15/32 | +1 | 0 |

The broad oracle upper bound says selection can matter. The strict sidecar
bound says the current source-specific clean residual surface is still too
thin: `idweighted_gate015` gives only one clean source-necessary residual ID
after subtracting translated-zero control retention.

## Implementation

Added:

- `scripts/analyze_svamp32_source_oracle_bound.py`
- `tests/test_analyze_svamp32_source_oracle_bound.py`

The analyzer:

- preserves exact raw method labels for gate-sweep rows
- checks exact ordered ID parity against the target row
- reports source correctness on teacher-only and clean residual targets
- reports candidate clean residual recovery
- computes oracle bounds for target and each baseline versus source/candidate
  rows
- emits clean residual provenance by example ID

Verification:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_oracle_bound.py -q
```

Result: `1 passed in 0.02s`

## Commands

Aggregate source/oracle diagnostic:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_oracle_bound.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --baseline target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --baseline selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate targetmem_gate020=path=.debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl,method=rotalign_kv_gate_0.20 \
  --candidate deltamem_gate017=path=.debug/svamp32_delta_memory_query_codec_20260424/preds/deltamem_konly_attention_gate_sweep.jsonl,method=rotalign_kv_gate_0.17 \
  --candidate control_contrast_gate012=path=.debug/svamp32_control_contrastive_innovation_20260423/preds/control_zero_shuffle_w010_m001_attention_gate_sweep.jsonl,method=rotalign_kv_gate_0.12 \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --expected-n 32 \
  --date 2026-04-24 \
  --output-json results/svamp32_source_oracle_bound_20260424/source_oracle_bound.json \
  --output-md results/svamp32_source_oracle_bound_20260424/source_oracle_bound.md
```

Strict sidecar-bound cross-check:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_sidecar_bound.py \
  --probe-json results/svamp32_idweighted_query_innovation_20260423/c2c_teacher_probe_gate015_targetself_translated_zero.json \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --candidate-label gate015 \
  --source-control-label translated_kv_zero \
  --output-json results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.json \
  --output-md results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.md
```

## Artifacts

- `results/svamp32_source_oracle_bound_20260424/source_oracle_bound.json`
  - sha256: `806a3b9e64a3562b386a77d6b8573bc529f8524dcaab9fbd82fbdcbf97378966`
- `results/svamp32_source_oracle_bound_20260424/source_oracle_bound.md`
  - sha256: `6de23b56930bca29c311a403b9ebd08be653a620a4f7d6b9b290cb3208c57557`
- `results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.json`
  - sha256: `678015b227017b2c679d2708ff89311fa749407814320c138be49789aeb3ad08`
- `results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.md`
  - sha256: `2b4958c03166acd9e78b55e9c3f9a65647c8136606093acf9a94f14e06c87ed8`

## Subagent Synthesis

All bounded side paths converged on the same next method family: a
source-control-contrastive learned query connector, not another direct KV
geometry tweak. The suggested minimal connector is `8-16` learned queries over
source K/V plus target-prior state, trained with C2C clean-residual
distillation, target-preservation, and matched-vs-zero/shuffled/ghost source
contrast.

Relevant primary-source anchors already present in local references:

- C2C cache fusion: https://arxiv.org/abs/2510.03215
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
- Flamingo gated cross-attention: https://arxiv.org/abs/2204.14198
- Perceiver IO learned query bottleneck: https://arxiv.org/abs/2107.14795
- AWQ selective salient-channel protection: https://arxiv.org/abs/2306.00978
- KIVI asymmetric KV quantization: https://arxiv.org/abs/2402.02750

## Hypothesis Update

- promoted: source/cache-level C2C distillation remains useful because C2C gets
  `16/32` while source final answers get `0/6` clean residual IDs
- promoted: a learned query connector with source-control contrast is the next
  highest-value positive-method branch
- weakened: verifier-only selection from existing rows as a paper method; the
  strict clean source-sidecar bound is only `15/32`
- killed for now: further direct target-memory/delta-memory variants without a
  source-discriminative connector objective
- saturated: target_self_repair as a comparator, current query-pool/idweighted
  latent rows, and text-to-text/source-final-answer relay on clean residual IDs

## Next Exact Gate

Implement the smallest source-control-contrastive learned query connector
smoke:

- frozen source and target models
- `8` learned connector queries
- C2C clean-residual distillation
- target-correct/self-repair preservation
- matched-vs-zero/shuffled source contrast
- exact SVAMP32 rows: matched, zero-source, shuffled-source, target-only or
  target-prior-only

Promotion threshold remains strict:

- `>=14/32` total correct
- `>=2/6` clean residual IDs
- at most `1` target-correct loss
- clean wins vanish under source controls
- exact ordered ID parity and numeric coverage `>=31/32`
