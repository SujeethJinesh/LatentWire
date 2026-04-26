# SVAMP32 C2C Mechanism Syndrome Probe - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: still one stable deployable positive method plus
  medium/large, seed, source-control, and cross-family gates
- current story: the C2C-derived syndrome remains a useful bound, but the
  tested deployable C2C mechanism summaries do not recover its source-necessary
  clean IDs
- blocker: no source/cache-derived signal has matched the C2C syndrome bound

## Gate

Test whether pre-generation C2C projector traces can predict the compact
SVAMP32 residue syndrome without parsing C2C final answers.

Promotion rule:

- matched `>=14/32`
- target-self `3/3`
- clean source-necessary `>=2/6`
- numeric coverage `>=31/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, target-only, and slots-only
  controls have clean union `0/6`

## Implementation

Added C2C prefill trace extraction in `latent_bridge/c2c_eval.py`:

- shared C2C KV-index builder
- projector scalar/gate summary extraction
- non-invasive projector forward hooks that record key/value residual summary
  statistics before answer generation

Added `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`, which trains
leave-one-out ridge residue classifiers from those features and reuses the
existing strict SVAMP32 target-candidate syndrome decoder.

Added focused tests in `tests/test_c2c_mechanism_trace.py`.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q
```

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --candidate text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --device mps \
  --max-new-tokens 1 \
  --date 2026-04-26 \
  --output-json results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.json \
  --output-md results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.md
```

## Evidence

Scalar trace:

- status: `c2c_mechanism_syndrome_probe_fails_gate`
- feature dim: `336`
- matched: `11/32`
- target-only: `14/32`
- zero-source: `14/32`
- shuffled-source: `9/32`
- label-shuffle: `14/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`

Residual trace:

- status: `c2c_mechanism_syndrome_probe_fails_gate`
- feature dim: `896`
- feature tensor sha256:
  `75ad00f84a99ae632ec5641fa53e66e987188ba693858079dfae319381de7e73`
- matched: `12/32`
- target-only: `14/32`
- zero-source: `13/32`
- shuffled-source: `9/32`
- label-shuffle: `14/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- teacher numeric coverage: `32/32`
- exact ordered ID parity: true

Artifacts:

- `results/svamp32_c2c_mechanism_syndrome_probe_20260426/manifest.md`
- `results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.json`
  - sha256: `685d76e3640b17084b25544c970ec8a95efe1555e5d36469fb49ba88325176f7`
- `results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.md`
  - sha256: `d5ab8c2dbbf68e18258001de7dec69b735288f7627b87f17e7030aa0e3193595`

## Decision

Weaken the C2C summary-feature syndrome-distillation branch. The richer
residual trace does not reach target-only, does not preserve target-self IDs,
and recovers none of the six clean residual IDs.

Do not scale this variant to SVAMP70/GSM70. A future C2C-derived attempt needs
a new mechanism-level reason, likely token/layer residual coding or a learned
contrastive objective with source-destroying controls, before consuming more
compute.

## Next Gate

Shift the live method work to the next highest-value branch: a source-control
contrastive innovation bottleneck on a surface with measured source headroom.
The first gate should be cheap and exact-ID: matched source positives against
zero/shuffled/wrong-source penalties, target-safe fallback, and the same clean
source-necessary accounting before any medium or large scale-up.
