# KV-Cache Latent Communication Baselines

Date: 2026-04-27

## Status

This memo updates the next-branch baseline contract after the durable
source-surface ranking and CPU source-hidden query smoke. The direct
source-hidden query-bottleneck diagnostic failed again, and no reusable offline
activation tensors exist. The next live learned/latent branch should therefore
start with target-preserving KV/cache communication baselines before adding a
new connector.

## Sources and Experiment Implications

### Cache-to-Cache

- Primary sources:
  - `https://openreview.net/forum?id=LeatkxrBCi`
  - `https://arxiv.org/abs/2510.03215`
  - `https://fuvty.github.io/C2C_Project_Page/`
- Problem it helps with: a reviewer will not accept text relay as the only
  communication baseline when direct KV-cache communication exists.
- Mechanism/design idea: project and fuse the source model KV cache into the
  target model cache, with target cache preserved and a learnable layer gate.
- Does it change the next experiment? Yes. The next MPS gate must include a
  receiver-cache-preserving C2C/KV baseline or explicitly explain why C2C is
  being deferred.
- Role: baseline and systems framing.

Bounded local translation:

- Run `latent_bridge.c2c_eval` or existing C2C materialization first when a
  compatible published artifact exists.
- Treat C2C as the main quality competitor; a new method must either approach
  it or beat it on bytes/latency/TTFT at comparable accuracy.

### KVComm

- Primary sources:
  - `https://arxiv.org/abs/2510.03346`
  - `https://openreview.net/pdf?id=F7rUng23nw`
- Problem it helps with: fixed communication budget and selective source-cache
  transfer.
- Mechanism/design idea: share selected KV layers/pairs using layer-wise
  attention-importance scores and a Gaussian prior.
- Does it change the next experiment? Yes. The next baseline should sweep
  selected-layer fractions such as `0.10`, `0.25`, `0.50`, and `1.00` with
  bytes/latency accounting.
- Role: baseline, ablation, and systems comparator.

Bounded local translation:

- Use `python -m latent_bridge.kvcomm_eval`, not direct script invocation, so
  repo-local imports resolve.
- First run a one-example smoke after MPS clears, then run `svamp70_live` only
  if the smoke preserves exact ID and numeric coverage.

### Direct Semantic Communication via Vector Translation

- Primary source: `https://arxiv.org/abs/2511.03945`
- Problem it helps with: conservative cross-family latent injection.
- Mechanism/design idea: learn a vector translator and inject translated source
  vectors with a fixed blend, reported at a conservative `30%` strength.
- Does it change the next experiment? Lightly. It motivates a small fixed-blend
  control sweep, but not before the C2C/KVComm baselines are logged.
- Role: ablation and cross-family inspiration.

Bounded local translation:

- If a projected latent slot branch is revived, include blend strengths
  `0.05`, `0.15`, and `0.30`, with zero-source and shuffled-source controls.
- Never call fixed-blend steering communication unless matched source separates
  from those controls on exact clean IDs.

## Local CPU Tooling Smoke

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 1 \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl
```

Result:

- Status: tooling smoke only, not a method gate.
- Accuracy: `0/1`.
- Generated tokens: `8`.
- Latency: `0.9505s` on CPU.
- Selected layers: `[1, 6, 2, 8, 7, 5, 4]`.
- Prediction artifact SHA256:
  `ddfa80b562ebcda86e0e2578e33d7d010f18cb003b9f1bb326e0c6f9940eb64e`
- Metadata SHA256:
  `b051921a3089b8af7f8f2c3ef89aed8ffaf6c6edb3b563313374ce3e75abed40`

Direct invocation as `./venv_arm64/bin/python latent_bridge/kvcomm_eval.py`
initially failed with `ModuleNotFoundError: No module named 'latent_bridge'`.
The wrapper now bootstraps the repo root onto `sys.path`, and direct `--help`
plus `py_compile` pass.

## Decision

Promote a fixed-budget KV/cache communication baseline as the next MPS-runnable
baseline once PID `31103` clears. Do not add a new latent connector before
running this baseline unless the objective is only a tiny no-op equivalence
test.
