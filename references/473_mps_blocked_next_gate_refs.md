# MPS-Blocked Next-Gate References

- date: `2026-04-27`
- status: `local_memo`
- purpose: consolidate the latest primary-source baseline implications while
  MPS is blocked and CPU-only method gates are exhausted.

## Current Blocker

The blocker this memo helps with is branch selection, not a runnable CPU method.
The latest artifact audits and answer-null scouts leave no CPU-only
evidence-bearing branch. Literature and baselines therefore sharpen the next
MPS gate: first find a stronger answer-masked source surface; only then spend
learned-interface effort.

## Sources and Experiment Implications

### Cache-to-Cache

- Primary source: `https://arxiv.org/abs/2510.03215`
- Official code: `https://github.com/thu-nics/C2C`
- Blocker helped: quality baseline for any latent/cache communication claim.
- Mechanism/design idea: projected source KV-cache fusion with target-cache
  preservation and learnable target-layer gates.
- Experiment change: any LatentWire learned interface must compare to C2C or
  clearly beat it on bytes/latency/TTFT at comparable accuracy. C2C also
  motivates per-layer gates and target-preserving residual integration rather
  than a single global source injection.
- Role: baseline, ablation design, and paper framing.

### KVCOMM

- Primary source: `https://arxiv.org/abs/2510.12872`
- Blocker helped: systems baseline for multi-agent prefill reuse and cache
  offset alignment.
- Mechanism/design idea: reuse overlapping-context KV caches by aligning cache
  offsets with online anchor examples.
- Experiment change: future systems claims should report TTFT/prefill latency
  and separate same-context reuse from true source-derived semantic
  communication.
- Role: systems baseline and threat model.

### KVComm

- Primary source: `https://openreview.net/forum?id=F7rUng23nw`
- Blocker helped: efficient inter-LLM communication baseline under selective KV
  sharing.
- Mechanism/design idea: layer-wise KV selection based on attention-importance
  with a prior, transmitting a fraction of layer KV pairs.
- Experiment change: local `latent_bridge.kvcomm_eval` remains the right
  source-control harness, but future runs must include matched, zero-source,
  shuffled-source, and target-only conditions with byte telemetry.
- Role: baseline, source-control comparator, and systems accounting.

### Q-KVComm

- Primary source: `https://arxiv.org/abs/2512.17914`
- Blocker helped: byte budget and compression comparator for raw KV/cache
  transfer.
- Mechanism/design idea: adaptive layer-wise quantization, hybrid information
  extraction, and heterogeneous calibration for compressed KV communication.
- Experiment change: raw KV byte counts are not enough. A positive method must
  either compare against a quantized/selective KV baseline or explicitly scope
  the systems claim to uncompressed transfer.
- Role: systems baseline and matched-byte ablation.

## Branch Decision

The next evidence-bearing gate is not another CPU sidecar over current
artifacts. The ranked next actions are:

1. Stronger-source, answer-masked surface discovery after MPS cleanup.
2. Erasure-aware learned syndrome over that surface if it has answer-unexplained
   clean target-pool headroom.
3. Zero-init target-query bottleneck only after the surface gate passes, with
   C2C/KVComm/Q-KVComm-style baselines visible from the first run.

Promotion still requires source-destroying controls, exact ID parity, high
numeric coverage, paired uncertainty, target-self preservation, and bytes /
latency / TTFT telemetry where relevant.
