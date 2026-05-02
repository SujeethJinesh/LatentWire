# Tokenwise Connector Runbook

## Method Boundary

Per-example source-conditioned rate-limited connector under source-destroying controls and explicit source-exposure accounting.

This must be compared against static prefix/prompt tuning and KV/cache fusion; otherwise the novelty claim is not defensible.

## Architecture

- Frozen source and frozen target.
- Tokenwise source activations or KV summaries, not per-choice mean caches.
- 16-64 learned queries or soft-prefix tokens.
- Train only the connector under target loss.
- Transmit only the rate-limited connector output or its packetized form.

## Required Controls

- target-only
- packet-only
- target-cache-only
- candidate-only
- target-derived packet
- row-shuffled source packet
- random same-rate packet
- label-permutation decoder
- candidate derangement
- same-byte visible text
- source-label-copy audit upper bound
- static prompt/prefix tuning
- zero-source connector
- wrong-row/shuffled-source connector
- label-shuffled training
- Qwen-substituted or target-family packet
- C2C/KVComm/KVCOMM cache-transfer baseline
- QJL/TurboQuant byte floors for source-state transfer

## Run Gates

### 1. Mac ARC smoke

Goal: Verify target-forward soft-prefix training code on 8-16 ARC rows without claiming evidence.

Pass rule: Training runs, controls are emitted, and no source answer/text/KV is transmitted.

### 2. ARC soft-prefix preflight

Goal: Train/select on ARC validation disagreement rows with frozen source and target endpoints.

Pass rule: Matched beats target-only, source-free, zero-source, row-shuffled, same-byte text, and Qwen-substituted controls.

### 3. OpenBookQA 3B receiver gate

Goal: Train-only receiver/query connector over the strongest OpenBookQA packet-only rows.

Pass rule: Packet-plus-receiver beats packet-only by >= +0.005 with positive paired CI95 low and source-destroying controls.

### 4. Frozen ARC/OpenBookQA test gate

Goal: Evaluate once on frozen larger slices after validation selection.

Pass rule: Mean delta >= 0.02 and CI95 low > 0 versus Qwen-substituted, packet-only, target-only, and same-byte text.

### 5. Cross-family falsification

Goal: Repeat with one true non-Qwen source-target pair.

Pass rule: Positive paired CI survives; same-family-only result is marked diagnostic.

### 6. Systems ingest

Goal: Fill native validator rows for LatentWire and all KV/cache baselines.

Pass rule: Validator allows native claims only after complete matched measurements.

## Primary Sources

- `blip2`: https://arxiv.org/abs/2301.12597
- `flamingo`: https://arxiv.org/abs/2204.14198
- `perceiver_io`: https://arxiv.org/abs/2107.14795
- `prefix_tuning`: https://arxiv.org/abs/2101.00190
- `c2c`: https://openreview.net/forum?id=LeatkxrBCi
- `kvcomm`: https://arxiv.org/abs/2510.03346
- `kvcomm_cross_context`: https://arxiv.org/abs/2510.12872
- `qjl`: https://arxiv.org/abs/2406.03482
- `turboquant`: https://arxiv.org/abs/2504.19874
- `dit`: https://arxiv.org/abs/2212.09748
- `consistency`: https://arxiv.org/abs/2303.01469
- `sae_universal`: https://arxiv.org/abs/2410.06981
