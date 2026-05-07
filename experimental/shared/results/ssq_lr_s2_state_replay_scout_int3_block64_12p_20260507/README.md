# SSQ-LR S2 INT3 Block-64 Held-Out Scout

Decision: `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a 12-prompt held-out Mac-local continuation replay. It is not GPU
evidence and cannot promote S2.

## Readout

- Evaluator-selected recipe: `mxfp4_primary_state_block64`
- Selected memory reduction: `3.765x`
- Selected accuracy CI high: `0.0`
- Selected NLL CI high: `0.04796`
- INT3 memory reduction: `4.923x`
- INT3 outcome: clears bytes but loses argmax fidelity on at least one prompt
- Prompt count: `12`
- Row count: `132`

The smaller block improves INT3 scaling granularity while still failing the
held-out quality gate.
