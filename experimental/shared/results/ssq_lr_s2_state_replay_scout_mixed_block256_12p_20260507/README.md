# SSQ-LR S2 Mixed-Block Held-Out Scout

Decision: `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`

This is a 12-prompt held-out Mac-local continuation replay. It is simulated
state quantization, not native GPU evidence, not task accuracy, and cannot
promote S2 by itself.

## Readout

- Evaluator-selected recipe: `mixed_int3_mxfp4_low_error_25pct`
- Selected memory reduction: `4.192x`
- Selected accuracy CI high: `0.0`
- Selected NLL CI high: `0.03956`
- Prompt count: `12`
- Row count: `156`

The result revives SSQ-LR as a Mac-side candidate: the mixed-block allocator
crosses the preregistered byte gate while preserving BF16 argmax on this short
continuation replay. The next gate is a non-resource-limited S2 packet with the
recipe hash frozen, longer continuation windows, paired uncertainty, and the
same controls.
