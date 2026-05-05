# SinkAware Approximate Attention Reference Gate

Status: **implemented and unit-testable on Mac**.

This reference defines the live SinkAware operator:

1. compute non-sink tail logits exactly;
2. replace only fixed-sink logits with a predictor;
3. run the ordinary softmax denominator over predicted sink logits plus exact
   tail logits;
4. report sink-mass, attention, and output drift against exact attention.

The reference is not a speed result and not a claim that approximation is safe
for generation. Its purpose is to make the future Triton/CUDA implementation
auditable: exact sink logits must reproduce exact attention, and approximate
sink logits must change only the sink side of the score vector.

Next gate: implement the same operator in Triton interpreter mode when `triton`
is available, then run native GPU timing only if the interpreter kernel matches
this reference.
