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

The Triton interpreter gate is now cleared locally against this reference. The
remaining gate is a native NVIDIA packet that passes
`phase2/check_native_gpu_packet.py` under `phase2/gpu_gate_runbook.md`: matched
quality drift, downstream loss/KL/top-1 checks, repeated same-shape latency for
every row/model/shape, NCU memory/HBM counters, and a promote/kill decision.
