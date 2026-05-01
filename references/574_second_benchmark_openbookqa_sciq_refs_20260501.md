# Second-Benchmark OpenBookQA/SciQ References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_openbookqa_bridge_contract_20260501/`
- `results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- `results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/`
- `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_3b/`
- `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/`
- `results/source_private_sciq_bridge_contract_20260501/`
- `results/source_private_sciq_fixed_packet_gate_20260501_qwen05_hashed_validation/`

OpenBookQA `main` was materialized as train/validation/test `4957/500/500`
with no cross-split content overlap. The `3B` hashed common-basis packet passes
`5/5` validation seeds and `5/5` test seeds. Validation matched/target/text is
`0.356/0.252/0.326`; test is `0.378/0.276/0.350`. The test minimum lift over
same-byte text is `0.028`, and the minimum paired CI95 lower bound versus
target is `0.038`.

SciQ was materialized as train/validation/test `11679/1000/1000` and passes the
target/control gate on validation, but same-byte text nearly saturates the
source signal (`0.712` packet versus `0.706` text at `12B`). It is therefore a
useful limitation/control rather than the promoted second benchmark.

## Primary Sources

- OpenBookQA. https://arxiv.org/abs/1809.02789
- OpenBookQA TensorFlow Datasets card. https://www.tensorflow.org/datasets/catalog/openbookqa
- OpenBookQA Hugging Face dataset card. https://huggingface.co/datasets/allenai/openbookqa
- SciQ. https://arxiv.org/abs/1707.06209
- SciQ ACL Anthology entry. https://aclanthology.org/W17-4413/
- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Relative Representations. https://arxiv.org/abs/2209.15430
- Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- QJL. https://arxiv.org/abs/2406.03482
- TurboQuant. https://arxiv.org/abs/2504.19874
- NVIDIA LLM inference metrics. https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html

## Safe Claim Boundary

Safe: the common-basis fixed-byte packet generalizes beyond ARC to OpenBookQA at
an even smaller `3B` payload while preserving destructive-control collapse and
projection-seed stability.

Safe: SciQ shows that same-byte text can catch up when answer strings are short;
this should be discussed as an access-model and byte-budget limitation.

Unsafe: claiming broad benchmark saturation, hidden-state communication, or
that byte packets universally beat text relay. The promoted OpenBookQA row still
uses a Qwen source-choice cache and a public hashed coordinate basis.
