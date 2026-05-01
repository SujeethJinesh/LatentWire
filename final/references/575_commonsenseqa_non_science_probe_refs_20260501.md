# CommonsenseQA Non-Science Probe References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_commonsenseqa_bridge_contract_20260501/`
- `results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/`
- `results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b/`
- `results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b_gap001/`

CommonsenseQA labeled train/validation splits were materialized as
`9741/1221` rows with no cross-split overlap. `question_concept` and answer
fields are forbidden to the source packet builder. The official Hugging Face
test split has empty labels, so the current Mac-feasible probe is validation
only.

The source signal is strong on non-science commonsense: at `12B`, validation
matched/target/text is `0.440/0.206/0.440`. At `2B`, the packet passes `5/5`
seeds under a relaxed `0.01` text-margin gate with matched mean/min/max
`0.438/0.437/0.439`, target `0.206`, text `0.424`, and CI95 lower bound versus
target `0.195`. It fails the stricter `0.02` same-byte-text margin (`0/5`), so
it is not yet a promoted headline benchmark.

## Primary Sources

- CommonsenseQA paper. https://arxiv.org/abs/1811.00937
- CommonsenseQA Hugging Face dataset card. https://huggingface.co/datasets/tau/commonsense_qa
- OpenBookQA. https://arxiv.org/abs/1809.02789
- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Relative Representations. https://arxiv.org/abs/2209.15430
- C2C / Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- TurboQuant. https://arxiv.org/abs/2504.19874
- NVIDIA LLM inference metrics. https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html

## Safe Claim Boundary

Safe: CommonsenseQA is a non-science validation diagnostic showing the source
model signal survives outside ARC/OpenBookQA.

Safe: same-byte text is too strong under the current packet protocol on
CommonsenseQA, so the row should not be promoted as a strict ICLR benchmark
result yet.

Unsafe: claiming broad non-science generalization, hidden-state communication,
or a text-baseline win on CommonsenseQA.
