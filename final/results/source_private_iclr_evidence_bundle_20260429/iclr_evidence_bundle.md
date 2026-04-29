# Source-Private ICLR Evidence Bundle

- pass gate: `True`
- created UTC: `2026-04-29T21:24:42.175443+00:00`
- current readiness: `scoped positive-method paper; not broad cross-family latent-transfer ready`

## Technical Contributions

| Contribution | Status | Headline evidence | Main metric | Remaining gap |
|---|---|---|---|---|
| Source-private evidence-packet benchmark and controls | strong scoped contribution | 160 examples x 3 seeds x 5 label/code/order stress transforms pass | matched=1.000, target=0.250, worst_control=0.263 | Still a protocol/candidate-decoder task; frame as source-private evidence communication, not universal semantics. |
| Extreme-rate candidate-syndrome packet method | headline method for scoped paper | packet oracle bytes max=2.0; matched-byte text accuracy max=0.250 | packet vs query-aware text >= 7.0x; packet vs full log >= 183.2x | Text becomes oracle at higher bytes, so claim only the far-left rate frontier. |
| Systems byte/KV-cache accounting frontier | systems contribution with clear caveat | minimum QJL-style 1-bit cache payload is 10752.0x packet | minimum KIVI-style 2-bit cache payload is 21504.0x packet | Derived byte accounting only; no production GPU serving throughput yet. |
| Endpoint paired uncertainty and local target-decoder evidence | paper-ready evidence rows exist, but systems scope is local | endpoint paired rows pass with min packet-vs-target CI lows 0.350/0.350 | paper-ready rows in ledger=3; total audited rows=104 | Mac-local proxy, not server TTFT/TPOT/throughput. |
| Learned receiver / latent-method diagnostics | bounded diagnostic contribution, not headline cross-family claim | ledger records 4 positive learned-receiver rows and explicit failed/pruned rows | same-distribution positives exist; simple cross-family masked innovation failed | Need shared-dictionary/crosscoder-style method with feature knockout before claiming cross-family latent communication. |

## Pass Checks

| Check | Pass |
|---|---|
| `required_artifacts_exist` | `True` |
| `rate_frontier_passes` | `True` |
| `matched_byte_text_stays_at_target` | `True` |
| `packet_beats_query_aware_text_by_7x` | `True` |
| `kv_cache_qjl_lower_bound_above_1000x` | `True` |
| `coded_label_risk_passes` | `True` |
| `composed_label_code_order_stress_passes` | `True` |
| `endpoint_core_uncertainty_passes` | `True` |
| `endpoint_holdout_uncertainty_passes` | `True` |
| `ledger_has_paper_ready_rows` | `True` |

## Novelty Matrix

| Comparison | Source | Communicated object | Source-private | Internals? | Extreme rate? | Controls? | Paper role |
|---|---|---|---|---|---|---|---|
| LatentWire source-private packet | this work | rate-capped private evidence packet decoded with target candidate side information | True | False | True | True | headline method |
| C2C cache-to-cache communication | https://arxiv.org/abs/2510.03215 | projected/fused source KV cache | partly | True | False | not same threat model | closest high-rate internal-state baseline/framing |
| KVComm selective KV sharing | https://openreview.net/forum?id=F7rUng23nw | selected KV pairs/layers | partly | True | False | not same threat model | high-rate KV communication baseline/framing |
| TurboQuant / vector-KV quantization | https://arxiv.org/abs/2504.19874 | quantized vectors or KV/cache states | False | True | False | False | systems byte-floor comparator and future vector-packet ablation |
| QJL 1-bit sign sketch | https://arxiv.org/abs/2406.03482 | JL-projected sign sketches for inner products/KV | False | True | low-bit but high-dimensional | False | matched-byte vector sketch baseline if latent branch is promoted |
| Prompt/text compression such as LLMLingua-family methods | https://arxiv.org/abs/2310.05736 | compressed visible prompt/context tokens | False | False | token-level | False | structured text/compression framing; query-aware text relay is the local control |
| Slepian-Wolf / Wyner-Ziv source coding | https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources | syndrome/source code with decoder side information | True | False | True | theory, not benchmark controls | theory framing, not empirical LLM baseline |
| JEPA / diffusion-transformer latent prediction | https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf | predicted latent/representation state | not primary | True | False | False | inspiration for future learned receiver; not current claim |

## Reproduction Commands

```bash
./venv_arm64/bin/python scripts/build_source_private_rate_frontier.py --output-dir results/source_private_rate_frontier_20260429
./venv_arm64/bin/python scripts/build_source_private_kv_cache_baseline_table.py --output-dir results/source_private_kv_cache_baseline_table_20260429
./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429
./venv_arm64/bin/python scripts/build_source_private_pass_fail_ledger.py --output-dir results/source_private_pass_fail_ledger_20260429
find final -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > final/MANIFEST.sha256
shasum -a 256 -c final/MANIFEST.sha256
./venv_arm64/bin/python -m pytest tests/test_build_source_private_rate_frontier.py tests/test_build_source_private_kv_cache_baseline_table.py tests/test_run_source_private_coded_label_risk_gate.py tests/test_build_source_private_pass_fail_ledger.py -q
```

## Remaining ICLR Risks

- Production serving TTFT/TPOT/throughput on NVIDIA GPUs is still missing.
- The headline method is protocol/candidate-side-information communication, not universal semantic latent transfer.
- Simple learned cross-family masked-innovation receivers failed; a future shared-dictionary/crosscoder method needs feature knockout before promotion.
- The final paper must show text relay catches up at higher byte budgets to avoid unfair-baseline criticism.
