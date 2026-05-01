# Source-Private Native Systems Benchmark Plan

- pass gate: `True`
- native systems complete: `False`
- required metrics: `44`
- required baselines: `11`
- headline benchmarks: `ARC-Challenge, OpenBookQA`
- diagnostic benchmarks: `HellaSwag`

## Baseline Rows

| Row | Method | Substrate | Source private | Source KV exposed | Role |
|---|---|---|---:|---:|---|
| `latentwire_packet_cached_source` | LatentWire packet receiver, cached source packet | vLLM and SGLang | `True` | `False` | headline systems row if quality is preserved |
| `latentwire_packet_end_to_end_source_scoring` | LatentWire end-to-end source scoring plus packet receiver | vLLM and SGLang | `True` | `False` | end-to-end cost row; may weaken speed claims but preserves privacy/bandwidth claims |
| `target_only_vllm` | Target-only vLLM/PagedAttention | vLLM | `True` | `False` | serving baseline for accuracy and latency |
| `target_only_sglang` | Target-only SGLang/RadixAttention | SGLang | `True` | `False` | second serving substrate / scheduler sensitivity check |
| `same_byte_visible_text` | Same-byte visible text packet | vLLM and SGLang | `False` | `False` | privacy and text-relay control |
| `source_label_copy_control` | Source-label-copy / trained-label-copy control | offline control plus serving row when applicable | `False` | `False` | fatal shortcut control for HellaSwag and MCQ surfaces |
| `c2c_cache_to_cache` | C2C cache-to-cache communication | native cache-fusion implementation | `False` | `True` | closest high-rate internal-state baseline |
| `kvcomm_selective_kv` | KVComm selective KV sharing | native KV-sharing implementation or faithful proxy | `False` | `True` | selective KV communication baseline |
| `kvcomm_online_cross_context` | KVCOMM online cross-context KV-cache communication | native or faithful online KV-cache communication proxy | `False` | `True` | multi-agent KV reuse systems neighbor |
| `qjl_1bit_source_state` | QJL 1-bit source-state sketch | native or faithful sign-sketch proxy | `False` | `True` | low-bit vector-state sketch baseline |
| `turboquant_lowbit_source_state` | TurboQuant-style low-bit source-state quantization | native low-bit vector/KV kernel or faithful proxy | `False` | `True` | quantized vector-state baseline |

## Required Metrics

| Metric | Unit | Required | Why |
|---|---|---:|---|
| `benchmark` | string | `True` | native rows must map back to a frozen benchmark surface |
| `split` | string | `True` | methods must run on the same split before paired claims are valid |
| `model_pair` | string | `True` | source and target model identities define the communication setting |
| `method` | string | `True` | each row must identify the protocol or baseline under test |
| `implementation` | string | `True` | native, faithful proxy, and byte-floor rows must remain separable |
| `commit_hash` | string | `True` | native baselines need exact reproducibility provenance |
| `gpu_name` | string | `True` | GPU SKU materially changes serving and memory traffic results |
| `cuda_version` | string | `True` | CUDA version affects kernels, profilers, and serving engines |
| `driver_version` | string | `True` | driver version is part of the native systems configuration |
| `serving_engine` | string | `True` | vLLM and SGLang rows must be distinguishable |
| `batch_size` | count | `True` | batch shape changes throughput and cache pressure |
| `concurrency` | count | `True` | online goodput depends on load concurrency |
| `context_len` | tokens | `True` | KV/cache traffic scales with context length |
| `max_new_tokens` | tokens | `True` | decode work must be fixed across native rows |
| `precision` | string | `True` | precision affects memory, bandwidth, and accuracy |
| `num_examples` | count | `True` | paired uncertainty requires the shared example count |
| `accuracy` | fraction | `True` | quality must be matched before any systems win is meaningful |
| `paired_delta_vs_target` | fraction | `True` | same-example accuracy deltas avoid independent-sample noise |
| `paired_ci95_low_vs_target` | fraction | `True` | reviewers need uncertainty on quality deltas |
| `ttft_ms_p50` | ms | `True` | time to first token captures prefill and packet/KV setup cost |
| `ttft_ms_p95` | ms | `True` | tail latency is a serving-quality constraint |
| `tpot_ms_p50` | ms/token | `True` | time per output token captures decode path efficiency |
| `tpot_ms_p95` | ms/token | `True` | tail decode latency catches scheduler/cache path regressions |
| `itl_ms_p50` | ms/token | `True` | inter-token latency is a standard online serving metric |
| `goodput_requests_per_s` | requests/s | `True` | systems claims need throughput at an accepted latency SLO |
| `generated_tokens_per_s` | tokens/s | `True` | token throughput makes vLLM/SGLang comparisons interpretable |
| `prefill_ms_p50` | ms | `True` | packet and KV baselines differ most in prefill/cache setup |
| `decode_ms_p50` | ms | `True` | decode latency separates one-time packet setup from token generation |
| `peak_gpu_memory_gb` | GB | `True` | KV/cache baselines may trade bytes for GPU memory |
| `hbm_read_bytes_per_request` | bytes/request | `True` | hardware-facing systems wins should show memory-traffic movement |
| `hbm_write_bytes_per_request` | bytes/request | `True` | cache-sharing and quantized-KV methods write different state volumes |
| `pcie_or_nvlink_rx_bytes_per_request` | bytes/request | `True` | multi-device or host-device communication must be accounted |
| `pcie_or_nvlink_tx_bytes_per_request` | bytes/request | `True` | transferred source state is central to the claim boundary |
| `payload_bytes_per_request` | bytes/request | `True` | packet/KV/text baselines must report actual communicated bytes |
| `framed_bytes_per_request` | bytes/request | `True` | wire-format overhead must be separated from raw payload bytes |
| `transferred_source_state_bytes` | bytes/request | `True` | cache/KV/vector baselines move source state that packets avoid exposing |
| `source_text_exposed` | bool | `True` | source-private claims fail if source text is exposed |
| `source_kv_exposed` | bool | `True` | C2C/KVComm/QJL/TurboQuant expose different source-state objects |
| `source_hidden_or_score_vector_exposed` | bool | `True` | raw hidden/score vectors are not equivalent to fixed-byte packets |
| `hardware` | string | `True` | GPU SKU and interconnect define systems comparability |
| `software_commit` | string | `True` | serving/runtime versions must be reproducible |
| `batch_size_or_concurrency` | count | `True` | serving throughput depends on load shape |
| `input_output_token_counts` | tokens | `True` | TTFT/TPOT/goodput are meaningless without token lengths |
| `wall_time_s` | seconds | `True` | total runtime catches benchmark harness and setup regressions |

## Checks

| Check | Pass |
|---|---:|
| `all_required_quality_latency_memory_traffic_metrics_listed` | `True` |
| `headline_benchmarks_are_arc_and_openbookqa` | `True` |
| `hellaswag_marked_diagnostic_until_label_copy_gate` | `True` |
| `packet_rows_include_cached_and_end_to_end_modes` | `True` |
| `serving_substrates_include_vllm_and_sglang` | `True` |
| `competitor_rows_include_cache_kv_quantized_baselines` | `True` |
| `source_exposure_flags_required_for_every_baseline` | `True` |
| `external_baselines_have_primary_sources` | `True` |
| `profiler_requirements_include_serving_json_nsight_and_nvml` | `True` |
| `no_ssh_policy_recorded` | `True` |
| `native_systems_complete_false_until_measurements_ingested` | `True` |
| `native_win_non_claims_recorded` | `True` |

## Runbook

1. Freeze ARC-Challenge test and OpenBookQA test as headline native rows; keep HellaSwag diagnostic until the label-copy gate passes.
2. Run target-only vLLM and target-only SGLang first to establish serving baselines.
3. Run LatentWire in cached-source-packet mode and end-to-end source-scoring mode.
4. Run same-byte visible text and source-label/trained-label controls on the same example IDs.
5. Run C2C, KVComm/KVCOMM, QJL, and TurboQuant rows only as native or faithful source-state baselines with exposure flags set.
6. Collect serving JSON plus Nsight/NVML traces for every row; do not use SSH inside this artifact.
7. Mark `native_systems_complete=true` only after every required baseline has accuracy, latency, memory, traffic, payload bytes, and exposure fields.

## Non-Claims

- Do not claim native throughput, TTFT, TPOT, HBM, or peak-memory wins until all required native rows are measured.
- Do not claim LatentWire beats C2C, KVComm, KVCOMM, QJL, TurboQuant, vLLM, or SGLang from byte-floor accounting alone.
- Do not run or require SSH in the artifact; remote execution must be done manually by the user from the runbook.
- Do not promote HellaSwag native rows until the HellaSwag method gate beats source-label and trained-label copy controls.
