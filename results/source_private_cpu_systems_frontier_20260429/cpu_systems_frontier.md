# Source-Private CPU Systems Frontier

- rows: `41`
- pass rows: `31`
- fail / near-miss rows: `10`
- minimum passing accuracy: `0.418`
- maximum passing payload bytes: `6.0`
- minimum passing model-packet valid rate: `0.537`

## Rows

| Contribution | Method | Surface | Status | Accuracy | Target | Best control | Bytes | Valid | CI low vs target | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| byte-rate systems frontier | 2-byte diagnostic packet | core seed29 | `pass` | 1.000 | 0.250 | 0.250 | 2.0 | - | - | Oracle at 2.0 bytes; JSON/free-text need 21.0/17.0 bytes; query-aware span needs 14.0 bytes; full log is 183.2x larger. |
| byte-rate systems frontier | 2-byte diagnostic packet | holdout seed30 | `pass` | 1.000 | 0.250 | 0.250 | 2.0 | - | - | Oracle at 2.0 bytes; JSON/free-text need 21.0/17.0 bytes; query-aware span needs 14.0 bytes; full log is 186.7x larger. |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | same-codebook | `pass` | 1.000 | 0.250 | 0.250 | 6.0 | - | 0.713 | raw_sign=0.307; scalar-control delta=0.750 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | same-codebook | `pass` | 1.000 | 0.250 | 0.258 | 6.0 | - | 0.715 | raw_sign=0.188; scalar-control delta=0.742 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | same-codebook | `pass` | 1.000 | 0.250 | 0.250 | 6.0 | - | 0.713 | raw_sign=0.207; scalar-control delta=0.750 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | same-codebook | `pass` | 1.000 | 0.250 | 0.262 | 6.0 | - | 0.713 | raw_sign=0.182; scalar-control delta=0.738 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | same-codebook | `pass` | 1.000 | 0.250 | 0.250 | 6.0 | - | 0.711 | raw_sign=0.201; scalar-control delta=0.750 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | remap 101 | `pass` | 0.463 | 0.250 | 0.264 | 6.0 | - | 0.156 | raw_sign=0.332; scalar-control delta=0.199 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | remap 103 | `pass` | 0.508 | 0.250 | 0.266 | 6.0 | - | 0.199 | raw_sign=0.316; scalar-control delta=0.242 |
| learned scalar packet | 6-byte slot/no-intercept scalar packet | remap 107 | `pass` | 0.492 | 0.250 | 0.250 | 6.0 | - | 0.186 | raw_sign=0.330; scalar-control delta=0.242 |
| learned Wyner-Ziv syndrome packet | 2-byte scalar WZ packet | remap 101 | `pass` | 0.418 | 0.250 | 0.264 | 2.0 | - | - | raw_sign=0.301; qjl=0.396; canonical_rasp=0.350; query_text_at_budget=0.250; packet_vs_query_text_oracle=7.0x |
| learned Wyner-Ziv syndrome packet | 4-byte scalar WZ packet | remap 101 | `pass` | 0.432 | 0.250 | 0.264 | 4.0 | - | - | raw_sign=0.326; qjl=0.461; canonical_rasp=0.494; query_text_at_budget=0.250; packet_vs_query_text_oracle=3.5x |
| learned Wyner-Ziv syndrome packet | 6-byte scalar WZ packet | remap 101 | `pass` | 0.463 | 0.250 | 0.264 | 6.0 | - | - | raw_sign=0.332; qjl=0.447; canonical_rasp=0.494; query_text_at_budget=0.250; packet_vs_query_text_oracle=2.3x |
| learned Wyner-Ziv syndrome packet | 2-byte scalar WZ packet | remap 103 | `pass` | 0.436 | 0.250 | 0.250 | 2.0 | - | - | raw_sign=0.303; qjl=0.439; canonical_rasp=0.363; query_text_at_budget=0.250; packet_vs_query_text_oracle=7.0x |
| learned Wyner-Ziv syndrome packet | 4-byte scalar WZ packet | remap 103 | `pass` | 0.475 | 0.250 | 0.266 | 4.0 | - | - | raw_sign=0.328; qjl=0.461; canonical_rasp=0.520; query_text_at_budget=0.250; packet_vs_query_text_oracle=3.5x |
| learned Wyner-Ziv syndrome packet | 6-byte scalar WZ packet | remap 103 | `pass` | 0.508 | 0.250 | 0.266 | 6.0 | - | - | raw_sign=0.316; qjl=0.484; canonical_rasp=0.520; query_text_at_budget=0.250; packet_vs_query_text_oracle=2.3x |
| learned Wyner-Ziv syndrome packet | 2-byte scalar WZ packet | remap 107 | `pass` | 0.418 | 0.250 | 0.246 | 2.0 | - | - | raw_sign=0.309; qjl=0.393; canonical_rasp=0.350; query_text_at_budget=0.250; packet_vs_query_text_oracle=7.0x |
| learned Wyner-Ziv syndrome packet | 4-byte scalar WZ packet | remap 107 | `pass` | 0.445 | 0.250 | 0.246 | 4.0 | - | - | raw_sign=0.326; qjl=0.453; canonical_rasp=0.506; query_text_at_budget=0.250; packet_vs_query_text_oracle=3.5x |
| learned Wyner-Ziv syndrome packet | 6-byte scalar WZ packet | remap 107 | `pass` | 0.492 | 0.250 | 0.232 | 6.0 | - | - | raw_sign=0.330; qjl=0.457; canonical_rasp=0.506; query_text_at_budget=0.250; packet_vs_query_text_oracle=2.3x |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 101 | `near-miss` | 0.494 | 0.250 | 0.295 | 4.0 | - | 0.184 | scalar=0.426; relative_minus_scalar=0.068 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 103 | `near-miss` | 0.520 | 0.250 | 0.256 | 4.0 | - | 0.213 | scalar=0.496; relative_minus_scalar=0.023 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 107 | `near-miss` | 0.506 | 0.250 | 0.355 | 4.0 | - | 0.199 | scalar=0.502; relative_minus_scalar=0.004 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 109 | `near-miss` | 0.477 | 0.250 | 0.279 | 4.0 | - | 0.170 | scalar=0.451; relative_minus_scalar=0.025 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 113 | `near-miss` | 0.473 | 0.250 | 0.279 | 4.0 | - | 0.164 | scalar=0.436; relative_minus_scalar=0.037 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 127 | `near-miss` | 0.453 | 0.250 | 0.275 | 4.0 | - | 0.146 | scalar=0.428; relative_minus_scalar=0.025 |
| canonical RASP remap robustness | 4-byte canonical RASP | remap 131 | `near-miss` | 0.506 | 0.250 | 0.281 | 4.0 | - | 0.197 | scalar=0.434; relative_minus_scalar=0.072 |
| canonical RASP larger-slice confirmation | 4-byte canonical RASP | remap 127 | `pass` | 0.442 | 0.250 | 0.275 | 4.0 | - | 0.152 | scalar=0.361; relative_minus_scalar=0.081 |
| canonical RASP cross-family falsification | 4-byte canonical RASP | core -> holdout | `fail` | 0.207 | 0.250 | 0.494 | 4.0 | - | - | scalar=0.225; controls_ok=False |
| canonical RASP cross-family falsification | 4-byte canonical RASP | holdout -> core | `pass` | 0.492 | 0.250 | 0.250 | 4.0 | - | - | scalar=0.375; controls_ok=True |
| consistency posterior negative ablation | 4-byte consistent posterior packet | core -> holdout large | `fail` | 0.354 | 0.250 | 0.355 | 4.0 | - | - | order-mismatch matched source in this failed row; scalar=0.370; controls_ok=False |
| consistency posterior negative ablation | 4-byte consistent posterior packet | holdout -> core large | `pass` | 0.495 | 0.250 | 0.250 | 4.0 | - | - | scalar=0.368; controls_ok=True |
| model-emitted source packet | Qwen3.5-0.8B | n160 seed29 | `pass` | 1.000 | 0.250 | 0.256 | 2.0 | 1.000 | - | n=160; exact_id_parity=True |
| model-emitted source packet | Qwen3.5-0.8B | n160 seed31 | `pass` | 1.000 | 0.250 | 0.250 | 2.0 | 1.000 | - | n=160; exact_id_parity=True |
| model-emitted source packet | Qwen3.5-2B | n160 seed29 | `pass` | 1.000 | 0.250 | 0.256 | 2.0 | 1.000 | - | n=160; exact_id_parity=True |
| model-emitted source packet | Qwen3.5-4B | n64 seed29 | `pass` | 1.000 | 0.250 | 0.250 | 2.0 | 1.000 | - | n=64; exact_id_parity=True |
| model-emitted source packet | Gemma 4 E2B | n64 seed29 | `pass` | 1.000 | 0.250 | 0.250 | 2.0 | 1.000 | - | n=64; exact_id_parity=True |
| model-emitted source packet | Granite 3.3 2B | n160 seed29 | `pass` | 0.631 | 0.250 | 0.256 | 1.1 | 0.537 | - | n=160; exact_id_parity=True |
| model-emitted source packet | Granite 3.3 2B | n160 seed31 | `pass` | 0.631 | 0.250 | 0.250 | 1.1 | 0.537 | - | n=160; exact_id_parity=True |
| model-emitted source packet | Granite raw-log/no-trace | n160 seed31 | `fail` | 0.250 | 0.250 | 0.250 | 0.0 | 0.000 | - | n=160; exact_id_parity=True |
| target model decoder ablation | Qwen3 target decoder | core n64 CPU | `pass` | 0.656 | 0.250 | 0.250 | 2.0 | 1.000 | - | n=64; generated_tokens=13.0 |
| target model decoder ablation | Qwen3 target decoder | holdout n64 CPU | `pass` | 0.719 | 0.250 | 0.266 | 2.0 | 1.000 | - | n=64; generated_tokens=13.0 |

## Caveat

CPU/local latency is not endpoint TTFT or throughput. Cross-family rows remain explicitly failed outside the promoted same-family/remap scope.
