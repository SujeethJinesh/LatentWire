# Citation Audit

| Paper Sentence | Citation | Evidence Source | Notes |
|---|---|---|---|
| Quamba2 argues for channel-order preservation and activation persistence. | `chiang2025quamba2` | arXiv:2503.22879 | Used as the headline assumption being stress-tested. |
| DecDEC demonstrates short-horizon dynamic salient channel selection. | `park2025decdec` | arXiv:2412.20185 | Paper differentiates by horizon, model class, architecture, and benchmarks. |
| ParoQuant is a reasoning W4A16 rotation baseline. | `liang2025paroquant` | arXiv:2511.10645 | Our implementation is algorithmic, not upstream kernel reproduction. |
| SLQ reports non-compounding in a near-lossless regime. | `helcig2026slq` | arXiv:2605.02404 | Our KL packet tests a different aggressive W4A16 regime. |
| SmoothQuant uses outlier migration for activation-to-weight migration. | `xiao2023smoothquant` | arXiv:2211.10438 | Terminology disambiguation footnote. |
| KIVI and KVQuant act on KV cache tensors. | `liu2024kivi`, `hooper2024kvquant` | arXiv:2402.02750, arXiv:2401.18079 | Differentiated from activation-channel protection. |
| PM-KVQ and PMPD address long-CoT quantization. | `liu2025pmkvq`, `chen2025pmpd` | arXiv:2505.18610, arXiv:2410.13461 | Adjacent long-decode work. |
| QEP and LAQuant motivate error-propagation caution. | `arai2025qep`, `choi2026laquant` | arXiv:2504.09629, arXiv:2605.08755 | Supports mechanism framing, not direct baseline. |

Every empirical sentence in the paper should also appear in
`docs/reproducing_results.md` with a reproduction script and tolerance.
