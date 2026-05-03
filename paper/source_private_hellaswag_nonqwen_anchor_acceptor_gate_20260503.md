# HellaSwag Non-Qwen Anchor Acceptor Gate

Date: 2026-05-03

## Status

This is a packet-preservation and receiver-falsification gate, not an ICLR
positive receiver result.

The gate was added after the score-simplex Fourier/SVD receiver failed. It
tests a stricter packet contract: the source can use cached TinyLlama scores
only to select a tiny discrete code, and the Phi receiver sees only the packet
candidate, the discrete code, and Phi's own target scores. It cannot inspect
TinyLlama text, hidden states, KV cache, logits, or raw scores.

## Artifact

- `results/source_private_hellaswag_nonqwen_anchor_acceptor_gate_20260503_validation1024_2048/`
- `scripts/build_source_private_hellaswag_nonqwen_anchor_acceptor_gate.py`
- `tests/test_build_source_private_hellaswag_nonqwen_anchor_acceptor_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_nonqwen_anchor_acceptor_gate.py \
  --output-dir results/source_private_hellaswag_nonqwen_anchor_acceptor_gate_20260503_validation1024_2048 \
  --run-date 2026-05-03
```

## Result

Across HellaSwag validation `1024:2048`:

| Method | Weighted eval accuracy |
| --- | ---: |
| Phi target-only | `0.263021` |
| TinyLlama packet-only | `0.506510` |
| cautious anchor acceptor | `0.506510` |
| target-or-packet oracle | `0.619792` |

The positive receiver gate fails because the acceptor does not improve over
packet-only. The preservation gate passes on `2/2` slices: the selected cautious
policy rejects all source-code override candidates and falls back to a
candidate-only packet requiring `1B` raw / `4B` framed.

Per-slice readout:

| Slice | Target | Packet | Acceptor | Oracle | Acceptor - packet | Raw bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024:1536` | `0.270833` | `0.489583` | `0.489583` | `0.611979` | `0.000000` | `1` |
| `1536:2048` | `0.255208` | `0.523438` | `0.523438` | `0.627604` | `0.000000` | `1` |

The strongest rejected non-fallback rows showed why the caution rule is
necessary. On `1024:1536`, `anchor_relative_k32` looked positive on the select
split by one example but hurt held-out eval by `-0.005208`. On `1536:2048`,
`packet_prob_q8` looked positive by one select example but hurt held-out eval
by `-0.052083`.

## Interpretation

The anchor/quantile code is not a positive receiver contribution yet. The useful
thing learned here is stricter: the current non-Qwen packet evidence does not
need the failed fusion machinery, and tiny train-prefix override gains are not
trustworthy. The safest method on this surface is still the packet itself.

Lay explanation: we let the source attach a tiny extra code saying what kind of
score pattern it saw. The receiver was allowed to use that code only if a small
held-out selector split said it was safe. The safe choice was to ignore the
extra code and keep the original hint.

## Decision

- Do not promote anchor-relative acceptors as a positive method.
- Keep the `1B` raw / `4B` framed candidate-only packet-preservation row as a
  systems/privacy contribution on this non-Qwen surface.
- The next ICLR-positive method branch must either improve the packet itself on
  strict Qwen `0:9216` or introduce a receiver with a stronger training signal
  than 64-row accept/reject supervision.
