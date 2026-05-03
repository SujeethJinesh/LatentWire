# HellaSwag Non-Qwen Score-Simplex Receiver Gate

Date: 2026-05-03

## Status

This is a negative receiver-fusion gate, not a promoted ICLR method.

The gate tests the highest-priority Mac-local common-basis branch after the
TinyLlama-to-Phi and Qwen-to-Phi receiver-family failures: both models score
the same four HellaSwag endings, so the source and target score vectors can be
row-centered, projected into an orthonormal 3D candidate-contrast basis, and
optionally aligned with train-prefix SVD before receiver fusion.

## Artifact

- `results/source_private_hellaswag_nonqwen_score_simplex_receiver_gate_20260503_validation1024_2048/`
- `scripts/build_source_private_hellaswag_nonqwen_score_simplex_receiver_gate.py`
- `tests/test_build_source_private_hellaswag_nonqwen_score_simplex_receiver_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_nonqwen_score_simplex_receiver_gate.py \
  --output-dir results/source_private_hellaswag_nonqwen_score_simplex_receiver_gate_20260503_validation1024_2048 \
  --run-date 2026-05-03
```

## Result

Across HellaSwag validation `1024:2048` with `128` train rows per slice and
`384` held-out eval rows per slice:

| Method | Weighted eval accuracy |
| --- | ---: |
| Phi target-only | `0.263021` |
| TinyLlama packet-only | `0.506510` |
| score-simplex receiver | `0.442708` |
| target-or-packet oracle | `0.619792` |

Per-slice readout:

| Slice | Target | Packet | Receiver | Oracle | Receiver - packet | CI95 low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024:1536` | `0.270833` | `0.489583` | `0.494792` | `0.611979` | `+0.005208` | `-0.031250` |
| `1536:2048` | `0.255208` | `0.523438` | `0.390625` | `0.627604` | `-0.132812` | `-0.182292` |

The receiver beats target-only on both slices, but it does not beat packet-only
on either slice under the strict paired-CI rule. Destructive controls also do
not separate cleanly.

## Interpretation

The score-simplex common basis is not sufficient as a positive receiver method.
It confirms that source score geometry contains useful signal, but the best
current use of that signal is still the source-private packet itself. Free
fusion overfits the small train prefix and overrules a strong packet too often.

Lay explanation: the tiny TinyLlama hint remains better than Phi alone. The new
combiner tried to compare how TinyLlama and Phi ranked the four endings, but it
made worse decisions than simply trusting the hint.

## Decision

- Weaken the score-simplex Fourier/SVD receiver branch.
- Do not promote this as cross-model latent reasoning.
- Keep the underlying packet result alive because the oracle still has
  `+0.113281` headroom over packet-only.
- Next highest-value branch should be packet-preserving and common-coordinate:
  anchor-relative score packets or a target-side acceptor that defaults to the
  packet and only overrides when trained source-destroying controls stay below
  the matched packet.
