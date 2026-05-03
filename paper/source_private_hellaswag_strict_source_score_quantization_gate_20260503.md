# HellaSwag Strict Source-Score Quantization Gate

Date: 2026-05-03

## Status

This is a reviewer-control closure gate, not an ICLR-positive method.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The current story is that source-private fixed-byte packets can carry
task evidence under strict controls. The exact ICLR blocker is still a
packet-beating receiver/common-basis method or native systems evidence on
NVIDIA hardware.

## Artifact

- `results/source_private_hellaswag_strict_source_score_quantization_gate_20260503_validation0_9216/`
- `scripts/build_source_private_hellaswag_strict_source_score_quantization_gate.py`
- `tests/test_build_source_private_hellaswag_strict_source_score_quantization_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_strict_source_score_quantization_gate.py \
  --output-dir results/source_private_hellaswag_strict_source_score_quantization_gate_20260503_validation0_9216 \
  --bootstrap-samples 2000 \
  --run-date 2026-05-03
```

## Result

The gate trains score-code decoders only on HellaSwag train-split source-score
caches (`1495` unique train rows from the existing bagged source caches), then
freezes them and evaluates on the strict Qwen HellaSwag validation `0:9216`
surface.

| Variant | Raw / framed bytes | Accuracy | Delta vs candidate-only | CI95 low |
| --- | ---: | ---: | ---: | ---: |
| candidate-only packet | `1B / 4B` | `0.525499` | reference | reference |
| source argmax | `1B / 4B` | `0.479384` | `-0.046115` | `-0.052083` |
| rank-order majority | `1B / 4B` | `0.479384` | `-0.046115` | `-0.052083` |
| z-score q2 vector majority | `1B / 4B` | `0.450629` | `-0.074870` | `-0.082794` |
| z-score q5 vector majority | `3B / 6B` | `0.444444` | `-0.081055` | `-0.089410` |
| top2-margin q16 majority | `1B / 4B` | `0.398655` | `-0.126845` | `-0.137807` |
| z-score q3 vector majority | `2B / 5B` | `0.385200` | `-0.140299` | `-0.151801` |
| z-score q4 vector majority | `2B / 5B` | `0.378906` | `-0.146593` | `-0.157986` |

Positive method pass: `False`.

Reviewer-control audit complete: `True`; all calibrated source-score code
variants remain below candidate-only.

## Interpretation

This closes the reviewer concern that we had not tested calibrated source-score
quantization on the strict HellaSwag row. Quantized score-vector, rank-order,
top-2 margin, and source-argmax packets at matched or larger byte budgets do
not explain or improve the current strict packet.

Scientifically, this weakens score-only and score-code branches as positive
methods. The live method branch should move away from shallow candidate-score
acceptors and toward target-loss query/soft-prefix connectors or conditional
hidden-innovation packets with stronger source-destroying controls.

Lay explanation: we tried sending compressed versions of the source model's
four answer scores instead of only the answer it picked. A small decoder learned
from train examples how to interpret those score codes. On the large frozen
validation set, those score codes still did worse than the tiny answer-choice
packet.

## Decision

- Do not promote source-score quantization as an ICLR-positive method.
- Keep the source-score rows in the main reviewer table as controls.
- The next Mac-local method gate should be target-loss query/soft-prefix or
  conditional hidden-innovation transfer, because score quantization and
  score-simplex receivers are now saturated on the current evidence.
