# HellaSwag Strict Channel-Selector Gate

Date: 2026-05-03

## Status

This is a strict packet-policy improvement, not a solved learned receiver or
common-basis method.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The current story is that source-private fixed-byte packets can carry
task evidence under strict controls. The exact ICLR blocker is still a
cross-family-stable learned receiver/common-basis method or native systems
evidence against KV/state-transfer baselines.

## Artifact

- `results/source_private_hellaswag_strict_channel_selector_gate_20260503_validation0_9216/`
- `scripts/build_source_private_hellaswag_strict_channel_selector_gate.py`
- `tests/test_build_source_private_hellaswag_strict_channel_selector_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_strict_channel_selector_gate.py \
  --output-dir results/source_private_hellaswag_strict_channel_selector_gate_20260503_validation0_9216 \
  --bootstrap-samples 5000 \
  --run-date 2026-05-03
```

## Result

The gate recomputes all nine strict Qwen HellaSwag `1024`-row slices from the
candidate-only audit prediction files and compares fixed channels plus
train-prefix selectors against the `1B` candidate-only packet.

| Method | Train rows | Eval rows | Accuracy | Delta vs candidate-only | CI95 low | Helps / harms | Improved slices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate-only packet | `0` | `9216` | `0.525499` | reference | reference | reference | reference |
| fixed hybrid vote-on-score-agreement | `0` | `9216` | `0.531141` | `+0.005642` | `+0.002713` | `125 / 73` | `9 / 9` |
| prefix global channel selector | `1024` | `8192` | `0.528320` | `+0.001221` | `-0.002930` | `170 / 160` | `6 / 8` |
| fixed vote | `0` | `9216` | `0.526801` | `+0.001302` | `-0.002713` | `191 / 179` | `7 / 9` |
| fixed trained label | `0` | `9216` | `0.483290` | `-0.042209` | `-0.049696` | `452 / 841` | `0 / 9` |
| fixed score mean/vote | `0` | `9216` | `0.479384` | `-0.046115` | `-0.052303` | `230 / 655` | `0 / 9` |

Positive method pass: `True` for the fixed hybrid packet policy.

The hybrid rule is:

```text
if hidden_mean_prediction == score_mean_prediction:
    emit vote_prediction
else:
    emit hidden_mean_prediction
```

The learned train-prefix selectors do not pass. The best learned selector picks
`vote_prediction` from the first `1024` rows, then gains only `+0.001221` on the
remaining `8192` rows with a negative CI95 low.

Oracle headroom remains large:

- selected/vote/trained/score oracle: `0.593099`
- all-channel oracle: `0.646159`

## Interpretation

This revives a narrow packet-policy contribution that the candidate-only audit
made look saturated. Candidate-only is the minimum packet contract, but a
source-side static policy over already-computed packet channels gives a small,
paired-significant accuracy lift on the strict `0:9216` surface.

This does not solve the stronger ICLR concern. The result is still a discrete
candidate-policy row on a multiple-choice benchmark. It should not be framed as
a general latent language, a learned receiver, or proof of cross-model latent
reasoning. The learned selectors in this same gate fail, so shallow per-row
selector learning remains weak.

Scientifically, the useful signal appears to be a conservative disagreement
rule: when hidden-mean and score-mean agree, the bagged vote can improve the
default; otherwise, the hidden-mean candidate is safer.

Lay explanation: we had several tiny hints from the source model. Sending only
the default hint worked well, but a fixed rule that switches to the vote hint
when another score hint agrees gives a small reliable improvement. A learned
rule trained on one slice did not find a better pattern.

## Decision

- Promote fixed hybrid vote-on-score-agreement as the strongest current strict
  HellaSwag packet-policy row: `0.531141` at `1B` raw / `4B` framed output
  packet, with source-side compute hidden from the receiver.
- Keep candidate-only as the minimum byte/exposure contract and as the key
  limitation: the benchmark surface is still candidate-choice communication.
- Do not promote train-prefix channel selection as a learned receiver method.
- Next exact gate: test whether this hybrid policy survives one strict
  cross-family receiver surface or whether a target-loss query/soft-prefix /
  conditional hidden-innovation method can beat this new strict packet-policy
  row with paired uncertainty.
