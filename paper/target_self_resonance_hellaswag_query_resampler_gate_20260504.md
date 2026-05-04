# Target Self-Resonance HellaSwag Query-Resampler Gate

## Status

This is a negative method gate. It should not be framed as a positive ICLR
result.

Current paper readiness remains below ICLR full-paper standard. The target
self-resonance oracle results show controllability/headroom, but the reusable
learned interface has not survived held-out target-only controls.

## Gate

Script:
`scripts/build_target_self_resonance_hellaswag_query_resampler_gate.py`

Tests:
`tests/test_build_target_self_resonance_hellaswag_query_resampler_gate.py`

Artifacts:

- `results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train32_validation72_80/`
- `results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train64_validation72_80/`

The gate trains a small Perceiver/Q-Former-style query bottleneck on official
HellaSwag train rows. Learned query slots attend over target prompt token
embeddings and emit `8` target-native soft-prefix slots. The frozen target
model then scores candidate continuations from only:

```text
soft slots + fixed anchor + candidate continuation
```

The compressed path does not receive the original HellaSwag context text.

Controls:

- full prompt reference;
- raw chunk-mean prefix;
- slots-only trained target cache;
- zero prefix;
- same-norm random prefix;
- wrong-row shuffled query-resampler prefix;
- candidate-score derangement.

Pass condition: query-resampler must have no nonfinite rows, match or beat
chunk and slots-only agreement, improve KL over chunk/slots/best control, and
stay under the mean-KL cap.

## Results

Default `32` train rows, validation `72:80`:

| condition | agreement | mean KL |
|---|---:|---:|
| query-resampler | 0.500000 | 0.244325 |
| chunk mean | 0.625000 | 0.426502 |
| slots-only | 0.000000 | 0.393548 |
| shuffled query | 0.500000 | 0.252318 |

The query-resampler improves KL over chunk and slots-only, but it fails
agreement and barely separates from the shuffled-prompt control. Mean normalized
attention entropy is `0.950380`, so the learned queries remain diffuse rather
than forming a sharp reusable prompt interface.

Capacity rescue with `64` train rows, same validation `72:80`:

| condition | agreement | mean KL |
|---|---:|---:|
| query-resampler | 0.500000 | 0.557950 |
| chunk mean | 0.625000 | 0.426502 |
| slots-only | 0.500000 | 0.198077 |

The rescue fails harder. Mean normalized attention entropy returns to
approximately `1.0`, and slots-only becomes the strongest KL control. More
target self-compression training is therefore not the next highest-value move.

## Decision

Demote the plain target query-resampler branch. The evidence says:

- compact target soft-prefix slots remain controllable in oracle mode;
- a small query bottleneck can fit training rows;
- held-out behavior is not reliably prompt-specific;
- wrong-row and target-only controls are too strong.

The next branch should add source-conditioned information only if it directly
attacks the control-separation failure. The highest-value direction is a
common-basis / relative-feature packet with strict wrong-source controls, not
more target-only query tuning.

## Systems Interpretation

This gate records useful systems accounting but is not yet a systems win.

- Packet: `8` Qwen soft slots.
- Raw fp16 packet size: `14,336` bytes.
- Default query-resampler parameters: `928,512`.
- Peak RSS on Mac default run: about `3425` MiB.

A real systems claim still needs source-present quality at the same byte budget
and native comparisons against cache/vector communication or serving baselines
such as C2C, KVComm, vLLM/SGLang, KIVI/QJL/TurboQuant-style compression, and
plain text/token baselines.

## Lay Explanation

This experiment asked whether a small learned module could read a question,
compress it into eight hidden "summary tokens", and let the model answer from
those hidden tokens instead of the original question. It learned something on
the training examples, but on new examples the hidden tokens were not specific
enough to the right question. A wrong-question hidden token often worked almost
as well, so this is not yet real communication.

## Next Exact Gate

Implement a source-conditioned common-basis packet gate:

- represent source and target signals relative to shared anchors or learned
  sparse features;
- include wrong-source, shuffled-anchor, target-only, and candidate-deranged
  controls;
- require paired held-out gains over chunk/slots/zero/random controls before
  expanding to more benchmarks.
