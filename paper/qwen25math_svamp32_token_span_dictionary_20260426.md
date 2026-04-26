# Qwen2.5-Math SVAMP32 Token/Span Dictionary Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-derived positive method plus
  medium, seed-repeat, uncertainty, source-control, systems, and cross-family
  gates
- current story: the SVAMP32 C2C-headroom surface remains valuable, but the
  source-readout family is now failing consistently
- blocker: fold-local token/span sparse dictionaries do not recover clean C2C
  headroom IDs

## Gate

This was the stricter follow-up to the negative sparse-anchor projection and
all-layer token query-bottleneck gates. It tested a fold-local sparse dictionary
over source token states, then decoded C2C numeric residue signatures through
the same strict SVAMP32 candidate pool.

Configuration:

- source model: `Qwen/Qwen2.5-Math-1.5B`
- target tokenizer/model: `Qwen/Qwen3-0.6B`
- surface:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- target set:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
- feature layers: `mid,last`
- outer folds: `8`
- atoms: `32`
- top-k atoms: `2`
- random projection dim: `128`
- dictionary iterations: `8`
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  boundary-only, target-only, and slots-only

Promotion rule:

- matched correct at least `10/32`
- target floor preserved
- at least `2/6` clean source-necessary recoveries
- clean destructive-control union `0/6`

## Evidence

| Condition | Correct | Clean Correct |
|---|---:|---:|
| matched | 7/32 | 0/6 |
| zero-source | 7/32 | 0/6 |
| shuffled-source | 8/32 | 0/6 |
| label-shuffled | 6/32 | 0/6 |
| same-norm-noise | 8/32 | 0/6 |
| boundary-only | 7/32 | 0/6 |
| target-only | 8/32 | 0/6 |
| slots-only | 6/32 | 0/6 |

Other diagnostics:

- candidate-pool clean gold coverage: `6/6`
- clean source-necessary IDs: `0`
- control clean union: `0`
- mean dead atom rate: `0.0000`
- mean codebook perplexity: `28.5363`
- estimated sidecar bytes: `22`

The dictionary learned active atoms and high codebook perplexity, so the
failure is not a dead-codebook artifact. It simply does not select the clean
C2C-headroom answers.

## Decision

Kill the current source-readout / sparse-dictionary family on the Qwen2.5-Math
SVAMP32 C2C-headroom surface.

This family now has multiple adjacent failures with the same root cause:

- sparse-anchor projection: `0/6` clean recovery
- constrained sparse-anchor projection: `0/6` clean recovery
- all-layer source-token query bottleneck: `0/6` clean recovery
- fold-local token/span sparse dictionary: `0/6` clean recovery

Do not spend another cycle tuning source-token readout, random projections,
dictionary seed, top-k, atom count, or byte budget on this same surface unless
a new objective introduces target-safe selection or an explicitly source-derived
teacher signal.

## Next Branch

Promote the next live branch to target-safe output-aware dynalign selector /
repair. The reason is evidence-based:

- raw GSM70 dynalign is seed-unstable and saturated as a method
- but dynalign remains the strongest real mechanism clue from older artifacts
- source-readout methods cannot recover the current SVAMP32 clean C2C IDs
- the next method needs to preserve target-correct rows while selectively
  accepting source-derived output-aware alignment wins

Next exact gate:

- design a strict target-safe accept/fallback or repair gate over output-aware
  dynalign candidates on the existing exact-ID surface
- include target-only, zero-source, shuffled-source, and selector-only controls
- promote only if it recovers at least `2/6` clean source-necessary IDs without
  target-floor regression

## Artifacts

- analyzer:
  `scripts/analyze_svamp32_token_span_dictionary_probe.py`
- tests:
  `tests/test_analyze_svamp32_token_span_dictionary_probe.py`
- JSON:
  `results/qwen25math_svamp32_token_span_dictionary_20260426/probe.json`
  - sha256:
    `877a5970ccd244cdaf0731426934c8764e6b63d16f12bebc2102dafb46e7a64e`
- Markdown:
  `results/qwen25math_svamp32_token_span_dictionary_20260426/probe.md`
  - sha256:
    `478a14c702ab16ccba98fe5a9656892f4c8a06aef0ab92e4ba943b2013fce54c`
- Log:
  `.debug/qwen25math_svamp32_token_span_dictionary_20260426/logs/probe.log`
  - sha256:
    `4a31f54820ec7dd9d5b14fbd26d0f02ca195198b154d229f80d5a23291038f46`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_token_span_dictionary_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_token_span_dictionary_probe.py`
