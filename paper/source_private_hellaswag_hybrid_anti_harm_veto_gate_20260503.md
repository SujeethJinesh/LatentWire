# HellaSwag Hybrid Anti-Harm Veto Gate

Date: 2026-05-03

## Status

This is a killed shallow-router branch, not a new positive method.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The current story is that source-private fixed-byte packets can carry
task evidence under strict destructive controls. The exact ICLR blocker is
still a learned receiver/common-basis method that beats packet-only, broader
cross-family stability, or native systems evidence against KV/state-transfer
baselines.

## Artifact

- `results/source_private_hellaswag_hybrid_anti_harm_veto_gate_20260503_validation0_9216/`
- `scripts/build_source_private_hellaswag_hybrid_anti_harm_veto_gate.py`
- `tests/test_build_source_private_hellaswag_hybrid_anti_harm_veto_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_hybrid_anti_harm_veto_gate.py \
  --run-date 2026-05-03
```

## Gate

The gate tested whether the current fixed hybrid packet policy has exploitable
anti-harm structure:

```text
candidate = selected_prediction
hybrid = vote_prediction if hidden_mean_prediction == score_mean_prediction else hidden_mean_prediction
vetoed = candidate when frozen source-side veto fires, otherwise hybrid
```

The rule is packet-preserving: the receiver still sees one final candidate id,
with `1B` raw / `4B` framed payload and no source text, KV cache, raw hidden
vectors, raw scores, or extra receiver-visible veto bit.

To avoid heldout tuning, the first strict `1024`-row slice is split into:

- `0:512`: define candidate one-rule vetoes from source-side features;
- `512:1024`: select one rule;
- `1024:9216`: primary strict evaluation.

The frozen same rule is then applied to cached Qwen-to-Phi rows `1024:2048`,
excluding the first `128` rows per `512`-row slice as in the prior
cross-family gate.

## Result

The predeclared fit/select veto fails. It selects the rule
`selected_id <= 1`, which is an option-position rule rather than a semantic
anti-harm detector.

| Row | Eval rows | Accuracy | Delta vs candidate-only | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed hybrid, full `0:9216` | `9216` | `0.531141` | `+0.005642` | reference | reference |
| candidate-only, full `0:9216` | `9216` | `0.525499` | reference | `-0.005642` | `-0.008681` |
| fit/select anti-harm veto, heldout `1024:9216` | `8192` | `0.529663` | `+0.002563` | `-0.003052` | `-0.005249` |
| margin-only fit/select veto, heldout `1024:9216` | `8192` | `0.532715` | `+0.005615` | `0.000000` | `0.000000` |
| leave-one-slice-out diagnostic veto, full `0:9216` | `9216` | `0.531033` | `+0.005534` | `-0.000109` | `-0.000977` |
| candidate/hybrid oracle veto, full `0:9216` | `9216` | `0.539063` | `+0.013563` | `+0.007921` | `+0.006293` |

Switch-level readout for the main heldout veto:

- fixed hybrid has `112` helps, `66` harms, and `69` neutral switches on
  heldout `1024:9216`;
- the veto fires on `121` rows;
- it avoids `32` hybrid harms but sacrifices `57` hybrid helps;
- it has `0/8` heldout slices positive versus fixed hybrid.

Cross-family falsification also fails:

| Row | Eval rows | Accuracy | Delta vs candidate-only | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen fixed hybrid to Phi | `768` | `0.467448` | `+0.011719` | reference | reference |
| frozen anti-harm veto to Phi | `768` | `0.462240` | `+0.006510` | `-0.005208` | `-0.014323` |

On the two cached Qwen-to-Phi slices, the frozen veto is negative versus fixed
hybrid on both slices. It avoids `4` harms but misses `8` hybrid helps.

Pass gate: `False`.

## Interpretation

The current shallow source-side packet features do not reliably separate
hybrid helps from hybrid harms. The main rule found by the fit/select gate is
an option-position rule, and it behaves like random same-coverage vetoing:
both lose `-0.003052` accuracy versus fixed hybrid on the strict heldout
surface. The margin-only variant selects no-op, which preserves fixed hybrid
but does not improve it.

Scientifically, this is still useful. The candidate/hybrid oracle reaches
`0.539063`, so anti-harm headroom exists. But shallow one-rule selective
classification over the current packet fields cannot access it. That weakens
further source-side veto searches and promotes the next branch: a target-aware
receiver/common-basis signal that can use target evidence without collapsing
below packet-only.

Lay explanation: the hybrid hint sometimes changes the answer choice and
sometimes that change is bad. We tried to train a simple warning rule that
would say, "this switch looks risky, keep the old hint." It learned a rule
based mostly on answer-choice position, and on new rows it threw away more
good switches than bad switches.

## Decision

- Kill shallow packet-preserving anti-harm vetoes as the next ICLR method
  branch unless a new feature source is introduced.
- Keep fixed hybrid vote-on-score-agreement as the strongest strict HellaSwag
  packet-policy row.
- Use the oracle gap as motivation for the next receiver/common-basis gate,
  not as evidence for another source-side one-rule selector.
- The next exact Mac-local gate should test a target-loss query/soft-prefix or
  conditional hidden-innovation receiver against fixed hybrid on the same
  strict `1024:9216` and cached Qwen-to-Phi surfaces.
