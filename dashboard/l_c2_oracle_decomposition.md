# L_C2 Oracle Decomposition

VERDICT: ORACLE_ONLY
DEPLOYABLE_RUNNER_ATTEMPT: BLOCKED_WITH_PROOF
PROMOTION_ALLOWED: false

## Strict Result

Source: `results/mps_first_strict_20260605/L_C2_control_trained_lcf_lite_proxy/summary.json`

- achieved gate rows: `500`
- receiver accuracy: `0.168`
- oracle fuser accuracy: `1.000`
- wrong-row accuracy: `0.022`
- zero-source accuracy: `0.168`
- oracle gain vs receiver: `+0.832`, CI `[+0.798, +0.864]`, MDE `0.034`
- oracle gain vs wrong-row: `+0.978`, CI `[+0.964, +0.990]`
- confirm rows scored: `0`

## Gold-Leakage Finding

The strict runner builds the fuser target from `tool_answer`, which is computed by evaluating the SVAMP gold equation:

- `safe_equation_value(row["metadata"]["equation"])`
- `out.append(row | {"strict_index": i, "tool_answer": answer})`
- in `l_c2`, `fuser.append(answer)` and `control_trained_fuser_prediction: answer`

This is a gold-aware oracle. It measures a ceiling and the receiver gap to perfect arithmetic, not transmissible latent information.

## Deployable Attempt Status

No gold-free source/receiver feature cache is present for this proxy in `results/mps_first_strict_20260605/L_C2_control_trained_lcf_lite_proxy/`. The available rows contain receiver predictions, answer-derived oracle predictions, wrong-row answer controls, and zero-source controls. They do not contain a trainable source packet feature that excludes the answer key.

Because the only available fuser feature is answer-derived, there is no honest deployable L_C2 runner to execute locally. The deployable attempt is therefore blocked with proof rather than marked positive or killed.

## Required Gold-Free Replacement

A runnable L_C2 candidate must produce, on frozen dev/gate rows only:

- receiver features and predictions before seeing any source answer,
- source hidden/logit/cache features that do not include `answer`, `correct`, `tool_answer`, equation value, or label fields,
- wrong-row, zero-source, label-shuffle, source-index, and equal-byte text controls,
- paired bootstrap gain versus receiver and each control,
- an access manifest with `confirm_rows_scored: 0`.

Until that exists, L_C2 remains an oracle ceiling and not a deployable method.
