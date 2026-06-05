# L_PC1 Reconciliation

VERDICT: PENDING_FRESH_DEPLOYABLE_OR_KILL
PROMOTION_ALLOWED: false
SCOPED_POSITIVE: false

## Cached Strict Result

Source: `results/mps_first_strict_20260605/L_PC1_cross_family_specialist_ceiling/summary.json`

- achieved rows: `512`
- receiver score-only accuracy: `0.384765625`
- selected receiver-plus-source accuracy: `0.427734375`
- source-label accuracy: `0.384765625`
- zero-hidden accuracy: `0.384765625`
- wrong-example-hidden accuracy: `0.376953125`
- paired gain vs receiver: `+0.04296875`, CI `[+0.015625, +0.068359375]`, MDE `0.02734375`
- conditional MI of source label given receiver prediction: `0.0` bits
- confirm rows scored: `0`

## Reconciliation

The strict cached row is not a raw source-label-copy result. In the raw rows:

- `source_label_prediction` equals `receiver_score_only_prediction` on `1.000000` of rows.
- `selected_prediction` equals `source_label_prediction` on `0.837890625` of rows.
- `selected_prediction` accuracy is `0.427734375`, while source-label and receiver score-only accuracy are both `0.384765625`.

So the `0.428` selected-packet accuracy can exceed the `0.385` source-label baseline because the selected packet differs from raw source-label on about `16.2%` of rows. This is consistent with a cached receiver-family packet selection effect, not with a deployable source-label-only communication win.

## Why It Is Not Claim-Eligible Yet

The prompt requested fresh cross-family pairs: Qwen to Phi plus at least one other pair, HellaSwag plus at least one other task, floor at least `500` rows per pair, and controls against receiver/source-index/equal-byte text. The current evidence is a single cached HellaSwag Qwen-strict-to-Phi slice and does not include the fresh pair/task matrix or equal-byte text/source-index controls.

## Required Next Evidence

L_PC1 can become `SCOPED_POSITIVE` only if a fresh non-confirm dev/gate run shows:

- at least two source-to-receiver model pairs,
- at least two tasks including HellaSwag,
- at least `500` rows per pair,
- packet accuracy beats receiver, source-index, and equal-byte text controls with positive paired CI,
- follow-rate and control-collapse diagnostics are logged,
- no confirm or confirmation source paths appear in the access manifest.

Until then, L_PC1 is a small cached ceiling signal and the confirm-or-kill verdict remains pending.
