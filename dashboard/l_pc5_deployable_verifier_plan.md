# L_PC5 Deployable Verifier Plan

VERDICT: ORACLE_ONLY
DEPLOYABLE_RUNNER_ATTEMPT: BLOCKED_WITH_PROOF
PROMOTION_ALLOWED: false

## Strict Result

Source: `results/mps_first_strict_20260605/L_PC5_private_verifier_receiver_candidates/summary.json`

- achieved prompts: `500`
- candidate rows: `8000`
- candidates per prompt: `16`
- prompts with at least one correct candidate: `500`
- receiver target accuracy: `0.334`
- receiver plus verifier accuracy: `1.000`
- receiver plus source accuracy: `1.000`
- oracle verifier gain vs target: `+0.666`, CI `[+0.624, +0.706]`, MDE `0.042`
- confirm rows scored: `0`

## Gold-Leakage Finding

The strict runner scores each candidate against the answer key:

- `answer = int(row["tool_answer"])`
- `correct = cand == answer`
- `verifier_scores.append(1.0 if correct else -float(distance))`
- `source_scores.append(0.8 if correct else -0.02 * distance)`

That verifier is gold-aware. It confirms the candidate pool has a large rerank ceiling, but it is not a deployable verifier packet.

## Deployable Attempt Status

The earlier gold-blind EXP4 verifier cache is below the screening floor: `results/mps_first/L_PC5_private_verifier_receiver_candidates/summary.json` reports `36` prompts with at least one correct candidate, below the `80`-prompt rerank gate. The strict `500`-prompt result reaches the floor only by constructing answer-centered candidates and answer-aware verifier/source scores.

No local artifact currently provides a gold-blind verifier score over at least `80` nondegenerate prompts. The deployable L_PC5 attempt is blocked with proof.

## Required Gold-Free Replacement

A runnable deployable L_PC5 candidate needs:

- at least `300` dev/gate prompts and `16` candidates per prompt from a stronger generator,
- at least `80` prompts with one or more correct candidates before screening,
- `target_score`, `source_score`, and `verifier_score` fields computed without showing the gold answer or equation value,
- receiver, source-index, and equal-byte text controls,
- paired gain versus the best control, not just versus target-only,
- `confirm_rows_scored: 0`.

Until that cache exists, L_PC5 remains an oracle ceiling and not a deployable method.
