# Workshop Hardening Convergence Log

Date: 2026-05-29

Target: 2nd Workshop on Efficient Reasoning @ COLM 2026, systems track.

## Phase 0: Desk-Rejection Gates

All desk-rejection gates pass after mechanical fixes:

- Main text ends on page 10; references begin on page 11.
- COLM 2026 submission style is used.
- Paper and final supplement pack were anonymized; local workspace paths in the pack were replaced by placeholders.
- Load-bearing claims are present in the main text.
- LLM Use Disclosure remains present and explicit.

See `desk_reject_gate.md`.

## Round 1

| Committee | Lens | Mean score |
|---|---|---:|
| A | Quantization / algorithms | 2.25 / 5 |
| B | Systems / efficiency | 2.25 / 5 |
| C | Methodology / statistics | 2.50 / 5 |

Overall mean: 2.33 / 5.

Top blocking themes:

1. Novelty versus ParoQuant and no-new-method risk.
2. Analytical-only systems evidence.
3. Small-N / no-gap / ratio-metric fragility.
4. Descriptive protocol language too strong.
5. MATH-500 risk of being overread as recovery replication.

Fixes applied:

- Changed protocol language to "provisionally suggests" and "checklist"; local confirmation packet is required.
- Added local ParoQuant-style implementation caveat.
- Clarified MATH-500 supports drift generalization, not intervention recovery.
- Changed paired margins to descriptive language.
- Added no-gap estimand caveat.
- Added aggregate systems-cost row and device/cache/batching caveat.
- Scoped ParoQuant+M11b sub-additivity to the tested Granite packet.

See `reviews_round_1.md` and `rebuttal_and_fix_plan_round_1.md`.

## Round 2

| Committee | Lens | Score |
|---|---|---|
| A | Quantization / algorithms | Borderline |
| B | Systems / efficiency | Weak accept |
| C | Methodology / statistics | Weak accept |

Mean score on a 5-point reject-to-accept scale: 3.33 / 5.

Round 2 found no remaining workshop-level blocker. Residual issues are honest limitations:

- The paper is not a new quantizer and does not beat official ParoQuant with a new method.
- Systems evidence is analytical, not profiler/kernel evidence.
- Intervention packets are small and ratio CIs are heavy-tailed.
- The calibration checklist is descriptive, not held-out validated.
- ParoQuant-style baseline is a local implementation following reported design, not official code.

One safe Round 2 fix was applied: the Reproducibility Statement now points to the exact E15 33-test Holm artifact and fields.

See `reviews_round_2.md` and `rebuttal_and_fix_plan_round_2.md`.

## Figure/Table Pass

Four main-text figures were regenerated with a shared colorblind-safe style. Figure 1 now marks the 53--67% final-position drift band. No data were changed.

See `figure_audit.md`.

## Convergence Decision

Stop condition reached: Round 2 produced no new blocking issue, and remaining concerns are honest limitations that cannot be fixed without new experiments, official baseline reruns, or kernel/profiler implementation. Those are explicitly out of scope for this final hardening loop.

Final verdict: `CONVERGED_READY_TO_SUBMIT` for a scoped COLM Efficient Reasoning workshop mechanism/regime paper.

