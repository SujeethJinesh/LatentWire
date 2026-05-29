# Numerical Consistency Report

Date: 2026-05-29

Scope: abstract, introduction, main results prose, Table 1, appendix method table, `CLAIM_AUDIT.md`, and the committed score-cache references listed in `CLAIM_AUDIT.md`.

## Fixes Applied

| Item | Status | Action |
|---|---|---|
| Nemotron ParoQuant recovery | FIX_APPLIED | Canonical main-paper rounding is now `1.05` with CI `[1.01, 1.29]` everywhere. `CLAIM_AUDIT.md` claim 15 was updated from `1.047` to `1.05`. |
| Nemotron M11b top-10 CI | PASS | All checked locations use `0.815`, CI `[0.158, 0.906]`, and margin `0.220`. No `[0.254, 0.923]` remnants remain. |
| DeepSeek ParoQuant tail risk | FIX_APPLIED | Abstract now states `0.756 on DeepSeek (high median, negative lower CI)`. Tables retain CI `[-0.246, 0.855]`. |
| Granite tight-clip worst trace | FIX_APPLIED | Table 1 now uses `-26.37 to 0.601`, matching prose and `CLAIM_AUDIT.md`. |
| Systems-cost scale | FIX_APPLIED | Section 6 now states that the `83--177M pJ/token` aggregate is the per-projection envelope multiplied by corrected modules and active experts. |
| "preregistered" language | FIX_APPLIED | Uncited uses were softened to `fixed` or `planned`. Remaining validation language is descriptive/future-facing. |

## Canonical Headline Numbers

| Claim family | Canonical value(s) | Status |
|---|---:|---|
| Four-model strict set-leaving band | `53--67%` | PASS |
| Per-model AIME set-leaving | Granite `0.566`, Nemotron `0.534`, DeepSeek `0.671`, Falcon `0.674` | PASS |
| MATH-500 set-leaving | Granite `0.561`, DeepSeek `0.651`, Falcon `0.675`; deltas `-0.005`, `-0.020`, `0.002` | PASS |
| Nemotron M11b top-10 | `0.815`, CI `[0.158, 0.906]`, static-top-10 margin `0.220` | PASS |
| ParoQuant rotation | Granite `0.754 [0.477, 1.00]`; Nemotron `1.05 [1.01, 1.29]`; DeepSeek `0.756 [-0.246, 0.855]`; Falcon `0.381 [0.0645, 0.547]` | PASS |
| ParoQuant vs M11b margins | Nemotron `0.232`; Falcon `0.337` | PASS |
| ParoQuant+M11b | Granite `0.565`, trails ParoQuant by `0.189` | PASS |
| DriftRot tight clip | Granite `0.754 -> 0.922`, worst `-26.37 -> 0.601`; DeepSeek `0.518`; Falcon `0.390` | PASS |
| K-RES | `-12.294` vs tight ParoQuant `5.940` in audit artifacts; main text keeps qualitative kill wording | PASS |
| M-SURFACE surfaces | `0.546 / 0.484 / 0.486 / 0.618 / 0.602 / 0.887 / 0.799` | PASS |

## Title Note

The title was not changed because the user requested options rather than an automatic edit. Options for the next human decision:

1. Current: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and When Rotation Helps`
   - Accurate for the regime map, but may sound slightly like a rotation-method contribution.
2. Softer mechanism title: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and Why Rotation Helps`
   - Better mechanism framing; slightly repetitive.
3. Most conservative: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails Under Long Decode`
   - Avoids implying a rotation contribution, but underplays the strongest baseline/remedy finding.

No numerical mismatches remain in the checked files.
