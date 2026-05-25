# Sprint Final Report

Generated: 2026-05-25 UTC

## Executive Status

Paper readiness: first-complete workshop draft is locked in source and PDF,
with six committee rounds complete and no open load-bearing critique.

Current story: decode-position channel-set drift is robust across the tested
long-reasoning W4A16 setting. Hard switching policies are actively harmful,
smoothing alone is insufficient, budget-tuned EMA gives a partial
cross-model remedy, and ParoQuant-style rotation is the stronger baseline on
Granite-Small.

Remaining blocker for ICLR-positive-method readiness: no channel-set method is
yet a clean deployable cross-model positive method. M11b is signal-bearing but
budget-sensitive across models.

## Final Paper State

- Title: **Decode-Position Channel Drift at Long-Decode Reasoning: Budget
  Tuning as a Partial Cross-Model Remedy**
- Source: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- PDF: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`
- Release copy: `release/paper/paper.{tex,pdf}`
- Build: `./build.sh` passes with underfull box warnings only.

## Headline Results

- Strict top-1% set-leaving: Granite `0.566234756098`, Nemotron
  `0.533713200380`, DeepSeek `0.670572916667`, Falcon-H1
  `0.673611111111`.
- M11b Granite top-5: `0.449284091125`, CI
  `[-1.300900018791, 1.000790626547]`.
- M11b Nemotron top-5: `0.456736183270`, CI
  `[0.345887792388, 0.794986745913]`, but below static top-10.
- M11b Nemotron top-10: `0.814739798903`, CI
  `[0.254439288178, 0.922552264880]`, margin over static top-10
  `0.220356055286`.
- ParoQuant Granite: `0.753776848891`, CI
  `[0.477044452405, 1.003769554323]`.
- KL dense-grid means: static `0.150302750276`, DecDEC proxy
  `0.132701884446`, M11 `0.131406585383`; AR decays `0.4387-0.5147`.

## Committee Review

Committee review completed through the six-round cap.

| Round | Efficient Reasoning | Context Beyond Window | Average | Load-bearing critiques |
|---|---:|---:|---:|---|
| 3 | 7.0 | 7.0 | 7.0 | Addressed |
| 4 | 7.5 | 8.0 | 7.75 | None new |
| 5 | 7.5 | 8.0 | 7.75 | None new |
| 6 | 7.5 | 8.0 | 7.75 | None new |

The paper fits Context Beyond Window slightly better because the state-summary
and long-horizon drift framing is central. Efficient Reasoning remains viable
because ParoQuant, W4A16, and deployment scope are explicitly discussed.

## Release Package

Release path: `release/`.

Implemented:

- minimal rewritten package, not copied from experimental runners;
- Apache-2.0 license;
- reproducibility scripts for every paper claim family;
- claim-to-script mapping and citation audit;
- extension guide and rotation+budget composition stub;
- CPU tests passing;
- clean-clone FAST_VERIFY verification passed twice.

Important limitation: `release/` does not implement full model-inference GPU
reruns. Scripts now fail explicitly outside `--dry-run` and `--fast-verify` so
frozen-claim replay is not misrepresented. Full GPU reproduction remains in the
archived experimental packet ecosystem and would require a dedicated runner
implementation.

Verification:

- `./.venv_gpu/bin/python -m pytest release/tests`: `7 passed`.
- `bash src/scripts/reproduce_all.sh --fast-verify` from `release/`: passed.
- Full-mode invocation without flags: correctly raises an explicit
  not-implemented error.

## Audits

Audit reports are under top-level `docs/`:

- `docs/code_audit_correctness.md`
- `docs/code_audit_hardcoding.md`
- `docs/code_audit_errors.md`
- `docs/code_audit_reproducibility.md`
- `docs/adversarial_review.md`
- `docs/statistical_audit.md`
- `docs/figure_table_audit.md`
- `docs/headline_number_audit.md`
- `docs/cross_document_consistency.md`
- `docs/audit_summary.md`

All critical audit findings were addressed. The main fix was removing the false
full-mode pass from release scripts.

## Autonomous Decisions

Key decisions are recorded in `swarm/vacation_decisions/`, including the
Nemotron Path C framing decision and committee rounds 3-6. The most important
judgment call was to frame M11b as a partial budget remedy rather than a clean
positive method, because the best Nemotron budget shifts to top-10 and the
Granite CI remains wide.

## GPU Hours

`swarm/state.json` last tracked cumulative GPU hours at approximately
`278.643`. The old 200-hour soft cap was intentionally exceeded under later
human authorization for full-quality KL, ParoQuant, and Nemotron evidence.

## Acceptance Estimate

Based on committee scores and fit:

- Context Beyond Window workshop: 55-70%.
- Efficient Reasoning workshop: 45-60%.
- ICLR main-track without new positive method: low, roughly 10-20%.

These are subjective estimates, not completion criteria.

## Recommended Next Move

Submit the current draft to the workshop after human copyedit and final venue
formatting. For ICLR follow-up, prioritize a true positive method rather than
more failed channel-set variants: rotation+budget composition, M18b
cross-tensor at top-5, or M11c budget-resolution sweep are the highest-value
next branches.
