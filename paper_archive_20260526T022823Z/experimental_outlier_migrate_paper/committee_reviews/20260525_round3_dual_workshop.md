# Committee Review Round 3: Post-Nemotron Integration

Draft reviewed:
`experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

Commit reviewed: `7b11d305`

Artifact basis:
- Valid Nemotron Path C packet:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`
- Granite M11b:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`
- ParoQuant Granite:
  `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
- KL accumulation:
  `experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z`
- M26 stable core:
  `experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z`

## Efficient Reasoning Workshop Rubric

Score: 7/10

The paper now has a credible workshop-level efficiency story: long-decode
reasoning quantization fails when channel protection assumes stable channel
identity, ParoQuant gives a stronger rotation baseline, and M11b shows a
partial budget-tuning remedy. The scope is honest and deployment-relevant:
small reasoning models, W4A16 PTQ, and hybrid Mamba-2 / parallel-hybrid
architectures. The main weakness is that the paper still does not provide a
runtime or memory-efficiency implementation for M11b. It should therefore
avoid presenting M11b as a systems contribution and instead frame it as a
mechanism-guided partial remedy.

Severity-tagged critiques:

- LOAD_BEARING: The abstract and contribution list need one explicit sentence
  saying M11b is not yet a deployable efficiency method because no runtime or
  memory implementation is provided.
- SUBSTANTIVE: The method-results table is dense; the paper needs a compact
  "what each result means" paragraph before or after the table to prevent
  reviewers from treating every PASS string as equally strong.
- NICE_TO_HAVE: Replace one numeric table with a plotted method-comparison
  figure before final submission.

## Context Beyond the Window Workshop Rubric

Score: 8/10

The paper fits this workshop well. It is about state over long generated
windows, internal memory/compression trade-offs, and why local calibration
statistics become stale over long decode. Negative and diagnostic results are
welcome in this venue, and the paper has enough mechanism detail to be useful:
set-leaving, FFT/autocorrelation, KL accumulation, layer stratification, and
budget tuning. The strongest remaining risk is that the paper currently says
little about state-management implications beyond quantization. A short
discussion paragraph should connect decode-position drift to cache/state
management explicitly.

Severity-tagged critiques:

- LOAD_BEARING: Add a short discussion sentence or paragraph tying
  channel-set drift to long-window state management, not only quantization.
- SUBSTANTIVE: The KL result is presented as weakening compound error; the
  limitations should clarify it is only Granite-Small and only the three
  tested regimes.
- NICE_TO_HAVE: A simple schematic of the three-mechanism framework would
  help this workshop more than another numeric table.

## Adversarial Reviewer

Score: 6/10

The draft is more honest after the Nemotron salvage, but a skeptical reviewer
will attack the M11b framing. Granite top-5 has a wide CI crossing zero;
Nemotron top-5 has a positive CI but loses to static top-10; Nemotron top-10
passes but changes the claimed best budget. The paper must keep saying
"budget sensitivity" rather than "M11b method solves it." Another attack is
ParoQuant: if rotation works better, why not stop there? The current answer is
reasonable but should be sharper: ParoQuant is a baseline/context, while this
paper characterizes why channel-set protection fails and when budget helps.

Severity-tagged critiques:

- LOAD_BEARING: Ensure every PASS label in prose is accompanied by the control
  comparison that qualifies it. In particular, do not let
  `PASS_M11B_NEMOTRON_REPLICATES` imply top-5 transfer.
- SUBSTANTIVE: State that ParoQuant's implementation is algorithmic, not a
  full upstream speed/quality reproduction with optimized kernels.
- NICE_TO_HAVE: Add a one-line "reviewer warning" in the table caption that
  checker decision strings are not identical to final paper framing.

## Statistical Reviewer

Score: 6/10

The paper reports medians and bootstrap CIs consistently, and it avoids
claiming significance where CIs are wide. The no-gap filtering is still a
statistical vulnerability. Positive-gap trace counts are reported, but the
paper should explicitly state that recovery is undefined for no-gap traces and
that this can change the effective sample size. Multiple method attempts are
also not corrected statistically; that is acceptable for preregistered
mechanism exploration only if the limitation is explicit.

Severity-tagged critiques:

- LOAD_BEARING: Add a limitations sentence that recovery CIs are over 8--10
  positive-gap traces in several packets, not all 12 traces.
- SUBSTANTIVE: Add one sentence that multiple Phase 9 methods were
  preregistered/audited and should be interpreted mechanistically, not as a
  post-hoc winner search.
- NICE_TO_HAVE: Add a table column with included trace counts for every Phase
  9 row.

## Overall Score

Efficient Reasoning: 7

Context Beyond the Window: 8

Adversarial/statistical average pressure: 6

Dual-workshop average used for convergence tracking: 7.0

This is the first post-Nemotron integrated round. Convergence has not yet
started because several new LOAD_BEARING critiques surfaced.

## Required Fixes Before Round 4

1. Add explicit M11b non-deployable-method language.
2. Add long-window state-management connection.
3. Qualify checker PASS labels versus paper framing.
4. Clarify KL scope and positive-gap effective sample size in limitations.
5. State that Phase 9 method attempts are preregistered/audited mechanism
   tests, not a post-hoc search over methods.
