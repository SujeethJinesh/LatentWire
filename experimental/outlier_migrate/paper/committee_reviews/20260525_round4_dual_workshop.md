# Committee Review Round 4: After Round 3 Fixes

Draft reviewed:
`experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

Commit reviewed: `e78c40ff`

## Efficient Reasoning Workshop Rubric

Score: 7.5/10

The revised draft now draws a clear boundary between mechanism evidence and
deployable systems contribution. The explicit statement that M11b is not a
runtime/memory implementation removes the main overclaim risk from round 3.
The ParoQuant contextualization is also stronger: it prevents the paper from
making an accidental "W4A16 is impossible" claim. The remaining weakness for
this venue is presentation: the method table is dense, and a reviewer may need
to read it twice to understand why ParoQuant and M11b are different kinds of
evidence.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: Add a compact visual or summary table before submission that
  separates "channel-set methods", "rotation baseline", and "mechanism
  analyses".
- NICE_TO_HAVE: Compress exact decimals in prose while keeping full precision
  in tables or reproducibility docs.

## Context Beyond the Window Workshop Rubric

Score: 8/10

The added state-management paragraph materially improves workshop fit. The
paper now explains channel protection as a stale internal state summary rather
than only a quantization detail. This makes the contribution relevant to
long-window compression and state internalization. The KL limitation is also
properly scoped. The main improvement left is a schematic of the three
mechanisms; without it, the paper is still numerically heavy.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: Add a three-mechanism schematic or compact textual callout in
  the final figure pass.
- NICE_TO_HAVE: Tie ThoughtFlow-FP8 into future work only if space remains;
  it is not needed for this submission.

## Adversarial Reviewer

Score: 7/10

The paper now handles the Nemotron nuance honestly. It says the checker passed
on top-10 but the top-5 transfer claim is not clean. A skeptical reviewer can
still argue that M11b is "just more budget", but the paper mostly agrees and
uses that as the point. The ParoQuant comparison is no longer damaging because
the draft positions it as a stronger baseline via a different mechanism.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: The phrase "first partially positive result" should remain
  paired with "not deployable" and "budget sensitivity" throughout copyedits.
- NICE_TO_HAVE: Add a short reviewer-facing note in the release docs mapping
  checker labels to paper framing.

## Statistical Reviewer

Score: 7/10

The positive-gap effective sample limitation is now explicit. The draft still
has many method attempts, but it labels them as preregistered/audit-authorized
mechanism tests rather than treating the best result as a post-hoc discovery.
The remaining statistical limitation is the wide Granite M11b CI. The paper
does not overclaim it, so this is acceptable for a workshop.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: In the final reproducibility package, make the included-trace
  filtering auditable for every recovery script.
- NICE_TO_HAVE: Add a one-row note to the method table with included trace
  counts if space allows.

## Overall Score

Efficient Reasoning: 7.5

Context Beyond the Window: 8

Adversarial/statistical average pressure: 7

Dual-workshop convergence score: 7.75

This is not yet convergence because the score improved by `0.75` from round 3.
No new LOAD_BEARING critiques surfaced, and no structural changes are
recommended. If the next two rounds remain within `0.5` score improvement with
no new LOAD_BEARING critiques, the paper will meet the convergence rule.

## Required Fixes Before Round 5

No LOAD_BEARING fixes. Recommended next work is release/reproducibility and
final figure/table polish rather than additional paper restructuring.
