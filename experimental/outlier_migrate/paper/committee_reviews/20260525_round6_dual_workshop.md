# Committee Review Round 6: Capstone Review

Draft reviewed:
`experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

Commit reviewed: `cf3b1a7d`

## Efficient Reasoning Workshop Rubric

Score: 7.5/10

The paper's best claim is now stable: channel-set protection is fragile under
long-decode W4A16 reasoning, budget tuning partially helps, and rotation is a
stronger baseline mechanism. This is workshop-suitable if the submission is
honest about not providing a runtime implementation.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: none requiring paper restructuring.
- NICE_TO_HAVE: figure polish.

## Context Beyond the Window Workshop Rubric

Score: 8/10

The paper remains well aligned. It has a long-window internal-state story,
state compression relevance, SSM/hybrid model relevance, and negative findings
with clear mechanism value.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: none requiring paper restructuring.
- NICE_TO_HAVE: schematic polish.

## Adversarial/Statistical Pressure

Score: 7/10

No new attack changes the paper's core framing. The strongest attacks remain
known and already disclosed: small effective sample sizes, ParoQuant beating
M11b, and M11b's model-dependent best budget.

Severity-tagged critiques:

- LOAD_BEARING: none.
- SUBSTANTIVE: none requiring paper restructuring.
- NICE_TO_HAVE: final consistency audit.

## Overall Score

Efficient Reasoning: 7.5

Context Beyond the Window: 8

Dual-workshop convergence score: 7.75

This is round 6. The committee loop stops by the six-round cap with no new
LOAD_BEARING critiques in rounds 4-6.
