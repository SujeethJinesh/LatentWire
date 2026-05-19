# Vacation Decision: KL Dense-Grid Fallback

## Situation

The Phase 9 KL accumulation preregistration asks for `KL(BF16 || Q)_t` over
decode positions `1-20000`, with a preregistered dense-grid fallback if
full-position evaluation is infeasible.

Implementing every-position full-vocabulary KL is technically possible, but it
would add a full-vocabulary `log_softmax` and CPU transfer at every one of
20,000 decode positions for every trace and quantized regime. This is
substantially heavier than the existing endpoint perplexity scorers, which only
materialize log-probabilities for a 512-token endpoint window. It risks
spending a large share of the remaining GPU budget before ParoQuant and M11b
replication, both of which are higher-priority reviewer-critical items.

## Options Considered

1. Run every-position KL exactly as the primary spec.
2. Use the preregistered dense grid before observing any KL values.
3. Skip KL accumulation and proceed to ParoQuant.

## Decision

Use option 2: run the preregistered dense grid:

- all positions `1-512`;
- every 10th position from `513-2000`;
- every 50th position from `2001-10000`;
- every 100th position from `10001-20000`.

The runner records `dense_grid_fallback=true` in `kl_positions.json`. The paper
must describe this as dense-grid KL, not every-position KL.

## Rationale

The dense grid preserves the scientific question that matters for the paper:
whether KL is flat, roughly linear/sublinear, superlinear, or too noisy to
classify across the long-decode horizon. It also keeps KL accumulation from
crowding out ParoQuant and M11b replication.

No KL values have been measured before this decision.

## What Would Invalidate This Decision

If the human wants an exact every-position KL curve regardless of runtime, this
decision should be revised and the full-position mode can be run later using
the same preregistration and runner.
