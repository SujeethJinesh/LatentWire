# Answer-Likelihood Control Analysis

- status: `answer_likelihood_controls_fail`
- score field: `answer_mean_logprob`
- unavailable controls: `target_only`

## Summaries

| row | n | finite | correct | mean score |
|---|---:|---:|---:|---:|
| live | 4 | 4 | 0 | -7.025400 |
| zero_source | 4 | 4 | 0 | -6.925437 |
| shuffled_source_salt1 | 4 | 4 | 0 | -7.048394 |
| slots_only | 4 | 4 | 0 | -7.025400 |

## Paired Controls

| control | mean live-control | wins | losses | ties | pass |
|---|---:|---:|---:|---:|---|
| zero_source | -0.099963 | 1 | 3 | 0 | False |
| shuffled_source_salt1 | 0.022994 | 3 | 1 | 0 | False |
| slots_only | 0.000000 | 0 | 0 | 4 | False |

## Best Control

- wins/losses/ties: `0/4/0`
- mean live-best-control delta: `-0.115530`

## Checks

- `row_count_parity`: `True`
- `finite_scores`: `True`
- `mean_delta_vs_each_control`: `False`
- `per_example_best_control_wins`: `False`
