# HellaSwag Score-Channel True Candidate-Text Permutation Gate

- smoke pass gate: `True`
- promotion pass gate: `False`
- eval rows: `128`
- permuted evaluations: `1024`
- permutations per example: `8`
- identity accuracy: `0.468750`
- remapped accuracy: `0.468750`
- canonical packet consistency: `1.000000`
- max accuracy delta from identity: `0.000000`
- accuracy std across permutations: `0.000000`
- wrong-remap accuracy: `0.176758`
- wrong-remap CI95 high vs remapped: `-0.248022`

## Interpretation

This physically reruns the Qwen HellaSwag continuation-likelihood source scorer after reordering candidate endings, then maps display predictions back to canonical candidate IDs. A pass means the score-channel candidate-id packet is content-stable under true display-order perturbation on this frozen slice. It does not prove the learned hidden fixed-hybrid row is candidate-order invariant; that requires rerunning the hidden-innovation pipeline under the same physical permutations.

## Lay Explanation

We actually shuffled the answer endings before asking the local source model to score them. Then we translated the displayed option number back to the original option number. If the same original answer is selected after shuffling, the score-channel hint is following the candidate text rather than the slot it appeared in.
