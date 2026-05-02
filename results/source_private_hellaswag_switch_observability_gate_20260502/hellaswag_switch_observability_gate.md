# HellaSwag Switch Observability Gate

- pass gate: `False`
- default row: `source_plus_qwen_rff`
- default delta vs packet-only: `-0.000398`
- default CI95 low vs packet-only: `-0.001095`
- default validation AUC help-vs-harm: `0.5539845123283766`
- best validation-oracle threshold delta: `0.000199`
- best diagnostic AUC help-vs-harm: `0.561171556341835`

## Lay Explanation

This experiment does not try to make a new model answer better. It checks whether the available signals contain enough information to tell when Qwen should overrule the TinyLlama packet. If that signal is not visible even to simple diagnostic learners, then more selector tuning is unlikely to produce an ICLR-strength method.

## Interpretation

This gate is a decision surface for the HellaSwag branch. A pass would mean the current packet/Qwen score surface still has learnable switch information worth turning into a source-private code. A fail, especially with a weak validation-oracle threshold row, means the branch should stop tuning selectors and move to a different source representation or a true joint connector.
