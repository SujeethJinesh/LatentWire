# HellaSwag Receiver Headroom Decomposition

- pass gate: `False`
- receiver headroom gate: `True`
- train-selected selector improvement gate: `False`
- train/eval split: validation `0:1024` train, `1024:10042` eval
- TinyLlama packet-only eval accuracy: `0.629741`
- Qwen target-score eval accuracy: `0.483034`
- best Tiny+Qwen oracle eval accuracy: `0.692947`
- best oracle delta vs packet-only: `0.063207`
- train-selected simple selector eval accuracy: `0.608228`
- train-selected simple selector delta vs packet-only: `-0.021513`

## Lay Explanation

This experiment asks a simple question: if TinyLlama sends a tiny answer hint and Qwen also has its own guess, are their mistakes different enough that a receiver could combine them? The oracle row is the unrealistic best case that peeks at the answer and picks whichever model was right. The train-selected selector row is the fair cheap version: it learns a rule only on the first validation prefix and then applies that frozen rule to the heldout rows.

## Interpretation

The decomposition is diagnostic, not a promoted positive method. A positive oracle gap means Qwen and TinyLlama contain complementary candidate information, so the receiver branch is alive. Failure of the train-prefix selector means simple confidence thresholds do not recover that complementarity. The next live branch should therefore learn a common-basis or selective residual receiver under train-only selection, while retaining the fixed 2B raw / 5B framed packet boundary and destructive controls.
