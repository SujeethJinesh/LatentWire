# HellaSwag Anchor-Relative Hidden-Innovation Multi-Slice Stress

- pass gate: `False`
- slice count: `5`
- total eval rows: `5120`
- contiguous validation prefix: `True`
- weighted selected accuracy: `0.469531`
- weighted best label-copy accuracy: `0.461523`
- weighted score-only accuracy: `0.456445`
- min delta vs best label-copy: `0.004883`
- min CI95 low vs best label-copy: `-0.014648`
- min delta vs score-only bagged: `0.008789`
- min score-only CI95 low: `0.000464`
- min delta vs zero-hidden: `0.008789`
- all corrupted-hidden controls below label-copy: `True`
- all anchor controls below label-copy: `True`
- jackknife slices passing: `0/5`
- packet: `2B` raw / `5B` framed

## Interpretation

This aggregate gate tests the strongest reviewer objection to the dense hidden-innovation result: whether the lift survives a train-only common-basis bottleneck. The answer is currently no. The anchor-relative packet preserves a small positive aggregate lift over label-copy and score-only controls, and anchor/corrupted controls stay below label-copy, but every slice misses the strict 0.02 margin and label-copy CI requirement. Treat this as a common-basis blocker and systems-friendly diagnostic, not as a promoted ICLR headline method.
