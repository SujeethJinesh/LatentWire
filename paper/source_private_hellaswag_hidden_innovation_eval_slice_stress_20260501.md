# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `1024:2048`
- eval rows: `1024`
- selected accuracy: `0.454102`
- source-label copy accuracy: `0.409180`
- best label-copy accuracy: `0.414062`
- delta vs best label-copy: `0.040039`
- CI95 vs best label-copy: `[0.019531, 0.058130]`
- score-only bagged control accuracy: `0.409180`
- delta vs score-only bagged: `0.044922`
- zero-hidden control accuracy: `0.409180`
- wrong-example hidden control accuracy: `0.386719`
- candidate-roll hidden control accuracy: `0.372070`
- jackknife subbags passing: `3/3`
- jackknife min delta vs best label-copy: `0.032227`
- jackknife min CI95 low vs best label-copy: `0.011206`
- packet: `2B` raw / `5B` framed

## Interpretation

This is the first frozen post-first1024 validation-slice stress for the
bagged hidden-innovation packet. The method, train samples, split seeds, ridge
family, aggregation rule, and packet contract are unchanged from the
three-sample jackknife gate. Only the evaluation rows change.

The result reduces the slice-overfit concern: the packet still beats
source-label/trained-label copy, score-only bagging, and zero-hidden controls
with positive paired uncertainty. Wrong-example and candidate-roll hidden
controls stay below best label-copy, and all `2-of-3` train-sample subbags
remain positive.

## Reviewer Boundary

This promotes HellaSwag from a live diagnostic to a headline-candidate hard
benchmark, not a finished ICLR result. The comfortable ICLR gate is still full
validation or a predeclared multi-slice validation stress, plus native
vLLM/SGLang systems rows on NVIDIA hardware. The method is also not a prefix
token, soft prompt, adapter, or KV-cache relay: the target receives only a
fixed discrete candidate/confidence packet, not source text, source KV, raw
hidden vectors, raw source scores, or learned prompt state.
