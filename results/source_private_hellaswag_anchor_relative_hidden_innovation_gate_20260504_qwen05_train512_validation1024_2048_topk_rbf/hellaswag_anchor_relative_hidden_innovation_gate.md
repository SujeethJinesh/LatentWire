# HellaSwag Anchor-Relative Hidden-Innovation Gate

- pass gate: `False`
- eval rows: `1024`
- anchor count: `128`
- anchor feature mode: `topk_rbf`
- component models: `9`
- selected accuracy: `0.409180`
- best label-copy accuracy: `0.414062`
- delta vs best label-copy: `-0.004883`
- CI95 vs best label-copy: `[-0.022461, 0.009766]`
- score-only bagged control: `0.409180`
- zero-hidden control: `0.409180`
- wrong-example hidden control: `0.409180`
- candidate-roll hidden control: `0.409180`
- anchor-id shuffle control: `0.409180`
- anchor-value roll control: `0.409180`
- jackknife subbags passing: `0/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate tests whether the HellaSwag hidden-innovation signal survives a common-basis bottleneck. Dense candidate hidden residuals are not sent to the receiver; each component first expresses them as similarities to a train-only anchor bank, then emits the same fixed candidate/confidence packet. A pass would support a stronger shared-coordinate story; a failure would demote anchor-relative common-basis as a mechanism while preserving the dense-packet result.
