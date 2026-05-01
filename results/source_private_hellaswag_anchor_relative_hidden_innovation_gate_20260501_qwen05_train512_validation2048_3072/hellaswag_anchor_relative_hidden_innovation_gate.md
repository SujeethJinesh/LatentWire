# HellaSwag Anchor-Relative Hidden-Innovation Gate

- pass gate: `False`
- eval rows: `1024`
- anchor count: `128`
- component models: `9`
- selected accuracy: `0.452148`
- best label-copy accuracy: `0.445312`
- delta vs best label-copy: `0.006836`
- CI95 vs best label-copy: `[-0.010278, 0.025391]`
- score-only bagged control: `0.433594`
- zero-hidden control: `0.433594`
- wrong-example hidden control: `0.437500`
- candidate-roll hidden control: `0.426758`
- anchor-id shuffle control: `0.433594`
- anchor-value roll control: `0.433594`
- jackknife subbags passing: `0/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate tests whether the HellaSwag hidden-innovation signal survives a common-basis bottleneck. Dense candidate hidden residuals are not sent to the receiver; each component first expresses them as similarities to a train-only anchor bank, then emits the same fixed candidate/confidence packet. A pass would support a stronger shared-coordinate story; a failure would demote anchor-relative common-basis as a mechanism while preserving the dense-packet result.
