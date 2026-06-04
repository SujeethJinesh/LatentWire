# LatentWire Oracle Ladder

- classification: `not-deployable`
- powered: `True`
- achieved delta_beyond_score MDE half-width: `0.030641`
- MDE target: `0.050000`
- scored dev/gate rows: `1500`
- split counts: `{'dev': 1141, 'gate': 359, 'confirm': 381}`
- confirm rows scored: `0`
- next estimated scored rows if underpowered: `not needed`

## Mutual Information

- I(source_scores; correct_option | source_top1): `2.098527` bits
- I(source_scores; correct_option | source_top1, target_scores): `0.281933` bits

## Gate Ladder

- best non-oracle baseline: `source_index_confidence`
- gate accuracy: `{'target_only': 0.17827298050139276, 'source_index': 0.16434540389972144, 'same_byte_text_proxy': 0.16434540389972144, 'random_same_byte': 0.11142061281337047, 'source_index_confidence': 0.19220055710306408, 'wz_1bit': 0.1671309192200557, 'wz_2bit': 0.17827298050139276, 'current_packet_4bit': 0.17827298050139276, 'wz_8bit': 0.17270194986072424, 'full_source_score_fusion_oracle': 0.17270194986072424, 'deployable_wz': 0.17827298050139276, 'best_equal_byte_score_sketch': 0.17827298050139276, 'source_target_at_encoder_upper_bound': 0.31754874651810583}`
- deployable WZ delta vs best baseline: `{'n': 359, 'delta': -0.013927576601671309, 'ci95_low': -0.05013927576601671, 'ci95_high': 0.022284122562674095, 'mde_half_width': 0.036211699164345405}`
- current packet delta vs best baseline: `{'n': 359, 'delta': -0.013927576601671309, 'ci95_low': -0.05013927576601671, 'ci95_high': 0.022284122562674095, 'mde_half_width': 0.036211699164345405}`
- full source-score oracle delta vs best baseline: `{'n': 359, 'delta': -0.019498607242339833, 'ci95_low': -0.055710306406685235, 'ci95_high': 0.016713091922005572, 'mde_half_width': 0.036211699164345405}`
- source+target-at-encoder upper bound delta vs best baseline: `{'n': 359, 'delta': 0.12534818941504178, 'ci95_low': 0.08913649025069638, 'ci95_high': 0.1615598885793872, 'mde_half_width': 0.03621169916434541}`

The source+target-at-encoder row is an upper-bound diagnostic only; it is not deployable and must not be reported as a method.

No confirm rows were scored; this is dev/gate screening evidence only.
