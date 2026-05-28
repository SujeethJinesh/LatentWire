# Paper Reframe Options

## A. Rotation-Dominant Framing

"Dynamic channel-set protection is fragile under long reasoning drift;
rotation-style conditioning is the robust positive remedy."

This framing becomes strongest if ParoQuant also works on DeepSeek and Falcon.
Figures/tables to update:

- Main method matrix: promote ParoQuant above M11b.
- Recovery comparison figure: show ParoQuant Granite and Nemotron with CIs.
- Failure table: channel-set methods become explanatory negatives.
- Discussion: explain why ParoQuant+M11b sub-additivity suggests overlap rather
  than composition.

## B. Regime-Aware Framing

"A calibration protocol selects rotation, dynamic channel protection, or
fallback precision depending on architecture and drift diagnostics."

This framing remains safer until DeepSeek/Falcon rotation is measured.
Figures/tables to update:

- Decision tree: rotation first, Falcon rescue only if rotation fails.
- Method matrix: Granite and Nemotron now both select rotation.
- Queue table: ParoQuant Falcon/DeepSeek smoke moves before Falcon HYST.
- Limitations: V1 is an algorithmic reproduction, not upstream kernel parity.
