# Decision: ParoQuant DeepSeek Rotation Dominates

Date: 2026-05-28T16:55Z

Run:
`experimental/outlier_migrate/phase9/results/om_v1_paroquant_deepseek_20260528T162858Z`

Decision: `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES`

Result:
- median recovery: 0.7562393091709771
- CI95: [-0.24565629003349176, 0.8553979703329783]
- included traces: 11/12
- margin vs DeepSeek static-top10 median: +0.37948669457657414

Interpretation:
ParoQuant-style rotation now passes all four measured models: Granite,
Nemotron, Falcon, and DeepSeek. This strongly shifts the paper toward a
rotation-dominant mechanism/protocol story. ParoQuant is still not our method;
the live positive-method question becomes whether DriftRot can improve
ParoQuant's tail risk, CI lower bound, or surface/branch robustness.

Caveat:
DeepSeek has a strong median but a negative lower CI because one included trace
has a large negative recovery tail. This keeps Scale/CVaR/Clip and residual
correction alive as tail-control gates.

Next gate:
Run Granite-focused DriftRot Scale/CVaR/Clip retune with exact ParoQuant
baseline included. A new method claim requires held-out confirmation or clear
tail/CI improvement over static ParoQuant, not just another baseline run.
