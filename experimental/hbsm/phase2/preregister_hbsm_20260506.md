# HBSM Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: hybrid boundary sensitivity mechanism

## B1: Sensitivity Heterogeneity Replication

Measure KL-divergence or downstream NLL/PPL sensitivity per layer under
simulated low precision on current hybrid reasoners.
The first admissible packet must target a live hybrid model from
`experimental/shared/results/hybrid_model_eligibility_20260506/`, use fixed
boundary flags from the shared architecture maps, and include an explicit
perturbation-off no-op row whose drift is zero within `1e-5` relative tolerance.

Pass if AutoQuantize- or boundary-flagged layers cluster in the top decile of
measured sensitivity. Top-decile enrichment must beat random top-decile flags
with Fisher exact `p < 0.05` or a bootstrap enrichment lower bound above 1.0.

## B2: Cheaper Predictor

Test no-forward-pass predictors such as weight kurtosis, condition number, and
channel imbalance ratio against the B1 sensitivity ranking.

Pass if the best no-forward-pass predictor reaches Spearman rho at least 0.6
against the full sensitivity ranking and the bootstrap 95% lower bound is at
least 0.3. Required baselines: random ranking, layer-index ranking,
parameter-count ranking, weight-norm ranking, and boundary-only ranking.
The selected predictor must survive a train/test or leave-one-model-out split
before it can be described as a recipe.

## B3: Mechanism

Inject equal-magnitude noise before attention and before SSM components.

Pass if attention-output drift is significantly higher than SSM-output drift
for matched noise and the effect aligns with HORN boundary-direction results.
If B3 is statistically indistinguishable from HORN H2, fold HBSM into HORN
instead of maintaining a standalone paper.

## Kill Rule

Kill if B1 does not reproduce on current hybrids, B2 predictors fail, or B3
does not add mechanism beyond existing sensitivity literature.
