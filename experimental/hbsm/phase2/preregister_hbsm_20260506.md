# HBSM Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: hybrid boundary sensitivity mechanism

## B1: Sensitivity Heterogeneity Replication

Measure KL-divergence or downstream NLL/PPL sensitivity per layer under
simulated low precision on current hybrid reasoners.

Pass if AutoQuantize- or boundary-flagged layers cluster in the top decile of
measured sensitivity.

## B2: Cheaper Predictor

Test no-forward-pass predictors such as weight kurtosis, condition number, and
channel imbalance ratio against the B1 sensitivity ranking.

Pass if the best no-forward-pass predictor reaches Spearman rho at least 0.6
against the full sensitivity ranking.

## B3: Mechanism

Inject equal-magnitude noise before attention and before SSM components.

Pass if attention-output drift is significantly higher than SSM-output drift
for matched noise and the effect aligns with HORN boundary-direction results.

## Kill Rule

Kill if B1 does not reproduce on current hybrids, B2 predictors fail, or B3
does not add mechanism beyond existing sensitivity literature.
