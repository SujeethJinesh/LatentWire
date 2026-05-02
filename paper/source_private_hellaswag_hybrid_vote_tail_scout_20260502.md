# HellaSwag Hybrid Vote-On-Score-Agreement Tail Scout

## Status

The hybrid vote-on-score-agreement policy improves the HellaSwag terminal-tail
aggregate, but it does not clear the strict jackknife gate. This is a useful
method scout, not an ICLR full-validation rescue.

Current paper story: LatentWire has a `2B` raw / `5B` framed source-private
hidden-innovation packet that passes contiguous HellaSwag validation `0:9216`.
The terminal tail remains positive but jackknife-fragile, so full HellaSwag
validation is still blocked.

## Method

The existing dense bagged packet has two hidden-private decoders:

1. a mean-zscore aggregate over candidate scores from the hidden-innovation
   model bank;
2. a vote aggregate over the same hidden-innovation model bank.

The hybrid policy keeps the mean-zscore prediction unless that prediction
collapses to the score-only model-bank prediction. On those rows, it switches
to the hidden vote prediction. The same policy is applied to zero-hidden,
wrong-example hidden, and candidate-roll hidden controls.

In lay terms: when the fancy hidden hint says the same thing as the score-only
shortcut, we ask the independent hidden-model votes to break the tie. This
tests whether model-bank agreement carries useful source-private information.

## Validation[0:1024] Smoke

Artifact:
`results/source_private_hellaswag_hidden_innovation_bagged_gate_20260502_qwen05_train512_validation1024_hybrid_vote_on_score_agreement/hellaswag_hidden_innovation_bagged_gate.json`

Result: `pass_gate=true`.

- selected accuracy: `0.518555`
- best label-copy: `0.463867`
- score-only bagged: `0.461914`
- delta vs best label-copy: `+0.054688`
- paired CI95 low vs best label-copy: `+0.031250`
- delta vs score-only: `+0.056641`
- wrong-example hidden: `0.427734`
- candidate-roll hidden: `0.381836`
- packet: `2B` raw / `5B` framed

## Validation[9216:10042] Terminal Tail

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260502_qwen05_train512_validation9216_10042_hybrid_vote_on_score_agreement/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=false`.

- terminal rows: `826`
- selected accuracy: `0.547215`
- best label-copy: `0.498789`
- score-only bagged: `0.497579`
- delta vs best label-copy: `+0.048426`
- paired CI95 low vs best label-copy: `+0.020581`
- delta vs score-only: `+0.049637`
- paired CI95 low vs score-only: `+0.026634`
- wrong-example hidden: `0.469734`
- candidate-roll hidden: `0.414044`
- jackknife: `2/3`
- jackknife min delta vs best label-copy: `+0.021792`
- jackknife min CI95 low vs best label-copy: `-0.011531`
- packet: `2B` raw / `5B` framed

## Four-Sample Hybrid Follow-Up

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260502_qwen05_train512_validation9216_10042_4sample_hybrid_vote_on_score_agreement/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=false`.

- selected accuracy: `0.535109`
- delta vs best label-copy: `+0.036320`
- paired CI95 low vs best label-copy: `+0.009685`
- jackknife: `2/4`
- jackknife min delta vs best label-copy: `+0.019370`
- jackknife min CI95 low vs best label-copy: `-0.003632`

Adding the fourth train sample weakens the hybrid policy, so the previous
four-sample conclusion still holds: larger dense bagging is not enough.

## Decision

Promoted:

1. Hybrid hidden-vote fallback is a real aggregate improvement on the terminal
   tail and the validation-first1024 smoke.
2. The improvement survives score-only, zero-hidden, wrong-hidden, and
   candidate-roll controls in aggregate.
3. The mechanism is still source-private: no source text, source KV, raw hidden
   vector, or raw score vector is transmitted.

Weakened:

1. The hybrid does not clear strict terminal-tail jackknife.
2. Four-sample hybrid bagging does not rescue the failure.
3. The method is an ensemble-consensus repair rule, not a top-2 trust-or-switch
   breakthrough or common-basis solution.

Ruled out for now:

1. Claiming full HellaSwag validation.
2. More dense-bag scaling without a new representation or training objective.
3. Framing this as a new selective-classification method.

## Next Gate

The next highest-value gate is a switch-only oracle decomposition on one passed
slice and the terminal tail:

- source-label copy;
- source top-2 oracle;
- score-margin switch;
- hidden-contrast switch;
- hidden-innovation candidate denoiser;
- label-permuted switch;
- wrong-example hidden switch;
- score-only switch.

If hidden switching cannot achieve high switch precision and low false-switch
rate over label-copy, cut top-2 trust-or-switch as a paper contribution and
keep candidate-wise hidden-innovation denoising as the main method branch.
