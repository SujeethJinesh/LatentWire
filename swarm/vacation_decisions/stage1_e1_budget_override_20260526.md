# Stage 1 E1 Budget Override

Timestamp: 2026-05-26T03:20:00Z

## Decision

Pause Stage 1 GPU work before continuing E1.

## Reason

The raised cap is 360 cumulative GPU hours, with a required pause at 340 hours
if E1+E2+E3 are not complete. The ledger before E1 was 278.643 hours. Two
short E1 launch attempts consumed about 0.495 hours, bringing the estimated
ledger to 279.138 hours.

The compact E1 attempt reached:

- Nemotron prompt 0, decode position 1000 at 2026-05-26T03:14:13Z
- Nemotron prompt 0, decode position 2000 at 2026-05-26T03:17:41Z

This implies about 3.5 minutes per 1000 generated tokens after setup, or about
70 minutes per 20K-token BF16 trace. The full preregistered Nemotron arm needs
12 BF16 traces plus static-1%, DecDEC proxy, and M11 alpha=0.5 quantized KL
passes. That projects to roughly 50+ GPU hours for Nemotron alone before
DeepSeek and Falcon.

## Interpretation

This is not a scientific E1 result and not a model-mechanism finding. It is a
runtime feasibility finding: the dense 20K manual-decode KL protocol is much
more expensive on Nemotron-3-Nano than the Stage 1 estimate assumed.

## Options Left Open

1. Narrow E1 while preserving 12 traces.
2. Run dense E1 on DeepSeek and Falcon first, deferring Nemotron dense KL.
3. Skip E1 and prioritize E2/E3.
4. Spend most of the remaining Stage 1 budget on full E1.

No option was selected autonomously because the budget override was explicitly
defined as a pause condition.
