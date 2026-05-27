# 2026-05-27T20:55Z positive-method queue additions

## Acknowledged queue order

Continue the in-flight M-PRED Granite run. After it completes:

1. M-PRED Nemotron, using the preregistered M-PRED protocol if budget permits.
2. M-PRED DeepSeek-R1-Distill and Falcon-H1, using the best alpha from the prior
   M-PRED arms and focusing on the top-10 budget.
3. M-FISH, if budget remains.
4. M-ROUTE on Nemotron after M-FISH, with a fresh scoop check before any run.
5. Decision-rule integration with all new results.
6. Final paper integration.

## M-PRED DeepSeek/Falcon criteria

Decision criterion added:

- `PASS_ARCHITECTURE_FILL`: M-PRED median recovery > 0.30 with CI lower > 0 on
  DeepSeek or Falcon, where M11b top-10 was ambiguous or failed.
- `KILL_UNIFORM`: M-PRED does not improve over M11b on DeepSeek/Falcon.

Priority: higher than M-ROUTE because it tests whether M-PRED fills the
cross-architecture gap left by V2.

## M-ROUTE criteria

M-ROUTE tests whether Nemotron's MoE routing structure explains why M11b works
there. Before starting, run a scoop check for:

- expert-conditional channel quantization, 2024--2026
- MoE per-expert outlier methods, 2024--2026
- EAQuant plus routing

Decision criterion added:

- `PASS_MECHANISM`: M-ROUTE beats M11b on Nemotron by >= 0.05 with
  non-overlapping CI.
- `PASS_FLAGSHIP`: M-ROUTE median recovery > 0.90 on Nemotron with CI lower
  > 0.5.
- `KILL`: M-ROUTE <= M11b; Nemotron success is not purely explained by
  expert routing.

## Budget priority

If budget pressure forces a choice, run M-PRED DeepSeek/Falcon before M-ROUTE.
If both additions pass, surface to the human because the paper would have two
architecture-grounded positive methods.
