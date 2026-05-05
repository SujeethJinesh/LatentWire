# ThoughtFlow-FP8 Real-Trace Retention Sweep

Status: **WEAKENED; no keep-rate band beats the strongest proxy on real generated traces.**

This sweeps saved generation traces across matched keep fractions.
It is still a text-proxy gate, not KV-cache telemetry and not a GPU result.

| Keep fraction | ThoughtFlow phase | Best other phase | Phase margin | ThoughtFlow math | Best other math | Math margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.281 | 0.281 | 0.000 | 0.234 | 0.352 | -0.117 |
| 0.15 | 0.707 | 0.707 | 0.000 | 0.328 | 0.398 | -0.070 |
| 0.20 | 0.941 | 0.941 | 0.000 | 0.484 | 0.557 | -0.073 |
| 0.25 | 0.988 | 0.988 | 0.000 | 0.677 | 0.692 | -0.014 |
| 0.30 | 0.998 | 0.998 | 0.000 | 0.801 | 0.806 | -0.004 |
| 0.35 | 1.000 | 1.000 | 0.000 | 0.877 | 0.877 | 0.000 |

## Decision

The current protected-token policy should advance only if it has a keep-rate band where it beats the strongest proxy rather than tying it.
If this sweep is weakened, the branch needs hidden/KV saliency before GPU work.
