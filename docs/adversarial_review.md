# Adversarial Review

Date: 2026-05-25

Scope: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`.

## Claim-Level Attacks

| Claim | Adversarial critique | Current defense | Adequacy |
|---|---|---|---|
| M11b is a partial remedy. | Granite top-5 CI is very wide, so the median may be unstable. | Paper reports CI `[-1.300900018791, 1.000790626547]`, positive-gap trace count, and Nemotron budget shift. | Adequate for workshop if framed as partial, not settled. |
| Cross-model drift is robust. | The models are small and not all 7B+ production targets. | Scope explicitly says small reasoning models and W4A16 PTQ; larger models are limitations/future work. | Adequate. |
| Quamba2 contradiction. | Per-component differences are small and block-output tensors are not internal SSM states. | Paper limits the contradiction to block-output persistence at long decode and says internal SSM state is untested. | Adequate if wording remains scoped. |
| ParoQuant context. | ParoQuant is stronger than M11b, so why study channel-set methods? | Paper says ParoQuant is a different rotation mechanism and constrains the negative claim. | Adequate; avoid selling M11b as deployable SOTA. |
| KL weakens compound-error hypothesis. | Only static, DecDEC proxy, and M11 were tested on Granite. | Paper says "in this measured packet" and does not generalize to all regimes. | Adequate. |
| Release reproducibility. | FAST_VERIFY is frozen replay, not full reproduction. | README and scripts now state this explicitly and fail in unimplemented full mode. | Scientifically honest, but weaker than a full artifact. |

## Recommendations

- Keep the title and abstract hedged around "partial cross-model remedy."
- Do not describe M11b as production-ready or SOTA.
- Preserve the exact positive-gap trace counts wherever M11b results appear.
- Treat full GPU release reproduction as a post-draft artifact gap unless a
  dedicated full runner is added.
