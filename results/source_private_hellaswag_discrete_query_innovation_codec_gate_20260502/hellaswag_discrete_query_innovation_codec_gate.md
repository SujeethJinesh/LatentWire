# HellaSwag Discrete Query Innovation Codec Gate

- pass gate: `False`
- default encoder: `prior_qwen_prob_q8_seed29_temp1.5_er10_k4`
- default accuracy: `0.605158`
- packet-only accuracy: `0.619199`
- default delta vs packet-only: `-0.014041`
- default CI95 low vs packet-only: `-0.018124`
- default oracle-headroom capture: `-0.240614`
- best scout accuracy: `0.619399`
- best scout delta vs packet-only: `0.000199`
- packet: `1B` raw / `4B` framed

## Interpretation

This gate is the bounded Mac-local test of the next materially different source-score branch after linear, nonlinear, and switch-observability selectors saturated. The source encoder uses fixed query summaries over source candidate tokens and is trained on an official-train target-conditioned residual objective, but inference transmits only a one-byte discrete code whose low bits preserve the source candidate packet. A pass would promote a decoder-conditioned innovation-code contribution; a fail means the current cached score surface still cannot convert HellaSwag complementarity into a positive learned method without stronger source representations or native connector training.
