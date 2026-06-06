# LatentWire Falsification Ladder

| Escape | What it would have shown | Result | Controlling baseline / audit | Package artifact |
| --- | --- | --- | --- | --- |
| Score/WZ one-way packet | A deployable low-byte source residual beats source-score controls | Held-out aggregate delta `-0.023622`, CI `[-0.060367, 0.013123]`, `n=381` | source-index+confidence | `paper/latentwire/figures/heldout_source_index_null.png` |
| Powered score ladder | Receiver has residual headroom after source scores | current packet delta `-0.013928`, CI `[-0.050139, 0.022284]`, gate `n=359` | best equal-byte score/source baseline | `paper/latentwire/figures/receiver_conditioning_bits.png` |
| L-B1 damage-avoidance trust packet | Packet improves repair without copying source answer | Killed in Stage-1: matched packet mostly equals source-selected answer and damage avoidance fails | source-index/source-selected metadata | `paper/latentwire/tables/provenance.md` |
| L-Q1 receiver-query packet | Two-way query/reply creates new usable evidence | delta `-0.022284`, CI `[-0.055710, 0.011142]`, `n=359` | source-index+confidence; query-only/reply-only ablations | `paper/latentwire/tables/provenance.md` |
| Exact discrete evidence | Opaque packet carries symbolic evidence better than visible code | visible exact code `1.000` vs packet `0.775`; delta `+0.225`, CI `[0.192188, 0.257812]` | equal-byte public code; random/shuffled/answer-only controls at `0.250` | `paper/latentwire/figures/discrete_evidence_bar.png` |
| L-IB1 privacy bottleneck | Utility can be preserved while hiding source/evidence identity | best bottleneck utility/leakage `0.602339/0.602339`; current packet `0.875/1.000` | same-byte visible code and adaptive anonymized text | `paper/latentwire/figures/l_ib_utility_leakage.png` |
| C2C/KVComm smokes | Dense/cache anchors become byte-scale method evidence | Anchor only; answer packets leak answer text or tie deterministic controls | zero-source, teacher/candidate, deterministic packet controls | `registry/L_C2_c2c_kv_lcf_anchor.yaml` |
| L-PC5/L-C2 oracle ceilings | Rerank/fuser ceilings imply deployable latent methods | Rejected as oracle-only or setup-blocked; gold-aware ceilings are not claims | gold-leakage audit and equal-byte text controls | `dashboard/l_pc5_deployable_verifier_plan.md`, `dashboard/l_c2_oracle_decomposition.md` |
