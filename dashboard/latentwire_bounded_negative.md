# LatentWire Bounded Negative

- status: `BOUNDED_NEGATIVE_LOCKED_FOR_ONE_WAY_AND_L_Q1`
- scope: MMLU-Pro one-way score packets plus L_Q1 receiver-query score packet.
- confirm rows scored for one-way negative closeout: `381`
- powered ladder: `results/mac_continue/latentwire_oracle_ladder/summary.json`
- held-out confirm closeout: `results/mac_continue/latentwire_one_way_confirm/summary.json`
- L_Q1 query-packet screen: `results/mac_continue/latentwire_query_packet/summary.json`

## Evidence

- Cached WZ/source-copy family: killed as a deployable positive method on row-safe cached screens.
- Fresh option-score WZ/L-B1 smoke: underpowered at `gate_n=23`; useful as a warning, not a kill.
- Powered oracle ladder: `1500` dev/gate scored rows, `359` gate rows, achieved MDE half-width `0.030641`.
- Deployable WZ: gate accuracy `0.178273`, below source-index+confidence baseline `0.192201`; delta `-0.013928`, CI `[-0.050139, 0.022284]`.
- Full source-score fusion oracle: also below the same baseline; delta `-0.019499`, CI `[-0.055710, 0.016713]`.
- Source+target-at-encoder upper bound: positive; delta `0.125348`, CI `[0.089136, 0.161560]`.
- Receiver-conditioned MI: `I(source_scores; correct | source_top1, target_scores) = 0.281933` bits, but it does not produce a deployable source-only win on this ladder.
- Held-out confirm: deployable WZ accuracy `0.181102` versus source-index+confidence `0.204724`; delta `-0.023622`, CI `[-0.060367, 0.013123]`.
- Held-out confirm full source-only oracle: delta `-0.031496`, CI `[-0.062992, 0.002625]`.
- Held-out confirm source+target-at-encoder upper bound: delta `0.110236`, CI `[0.081365, 0.141732]`.
- L_Q1 receiver-query packet: killed on gate; accuracy `0.169916` versus source-index+confidence `0.192201`, delta `-0.022284`, CI `[-0.055710, 0.011142]`; controls did not collapse and query-only/reply-only ablations explain the signal.
- L-B1: no AURC signal on the prior fresh screen; do not promote damage avoidance.

## Classification

`not-deployable`: only the oracle that sees target-side information at encode time helps. Source-only score packets and full source-only fusion do not beat the best non-oracle baseline on the powered gate.

## Claim Boundary

This supports a bounded-negative workshop story about source-copy leakage, non-deployable oracle headroom, and the importance of receiver-conditioned controls. It does not support a LatentWire positive-method claim. L-A2 remains parked until a real generated-solution candidate-pool cache exists; do not chase additional score-packet variants.
