# OutlierMigrate Experiment D: Decomposition Formalization

## Definitions

- Strict set-leaving: a channel with rank inside the top-1% boundary at decode position 100 has rank outside that boundary at the final decode position.
- Within-set rank shuffling: the channel remains inside the final top-1% set but moves by more than 2 rank positions.
- Gate migration: the preregistered original migration metric from each packet's `metrics.json`.

## Component Summary

| Packet | Gate migration | Gate CI95 | Strict set-leaving | Within-set shuffling | Strict original |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 0 Granite-4.0-H-Tiny | 0.817838541667 | [0.797265625000, 0.836848958333] | 0.634244791667 [0.607031250000, 0.666276041667] | 0.175260416667 [0.159244791667, 0.191406250000] | 0.808072916667 [0.786979166667, 0.830338541667] |
| Phase 1 Granite-4.0-H-Small | 0.843165650407 | [0.833434959350, 0.851143292683] | 0.566234756098 [0.549186991870, 0.580640243902] | 0.270934959350 [0.261382113821, 0.280995934959] | 0.836763211382 [0.827159552846, 0.844969512195] |
| Phase 2 Nemotron-3-Nano partial | 0.820809591643 | [0.786532526116, 0.854493114910] | 0.533713200380 [0.470589981007, 0.601109924027] | 0.269082383666 [0.238336894587, 0.297898860399] | 0.801667853751 [0.765105650522, 0.842117758784] |
| Phase 5' pure-Transformer control | 0.839378720238 | [0.827845982143, 0.850725446429] | 0.670572916667 [0.651506696429, 0.691127232143] | 0.165736607143 [0.152436755952, 0.176804315476] | 0.834914434524 [0.821707589286, 0.847656250000] |

## Kendall Tau Summary

| Packet | Position | Mean Kendall tau | 95% bootstrap CI |
| --- | ---: | ---: | ---: |
| Phase 0 Granite-4.0-H-Tiny | 500 | 0.104244095384 | [0.097278041446, 0.112469469610] |
| Phase 0 Granite-4.0-H-Tiny | 1000 | 0.101618976628 | [0.094642658710, 0.108111558570] |
| Phase 0 Granite-4.0-H-Tiny | 5000 | 0.101281078227 | [0.095699594106, 0.106858374050] |
| Phase 0 Granite-4.0-H-Tiny | 10000 | 0.095205618893 | [0.089710385422, 0.101633853036] |
| Phase 1 Granite-4.0-H-Small | 500 | 0.054401941668 | [0.051825047556, 0.057856737198] |
| Phase 1 Granite-4.0-H-Small | 1000 | 0.061946421566 | [0.052497679323, 0.078341621147] |
| Phase 1 Granite-4.0-H-Small | 5000 | 0.050511015045 | [0.048494330015, 0.052571337354] |
| Phase 1 Granite-4.0-H-Small | 10000 | 0.050637880480 | [0.048261605722, 0.052906324723] |
| Phase 1 Granite-4.0-H-Small | 20000 | 0.051319648392 | [0.049121201810, 0.053680640761] |
| Phase 2 Nemotron-3-Nano partial | 500 | 0.221169690616 | [0.159722048268, 0.277718669789] |
| Phase 2 Nemotron-3-Nano partial | 1000 | 0.193113005034 | [0.145923161518, 0.238895370807] |
| Phase 2 Nemotron-3-Nano partial | 5000 | 0.163484425401 | [0.124206673684, 0.201270504283] |
| Phase 2 Nemotron-3-Nano partial | 10000 | 0.156416428761 | [0.120022877780, 0.192418250312] |
| Phase 2 Nemotron-3-Nano partial | 20000 | 0.150317988534 | [0.115821529823, 0.184060309848] |
| Phase 5' pure-Transformer control | 500 | 0.127249408539 | [0.115026260296, 0.141995292748] |
| Phase 5' pure-Transformer control | 1000 | 0.130608809290 | [0.110507284355, 0.161912833024] |
| Phase 5' pure-Transformer control | 5000 | 0.122138340144 | [0.114204407291, 0.130692875663] |
| Phase 5' pure-Transformer control | 10000 | 0.125824496253 | [0.114999075495, 0.141162583998] |
| Phase 5' pure-Transformer control | 20000 | 0.118480037853 | [0.111061553379, 0.127009968703] |

## Interpretation

Phase 5' shows that high rank migration is not confined to the measured Mamba-2 hybrid models. The decomposition remains useful for static-protection methods because strict set-leaving directly measures channels that leave a fixed protected set.
