# OutlierMigrate Experiment E: Threshold Sensitivity

## Scope

This post-hoc analysis recomputes decomposition at top-channel thresholds 0.5%, 1%, 2%, and 5%. It reports all thresholds; no threshold is selected post hoc.

## Summary

| Packet | Top % | Gate-style migration | Strict set-leaving | Within-set shuffling | Stable vs top-1%? |
| --- | ---: | ---: | ---: | ---: | --- |
| Phase 0 Granite-Tiny | 0.5 | 0.690625000000 | 0.583333333333 | 0.100000000000 | yes |
| Phase 0 Granite-Tiny | 1.0 | 0.817838541667 | 0.634244791667 | 0.175260416667 | yes |
| Phase 0 Granite-Tiny | 2.0 | 0.893279569892 | 0.669153225806 | 0.216666666667 | yes |
| Phase 0 Granite-Tiny | 5.0 | 0.949242424242 | 0.658008658009 | 0.286958874459 | no |
| Phase 1 Granite-Small | 0.5 | 0.723412698413 | 0.538293650794 | 0.177876984127 | yes |
| Phase 1 Granite-Small | 1.0 | 0.843165650407 | 0.566234756098 | 0.270934959350 | yes |
| Phase 1 Granite-Small | 2.0 | 0.913300304878 | 0.574123475610 | 0.335721544715 | yes |
| Phase 1 Granite-Small | 5.0 | 0.961966463415 | 0.689161585366 | 0.272500000000 | no |
| Phase 2 Nemotron-3 | 0.5 | 0.708390567766 | 0.520890567766 | 0.168498168498 | no |
| Phase 2 Nemotron-3 | 1.0 | 0.820809591643 | 0.533713200380 | 0.269082383666 | yes |
| Phase 2 Nemotron-3 | 2.0 | 0.894364316239 | 0.520922364672 | 0.363351733143 | yes |
| Phase 2 Nemotron-3 | 5.0 | 0.949163105413 | 0.566001899335 | 0.380359686610 | no |
| Phase 5' Transformer | 0.5 | 0.724702380952 | 0.639322916667 | 0.093005952381 | yes |
| Phase 5' Transformer | 1.0 | 0.839378720238 | 0.670572916667 | 0.165736607143 | yes |
| Phase 5' Transformer | 2.0 | 0.905817972350 | 0.688028033794 | 0.212605606759 | yes |
| Phase 5' Transformer | 5.0 | 0.955705009276 | 0.680967841682 | 0.270852659246 | no |

## Stability Rule

A threshold is marked stable when both strict set-leaving and within-set rank-shuffling remain within 0.10 absolute fraction of the top-1% values for the same packet.

## Figure

`threshold_sensitivity.pdf` plots strict set-leaving and within-set rank-shuffling as a function of the top-channel threshold.
