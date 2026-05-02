# ARC Cross-Family Failure Decomposition

- wrappers analyzed: `3`
- selected next gate: `common_feature_connector_with_stronger_source`

## Lay Explanation

This diagnostic asks whether cross-family transfer failed because the sender chose weak answers, because the tiny packet failed to carry the sender's choice, or because the receiver needs a better common feature space for deciding which model to trust.

## Per-Wrapper Decisions

### phi3_mini_4k

- pass gate: `False`
- primary blocker: `source_endpoint_quality`
- source quality gap vs target: `-0.019625`
- packet decode gap vs source: `-0.001365`
- packet follows source rate: `0.996928`
- matched minus Qwen-substituted on disagreement: `-0.140096`
- next gate: replace or improve the cross-family source endpoint before revising the 8B packet codec

### qwen2.5_1.5b

- pass gate: `False`
- primary blocker: `mixed_or_unresolved`
- source quality gap vs target: `0.179181`
- packet decode gap vs source: `-0.002048`
- packet follows source rate: `0.996075`
- matched minus Qwen-substituted on disagreement: `0.297938`
- next gate: rerun with larger slices and a stronger alternate source to disambiguate

### tinyllama_1.1b

- pass gate: `False`
- primary blocker: `source_family_mismatch`
- source quality gap vs target: `0.060580`
- packet decode gap vs source: `-0.001365`
- packet follows source rate: `0.996075`
- matched minus Qwen-substituted on disagreement: `-0.047780`
- next gate: learn a common-feature selector/router that decides when to trust the alternate source
