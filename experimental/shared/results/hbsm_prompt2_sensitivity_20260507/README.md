# HBSM Prompt-Repeat Sensitivity Scout

Resource-limited two-prompt Granite Tiny HBSM B1 prompt-repeat packet.

- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- prompt count: `2`
- scoring layers: `8`
- row count: `64`
- Fisher boundary top-decile p-value: `1.0`
- boundary top-decile count: `0`
- non-boundary top-decile count: `1`
- cheap-predictor Spearman: `-0.667`

This packet is not promotable B1 evidence because it remains resource-limited by
prompt count and layer count. It is useful demotion evidence: the one-prompt
smoke's boundary top-decile event does not survive the first prompt-repeat
scout, and the cheap predictor remains negative.
