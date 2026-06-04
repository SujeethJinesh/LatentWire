# LatentWire Query Packet

- status: `KILLED`
- protocol: receiver-query-conditioned two-way packet, not a one-way source-private packet.
- dev rows: `1141`
- gate rows: `359`
- byte accounting: query `2` + reply `1` = `3` total bytes
- selected alpha: `0.7`
- best equal-total-byte baseline: `source_index_confidence`
- gate accuracy: `{'target_only': 0.17827298050139276, 'source_index': 0.16434540389972144, 'source_index_confidence': 0.19220055710306408, 'equal_total_byte_score_sketch': 0.17827298050139276, 'equal_total_byte_text_proxy': 0.16434540389972144, 'query_only': 0.17827298050139276, 'reply_only': 0.1392757660167131, 'l_q1': 0.16991643454038996, 'wrong_row_query': 0.11142061281337047, 'wrong_row_reply': 0.17827298050139276, 'derangement': 0.18384401114206128, 'coordinate_shuffle': 0.15877437325905291, 'full_source_oracle': 0.17270194986072424, 'source_target_at_encoder_upper_bound': 0.31754874651810583}`
- delta vs best equal-total-byte baseline: `{'n': 359, 'delta': -0.022284122562674095, 'ci95_low': -0.055710306406685235, 'ci95_high': 0.011142061281337047, 'mde_half_width': 0.033426183844011144}`
- controls collapse: `False`
- query-only/reply-only explain: `True`
- source-copy leakage MI: `{'l_q1_pred_vs_source_top1': 0.45741866192334546, 'reply_only_pred_vs_source_top1': 0.5948578395117702}`

Gate rule: point delta must be positive versus the best equal-total-byte baseline, controls must collapse, and query-only/reply-only ablations must not explain the result.
