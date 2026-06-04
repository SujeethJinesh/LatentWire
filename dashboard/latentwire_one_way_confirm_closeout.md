# LatentWire One-Way Confirm Closeout

- scope: held-out confirm closeout for the one-way negative only; no positive method selection used confirm.
- confirm rows: `381`
- best non-oracle baseline: `source_index_confidence`
- confirm accuracy: `{'target_only': 0.1942257217847769, 'source_index': 0.16010498687664043, 'source_index_confidence': 0.2047244094488189, 'current_packet_4bit': 0.18110236220472442, 'deployable_wz': 0.18110236220472442, 'full_source_score_fusion_oracle': 0.1732283464566929, 'source_target_at_encoder_upper_bound': 0.31496062992125984, 'random_same_byte': 0.09711286089238845}`
- deployable delta vs best: `{'n': 381, 'delta': -0.023622047244094488, 'ci95_low': -0.06036745406824147, 'ci95_high': 0.013123359580052493, 'mde_half_width': 0.03674540682414698}`
- full source-only oracle delta vs best: `{'n': 381, 'delta': -0.031496062992125984, 'ci95_low': -0.06299212598425197, 'ci95_high': 0.0026246719160104987, 'mde_half_width': 0.03412073490813648}`
- source+target-at-encoder upper bound delta vs best: `{'n': 381, 'delta': 0.11023622047244094, 'ci95_low': 0.08136482939632546, 'ci95_high': 0.14173228346456693, 'mde_half_width': 0.031496062992125984}`

The source+target-at-encoder row is an upper-bound diagnostic only and remains non-deployable.
