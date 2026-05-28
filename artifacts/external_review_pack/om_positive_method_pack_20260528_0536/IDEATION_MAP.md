# Ideation Map

method_failed | failure_signal | inferred_bad_assumption | unexplored_fix | required_data | estimated_cost | novelty_risk
---|---|---|---|---|---|---
M2 position switching | loses to random | hard boundaries are cheap to track | soft/hysteretic transitions | churn/local-pool traces | cheap | low
M10 position bins | coarse bins underperform | position is sufficient state | content/layer-conditioned smooth budgets | per-layer drift and trace metadata | medium | medium
M11 EMA top-1% | weak recovery | 1% budget covers moving support | budget expansion or waterfilling | budget curves and no-gap rows | cheap | low
M11b Granite | positive but wide CI | budget alone solves all models | regime selector or rotation for Granite | ParoQuant/V1 and no-gap analysis | cheap | low
M11b DeepSeek | ambiguous | EMA transfers to dense Transformer | LAMBDA/HYST smoke or surface placement | stratified smoke traces | cheap | medium
M11b Falcon | near zero | parallel hybrid drift is EMA-trackable | branch/surface-local protection | branch-local activations | medium | medium
E3 composition | sub-additive | rotation and channel protection address independent errors | method selector rather than stacking | paired ParoQuant/M11b/control rows | cheap | low
M18 activation+K | negative recovery | cross-tensor coupling is stabilizing | loss-sensitive single-surface scoring | KLLOOK/Fisher proxy | medium | medium
M26 stable core | small ambiguous signal | stable channels are sufficient | stable core plus dynamic top-up with smoke gate | budget split curves | medium | medium
DecDEC proxy | weak/negative | short-horizon saliency transfers to long decode | long-horizon local surface diagnostics | per-position activations | medium | medium
M-PRED | large negative on Granite, weak elsewhere | drift is predictable by simple AR/innovation | avoid predictors; use robust selection or hysteresis | FFT entropy and per-trace recoveries | cheap | low
WJAC | offline kill diagnostics | weight norms change channel ranking materially | full Fisher only if KLLOOK shows headroom | WJAC parquet and KLLOOK samples | expensive | medium
