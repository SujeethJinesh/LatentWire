# Compact pseudocode for the V1 scoring path.
# 1. Reuse BF16 traces and static top-1% score cache from Nemotron M11b.
# 2. Apply deterministic ParoQuant-style pairwise rotations to eligible weights.
# 3. Fold rotated W4A16 dequantized weights back into the model.
# 4. Score the fixed 512-token window ending at decode position 10000.
# 5. Compute unclipped recovery against the same BF16/static baseline.
