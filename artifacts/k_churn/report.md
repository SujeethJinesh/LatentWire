# C2 K-CHURN Rotated Residual-Set Churn

Status: `INCOMPLETE_NO_POSITIONAL_ROTATED_SET_CACHE`.

The available residual-correction artifacts contain residual norms and one aggregate long-decode activation EMA for Granite tail trace I_4. They do not contain position-resolved residual-benefit top-k sets in the rotated basis. That means this gate cannot choose static P1 versus drift-tracked P2, hysteretic P3, or phased P9. Do not run dynamic residual GPU jobs until a position-resolved residual-benefit cache exists.
