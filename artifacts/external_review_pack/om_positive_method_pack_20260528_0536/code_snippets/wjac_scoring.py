"""WJAC score approximation."""

def wjac_score(ema_x2, weight_column_norm_sq):
    return weight_column_norm_sq * ema_x2
