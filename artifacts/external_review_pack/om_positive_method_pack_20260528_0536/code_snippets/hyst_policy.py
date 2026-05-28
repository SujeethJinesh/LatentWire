"""Hysteresis protected-set update."""

def hyst_update(previous, top_k, top_2k, budget):
    keep = set(previous) & set(top_2k)
    keep.update(top_k)
    return set(list(keep)[:budget])
