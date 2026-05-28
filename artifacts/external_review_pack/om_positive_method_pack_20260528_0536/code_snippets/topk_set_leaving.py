"""Strict top-k set leaving."""

def set_leaving(s0, st):
    s0, st = set(s0), set(st)
    return 1.0 - len(s0 & st) / max(1, len(s0))
