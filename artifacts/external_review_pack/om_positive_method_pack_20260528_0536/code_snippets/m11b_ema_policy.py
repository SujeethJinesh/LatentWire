"""M11b endpoint EMA policy sketch."""

def ema_update(scores, topk_indicator, alpha=0.3):
    return [alpha * ind + (1 - alpha) * old for old, ind in zip(scores, topk_indicator)]
