"""LAMBDA smoke waterfill rule."""

def waterfill(global_scores, total_budget):
    # global_scores: [(score, layer, channel), ...]
    return sorted(global_scores, reverse=True)[:total_budget]
