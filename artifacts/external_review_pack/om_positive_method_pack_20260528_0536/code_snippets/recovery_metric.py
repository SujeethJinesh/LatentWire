"""Recovery metric used in the review pack."""

def recovery(perplexity_bf16, perplexity_static, perplexity_method):
    gap = perplexity_static - perplexity_bf16
    if gap <= 0:
        return None  # no recoverable static gap
    return 1.0 - (perplexity_method - perplexity_bf16) / gap
