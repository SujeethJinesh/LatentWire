def recovery(l_bf16, l_static, l_method):
    gap = l_static - l_bf16
    if gap <= 0:
        return None
    return 1.0 - (l_method - l_bf16) / gap
