"""Restricted KLLOOK oracle sketch."""

def delta_kl(kl_without_channel, kl_with_channel):
    return kl_without_channel - kl_with_channel
