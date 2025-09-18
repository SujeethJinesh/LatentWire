import torch
import logging

LOG = logging.getLogger("latentwire.anchor_utils")


def apply_anchor_normalization(args):
    """Placeholder for backwards compatibility."""
    return None


def apply_anchor_and_bos(prefix_embeds: torch.Tensor, anchor_embeds: torch.Tensor, append_bos_after_prefix: str = "no"):
    """
    Concatenate [prefix, anchor] and control whether a BOS is appended by the caller.
    prefix_embeds: [M, d_model]
    anchor_embeds: [A, d_model]
    """
    if prefix_embeds.dim() == 2:
        prefix_embeds = prefix_embeds.unsqueeze(0)
    if anchor_embeds.dim() == 2:
        anchor_embeds = anchor_embeds.unsqueeze(0)
    out = torch.cat([prefix_embeds, anchor_embeds], dim=1)
    LOG.debug(f"[anchor] prefix+anchor -> {tuple(out.shape)}")
    return out
