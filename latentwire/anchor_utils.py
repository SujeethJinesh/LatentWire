import torch
import logging

LOG = logging.getLogger("latentwire.anchor_utils")


def _normalize_anchor_text(text: str) -> str:
    if not text:
        return text
    normalized = text.rstrip()
    if not normalized.endswith(" "):
        normalized = normalized + " "
    return normalized


def apply_anchor_normalization(args):
    """Normalize anchor text inputs and enforce BOS policy consistency."""
    if args is None:
        return None

    if hasattr(args, "warm_anchor_text") and isinstance(args.warm_anchor_text, str):
        args.warm_anchor_text = _normalize_anchor_text(args.warm_anchor_text)

    if hasattr(args, "latent_anchor_text") and isinstance(args.latent_anchor_text, str):
        args.latent_anchor_text = _normalize_anchor_text(args.latent_anchor_text)

    has_anchor = bool(getattr(args, "warm_anchor_text", "").strip()) or bool(getattr(args, "latent_anchor_text", "").strip())

    if has_anchor:
        if hasattr(args, "append_bos_after_prefix"):
            setattr(args, "append_bos_after_prefix", "no")
        if hasattr(args, "train_append_bos_after_prefix"):
            setattr(args, "train_append_bos_after_prefix", "no")

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
