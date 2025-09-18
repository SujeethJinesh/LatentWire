from typing import Optional, Dict
import torch
from .common import rms, LOG

def build_anchor_prefix_text(anchor_text: str = "Answer: ", append_bos_after_prefix: str = "no"):
    """Return the literal anchor string and BOS behavior flag."""
    if not anchor_text.endswith(" "):
        LOG.warning("Anchor text should have trailing space; appending one.")
        anchor_text = anchor_text + " "
    return {"anchor_text": anchor_text, "append_bos_after_prefix": append_bos_after_prefix}

@torch.no_grad()
def calibrate_adapter_outputs(adapter_out: torch.Tensor, lm_embed_weight: torch.Tensor, mode: str = "embed_rms", gain: float = 1.0):
    """
    Calibrate adapter outputs P in [M, d_model] to match LM embedding stats.
    mode="embed_rms": scale P so that its RMS matches token embedding RMS, then multiply by gain.
    """
    if mode != "embed_rms":
        return adapter_out
    tgt = rms(lm_embed_weight)  # reference RMS of token embeddings
    src = rms(adapter_out)
    scale = (tgt / (src + 1e-8)) * gain
    return adapter_out * scale
