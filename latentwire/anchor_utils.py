# latentwire/anchor_utils.py
import re

def _norm_space(s: str) -> str:
    s = (s or "").replace("\t"," ").replace("\r"," ").replace("\n"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_anchor_text(s: str) -> str:
    # Ensure exactly one trailing space (tokenizers often expect it)
    s = _norm_space(s)
    if not s:
        return s
    if not s.endswith(" "):
        s = s + " "
    return s

def apply_anchor_normalization(args):
    # Train path uses --warm_anchor_text; eval uses --latent_anchor_text
    if hasattr(args, "warm_anchor_text") and args.warm_anchor_text:
        args.warm_anchor_text = normalize_anchor_text(args.warm_anchor_text)
    if hasattr(args, "latent_anchor_text") and args.latent_anchor_text:
        args.latent_anchor_text = normalize_anchor_text(args.latent_anchor_text)
    # If anchor text is present, discourage BOS insertion to avoid t=0 drift
    if (getattr(args, "warm_anchor_text", "") or getattr(args, "latent_anchor_text","")) and hasattr(args, "append_bos_after_prefix"):
        if str(getattr(args, "append_bos_after_prefix")).lower() != "no":
            setattr(args, "append_bos_after_prefix", "no")
    return args