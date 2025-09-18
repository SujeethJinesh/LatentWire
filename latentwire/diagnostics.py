import os, json
from typing import Dict
import torch
from .common import rms, save_json, LOG

@torch.no_grad()
def capture_stats(run_dir: str, model_name: str, lm_embed_weight: torch.Tensor, adapter_out: torch.Tensor, z: torch.Tensor, extra: Dict=None):
    stats = {
        "model": model_name,
        "embed_weight_rms": float(rms(lm_embed_weight)),
        "adapter_out_rms": float(rms(adapter_out)),
        "z_rms": float(rms(z)),
        "adapter_out_mean": float(adapter_out.float().mean()),
        "adapter_out_std": float(adapter_out.float().std()),
        "z_mean": float(z.float().mean()),
        "z_std": float(z.float().std()),
    }
    if extra:
        stats.update(extra)
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_stats.json")
    save_json(path, stats)
    LOG.info(f"[diag] wrote {path}")
    return stats
