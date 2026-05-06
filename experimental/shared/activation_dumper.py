"""Lightweight tensor packet helpers for Mac-local activation/state traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch


def save_tensor_packet(
    output_dir: Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> None:
    """Save a small trace packet with tensors and JSON metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    for name, tensor in tensors.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        torch.save(tensor.detach().cpu(), output_dir / f"{safe_name}.pt")


def load_tensor_packet(input_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(input_dir.glob("*.pt")):
        tensors[path.stem] = torch.load(path, map_location="cpu")
    return tensors, metadata
