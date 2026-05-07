"""Lightweight tensor packet helpers for Mac-local activation/state traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_tensor_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def save_tensor_packet(
    output_dir: Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> None:
    """Save a small trace packet with tensors and JSON metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    safe_names: dict[str, str] = {}
    for name in tensors:
        safe_name = _safe_tensor_name(name)
        if safe_name in safe_names:
            raise ValueError(
                "tensor names collide after packet-safe normalization: "
                f"{safe_names[safe_name]!r} and {name!r}"
            )
        safe_names[safe_name] = name
    manifest: dict[str, dict[str, object]] = {}
    for name, tensor in tensors.items():
        safe_name = _safe_tensor_name(name)
        storage_name = f"{safe_name}.pt"
        path = output_dir / storage_name
        cpu_tensor = tensor.detach().cpu()
        torch.save(cpu_tensor, path)
        manifest[safe_name] = {
            "original_name": name,
            "safe_name": safe_name,
            "storage_name": storage_name,
            "sha256": _sha256(path),
            "dtype": str(cpu_tensor.dtype),
            "shape": list(cpu_tensor.shape),
            "numel": int(cpu_tensor.numel()),
        }
    (output_dir / "tensor_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_tensor_packet(input_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(input_dir.glob("*.pt")):
        tensors[path.stem] = torch.load(path, map_location="cpu")
    return tensors, metadata


def load_tensor_manifest(input_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = input_dir / "tensor_manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tensor_manifest.json must be a JSON object")
    manifest: dict[str, dict[str, object]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            manifest[str(key)] = value
    return manifest
