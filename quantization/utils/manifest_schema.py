"""Manifest schema validation helpers."""
from __future__ import annotations

from typing import Dict, List


REQUIRED_FIELDS = [
    "bytes_estimate",
    "bytes_estimated_total",
    "bytes_measured_total",
    "bytes_measured_breakdown",
]


def validate_manifest(manifest: Dict) -> List[str]:
    missing = []
    for key in REQUIRED_FIELDS:
        if key not in manifest:
            missing.append(key)
    return missing
