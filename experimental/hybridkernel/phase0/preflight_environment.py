"""Record local HybridKernel runtime/dependency preflight status.

This script is a handoff artifact generator, not an installer. It records the
local Python/PyTorch accelerator surface and, when requested, checks whether
Triton packages are visible to the active pip index without installing them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRITON_PACKAGE_CANDIDATES = ("triton", "triton-cpu", "triton-nightly")


def _import_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"importable": False, "version": None, "error": "module not found"}
    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "importable": False,
            "version": None,
            "origin": spec.origin,
            "module_file": None,
            "error": repr(exc),
        }
    return {
        "importable": True,
        "version": getattr(module, "__version__", None),
        "origin": spec.origin,
        "module_file": getattr(module, "__file__", None),
        "error": None,
    }


def _torch_status() -> dict[str, Any]:
    status = _import_status("torch")
    if not status["importable"]:
        return status

    import torch

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    return {
        **status,
        "cuda_version": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "mps_built": bool(mps_backend is not None and mps_backend.is_built()),
        "mps_available": bool(mps_backend is not None and mps_backend.is_available()),
    }


def _run_pip_index(package: str, timeout_seconds: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "pip", "index", "versions", package]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "package": package,
            "command": command,
            "returncode": None,
            "available": False,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    no_match = "No matching distribution found" in output
    available = completed.returncode == 0 and not no_match
    return {
        "package": package,
        "command": command,
        "returncode": completed.returncode,
        "available": available,
        "timed_out": False,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect_preflight(
    *,
    check_pip_index: bool = False,
    pip_timeout_seconds: int = 30,
) -> dict[str, Any]:
    torch = _torch_status()
    triton_import = _import_status("triton")
    pip_index = []
    if check_pip_index:
        pip_index = [
            _run_pip_index(package, pip_timeout_seconds)
            for package in TRITON_PACKAGE_CANDIDATES
        ]

    any_triton_candidate_available = any(row["available"] for row in pip_index)
    if check_pip_index:
        triton_install_possible: bool | None = any_triton_candidate_available
    else:
        triton_install_possible = None
    triton_blocker = None
    if check_pip_index and not triton_import["importable"] and not any_triton_candidate_available:
        triton_blocker = (
            "Triton is not importable and pip index found no matching package "
            "among triton, triton-cpu, and triton-nightly for this environment."
        )

    status = "PASS"
    if not torch["importable"]:
        status = "BLOCKED_NO_TORCH"
    elif triton_blocker:
        status = "BLOCKED_TRITON_UNAVAILABLE"
    elif not triton_import["importable"]:
        status = "PENDING_TRITON_INDEX_CHECK"

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "torch": torch,
        "triton": {
            "import": triton_import,
            "usable_in_current_env": triton_import["importable"],
            "pip_index_checked": check_pip_index,
            "pip_index": pip_index,
            "install_possible_from_current_index": triton_install_possible,
            "blocker": triton_blocker,
        },
        "interpretation": _interpretation(status, torch, triton_blocker),
    }


def _interpretation(
    status: str,
    torch: dict[str, Any],
    triton_blocker: str | None,
) -> str:
    if status == "BLOCKED_NO_TORCH":
        return "HybridKernel local tests cannot run because torch is not importable."
    if status == "BLOCKED_TRITON_UNAVAILABLE":
        return (
            "Mac-local Phase 4 Triton interpreter tests should skip; do not add "
            "more kernels or claim performance until a native NVIDIA handoff "
            "environment is available. " + str(triton_blocker)
        )
    if status == "PENDING_TRITON_INDEX_CHECK":
        return (
            "Torch is available, but Triton is not importable. Re-run with "
            "--check-pip-index before handing this environment to another operator."
        )
    accelerator = "cuda" if torch.get("cuda_available") else "mps" if torch.get("mps_available") else "cpu"
    return f"Local runtime preflight passed for correctness work on {accelerator}."


def write_outputs(payload: dict[str, Any], json_path: Path, markdown_path: Path | None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    torch = payload["torch"]
    triton = payload["triton"]
    lines = [
        "# HybridKernel Local Preflight",
        "",
        f"- created UTC: `{payload['created_utc']}`",
        f"- status: `{payload['status']}`",
        f"- python: `{payload['platform']['python_version']}`",
        f"- executable: `{payload['platform']['python_executable']}`",
        f"- platform: `{payload['platform']['platform']}`",
        "",
        "## Runtime",
        "",
        f"- torch importable: `{torch['importable']}`",
        f"- torch version: `{torch.get('version')}`",
        f"- cuda available: `{torch.get('cuda_available')}`",
        f"- cuda version: `{torch.get('cuda_version')}`",
        f"- cuda device count: `{torch.get('cuda_device_count')}`",
        f"- mps built: `{torch.get('mps_built')}`",
        f"- mps available: `{torch.get('mps_available')}`",
        "",
        "## Triton",
        "",
        f"- importable: `{triton['import']['importable']}`",
        f"- version: `{triton['import'].get('version')}`",
        f"- origin: `{triton['import'].get('origin')}`",
        f"- module file: `{triton['import'].get('module_file')}`",
        f"- usable in current env: `{triton['usable_in_current_env']}`",
        f"- pip index checked: `{triton['pip_index_checked']}`",
        f"- install possible from current index: `{triton['install_possible_from_current_index']}`",
        f"- blocker: `{triton['blocker']}`",
        "",
        "## Pip Index Results",
        "",
        "| package | returncode | available | timed out |",
        "|---|---:|---:|---:|",
    ]
    for row in triton["pip_index"]:
        lines.append(
            f"| `{row['package']}` | `{row['returncode']}` | `{row['available']}` | `{row['timed_out']}` |"
        )
    if not triton["pip_index"]:
        lines.append("| none checked |  |  |  |")
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experimental/hybridkernel/phase0/local_preflight.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experimental/hybridkernel/phase0/local_preflight.md"),
    )
    parser.add_argument("--check-pip-index", action="store_true")
    parser.add_argument("--pip-timeout-seconds", type=int, default=30)
    args = parser.parse_args(argv)

    payload = collect_preflight(
        check_pip_index=args.check_pip_index,
        pip_timeout_seconds=args.pip_timeout_seconds,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(f"{payload['status']} wrote {args.output_json}")
    if args.output_md is not None:
        print(f"wrote {args.output_md}")
    return 0 if payload["status"] in {"PASS", "BLOCKED_TRITON_UNAVAILABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
