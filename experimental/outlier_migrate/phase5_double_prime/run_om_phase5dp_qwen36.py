#!/usr/bin/env python3
"""Run/probe OutlierMigrate Phase 5'' Qwen3.6 validation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase5_double_prime/results"
PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_SNAPSHOT_COMMIT = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SCHEMA_VERSION = "om_phase5dp_qwen36_v1"


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_environment() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in ["torch", "sglang", "transformers", "huggingface_hub"]:
        try:
            module = __import__(name)
            packages[name] = {"available": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            packages[name] = {"available": False, "import_error": repr(exc)}
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        torch_info = {"import_error": repr(exc)}
    return {
        "schema_version": f"{SCHEMA_VERSION}_environment",
        "created_at_utc": utc_now(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "repo_git_sha": command_output(["git", "rev-parse", "HEAD"], timeout=30)["stdout"],
        "packages": packages,
        "torch": torch_info,
        "commands": {
            "nvidia_smi": command_output(["nvidia-smi"], timeout=30),
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"], timeout=120),
        },
        "environment_variables": {
            key: os.environ.get(key)
            for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES"]
        },
    }


def build_prompt_manifest() -> dict[str, Any]:
    prompts = []
    for row_index, line in enumerate(PROMPT_FILE.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        prompts.append(
            {
                "index": int(item.get("index", row_index)),
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "answer": item.get("answer"),
                "source_dataset": item.get("source_dataset"),
                "source_file": item.get("source_file"),
                "source_commit": item.get("source_commit"),
            }
        )
    payload = "".join(str(row["prompt"]) for row in sorted(prompts, key=lambda row: row["index"])).encode("utf-8")
    return {
        "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
        "created_at_utc": utc_now(),
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_23",
        "prompt_file": str(PROMPT_FILE),
        "prompt_file_sha256": file_sha256(PROMPT_FILE),
        "prompt_count": len(prompts),
        "prompt_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }


def resolve_model_snapshot() -> dict[str, Any]:
    safe_id = "models--" + MODEL_ID.replace("/", "--")
    repo_dir = Path(os.environ.get("HF_HOME", "/workspace/hf_cache")) / "hub" / safe_id
    snapshot = repo_dir / "snapshots" / MODEL_SNAPSHOT_COMMIT
    return {
        "schema_version": f"{SCHEMA_VERSION}_model_provenance",
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "hf_snapshot_commit": MODEL_SNAPSHOT_COMMIT,
        "snapshot_path": str(snapshot) if snapshot.exists() else None,
        "cache_repo_path": str(repo_dir),
        "config_exists": (snapshot / "config.json").exists(),
        "weights_downloaded": any(snapshot.glob("*.safetensors")) if snapshot.exists() else False,
        "note": "Config snapshot was resolved before inference; full weights are not required for this infra capability probe.",
    }


def build_capability_probe() -> dict[str, Any]:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    try:
        # Registers Qwen3.5/Qwen3.6 configs with Transformers AutoConfig.
        import sglang.srt.utils.hf_transformers_utils  # noqa: F401
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        details["autoconfig"] = {
            "ok": True,
            "class": type(cfg).__name__,
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
        }
    except Exception as exc:
        details["autoconfig"] = {"ok": False, "error": repr(exc)}
        reasons.append(f"AutoConfig failed after SGLang registry import: {exc!r}")

    try:
        from sglang.srt.entrypoints.engine import Engine

        signature = inspect.signature(Engine.generate)
        details["engine_generate_parameters"] = list(signature.parameters)
        details["engine_has_return_hidden_states"] = "return_hidden_states" in signature.parameters
        details["engine_has_layer_hook_parameter"] = any(
            "hook" in name or "activation" in name or "layer" in name
            for name in signature.parameters
        )
    except Exception as exc:
        details["engine_generate_inspection"] = {"ok": False, "error": repr(exc)}
        reasons.append(f"Could not inspect SGLang Engine.generate: {exc!r}")

    engine_source = Path(
        "/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/entrypoints/engine.py"
    )
    if engine_source.is_file():
        text = engine_source.read_text(encoding="utf-8")
        details["engine_architecture_note"] = {
            "engine_source": str(engine_source),
            "scheduler_subprocess_doc_present": "Scheduler (subprocess)" in text,
            "tokenizer_manager_main_process_doc_present": "TokenizerManager" in text,
        }
    else:
        details["engine_architecture_note"] = {"engine_source_missing": str(engine_source)}

    activation_capture_accessible = False
    reasons.append(
        "SGLang Engine exposes request-level return_hidden_states but no public per-layer residual or block-output hook API; "
        "the model is owned by scheduler subprocesses, and source modification is forbidden by the preregistration."
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}_capability_probe",
        "created_at_utc": utc_now(),
        "activation_capture_accessible": activation_capture_accessible,
        "failure_classification": "per_layer_activation_capture_unavailable",
        "details": details,
        "reasons": reasons,
    }


def build_artifact_hashes(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}_artifact_hashes",
        "created_at_utc": utc_now(),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase5dp_qwen36_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, stdout_log)
    sys.stderr = Tee(sys.__stderr__, stderr_log)
    run_events = run_dir / "run_events.jsonl"
    run_events.write_text(json.dumps({"created_at_utc": utc_now(), "event": "run_started"}) + "\n", encoding="utf-8")

    environment = build_environment()
    prompt_manifest = build_prompt_manifest()
    model_provenance = resolve_model_snapshot()
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": utc_now(),
        "argv": sys.argv if argv is None else ["run_om_phase5dp_qwen36.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase5_double_prime_qwen36",
        "required_venv": "/workspace/.sglang",
        "actual_python": sys.executable,
        "inference_attempted": False,
        "reason_inference_not_attempted": "per-layer activation capture is not accessible without forbidden source modification",
    }
    capability_probe = build_capability_probe()
    infra_error = {
        "schema_version": f"{SCHEMA_VERSION}_infra_error",
        "created_at_utc": utc_now(),
        "decision": "FAIL_INFRA_QWEN36",
        "reasons": capability_probe["reasons"],
    }
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    write_json(run_dir / "model_provenance.json", model_provenance)
    write_json(run_dir / "command_metadata.json", command_metadata)
    write_json(run_dir / "capability_probe.json", capability_probe)
    write_json(run_dir / "infra_error.json", infra_error)
    run_events.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": utc_now(), "event": "run_completed", "decision": "FAIL_INFRA_QWEN36"}) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "decision": "FAIL_INFRA_QWEN36"}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
